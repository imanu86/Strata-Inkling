"""Short native Inkling smoke run; no HTTP and no model downloads.

Example (PowerShell, choose an actual GPU UUID and a new artifact directory):
  python tools/inkling_smoke.py --gpu-uuid GPU-... --artifact-dir D:/runs/inkling-smoke --runs 2

The default prompt/effort tests a short answer without reasoning. This is a smoke test,
not a quality benchmark or an SSD-cache benchmark. Context capacity is always 262144.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import queue
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[1]
CAPACITY = 262144
DEFAULT_PROMPT = "Qual è la capitale d'Italia? Rispondi soltanto con il nome della città, senza spiegazioni."
GPU_UUID = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\Z")


def stamp():
    return {"utc": datetime.now(timezone.utc).isoformat(), "monotonic_s": time.monotonic()}


def parse_done(line):
    """Inkling's positional DONE record; timings are emitted by the engine, in milliseconds."""
    fields = line.split()
    if len(fields) < 6 or fields[0] != "DONE":
        raise ValueError(f"malformed DONE: {line!r}")
    out = {"generated": int(fields[1]), "prompt_tokens": int(fields[2]),
           "prefill_ms": float(fields[3]), "decode_ms": float(fields[4]), "finish": fields[5], "raw": line}
    if min(out["generated"], out["prompt_tokens"]) < 0 or any(
            not math.isfinite(out[key]) or out[key] < 0 for key in ("prefill_ms", "decode_ms")):
        raise ValueError(f"invalid counts/timings in DONE: {line!r}")
    for index, key, convert in ((6, "drafts_accepted", int), (7, "drafts_offered", int), (8, "reused", int),
                                (9, "hits", int), (10, "lookups", int), (11, "ram_blobs", int),
                                (12, "file_blobs", int), (13, "file_mb", float), (14, "prompt_read", int)):
        if len(fields) > index:
            out[key] = convert(fields[index])
    # The first sampled token is produced by prefill. The timed decode loop performs N-1 forwards.
    steps = max(0, out["generated"] - 1)
    out["decode_forward_steps"] = steps
    out["decode_forward_tok_s"] = steps * 1000 / out["decode_ms"] if out["decode_ms"] > 0 else None
    out["emitted_tokens_per_decode_s"] = out["generated"] * 1000 / out["decode_ms"] if out["decode_ms"] > 0 else None
    fresh = out.get("prompt_read")
    out["prefill_tok_s"] = fresh * 1000 / out["prefill_ms"] if fresh is not None and out["prefill_ms"] > 0 else None
    return out


def gen_line(ids, max_new, seed, greedy=False):
    if not ids or len(ids) >= CAPACITY or type(max_new) is not int or not 32 <= max_new <= 1000 or seed <= 0:
        raise ValueError("GEN requires a nonempty prompt below capacity, max_new 32..1000 and a positive seed")
    if any(type(token) is not int or token < 0 for token in ids):
        raise ValueError("token IDs must be nonnegative integers")
    return f"GEN {max_new} temperature={0 if greedy else 1} top_p=1 top_k=0 min_p=0 seed={seed} " + ",".join(map(str, ids))


class NativeSession:
    """Own exactly one process and its pipes. No global process names or GPU process killing."""

    def __init__(self, command, env, stdout_path, stderr_path):
        self.command, self.env = command, env
        self.stdout_path, self.stderr_path = Path(stdout_path), Path(stderr_path)
        self.lines = queue.Queue()
        self.proc = None
        self.thread = None
        self.stderr = None
        self.cleanup = {}

    def start(self):
        self.stderr = self.stderr_path.open("xb")
        try:
            self.proc = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=self.stderr, text=True, encoding="utf-8", errors="replace", bufsize=1,
                                         env=self.env, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except BaseException:
            self.stderr.close()
            raise
        self.thread = threading.Thread(target=self._read, name="inkling-stdout", daemon=True)
        self.thread.start()

    def _read(self):
        try:
            with self.stdout_path.open("x", encoding="utf-8") as log:
                for line in self.proc.stdout:
                    text = line.rstrip("\r\n")
                    log.write(json.dumps({**stamp(), "line": text}, ensure_ascii=False) + "\n")
                    log.flush()
                    self.lines.put((stamp(), text))
        except BaseException as error:
            self.lines.put((stamp(), f"ERR stdout reader: {error}"))
        finally:
            self.lines.put((stamp(), None))

    def next_line(self, deadline):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("native protocol deadline exceeded")
        try:
            timestamp, line = self.lines.get(timeout=remaining)
        except queue.Empty:
            raise TimeoutError("native protocol deadline exceeded") from None
        if line is None:
            raise RuntimeError(f"engine stdout closed (exit={self.proc.poll()})")
        if line.startswith("ERR"):
            raise RuntimeError(line)
        return timestamp, line

    def ready(self, timeout):
        deadline = time.monotonic() + timeout
        initial = []
        while True:
            timestamp, line = self.next_line(deadline)
            initial.append({**timestamp, "line": line})
            if line.startswith("READY "):
                fields = line.split()
                if len(fields) < 2 or int(fields[1]) != CAPACITY:
                    raise ValueError(f"engine did not confirm required capacity {CAPACITY}: {line}")
                return initial

    def generate(self, ids, max_new, seed, timeout, greedy=False, record=None):
        record = record if record is not None else {}
        record.update({"started": stamp(), "input_token_ids": ids, "output_token_ids": [], "events": [],
                       "request": gen_line(ids, max_new, seed, greedy)})
        deadline = time.monotonic() + timeout
        self.proc.stdin.write(record["request"] + "\n")
        self.proc.stdin.flush()
        while True:
            timestamp, line = self.next_line(deadline)
            record["events"].append({**timestamp, "line": line})
            if line.startswith("T "):
                record["output_token_ids"].append(int(line.split()[1]))
                record.setdefault("first_token", timestamp)
            elif line.startswith("DONE "):
                record["done"] = parse_done(line)
                record["ended"] = timestamp
                record["wall_request_s"] = timestamp["monotonic_s"] - record["started"]["monotonic_s"]
                if record["done"]["generated"] != len(record["output_token_ids"]):
                    raise ValueError("DONE generated count differs from streamed token IDs")
                if record["done"]["prompt_tokens"] != len(ids):
                    raise ValueError("DONE prompt count differs from input token IDs")
                return record

    def close(self):
        proc = self.proc
        if proc is None:
            return
        self.cleanup = {"pid": proc.pid, "command": self.command, "started_by_this_script": True}
        if proc.poll() is None:
            try:
                proc.stdin.write("STOP\nQUIT\n")
                proc.stdin.flush()
            except (OSError, ValueError):
                pass
        # QUIT alone leaves the native reader blocked; EOF lets its join finish.
        if proc.stdin and not proc.stdin.closed:
            try:
                proc.stdin.close()
            except OSError:
                pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            # Popen owns the original process handle: this cannot target a reused unrelated PID.
            self.cleanup["terminated"] = True
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.cleanup["killed"] = True
                proc.kill()
                proc.wait(timeout=3)
        if self.thread:
            self.thread.join(timeout=3)
        if proc.stdout:
            proc.stdout.close()
        if self.stderr:
            self.stderr.close()
        self.cleanup.update({"exit_code": proc.returncode, "ended": stamp()})


def machine_sample(gpu_uuid, phase, process=None):
    import psutil
    ram = psutil.virtual_memory()
    out = {**stamp(), "phase": phase, "ram": {"total_bytes": ram.total, "available_bytes": ram.available,
                                               "used_bytes": ram.used, "percent": ram.percent}}
    if process is not None:
        try:
            mem = psutil.Process(process.pid).memory_info()
            out["engine_memory"] = {"pid": process.pid, "rss_bytes": mem.rss, "vms_bytes": mem.vms}
        except psutil.Error as error:
            out["engine_memory_error"] = str(error)
    # Observe the display card as well as both possible inference devices.
    out["main_gpu_uuid"] = gpu_uuid
    command = ["nvidia-smi",
               "--query-gpu=uuid,index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw,clocks.current.sm,clocks.current.memory,pcie.link.gen.current,pcie.link.width.current",
               "--format=csv,noheader,nounits"]
    out["nvidia_smi_command"] = command
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=5,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        out["nvidia_smi"] = {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr,
                              "columns": ["uuid", "index", "name", "memory.total MiB", "memory.used MiB", "memory.free MiB",
                                          "utilization.gpu %", "temperature.gpu C", "power.draw W", "SM MHz",
                                          "memory MHz", "PCIe generation", "PCIe width"]}
    except (OSError, subprocess.TimeoutExpired) as error:
        out["nvidia_smi"] = {"error": str(error)}
    out["sample_ended"] = stamp()
    return out


def file_record(path, digest=False):
    path = Path(path).resolve()
    info = path.stat()
    out = {"path": str(path), "size_bytes": info.st_size, "mtime_ns": info.st_mtime_ns}
    if digest:
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(block)
        out["sha256"] = hasher.hexdigest()
    return out


def prepare_prompt(tokenizer_dir, prompt, effort):
    sys.path.insert(0, str(ROOT / "terminal"))
    from serve.frontend import ChatTemplate
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(Path(tokenizer_dir) / "tokenizer.json"))
    template = ChatTemplate(Path(tokenizer_dir) / "chat_template.jinja")
    rendered = template.render([{"role": "user", "content": prompt}], reasoning_effort=effort)
    return tokenizer, rendered, tokenizer.encode(rendered, add_special_tokens=False).ids


def output_channels(tokenizer, ids):
    from serve.frontend import InklingOutputParser
    raw = tokenizer.decode(ids, skip_special_tokens=False)
    parser = InklingOutputParser()
    events = parser.feed(raw) + parser.finish()
    return {"raw_text": raw, "content": "".join(event.text or "" for event in events if event.kind == "content"),
            "reasoning": "".join(event.text or "" for event in events if event.kind == "reasoning")}


def bounded_seconds(maximum):
    def parse(value):
        number = float(value)
        if not math.isfinite(number) or not 0 < number <= maximum:
            raise argparse.ArgumentTypeError(f"expected 0 < seconds <= {maximum}")
        return number
    return parse


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpu-uuid", required=True, help="Full GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx UUID; never an index")
    parser.add_argument("--core-gpu-uuid", help="Enable layer split; the same UUID selects same-device parity")
    parser.add_argument("--split-core-graph", action="store_true", help="Use the opt-in split core decode graph")
    parser.add_argument("--split-outer-graph", action="store_true", help="Use the opt-in outer decode graphs")
    parser.add_argument("--artifact-dir", type=Path, required=True, help="Directory with no existing smoke artifact files")
    parser.add_argument("--exe", type=Path, default=ROOT / "build-windows-inkling/strata-inkling2.exe")
    parser.add_argument("--model", type=Path, default=Path("D:/ds4_work/inkling2/models/Inkling-2-Mini-UD-Q3_K_XL.gguf"))
    parser.add_argument("--draft", type=Path, help="Optional official DSpark GGUF; absent means no speculation")
    parser.add_argument("--spec", type=int, choices=range(1, 8), default=3)
    parser.add_argument("--batch-qkv-control-t1", action="store_true")
    parser.add_argument("--tokenizer", type=Path, default=Path("D:/ds4_work/inkling2/models/tokenizer"))
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--effort", choices=("none", "minimal", "low", "medium", "high", "xhigh", "max"), default="none")
    parser.add_argument("--max-new", type=int, choices=(32, 64, 128, 256, 512, 1000), default=128)
    parser.add_argument("--lengths", type=int, nargs="+", choices=(128, 256, 512, 1000),
                        help="Output lengths to run in sequence without reloading the model")
    parser.add_argument("--chunk", type=int, choices=(1, 64), default=64,
                        help="1 isolates decode parity from architecture-dependent batched prefill")
    parser.add_argument("--seed", type=int, default=20261121)
    parser.add_argument("--greedy", action="store_true", help="Explicit temperature=0 parity run; still recorded as greedy")
    parser.add_argument("--runs", type=int, choices=(1, 2), default=1, help="Repeat the identical GEN in the same process")
    parser.add_argument("--load-timeout", type=bounded_seconds(180), default=180)
    parser.add_argument("--generate-timeout", type=bounded_seconds(90), default=90)
    parser.add_argument("--sample-interval", type=bounded_seconds(60), default=2)
    parser.add_argument("--expert-vram", type=float, default=None, help="Optional native expert-placement budget, GiB")
    parser.add_argument("--cpu-threads", type=int, default=None)
    args = parser.parse_args(argv)
    if not GPU_UUID.fullmatch(args.gpu_uuid):
        parser.error("--gpu-uuid must be a full GPU UUID, not a CUDA/nvidia-smi index")
    if args.core_gpu_uuid is not None and not GPU_UUID.fullmatch(args.core_gpu_uuid):
        parser.error("--core-gpu-uuid must be a full GPU UUID, not an index")
    if (args.split_core_graph or args.split_outer_graph) and args.core_gpu_uuid is None:
        parser.error("split graphs require --core-gpu-uuid")
    if args.seed <= 0 or args.seed > 2 ** 64 - 1:
        parser.error("--seed must be an integer in 1..2^64-1")
    if args.expert_vram is not None and (not math.isfinite(args.expert_vram) or args.expert_vram < 0):
        parser.error("--expert-vram must be finite and nonnegative")
    if args.cpu_threads is not None and args.cpu_threads <= 0:
        parser.error("--cpu-threads must be positive")
    return args


def device_selection(args):
    visible = [args.gpu_uuid]
    split_args = []
    if args.core_gpu_uuid is not None:
        if args.core_gpu_uuid != args.gpu_uuid:
            visible.append(args.core_gpu_uuid)
        split_args = ["--layer-split-device", str(visible.index(args.core_gpu_uuid))]
        if args.split_core_graph:
            split_args.append("--split-core-graph")
        if getattr(args, "split_outer_graph", False):
            split_args.append("--split-outer-graph")
    return ",".join(visible), split_args


def main(argv=None):
    args = arguments(argv)
    folder = args.artifact_dir.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    names = ("report.json", "input.json", "stdout.log", "stderr.log", "telemetry.jsonl")
    if any((folder / name).exists() for name in names):
        raise SystemExit("artifact files already exist; use a new directory")
    report = {"schema": 1, "status": "preparing", "started": stamp(), "machine": platform.uname()._asdict(),
              "gpu_uuid": args.gpu_uuid, "core_gpu_uuid": args.core_gpu_uuid,
              "capacity": CAPACITY, "chunk": args.chunk, "kv": "q8", "runs": [],
              "measurement_notes": ["Timing fields come directly from native DONE, not request/HTTP wall time.",
                                    "First token comes from prefill; the decode rate counts generated-1 emitted tokens. With speculation this is not the forward count.",
                                    "Repeat is same-process RAM/resident-weight/KV reuse, not an SSD-cache benchmark.",
                                    "Native ram/file tier counters are placeholders; their zero values do not measure SSD reads."]}
    session = None
    sampler = None
    sampling_stop = threading.Event()
    tokenizer = None
    telemetry = None
    try:
        report["files"] = {"exe": file_record(args.exe, True), "model": file_record(args.model),
                           "tokenizer": file_record(args.tokenizer / "tokenizer.json", True),
                           "chat_template": file_record(args.tokenizer / "chat_template.jinja", True)}
        report["speculation"] = {"enabled": args.draft is not None, "draft_max": args.spec if args.draft else 0}
        if args.draft:
            report["files"]["draft"] = file_record(args.draft, True)
        manifest = args.tokenizer / "source-manifest.json"
        if manifest.exists():
            report["tokenizer_source"] = json.loads(manifest.read_text(encoding="utf-8-sig"))
        tokenizer, rendered, ids = prepare_prompt(args.tokenizer, args.prompt, args.effort)
        report["input"] = {"prompt": args.prompt, "effort": args.effort, "rendered_prompt": rendered, "input_token_ids": ids,
                           "max_new": args.max_new, "seed": args.seed, "temperature": 0 if args.greedy else 1,
                           "top_p": 1, "top_k": 0, "min_p": 0}
        schedule = (args.lengths or [args.max_new]) * args.runs
        report["input"]["max_new_schedule"] = schedule
        for length in schedule:
            gen_line(ids, length, args.seed, args.greedy)
        (folder / "input.json").write_text(json.dumps(report["input"], ensure_ascii=False, indent=2), encoding="utf-8")
        command = [str(args.exe.resolve()), "--model", str(args.model.resolve()), "--serve", "--max-context", str(CAPACITY),
                   "--chunk", str(args.chunk), "--kv-q8"]
        if args.draft:
            command += ["--draft", str(args.draft.resolve()), "--spec", str(args.spec)]
        if args.batch_qkv_control_t1:
            command.append("--batch-qkv-control-t1")
        if args.expert_vram is not None:
            command += ["--expert-vram", str(args.expert_vram)]
        if args.cpu_threads is not None:
            command += ["--cpu-threads", str(args.cpu_threads)]
        visible, split_args = device_selection(args)
        command += split_args
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": visible, "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
        report["command"] = command
        report["cuda_environment"] = {key: env[key] for key in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER")}
        report["timeouts_s"] = {"load": args.load_timeout, "generate": args.generate_timeout}
        telemetry = (folder / "telemetry.jsonl").open("x", encoding="utf-8")

        def sample(phase):
            value = machine_sample(args.gpu_uuid, phase, session.proc if session else None)
            telemetry.write(json.dumps(value, ensure_ascii=False) + "\n")
            telemetry.flush()

        sample("before")
        session = NativeSession(command, env, folder / "stdout.log", folder / "stderr.log")
        report["load_started"] = stamp()
        session.start()
        report["engine_pid"] = session.proc.pid

        def sample_during():
            while not sampling_stop.is_set():
                try:
                    sample("during")
                except Exception as error:
                    report["telemetry_error"] = str(error)
                    return
                sampling_stop.wait(args.sample_interval)

        sampler = threading.Thread(target=sample_during, name="inkling-machine-sampler", daemon=True)
        sampler.start()
        report["startup_protocol"] = session.ready(args.load_timeout)
        report["ready"] = stamp()
        report["load_wall_s"] = report["ready"]["monotonic_s"] - report["load_started"]["monotonic_s"]
        for index, length in enumerate(schedule):
            record = {"index": index + 1, "requested_max_new": length}
            report["runs"].append(record)
            session.generate(ids, length, args.seed, args.generate_timeout, args.greedy, record)
            if args.draft:
                record["done"]["decode_generated_tokens"] = record["done"].pop("decode_forward_steps", None)
                record["done"]["decode_generated_tok_s"] = record["done"].pop("decode_forward_tok_s", None)
            record.update(output_channels(tokenizer, record["output_token_ids"]))
        report["status"] = "ok"
    except (Exception, KeyboardInterrupt) as error:
        report["status"] = "error"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        sampling_stop.set()
        if sampler:
            sampler.join(timeout=7)
        if session:
            try:
                session.close()
            except Exception as error:
                report["status"] = "error"
                report["cleanup_error"] = str(error)
            report["cleanup"] = session.cleanup
        if tokenizer:
            for record in report["runs"]:
                if "raw_text" not in record:
                    record.update(output_channels(tokenizer, record.get("output_token_ids", [])))
        if telemetry:
            try:
                sample("after")
            except Exception as error:
                report["telemetry_after_error"] = str(error)
            telemetry.close()
        report["ended"] = stamp()
        with (folder / "report.json").open("x", encoding="utf-8") as output:
            json.dump(report, output, ensure_ascii=False, indent=2)
    print(json.dumps({"status": report["status"], "report": str(folder / "report.json"), "error": report.get("error")}, ensure_ascii=False))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
