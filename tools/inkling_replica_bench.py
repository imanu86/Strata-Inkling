"""Opt-in replica bench. Never starts HTTP or stops another process.

run: native T1/P1/K0/minP0/seed20261121, 99 frozen Italian IDs, chunk64,
     256K capacity, Q8 KV, no draft; warmup then 128/512/1000 in one process.
logits: separate cold native input-logit dumps at chunk1 and chunk64. Native
        CLI is greedy; input logits precede sampling and are sampler-independent.
compare: strict input/output/logit parity; performance alone cannot pass parity.

Pass engine flags as a JSON array file. Profiling/training runs must precede
the measured BAB and use the same frozen profile for both B processes. This
harness has no SSD restore: native RAM prefix reuse is reported separately.
An external coordinator must arrange machine idle, server handoff and BAB order.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time

from inkling_smoke import CAPACITY, NativeSession, file_record, gen_line, machine_sample, prepare_prompt, stamp

PRIMARY = "GPU-3ec189a9-0fe7-a128-af2c-8df8fb3e15c7"
SECONDARY = "GPU-d945c2f8-983f-1aed-1e7d-5ba1510f13aa"
SEED = 20261121
LENGTHS = [128, 512, 1000]
PROMPT_IDS_SHA256 = "7335795005314e5f91d7d81bb478709627e618455dbd01f4c1281e5937e88cdf"
DEFAULT_INPUT = Path("D:/ds4_work/inkling2/runs/20261121-long-bab-A/input.json")
MODEL = Path("D:/ds4_work/inkling2/models/Inkling-2-Mini-UD-Q3_K_XL.gguf")
BASELINE = Path("D:/ds4_work/inkling2/builds/qkv-control-0dc78e64/strata-inkling2.exe")
FORBIDDEN = {"--native", "--model", "--serve", "--max-context", "--chunk", "--prefill", "--kv-f16", "--kv-q8",
             "--tokens", "--max-new", "-n", "--dump-logits", "--draft", "--draft-model", "--spec", "--draft-max",
             "--bench", "--verify-row0-oracle", "--oracle-force-t1"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def frozen_input(path):
    value = read_json(path)
    value = value.get("input", value)
    ids = value["input_token_ids"]
    if len(ids) != 99:
        raise ValueError("requires the frozen 99-token Italian prompt")
    gen_line(ids, 128, SEED)
    if hashlib.sha256(",".join(map(str, ids)).encode("ascii")).hexdigest() != PROMPT_IDS_SHA256:
        raise ValueError("input differs from the frozen Italian baseline token IDs")
    for key, expected in (("seed", SEED), ("temperature", 1), ("top_p", 1), ("top_k", 0), ("min_p", 0)):
        if value.get(key, expected) != expected:
            raise ValueError(f"frozen input has incompatible {key}")
    if value.get("effort", "none") != "none":
        raise ValueError("frozen input must have effort none")
    return {"input_token_ids": ids, "prompt": value.get("prompt"), "effort": "none", "seed": SEED,
            "temperature": 1, "top_p": 1, "top_k": 0, "min_p": 0}


def engine_flags(path):
    flags = read_json(path) if path else []
    if not isinstance(flags, list) or any(not isinstance(s, str) or not s for s in flags):
        raise ValueError("engine arguments JSON must be an array of nonempty strings")
    if any(s.split("=", 1)[0] in FORBIDDEN for s in flags):
        raise ValueError("engine arguments cannot override bench invariants or enable draft")
    return flags


def warmup_input(inputs, tokenizer_directory):
    prefix = "Scrivi un racconto originale"
    if not inputs.get("prompt", "").startswith(prefix):
        raise ValueError("frozen Italian prompt text is required to construct warmup")
    prompt = "Racconta una storia originale" + inputs["prompt"][len(prefix):]
    _, rendered, ids = prepare_prompt(tokenizer_directory, prompt, "none")
    lcp = 0
    for a, b in zip(inputs["input_token_ids"], ids):
        if a != b:
            break
        lcp += 1
    if lcp >= 64 or ids == inputs["input_token_ids"]:
        raise ValueError("warmup must diverge before the first 64-token checkpoint")
    gen_line(ids, 128, SEED)
    return {"prompt": prompt, "rendered_prompt": rendered, "input_token_ids": ids, "token_count": len(ids),
            "ids_sha256": hashlib.sha256(",".join(map(str, ids)).encode("ascii")).hexdigest(),
            "common_prefix_tokens": lcp, "effort": "none", "max_new": 128,
            "seed": SEED, "temperature": 1, "top_p": 1, "top_k": 0, "min_p": 0}


def command(exe, model, flags, chunk=64, logits=None, ids=None):
    result = [str(Path(exe).resolve()), "--model", str(Path(model).resolve()), "--max-context", str(CAPACITY),
              "--chunk", str(chunk), "--kv-q8", *flags]
    if logits is None:
        return result + ["--serve"]
    return result + ["--tokens", ",".join(map(str, ids)), "--max-new", "1", "--dump-logits", str(logits)]


def telemetry_peaks(samples):
    result = {"gpu": {}, "ram_used_bytes": 0, "engine_rss_bytes": None}
    for sample in samples:
        result["ram_used_bytes"] = max(result["ram_used_bytes"], sample.get("ram", {}).get("used_bytes", 0))
        rss = sample.get("engine_memory", {}).get("rss_bytes")
        if rss is not None:
            result["engine_rss_bytes"] = max(result["engine_rss_bytes"] or 0, rss)
        for row in csv.reader(sample.get("nvidia_smi", {}).get("stdout", "").splitlines(), skipinitialspace=True):
            if len(row) != 13:
                continue
            gpu = result["gpu"].setdefault(row[0].strip(), {})
            for index, key in ((4, "vram_mib"), (6, "utilization_pct"), (8, "power_w"), (9, "sm_mhz"),
                               (10, "memory_mhz"), (11, "pcie_gen"), (12, "pcie_width")):
                try:
                    gpu[key] = max(gpu.get(key, 0), float(row[index]))
                except ValueError:
                    pass
    return result


def replica_events(path):
    events = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("REPLICA "):
            event = json.loads(line[len("REPLICA "):])
            if not isinstance(event, dict) or "event" not in event:
                raise ValueError("malformed REPLICA diagnostic")
            events.append(event)
    return events


def replica_checks(events, measured_count, cli=False):
    ready = [e for e in events if e["event"] == "READY"]
    done = [e for e in events if e["event"] == "DONE"]
    measured = done if cli else done[1::2]
    done_positions = [i for i, e in enumerate(events) if e["event"] == "DONE"]
    hot_before_measure = len(done_positions) == 2 * measured_count and all(
        any(e["event"] == "RESET" and e.get("hot_loaded", 0) > 0 for e in events[done_positions[i] + 1:done_positions[i + 1]])
        for i in range(0, len(done_positions), 2))
    if cli:
        hot_before_measure = bool(done_positions) and any(e["event"] == "RESET" and e.get("hot_loaded", 0) > 0
                                                                        for e in events[:done_positions[0]])
    checks = {"full_residency": len(ready) == 1 and ready[0].get("full_experts") == 3072,
              "placement": len(ready) == 1 and ready[0].get("dense_device") == 1 and ready[0].get("replica_device") == 0,
              "hot_before_measure": hot_before_measure,
              "measured_done_count": len(done) == measured_count * (1 if cli else 2),
              "both_gpus_computed": bool(measured) and all(e.get("local_hits", 0) > 0 and e.get("remote_hits", 0) > 0 for e in measured),
              "computed_once": bool(measured) and all(e.get("duplicate_entries") == 0 and
                  e.get("computed_entries", -1) == e.get("local_hits", 0) + e.get("remote_hits", 0) for e in measured)}
    checks["no_missing_entries"] = bool(measured) and all(e.get("missing_entries") == 0 for e in measured)
    return {"passed": all(checks.values()), "checks": checks}


def reset_native(session, timeout):
    session.proc.stdin.write("RESET\n")
    session.proc.stdin.flush()
    deadline, events = time.monotonic() + timeout, []
    while True:
        timestamp, line = session.next_line(deadline)
        events.append({**timestamp, "line": line})
        if line == "RESET_OK":
            return events


def guardrail(rate, baseline, max_slowdown):
    if not isinstance(rate, (int, float)) or rate <= 0 or not isinstance(baseline, (int, float)) or baseline <= 0:
        raise ValueError("guardrail requires positive native decode rates")
    return {"passed": rate >= baseline * (1 - max_slowdown), "rate": rate, "baseline_rate": baseline,
            "max_slowdown": max_slowdown, "speed_ratio": rate / baseline}


def compare(a, b):
    checks = {"successful": a.get("status") == b.get("status") == "ok",
              "input_identical": a.get("input") == b.get("input"),
              "settings_identical": a.get("settings") == b.get("settings")}
    if a.get("mode") == b.get("mode") == "logits":
        checks["both_chunks_present"] = set(a.get("logits", {})) == set(b.get("logits", {})) == {"1", "64"}
        for chunk in ("1", "64"):
            left, right = a.get("logits", {}).get(chunk, {}), b.get("logits", {}).get(chunk, {})
            checks[f"chunk{chunk}_bitwise"] = bool(left.get("sha256")) and left.get("sha256") == right.get("sha256")
            checks[f"chunk{chunk}_full_vocab"] = left.get("size_bytes") == right.get("size_bytes") == 200058 * 4
    elif a.get("mode") == b.get("mode") == "run":
        checks["warmup_identical"] = a.get("warmup_input") == b.get("warmup_input")
        ar, br = a.get("runs", []), b.get("runs", [])
        schedule = [r.get("requested_max_new") for r in ar]
        checks["matching_schedule"] = schedule == [r.get("requested_max_new") for r in br] and schedule in ([128], LENGTHS)
        for index, length in enumerate(schedule):
            left = ar[index] if index < len(ar) else {}
            right = br[index] if index < len(br) else {}
            checks[f"ids_{length}"] = len(left.get("output_token_ids", [])) == length and left.get("output_token_ids") == right.get("output_token_ids")
            checks[f"finish_{length}"] = left.get("done", {}).get("finish") == right.get("done", {}).get("finish") == "length"
    else:
        checks["matching_modes"] = False
    return {"passed": all(checks.values()), "checks": checks,
            "final_lengths_completed": a.get("mode") == "run" and [r.get("requested_max_new") for r in a.get("runs", [])] == LENGTHS,
            "note": "Parity only. Residency, routing, idle machine and BAB performance require independent evidence."}


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=("run", "logits", "compare"))
    p.add_argument("--artifact-dir", type=Path)
    p.add_argument("--input-json", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--exe", type=Path, default=BASELINE)
    p.add_argument("--model", type=Path, default=MODEL)
    p.add_argument("--engine-args-json", type=Path)
    p.add_argument("--single-gpu", action="store_true", help="2080 only; default exposes only 2080 and 3080")
    p.add_argument("--tokenizer", type=Path, default=Path("D:/ds4_work/inkling2/models/tokenizer"))
    p.add_argument("--lengths", type=int, nargs="+", default=LENGTHS, help="128 for initial guardrail; 128 512 1000 for final bench")
    p.add_argument("--replica", action="store_true", help="Require RESET protocol and residency/routing diagnostics")
    p.add_argument("--logit-chunks", type=int, nargs="+", choices=(1, 64), default=[1, 64])
    p.add_argument("--cli-bench", action="store_true", help="Native CLI warmup for controlled zero-hot logits; no hot gate")
    p.add_argument("--guardrail-report", type=Path, help="Prior baseline run report: stop after128 if decode rate is too low")
    p.add_argument("--max-slowdown", type=float, default=0.30)
    p.add_argument("--timeout", type=float, default=180)
    p.add_argument("--left", type=Path)
    p.add_argument("--right", type=Path)
    p.add_argument("--plan", action="store_true", help="Print commands and exit without GPU or process activity")
    args = p.parse_args(argv)
    if not 0 < args.timeout <= 1800:
        p.error("timeout must be finite and in (0,1800]")
    if not 0 <= args.max_slowdown < 1:
        p.error("max-slowdown must be finite and in [0,1)")
    if args.lengths not in ([128], LENGTHS):
        p.error("lengths must be 128 or 128 512 1000")
    if args.replica and args.single_gpu:
        p.error("replica requires both explicit GPU UUIDs")
    if args.mode != "compare" and not args.single_gpu and not args.engine_args_json:
        p.error("dual GPU runs require explicit --engine-args-json; use the frozen baseline flags or prototype flags")
    if args.mode == "compare":
        if not args.left or not args.right:
            p.error("compare requires --left and --right report paths")
    elif not args.artifact_dir:
        p.error("run/logits require --artifact-dir")
    return args


def main(argv=None):
    args = arguments(argv)
    if args.mode == "compare":
        result = compare(read_json(args.left), read_json(args.right))
        print(json.dumps(result, indent=2))
        return 0 if result["passed"] else 1
    inputs, flags = frozen_input(args.input_json), engine_flags(args.engine_args_json)
    warmup = warmup_input(inputs, args.tokenizer) if args.mode == "run" else None
    folder = args.artifact_dir.resolve()
    visible = PRIMARY if args.single_gpu else PRIMARY + "," + SECONDARY
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": visible, "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
    commands = [command(args.exe, args.model, flags)] if args.mode == "run" else [
        command(args.exe, args.model, flags, c, folder / f"chunk{c}.f32", inputs["input_token_ids"]) for c in args.logit_chunks]
    if (args.replica or args.cli_bench) and args.mode == "logits":
        for cmd in commands:
            cmd.append("--bench")
    if args.plan:
        print(json.dumps({"commands": commands, "CUDA_VISIBLE_DEVICES": visible, "input": inputs, "warmup_input": warmup}, indent=2))
        return 0
    folder.mkdir(parents=True, exist_ok=False)
    report = {"schema": 1, "mode": args.mode, "status": "running", "started": stamp(), "input": inputs,
              "settings": {"capacity": CAPACITY, "kv": "q8", "chunk": 64, "mtp": False, "warmup": 128},
              "commands": commands, "CUDA_VISIBLE_DEVICES": visible, "runs": [], "logits": {},
              "warmup_input": warmup, "warmups": [],
              "notes": ["No SSD restore. RAM prefix reuse, prompt_read and prefill_ms come from DONE.",
                        "Sampled telemetry every 2 seconds; peaks can miss sub-interval excursions.",
                        "First token from prefill; decode steps=(generated-1), wall includes protocol overhead.",
                        "Native CLI logits use greedy; input logits are captured before sampling."]}
    samples, stop, session, sampler, cli_process = [], threading.Event(), None, None, None
    try:
        report["files"] = {"exe": file_record(args.exe, True), "model": file_record(args.model),
                           "input": file_record(args.input_json, True)}
        if args.engine_args_json:
            report["files"]["engine_args"] = file_record(args.engine_args_json, True)
        if warmup:
            report["files"]["tokenizer"] = file_record(args.tokenizer / "tokenizer.json", True)
            report["files"]["template"] = file_record(args.tokenizer / "chat_template.jinja", True)
        with (folder / "telemetry.jsonl").open("x", encoding="utf-8") as telemetry:
            def sample():
                value = machine_sample(PRIMARY, "during", session.proc if session else cli_process)
                samples.append(value)
                telemetry.write(json.dumps(value) + "\n")
                telemetry.flush()

            def sampling():
                while not stop.is_set():
                    try:
                        sample()
                    except Exception as error:
                        report["telemetry_error"] = str(error)
                        return
                    stop.wait(2)

            sample()
            sampler = threading.Thread(target=sampling, daemon=True)
            sampler.start()
            try:
                if args.mode == "run":
                    session = NativeSession(commands[0], env, folder / "stdout.log", folder / "stderr.log")
                    start = time.monotonic()
                    session.start()
                    report["startup"] = session.ready(args.timeout)
                    report["load_wall_s"] = time.monotonic() - start
                    for length in args.lengths:
                        training = {"before_length": length}
                        report["warmups"].append(training)
                        session.generate(warmup["input_token_ids"], 128, SEED, args.timeout, record=training)
                        if args.replica:
                            training["reset"] = reset_native(session, args.timeout)
                            current_events = replica_events(folder / "stderr.log")
                            resets = [e for e in current_events if e["event"] == "RESET"]
                            if not resets or resets[-1].get("hot_loaded", 0) <= 0:
                                raise ValueError("warmup reset did not confirm hot expert residency")
                        record = {"requested_max_new": length}
                        report["runs"].append(record)
                        session.generate(inputs["input_token_ids"], length, SEED, args.timeout, record=record)
                        if record["done"].get("prompt_read") != 99 or record["done"].get("reused") != 0:
                            raise ValueError("measured request must prefill exactly99 fresh tokens with zero prefix reuse")
                        if record["done"]["generated"] != length or record["done"]["finish"] != "length":
                            raise ValueError("output ended before requested length; no forced EOS suppression is used")
                        if args.guardrail_report and length == 128:
                            baseline = read_json(args.guardrail_report)
                            if baseline.get("status") != "ok" or baseline.get("input") != inputs or baseline.get("settings") != report["settings"] or baseline.get("warmup_input") != warmup:
                                raise ValueError("guardrail baseline must have identical input/settings")
                            baseline_run = next(r for r in baseline["runs"] if r["requested_max_new"] == 128)
                            report["guardrail"] = guardrail(record["done"]["decode_forward_tok_s"],
                                baseline_run["done"]["decode_forward_tok_s"], args.max_slowdown)
                            if not report["guardrail"]["passed"]:
                                raise ValueError("128-token slowdown guardrail stopped longer runs; coordinator must decide")
                else:
                    for chunk, cmd in zip(args.logit_chunks, commands):
                        start = time.monotonic()
                        with (folder / f"chunk{chunk}.stdout.log").open("xb") as out, (folder / f"chunk{chunk}.stderr.log").open("xb") as err:
                            cli_process = subprocess.Popen(cmd, env=env, stdout=out, stderr=err,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                            report.setdefault("cli_processes", []).append({"chunk": chunk, "pid": cli_process.pid})
                            try:
                                exit_code = cli_process.wait(timeout=args.timeout)
                            finally:
                                if cli_process.poll() is None:
                                    cli_process.terminate()
                                    try:
                                        cli_process.wait(timeout=3)
                                    except subprocess.TimeoutExpired:
                                        cli_process.kill()
                                        cli_process.wait(timeout=3)
                        if exit_code:
                            raise RuntimeError(f"chunk{chunk} exited {exit_code}")
                        path = folder / f"chunk{chunk}.f32"
                        report["logits"][str(chunk)] = {**file_record(path, True), "wall_process_s": time.monotonic() - start}
                        if path.stat().st_size != 200058 * 4:
                            raise ValueError("logits dump must contain all 200058 float32 logits")
                        if args.replica:
                            events = replica_events(folder / f"chunk{chunk}.stderr.log")
                            checks = replica_checks(events, 1, cli=True)
                            report["logits"][str(chunk)]["replica_events"] = events
                            report["logits"][str(chunk)]["replica_validation"] = checks
                            if not checks["passed"]:
                                raise ValueError(f"chunk{chunk} replica gate failed")
            finally:
                if session:
                    session.close()
                    report["cleanup"] = session.cleanup
                stop.set()
                sampler.join(timeout=12)
                if sampler.is_alive():
                    raise RuntimeError("telemetry sampler did not stop")
                sample()
        if report.get("telemetry_error"):
            raise RuntimeError("telemetry sampling failed: " + report["telemetry_error"])
        if args.replica and args.mode == "run":
            report["replica_events"] = replica_events(folder / "stderr.log")
            report["replica_validation"] = replica_checks(report["replica_events"], len(report["runs"]))
            if not report["replica_validation"]["passed"]:
                raise ValueError("replica residency/routing/computed-once gate failed")
        report["status"] = "ok"
    except (Exception, KeyboardInterrupt) as error:
        report["status"] = "error"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        stop.set()
        if args.replica and (folder / "stderr.log").exists():
            try:
                report["replica_events"] = replica_events(folder / "stderr.log")
                report["replica_validation"] = replica_checks(report["replica_events"], len(report["runs"]))
                report["replica_counter_scope"] = "Host per-entry ownership plan; actual GPU execution needs independent coverage test and Nsight overlap trace."
            except Exception as error:
                report["diagnostic_error"] = str(error)
                report["status"] = "error"
        report["telemetry_peaks"] = telemetry_peaks(samples)
        report["ended"] = stamp()
        (folder / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str(folder / "report.json"), "error": report.get("error")}))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
