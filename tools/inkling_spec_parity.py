"""Sequential native greedy speculation parity and STOP/reset bank.

Exact 1100-token context, capacity 262144, Q8 KV, chunk 64, 128 outputs.
Token divergences remain failures, including possible numerical batch differences.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from inkling_context_bench import inputs
from inkling_smoke import (CAPACITY, GPU_UUID, ROOT, NativeSession, bounded_seconds,
                           device_selection, file_record, output_channels, stamp)


class StopSession(NativeSession):
    """Inject STOP while retaining NativeSession's protocol parser and sampler."""
    stop_record = None

    def next_line(self, deadline):
        timestamp, line = super().next_line(deadline)
        record = self.stop_record
        if record is not None and line.startswith("T "):
            record["received_before_stop"] += 1
            if record["received_before_stop"] == 16:
                self.proc.stdin.write("STOP\n")
                self.proc.stdin.flush()
                record["stop_sent"] = stamp()
                record["stop_writes"] = 1
                self.stop_record = None
        return timestamp, line


def first_difference(reference, actual):
    for i in range(max(len(reference), len(actual))):
        a = reference[i] if i < len(reference) else None
        b = actual[i] if i < len(actual) else None
        if a != b:
            return {"index": i, "reference_token": a, "actual_token": b}
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--core-gpu-uuid")
    parser.add_argument("--split-core-graph", action="store_true")
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("D:/ds4_work/inkling2/models/Inkling-2-Mini-UD-Q3_K_XL.gguf"))
    parser.add_argument("--tokenizer", type=Path, default=Path("D:/ds4_work/inkling2/models/tokenizer"))
    parser.add_argument("--seed", type=int, default=20261121)
    parser.add_argument("--load-timeout", type=bounded_seconds(180), default=180)
    parser.add_argument("--generate-timeout", type=bounded_seconds(3600), default=600)
    args = parser.parse_args(argv)
    for value in (args.gpu_uuid, args.core_gpu_uuid):
        if value is not None and not GPU_UUID.fullmatch(value):
            parser.error("GPU selection requires full UUIDs")
    if args.split_core_graph and not args.core_gpu_uuid:
        parser.error("--split-core-graph requires --core-gpu-uuid")
    if not 0 < args.seed <= 2**64 - 1:
        parser.error("seed must be in 1..2^64-1")
    folder = args.artifact_dir.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    if any(folder.iterdir()):
        parser.error("use a new empty artifact directory")
    report = {"status": "preparing", "started": stamp(), "capacity": CAPACITY,
              "kv": "q8", "chunk": 64, "context": 1100, "max_new": 128,
              "sampling": {"temperature": 0, "top_p": 1, "top_k": 0, "min_p": 0, "seed": args.seed},
              "processes": [], "checks": {},
              "notes": ["Greedy token divergence is reported, including possible numerical batch differences.",
                        "STOP is sent after receiving 16 tokens; buffering permits additional emitted tokens."]}

    def save():
        temporary = folder / "report.json.tmp"
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(folder / "report.json")

    try:
        report["files"] = {"exe": file_record(args.exe, True), "draft": file_record(args.draft, True),
                           "model": file_record(args.model),
                           "tokenizer": file_record(args.tokenizer / "tokenizer.json", True)}
        sources = [Path(__file__), ROOT / "tools/inkling_smoke.py", ROOT / "tools/inkling_context_bench.py"]
        sources += sorted((ROOT / "engine/src/inkling2").glob("*"))
        report["source_hashes"] = [file_record(p, True) for p in sources if p.is_file()]
        tokenizer, prompts = inputs(args.tokenizer, [1100])
        prompt = prompts[0]
        (folder / "input.json").write_text(json.dumps(prompt, ensure_ascii=False, indent=2), encoding="utf-8")
        visible, split_args = device_selection(args)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": visible, "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
        report["cuda_environment"] = {k: env[k] for k in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER")}
        reference = None
        for spec in (0, 1, 3, 7):
            directory = folder / ("reference" if spec == 0 else f"spec{spec}")
            directory.mkdir()
            command = [str(args.exe.resolve()), "--model", str(args.model.resolve()), "--serve",
                       "--max-context", str(CAPACITY), "--kv-q8", "--chunk", "64"] + split_args
            if spec:
                command += ["--draft", str(args.draft.resolve()), "--spec", str(spec)]
            process = {"spec": spec, "command": command, "runs": [], "started": stamp(),
                       "stdout_log": str(directory / "stdout.log"), "stderr_log": str(directory / "stderr.log")}
            report["processes"].append(process)
            session = StopSession(command, env, directory / "stdout.log", directory / "stderr.log")
            save()
            try:
                session.start()
                process["pid"] = session.proc.pid
                process["startup"] = session.ready(args.load_timeout)
                cases = [("parity", 128)]
                if spec == 3:
                    cases += [("stop", 1000), ("after_stop", 128)]
                for case, length in cases:
                    run = {"case": case, "requested_max_new": length}
                    process["runs"].append(run)
                    if case == "stop":
                        run.update({"received_before_stop": 0, "stop_writes": 0})
                        session.stop_record = run
                    session.generate(prompt["input_token_ids"], length, args.seed,
                                     args.generate_timeout, greedy=True, record=run)
                    session.stop_record = None
                    run.update(output_channels(tokenizer, run["output_token_ids"]))
                    if spec:
                        run["done"]["decode_generated_tokens"] = run["done"].pop("decode_forward_steps", None)
                        run["done"]["decode_generated_tok_s"] = run["done"].pop("decode_forward_tok_s", None)
                    if case == "stop":
                        run["alive_after_done"] = session.proc.poll() is None
                        # Native STOP cancellation is "cancel"; "stop" denotes EOS.
                        report["checks"]["stop"] = (run["stop_writes"] == 1 and run["done"]["finish"] == "cancel"
                            and len(run["output_token_ids"]) < 1000 and run["alive_after_done"])
                    elif spec == 0:
                        reference = run
                        report["checks"]["reference_length"] = len(run["output_token_ids"]) == 128
                    else:
                        run["first_difference"] = first_difference(reference["output_token_ids"], run["output_token_ids"])
                        report["checks"][f"spec{spec}_{case}_ids"] = run["first_difference"] is None
                        report["checks"][f"spec{spec}_{case}_finish"] = run["done"]["finish"] == reference["done"]["finish"]
                        if case == "after_stop":
                            initial = process["runs"][0]
                            run["reset_first_difference"] = first_difference(
                                initial["output_token_ids"], run["output_token_ids"])
                            report["checks"]["spec3_reset_ids"] = run["reset_first_difference"] is None
                            report["checks"]["spec3_reset_finish"] = run["done"]["finish"] == initial["done"]["finish"]
                    save()
            finally:
                session.stop_record = None
                try:
                    session.close()
                finally:
                    process["cleanup"] = session.cleanup
                    process["ended"] = stamp()
                    save()
            if session.proc is None or session.proc.poll() is None or session.proc.returncode != 0:
                raise RuntimeError("owned native process did not exit cleanly; refusing the next process")
        report["status"] = "ok" if all(report["checks"].values()) else "failed"
    except (Exception, KeyboardInterrupt) as error:
        report["status"] = "error"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        report["ended"] = stamp()
        save()
    print(json.dumps({"status": report["status"], "report": str(folder / "report.json"),
                      "checks": report["checks"], "error": report.get("error")}), flush=True)
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
