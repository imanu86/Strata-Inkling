"""Check native Inkling RAM checkpoint/restore and shorter-prompt reset against cold runs.

Run from the checkout with explicit GPU UUIDs and a new artifact directory:
  python tools/inkling_state_parity.py --gpu-uuid GPU-... --core-gpu-uuid GPU-... --artifact-dir D:/runs/state-parity

Two sequential processes: A -> B warm, then B cold -> A reset. B is a valid conversation
whose exact prefix is A. Always context 262144, KV Q8, chunk 1, greedy, max_new 32.
No SSD cache claim: this tests native in-process RAM checkpoints and reset state.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import platform
import sys

from inkling_smoke import (CAPACITY, DEFAULT_PROMPT, GPU_UUID, ROOT, NativeSession, bounded_seconds,
                           device_selection, file_record, output_channels, prepare_prompt, stamp)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--core-gpu-uuid", help="Optional explicit split core UUID; may equal the primary UUID")
    parser.add_argument("--split-core-graph", action="store_true")
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--exe", type=Path, default=ROOT / "build-windows-inkling/strata-inkling2.exe")
    parser.add_argument("--model", type=Path, default=Path("D:/ds4_work/inkling2/models/Inkling-2-Mini-UD-Q3_K_XL.gguf"))
    parser.add_argument("--tokenizer", type=Path, default=Path("D:/ds4_work/inkling2/models/tokenizer"))
    parser.add_argument("--seed", type=int, default=20261121)
    parser.add_argument("--load-timeout", type=bounded_seconds(180), default=180)
    parser.add_argument("--generate-timeout", type=bounded_seconds(90), default=90)
    parser.add_argument("--expert-vram", type=float, default=None)
    parser.add_argument("--cpu-threads", type=int, default=None)
    args = parser.parse_args(argv)
    for name in ("gpu_uuid", "core_gpu_uuid"):
        value = getattr(args, name)
        if value is not None and not GPU_UUID.fullmatch(value):
            parser.error(f"--{name.replace('_', '-')} requires a full GPU UUID, never an index")
    if args.split_core_graph and args.core_gpu_uuid is None:
        parser.error("--split-core-graph requires --core-gpu-uuid")
    if not 0 < args.seed <= 2 ** 64 - 1:
        parser.error("--seed must be in 1..2^64-1")
    if args.expert_vram is not None and (not math.isfinite(args.expert_vram) or args.expert_vram < 0):
        parser.error("--expert-vram must be finite and nonnegative")
    if args.cpu_threads is not None and args.cpu_threads <= 0:
        parser.error("--cpu-threads must be positive")
    return args


def prepare_inputs(tokenizer_directory):
    tokenizer, text_a, ids_a = prepare_prompt(tokenizer_directory, DEFAULT_PROMPT, "none")
    # prepare_prompt loads the official local tokenizer and makes serve.frontend available, as in the smoke bank.
    from serve.frontend import ChatTemplate
    messages_a = [{"role": "user", "content": DEFAULT_PROMPT}]
    messages_b = messages_a + [{"role": "assistant", "content": "Roma"},
                               {"role": "user", "content": "E quella della Francia?"}]
    text_b = ChatTemplate(Path(tokenizer_directory) / "chat_template.jinja").render(messages_b, reasoning_effort="none")
    ids_b = tokenizer.encode(text_b, add_special_tokens=False).ids
    if not text_b.startswith(text_a) or len(ids_b) <= len(ids_a) or ids_b[:len(ids_a)] != ids_a:
        raise ValueError("B must extend A exactly, both rendered text and input token IDs")
    if not ids_a or len(ids_b) >= CAPACITY:
        raise ValueError("invalid prompt size")
    return tokenizer, {"A": {"messages": messages_a, "rendered_prompt": text_a, "input_token_ids": ids_a},
                       "B": {"messages": messages_b, "rendered_prompt": text_b, "input_token_ids": ids_b},
                       "exact_prefix": True, "prefix_token_count": len(ids_a)}


def first_difference(left, right):
    for index in range(max(len(left), len(right))):
        a = left[index] if index < len(left) else None
        b = right[index] if index < len(right) else None
        if a != b:
            return {"index": index, "left_token": a, "right_token": b}
    return None


def compare_runs(processes):
    by_case = {run["case"]: run for process in processes for run in process["runs"]}
    a, warm, cold, reset = (by_case[name] for name in ("A_initial", "B_warm", "B_cold", "A_reset"))
    differences = {"B_warm_vs_cold": first_difference(warm["output_token_ids"], cold["output_token_ids"]),
                   "A_initial_vs_reset": first_difference(a["output_token_ids"], reset["output_token_ids"])}
    checks = {"B_output_ids_match": differences["B_warm_vs_cold"] is None,
              "A_output_ids_match": differences["A_initial_vs_reset"] is None,
              "B_warm_reused_positive": warm["done"].get("reused", 0) > 0,
              "B_cold_reused_zero": cold["done"].get("reused") == 0,
              "A_reset_reused_zero": reset["done"].get("reused") == 0,
              "A_initial_reused_zero": a["done"].get("reused") == 0,
              "B_finish_matches": warm["done"]["finish"] == cold["done"]["finish"],
              "A_finish_matches": a["done"]["finish"] == reset["done"]["finish"],
              "all_runs_nonempty": all(run["output_token_ids"] for run in by_case.values()),
              "native_finishes_valid": all(run["done"]["finish"] in ("stop", "length", "eos") for run in by_case.values())}
    return {"passed": all(checks.values()), "checks": checks, "first_differences": differences,
            "reused": {name: run["done"].get("reused") for name, run in by_case.items()}}


def run_process(command, env, directory, cases, inputs, tokenizer, args, record):
    directory.mkdir()
    session = NativeSession(command, env, directory / "stdout.log", directory / "stderr.log")
    record.update({"command": command, "started": stamp(), "runs": [], "stdout_log": str(directory / "stdout.log"),
                   "stderr_log": str(directory / "stderr.log")})
    try:
        session.start()
        record["pid"] = session.proc.pid
        record["startup_protocol"] = session.ready(args.load_timeout)
        record["ready"] = stamp()
        record["load_wall_s"] = record["ready"]["monotonic_s"] - record["started"]["monotonic_s"]
        for case, prompt in cases:
            run = {"case": case, "prompt": prompt}
            record["runs"].append(run)
            session.generate(inputs[prompt]["input_token_ids"], 32, args.seed, args.generate_timeout, greedy=True, record=run)
            run.update(output_channels(tokenizer, run["output_token_ids"]))
    finally:
        try:
            session.close()
        finally:
            record["cleanup"] = session.cleanup
            record["ended"] = stamp()
            for run in record["runs"]:
                if "raw_text" not in run:
                    run.update(output_channels(tokenizer, run.get("output_token_ids", [])))
    if session.proc is not None and session.proc.poll() is None:
        raise RuntimeError("owned native process is still running; refuse to start the next process")
    if session.proc is None or session.proc.returncode != 0:
        raise RuntimeError(f"native process did not close cleanly: {session.cleanup}")


def main(argv=None):
    args = arguments(argv)
    folder = args.artifact_dir.resolve()
    if folder.exists() and any(folder.iterdir()):
        raise SystemExit("artifact directory is not empty; choose a new directory")
    folder.mkdir(parents=True, exist_ok=True)
    report = {"schema": 1, "status": "preparing", "started": stamp(), "machine": platform.uname()._asdict(),
              "capacity": CAPACITY, "chunk": 1, "kv": "q8", "effort": "none", "max_new": 32,
              "sampling": {"temperature": 0, "top_p": 1, "top_k": 0, "min_p": 0, "seed": args.seed},
              "gpu_uuid": args.gpu_uuid, "core_gpu_uuid": args.core_gpu_uuid, "split_core_graph": args.split_core_graph,
              "processes": [], "scope": "Native RAM checkpoint/restore and shorter-prompt reset; not an SSD cache test."}
    try:
        report["files"] = {"exe": file_record(args.exe, True), "model": file_record(args.model),
                           "tokenizer": file_record(args.tokenizer / "tokenizer.json", True),
                           "chat_template": file_record(args.tokenizer / "chat_template.jinja", True)}
        tokenizer, inputs = prepare_inputs(args.tokenizer)
        with (folder / "input.json").open("x", encoding="utf-8") as output:
            json.dump(inputs, output, ensure_ascii=False, indent=2)
        report["input"] = inputs
        report["timeouts_s"] = {"load": args.load_timeout, "generate": args.generate_timeout}
        command = [str(args.exe.resolve()), "--model", str(args.model.resolve()), "--serve", "--max-context", str(CAPACITY),
                   "--chunk", "1", "--kv-q8"]
        if args.expert_vram is not None:
            command += ["--expert-vram", str(args.expert_vram)]
        if args.cpu_threads is not None:
            command += ["--cpu-threads", str(args.cpu_threads)]
        visible, split_args = device_selection(args)
        command += split_args
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": visible, "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
        report["cuda_environment"] = {key: env[key] for key in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER")}
        for number, cases in enumerate(((('A_initial', 'A'), ('B_warm', 'B')), (('B_cold', 'B'), ('A_reset', 'A'))), 1):
            record = {"index": number}
            report["processes"].append(record)
            run_process(command, env, folder / f"process{number}", cases, inputs, tokenizer, args, record)
        report["comparison"] = compare_runs(report["processes"])
        report["status"] = "passed" if report["comparison"]["passed"] else "failed"
    except (Exception, KeyboardInterrupt) as error:
        report["status"] = "error"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        report["ended"] = stamp()
        with (folder / "report.json").open("x", encoding="utf-8") as output:
            json.dump(report, output, ensure_ascii=False, indent=2)
    print(json.dumps({"status": report["status"], "report": str(folder / "report.json"),
                      "comparison": report.get("comparison"), "error": report.get("error")}, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
