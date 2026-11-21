"""Measure cold prefill and decode at exact input lengths in one native process.

Uses the smoke bank's protocol, sampling and telemetry. Prompts are synthetic
maintenance logs, not a quality benchmark. Capacity stays 262144 with Q8 KV.
Each prompt has a different opening to prevent RAM prefix reuse. No SSD restore.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import threading

from inkling_smoke import (CAPACITY, GPU_UUID, NativeSession, device_selection,
                           file_record, machine_sample, output_channels, prepare_prompt, stamp)


def inputs(tokenizer_dir, lengths):
    marker = "__STRATA_CONTEXT_BODY__"
    tokenizer, _, _ = prepare_prompt(tokenizer_dir, "Prova", "none")
    body = []
    for i in range(3000):
        text = (f"Registro {i:04d}: la tecnica Elena controlla il settore {i % 17} della stazione orbitale. "
                f"Pressione stabile, temperatura {18 + i % 9} gradi; il sensore {i % 43} registra "
                "un impulso radio inatteso. Marco confronta gli orari con il diario di manutenzione. "
                "La squadra conserva i dati, verifica i cablaggi e prepara il controllo successivo.\n")
        body.extend(tokenizer.encode(text, add_special_tokens=False).ids)
        if len(body) >= max(lengths):
            break
    result = []
    for length in lengths:
        prompt = (f"Caso di prova {length}. Leggi il seguente archivio di manutenzione.\n{marker}\n"
                  "Fine archivio. Scrivi un racconto originale in italiano di almeno 1600 parole "
                  "sulla scoperta del segnale e sulle indagini di Elena e Marco. Sviluppa eventi e dialoghi "
                  "in ordine cronologico. Scrivi solo il racconto, senza titolo o introduzione.")
        _, rendered, _ = prepare_prompt(tokenizer_dir, prompt, "none")
        before, after = rendered.split(marker)
        prefix = tokenizer.encode(before, add_special_tokens=False).ids
        suffix = tokenizer.encode(after, add_special_tokens=False).ids
        count = length - len(prefix) - len(suffix)
        if count <= 0 or count > len(body):
            raise ValueError("insufficient input body or context too short")
        ids = prefix + body[:count] + suffix
        assert len(ids) == length
        result.append({"context": length, "input_token_ids": ids,
                       "rendered_text": tokenizer.decode(ids, skip_special_tokens=False),
                       "construction": "Official template framing; body cut at a token boundary for exact length."})
    return tokenizer, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--core-gpu-uuid")
    parser.add_argument("--split-core-graph", action="store_true")
    parser.add_argument("--contexts", type=int, nargs="+", default=[10000, 20000, 40000])
    parser.add_argument("--max-new", type=int, choices=[128, 256, 512, 1000], default=512)
    parser.add_argument("--chunk", type=int, choices=[1, 64], default=64)
    parser.add_argument("--seed", type=int, default=20261121)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--draft", type=Path)
    parser.add_argument("--spec", type=int, choices=range(1, 8), default=3)
    parser.add_argument("--model", type=Path, default=Path("D:/ds4_work/inkling2/models/Inkling-2-Mini-UD-Q3_K_XL.gguf"))
    parser.add_argument("--tokenizer", type=Path, default=Path("D:/ds4_work/inkling2/models/tokenizer"))
    args = parser.parse_args()
    for value in [args.gpu_uuid, args.core_gpu_uuid]:
        if value is not None and not GPU_UUID.fullmatch(value):
            parser.error("GPU selection requires full UUIDs")
    if args.split_core_graph and not args.core_gpu_uuid:
        parser.error("core graph requires a core GPU")
    if any(n < 256 or n + args.max_new > CAPACITY for n in args.contexts):
        parser.error("input and output must fit the fixed capacity")
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    folder = args.artifact_dir.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    if any(folder.iterdir()):
        parser.error("use a new, empty artifact directory")
    tokenizer, prompts = inputs(args.tokenizer, args.contexts)
    (folder / "inputs.json").write_text(json.dumps(prompts, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"prepared_input_lengths": [len(p["input_token_ids"]) for p in prompts], "folder": str(folder)}), flush=True)
    if args.prepare_only:
        return 0
    visible, split_args = device_selection(args)
    command = [str(args.exe.resolve()), "--model", str(args.model.resolve()), "--serve",
               "--max-context", str(CAPACITY), "--kv-q8", "--chunk", str(args.chunk)] + split_args
    if args.draft:
        command += ["--draft", str(args.draft.resolve()), "--spec", str(args.spec)]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": visible, "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
    report = {"status": "loading", "started": stamp(), "capacity": CAPACITY, "kv": "q8", "chunk": args.chunk,
              "command": command, "exe": file_record(args.exe, True), "model": file_record(args.model),
              "tokenizer": file_record(args.tokenizer / "tokenizer.json", True),
              "sampling": {"temperature": 1, "top_p": 1, "top_k": 0, "min_p": 0, "seed": args.seed, "effort": "none"},
              "notes": ["Cold native prefill, no SSD restore; no HTTP timings.",
                        "Decode metric is (generated-1)/decode_seconds only without speculation.",
                        "Synthetic logs; this does not establish long-context model quality."], "runs": []}
    def save():
        temporary = folder / "report.json.tmp"
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(folder / "report.json")
    stop = threading.Event()
    session = NativeSession(command, env, folder / "stdout.log", folder / "stderr.log")
    def observe():
        with (folder / "telemetry.jsonl").open("x", encoding="utf-8") as telemetry:
            while not stop.is_set():
                try:
                    value = machine_sample(args.gpu_uuid, "context_bench", session.proc)
                    telemetry.write(json.dumps(value) + "\n")
                    telemetry.flush()
                except Exception as error:
                    telemetry.write(json.dumps({"error": str(error)}) + "\n")
                stop.wait(2)
    thread = threading.Thread(target=observe, daemon=True)
    try:
        save()
        session.start()
        report["pid"] = session.proc.pid
        thread.start()
        report["startup"] = session.ready(180)
        for prompt in prompts:
            run = {"context": prompt["context"], "requested_max_new": args.max_new}
            report["runs"].append(run)
            report["status"] = "generating"
            save()
            print(f"Starting input={prompt['context']} output={args.max_new}", flush=True)
            session.generate(prompt["input_token_ids"], args.max_new, args.seed, args.timeout, record=run)
            if args.draft:
                # Emitted tokens are not forward calls when a verification accepts a block.
                run["done"]["decode_generated_tokens"] = run["done"].pop("decode_forward_steps", None)
                run["done"]["decode_generated_tok_s"] = run["done"].pop("decode_forward_tok_s", None)
            run.update(output_channels(tokenizer, run["output_token_ids"]))
            if run["done"].get("reused") != 0:
                raise ValueError("unexpected prefix reuse: this is a cold-prefill bank")
            save()
            print(json.dumps({"context": prompt["context"], "done": run["done"]}), flush=True)
        report["status"] = "ok"
    except (Exception, KeyboardInterrupt) as error:
        report["status"] = "error"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        stop.set()
        if thread.ident:
            thread.join(timeout=7)
        session.close()
        report["cleanup"] = session.cleanup
        report["ended"] = stamp()
        save()
    print(json.dumps({"status": report["status"], "report": str(folder / "report.json"), "error": report.get("error")}), flush=True)
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
