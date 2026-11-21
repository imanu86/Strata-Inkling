"""CPU-only contract tests; no CUDA, nvidia-smi or native process launch."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import inkling_replica_bench as bench
from inkling_smoke import parse_done


class ReplicaBenchTests(unittest.TestCase):
    def test_command_keeps_native_invariants(self):
        cmd = bench.command("engine.exe", "model.gguf", ["--expert-dense-device", "1"])
        self.assertEqual(cmd[cmd.index("--max-context") + 1], "262144")
        self.assertEqual(cmd[cmd.index("--chunk") + 1], "64")
        self.assertIn("--kv-q8", cmd)
        self.assertEqual(cmd[-1], "--serve")
        self.assertNotIn("--draft", cmd)
        cli = bench.command("engine.exe", "model.gguf", [], 1, Path("a.f32"), [1, 2])
        self.assertNotIn("--serve", cli)
        self.assertEqual(cli[cli.index("--tokens") + 1], "1,2")

    def test_flags_cannot_silently_change_sampler_context_or_draft(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "args.json"
            for flags in (["--chunk", "1"], ["--draft=model"], ["--kv-f16"], [1], {"flags": []}):
                path.write_text(json.dumps(flags))
                with self.assertRaises(ValueError):
                    bench.engine_flags(path)
            path.write_text('["--expert-hot-mib", "4096"]')
            self.assertEqual(bench.engine_flags(path), ["--expert-hot-mib", "4096"])

    def test_decode_metric_excludes_first_prefill_token_and_wall(self):
        done = parse_done("DONE 128 99 500 2000 length")
        self.assertEqual(done["decode_forward_steps"], 127)
        self.assertEqual(done["decode_forward_tok_s"], 63.5)
        self.assertEqual(bench.guardrail(44.45, 63.5, .30)["passed"], True)
        self.assertEqual(bench.guardrail(44.44, 63.5, .30)["passed"], False)

    def test_peaks_parse_all_gpu_rows_and_unavailable_values(self):
        samples = [{"ram": {"used_bytes": 123}, "engine_memory": {"rss_bytes": 99}, "nvidia_smi": {
            "stdout": "GPU-a,0,RTX,12000,8000,4000,85,65,200,1500,9500,3,16\nGPU-b,1,RTX,22000,17000,5000,90,60,N/A,1400,7000,3,8"}}]
        peaks = bench.telemetry_peaks(samples)
        self.assertEqual(peaks["gpu"]["GPU-a"]["vram_mib"], 8000)
        self.assertEqual(peaks["gpu"]["GPU-b"]["pcie_width"], 8)
        self.assertNotIn("power_w", peaks["gpu"]["GPU-b"])
        self.assertEqual(peaks["engine_rss_bytes"], 99)

    def test_replica_gate_requires_hot_reset_and_both_routes(self):
        ready = {"event": "READY", "full_experts": 3072, "dense_device": 1, "replica_device": 0}
        done = {"event": "DONE", "local_hits": 10, "remote_hits": 20, "computed_entries": 30, "duplicate_entries": 0, "missing_entries": 0}
        events = [ready] + [done, {"event": "RESET", "hot_loaded": 100}, done] * 3
        self.assertTrue(bench.replica_checks(events, 3)["passed"])
        bad = copy.deepcopy(events)
        bad[2]["hot_loaded"] = 0
        self.assertFalse(bench.replica_checks(bad, 3)["passed"])
        bad = copy.deepcopy(events)
        bad[-1]["duplicate_entries"] = 1
        self.assertFalse(bench.replica_checks(bad, 3)["passed"])

    def test_warmup_preserves_effort_template_and_diverges_before_checkpoint(self):
        inputs = {"prompt": "Scrivi un racconto originale in italiano.", "input_token_ids": [1, 2, 3, 4]}
        with patch.object(bench, "prepare_prompt", return_value=(None, "rendered", [1, 2, 7, 8])) as prepare:
            warmup = bench.warmup_input(inputs, "tokenizer")
        self.assertEqual(prepare.call_args.args, ("tokenizer", "Racconta una storia originale in italiano.", "none"))
        self.assertEqual(warmup["common_prefix_tokens"], 2)
        self.assertEqual(warmup["token_count"], 4)
        self.assertEqual(warmup["max_new"], 128)

    def test_exact_output_parity_rejects_length_and_token_changes(self):
        a = {"status": "ok", "mode": "run", "input": {}, "settings": {}, "runs": [
            {"requested_max_new": n, "output_token_ids": list(range(n)), "done": {"finish": "length"}} for n in bench.LENGTHS]}
        self.assertTrue(bench.compare(a, copy.deepcopy(a))["passed"])
        b = copy.deepcopy(a)
        b["runs"][1]["output_token_ids"][400] = 10000
        self.assertFalse(bench.compare(a, b)["passed"])
        b = copy.deepcopy(a)
        b["runs"][-1]["output_token_ids"].pop()
        self.assertFalse(bench.compare(a, b)["passed"])

    def test_logit_parity_requires_full_vocab_and_hash(self):
        a = {"status": "ok", "mode": "logits", "input": {}, "settings": {}, "logits": {
            str(c): {"sha256": "abc", "size_bytes": 200058 * 4} for c in (1, 64)}}
        self.assertTrue(bench.compare(a, copy.deepcopy(a))["passed"])
        b = copy.deepcopy(a)
        b["logits"]["64"]["sha256"] = "different"
        self.assertFalse(bench.compare(a, b)["passed"])
        b = copy.deepcopy(a)
        b["logits"]["1"]["size_bytes"] = 4
        self.assertFalse(bench.compare(a, b)["passed"])


if __name__ == "__main__":
    unittest.main()
