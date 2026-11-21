"""CPU-only protocol/timeout tests; uses a Python fake child, never the native engine or CUDA."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inkling_smoke import CAPACITY, NativeSession, arguments, device_selection, gen_line, main, parse_done


class InklingSmoke(unittest.TestCase):
    def test_done_timings_are_engine_timings(self):
        done = parse_done("DONE 3 10 20.00 40.00 stop 0 0 4 7 8 0 0 0.0 6")
        self.assertEqual((done["prefill_ms"], done["decode_ms"], done["reused"]), (20, 40, 4))
        self.assertEqual(done["prefill_tok_s"], 300)
        self.assertEqual(done["decode_forward_steps"], 2)
        self.assertEqual(done["decode_forward_tok_s"], 50)
        self.assertEqual(done["emitted_tokens_per_decode_s"], 75)
        for line in ("DONE", "DONE 3 10 nan 4 stop", "DONE 3 10 1 -1 stop"):
            with self.assertRaises(ValueError):
                parse_done(line)

    def test_gen_defaults_are_explicit(self):
        self.assertEqual(gen_line([1, 2], 32, 42), "GEN 32 temperature=1 top_p=1 top_k=0 min_p=0 seed=42 1,2")
        self.assertIn("temperature=0", gen_line([1], 64, 42, greedy=True))
        self.assertTrue(gen_line([1], 256, 42).startswith("GEN 256 "))
        for n in (128, 512, 1000):
            self.assertTrue(gen_line([1], n, 42).startswith(f"GEN {n} "))
        for ids, count, seed in (([], 32, 42), ([True], 32, 42), ([1], 1024, 42), ([1], 32, 0)):
            with self.assertRaises(ValueError):
                gen_line(ids, count, seed)

    def test_cli_rejects_indices_and_unbounded_timeouts(self):
        base = ["--gpu-uuid", "GPU-12345678-1234-1234-1234-123456789abc", "--artifact-dir", "unused"]
        args = arguments(base)
        self.assertEqual((args.max_new, args.effort, args.chunk), (128, "none", 64))
        self.assertEqual(arguments(base + ["--lengths", "128", "512", "1000"]).lengths, [128, 512, 1000])
        self.assertEqual(arguments(base + ["--chunk", "1"]).chunk, 1)
        for extra in (("--gpu-uuid", "0"), ("--load-timeout", "181"), ("--generate-timeout", "91"),
                      ("--generate-timeout", "nan"), ("--seed", "0"), ("--core-gpu-uuid", "1"), ("--split-core-graph",), ("--chunk", "0")):
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                arguments(base + list(extra))

    def test_split_device_selection_preserves_main_and_excludes_other_cards(self):
        outer = "GPU-12345678-1234-1234-1234-123456789abc"
        core = "GPU-87654321-1234-1234-1234-123456789abc"
        base = ["--gpu-uuid", outer, "--artifact-dir", "unused"]
        self.assertEqual(device_selection(arguments(base)), (outer, []))
        self.assertEqual(device_selection(arguments(base + ["--core-gpu-uuid", outer])),
                         (outer, ["--layer-split-device", "0"]))
        self.assertEqual(device_selection(arguments(base + ["--core-gpu-uuid", core])),
                         (outer + "," + core, ["--layer-split-device", "1"]))
        self.assertEqual(device_selection(arguments(base + ["--core-gpu-uuid", core, "--split-core-graph"])),
                         (outer + "," + core, ["--layer-split-device", "1", "--split-core-graph"]))

    def test_fake_native_two_gen_same_child_and_cleanup(self):
        script = '''import sys
print("INFO engine=fake", flush=True)
print("READY 262144 stop", flush=True)
for line in sys.stdin:
    if line.startswith("GEN "):
        print("RESUME 0", flush=True)
        print("PP 2 2 1 2000", flush=True)
        print("T 7", flush=True)
        print("T 8", flush=True)
        print("DONE 2 2 1 2 stop 0 0 0 0 0 0 0 0.0 2", flush=True)
    elif line.strip() == "QUIT":
        break
'''
        with tempfile.TemporaryDirectory() as folder:
            session = NativeSession([sys.executable, "-u", "-c", script], dict(os.environ),
                                    Path(folder) / "stdout.log", Path(folder) / "stderr.log")
            try:
                session.start()
                pid = session.proc.pid
                self.assertTrue(session.ready(3)[-1]["line"].startswith(f"READY {CAPACITY}"))
                for _ in range(2):
                    record = session.generate([1, 2], 32, 42, 3)
                    self.assertEqual(record["output_token_ids"], [7, 8])
                    self.assertEqual(session.proc.pid, pid)
            finally:
                session.close()
            self.assertEqual(session.cleanup["pid"], pid)
            self.assertEqual(session.cleanup["exit_code"], 0)
            self.assertTrue((Path(folder) / "stdout.log").read_text().count('"line": "DONE') == 2)

    def test_timeout_keeps_partial_record_and_closes_only_owned_child(self):
        script = 'import time; print("READY 262144 stop",flush=True); time.sleep(30)'
        with tempfile.TemporaryDirectory() as folder:
            session = NativeSession([sys.executable, "-u", "-c", script], dict(os.environ),
                                    Path(folder) / "stdout.log", Path(folder) / "stderr.log")
            record = {}
            try:
                session.start()
                session.ready(3)
                with self.assertRaises(TimeoutError):
                    session.generate([1], 32, 42, 0.02, record=record)
                self.assertEqual(record["input_token_ids"], [1])
                self.assertEqual(record["output_token_ids"], [])
            finally:
                session.close()
            self.assertIsNotNone(session.proc.poll())
            self.assertTrue(session.cleanup["started_by_this_script"])

    def test_prelaunch_failure_still_saves_report_without_starting_any_child(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            code = main(["--gpu-uuid", "GPU-12345678-1234-1234-1234-123456789abc", "--artifact-dir", folder,
                         "--exe", str(Path(folder) / "missing.exe")])
            report = json.loads((Path(folder) / "report.json").read_text(encoding="utf-8"))
            self.assertEqual((code, report["status"], report["capacity"]), (1, "error", 262144))
            self.assertNotIn("engine_pid", report)


if __name__ == "__main__":
    unittest.main()
