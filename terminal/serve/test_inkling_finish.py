"""Inkling DONE stop propagation, including a native iterator that consumes EOS itself (CPU only)."""
import json
from pathlib import Path
import subprocess
import sys
import threading
import unittest
from unittest import mock
import urllib.request

from serve import frontend
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, EngineDied, MockEngine, Service, StrataEngine, serve


TEMPLATE = Path(__file__).with_name("test_fixtures") / "inkling2_chat_template.jinja"
SCRIPT = "<|message_model|><|content_text|>Roma<|end_message|>"


class DoneEngine(MockEngine):
    """Replay content without an EOS token; parse a real DONE record on exhaustion/close."""
    finish = "stop"
    update_done = True
    cancel_at_end = False
    error_at_end = False
    eos_token = None

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.last_prompt = list(ids)
        emitted = 0
        try:
            for token in self.tok.encode(SCRIPT)[:max_new]:
                emitted += 1
                yield token
            if self.eos_token is not None:
                emitted += 1
                yield self.eos_token
            if self.cancel_at_end:
                cancel.set()
            if self.error_at_end:
                raise EngineDied("fake native engine died")
        finally:
            if self.update_done:
                StrataEngine._parse_done(self, f"DONE {emitted} {len(ids)} 12.0 34.0 {self.finish}")


class InklingFinish(unittest.TestCase):
    def setUp(self):
        self.tok = ByteTokenizer()
        self.engine = DoneEngine(self.tok, "", max_context=262144)
        self.svc = Service(self.engine, self.tok, ChatTemplate(TEMPLATE))
        self.svc.set_family("inkling2")
        self.svc.model_arch = "inkling2"
        self.addCleanup(frontend.use_family, None)

    def run_service(self, cancel=None):
        return list(self.svc.run(self.tok.encode("prompt"), False, None, 256, {}, cancel or threading.Event()))

    def test_pipe_budget_reply_reaches_waiting_engine(self):
        script = """
import sys
print('READY 4096 stop budget_splice', flush=True)
assert 'budget_check=2' in input().split()
print('T 17\\nT 18\\nBUDGET_WAIT', flush=True)
assert input().split() == ['BUDGET_RESUME', '0', '20,21']
print('T 20\\nT 21\\nT 22\\nDONE 5 1 0 0 stop', flush=True)
"""
        with mock.patch("serve.server.popen", side_effect=lambda what, args, **kw:
                        subprocess.Popen([sys.executable, "-u", "-c", script], **kw)):
            engine = StrataEngine("test-budget-pipe", [])
        self.addCleanup(engine.close)
        self.assertTrue(engine.can_budget_splice)
        engine.silence_s = 2
        tokens = engine.generate([1], 5, {}, threading.Event(), budget_boundary=(2, lambda: (0, [20, 21])))
        self.assertEqual(list(tokens), [17, 18, 20, 21, 22])

    def test_budget_splices_native_channels_in_one_generation(self):
        self.engine.can_budget_splice = True
        for tools, force, after_header, expected_header in (
            (None, None, "Roma", "<|message_model|><|content_text|>"),
            ([{"name": "lookup"}], None, "<|content_text|>Roma", "<|message_model|>"),
            ([{"name": "lookup"}], frontend.inkling_force_text("<tool_call>\n<function=lookup>"),
             '{}}', '<|message_model|>lookup<|content_invoke_tool_json|>{"name":"lookup","args":'),
        ):
            with self.subTest(header=expected_header):
                splices = []
                passes = []
                def generate(ids, max_new, sampling, cancel, budget_boundary=None):
                    passes.append(1)
                    at, decide = budget_boundary
                    stream = iter(self.tok.encode("<|message_model|><|content_thinking|>" + "abc" * 100))
                    for pos in range(max_new):
                        if pos == at:
                            more, tokens = decide()
                            at = pos + more if more else None
                            if tokens:
                                splices.append(self.tok.decode(tokens))
                                stream = iter(list(tokens) + self.tok.encode(after_header + "<|end_message|>"))
                        token = next(stream, None)
                        if token is None:
                            return
                        yield token
                self.engine.generate = generate
                events = list(self.svc.run(self.tok.encode("prompt"), True, tools, 512,
                                           {"reasoning_budget_tokens": 64}, threading.Event(), force=force))
                self.assertEqual(passes, [1])
                self.assertEqual(splices, ["<|end_message|>" + expected_header])
                if force is None:
                    self.assertEqual("".join(e.text for k, e in events if k == "event" and e.kind == "content"), "Roma")
                else:
                    self.assertTrue(any(k == "event" and e.kind == "tool_call" for k, e in events))

    def test_native_stop_without_eos_is_a_stop_for_service_and_history(self):
        for native_finish in ("stop", "eos"):
            self.engine.finish = native_finish
            result = self.run_service()
            done = next(value for kind, value in result if kind == "done")
            self.assertEqual(done["finish"], "stop")
            self.assertLess(done["completion_tokens"], 256)
            self.assertEqual(self.svc.history[-1]["finish"], "stop")

    def test_streamed_and_whole_chat_report_stop_without_an_eos_token(self):
        httpd = serve(self.svc, port=0)
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        url = f"http://127.0.0.1:{httpd.server_address[1]}/v1/chat/completions"
        for stream in (False, True):
            body = {"messages": [{"role": "user", "content": "Qual è la capitale d'Italia?"}],
                    "max_tokens": 256, "reasoning_effort": "none", "stream": stream}
            request = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=5) as response:
                raw = response.read().decode()
            if stream:
                chunks = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: {")]
                self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")
            else:
                choice = json.loads(raw)["choices"][0]
                self.assertEqual(choice["message"]["content"], "Roma")
                self.assertEqual(choice["finish_reason"], "stop")

    def test_complete_gguf_metadata_keys_supply_inkling_stop_ids(self):
        self.tok.special_ids = {"tokenizer.ggml.eos_token_id": 200006, "tokenizer.ggml.eot_token_id": 200006}
        self.svc.set_family("inkling2")
        self.assertIn(200006, self.svc.stop_ids)
        self.engine.eos_token = 200006
        result = self.run_service()
        done = next(value for kind, value in result if kind == "done")
        self.assertEqual(done["finish"], "stop")
        self.assertEqual("".join(value.text or "" for kind, value in result if kind == "event" and value.kind == "content"), "Roma")

    def test_length_and_stale_stop_are_not_upgraded(self):
        self.engine.finish = "length"
        done = next(value for kind, value in self.run_service() if kind == "done")
        self.assertEqual(done["finish"], "length")
        self.engine.last = {"finish": "stop", "generated": 1}
        self.engine.update_done = False
        done = next(value for kind, value in self.run_service() if kind == "done")
        self.assertEqual(done["finish"], "length")

    def test_early_close_cancel_and_error_are_not_changed_to_stop(self):
        self.svc.repeat_stop_tokens = 1  # client-side stop closes the iterator; DONE stop must not override length
        done = next(value for kind, value in self.run_service() if kind == "done")
        self.assertEqual(done["finish"], "length")
        self.svc.repeat_stop_tokens = 0
        self.engine.cancel_at_end = True
        done = next(value for kind, value in self.run_service() if kind == "done")
        self.assertEqual(done["finish"], "cancel")
        self.engine.cancel_at_end = False
        self.engine.error_at_end = True
        with self.assertRaises(EngineDied):
            self.run_service()

    def test_qwen_iterator_exhaustion_retains_its_previous_behavior(self):
        self.svc.set_family(None)
        self.svc.model_arch = "qwen4exp"
        done = next(value for kind, value in self.run_service() if kind == "done")
        self.assertEqual(done["finish"], "length")


if __name__ == "__main__":
    unittest.main()
