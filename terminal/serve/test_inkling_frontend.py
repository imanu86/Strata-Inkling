"""Inkling effort, template/tool calls and sampling over the three APIs, without a GPU or downloads.

The fixture is thinkingmachines/Inkling-2-Mini@ed40e7da625f594fc9c4990f5f66be24873c22d7/chat_template.jinja,
SHA256 53a9035f3255945704b5857657cfb00c1165090c0641c4f782a0cbf268cc16dc.
Run from terminal/: python -m unittest serve.test_inkling_frontend -v
"""
from __future__ import annotations

import json
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from serve import frontend
from serve.frontend import ChatTemplate, InklingOutputParser, forced_call
from serve.server import (ByteTokenizer, MockEngine, Service, StrataEngine, clean_shared_defaults,
                          model_card_sampling, sampling_defaults_from_config, serve)


FIXTURE = Path(__file__).with_name("test_fixtures") / "inkling2_chat_template.jinja"
MESSAGE = [{"role": "user", "content": "hello"}]
SCRIPT = "<|message_model|><|content_thinking|>think<|end_message|>" \
         "<|message_model|><|content_text|>ok<|end_message|><|content_model_end_sampling|>"
SAMPLING = {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0}


class RecordingEngine(MockEngine):
    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.sampling = dict(sampling or {})
        yield from super().generate(ids, max_new, sampling, cancel, embeddings)


class InklingFrontend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.template = ChatTemplate(FIXTURE)
        cls.tok = ByteTokenizer()
        cls.engine = RecordingEngine(cls.tok, SCRIPT, max_context=262144)
        cls.svc = Service(cls.engine, cls.tok, cls.template, model_name="Inkling-2-Mini")
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def setUp(self):
        self.svc.set_family("inkling2")
        self.svc.model_arch = "inkling2"
        self.svc.config_effort = None
        self.svc.shared = {}
        self.svc.sampling_defaults = {}
        self.svc.effort_end = False
        self.engine.scripts = [self.tok.encode(SCRIPT)]
        self.engine.script = self.engine.scripts[0]

    def tearDown(self):
        frontend.use_family(None)

    def post(self, path, body):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        self.engine.last_prompt = []
        self.engine.sampling = {}
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as error:
            with error:
                return error.code, json.load(error)

    def prompt(self, path, body):
        status, response = self.post(path, body)
        self.assertEqual(status, 200, response)
        return self.tok.decode(self.engine.last_prompt)

    def assert_effort(self, effort, expected):
        forms = (("/v1/chat/completions", {"messages": MESSAGE, "reasoning_effort": effort}),
                 ("/v1/chat/completions", {"messages": MESSAGE, "reasoning": {"effort": effort}}),
                 ("/v1/chat/completions", {"messages": MESSAGE, "chat_template_kwargs": {"reasoning_effort": effort}}),
                 ("/v1/responses", {"input": "hello", "reasoning": {"effort": effort}}),
                 ("/v1/messages", {"messages": MESSAGE, "max_tokens": 1024, "output_config": {"effort": effort}}))
        for path, body in forms:
            with self.subTest(path=path, effort=effort, body=body):
                prompt = self.prompt(path, body)
                self.assertIn(f"Thinking effort level: {expected}<|end_message|>", prompt)
                self.assertNotIn("Reasoning effort is set", prompt)

    def test_named_and_numeric_effort_reaches_each_api_unchanged(self):
        for effort, expected in (("none", "0"), ("minimal", "0.1"), ("low", "0.2"), ("medium", "0.7"),
                                 ("high", "0.9"), ("xhigh", "0.99"), ("max", "0.99"),
                                 (0, "0"), (0.35, "0.35"), (0.99, "0.99")):
            self.assert_effort(effort, expected)

    def test_template_default_is_high_point_nine(self):
        for path, body in (("/v1/chat/completions", {"messages": MESSAGE}),
                           ("/v1/responses", {"input": "hello"}),
                           ("/v1/messages", {"messages": MESSAGE, "max_tokens": 1024})):
            self.assertIn("Thinking effort level: 0.9<|end_message|>", self.prompt(path, body))

    def test_invalid_effort_is_a_400_on_each_api(self):
        for value in (True, False, -0.01, 1, float("nan"), float("inf"), float("-inf"), "huge", "0.35", [], {}):
            for path, body in (("/v1/chat/completions", {"messages": MESSAGE, "reasoning_effort": value}),
                               ("/v1/responses", {"input": "hello", "reasoning": {"effort": value}}),
                               ("/v1/messages", {"messages": MESSAGE, "max_tokens": 1024, "output_config": {"effort": value}})):
                with self.subTest(value=value, path=path):
                    self.assertEqual(self.post(path, body)[0], 400)
                    self.assertEqual(self.engine.last_prompt, [])

    def test_config_and_shared_zero_are_not_lost(self):
        forms = (("/v1/chat/completions", {"messages": MESSAGE}), ("/v1/responses", {"input": "hello"}),
                 ("/v1/messages", {"messages": MESSAGE, "max_tokens": 1024}))
        for config, shared in ((0, {}), ("high", {"reasoning_effort": 0}), ("medium", {"reasoning_effort": "high"})):
            self.svc.config_effort, self.svc.shared = config, shared
            expected = "0.9" if shared.get("reasoning_effort") == "high" else "0"
            for path, body in forms:
                with self.subTest(config=config, shared=shared, path=path):
                    self.assertIn(f"Thinking effort level: {expected}<|end_message|>", self.prompt(path, body))
        self.assert_effort("medium", "0.7")   # request still beats both default layers

    def test_anthropic_budget_maps_to_family_levels_and_disabled_thinking(self):
        for budget, expected in ((1024, "0.2"), (4096, "0.7"), (8192, "0.9")):
            prompt = self.prompt("/v1/messages", {"messages": MESSAGE, "max_tokens": 1024,
                                                "thinking": {"type": "enabled", "budget_tokens": budget}})
            self.assertIn(f"Thinking effort level: {expected}<|end_message|>", prompt)
        prompt = self.prompt("/v1/messages", {"messages": MESSAGE, "max_tokens": 1024,
                                             "thinking": {"type": "disabled"}})
        self.assertIn("Thinking effort level: 0<|end_message|>", prompt)
        prompt = self.prompt("/v1/chat/completions", {"messages": MESSAGE,
                                                    "chat_template_kwargs": {"enable_thinking": False}})
        self.assertIn("Thinking effort level: 0<|end_message|>", prompt)

    def test_inkling_does_not_get_qwen_effort_tail(self):
        self.svc.effort_end = True
        self.assert_effort("medium", "0.7")

    def test_template_generation_blocks_and_tool_capabilities(self):
        self.assertIs(self.svc.parser_cls, InklingOutputParser)
        self.assertTrue(all(self.template.caps.values()), self.template.caps)
        tools = [{"name": "weather", "description": "Get weather", "parameters": {"type": "object"}}]
        history = MESSAGE + [{"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_1", "function": {"name": "weather", "arguments": {"city": "Rome"}}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "sunny"}]
        prompt = self.template.render(history, tools=tools)
        self.assertIn('<|message_model|>weather<|content_invoke_tool_json|>{"name":"weather","args":{"city":"Rome"}}', prompt)
        self.assertIn('<|message_tool|>weather<|content_text|>sunny', prompt)
        self.assertNotIn("<tool_call>", prompt)
        parser = self.svc.parser_cls(tools=tools)
        events = parser.feed(prompt) + parser.finish()
        self.assertEqual([(e.call.name, e.call.arguments) for e in events if e.kind == "tool_call"],
                         [("weather", {"city": "Rome"})])

    def test_forced_call_uses_inkling_json_channel(self):
        tools = [{"name": "weather", "parameters": {"type": "object"}}]
        prefix = self.svc.force_text(forced_call("required", tools))
        parser = InklingOutputParser(tools=tools)
        events = parser.feed(prefix + '{"city":"Rome"}}<|end_message|>') + parser.finish()
        self.assertEqual([(e.call.name, e.call.arguments) for e in events if e.kind == "tool_call"],
                         [("weather", {"city": "Rome"})])
        self.assertNotIn("<tool_call>", prefix)

    def test_tool_call_results_over_each_api(self):
        script = '<|message_model|>weather<|content_invoke_tool_json|>{"name":"weather","args":{"city":"Rome"}}' \
                 '<|end_message|><|content_model_end_sampling|>'
        self.engine.scripts = [self.tok.encode(script)]
        self.engine.script = self.engine.scripts[0]
        function = {"name": "weather", "description": "Get weather", "parameters": {"type": "object"}}
        status, response = self.post("/v1/chat/completions", {"messages": MESSAGE,
                                     "tools": [{"type": "function", "function": function}]})
        self.assertEqual(status, 200, response)
        call = response["choices"][0]["message"]["tool_calls"][0]["function"]
        self.assertEqual((call["name"], json.loads(call["arguments"])), ("weather", {"city": "Rome"}))
        status, response = self.post("/v1/responses", {"input": "hello", "tools": [{"type": "function", **function}]})
        self.assertEqual(status, 200, response)
        call = next(item for item in response["output"] if item["type"] == "function_call")
        self.assertEqual((call["name"], json.loads(call["arguments"])), ("weather", {"city": "Rome"}))
        status, response = self.post("/v1/messages", {"messages": MESSAGE, "max_tokens": 1024,
                                     "tools": [{"name": "weather", "input_schema": {"type": "object"}}]})
        self.assertEqual(status, 200, response)
        call = next(item for item in response["content"] if item["type"] == "tool_use")
        self.assertEqual((call["name"], call["input"]), ("weather", {"city": "Rome"}))

    def test_sampling_defaults_and_override_reach_native_keys(self):
        for thinking in (True, False):
            self.assertEqual(model_card_sampling("inkling2", thinking), SAMPLING)
        self.assertEqual(sampling_defaults_from_config({"arch": "inkling2", "sampling": SAMPLING}), SAMPLING)
        for path, body in (("/v1/chat/completions", {"messages": MESSAGE, "reasoning_effort": "high"}),
                           ("/v1/chat/completions", {"messages": MESSAGE, "reasoning_effort": "none"}),
                           ("/v1/responses", {"input": "hello"}),
                           ("/v1/messages", {"messages": MESSAGE, "max_tokens": 1024})):
            self.prompt(path, body)
            for key, value in SAMPLING.items():
                self.assertEqual(self.engine.sampling[key], value)
            keys = StrataEngine.sampling_keys(self.engine.sampling)
            self.assertIn("temperature=1.0", keys)
            self.assertIn("top_k=0", keys)
            self.assertNotIn("_strata_arch", keys)
        self.svc.sampling_defaults = {"temperature": 0.6, "top_k": 16}
        self.svc.shared = {"temperature": 0.8}
        self.prompt("/v1/chat/completions", {"messages": MESSAGE, "temperature": 0.4, "top_p": 0.7, "top_k": 40,
                                            "min_p": 0.1, "_strata_arch": "qwen4exp"})
        keys = StrataEngine.sampling_keys(self.engine.sampling)
        for expected in ("temperature=0.4", "top_p=0.7", "top_k=40", "min_p=0.1"):
            self.assertIn(expected, keys)
        self.assertEqual(self.engine.sampling["_strata_arch"], "inkling2")

    def test_shared_effort_validation_and_qwen_regression(self):
        for value in (0, 0.35, "minimal", "high", "max"):
            self.assertEqual(clean_shared_defaults({"reasoning_effort": value}), {"reasoning_effort": value})
        for value in (True, False, float("nan"), -1, 1):
            with self.assertRaises(ValueError):
                clean_shared_defaults({"reasoning_effort": value})
        self.assertEqual(clean_shared_defaults({"top_k": 0}), {"top_k": 0})
        for family in (None, "qwen4exp", "qwen35moe"):
            self.svc.set_family(family)
            self.assertEqual(frontend.effort_kwargs("high"), {"reasoning_effort": "xhigh"})
            self.assertEqual(frontend.effort_kwargs("minimal"), {"enable_thinking": False})
            self.assertEqual(frontend.effort_kwargs(False), {"enable_thinking": False})
            with self.assertRaises(ValueError):
                frontend.effort_kwargs(0.35)
            with self.assertRaises(ValueError):
                clean_shared_defaults({"top_k": 0})
            with self.assertRaises(SystemExit):
                sampling_defaults_from_config({"arch": family, "sampling": {"top_k": 0}})
            self.assertEqual(frontend.anthropic_to_messages({"messages": MESSAGE,
                                                            "output_config": {"effort": False}})[2], {})
        self.assertIn("top_k=64", StrataEngine.sampling_keys({"top_k": 0}))
        self.svc.model_arch = "qwen4exp"
        self.prompt("/v1/chat/completions", {"messages": MESSAGE, "top_k": 0, "_strata_arch": "inkling2"})
        self.assertNotIn("_strata_arch", self.engine.sampling)
        self.assertIn("top_k=64", StrataEngine.sampling_keys(self.engine.sampling))


if __name__ == "__main__":
    unittest.main()
