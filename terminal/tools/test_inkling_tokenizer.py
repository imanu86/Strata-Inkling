"""Inkling GGUF/pack token IDs against the official local Hugging Face Rust tokenizer.

No downloads, weight reads, engine, or GPU. Set STRATA_INKLING_GGUF to a model GGUF and
STRATA_INKLING_HF_TOKENIZER to its official tokenizer directory, then run from terminal/:
  python -m unittest tools.test_inkling_tokenizer -v
The GGUF reader parses metadata/tensor descriptors only. Other families keep their existing rules.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import random
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
import strata_tokenizer as ST


class InklingRule(unittest.TestCase):
    def test_family_is_distinct_and_other_names_are_unchanged(self):
        for pre, expected in (("inkling2", "inkling2"), (None, "qwen35"), ("", "qwen35"),
                              ("qwen35", "qwen35"), ("qwen2", "qwen2"), ("deepseek-v3", "deepseek3")):
            self.assertEqual(ST.pre_tokenizer_kind(pre), expected)
        tokens = list(ST.BYTE_TO_UNICODE.values())
        tokenizer = ST.Tokenizer(tokens, [], pre="inkling2")
        self.assertEqual(tokenizer._split("a1234567"), ["a", "123", "456", "7"])
        self.assertEqual(tokenizer.decode(tokenizer.encode("città\r\n日本語🙂")), "città\r\n日本語🙂")


class InklingOracle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model = os.environ.get("STRATA_INKLING_GGUF")
        directory = os.environ.get("STRATA_INKLING_HF_TOKENIZER")
        if not model or not directory:
            raise unittest.SkipTest("set STRATA_INKLING_GGUF and STRATA_INKLING_HF_TOKENIZER for the local HF oracle")
        try:
            from tokenizers import Tokenizer
        except ImportError:
            raise unittest.SkipTest("the optional Hugging Face tokenizers package is not installed")
        cls.model, cls.directory = Path(model), Path(directory)
        cls.source = json.loads((cls.directory / "tokenizer.json").read_text(encoding="utf-8"))
        cls.hf = Tokenizer.from_file(str(cls.directory / "tokenizer.json"))
        cls.actual = ST.Tokenizer.from_gguf(cls.model)
        # Disabling added-token matching provides an independent oracle for protected literal text.
        plain_source = {**cls.source, "added_tokens": []}
        cls.hf_plain = Tokenizer.from_str(json.dumps(plain_source, ensure_ascii=False))
        cls.corpus = ["", "Qual è la capitale d'Italia? Rispondi soltanto: Roma.",
                      "L'ITALIA, È bello, città e perché; DON'T I'm we're THEY'LL she'd foo'ſ.",
                      "CamelCase snake_case XMLHTTPReader mixedCASE_and-dashes 1234567890 00000123456789",
                      "\r\n\r\n  righe\r\nlinea\n / //\r\n\t  \n fine  ",
                      "日本語 カタカナ 中文 한글 Привет العربية हिन्दी Ελληνικά",
                      "café e\u0301 \u0301\u0301 👩🏽‍💻🙂🚀🇮🇹 \x00\x01\x7f \u00a0\u2028\u2029",
                      '{"name":"weather","args":{"city":"Roma","n":1234567,"ok":true,"quote":"a\\n\\\"b"}}',
                      "a" * 5000, "中文" * 100, "\t \r\n" * 60]
        rng = random.Random(20261121)
        words = ["Roma", " città", "Perché", "L'Italia", "DON'T", "CamelCase", "1234567890", "000001", "12.345",
                 "中文", "日本語", "e\u0301", "🙂", "👩🏽‍💻", "\r\n", "\t", "   ", "/\r\n", "{\"args\":", "}", "ßſİı", "\u00a0"]
        cls.corpus += ["".join(rng.choice(words) for _ in range(rng.randrange(1, 35))) for _ in range(500)]

    def test_pattern_is_exactly_the_official_split_rule(self):
        split, bytelevel = self.source["pre_tokenizer"]["pretokenizers"]
        self.assertEqual(ST.INKLING2_PATTERN, split["pattern"]["Regex"])
        self.assertEqual((split["behavior"], split["invert"], bytelevel["use_regex"], bytelevel["add_prefix_space"]),
                         ("Isolated", False, False, False))
        for text in self.corpus:
            got = ["".join(ST.BYTE_TO_UNICODE[b] for b in part.encode("utf-8")) for part in self.actual._split(text)]
            self.assertEqual(got, [part for part, _ in self.hf.pre_tokenizer.pre_tokenize_str(text)], repr(text))

    def test_vocab_and_every_control_id_match_hugging_face(self):
        self.assertEqual(self.actual.pre, "inkling2")
        self.assertEqual(self.actual.ids, self.hf.get_vocab())
        for token in self.source["added_tokens"]:
            self.assertTrue(token["special"])
            self.assertEqual(self.actual.special_tokens[token["content"]], token["id"])
            self.assertEqual(self.actual.encode(token["content"], parse_special=True), [token["id"]])
            self.assertEqual(self.actual.decode([token["id"]]), token["content"])
        self.assertEqual(self.actual.special_ids["tokenizer.ggml.eos_token_id"], 200006)
        self.assertEqual(self.hf.token_to_id("<|message_user|>"), 200000)
        self.assertEqual(self.hf.token_to_id("<|message_model|>"), 200001)
        self.assertEqual(self.hf.token_to_id("<|endoftext|>"), 199999)

    def test_encode_decode_corpus_matches_hugging_face_exactly(self):
        for text in self.corpus:
            expected = self.hf.encode(text, add_special_tokens=False).ids
            actual = self.actual.encode(text, parse_special=True)
            self.assertEqual(actual, expected, repr(text))
            self.assertEqual(self.actual.decode(actual), text)
            self.assertEqual(self.actual.decode(expected), self.hf.decode(expected, skip_special_tokens=False))

    def test_literal_specials_are_plain_when_protected(self):
        for text in ("Cita <|message_user|> e <|end_message|> alla lettera.",
                     '<|message_model|>weather<|content_invoke_tool_json|>{"args":{}}<|end_message|>',
                     "Prima <|content_thinking|> ragiona <|content_model_end_sampling|> dopo."):
            expected = self.hf_plain.encode(text, add_special_tokens=False).ids
            self.assertEqual(self.actual.encode(text, parse_special=False), expected)
            self.assertEqual(self.actual.encode(text, parse_special=True, plain=[(0, len(text))]), expected)
            self.assertEqual(self.actual.decode(expected), text)
            prefix, suffix = "<|message_user|>", "<|end_message|>"
            wrapped = prefix + text + suffix
            expected_wrapped = [self.hf.token_to_id(prefix)] + expected + [self.hf.token_to_id(suffix)]
            self.assertEqual(self.actual.encode(wrapped, parse_special=True, plain=[(len(prefix), len(prefix) + len(text))]),
                             expected_wrapped)
            self.assertEqual(self.actual.encode(wrapped, parse_special=True), self.hf.encode(wrapped, add_special_tokens=False).ids)

    def test_official_chat_template_prompt_ids_match_hugging_face(self):
        from serve.frontend import ChatTemplate
        template = ChatTemplate(self.directory / "chat_template.jinja")
        messages = [{"role": "user", "content": "Qual è la capitale d'Italia?"},
                    {"role": "assistant", "content": "", "tool_calls": [{"id": "t1", "function": {
                        "name": "lookup", "arguments": {"city": "Roma", "count": 123456}}}]},
                    {"role": "tool", "tool_call_id": "t1", "content": "Roma è la capitale."},
                    {"role": "user", "content": "Ripeti il nome."}]
        for effort in ("none", "medium", "high", 0.35):
            prompt = template.render(messages, tools=[{"name": "lookup", "parameters": {"type": "object"}}], reasoning_effort=effort)
            expected = self.hf.encode(prompt, add_special_tokens=False).ids
            self.assertEqual(self.actual.encode(prompt, parse_special=True), expected)
            self.assertEqual(self.actual.decode(expected), prompt)

    def test_exported_pack_keeps_pattern_and_hf_parity(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg = ST.extract(self.model, folder)
            self.assertEqual(cfg["pre_pattern"], ST.INKLING2_PATTERN)
            self.assertEqual(cfg["pre_pattern_source"], ST.INKLING2_PATTERN_SOURCE)
            directory = Path(folder) / "tokenizer"
            self.assertEqual((directory / "chat_template.jinja").read_text(encoding="utf-8"),
                             (self.directory / "chat_template.jinja").read_text(encoding="utf-8"))
            vocab = json.loads((directory / "vocab.json").read_text(encoding="utf-8"))
            tokens = [None] * len(vocab)
            for token, index in vocab.items():
                tokens[index] = token
            pack = ST.Tokenizer(tokens, (directory / "merges.txt").read_text(encoding="utf-8").splitlines(),
                                json.loads((directory / "token_type.json").read_text(encoding="utf-8")),
                                cfg["pre"], cfg["special_ids"])
            for text in self.corpus[:11]:
                self.assertEqual(pack.encode(text, parse_special=True), self.hf.encode(text, add_special_tokens=False).ids)


if __name__ == "__main__":
    unittest.main()
