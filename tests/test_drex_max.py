"""Drex-Max pieces that run on CPU without the 49 GB checkpoint."""
import os
import sys
import unittest
from pathlib import Path

import torch
from transformers import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drex_max.model import _context_limit, _bidirectional_full_attention, _prompt, answer, options


def tiny_text_model():
    cfg = Qwen3_5TextConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=8, linear_value_head_dim=8,
        max_position_embeddings=128, attn_implementation="sdpa",
    )
    torch.manual_seed(0)
    return Qwen3_5TextModel(cfg).eval()


def first_token_shift(model):
    a = torch.tensor([[1, 2, 3, 4, 5, 6]])
    b = a.clone()
    b[0, -1] = 9
    with torch.no_grad():
        ha = model(input_ids=a, use_cache=False).last_hidden_state
        hb = model(input_ids=b, use_cache=False).last_hidden_state
    return (ha[0, 0] - hb[0, 0]).abs().max().item()


class DrexMaxTests(unittest.TestCase):
    def test_full_attention_is_non_causal_after_hook(self):
        self.assertEqual(first_token_shift(tiny_text_model()), 0.0)
        model = tiny_text_model()
        _bidirectional_full_attention(model)
        self.assertGreater(first_token_shift(model), 1e-4)

    def test_hook_rejects_kv_cache(self):
        model = tiny_text_model()
        _bidirectional_full_attention(model)
        with self.assertRaisesRegex(ValueError, "KV cache"):
            model(input_ids=torch.tensor([[1, 2, 3]]), use_cache=True)

    def test_answers(self):
        choice = {"type": "choice", "criteria": {"a": "x", "b": None, "c": "z"}}
        out = answer(choice, [0.2, 0.7, 0.1])
        self.assertEqual(out["choice"], "b")
        self.assertAlmostEqual(sum(out["probabilities"].values()), 1.0)
        self.assertAlmostEqual(answer({"type": "noul"}, [0.25, 0.75])["noul"], 0.75)
        score = answer({"type": "score", "criteria": ["low", "high"]}, [0.5, 0.5])
        self.assertAlmostEqual(score["score"], 0.5)
        with self.assertRaises(ValueError):
            answer(choice, [0.5, 0.5])
        with self.assertRaises(ValueError):
            answer(choice, [float("nan"), 0.5, 0.5])

    def test_prompt_labels_options_and_counts_images(self):
        row = {"state": {"ticket": "hi"}, "images": ["a.png", "b.png"],
               "question": {"type": "choice", "instructions": "Pick.", "criteria": {"x": "first", "y": None}}}
        system, user = _prompt(row, ["A", "B"])
        self.assertEqual([part["type"] for part in user["content"]], ["image", "image", "text"])
        text = user["content"][-1]["text"]
        self.assertIn("A: x: first", text)
        self.assertIn("B: y", text)
        with self.assertRaises(ValueError):
            _prompt(row, ["A"])
        self.assertEqual(options({"type": "noul"})[0], ["false", "true"])

    def test_context_limit(self):
        old = os.environ.pop("DREX_MAX_CONTEXT", None)
        try:
            self.assertEqual(_context_limit(), 16384)
            for value, expected in (("65536", 65536), ("262144", 262144)):
                os.environ["DREX_MAX_CONTEXT"] = value
                self.assertEqual(_context_limit(), expected)
            for bad in ("0", "262145", "x"):
                os.environ["DREX_MAX_CONTEXT"] = bad
                with self.assertRaises(ValueError):
                    _context_limit()
        finally:
            os.environ.pop("DREX_MAX_CONTEXT", None)
            if old is not None:
                os.environ["DREX_MAX_CONTEXT"] = old


if __name__ == "__main__":
    unittest.main()
