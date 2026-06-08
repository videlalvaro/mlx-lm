import json
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_lm.aion_onnx_convert import (
    QuantSpec,
    build_quantization_config,
    copy_tokenizer_files,
    pack_aion_quantized,
)
from mlx_lm.chat_templates.aion import apply_chat_template
from mlx_lm.models.aion_onnx_mlx import Model, ModelArgs


class TestAionOnnxConvert(unittest.TestCase):
    def test_pack_aion_4bit_quantized_weight_for_mlx(self):
        q = np.arange(16, dtype=np.uint8).reshape(2, 8)
        packed = np.zeros((2, 2, 2), dtype=np.uint8)
        for row in range(2):
            for block in range(2):
                values = q[row, block * 4 : (block + 1) * 4]
                packed[row, block, 0] = values[0] | (values[1] << 4)
                packed[row, block, 1] = values[2] | (values[3] << 4)
        scales = np.array([[0.5, 0.25], [0.125, 0.0625]], dtype=np.float16)
        spec = QuantSpec(bits=4, block_size=4, k=8, n=2)

        qweight, qscales = pack_aion_quantized(packed, scales, spec)

        expected_words = np.array(
            [
                sum(int(q[0, offset]) << (4 * offset) for offset in range(8)),
                sum(int(q[1, offset]) << (4 * offset) for offset in range(8)),
            ],
            dtype=np.uint32,
        ).reshape(2, 1)
        np.testing.assert_array_equal(qweight, expected_words)
        np.testing.assert_array_equal(qscales, scales)

    def test_pack_aion_8bit_quantized_weight_for_mlx(self):
        packed = np.array([[[1, 2, 3, 4]]], dtype=np.uint8)
        scales = np.array([[0.5]], dtype=np.float16)
        spec = QuantSpec(bits=8, block_size=4, k=4, n=1)

        qweight, qscales = pack_aion_quantized(packed, scales, spec)

        expected = np.array([[1 | (2 << 8) | (3 << 16) | (4 << 24)]], dtype=np.uint32)
        np.testing.assert_array_equal(qweight, expected)
        np.testing.assert_array_equal(qscales, scales)

    def test_build_quantization_config_uses_common_default_and_overrides(self):
        config = build_quantization_config(
            {
                "model.layers.0.self_attn.q_proj": (4, 32),
                "model.layers.0.self_attn.k_proj": (4, 32),
                "lm_head": (8, 32),
            }
        )

        self.assertEqual(config["bits"], 4)
        self.assertEqual(config["group_size"], 32)
        self.assertEqual(config["mode"], "affine")
        self.assertEqual(config["lm_head"], {"bits": 8, "group_size": 32, "mode": "affine"})

    def test_copy_tokenizer_files_sets_mistral_regex_flag(self):
        with tempfile.TemporaryDirectory() as source_dir, tempfile.TemporaryDirectory() as out_dir:
            source = Path(source_dir)
            output = Path(out_dir)
            (source / "tokenizer.json").write_text("{}", encoding="utf-8")
            (source / "tokenizer_config.json").write_text(json.dumps({"tokenizer_file": "tokenizer.json"}), encoding="utf-8")

            copy_tokenizer_files(source, output)

            tokenizer_config = json.loads((output / "tokenizer_config.json").read_text(encoding="utf-8"))
            self.assertTrue(tokenizer_config["fix_mistral_regex"])
            self.assertEqual(tokenizer_config["chat_template_type"], "aion")
            self.assertTrue((output / "tokenizer.json").exists())

    def test_aion_chat_template_uses_native_role_tokens(self):
        rendered = apply_chat_template(
            [{"role": "user", "content": "Who made you?"}],
            add_generation_prompt=True,
        )

        self.assertEqual(rendered, "<|user|>\nWho made you?<|end|>\n<|assistant|>\n")

    def test_sanitize_computes_4bit_biases_from_scales(self):
        """Biases are not stored on disk; sanitize must reconstruct them from scales."""
        args = ModelArgs(
            model_type="aion_onnx_mlx",
            hidden_size=64,
            num_hidden_layers=1,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            vocab_size=32,
            max_position_embeddings=16,
            quantization={"group_size": 4, "bits": 4, "mode": "affine"},
        )
        model = Model(args)
        scales = mx.array([[0.5, 0.25], [0.125, 0.0625]], dtype=mx.float16)
        weights = {
            "model.layers.0.self_attn.q_proj.weight": mx.zeros((8, 2), dtype=mx.uint32),
            "model.layers.0.self_attn.q_proj.scales": scales,
            "model.cos_cache": mx.zeros((16, 8)),
            "model.sin_cache": mx.zeros((16, 8)),
        }

        sanitized = model.sanitize(weights)

        bias_key = "model.layers.0.self_attn.q_proj.biases"
        self.assertIn(bias_key, sanitized)
        # 4-bit zero_point = 8
        expected_biases = (-8.0 * np.array(scales, dtype=np.float32)).astype(np.float16)
        np.testing.assert_array_equal(np.array(sanitized[bias_key]), expected_biases)

    def test_sanitize_computes_8bit_biases_from_scales(self):
        """Per-layer quantization overrides are respected when computing biases."""
        args = ModelArgs(
            model_type="aion_onnx_mlx",
            hidden_size=64,
            num_hidden_layers=1,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            vocab_size=32,
            max_position_embeddings=16,
            quantization={
                "group_size": 4,
                "bits": 4,
                "mode": "affine",
                "model.embed_tokens": {"group_size": 4, "bits": 8, "mode": "affine"},
            },
        )
        model = Model(args)
        scales = mx.array([[0.5, 0.25]], dtype=mx.float16)
        weights = {
            "model.embed_tokens.weight": mx.zeros((1, 2), dtype=mx.uint32),
            "model.embed_tokens.scales": scales,
            "model.cos_cache": mx.zeros((16, 8)),
            "model.sin_cache": mx.zeros((16, 8)),
        }

        sanitized = model.sanitize(weights)

        bias_key = "model.embed_tokens.biases"
        self.assertIn(bias_key, sanitized)
        # 8-bit zero_point = 128
        expected_biases = (-128.0 * np.array(scales, dtype=np.float32)).astype(np.float16)
        np.testing.assert_array_equal(np.array(sanitized[bias_key]), expected_biases)

    def test_sanitize_preserves_existing_biases(self):
        """If biases are already in the weights (old format), sanitize does not overwrite."""
        args = ModelArgs(
            model_type="aion_onnx_mlx",
            hidden_size=64,
            num_hidden_layers=1,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            vocab_size=32,
            max_position_embeddings=16,
            quantization={"group_size": 4, "bits": 4, "mode": "affine"},
        )
        model = Model(args)
        existing_biases = mx.array([[1.0, 2.0]], dtype=mx.float16)
        weights = {
            "model.layers.0.self_attn.q_proj.weight": mx.zeros((1, 2), dtype=mx.uint32),
            "model.layers.0.self_attn.q_proj.scales": mx.array([[0.5, 0.25]], dtype=mx.float16),
            "model.layers.0.self_attn.q_proj.biases": existing_biases,
            "model.cos_cache": mx.zeros((16, 8)),
            "model.sin_cache": mx.zeros((16, 8)),
        }

        sanitized = model.sanitize(weights)

        np.testing.assert_array_equal(
            np.array(sanitized["model.layers.0.self_attn.q_proj.biases"]),
            np.array(existing_biases),
        )


if __name__ == "__main__":
    unittest.main()
