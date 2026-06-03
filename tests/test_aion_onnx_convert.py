import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mlx_lm.aion_onnx_convert import (
    QuantSpec,
    build_quantization_config,
    copy_tokenizer_files,
    pack_aion_quantized,
)
from mlx_lm.chat_templates.aion import apply_chat_template


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

        qweight, qscales, qbiases = pack_aion_quantized(packed, scales, spec)

        expected_words = np.array(
            [
                sum(int(q[0, offset]) << (4 * offset) for offset in range(8)),
                sum(int(q[1, offset]) << (4 * offset) for offset in range(8)),
            ],
            dtype=np.uint32,
        ).reshape(2, 1)
        np.testing.assert_array_equal(qweight, expected_words)
        np.testing.assert_array_equal(qscales, scales)
        np.testing.assert_array_equal(qbiases, (-8.0 * scales.astype(np.float32)).astype(np.float16))

    def test_pack_aion_8bit_quantized_weight_for_mlx(self):
        packed = np.array([[[1, 2, 3, 4]]], dtype=np.uint8)
        scales = np.array([[0.5]], dtype=np.float16)
        spec = QuantSpec(bits=8, block_size=4, k=4, n=1)

        qweight, qscales, qbiases = pack_aion_quantized(packed, scales, spec)

        expected = np.array([[1 | (2 << 8) | (3 << 16) | (4 << 24)]], dtype=np.uint32)
        np.testing.assert_array_equal(qweight, expected)
        np.testing.assert_array_equal(qscales, scales)
        np.testing.assert_array_equal(qbiases, np.array([[-64.0]], dtype=np.float16))

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


if __name__ == "__main__":
    unittest.main()
