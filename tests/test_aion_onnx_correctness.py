import os
import shutil
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx
import numpy as np
from tokenizers import Tokenizer

from mlx_lm.aion_onnx_convert import convert
from mlx_lm.models import cache as cache_lib
from mlx_lm.utils import load

try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    import onnx
except ImportError:
    onnx = None


AION_ONNX_BUNDLE = os.getenv("AION_ONNX_BUNDLE")


def make_exact_reference_onnx(bundle: Path, out_dir: Path) -> Path:
    model = onnx.load(str(bundle / "model.onnx"), load_external_data=False)
    patched = 0
    for node in model.graph.node:
        if node.op_type != "MatMulNBits":
            continue
        for attr in node.attribute:
            if attr.name == "accuracy_level":
                attr.CopyFrom(onnx.helper.make_attribute("accuracy_level", 1))
                patched += 1
                break
    if patched == 0:
        raise AssertionError("expected at least one MatMulNBits accuracy_level attribute")

    patched_path = out_dir / "model.accuracy_level_1.onnx"
    onnx.save_model(model, str(patched_path))

    external_data = bundle / "model.onnx.data"
    if external_data.exists():
        linked_external_data = out_dir / "model.onnx.data"
        try:
            os.link(external_data, linked_external_data)
        except OSError:
            shutil.copy2(external_data, linked_external_data)
    return patched_path


@unittest.skipUnless(AION_ONNX_BUNDLE, "set AION_ONNX_BUNDLE to run Aion ONNX parity tests")
@unittest.skipUnless(ort is not None, "onnxruntime is required for Aion ONNX parity tests")
@unittest.skipUnless(onnx is not None, "onnx is required to build the exact Aion ONNX reference")
class TestAionOnnxCorrectness(unittest.TestCase):
    maxDiff = None

    prompt_cases = [
        ("single_token", "Hello"),
        ("single_token", "."),
        ("single_token", ":"),
        ("short_chat", "Hello, who are you?"),
        ("short_chat", "Can you help me write a note?"),
        ("completion", "The capital of France is"),
        ("completion", "A list of colors: red, green, blue,"),
        ("completion", "Once upon a time,"),
        ("instruction", "Translate to Spanish: The weather is nice today."),
        ("instruction", "Summarize this sentence in three words: Apple silicon runs neural networks efficiently."),
        ("instruction", "Rewrite politely: send me the report now."),
        ("math", "Question: What is 17 * 23?\nAnswer:"),
        ("math", "Compute 144 / 12 ="),
        ("math", "The sequence is 2, 4, 8, 16,"),
        ("code", "def fibonacci(n):\n    if n <= 1:"),
        ("code", "for item in items:\n    print("),
        ("code", "SELECT name FROM users WHERE"),
        ("json", '{"name": "Aion", "type":'),
        ("markdown", "# Project Notes\n\n- First item\n-"),
        ("punctuation", "Wait... what happened?!"),
        ("punctuation", "alpha,beta,gamma,"),
        ("whitespace", "Leading spaces:     value"),
        ("whitespace", "Line one\nLine two\nLine three"),
        ("multilingual", "日本語で短く自己紹介してください。"),
        ("multilingual", "Explique en francais pourquoi le ciel est bleu."),
        ("multilingual", "Traducir al ingles: buenos dias a todos."),
        ("symbols", "Path: /workspace/project/file.txt"),
        ("symbols", "Email test@example.com about version 2.0."),
        ("retrieval", "Context: Paris is in France. Madrid is in Spain. Question: Paris is in"),
        ("retrieval", "Facts: red=apple, yellow=banana, green=lime. The yellow fruit is"),
        ("dialogue", "User: I lost my keys.\nAssistant:"),
        ("dialogue", "System: Be concise.\nUser: Explain gravity.\nAssistant:"),
        ("long_context", "In a tiny village near the mountains, the old clock tower rang at dawn. " * 12 + "Then"),
        ("long_context", "The experiment recorded temperature, pressure, and voltage every second. " * 10 + "The next reading"),
        ("long_context", "A careful engineer checked the logs, compared the traces, and reran the benchmark. " * 8 + "Finally"),
    ]

    @classmethod
    def setUpClass(cls):
        cls.bundle = Path(AION_ONNX_BUNDLE)
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.model_path = Path(cls.tmpdir.name) / "aion-mlx"
        convert(cls.bundle, cls.model_path, max_seq_len=4096)
        cls.tokenizer = Tokenizer.from_file(str(cls.bundle / "tokenizer.json"))
        cls.exact_onnx_path = make_exact_reference_onnx(cls.bundle, Path(cls.tmpdir.name))
        cls.onnx = ort.InferenceSession(str(cls.exact_onnx_path), providers=["CPUExecutionProvider"])
        cls.model, _tokenizer, cls.config = load(
            str(cls.model_path),
            return_config=True,
            tokenizer_config={"trust_remote_code": True},
        )
        mx.eval(cls.model.parameters())
        cls.num_layers = int(cls.config["num_hidden_layers"])
        cls.num_kv_heads = int(cls.config["num_key_value_heads"])
        cls.head_dim = int(cls.config["head_dim"])

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def onnx_incremental_logits(self, token_ids):
        past = {
            layer: (
                np.zeros((1, self.num_kv_heads, 0, self.head_dim), dtype=np.float16),
                np.zeros((1, self.num_kv_heads, 0, self.head_dim), dtype=np.float16),
            )
            for layer in range(self.num_layers)
        }
        logits = None
        for position, token_id in enumerate(token_ids):
            inputs = {
                "input_ids": np.array([[token_id]], dtype=np.int64),
                "attention_mask": np.ones((1, position + 1), dtype=np.int64),
            }
            for layer in range(self.num_layers):
                inputs[f"past_key_values.{layer}.key"] = past[layer][0]
                inputs[f"past_key_values.{layer}.value"] = past[layer][1]
            outputs = self.onnx.run(None, inputs)
            logits = outputs[0][0, -1].astype(np.float32)
            for layer in range(self.num_layers):
                past[layer] = (outputs[1 + 2 * layer], outputs[2 + 2 * layer])
        return logits

    def mlx_cached_logits(self, token_ids):
        prompt_cache = cache_lib.make_prompt_cache(self.model)
        logits = None
        for token_id in token_ids:
            logits = self.model(mx.array([[token_id]], dtype=mx.int32), cache=prompt_cache)[:, -1, :]
            mx.eval(logits)
        return np.array(logits).reshape(-1).astype(np.float32)

    def test_next_token_logits_match_onnx_for_prompt_suite(self):
        failures = []
        for category, prompt in self.prompt_cases:
            token_ids = self.tokenizer.encode(prompt).ids
            self.assertGreater(len(token_ids), 0, prompt)
            onnx_logits = self.onnx_incremental_logits(token_ids)
            mlx_logits = self.mlx_cached_logits(token_ids)
            onnx_argmax = int(np.argmax(onnx_logits))
            mlx_argmax = int(np.argmax(mlx_logits))
            cosine = float(
                np.dot(onnx_logits, mlx_logits)
                / (np.linalg.norm(onnx_logits) * np.linalg.norm(mlx_logits))
            )
            if onnx_argmax != mlx_argmax or cosine < 0.995:
                top_onnx = np.argsort(onnx_logits)[-5:][::-1]
                top_mlx = np.argsort(mlx_logits)[-5:][::-1]
                onnx_rank_of_mlx_argmax = int(np.where(np.argsort(onnx_logits)[::-1] == mlx_argmax)[0][0])
                mlx_rank_of_onnx_argmax = int(np.where(np.argsort(mlx_logits)[::-1] == onnx_argmax)[0][0])
                failures.append(
                    {
                        "category": category,
                        "prompt": prompt,
                        "tokens": len(token_ids),
                        "onnx_argmax": onnx_argmax,
                        "mlx_argmax": mlx_argmax,
                        "cosine": cosine,
                        "max_abs": float(np.max(np.abs(onnx_logits - mlx_logits))),
                        "top5_overlap": len(set(top_onnx.tolist()) & set(top_mlx.tolist())),
                        "onnx_rank_of_mlx_argmax": onnx_rank_of_mlx_argmax,
                        "mlx_rank_of_onnx_argmax": mlx_rank_of_onnx_argmax,
                        "onnx_margin_over_mlx_argmax": float(onnx_logits[onnx_argmax] - onnx_logits[mlx_argmax]),
                        "mlx_margin_over_onnx_argmax": float(mlx_logits[mlx_argmax] - mlx_logits[onnx_argmax]),
                        "onnx_top5": [(int(i), float(onnx_logits[i])) for i in top_onnx],
                        "mlx_top5": [(int(i), float(mlx_logits[i])) for i in top_mlx],
                    }
                )
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
