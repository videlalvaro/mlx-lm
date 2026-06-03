#!/usr/bin/env python3
"""Convert a local Aion ONNX bundle into an MLX-LM safetensors directory."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import mlx.core as mx
import numpy as np


class QuantSpec:
    def __init__(self, bits: int, block_size: int, k: int, n: int) -> None:
        self.bits = bits
        self.block_size = block_size
        self.k = k
        self.n = n


class OnnxWeightStore:
    def __init__(self, onnx_path: Path) -> None:
        try:
            import onnx
            from onnx import numpy_helper
        except ImportError as exc:
            raise ImportError("Aion ONNX conversion requires `onnx`. Install it with `pip install onnx`.") from exc

        self.weights: dict[str, np.ndarray] = {}
        self.quant_specs: dict[str, QuantSpec] = {}
        model = onnx.load(str(onnx_path), load_external_data=True)
        for init in model.graph.initializer:
            self.weights[init.name] = numpy_helper.to_array(init)
        for node in model.graph.node:
            if node.op_type != "MatMulNBits" or len(node.input) < 2:
                continue
            attrs = {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}
            if {"bits", "block_size", "K", "N"}.issubset(attrs):
                self.quant_specs[node.input[1]] = QuantSpec(
                    bits=int(attrs["bits"]),
                    block_size=int(attrs["block_size"]),
                    k=int(attrs["K"]),
                    n=int(attrs["N"]),
                )

    @staticmethod
    def _expand(candidate: str) -> list[str]:
        names = [candidate]
        if candidate.endswith(".weight"):
            names.append(candidate.replace(".weight", ".matmul.backbone.weight_quantized"))
            names.append(candidate.replace(".weight", ".weight_quantized"))
        return names

    @staticmethod
    def _dequantize_blockwise(packed: np.ndarray, scales: np.ndarray, spec: QuantSpec) -> np.ndarray:
        if packed.dtype != np.uint8:
            raise TypeError(f"expected uint8 packed weights, got {packed.dtype}")
        out_dim, blocks, _blob = packed.shape
        expected_blocks = spec.k // spec.block_size
        if out_dim != spec.n or blocks != expected_blocks:
            raise ValueError(f"packed shape {packed.shape} does not match MatMulNBits K/N/block metadata")
        if scales.ndim == 1:
            scales = scales.reshape(spec.n, expected_blocks)
        if spec.bits == 8:
            q = packed.astype(np.int16)
        elif spec.bits == 4:
            low = packed & 0x0F
            high = (packed >> 4) & 0x0F
            q = np.stack([low, high], axis=-1).reshape(spec.n, expected_blocks, spec.block_size).astype(np.int16)
        else:
            raise ValueError(f"unsupported bits {spec.bits}")
        zero_point = 1 << (spec.bits - 1)
        deq = (q - zero_point).astype(np.float32) * scales.astype(np.float32)[..., None]
        return deq.reshape(spec.n, spec.k).astype(np.float16)

    def _scale_for(self, name: str) -> np.ndarray:
        if ".matmul.backbone.weight_quantized" in name:
            scale_name = name.replace(".matmul.backbone.weight_quantized", ".matmul.quantizers.weight.scale_for_export")
        else:
            scale_name = name.replace(".weight_quantized", ".quantizers.weight.scale_for_export")
        if scale_name not in self.weights:
            raise KeyError(f"missing scale tensor {scale_name}")
        return self.weights[scale_name]

    def get_any(self, names: list[str]) -> np.ndarray:
        for name in names:
            for candidate in self._expand(name):
                if candidate not in self.weights:
                    continue
                value = self.weights[candidate]
                if candidate.endswith("weight_quantized") and value.dtype == np.uint8:
                    spec = self.quant_specs[candidate]
                    return self._dequantize_blockwise(value, self._scale_for(candidate), spec)
                return value.astype(np.float16)
        raise KeyError(f"unable to resolve: {names}")

    def get_quantized_any(self, names: list[str]) -> tuple[np.ndarray, np.ndarray, QuantSpec] | None:
        for name in names:
            for candidate in self._expand(name):
                value = self.weights.get(candidate)
                if value is None or value.dtype != np.uint8 or not candidate.endswith("weight_quantized"):
                    continue
                spec = self.quant_specs.get(candidate)
                if spec is not None:
                    return value, self._scale_for(candidate), spec
        return None


def pack_aion_quantized(packed: np.ndarray, scales: np.ndarray, spec: QuantSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if spec.bits not in (4, 8):
        raise ValueError(f"unsupported bit width for MLX export: {spec.bits}")
    if spec.bits == 4:
        low = packed & 0x0F
        high = (packed >> 4) & 0x0F
        q = np.stack([low, high], axis=-1).reshape(spec.n, spec.k).astype(np.uint32)
    else:
        q = packed.reshape(spec.n, spec.k).astype(np.uint32)

    values_per_word = 32 // spec.bits
    q32 = np.zeros((spec.n, spec.k // values_per_word), dtype=np.uint32)
    mask = (1 << spec.bits) - 1
    for word_offset in range(values_per_word):
        q32 |= (q[:, word_offset::values_per_word] & mask) << (spec.bits * word_offset)

    scale = scales.reshape(spec.n, spec.k // spec.block_size).astype(np.float16)
    zero_point = 1 << (spec.bits - 1)
    bias = (-float(zero_point) * scale.astype(np.float32)).astype(np.float16)
    return q32, scale, bias


def add_linear(weights: dict[str, mx.array], store: OnnxWeightStore, out_name: str, source_names: list[str]) -> tuple[int | None, int | None]:
    quantized = store.get_quantized_any(source_names)
    if quantized is not None:
        packed, scales, spec = quantized
        qweight, qscales, qbiases = pack_aion_quantized(packed, scales, spec)
        weights[f"{out_name}.weight"] = mx.array(qweight)
        weights[f"{out_name}.scales"] = mx.array(qscales)
        weights[f"{out_name}.biases"] = mx.array(qbiases)
        return spec.bits, spec.block_size
    weights[f"{out_name}.weight"] = mx.array(store.get_any(source_names).astype(np.float16))
    return None, None


def add_norm(weights: dict[str, mx.array], store: OnnxWeightStore, out_name: str, source_names: list[str]) -> None:
    weights[f"{out_name}.weight"] = mx.array(store.get_any(source_names).astype(np.float16))


def copy_tokenizer_files(bundle: Path, out_dir: Path) -> None:
    for name in [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
    ]:
        source = bundle / name
        if source.exists():
            shutil.copy2(source, out_dir / name)
    tokenizer_config_path = out_dir / "tokenizer_config.json"
    if tokenizer_config_path.exists():
        tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
    else:
        tokenizer_config = {"tokenizer_file": "tokenizer.json"}
    tokenizer_config["fix_mistral_regex"] = True
    tokenizer_config["chat_template_type"] = "aion"
    tokenizer_config_path.write_text(json.dumps(tokenizer_config, indent=2), encoding="utf-8")


def build_quantization_config(linear_quantization: dict[str, tuple[int, int]]) -> dict | None:
    if not linear_quantization:
        return None
    (default_bits, default_group_size), _count = Counter(linear_quantization.values()).most_common(1)[0]
    quantization = {"group_size": int(default_group_size), "bits": int(default_bits), "mode": "affine"}
    for module_name, (bits, group_size) in linear_quantization.items():
        if bits != default_bits or group_size != default_group_size:
            quantization[module_name] = {"group_size": int(group_size), "bits": int(bits), "mode": "affine"}
    return quantization


def convert(bundle: Path, out_dir: Path, max_seq_len: int | None = None) -> dict:
    bundle = Path(bundle)
    out_dir.mkdir(parents=True, exist_ok=True)
    store = OnnxWeightStore(bundle / "model.onnx")
    genai = json.loads((bundle / "genai_config.json").read_text(encoding="utf-8"))
    model_cfg = genai["model"]
    cfg = model_cfg["decoder"]
    hidden_size = int(cfg["hidden_size"])
    num_layers = int(cfg["num_hidden_layers"])
    num_heads = int(cfg["num_attention_heads"])
    num_kv_heads = int(cfg["num_key_value_heads"])
    head_dim = int(cfg["head_size"])
    vocab_size = int(model_cfg["vocab_size"])
    context_length = min(int(model_cfg["context_length"]), max_seq_len or int(model_cfg["context_length"]))

    weights: dict[str, mx.array] = {}
    linear_quantization: dict[str, tuple[int, int]] = {}

    embed = store.get_any(["lm_head.weight_quantized", "model.embed_tokens.weight"]).astype(np.float16)
    weights["model.embed_tokens.weight"] = mx.array(embed)
    bits, group_size = add_linear(weights, store, "lm_head", ["lm_head.weight_quantized", "lm_head.weight"])
    if bits is not None and group_size is not None:
        linear_quantization["lm_head"] = (bits, group_size)
    add_norm(weights, store, "model.norm", ["model.norm.weight", "transformer.ln_f.weight"])

    intermediate_size = None
    for layer_idx in range(num_layers):
        source_prefix = f"model.layers.{layer_idx}."
        target_prefix = f"model.layers.{layer_idx}."
        add_norm(weights, store, target_prefix + "input_layernorm", [source_prefix + "input_layernorm.weight", f"layers.{layer_idx}.input_layernorm.weight"])
        add_norm(weights, store, target_prefix + "post_attention_layernorm", [source_prefix + "post_attention_layernorm.weight", f"layers.{layer_idx}.post_attention_layernorm.weight"])
        add_norm(weights, store, target_prefix + "self_attn.q_norm", [source_prefix + "self_attn.q_norm.weight", f"layers.{layer_idx}.self_attn.q_norm.weight"])
        add_norm(weights, store, target_prefix + "self_attn.k_norm", [source_prefix + "self_attn.k_norm.weight", f"layers.{layer_idx}.self_attn.k_norm.weight"])
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            bits, group_size = add_linear(
                weights,
                store,
                target_prefix + "self_attn." + proj,
                [source_prefix + f"self_attn.{proj}.matmul.backbone.weight_quantized", source_prefix + f"self_attn.{proj}.weight", f"layers.{layer_idx}.self_attn.{proj}.weight"],
            )
            if bits is not None and group_size is not None:
                linear_quantization[target_prefix + "self_attn." + proj] = (bits, group_size)
        for proj in ["gate_proj", "up_proj", "down_proj"]:
            bits, group_size = add_linear(
                weights,
                store,
                target_prefix + "mlp." + proj,
                [source_prefix + f"mlp.{proj}.matmul.backbone.weight_quantized", source_prefix + f"mlp.{proj}.weight", f"layers.{layer_idx}.mlp.{proj}.weight"],
            )
            if bits is not None and group_size is not None:
                linear_quantization[target_prefix + "mlp." + proj] = (bits, group_size)
        if intermediate_size is None:
            intermediate_size = weights[target_prefix + "mlp.gate_proj.weight"].shape[0]
        try:
            cos_cache = store.get_any([source_prefix + "self_attn.cos_cached_export", f"layers.{layer_idx}.self_attn.cos_cached_export"])
            sin_cache = store.get_any([source_prefix + "self_attn.sin_cached_export", f"layers.{layer_idx}.self_attn.sin_cached_export"])
        except KeyError:
            cos_cache = store.get_any(["model.layers.0.self_attn.cos_cached_export", "layers.0.self_attn.cos_cached_export"])
            sin_cache = store.get_any(["model.layers.0.self_attn.sin_cached_export", "layers.0.self_attn.sin_cached_export"])
        weights[target_prefix + "self_attn.cos_cache"] = mx.array(cos_cache[:context_length].astype(np.float16))
        weights[target_prefix + "self_attn.sin_cache"] = mx.array(sin_cache[:context_length].astype(np.float16))

    quantization = build_quantization_config(linear_quantization)

    config = {
        "model_type": "aion_onnx_mlx",
        "hidden_size": hidden_size,
        "num_hidden_layers": num_layers,
        "intermediate_size": int(intermediate_size),
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_kv_heads,
        "head_dim": head_dim,
        "vocab_size": vocab_size,
        "max_position_embeddings": context_length,
        "rms_norm_eps": 1e-5,
        "eos_token_id": model_cfg.get("eos_token_id", []),
        "quantization": quantization,
    }
    if config["quantization"] is None:
        config.pop("quantization")

    mx.save_safetensors(str(out_dir / "model.safetensors"), weights)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    copy_tokenizer_files(bundle, out_dir)
    return {"output": str(out_dir), "weights": len(weights), "config": config}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True, help="Path to the local Aion ONNX bundle containing model.onnx and genai_config.json.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-seq-len", type=int, default=None)
    args = parser.parse_args()
    print(json.dumps({"status": "ok", **convert(args.bundle, args.out_dir, args.max_seq_len)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
