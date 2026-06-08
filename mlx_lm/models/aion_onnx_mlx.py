from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    max_position_embeddings: int
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = False
    quantization: Optional[dict] = None


class AionRMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, self.eps)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = args.head_dim**-0.5
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * args.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, args.num_key_value_heads * args.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, args.num_key_value_heads * args.head_dim, bias=False)
        self.o_proj = nn.Linear(args.num_attention_heads * args.head_dim, args.hidden_size, bias=False)
        self.q_norm = AionRMSNorm(args.head_dim, eps=args.rms_norm_eps)
        self.k_norm = AionRMSNorm(args.head_dim, eps=args.rms_norm_eps)

    def _apply_rope(self, x: mx.array, cos: mx.array, sin: mx.array, offset: int) -> mx.array:
        B, H, L, D = x.shape
        rope_half = cos.shape[-1]
        rope_dim = rope_half * 2
        cos_slice = cos[offset : offset + L].reshape(1, 1, L, rope_half)
        sin_slice = sin[offset : offset + L].reshape(1, 1, L, rope_half)
        x_rot = x[..., :rope_dim]
        x_pass = x[..., rope_dim:]
        x_lo = x_rot[..., :rope_half]
        x_hi = x_rot[..., rope_half:]
        y_lo = x_lo * cos_slice - x_hi * sin_slice
        y_hi = x_lo * sin_slice + x_hi * cos_slice
        return mx.concatenate([y_lo, y_hi, x_pass], axis=-1)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array, mask: Optional[mx.array] = None, cache: Optional[Any] = None) -> mx.array:
        B, L, _ = x.shape
        queries = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        keys = self.k_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        values = self.v_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        queries = self.q_norm(queries)
        keys = self.k_norm(keys)
        offset = cache.offset if cache is not None else 0
        queries = self._apply_rope(queries, cos, sin, offset)
        keys = self._apply_rope(keys, cos, sin, offset)
        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)
        output = scaled_dot_product_attention(queries, keys, values, cache=cache, scale=self.scale, mask=mask)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj((gate * mx.sigmoid(gate)) * up)


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.self_attn = Attention(args)
        self.mlp = MLP(args)
        self.input_layernorm = AionRMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = AionRMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array, mask: Optional[mx.array] = None, cache: Optional[Any] = None) -> mx.array:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, mask, cache)
        return x + self.mlp(self.post_attention_layernorm(x))


class AionModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [TransformerBlock(args) for _ in range(args.num_hidden_layers)]
        self.norm = AionRMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.cos_cache = mx.zeros((args.max_position_embeddings, args.head_dim // 2))
        self.sin_cache = mx.zeros((args.max_position_embeddings, args.head_dim // 2))

    def __call__(self, inputs: mx.array, cache=None, input_embeddings: Optional[mx.array] = None):
        h = input_embeddings if input_embeddings is not None else self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        mask = create_attention_mask(h, cache[0])
        for layer, layer_cache in zip(self.layers, cache):
            h = layer(h, self.cos_cache, self.sin_cache, mask, layer_cache)
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = AionModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache=None, input_embeddings: Optional[mx.array] = None):
        out = self.model(inputs, cache, input_embeddings)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    @property
    def layers(self):
        return self.model.layers

    def sanitize(self, weights):
        # Compute biases from scales for quantized layers when not stored on disk.
        # For affine quantization: bias = -zero_point * scale where zero_point = 2^(bits-1).
        quantization = self.args.quantization
        if quantization:
            default_bits = quantization.get("bits", 4)
            for key in list(weights.keys()):
                if not key.endswith(".scales"):
                    continue
                bias_key = key[:-len(".scales")] + ".biases"
                if bias_key in weights:
                    continue
                layer_path = key[:-len(".scales")]
                layer_quant = quantization.get(layer_path)
                bits = layer_quant["bits"] if layer_quant else default_bits
                zero_point = 1 << (bits - 1)
                weights[bias_key] = (-float(zero_point) * weights[key].astype(mx.float32)).astype(mx.float16)
        # Remove lm_head weights when embeddings are tied.
        if self.args.tie_word_embeddings:
            for key in list(weights.keys()):
                if key.startswith("lm_head."):
                    weights.pop(key)
        # Migrate per-layer cos/sin cache to shared model-level cache (backward compat).
        layer0_cos = "model.layers.0.self_attn.cos_cache"
        if layer0_cos in weights and "model.cos_cache" not in weights:
            weights["model.cos_cache"] = weights[layer0_cos]
            weights["model.sin_cache"] = weights["model.layers.0.self_attn.sin_cache"]
        for key in list(weights.keys()):
            if "self_attn.cos_cache" in key or "self_attn.sin_cache" in key:
                weights.pop(key)
        return weights
