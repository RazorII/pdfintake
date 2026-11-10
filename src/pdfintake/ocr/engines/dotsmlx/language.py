"""dots.mocr's language model: Qwen2 with a plain (1-D) rotary embedding, and its KV cache.

Ported from mlx-vlm 0.7.4 (``mlx_vlm/models/llava_bunny/language.py``, ``models/mlp.py``,
``models/activations.py`` for ``swiglu``, and ``KVCache`` from ``models/cache.py``), MIT,
Copyright (c) 2025 Prince Canuma; the architecture is Qwen2's as dots.ocr uses it (MIT, Copyright
(c) 2025 rednote-hilab). Module and attribute names are kept, because the checkpoint's weight keys
are those paths. Only what dots.mocr-8bit uses is here: no rope scaling, no tied embeddings, one
sequence, a causal mask during prefill and none for a single decoded token.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from ._mlx import mx, nn

if TYPE_CHECKING:
    from .config import TextConfig


@partial(mx.compile, shapeless=True)
def swiglu(gate, x):
    # Compiled as mlx-vlm compiles it: the fused kernel rounds differently from separate ops.
    return nn.silu(gate) * x


class KVCache:
    """Keys and values per layer, grown in steps of 256 positions."""

    step = 256

    def __init__(self) -> None:
        self.keys = None
        self.values = None
        self.offset = 0

    def update_and_fetch(self, keys, values):
        prev = self.offset
        if self.keys is None or (prev + keys.shape[2]) > self.keys.shape[2]:
            B, n_kv_heads, _, k_head_dim = keys.shape
            v_head_dim = values.shape[3]
            n_steps = (self.step + keys.shape[2] - 1) // self.step
            new_k = mx.zeros((B, n_kv_heads, n_steps * self.step, k_head_dim), keys.dtype)
            new_v = mx.zeros((B, n_kv_heads, n_steps * self.step, v_head_dim), values.dtype)
            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v
        self.offset += keys.shape[2]
        self.keys[..., prev:self.offset, :] = keys
        self.values[..., prev:self.offset, :] = values
        return self.keys[..., :self.offset, :], self.values[..., :self.offset, :]

    @property
    def state(self):
        if self.keys is None or self.offset == self.keys.shape[2]:
            return self.keys, self.values
        return self.keys[..., :self.offset, :], self.values[..., :self.offset, :]


class Attention(nn.Module):
    def __init__(self, config: TextConfig) -> None:
        super().__init__()
        dim = config.hidden_size
        self.n_heads = n_heads = config.num_attention_heads
        self.n_kv_heads = n_kv_heads = config.num_key_value_heads
        head_dim = config.hidden_size // n_heads
        self.scale = head_dim**-0.5
        bias = config.attention_bias
        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)
        self.rope = nn.RoPE(head_dim, traditional=False, base=config.rope_theta, scale=1)

    def __call__(self, x, mask, cache: KVCache):
        B, L, _ = x.shape
        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        queries = self.rope(queries, offset=cache.offset)
        keys = self.rope(keys, offset=cache.offset)
        keys, values = cache.update_and_fetch(keys, values)
        output = mx.fast.scaled_dot_product_attention(queries, keys, values, scale=self.scale, mask=mask)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def __call__(self, x):
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class TransformerBlock(nn.Module):
    def __init__(self, config: TextConfig) -> None:
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = MLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x, mask, cache: KVCache):
        h = x + self.self_attn(self.input_layernorm(x), mask, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class Qwen2Model(nn.Module):
    def __init__(self, config: TextConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [TransformerBlock(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def hidden(self, h, cache: list[KVCache]):
        """Run the layers over embeddings ``h`` (1, L, D), appending to ``cache``."""
        mask = "causal" if h.shape[1] > 1 else None
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)
        return h

    def logits(self, h):
        return self.lm_head(self.norm(h))


class LanguageModel(nn.Module):
    def __init__(self, config: TextConfig) -> None:
        super().__init__()
        self.model = Qwen2Model(config)

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.model.layers]
