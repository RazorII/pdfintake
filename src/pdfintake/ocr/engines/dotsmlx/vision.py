"""dots.mocr's vision tower: the dots ViT (patch embed, 42 blocks with 2-D rotary attention, merger).

Ported from mlx-vlm 0.7.4 (``mlx_vlm/models/dots_ocr/vision.py``), MIT, Copyright (c) 2025 Prince
Canuma; the architecture is dots.ocr's (MIT, Copyright (c) 2025 rednote-hilab). Module and
attribute names are kept, because the checkpoint's weight keys are those paths. One image per
call, so attention runs over the whole patch sequence; the checkpoint's patch-embed weight is
already in mlx's (out, kh, kw, in) layout, so there's no weight sanitising.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._mlx import mx, nn

if TYPE_CHECKING:
    from .config import VisionConfig


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb_vision(tensor, freqs):
    orig_dtype = tensor.dtype
    cos = mx.expand_dims(mx.tile(mx.expand_dims(mx.cos(freqs), axis=1), (1, 1, 2)), axis=0)
    sin = mx.expand_dims(mx.tile(mx.expand_dims(mx.sin(freqs), axis=1), (1, 1, 2)), axis=0)
    output = (tensor * cos) + (rotate_half(tensor) * sin)
    return output.astype(orig_dtype)


def rotary_freqs(dim: int, seqlen: int, theta: float = 10000.0):
    """The (seqlen, dim / 2) rotary angles for positions 0..seqlen-1."""
    inv_freq = 1.0 / (theta ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
    seq = mx.arange(int(seqlen), dtype=inv_freq.dtype)
    return mx.outer(seq, inv_freq)


class RMSNorm(nn.Module):
    """The ViT's RMSNorm: computed in float32, unlike mlx's fused one."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x):
        input_dtype = x.dtype
        x = x.astype(mx.float32)
        x = x * mx.rsqrt(mx.mean(x**2, axis=-1, keepdims=True) + self.eps)
        return x.astype(input_dtype) * self.weight.astype(input_dtype)


class PatchMerger(nn.Module):
    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = nn.LayerNorm(context_dim, eps=1e-6)
        self.mlp = [nn.Linear(self.hidden_size, self.hidden_size), nn.GELU(),
                    nn.Linear(self.hidden_size, dim)]

    def __call__(self, x):
        x = self.ln_q(x).reshape(-1, self.hidden_size)
        for layer in self.mlp:
            x = layer(x)
        return x


class VisionAttention(nn.Module):
    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.scale = (config.embed_dim // config.num_attention_heads) ** -0.5
        self.qkv = nn.Linear(config.embed_dim, config.embed_dim * 3, bias=config.use_bias)
        self.proj = nn.Linear(config.embed_dim, config.embed_dim, bias=config.use_bias)

    def __call__(self, hidden_states, rotary_pos_emb):
        seq_length = hidden_states.shape[0]
        qkv = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).transpose(1, 0, 2, 3)
        q, k, v = mx.split(qkv, 3)
        q = apply_rotary_pos_emb_vision(mx.expand_dims(q, 0), rotary_pos_emb)[0]
        k = apply_rotary_pos_emb_vision(mx.expand_dims(k, 0), rotary_pos_emb)[0]
        q, k, v = (a.transpose(0, 2, 1, 3) for a in (q, k, v))
        output = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        output = output.transpose(0, 2, 1, 3).reshape(seq_length, -1)
        return self.proj(output)


class DotsSwiGLUFFN(nn.Module):
    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.embed_dim, config.intermediate_size, bias=config.use_bias)
        self.fc2 = nn.Linear(config.intermediate_size, config.embed_dim, bias=config.use_bias)
        self.fc3 = nn.Linear(config.embed_dim, config.intermediate_size, bias=config.use_bias)

    def __call__(self, x):
        return self.fc2(nn.silu(self.fc1(x)) * self.fc3(x))


class DotsPatchEmbed(nn.Module):
    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.num_channels = config.num_channels
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.embed_dim = config.embed_dim
        self.proj = nn.Conv2d(config.num_channels, config.embed_dim,
                              kernel_size=(config.patch_size, config.patch_size),
                              stride=(config.patch_size, config.patch_size), bias=True)
        self.norm = RMSNorm(config.embed_dim, eps=config.rms_norm_eps)

    def __call__(self, x):
        x = x.reshape(-1, self.num_channels, self.temporal_patch_size, self.patch_size, self.patch_size)[:, :, 0]
        x = x.transpose(0, 2, 3, 1)                                   # (N, P, P, C) for mlx's conv
        x = self.proj(x).reshape(-1, self.embed_dim)
        return self.norm(x)


class DotsViTPreprocessor(nn.Module):
    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.patchifier = DotsPatchEmbed(config)

    def __call__(self, x):
        return self.patchifier(x)


class DotsVisionBlock(nn.Module):
    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.attn = VisionAttention(config)
        self.norm1 = RMSNorm(config.embed_dim, eps=config.rms_norm_eps)
        self.mlp = DotsSwiGLUFFN(config)
        self.norm2 = RMSNorm(config.embed_dim, eps=config.rms_norm_eps)

    def __call__(self, hidden_states, rotary_pos_emb):
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), rotary_pos_emb)
        return hidden_states + self.mlp(self.norm2(hidden_states))


class VisionModel(nn.Module):
    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.head_dim = config.embed_dim // config.num_attention_heads
        self.patch_embed = DotsViTPreprocessor(config)
        self.blocks = [DotsVisionBlock(config) for _ in range(config.num_hidden_layers)]
        if config.post_norm:
            self.post_trunk_norm = RMSNorm(config.embed_dim, eps=config.rms_norm_eps)
        self.merger = PatchMerger(dim=config.hidden_size, context_dim=config.embed_dim,
                                  spatial_merge_size=config.spatial_merge_size)

    def rot_pos_emb(self, grid: tuple[int, int, int]):
        """Each patch's (row, column) rotary angles, in the merge-block order the patches come in."""
        t, h, w = grid
        s = self.spatial_merge_size
        hpos = mx.repeat(mx.arange(h).reshape(h, 1), w, axis=1)
        hpos = hpos.reshape(h // s, s, w // s, s).transpose(0, 2, 1, 3).flatten()
        wpos = mx.repeat(mx.arange(w).reshape(1, w), h, axis=0)
        wpos = wpos.reshape(h // s, s, w // s, s).transpose(0, 2, 1, 3).flatten()
        pos_ids = mx.tile(mx.stack([hpos, wpos], axis=-1), (t, 1))
        full = rotary_freqs(self.head_dim // 2, max(h, w))
        return full[pos_ids].reshape(pos_ids.shape[0], -1)

    def __call__(self, pixel_values, grid: tuple[int, int, int]):
        hidden_states = self.patch_embed(pixel_values)
        rotary_pos_emb = self.rot_pos_emb(grid)
        for block in self.blocks:
            hidden_states = block(hidden_states, rotary_pos_emb)
        if self.config.post_norm:
            hidden_states = self.post_trunk_norm(hidden_states)
        return self.merger(hidden_states)
