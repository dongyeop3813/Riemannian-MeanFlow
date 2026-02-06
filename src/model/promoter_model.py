import math
from typing import Optional

import jax
import jax.numpy as jnp
import flax.nnx as nnx


class GaussianFourierProjection(nnx.Module):
    def __init__(self, embed_dim: int, scale: float = 30.0, rngs: nnx.Rngs = None):
        self.embed_dim = embed_dim
        self.scale = scale
        # Non-trainable random weights
        self.W = nnx.Variable(
            jax.random.normal(rngs.params(), (self.embed_dim // 2,)) * self.scale
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: [N]
        x_proj = x[:, None] * self.W.value[None, :] * (2.0 * jnp.pi)
        return jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)


class PositionalEmbedding(nnx.Module):
    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim

        half = embed_dim // 2
        self.freqs = jnp.exp(
            -math.log(10000) * jnp.arange(start=0, stop=half, dtype=jnp.float32) / half
        )

    def __call__(self, t):
        emb = t[:, None].astype(jnp.float32) * self.freqs[None, :]
        emb = jnp.concat([jnp.cos(emb), jnp.sin(emb)], axis=-1)

        if self.embed_dim % 2 == 1:
            emb = jnp.concatenate([emb, jnp.zeros_like(emb[:, :1])], axis=-1)

        return emb


class PromoterModel(nnx.Module):
    def __init__(
        self,
        mode: str,
        embed_dim: int = 256,
        time_dependent_weights: Optional[jnp.ndarray] = None,
        time_step: float = 0.01,
        scale: float = 30.0,
        rngs: nnx.Rngs = None,
    ):
        self.mode = mode
        self.embed_dim = embed_dim
        self.time_dependent_weights = time_dependent_weights
        self.time_step = time_step
        self.alphabet_size = 4

        # Time embedding
        self.embed = nnx.Sequential(
            GaussianFourierProjection(embed_dim=self.embed_dim, scale=scale, rngs=rngs),
            nnx.Linear(self.embed_dim, self.embed_dim, rngs=rngs),
        )

        n = 256
        expanded_simplex_input = self.mode == "dirichlet" or self.mode == "riemannian"
        inp_size = self.alphabet_size * (2 if expanded_simplex_input else 1) + 1
        if self.mode == "ardm" or self.mode == "lrar":
            inp_size += 1

        # Convolution blocks (channels last). kernel_size=9, with some dilations
        def conv(out_features: int, dilation: int = 1, in_features: int = None):
            if in_features is None:
                in_features = out_features
            # Use fixed padding values to match PyTorch implementation exactly
            if dilation == 1:
                padding = 4
            elif dilation == 4:
                padding = 16
            elif dilation == 16:
                padding = 64
            elif dilation == 64:
                padding = 256
            else:
                padding = dilation * (9 - 1) // 2
            return nnx.Conv(
                in_features=in_features,
                out_features=out_features,
                kernel_size=(9,),
                padding=((padding, padding),),
                kernel_dilation=(dilation,),
                rngs=rngs,
            )

        self.input_conv = conv(n, in_features=inp_size)
        self.blocks = [
            conv(n),
            conv(n),
            conv(n, dilation=4),
            conv(n, dilation=16),
            conv(n, dilation=64),
            conv(n),
            conv(n),
            conv(n, dilation=4),
            conv(n, dilation=16),
            conv(n, dilation=64),
            conv(n),
            conv(n),
            conv(n, dilation=4),
            conv(n, dilation=16),
            conv(n, dilation=64),
            conv(n),
            conv(n),
            conv(n, dilation=4),
            conv(n, dilation=16),
            conv(n, dilation=64),
        ]

        self.denses = [nnx.Linear(self.embed_dim, n, rngs=rngs) for _ in range(20)]
        self.norms = [
            nnx.GroupNorm(num_features=n, num_groups=1, epsilon=1e-5, rngs=rngs)
            for _ in range(20)
        ]

        self.final_conv1 = nnx.Conv(
            in_features=n, out_features=n, kernel_size=(1,), rngs=rngs
        )
        self.final_conv2 = nnx.Conv(
            in_features=n, out_features=4, kernel_size=(1,), rngs=rngs
        )

    def __call__(
        self, x: jnp.ndarray, signal: jnp.ndarray, t: jnp.ndarray
    ) -> jnp.ndarray:
        # x: [N, L, Cx], signal: [N, L, 1], t: [N]
        # Time embedding
        t = jnp.squeeze(t)
        if t.ndim == 0:
            t = jnp.repeat(t[None], x.shape[0])
        embed = nnx.silu(self.embed(t / 2.0))  # [N, embed_dim]

        # Concatenate signal to inputs along channel dimension (last axis)
        out = jnp.concatenate([x, signal], axis=-1)  # [N, L, inp_size]

        # Input conv (channels last)
        out = nnx.silu(self.input_conv(out))  # [N, L, n]

        # Positional/time encoding added through dense layers
        for block, dense, norm in zip(self.blocks, self.denses, self.norms):
            # Apply normalization to the sum of out and embedding projection (matching PyTorch)
            emb_proj = dense(embed)[:, None, :]  # [N, 1, n]

            h = nnx.silu(block(norm(out + emb_proj)))

            if h.shape == out.shape:
                out = h + out
            else:
                out = h

        out = self.final_conv1(out)  # [N, L, n]
        out = nnx.gelu(out)
        out = self.final_conv2(out)  # [N, L, 4]

        # Time dependent weights interpolation
        if self.time_dependent_weights is not None:
            t_step = (t / self.time_step) - 1.0
            t0 = jnp.floor(t_step).astype(jnp.int32)
            t1 = jnp.clip(
                t0 + 1, a_min=0, a_max=self.time_dependent_weights.shape[0] - 1
            )
            w0 = self.time_dependent_weights[t0]
            w1 = self.time_dependent_weights[t1]
            w = w0 + (t_step - jnp.floor(t_step)) * (w1 - w0)
            out = out * w[:, None, None]

        # Project to zero-mean across channels for modes other than sfm
        if self.mode != "sfm":
            out = out - jnp.mean(out, axis=-1, keepdims=True)
        return out


class PromoterModelWithTimeInterval(nnx.Module):
    def __init__(
        self,
        mode: str,
        embed_dim: int = 256,
        time_dependent_weights: Optional[jnp.ndarray] = None,
        time_step: float = 0.01,
        time_emb_type: str = "fourier",
        scale: float = 30.0,
        rngs: nnx.Rngs = None,
    ):
        self.mode = mode
        self.embed_dim = embed_dim
        self.time_dependent_weights = time_dependent_weights
        self.time_step = time_step
        self.alphabet_size = 4

        self.init_time_embedder(time_emb_type, scale, rngs)

        n = 256
        expanded_simplex_input = self.mode == "dirichlet" or self.mode == "riemannian"
        inp_size = self.alphabet_size * (2 if expanded_simplex_input else 1) + 1
        if self.mode == "ardm" or self.mode == "lrar":
            inp_size += 1

        # Convolution blocks (channels last). kernel_size=9, with some dilations
        def conv(out_features: int, dilation: int = 1, in_features: int = None):
            if in_features is None:
                in_features = out_features
            # Use fixed padding values to match PyTorch implementation exactly
            if dilation == 1:
                padding = 4
            elif dilation == 4:
                padding = 16
            elif dilation == 16:
                padding = 64
            elif dilation == 64:
                padding = 256
            else:
                padding = dilation * (9 - 1) // 2
            return nnx.Conv(
                in_features=in_features,
                out_features=out_features,
                kernel_size=(9,),
                padding=((padding, padding),),
                kernel_dilation=(dilation,),
                rngs=rngs,
            )

        self.input_conv = conv(n, in_features=inp_size)
        self.blocks = [
            conv(n),
            conv(n),
            conv(n, dilation=4),
            conv(n, dilation=16),
            conv(n, dilation=64),
            conv(n),
            conv(n),
            conv(n, dilation=4),
            conv(n, dilation=16),
            conv(n, dilation=64),
            conv(n),
            conv(n),
            conv(n, dilation=4),
            conv(n, dilation=16),
            conv(n, dilation=64),
            conv(n),
            conv(n),
            conv(n, dilation=4),
            conv(n, dilation=16),
            conv(n, dilation=64),
        ]

        self.denses = [nnx.Linear(self.embed_dim * 2, n, rngs=rngs) for _ in range(20)]
        self.norms = [
            nnx.GroupNorm(num_features=n, num_groups=1, epsilon=1e-5, rngs=rngs)
            for _ in range(20)
        ]

        self.final_linear1 = nnx.Linear(n, n, rngs=rngs)
        self.final_linear2 = nnx.Linear(n, 4, rngs=rngs)

    def init_time_embedder(self, time_emb_type: str, scale: float, rngs: nnx.Rngs):
        if time_emb_type == "fourier":
            # Time embedding
            self.embed = nnx.Sequential(
                GaussianFourierProjection(
                    embed_dim=self.embed_dim, scale=scale, rngs=rngs
                ),
                nnx.Linear(self.embed_dim, self.embed_dim, rngs=rngs),
            )

            # Time gap embedding
            self.time_gap_embed = nnx.Sequential(
                GaussianFourierProjection(
                    embed_dim=self.embed_dim, scale=scale, rngs=rngs
                ),
                nnx.Linear(self.embed_dim, self.embed_dim, rngs=rngs),
            )
        elif time_emb_type == "position":
            self.embed = nnx.Sequential(
                PositionalEmbedding(embed_dim=self.embed_dim),
                nnx.Linear(self.embed_dim, self.embed_dim, rngs=rngs),
            )
            self.time_gap_embed = nnx.Sequential(
                PositionalEmbedding(embed_dim=self.embed_dim),
                nnx.Linear(self.embed_dim, self.embed_dim, rngs=rngs),
            )
        else:
            raise ValueError(f"Invalid time embedding type: {time_emb_type}")

    def __call__(
        self, x: jnp.ndarray, signal: jnp.ndarray, t: jnp.ndarray, s: jnp.ndarray
    ) -> jnp.ndarray:
        # x: [N, L, Cx], signal: [N, L, 1], t: [N]
        # Time embedding
        t = jnp.squeeze(t)
        if t.ndim == 0:
            t = jnp.repeat(t[None], x.shape[0])
        embed = nnx.silu(self.embed(t / 2.0))  # [N, embed_dim]

        # Time gap embedding
        s_minus_t = s - t
        s_minus_t = jnp.squeeze(s_minus_t)
        if s_minus_t.ndim == 0:
            s_minus_t = jnp.repeat(s_minus_t[None], x.shape[0])
        gap_embed = nnx.silu(self.time_gap_embed(s_minus_t / 2.0))  # [N, embed_dim]

        embed = jnp.concatenate([embed, gap_embed], axis=-1)

        # Concatenate signal to inputs along channel dimension (last axis)
        out = jnp.concatenate([x, signal], axis=-1)  # [N, L, inp_size]

        # Input conv (channels last)
        out = nnx.silu(self.input_conv(out))  # [N, L, n]

        # Positional/time encoding added through dense layers
        for block, dense, norm in zip(self.blocks, self.denses, self.norms):
            # Apply normalization to the sum of out and embedding projection (matching PyTorch)
            emb_proj = dense(embed)[:, None, :]  # [N, 1, n]

            h = nnx.silu(block(norm(out + emb_proj)))
            if h.shape == out.shape:
                out = h + out
            else:
                out = h

        out = self.final_linear1(out)  # [N, L, n]
        out = nnx.gelu(out)
        out = self.final_linear2(out)  # [N, L, 4]

        # Time dependent weights interpolation
        if self.time_dependent_weights is not None:
            t_step = (t / self.time_step) - 1.0
            t0 = jnp.floor(t_step).astype(jnp.int32)
            t1 = jnp.clip(
                t0 + 1, a_min=0, a_max=self.time_dependent_weights.shape[0] - 1
            )
            w0 = self.time_dependent_weights[t0]
            w1 = self.time_dependent_weights[t1]
            w = w0 + (t_step - jnp.floor(t_step)) * (w1 - w0)
            out = out * w[:, None, None]

        # Project to zero-mean across channels for modes other than sfm
        if self.mode != "sfm":
            out = out - jnp.mean(out, axis=-1, keepdims=True)

        return out
