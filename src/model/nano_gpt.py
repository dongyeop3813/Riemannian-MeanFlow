import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import flax.nnx as nnx
from jax import Array


def sinusoidal_timestep_embedding(
    timesteps: Array, embedding_dim: int, max_positions: int = 10000
) -> Array:
    assert timesteps.ndim == 1
    half_dim = embedding_dim // 2
    emb_scale = math.log(max_positions) / (half_dim - 1)
    freqs = jnp.exp(jnp.arange(half_dim, dtype=jnp.float32) * (-emb_scale))
    args = timesteps.astype(jnp.float32)[:, None] * freqs[None, :]
    emb = jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)
    if embedding_dim % 2 == 1:
        emb = jnp.pad(emb, ((0, 0), (0, 1)))
    return emb


class LayerNorm(nnx.Module):
    def __init__(self, features: int, use_bias: bool):
        self.scale = nnx.Param(jnp.ones((features,), dtype=jnp.float32))
        self.bias = (
            nnx.Param(jnp.zeros((features,), dtype=jnp.float32)) if use_bias else None
        )

    def __call__(self, x: Array) -> Array:
        mean = x.mean(axis=-1, keepdims=True)
        var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
        y = (x - mean) * jax.lax.rsqrt(var + 1e-5)
        y = y * self.scale
        if self.bias is not None:
            y = y + self.bias
        return y


class SelfAttention(nnx.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        dropout: float,
        use_bias: bool,
        qk_layernorm: bool,
        rngs: nnx.Rngs,
    ):
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.qkv = nnx.Linear(n_embd, 3 * n_embd, use_bias=use_bias, rngs=rngs)
        self.proj = nnx.Linear(n_embd, n_embd, use_bias=use_bias, rngs=rngs)
        self.attn_dropout = dropout
        self.resid_dropout = dropout
        self.qk_layernorm = qk_layernorm
        if qk_layernorm:
            head_dim = n_embd // n_head
            self.q_ln = LayerNorm(head_dim, use_bias)
            self.k_ln = LayerNorm(head_dim, use_bias)

    def __call__(
        self, x: Array, attn_mask: Array | None = None, *, rngs: nnx.Rngs
    ) -> Array:
        b, t, c = x.shape
        qkv = self.qkv(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        head_dim = c // self.n_head

        def split_heads(y):
            y = y.reshape(b, t, self.n_head, head_dim)
            return jnp.transpose(y, (0, 2, 1, 3))

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        if self.qk_layernorm:
            q = self.q_ln(q)
            k = self.k_ln(k)

        scale = 1.0 / math.sqrt(head_dim)
        att = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale
        if attn_mask is not None:
            att = jnp.where(attn_mask[:, None, :, :], att, -1e30)
        att = jax.nn.softmax(att, axis=-1)
        if self.attn_dropout > 0.0 and nnx.is_training():
            keep = 1.0 - self.attn_dropout
            dropout_mask = jax.random.bernoulli(rngs.dropout(), keep, att.shape)
            att = att * dropout_mask / keep
        y = jnp.einsum("bhqk,bhkd->bhqd", att, v)
        y = jnp.transpose(y, (0, 2, 1, 3)).reshape(b, t, c)
        y = self.proj(y)
        if self.resid_dropout > 0.0 and nnx.is_training():
            keep = 1.0 - self.resid_dropout
            dropout_mask = jax.random.bernoulli(rngs.dropout(), keep, y.shape)
            y = y * dropout_mask / keep
        return y


class MLP(nnx.Module):
    def __init__(self, n_embd: int, dropout: float, use_bias: bool, rngs: nnx.Rngs):
        self.fc = nnx.Linear(n_embd, 4 * n_embd, use_bias=use_bias, rngs=rngs)
        self.proj = nnx.Linear(4 * n_embd, n_embd, use_bias=use_bias, rngs=rngs)
        self.dropout = dropout

    def __call__(self, x: Array, *, rngs: nnx.Rngs) -> Array:
        x = self.fc(x)
        x = nnx.gelu(x)
        x = self.proj(x)
        if self.dropout > 0.0 and nnx.is_training():
            keep = 1.0 - self.dropout
            dropout_mask = jax.random.bernoulli(rngs.dropout(), keep, x.shape)
            x = x * dropout_mask / keep
        return x


class Block(nnx.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        dropout: float,
        use_bias: bool,
        qk_layernorm: bool,
        rngs: nnx.Rngs,
    ):
        self.ln1 = LayerNorm(n_embd, use_bias)
        self.attn = SelfAttention(
            n_embd, n_head, dropout, use_bias, qk_layernorm, rngs=rngs
        )
        self.ln2 = LayerNorm(n_embd, use_bias)
        self.mlp = MLP(n_embd, dropout, use_bias, rngs=rngs)

    def __call__(self, x: Array, attn_mask: Array | None, *, rngs: nnx.Rngs) -> Array:
        x = x + self.attn(self.ln1(x), attn_mask, rngs=rngs)
        x = x + self.mlp(self.ln2(x), rngs=rngs)
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    qk_layernorm: bool = False
    do_x1_sc: bool = False
    mask_token_id: int = 0
    proper_timestep_emb: bool = False
    d3pm_loss_weighting: bool = False
    d3pm_loss_weighting_maxT: int = 1000


class GPT(nnx.Module):
    def __init__(self, config: GPTConfig, rngs: nnx.Rngs):
        assert config.vocab_size is not None and config.block_size is not None
        self.config = config
        self.wte = nnx.Embed(config.vocab_size, config.n_embd, rngs=rngs)
        self.wpe = nnx.Embed(config.block_size, config.n_embd, rngs=rngs)
        self.drop = config.dropout
        self.h = [
            Block(
                config.n_embd,
                config.n_head,
                config.dropout,
                config.bias,
                config.qk_layernorm,
                rngs=rngs,
            )
            for _ in range(config.n_layer)
        ]
        self.ln_f = LayerNorm(config.n_embd, config.bias)
        self.lm_head = nnx.Linear(
            config.n_embd, config.vocab_size, use_bias=False, rngs=rngs
        )

    def _run_net(
        self,
        idx: Array,
        time: Array,
        x1: Array | None,
        attn_mask: Array | None,
        *,
        rngs: nnx.Rngs,
    ) -> Array:
        b, t = idx.shape
        n_embd = self.config.n_embd
        pos = jnp.arange(t, dtype=jnp.int32)
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        if self.config.do_x1_sc and x1 is not None:
            x1_tok_emb = self.wte(x1)
            tok_emb = nnx.Linear(
                2 * n_embd, n_embd, use_bias=self.config.bias, rngs=rngs
            )(jnp.concatenate([tok_emb, x1_tok_emb], axis=-1))

        if self.config.proper_timestep_emb:
            time_emb = sinusoidal_timestep_embedding(time * 1000.0, n_embd)
        else:
            time_emb = sinusoidal_timestep_embedding(time, n_embd)

        x = tok_emb + pos_emb[None, :, :] + time_emb[:, None, :]
        if self.drop > 0.0 and nnx.is_training():
            keep = 1.0 - self.drop
            dropout_mask = jax.random.bernoulli(rngs.dropout(), keep, x.shape)
            x = x * dropout_mask / keep
        for block in self.h:
            x = block(x, attn_mask, rngs=rngs)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def __call__(
        self,
        idx: Array,
        time: Array,
        targets: Array | None = None,
        target_mask: Array | None = None,
        do_self_cond_loop: bool = False,
        attn_mask: Array | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        b, t = idx.shape
        if self.config.do_x1_sc:
            if do_self_cond_loop:
                x1_mask = jnp.full_like(idx, self.config.mask_token_id)
                logits = self._run_net(idx, time, x1_mask, attn_mask, rngs=rngs)
                probs = jax.nn.softmax(logits, axis=-1)
                x1_sample = jax.random.categorical(
                    rngs.params(), jnp.log(probs).reshape(b * t, -1)
                ).reshape(b, t)
                logits = self._run_net(idx, time, x1_sample, attn_mask, rngs=rngs)
            else:
                x1_mask = jnp.full_like(idx, self.config.mask_token_id)
                logits = self._run_net(idx, time, x1_mask, attn_mask, rngs=rngs)
        else:
            logits = self._run_net(idx, time, None, attn_mask, rngs=rngs)

        loss = None
        if targets is not None:
            per_token_loss = -jnp.sum(
                jax.nn.log_softmax(logits, axis=-1)
                * jax.nn.one_hot(targets, logits.shape[-1]),
                axis=-1,
            )
            if target_mask is not None:
                masked_loss = per_token_loss * target_mask
                if self.config.d3pm_loss_weighting:
                    scaled_up_time = self.config.d3pm_loss_weighting_maxT * time
                    scaled_up_time = jnp.clip(
                        scaled_up_time[:, None].repeat(t, axis=1),
                        0.01 * self.config.d3pm_loss_weighting_maxT,
                        0.99 * self.config.d3pm_loss_weighting_maxT,
                    )
                    masked_loss = masked_loss / scaled_up_time
                loss = jnp.sum(masked_loss) / (jnp.sum(target_mask) + 1e-5)
            else:
                loss = jnp.mean(per_token_loss)
        return logits, loss


if __name__ == "__main__":
    config = GPTConfig(
        n_layer=1,
        n_head=1,
        n_embd=128,
        block_size=1024,
    )
    model = GPT(config, rngs=nnx.Rngs(0))
    nnx.display(model)
