import optax
import flax.nnx as nnx

from typing import Optional


def init_optim(cfg, model: nnx.Module, iter_per_epoch: Optional[int] = None):

    if "schedule" in cfg:
        schedule = optax.cosine_decay_schedule(
            init_value=cfg.lr,
            decay_steps=iter_per_epoch * cfg.schedule.decay_steps,
            alpha=cfg.schedule.alpha,
        )

        optim = optax.adamw(schedule, weight_decay=cfg.weight_decay)
    else:
        optim = optax.adamw(cfg.lr, weight_decay=cfg.weight_decay)

    if "grad_clip" in cfg:
        optim = optax.chain(
            optax.clip_by_global_norm(cfg.grad_clip),
            optim,
        )
    elif "grad_clip_by_val" in cfg:
        optim = optax.chain(
            optax.clip(cfg.grad_clip_by_val),
            optim,
        )

    optimizer = nnx.Optimizer(model, optim, wrt=nnx.Param)

    return optimizer
