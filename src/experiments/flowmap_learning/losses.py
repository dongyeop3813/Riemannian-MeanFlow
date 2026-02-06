# Riemannian MeanFlow losses

import jax
from jax.lax import stop_gradient
from jax import jvp
import jax.numpy as jnp

from src.utils import *


def meanflow_loss(
    model,
    batch,
    loss_p=0.0,
    tangent_clip: None | float = None,
    return_stratified_loss: bool = True,
    loss_down_scale: bool = False,
    t_clip: float = 0.1,
):
    path, (t, s), cond = batch
    x_t = path.x_t
    v_t = path.dx_t

    manifold = model.manifold

    s_minus_t = expand_to(s - t, x_t.ndim)

    fn = lambda x, t, s: model(x=x, t=t, s=s, **cond)

    u_t, du_dt = jvp(fn, (x_t, t, s), (v_t, jnp.ones_like(t), jnp.zeros_like(s)))

    if tangent_clip is not None:
        du_dt = norm_clip(du_dt, tangent_clip)
    du_dt = manifold.proju(x_t, du_dt)
    x_s = manifold.exp_map(x_t, s_minus_t * u_t)

    def log_map_fn(x):
        return manifold.log_map(x, x_s)

    dlog_dx = jvp(log_map_fn, (x_t,), (v_t,))[1]
    dlog_dx = manifold.proju(x_t, dlog_dx)
    u_tgt = s_minus_t * du_dt - dlog_dx

    error = u_t - jax.lax.stop_gradient(u_tgt)
    loss = manifold.square_norm_at(x_t, error)

    if loss_down_scale:
        scaler = jnp.where(t > 1 - t_clip, (1 - t) / t_clip, 1.0)
        loss = scaler**2 * loss

    if return_stratified_loss:
        return weighted_loss(loss, loss_p), t_stratified_loss(t, loss)
    else:
        return weighted_loss(loss, loss_p)


def meanflow_loss_self_distillation(
    model,
    batch,
    loss_p=0.0,
    tangent_clip: None | float = None,
    return_stratified_loss: bool = True,
    partial_sg: bool = False,
):
    path, (t, s), cond = batch
    x_t = path.x_t
    v_t = path.dx_t
    manifold = model.manifold

    # Flow matching loss
    u_tt = model.instant_velocity(x_t, t, **cond)
    loss = manifold.square_norm_at(x_t, u_tt - v_t)

    # Self-distillation loss
    s_minus_t = expand_to(s - t, x_t.ndim)

    fn = lambda x, t, s: model(x=x, t=t, s=s, **cond)
    u_ts, du_dt = jvp(
        fn, (x_t, t, s), (stop_gradient(u_tt), jnp.ones_like(t), jnp.zeros_like(s))
    )

    if tangent_clip is not None:
        du_dt = norm_clip(du_dt, tangent_clip)
    du_dt = manifold.proju(x_t, du_dt)
    x_s = manifold.exp_map(x_t, s_minus_t * u_ts)

    def log_map_fn(x):
        return manifold.log_map(x, x_s)

    dlog_dx = jvp(log_map_fn, (x_t,), (u_tt,))[1]
    dlog_dx = manifold.proju(x_t, dlog_dx)

    if partial_sg:
        u_tgt = s_minus_t * du_dt - stop_gradient(dlog_dx)
    else:
        u_tgt = s_minus_t * du_dt - dlog_dx
        u_tgt = stop_gradient(u_tgt)

    loss += manifold.square_norm_at(x_t, u_ts - u_tgt)

    if return_stratified_loss:
        return weighted_loss(loss, loss_p), t_stratified_loss(t, loss)
    else:
        return weighted_loss(loss, loss_p)


def meanflow_loss_with_jvp_bp(
    model,
    batch,
    loss_p=0.0,
    tangent_clip: None | float = None,
    return_stratified_loss: bool = True,
):
    path, (t, s), cond = batch
    x_t = path.x_t
    v_t = path.dx_t

    manifold = model.manifold

    s_minus_t = expand_to(s - t, x_t.ndim)

    fn = lambda x, t, s: model(x=x, t=t, s=s, **cond)

    u_t, du_dt = jvp(fn, (x_t, t, s), (v_t, jnp.ones_like(t), jnp.zeros_like(s)))

    if tangent_clip is not None:
        du_dt = norm_clip(du_dt, tangent_clip)
    du_dt = manifold.proju(x_t, du_dt)
    x_s = manifold.exp_map(x_t, s_minus_t * u_t)

    def log_map_fn(x):
        return manifold.log_map(x, x_s)

    dlog_dx = jvp(log_map_fn, (x_t,), (v_t,))[1]
    dlog_dx = manifold.proju(x_t, dlog_dx)

    error = u_t - s_minus_t * du_dt + jax.lax.stop_gradient(dlog_dx)
    loss = manifold.square_norm_at(x_t, error)

    if return_stratified_loss:
        return weighted_loss(loss, loss_p), t_stratified_loss(t, loss)
    else:
        return weighted_loss(loss, loss_p)


def split_flow_loss(
    model,
    batch,
    loss_p1=0.0,
    loss_p2=0.0,
    weight=1.0,
    return_stratified_loss: bool = True,
    v_loss_down_scale: bool = True,
    split_loss_down_scale: bool = True,
    split_loss_type: str = "x_loss",
):
    loss1, stratified_loss1 = semigroup_identity_loss(
        model,
        batch,
        loss_p1,
        loss_down_scale=split_loss_down_scale,
        split_loss_type=split_loss_type,
    )

    path, (t, _, _), cond = batch
    loss2, stratified_loss2 = flow_matching_loss(
        model, (path, t, cond), loss_p2, v_loss_down_scale
    )

    loss = weight * loss1 + loss2

    if return_stratified_loss:
        return loss, {**stratified_loss1, **stratified_loss2}
    else:
        return loss


def semigroup_identity_loss(
    model,
    batch,
    loss_p=0.0,
    return_stratified_loss: bool = True,
    prefix: str = "split_",
    loss_down_scale: bool = True,
    t_clip: float = 0.1,
    split_loss_type: str = "x_loss",
):
    # t < s < r
    path, (t, s, r), cond = batch

    x_t = path.x_t
    dist = model.manifold.geodesic_distance

    x_s = model.forward_flow(x_t, t, s, **cond)
    x_r = model.forward_flow(x_s, s, r, **cond)

    if split_loss_type == "x_loss":
        loss = dist(model.forward_flow(x_t, t, r, **cond), stop_gradient(x_r)) ** 2
    elif split_loss_type == "v_loss":
        u_target = stop_gradient(
            model.manifold.log_map(x_t, x_r)
            / expand_to(jnp.clip(r - t, min=1e-8), x_t.ndim)
        )
        u_t = model(x_t, t, r, **cond)
        loss = model.manifold.square_norm_at(x_t, u_t - u_target)
    else:
        raise ValueError(f"Unknown split loss type: {split_loss_type}")

    if loss_down_scale:
        scaler = jnp.where(t > 1 - t_clip, (1 - t) / t_clip, 1.0)
        loss = scaler**2 * loss

    if return_stratified_loss:
        return weighted_loss(loss, loss_p), t_stratified_loss(
            r - t, loss, prefix=prefix
        )
    else:
        return weighted_loss(loss, loss_p)


def flow_matching_loss(
    model, batch, loss_p=0.0, v_loss_down_scale: bool = True, t_clip: float = 0.1
):
    path, t, cond = batch

    u_t = model.instant_velocity(path.x_t, t, **cond)

    error = u_t - path.dx_t
    if v_loss_down_scale:
        scaler = jnp.where(t > 1 - t_clip, (1 - t) / t_clip, 1.0)
        error = error * expand_to(scaler, error.ndim)

    loss = model.manifold.square_norm_at(path.x_t, error)

    return weighted_loss(loss, loss_p), t_stratified_loss(t, loss, prefix="flow_")


def euler_identity_loss(model, batch):
    path, (t, s), cond = batch

    x_t = path.x_t
    v_t = path.dx_t
    manifold = model.manifold

    s_minus_t = expand_to(s - t, x_t.ndim)

    fn = lambda x, t, s: model(x=x, t=t, s=s, **cond)

    u_t, du_dt = jvp(fn, (x_t, t, s), (v_t, jnp.ones_like(t), jnp.zeros_like(s)))
    du_dt = manifold.proju(x_t, du_dt)
    x_s = manifold.exp_map(x_t, s_minus_t * u_t)

    def log_map_fn(x):
        return manifold.log_map(x, x_s)

    dlog_dx = jvp(log_map_fn, (x_t,), (v_t,))[1]
    dlog_dx = manifold.proju(x_t, dlog_dx)
    u_tgt = s_minus_t * du_dt - dlog_dx

    error = u_t - jax.lax.stop_gradient(u_tgt)
    loss = manifold.square_norm_at(x_t, error)

    return loss, t_stratified_loss(t, loss, prefix="euler_")


def meanflow_loss_with_split(
    model,
    batch,
    loss_p=0.0,
    split_idx=None,
):
    # First (split_idx) samples are used for flow matching
    # Second (batch_size - split_idx) samples are used for consistency loss
    batch1, batch2 = split_pytree(batch, split_idx)

    loss1, binned_loss1 = flow_matching_loss(model, batch1)
    loss2, binned_loss2 = euler_identity_loss(model, batch2)

    loss = loss1 + weighted_loss(loss2, loss_p).mean()

    return loss, {**binned_loss1, **binned_loss2}


def lagrange_pushforward_loss(
    model,
    batch,
    loss_p=0.0,
    tangent_clip=None,
    return_stratified_loss=True,
):
    # This loss compares avg. velocity u at x_t,
    # so final tangent vectors should be projected to x_t.
    # path is sampled at x_t.
    path, (s, t), cond = batch
    x_t, v_t = path.x_t, path.dx_t
    manifold = model.manifold
    t_minus_s = expand_to(t - s, x_t.ndim)

    x_s = model.forward_flow(x_t, t, s, **cond)
    x_s = stop_gradient(x_s)

    fn = lambda x, s, t: model(x, s, t, **cond)
    u, du_dt = jvp(
        fn,
        (x_s, s, t),
        (jnp.zeros_like(x_s), jnp.zeros_like(s), jnp.ones_like(t)),
    )
    if tangent_clip is not None:
        du_dt = norm_clip(du_dt, tangent_clip)
    du_dt = manifold.proju(x_s, du_dt)

    def d2_exp(v):
        def exp_fn(w):
            return manifold.exp_map(x_s, w)

        return jvp(exp_fn, (manifold.log_map(x_s, x_t),), (v,))[1]

    d2_exp_w = d2_exp(u + stop_gradient(t_minus_s * du_dt))
    d2_exp_w = manifold.proju(x_t, d2_exp_w)

    error = d2_exp_w - v_t
    loss = manifold.square_norm_at(x_t, error)

    if return_stratified_loss:
        return weighted_loss(loss, loss_p), t_stratified_loss(t, loss)
    else:
        return weighted_loss(loss, loss_p)


def lagrange_pullback_loss(
    model, batch, loss_p=0.0, tangent_clip=None, return_stratified_loss=True
):
    # This loss compares avg. velocity u at x_s,
    # so final tangent vector should be projected to x_s.
    # path is sampled at x_t.
    path, (s, t), cond = batch
    x_t, v_t = path.x_t, path.dx_t
    manifold = model.manifold
    t_minus_s = expand_to(t - s, x_t.ndim)

    x_s = model.forward_flow(x_t, t, s, **cond)
    x_s = stop_gradient(x_s)

    fn = lambda x, s, t: model(x, s, t, **cond)
    u, du_dt = jvp(
        fn, (x_s, s, t), (jnp.zeros_like(x_s), jnp.zeros_like(s), jnp.ones_like(t))
    )
    if tangent_clip is not None:
        du_dt = norm_clip(du_dt, tangent_clip)
    du_dt = manifold.proju(x_s, du_dt)

    def dlog(v):
        def log(y):
            return manifold.log_map(x_s, y)

        return jvp(log, (x_t,), (v,))[1]

    dlog_v = dlog(v_t)
    dlog_v = manifold.proju(x_s, dlog_v)
    u_tgt = dlog_v - t_minus_s * du_dt

    error = u - stop_gradient(u_tgt)
    loss = manifold.square_norm_at(x_s, error)

    if return_stratified_loss:
        return weighted_loss(loss, loss_p), t_stratified_loss(t, loss)
    else:
        return weighted_loss(loss, loss_p)


def AYF_EMD_objective(
    model, batch, loss_p=0.0, tangent_clip=None, return_stratified_loss=True
):
    path, (s, t), cond = batch
    x_s = path.x_t
    v_s = path.dx_t
    manifold = model.manifold

    fn = lambda x, s, t: model.forward_flow(x, s, t, **cond)
    xt, dxt_ds = jvp(fn, (x_s, s, t), (v_s, jnp.ones_like(s), jnp.zeros_like(t)))
    sg_xt = stop_gradient(xt)

    if tangent_clip is not None:
        dxt_ds = norm_clip(dxt_ds, tangent_clip)
    dxt_ds = manifold.proju(x_s, dxt_ds)
    dxt_ds = stop_gradient(dxt_ds)

    d1_l = jvp(
        manifold.geodesic_distance,
        (sg_xt, xt),
        (dxt_ds, jnp.zeros_like(xt)),
    )[1]

    loss = d1_l**2
    if return_stratified_loss:
        return weighted_loss(loss, loss_p), t_stratified_loss(t, loss)
    else:
        return weighted_loss(loss, loss_p)


def create_loss_metric(name):
    import flax.nnx as nnx
    from src.factory import create_stratified_metric

    if name == "meanflow" or name == "meanflow_self_distillation":
        metrics = {"loss": nnx.metrics.Average("loss")}
        metrics.update(create_stratified_metric(return_as_dict=True))
        return nnx.metrics.MultiMetric(**metrics)
    elif name == "meanflow_with_split":
        metrics = {"loss": nnx.metrics.Average("loss")}
        metrics.update(create_stratified_metric("flow_", return_as_dict=True))
        metrics.update(create_stratified_metric("euler_", return_as_dict=True))
        return nnx.metrics.MultiMetric(**metrics)
    elif name == "split_flow":
        metrics = {"loss": nnx.metrics.Average("loss")}
        metrics.update(create_stratified_metric("flow_", return_as_dict=True))
        metrics.update(create_stratified_metric("split_", return_as_dict=True))
        return nnx.metrics.MultiMetric(**metrics)
    elif name in ["lagrange_pullback", "lagrange_pushforward"]:
        metrics = {"loss": nnx.metrics.Average("loss")}
        metrics.update(create_stratified_metric(return_as_dict=True))
        return nnx.metrics.MultiMetric(**metrics)
    elif name == "AYF_Euler":
        metrics = {"loss": nnx.metrics.Average("loss")}
        metrics.update(create_stratified_metric(return_as_dict=True))
        return nnx.metrics.MultiMetric(**metrics)
    else:
        raise ValueError(f"Unknown loss: {name}")
