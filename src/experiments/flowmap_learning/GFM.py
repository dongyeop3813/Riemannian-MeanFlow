from functools import partial
import flax.nnx as nnx

import jax
from jax.lax import stop_gradient
from jax import jvp
import jax.numpy as jnp

from src.experiments.flowmap_learning.mean_flow import MeanFlowTrainer
from src.utils import *
from src.factory import *
from src.eval import *


def gfm_esd(model, batch, loss_p=0.0):
    path, (t, s), cond = batch
    x_t = path.x_t
    v_t = path.dx_t
    manifold = model.manifold

    u_t = model.instant_velocity(x_t, t, **cond)

    # Flow matching loss
    loss = manifold.square_norm_at(x_t, u_t - v_t)

    # ESD loss
    flow_fn_t = lambda _t: model.forward_flow(x_t, _t, s, **cond)

    x_s, dxs_dt = jvp(flow_fn_t, (t,), (jnp.ones_like(t),))

    flow_fn = lambda _x: model.forward_flow(_x, t, s, **cond)
    dX_s = jvp(flow_fn, (x_t,), (u_t,))[1]
    dX_s = manifold.proju(x_s, dX_s)

    # dX_s = manifold.proju(x_t, u_t)
    # This is the version used in the paper, but not correct.

    error = dxs_dt + jax.lax.stop_gradient(dX_s)
    loss += manifold.square_norm_at(x_s, error)

    return weighted_loss(loss, loss_p), t_stratified_loss(t, loss)


def gfm_lsd(model, batch, loss_p=0.0):
    path, (s, t), cond = batch
    x_s = path.x_t
    v_s = path.dx_t
    manifold = model.manifold

    # Flow matching loss
    u_s = model.instant_velocity(x_s, s, **cond)
    loss = manifold.square_norm_at(x_s, u_s - v_s)

    # LSD loss
    flow_fn_t = lambda _t: model.forward_flow(x_s, s, _t, **cond)

    x_t, dxt_dt = jvp(flow_fn_t, (t,), (jnp.ones_like(t),))
    v_t = model.instant_velocity(x_t, t, **cond)

    error = dxt_dt - stop_gradient(v_t)
    loss += weighted_loss(manifold.square_norm_at(x_t, error), loss_p)

    return loss.mean(), t_stratified_loss(t, loss)


def gfm_lsd_with_sg(model, batch, loss_p=0.0):
    path, (s, t), cond = batch
    x_s = path.x_t
    v_s = path.dx_t
    manifold = model.manifold

    # Flow matching loss
    u_s = model.instant_velocity(x_s, s, **cond)
    loss = manifold.square_norm_at(x_s, u_s - v_s)

    # LSD loss
    # partial_t x_st = exp_{x_s}{(t-s)*u_st} = d(exp_{x_s})_{(t-s)*u_st}[u_st + (t-s)*partial_t_u_st]
    # flow_fn_t = lambda _t: model.forward_flow(x_s, s, _t, **cond)

    u, partial_t_u = jvp(
        model.avg_velocity, (x_s, s, t), (u_s, jnp.zeros_like(s), jnp.ones_like(t))
    )
    t_minus_s = expand_to(t - s, x_s.ndim)
    x_t = manifold.exp_map(x_s, t_minus_s * u)

    def Dexp(v):
        def exp_fn(w):
            return manifold.exp_map(x_s, w)

        return jvp(exp_fn, (t_minus_s * stop_gradient(u),), (v,))[1]

    dxt_dt = Dexp(u_s + t_minus_s * stop_gradient(partial_t_u))
    dxt_dt = manifold.proju(x_t, dxt_dt)
    v_t = model.instant_velocity(x_t, t, **cond)

    error = dxt_dt - stop_gradient(v_t)
    loss += weighted_loss(manifold.square_norm_at(x_t, error), loss_p)

    return loss.mean(), t_stratified_loss(t, loss)


class GFM_ESD_Trainer(MeanFlowTrainer):

    def setup(self):
        cfg = self.cfg

        self.init_components()

        self.loss_fn = partial(gfm_esd, loss_p=self.cfg.get("loss_p", 0.0))
        self.train_step = nnx.jit(make_train_step(self.loss_fn, has_aux=True))

        metrics = {"loss": nnx.metrics.Average("loss")}
        metrics.update(create_stratified_metric(return_as_dict=True))
        self.train_loss = nnx.metrics.MultiMetric(**metrics)
        self.val_loss = nnx.metrics.Average("loss")
        self.checkpoint = create_checkpoint(cfg.checkpoint)
        self.set_eval_fn()


class GFM_LSD_Trainer(MeanFlowTrainer):

    def setup(self):
        cfg = self.cfg

        self.init_components()

        self.loss_fn = partial(gfm_lsd, loss_p=self.cfg.get("loss_p", 0.0))
        self.train_step = nnx.jit(make_train_step(self.loss_fn, has_aux=True))

        metrics = {"loss": nnx.metrics.Average("loss")}
        metrics.update(create_stratified_metric(return_as_dict=True))
        self.train_loss = nnx.metrics.MultiMetric(**metrics)
        self.val_loss = nnx.metrics.Average("loss")
        self.checkpoint = create_checkpoint(cfg.checkpoint)
        self.set_eval_fn()
