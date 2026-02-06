from functools import partial
from typing import Tuple, List

import jax.numpy as jnp
from jax import Array

import flax.nnx as nnx
from torch.utils.data import DataLoader
from hydra.utils import instantiate

from src.experiments.base_class import Experiment
from src.probability_path import GeodesicProbPath
from src.utils import *
from src.factory import *
from src.eval import *
from src.manifold import sphere_to_label, sphere_to_simplex
from src.utils.ema_model import ema_step

from .losses import *


class MeanFlowTrainer(Experiment):

    def setup(self):
        cfg = self.cfg

        self.init_components()

        if self.cfg.loss_type == "meanflow_self_distillation":
            self.loss_fn = partial(
                meanflow_loss_self_distillation,
                loss_p=self.cfg.loss_p,
                tangent_clip=self.cfg.tangent_clip,
                partial_sg=self.cfg.get("partial_sg", False),
            )
        else:
            self.loss_fn = partial(
                meanflow_loss,
                loss_p=self.cfg.loss_p,
                tangent_clip=self.cfg.tangent_clip,
                loss_down_scale=self.cfg.get("loss_down_scale", False),
                t_clip=self.cfg.get("t_clip", 0.1),
            )
        self.train_step = nnx.jit(make_train_step(self.loss_fn, has_aux=True))

        self.train_loss = create_loss_metric(self.cfg.loss_type)
        self.val_loss = nnx.metrics.Average("loss")
        self.checkpoint = create_checkpoint(cfg.checkpoint)

        self.set_eval_fn()

    def init_components(self):
        cfg = self.cfg

        manifold = instantiate(cfg.manifold)
        prior = UniformPrior(manifold, self.rngs())
        prob_path = GeodesicProbPath(manifold)
        time_sampler = create_time_sampler(cfg.time_sampler, self.rngs())

        self.manifold = manifold
        self.prior = prior
        self.prob_path = prob_path
        self.time_sampler = time_sampler

        self.model = init_flow_map_model(
            cfg.model, manifold, cfg.x_shape, seed=self.rngs()
        )
        iter_per_epoch = len(self.train_dataset) // cfg.batch_size
        self.optimizer = init_optim(cfg.optim, self.model, iter_per_epoch)

    def set_eval_fn(self):
        # Pick the appropriate eval function based on the dataset
        if is_promoter_dataset(self.train_dataset):
            # create_promoter_metrics(trainer=self)
            # self.eval_epoch = partial(promoter_eval_fn, trainer=self)
            self.eval_epoch = PromoterEvalFunction(trainer=self)
            if self.cfg.eval.eval_sei:
                self.sei = SeiEval()
        elif is_toy_dataset(self.train_dataset):
            self.eval_data = self.train_dataset[: self.cfg.eval.num_samples]
            self.eval_epoch = self.toy_eval_fn
        elif is_toy_helix_dataset(self.train_dataset):
            self.eval_data = self.train_dataset[: self.cfg.eval.num_samples]
            self.eval_epoch = partial(toy_helix_eval_fn, trainer=self)
        elif is_earth_dataset(self.train_dataset):
            self.eval_data = jnp.array(self.val_dataset[: len(self.val_dataset)])
            self.eval_epoch = partial(earth_eval_fn, trainer=self)
        elif is_amino_dataset(self.train_dataset):
            self.eval_data = jnp.array(self.val_dataset[: len(self.val_dataset)])
            self.eval_epoch = partial(amino_eval_fn, trainer=self)
        else:
            raise ValueError(f"Unknown dataset: {type(self.train_dataset)}")

    def train_epoch(self):
        for batch in train_iter(self.train_dataloader):
            batch = self.prepare_batch(batch)
            loss, grad_norm = self.train_step(
                self.model, self.optimizer, self.train_loss, batch
            )

            if self.do_ema:
                ema_step(self.model, self.ema_optimizer)

            # This block executes only when debug option is set.
            if self.logger.debug:
                self.logger.log_for_debug(self.epoch, loss, grad_norm)
                self.detect_anomlay_and_save_ckpt(self.epoch, loss, grad_norm, batch)

        self.logger.log_loss(self.epoch, self.train_loss, "train")

    def prepare_batch(self, data):
        if is_promoter_dataset(self.train_dataset):
            # Conditional case
            x_1, signal = data
            x_1, signal = prepare_promoter_data(x_1, signal)
            cond = {"signal": signal}
        elif is_toy_dataset(self.train_dataset):
            # Unconditional case
            x_1 = data
            x_1 = jnp.asarray(x_1)
            cond = {}
        elif is_toy_helix_dataset(self.train_dataset):
            # Unconditional case
            x_1 = data
            x_1 = jnp.asarray(x_1)
            cond = {}
        elif is_earth_dataset(self.train_dataset):
            x_1 = data
            x_1 = jnp.asarray(x_1)
            cond = {}
        elif is_amino_dataset(self.train_dataset):
            x_1 = data
            x_1 = jnp.asarray(x_1)
            cond = {}
        else:
            raise ValueError(f"Unknown dataset: {type(self.train_dataset)}")

        x_0 = self.prior.sample(*x_1.shape)
        t, s = self.time_sampler.sample(x_1.shape[0])
        path = self.prob_path.sample(x_0, x_1, t)

        return (path, (t, s), cond)

    def detect_anomlay_and_save_ckpt(self, epoch, loss, grad_norm, batch):
        self.check_nan_in_grad_norm(epoch, loss, grad_norm, batch)
        self.check_anomaly_grad_norm(epoch, loss, grad_norm, batch)

    def check_nan_in_grad_norm(self, epoch, loss, grad_norm, batch):
        # This line checks the value of grad_norm, blocking the dispatching of the next batch.
        # So this block incur a significant overhead. Only use it in debug mode.
        if jnp.isnan(grad_norm):
            print("Nan in grad_norm")
            self.logger.checkpoint_and_exit(
                epoch,
                self.model,
                self.optimizer,
                loss=loss,
                grad_norm=grad_norm,
                batch=batch,
            )

    def check_anomaly_grad_norm(self, epoch, loss, grad_norm, batch):
        threshold = 1000
        if epoch > 30 and grad_norm > threshold:
            print("Large value in grad_norm")
            self.logger.checkpoint_and_exit(
                epoch,
                self.model,
                self.optimizer,
                loss=loss,
                grad_norm=grad_norm,
                batch=batch,
            )

    def toy_eval_fn(self):

        if not should_eval(self.epoch, self.cfg.eval.interval):
            return

        # Generate samples from model
        x0 = self.prior.sample(self.cfg.eval.num_samples, *self.cfg.x_shape)
        gen_x1 = self.sample_x1(self.model, x0)
        xts = self.sample_trajectories(self.model, x0)

        # Compute L2 distance
        l2_metric = compute_histogram_l2(
            sphere_to_label(gen_x1), sphere_to_label(self.eval_data)
        )

        self.logger.log_loss(self.epoch, {"l2": l2_metric}, "val")

        # Draw plots and log them
        hist_fig = plot_token_histogram(sphere_to_label(gen_x1))
        simplex_fig = plot_on_simplex(sphere_to_simplex(gen_x1))
        xts_fig = plot_xts_on_simplex(xts, self.prior, self.prob_path, self.eval_data)

        self.logger.log_figures(
            self.epoch,
            {
                "token_histogram": hist_fig,
                "simplex_scatter": simplex_fig,
                **xts_fig,
            },
        )

    @staticmethod
    @nnx.jit
    def compute_val_loss(model: nnx.Module, val_metric: nnx.metrics.Metric, batch):
        loss = meanflow_loss(model, batch, return_stratified_loss=False)
        val_metric.update(loss=loss)

    @staticmethod
    @nnx.jit
    def sample_trajectories(model: nnx.Module, x_0: Array, **kwargs):
        num_eval_samples = x_0.shape[0]
        t0 = jnp.zeros(num_eval_samples)

        xts = []
        for t in jnp.linspace(0, 1.0, 5):
            xt_sample = model.forward_flow(
                x_0, t0, jnp.ones(num_eval_samples) * t, **kwargs
            )
            xts.append(xt_sample)

        return xts

    @staticmethod
    @nnx.jit
    def sample_x1_with_integration(model: nnx.Module, x_0: Array, **kwargs):
        return model.sample_x1_with_integration(x_0, **kwargs)

    @staticmethod
    @nnx.jit
    def sample_x1(model: nnx.Module, x_0: Array, **kwargs):
        return model.sample_x1(x_0, **kwargs)

    @staticmethod
    @nnx.jit(static_argnames=["n_steps"])
    def sample_x1_with_n_steps(model: nnx.Module, x_0: Array, n_steps: int, **kwargs):
        return model.sample_x1_with_n_steps(x_0, n_steps, **kwargs)


class MeanFlowTrainerWithJVPBP(MeanFlowTrainer):

    def setup(self):
        cfg = self.cfg

        self.init_components()

        self.loss_fn = partial(
            meanflow_loss_with_jvp_bp,
            loss_p=self.cfg.loss_p,
            tangent_clip=self.cfg.tangent_clip,
        )
        self.train_step = nnx.jit(make_train_step(self.loss_fn, has_aux=True))

        self.train_loss = create_loss_metric(self.cfg.loss_type)
        self.val_loss = nnx.metrics.Average("loss")
        self.checkpoint = create_checkpoint(cfg.checkpoint)

        if is_promoter_dataset(self.train_dataset):
            self.create_promoter_metrics()

        self.set_eval_fn()
