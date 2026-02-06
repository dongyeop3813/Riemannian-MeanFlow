# Fisher flow matching for promoter DNA generation
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
from src.manifold import sphere_to_onehot


def flow_matching_loss(model, batch, loss_p=0.0, return_stratified_loss=True):
    path, t, cond = batch

    u_t = model.instant_velocity(path.x_t, t, **cond)

    loss = model.manifold.square_norm_at(path.x_t, u_t - path.dx_t)

    if return_stratified_loss:
        return weighted_loss(loss, loss_p), t_stratified_loss(t, loss, prefix="flow_")
    else:
        return weighted_loss(loss, loss_p)


class FlowMatchingTrainer(Experiment):

    def setup(self):
        cfg = self.cfg

        self.init_components()

        self.loss_fn = partial(flow_matching_loss, loss_p=self.cfg.loss_p)
        self.train_step = nnx.jit(make_train_step(self.loss_fn, has_aux=True))

        self.train_loss = nnx.metrics.MultiMetric(
            loss=nnx.metrics.Average("loss"),
            **create_stratified_metric(prefix="flow_", return_as_dict=True),
        )
        self.val_loss = nnx.metrics.Average("loss")
        self.checkpoint = create_checkpoint(cfg.checkpoint)

        # Pick the appropriate eval function based on the dataset
        if is_promoter_dataset(self.train_dataset):
            self.eval_epoch = PromoterEvalFunction(trainer=self)
            if self.cfg.eval.eval_sei:
                self.sei = SeiEval()
        elif is_toy_dataset(self.train_dataset):
            self.eval_data = self.train_dataset[: cfg.eval.num_samples]
            self.eval_epoch = self.toy_eval_fn
        elif is_amino_dataset(self.train_dataset):
            self.eval_data = jnp.array(self.val_dataset[: len(self.val_dataset)])
            self.eval_epoch = partial(amino_eval_fn, trainer=self)
        else:
            raise ValueError(f"Unknown dataset: {type(self.train_dataset)}")

    def init_components(self):
        cfg = self.cfg

        self.manifold = instantiate(cfg.manifold)
        self.prior = UniformPrior(self.manifold, self.rngs())
        self.prob_path = GeodesicProbPath(self.manifold)
        self.time_sampler = create_time_sampler(cfg.time_sampler, self.rngs())

        self.model = init_flow_map_model(
            cfg.model, self.manifold, cfg.x_shape, seed=self.rngs()
        )
        iter_per_epoch = len(self.train_dataset) // cfg.batch_size
        self.optimizer = init_optim(cfg.optim, self.model, iter_per_epoch)

    def train_epoch(self):
        for batch in train_iter(self.train_dataloader):
            batch = self.prepare_batch(batch)
            loss, grad_norm = self.train_step(
                self.model, self.optimizer, self.train_loss, batch
            )

            # This block executes only when debug option is set.
            if self.logger.debug:
                self.logger.log_for_debug(self.epoch, loss, grad_norm)
                self.check_nan_in_grad_norm(self.epoch, loss, grad_norm, batch)

        self.logger.log_loss(self.epoch, self.train_loss, "train")

    def prepare_batch(self, data):
        if is_promoter_dataset(self.train_dataset):
            x_1, signal = data
            x_1, signal = prepare_promoter_data(x_1, signal)
            cond = {"signal": signal}
        elif is_toy_dataset(self.train_dataset):
            x_1 = data
            x_1 = jnp.asarray(x_1)
            cond = {}
        elif is_toy_helix_dataset(self.train_dataset):
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
        t = self.time_sampler.sample(x_1.shape[0])
        path = self.prob_path.sample(x_0, x_1, t)
        return (path, t, cond)

    def check_nan_in_grad_norm(self, epoch, loss, grad_norm, batch):
        # This line checks the value of grad_norm, blocking the dispatching of the next batch.
        # So this block incur a significant overhead. Only use it in debug mode.
        if jnp.isnan(grad_norm):
            self.logger.checkpoint_and_exit(
                epoch,
                self.model,
                self.optimizer,
                loss=loss,
                grad_norm=grad_norm,
                batch=batch,
            )

    def toy_eval_fn(self):
        from src.manifold import sphere_to_label, sphere_to_simplex

        if not should_eval(self.epoch, self.cfg.eval.interval):
            return

        # Generate samples from model
        x0 = self.prior.sample(self.cfg.eval.num_samples, *self.cfg.x_shape)
        gen_x1 = self.sample_x1(self.model, x0)

        # Compute L2 distance
        l2_metric = compute_histogram_l2(
            sphere_to_label(gen_x1), sphere_to_label(self.eval_data)
        )

        self.logger.log_loss(self.epoch, {"l2": l2_metric}, "val")

        # Draw plots and log them
        hist_fig = plot_token_histogram(sphere_to_label(gen_x1))
        simplex_fig = plot_on_simplex(sphere_to_simplex(gen_x1))

        self.logger.log_figures(
            self.epoch,
            {"token_histogram": hist_fig, "simplex_scatter": simplex_fig},
        )

    @staticmethod
    @nnx.jit
    def compute_val_loss(model: nnx.Module, val_metric: nnx.metrics.Metric, batch):
        # flow matching loss with no adaptive weighting
        # No weighting allows us to compare runs with different loss_p
        loss = flow_matching_loss(model, batch, return_stratified_loss=False)
        val_metric.update(loss=loss)

    @staticmethod
    @nnx.jit
    def sample_x1(model: nnx.Module, x_0: Array, **kwargs):
        return model.sample_x1(x_0, **kwargs)

    @staticmethod
    @nnx.jit(static_argnames=["n_steps"])
    def sample_x1_with_n_steps(model: nnx.Module, x_0: Array, n_steps: int, **kwargs):
        return model.integrate(x_0, inference_steps=n_steps, **kwargs)
