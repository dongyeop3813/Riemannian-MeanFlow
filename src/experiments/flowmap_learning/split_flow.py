from functools import partial

import flax.nnx as nnx
from src.factory import *
from src.utils import *

from .losses import *
from .mean_flow import MeanFlowTrainer


class SplitFlowTrainer(MeanFlowTrainer):

    def setup(self):
        cfg = self.cfg

        self.init_components()

        self.loss_fn = partial(
            split_flow_loss,
            loss_p1=self.cfg.loss_p1,
            loss_p2=self.cfg.loss_p2,
            weight=self.cfg.loss_weight,
            v_loss_down_scale=self.cfg.v_loss_down_scale,
            split_loss_down_scale=self.cfg.split_loss_down_scale,
            split_loss_type=self.cfg.split_loss_type,
        )
        self.train_step = nnx.jit(make_train_step(self.loss_fn, has_aux=True))

        self.train_loss = create_loss_metric(self.cfg.loss_type)
        self.val_loss = nnx.metrics.Average("loss")
        self.checkpoint = create_checkpoint(cfg.checkpoint)
        self.set_eval_fn()

    def prepare_batch(self, data):
        if is_promoter_dataset(self.train_dataset):

            x_1, signal = data
            x_1, signal = prepare_promoter_data(x_1, signal)
            x_0 = self.prior.sample(*x_1.shape)
            t, s, r = self.time_sampler.sample(x_1.shape[0])
            path = self.prob_path.sample(x_0, x_1, t)

            return (path, (t, s, r), {"signal": signal})

        elif is_toy_dataset(self.train_dataset):

            x_1 = data
            x_1 = jnp.asarray(x_1)
            x_0 = self.prior.sample(*x_1.shape)
            t, s, r = self.time_sampler.sample(x_1.shape[0])
            path = self.prob_path.sample(x_0, x_1, t)

            return (path, (t, s, r), {})

        elif is_toy_helix_dataset(self.train_dataset):
            x_1 = data
            x_1 = jnp.asarray(x_1)
            x_0 = self.prior.sample(*x_1.shape)
            t, s, r = self.time_sampler.sample(x_1.shape[0])
            path = self.prob_path.sample(x_0, x_1, t)

            return (path, (t, s, r), {})

        elif is_earth_dataset(self.train_dataset):
            x_1 = data
            x_1 = jnp.asarray(x_1)
            x_0 = self.prior.sample(*x_1.shape)
            t, s, r = self.time_sampler.sample(x_1.shape[0])
            path = self.prob_path.sample(x_0, x_1, t)

            return (path, (t, s, r), {})

        elif is_amino_dataset(self.train_dataset):
            x_1 = data
            x_1 = jnp.asarray(x_1)
            x_0 = self.prior.sample(*x_1.shape)
            t, s, r = self.time_sampler.sample(x_1.shape[0])
            path = self.prob_path.sample(x_0, x_1, t)

            return (path, (t, s, r), {})
        else:
            raise ValueError(f"Unknown dataset: {type(self.train_dataset)}")

    @staticmethod
    @nnx.jit
    def compute_val_loss(model: nnx.Module, val_metric: nnx.metrics.Metric, batch):
        loss = split_flow_loss(model, batch, return_stratified_loss=False)
        val_metric.update(loss=loss)


class SplitFlowTrainerV2(MeanFlowTrainer):
    # Split flow loss with stochastic choice of loss type
    # loss_prob chance of using flow matching loss
    # (1 - loss_prob) chance of using semigroup identity loss

    # Return the train loss function and metric object to track it.

    def setup(self):
        cfg = self.cfg

        self.flow_loss_fn = partial(
            flow_matching_loss,
            loss_p=self.cfg.loss_p1,
        )
        self.semigroup_loss_fn = partial(
            semigroup_identity_loss,
            loss_p=self.cfg.loss_p2,
        )
        self.fm_step = nnx.jit(make_train_step(self.flow_loss_fn, has_aux=True))
        self.semigroup_step = nnx.jit(
            make_train_step(self.semigroup_loss_fn, has_aux=True)
        )

        self.flow_loss = nnx.metrics.MultiMetric(
            loss=nnx.metrics.Average("loss"),
            **create_stratified_metric("flow_", return_as_dict=True),
        )
        self.semigroup_loss = nnx.metrics.MultiMetric(
            loss=nnx.metrics.Average("loss"),
            **create_stratified_metric("semigroup_", return_as_dict=True),
        )

        self.val_loss = nnx.metrics.Average("loss")
        self.checkpoint = create_checkpoint(cfg.checkpoint)

        if is_promoter_dataset(self.train_dataset):
            self.create_promoter_metrics()

        self.set_eval_fn()

    def init_train_components(self):
        super().init_train_components()
        self.FM_time_sampler = create_time_sampler(
            self.cfg.FM_time_sampler, self.rngs()
        )

    def train(self):
        for epoch in epoch_iter(self.epoch, self.cfg.num_epochs):
            self.epoch = epoch
            self.checkpoint.save(epoch, self.model, self.optimizer)

            self.model.train()
            for batch in train_iter(self.train_dataloader):
                self.step(batch)

            fm_loss_val = self.flow_loss.compute()
            semi_loss_val = self.semigroup_loss.compute()
            self.logger.log_loss(epoch, fm_loss_val, "train")
            self.logger.log_loss(epoch, semi_loss_val, "train")

            self.model.eval()
            self.eval_fn()

            self.reset_metrics()

    def step(self, batch):
        do_FM = self.choice_fn()
        if do_FM:
            batch = self.prepare_batch(batch, method="flow")
            loss, grad_norm = self.fm_step(
                self.model, self.optimizer, self.flow_loss, batch
            )
        else:
            batch = self.prepare_batch(batch, method="semigroup")
            loss, grad_norm = self.semigroup_step(
                self.model, self.optimizer, self.semigroup_loss, batch
            )

        # This block executes only when debug option is set.
        if self.logger.debug:
            self.logger.log_for_debug(self.epoch, loss, grad_norm)
            self.check_nan_in_grad_norm(self.epoch, loss, grad_norm, batch)

    def prepare_batch(self, data, method="semigroup"):
        if is_promoter_dataset(self.train_dataset):
            x_1, signal = data
            x_1, signal = prepare_promoter_data(x_1, signal)
            cond = {"signal": signal}
        elif is_toy_dataset(self.train_dataset):
            x_1 = data
            x_1 = jnp.asarray(x_1)
            cond = {}
        elif is_earth_dataset(self.train_dataset):
            x_1 = data
            x_1 = jnp.asarray(x_1)
            cond = {}
        else:
            raise ValueError(f"Unknown dataset: {type(self.train_dataset)}")

        x_0 = self.prior.sample(*x_1.shape)

        if method == "semigroup":
            t, s, r = self.time_sampler.sample(x_1.shape[0])
            path = self.prob_path.sample(x_0, x_1, t)
            times = (t, s, r)
        elif method == "flow":
            t = self.FM_time_sampler.sample(x_1.shape[0])
            path = self.prob_path.sample(x_0, x_1, t)
            times = t
        else:
            raise ValueError(f"Unknown method: {method}")

        return (path, times, cond)

    def choice_fn(self):
        return jax.random.bernoulli(self.rngs(), self.cfg.loss_prob)

    @staticmethod
    @nnx.jit
    def compute_val_loss(model: nnx.Module, val_metric: nnx.metrics.Metric, batch):
        loss = split_flow_loss(model, batch, return_stratified_loss=False)
        val_metric.update(loss=loss)
