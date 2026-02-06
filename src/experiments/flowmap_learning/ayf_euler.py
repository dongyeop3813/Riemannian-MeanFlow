from functools import partial

import flax.nnx as nnx
from src.factory import *
from src.utils import *

from .losses import *
from .mean_flow import MeanFlowTrainer


class AYFEulerTrainer(MeanFlowTrainer):

    def setup(self):
        cfg = self.cfg

        self.init_components()

        self.loss_fn = partial(
            AYF_EMD_objective,
            loss_p=cfg.loss_p,
            tangent_clip=cfg.tangent_clip,
        )
        self.train_step = nnx.jit(make_train_step(self.loss_fn, has_aux=True))

        self.train_loss = create_loss_metric(cfg.loss_type)
        self.val_loss = nnx.metrics.Average("loss")
        self.checkpoint = create_checkpoint(cfg.checkpoint)

        if is_promoter_dataset(self.train_dataset):
            self.create_promoter_metrics()

        self.set_eval_fn()

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
        else:
            raise ValueError(f"Unknown dataset: {type(self.train_dataset)}")

        x_0 = self.prior.sample(*x_1.shape)
        s, t = self.time_sampler.sample(x_1.shape[0])
        path = self.prob_path.sample(x_0, x_1, s)

        return (path, (s, t), cond)

    @staticmethod
    @nnx.jit
    def compute_val_loss(model: nnx.Module, val_metric: nnx.metrics.Metric, batch):
        loss = AYF_EMD_objective(model, batch, return_stratified_loss=False)
        val_metric.update(loss=loss)
