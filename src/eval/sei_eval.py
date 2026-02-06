# Adaptation from: https://github.com/HannesStark/dirichlet-flow-matching/blob/main/lightning_modules/promoter_module.py
import re
import pandas as pd

import jax
from jax import Array, numpy as jnp, device_put, device_get
import flax.nnx as nnx
import orbax.checkpoint as ocp
from pathlib import Path
from typing import Callable

from .sei import Sei, NonStrandSpecific
from ..manifold import sphere_to_simplex


def upgrade_state_dict(state_dict, prefixes=["encoder.sentence_encoder.", "encoder."]):
    """Removes prefixes 'model.encoder.sentence_encoder.' and 'model.encoder.'."""
    pattern = re.compile("^" + "|".join(prefixes))
    state_dict = {pattern.sub("", name): param for name, param in state_dict.items()}
    return state_dict


def put_model(model, device):
    state = nnx.state(model)
    new_state = jax.device_put(state, device)
    nnx.update(model, new_state)


class SeiEval:
    """SEI evaluation class."""

    ckpt_path = Path("data/promoter_sei/best_promoter_sei").absolute()

    def __init__(self):
        self._sei = NonStrandSpecific(Sei(4096, 21907, nnx.Rngs(params=0, dropout=0)))
        self._sei_features = pd.read_csv(
            "data/promoter_sei/target.sei.names",
            sep="|",
            header=None,
        )
        self._sei_cache = {}

        self.load_sei_model()
        self.to_device("cpu")

    def load_sei_model(self):
        # Load the SEI model from the checkpoint
        checkpoint = ocp.StandardCheckpointer()

        abstract_model = nnx.eval_shape(lambda: self._sei)
        graphdef, _ = nnx.split(abstract_model)
        state = checkpoint.restore(self.ckpt_path)
        self._sei = nnx.merge(graphdef, state)

        self._sei.eval()

    def to_device(self, device: str):
        if device not in ("cpu", "gpu"):
            raise ValueError(f"Invalid device: {device}")

        devs = jax.devices(device)
        if not devs:
            raise RuntimeError(f"No {device.upper()} device available.")

        put_model(self._sei, devs[0])

    def get_pure_model(self):
        """Return split model (graphdef, state) for use in JAX transformations."""
        return nnx.split(self._sei)

    def _prepare_sei_input(self, seq_one_hot: Array) -> Array:
        """Prepare input for SEI model."""
        B, _, _ = seq_one_hot.shape
        sei_inp = jnp.concatenate(
            [
                jnp.ones((B, 4, 1536)) * 0.25,
                seq_one_hot.transpose((0, 2, 1)),
                jnp.ones((B, 4, 1536)) * 0.25,
            ],
            2,
        )
        return sei_inp.transpose((0, 2, 1))

    def _postprocess_sei_output(self, sei_out: Array) -> Array:
        """Extract H3K4me3 predictions from SEI output."""
        sei_out = sei_out[
            :, self._sei_features[1].str.strip().values == "H3K4me3"
        ]
        return sei_out.mean(axis=1)

    def get_sei_profile_pure(self, seq_one_hot: Array, graphdef, state) -> tuple[Array, Array]:
        """
        Pure functional version of get_sei_profile for use inside JAX transformations.

        Use with get_pure_model() to get graphdef and state before entering traced context.
        """
        sei_inp = self._prepare_sei_input(seq_one_hot)
        model = nnx.merge(graphdef, state)
        sei_out, feat = model(sei_inp, return_features=True)
        predh3k4me3 = self._postprocess_sei_output(sei_out)
        return predh3k4me3, feat

    def get_sei_profile(self, seq_one_hot: Array) -> Array:
        """
        Get the SEI profile from the one-hot encoded sequence.

        Parameters:
            - `seq_one_hot`: The one-hot encoded sequence tensor.

        Returns:
            The SEI profile tensor.
        """
        B, _, _ = seq_one_hot.shape
        sei_inp = jnp.concatenate(
            [
                jnp.ones((B, 4, 1536)) * 0.25,
                seq_one_hot.transpose((0, 2, 1)),
                jnp.ones((B, 4, 1536)) * 0.25,
            ],
            2,
        )  # batchsize x 4 x 4,096
        sei_out, feat = self._sei(
            sei_inp.transpose((0, 2, 1)), return_features=True
        )  # batchsize x 21,907

        sei_out = sei_out[
            :, self._sei_features[1].str.strip().values == "H3K4me3"
        ]  # batchsize x 2,350
        predh3k4me3 = sei_out.mean(axis=1)  # batchsize
        return predh3k4me3, feat

    def eval_sp_mse(
        self,
        seq_one_hot: Array,
        target: Array,
        b_index: int | None = None,
        return_features: bool = False,
    ) -> Array:
        """
        Evaluate the mean squared error of the SEI profile prediction.

        Parameters:
            - `seq_one_hot`: The one-hot encoded sequence tensor.
            - `target`: The target tensor;
            - `b_index`: The batch index of the target Tensor; avoids recalculating
                the profile all the time; if `None` always calculates profile (useful
                for testing).

        Returns:
            The mean squared error tensor.
        """
        if b_index is not None and b_index in self._sei_cache:
            target_prof, target_feat = self._sei_cache[b_index]
            target_prof, target_feat = device_put(target_prof), device_put(target_feat)
        else:
            target_prof, target_feat = self.get_sei_profile(target)
            self._sei_cache[b_index] = (
                device_get(target_prof),
                device_get(target_feat),
            )
        pred_prof, pred_feat = self.get_sei_profile(seq_one_hot)
        mse = (pred_prof - target_prof) ** 2

        if return_features:
            return mse, (pred_feat, target_feat)
        else:
            return mse

    def create_reward_fn(self, target: Array = None, signal=None) -> Callable[[Array], Array]:
        """
        Create a pure reward function for use with JAX grad (e.g., reward guidance).

        The returned function takes sphere-representation samples and returns
        per-sample H3K4me3 predictions as rewards. If target_signal is provided,
        returns negative MSE (to maximize similarity to target).

        Args:
            target_signal: Optional target H3K4me3 signal to match. If provided,
                the reward is -MSE(pred, target). If None, reward is the raw
                H3K4me3 prediction (higher = better promoter activity).

        Returns:
            A pure function reward_fn(x_sphere) -> rewards of shape (batch_size,)
        """
        graphdef, state = self.get_pure_model()

        # Pre-compute target profile if provided
        if target is not None:
            target_profile, _ = self.get_sei_profile_pure(target, graphdef, state)
        else:
            target_profile = None
        

        def reward_fn(seq_one_hot: Array) -> Array:
            # Convert sphere to simplex (soft probabilities for differentiability)
            #x_simplex = sphere_to_simplex(x_sphere)
            # Simplex is (batch, seq_len, 4), need one-hot format (batch, seq_len, 4)
            #seq_one_hot = x_simplex

            # Get H3K4me3 prediction
            pred_profile, _ = self.get_sei_profile_pure(seq_one_hot, graphdef, state)

            if target is not None:
                # Negative MSE: maximize to minimize distance to target
                reward = -((pred_profile - target_profile) ** 2)
            else:
                # Raw H3K4me3 prediction: maximize promoter activity
                reward = pred_profile

            return reward

        return reward_fn
