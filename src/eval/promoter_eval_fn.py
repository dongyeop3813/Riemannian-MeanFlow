import shutil
from pathlib import Path
from hydra.core.hydra_config import HydraConfig

from src.utils import *
from src.manifold import *
from src.experiments.base_class import Experiment
from src.eval.fbd import FBD
from src.eval.kmer_freq import KmerFreq, compute_kmer_freq

import flax.nnx as nnx


class CategoricalEntropy(nnx.metrics.Metric):
    """Compute average entropy of categorical distribution over sequence positions.

    Treats input as probs,
    then computes entropy at each position and averages over the sequence.
    """

    def __init__(self):
        self.total_entropy = 0.0
        self.n_samples = 0

    def update(self, probs):
        """Update with probs.

        Args:
            probs: Probs of shape (B, L, 4)
        """
        probs = np.asarray(probs)

        # Compute entropy at each position: H = -sum(p * log2(p))
        eps = 1e-6
        entropy_per_pos = -np.sum(probs * np.log2(probs + eps), axis=-1)  # (B, L)

        # Average entropy over sequence positions for each sample: (B,)
        entropy_per_sample = entropy_per_pos.mean(axis=-1)

        # Accumulate batch sum and count
        self.total_entropy += entropy_per_sample.sum()
        self.n_samples += probs.shape[0]

    def compute(self):
        """Compute batch-averaged entropy.

        Returns:
            Average entropy in bits (log2)
        """
        if self.n_samples == 0:
            return 0.0
        return float(self.total_entropy / self.n_samples)

    def reset(self):
        self.total_entropy = 0.0
        self.n_samples = 0


class PromoterEvalFunction:
    def __init__(self, trainer: Experiment):
        self.trainer = trainer
        self.create_metrics()
        self.tmp_dir = Path(HydraConfig.get().runtime.output_dir) / "tmp_eval_dir"

    def create_metrics(self):
        self.metrics = {}
        for n_steps in [1, 2, 4, 8, 100]:
            self.metrics[f"{n_steps}_step_kmer_corr"] = KmerFreq()
            self.metrics[f"{n_steps}_step_entropy"] = CategoricalEntropy()

        if self.trainer.cfg.eval.eval_sei:
            for n_steps in [1, 2, 4, 8, 100]:
                self.metrics[f"{n_steps}_step_mse"] = nnx.metrics.Average()
                self.metrics[f"{n_steps}_step_fbd"] = FBD()

    def clear_tmp_dir(self):
        if self.tmp_dir.exists():
            shutil.rmtree(self.tmp_dir)

    def clear_metrics(self):
        for metric in self.metrics.values():
            metric.reset()

    def metric_values(self):
        metrics = {}
        should_eval_sample_metrics = should_eval(
            self.trainer.epoch, self.trainer.cfg.eval.sample_every
        )

        for k, v in self.metrics.items():
            skip_mse = k.endswith("mse")
            skip_mse = skip_mse and not (
                should_eval_sample_metrics and self.trainer.cfg.eval.eval_sei
            )
            skip_fbd = k.endswith("fbd") and not should_eval_sample_metrics
            skip_kmer_corr = k.endswith("kmer_corr") and not should_eval_sample_metrics
            skip_entropy = k.endswith("entropy") and not should_eval_sample_metrics
            if skip_mse or skip_fbd or skip_kmer_corr or skip_entropy:
                continue

            metrics[k] = v.compute()
        return metrics

    def sample_with_n_steps(self, n_steps: int, x_0: Array, **cond):
        trainer = self.trainer
        x1 = trainer.sample_x1_with_n_steps(trainer.model, x_0, n_steps, **cond)
        return x1

    def __call__(self):
        trainer = self.trainer
        cfg = trainer.cfg

        should_eval_sample_metrics = should_eval(trainer.epoch, cfg.eval.sample_every)

        # Create tmp_dir only if SEI evaluation is needed
        if should_eval_sample_metrics and cfg.eval.eval_sei:
            self.tmp_dir.mkdir(parents=True, exist_ok=True)

        self.eval_val_loss_and_generate_samples(do_sample=should_eval_sample_metrics)

        if should_eval_sample_metrics and cfg.eval.eval_sei:
            self.eval_sei_metrics()
            self.clear_tmp_dir()

        trainer.logger.log_loss(trainer.epoch, self.metric_values(), "val")
        self.clear_metrics()

    def eval_val_loss_and_generate_samples(self, do_sample: bool):
        trainer = self.trainer
        cfg = trainer.cfg

        for batch_idx, data in enumerate(val_iter(trainer.val_dataloader)):
            (path, _, cond) = batch = trainer.prepare_batch(data)

            trainer.compute_val_loss(trainer.model, trainer.val_loss, batch)

            if do_sample:
                true_freq = compute_kmer_freq(path.x_1, k=6)

                samples = {} if cfg.eval.eval_sei else None
                for n_steps in [1, 2, 4, 8, 100]:
                    x1 = trainer.manifold.projx(
                        self.sample_with_n_steps(n_steps, path.x_0, **cond)
                    )
                    prob_x1 = sphere_to_simplex(x1)
                    ohe_x1 = sphere_to_onehot(x1)

                    self.metrics[f"{n_steps}_step_kmer_corr"].update_with_true_freq(
                        true_freq=true_freq, pred=ohe_x1
                    )
                    self.metrics[f"{n_steps}_step_entropy"].update(prob_x1)

                    if cfg.eval.eval_sei:
                        samples[f"{n_steps}_step"] = np.asarray(ohe_x1)

                if cfg.eval.eval_sei:
                    samples["true_x1"] = np.asarray(path.x_1)
                    np.savez(self.tmp_dir / f"batch_{batch_idx}.npz", **samples)

        trainer.logger.log_loss(trainer.epoch, trainer.val_loss, "val")

    def eval_sei_metrics(self):
        trainer = self.trainer
        trainer.sei.to_device("gpu")

        sample_files = list(self.tmp_dir.glob("batch_*.npz"))

        for sample_file in sample_files:
            samples = np.load(sample_file)
            batch_idx = int(sample_file.stem.split("_")[-1])
            true_seq = samples["true_x1"]

            for n_steps in [1, 2, 4, 8, 100]:
                gen_seq = samples[f"{n_steps}_step"]

                mse, feats = trainer.sei.eval_sp_mse(
                    gen_seq,
                    true_seq,
                    b_index=batch_idx,
                    return_features=True,
                )

                self.metrics[f"{n_steps}_step_mse"].update(values=mse)
                self.metrics[f"{n_steps}_step_fbd"].update(*feats)

        trainer.sei.to_device("cpu")
