from src.experiments.base_class import Experiment
from src.manifold import *
from src.utils.etc import should_eval
import numpy as np
from src.eval.mmd import compute_mmd


def toy_helix_eval_fn(trainer: Experiment):
    if not should_eval(trainer.epoch, trainer.cfg.eval.interval):
        return

    x0 = trainer.prior.sample(trainer.cfg.eval.num_samples, *trainer.cfg.x_shape)
    gen_x1 = trainer.sample_x1(trainer.model, x0)
    one_step_fig = trainer.train_dataset.visualize_projection_to_S2(np.array(gen_x1))

    gen_x1 = trainer.sample_x1_with_integration(trainer.model, x0)
    multi_step_fig = trainer.train_dataset.visualize_projection_to_S2(np.array(gen_x1))

    trainer.logger.log_figures(
        trainer.epoch,
        {
            "one_step_projection": one_step_fig,
            "multi_step_projection": multi_step_fig,
        },
    )


def earth_eval_fn(trainer: Experiment):
    if not should_eval(trainer.epoch, trainer.cfg.eval.interval):
        return

    from src.eval import compute_mmd

    x0 = trainer.prior.sample(len(trainer.val_dataset), *trainer.cfg.x_shape)
    one_stpe_x1 = trainer.sample_x1(trainer.model, x0)
    x1_steps_2 = trainer.sample_x1_with_n_steps(trainer.model, x0, n_steps=2)
    x1_steps_8 = trainer.sample_x1_with_n_steps(trainer.model, x0, n_steps=8)
    x1_steps_16 = trainer.sample_x1_with_n_steps(trainer.model, x0, n_steps=16)
    x1_steps_32 = trainer.sample_x1_with_n_steps(trainer.model, x0, n_steps=32)
    x1_steps_64 = trainer.sample_x1_with_n_steps(trainer.model, x0, n_steps=64)
    integrated_x1 = trainer.sample_x1_with_integration(trainer.model, x0)

    # Compute MMD metric and log the figure
    one_step_mmd = compute_mmd(one_stpe_x1, trainer.eval_data, trainer.manifold)
    x1_steps_2_mmd = compute_mmd(x1_steps_2, trainer.eval_data, trainer.manifold)
    x1_steps_8_mmd = compute_mmd(x1_steps_8, trainer.eval_data, trainer.manifold)
    x1_steps_16_mmd = compute_mmd(x1_steps_16, trainer.eval_data, trainer.manifold)
    x1_steps_32_mmd = compute_mmd(x1_steps_32, trainer.eval_data, trainer.manifold)
    x1_steps_64_mmd = compute_mmd(x1_steps_64, trainer.eval_data, trainer.manifold)
    integrated_mmd = compute_mmd(integrated_x1, trainer.eval_data, trainer.manifold)
    metrics = {
        "1_step_mmd": one_step_mmd,
        "2_steps_mmd": x1_steps_2_mmd,
        "8_steps_mmd": x1_steps_8_mmd,
        "16_steps_mmd": x1_steps_16_mmd,
        "32_steps_mmd": x1_steps_32_mmd,
        "64_steps_mmd": x1_steps_64_mmd,
        "100_steps_mmd": integrated_mmd,
    }
    trainer.logger.log_loss(trainer.epoch, metrics, "val")

    # Draw plots and log them
    one_step_fig = trainer.train_dataset.visualize(np.array(one_stpe_x1))
    integrated_fig = trainer.train_dataset.visualize(np.array(integrated_x1))

    trainer.logger.log_figures(
        trainer.epoch,
        {
            "one_step": one_step_fig,
            "multi_step": integrated_fig,
        },
    )


def amino_eval_fn(trainer: Experiment):
    if not should_eval(trainer.epoch, trainer.cfg.eval.interval):
        return

    x0 = trainer.prior.sample(len(trainer.val_dataset), *trainer.cfg.x_shape)
    one_stpe_x1 = trainer.sample_x1(trainer.model, x0)
    x1_steps_2 = trainer.sample_x1_with_n_steps(trainer.model, x0, n_steps=2)
    x1_steps_8 = trainer.sample_x1_with_n_steps(trainer.model, x0, n_steps=8)
    x1_steps_16 = trainer.sample_x1_with_n_steps(trainer.model, x0, n_steps=16)
    x1_steps_32 = trainer.sample_x1_with_n_steps(trainer.model, x0, n_steps=32)
    x1_steps_64 = trainer.sample_x1_with_n_steps(trainer.model, x0, n_steps=64)
    integrated_x1 = trainer.sample_x1_with_integration(trainer.model, x0)

    # Compute MMD metric and log the figure
    one_step_mmd = compute_mmd(one_stpe_x1, trainer.eval_data, trainer.manifold)
    x1_steps_2_mmd = compute_mmd(x1_steps_2, trainer.eval_data, trainer.manifold)
    x1_steps_8_mmd = compute_mmd(x1_steps_8, trainer.eval_data, trainer.manifold)
    x1_steps_16_mmd = compute_mmd(x1_steps_16, trainer.eval_data, trainer.manifold)
    x1_steps_32_mmd = compute_mmd(x1_steps_32, trainer.eval_data, trainer.manifold)
    x1_steps_64_mmd = compute_mmd(x1_steps_64, trainer.eval_data, trainer.manifold)
    integrated_mmd = compute_mmd(integrated_x1, trainer.eval_data, trainer.manifold)
    metrics = {
        "1_step_mmd": one_step_mmd,
        "2_steps_mmd": x1_steps_2_mmd,
        "8_steps_mmd": x1_steps_8_mmd,
        "16_steps_mmd": x1_steps_16_mmd,
        "32_steps_mmd": x1_steps_32_mmd,
        "64_steps_mmd": x1_steps_64_mmd,
        "100_steps_mmd": integrated_mmd,
    }
    trainer.logger.log_loss(trainer.epoch, metrics, "val")
