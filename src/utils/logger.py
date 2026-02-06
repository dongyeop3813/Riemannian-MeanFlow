from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig
import wandb
import matplotlib.pyplot as plt

from flax.nnx import metrics

from pathlib import Path
from src.utils.checkpoint import save_ckpt
from src.utils.etc import add_prefix_to_keys
from src.utils.debug_util import DebugLogger
from src.utils.run_name import get_run_name, get_run_name_for_toy_helix


class WandbLogger:
    def __init__(self, cfg):
        self.cfg = cfg
        self.run_dir = Path(HydraConfig.get().runtime.output_dir)

        if cfg.data.train._target_.split(".")[-1] == "SnEmbeddedHelixDataset":
            name = get_run_name_for_toy_helix(cfg)
        else:
            name = get_run_name(cfg)

        wandb.init(
            project=cfg.wandb.project,
            name=name,
            config=OmegaConf.to_container(cfg, resolve=True),
            settings=wandb.Settings(console="off"),
            dir=self.run_dir,
            tags=cfg.wandb.get("tags", None),
        )
        wandb.run.config["run_dir"] = self.run_dir

        self.debug = cfg.get("debug", False)
        if self.debug:
            self.debug_logger = DebugLogger(self.run_dir)

            # Add "debug" tag to wandb run if in debug mode
            wandb.run.tags += ("debug",)

    def log_for_debug(self, epoch, loss, grad_norm):
        self.debug_logger.loss_grad_and_loss(epoch, loss, grad_norm)

    def checkpoint_and_exit(self, epoch, model, optimizer, **kwargs):
        self.debug_logger.checkpoint_and_exit(epoch, model, optimizer, **kwargs)

    def log_loss(self, epoch, loss_dict, prefix="train"):
        loss_dict = metric_to_float(loss_dict)
        wandb.log(add_prefix_to_keys(loss_dict, prefix), step=epoch)

    def log_metrics(self, epoch, metrics):
        metrics = metric_to_float(metrics)
        wandb.log(metrics, step=epoch)

    def log_figures(self, epoch, figs):
        wandb.log(self.to_wandb_images(figs), step=epoch)
        plt.close("all")

    def finish(self):
        wandb.finish()

    def to_wandb_images(self, figs_dict) -> dict[str, wandb.Image]:
        return {k: wandb.Image(v) for k, v in figs_dict.items()}

    def save_ckpt(self, model, optimizer, epoch):
        save_ckpt(self.run_dir, model, optimizer, epoch)


class Logger:
    def __init__(self, cfg):
        self.cfg = cfg
        self.run_dir = Path(HydraConfig.get().runtime.output_dir)
        self.debug = cfg.get("debug", False)
        if self.debug:
            self.debug_logger = DebugLogger(self.run_dir)

    def log_for_debug(self, epoch, loss, grad_norm):
        self.debug_logger.loss_grad_and_loss(epoch, loss, grad_norm)

    def checkpoint_and_exit(self, epoch, model, optimizer, **kwargs):
        self.debug_logger.checkpoint_and_exit(epoch, model, optimizer, **kwargs)

    def log_loss(self, epoch, loss_dict, prefix="train"):
        loss_dict = metric_to_float(loss_dict)
        msg = f"Epoch {epoch}: "
        msg += ", ".join([f"{prefix}/{k}: {v}" for k, v in loss_dict.items()])
        print(msg)

    def log_metrics(self, epoch, metrics):
        metrics = metric_to_float(metrics)
        msg = f"Epoch {epoch}: "
        msg += ", ".join([f"{k}: {v}" for k, v in metrics.items()])
        print(msg)

    def log_figures(self, epoch, figs):
        plt.close("all")

    def save_ckpt(self, model, optimizer, epoch):
        save_ckpt(self.run_dir, model, optimizer, epoch)

    def finish(self):
        pass


def get_logger(cfg):
    if cfg.get("wandb", False):
        return WandbLogger(cfg)
    else:
        return Logger(cfg)


def metric_to_float(loss_dict):
    if isinstance(loss_dict, metrics.MultiMetric):
        return loss_dict.compute()
    elif isinstance(loss_dict, metrics.Metric):
        return {loss_dict.argname: loss_dict.compute()}
    elif isinstance(loss_dict, dict):
        for k, metric in loss_dict.items():
            if isinstance(metric, metrics.Metric):
                loss_dict[k] = metric.compute()
        return loss_dict
    else:
        return loss_dict
