import flax.nnx as nnx
from torch.utils.data import Dataset, DataLoader
from src.utils import resume_run_if_needed
from src.utils import get_logger, epoch_iter
from src.utils.checkpoint import Checkpoint
from src.utils.ema_model import init_ema, ema_step


class Experiment:
    """
    Base class for all experiments.

    The subclass must implement the setup, train_epoch, and eval_epoch methods.
    setup is called before to build the model, optimizer, and other components for training.
    train is called to run the training loop.
    """

    def __init__(self, cfg, train_dataset: Dataset, val_dataset: Dataset):
        self.cfg = cfg
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.rngs = nnx.Rngs(cfg.seed)
        self.metrics = []

        self.do_ema = cfg.get("ema", False)

    def run(self):
        self.init_dataloaders()

        self.setup()

        self.model, self.optimizer, self.epoch = resume_run_if_needed(
            self.cfg, self.model, self.optimizer
        )

        if self.do_ema:
            self.ema_model, self.ema_optimizer = init_ema(
                self.model, self.cfg.ema_decay
            )

        try:
            self.logger = get_logger(self.cfg)

            metric = self.train()

        finally:
            self.logger.finish()

        return metric

    def setup(self) -> dict:
        raise NotImplementedError("Subclass must implement setup method")

    def train(self):
        for epoch in epoch_iter(self.epoch, self.cfg.num_epochs):
            self.epoch = epoch
            if self.do_ema:
                self.checkpoint.save(epoch, self.ema_model, self.optimizer)
            else:
                self.checkpoint.save(epoch, self.model, self.optimizer)

            if self.do_ema:
                temp_model = self.model
                self.model = self.ema_model
                self.model.eval()
                self.eval_epoch()
                self.model = temp_model
            else:
                self.model.eval()
                self.eval_epoch()

            self.model.train()
            if self.do_ema:
                self.ema_model.train()
            self.train_epoch()

            self.reset_metrics()

        # Final checkpoint
        self.epoch += 1
        if self.do_ema:
            self.checkpoint.save(self.epoch, self.ema_model, self.optimizer)
        else:
            self.checkpoint.save(self.epoch, self.model, self.optimizer)

    def train_epoch(self):
        raise NotImplementedError("Subclass must implement train_epoch method")

    def eval_epoch(self):
        raise NotImplementedError("Subclass must implement eval_epoch method")

    def __setattr__(self, name, value):
        if isinstance(value, nnx.metrics.Metric):
            self.metrics.append(value)
        return super().__setattr__(name, value)

    def reset_metrics(self):
        for metric in self.metrics:
            metric.reset()

    def init_dataloaders(self):
        self.train_dataloader = DataLoader(
            dataset=self.train_dataset, batch_size=self.cfg.batch_size, shuffle=False
        )

        if self.val_dataset is not None:
            self.val_dataloader = DataLoader(
                dataset=self.val_dataset, batch_size=self.cfg.batch_size, shuffle=False
            )
        else:
            self.val_dataloader = None
