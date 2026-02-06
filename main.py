import torch
import numpy as np

import absl.logging as absl_logging

import hydra
from hydra.utils import instantiate

import logging

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)


@hydra.main(config_path="config", config_name="main.yaml", version_base="1.3")
def main(cfg):

    # Seed everything
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    absl_logging.set_verbosity(absl_logging.ERROR)

    logging.info(f"Setting seed to {cfg.seed}")

    # Instantiate the dataset
    train_dataset = instantiate(cfg.data.train)
    val_dataset = instantiate(cfg.data.val)

    logging.info(f"Train and val datasets are instantiated")

    trainer = instantiate(cfg.trainer, cfg, train_dataset, val_dataset)

    return trainer.run()


if __name__ == "__main__":
    main()
