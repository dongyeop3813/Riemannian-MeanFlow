from src.utils.checkpoint import CheckpointEveryNEpochs, CheckpointOnBestValLoss


def create_checkpoint(cfg, metric_to_monitor=None):
    from src.utils.etc import hydra_run_dir

    run_dir = hydra_run_dir()
    if cfg.mode == "every_n_epochs":
        return CheckpointEveryNEpochs(run_dir, cfg.save_interval, cfg.max_to_keep)
    elif cfg.mode == "best_val_loss":
        return CheckpointOnBestValLoss(run_dir, metric_to_monitor, cfg.max_to_keep)
    else:
        raise ValueError(f"Checkpoint mode {cfg.mode} not supported")
