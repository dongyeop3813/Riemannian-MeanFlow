from omegaconf import DictConfig


def get_run_name(cfg: DictConfig) -> str:

    # Extract the name of the data
    experiment = cfg.wandb.tags[0] + "_" + cfg.wandb.tags[1]

    lr = f"_lr={cfg.optim.lr}"

    if cfg.optim.get("grad_clip", None):
        grad_clip = f"_gc={cfg.optim.grad_clip}"
    else:
        grad_clip = ""

    seed = f"_seed={cfg.seed}"

    batch_size = f"_bs={cfg.batch_size}"

    return f"{experiment}{lr}{seed}{batch_size}{grad_clip}"


def get_run_name_for_toy_helix(cfg: DictConfig) -> str:
    experiment = "ToyHelix"

    if cfg.model.parametrization == "cond_velocity":
        parametrization = "cond"
    elif cfg.model.parametrization == "average_velocity":
        parametrization = "avg"
    else:
        parametrization = cfg.model.parametrization

    dim = cfg.data.train.get("n", "NA")
    helix_a = str(cfg.data.train.helix_a)
    hdim = str(cfg.model.net.hidden_dim)

    return f"{experiment}_{parametrization}_D={dim}_a={helix_a}_h={hdim}"
