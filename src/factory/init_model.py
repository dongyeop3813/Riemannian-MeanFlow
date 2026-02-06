import flax.nnx as nnx

from src.model import *


def init_net(net_cfg, x_shape, rngs: nnx.Rngs):
    if net_cfg.name == "MLPwithShape":
        return MLPwithShape(
            net_cfg.x_dim,
            net_cfg.hidden_dim,
            net_cfg.n_layers,
            x_shape,
            rngs=rngs,
        )
    elif net_cfg.name == "1D_CNN":
        return PromoterModelWithTimeInterval(
            mode=net_cfg.mode,
            embed_dim=net_cfg.embed_dim,
            time_step=net_cfg.time_step,
            scale=net_cfg.scale,
            time_emb_type=net_cfg.time_emb_type,
            rngs=rngs,
        )
    elif net_cfg.name == "WeakMLPwithShape":
        return WeakMLPwithShape(
            net_cfg.x_dim,
            net_cfg.hidden_dim,
            net_cfg.n_layers,
            x_shape,
            rngs=rngs,
        )
    else:
        raise ValueError(f"Unknown net: {net_cfg.name}")


def init_flow_map_model(model_cfg, manifold, x_shape: tuple[int, ...], seed: int):
    net = init_net(model_cfg.net, x_shape, nnx.Rngs(params=seed))

    if model_cfg.parametrization == "average_velocity":
        return AverageVelocityWrapper(
            net, manifold, clip_mode=model_cfg.get("clip_mode", True)
        )
    elif model_cfg.parametrization == "cond_velocity":
        return CondVelocityParameterizedWrapper(
            net, manifold, projection_mode=model_cfg.get("projection_mode", "default")
        )
    elif model_cfg.parametrization == "flow_map":
        return FlowMapParameterizedWrapper(net, manifold, clip_mode=model_cfg.clip_mode)
    elif model_cfg.parametrization == "flow_ode":
        return TangentWrapper(net, manifold)
    else:
        raise ValueError(f"Unknown model: {model_cfg.parametrization}")
