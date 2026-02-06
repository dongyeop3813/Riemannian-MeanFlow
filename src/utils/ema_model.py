import flax.nnx as nnx
import optax


class ModelEMA(nnx.Optimizer):

    def __init__(
        self,
        model: nnx.Module,
        tx: optax.GradientTransformation,
    ):
        super().__init__(model, tx, wrt=[nnx.Param, nnx.BatchStat])

    def update(self, model_orginal: nnx.Module):
        params = nnx.state(model_orginal, self.wrt)
        opt_state = nnx.optimizer._opt_state_variables_to_state(self.opt_state)

        _, new_opt_state = self.tx.update(params, opt_state)

        self.step.value += 1
        nnx.update(self.model, new_opt_state.ema)
        nnx.optimizer._update_opt_state(self.opt_state, new_opt_state)

        params = nnx.state(self.model, self.wrt)
        opt_state = nnx.optimizer._opt_state_variables_to_state(self.opt_state)


def init_ema(model: nnx.Module, ema_decay: float = 0.9999):
    ema_model = nnx.clone(model)

    ema_tx = optax.ema(ema_decay)
    ema_optimizer = ModelEMA(ema_model, ema_tx)

    return ema_model, ema_optimizer


@nnx.jit
def ema_step(model: nnx.Module, ema_optimizer: nnx.Optimizer):
    ema_optimizer.update(model)
