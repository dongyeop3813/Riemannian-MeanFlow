from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from jax import Array

from jax import tree_util
from functools import partial


@partial(
    tree_util.register_dataclass,
    data_fields=["x_1", "x_0", "t", "x_t", "dx_t"],
    meta_fields=[],
)
@dataclass
class PathSample:
    x_1: Array = field(metadata={"help": "target samples X_1 (batch_size, ...)."})
    x_0: Array = field(metadata={"help": "source samples X_0 (batch_size, ...)."})
    t: Array = field(metadata={"help": "time samples t (batch_size)."})
    x_t: Array = field(
        metadata={"help": "samples x_t ~ p_t(X_t), shape (batch_size, ...)."}
    )
    dx_t: Array = field(
        metadata={"help": "conditional target dX_t, shape: (batch_size, ...)."}
    )


class ProbPath(ABC):
    r"""Abstract class, representing a probability path.

    A probability path transforms the distribution :math:`p(X_0)` into :math:`p(X_1)` over :math:`t=0\rightarrow 1`.

    The ``ProbPath`` class is designed to support model training in the flow matching framework. It supports two key functionalities: (1) sampling the conditional probability path and (2) conversion between various training objectives.
    Here is a high-level example

    .. code-block:: python

        # Instantiate a probability path
        my_path = ProbPath(...)

        for x_0, x_1 in dataset:
            # Sets t to a random value in [0,1]
            t = torch.rand()

            # Samples the conditional path X_t ~ p_t(X_t|X_0,X_1)
            path_sample = my_path.sample(x_0=x_0, x_1=x_1, t=t)

            # Optimizes the model. The loss function varies, depending on model and path.
            loss(path_sample, my_model(x_t, t)).backward()

    """

    @abstractmethod
    def sample(self, x_0: Array, x_1: Array, t: Array) -> PathSample:
        r"""Sample from an abstract probability path:

        | given :math:`(X_0,X_1) \sim \pi(X_0,X_1)`.
        | returns :math:`X_0, X_1, X_t \sim p_t(X_t)`, and a conditional target :math:`Y`, all objects are under ``PathSample``.

        Args:
            x_0 (Tensor): source data point, shape (batch_size, ...).
            x_1 (Tensor): target data point, shape (batch_size, ...).
            t (Tensor): times in [0,1], shape (batch_size).

        Returns:
            PathSample: a conditional sample.
        """
