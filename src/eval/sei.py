"""
Sei architecture (JAX/Flax nnx port)

Notes
- Convolution layout is channels-last: inputs are shaped [batch, length, channels].
- MaxPool uses nnx.max_pool.
- BSpline basis is precomputed using NumPy/SciPy and cached in the module.
"""

from typing import Optional, Tuple, Union
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import jax.nn as jnn
import flax.nnx as nnx
from scipy.interpolate import splev


def bs(
    x: np.ndarray,
    df: Optional[int] = None,
    knots: Optional[np.ndarray] = None,
    degree: int = 3,
    intercept: bool = False,
) -> np.ndarray:
    """
    Natural cubic B-spline basis evaluation (NumPy/SciPy).

    Parameters
    ----------
    x : np.ndarray
        Evaluation positions (1D array with shape [n]).
    df : int
        The number of degrees of freedom to use for this spline. You must specify at
        least one of `df` and `knots`.
    knots : np.ndarray
        The interior knots of the spline.
    degree : int
        Degree of piecewise polynomial. Default is 3.
    intercept : bool
        If True, include the intercept basis.
    """

    order = degree + 1
    inner_knots: Union[np.ndarray, list] = []
    if df is not None and knots is None:
        n_inner_knots = df - order + (1 - int(intercept))
        if n_inner_knots < 0:
            n_inner_knots = 0
            print("df was too small; have used %d" % (order - (1 - int(intercept))))

        if n_inner_knots > 0:
            inner_knots = np.percentile(
                x, 100 * np.linspace(0, 1, n_inner_knots + 2)[1:-1]
            )
    elif knots is not None:
        inner_knots = knots

    all_knots = np.concatenate(
        ([np.min(x), np.max(x)] * order, np.asarray(inner_knots))
    )
    all_knots.sort()

    n_basis = len(all_knots) - (degree + 1)
    basis = np.empty((x.shape[0], n_basis), dtype=float)

    for i in range(n_basis):
        coefs = np.zeros((n_basis,))
        coefs[i] = 1
        basis[:, i] = splev(x, (all_knots, coefs, degree))

    if not intercept:
        basis = basis[:, 1:]
    return basis


def spline_factory(n: int, df: int, log: bool = False) -> jnp.ndarray:
    """Create a spline basis matrix of shape [n, df] as a JAX array."""
    if log:
        dist = np.array(np.arange(n) - n / 2.0)
        dist = np.log(np.abs(dist) + 1) * (2 * (dist > 0) - 1)
        n_knots = df - 4
        knots = np.linspace(np.min(dist), np.max(dist), n_knots + 2)[1:-1]
        basis = bs(dist, knots=knots, intercept=True)
    else:
        dist = np.arange(n)
        basis = bs(dist, df=df, intercept=True)
    return jnp.asarray(basis, dtype=jnp.float32)


class BSplineTransformation(nnx.Module):
    def __init__(
        self, degrees_of_freedom: int, log: bool = False, scaled: bool = False
    ):
        self._spline_tr: Optional[jnp.ndarray] = None
        self._log = log
        self._scaled = scaled
        self._df = degrees_of_freedom

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Apply spline basis along the sequence axis.

        x: [batch, length, channels]
        returns: [batch, channels, df]
        """
        spatial_dim = x.shape[-2]
        if (self._spline_tr is None) or (self._spline_tr.shape[0] != spatial_dim):
            spline_tr = spline_factory(spatial_dim, self._df, log=self._log)
            if self._scaled:
                spline_tr = spline_tr / float(spatial_dim)
            self._spline_tr = spline_tr

        # x: [B, L, C], spline_tr: [L, df] -> out: [B, C, df]
        return jnp.einsum("blc,ld->bcd", x, self._spline_tr)


class Sei(nnx.Module):
    def __init__(
        self,
        sequence_length: int = 4096,
        n_genomic_features: int = 21907,
        rngs: Optional[nnx.Rngs] = None,
    ):
        # Initial conv stack (channels-last)
        self.max_pool_4 = partial(
            nnx.max_pool, window_shape=(4,), strides=(4,), padding="VALID"
        )
        self.lconv1 = nnx.Sequential(
            nnx.Conv(
                in_features=4,
                out_features=480,
                kernel_size=(9,),
                padding=((4, 4),),
                rngs=rngs,
            ),
            nnx.Conv(
                in_features=480,
                out_features=480,
                kernel_size=(9,),
                padding=((4, 4),),
                rngs=rngs,
            ),
        )

        self.conv1 = nnx.Sequential(
            nnx.Conv(
                in_features=480,
                out_features=480,
                kernel_size=(9,),
                padding=((4, 4),),
                rngs=rngs,
            ),
            nnx.relu,
            nnx.Conv(
                in_features=480,
                out_features=480,
                kernel_size=(9,),
                padding=((4, 4),),
                rngs=rngs,
            ),
            nnx.relu,
        )

        # Second conv block (pooling is applied in forward)
        self.lconv2 = nnx.Sequential(
            self.max_pool_4,
            nnx.Dropout(rate=0.2, rngs=rngs),
            nnx.Conv(
                in_features=480,
                out_features=640,
                kernel_size=(9,),
                padding=((4, 4),),
                rngs=rngs,
            ),
            nnx.Conv(
                in_features=640,
                out_features=640,
                kernel_size=(9,),
                padding=((4, 4),),
                rngs=rngs,
            ),
        )

        self.conv2 = nnx.Sequential(
            nnx.Dropout(rate=0.2, rngs=rngs),
            nnx.Conv(
                in_features=640,
                out_features=640,
                kernel_size=(9,),
                padding=((4, 4),),
                rngs=rngs,
            ),
            nnx.relu,
            nnx.Conv(
                in_features=640,
                out_features=640,
                kernel_size=(9,),
                padding=((4, 4),),
                rngs=rngs,
            ),
            nnx.relu,
        )

        # Third conv block (pooling is applied in forward)
        self.lconv3 = nnx.Sequential(
            self.max_pool_4,
            nnx.Dropout(rate=0.2, rngs=rngs),
            nnx.Conv(
                in_features=640,
                out_features=960,
                kernel_size=(9,),
                padding=((4, 4),),
                rngs=rngs,
            ),
            nnx.Conv(
                in_features=960,
                out_features=960,
                kernel_size=(9,),
                padding=((4, 4),),
                rngs=rngs,
            ),
        )

        self.conv3 = nnx.Sequential(
            nnx.Dropout(rate=0.2, rngs=rngs),
            nnx.Conv(
                in_features=960,
                out_features=960,
                kernel_size=(9,),
                padding=((4, 4),),
                rngs=rngs,
            ),
            nnx.relu,
            nnx.Conv(
                in_features=960,
                out_features=960,
                kernel_size=(9,),
                padding=((4, 4),),
                rngs=rngs,
            ),
            nnx.relu,
        )

        # Dilated conv stack (kernel_size=5)
        def dconv_seq(dilation: int):
            pad = dilation * (5 - 1) // 2
            conv = nnx.Conv(
                in_features=960,
                out_features=960,
                kernel_size=(5,),
                padding=((pad, pad),),
                kernel_dilation=(dilation,),
                rngs=rngs,
            )
            return nnx.Sequential(nnx.Dropout(rate=0.1, rngs=rngs), conv, nnx.relu)

        self.dconv1 = dconv_seq(2)
        self.dconv2 = dconv_seq(4)
        self.dconv3 = dconv_seq(8)
        self.dconv4 = dconv_seq(16)
        self.dconv5 = dconv_seq(25)

        # Spline transformation and classifier
        self._spline_df = int(128 / 8)
        self.spline_tr = nnx.Sequential(
            nnx.Dropout(rate=0.5, rngs=rngs),
            BSplineTransformation(self._spline_df, scaled=False),
        )

        self.classifier = nnx.Sequential(
            nnx.Linear(960 * self._spline_df, n_genomic_features, rngs=rngs),
            nnx.relu,
            nnx.Linear(n_genomic_features, n_genomic_features, rngs=rngs),
            jnn.sigmoid,
        )

    def __call__(self, x: jnp.ndarray, return_features: bool = False) -> jnp.ndarray:
        """
        Forward propagation.

        x: [batch, length, 4]
        returns: [batch, n_genomic_features]
        """
        lout1 = self.lconv1(x)
        out1 = self.conv1(lout1)

        lout2 = self.lconv2(out1 + lout1)
        out2 = self.conv2(lout2)

        lout3 = self.lconv3(out2 + lout2)
        out3 = self.conv3(lout3)

        dconv_out1 = self.dconv1(out3 + lout3)
        cat_out1 = out3 + dconv_out1

        dconv_out2 = self.dconv2(cat_out1)
        cat_out2 = cat_out1 + dconv_out2

        dconv_out3 = self.dconv3(cat_out2)
        cat_out3 = cat_out2 + dconv_out3

        dconv_out4 = self.dconv4(cat_out3)
        cat_out4 = cat_out3 + dconv_out4

        dconv_out5 = self.dconv5(cat_out4)
        out = cat_out4 + dconv_out5

        spline_out = self.spline_tr(out)
        reshape_out = jnp.reshape(spline_out, (spline_out.shape[0], -1))

        if return_features:
            return self.classifier(reshape_out), reshape_out
        else:
            return self.classifier(reshape_out)


def _flip_last_two_axes(x: jnp.ndarray) -> jnp.ndarray:
    """Flip along channels (last axis) and length (axis=1) to simulate reverse complement."""
    return jnp.flip(jnp.flip(x, axis=2), axis=1)


class NonStrandSpecific(nnx.Module):
    """
    Wrap a model to be non-strand specific by averaging or maxing predictions
    over the input and its reverse-complement.
    """

    def __init__(self, model: nnx.Module, mode: str = "mean"):
        self.model = model
        if mode not in ("mean", "max"):
            raise ValueError(f"Mode should be 'mean' or 'max' but was {mode}.")
        self.mode = mode

    @partial(nnx.jit, static_argnames=("return_features"))
    def __call__(self, x: jnp.ndarray, return_features: bool = False):
        reverse_input = _flip_last_two_axes(x)
        output = self.model(x, return_features)
        if return_features:
            output, feat = output

        output_from_rev = self.model(reverse_input)

        if isinstance(output, tuple):
            output1, output2 = output
            output_from_rev1, output_from_rev2 = output_from_rev
            if self.mode == "mean":
                ret = (
                    (output1 + output_from_rev1) / 2.0,
                    (output2 + output_from_rev2) / 2.0,
                )
            else:
                ret = (
                    jnp.maximum(output1, output_from_rev1),
                    jnp.maximum(output2, output_from_rev2),
                )
        else:
            if self.mode == "mean":
                ret = (output + output_from_rev) / 2.0
            else:
                ret = jnp.maximum(output, output_from_rev)

        if return_features:
            return ret, feat
        else:
            return ret
