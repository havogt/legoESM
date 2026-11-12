"""The vertical coordinate of a column lane whose dycore is not on sigma levels: each column's own
full- and half-level pressures, behind the pressure methods the column physics calls on a sigma
coordinate. Levels run top to bottom, ``(ncol, nlev)`` and ``(ncol, nlev + 1)``."""

from __future__ import annotations

from typing import NamedTuple

import jax


class ColumnPressures(NamedTuple):
    """Per-column pressures [Pa] standing in for ``sigma_coord``. The methods take the
    sigma-coordinate signature and ignore ``p_s``: the pressures are the dycore's. A sigma-only
    attribute (``sigma_full``, ``dsigma``, ...) is absent on purpose, so code that needs one
    fails instead of reading a wrong coordinate."""

    p_full: jax.Array
    p_half: jax.Array

    @property
    def n_levels(self) -> int:
        return int(self.p_full.shape[-1])

    def pressure_at_full(self, p_s=None):
        del p_s
        return self.p_full

    def pressure_at_half(self, p_s=None):
        del p_s
        return self.p_half

    def layer_thickness_dp(self, p_s=None):
        del p_s
        return self.p_half[..., 1:] - self.p_half[..., :-1]


class NoSigmaCoordinate(NamedTuple):
    """For setup code that sizes itself from a sigma coordinate on a lane that has none: the
    sigma arrays are NaN of the right length (any arithmetic on them shows), and the pressure
    methods raise. The lane's physics gets the columns' pressures per step instead."""

    n_levels: int

    @property
    def sigma_full(self):
        import jax.numpy as jnp

        return jnp.full((self.n_levels,), jnp.nan)

    @property
    def sigma_half(self):
        import jax.numpy as jnp

        return jnp.full((self.n_levels + 1,), jnp.nan)

    @property
    def dsigma(self):
        import jax.numpy as jnp

        return jnp.full((self.n_levels,), jnp.nan)

    def _refuse(self, *_):
        raise ValueError("this lane has no sigma coordinate: its physics takes each step's "
                         "column pressures (ColumnPressures)")

    pressure_at_full = pressure_at_half = layer_thickness_dp = _refuse
