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
