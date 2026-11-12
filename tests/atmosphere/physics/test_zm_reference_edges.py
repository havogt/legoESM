"""ZM's reference interfaces: on a sigma coordinate exactly the expression the integration
passed before (``pressure_at_half(p_ref)``), on per-column pressures none (each column's own
interfaces set the convection-top limit)."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from legoesm.atmosphere.physics.convection.integration import _reference_edges  # noqa: E402
from legoesm.grids.column_pressures import ColumnPressures  # noqa: E402
from legoesm.grids.vertical import create_sigma_coordinate, make_cam6_l32_levels  # noqa: E402

from legoesm import constants  # noqa: E402


@pytest.mark.parametrize("make", [make_cam6_l32_levels, lambda: create_sigma_coordinate(26)])
def test_a_sigma_coordinate_gets_the_same_reference_edges(make):
    sigma = make()
    before = sigma.pressure_at_half(jnp.asarray(constants.p_ref, dtype=jnp.float64))
    after = _reference_edges(sigma, jnp.float64)
    assert after.ndim == 1 and after.shape == (sigma.n_levels + 1,)
    np.testing.assert_array_equal(np.asarray(after), np.asarray(before))


def test_per_column_pressures_get_none():
    cp = ColumnPressures(jnp.ones((4, 3)), jnp.ones((4, 4)))
    assert _reference_edges(cp, jnp.float64) is None
