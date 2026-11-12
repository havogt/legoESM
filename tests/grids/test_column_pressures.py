"""ColumnPressures answers the sigma coordinate's pressure calls with the columns' own pressures,
and has no sigma attributes."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from legoesm.grids.column_pressures import ColumnPressures  # noqa: E402
from legoesm.grids.vertical import make_cam6_l32_levels  # noqa: E402


def test_it_answers_like_the_sigma_coordinate_it_stands_for():
    sigma = make_cam6_l32_levels()
    p_s = jnp.asarray([1.0e5, 9.0e4, 7.5e4])
    cp = ColumnPressures(sigma.pressure_at_full(p_s), sigma.pressure_at_half(p_s))
    assert cp.n_levels == sigma.n_levels
    np.testing.assert_array_equal(cp.pressure_at_full(p_s), sigma.pressure_at_full(p_s))
    np.testing.assert_array_equal(cp.pressure_at_half(p_s), sigma.pressure_at_half(p_s))
    # sigma computes dA + dB p_s, here the half-level difference: equal to the coordinate's
    # float32 rounding
    np.testing.assert_allclose(cp.layer_thickness_dp(p_s), sigma.layer_thickness_dp(p_s),
                               rtol=1e-6)
    # the surface pressure argument is ignored: the pressures are the columns'
    np.testing.assert_array_equal(cp.pressure_at_full(2.0 * p_s), cp.p_full)


def test_it_has_no_sigma_attributes_and_is_a_pytree():
    cp = ColumnPressures(jnp.ones((2, 3)), jnp.ones((2, 4)))
    for name in ("sigma_full", "sigma_half", "dsigma"):
        with pytest.raises(AttributeError):
            getattr(cp, name)
    doubled = jax.jit(lambda c: jax.tree.map(lambda x: 2.0 * x, c))(cp)
    assert isinstance(doubled, ColumnPressures) and float(doubled.p_half[0, 0]) == 2.0
