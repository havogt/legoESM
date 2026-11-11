"""The land-day probe's budget residuals: zero for a column that obeys the slab's update, and
the size of a planted imbalance otherwise."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "validate" / "icon_lane"))

import land_day  # noqa: E402

from legoesm import constants  # noqa: E402


def _cur(**kw):
    base = {"sw_down": np.array([400.0]), "lw_down": np.array([350.0]),
            "albedo": np.array([0.2]), "emissivity": np.array([0.95]),
            "shflx": np.array([30.0]), "lhflx": np.array([80.0]),
            "precipitation": np.array([1.0e-4]), "evaporation": np.array([3.0e-5]),
            "runoff": np.array([1.0e-5])}
    return {**base, **kw}


def test_the_energy_residual_is_zero_for_the_slab_update_and_sees_an_imbalance():
    cur, t_old, dt, heat = _cur(), np.array([290.0]), 300.0, 2.0e6
    net = (1.0 - 0.2) * 400.0 + 0.95 * (350.0 - constants.sigma_sb * 290.0**4) - 30.0 - 80.0
    t_new = t_old + dt * net / heat
    np.testing.assert_allclose(land_day.surface_energy_residual(cur, t_old, t_new, heat, dt),
                               0.0, atol=1e-9)
    np.testing.assert_allclose(
        land_day.surface_energy_residual(cur, t_old, t_new + 1.0e-3, heat, dt),
        -heat * 1.0e-3 / dt, rtol=1e-9)


def test_the_water_residual_closes_and_sees_a_leak():
    cur, dt = _cur(), 300.0
    gain = (1.0e-4 - 3.0e-5 - 1.0e-5) * dt
    w_old, snow = np.array([75.0]), np.array([0.0])
    np.testing.assert_allclose(
        land_day.water_residual(w_old, w_old + gain, snow, snow, cur, dt), 0.0, atol=1e-12)
    np.testing.assert_allclose(
        land_day.water_residual(w_old, w_old + gain + 0.5, snow, snow, cur, dt), 0.5)
