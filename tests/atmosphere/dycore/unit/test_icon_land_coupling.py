"""legoESM land fluxes (positive up) into ICON's SurfaceFluxState (positive down): each sign is
checked through icon4py's own turbulence scheme, and the flipped mapping gives the opposite sign."""

import os
import pathlib

import numpy as np
import pytest

pytest.importorskip("icon4py.model.driver.jax_api")
if "ICON4PY_TEST_DATA_PATH" not in os.environ:
    pytest.skip("ICON4PY_TEST_DATA_PATH is not set", allow_module_level=True)

import jax.numpy as jnp  # noqa: E402

from legoesm import constants  # noqa: E402
from legoesm.atmosphere.dynamics.gcm import icon4py_jax  # noqa: E402

pytestmark = pytest.mark.slow

DATA = pathlib.Path(os.environ["ICON4PY_TEST_DATA_PATH"])
GRID_FILE = DATA / "grids" / "icon_grid_0013_R02B04_R" / "icon_grid_0013_R02B04_R.nc"
APE = DATA / "ser_icondata" / "mpitask1_exclaim_ape_aesPhys_v14" / "config.yml"


@pytest.fixture(scope="module")
def column(tmp_path_factory):
    from icon4py.model.driver import config as driver_config
    from icon4py.model.driver import jax_api

    cfg = driver_config.read_experiment_config_from_yaml(APE).with_overrides(
        driver={"output_path": tmp_path_factory.mktemp("ape")}
    )
    static, state = jax_api.init_from_grid_file(GRID_FILE, cfg)
    col = jax_api.dyn2phy(static, state)
    template = jax_api.surface(static, jax_api.surface_input(static, col))
    return static, col, template


def _lowest_tendencies(column, arrays):
    from icon4py.model.driver import jax_api

    static, col, template = column
    tend, _ = jax_api.tmx(static, col, icon4py_jax.surface_flux_state(template, arrays))
    return {
        "T": np.asarray(tend.temperature.ndarray)[:, -1],
        "qv": np.asarray(tend.q["qv"].ndarray)[:, -1],
        "u": np.asarray(tend.u.ndarray)[:, -1],
        "v": np.asarray(tend.v.ndarray)[:, -1],
    }


def _land(column, *, shflx=0.0, lhflx=0.0, drag=0.0):
    _, col, _ = column
    u = jnp.asarray(col.u.ndarray)[:, -1]
    v = jnp.asarray(col.v.ndarray)[:, -1]
    speed = jnp.sqrt(u**2 + v**2)
    n = u.shape[0]
    return icon4py_jax.land_surface_fluxes(
        shflx=jnp.full(n, shflx),
        lhflx=jnp.full(n, lhflx),
        surface_mass_flux=jnp.full(n, lhflx / constants.L_v),
        tau_x=-drag * u / speed,
        tau_y=-drag * v / speed,
        T_lowest=jnp.asarray(col.temperature.ndarray)[:, -1],
    )


def _response(column, arrays, sign=1.0):
    zero = {k: jnp.zeros_like(v) for k, v in arrays.items()}
    base = _lowest_tendencies(column, zero)
    got = _lowest_tendencies(column, {k: sign * v for k, v in arrays.items()})
    return {k: got[k] - base[k] for k in got}


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_evaporating_land_moistens_the_lowest_level(column, sign):
    d = _response(column, _land(column, lhflx=100.0), sign)
    assert (sign * d["qv"] > 0.0).all()


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_sensible_heat_from_the_land_warms_the_lowest_level(column, sign):
    d = _response(column, _land(column, shflx=50.0), sign)
    assert (sign * d["T"] > 0.0).all()


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_land_drag_decelerates_the_lowest_level_wind(column, sign):
    _, col, _ = column
    u = np.asarray(col.u.ndarray)[:, -1]
    v = np.asarray(col.v.ndarray)[:, -1]
    d = _response(column, _land(column, drag=0.2), sign)
    # the stress is projected to edges and back, so a few cells in weak or turning flow may
    # not follow their own wind; the bulk must
    assert np.mean(sign * (d["u"] * u + d["v"] * v) < 0.0) > 0.95


def test_no_canopy_snow_melt_heat_from_the_slab(column):
    arrays = _land(column, shflx=50.0, lhflx=100.0, drag=0.2)
    assert not np.any(np.asarray(arrays["q_snocpymlt"]))
