"""legoESM's RRTMGP on the ICON lane: the step's radiation equals the solver called directly on
the same columns, its heating is the column's net flux convergence at constant volume, and the
D3 land patch runs a day under it."""

import os
import pathlib

import numpy as np
import pytest

pytest.importorskip("icon4py.model.driver.jax_api")
if "ICON4PY_TEST_DATA_PATH" not in os.environ:
    pytest.skip("ICON4PY_TEST_DATA_PATH is not set", allow_module_level=True)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from legoesm.config import Config  # noqa: E402

pytestmark = pytest.mark.slow

REPO = pathlib.Path(__file__).resolve().parents[4]
DATA = pathlib.Path(os.environ["ICON4PY_TEST_DATA_PATH"])
GRID_FILE = DATA / "grids" / "icon_grid_0013_R02B04_R" / "icon_grid_0013_R02B04_R.nc"


def _config(out):
    config = Config.from_yaml(str(REPO / "config" / "icon" / "ape_aes_land_r2b4.yaml"))
    config.set("grid.grid_file", str(GRID_FILE))
    config.set("output.path", str(out))
    return config.to_experiment_config()


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    from icon4py.model.driver import jax_api
    from legoesm.atmosphere.dynamics.gcm import icon4py_jax
    from legoesm.driver import component_factory, icon_radiation

    cfg = _config(tmp_path_factory.mktemp("rad"))
    static, state = jax_api.init_from_grid_file(
        GRID_FILE, icon4py_jax.build_icon4py_experiment_config(cfg))
    col = jax_api.dyn2phy(static, state)
    coupling = component_factory._icon_radiation_coupling(cfg, False)
    clock = coupling.clock(0)
    heating, out = jax.jit(coupling.step)(col, coupling.ocean_surface, *clock)
    cv = np.asarray(icon_radiation.cv_air(col))
    return cfg, col, coupling, clock, np.asarray(heating), out, cv


def test_the_step_is_the_solver_on_the_same_columns(setup):
    from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP, standard_o3_profile
    from legoesm.atmosphere.physics.radiation.solar import daily_mean_insolation, daylight_fraction
    from legoesm.driver.physics_pipeline import rrtmgp_config_from_experiment
    from legoesm.grids.factory import create_grid

    cfg, col, coupling, (doy, _), _, out, _ = setup
    rad = cfg.icon.radiation
    assert not rad.diurnal_cycle
    mesh = create_grid("icon", grid_file=GRID_FILE)
    solver = RRTMGP.from_legoesm_config(rrtmgp_config_from_experiment(
        rad, sfc_emissivity=rad.sfc_emissivity, sfc_albedo=rad.albedo_ocean,
        include_clouds=False))
    lat = jnp.asarray(mesh.latCell)
    insol = daily_mean_insolation(lat, doy, rad.S_0)
    f_day = daylight_fraction(lat, doy)
    # jitted like the step: the solver's optics are float32, so XLA's fusion order shows
    witness = jax.jit(solver.solve_columns)(
        T=col.temperature.ndarray, p_full=col.pressure.ndarray, p_half=col.pressure_ifc.ndarray,
        sfc_temperature=coupling.ocean_surface["T_sfc"], q_v=col.q["qv"].ndarray,
        cos_zenith=jnp.clip(insol / (rad.S_0 * jnp.maximum(f_day, 1.0e-6)), 0.0, 1.0),
        sfc_albedo=coupling.ocean_surface["albedo"],
        sfc_emissivity=coupling.ocean_surface["emissivity"],
        # given explicitly, as every legoESM lane does: without it the solver evaluates the
        # profile at its lid instead of copying the top layer up
        o3_vmr=standard_o3_profile(col.pressure.ndarray))
    for name in ("lw_flux_up", "lw_flux_down"):
        np.testing.assert_allclose(np.asarray(getattr(out, name)),
                                   np.asarray(getattr(witness, name)), rtol=1e-9, err_msg=name)
    for name in ("sw_flux_up", "sw_flux_down"):
        np.testing.assert_allclose(np.asarray(getattr(out, name)),
                                   np.asarray(getattr(witness, name)) * np.asarray(f_day)[:, None],
                                   rtol=1e-9, atol=1e-9, err_msg=name)


def test_the_heating_is_the_net_flux_convergence_at_constant_volume(setup):
    from icon4py.model.driver import jax_api

    _, col, _, _, heating, out, cv = setup
    net = (np.asarray(out.sw_flux_down) - np.asarray(out.sw_flux_up)
           + np.asarray(out.lw_flux_down) - np.asarray(out.lw_flux_up))
    np.testing.assert_allclose((cv * heating).sum(axis=1), net[:, 0] - net[:, -1],
                               rtol=1e-10, atol=1e-8)
    # ICON applies the tendency at constant volume: cv times the temperature change it makes
    # is the energy radiation put in
    dt = 300.0
    tend = jax_api.ColumnTendencies(
        temperature=type(col.temperature).from_array(jnp.asarray(heating),
                                                     domain=col.temperature.domain))
    new = jax_api.apply_column_tendencies(col, tend, dt, recompute_virtual_temperature=False)
    gained = (cv * (np.asarray(new.temperature.ndarray) - np.asarray(col.temperature.ndarray))
              ).sum(axis=1)
    np.testing.assert_allclose(gained, (net[:, 0] - net[:, -1]) * dt, rtol=1e-8, atol=1e-6)


def test_the_global_mean_insolation_and_outgoing_longwave(setup):
    from legoesm.grids.factory import create_grid

    cfg, _, _, _, _, out, _ = setup
    area = np.asarray(create_grid("icon", grid_file=GRID_FILE).areaCell)
    mean = lambda x: float(np.sum(area * np.asarray(x)) / np.sum(area))  # noqa: E731
    # the daily-mean insolation on a circular orbit averages to S_0 / 4 over the sphere
    assert mean(out.toa_insolation) == pytest.approx(cfg.icon.radiation.S_0 / 4.0, rel=2e-3)
    # clear-sky aquaplanet at 303 K
    assert 230.0 < mean(out.lw_flux_up[:, 0]) < 330.0


def test_the_land_patch_runs_a_day(tmp_path):
    from legoesm.driver.model_driver import ModelDriver

    cfg = _config(tmp_path / "land")
    driver = ModelDriver(cfg)
    driver.setup()
    assert driver.model.land.mask.sum() > 0 and driver.model.radiation is not None
    assert driver.run() == "COMPLETED"
    t_soil = np.asarray(driver.model.carry["land"].T_soil.data)[np.asarray(driver.model.land.mask)]
    assert np.isfinite(t_soil).all() and (200.0 < t_soil).all() and (t_soil < 350.0).all()
    snap = dict(np.load(tmp_path / "land" / "icon_snapshot_step_000288_rank0.npz"))
    assert all(np.isfinite(v).all() for k, v in snap.items() if k.endswith("_0"))
