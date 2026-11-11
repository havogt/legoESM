"""legoESM land on the ICON lane, end to end: the land step on ICON's columns, the patch refused
until the lane has radiation, and a patch without cells reproducing APE_AES bit for bit."""

import os
import pathlib

import numpy as np
import pytest

pytest.importorskip("icon4py.model.driver.jax_api")
if "ICON4PY_TEST_DATA_PATH" not in os.environ:
    pytest.skip("ICON4PY_TEST_DATA_PATH is not set", allow_module_level=True)

from legoesm.config import Config  # noqa: E402

pytestmark = pytest.mark.slow

REPO = pathlib.Path(__file__).resolve().parents[4]
DATA = pathlib.Path(os.environ["ICON4PY_TEST_DATA_PATH"])
GRID_FILE = DATA / "grids" / "icon_grid_0013_R02B04_R" / "icon_grid_0013_R02B04_R.nc"
STEPS = 288


def _config(deck, out, **sets):
    config = Config.from_yaml(str(REPO / "config" / "icon" / f"{deck}.yaml"))
    config.set("grid.grid_file", str(GRID_FILE))
    config.set("output.path", str(out))
    for key, value in sets.items():
        config.set(key, value)
    return config.to_experiment_config()


NO_RADIATION = {"radiation.scheme": "none", "icon.radiation": None}


def test_land_cells_without_radiation_are_refused(tmp_path):
    from legoesm.driver import component_factory

    with pytest.raises(ValueError, match=r"\d+ cells of the patch are land and icon.radiation"):
        component_factory.create_icon_model(
            _config("ape_aes_land_r2b4", tmp_path, **NO_RADIATION))


def test_the_land_step_on_icon_columns(tmp_path):
    import dataclasses

    from icon4py.model.driver import jax_api
    from legoesm.atmosphere.dynamics.gcm import icon4py_jax
    from legoesm.driver import icon_land
    from legoesm.grids.factory import create_grid

    cfg = _config("ape_aes_land_r2b4", tmp_path)
    land = cfg.icon.land
    static, state = jax_api.init_from_grid_file(
        GRID_FILE, icon4py_jax.build_icon4py_experiment_config(cfg))
    col = jax_api.dyn2phy(static, state)
    _, muphys = jax_api.muphys(static, col)
    low = icon_land.lowest_level(col)
    assert (np.asarray(low["z"]) > 0.0).all()
    mesh = create_grid("icon", grid_file=GRID_FILE)
    land_state, _ = icon_land.initial_land_state(land, low["T"])
    fluxes, new, _ = icon_land.land_step(land, mesh.latCell, static.dtime)(
        col, muphys, land_state, None)
    template = jax_api.surface(static, jax_api.surface_input(static, col))
    assert set(fluxes) == {f.name for f in dataclasses.fields(template)}
    for name, value in fluxes.items():
        assert np.isfinite(np.asarray(value)).all(), name
    # the soil starts at the air temperature, so the sensible heat flux is small but present
    assert np.any(np.asarray(fluxes["thermal_energy_flux"]) != 0.0)
    assert np.any(np.asarray(new.T_soil.data) != np.asarray(land_state.T_soil.data))
    np.testing.assert_array_equal(np.asarray(land_state.W_bucket.data),
                                  land.bucket_init_frac * land.slab.W_max)


def _run(cfg):
    from legoesm.driver.model_driver import ModelDriver

    driver = ModelDriver(cfg)
    driver.setup()
    assert driver.run() == "COMPLETED"
    out = pathlib.Path(cfg.output.output_dir)
    return driver, dict(np.load(out / f"icon_snapshot_step_{STEPS:06d}_rank0.npz"))


def test_a_patch_without_cells_reproduces_ape_aes_bitwise(tmp_path):
    empty = {"icon.land.patch_lat_min_deg": 0.0, "icon.land.patch_lat_max_deg": 0.0,
             **NO_RADIATION}
    driver, with_land_path = _run(_config("ape_aes_land_r2b4", tmp_path / "land", **empty))
    assert driver.model.land is not None and not driver.model.land.mask.any()
    _, reference = _run(_config("ape_aes_r2b4", tmp_path / "ape"))
    names = [k for k in reference if k.endswith("_0")]
    assert len(names) >= 11
    for name in names:
        np.testing.assert_array_equal(with_land_path[name], reference[name], err_msg=name)


def test_land_cells_change_the_state(tmp_path):
    """The gate above is not vacuous: the same land path with the patch's cells (stepped
    directly, past the factory's refusal) moves the state away from APE_AES."""
    import functools

    from legoesm.atmosphere.dynamics.gcm import icon4py_jax
    from legoesm.driver import icon_land
    from legoesm.grids.factory import create_grid

    cfg = _config("ape_aes_land_r2b4", tmp_path / "land", **NO_RADIATION)
    land = cfg.icon.land
    mesh = create_grid("icon", grid_file=GRID_FILE)
    coupling = icon4py_jax.IconLandCoupling(
        mask=icon_land.patch_mask(mesh, land),
        initial_state=functools.partial(icon_land.initial_land_state, land),
        step=icon_land.land_step(land, mesh.latCell, float(cfg.dycore.dt)),
    )
    assert coupling.mask.any()
    experiment = icon4py_jax.build_icon4py_experiment_config(cfg)
    with_land = icon4py_jax.Icon4pyJaxDycore(
        grid_file=GRID_FILE, experiment_config=experiment, land=coupling)
    without = icon4py_jax.Icon4pyJaxDycore(grid_file=GRID_FILE, experiment_config=experiment)
    for _ in range(2):
        with_land.step()
        without.step()
    a, b = with_land.prognostics(), without.prognostics()
    differs = np.asarray(a["theta_v"] != b["theta_v"]).any(axis=1)
    assert differs[coupling.mask].all()
