"""The ICON lane end to end: legoESM's driver runs the JW deck for a day, bitwise equal to the
adapter on icon4py's own EXCLAIM experiment file."""

import os
import pathlib

import numpy as np
import pytest

pytest.importorskip("icon4py.model.driver.jax_api")
if "ICON4PY_TEST_DATA_PATH" not in os.environ:
    pytest.skip("ICON4PY_TEST_DATA_PATH is not set", allow_module_level=True)

from legoesm.atmosphere.dynamics.gcm import icon4py_jax  # noqa: E402

from legoesm.config import Config  # noqa: E402

pytestmark = pytest.mark.slow

REPO = pathlib.Path(__file__).resolve().parents[4]
DATA = pathlib.Path(os.environ["ICON4PY_TEST_DATA_PATH"])
GRID_FILE = DATA / "grids" / "icon_grid_0013_R02B04_R" / "icon_grid_0013_R02B04_R.nc"
EXCLAIM = DATA / "ser_icondata" / "mpitask1_exclaim_nh35_tri_jws_v12" / "config.yml"
STEPS = 288


def test_jw_deck_through_the_driver_matches_the_adapter(tmp_path):
    from icon4py.model.driver import config as driver_config
    from legoesm.driver.model_driver import ModelDriver

    config = Config.from_yaml(str(REPO / "config" / "icon" / "jw_r2b4.yaml"))
    config.set("grid.grid_file", str(GRID_FILE))
    config.set("output.path", str(tmp_path / "lego"))
    driver = ModelDriver(config.to_experiment_config())
    driver.setup()
    assert driver.run() == "COMPLETED"
    assert (tmp_path / "lego" / "icon_status.txt").read_text().strip() == "COMPLETED"
    lego = dict(np.load(tmp_path / "lego" / f"icon_snapshot_step_{STEPS:06d}_rank0.npz"))

    reference = icon4py_jax.Icon4pyJaxDycore(
        grid_file=GRID_FILE,
        experiment_config=driver_config.read_experiment_config_from_yaml(EXCLAIM).with_overrides(
            driver={"output_path": tmp_path / "reference"}
        ),
    )
    for _ in range(STEPS):
        reference.step()
    expected = reference.prognostics()
    assert set(expected) == set(icon4py_jax.PROGNOSTIC_FIELDS)
    for name, value in expected.items():
        np.testing.assert_array_equal(lego[f"{name}_0"], value, err_msg=name)
