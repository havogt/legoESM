"""The ICON column lane through legoESM's driver: a day of icon4py's dynamics under the column loop
(every legoESM physics scheme off) completes, stays finite and keeps the dycore's air mass."""

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


def _air_mass(model, native):
    col = model._api.dyn2phy(model._static, native["icon"])
    area = np.asarray(model.mesh.areaCell)
    return float(np.sum(area[:, None] * np.asarray(col.air_mass.ndarray)))


def test_a_day_of_the_column_lane(tmp_path):
    from legoesm.atmosphere.dynamics.gcm.icon_column import IconColumnModel
    from legoesm.driver.model_driver import ModelDriver
    from legoesm.grids.column_pressures import ColumnPressures

    config = Config.from_yaml(str(REPO / "config" / "icon" / "ape_column_r2b4.yaml"))
    config.set("grid.grid_file", str(GRID_FILE))
    config.set("output.path", str(tmp_path / "out"))
    driver = ModelDriver(config.to_experiment_config())
    driver.setup()
    assert isinstance(driver.model, IconColumnModel)
    assert isinstance(driver.model.sigma_coord, ColumnPressures)
    mass0 = _air_mass(driver.model, driver.state.native)
    assert driver.run() == "COMPLETED"
    native = driver.state.native
    for name, value in native["icon"].prognostic.items():
        assert np.isfinite(np.asarray(value.ndarray)).all(), name
    # icon4py's dycore conserves the total air mass; the column loop with no physics must too
    assert _air_mass(driver.model, native) == pytest.approx(mass0, rel=1e-12)
