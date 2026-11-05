"""icon4py's JAX dycore through the legoESM adapter: JW R02B04, 3 steps, vs standalone icon4py."""

import os
import pathlib
import subprocess
import sys
import textwrap

import numpy as np
import pytest

pytest.importorskip("icon4py.model.driver.jax_driver")
if "ICON4PY_TEST_DATA_PATH" not in os.environ:
    pytest.skip("ICON4PY_TEST_DATA_PATH is not set", allow_module_level=True)

from legoesm.atmosphere.dynamics.gcm import icon4py_jax  # noqa: E402

pytestmark = pytest.mark.slow

DATA = pathlib.Path(os.environ["ICON4PY_TEST_DATA_PATH"])
EXPERIMENT = DATA / "ser_icondata" / "mpitask1_exclaim_nh35_tri_jws_v12"
GRID_FILE = DATA / "grids" / "icon_grid_0013_R02B04_R" / "icon_grid_0013_R02B04_R.nc"
STEP_DATES = ("2008-09-01T00:05:00.000", "2008-09-01T00:10:00.000", "2008-09-01T00:15:00.000")

# icon4py's JW double-precision tolerances (atol, rtol), model/driver/tests/.../test_driver.py
SAVEPOINT_TOLERANCES = {
    "vn": (5.3e-7, 0.0),
    "w": (8e-9, 0.0),
    "exner": (4.5e-11, 5.5e-11),
    "theta_v": (5.5e-8, 1.3e-10),
    "rho": (1.5e-10, 2.2e-10),
}

_REFERENCE = textwrap.dedent(
    """
    import pathlib, sys
    import numpy as np
    from icon4py.model.common import model_backends
    from icon4py.model.common.decomposition import definitions as decomposition
    from icon4py.model.driver import config as driver_config, driver, driver_utils

    grid_file, experiment_config, output_path, out = map(pathlib.Path, sys.argv[1:5])
    config = driver_config.read_experiment_config_from_yaml(experiment_config)
    n_steps = int(sys.argv[5])
    config = config.with_overrides(
        driver={"output_path": output_path, "end_of_simulation": n_steps}
    )
    process_props = decomposition.SingleNodeProcessProperties()
    grid_manager = driver_utils.create_grid_manager(
        grid_file_path=grid_file,
        vertical_grid_config=config.vertical_grid,
        allocator=model_backends.get_allocator(None),
        process_props=process_props,
    )
    fields = ("vn", "w", "rho", "exner", "theta_v")

    def record(time_step, state):
        np.savez(out / f"step{time_step}.npz", **{n: getattr(state, n).asnumpy() for n in fields})

    driver.run_driver(
        config=config, grid_manager=grid_manager, process_props=process_props, backend=None,
        jax=True, jax_jit_time_step=True, on_step_end=record,
    )
    """
)


def _savepoint_fields(date: str) -> dict[str, np.ndarray]:
    import serialbox

    serializer = serialbox.Serializer(
        serialbox.OpenModeKind.Read, str(EXPERIMENT / "ser_data"), "icon_pydycore_rank0"
    )
    savepoint = serializer.savepoint["time-step-exit"].id[1].date[date].as_savepoint()
    return {
        name: np.squeeze(serializer.read(name, savepoint).astype(np.float64))
        for name in icon4py_jax.PROGNOSTIC_FIELDS
    }


@pytest.fixture(scope="module")
def adapter_steps(tmp_path_factory):
    dycore = icon4py_jax.Icon4pyJaxDycore(
        grid_file=GRID_FILE,
        experiment_config=EXPERIMENT / "config.yml",
        output_path=tmp_path_factory.mktemp("adapter_output"),
    )
    steps = []
    for _ in STEP_DATES:
        dycore.step()
        steps.append(dycore.prognostics())
    return steps


@pytest.fixture(scope="module")
def reference_steps(tmp_path_factory):
    out = tmp_path_factory.mktemp("reference")
    subprocess.run(
        [
            sys.executable,
            "-c",
            _REFERENCE,
            str(GRID_FILE),
            str(EXPERIMENT / "config.yml"),
            str(tmp_path_factory.mktemp("reference_output")),
            str(out),
            str(len(STEP_DATES)),
        ],
        check=True,
    )
    return [dict(np.load(out / f"step{i}.npz")) for i in range(len(STEP_DATES))]


def test_bitwise_against_standalone_icon4py(adapter_steps, reference_steps):
    assert len(reference_steps) == len(adapter_steps) == len(STEP_DATES)
    for step, (ours, theirs) in enumerate(zip(adapter_steps, reference_steps, strict=True)):
        for name in icon4py_jax.PROGNOSTIC_FIELDS:
            assert ours[name].dtype == theirs[name].dtype == np.float64
            np.testing.assert_array_equal(ours[name], theirs[name], err_msg=f"step {step} {name}")


def test_against_icon_savepoints(adapter_steps):
    for step, date in enumerate(STEP_DATES):
        reference = _savepoint_fields(date)
        for name, (atol, rtol) in SAVEPOINT_TOLERANCES.items():
            ours = adapter_steps[step][name]
            theirs = reference[name][: ours.shape[0]]
            assert theirs.shape == ours.shape, (name, theirs.shape, ours.shape)
            print(f"{date} {name}: max |adapter - savepoint| = {np.max(np.abs(ours - theirs)):.2e}")
            np.testing.assert_allclose(ours, theirs, atol=atol, rtol=rtol, err_msg=f"{date} {name}")
