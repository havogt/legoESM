"""icon4py's dynamics behind legoESM's column loop: with no physics increment the step is ICON's
own dynamics + column interface round trip, legoESM's heating lands as equal energy at constant
volume, the physics' water changes the density with the dry air held, and the tracer tendencies
arrive."""

import os
import pathlib

import numpy as np
import pytest

pytest.importorskip("icon4py.model.driver.jax_api")
if "ICON4PY_TEST_DATA_PATH" not in os.environ:
    pytest.skip("ICON4PY_TEST_DATA_PATH is not set", allow_module_level=True)

import jax.numpy as jnp  # noqa: E402
from legoesm.core.field import Field  # noqa: E402
from legoesm.core.state import HydrostaticTendencies  # noqa: E402

from legoesm import constants  # noqa: E402

pytestmark = pytest.mark.slow

DATA = pathlib.Path(os.environ["ICON4PY_TEST_DATA_PATH"])
GRID_FILE = DATA / "grids" / "icon_grid_0013_R02B04_R" / "icon_grid_0013_R02B04_R.nc"
APE = DATA / "ser_icondata" / "mpitask1_exclaim_ape_aesPhys_v14" / "config.yml"
DT = 300.0


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    from icon4py.model.driver import config as driver_config
    from legoesm.atmosphere.dynamics.gcm.icon_column import IconColumnModel
    from legoesm.grids.factory import create_grid

    cfg = driver_config.read_experiment_config_from_yaml(APE).with_overrides(
        driver={"output_path": tmp_path_factory.mktemp("col")})
    return IconColumnModel(grid_file=GRID_FILE, experiment_config=cfg,
                           mesh=create_grid("icon", grid_file=GRID_FILE), config=None,
                           held_tracers=("N_c",))


def _physics(heating=0.0, dqv=0.0, dnc=0.0):
    def physics_fn(state, mesh, sigma_coord, phys_state=None, forcing=None):
        zero = jnp.zeros_like(state.T.data)

        def f(x, name):
            return Field(data=zero + x, name=name, dims=(), units="1/s")
        tracers = {name: f(0.0, name) for name in state.tracers}
        tracers["q_v"], tracers["N_c"] = f(dqv, "q_v"), f(dnc, "N_c")
        return HydrostaticTendencies(
            du_dt=f(0.0, "u"), dT_dt=f(heating, "T"),
            dp_s_dt=Field(data=jnp.zeros_like(state.p_s.data), name="p_s", dims=(), units="Pa/s"),
            dphis_dt=Field(data=jnp.zeros_like(state.p_s.data), name="phis", dims=(), units=""),
            dv_dt=f(0.0, "v"), tracer_tendencies=tracers)
    return physics_fn


def _step(model, physics_fn):
    state = model.from_bundle(model.initial_native())
    return model.step(state, DT, physics_fn=physics_fn)


def _prognostics(native):
    return {k: np.asarray(v.ndarray) for k, v in native["icon"].prognostic.items()}


def test_no_increment_is_icon_dynamics_and_the_column_round_trip(model):
    import jax

    api, static = model._api, model._static
    new = _step(model, _physics())
    held = model.initial_native()["held"]

    # the same composition, jitted like the step (eager and jitted differ by XLA's fusion)
    @jax.jit
    def reference(icon0):
        icon = api.dynamics_step(static, icon0)
        col = api.dyn2phy(static, icon)
        zero = model.icon_tendencies(
            col, _physics()(model._view_of({"icon": icon, "held": held}, col), model.mesh, None))
        before = sum(col.q[sp].ndarray for sp in ("qv", "qc", "qi", "qr", "qs", "qg"))
        dyn_prognostic = icon.prognostic
        col = api.apply_column_tendencies(col, zero, static.dtime,
                                          recompute_virtual_temperature=False)
        col, kept = model._positivity(col, held)
        icon, col, _ = model._hold_dry_mass(icon, col, kept, before)
        return api.phy2dyn(static, icon, col, zero).prognostic, dyn_prognostic

    ref, dyn = reference(model.initial_native()["icon"])
    got = _prognostics(new.native)
    for name, value in ref.items():
        np.testing.assert_allclose(got[name], np.asarray(value.ndarray), rtol=1e-14, atol=1e-14,
                                   err_msg=name)
    # against the dynamics alone: icon4py's dyn2phy/phy2dyn round trip (the closed lane's too)
    # moves exner and theta_v; measured 3.07e-8 relative on exner after one APE_AES step
    dyn = {k: np.asarray(v.ndarray) for k, v in dyn.items()}
    for name in ("exner", "theta_v"):
        drift = np.max(np.abs(got[name] - dyn[name]) / np.abs(dyn[name]))
        print(f"{name}: drift of the column round trip against the dynamics {drift:.3e}")
        assert 0.0 < drift < 1e-7, name
    # the drift is the equation of state re-diagnosed from the dynamics' rho and Tv: an
    # independent NumPy diagnosis (icon4py's constants) reproduces the step's exner and theta_v
    from icon4py.model.common import constants as c

    virtual_temperature = dyn["theta_v"] * dyn["exner"]
    exner = np.exp(c.RD_O_CPD * np.log(c.RD / c.P0REF * dyn["rho"] * virtual_temperature))
    for name, expected in (("exner", exner), ("theta_v", virtual_temperature / exner)):
        mismatch = np.max(np.abs(got[name] - expected) / np.abs(expected))
        print(f"{name}: against its equation-of-state diagnosis {mismatch:.3e}")
        assert mismatch < 1e-13, name
    for name in ("rho", "vn", "w"):
        np.testing.assert_allclose(got[name], dyn[name], rtol=1e-14, atol=1e-14, err_msg=name)


def test_heating_lands_as_equal_energy_at_constant_volume(model):
    from legoesm.driver import icon_radiation

    rate = 1.0e-4
    cold, warm = _step(model, _physics()), _step(model, _physics(heating=rate))
    col_cold = model._api.dyn2phy(model._static, cold.native["icon"])
    col_warm = model._api.dyn2phy(model._static, warm.native["icon"])
    cv = np.asarray(icon_radiation.cv_air(col_cold))
    gained = cv * (np.asarray(col_warm.temperature.ndarray)
                   - np.asarray(col_cold.temperature.ndarray))
    expected = constants.c_pd * np.asarray(col_cold.air_mass.ndarray) * rate * DT
    np.testing.assert_allclose(gained, expected, rtol=1e-6)
    # constant volume: the density is untouched
    np.testing.assert_array_equal(np.asarray(warm.native["icon"].prognostic["rho"].ndarray),
                                  np.asarray(cold.native["icon"].prognostic["rho"].ndarray))


def _rho(step):
    return np.asarray(step.native["icon"].prognostic["rho"].ndarray)


def _water(step):
    return sum(np.asarray(step.tracers[k].data) for k in ("q_v", "q_c", "q_i", "q_r", "q_s", "q_g"))


def test_the_physics_water_changes_the_density_with_the_dry_air_held(model):
    from icon4py.model.common import constants as c

    dqv = 1.0e-8
    base, wet = _step(model, _physics()), _step(model, _physics(dqv=dqv))
    rho_b, rho_w = _rho(base), _rho(wet)
    # the dry air is held, and the layer gains the water: rho' = rho (1 + dq)
    np.testing.assert_allclose(rho_w * (1.0 - _water(wet)), rho_b * (1.0 - _water(base)),
                               rtol=1e-13)
    np.testing.assert_allclose(rho_w / rho_b, 1.0 + dqv * DT, rtol=1e-12)
    # exner and theta_v: the equation of state with the new density and the virtual temperature
    # of the unchanged temperature and the new moisture (independent NumPy diagnosis)
    pb = base.native["icon"].prognostic
    pw = wet.native["icon"].prognostic
    def factor(st):
        q = {k: np.asarray(st.tracers[k].data) for k in ("q_v", "q_c", "q_i", "q_r", "q_s", "q_g")}
        return 1.0 + c.RV_O_RD_MINUS_1 * q.pop("q_v") - sum(q.values())

    tv_b = np.asarray(pb["theta_v"].ndarray) * np.asarray(pb["exner"].ndarray)
    tv_w = tv_b / factor(base) * factor(wet)
    exner = np.exp(c.RD_O_CPD * np.log(c.RD / c.P0REF * rho_w * tv_w))
    for name, expected in (("exner", exner), ("theta_v", tv_w / exner)):
        np.testing.assert_allclose(np.asarray(pw[name].ndarray), expected, rtol=1e-13,
                                   err_msg=name)


def test_tracer_tendencies_arrive(model):
    base, wet = _step(model, _physics()), _step(model, _physics(dqv=1.0e-8, dnc=1.0e3))
    rho_b, rho_w = _rho(base), _rho(wet)
    # per volume: the physics' increment on the pre-physics density
    added = (rho_w * np.asarray(wet.tracers["q_v"].data)
             - rho_b * np.asarray(base.tracers["q_v"].data))
    np.testing.assert_allclose(added, rho_b * 1.0e-8 * DT, rtol=1e-6, atol=1e-18)
    np.testing.assert_allclose(rho_w * np.asarray(wet.tracers["N_c"].data), rho_b * 1.0e3 * DT,
                               rtol=1e-12)
