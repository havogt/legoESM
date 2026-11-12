"""icon4py's JAX non-hydrostatic dynamics behind legoESM's column loop
(``legoesm.grids.column_mesh.ColumnModel``): ICON is the dynamical core and legoESM's column
physics is applied once per ICON time step, through icon4py's own column interface.

The step is ``dynamics_step``, ``dyn2phy``, legoESM's physics on the column view,
``apply_column_tendencies`` and ``phy2dyn``. The view is T, cell u/v, the total surface pressure
(ICON's lowest half-level pressure) and the tracers as mass fractions of total air; the vertical
coordinate handed to the physics is the columns' own pressures (:class:`ColumnPressures`).

legoESM's physics heats at constant pressure; ICON's physics interface applies heating at
constant volume (rho is held). The temperature tendency is therefore converted at equal energy:
``dT_icon = dT * c_pd * air_mass / cv_air``. Tracers legoESM carries beyond ICON's six moisture
species (number concentrations) are held by the model and are not advected.
"""

from __future__ import annotations

import pathlib
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from legoesm.core.field import Field
from legoesm.grids.column_pressures import ColumnPressures
from legoesm.timestepping.integration import refuse_unthreaded_stateful_physics

from legoesm import constants

#: legoESM tracer name -> icon4py moisture species
ICON_SPECIES = {"q_v": "qv", "q_c": "qc", "q_r": "qr", "q_i": "qi", "q_s": "qs", "q_g": "qg"}


class IconColumnState(NamedTuple):
    """``HydrostaticState`` field order plus the native state (icon4py's and the held tracers)."""

    u: Field
    T: Field
    p_s: Field
    phis: Field
    v: Field | None = None
    tracers: dict | None = None
    native: Any = None


def _field(data, name, units):
    return Field(data=data, name=name, dims=(), units=units)


class IconColumnModel:
    """icon4py's experiment on one rank, stepped under the column loop's contract.

    ``experiment_config`` is icon4py's ``ExperimentConfig`` (with a physics group configured, so
    its column interface exists; the column model never calls it), ``mesh`` the ICON
    ``ColumnMesh``, ``held_tracers`` the legoESM tracers beyond ICON's six."""

    def __init__(self, *, grid_file, experiment_config, mesh, config, held_tracers=()):
        from icon4py.model.driver import jax_api

        self._api = jax_api
        self.config = config
        self.mesh = mesh
        self._state_type = IconColumnState
        self.held_tracers = tuple(held_tracers)
        self._static, self._initial = jax_api.init_from_grid_file(
            pathlib.Path(grid_file), experiment_config
        )
        self.dt = float(self._static.dtime)
        self._phys_state = None
        self._sfc_diag = None
        self._last = None
        self._fns = {}
        col = jax_api.dyn2phy(self._static, self._initial)
        self.sigma_coord = ColumnPressures(col.pressure.ndarray, col.pressure_ifc.ndarray)

    def init_state(self, driver) -> None:
        """Fresh IC: icon4py's own (the experiment's analytic initial state)."""
        driver.state = self.from_bundle(self.initial_native())

    def initial_native(self):
        """icon4py's initial state and zero held tracers."""
        n, k = self.sigma_coord.p_full.shape
        return {"icon": self._initial,
                "held": {name: jnp.zeros((n, k)) for name in self.held_tracers}}

    # ------------------------------------------------------------------
    # views
    # ------------------------------------------------------------------

    def _view_of(self, native, col):
        tracers = {name: _field(col.q[sp].ndarray, name, "kg/kg")
                   for name, sp in ICON_SPECIES.items()}
        tracers.update({name: _field(v, name, "1/kg") for name, v in native["held"].items()})
        p_s = col.pressure_ifc.ndarray[:, -1]
        return IconColumnState(
            u=_field(col.u.ndarray, "u", "m/s"),
            T=_field(col.temperature.ndarray, "T", "K"),
            p_s=_field(p_s, "p_s", "Pa"),
            # flat lower boundary only, so far
            phis=_field(jnp.zeros_like(p_s), "phis", "m2/s2"),
            v=_field(col.v.ndarray, "v", "m/s"),
            tracers=tracers,
            native=native,
        )

    def _view(self, native):
        return self._view_of(native, self._api.dyn2phy(self._static, native["icon"]))

    def column_view(self, native):
        return self._view(native)

    def from_bundle(self, native):
        fn = self._fns.get("view")
        if fn is None:
            fn = jax.jit(self._view)
            self._fns["view"] = fn
        self._last = fn(native)
        return self._last

    def to_bundle(self, state):
        return self._native_of(state)

    def _native_of(self, state):
        """The columns are a view; the driver may not edit them on this lane yet."""
        last = self._last
        if last is None or state.native is not last.native:
            raise ValueError(
                "IconColumnModel.step: the column state is not the one this model returned "
                "(build states with from_bundle and thread step's output)")
        for name in ("u", "v", "T", "p_s", "phis", "tracers"):
            if getattr(state, name) is not getattr(last, name):
                raise ValueError(
                    f"IconColumnModel.step: state.{name} was edited outside the model; the "
                    "ICON column lane writes no edits back yet")
        return state.native

    # ------------------------------------------------------------------
    # the step
    # ------------------------------------------------------------------

    def icon_tendencies(self, col, tend):
        """legoESM's physics tendencies as icon4py ``ColumnTendencies``: heating converted from
        constant pressure to constant volume at equal energy, the six species, u and v."""
        from icon4py.model.common.physics.thermodynamics import compute_energy

        q = col.q
        cv_air = compute_energy._compute_moist_air_isochoric_heat_capacity_per_area.definition(
            q["qv"], q["qc"], q["qi"], q["qr"], q["qs"], q["qg"], col.air_mass
        ).ndarray
        heating = tend.dT_dt.data * constants.c_pd * col.air_mass.ndarray / cv_air

        def field(like, value):
            return type(like).from_array(value, domain=like.domain)

        species = {}
        for name, sp in ICON_SPECIES.items():
            rate = (tend.tracer_tendencies or {}).get(name)
            if rate is not None:
                species[sp] = field(q[sp], rate.data)
        if tend.dv_dt is None:
            raise ValueError("IconColumnModel: the physics returned dv_dt=None; the ICON column "
                             "lane carries both cell wind components")
        return self._api.ColumnTendencies(
            temperature=field(col.temperature, heating),
            q=species,
            u=field(col.u, tend.du_dt.data),
            v=field(col.v, tend.dv_dt.data),
        )

    def step(self, state, dt, physics_fn=None, forcing=None, phys_state=None):
        """One ICON time step (``dt`` must be ICON's ``dtime``), the physics after the
        dynamics. Returns the new column state; stashes ``_phys_state``/``_sfc_diag`` like the
        other column models."""
        from legoesm.core.state import MPAS_SFC_DIAG_EXTRA_KEYS

        if float(dt) != self.dt:
            raise ValueError(f"IconColumnModel.step: dt={dt} is not ICON's dtime {self.dt}")
        refuse_unthreaded_stateful_physics(
            physics_fn, phys_state, where="IconColumnModel.step()")
        native = self._native_of(state)
        fn = self._fns.get(physics_fn)
        if fn is None:
            api, static = self._api, self._static

            def _step(nat, fo, ps):
                icon = api.dynamics_step(static, nat["icon"])
                col = api.dyn2phy(static, icon)
                held = nat["held"]
                if physics_fn is None:
                    new = {"icon": icon, "held": held}
                    return self._view_of(new, col), ps, None
                view = self._view_of({"icon": icon, "held": held}, col)
                pressures = ColumnPressures(col.pressure.ndarray, col.pressure_ifc.ndarray)
                pr = physics_fn(view, self.mesh, pressures, phys_state=ps, forcing=fo)
                tend, ps_out = (pr[0], pr[1]) if type(pr) is tuple else (pr, ps)
                icon_tend = self.icon_tendencies(col, tend)
                col = api.apply_column_tendencies(
                    col, icon_tend, static.dtime, recompute_virtual_temperature=False)
                icon = api.phy2dyn(static, icon, col, icon_tend)
                held = {name: v + static.dtime * tend.tracer_tendencies[name].data
                        if tend.tracer_tendencies and name in tend.tracer_tendencies else v
                        for name, v in held.items()}
                sfc = (getattr(tend, "sw_net_sfc", None), getattr(tend, "lw_net_sfc", None),
                       getattr(tend, "precip", None)) + tuple(
                    getattr(tend, k, None) for k in MPAS_SFC_DIAG_EXTRA_KEYS)
                new = {"icon": icon, "held": held}
                return (self._view(new), ps_out,
                        None if all(s is None for s in sfc) else sfc)

            fn = jax.jit(_step)
            self._fns[physics_fn] = fn
        state_new, phys_out, sfc_diag = fn(native, forcing, phys_state)
        self._last = state_new
        col_new = self._api.dyn2phy(self._static, state_new.native["icon"])
        self.sigma_coord = ColumnPressures(col_new.pressure.ndarray, col_new.pressure_ifc.ndarray)
        if physics_fn is not None:
            self._phys_state = phys_out
        if sfc_diag is not None:
            prev = self._sfc_diag or (None,) * len(sfc_diag)
            self._sfc_diag = tuple(new if new is not None else old
                                   for new, old in zip(sfc_diag, prev, strict=True))
        return state_new
