"""icon4py's JAX non-hydrostatic dynamics behind legoESM's column loop
(``legoesm.grids.column_mesh.ColumnModel``): ICON is the dynamical core and legoESM's column
physics is applied once per ICON time step, through icon4py's own column interface.

The step is ``dynamics_step``, ``dyn2phy``, legoESM's physics on the column view,
``apply_column_tendencies`` and ``phy2dyn``. The view is T, cell u/v, the total surface pressure
(ICON's lowest half-level pressure) and the tracers as mass fractions of total air; the vertical
coordinate handed to the physics is the columns' own pressures (:class:`ColumnPressures`).

legoESM's physics heats at constant pressure; ICON's physics interface applies heating at
constant volume. The temperature tendency is therefore converted at equal energy:
``dT_icon = dT * c_pd * air_mass / cv_air``. The physics' water stays in the air's mass: each
layer's density changes by the water the physics added or removed, with the dry air held
(``rho' = rho * (1 + q_t - q_t_before)``, every tracer keeping its mass per volume), so the dry
mass is conserved and the total follows the surface water exchange. ``phy2dyn`` then diagnoses
exner and theta_v from the equation of state with ``rho'`` and the post-physics virtual
temperature. Tracers legoESM carries beyond ICON's six moisture
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


def _water_fraction(col):
    """Each layer's total water [kg/kg]: the six species."""
    return sum(col.q[sp].ndarray for sp in ICON_SPECIES.values())


def _column_water(col):
    """Each column's water [kg/m2]: the six species weighted by ICON's layer air mass."""
    return jnp.sum(_water_fraction(col) * col.air_mass.ndarray, axis=-1)


def _rate(tend, key, like):
    """A surface water rate of the physics output [kg/m2/s], zero where it publishes none."""
    rate = getattr(tend, key, None)
    return jnp.zeros_like(like) if rate is None else getattr(rate, "data", rate)


def _field(data, name, units):
    return Field(data=data, name=name, dims=(), units=units)


class IconColumnModel:
    """icon4py's experiment on one rank, stepped under the column loop's contract.

    ``experiment_config`` is icon4py's ``ExperimentConfig`` (``external_physics`` on, or a physics
    group configured, so its column interface exists; the model never calls icon4py's physics),
    ``mesh`` the ICON
    ``ColumnMesh``, ``held_tracers`` the legoESM tracers beyond ICON's six."""

    def __init__(self, *, grid_file, experiment_config, mesh, config, held_tracers=(),
                 conservative_tracer_clamp=True):
        from icon4py.model.driver import jax_api

        self._api = jax_api
        self.config = config
        self.mesh = mesh
        self._state_type = IconColumnState
        self.held_tracers = tuple(held_tracers)
        #: the other lanes' end-of-step positivity stage (apply_water_positivity): borrow the
        #: negative water back within each column (True) or floor it (False)
        self.conservative_tracer_clamp = bool(conservative_tracer_clamp)
        self._static, self._initial = jax_api.init_from_grid_file(
            pathlib.Path(grid_file), experiment_config
        )
        self.dt = float(self._static.dtime)
        self._vertical_cfl_threshold = float(experiment_config.driver.vertical_cfl_threshold)
        self.steps_done = 0
        #: state snapshots: ``snapshot_dir`` None = off; every ``snapshot_every`` steps, and on a
        #: non-finite step the failing state and the last ``snapshot_ring`` finite ones
        self.snapshot_dir = None
        self.snapshot_every = 0
        self.snapshot_ring = 12
        self._ring = []
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
        # the surface water the physics exchanged since the start [kg/m2]: precipitation down,
        # evaporation up
        return {"icon": self._initial,
                "held": {name: jnp.zeros((n, k)) for name in self.held_tracers},
                "acc": {"precip": jnp.zeros((n,)), "evap": jnp.zeros((n,)),
                        # the column water [kg/m2] the physics added (ICON's layer air mass as
                        # the weight) and the positivity stage added
                        "water_physics": jnp.zeros((n,)), "water_positivity": jnp.zeros((n,))}}

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
    # snapshots
    # ------------------------------------------------------------------

    def write_snapshot(self, native, step, tag):
        """All prognostics, the six species, air mass, T and the lowest half-level pressure of
        ``native``, with the cell areas, to ``snapshot_dir``/``{tag}_step_{step:06d}.npz``."""
        import numpy as np

        col = self._api.dyn2phy(self._static, native["icon"])
        fields = {f"prog_{k}": v.ndarray for k, v in native["icon"].prognostic.items()}
        fields.update({f"q_{k}": v.ndarray for k, v in col.q.items()})
        fields.update({f"held_{k}": v for k, v in native["held"].items()})
        fields.update({f"acc_{k}": v for k, v in native["acc"].items()})
        fields.update(air_mass=col.air_mass.ndarray, temperature=col.temperature.ndarray,
                      p_surface=col.pressure_ifc.ndarray[:, -1], area=self.mesh.areaCell)
        out = pathlib.Path(self.snapshot_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.savez(out / f"{tag}_step_{step:06d}.npz", step=step, dt=self.dt,
                 **{k: np.asarray(v) for k, v in fields.items()})

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

    def _positivity(self, col, held):
        """The shared positivity stage on the post-physics columns, weighted by ICON's layer air
        mass: the six water species borrowed within each column, the held tracers floored."""
        import dataclasses

        from legoesm.core.conservation import apply_water_positivity

        tracers = {name: col.q[sp].ndarray for name, sp in ICON_SPECIES.items()}
        tracers.update(held)
        fixed, _ = apply_water_positivity(
            tracers, None, col.air_mass.ndarray, conservative=self.conservative_tracer_clamp,
            energy_consistent=False, area=self.mesh.areaCell)
        q = dict(col.q)
        for name, sp in ICON_SPECIES.items():
            q[sp] = type(col.q[sp]).from_array(fixed[name], domain=col.q[sp].domain)
        return dataclasses.replace(col, q=q), {name: fixed[name] for name in held}

    def _hold_dry_mass(self, icon, col, held, water_before):
        """``icon``'s density with the water the physics added since ``water_before`` (each
        layer's total water fraction before it), the dry air held; the column's and the held
        tracers rescaled to keep their mass per volume."""
        import dataclasses

        ratio = 1.0 + _water_fraction(col) - water_before
        rho = icon.prognostic["rho"]
        rho = type(rho).from_array(rho.ndarray * ratio, domain=rho.domain)
        q = {sp: type(v).from_array(v.ndarray / ratio, domain=v.domain) for sp, v in col.q.items()}
        return (dataclasses.replace(icon, prognostic={**icon.prognostic, "rho": rho}),
                dataclasses.replace(col, q=q), {name: v / ratio for name, v in held.items()})

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
                    new = {"icon": icon, "held": held, "acc": nat["acc"]}
                    return self._view_of(new, col), ps, None
                view = self._view_of({"icon": icon, "held": held, "acc": nat["acc"]}, col)
                pressures = ColumnPressures(col.pressure.ndarray, col.pressure_ifc.ndarray)
                pr = physics_fn(view, self.mesh, pressures, phys_state=ps, forcing=fo)
                tend, ps_out = (pr[0], pr[1]) if type(pr) is tuple else (pr, ps)
                icon_tend = self.icon_tendencies(col, tend)
                water_dyn = _column_water(col)
                fraction_dyn = _water_fraction(col)
                col = api.apply_column_tendencies(
                    col, icon_tend, static.dtime, recompute_virtual_temperature=False)
                held = {name: v + static.dtime * tend.tracer_tendencies[name].data
                        if tend.tracer_tendencies and name in tend.tracer_tendencies else v
                        for name, v in held.items()}
                water_phys = _column_water(col)
                col, held = self._positivity(col, held)
                water_pos = _column_water(col)
                icon, col, held = self._hold_dry_mass(icon, col, held, fraction_dyn)
                icon = api.phy2dyn(static, icon, col, icon_tend)
                sfc = (getattr(tend, "sw_net_sfc", None), getattr(tend, "lw_net_sfc", None),
                       getattr(tend, "precip", None)) + tuple(
                    getattr(tend, k, None) for k in MPAS_SFC_DIAG_EXTRA_KEYS)
                acc = {name: nat["acc"][name] + static.dtime * _rate(tend, key, nat["acc"][name])
                       for name, key in (("precip", "precip"), ("evap", "evap_sfc"))}
                acc["water_physics"] = nat["acc"]["water_physics"] + water_phys - water_dyn
                acc["water_positivity"] = nat["acc"]["water_positivity"] + water_pos - water_phys
                new = {"icon": icon, "held": held, "acc": acc}
                return (self._view(new), ps_out,
                        None if all(s is None for s in sfc) else sfc)

            fn = jax.jit(_step)
            self._fns[physics_fn] = fn
        state_new, phys_out, sfc_diag = fn(native, forcing, phys_state)
        self.steps_done += 1
        prognostic = {**state_new.native["icon"].prognostic, **state_new.native["icon"].tracers}
        finite = jnp.stack([jnp.all(jnp.isfinite(getattr(v, "ndarray", v)))
                            for v in prognostic.values()])
        if not bool(jnp.all(finite)):
            bad = [k for k, ok in zip(prognostic, list(finite), strict=True) if not bool(ok)]
            if self.snapshot_dir is not None:
                self.write_snapshot(state_new.native, self.steps_done, "nonfinite")
                for step, kept in self._ring:
                    self.write_snapshot(kept, step, "before_nonfinite")
            raise FloatingPointError(
                f"step {self.steps_done}: first non-finite fields {bad}")
        if self.snapshot_dir is not None:
            self._ring = (self._ring + [(self.steps_done, state_new.native)])[-self.snapshot_ring:]
            if self.snapshot_every and self.steps_done % self.snapshot_every == 0:
                self.write_snapshot(state_new.native, self.steps_done, "daily")
        cfl = state_new.native["icon"].carry["diagnostic"]["max_vertical_cfl"]
        cfl = float(getattr(cfl, "ndarray", cfl))
        if not cfl <= self._vertical_cfl_threshold:
            # the closed lane's guard: icon4py's driver would add dynamical substeps here, the
            # pure-step API runs a fixed number
            if self.snapshot_dir is not None:
                self.write_snapshot(state_new.native, self.steps_done, "cfl")
                for step, kept in self._ring[:-1]:
                    self.write_snapshot(kept, step, "before_cfl")
            raise RuntimeError(
                f"step {self.steps_done}: maximum vertical CFL {cfl:.6g} exceeds "
                f"vertical_cfl_threshold {self._vertical_cfl_threshold:g}")
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
