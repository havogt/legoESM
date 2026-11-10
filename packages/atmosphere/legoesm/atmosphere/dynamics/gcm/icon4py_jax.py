"""icon4py's JAX non-hydrostatic model, stepped from legoESM.

On one rank with the whole time step jitted, a step goes through icon4py's pure-step API on
``(IconStatic, IconState)``: ``jax_api.time_step`` without physics, or, with all of muphys,
surface and tmx configured, ``dynamics_step`` followed by the physics per process group
(``per_group_time_step``, ICON's order, the surface fluxes from one replaceable call).
Otherwise (several ranks, part of the physics, or per-substep jit) ``step`` is the body of
``icon4py.model.driver.jax_driver.JaxIcon4pyDriver``'s time loop, on one rank or as icon4py's
SPMD program over all ranks. Both come from the optional ``icon`` extra.

On the ``jax_api`` path the number of dynamical substeps is fixed by the experiment. Where the
driver path would add substeps, the step's maximum vertical CFL exceeds
``vertical_cfl_threshold``, and ``step`` raises instead (checked on the host, after the jitted
step).
"""

from __future__ import annotations

import functools
import pathlib

import numpy as np

PROGNOSTIC_FIELDS = ("vn", "w", "rho", "exner", "theta_v")


def per_group_time_step(static, state, *, surface):
    """One time step: dynamics, then muphys, ``surface`` and tmx applied in turn on the columns.

    ``surface(static, surface_input)`` returns the ``SurfaceFluxState`` tmx reads. The virtual
    temperature is kept at its dyn2phy value, as icon4py's ``physics_step`` does.
    """
    from icon4py.model.driver import jax_api

    state = jax_api.dynamics_step(static, state)
    col = jax_api.dyn2phy(static, state)
    tend_muphys, _ = jax_api.muphys(static, col)
    col = jax_api.apply_column_tendencies(
        col, tend_muphys, static.dtime, recompute_virtual_temperature=False
    )
    fluxes = surface(static, jax_api.surface_input(static, col))
    tend_tmx, _ = jax_api.tmx(static, col, fluxes)
    col = jax_api.apply_column_tendencies(
        col, tend_tmx, static.dtime, recompute_virtual_temperature=False
    )
    return jax_api.phy2dyn(static, state, col, jax_api.sum_column_tendencies(tend_muphys, tend_tmx))


class Icon4pyJaxDycore:
    """Set up an icon4py experiment on the embedded backend and step it under ``jax.jit``.

    ``experiment_config`` is icon4py's ``ExperimentConfig``, from
    :func:`build_icon4py_experiment_config` on the ICON lane.
    """

    def __init__(
        self,
        *,
        grid_file: str | pathlib.Path,
        experiment_config,
        jit_time_step: bool = True,
        process_props=None,
        spmd: bool = False,
        finite_halo_padding: bool = False,
    ) -> None:
        from icon4py.model.common import dimension as dims
        from icon4py.model.common import model_backends
        from icon4py.model.common.decomposition import definitions as decomposition
        from icon4py.model.driver import driver, driver_utils, jax_driver, jax_utils

        self._jax_driver = jax_driver
        config = experiment_config
        if process_props is None:
            process_props = decomposition.SingleNodeProcessProperties()
        multi_rank = not process_props.is_single_rank()
        self.process_props = process_props
        self.cell_and_edge_dims = (dims.CellDim, dims.EdgeDim)
        groups = [config.muphys is not None, config.surface is not None, config.tmx is not None]
        self._field_dim = {"vn": dims.EdgeDim}
        self.uses_jax_api = not multi_rank and jit_time_step and (all(groups) or not any(groups))
        if self.uses_jax_api:
            import jax
            from icon4py.model.driver import jax_api

            self._api_static, self._api_state = jax_api.init_from_grid_file(
                pathlib.Path(grid_file), config
            )
            self._vertical_cfl_threshold = float(config.driver.vertical_cfl_threshold)
            self._api_step = jax.jit(
                functools.partial(per_group_time_step, surface=jax_api.surface)
                if all(groups)
                else jax_api.time_step
            )
            self.driver = None
            self.owned_global_index = {
                dims.CellDim: np.arange(self._api_state.prognostic["rho"].ndarray.shape[0]),
                dims.EdgeDim: np.arange(self._api_state.prognostic["vn"].ndarray.shape[0]),
            }
            self.steps_done = 0
            return
        if spmd and multi_rank:
            import jax

            if not jax.distributed.is_initialized():
                jax_utils.initialize_distributed(process_props.comm)
        allocator = model_backends.get_allocator(None)
        grid_manager = driver_utils.create_grid_manager(
            grid_file_path=pathlib.Path(grid_file),
            vertical_grid_config=config.vertical_grid,
            allocator=allocator,
            process_props=process_props,
            extra_halo_rings=driver_utils.jax_extra_halo_rings(config) if multi_rank else 0,
        )
        info = grid_manager.decomposition_info
        owned = decomposition.DecompositionInfo.EntryType.OWNED
        self.owned_global_index = {
            dim: np.asarray(info.global_index(dim, owned)) for dim in (dims.CellDim, dims.EdgeDim)
        }
        self.driver = driver.initialize_driver(
            config=config,
            grid_manager=grid_manager,
            process_props=process_props,
            backend=None,
            jax=True,
            jax_jit_time_step=jit_time_step,
            jax_exchange_read_fields_only=multi_rank,
            jax_spmd=spmd,
        )
        self.driver.finite_halo_padding = finite_halo_padding
        ds = driver.initialize_driver_states(
            icon4py_driver=self.driver, allocator=allocator, validate_granules=False
        )
        (
            self._prognostic,
            diagnostic,
            self._intermediate,
            self._prep_adv,
            self._tracers,
            self._constants,
        ) = self.driver._prepare(ds)
        self._diagnostic_state = jax_driver._diagnostic_from_dict(diagnostic)
        self.steps_done = 0

    def step(self) -> None:
        if self.uses_jax_api:
            self._api_state = self._api_step(self._api_static, self._api_state)
            self._api_state.prognostic["vn"].ndarray.block_until_ready()
            self.steps_done += 1
            cfl = self._api_state.carry["diagnostic"]["max_vertical_cfl"]
            cfl = float(np.asarray(getattr(cfl, "ndarray", cfl)))
            if not cfl <= self._vertical_cfl_threshold:
                raise RuntimeError(
                    f"step {self.steps_done}: maximum vertical CFL {cfl:.6g} exceeds "
                    f"vertical_cfl_threshold {self._vertical_cfl_threshold:g}; the driver path "
                    "would add dynamical substeps, the jax_api path runs a fixed number"
                )
            return
        d = self.driver
        time_vars = d.model_time_variables
        time_vars.advance_simulation_datetime()
        key = (
            time_vars.is_first_step_in_simulation,
            float(time_vars.substep_timestep),
            time_vars.ndyn_substeps_var,
            float(d._second_order_divdamp_factor()),
            float(time_vars.dtime_in_seconds),
            d._even_timestep,
        )
        args = (
            self._constants,
            self._prognostic,
            self._jax_driver._diagnostic_to_dict(self._diagnostic_state),
            self._intermediate,
            self._prep_adv,
            self._tracers,
        )
        if d._layout is not None:
            spmd_step, state_treedef = d._jitted_spmd_time_step(key, args)
            state = d._jax.tree_util.tree_unflatten(
                state_treedef, spmd_step(*d._jax.tree_util.tree_leaves(args))
            )
        else:
            if d.jit_time_step:
                d._compute_statistics(0, self._prognostic_state())
                step = d._jitted_time_step(key)
            else:
                step = functools.partial(d._time_step, key=key)
            state = step(*args)
        (
            self._prognostic,
            diagnostic,
            self._intermediate,
            self._prep_adv,
            self._tracers,
        ) = state
        self._diagnostic_state = self._jax_driver._diagnostic_from_dict(diagnostic)
        d._jax.block_until_ready(self._prognostic["vn"].ndarray)
        time_vars.is_first_step_in_simulation = False
        if self._tracers:
            d._even_timestep = not d._even_timestep
        d._adjust_ndyn_substeps_var(self._diagnostic_state)
        self.steps_done += 1

    def _prognostic_state(self):
        from icon4py.model.common.states import prognostic_state

        return prognostic_state.PrognosticState(**self._prognostic)

    def prognostics(self) -> dict[str, np.ndarray]:
        """This rank's owned points of the prognostic fields and tracers, in local order."""
        from icon4py.model.common import dimension as dims

        if self.uses_jax_api:
            local = {**self._api_state.prognostic, **self._api_state.tracers}
        else:
            local = self.driver._local({**self._prognostic, **self._tracers})
        return {
            name: np.asarray(local[name].asnumpy())[
                : self.owned_global_index[self._field_dim.get(name, dims.CellDim)].size
            ]
            for name in local
        }


# icon4py section.field -> (legoESM IconConfig block, legoESM field)
_ICON_FIELD_MAP = {
    "metrics.exner_expol": ("dycore", "exner_expol"),
    "metrics.vwind_offctr": ("dycore", "vwind_offctr"),
    "metrics.thslp_zdiffu": ("dycore", "thslp_zdiffu"),
    "metrics.thhgtd_zdiffu": ("dycore", "thhgtd_zdiffu_m"),
    "metrics.rayleigh_coeff": ("dycore", "rayleigh_coeff"),
    "metrics.divdamp_trans_start": ("dycore", "divdamp_trans_start_m"),
    "metrics.divdamp_trans_end": ("dycore", "divdamp_trans_end_m"),
    "interpolation.divergence_averaging_central_cell_weight": (
        "dycore",
        "divergence_averaging_central_cell_weight",
    ),
    "interpolation.lsq_high_ord": ("dycore", "lsq_high_ord"),
    "driver.apply_extra_second_order_divdamp": ("dycore", "apply_extra_second_order_divdamp"),
    "driver.diffuse_before_time_loop": ("dycore", "diffuse_before_time_loop"),
    "driver.vertical_cfl_threshold": ("dycore", "vertical_cfl_threshold"),
    "driver.ndyn_substeps": ("dycore", "ndyn_substeps"),
    "nonhydrostatic.itime_scheme": ("dycore", "itime_scheme"),
    "nonhydrostatic.iadv_rhotheta": ("dycore", "iadv_rhotheta"),
    "nonhydrostatic.igradp_method": ("dycore", "igradp_method"),
    "nonhydrostatic.rayleigh_type": ("dycore", "rayleigh_type"),
    "nonhydrostatic.divdamp_order": ("dycore", "divdamp_order"),
    "nonhydrostatic.divdamp_type": ("dycore", "divdamp_type"),
    "nonhydrostatic.extra_diffu": ("dycore", "extra_diffu"),
    "nonhydrostatic.rhotheta_offctr": ("dycore", "rhotheta_offctr"),
    "nonhydrostatic.veladv_offctr": ("dycore", "veladv_offctr"),
    "nonhydrostatic.fourth_order_divdamp_factor": ("dycore", "fourth_order_divdamp_factor"),
    "nonhydrostatic.fourth_order_divdamp_factor2": ("dycore", "fourth_order_divdamp_factor2"),
    "nonhydrostatic.fourth_order_divdamp_factor3": ("dycore", "fourth_order_divdamp_factor3"),
    "nonhydrostatic.fourth_order_divdamp_factor4": ("dycore", "fourth_order_divdamp_factor4"),
    "nonhydrostatic.fourth_order_divdamp_z": ("dycore", "fourth_order_divdamp_z_m"),
    "nonhydrostatic.fourth_order_divdamp_z2": ("dycore", "fourth_order_divdamp_z2_m"),
    "nonhydrostatic.fourth_order_divdamp_z3": ("dycore", "fourth_order_divdamp_z3_m"),
    "nonhydrostatic.fourth_order_divdamp_z4": ("dycore", "fourth_order_divdamp_z4_m"),
    "vertical_grid.maximal_layer_thickness": ("vertical", "maximal_layer_thickness_m"),
    "vertical_grid.top_height_limit_for_maximal_layer_thickness": (
        "vertical",
        "top_height_limit_for_maximal_layer_thickness_m",
    ),
    "vertical_grid.lowest_layer_thickness": ("vertical", "lowest_layer_thickness_m"),
    "vertical_grid.model_top_height": ("vertical", "model_top_height_m"),
    "vertical_grid.flat_height": ("vertical", "flat_height_m"),
    "vertical_grid.stretch_factor": ("vertical", "stretch_factor"),
    "vertical_grid.rayleigh_damping_height": ("vertical", "rayleigh_damping_height_m"),
    "vertical_grid.htop_moist_proc": ("vertical", "htop_moist_proc_m"),
    "vertical_grid.SLEVE_decay_scale_1": ("vertical", "sleve_decay_scale_1_m"),
    "vertical_grid.SLEVE_decay_scale_2": ("vertical", "sleve_decay_scale_2_m"),
    "vertical_grid.SLEVE_decay_exponent": ("vertical", "sleve_decay_exponent"),
    "initial_condition.p_sfc": ("jw", "p_sfc_Pa"),
    "initial_condition.baroclinic_amplitude": ("jw", "baroclinic_amplitude"),
    "initial_condition.u0": ("jw", "u0_m_s"),
    "initial_condition.temp0": ("jw", "temp0_K"),
    "initial_condition.eta_0": ("jw", "eta_0"),
    "initial_condition.eta_t": ("jw", "eta_t"),
    "initial_condition.gamma": ("jw", "gamma_K_m"),
    "initial_condition.dtemp": ("jw", "dtemp_K"),
    "initial_condition.lon_perturbation_center": ("jw", "lon_perturbation_center_rad"),
    "initial_condition.lat_perturbation_center": ("jw", "lat_perturbation_center_rad"),
    "initial_condition.rh_at_1000hpa": ("jw", "rh_at_1000hpa"),
    "initial_condition.qv_max": ("jw", "qv_max_kg_kg"),
    "initial_condition.moisture_init_iterations": ("jw", "moisture_init_iterations"),
    "initial_condition.global_moisture_content": ("jw", "global_moisture_content_kg_m2"),
    "initial_condition.normalize_global_moisture": ("jw", "normalize_global_moisture"),
    "diffusion.diffusion_type": ("diffusion", "diffusion_type"),
    "diffusion.apply_to_vertical_wind": ("diffusion", "apply_to_vertical_wind"),
    "diffusion.apply_to_horizontal_wind": ("diffusion", "apply_to_horizontal_wind"),
    "diffusion.apply_to_temperature": ("diffusion", "apply_to_temperature"),
    "diffusion.apply_smag_diff_to_vertical_wind": ("diffusion", "apply_smag_diff_to_vertical_wind"),
    "diffusion.compute_3d_smag_coeff": ("diffusion", "compute_3d_smag_coeff"),
    "diffusion.type_vn_diffu": ("diffusion", "type_vn_diffu"),
    "diffusion.type_t_diffu": ("diffusion", "type_t_diffu"),
    "diffusion.hdiff_efdt_ratio": ("diffusion", "hdiff_efdt_ratio"),
    "diffusion.hdiff_w_efdt_ratio": ("diffusion", "hdiff_w_efdt_ratio"),
    "diffusion.smagorinski_scaling_factor": ("diffusion", "smagorinski_scaling_factor"),
    "diffusion.smagorinski_scaling_factor2": ("diffusion", "smagorinski_scaling_factor2"),
    "diffusion.smagorinski_scaling_factor3": ("diffusion", "smagorinski_scaling_factor3"),
    "diffusion.smagorinski_scaling_factor4": ("diffusion", "smagorinski_scaling_factor4"),
    "diffusion.smagorinski_scaling_height": ("diffusion", "smagorinski_scaling_height_m"),
    "diffusion.smagorinski_scaling_height2": ("diffusion", "smagorinski_scaling_height2_m"),
    "diffusion.smagorinski_scaling_height3": ("diffusion", "smagorinski_scaling_height3_m"),
    "diffusion.smagorinski_scaling_height4": ("diffusion", "smagorinski_scaling_height4_m"),
    "diffusion.apply_zdiffusion_t": ("diffusion", "apply_zdiffusion_t"),
    "diffusion.temperature_boundary_diffusion_denominator": (
        "diffusion",
        "temperature_boundary_diffusion_denominator",
    ),
    "diffusion.velocity_boundary_diffusion_denominator": (
        "diffusion",
        "velocity_boundary_diffusion_denominator",
    ),
    "diffusion.shear_type": ("diffusion", "shear_type"),
    "diffusion.a_hshr": ("diffusion", "a_hshr"),
    "tracer_advection.horizontal_advection_type": ("tracer_advection", "horizontal_advection_type"),
    "tracer_advection.horizontal_advection_limiter": (
        "tracer_advection",
        "horizontal_advection_limiter",
    ),
    "tracer_advection.vertical_advection_type": ("tracer_advection", "vertical_advection_type"),
    "tracer_advection.vertical_advection_limiter": (
        "tracer_advection",
        "vertical_advection_limiter",
    ),
    "muphys.qnc": ("muphys", "qnc_per_m3"),
    "tmx.solver_type": ("tmx", "solver_type"),
    "tmx.energy_type": ("tmx", "energy_type"),
    "tmx.dissipation_factor": ("tmx", "dissipation_factor"),
    "tmx.use_louis": ("tmx", "use_louis"),
    "tmx.use_louis_land": ("tmx", "use_louis_land"),
    "tmx.use_louis_ice": ("tmx", "use_louis_ice"),
    "tmx.louis_constant_b": ("tmx", "louis_constant_b"),
    "tmx.use_km_const": ("tmx", "use_km_const"),
    "tmx.km_const": ("tmx", "km_const_m2_s"),
    "tmx.use_scale_turb_energy_flux": ("tmx", "use_scale_turb_energy_flux"),
    "tmx.scale_turb_energy_flux": ("tmx", "scale_turb_energy_flux"),
    "tmx.smag_constant": ("tmx", "smag_constant"),
    "tmx.turb_prandtl": ("tmx", "turb_prandtl"),
    "tmx.km_min": ("tmx", "km_min_m2_s"),
    "tmx.max_turb_scale": ("tmx", "max_turb_scale_m"),
    "surface.surface_flux_type": ("surface", "surface_flux_type"),
    "surface.kinematic_sensible_heat_flux": ("surface", "kinematic_sensible_heat_flux_K_m_s"),
    "surface.kinematic_latent_heat_flux": ("surface", "kinematic_latent_heat_flux_m_s"),
    "surface.sea_surface_temperature": ("surface", "sea_surface_temperature_K"),
    "metrics.igradp_method": ("dycore", "igradp_method"),
    "metrics.rayleigh_type": ("dycore", "rayleigh_type"),
    "metrics.divdamp_type": ("dycore", "divdamp_type"),
}

# Settings with no legoESM counterpart: ICON discretisation conventions, inert lateral-boundary
# nudging (pinned to the EXCLAIM experiments' values), and the switches legoESM owns itself.
_ICON_DEFAULT = object()
_ICON_PINNED = {
    "geometry.use_analytical_means": _ICON_DEFAULT,
    "vertical_grid.file_path": None,
    "vertical_grid._SLEVE_minimum_layer_thickness_1": _ICON_DEFAULT,
    "vertical_grid._SLEVE_minimum_layer_thickness_2": _ICON_DEFAULT,
    "vertical_grid._SLEVE_minimum_relative_layer_thickness_1": _ICON_DEFAULT,
    "vertical_grid._SLEVE_minimum_relative_layer_thickness_2": _ICON_DEFAULT,
    "interpolation.rbf_kernel_cell": _ICON_DEFAULT,
    "interpolation.rbf_kernel_edge": _ICON_DEFAULT,
    "interpolation.rbf_kernel_vertex": _ICON_DEFAULT,
    "interpolation.lsq_dim_unk": _ICON_DEFAULT,
    "interpolation.lsq_dim_c": _ICON_DEFAULT,
    "interpolation.lsq_wgt_exp": _ICON_DEFAULT,
    "interpolation.max_nudging_coefficient": 0.1,
    "interpolation.nudge_efold_width": 2.0,
    "interpolation.nudge_zone_width": 8,
    "prescribed_tendencies.data_path": None,
    "driver.profiling_options": None,
    "driver.enable_statistics_logging": False,
    "driver.enable_output": False,
    "driver.output_backend": _ICON_DEFAULT,
    "driver.output_mode": _ICON_DEFAULT,
    "driver.backend_config": None,
    "nonhydrostatic.l_vert_nested": False,
    "nonhydrostatic.deepatmos_mode": False,
    "nonhydrostatic.iau_init": False,
    "diffusion.loutshs": False,
}
_ICON_PHYSICS_FAMILIES = {
    "muphys": ("microphysics", "icon_muphys"),
    "tmx": ("turbulence", "icon_tmx"),
    "surface": ("surface_bulk_scheme", "icon_fixed_heat_fluxes"),
}


def _construct(cls, values: dict):
    """``cls`` with every one of its fields given explicitly; nothing falls back to a default."""
    import dataclasses
    import enum
    import typing

    fields = {f.name: f for f in dataclasses.fields(cls)}
    if set(values) != set(fields):
        raise ValueError(
            f"{cls.__qualname__}: missing {sorted(set(fields) - set(values))}, "
            f"unknown {sorted(set(values) - set(fields))}"
        )
    hints = typing.get_type_hints(cls)
    kwargs = {}
    for name, value in values.items():
        if value is _ICON_DEFAULT:
            f = fields[name]
            value = f.default if f.default is not dataclasses.MISSING else f.default_factory()
        hint = hints[name]
        if isinstance(value, str) and isinstance(hint, type) and issubclass(hint, enum.Enum):
            if value.upper() not in hint.__members__:
                raise ValueError(
                    f"{cls.__qualname__}.{name}={value!r}: one of "
                    f"{sorted(m.lower() for m in hint.__members__)}"
                )
            value = hint[value.upper()]
        kwargs[name] = value
    return cls(**kwargs)


def build_icon4py_experiment_config(cfg):
    """icon4py's experiment configuration for the legoESM ``ExperimentConfig`` ``cfg``."""
    import dataclasses
    import datetime

    from icon4py.model.atmosphere.diffusion import diffusion
    from icon4py.model.atmosphere.dycore import dycore_states, solve_nonhydro
    from icon4py.model.atmosphere.subgrid_scale_physics.muphys import config as muphys_config
    from icon4py.model.atmosphere.subgrid_scale_physics.surface import config as surface_config
    from icon4py.model.atmosphere.subgrid_scale_physics.tmx import config as tmx_config
    from icon4py.model.atmosphere.tracer_advection import tracer_advection
    from icon4py.model.common import prescribed_tendencies
    from icon4py.model.common.grid import geometry_config, vertical
    from icon4py.model.common.initial_condition.analytical import jablonowski_williamson as jw_ic
    from icon4py.model.common.interpolation import interpolation_factory
    from icon4py.model.common.metrics import metrics_factory
    from icon4py.model.common.states import tracer_states
    from icon4py.model.common.topography.analytical import flat_topography
    from icon4py.model.common.topography.analytical import jablonowski_williamson as jw_topo
    from icon4py.model.driver import config as driver_config

    icon = cfg.icon
    if icon is None:
        raise ValueError("the ICON lane needs ExperimentConfig.icon")
    if cfg.ic != "icon_jablonowski_williamson":
        raise ValueError(f"ic={cfg.ic!r}: the ICON lane has ic='icon_jablonowski_williamson' only")
    for block, (key, value) in _ICON_PHYSICS_FAMILIES.items():
        selected = getattr(cfg, key) == value
        if selected != (getattr(icon, block) is not None):
            raise ValueError(
                f"{key}={getattr(cfg, key)!r} and ExperimentConfig.icon.{block} disagree: the "
                f"{block} block is given exactly when {key}={value!r}"
            )
    dt = float(cfg.dycore.dt)
    if dt <= 0 or dt != round(dt):
        raise ValueError(
            f"dycore.dt={dt!r}: the ICON lane needs a positive whole number of seconds"
        )
    n_steps = cfg.days * 86400.0 / dt
    if n_steps != round(n_steps) or n_steps < 1:
        raise ValueError(f"days={cfg.days!r} is not a whole number of steps of {dt:g} s")
    if not cfg.output.output_dir:
        raise ValueError("output.output_dir is required on the ICON lane (never the cwd)")
    start = datetime.datetime(cfg.start_year, 1, 1, tzinfo=datetime.UTC) + datetime.timedelta(
        days=cfg.start_day
    )
    legoesm_values = {
        "vertical_grid.num_levels": cfg.grid.nlev,
        "driver.dtime": datetime.timedelta(seconds=dt),
        "driver.start_of_simulation": start,
        "driver.start_of_timestepping": start,
        "driver.end_of_simulation": start + datetime.timedelta(seconds=round(n_steps) * dt),
        "driver.output_path": pathlib.Path(cfg.output.output_dir).resolve() / "icon4py",
        "driver.experiment_name": pathlib.Path(cfg.output.output_dir).name,
        "diffusion.iforcing": "aes" if icon.muphys or icon.tmx or icon.surface else "no_forcing",
    }
    for q in ("qv", "qc", "qi", "qr", "qs", "qg"):
        legoesm_values[f"tracer_config.{q}"] = icon.muphys is not None

    def values(section: str, cls) -> dict:
        out = {}
        for f in dataclasses.fields(cls):
            key = f"{section}.{f.name}"
            if key in legoesm_values:
                out[f.name] = legoesm_values[key]
            elif key in _ICON_PINNED:
                out[f.name] = _ICON_PINNED[key]
            elif key in _ICON_FIELD_MAP:
                block, name = _ICON_FIELD_MAP[key]
                out[f.name] = getattr(getattr(icon, block), name)
        return out

    def section(name: str, cls):
        return _construct(cls, values(name, cls))

    metrics = values("metrics", metrics_factory.MetricsConfig)
    metrics["igradp_method"] = int(
        dycore_states.HorizontalPressureDiscretizationType[icon.dycore.igradp_method.upper()].value
    )
    metrics["divdamp_type"] = int(
        dycore_states.DivergenceDampingType[icon.dycore.divdamp_type.upper()].value
    )
    if cfg.topography == "flat":
        topography = _construct(flat_topography.FlatTopographyConfig, {})
    elif cfg.topography == "icon_jablonowski_williamson":
        topography = _construct(
            jw_topo.JablonowskiWilliamsonConfig,
            {"u0": icon.jw.u0_m_s, "eta_0": icon.jw.eta_0},
        )
    else:
        raise ValueError(
            f"topography={cfg.topography!r}: the ICON lane has 'flat' and "
            "'icon_jablonowski_williamson'"
        )
    return _construct(
        driver_config.ExperimentConfig,
        {
            "geometry": section("geometry", geometry_config.GeometryConfig),
            "metrics": _construct(metrics_factory.MetricsConfig, metrics),
            "interpolation": section("interpolation", interpolation_factory.InterpolationConfig),
            "vertical_grid": section("vertical_grid", vertical.VerticalGridConfig),
            "initial_condition": section("initial_condition", jw_ic.JablonowskiWilliamsonConfig),
            "topography": topography,
            "prescribed_tendencies": section(
                "prescribed_tendencies", prescribed_tendencies.PrescribedTendenciesConfig
            ),
            "driver": section("driver", driver_config.DriverConfig),
            "nonhydrostatic": section("nonhydrostatic", solve_nonhydro.NonHydrostaticConfig),
            "diffusion": section("diffusion", diffusion.DiffusionConfig),
            "tracer_config": section("tracer_config", tracer_states.TracerConfig),
            "tracer_advection": section("tracer_advection", tracer_advection.AdvectionConfig),
            "graupel": None,
            "muphys": section("muphys", muphys_config.MuphysConfig) if icon.muphys else None,
            "tmx": section("tmx", tmx_config.TmxConfig) if icon.tmx else None,
            "surface": section("surface", surface_config.SurfaceConfig) if icon.surface else None,
            "shared": _ICON_DEFAULT,
        },
    )
