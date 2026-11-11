"""Settings of the ICON lane (icon4py's JAX model), in legoESM's naming.

Every field is required: a run states each ICON setting explicitly (no defaults). The
adapter maps them one to one onto icon4py's experiment configuration; the enum-valued
fields hold the lower-case name of icon4py's enum member.
"""

from __future__ import annotations

from typing import Any, NamedTuple


class IconDycoreConfig(NamedTuple):
    """Non-hydrostatic solver, metrics and time-step settings."""

    exner_expol: float
    vwind_offctr: float
    thslp_zdiffu: float
    thhgtd_zdiffu_m: float
    rayleigh_coeff: float
    divdamp_trans_start_m: float
    divdamp_trans_end_m: float
    divergence_averaging_central_cell_weight: float
    lsq_high_ord: int
    apply_extra_second_order_divdamp: bool
    diffuse_before_time_loop: bool
    vertical_cfl_threshold: float
    ndyn_substeps: int
    itime_scheme: str
    iadv_rhotheta: str
    igradp_method: str
    rayleigh_type: str
    divdamp_order: str
    divdamp_type: str
    extra_diffu: bool
    rhotheta_offctr: float
    veladv_offctr: float
    fourth_order_divdamp_factor: float
    fourth_order_divdamp_factor2: float
    fourth_order_divdamp_factor3: float
    fourth_order_divdamp_factor4: float
    fourth_order_divdamp_z_m: float
    fourth_order_divdamp_z2_m: float
    fourth_order_divdamp_z3_m: float
    fourth_order_divdamp_z4_m: float


class IconVerticalConfig(NamedTuple):
    """Analytic vertical grid (vct_a/vct_b); levels count is GridConfig.nlev."""

    maximal_layer_thickness_m: float
    top_height_limit_for_maximal_layer_thickness_m: float
    lowest_layer_thickness_m: float
    model_top_height_m: float
    flat_height_m: float
    stretch_factor: float
    rayleigh_damping_height_m: float
    htop_moist_proc_m: float
    sleve_decay_scale_1_m: float
    sleve_decay_scale_2_m: float
    sleve_decay_exponent: float


class IconJWConfig(NamedTuple):
    """Jablonowski-Williamson initial condition (and its analytic terrain's u0/eta_0)."""

    p_sfc_Pa: float
    baroclinic_amplitude: float
    u0_m_s: float
    temp0_K: float
    eta_0: float
    eta_t: float
    gamma_K_m: float
    dtemp_K: float
    lon_perturbation_center_rad: float
    lat_perturbation_center_rad: float
    rh_at_1000hpa: float
    qv_max_kg_kg: float
    moisture_init_iterations: int
    global_moisture_content_kg_m2: float
    normalize_global_moisture: bool


class IconDiffusionConfig(NamedTuple):
    """Horizontal diffusion."""

    diffusion_type: str
    apply_to_vertical_wind: bool
    apply_to_horizontal_wind: bool
    apply_to_temperature: bool
    apply_smag_diff_to_vertical_wind: bool
    compute_3d_smag_coeff: bool
    type_vn_diffu: str
    type_t_diffu: str
    hdiff_efdt_ratio: float
    hdiff_w_efdt_ratio: float
    smagorinski_scaling_factor: float
    smagorinski_scaling_factor2: float
    smagorinski_scaling_factor3: float
    smagorinski_scaling_factor4: float
    smagorinski_scaling_height_m: float
    smagorinski_scaling_height2_m: float
    smagorinski_scaling_height3_m: float
    smagorinski_scaling_height4_m: float
    apply_zdiffusion_t: bool
    temperature_boundary_diffusion_denominator: float
    velocity_boundary_diffusion_denominator: float
    shear_type: str
    a_hshr: float


class IconTracerAdvectionConfig(NamedTuple):
    """Tracer transport."""

    horizontal_advection_type: str
    horizontal_advection_limiter: str
    vertical_advection_type: str
    vertical_advection_limiter: str


class IconMuphysConfig(NamedTuple):
    """muphys microphysics (ExperimentConfig.microphysics == 'icon_muphys')."""

    qnc_per_m3: float


class IconTmxConfig(NamedTuple):
    """tmx turbulent mixing (ExperimentConfig.turbulence == 'icon_tmx')."""

    solver_type: str
    energy_type: str
    dissipation_factor: float
    use_louis: bool
    use_louis_land: bool
    use_louis_ice: bool
    louis_constant_b: float
    use_km_const: bool
    km_const_m2_s: float
    use_scale_turb_energy_flux: bool
    scale_turb_energy_flux: float
    smag_constant: float
    turb_prandtl: float
    km_min_m2_s: float
    max_turb_scale_m: float


class IconSurfaceConfig(NamedTuple):
    """Prescribed surface fluxes (surface_bulk_scheme == 'icon_fixed_heat_fluxes')."""

    surface_flux_type: str
    kinematic_sensible_heat_flux_K_m_s: float
    kinematic_latent_heat_flux_m_s: float
    sea_surface_temperature_K: float


class IconLandConfig(NamedTuple):
    """legoESM slab land on the cells of a lat/lon box, ICON's own surface elsewhere.

    A cell is land when its centre has ``patch_lat_min_deg <= lat < patch_lat_max_deg`` and
    ``patch_lon_min_deg <= lon < patch_lon_max_deg``, longitude in [0, 360). ``slab`` is the
    land package's ``LandConfig``; this package does not import it.
    """

    patch_lat_min_deg: float
    patch_lat_max_deg: float
    patch_lon_min_deg: float
    patch_lon_max_deg: float
    #: initial bucket water as a fraction of the bucket capacity ``slab.W_max``
    bucket_init_frac: float
    #: wind-speed floor of the land's bulk fluxes
    wind_speed_min_m_s: float
    slab: Any


class IconRadiationConfig(NamedTuple):
    """legoESM's RRTMGP on ICON's columns (``radiation == 'rrtmgp'``), clear sky: the
    settings legoESM's RRTMGP builder reads, under their ExperimentConfig names. Ozone is an
    inline profile (``ozone_source``); there is no aerosol."""

    diurnal_cycle: bool
    S_0: float
    orbital_insolation: bool
    co2_ppmv: float
    ch4_ppbv: float
    n2o_ppbv: float
    ozone_source: str
    #: the ocean surface's albedo and emissivity; land takes its own from the land model
    albedo_ocean: float
    sfc_emissivity: float
    rrtmgp_use_scan: bool | None
    rrtmgp_gpoint_batch_size: int
    rrtmgp_gpoint_checkpoint: bool
    rrtmgp_column_chunk_size: int
    rrtmgp_overhead_layer: bool


class IconConfig(NamedTuple):
    """The ICON lane's settings; the physics blocks are None when that family is off, and
    ``land`` is None on a surface without legoESM land."""

    dycore: IconDycoreConfig
    vertical: IconVerticalConfig
    jw: IconJWConfig
    diffusion: IconDiffusionConfig
    tracer_advection: IconTracerAdvectionConfig
    muphys: IconMuphysConfig | None
    tmx: IconTmxConfig | None
    surface: IconSurfaceConfig | None
    land: IconLandConfig | None
    radiation: IconRadiationConfig | None


__param_spec__ = {
    "IconDycoreConfig": {
        "scheme_key": "icon_dycore",
        "excluded": {
            "exner_expol": "exposure deferred (2026-11-10)",
            "vwind_offctr": "exposure deferred (2026-11-10)",
            "thslp_zdiffu": "exposure deferred (2026-11-10)",
            "thhgtd_zdiffu_m": "exposure deferred (2026-11-10)",
            "rayleigh_coeff": "exposure deferred (2026-11-10)",
            "divdamp_trans_start_m": "exposure deferred (2026-11-10)",
            "divdamp_trans_end_m": "exposure deferred (2026-11-10)",
            "divergence_averaging_central_cell_weight": "exposure deferred (2026-11-10)",
            "vertical_cfl_threshold": "exposure deferred (2026-11-10)",
            "rhotheta_offctr": "exposure deferred (2026-11-10)",
            "veladv_offctr": "exposure deferred (2026-11-10)",
            "fourth_order_divdamp_factor": "exposure deferred (2026-11-10)",
            "fourth_order_divdamp_factor2": "exposure deferred (2026-11-10)",
            "fourth_order_divdamp_factor3": "exposure deferred (2026-11-10)",
            "fourth_order_divdamp_factor4": "exposure deferred (2026-11-10)",
            "fourth_order_divdamp_z_m": "exposure deferred (2026-11-10)",
            "fourth_order_divdamp_z2_m": "exposure deferred (2026-11-10)",
            "fourth_order_divdamp_z3_m": "exposure deferred (2026-11-10)",
            "fourth_order_divdamp_z4_m": "exposure deferred (2026-11-10)",
        },
        "params": {},
    },
    "IconVerticalConfig": {
        "scheme_key": "icon_vertical",
        "excluded": {
            "maximal_layer_thickness_m": "exposure deferred (2026-11-10)",
            "top_height_limit_for_maximal_layer_thickness_m": "exposure deferred (2026-11-10)",
            "lowest_layer_thickness_m": "exposure deferred (2026-11-10)",
            "model_top_height_m": "exposure deferred (2026-11-10)",
            "flat_height_m": "exposure deferred (2026-11-10)",
            "stretch_factor": "exposure deferred (2026-11-10)",
            "rayleigh_damping_height_m": "exposure deferred (2026-11-10)",
            "htop_moist_proc_m": "exposure deferred (2026-11-10)",
            "sleve_decay_scale_1_m": "exposure deferred (2026-11-10)",
            "sleve_decay_scale_2_m": "exposure deferred (2026-11-10)",
            "sleve_decay_exponent": "exposure deferred (2026-11-10)",
        },
        "params": {},
    },
    "IconJWConfig": {
        "scheme_key": "icon_jw",
        "excluded": {
            "p_sfc_Pa": "exposure deferred (2026-11-10)",
            "baroclinic_amplitude": "exposure deferred (2026-11-10)",
            "u0_m_s": "exposure deferred (2026-11-10)",
            "temp0_K": "exposure deferred (2026-11-10)",
            "eta_0": "exposure deferred (2026-11-10)",
            "eta_t": "exposure deferred (2026-11-10)",
            "gamma_K_m": "exposure deferred (2026-11-10)",
            "dtemp_K": "exposure deferred (2026-11-10)",
            "lon_perturbation_center_rad": "exposure deferred (2026-11-10)",
            "lat_perturbation_center_rad": "exposure deferred (2026-11-10)",
            "rh_at_1000hpa": "exposure deferred (2026-11-10)",
            "qv_max_kg_kg": "exposure deferred (2026-11-10)",
            "global_moisture_content_kg_m2": "exposure deferred (2026-11-10)",
        },
        "params": {},
    },
    "IconDiffusionConfig": {
        "scheme_key": "icon_diffusion",
        "excluded": {
            "hdiff_efdt_ratio": "exposure deferred (2026-11-10)",
            "hdiff_w_efdt_ratio": "exposure deferred (2026-11-10)",
            "smagorinski_scaling_factor": "exposure deferred (2026-11-10)",
            "smagorinski_scaling_factor2": "exposure deferred (2026-11-10)",
            "smagorinski_scaling_factor3": "exposure deferred (2026-11-10)",
            "smagorinski_scaling_factor4": "exposure deferred (2026-11-10)",
            "smagorinski_scaling_height_m": "exposure deferred (2026-11-10)",
            "smagorinski_scaling_height2_m": "exposure deferred (2026-11-10)",
            "smagorinski_scaling_height3_m": "exposure deferred (2026-11-10)",
            "smagorinski_scaling_height4_m": "exposure deferred (2026-11-10)",
            "temperature_boundary_diffusion_denominator": "exposure deferred (2026-11-10)",
            "velocity_boundary_diffusion_denominator": "exposure deferred (2026-11-10)",
            "a_hshr": "exposure deferred (2026-11-10)",
        },
        "params": {},
    },
    "IconMuphysConfig": {
        "scheme_key": "icon_muphys",
        "excluded": {
            "qnc_per_m3": "exposure deferred (2026-11-10)",
        },
        "params": {},
    },
    "IconTmxConfig": {
        "scheme_key": "icon_tmx",
        "excluded": {
            "dissipation_factor": "exposure deferred (2026-11-10)",
            "louis_constant_b": "exposure deferred (2026-11-10)",
            "km_const_m2_s": "exposure deferred (2026-11-10)",
            "scale_turb_energy_flux": "exposure deferred (2026-11-10)",
            "smag_constant": "exposure deferred (2026-11-10)",
            "turb_prandtl": "exposure deferred (2026-11-10)",
            "km_min_m2_s": "exposure deferred (2026-11-10)",
            "max_turb_scale_m": "exposure deferred (2026-11-10)",
        },
        "params": {},
    },
    "IconSurfaceConfig": {
        "scheme_key": "icon_surface",
        "excluded": {
            "kinematic_sensible_heat_flux_K_m_s": "exposure deferred (2026-11-10)",
            "kinematic_latent_heat_flux_m_s": "exposure deferred (2026-11-10)",
            "sea_surface_temperature_K": "exposure deferred (2026-11-10)",
        },
        "params": {},
    },
    "IconLandConfig": {
        "scheme_key": "icon_land",
        "excluded": {
            "patch_lat_min_deg": "case geometry, not a parameter",
            "patch_lat_max_deg": "case geometry, not a parameter",
            "patch_lon_min_deg": "case geometry, not a parameter",
            "patch_lon_max_deg": "case geometry, not a parameter",
            "bucket_init_frac": "initial condition, not a parameter",
            "wind_speed_min_m_s": "numerics floor",
        },
        "params": {},
    },
    "IconRadiationConfig": {
        "scheme_key": "icon_radiation",
        "excluded": {
            "S_0": "radiative forcing, exposure deferred",
            "co2_ppmv": "radiative forcing, exposure deferred",
            "ch4_ppbv": "radiative forcing, exposure deferred",
            "n2o_ppbv": "radiative forcing, exposure deferred",
            "albedo_ocean": "surface property, exposure deferred",
            "sfc_emissivity": "surface property, exposure deferred",
        },
        "params": {},
    },
}
