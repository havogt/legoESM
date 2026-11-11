"""legoESM's RRTMGP on ICON's columns (``icon.radiation``, :class:`IconRadiationConfig`): ICON's
own pressures, temperature and water, and the heating returned the way ICON's physics heats,
at constant volume."""

from __future__ import annotations

import types

import jax.numpy as jnp
from legoesm.driver.physics_pipeline import build_rrtmgp_radiation_fn
from legoesm.driver.radiation_forcing import radiation_forcing_on_columns

_CONDENSATE = ("qc", "qi", "qr", "qs", "qg")


def _builder_config(rad):
    """The settings legoESM's RRTMGP builder reads, every one from the block; clear sky."""
    return types.SimpleNamespace(**rad._asdict(), radiation="rrtmgp", cloud_scheme="none")


def cv_air(col):
    """ICON's moist air heat capacity at constant volume per area [J/K/m^2], as tmx takes it."""
    from icon4py.model.common.physics.thermodynamics import compute_energy

    q = col.q
    return compute_energy._compute_moist_air_isochoric_heat_capacity_per_area.definition(
        q["qv"], q["qc"], q["qi"], q["qr"], q["qs"], q["qg"], col.air_mass
    ).ndarray


def radiation_step(rad, lat, lon):
    """``step(col, surface, day_of_year, seconds_of_day) -> (dT/dt, out)``: the temperature
    tendency [K/s] at constant volume and RRTMGP's output (interface fluxes, index 0 at the
    model top). ``surface`` holds ``T_sfc`` [K], ``albedo`` and ``emissivity`` per cell;
    ``lat``/``lon`` [rad]."""
    radiation_fn, _ = build_rrtmgp_radiation_fn(_builder_config(rad))
    lat = jnp.asarray(lat)
    lon = jnp.asarray(lon)

    def step(col, surface, day_of_year, seconds_of_day):
        temperature = col.temperature.ndarray
        p_full = col.pressure.ndarray
        p_half = col.pressure_ifc.ndarray
        qv = col.q["qv"].ndarray
        condensate = sum(col.q[k].ndarray for k in _CONDENSATE)
        forcing = radiation_forcing_on_columns(
            day_of_year, p_full, p_half, lat, None, None,
            ozone_ext_config=None, ozone_source=rad.ozone_source, aerosol_config=None,
            aerosol_active=False, aerosol_lw_active=False, dtype=temperature.dtype)
        out = radiation_fn(
            temperature, p_full, p_half,
            # RRTMGP's wrapper takes the vapour-to-dry-air ratio (and turns it into r / (1 + r))
            qv / (1.0 - qv - condensate),
            surface["T_sfc"], lat, lon, day_of_year, seconds_of_day,
            surface["albedo"], surface["emissivity"], forcing.o3_vmr, forcing.aerosol_od,
            jnp.zeros((0,)), s_0=rad.S_0,
            o3_top_vmr_col=forcing.o3_top_vmr if rad.rrtmgp_overhead_layer else None,
        )
        net_down = (out.sw_flux_down - out.sw_flux_up) + (out.lw_flux_down - out.lw_flux_up)
        # each layer gains the net flux into its top minus what leaves through its bottom
        heating = net_down[:, :-1] - net_down[:, 1:]
        return heating / cv_air(col), out

    return step
