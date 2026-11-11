"""legoESM slab land on the ICON lane: the patch's cells, their initial land state, and the land
step that turns ICON's lowest model level and muphys's surface precipitation into ICON's
surface fluxes (``icon.land``, :class:`IconLandConfig`)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from legoesm.atmosphere.dynamics.gcm.icon4py_jax import land_surface_fluxes
from legoesm.core.coupling_fields import AtmToSurface
from legoesm.core.field import Field
from legoesm.land.slab_land import step_land_with_diagnostics
from legoesm.land.state import LandState


def patch_mask(mesh, land) -> np.ndarray:
    """True on the cells whose centre lies in the patch: ``min <= lat < max`` and
    ``min <= lon < max``, longitude in [0, 360)."""
    lat = np.rad2deg(np.asarray(mesh.latCell, dtype=np.float64))
    lon = np.mod(np.rad2deg(np.asarray(mesh.lonCell, dtype=np.float64)), 360.0)
    return ((lat >= land.patch_lat_min_deg) & (lat < land.patch_lat_max_deg)
            & (lon >= land.patch_lon_min_deg) & (lon < land.patch_lon_max_deg))


def initial_land_state(land, T_lowest):  # noqa: N803
    """The soil at the lowest-level air temperature, the bucket at ``bucket_init_frac`` of its
    capacity, no snow; and the surface radiation first sees: the soil temperature and the
    slab's albedo and emissivity."""
    t_soil = jnp.asarray(T_lowest)

    def field(name, data, units):
        return Field(data, name=name, dims=("column",), units=units)

    zero = jnp.zeros_like(t_soil)
    state = LandState(
        T_soil=field("T_soil", t_soil, "K"),
        W_bucket=field("W_bucket", zero + land.bucket_init_frac * land.slab.W_max, "kg/m2"),
        snow_depth=field("snow_depth", zero, "kg/m2"),
        snow_age=field("snow_age", zero, "s"),
        # an array from the start, so the step's input and output pytrees agree
        runoff=zero,
    )
    surface = {"T_sfc": t_soil, "albedo": zero + land.slab.albedo_land,
               "emissivity": zero + land.slab.emissivity_land}
    return state, surface


def lowest_level(col) -> dict:
    """ICON's lowest full level (the last of its top-to-bottom levels) as plain arrays."""
    def last(f):
        return f.ndarray[:, -1]

    return {
        "T": last(col.temperature),
        "qv": last(col.q["qv"]),
        "u": last(col.u),
        "v": last(col.v),
        "p": last(col.pressure),
        "p_surface": last(col.pressure_ifc),
        "rho": last(col.rho),
        # z_mc is the mean of the two interface heights, so the lowest full level sits dz/2
        # above the ground
        "z": 0.5 * last(col.dz),
    }


def land_step(land, lat, dt: float):
    """``step(col, muphys_diagnostics, land_state, radiation) -> (SurfaceFluxState arrays,
    land_state, land_surface)`` on every cell; the caller keeps ICON's own fluxes off the
    patch. ``radiation`` is None or the surface's ``sw_down``/``lw_down``; ``land_surface`` is
    what radiation sees of the land next step. ``lat`` [rad]."""
    lat = jnp.asarray(lat)

    def step(col, muphys, state, radiation):
        low = lowest_level(col)
        frozen = sum(muphys[k].ndarray[:, -1] for k in ("ps", "pi", "pg"))
        zero = jnp.zeros_like(low["T"])
        forcing = AtmToSurface(
            # without radiation a run with land cells is refused
            sw_down=zero if radiation is None else radiation["sw_down"],
            lw_down=zero if radiation is None else radiation["lw_down"],
            precip_total=muphys["pr"].ndarray[:, -1] + frozen,
            precip_snow=frozen,
            T_lowest=low["T"],
            q_lowest=low["qv"],
            u_lowest=low["u"],
            v_lowest=low["v"],
            p_lowest=low["p"],
            p_surface=low["p_surface"],
            rho_lowest=low["rho"],
            cos_zenith=zero,
            co2_ppmv=zero,
            has_radiation=zero if radiation is None else zero + 1.0,
            has_precipitation=zero + 1.0,
            z_lowest=low["z"],
        )
        new, response, _, _ = step_land_with_diagnostics(
            state, forcing, land.slab, land.wind_speed_min_m_s, dt, lat=lat)
        fluxes = land_surface_fluxes(
            shflx=response.shflx,
            lhflx=response.lhflx,
            surface_mass_flux=response.surface_mass_flux,
            tau_x=response.tau_x,
            tau_y=response.tau_y,
            T_lowest=low["T"],
        )
        surface = {"T_sfc": response.T_rad, "albedo": response.albedo,
                   "emissivity": response.emissivity}
        return fluxes, new, surface

    return step
