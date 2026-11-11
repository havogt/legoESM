"""Radiation's external forcing (ozone, aerosol) on columns given by their own pressures, for
any vertical coordinate: the sigma lanes pass sigma's pressures, the ICON lane its own."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp


class RadiationForcing(NamedTuple):
    o3_vmr: Any
    #: mean ozone VMR above the model top, for RRTMGP's overhead layer
    o3_top_vmr: Any
    aerosol_od: Any
    aerosol_ccn_aod: Any
    aerosol_lw_od: Any


def radiation_forcing_on_columns(day, p_full_col, p_half_col, lat_col, lon_col, temp_col, *,
                                 ozone_ext_config, ozone_source, aerosol_config,
                                 aerosol_active, aerosol_lw_active, dtype) -> RadiationForcing:
    """Ozone and aerosol on ``(ncol, nlev)`` full / ``(ncol, nlev + 1)`` half-level pressures.

    ``ozone_ext_config`` None = the inline ``ozone_source`` profile; ``lon_col`` [rad] is read
    only with aerosol active; ``temp_col`` None = climatological tropopause."""
    from legoesm.forcing.external import get_ozone_at_time
    from legoesm.forcing.surface_utils import (
        aerosol_layer_fields,
        stratospheric_layer_weight,
    )

    ncol, nlev = p_full_col.shape
    # Ozone is evaluated on the model levels PLUS 16 midpoints of equal
    # pressure sub-layers above the model top (0 < p < p_top); their mean
    # is the overhead column-mean VMR for the RRTMGP overhead layer
    # (RRTMGPConfig.overhead_layer; unused when that is off).  The inline sources other than
    # "mls" are pointwise in p, so the model-level values are unchanged;
    # "mls" (gSAM layer means) is handled from the interfaces below.
    _p_o3 = jnp.concatenate([
        p_full_col,
        p_half_col[:, :1] * ((jnp.arange(16) + 0.5) / 16.0),
    ], axis=1)

    w_strat = None
    if aerosol_active or aerosol_lw_active:
        w_strat = stratospheric_layer_weight(
            p_half_col, lat_col, p_full_col, temp_col)

    if ozone_ext_config is not None:
        o3_vmr = jnp.asarray(get_ozone_at_time(
            ozone_ext_config, day,
            lat_grid=lat_col, p_grid=_p_o3,
        ))
    else:
        # External ozone forcing inactive — build the inline profile
        # HERE so the radiation JIT signature stays stable (it always
        # sees a real array).  Previously ``o3_vmr`` was initialised to
        # zeros and threaded into the solver, which then clipped it to
        # 1e-10 and effectively disabled stratospheric ozone heating
        # (audit 2026-05-12 HIGH #3).
        #
        # Honor ``ozone_source`` (2026-07-21 audit): this precomputed
        # array reaches the compiled radiation as ``o3_vmr_override``,
        # which takes precedence over the backend's own
        # ``_compute_ozone_vmr`` — so building only the "standard"
        # profile here silently ignored ``--ozone-source analytical``/
        # ``none`` on the cd-grid/latlon pipeline while MPAS/spectral
        # honored them (grid-dependent physics from the same config).
        # Reuse the backend's dispatcher for exact parity.
        from legoesm.atmosphere.physics.radiation.config import OzoneProfileConfig
        from legoesm.atmosphere.physics.radiation.integration import (
            compute_ozone_vmr as _compute_ozone_vmr,
        )
        if ozone_source == "mls":
            # gSAM layer means are not pointwise in p: evaluate them on the
            # model interfaces; the overhead layer takes gSAM's own extra
            # layer above the model top (p_top to min(1e-4 hPa, p_top/8)).
            # With p_top = 0 there is no overhead layer: it keeps the top
            # model layer's value, and the extra layer's 0/0 is never formed
            # (p_top replaced by 1 Pa only inside that branch).
            from legoesm.atmosphere.physics.radiation.ozone_mls import mls_ozone_layer_vmr
            _lay = _compute_ozone_vmr(
                p_full_col, lat_col, OzoneProfileConfig(source="mls"),
                p_half=p_half_col,
            )
            _ptop = p_half_col[:, 0]
            _ph_safe = p_half_col.at[:, 0].set(jnp.where(_ptop > 0.0, _ptop, 1.0))
            _above = jnp.where(_ptop > 0.0, mls_ozone_layer_vmr(_ph_safe)[1],
                               _lay[:, 0])
            _o3_inline = jnp.concatenate(
                [_lay, jnp.broadcast_to(_above[:, None], (ncol, _p_o3.shape[1] - nlev))],
                axis=1)
        else:
            _o3_inline = _compute_ozone_vmr(
                _p_o3, lat_col,
                OzoneProfileConfig(source=ozone_source),
            )
        if _o3_inline is None:
            # source="standard": the backend contract is o3_vmr=None →
            # its built-in US Std Atm 1976 fit; materialize the same
            # profile for the stable-signature override path.
            from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import standard_o3_profile
            _o3_inline = standard_o3_profile(_p_o3)
        o3_vmr = jnp.asarray(_o3_inline).astype(dtype)
    o3_top_vmr = jnp.mean(o3_vmr[:, nlev:], axis=1)
    o3_vmr = o3_vmr[:, :nlev]

    aerosol_od, ccn_aod, aerosol_lw_od = aerosol_layer_fields(
        aerosol_config, day, lat_col, p_half_col, w_strat,
        active=aerosol_active, lw_active=aerosol_lw_active,
        dtype=dtype, lon_col=lon_col)
    return RadiationForcing(o3_vmr, o3_top_vmr, aerosol_od, ccn_aod, aerosol_lw_od)
