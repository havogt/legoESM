"""Radiation forcing on columns given by their own pressures: the sigma lanes' precompute,
now a sigma wrapper around the pressure-level core, gives the same bits as before the split."""

import hashlib

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from tests.unit.test_aerosol_band_ccn import (  # noqa: E402
    _driver_forcing,
    _kinne_file,
    _std_temp,
    _volc_file,
)

# sha256 of every output of the precompute, recorded on a5480ee18c (before the split)
GOLDEN = {
    "standard": "826c66d8e0236a8b98e7f91696139290869c768653588e3c8f20b3f3267b447d",
    "analytical": "8ae86ee7caf2b4dd56c4db92f9d50a5f679aa4eefb3bab4adfbb768655cbfd2f",
    "mls": "5257e1a1b8f615072d5dea9b562c626468f7f4c13b05e4ff8c8b80d3f81482d3",
    "none": "7a9ae1b7c9f119df9f9b1511be1ed7ced3842fa6947303b8989d4b5bd4289d9a",
    "aerosol": "a4990619366014c0bac45244377e95bc05d4f025ea8a88ef0a8751ba4b8f381a",
    "aerosol_state_tropopause": "11930034b6d3e46c9a173eece039566c90a1deb224ff236a1d0783bf8c6f896d",
}


def _digest(fake, o3_and_aer):
    h = hashlib.sha256()
    for a in (*o3_and_aer, fake._o3_top_vmr, getattr(fake, "_aerosol_ccn_aod", None),
              fake._aerosol_lw_od):
        h.update(b"None" if a is None else np.ascontiguousarray(a, dtype=np.float64).tobytes())
    return h.hexdigest()


def _run(case, tmp_path):
    from legoesm.driver.config import ExperimentConfig
    from legoesm.driver.model_driver import ModelDriver
    from legoesm.forcing import external

    # the file readers cache by path, and other tests leave files behind
    for f in vars(external).values():
        if hasattr(f, "cache_clear"):
            f.cache_clear()

    kw, temp_fn = {}, None
    if case.startswith("aerosol"):
        kw = dict(aerosol_forcing="external",
                  aerosol_file=_kinne_file(tmp_path / "t.nc", [0.05, 0.20, 0.40], lon=True),
                  volcanic_aerosol_file=_volc_file(tmp_path / "v.nc"))
        if case == "aerosol_state_tropopause":
            def temp_fn(p):
                return _std_temp(p, z_tp_m=19.0e3, t0=300.0)[0]
    else:
        kw = dict(ozone_source=case)
    cfg = ExperimentConfig(radiation="rrtmgp", **kw)
    captured = {}
    real = ModelDriver._precompute_external_forcing

    def spy(fake, *args, **kwargs):
        captured["out"] = real(fake, *args, **kwargs)
        captured["fake"] = fake
        return captured["out"]

    ModelDriver._precompute_external_forcing = spy
    try:
        _driver_forcing(cfg, temp_fn=temp_fn)
    finally:
        ModelDriver._precompute_external_forcing = real
    o3, aer, _ = captured["out"]
    return captured["fake"], (o3, aer)


@pytest.mark.parametrize("case", sorted(GOLDEN))
def test_the_sigma_precompute_is_unchanged_bitwise(case, tmp_path):
    fake, out = _run(case, tmp_path)
    assert _digest(fake, out) == GOLDEN[case]


def test_the_core_on_pressures_is_the_sigma_precompute(tmp_path):
    from legoesm.driver.radiation_forcing import radiation_forcing_on_columns
    from legoesm.grids.vertical import make_cam6_l32_levels

    fake, (o3, _) = _run("analytical", tmp_path)
    sigma = make_cam6_l32_levels()
    p_s = jnp.full((7,), 1.0e5)
    got = radiation_forcing_on_columns(
        15.5, sigma.pressure_at_full(p_s).reshape(7, -1),
        sigma.pressure_at_half(p_s).reshape(7, -1),
        jnp.radians(jnp.linspace(-90.0, 90.0, 7)), None, None,
        ozone_ext_config=None, ozone_source="analytical", aerosol_config=fake._aerosol_config,
        aerosol_active=False, aerosol_lw_active=False, dtype=p_s.dtype)
    np.testing.assert_array_equal(np.asarray(got.o3_vmr), np.asarray(o3))
