"""The DCMIP-2016 baroclinic wave as the MPAS lane builds it (on pressure levels) and as the ICON
column lane builds it (on heights) agree at the same points: temperature, the eastward wind with
its perturbation, and the water vapour."""

from types import SimpleNamespace

import jax
import numpy as np
import pytest

pytest.importorskip("icon4py.model.common.initial_condition.from_arrays")

from legoesm.atmosphere.dynamics.gcm.icon_column import dcmip2016_initial_state  # noqa: E402
from legoesm.atmosphere.forcing.idealized.baroclinic_wave import (  # noqa: E402
    P0,
    baroclinic_wave_init_mpas,
    find_z_for_pressure,
)
from legoesm.thermo import initial_water_vapour  # noqa: E402

RH = 0.7


@pytest.fixture
def points():
    if not jax.config.jax_enable_x64:
        pytest.skip("round-off agreement needs JAX_ENABLE_X64=1")
    lat = np.deg2rad(np.array([-70.0, -35.0, 0.0, 20.0, 40.0, 41.0, 60.0, 85.0]))
    # 20E/40N is the perturbation centre: two points inside it
    lon = np.deg2rad(np.array([0.0, 100.0, 200.0, 20.0, 20.0, 22.0, 300.0, 10.0]))
    sigma = np.array([0.01, 0.1, 0.25, 0.5, 0.85, 0.99])
    # cells and edges at the same points, edge normals eastward: u_normal is the eastward wind
    mesh = SimpleNamespace(nCells=lat.size, nEdges=lat.size, latCell=lat, lonCell=lon,
                           latEdge=lat, lonEdge=lon, angleEdge=np.zeros_like(lat))
    return mesh, SimpleNamespace(sigma_full=sigma, n_levels=sigma.size)


def test_the_two_lanes_evaluate_the_same_state(points):
    mesh, sigma = points
    mpas = baroclinic_wave_init_mpas(mesh, sigma, perturbed=True)
    p_mpas = np.broadcast_to(P0 * sigma.sigma_full[None, :], (mesh.nCells, sigma.n_levels))
    qv_mpas = initial_water_vapour(mpas.T.data, p_mpas, sigma.sigma_full, RH)

    # the ICON lane on the heights where the MPAS lane's pressure levels sit
    lat = np.broadcast_to(mesh.latCell[:, None], p_mpas.shape)
    z = np.asarray(find_z_for_pressure(p_mpas, lat))
    icon = dcmip2016_initial_state(mesh, RH)(z)

    np.testing.assert_allclose(icon.surface_pressure, P0, rtol=1e-15)
    np.testing.assert_allclose(icon.temperature, np.asarray(mpas.T.data), rtol=1e-13)
    np.testing.assert_allclose(icon.u, np.asarray(mpas.u.data), rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(icon.v, 0.0)
    np.testing.assert_allclose(icon.q["qv"], np.asarray(qv_mpas), rtol=1e-12)
    assert all(not np.any(icon.q[sp]) for sp in ("qc", "qi", "qr", "qs", "qg"))
    # the perturbation is in both (not a vacuous zero-vs-zero wind)
    assert np.ptp(np.asarray(mpas.u.data)[3:6, -1]) > 0.1
