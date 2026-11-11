"""Terrain ownership on k-corner cells: the 4-corner column meshes give the same bits as before
the k-corner generalisation, and triangles own every source point."""

import hashlib

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from legoesm.grids import topography  # noqa: E402

# sha256 of (z_s, f_land) from grid_terrain_product, recorded on 01064d0808 (4-corner code)
GOLDEN = {
    "duo": "6e7b58b997dd0ce0e628e76b690c916bbb116ab18bf31e7d304729a73d8c7927",
    "latlon": "7d2246e3cdc00686f7c4bf29af429587a994024818ab0b7625f37180b120d181",
}


def _elevation():
    lat = np.arange(-89.75, 90.0, 0.5)
    lon = np.arange(0.25, 360.0, 0.5)
    lo, la = np.meshgrid(np.deg2rad(lon), np.deg2rad(lat))
    return lat, lon, 3000.0 * np.sin(3.0 * la) * np.cos(2.0 * lo) + 500.0 * np.cos(5.0 * lo)


def _mesh(kind):
    if kind == "duo":
        from legoesm.atmosphere.dynamics.gcm.fv3_duo_column import build_duo_column_mesh
        from legoesm.grids.factory import create_fv3_duo_grid

        return build_duo_column_mesh(create_fv3_duo_grid(12, 3).ctx_np, 12, 3)
    from legoesm.atmosphere.dynamics.gcm.latlon_cgrid_column import build_latlon_column_mesh
    from legoesm.grids.factory import create_grid

    return build_latlon_column_mesh(create_grid("latlon", 16))


def _digest(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
    return h.hexdigest()


@pytest.mark.parametrize("kind", sorted(GOLDEN))
def test_four_corner_terrain_product_is_unchanged_bitwise(kind):
    z_s, f_land = topography.grid_terrain_product(_mesh(kind), *_elevation())
    assert _digest(z_s, f_land) == GOLDEN[kind]


def _icon_grid_file():
    import os
    import pathlib

    pytest.importorskip("icon4py.model.common.grid.grid_manager")
    if "ICON4PY_TEST_DATA_PATH" not in os.environ:
        pytest.skip("ICON4PY_TEST_DATA_PATH is not set")
    return (pathlib.Path(os.environ["ICON4PY_TEST_DATA_PATH"]) / "grids" / "icon_grid_0013_R02B04_R"
            / "icon_grid_0013_R02B04_R.nc")


@pytest.fixture(scope="module")
def icon_mesh():
    from legoesm.grids.factory import create_grid

    return create_grid("icon", grid_file=_icon_grid_file())


def test_icon_mesh_is_icon4py_geometry(icon_mesh):
    import netCDF4
    from icon4py.model.common import constants as icon_constants

    m = icon_mesh
    assert m.nCells == 20480 and m.cornerLat.shape == (20480, 3)
    assert m.radius == icon_constants.EARTH_RADIUS
    assert m.omega == icon_constants.EARTH_ANGULAR_VELOCITY
    with netCDF4.Dataset(_icon_grid_file()) as ds:
        np.testing.assert_array_equal(np.asarray(m.latCell), ds["clat"][:])
        np.testing.assert_array_equal(np.asarray(m.cornerLat), ds["clat_vertices"][:])
        # icon4py reads vlon, which differs from the file's clon_vertices by ~2e-16 at lon 0
        np.testing.assert_allclose(
            np.asarray(m.cornerLon), ds["clon_vertices"][:], rtol=0, atol=1e-15
        )
    # icon4py's own bound for a global icosahedral grid: the mean cell area from the area sum
    # agrees with 4 pi R^2 / n_cells to rel 2e-3
    # (test_geometry.py::test_analytical_and_global_reduction_mean_fields_agree)
    assert float(np.sum(m.areaCell)) == pytest.approx(4.0 * np.pi * m.radius**2, rel=2e-3)


def test_icon_triangles_own_their_centres_and_every_source_point(icon_mesh):
    m = icon_mesh
    tris = topography._xyz(m.cornerLat, m.cornerLon).reshape(-1, 3, 3)
    centres = topography._xyz(m.latCell, m.lonCell)
    assert topography._inside_quads(centres, tris, centres).all()
    nb = topography._corner_neighbours(tris)
    # a triangle has 12 corner neighbours away from the 12 pentagon vertices (11 there); a
    # table cut at the quads' 8 would leave source points unclaimed
    assert nb.shape[0] == 12 and (nb[:11] >= 0).all()
    assert not topography._inside_quads(centres[nb[0]], tris, centres).any()
    lat = np.deg2rad(np.arange(-89.75, 90.0, 0.5))
    lon = np.deg2rad(np.arange(0.25, 360.0, 0.5))
    lo, la = np.meshgrid(lon, lat)
    pts = topography._xyz(la.ravel(), lo.ravel())
    owner, n_unclaimed = topography.owner_by_quads(pts, centres, tris, nb)
    assert n_unclaimed == 0
    assert topography._inside_quads(pts, tris[owner], centres[owner]).all()
    _, n_cut = topography.owner_by_quads(pts, centres, tris, nb[:8])
    assert n_cut > 0


def test_icon_terrain_product_conserves_the_land_area(icon_mesh):
    lat, lon, elev = _elevation()
    z_s, f_land = topography.grid_terrain_product(icon_mesh, lat, lon, elev)
    area = np.asarray(icon_mesh.areaCell)
    src = topography._source_cell_areas(lat, lon)
    np.testing.assert_allclose(np.sum(area * f_land) / np.sum(area),
                               np.sum(src * (elev > 0.0)) / np.sum(src), rtol=2e-3)
    assert z_s.shape == (icon_mesh.nCells,) and (z_s[f_land == 0.0] == 0.0).all()


@pytest.mark.parametrize("kwargs", [{}, {"grid_file": "g.nc", "resolution": 4},
                                    {"grid_file": "g.nc", "radius": 1.0}])
def test_create_grid_icon_refuses_a_resolution_or_extra_keywords(kwargs):
    from legoesm.grids.factory import create_grid

    resolution = kwargs.pop("resolution", None)
    with pytest.raises(ValueError, match="icon is a file-backed grid"):
        create_grid("icon", resolution, **kwargs)
