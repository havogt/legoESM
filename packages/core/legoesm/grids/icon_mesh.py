"""The ICON grid as a :class:`ColumnMesh`: icon4py's own grid manager reads the grid file, and the
mesh is a view of its cell geometry (no legoESM re-implementation of ICON geometry)."""

from __future__ import annotations

import pathlib

import jax.numpy as jnp
import numpy as np
from legoesm.grids.column_mesh import ColumnMesh


def build_icon_column_mesh(grid_file: str | pathlib.Path) -> ColumnMesh:
    """All cells of an ICON grid file in its global order: centres and areas [m^2] from
    icon4py's grid manager, triangle corners ``(nCells, 3)`` from its cell-to-vertex table
    (ring order), icon4py's radius (``grid_params``) and rotation rate."""
    try:
        from icon4py.model.common import constants as icon_constants
        from icon4py.model.common import dimension as dims
        from icon4py.model.common.decomposition import definitions as decomposition
        from icon4py.model.common.grid import grid_manager as gm
        from icon4py.model.common.grid import gridfile
        from icon4py.model.common.grid import vertical as v_grid
    except ImportError as err:
        raise ImportError(
            "create_grid('icon') needs icon4py: pip install legoesm[icon]"
        ) from err

    # the grid manager insists on a vertical config; the horizontal mesh does not depend on it
    manager = gm.GridManager(
        grid_file=pathlib.Path(grid_file),
        config=v_grid.VerticalGridConfig(num_levels=1),
        offset_transformation=gridfile.ToZeroBasedIndexTransformation(),
    )
    manager(
        allocator=None,
        keep_skip_values=True,
        process_props=decomposition.SingleNodeProcessProperties(),
    )
    radius = manager.grid.grid_params.radius
    if radius is None:
        raise ValueError(f"create_grid('icon'): {grid_file} is not a spherical (icosahedral) grid")
    cell = manager.coordinates[dims.CellDim]
    vertex = manager.coordinates[dims.VertexDim]
    c2v = manager.grid.get_connectivity("C2V").asnumpy()
    lat = jnp.asarray(cell["lat"].asnumpy())
    lon = jnp.asarray(cell["lon"].asnumpy())
    area = manager.geometry_fields[gridfile.GeometryName.CELL_AREA.value].asnumpy()
    return ColumnMesh(
        latCell=lat,
        lonCell=lon,
        areaCell=jnp.asarray(area),
        nCells=int(lat.shape[0]),
        grid_lat=lat,
        grid_lon=lon,
        grid_shape_2d=(int(lat.shape[0]),),
        lat=lat,
        lon=lon,
        cornerLat=jnp.asarray(np.asarray(vertex["lat"].asnumpy())[c2v]),
        cornerLon=jnp.asarray(np.asarray(vertex["lon"].asnumpy())[c2v]),
        radius=float(radius),
        omega=float(icon_constants.EARTH_ANGULAR_VELOCITY),
    )
