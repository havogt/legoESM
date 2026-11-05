"""assemble() puts every rank's owned points back in global order."""

import numpy as np
import pytest

from legoesm.atmosphere.dynamics.gcm import icon4py_jax_spmd


def _write_rank(path, rank, gid_cell, gid_edge, steps):
    fields = {}
    for k in range(steps):
        fields[f"vn_{k}"] = 10.0 * k + gid_edge.astype(np.float64)
        for name in ("w", "rho", "exner", "theta_v"):
            fields[f"{name}_{k}"] = 10.0 * k + gid_cell.astype(np.float64)
    np.savez(path / f"rank{rank}.npz", gid_cell=gid_cell, gid_edge=gid_edge, **fields)


def test_assemble_orders_by_global_index(tmp_path):
    _write_rank(tmp_path, 0, np.array([3, 0]), np.array([4, 1, 2]), steps=2)
    _write_rank(tmp_path, 1, np.array([1, 2]), np.array([0, 3]), steps=2)
    steps = icon4py_jax_spmd.assemble(tmp_path, 2)
    for k, fields in enumerate(steps):
        np.testing.assert_array_equal(fields["vn"], 10.0 * k + np.arange(5))
        np.testing.assert_array_equal(fields["rho"], 10.0 * k + np.arange(4))


def test_assemble_refuses_overlapping_owners(tmp_path):
    _write_rank(tmp_path, 0, np.array([0, 1]), np.array([0]), steps=1)
    _write_rank(tmp_path, 1, np.array([1, 2]), np.array([1]), steps=1)
    with pytest.raises(AssertionError, match="no partition"):
        icon4py_jax_spmd.assemble(tmp_path, 1)
