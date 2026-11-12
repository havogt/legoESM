"""The column-mass probe: dry plus water is the total, and a rain-out at constant air mass
turns water into dry air."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "validate" / "icon_lane"))

import column_mass  # noqa: E402


def _snap(qv):
    z = np.zeros((2, 3))
    return {"area": np.array([1.0, 3.0]), "air_mass": np.full((2, 3), 100.0),
            "q_qv": z + qv, **{f"q_{k}": z for k in column_mass.SPECIES[1:]}}


def test_dry_and_water_add_to_the_total_and_rain_out_moves_water_to_dry():
    wet, dry = column_mass.masses(_snap(0.01)), column_mass.masses(_snap(0.0))
    assert wet["dry_kg"] + wet["water_kg"] == np.float64(wet["total_kg"])
    np.testing.assert_allclose(wet["water_kg"], 0.01 * 1200.0)
    assert dry["total_kg"] == wet["total_kg"] and dry["dry_kg"] > wet["dry_kg"]
