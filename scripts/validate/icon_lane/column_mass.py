"""Total and dry air mass of the ICON column lane from its state snapshots (area-weighted, fp64):
total = sum(area * air_mass), dry = sum(area * air_mass * (1 - q_total)), water = the rest.

    python scripts/validate/icon_lane/column_mass.py <snapshot dir> [--tag daily]
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np

SPECIES = ("qv", "qc", "qi", "qr", "qs", "qg")


def masses(snapshot) -> dict:
    """Global total, dry and water mass [kg] of one snapshot."""
    area = np.asarray(snapshot["area"], dtype=np.float64)[:, None]
    air = area * np.asarray(snapshot["air_mass"], dtype=np.float64)
    water = sum(np.asarray(snapshot[f"q_{k}"], dtype=np.float64) for k in SPECIES)
    return {"total_kg": float(air.sum()), "dry_kg": float((air * (1.0 - water)).sum()),
            "water_kg": float((air * water).sum())}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshots")
    ap.add_argument("--tag", default="daily")
    a = ap.parse_args(argv)
    rows = []
    for path in sorted(pathlib.Path(a.snapshots).glob(f"{a.tag}_step_*.npz")):
        with np.load(path) as snap:
            rows.append({"step": int(snap["step"]), **masses(snap)})
    base = rows[0] if rows else None
    for r in rows:
        d_total = r["total_kg"] / base["total_kg"] - 1.0
        d_dry = r["dry_kg"] / base["dry_kg"] - 1.0
        print(f"step {r['step']:6d}  total {r['total_kg']:.12e}  dry {r['dry_kg']:.12e}  "
              f"water {r['water_kg']:.6e}  d_total/total {d_total:+.3e}  d_dry/dry {d_dry:+.3e}")
    return rows


if __name__ == "__main__":
    main()
