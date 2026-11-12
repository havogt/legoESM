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
    out = {"total_kg": float(air.sum()), "dry_kg": float((air * (1.0 - water)).sum()),
           "water_kg": float((air * water).sum())}
    if "acc_precip" in snapshot:
        a = area[:, 0]
        out["precip_kg"] = float((a * np.asarray(snapshot["acc_precip"], dtype=np.float64)).sum())
        out["evap_kg"] = float((a * np.asarray(snapshot["acc_evap"], dtype=np.float64)).sum())
        for k in ("water_physics", "water_positivity"):
            if f"acc_{k}" in snapshot:
                out[f"{k}_kg"] = float((a * np.asarray(snapshot[f"acc_{k}"], np.float64)).sum())
    return out


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
        line = (f"step {r['step']:6d}  d_total/total {d_total:+.3e}  d_dry/dry {d_dry:+.3e}  "
                f"d_dry {r['dry_kg'] - base['dry_kg']:+.6e} kg  "
                f"d_water {r['water_kg'] - base['water_kg']:+.6e} kg")
        if "precip_kg" in r:
            p_e = (r["precip_kg"] - base["precip_kg"]) - (r["evap_kg"] - base["evap_kg"])
            line += f"  P-E {p_e:+.6e} kg  E-P since start {r['evap_kg'] - r['precip_kg']:+.6e} kg"
        for k in ("water_physics", "water_positivity"):
            if f"{k}_kg" in r:
                line += f"  {k} since start {r[f'{k}_kg']:+.6e} kg"
        print(line)
    return rows


if __name__ == "__main__":
    main()
