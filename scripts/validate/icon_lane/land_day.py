"""One ICON-lane day with legoESM land: hourly land surface temperature, the land's surface
energy and water budget residuals, and the atmosphere's total water and internal energy.

    python scripts/validate/icon_lane/land_day.py config/icon/ape_aes_land_r2b4.yaml \\
        --grid-file <icon_grid_0013_R02B04_R.nc> --out land_day.json [--set KEY=VALUE ...]
"""

from __future__ import annotations

import argparse
import json
import subprocess

import numpy as np


def surface_energy_residual(cur, t_old, t_new, heat_capacity, dt):
    """Net radiation minus turbulent heat minus soil storage [W/m2] on the land cells: the
    radiation at the step's start temperature, the fluxes of the step (slab: no melt term)."""
    from legoesm.core.surface_energy import surface_radiation_fluxes

    sw_net, lw_net, _ = surface_radiation_fluxes(
        cur["sw_down"], cur["lw_down"], t_old, cur["albedo"], cur["emissivity"])
    return (np.asarray(sw_net) + np.asarray(lw_net) - cur["shflx"] - cur["lhflx"]
            - heat_capacity * (t_new - t_old) / dt)


def water_residual(w_old, w_new, snow_old, snow_new, cur, dt):
    """Change of bucket and snow water minus precipitation less evaporation and runoff
    [kg/m2] over the step."""
    return (w_new - w_old) + (snow_new - snow_old) - (
        cur["precipitation"] - cur["evaporation"] - cur["runoff"]) * dt


def _atmosphere_totals(model, area):
    from icon4py.model.driver import jax_api
    from legoesm.driver import icon_radiation

    col = jax_api.dyn2phy(model._api_static, model._api_state)
    q_total = sum(np.asarray(col.q[k].ndarray) for k in ("qv", "qc", "qi", "qr", "qs", "qg"))
    air_mass = np.asarray(col.air_mass.ndarray)
    cv = np.asarray(icon_radiation.cv_air(col))
    temperature = np.asarray(col.temperature.ndarray)
    return {
        "water_kg": float(np.sum(area[:, None] * air_mass * q_total)),
        "internal_energy_J": float(np.sum(area[:, None] * cv * temperature)),
    }


def main(argv=None) -> dict:
    from legoesm.driver import component_factory

    from legoesm.config import Config

    ap = argparse.ArgumentParser()
    ap.add_argument("deck")
    ap.add_argument("--grid-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--steps", type=int, default=288)
    ap.add_argument("--every", type=int, default=12)
    args = ap.parse_args(argv)

    config = Config.from_yaml(args.deck)
    config.set("grid.grid_file", args.grid_file)
    for item in args.set:
        key, _, value = item.partition("=")
        config.set(key, json.loads(value))
    cfg = config.to_experiment_config()
    model = component_factory.create_icon_model(cfg)
    dt = float(cfg.dycore.dt)
    area = np.asarray(component_factory._icon_mesh(cfg).areaCell)
    mask = np.asarray(model.land.mask) if model.land is not None else np.zeros(area.size, bool)
    slab = cfg.icon.land.slab if cfg.icon.land is not None else None
    record = {
        "deck": args.deck, "sets": args.set, "steps": args.steps, "dt_s": dt,
        "land_cells": int(mask.sum()),
        "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True).stdout.strip(),
        "start": _atmosphere_totals(model, area), "hours": [],
    }
    worst_seb, worst_water, worst_seb_snow_free, snow_max, snow_cells = 0.0, 0.0, 0.0, 0.0, 0
    for step in range(1, args.steps + 1):
        before = None
        if mask.any():
            land = model.carry["land"]
            before = (np.asarray(land.T_soil.data)[mask], np.asarray(land.W_bucket.data)[mask],
                      np.asarray(land.snow_depth.data)[mask])
        model.step()
        if before is not None:
            land, cur = model.carry["land"], model.carry["land_surface"]
            cur = {k: np.asarray(v)[mask] for k, v in cur.items()}
            t_new = np.asarray(land.T_soil.data)[mask]
            seb = surface_energy_residual(cur, before[0], t_new,
                                          slab.C_soil * slab.d_soil, dt)
            water = water_residual(before[1], np.asarray(land.W_bucket.data)[mask], before[2],
                                   np.asarray(land.snow_depth.data)[mask], cur, dt)
            worst_seb = max(worst_seb, float(np.max(np.abs(seb))))
            snow_new = np.asarray(land.snow_depth.data)[mask]
            # the slab's storage also loses the snow-melt energy, which the residual leaves out
            snowy = (before[2] > 0.0) | (snow_new > 0.0)
            if (~snowy).any():
                worst_seb_snow_free = max(worst_seb_snow_free, float(np.max(np.abs(seb[~snowy]))))
            snow_max = max(snow_max, float(snow_new.max()))
            snow_cells = max(snow_cells, int(snowy.sum()))
            worst_water = max(worst_water, float(np.max(np.abs(water))))
        if step % args.every == 0:
            hour = {"step": step, **_atmosphere_totals(model, area)}
            if before is not None:
                hour.update(t_soil_min=float(t_new.min()), t_soil_mean=float(t_new.mean()),
                            t_soil_max=float(t_new.max()),
                            sw_down_mean=float(cur["sw_down"].mean()),
                            shflx_mean=float(cur["shflx"].mean()),
                            lhflx_mean=float(cur["lhflx"].mean()))
            record["hours"].append(hour)
            print(json.dumps(hour), flush=True)
    prognostics = model.prognostics()
    record["finite"] = {k: bool(np.isfinite(v).all()) for k, v in prognostics.items()}
    record["max_abs_land_seb_residual_W_m2"] = worst_seb
    record["max_abs_land_water_residual_kg_m2"] = worst_water
    record["max_abs_land_seb_residual_snow_free_W_m2"] = worst_seb_snow_free
    record["max_snow_kg_m2"] = snow_max
    record["max_snowy_land_cells"] = snow_cells
    with open(args.out, "w") as fh:
        json.dump(record, fh, indent=1)
    return record


if __name__ == "__main__":
    main()
