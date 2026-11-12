"""The ICON column lane's aquaplanet (stage S2): icon4py's dynamics under legoESM's column loop
with the MPAS AMIP production deck's physics suite unchanged, on an analytic-SST aquaplanet.

    python scripts/experiment/icon_column_aquaplanet.py --grid-file <icon_grid_0013_R02B04_R.nc> \\
        --out <run dir> --days 1

The physics settings are run_amip's for config/amip/amip_production.yaml. Replaced: the grid,
dycore and ICON blocks (those of config/icon/ape_column_r2b4.yaml), the cadences (physics every
ICON step, radiation hourly, as on the MPAS deck), and what an aquaplanet without the deck's
data files needs (analytic SST/SIC, no land, inline ozone, constant greenhouse gases and solar
constant, no prescribed aerosol).
"""

from __future__ import annotations

import argparse
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
PRODUCTION_DECK = REPO / "config" / "amip" / "amip_production.yaml"
ICON_DECK = REPO / "config" / "icon" / "ape_column_r2b4.yaml"
DT_S = 300.0
RADIATION_S = 3600.0


def build_config(grid_file, out, days: int):
    """The S2 ExperimentConfig."""
    sys.path.insert(0, str(REPO / "scripts" / "run"))
    import run_amip
    from legoesm.driver.config import DycoreConfig, ExperimentConfig, GridConfig, OutputConfig

    from legoesm.config import Config

    parser = run_amip.build_arg_parser()
    argv = ["--config", str(PRODUCTION_DECK)]
    args = parser.parse_args(argv)
    run_amip._postprocess_args(args, parser, argv)
    amip = run_amip.build_config_from_args(args)

    icon_deck = Config.from_yaml(str(ICON_DECK))
    icon_deck.set("grid.grid_file", str(grid_file))
    icon_deck.set("output.path", str(out))
    icon = icon_deck.to_experiment_config()
    defaults = ExperimentConfig._field_defaults
    return amip._replace(
        **{k: defaults[k] for k in ExperimentConfig._fields if k.startswith("mpas_")},
        grid=GridConfig(grid_type="icon", nlev=icon.grid.nlev, grid_file=str(grid_file)),
        dycore=DycoreConfig(
            model_type="nonhydrostatic", discretization="icon", dt=DT_S, icon_spmd=False,
            icon_finite_halo_padding=False, column_lane=True, energy_fixer=False),
        icon=icon.icon,
        ic=icon.ic,
        topography="flat",
        days=int(days),
        start_year=icon.start_year,
        start_day=icon.start_day,
        precision="fp64",
        # run_amip turns MPI on under srun; the ICON column lane is one device so far
        distributed=False,
        ensemble_size=1,
        physics_update_steps=1,
        rad_update_steps=int(RADIATION_S / DT_S),
        dataset="analytical",
        use_multilayer_land=False,
        ozone_forcing="inline",
        ghg_forcing="constant",
        aerosol_forcing="off",
        nc_from_aerosol=False,
        solar_source="constant",
        output=OutputConfig(output_dir=str(out), diag_days=1.0),
    )


def main(argv=None):
    from legoesm.driver.model_driver import ModelDriver

    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--days", type=int, default=1)
    ap.add_argument("--validate-only", action="store_true")
    a = ap.parse_args(argv)
    cfg = build_config(a.grid_file, a.out, a.days)
    cfg.validate_strict()
    if a.validate_only:
        print("admitted")
        return cfg
    driver = ModelDriver(cfg)
    driver.setup()
    status = driver.run()
    print("status", status)
    return status


if __name__ == "__main__":
    main()
