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
AQUAPLANET_FLAGS = (
    "--dataset", "analytical", "--ic", "default", "--solar-source", "constant",
    "--ozone-forcing", "inline", "--ghg-forcing", "constant", "--aerosol-forcing", "off",
    "--no-aerosol-ccn", "--land-model", "none", "--land-ic", "",
    # land settings are inert without land, but run_amip refuses a canopy without its tile
    "--land-surface-scheme", "simple_seb",
)


def amip_config(run_amip, deck, cli=()):
    """run_amip's ExperimentConfig for ``deck`` and command-line flags ``cli``, by run_amip.main's
    own sequence: the deck's keys become the parser's defaults, the flags override them, then
    the land model and the sentinels resolve."""
    from legoesm.driver.run_config_yaml import load_yaml_config

    argv = ["--config", str(deck), *cli]
    parser = run_amip.build_arg_parser()
    keys = load_yaml_config(str(deck), parser, example_keys="")
    parser.set_defaults(**keys)
    parser.set_defaults(_config_keys=frozenset(keys))
    args = parser.parse_args(argv)
    run_amip._resolve_land_model(args, parser, argv)
    run_amip._postprocess_args(args, parser, argv)
    return run_amip.build_config_from_args(args)


def _aquaplanet(amip, out, days, start_year, start_day):
    """The aquaplanet's settings, the same on both lanes."""
    from legoesm.driver.config import ExperimentConfig, OutputConfig

    defaults = ExperimentConfig._field_defaults
    return amip._replace(
        # an aquaplanet has no land: the deck's land settings would be inert (and are refused)
        **{k: defaults[k] for k in ExperimentConfig._fields
           if k.startswith("land_") or k in ("ocean_flux_on_ocean_surface",)},
        topography="flat",
        days=int(days),
        start_year=start_year,
        start_day=start_day,
        precision="fp64",
        # run_amip turns MPI on under srun; both runs are one device
        distributed=False,
        ensemble_size=1,
        dataset="analytical",
        use_multilayer_land=False,
        # ZM's land/ocean autoconversion split has no land to read on an aquaplanet
        zm_land_fraction="none",
        ozone_forcing="inline",
        ghg_forcing="constant",
        aerosol_forcing="off",
        nc_from_aerosol=False,
        solar_source="constant",
        output=OutputConfig(output_dir=str(out), diag_days=1.0),
    )


def build_config(grid_file, out, days: int, lane: str = "icon"):
    """The S2 ExperimentConfig: ``lane`` 'icon' (the ICON column lane) or 'mpas' (the MPAS lane
    as the production deck has it: its own grid, dycore, time step and physics cadence)."""
    sys.path.insert(0, str(REPO / "scripts" / "run"))
    import run_amip
    from legoesm.driver.config import DycoreConfig, ExperimentConfig, GridConfig

    from legoesm.config import Config

    # the deck's production launcher passes the data paths on the command line; the aquaplanet
    # takes the analytic SST/SIC instead and needs none of them
    amip = amip_config(run_amip, PRODUCTION_DECK, cli=AQUAPLANET_FLAGS)
    icon_deck = Config.from_yaml(str(ICON_DECK))
    icon_deck.set("grid.grid_file", str(grid_file))
    icon_deck.set("output.path", str(out))
    icon = icon_deck.to_experiment_config()
    planet = _aquaplanet(amip, out, days, icon.start_year, icon.start_day)
    if lane == "mpas":
        return planet
    if lane != "icon":
        raise ValueError(f"lane={lane!r}: 'icon' or 'mpas'")
    defaults = ExperimentConfig._field_defaults
    return planet._replace(
        **{k: defaults[k] for k in ExperimentConfig._fields if k.startswith("mpas_")},
        grid=GridConfig(grid_type="icon", nlev=icon.grid.nlev, grid_file=str(grid_file)),
        dycore=DycoreConfig(
            model_type="nonhydrostatic", discretization="icon", dt=DT_S, icon_spmd=False,
            icon_finite_halo_padding=False, column_lane=True, energy_fixer=False),
        icon=icon.icon,
        ic=icon.ic,
        physics_update_steps=1,
        rad_update_steps=int(RADIATION_S / DT_S),
    )


def main(argv=None):
    from legoesm.driver.model_driver import ModelDriver

    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--days", type=int, default=1)
    ap.add_argument("--lane", choices=("icon", "mpas"), default="icon")
    ap.add_argument("--validate-only", action="store_true")
    a = ap.parse_args(argv)
    cfg = build_config(a.grid_file, a.out, a.days, a.lane)
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
