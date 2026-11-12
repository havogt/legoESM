"""The ICON column lane's aquaplanet configuration: the MPAS AMIP production deck's physics suite
unchanged, on ICON's grid and dycore, admitted by validate_strict."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "experiment"))
sys.path.insert(0, str(REPO / "scripts" / "run"))

import icon_column_aquaplanet  # noqa: E402

SUITE = ("radiation", "cloud_scheme", "convection", "microphysics", "turbulence",
         "gravity_wave_drag", "surface_bulk_scheme", "diurnal_cycle", "co2_ppmv")


def _amip():
    import run_amip

    parser = run_amip.build_arg_parser()
    argv = ["--config", str(icon_column_aquaplanet.PRODUCTION_DECK)]
    args = parser.parse_args(argv)
    run_amip._postprocess_args(args, parser, argv)
    return run_amip.build_config_from_args(args)


def test_the_suite_is_the_production_decks_on_icon(tmp_path):
    cfg = icon_column_aquaplanet.build_config("grid.nc", tmp_path, 1)
    cfg.validate_strict()
    amip = _amip()
    for name in SUITE:
        assert getattr(cfg, name) == getattr(amip, name), name
    assert cfg.dycore.discretization == "icon" and cfg.dycore.column_lane
    assert cfg.physics_update_steps == 1 and cfg.rad_update_steps * cfg.dycore.dt == 3600.0
    assert cfg.dataset == "analytical" and not cfg.use_multilayer_land
