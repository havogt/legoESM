"""The ICON column lane's aquaplanet configuration: the MPAS AMIP production deck's physics suite
unchanged, on ICON's grid and dycore, admitted by validate_strict."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "experiment"))
sys.path.insert(0, str(REPO / "scripts" / "run"))

import icon_column_aquaplanet  # noqa: E402


def test_the_suite_is_the_production_decks_on_icon(tmp_path):
    import yaml

    cfg = icon_column_aquaplanet.build_config("grid.nc", tmp_path, 1)
    cfg.validate_strict()
    # the witness is the deck file itself, not run_amip's reading of it
    deck = yaml.safe_load(icon_column_aquaplanet.PRODUCTION_DECK.read_text())
    for field, key in (("convection", "convection"), ("microphysics", "microphysics"),
                       ("turbulence", "turbulence"), ("gravity_wave_drag", "gravity_wave_drag"),
                       ("surface_bulk_scheme", "surface_bulk_scheme"),
                       ("cloud_scheme", "clouds"), ("diurnal_cycle", "diurnal_cycle")):
        assert getattr(cfg, field) == deck[key], field
    assert deck["radiation"] == "rrtmg" and cfg.radiation in ("rrtmg", "rrtmgp")
    assert cfg.dycore.discretization == "icon" and cfg.dycore.column_lane
    assert cfg.physics_update_steps == 1 and cfg.rad_update_steps * cfg.dycore.dt == 3600.0
    assert cfg.dataset == "analytical" and not cfg.use_multilayer_land
    assert not cfg.distributed


def test_both_lanes_start_from_the_same_state(tmp_path):
    icon = icon_column_aquaplanet.build_config("grid.nc", tmp_path, 1, "icon")
    mpas = icon_column_aquaplanet.build_config("grid.nc", tmp_path, 1, "mpas")
    mpas.validate_strict()
    for cfg in (icon, mpas):
        assert (cfg.ic, cfg.rh_init, cfg.topography) == ("dcmip2016_baroclinic_wave", 0.7, "flat")
