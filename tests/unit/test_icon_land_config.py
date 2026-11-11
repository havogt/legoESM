"""The ICON lane's land block: the deck spells out legoESM's slab defaults, and the land
settings the lane cannot honour are refused."""

import pathlib
import types

import numpy as np
import pytest
import yaml
from legoesm.driver.config import _icon_config_from_dict
from legoesm.driver.icon_land import patch_mask
from legoesm.land.config import LandConfig
from legoesm.land.surface_scheme import SimpleSEBConfig

REPO = pathlib.Path(__file__).resolve().parents[2]
DECK = REPO / "config" / "icon" / "ape_aes_land_r2b4.yaml"


def _icon():
    return yaml.safe_load(DECK.read_text())["icon"]


def test_the_deck_writes_out_the_slab_defaults_and_the_patch():
    land = _icon_config_from_dict(_icon(), strict=True).land
    assert land.slab == LandConfig(surface_scheme=SimpleSEBConfig())
    assert (land.patch_lat_min_deg, land.patch_lat_max_deg) == (0.0, 45.0)
    assert (land.patch_lon_min_deg, land.patch_lon_max_deg) == (0.0, 60.0)
    assert land.bucket_init_frac == 0.5


@pytest.mark.parametrize(
    "edit, message",
    [
        (lambda s: s["carbon"].pop("epsilon"), r"icon\.land\.slab\.carbon\.epsilon \(missing\)"),
        (lambda s: s.update(W_max=150), r"icon\.land\.slab\.W_max \(int, expected float\)"),
        (lambda s: s.update(surface_scheme={"two_leaf": {}}), r"surface_scheme \(only"),
        (lambda s: s["stomata"].update(sif={}), r"stomata\.sif \(only null"),
        (lambda s: s.update(not_a_field=1.0), r"icon\.land\.slab\.not_a_field"),
    ],
)
def test_the_slab_block_is_strict(edit, message):
    icon = _icon()
    edit(icon["land"]["slab"])
    with pytest.raises(ValueError, match=message):
        _icon_config_from_dict(icon, strict=True)


def _lane_config(**land_edits):
    from legoesm.config import Config

    config = Config.from_yaml(str(DECK))
    config.set("grid.grid_file", "grid.nc")
    for key, value in land_edits.items():
        config.set(f"icon.land.{key}", value)
    return config.to_experiment_config()


def test_the_land_deck_is_admitted():
    _lane_config().validate_strict()


@pytest.mark.parametrize(
    "edits, message",
    [
        ({"patch_lat_min_deg": 50.0}, "patch_lat_min_deg <= patch_lat_max_deg"),
        ({"patch_lon_max_deg": 400.0}, "patch_lon_max_deg <= 360"),
        ({"bucket_init_frac": 1.5}, "bucket_init_frac must be in"),
        ({"wind_speed_min_m_s": 0.0}, "wind_speed_min_m_s must be positive"),
        ({"slab": {**_icon()["land"]["slab"],
                   "carbon": {**_icon()["land"]["slab"]["carbon"], "scheme": "seasonal"}}},
         "carbon.scheme='seasonal'"),
    ],
)
def test_validate_strict_refuses_land(edits, message):
    with pytest.raises(ValueError, match=message):
        _lane_config(**edits).validate_strict()


def test_land_needs_icon_physics():
    cfg = _lane_config()
    cfg = cfg._replace(surface_bulk_scheme="none", icon=cfg.icon._replace(surface=None))
    with pytest.raises(ValueError, match="icon.land needs icon.muphys, icon.tmx and icon.surface"):
        cfg.validate_strict()


def test_the_patch_takes_its_lower_bounds_and_wraps_longitude():
    land = _icon_config_from_dict(_icon(), strict=True).land

    mesh = types.SimpleNamespace(
        latCell=np.deg2rad([0.0, 44.999, 45.0, 10.0, 10.0, 10.0, -0.001]),
        lonCell=np.deg2rad([0.0, 59.999, 30.0, 60.001, -1.0, 360.0 - 1e-9, 30.0]),
    )
    np.testing.assert_array_equal(patch_mask(mesh, land),
                                  [True, True, False, False, False, False, False])
