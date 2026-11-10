"""The ICON lane's admission: validate_strict's ICON block and the factory's default-deny wall."""

import pathlib

import pytest
import yaml

from legoesm.driver import component_factory
from legoesm.driver.config import (
    DycoreConfig,
    ExperimentConfig,
    GridConfig,
    OutputConfig,
    _icon_config_from_dict,
)

REPO = pathlib.Path(__file__).resolve().parents[2]


def _icon_config(physics: bool = False) -> ExperimentConfig:
    deck = "ape_aes_r2b4" if physics else "jw_r2b4"
    icon = yaml.safe_load((REPO / "config" / "icon" / f"{deck}.yaml").read_text())["icon"]
    families = ("icon_muphys", "icon_tmx", "icon_fixed_heat_fluxes") if physics else ("none",) * 3
    return ExperimentConfig(
        grid=GridConfig(grid_type="icon", nlev=35, grid_file="grid.nc"),
        dycore=DycoreConfig(
            model_type="nonhydrostatic",
            discretization="icon",
            dt=300.0,
            icon_spmd=True,
            icon_finite_halo_padding=False,
        ),
        output=OutputConfig(output_dir="out", diag_days=1.0),
        days=1,
        start_day=244.0,
        start_year=2008,
        precision="fp64",
        ic="icon_jablonowski_williamson",
        topography="flat",
        radiation="none",
        convection="none",
        cloud_scheme="none",
        gravity_wave_drag="none",
        microphysics=families[0],
        turbulence=families[1],
        surface_bulk_scheme=families[2],
        icon=_icon_config_from_dict(icon, strict=True),
    )


@pytest.mark.parametrize("physics", [False, True])
def test_the_decks_are_admitted(physics):
    _icon_config(physics).validate_strict()


def _changed(value):
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value + 1.0 if value == value else 0.0
    if isinstance(value, str):
        return value + "_changed"
    if isinstance(value, tuple):
        return (*value, "changed")
    return "changed"


def _replace(cfg, path: str, value):
    head, _, rest = path.partition(".")
    if not rest:
        return cfg._replace(**{head: value})
    return cfg._replace(**{head: _replace(getattr(cfg, head), rest, value)})


def _unread_fields():
    cfg = _icon_config()
    return [
        path
        for path, _ in component_factory._flatten_config_fields(cfg)
        if not (path == "icon" or path.startswith("icon."))
        and path not in component_factory._ICON_ALLOWED_NONDEFAULT
    ]


def test_the_wall_covers_the_skipped_setup_fields():
    unread = set(_unread_fields())
    # settings of the setup steps the lane skips: runtime bootstrap, forcing, topography,
    # physics, initial state and parallel layout
    for path in ("n_devices", "distributed_mode", "dataset", "forcing_path", "co2_ppmv",
                 "topography_file", "use_multilayer_land", "ic_path", "T_init",
                 "enable_latlon_spmd", "output.checkpoint_format", "dycore.column_lane"):
        if path in dict(component_factory._flatten_config_fields(_icon_config())):
            assert path in unread, path
    assert len(unread) > 300


@pytest.mark.parametrize("path", _unread_fields())
def test_every_unread_field_is_refused_away_from_its_default(path):
    cfg = _icon_config()
    current = dict(component_factory._flatten_config_fields(cfg))[path]
    with pytest.raises(ValueError, match=rf"\b{path.replace('.', r'\.')}="):
        component_factory.create_icon_model(_replace(cfg, path, _changed(current)))


@pytest.mark.parametrize(
    "edit, message",
    [
        ({"convection": "sbm"}, "convection='sbm'"),
        ({"radiation": "gray"}, "radiation='gray'"),
        ({"microphysics": "icon_muphys"}, "icon.muphys disagree"),
        ({"precision": "fp32"}, "fp64 only"),
        ({"ic": "default"}, "ic='default'"),
        ({"icon": None}, "ExperimentConfig.icon is required"),
    ],
)
def test_validate_strict_refuses(edit, message):
    with pytest.raises(ValueError, match=message):
        _icon_config()._replace(**edit).validate_strict()


@pytest.mark.parametrize(
    "path, value, message",
    [
        ("grid.grid_file", None, "grid.grid_file is required"),
        ("dycore.icon_spmd", None, "dycore.icon_spmd must be set explicitly"),
        ("dycore.icon_finite_halo_padding", None, "icon_finite_halo_padding must be set"),
        ("dycore.dt", 300.5, "whole number of seconds"),
        ("output.checkpoint_days", 1, "no checkpoints yet"),
        ("output.output_dir", "", "output.output_dir is required"),
    ],
)
def test_validate_strict_refuses_nested(path, value, message):
    with pytest.raises(ValueError, match=message):
        _replace(_icon_config(), path, value).validate_strict()


def test_an_icon_field_left_out_of_a_deck_is_refused():
    icon = yaml.safe_load((REPO / "config" / "icon" / "jw_r2b4.yaml").read_text())["icon"]
    del icon["dycore"]["rayleigh_coeff"]
    with pytest.raises(ValueError, match=r"icon\.dycore\.rayleigh_coeff \(missing\)"):
        _icon_config_from_dict(icon, strict=True)
