"""The ICON-lane keys of the `legoesm run` YAML reach ExperimentConfig only when given."""

from legoesm.atmosphere.dynamics.gcm import icon_config
from legoesm.config import Config
from legoesm.driver.config import ExperimentConfig

_BLOCKS = dict(
    zip(
        icon_config.IconConfig._fields,
        (
            icon_config.IconDycoreConfig,
            icon_config.IconVerticalConfig,
            icon_config.IconJWConfig,
            icon_config.IconDiffusionConfig,
            icon_config.IconTracerAdvectionConfig,
            icon_config.IconMuphysConfig,
            icon_config.IconTmxConfig,
            icon_config.IconSurfaceConfig,
        ),
        strict=True,
    )
)
_ICON_FIELDS = ("grid_file", "icon_spmd", "icon_finite_halo_padding", "ic", "topography")


def _icon_block():
    return {name: {f: 1.0 for f in cls._fields} for name, cls in _BLOCKS.items()}


def _present():
    return {
        "grid": {"grid_file": "/data/grid.nc"},
        "atmosphere": {"icon_spmd": True, "icon_finite_halo_padding": False},
        "ic": "icon_jablonowski_williamson",
        "topography": "icon_jablonowski_williamson",
        "time": {"start_year": 2008},
        "icon": _icon_block(),
    }


def test_each_mapping_reaches_experiment_config():
    cfg = Config.from_dict(_present()).to_experiment_config()
    assert cfg.grid.grid_file == "/data/grid.nc"
    assert cfg.dycore.icon_spmd is True
    assert cfg.dycore.icon_finite_halo_padding is False
    assert cfg.ic == "icon_jablonowski_williamson"
    assert cfg.topography == "icon_jablonowski_williamson"
    assert cfg.start_year == 2008
    expected = icon_config.IconConfig(
        **{name: cls(**block) for (name, cls), block in zip(
            _BLOCKS.items(), _icon_block().values(), strict=True)}
    )
    assert cfg.icon == expected


def test_absent_keys_keep_experiment_config_defaults():
    default = ExperimentConfig()
    cfg = Config.from_dict({}).to_experiment_config()
    assert cfg.grid.grid_file == default.grid.grid_file
    assert cfg.dycore.icon_spmd == default.dycore.icon_spmd
    assert cfg.dycore.icon_finite_halo_padding == default.dycore.icon_finite_halo_padding
    assert cfg.ic == default.ic
    assert cfg.topography == default.topography
    assert cfg.start_year == default.start_year
    assert cfg.icon is None


def test_mappings_change_only_their_own_fields():
    from legoesm.driver.config import experiment_config_to_dict

    base = experiment_config_to_dict(Config.from_dict({}).to_experiment_config())
    new = experiment_config_to_dict(Config.from_dict(_present()).to_experiment_config())
    changed = {k for k in base if base[k] != new[k]}
    assert changed == {"grid", "dycore", "ic", "topography", "start_year", "icon"}
    for section, field in (("grid", "grid_file"), ("dycore", "icon_spmd"),
                           ("dycore", "icon_finite_halo_padding")):
        assert base[section][field] != new[section][field]
    for section in ("grid", "dycore"):
        diff = {k for k in base[section] if base[section][k] != new[section][k]}
        assert diff <= set(_ICON_FIELDS)
