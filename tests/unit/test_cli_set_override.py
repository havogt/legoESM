"""`legoesm run --set KEY=VALUE` reaches the driver's config and the run manifest."""

import argparse

import pytest

from legoesm import cli
from legoesm.config import Config


class _CapturingDriver:
    captured = None

    def __init__(self, config, *args, **kwargs):
        type(self).captured = config

    def setup(self):
        pass

    def run(self):
        return "COMPLETED"


def test_set_override_reaches_driver_config_and_manifest(tmp_path, monkeypatch):
    import legoesm.driver.model_driver as model_driver
    from legoesm.driver.restart import build_run_manifest

    deck = tmp_path / "deck.yaml"
    deck.write_text("grid:\n  type: cubed_sphere\n")
    monkeypatch.setattr(model_driver, "ModelDriver", _CapturingDriver)
    _CapturingDriver.captured = None
    args = argparse.Namespace(
        config=str(deck), overrides=["grid.grid_file=/data/grid.nc", "seed=7"]
    )
    cli.cmd_run(args)
    cfg = _CapturingDriver.captured
    assert cfg is not None
    assert cfg.grid.grid_file == "/data/grid.nc"
    assert cfg.seed == 7
    resolved = build_run_manifest(cfg)["config"]["resolved_config"]
    assert resolved["grid"]["grid_file"] == "/data/grid.nc"
    assert resolved["seed"] == 7


@pytest.mark.parametrize("item", ["noequals", "=value", "grid..type=x"])
def test_malformed_override_is_refused(item):
    with pytest.raises(ValueError, match="KEY=VALUE"):
        cli.apply_overrides(Config(), [item])
