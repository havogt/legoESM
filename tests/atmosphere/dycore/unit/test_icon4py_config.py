"""The ICON lane's decks build icon4py's EXCLAIM experiment configurations, field by field."""

import dataclasses
import os
import pathlib

import pytest

pytest.importorskip("icon4py.model.driver.jax_api")
if "ICON4PY_TEST_DATA_PATH" not in os.environ:
    pytest.skip("ICON4PY_TEST_DATA_PATH is not set", allow_module_level=True)

from legoesm.atmosphere.dynamics.gcm import icon4py_jax  # noqa: E402
from legoesm.config import Config  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[4]
DATA = pathlib.Path(os.environ["ICON4PY_TEST_DATA_PATH"]) / "ser_icondata"
DECKS = {
    "jw_r2b4": "mpitask1_exclaim_nh35_tri_jws_v12",
    "ape_aes_r2b4": "mpitask1_exclaim_ape_aesPhys_v14",
}
# what the lane sets from legoESM itself: the run label and directory, and whole-day run length
NOT_COMPARED = {"driver.experiment_name", "driver.output_path", "driver.end_of_simulation"}


def _legoesm_config(deck: str, tmp_path):
    config = Config.from_yaml(str(REPO / "config" / "icon" / f"{deck}.yaml"))
    config.set("grid.grid_file", "unused-by-the-builder.nc")
    config.set("output.path", str(tmp_path / deck))
    return config.to_experiment_config()


def _differences(ours, theirs, prefix=""):
    if dataclasses.is_dataclass(theirs) and not isinstance(theirs, type):
        if type(ours) is not type(theirs):
            return [f"{prefix}: {type(ours).__name__} != {type(theirs).__name__}"]
        out = []
        for f in dataclasses.fields(theirs):
            key = f"{prefix}.{f.name}" if prefix else f.name
            if key not in NOT_COMPARED:
                out += _differences(getattr(ours, f.name), getattr(theirs, f.name), key)
        return out
    return [] if ours == theirs else [f"{prefix}: {ours!r} != {theirs!r}"]


@pytest.mark.parametrize("deck", sorted(DECKS))
def test_deck_builds_the_exclaim_experiment(deck, tmp_path):
    from icon4py.model.driver import config as driver_config

    cfg = _legoesm_config(deck, tmp_path)
    cfg.validate_strict()
    built = icon4py_jax.build_icon4py_experiment_config(cfg)
    exclaim = driver_config.read_experiment_config_from_yaml(DATA / DECKS[deck] / "config.yml")
    assert _differences(built, exclaim) == []


def test_every_field_must_be_given():
    from icon4py.model.common.states import tracer_states

    names = [f.name for f in dataclasses.fields(tracer_states.TracerConfig)]
    full = dict.fromkeys(names, False)
    icon4py_jax._construct(tracer_states.TracerConfig, full)
    with pytest.raises(ValueError, match="missing"):
        icon4py_jax._construct(tracer_states.TracerConfig, {n: False for n in names[1:]})
    with pytest.raises(ValueError, match="unknown"):
        icon4py_jax._construct(tracer_states.TracerConfig, {**full, "not_a_field": False})
