"""The lane comparator averages over the named days only, lists NaN series apart, and refuses runs
whose windows hold different days."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "validate" / "icon_lane"))

import compare_lanes  # noqa: E402


def _run(tmp_path, name, days, temperature, precip):
    d = tmp_path / name
    d.mkdir()
    np.savez(d / "timeseries.npz", days=np.asarray(days, float),
             T_atm=np.asarray(temperature, float), precip=np.asarray(precip, float))
    return d


def test_the_window_mean_and_the_nan_series(tmp_path):
    days = np.arange(1, 31)
    a = _run(tmp_path, "a", days, days * 1.0, np.full(30, np.nan))
    b = _run(tmp_path, "b", days, np.full(30, 250.0), days * 0.0)
    out = compare_lanes.main([f"a={a}", f"b={b}", "--days", "21", "30"])
    # the mean of days 21..30 of a linear series is 25.5, not the 30-day mean 15.5
    assert "| T_atm | 25.5 | 250 |" in out
    assert "days 21-30 (10 days)" in out
    assert "not compared): precip" in out and "| precip |" not in out


def test_unmatched_windows_are_refused(tmp_path):
    a = _run(tmp_path, "a", np.arange(1, 31), np.zeros(30), np.zeros(30))
    b = _run(tmp_path, "b", np.arange(1, 31) + 0.5, np.zeros(30), np.zeros(30))
    with pytest.raises(ValueError, match="different|days"):
        compare_lanes.main([f"a={a}", f"b={b}", "--days", "21", "30"])
