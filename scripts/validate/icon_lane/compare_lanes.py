"""Side-by-side means of two or more runs' daily series (``timeseries.npz``) over one window of
model days, as a Markdown table: one row per series, one column per run. A series that is NaN in
the window of any run is listed below the table instead of averaged.

    python scripts/validate/icon_lane/compare_lanes.py mpas=<run dir> icon=<run dir> \\
        --days 21 30
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np


def window_means(series: dict, first: float, last: float) -> tuple[dict, np.ndarray]:
    """Each series' mean over the days ``first..last`` (inclusive, by the ``days`` series), and
    the days taken. A series with a NaN in the window is NaN."""
    days = np.asarray(series["days"], dtype=np.float64)
    take = (days >= first) & (days <= last)
    if not take.any():
        raise ValueError(f"no day in [{first}, {last}]; the run has days "
                         f"{days.min()}..{days.max()}")
    return ({k: float(np.mean(np.asarray(v, dtype=np.float64)[take]))
             for k, v in series.items() if k != "days"}, days[take])


def table(runs: dict, first: float, last: float) -> str:
    """The Markdown table of ``runs`` (label -> series) over ``first..last``; refuses runs whose
    windows hold different days."""
    means, taken = {}, None
    for label, series in runs.items():
        means[label], days = window_means(series, first, last)
        if taken is not None and not np.array_equal(days, taken):
            raise ValueError(f"run {label!r} has days {days.tolist()} in the window, "
                             f"the first run {taken.tolist()}")
        taken = days
    labels = list(runs)
    keys = [k for k in means[labels[0]] if all(k in means[lb] for lb in labels)]
    fed = [k for k in keys if all(np.isfinite(means[lb][k]) for lb in labels)]
    lines = [f"days {taken.min():g}-{taken.max():g} ({taken.size} days)", "",
             "| series | " + " | ".join(labels) + " |",
             "|---|" + "---|" * len(labels)]
    lines += [f"| {k} | " + " | ".join(f"{means[lb][k]:.4g}" for lb in labels) + " |"
              for k in fed]
    missing = [k for k in keys if k not in fed]
    if missing:
        lines += ["", "NaN in the window (not compared): " + ", ".join(missing)]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="label=run directory (holding timeseries.npz)")
    ap.add_argument("--days", nargs=2, type=float, required=True, metavar=("FIRST", "LAST"))
    a = ap.parse_args(argv)
    runs = {}
    for spec in a.runs:
        label, _, path = spec.partition("=")
        with np.load(pathlib.Path(path) / "timeseries.npz") as f:
            runs[label] = {k: f[k] for k in f.files}
    out = table(runs, *a.days)
    print(out)
    return out


if __name__ == "__main__":
    main()
