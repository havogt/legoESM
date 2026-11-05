"""Run icon4py's JAX dycore on all MPI ranks, through the legoESM adapter or icon4py's own driver.

``run`` steps the experiment and writes each rank's owned prognostic points and their global
indices to ``<out>/<label>/rank<r>.npz``; ``compare`` assembles runs on the global grid and checks
them against each other and against the ICON time-step-exit savepoints.

    srun python -m legoesm.atmosphere.dynamics.gcm.icon4py_jax_spmd run --mode adapter ...
    python -m legoesm.atmosphere.dynamics.gcm.icon4py_jax_spmd compare --out ... --experiment ...
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np

PROGNOSTIC_FIELDS = ("vn", "w", "rho", "exner", "theta_v")

# icon4py's JW double-precision tolerances (atol, rtol), model/driver/tests/.../test_driver.py
JW_SAVEPOINT_TOLERANCES = {
    "vn": (5.3e-7, 0.0),
    "w": (8e-9, 0.0),
    "exner": (4.5e-11, 5.5e-11),
    "theta_v": (5.5e-8, 1.3e-10),
    "rho": (1.5e-10, 2.2e-10),
}


def savepoint_fields(experiment: pathlib.Path, date: str) -> dict[str, np.ndarray]:
    """The single-rank ICON run's prognostic fields at the time-step-exit savepoint of `date`."""
    import serialbox

    serializer = serialbox.Serializer(
        serialbox.OpenModeKind.Read, str(experiment / "ser_data"), "icon_pydycore_rank0"
    )
    savepoint = serializer.savepoint["time-step-exit"].id[1].date[date].as_savepoint()
    return {
        name: np.squeeze(serializer.read(name, savepoint).astype(np.float64))
        for name in PROGNOSTIC_FIELDS
    }


def _process_props(mpi: bool):
    if mpi:
        # MPI must come up with thread level MULTIPLE before icon4py imports it
        import mpi4py

        mpi4py.rc.thread_level = "multiple"
        from mpi4py import MPI  # noqa: F401
    from icon4py.model.common.decomposition import definitions as decomposition

    if not mpi:
        return decomposition.SingleNodeProcessProperties()
    return decomposition.get_process_properties(decomposition.get_runtype(with_mpi=True))


def run(args: argparse.Namespace) -> None:
    process_props = _process_props(args.mpi)
    from icon4py.model.common import dimension as dims
    from legoesm.atmosphere.dynamics.gcm import icon4py_jax

    rank = process_props.rank
    out = pathlib.Path(args.out) / args.label
    out.mkdir(parents=True, exist_ok=True)
    multi_rank = not process_props.is_single_rank()
    steps: list[dict[str, np.ndarray]] = []

    if args.mode == "adapter":
        dycore = icon4py_jax.Icon4pyJaxDycore(
            grid_file=args.grid,
            experiment_config=pathlib.Path(args.experiment) / "config.yml",
            output_path=out / f"driver_output_rank{rank}",
            process_props=process_props,
            spmd=multi_rank,
            finite_halo_padding=True,
        )
        global_index = dycore.owned_global_index
        for _ in range(args.steps):
            dycore.step()
            steps.append(dycore.prognostics())
    else:
        from icon4py.model.common import model_backends
        from icon4py.model.common.decomposition import definitions as decomposition
        from icon4py.model.driver import config as driver_config
        from icon4py.model.driver import driver, driver_utils

        config = driver_config.read_experiment_config_from_yaml(
            pathlib.Path(args.experiment) / "config.yml"
        )
        config = config.with_overrides(
            driver={
                "output_path": out / f"driver_output_rank{rank}",
                "end_of_simulation": args.steps,
            }
        )
        grid_manager = driver_utils.create_grid_manager(
            grid_file_path=pathlib.Path(args.grid),
            vertical_grid_config=config.vertical_grid,
            allocator=model_backends.get_allocator(None),
            process_props=process_props,
            extra_halo_rings=driver_utils.JAX_EXTRA_HALO_RINGS if multi_rank else 0,
        )
        owned = decomposition.DecompositionInfo.EntryType.OWNED
        info = grid_manager.decomposition_info
        global_index = {
            dim: np.asarray(info.global_index(dim, owned)) for dim in (dims.CellDim, dims.EdgeDim)
        }

        def record(time_step, state):
            steps.append(
                {
                    name: getattr(state, name).asnumpy()[
                        : global_index[dims.EdgeDim if name == "vn" else dims.CellDim].size
                    ]
                    for name in PROGNOSTIC_FIELDS
                }
            )

        driver.run_driver(
            config=config,
            grid_manager=grid_manager,
            process_props=process_props,
            backend=None,
            jax=True,
            jax_jit_time_step=True,
            jax_exchange_read_fields_only=multi_rank,
            jax_spmd=multi_rank,
            on_step_end=record,
        )

    assert len(steps) == args.steps
    np.savez(
        out / f"rank{rank}.npz",
        gid_cell=global_index[dims.CellDim],
        gid_edge=global_index[dims.EdgeDim],
        **{f"{name}_{k}": fields[name] for k, fields in enumerate(steps) for name in fields},
    )
    print(f"rank {rank}: wrote {out / f'rank{rank}.npz'} ({args.steps} steps)", flush=True)


def assemble(run_dir: pathlib.Path, steps: int) -> list[dict[str, np.ndarray]]:
    """The global fields of a run, from every rank's owned points ordered by global index."""
    ranks = [dict(np.load(f)) for f in sorted(run_dir.glob("rank*.npz"))]
    assert ranks, f"no rank*.npz in {run_dir}"
    gid = {k: np.concatenate([r[k] for r in ranks]) for k in ("gid_cell", "gid_edge")}
    for index in gid.values():
        assert np.array_equal(np.sort(index), np.arange(index.size)), "owned points: no partition"
    return [
        {
            name: np.concatenate([r[f"{name}_{k}"] for r in ranks])[
                np.argsort(gid["gid_edge" if name == "vn" else "gid_cell"])
            ]
            for name in PROGNOSTIC_FIELDS
        }
        for k in range(steps)
    ]


def step_dates(experiment: pathlib.Path, steps: int) -> list[str]:
    from icon4py.model.driver import config as driver_config

    config = driver_config.read_experiment_config_from_yaml(experiment / "config.yml")
    start = config.driver.start_of_timestepping
    dtime = config.driver.dtime
    return [(start + (k + 1) * dtime).strftime("%Y-%m-%dT%H:%M:%S.000") for k in range(steps)]


def compare(args: argparse.Namespace) -> int:
    out = pathlib.Path(args.out)
    experiment = pathlib.Path(args.experiment)
    runs = {label: assemble(out / label, args.steps) for label in args.labels}
    failures = []
    first, *rest = args.labels
    for label in rest if args.bitwise else ():
        different = [
            f"{first} != {label}: step {k} {name}"
            for k in range(args.steps)
            for name in PROGNOSTIC_FIELDS
            if not np.array_equal(runs[first][k][name], runs[label][k][name])
        ]
        failures += different
        print(f"bitwise {first} vs {label}: {'DIFFERENT' if different else 'equal'}")
    for label in args.labels:
        for k, date in enumerate(step_dates(experiment, args.steps)):
            reference = savepoint_fields(experiment, date)
            errors = []
            for name, (atol, rtol) in JW_SAVEPOINT_TOLERANCES.items():
                ours = runs[label][k][name]
                theirs = reference[name][: ours.shape[0]]
                assert theirs.shape == ours.shape, (label, name, theirs.shape, ours.shape)
                errors.append(f"{name}={np.max(np.abs(ours - theirs)):.2e}")
                if not np.allclose(ours, theirs, atol=atol, rtol=rtol):
                    failures.append(f"{label} vs savepoint: {date} {name}")
            print(f"{label} vs savepoint {date}: " + " ".join(errors))
    for label in args.max_diff_against or ():
        ref = assemble(out / label, args.steps)
        for other in args.labels:
            print(
                f"max |{other} - {label}| last step: "
                + " ".join(
                    f"{n}={np.max(np.abs(runs[other][-1][n] - ref[-1][n])):.2e}"
                    for n in PROGNOSTIC_FIELDS
                )
            )
    for failure in failures:
        print("FAIL", failure)
    print("RESULT", "PASS" if not failures else "FAIL")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--mode", choices=("adapter", "reference"), required=True)
    r.add_argument("--label", required=True)
    r.add_argument("--grid", required=True)
    r.add_argument("--experiment", required=True)
    r.add_argument("--steps", type=int, required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--mpi", action="store_true", help="decompose over the MPI ranks")
    c = sub.add_parser("compare")
    c.add_argument("--out", required=True)
    c.add_argument("--experiment", required=True)
    c.add_argument("--steps", type=int, required=True)
    c.add_argument("--labels", nargs="+", required=True)
    c.add_argument("--bitwise", action="store_true")
    c.add_argument("--max-diff-against", nargs="*")
    args = parser.parse_args(argv)
    if args.command == "run":
        run(args)
        return 0
    return compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
