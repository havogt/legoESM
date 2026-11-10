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
TRACERS = ("qv", "qc", "qi", "qr", "qs", "qg")

# icon4py's double-precision tolerances (atol, rtol), model/driver/tests/.../test_driver.py
JW_SAVEPOINT_TOLERANCES = {
    "vn": (5.3e-7, 0.0),
    "w": (8e-9, 0.0),
    "exner": (4.5e-11, 5.5e-11),
    "theta_v": (5.5e-8, 1.3e-10),
    "rho": (1.5e-10, 2.2e-10),
}
APE_AES_SAVEPOINT_TOLERANCES = {
    "vn": (6e-7, 0.0),
    "w": (1e-8, 0.0),
    "rho": (9e-10, 0.0),
    "exner": (1e-8, 0.0),
    "theta_v": (0.0, 3e-8),
    "qv": (1e-8, 0.0),
    "qc": (1e-10, 0.0),
    "qr": (1e-10, 0.0),
    "qs": (1e-10, 0.0),
    "qi": (1e-10, 0.0),
    "qg": (1e-10, 0.0),
}
SAVEPOINT_TOLERANCES = {
    "exclaim_nh35_tri_jws_sb": JW_SAVEPOINT_TOLERANCES,
    "exclaim_ape_aesPhys_sb": APE_AES_SAVEPOINT_TOLERANCES,
}


def savepoint_fields(experiment: pathlib.Path, date: str) -> dict[str, np.ndarray]:
    """The single-rank ICON run's prognostic fields at the time-step-exit savepoint of `date`."""
    import serialbox

    serializer = serialbox.Serializer(
        serialbox.OpenModeKind.Read, str(experiment / "ser_data"), "icon_pydycore_rank0"
    )
    savepoint = serializer.savepoint["time-step-exit"].id[1].date[date].as_savepoint()
    fields = {
        name: np.squeeze(serializer.read(name, savepoint).astype(np.float64))
        for name in PROGNOSTIC_FIELDS
    }
    if "tracers" in serializer.fields_at_savepoint(savepoint):
        from icon4py.model.common.states import data

        tracers = np.squeeze(serializer.read("tracers", savepoint).astype(np.float64))
        for name in TRACERS:
            fields[name] = tracers[:, :, getattr(data, name.upper())]
    return fields


def _process_props(mpi: bool):
    if mpi:
        # MPI must come up with thread level MULTIPLE before icon4py imports it
        import mpi4py

        mpi4py.rc.thread_level = "multiple"
        from mpi4py import MPI  # noqa: F401
    from icon4py.model.common.decomposition import definitions as decomposition

    if not mpi:
        return decomposition.SingleNodeProcessProperties()
    from icon4py.model.common.decomposition import mpi_decomposition  # noqa: F401  # registers MPI

    return decomposition.get_process_properties(decomposition.get_runtype(with_mpi=True))


def run(args: argparse.Namespace) -> None:
    process_props = _process_props(args.mpi)
    from icon4py.model.common import dimension as dims
    from legoesm.atmosphere.dynamics.gcm import icon4py_jax

    rank = process_props.rank
    out = pathlib.Path(args.out) / args.label
    out.mkdir(parents=True, exist_ok=True)
    multi_rank = not process_props.is_single_rank()
    spmd = multi_rank and not args.no_spmd
    steps: list[dict[str, np.ndarray]] = []

    if args.mode == "adapter":
        from icon4py.model.driver import config as driver_config

        experiment_config = driver_config.read_experiment_config_from_yaml(
            pathlib.Path(args.experiment) / "config.yml"
        ).with_overrides(driver={"output_path": out / f"driver_output_rank{rank}"})
        dycore = icon4py_jax.Icon4pyJaxDycore(
            grid_file=args.grid,
            experiment_config=experiment_config,
            process_props=process_props,
            spmd=spmd,
            jit_time_step=not args.no_spmd,
            finite_halo_padding=True,
        )
        global_index = dycore.owned_global_index
        if args.diagnostics_every:
            if dycore.uses_jax_api:
                raise NotImplementedError(
                    "--diagnostics-every reads cell_area and ddqz_z_full from the driver's "
                    "static-field factories, which the jax_api path (one rank, no physics) "
                    "does not build"
                )
            _diagnosed_run(dycore, process_props, args, out)
            return
        for k in range(args.steps):
            dycore.step()
            if not args.save_last or k == args.steps - 1:
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
            extra_halo_rings=driver_utils.jax_extra_halo_rings(config) if multi_rank else 0,
        )
        owned = decomposition.DecompositionInfo.EntryType.OWNED
        info = grid_manager.decomposition_info
        global_index = {
            dim: np.asarray(info.global_index(dim, owned)) for dim in (dims.CellDim, dims.EdgeDim)
        }

        def record(time_step, state):
            if args.save_last and time_step != args.steps - 1:
                return
            steps.append(
                {
                    name: getattr(state, name).asnumpy()[
                        : global_index[dims.EdgeDim if name == "vn" else dims.CellDim].size
                    ]
                    for name in PROGNOSTIC_FIELDS
                }
            )

        ds, _ = driver.run_driver(
            config=config,
            grid_manager=grid_manager,
            process_props=process_props,
            backend=None,
            jax=True,
            jax_jit_time_step=not args.no_spmd,
            jax_exchange_read_fields_only=multi_rank,
            jax_spmd=spmd,
            on_step_end=record,
        )
        # run_driver's per-step hook sees no tracers: they are compared after the last step
        num_owned_cells = global_index[dims.CellDim].size
        steps[-1].update(
            {
                tracer.name: tracer.field.asnumpy()[:num_owned_cells]
                for tracer in ds.tracers.current.active_fields()
            }
        )

    assert len(steps) == (1 if args.save_last else args.steps)
    np.savez(
        out / f"rank{rank}.npz",
        gid_cell=global_index[dims.CellDim],
        gid_edge=global_index[dims.EdgeDim],
        **{f"{name}_{k}": fields[name] for k, fields in enumerate(steps) for name in fields},
    )
    print(f"rank {rank}: wrote {out / f'rank{rank}.npz'} ({args.steps} steps)", flush=True)


def _dry_mass(dycore, prognostics: dict[str, np.ndarray]) -> float:
    """Total density mass Σ rho·cell_area·ddqz_z_full over owned cells, as icon4py's statistics."""
    from icon4py.model.common.grid import geometry_attributes
    from icon4py.model.common.metrics import metrics_attributes

    factories = dycore.driver.static_field_factories
    rho = prognostics["rho"]
    area = np.asarray(factories.geometry.get(geometry_attributes.CELL_AREA).asnumpy())
    dz = np.asarray(factories.metrics.get(metrics_attributes.DDQZ_Z_FULL).asnumpy())
    n = rho.shape[0]
    return float(np.sum(rho * area[:n, np.newaxis] * dz[:n]))


def _water_mass(dycore, prognostics: dict[str, np.ndarray]) -> float:
    """Total water Σ (qv+qc+qi+qr+qs+qg)·rho·cell_area·ddqz_z_full over owned cells."""
    q = sum(prognostics[name] for name in TRACERS if name in prognostics)
    if isinstance(q, int):
        return 0.0
    return _dry_mass(dycore, {"rho": prognostics["rho"] * q})


def _diagnosed_run(dycore, process_props, args: argparse.Namespace, out: pathlib.Path) -> None:
    """Step and sample every N steps: finiteness, max |vn|, total mass; time unsampled steps."""
    import json
    import time

    def global_reduce(value: float, op: str) -> float:
        if process_props.is_single_rank():
            return value
        from mpi4py import MPI

        return process_props.comm.allreduce(value, op={"sum": MPI.SUM, "max": MPI.MAX}[op])

    def sample(step: int) -> dict:
        fields = dycore.prognostics()
        finite = all(bool(np.isfinite(v).all()) for v in fields.values())
        return {
            "step": step,
            "finite": bool(global_reduce(float(not finite), "sum") == 0.0),
            "max_abs_vn": global_reduce(float(np.max(np.abs(fields["vn"]))), "max"),
            "mass": global_reduce(_dry_mass(dycore, fields), "sum"),
            "water": global_reduce(_water_mass(dycore, fields), "sum"),
        }

    samples = [sample(0)]
    step_seconds = []
    for k in range(1, args.steps + 1):
        start = time.perf_counter()
        dycore.step()
        elapsed = time.perf_counter() - start
        if k > 2:
            step_seconds.append(elapsed)
        if k % args.diagnostics_every == 0 or k == args.steps:
            samples.append(sample(k))
    if process_props.rank == 0:
        mass0 = samples[0]["mass"]
        summary = {
            "ranks": process_props.comm_size,
            "steps": args.steps,
            "all_finite": all(s["finite"] for s in samples),
            "max_abs_vn_over_time": max(s["max_abs_vn"] for s in samples),
            "relative_mass_drift_final": (samples[-1]["mass"] - mass0) / mass0,
            "max_abs_relative_mass_drift": max(abs(s["mass"] - mass0) / mass0 for s in samples),
            "water_initial": samples[0]["water"],
            "water_final": samples[-1]["water"],
            "step_seconds_median": float(np.median(step_seconds)),
            "step_seconds_mean": float(np.mean(step_seconds)),
            "timed_steps": len(step_seconds),
            "samples": samples,
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
        print(json.dumps({k: v for k, v in summary.items() if k != "samples"}), flush=True)


def assemble_owned(ranks: list[dict], suffix: str = "") -> dict[str, np.ndarray]:
    """Global fields from every rank's owned points (``gid_cell``/``gid_edge`` plus
    ``<name><suffix>`` arrays), ordered by global index."""
    gid = {k: np.concatenate([r[k] for r in ranks]) for k in ("gid_cell", "gid_edge")}
    for index in gid.values():
        assert np.array_equal(np.sort(index), np.arange(index.size)), "owned points: no partition"
    return {
        name: np.concatenate([r[f"{name}{suffix}"] for r in ranks])[
            np.argsort(gid["gid_edge" if name == "vn" else "gid_cell"])
        ]
        for name in (*PROGNOSTIC_FIELDS, *TRACERS)
        if f"{name}{suffix}" in ranks[0]
    }


def assemble(
    run_dir: pathlib.Path, steps: int, pattern: str = "rank*.npz"
) -> list[dict[str, np.ndarray]]:
    """The global fields of a run, from every rank's owned points ordered by global index."""
    ranks = [dict(np.load(f)) for f in sorted(run_dir.glob(pattern))]
    assert ranks, f"no {pattern} in {run_dir}"
    return [assemble_owned(ranks, f"_{k}") for k in range(steps)]


def experiment_name(experiment: pathlib.Path) -> str:
    from icon4py.model.driver import config as driver_config

    return driver_config.read_experiment_config_from_yaml(
        experiment / "config.yml"
    ).driver.experiment_name


def step_dates(experiment: pathlib.Path, steps: int) -> list[str]:
    from icon4py.model.driver import config as driver_config

    config = driver_config.read_experiment_config_from_yaml(experiment / "config.yml")
    start = config.driver.start_of_timestepping
    dtime = config.driver.dtime
    return [(start + (k + 1) * dtime).strftime("%Y-%m-%dT%H:%M:%S.000") for k in range(steps)]


def compare(args: argparse.Namespace) -> int:
    out = pathlib.Path(args.out)
    experiment = pathlib.Path(args.experiment)
    runs = {
        label: assemble(out / label, args.steps, args.pattern.get(label, "rank*.npz"))
        for label in args.labels
    }
    failures = []
    first, *rest = args.labels
    for label in rest if args.bitwise else ():
        different = [
            f"{first} != {label}: step {k} {name}"
            for k in range(args.steps)
            for name in runs[first][k].keys() & runs[label][k].keys()
            if not np.array_equal(runs[first][k][name], runs[label][k][name])
        ]
        failures += different
        print(f"bitwise {first} vs {label}: {'DIFFERENT' if different else 'equal'}")
    tolerances = SAVEPOINT_TOLERANCES[experiment_name(experiment)]
    for label in args.labels if not args.no_savepoints else ():
        for k, date in enumerate(step_dates(experiment, args.steps)):
            reference = savepoint_fields(experiment, date)
            errors = []
            for name, (atol, rtol) in tolerances.items():
                if name not in runs[label][k]:
                    continue
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
                    for n in runs[other][-1]
                    if n in ref[-1]
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
    r.add_argument("--save-last", action="store_true", help="keep only the final step (as step 0)")
    r.add_argument(
        "--no-spmd",
        action="store_true",
        help="host halo exchange between per-substep jitted steps instead of one SPMD program",
    )
    r.add_argument(
        "--diagnostics-every", type=int, help="adapter only: sample every N steps, no npz output"
    )
    c = sub.add_parser("compare")
    c.add_argument("--out", required=True)
    c.add_argument("--experiment", required=True)
    c.add_argument("--steps", type=int, required=True)
    c.add_argument("--labels", nargs="+", required=True)
    c.add_argument("--bitwise", action="store_true")
    c.add_argument("--max-diff-against", nargs="*")
    c.add_argument("--no-savepoints", action="store_true", help="skip the savepoint check")
    c.add_argument(
        "--pattern",
        nargs="*",
        default=[],
        metavar="LABEL=GLOB",
        help="per-run file pattern (default rank*.npz)",
    )
    args = parser.parse_args(argv)
    if args.command == "compare":
        args.pattern = dict(item.split("=", 1) for item in args.pattern)
    if args.command == "run":
        run(args)
        return 0
    return compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
