"""icon4py's JAX non-hydrostatic dynamical core, stepped from legoESM (no physics).

A thin wrapper over ``icon4py.model.driver.jax_driver.JaxIcon4pyDriver`` from the icon4py
``jax-physics`` line (the optional ``icon`` extra). ``step`` is the body of that driver's time
loop, on one rank or as icon4py's SPMD program over all ranks; it goes away once icon4py
provides a pure ``dynamics_step``.
"""

from __future__ import annotations

import functools
import pathlib

import numpy as np

PROGNOSTIC_FIELDS = ("vn", "w", "rho", "exner", "theta_v")


class Icon4pyJaxDycore:
    """Set up an icon4py experiment on the embedded backend and step it under ``jax.jit``."""

    def __init__(
        self,
        *,
        grid_file: str | pathlib.Path,
        experiment_config: str | pathlib.Path,
        output_path: str | pathlib.Path,
        jit_time_step: bool = True,
        process_props=None,
        spmd: bool = False,
        finite_halo_padding: bool = False,
    ) -> None:
        from icon4py.model.common import dimension as dims
        from icon4py.model.common import model_backends
        from icon4py.model.common.decomposition import definitions as decomposition
        from icon4py.model.driver import config as driver_config
        from icon4py.model.driver import driver, driver_utils, jax_driver, jax_utils

        self._jax_driver = jax_driver
        config = driver_config.read_experiment_config_from_yaml(pathlib.Path(experiment_config))
        config = config.with_overrides(driver={"output_path": pathlib.Path(output_path)})
        if process_props is None:
            process_props = decomposition.SingleNodeProcessProperties()
        multi_rank = not process_props.is_single_rank()
        if spmd and multi_rank:
            jax_utils.initialize_distributed(process_props.comm)
        allocator = model_backends.get_allocator(None)
        grid_manager = driver_utils.create_grid_manager(
            grid_file_path=pathlib.Path(grid_file),
            vertical_grid_config=config.vertical_grid,
            allocator=allocator,
            process_props=process_props,
            extra_halo_rings=driver_utils.JAX_EXTRA_HALO_RINGS if multi_rank else 0,
        )
        info = grid_manager.decomposition_info
        owned = decomposition.DecompositionInfo.EntryType.OWNED
        self.owned_global_index = {
            dim: np.asarray(info.global_index(dim, owned)) for dim in (dims.CellDim, dims.EdgeDim)
        }
        self._field_dim = {"vn": dims.EdgeDim}
        self.driver = driver.initialize_driver(
            config=config,
            grid_manager=grid_manager,
            process_props=process_props,
            backend=None,
            jax=True,
            jax_jit_time_step=jit_time_step,
            jax_exchange_read_fields_only=multi_rank,
            jax_spmd=spmd,
        )
        self.driver.finite_halo_padding = finite_halo_padding
        ds = driver.initialize_driver_states(
            icon4py_driver=self.driver, allocator=allocator, validate_granules=False
        )
        (
            self._prognostic,
            diagnostic,
            self._intermediate,
            self._prep_adv,
            self._tracers,
            self._constants,
        ) = self.driver._prepare(ds)
        self._diagnostic_state = jax_driver._diagnostic_from_dict(diagnostic)
        self.steps_done = 0

    def step(self) -> None:
        d = self.driver
        time_vars = d.model_time_variables
        time_vars.advance_simulation_datetime()
        key = (
            time_vars.is_first_step_in_simulation,
            float(time_vars.substep_timestep),
            time_vars.ndyn_substeps_var,
            float(d._second_order_divdamp_factor()),
            float(time_vars.dtime_in_seconds),
            d._even_timestep,
        )
        args = (
            self._constants,
            self._prognostic,
            self._jax_driver._diagnostic_to_dict(self._diagnostic_state),
            self._intermediate,
            self._prep_adv,
            self._tracers,
        )
        if d._layout is not None:
            spmd_step, state_treedef = d._jitted_spmd_time_step(key, args)
            state = d._jax.tree_util.tree_unflatten(
                state_treedef, spmd_step(*d._jax.tree_util.tree_leaves(args))
            )
        else:
            if d.jit_time_step:
                d._compute_statistics(0, self._prognostic_state())
                step = d._jitted_time_step(key)
            else:
                step = functools.partial(d._time_step, key=key)
            state = step(*args)
        (
            self._prognostic,
            diagnostic,
            self._intermediate,
            self._prep_adv,
            self._tracers,
        ) = state
        self._diagnostic_state = self._jax_driver._diagnostic_from_dict(diagnostic)
        d._jax.block_until_ready(self._prognostic["vn"].ndarray)
        time_vars.is_first_step_in_simulation = False
        if self._tracers:
            d._even_timestep = not d._even_timestep
        d._adjust_ndyn_substeps_var(self._diagnostic_state)
        self.steps_done += 1

    def _prognostic_state(self):
        from icon4py.model.common.states import prognostic_state

        return prognostic_state.PrognosticState(**self._prognostic)

    def prognostics(self) -> dict[str, np.ndarray]:
        """This rank's owned points of the prognostic fields, in local order."""
        from icon4py.model.common import dimension as dims

        local = self.driver._local(self._prognostic)
        return {
            name: np.asarray(local[name].asnumpy())[
                : self.owned_global_index[self._field_dim.get(name, dims.CellDim)].size
            ]
            for name in PROGNOSTIC_FIELDS
        }
