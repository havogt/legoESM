#!/bin/bash -l
# icon4py's JAX model (R02B04, experiment EXPERIMENT) through the legoESM adapter for STEPS steps on all ranks,
# sampled every DIAGNOSTICS_EVERY steps (finiteness, max |vn|, total mass, time per step):
#   sbatch --nodes=2 --ntasks-per-node=4 --gpus-per-node=4 icon4py_jax_long.sbatch
# Environment: VENV (built in this uenv with --extra icon --extra icon-mpi), OUT, STEPS,
# DIAGNOSTICS_EVERY, ICON4PY_TEST_DATA_PATH, EXPERIMENT (directory under ser_icondata/,
# e.g. mpitask1_exclaim_nh35_tri_jws_v12 or mpitask1_exclaim_ape_aesPhys_v14).
#SBATCH --job-name=icon4py-jax-long
#SBATCH --account=csstaff
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --uenv=icon/26.7:v1
#SBATCH --view=default
set -euo pipefail
: "${VENV:?}" "${OUT:?}" "${STEPS:?}" "${DIAGNOSTICS_EVERY:?}" "${ICON4PY_TEST_DATA_PATH:?}" "${EXPERIMENT:?}"
EXP=$ICON4PY_TEST_DATA_PATH/ser_icondata/$EXPERIMENT
GRID=$ICON4PY_TEST_DATA_PATH/grids/icon_grid_0013_R02B04_R/icon_grid_0013_R02B04_R.nc
export XLA_PYTHON_CLIENT_PREALLOCATE=false MPICH_MAX_THREAD_SAFETY=multiple MPICH_GPU_SUPPORT_ENABLED=1
export NCCL_DEBUG=WARN
mkdir -p "$OUT"
echo "== start $(date -Is)"
srun -n "$SLURM_NTASKS" --cpus-per-task=72 bash -c \
  "export CUDA_VISIBLE_DEVICES=\$SLURM_LOCALID; exec $VENV/bin/python \
   -m legoesm.atmosphere.dynamics.gcm.icon4py_jax_spmd run --mpi --mode adapter --label long \
   --grid $GRID --experiment $EXP --steps $STEPS --out $OUT --diagnostics-every $DIAGNOSTICS_EVERY"
echo "== end $(date -Is)"
