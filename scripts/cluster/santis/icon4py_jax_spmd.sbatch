#!/bin/bash -l
# icon4py's JAX model (R02B04, experiment EXPERIMENT) on all ranks: legoESM adapter vs icon4py's own SPMD driver,
# a 1-rank adapter run, then the comparison. The layout comes from the sbatch command line:
#   sbatch --nodes=2 --ntasks-per-node=1 --gpus-per-node=1 icon4py_jax_spmd.sbatch
#   sbatch --nodes=2 --ntasks-per-node=4 --gpus-per-node=4 icon4py_jax_spmd.sbatch
# Environment: VENV (built in this uenv with --extra icon --extra icon-mpi), OUT, STEPS,
# ICON4PY_TEST_DATA_PATH, EXPERIMENT (directory under ser_icondata/,
# e.g. mpitask1_exclaim_nh35_tri_jws_v12 or mpitask1_exclaim_ape_aesPhys_v14).
#SBATCH --job-name=icon4py-jax-spmd
#SBATCH --account=csstaff
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --uenv=icon/26.7:v1
#SBATCH --view=default
set -euo pipefail
: "${VENV:?}" "${OUT:?}" "${STEPS:?}" "${ICON4PY_TEST_DATA_PATH:?}" "${EXPERIMENT:?}"
EXP=$ICON4PY_TEST_DATA_PATH/ser_icondata/$EXPERIMENT
GRID=$ICON4PY_TEST_DATA_PATH/grids/icon_grid_0013_R02B04_R/icon_grid_0013_R02B04_R.nc
RUNNER="$VENV/bin/python -m legoesm.atmosphere.dynamics.gcm.icon4py_jax_spmd"
export XLA_PYTHON_CLIENT_PREALLOCATE=false MPICH_MAX_THREAD_SAFETY=multiple MPICH_GPU_SUPPORT_ENABLED=1
export NCCL_DEBUG=WARN
mkdir -p "$OUT"

run() {  # ntasks mode label
  echo "== $3: $1 ranks, $2, start $(date -Is)"
  srun -n "$1" --cpus-per-task=72 bash -c \
    "export CUDA_VISIBLE_DEVICES=\$SLURM_LOCALID; exec $RUNNER run --mpi --mode $2 --label $3 \
     --grid $GRID --experiment $EXP --steps $STEPS --out $OUT"
  echo "== $3: end $(date -Is)"
}

run "$SLURM_NTASKS" reference reference
run "$SLURM_NTASKS" adapter adapter
run 1 adapter adapter1
status=0
$RUNNER compare --out "$OUT" --experiment "$EXP" --steps "$STEPS" \
  --labels adapter reference --bitwise --max-diff-against adapter1 || status=1
$RUNNER compare --out "$OUT" --experiment "$EXP" --steps "$STEPS" --labels adapter1 || status=1
exit $status
