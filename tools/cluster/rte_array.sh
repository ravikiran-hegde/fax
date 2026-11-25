#!/bin/bash
# Dense-grid fluxes as a SLURM array, one task per variant and block of 8
# columns. See README.md.
#
#   sbatch --account=<project> --partition=<partition> --array=0-38 \
#       tools/cluster/rte_array.sh arts-highres rfmip
#
# Array indices: rfmip 0-38 (variants 0 1 2), rce 0-7 (variants 0 1).
# STRIDE=n uses every n-th frequency.

#SBATCH --job-name=fax-rte
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.log

set -eo pipefail

usage="usage: sbatch --array=... tools/cluster/rte_array.sh fax-highres|arts-highres rfmip|rce"
BACKEND=${1:?$usage}
CASE=${2:?$usage}
STRIDE=${STRIDE:-1}
WORKERS=8
THREADS=$(( SLURM_CPUS_PER_TASK / WORKERS ))

case $CASE in
    rfmip) VARIANTS=(0 1 2); N_COLUMNS=100 ;;
    rce)   VARIANTS=(0 1);   N_COLUMNS=32 ;;
    *) echo "unknown case $CASE"; exit 1 ;;
esac
N_BLOCKS=$(( (N_COLUMNS + WORKERS - 1) / WORKERS ))
VARIANT=${VARIANTS[$(( SLURM_ARRAY_TASK_ID / N_BLOCKS ))]}
FIRST=$(( (SLURM_ARRAY_TASK_ID % N_BLOCKS) * WORKERS ))

source "${CONDA_SH:?set CONDA_SH to the conda.sh that activates your environment}"
conda activate "${CONDA_ENV:?set CONDA_ENV to the environment with faxsec installed}"
cd "$SLURM_SUBMIT_DIR"
OUT_DIR=$(python -c "from faxsec.constants import DATA_DIR; print(DATA_DIR / 'rte_examples')")
echo "$BACKEND $CASE variant $VARIANT, columns from $FIRST, $WORKERS x $THREADS threads on $(hostname)"

STRIDE_TAG=""
[[ $STRIDE -gt 1 ]] && STRIDE_TAG="_s$STRIDE"
pids=()
for (( c = FIRST; c < FIRST + WORKERS && c < N_COLUMNS; c++ )); do
    out=$OUT_DIR/fluxes_${BACKEND}_${CASE}_v$(printf %03d "$VARIANT")_c${c}-$((c + 1))${STRIDE_TAG}.nc
    if [[ -f $out ]]; then
        echo "$out exists, skipping"
        continue
    fi
    OMP_NUM_THREADS=$THREADS python examples/rte_example.py --gas-optics "$BACKEND" \
        --case "$CASE" --variant "$VARIANT" --columns "$c:$((c + 1))" \
        --frequency-chunk 5000 --stride "$STRIDE" \
        > "logs/${BACKEND}_${CASE}_v${VARIANT}_c${c}.log" 2>&1 &
    pids+=($!)
done
# fail the task if any column failed
failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || failed=$(( failed + 1 ))
done
echo "done, $failed of ${#pids[@]} columns failed"
exit $(( failed > 0 ))
