#!/bin/bash
# Reference and training of the dense-grid FAX model for one band. See README.md.
#
#   sbatch --account=<project> --partition=<partition> tools/cluster/train.sh LW
#
# Further arguments go to examples/train_fgrid.py.

#SBATCH --job-name=fax-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.log

set -eo pipefail

BAND=${1:?usage: sbatch tools/cluster/train.sh LW|SW [train_fgrid.py options]}
shift

source "${CONDA_SH:?set CONDA_SH to the conda.sh that activates your environment}"
conda activate "${CONDA_ENV:?set CONDA_ENV to the environment with artsfax installed}"
cd "$SLURM_SUBMIT_DIR"
echo "training $BAND on $(hostname) with $SLURM_CPUS_PER_TASK cores"

# 800 frequencies per chunk gives every core a chunk on the 1e5-point grid
python examples/train_fgrid.py --bands "$BAND" --workers "$SLURM_CPUS_PER_TASK" \
    --frequency-chunk 800 "$@"
