# Cluster runs

Example SLURM scripts for large jobs: training the
dense-grid FAX model, and line-by-line fluxes on that grid.

Submit from the repository root. The scripts need two variables to activate the
Python environment that has artsfax installed, and a `logs/` folder:

```bash
export CONDA_SH=/path/to/conda/etc/profile.d/conda.sh
export CONDA_ENV=<env with artsfax, pyarts3 and pyrte_rrtmgp>
mkdir -p logs
```

`ARTSFAX_DATA` sets a different data directory.

## Training

```bash
sbatch --account=<project> --partition=<partition> tools/cluster/train.sh LW
```

Computes the ARTS reference unless cached, then fits. Further arguments go to
`examples/train_fgrid.py`.

## Fluxes

```bash
sbatch --account=<project> --partition=<partition> --array=0-38 \
    tools/cluster/rte_array.sh arts-highres rfmip
python tools/cluster/join_columns.py arts-highres rfmip
```

One array task per variant and block of 8 columns (`rfmip`: 0-38, `rce`: 0-7),
each running 8 processes of 16 threads. Columns already on disk are skipped.

Quick test on every 50th frequency:

```bash
STRIDE=50 sbatch --account=<project> --partition=<partition> --array=0 \
    --time=00:30:00 tools/cluster/rte_array.sh arts-highres rce
```
