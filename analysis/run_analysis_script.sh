#!/bin/bash
#SBATCH --job-name=analysis_py
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log

# run_analysis_script.sh
# Run one analysis/*.py script as a Slurm job in the analysis conda env.
# Submit from analysis/ (logs go to analysis/logs/); --time and --mem are set per
# script at submission.
#
# Usage:
#   sbatch --job-name=gene_motif_linking --time=3:00:00 --mem=64G \
#       run_analysis_script.sh 1.0.gene_motif_linking.py
#
# Rerun order: 1.0.gene_motif_linking.py -> 2.tf_enrichment_in_programs.py
#   -> 3.chrombpnet_figures.py (chain with --dependency=afterok:<jobid>)

set -euo pipefail

script="${1:?usage: sbatch run_analysis_script.sh <script.py>}"
analysis_env="/path/to/conda/envs/<analysis_env>"

cd "${SLURM_SUBMIT_DIR}"
if [[ ! -f "${script}" ]]; then
    echo "script not found: ${SLURM_SUBMIT_DIR}/${script}" >&2
    exit 1
fi

# pybedtools temp files: point TMPDIR to node-local scratch if your cluster has one;
# non-interactive plotting backend
export TMPDIR="${TMPDIR:-/tmp}"
export MPLBACKEND=Agg
export PATH="${analysis_env}/bin:${PATH}"

echo "[$(date)] ${script} on $(hostname), job ${SLURM_JOB_ID}"
python -u "${script}"
echo "[$(date)] done: ${script}"
