#!/bin/bash
#SBATCH --job-name=finemo_postprocess
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --partition=normal
#SBATCH --array=0-4
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# 11.postprocess_finemo.sh
# Purpose: Post-process Fi-NeMo hit calls per day:
#            1. finemo report        - compute per-motif instance-CWM vs input-CWM
#                                     similarity (cwm_similarity in motif_report.tsv)
#            2. finemo collapse-hits - applied to hits_unique.tsv (genomically
#                                     deduplicated); adds is_primary column to
#                                     mark the best non-overlapping hit per group
#
# Input  (from step 10, per day):
#   {finemo_unified_dir}/{day}_{peak_type}/hits_unique.tsv
#   {finemo_unified_dir}/{day}_{peak_type}/intermediate_inputs.npz
#
# Output (per day):
#   finemo_report/motif_report.tsv  - per-motif CWM similarity and stats (used by 0.2)
#
# Note: finemo collapse-hits is intentionally omitted. The algorithm has inverted
# selection logic (demotes higher-similarity hits), so is_primary is not a useful
# filter. Downstream filtering (0.2.finemo_hit_qc.py) uses hits_unique.tsv directly
# and applies cwm_similarity threshold from motif_report.tsv.
#
# Usage:
#   sbatch 11.postprocess_finemo.sh            # all days
#   sbatch --array=0 11.postprocess_finemo.sh  # day 0 only
#
# Prerequisites: 10.run_finemo_unified.sh must have completed.
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "${SCRIPT_DIR}/config.sh"

day="${days[${SLURM_ARRAY_TASK_ID}]}"
[[ -z "${day}" ]] && { echo "No day at array index ${SLURM_ARRAY_TASK_ID}, exiting."; exit 0; }

out_dir="${finemo_unified_dir}/${day}_${peak_type}"
hits_tsv="${out_dir}/hits.tsv"
finemo_npz="${out_dir}/intermediate_inputs.npz"
report_dir="${out_dir}/finemo_report"

if [[ -f "${report_dir}/motif_report.tsv" ]]; then
    echo "[${day}] Already post-processed, skipping."
    exit 0
fi

if [[ ! -f "${hits_tsv}" ]]; then
    echo "ERROR: ${hits_tsv} not found. Run 11.run_finemo_unified.sh first." >&2
    exit 1
fi

if [[ ! -f "${finemo_npz}" ]]; then
    echo "ERROR: ${finemo_npz} not found. Run 11.run_finemo_unified.sh first." >&2
    exit 1
fi

ml biology samtools

source "${CONDA_INIT}"
conda activate "${finemo_conda}"

# Compute cwm_similarity for each motif by comparing
# the average CWM reconstructed from instances against the input CWM
echo "[$(date)] [${day}] Running finemo report..."
finemo report \
    -r "${finemo_npz}" \
    -H "${out_dir}" \
    -o "${report_dir}"

motif_report="${report_dir}/motif_report.tsv"
if [[ ! -f "${motif_report}" ]]; then
    echo "ERROR: ${motif_report} not produced. Check finemo report output above." >&2
    exit 1
fi
echo "[$(date)] [${day}] Report done: ${motif_report}"
