#!/bin/bash
#SBATCH --job-name=finemo_postprocess_profile
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --partition=normal
#SBATCH --array=0-4
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# 11.1.postprocess_finemo_profile.sh
# Purpose: Post-process Fi-NeMo profile-score hit calls per day. Mirrors
#          11.postprocess_finemo.sh exactly, but points at the profile
#          output dir from step 10.1 so counts post-processing is untouched.
#            1. finemo report        - compute per-motif instance-CWM vs input-CWM
#                                     similarity (cwm_similarity in motif_report.tsv)
#
# Input  (from step 10.1, per day):
#   {finemo_unified_dir}/{day}_{peak_type}_profile/hits_unique.tsv
#   {finemo_unified_dir}/{day}_{peak_type}_profile/intermediate_inputs.npz
#
# Output (per day):
#   finemo_report/motif_report.tsv  - per-motif CWM similarity and stats
#
# Note: finemo collapse-hits is intentionally omitted, same as step 11
# (inverted selection logic - see that script's note).
#
# Usage:
#   sbatch 11.1.postprocess_finemo_profile.sh            # all days
#   sbatch --array=0 11.1.postprocess_finemo_profile.sh  # day 0 only
#
# Prerequisites: 10.1.run_finemo_unified_profile.sh must have completed.
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "${SCRIPT_DIR}/config.sh"

day="${days[${SLURM_ARRAY_TASK_ID}]}"
[[ -z "${day}" ]] && { echo "No day at array index ${SLURM_ARRAY_TASK_ID}, exiting."; exit 0; }

out_dir="${finemo_unified_dir}/${day}_${peak_type}_profile"
hits_tsv="${out_dir}/hits.tsv"
finemo_npz="${out_dir}/intermediate_inputs.npz"
report_dir="${out_dir}/finemo_report"

if [[ -f "${report_dir}/motif_report.tsv" ]]; then
    echo "[${day}] Already post-processed, skipping."
    exit 0
fi

if [[ ! -f "${hits_tsv}" ]]; then
    echo "ERROR: ${hits_tsv} not found. Run 10.1.run_finemo_unified_profile.sh first." >&2
    exit 1
fi

if [[ ! -f "${finemo_npz}" ]]; then
    echo "ERROR: ${finemo_npz} not found. Run 10.1.run_finemo_unified_profile.sh first." >&2
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
