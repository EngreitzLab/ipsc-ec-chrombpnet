#!/bin/bash
#SBATCH --job-name=avg_contribs_bw
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --time=2:00:00
#SBATCH --partition=normal
#SBATCH --array=0-4
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# 07.contribs_to_bigwig.sh
# Purpose: Convert the fold-averaged contribution score H5 (from step 06)
#          into a bigwig for each day. One SLURM array job per day.
#
# score_types below mirrors step 06: only "profile" is (re-)generated here
# since a counts bigwig already exists; add "counts" back to regenerate it.
#
# Input:  {averaged_dir}/{day}/{day}_average_shaps.{score_type}.h5  (step 06 output)
#         {full_model_dir}/{day}_all_fold_0/interpretation/
#             interpretation.interpreted_regions.bed              (any fold)
# Output: {averaged_dir}/{day}/{day}_average_shaps.{score_type}.bw
#
# Usage:
#   sbatch 07.contribs_to_bigwig.sh            # all days (array 0-4)
#   sbatch --array=0 07.contribs_to_bigwig.sh  # d0 only
#
# Prerequisites: 06.average_contrib_scores.sh must have completed.

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "${SCRIPT_DIR}/config.sh"

day="${days[${SLURM_ARRAY_TASK_ID}]}"
[[ -z "${day}" ]] && { echo "No day at array index ${SLURM_ARRAY_TASK_ID}, exiting."; exit 0; }

score_types=("profile")  # counts bigwig already exists; add "counts" here to redo/extend

source "${CONDA_INIT}"
conda activate "${CONDA_ENV}"

# Use fold 0's interpreted regions — identical across folds (same peaks input)
regions_file="${full_model_dir}/${day}_${peak_type}_fold_0/interpretation/interpretation.interpreted_regions.bed"

if [[ ! -f "${regions_file}" ]]; then
    echo "[${day}] Regions BED not found: ${regions_file}" >&2
    exit 1
fi

for score_type in "${score_types[@]}"; do
    h5_file="${averaged_dir}/${day}/${day}_average_shaps.${score_type}.h5"
    out_bw="${averaged_dir}/${day}/${day}_average_shaps.${score_type}.bw"

    if [[ ! -f "${h5_file}" ]]; then
        echo "[${day} ${score_type}] Averaged H5 not found: ${h5_file}" >&2
        echo "  Run 06.average_contrib_scores.sh first." >&2
        exit 1
    fi

    echo "[$(date)] [${day} ${score_type}] Writing averaged contribution score bigwig..."

    python3 "${SCRIPT_DIR}/contribs_to_bigwig.py" \
        --h5 "${h5_file}" \
        --regions "${regions_file}" \
        --chrom-sizes "${chrom_sizes_bigwig}" \
        --output-bw "${out_bw}"

    echo "[$(date)] [${day} ${score_type}] Done: ${out_bw}"
done
