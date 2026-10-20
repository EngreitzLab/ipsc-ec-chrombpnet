#!/bin/bash
#SBATCH --job-name=avg_contribs
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --partition=normal
#SBATCH --array=0-4
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# 06.average_contrib_scores.sh
# Purpose: Average DeepLIFT contribution scores across all 5 folds for
#   each day. One SLURM array job per day.
#
# Why: Each fold's model was trained on a different 80/20 data split, so
#   its contribution scores carry fold-specific noise. Averaging reduces
#   this noise and gives a more robust signal for motif discovery.
#   This follows the Greenleaf lab approach (their step 05-average_deepshaps).
#
# score_types below controls which contribution head(s) get (re-)averaged.
#   Counts was already run previously; it is left out here so reruns of
#   this script do not touch counts.h5. average_contrib_scores.py also
#   skips a score-type if its output already exists, so re-adding "counts"
#   to score_types later is safe and will not recompute it.
#
# Input:  ${full_model_dir}/{day}_{peak_type}_fold_{0..4}/interpretation/
#             interpretation.{score_type}_scores.h5  (step 05 output)
# Output: ${averaged_dir}/{day}/{day}_average_shaps.{score_type}.h5
#
# Usage:
#   sbatch 06.average_contrib_scores.sh            # all days (array 0-4)
#   sbatch --array=0 06.average_contrib_scores.sh  # day 0 only
#
# Prerequisites: 05.get_contrib_scores.sh must have completed for all folds.
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "${SCRIPT_DIR}/config.sh"

day="${days[${SLURM_ARRAY_TASK_ID}]}"
[[ -z "${day}" ]] && { echo "No day at array index ${SLURM_ARRAY_TASK_ID}, exiting."; exit 0; }

score_types=("profile")  # counts already done; add "counts" here to redo/extend

out_dir="${averaged_dir}/${day}"
mkdir -p "${out_dir}" "${log_dir}"

source "${CONDA_INIT}"
conda activate "${CONDA_ENV}"

for score_type in "${score_types[@]}"; do
    echo "[$(date)] [${day} ${score_type}] Averaging contribution scores..."
    python "${SCRIPT_DIR}/average_contrib_scores.py" \
        --day "${day}" \
        --folds "${folds[@]}" \
        --full-model-dir "${full_model_dir_selected}" \
        --peak-type "${peak_type}" \
        --score-type "${score_type}" \
        --out-dir "${out_dir}"
done

echo "Done"
