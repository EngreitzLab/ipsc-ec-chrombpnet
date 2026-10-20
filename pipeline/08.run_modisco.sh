#!/bin/bash
#SBATCH --job-name=modisco_avg
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --time=4-0
#SBATCH --partition=normal
#SBATCH --array=0-4
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# 08.run_modisco.sh
# Purpose: Run TF-MoDISco on fold-averaged contribution scores (step 07).
#          One SLURM array job per day; produces ONE modisco result per day
#          per score type (rather than one per fold), which is then used for
#          the unified motif compendium (step 09 / 09.1).
#
# Why run on averaged scores:
#   Averaging contribution scores across folds before MoDISco improves
#   signal-to-noise ratio, so the discovered patterns are more reproducible
#   and biologically meaningful. This is the standard approach in the
#   Greenleaf lab ChromBPNet pipeline.
#
# score_types below: only "profile" is (re-)run here since counts modisco
# results already exist; add "counts" back to redo/extend.
#
# Partition/GPU/QOS: modisco motifs (tfmodisco-lite) is CPU-only, so this
# runs on a CPU partition (no GPU request). Per-day runtime varies a lot
# (d1/d2/d3 took 1.4-1.9 days; d0/d4 exceeded 2 days), so request a
# partition/QOS that allows at least 4 days of walltime (add a --qos line
# if your cluster's default QOS caps walltime below --time).
#
# Input:  results/contrib_scores/{day}/{day}_average_shaps.{score_type}.h5  (step 06/07)
# Output: ${averaged_dir}/{day}/modisco/
#             modisco_{score_type}_results.h5
#             {score_type}_report/
#
# Usage:
#   sbatch 08.run_modisco.sh            # all days (array 0-4)
#   sbatch --array=0 08.run_modisco.sh  # d0 only
#
# Prerequisites: 06.average_contrib_scores.sh must have completed.

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "${SCRIPT_DIR}/config.sh"

day="${days[${SLURM_ARRAY_TASK_ID}]}"
[[ -z "${day}" ]] && { echo "No day at array index ${SLURM_ARRAY_TASK_ID}, exiting."; exit 0; }

score_types=("profile")  # counts modisco already done; add "counts" here to redo/extend

mkdir -p "${log_dir}"
for score_type in "${score_types[@]}"; do
    mkdir -p "${averaged_dir}/${day}/modisco/${score_type}_report"
done

ml devel
ml system
ml cairo
ml pango/1.40.10

source "${CONDA_INIT}"
conda activate "${CONDA_ENV}"

for score_type in "${score_types[@]}"; do
    echo "[$(date)] Day ${day}: running MoDISco on averaged ${score_type} scores..."

    modisco motifs \
        -i "${averaged_dir}/${day}/${day}_average_shaps.${score_type}.h5" \
        -n 500000 \
        -o "${averaged_dir}/${day}/modisco/modisco_${score_type}_results.h5" \
        -w 500 \
        -v

    # modisco report \
    #     -i "${averaged_dir}/${day}/modisco/modisco_${score_type}_results.h5" \
    #     -o "${averaged_dir}/${day}/modisco/${score_type}_report" \
    #     -s "${averaged_dir}/${day}/modisco/${score_type}_report" \
    #     -m "${ref_db_meme}"

    echo "[$(date)] Day ${day}: ${score_type} MoDISco complete"
done
