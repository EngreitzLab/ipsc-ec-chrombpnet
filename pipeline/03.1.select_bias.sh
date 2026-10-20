#!/bin/bash
#SBATCH --job-name=select_bias
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=1:00:00
#SBATCH --partition=normal
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# 03.1.select_bias.sh
# Purpose: Pick the best bias-factor per fold from the fast QC metrics
#   (evaluation/*_bias_metrics.json) that 03.0.train_bias_model.sh computed
#   for all fold x bias-factor combos. Update fold_bias_suffix in
#   dataset_config.sh with the result, then run 03.2.qc_selected_bias.sh on
#   just the selected combos before 04.0.train_full_model.sh.

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "${SCRIPT_DIR}/config.sh"

source "${CONDA_INIT}"
conda activate "${CONDA_ENV}"

python "${SCRIPT_DIR}/select_bias_model.py" \
    --core-path "${core_path}" \
    --biases "${bias_suffixes_sweep[@]#_}" \
    --folds "${folds[@]}" \
    --day "${bias_day}" \
    --peak-type "${peak_type}"


