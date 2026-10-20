#!/bin/bash
#SBATCH --job-name=qc_selected_bias
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=1-0
#SBATCH --partition=gpu
#SBATCH --array=0-4
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# 03.2.qc_selected_bias.sh
# Purpose: Full QC (marginal footprinting + DeepLIFT interpretation +
#   TF-MoDISco) on only the bias model selected per fold in
#   dataset_config.sh (fold_bias_suffix), populated by 03.1.select_bias.sh.
#   This is the expensive step that verifies the Tn5 signal has actually
#   been learned - inspect evaluation/*_bias_profile.pdf for Tn5 vs GC-rich
#   motifs. Deliberately not run on the full fold x bias-factor sweep (03.0).
#
# Array index = fold index (one task per fold, not per bias-factor).
#
# Usage:
#   sbatch 03.2.qc_selected_bias.sh            # all folds (array 0-4)
#   sbatch --array=0 03.2.qc_selected_bias.sh  # fold 0 only (quick test)
#
# Prerequisites: 03.0.train_bias_model.sh and 03.1.select_bias.sh must have
#   completed, with fold_bias_suffix set in dataset_config.sh.

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "${SCRIPT_DIR}/config.sh"

fold="${folds[$SLURM_ARRAY_TASK_ID]}"
[[ -z "${fold}" ]] && { echo "Invalid array index ${SLURM_ARRAY_TASK_ID}, exiting."; exit 0; }

suffix="${fold_bias_suffix[${fold}]}"
if [[ -z "${suffix}" ]]; then
    echo "ERROR: No bias suffix defined for fold ${fold} in fold_bias_suffix (dataset_config.sh)." >&2
    exit 1
fi

ml devel
ml system
ml cairo
ml pango/1.40.10
ml cuda/11.5.0
ml cudnn/8.6.0.163

source "${CONDA_INIT}"
conda activate "${CONDA_ENV}"

export CUDA_VISIBLE_DEVICES=0
export TF_FORCE_GPU_ALLOW_GROWTH=true

fold_json="${folds_dir}/fold_${fold}.json"
file_prefix="${bias_day}_${peak_type}_fold_${fold}"
out_dir="${results_path}/bias_models/bias_model${suffix}/${bias_day}_${peak_type}_fold_${fold}"
model_file="${out_dir}/models/${file_prefix}_bias.h5"

echo "[$(date)] [fold ${fold}] Full QC on selected bias model (suffix ${suffix})"
echo "  model      : ${model_file}"
echo "  output dir : ${out_dir}"

if [[ ! -f "${model_file}" ]]; then
    echo "ERROR: Bias model not found for fold ${fold} (suffix ${suffix}):" >&2
    echo "  ${model_file}" >&2
    echo "  Run 03.0.train_bias_model.sh first." >&2
    exit 1
fi

python "${SCRIPT_DIR}/run_bias_qc.py" \
    --bias-model "${model_file}" \
    --output-dir "${out_dir}" \
    --file-prefix "${file_prefix}" \
    --genome "${genome_fa}" \
    --chrom-sizes "${chrom_sizes}" \
    --fold-json "${fold_json}"

echo "[$(date)] [fold ${fold}] Done."
