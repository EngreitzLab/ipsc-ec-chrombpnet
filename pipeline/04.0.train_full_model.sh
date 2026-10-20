#!/bin/bash
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=2-0
#SBATCH --partition=gpu
#SBATCH --job-name=full_model
#SBATCH --array=0-24

# 04.0.train_full_model.sh
# Purpose: Train the bias-factorised ChromBPNet full model for all days and
#   folds, using the per-fold optimal bias model selected in step 03
#   (03.1.select_bias.sh / select_bias_model.py). Each fold uses the bias
#   suffix recorded in fold_bias_suffix in dataset_config.sh.
#
# Array index maps to fold x day: task_id = fold_idx * n_days + day_idx
#   tasks 0-4  = fold 0 (d0..d4)
#   tasks 5-9  = fold 1 (d0..d4)
#   tasks 10-14 = fold 2, etc.
#
# Output directory: ${full_model_dir_selected} (set in config.sh)
#
# Usage:
#   sbatch 04.0.train_full_model.sh            # all folds x days (array 0-24)
#   sbatch --array=0 04.0.train_full_model.sh  # fold 0, d0 only (quick test)
#
# Prerequisites: 03.0.train_bias_model.sh, 03.1.select_bias.sh, and
#   03.2.qc_selected_bias.sh must have completed for all folds. Bias suffixes
#   must be set in dataset_config.sh.

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "${SCRIPT_DIR}/config.sh"

n_days=${#days[@]}
fold_idx=$(( SLURM_ARRAY_TASK_ID / n_days ))
day_idx=$(( SLURM_ARRAY_TASK_ID % n_days ))
fold="${folds[$fold_idx]}"
day="${days[$day_idx]}"
[[ -z "${fold}" || -z "${day}" ]] && { echo "Invalid array index ${SLURM_ARRAY_TASK_ID}, exiting."; exit 0; }

suffix="${fold_bias_suffix[${fold}]}"
if [[ -z "${suffix}" ]]; then
    echo "ERROR: No bias suffix defined for fold ${fold} in fold_bias_suffix (dataset_config.sh)." >&2
    exit 1
fi

bias_model="${results_path}/bias_models/bias_model${suffix}/${bias_day}_${peak_type}_fold_${fold}/models/${bias_day}_${peak_type}_fold_${fold}_bias.h5"

if [[ ! -f "${bias_model}" ]]; then
    echo "ERROR: Bias model not found for fold ${fold} (suffix ${suffix}):" >&2
    echo "  ${bias_model}" >&2
    echo "  Run 03.0.train_bias_model.sh first." >&2
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

out_dir="${full_model_dir_selected}/${day}_${peak_type}_fold_${fold}"
model_file="${out_dir}/models/chrombpnet_nobias.h5"
# Last file written by the pipeline's evaluation stage; absence means training
# finished (model_file exists) but evaluation was cut short, e.g. by preemption.
eval_marker="${out_dir}/evaluation/chrombpnet_nobias_profile.pdf"

frag_file="${frag_path}/${day}.fragments.tsv.gz"
peaks_file="${data_path}/${day}/${day}_${peak_type}_peaks_no_blacklist.narrowPeak"
negatives_file="${data_path}/${day}/output_${peak_type}_fold_${fold}_negatives.bed"
fold_json="${folds_dir}/fold_${fold}.json"

echo "[$(date)] [${day} fold ${fold}] Training full model (bias suffix ${suffix})"
echo "  bias model : ${bias_model}"
echo "  output dir : ${out_dir}"

if [[ -f "${model_file}" && -f "${eval_marker}" ]]; then
    echo "  Already done, skipping."
    exit 0
fi

for f in "${frag_file}" "${peaks_file}" "${negatives_file}" "${fold_json}"; do
    [[ -f "${f}" ]] || { echo "  Missing input: ${f}" >&2; exit 1; }
done

rm -rf "${out_dir}"
mkdir -p "${out_dir}"

echo "[$(date)] [${day} fold ${fold}] Training..."

chrombpnet pipeline \
    -ifrag "${frag_file}" \
    -d "ATAC" \
    -g "${genome_fa}" \
    -c "${chrom_sizes}" \
    -p "${peaks_file}" \
    -n "${negatives_file}" \
    -fl "${fold_json}" \
    -b "${bias_model}" \
    -o "${out_dir}"

echo "[$(date)] [${day} fold ${fold}] Done."
