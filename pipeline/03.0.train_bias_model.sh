#!/bin/bash
#SBATCH --job-name=bias_sweep
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=2-0
#SBATCH --partition=gpu
#SBATCH --array=0-29
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# 03.0.train_bias_model.sh
# Purpose: Train Tn5 bias models for a sweep of bias threshold factors on
#   bias_day (defined in dataset_config.sh), then compute fast QC metrics
#   (counts/profile Pearson r, JSD) for each - the metrics select_bias_model.py
#   (03.1) needs to pick a winner per fold. Deliberately skips the expensive
#   interpretation + TF-MoDISco QC (that only runs on the selected bias model,
#   in 03.2.qc_selected_bias.sh) - running it on all 25 combos is what made
#   this step slow before.
#
#   Array index maps to fold x bias_factor: task_id = fold_idx * n_factors + factor_idx
#     tasks 0-5   = fold 0 (bias 0.5, 0.6, 0.7, 0.8, 0.9, 1)
#     tasks 6-11  = fold 1, etc.
#
# Usage:
#   sbatch 03.0.train_bias_model.sh            # all folds x factors (array 0-29)
#   sbatch --array=0 03.0.train_bias_model.sh  # fold 0, bias 0.5 only (quick test)
#
# After all jobs complete, run 03.1.select_bias.sh, then 03.2.qc_selected_bias.sh,
# then 04.0.train_full_model.sh.

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "${SCRIPT_DIR}/config.sh"

n_factors=${#bias_factors[@]}
fold_idx=$(( SLURM_ARRAY_TASK_ID / n_factors ))
factor_idx=$(( SLURM_ARRAY_TASK_ID % n_factors ))
fold="${folds[$fold_idx]}"
bf="${bias_factors[$factor_idx]}"
suffix="${bias_suffixes_sweep[$factor_idx]}"
[[ -z "${fold}" || -z "${bf}" ]] && { echo "Invalid array index ${SLURM_ARRAY_TASK_ID}, exiting."; exit 0; }

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

peaks_file="${data_path}/${bias_day}/${bias_day}_${peak_type}_peaks_no_blacklist.narrowPeak"
negatives_file="${data_path}/${bias_day}/output_${peak_type}_fold_${fold}_negatives.bed"
fold_json="${folds_dir}/fold_${fold}.json"
file_prefix="${bias_day}_${peak_type}_fold_${fold}"
out_dir="${results_path}/bias_models/bias_model${suffix}/${bias_day}_${peak_type}_fold_${fold}"
model_file="${out_dir}/models/${file_prefix}_bias.h5"

echo "[$(date)] [fold ${fold} bias=${bf}] Training bias model"
echo "  output dir : ${out_dir}"

if [[ -f "${model_file}" ]]; then
    echo "  Model already trained, skipping training."
else
    for f in "${frag_d0}" "${peaks_file}" "${negatives_file}" "${fold_json}"; do
        [[ -f "${f}" ]] || { echo "  Missing input: ${f}" >&2; exit 1; }
    done

    rm -rf "${out_dir}"
    mkdir -p "${out_dir}"

    chrombpnet bias train \
        -ifrag "${frag_d0}" \
        -d "ATAC" \
        -g "${genome_fa}" \
        -c "${chrom_sizes}" \
        -p "${peaks_file}" \
        -n "${negatives_file}" \
        -fl "${fold_json}" \
        -b "${bf}" \
        -o "${out_dir}" \
        -fp "${file_prefix}"

    if [[ $? -ne 0 || ! -f "${model_file}" ]]; then
        echo "ERROR: chrombpnet bias train failed for fold ${fold} bias=${bf} (bias threshold factor may be too low/high for this fold - see stdout above)." >&2
        exit 1
    fi
fi

echo "[$(date)] [fold ${fold} bias=${bf}] Computing fast QC metrics."

python "${SCRIPT_DIR}/predict_bias_metrics.py" \
    --bias-model "${model_file}" \
    --output-dir "${out_dir}" \
    --file-prefix "${file_prefix}" \
    --genome "${genome_fa}" \
    --fold-json "${fold_json}"

echo "[$(date)] [fold ${fold} bias=${bf}] Done."
