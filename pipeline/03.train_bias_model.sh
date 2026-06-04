#!/bin/bash
#SBATCH --job-name=bias_sweep
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --time=2-0
#SBATCH --partition=gpu
#SBATCH --array=0-4
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# 03.train_bias_model.sh
# Purpose: Train Tn5 bias models for a sweep of bias threshold factors on
#          bias_day (defined in config.sh). One SLURM array job per fold;
#          each job trains all bias_factors sequentially for that fold.
#
# The resulting bias models are evaluated with qc_bias_selection.py, then the
# best factor per fold is recorded in fold_bias_suffix in config.sh.
#
# Usage:
#   sbatch 03.train_bias_model.sh            # all folds (array 0-4)
#   sbatch --array=0 03.train_bias_model.sh  # fold 0 only (quick test)
#
# After all jobs complete, run 04.select_bias.sh then 05.train_full_model.sh.

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "${SCRIPT_DIR}/config.sh"

fold="${folds[${SLURM_ARRAY_TASK_ID}]}"
[[ -z "${fold}" ]] && { echo "No fold at array index ${SLURM_ARRAY_TASK_ID}, exiting."; exit 0; }

ml devel
ml system
ml cairo
ml pango/1.40.10
ml cuda/11.5.0
ml cudnn/8.6.0.163

source "${CONDA_INIT}"
conda activate "${CONDA_ENV}"

export CUDA_VISIBLE_DEVICES=0,1
export TF_FORCE_GPU_ALLOW_GROWTH=true

bam_file="${bam_path}/${bias_day}_main_chrs.bam"
peaks_file="${data_path}/${bias_day}/${bias_day}_${peak_type}_peaks_no_blacklist.narrowPeak"
negatives_file="${data_path}/${bias_day}/output_${peak_type}_fold_${fold}_negatives.bed"
fold_json="${folds_dir}/fold_${fold}.json"
file_prefix="${bias_day}_${peak_type}_fold_${fold}"

echo "[$(date)] Fold ${fold}: starting bias factor sweep: ${bias_factors[*]}"

for bf in "${bias_factors[@]}"; do
    # Derive suffix: 0.8 -> _08, 0.5 -> _05, 0.65 -> _065, etc.
    suffix="_$(echo "${bf}" | tr -d '.')"
    out_dir="${results_path}/bias_models/bias_model${suffix}/${bias_day}_${peak_type}_fold_${fold}"
    model_file="${out_dir}/models/${file_prefix}_bias.h5"

    if [[ -f "${model_file}" ]]; then
        echo "  [bias=${bf}] Already done, skipping: ${model_file}"
        continue
    fi

    mkdir -p "${out_dir}"
    echo "[$(date)] [fold ${fold} bias=${bf}] Training..."

    chrombpnet bias pipeline \
        -ibam "${bam_file}" \
        -d "ATAC" \
        -g "${genome_fa}" \
        -c "${chrom_sizes}" \
        -p "${peaks_file}" \
        -n "${negatives_file}" \
        -fl "${fold_json}" \
        -b "${bf}" \
        -o "${out_dir}" \
        -fp "${file_prefix}"

    echo "[$(date)] [fold ${fold} bias=${bf}] Done."
done

echo "[$(date)] Fold ${fold}: bias sweep complete."
