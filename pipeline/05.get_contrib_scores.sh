#!/bin/bash
#SBATCH --job-name=contribs
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=2-0
#SBATCH --partition=gpu
#SBATCH --array=0-24
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# 05.get_contrib_scores.sh
# Purpose: Compute DeepLIFT contribution scores for each day x fold using the
#   bias-corrected chrombpnet_nobias model. One SLURM array task per day x fold.
#   task_id = fold_idx * n_days + day_idx
#     tasks 0-4   = fold 0 (d0..d4)
#     tasks 5-9   = fold 1 (d0..d4)
#     tasks 10-14 = fold 2, etc.
#
# Outputs per day/fold (inside ${full_model_dir}/{day}_{peak_type}_fold_{fold}/interpretation/):
#   interpretation.counts_scores.h5 / .bw
#
# Usage:
#   sbatch 05.get_contrib_scores.sh            # all folds x days (array 0-24)
#   sbatch --array=0 05.get_contrib_scores.sh  # fold 0, d0 only
#
# Prerequisites: 04.0.train_full_model.sh must have completed.

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "${SCRIPT_DIR}/config.sh"

n_days=${#days[@]}
fold_idx=$(( SLURM_ARRAY_TASK_ID / n_days ))
day_idx=$(( SLURM_ARRAY_TASK_ID % n_days ))
fold="${folds[$fold_idx]}"
day="${days[$day_idx]}"
[[ -z "${fold}" || -z "${day}" ]] && { echo "Invalid array index ${SLURM_ARRAY_TASK_ID}, exiting."; exit 0; }

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

model_file="${full_model_dir}/${day}_${peak_type}_fold_${fold}/models/chrombpnet_nobias.h5"

if [[ ! -f "${model_file}" ]]; then
    echo "ERROR: Model not found for ${day} fold ${fold}: ${model_file}" >&2
    echo "  Run 04.0.train_full_model.sh first." >&2
    exit 1
fi

interp_dir="${full_model_dir}/${day}_${peak_type}_fold_${fold}/interpretation"
peaks_file="${data_path}/${day}/${day}_${peak_type}_peaks_no_blacklist.narrowPeak"
done_file_h5="${interp_dir}/interpretation.counts_scores.h5"
done_file_bw="${interp_dir}/interpretation.counts_scores.bw"

echo "[$(date)] [${day} fold ${fold}] Computing contribution scores"
echo "  model : ${model_file}"
echo "  output: ${interp_dir}/"

if [[ -f "${done_file_h5}" && -f "${done_file_bw}" ]]; then
    echo "  Already done, skipping."
    exit 0
fi

for f in "${model_file}" "${peaks_file}"; do
    [[ -f "${f}" ]] || { echo "  Missing input: ${f}" >&2; exit 1; }
done

mkdir -p "${interp_dir}"

chrombpnet contribs_bw \
    -m "${model_file}" \
    -r "${peaks_file}" \
    -g "${genome_fa}" \
    -c "${chrom_sizes}" \
    -op "${interp_dir}/interpretation"

echo "[$(date)] [${day} fold ${fold}] Done."
