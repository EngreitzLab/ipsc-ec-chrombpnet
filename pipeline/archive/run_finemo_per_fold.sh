#!/bin/bash
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu,owners
#SBATCH --job-name=finemo
#SBATCH --array=0-4

# =============================================================================
# 4.1-run_finemo.sh
# Purpose: Call motif hits (predictive instances) using Fi-NeMo GPU.
#          https://github.com/austintwang/finemo_gpu
#          One SLURM array job per fold; each job processes all days.
#
# Input per day/fold:
#   interpretation.counts_scores.h5   – DeepLIFT contribution scores (step 3.0)
#   modisco/modisco_counts_results.h5  – MoDISco motif patterns        (step 07)
#
# Output per day/fold (inside <finemo_dir>/<day>_<peak_type>_fold_<fold>/):
#   hits.bed.gz + hits.bed.gz.tbi  – tabix-indexed hit calls
#   hits.tsv                       – full hit table
#   finemo_report/                 – HTML report
#
# Usage:
#   sbatch 4.1-run_finemo.sh            # all folds (array 0-4)
#   sbatch --array=0 4.1-run_finemo.sh  # fold 0 only
#
# Prerequisites: 3.0-get_contrib_scores.sh and 3.1-run_modisco.sh must
#               have completed. Requires the 'finemo' conda environment.
# =============================================================================

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "${SCRIPT_DIR}/config.sh"

# ---------------------------------------------------------------------------
# Parameters – edit here; shared settings live in config.sh
# ---------------------------------------------------------------------------
# finemo_dir, finemo_alpha, finemo_conda are set in config.sh.
# ---------------------------------------------------------------------------

fold="${folds[${SLURM_ARRAY_TASK_ID}]}"
[[ -z "${fold}" ]] && { echo "No fold at array index ${SLURM_ARRAY_TASK_ID}, exiting."; exit 0; }

ml devel
ml system
ml cuda/11.5.0
ml cudnn/8.6.0.163
ml biology samtools

source "${CONDA_INIT}"
conda activate "${finemo_conda}"

export CUDA_VISIBLE_DEVICES=0
export TF_FORCE_GPU_ALLOW_GROWTH=true

echo "[$(date)] Fold ${fold}: running Fi-NeMo for days [${days[*]}]"

for day in "${days[@]}"; do
    interp_dir="${full_model_dir}/${day}_${peak_type}_fold_${fold}/interpretation"
    counts_h5="${interp_dir}/interpretation.counts_scores.h5"
    modisco_h5="${interp_dir}/modisco/modisco_counts_results.h5"
    peaks_file="${data_path}/${day}/${day}_${peak_type}_peaks_no_blacklist.narrowPeak"

    if [[ ! -f "${counts_h5}" ]]; then
        echo "  [${day} fold ${fold}] counts_scores.h5 not found, skipping." >&2
        echo "  Run 06.get_contrib_scores.sh first." >&2
        continue
    fi
    if [[ ! -f "${modisco_h5}" ]]; then
        echo "  [${day} fold ${fold}] modisco_counts_results.h5 not found, skipping." >&2
        echo "  Run 3.1.run_modisco.sh first." >&2
        continue
    fi

    out_dir="${finemo_dir}/${day}_${peak_type}_fold_${fold}"
    hits_file="${out_dir}/hits.bed.gz"

    if [[ -f "${hits_file}" ]]; then
        echo "  [${day} fold ${fold}] Hit calls already exist, skipping."
        continue
    fi

    finemo_npz="${out_dir}/intermediate_inputs.npz"
    report_dir="${out_dir}/finemo_report"
    mkdir -p "${report_dir}"

    echo "[$(date)] [${day} fold ${fold}] Extracting regions..."

    finemo extract-regions-chrombpnet-h5 \
        --h5s          "${counts_h5}" \
        --peaks        "${peaks_file}" \
        --out-path     "${finemo_npz}" \
        --region-width 1000

    echo "[$(date)] [${day} fold ${fold}] Calling hits..."

    finemo call-hits \
        -r "${finemo_npz}" \
        -m "${modisco_h5}" \
        -l "${finemo_alpha}" \
        -o "${out_dir}" \
        -b 200

    echo "[$(date)] [${day} fold ${fold}] Generating report..."

    finemo report \
        -r "${finemo_npz}" \
        -H "${out_dir}" \
        -m "${modisco_h5}" \
        -o "${report_dir}" \
        --no-recall

    if [[ -f "${out_dir}/hits.bed" ]]; then
        echo "[$(date)] [${day} fold ${fold}] Compressing and indexing hits..."
        bgzip -c "${out_dir}/hits.bed" > "${hits_file}"
        tabix -p bed "${hits_file}"
    fi

    echo "[$(date)] [${day} fold ${fold}] Fi-NeMo complete."
done

echo "[$(date)] Fold ${fold}: Fi-NeMo run complete."
