#!/bin/bash
#SBATCH --job-name=finemo_per_day
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu,owners
#SBATCH --array=0-4
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# _11.run_finemo.sh  (legacy, not part of main pipeline)
# Purpose: Call motif hits per day using fold-averaged contribution scores
#          and the per-day MoDISco motif set. Kept for reference; use
#          12.run_finemo_unified.sh for cross-day comparable hit calls.
#
# Input per day:
#   average_shaps.counts.h5        – fold-averaged DeepLIFT scores (step 07)
#   modisco_counts_results.h5      – per-day MoDISco patterns (step 09)
#
# Output (inside ${finemo_dir}/{day}_{peak_type}/):
#   hits.bed.gz + hits.bed.gz.tbi  – tabix-indexed hit calls
#   hits.tsv                       – full hit table
#   finemo_report/                 – HTML report
#
# Usage:
#   sbatch _11.run_finemo.sh            # all days (array 0-4)
#   sbatch --array=0 _11.run_finemo.sh  # day 0 only
#
# Prerequisites: 07.average_contrib_scores.sh and 09.run_modisco.sh must have completed.

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "${SCRIPT_DIR}/config.sh"

day="${days[${SLURM_ARRAY_TASK_ID}]}"
[[ -z "${day}" ]] && { echo "No day at array index ${SLURM_ARRAY_TASK_ID}, exiting."; exit 0; }

counts_h5="${averaged_dir}/${day}/${day}_average_shaps.counts.h5"
modisco_h5="${averaged_dir}/${day}/modisco/modisco_counts_results.h5"
peaks_file="${data_path}/${day}/${day}_${peak_type}_peaks_no_blacklist.narrowPeak"

if [[ ! -f "${counts_h5}" ]]; then
    echo "[${day}] Averaged counts H5 not found: ${counts_h5}" >&2
    echo "  Run 07.average_contrib_scores.sh first." >&2
    exit 1
fi

if [[ ! -f "${modisco_h5}" ]]; then
    echo "[${day}] Per-day MoDISco H5 not found: ${modisco_h5}" >&2
    echo "  Run 09.run_modisco.sh first." >&2
    exit 1
fi

ml devel
ml system
ml cuda/11.5.0
ml cudnn/8.6.0.163
ml biology samtools

source "${CONDA_INIT}"
conda activate "${finemo_conda}"

export CUDA_VISIBLE_DEVICES=0
export TF_FORCE_GPU_ALLOW_GROWTH=true

out_dir="${finemo_dir}/${day}_${peak_type}"
hits_file="${out_dir}/${day}_hits.bed.gz"

if [[ -f "${hits_file}" ]]; then
    echo "[${day}] Hit calls already exist, skipping."
    exit 0
fi

finemo_npz="${out_dir}/intermediate_inputs.npz"
report_dir="${out_dir}/finemo_report"
mkdir -p "${report_dir}"

echo "[$(date)] [${day}] Extracting regions from averaged H5..."

finemo extract-regions-chrombpnet-h5 \
    --h5s          "${counts_h5}" \
    --peaks        "${peaks_file}" \
    --out-path     "${finemo_npz}" \
    --region-width 1000

echo "[$(date)] [${day}] Calling hits (per-day modisco)..."

finemo call-hits \
    -r "${finemo_npz}" \
    -m "${modisco_h5}" \
    -l "${finemo_alpha}" \
    -o "${out_dir}" \
    -b 200

echo "[$(date)] [${day}] Generating report..."

finemo report \
    -r "${finemo_npz}" \
    -H "${out_dir}" \
    -m "${modisco_h5}" \
    -o "${report_dir}" \
    --no-recall

if [[ -f "${out_dir}/${day}_hits.bed" ]]; then
    echo "[$(date)] [${day}] Compressing and indexing hits..."
    bgzip -c "${out_dir}/${day}_hits.bed" > "${hits_file}"
    tabix -p bed "${hits_file}"
fi

echo "[$(date)] [${day}] Fi-NeMo (per-day) complete."
