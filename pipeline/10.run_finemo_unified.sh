#!/bin/bash
#SBATCH --job-name=finemo_unified
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --array=0-4
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# 10.run_finemo_unified.sh
# Purpose: Call motif hits using the unified (compendium) MoDISco H5.
#          Uses modisco_compiled.h5 built in step 09 and fold-averaged
#          contribution scores (step 06) so that a single hit set per
#          day is produced, enabling direct cross-day comparisons.
#
#          One SLURM array job per day.
#
# Input per day:
#   {averaged_dir}/{day}/{day}_average_shaps.counts.h5  – averaged DeepLIFT scores (step 06)
#   modisco_compiled.h5                                 – unified MoDISco patterns (step 09)
#
# Output (inside finemo_unified_dir/{day}_{peak_type}/):
#   hits.bed.gz + hits.bed.gz.tbi     – tabix-indexed hit calls
#   hits.tsv                          – full hit table
#   finemo_report/                    – HTML report
#
# Usage:
#   sbatch 10.run_finemo_unified.sh            # all days
#   sbatch --array=0 10.run_finemo_unified.sh  # day 0 only
#
# Prerequisites: steps 08 and 09 must have completed.
#   Requires the 'finemo' conda environment.

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "${SCRIPT_DIR}/config.sh"

day="${days[${SLURM_ARRAY_TASK_ID}]}"
[[ -z "${day}" ]] && { echo "No day at array index ${SLURM_ARRAY_TASK_ID}, exiting."; exit 0; }

compiled_h5="${modisco_compiled_dir}/modisco_compiled.h5"
if [[ ! -f "${compiled_h5}" ]];
then
    echo "ERROR: ${compiled_h5} not found. Run 09.motif_compendium.sh first." >&2
    exit 1
fi

ml devel
ml gcc/10.1.0
ml system
ml cuda/11.5.0
ml cudnn/8.6.0.163
ml biology samtools

source "${CONDA_INIT}"
conda activate "${finemo_conda}"

export CUDA_VISIBLE_DEVICES=0
export TF_FORCE_GPU_ALLOW_GROWTH=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

counts_h5="${averaged_dir}/${day}/${day}_average_shaps.counts.h5"
# NOTE: chrombpnet contribs_bw (step 05) silently drops peaks whose input
# window would run past a chromosome boundary (e.g. peaks near the end of
# the short chrM contig). The averaged h5 (step 06) therefore has fewer
# regions than the raw peaks file. Use the retained-region bed that
# chrombpnet itself wrote out (identical across folds) so peaks_file lines
# up 1:1 with the h5 rows that finemo expects.
peaks_file="${full_model_dir_selected}/${day}_${peak_type}_fold_${folds[0]}/interpretation/interpretation.interpreted_regions.bed"

if [[ ! -f "${counts_h5}" ]]; then
    echo "ERROR: ${counts_h5} not found. Run 06.average_contrib_scores.sh first." >&2
    exit 1
fi

if [[ ! -f "${peaks_file}" ]]; then
    echo "ERROR: ${peaks_file} not found. Run 05.get_contrib_scores.sh first." >&2
    exit 1
fi

out_dir="${finemo_unified_dir}/${day}_${peak_type}"
hits_file="${out_dir}/hits.bed.gz"

if [[ -f "${hits_file}" ]]; then
    echo "[${day}] Hit calls already exist, skipping."
    exit 0
fi

finemo_npz="${out_dir}/intermediate_inputs.npz"
mkdir -p "${out_dir}"

# echo "[$(date)] [${day}] Extracting regions from averaged scores..."

finemo extract-regions-chrombpnet-h5 \
    --h5s "${counts_h5}" \
    --peaks "${peaks_file}" \
    --out-path "${finemo_npz}" \
    --region-width 1000

if [[ ! -f "${finemo_npz}" ]]; then
    echo "ERROR: [${day}] extract-regions failed, ${finemo_npz} not produced." >&2
    exit 1
fi

echo "[$(date)] [${day}] Calling hits (unified modisco)..."

finemo call-hits \
    -r "${finemo_npz}" \
    -m "${compiled_h5}" \
    -l "${finemo_alpha}" \
    -o "${out_dir}" \
    -b 500

if [[ ! -f "${out_dir}/hits.bed" ]]; then
    echo "ERROR: [${day}] call-hits failed, ${out_dir}/hits.bed not produced." >&2
    exit 1
fi

echo "[$(date)] [${day}] Compressing and indexing hits..."
bgzip -c "${out_dir}/hits.bed" > "${hits_file}"
tabix -p bed "${hits_file}"

echo "[$(date)] [${day}] Fi-NeMo (unified) complete."
echo "  Hits: ${hits_file}"
