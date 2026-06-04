#!/bin/bash
#SBATCH --job-name=hits_to_bed
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --partition=normal,engreitz
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# =============================================================================
# 13.hits_to_bed.sh
# Purpose: Convert per-day FiNeMo hits to bed files with TF names for IGV.
#
# Input:
#   modisco_compendium_meta.tsv     – cluster -> TF annotation (step 12)
#   {finemo_unified_dir}/{day}_{peak_type}/hits.bed.gz  (step 12)
#
# Output (inside {compendium_dir}/bed/):
#   {day}_{peak_type}_hits.bed    – one file per day
#
# Usage:
#   sbatch 13.hits_to_bed.sh
# =============================================================================

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "${SCRIPT_DIR}/config.sh"

meta_tsv="${modisco_compiled_dir}/modisco_compendium_meta.tsv"
bed_dir="${compendium_dir}/bed"

if [[ ! -f "${meta_tsv}" ]]; then
    echo "ERROR: ${meta_tsv} not found. Run 11.motif_compendium.sh first." >&2
    exit 1
fi

echo "[$(date)] Converting hits to bed (TF names from annotation database)..."

python "${SCRIPT_DIR}/13.hits_to_bed.py" \
    --meta       "${meta_tsv}" \
    --hits-dir   "${finemo_unified_dir}" \
    --days       "${days[@]}" \
    --peak-type  "${peak_type}" \
    --out-dir    "${bed_dir}"

echo "[$(date)] Done. BED files in: ${bed_dir}"
