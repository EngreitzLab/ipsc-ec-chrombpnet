#!/bin/bash
#SBATCH --job-name=hits_to_bed
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --partition=normal
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# 0.3.hits_to_bed.sh
# Purpose: Convert per-day FiNeMo hits to BED9 files with tf_motif_id labels
#          and per-TF colors for IGV.
#
# Requires:
#   analysis/0.0.motif_id_annotation/tables/cluster_to_tf_motif_id_mapping.tsv
#   analysis/0.2.finemo_hit_qc/hits_filtered_{day}.bed.gz  (from 0.2.finemo_hit_qc.py)
#
# Output:
#   results/igv_files/finemo_filtered_hits/{day}_{peak_type}_hits.bed.gz   (bgzipped, one per day)
#
# Usage:
#   sbatch 0.3.hits_to_bed.sh

ANALYSIS_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PIPELINE_DIR="$(cd "${ANALYSIS_DIR}/../pipeline" && pwd)"
SLURM_SUBMIT_DIR="${PIPELINE_DIR}"
source "${PIPELINE_DIR}/config.sh"

ml biology samtools

mapping_tsv="${core_path}/analysis/0.0.motif_id_annotation/tables/cluster_to_tf_motif_id_mapping.tsv"
hits_qc_dir="${core_path}/analysis/0.2.finemo_hit_qc"
bed_dir="${core_path}/results/igv_files/finemo_filtered_hits"

if [[ ! -f "${mapping_tsv}" ]]; then
    echo "ERROR: ${mapping_tsv} not found. Run 0.0.motif_id_annotation.py first." >&2
    exit 1
fi

echo "[$(date)] Compressing input hits BED files..."
for day in "${days[@]}"; do
    bed="${hits_qc_dir}/hits_filtered_${day}.bed"
    if [[ -f "${bed}" && ! -f "${bed}.gz" ]]; then
        bgzip -f "${bed}"
        echo "  ${day}: $(basename "${bed}").gz"
    fi
done

echo "[$(date)] Converting hits to BED (tf_motif_id labels)..."

python "${ANALYSIS_DIR}/hits_to_bed.py" \
    --mapping    "${mapping_tsv}" \
    --hits-dir   "${hits_qc_dir}" \
    --days       "${days[@]}" \
    --peak-type  "${peak_type}" \
    --out-dir    "${bed_dir}"

echo "[$(date)] Compressing BED files..."
for day in "${days[@]}"; do
    bed="${bed_dir}/${day}_${peak_type}_hits.bed"
    if [[ -f "${bed}" ]]; then
        bgzip -f "${bed}"
        echo "  ${day}: $(basename "${bed}").gz"
    fi
done

echo "[$(date)] Done. BED files in: ${bed_dir}"

igv_dir="${core_path}/results/igv_files"
ln -sfn "${hits_qc_dir}" "${igv_dir}/finemo_hit_qc"
echo "[$(date)] Symlink created: ${igv_dir}/finemo_hit_qc -> ${hits_qc_dir}"
