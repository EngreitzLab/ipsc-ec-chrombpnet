#!/bin/bash
# 01.preprocess_peaks.sh
# Purpose: Remove blacklisted regions from peak files and reformat to
#          narrowPeak for chrombpnet (summit = midpoint of peak).
#
# Run interactively:
#   srun --mem 100G --time 6:00:00 --partition engreitz --pty bash
#   source ${CONDA_INIT} && conda activate ${CONDA_ENV}
#   ml biology && ml bedtools/2.30.0
#   bash 01.preprocess_peaks.sh

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "${SCRIPT_DIR}/config.sh"

set -euo pipefail

mkdir -p "${data_path}"

# Expand the blacklist by 1057 bp on each side (half of the 2114 bp input window)
# so that no training window overlaps a blacklisted region
bedtools slop \
    -i "${blacklist}" \
    -g "${chrom_sizes}" \
    -b 1057 > "${data_path}/blacklist_slop.bed"

# Keep only canonical autosomes + chrX/Y (first 24 lines of chrom.sizes)
# head -n 24 "${chrom_sizes}" > "${data_path}/hg38.chrom.subset.sizes"

for day in "${days[@]}"; do
    echo "Processing peaks for ${day}..."
    mkdir -p "${data_path}/${day}"

    # Remove peaks overlapping (slopped) blacklist
    bedtools intersect \
        -v \
        -a "${core_path}/data/${day}_${peak_type}_peaks.bed" \
        -b "${data_path}/blacklist_slop.bed" \
        > "${data_path}/${day}/${day}_${peak_type}_peaks_no_blacklist.bed"

    # Convert to narrowPeak format; summit = midpoint of peak
    awk 'BEGIN{OFS="\t"} {
        summit = int(($3-$2)/2);
        print $1, $2, $3, "peak_"NR, "0", ".", "0", "-1", "-1", summit
    }' "${data_path}/${day}/${day}_${peak_type}_peaks_no_blacklist.bed" \
        > "${data_path}/${day}/${day}_${peak_type}_peaks_no_blacklist.narrowPeak"

    n=$(wc -l < "${data_path}/${day}/${day}_${peak_type}_peaks_no_blacklist.narrowPeak")
    echo "  -> ${n} peaks written to ${day}_${peak_type}_peaks_no_blacklist.narrowPeak"
done

echo "Done: 01.preprocess_peaks.sh"
