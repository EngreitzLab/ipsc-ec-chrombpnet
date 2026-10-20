#!/bin/bash
# 00.3.tss_profiles.sh
# Aggregate TSS insertion profile (+/- 2 kb) per timepoint, for the TSS
# enrichment profile figure produced by 00.3.tss_profiles.py.
#
# Convention follows ArchR / ENCODE: two 1 bp insertion sites per fragment
# (both fragment ends), distance measured strand aware to every TSS within
# 2 kb, tallied into 1 bp bins. Normalization to the flanking background is
# done in the plotting script, not here.
#
# The TSS set is the same annotation the ArchR per cell QC used
# (compute_atac_percell_qc.R): scE2G's CollapsedGeneBounds hg38 GENCODE v43,
# deduplicated by gene symbol, so the profiles are consistent with the
# per cell TSS enrichment values in 00.2.atac_percell_qc.
#
# Only barcodes present in that day's per cell QC report are counted, i.e.
# the RNA called cells, matching 00.2.
#
# Usage
#   bash analysis/00.3.tss_profiles.sh              # all five days
#   bash analysis/00.3.tss_profiles.sh d0           # one day
#   SMOKE_LINES=2000000 bash analysis/00.3.tss_profiles.sh d0   # subset test
#
# Requires: ml biology bedtools/2.30.0

set -euo pipefail

core_path="/path/to/ipsc-ec-chrombpnet"
frag_dir="/path/to/10x_5_timepoints/Pseudobulks/ATAC"
qc_dir="/path/to/scE2G_qc/generated_for_submission"
gene_bed="/path/to/scE2G/resources/genome_annotations/CollapsedGeneBounds.hg38.intGENCODEv43.bed"

flank_bp=2000            # profile half width
output_path="${core_path}/analysis/00.3.tss_profiles"
tables_dir="${output_path}/tables"
work_dir="${output_path}/work"

smoke_lines="${SMOKE_LINES:-0}"   # >0 truncates the fragment stream, for testing

mkdir -p "$tables_dir" "$work_dir"

days=("$@")
if [ ${#days[@]} -eq 0 ]; then
    days=(d0 d1 d2 d3 d4)
fi

command -v bedtools >/dev/null || {
    echo "bedtools not on PATH; run: ml biology bedtools/2.30.0" >&2
    exit 1
}

# TSS windows: one per deduplicated gene symbol, strand aware, 0 based.
# For + strand the TSS is the BED start; for - strand it is end-1. This
# reproduces GenomicRanges resize(fix="start") as used in the ArchR QC script.
tss_windows="${work_dir}/tss_windows_${flank_bp}bp.bed"
if [ ! -s "$tss_windows" ]; then
    echo "building TSS windows -> $tss_windows"
    awk -v OFS='\t' -v flank="$flank_bp" '
        /^#/ { next }
        {
            symbol = $4
            if (symbol in seen) next
            seen[symbol] = 1
            if ($1 !~ /^chr([1-9]|1[0-9]|2[0-2]|X|Y)$/) next
            strand = $6
            tss = (strand == "+") ? $2 : $3 - 1
            start = tss - flank
            if (start < 0) start = 0
            print $1, start, tss + flank + 1, tss, strand
        }
    ' "$gene_bed" | sort -k1,1 -k2,2n > "$tss_windows"
fi
echo "TSS windows: $(wc -l < "$tss_windows")"

for day in "${days[@]}"; do
    frag_file="${frag_dir}/${day}.fragments.tsv.gz"
    qc_file="${qc_dir}/${day}/WTC11-EC_10xmultiome_5timepoints_${day}_percell_qc_report.tsv.gz"
    keep_file="${work_dir}/${day}_barcodes.txt"
    out_file="${tables_dir}/tss_profile_${day}.tsv"

    [ -s "$frag_file" ] || { echo "missing $frag_file" >&2; exit 1; }
    [ -s "$qc_file" ] || { echo "missing $qc_file" >&2; exit 1; }

    if [ ! -s "$keep_file" ]; then
        zcat "$qc_file" | tail -n +2 | cut -f1 > "$keep_file"
    fi
    echo "[${day}] barcodes to count: $(wc -l < "$keep_file")"

    if [ "$smoke_lines" -gt 0 ]; then
        echo "[${day}] SMOKE MODE, first ${smoke_lines} fragment lines only"
        stream="zcat ${frag_file} | head -n ${smoke_lines}"
        out_file="${tables_dir}/tss_profile_${day}_smoke.tsv"
        # head closes the pipe early, which makes zcat exit on SIGPIPE; that
        # is expected in smoke mode and must not abort the script
        set +o pipefail
    else
        stream="zcat ${frag_file}"
        set -o pipefail
    fi

    echo "[${day}] streaming insertions through bedtools ..."
    # Stage 1: keep called cell fragments, emit both 1 bp insertion sites.
    # Stage 2: intersect against the TSS windows (B held in memory, A streamed).
    # Stage 3: strand aware distance to TSS, tallied in 1 bp bins.
    eval "$stream" \
        | awk -v OFS='\t' -v keep="$keep_file" '
            BEGIN { while ((getline line < keep) > 0) k[line] = 1 }
            ($4 in k) { print $1, $2, $2 + 1; print $1, $3 - 1, $3 }
          ' \
        | bedtools intersect -a stdin -b "$tss_windows" -wa -wb \
        | awk -v OFS='\t' -v flank="$flank_bp" '
            {
                d = $2 - $7
                if ($8 == "-") d = -d
                if (d >= -flank && d <= flank) { h[d]++; total++ }
            }
            END {
                print "distance_from_tss", "insertions"
                for (d = -flank; d <= flank; d++) print d, (d in h) ? h[d] : 0
                printf("tallied %d TSS proximal insertions\n", total) > "/dev/stderr"
            }
          ' > "$out_file"

    echo "[${day}] wrote $out_file ($(wc -l < "$out_file") bins)"
done

echo "done"
