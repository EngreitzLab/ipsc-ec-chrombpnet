#!/usr/bin/env python3
# =============================================================================
# 5.3.hits_to_bed.py
# Purpose: Convert FiNeMo hits.bed.gz to BED9 format with TF names and
#          per-TF colors for IGV.
#
# Output is BED9 (chrom, start, end, name, score, strand,
#                 thickStart, thickEnd, itemRgb).
# IGV renders these as labelled rectangles with strand arrows and per-TF colors.
# Requires "track ... itemRgb=On" header (written automatically).
#
# TF name is extracted from JASPAR annotation: MA0103.3.ZEB1 -> ZEB1
# Name format: "{TF}_{pattern_id}"  e.g. ZEB1_neg_patterns.194
# Unannotated clusters use the pattern ID alone and are colored gray.
# All patterns sharing the same TF name get the same color.
#
# Input:
#   --meta      modisco_compendium_meta.tsv   (from step 5.1)
#   --hits-dir  directory containing {day}_{peak_type}/hits.bed.gz
#   --out-dir   output directory for per-day .bed files
#
# Usage:
#   python 5.3.hits_to_bed.py \
#       --meta    /path/to/modisco_compendium_meta.tsv \
#       --hits-dir /path/to/chrombpnet_finemo_unified \
#       --days    d0 d1 d2 d3 d4 \
#       --peak-type all \
#       --out-dir /path/to/output
# =============================================================================

import argparse
import colorsys
import csv
import gzip
import os
import sys


def parse_args():
    p = argparse.ArgumentParser(
        description="Convert FiNeMo hits to bed with TF names for IGV."
    )
    p.add_argument(
        "--meta",
        required=True,
        help="modisco_compendium_meta.tsv from step 5.1",
    )
    p.add_argument(
        "--hits-dir",
        required=True,
        help="Directory containing {day}_{peak_type}/hits.bed.gz",
    )
    p.add_argument(
        "--days",
        nargs="+",
        default=["d0", "d1", "d2", "d3", "d4"],
        help="Days to process (default: d0 d1 d2 d3 d4)",
    )
    p.add_argument(
        "--peak-type",
        default="all",
        help="Peak type suffix used in directory names (default: all)",
    )
    p.add_argument(
        "--out-dir", required=True, help="Output directory for .bed files"
    )
    return p.parse_args()


def extract_tf_name(annotation_name):
    """Return first TF from MotifCompendium annotation (e.g. 'CTCF,CTCFL' -> 'CTCF')."""
    if not annotation_name or annotation_name in ("", "nan", "NA"):
        return None
    return str(annotation_name).split(",")[0].strip()


def build_cluster_map(meta_path):
    """Map cluster_id -> TF name using the highest-scoring annotation hit per cluster."""
    # Read all rows, then pick the best-scoring annotation per cluster
    rows = []
    with open(meta_path) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            rows.append(row)

    # Sort by annotation_score0 descending so the best annotation wins
    rows.sort(
        key=lambda r: float(r["annotation_score0"]) if r["annotation_score0"] else 0.0,
        reverse=True,
    )

    cluster_map = {}
    for row in rows:
        cid = int(row["cluster_id"])
        if cid not in cluster_map:
            tf = extract_tf_name(row.get("annotation_name0", ""))
            cluster_map[cid] = tf if tf else f"cluster_{cid}"
    return cluster_map


def make_color_map(cluster_map):
    """Assign a distinct RGB color to each unique TF name.

    Uses evenly spaced hues in HSV space. Unannotated clusters get gray.
    Returns {tf_name: "R,G,B"}.
    """
    unique_tfs = sorted(set(cluster_map.values()))
    n = len(unique_tfs)
    color_map = {}
    for i, tf in enumerate(unique_tfs):
        h = i / n
        r, g, b = colorsys.hsv_to_rgb(h, 0.80, 0.85)
        color_map[tf] = f"{int(r * 255)},{int(g * 255)},{int(b * 255)}"
    return color_map


GRAY = "128,128,128"


def convert_hits(hits_gz, cluster_map, color_map, out_path, day, peak_type):
    """Read hits.bed.gz and write a BED9 file with TF names, strand, and colors."""
    n_written = 0
    n_unknown = 0
    with gzip.open(hits_gz, "rt") as fh_in, open(out_path, "w") as fh_out:
        fh_out.write(f'track name="{day}_{peak_type}_hits" itemRgb=On\n')
        for line in fh_in:
            if line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            chrom, start, end, pattern_name, score, strand = cols[:6]

            # pattern_name is like "pos_patterns.15" or "neg_patterns.194"
            cluster_id = int(pattern_name.rsplit(".", 1)[-1])
            tf_name = cluster_map.get(cluster_id)
            if tf_name is None:
                name = pattern_name
                rgb = GRAY
                n_unknown += 1
            else:
                name = f"{tf_name}_{pattern_name}"
                rgb = color_map[tf_name]

            # BED9: thickStart/thickEnd = start/end (full block colored)
            fh_out.write(
                f"{chrom}\t{start}\t{end}\t{name}\t{score}\t{strand}\t"
                f"{start}\t{end}\t{rgb}\n"
            )
            n_written += 1

    return n_written, n_unknown


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"@ loading cluster -> TF map from {args.meta}")
    cluster_map = build_cluster_map(args.meta)
    print(f"@ {len(cluster_map)} clusters mapped")

    color_map = make_color_map(cluster_map)
    unique_tfs = len(set(cluster_map.values()))
    print(f"@ {unique_tfs} unique TF colors assigned")

    for day in args.days:
        hits_gz = os.path.join(
            args.hits_dir, f"{day}_{args.peak_type}", "hits.bed.gz"
        )
        if not os.path.exists(hits_gz):
            print(
                f"  [WARN] {day}: hits.bed.gz not found, skipping: {hits_gz}",
                file=sys.stderr,
            )
            continue

        out_path = os.path.join(
            args.out_dir, f"{day}_{args.peak_type}_hits.bed"
        )
        n, n_unk = convert_hits(
            hits_gz, cluster_map, color_map, out_path, day, args.peak_type
        )
        unk_str = f" ({n_unk} gray, no TF annotation)" if n_unk else ""
        print(f"  {day}: {n:,} hits written -> {out_path}{unk_str}")

    print("@ done")


if __name__ == "__main__":
    main()
