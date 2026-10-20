#!/usr/bin/env python3
"""
hits_to_bed.py
Convert FiNeMo hits.bed.gz to BED9 format with tf_motif_id labels and
per-TF colors for IGV.

Uses tf_motif_id from cluster_to_tf_motif_id_mapping.tsv produced by
0.0.motif_id_annotation.py. Each cluster has one tf_motif_id from the
cluster-averaged PWM annotation.

Output is BED9 (chrom, start, end, name, score, strand,
                thickStart, thickEnd, itemRgb).
IGV renders these as labelled rectangles with strand arrows and per-TF colors.
Requires "track ... itemRgb=On" header (written automatically).

Input:
  --mapping   cluster_to_tf_motif_id_mapping.tsv   (from 0.0.motif_id_annotation.py)
  --hits-dir  directory containing hits_filtered_{day}.bed.gz (from 0.2.finemo_hit_qc.py)
  --out-dir   output directory for per-day .bed files

Usage:
  python 0.2.hits_to_bed.py \\
      --mapping   analysis/0.0.motif_id_annotation/tables/cluster_to_tf_motif_id_mapping.tsv \\
      --hits-dir  analysis/0.2.finemo_hit_qc \\
      --days      d0 d1 d2 d3 d4 \\
      --peak-type all \\
      --out-dir   results/compendium/bed
"""

import argparse
import colorsys
import gzip
import os
import sys

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(
        description="Convert FiNeMo hits to BED9 with tf_motif_id labels for IGV."
    )
    p.add_argument(
        "--mapping",
        required=True,
        help="cluster_to_tf_motif_id_mapping.tsv from 0.0.motif_id_annotation.py",
    )
    p.add_argument(
        "--hits-dir",
        required=True,
        help="analysis/0.2.finemo_hit_qc directory containing hits_filtered_{day}.bed.gz",
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
    p.add_argument("--out-dir", required=True, help="Output directory for .bed files")
    return p.parse_args()


def build_cluster_map(mapping_tsv):
    """Map cluster_id -> tf_motif_id from the cluster-level annotation table."""
    mapping = pd.read_csv(mapping_tsv, sep="\t")
    mapping["tf_motif_id"] = mapping["tf_motif_id"].astype(str)
    return mapping.set_index("cluster_id")["tf_motif_id"].to_dict()


def make_color_map(cluster_map):
    """Assign a distinct RGB color to each unique tf_motif_id.

    Uses evenly spaced hues in HSV space.
    Returns {tf_motif_id: "R,G,B"}.
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


def convert_hits(hits_gz, cluster_map, color_map, out_path, day):
    """Read hits.bed.gz and write a BED9 file with tf_motif_id names and colors."""
    n_written = 0
    n_unknown = 0
    with gzip.open(hits_gz, "rt") as fh_in, open(out_path, "w") as fh_out:
        fh_out.write(f'track name="hits_filtered_{day}" itemRgb=On\n')
        for line in fh_in:
            if line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            chrom, start, end, pattern_name, score, strand = cols[:6]

            cluster_id = int(pattern_name.rsplit(".", 1)[-1])
            tf_name = cluster_map.get(cluster_id)
            if tf_name is None:
                name = pattern_name
                rgb = GRAY
                n_unknown += 1
            else:
                name = f"{tf_name}"
                rgb = color_map[tf_name]

            fh_out.write(
                f"{chrom}\t{start}\t{end}\t{name}\t{score}\t{strand}\t"
                f"{start}\t{end}\t{rgb}\n"
            )
            n_written += 1

    return n_written, n_unknown


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"@ loading cluster -> tf_motif_id map from {args.mapping}")
    cluster_map = build_cluster_map(args.mapping)
    print(f"@ {len(cluster_map)} clusters mapped")

    color_map = make_color_map(cluster_map)
    print(f"@ {len(set(cluster_map.values()))} unique tf_motif_id colors assigned")

    for day in args.days:
        hits_gz = os.path.join(args.hits_dir, f"hits_filtered_{day}.bed.gz")
        if not os.path.exists(hits_gz):
            print(
                f"  [WARN] {day}: hits_filtered_{day}.bed.gz not found, skipping: {hits_gz}",
                file=sys.stderr,
            )
            continue

        out_path = os.path.join(args.out_dir, f"{day}_{args.peak_type}_hits.bed")
        n, n_unk = convert_hits(hits_gz, cluster_map, color_map, out_path, day)
        unk_str = f" ({n_unk} gray, no cluster match)" if n_unk else ""
        print(f"  {day}: {n:,} hits written -> {out_path}{unk_str}")

    print("@ done")


if __name__ == "__main__":
    main()
