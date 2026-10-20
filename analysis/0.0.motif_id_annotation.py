#!/usr/bin/env python3
#
# Purpose: Annotate cluster-averaged motifs against the MotifCompendium reference
#          database using assign_label_from_pfms() on the cluster-averaged PWMs.
#
# tf_motif_id column:
#   - Reference DB match with exactly one TF -> gene name (e.g. CTCF, REST)
#   - Reference DB match with multiple TFs   -> annotation_name0 (e.g. FOS-JUN_0)
#   - No match above min_annotation_score    -> "cluster_<cluster_id>"
#
# expressed_TFs_dX columns:
#   Comma-separated subset of TFs with mean log-normalised (pflog layer)
#   expression > 0 across cells in that day's scRNA-seq file. Empty string
#   if none expressed. This is a permissive detection threshold (any nonzero
#   signal counts), not an expression-level cutoff -- see the companion
#   tf_expression_by_day.tsv for continuous mean expression / pct-cells-
#   expressing per TF per day if a stricter threshold is needed downstream.
#
# Previously this joined against data/gex/seurat_rna.h5ad (a Seurat-derived
# object with a "cell.state" column covering only d0-d3), which no longer
# exists on disk. Replaced 2026-08-24 with the per-day scRNA-seq processed
# upstream, which covers all 5 days and is already fully QC'd/doublet-
# filtered (every cell in each file passes all pass_* obs flags).
#
# Depends on:
#   results/compendium/modisco_compiled/modisco_compendium.mc
#   MotifCompendium-Database-Human.meme.txt
#   MotifCompendium-Database-Human.metadata.tsv
#   /path/to/10x_5_timepoints/Pseudobulks/RNA/rna_{day}.h5ad
#
# Output:
#   analysis/0.0.motif_id_annotation/tables/cluster_to_tf_motif_id_mapping.tsv
#   analysis/0.0.motif_id_annotation/tables/tf_expression_by_day.tsv
#
# Before running: conda activate motif_compendium
# %%
import os
import pickle
import re

import anndata as ad
import MotifCompendium.utils.analysis as utils_analysis
import numpy as np
import pandas as pd
from scipy.sparse import issparse


# %%
def get_expressed_tfs(tfs_str, expressed_genes):
    if pd.isna(tfs_str) or not tfs_str:
        return ""
    tfs = [t.strip() for t in re.split(r"[,@;]", tfs_str) if t.strip()]
    return ",".join(t for t in tfs if t in expressed_genes)


# %%
core_path = "/path/to/ipsc-ec-chrombpnet"
save_outputs = True
min_annotation_score = 0.8

output_path = os.path.join(core_path, "analysis", "0.0.motif_id_annotation")
os.makedirs(os.path.join(output_path, "tables"), exist_ok=True)

mc_path = os.path.join(
    core_path,
    "results",
    "compendium",
    "modisco_compiled",
    "modisco_compendium.mc",
)
pfm_path = (
    "/path/to/MotifCompendium/pipeline/data/MotifCompendium-Database-Human.meme.txt"
)
db_metadata_path = (
    "/path/to/MotifCompendium/pipeline/data/MotifCompendium-Database-Human.metadata.tsv"
)
gex_dir = "/path/to/10x_5_timepoints/Pseudobulks/RNA"

expression_days = ["d0", "d1", "d2", "d3", "d4"]

# %%
print("Loading MotifCompendium...")
with open(mc_path, "rb") as fh:
    mc = pickle.load(fh)
print(f"  {len(mc.metadata)} patterns, {mc.metadata['cluster_id'].nunique()} clusters")
# %%
# Compute cluster averages
mc_avg = mc.cluster_averages(
    "cluster_id",
    aggregations=[
        ("annotation_name0", "concat", "annotation_names"),
        ("annotation_score0", "avg", "annotation_score_mean"),
        ("name", "count", "n_patterns"),
        ("model", "concat", "days_present"),
        ("avg_dist_from_summit", "avg", "avg_dist_from_summit"),
        ("num_seqlets", "sum", "total_seqlets"),
        ("posneg", "concat", "posneg"),
    ],
    weight_col="num_seqlets",
    compute_quality_stats=True,
)
# %%
mc_avg
# %%
# Annotate cluster-averaged PWMs directly against the reference database.
# This gives each cluster one best-match annotation independent of which
# constituent patterns happened to be most abundant.
utils_analysis.assign_label_from_pfms(
    mc=mc_avg,
    pfm_file=pfm_path,
    save_col_prefix="annotation",
    min_score=min_annotation_score,
    save_images=False,
)
n_annotated = mc_avg.metadata["annotation_name0"].notna().sum()
print(f"  {n_annotated} / {len(mc_avg.metadata)} annotated clusters")
# %%
cluster_df = mc_avg.metadata.copy()
cluster_df = cluster_df.rename(columns={"source_cluster": "cluster_id"})

# %%
# Join TF gene names from reference DB metadata using the cluster-level annotation
db = pd.read_csv(db_metadata_path, sep="\t", usecols=["name", "TF"])
db = db.rename(columns={"name": "annotation_name0", "TF": "TFs"})

cluster_df = cluster_df.merge(db, on="annotation_name0", how="left")
cluster_df["n_TFs"] = cluster_df["TFs"].str.split(r"[,@;]").str.len()

# %%
# Single-TF clusters use the gene name directly, multi-TF use the DB motif name
is_single_tf = cluster_df["TFs"].notna() & (cluster_df["n_TFs"] == 1)

cluster_df["tf_motif_id"] = cluster_df["annotation_name0"].copy()
cluster_df["tf_motif_id"] = cluster_df["tf_motif_id"].where(
    ~is_single_tf, cluster_df["TFs"]
)

unmatched_mask = cluster_df["tf_motif_id"].isna()
cluster_df["tf_motif_id"] = cluster_df["tf_motif_id"].fillna(
    "cluster_" + cluster_df["cluster_id"].astype(str)
)

n_single = is_single_tf.sum()
n_unmatched = unmatched_mask.sum()
print(
    f"  {n_single} single-TF, {len(cluster_df) - n_single - n_unmatched} multi-TF, {n_unmatched} unmatched clusters"
)

# %%
print("\nLoading expression data...")
all_tfs = set()
for tfs_str in cluster_df["TFs"].dropna():
    all_tfs.update(t.strip() for t in re.split(r"[,@;]", tfs_str) if t.strip())

expressed_per_day = {}
tf_expression_records = []
for day in expression_days:
    day_gex_path = os.path.join(gex_dir, f"rna_{day}.h5ad")
    adata = ad.read_h5ad(day_gex_path)
    n_cells = adata.n_obs
    print(f"  {day}: {n_cells} cells, {adata.n_vars} genes ({day_gex_path})")

    tf_cols = [tf for tf in all_tfs if tf in adata.var_names]
    missing_tfs = all_tfs - set(tf_cols)
    if missing_tfs:
        print(f"  {day}: {len(missing_tfs)} TFs not found in var_names, skipping them")

    pflog = adata[:, tf_cols].layers["pflog"]
    pflog = pflog.toarray() if issparse(pflog) else np.asarray(pflog)
    counts = adata[:, tf_cols].layers["counts"]
    counts = counts.toarray() if issparse(counts) else np.asarray(counts)

    mean_pflog = pflog.mean(axis=0)
    pct_expressing = (counts > 0).mean(axis=0) * 100

    expressed_per_day[day] = {tf for tf, m in zip(tf_cols, mean_pflog) if m > 0}
    print(f"  {day}: {len(expressed_per_day[day])}/{len(tf_cols)} TFs expressed")

    tf_expression_records.append(
        pd.DataFrame(
            {
                "TF": tf_cols,
                "day": day,
                "n_cells": n_cells,
                "mean_pflog_expression": mean_pflog,
                "pct_cells_expressing": pct_expressing,
            }
        )
    )

tf_expression_df = pd.concat(tf_expression_records, ignore_index=True)

# %%
for day in expression_days:
    cluster_df[f"expressed_TFs_{day}"] = cluster_df["TFs"].apply(
        get_expressed_tfs, expressed_genes=expressed_per_day[day]
    )
# %%
cluster_df["tf_motif_id"] = np.where(
    cluster_df["tf_motif_id"] == "",
    "cluster_" + cluster_df["cluster_id"].astype(str),
    cluster_df["tf_motif_id"],
)
# %%
out_cols = [
    "cluster_id",
    "total_seqlets",
    "n_patterns",
    "days_present",
    "posneg",
    "avg_dist_from_summit",
    "annotation_name0",
    "annotation_score0",
    "annotation_score_mean",
    "annotation_names",
    "TFs",
    "n_TFs",
    "tf_motif_id",
] + [f"expressed_TFs_{d}" for d in expression_days]

if save_outputs:
    out_path = os.path.join(output_path, "tables", "cluster_to_tf_motif_id_mapping.tsv")
    cluster_df[out_cols].to_csv(out_path, sep="\t", index=False)
    print(f"\nSaved -> {out_path}")

    tf_expr_path = os.path.join(output_path, "tables", "tf_expression_by_day.tsv")
    tf_expression_df.to_csv(tf_expr_path, sep="\t", index=False)
    print(f"Saved -> {tf_expr_path}")

# %%
