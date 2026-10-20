"""
1.0.gene_motif_linking.py
Compute distance-weighted TF contribution scores per gene per day,
following the scE2G gene-motif linking approach.

Pipeline
--------
1. Build gene to linked peaks map: union all scE2G-linked peaks across days.
2. For each element type (promoter / enhancer / all_cres) and each day:
   find FiNeMo hits overlapping any linked peak, then compute a weighted score:
     weighted_score = sign * hit_score_col * weight
   where weight depends on the chosen strategy (e2g / abc / distance).
   neg_patterns hits receive negative scores (repressive motif orientation).
3. Aggregate by (TargetGene, tf_motif_id, day) and save.
4. Plot mean TF score per day:
   - 3-panel heatmap (promoter | enhancer | all_cres) for top TFs
   - Per-day scatter of enhancer vs promoter mean score

Outputs (analysis/1.0.gene_motif_linking/)
------------------------------------------
  data/{strategy}_{hit_score_col}_{et}_gene_motif_scores.tsv.gz
  plots/score_heatmap_{strategy}_{hit_score_col}.pdf / .png
  plots/score_heatmap_{strategy}_{hit_score_col}_{et}_data.tsv
  plots/enhancer_vs_promoter_{strategy}_{hit_score_col}.pdf / .png
  plots/enhancer_vs_promoter_{strategy}_{hit_score_col}_data.tsv

Prerequisites:
  analysis/0.0.motif_id_annotation/tables/cluster_to_tf_motif_id_mapping.tsv
  analysis/0.2.finemo_hit_qc/hits_filtered_{day}.tsv
  {scE2G_results_dir}/{day}/multiome_powerlaw_v3/scE2G_predictions_threshold0.177.tsv.gz
    (per pipeline/dataset_config.sh's peaks_path)
"""

# %%
import os
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["axes.spines.right"] = False
matplotlib.rcParams["font.size"] = 14
matplotlib.rcParams["savefig.dpi"] = 300
matplotlib.rcParams["savefig.bbox"] = "tight"
matplotlib.rcParams["savefig.transparent"] = True

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from genomics import intersect_hits_peaks


# %%
def _compute_scores(hits_per_day, e2g_df, strategy, hit_score_col, decay_dist):
    """Compute weighted gene x TF x day scores for one element-type subset."""
    records = []
    for day, hits_df in hits_per_day.items():
        overlapping = intersect_hits_peaks(hits_df, e2g_df)
        if overlapping.empty:
            print("no overlaps")
            continue
        overlapping["dist_to_tss"] = (
            overlapping["midpoint"] - overlapping["TargetGeneTSS"]
        ).abs()
        if strategy == "distance":
            overlapping["weight"] = np.exp(
                -overlapping["dist_to_tss"] / decay_dist
            ) + np.exp(-1)
        elif strategy == "abc":
            overlapping["weight"] = overlapping["ABC.Score"]
        elif strategy == "e2g":
            overlapping["weight"] = overlapping["E2G.Score.qnorm"]
        overlapping["weighted_score"] = (
            overlapping["sign"] * overlapping[hit_score_col] * overlapping["weight"]
        )
        agg = (
            overlapping.groupby(["TargetGene", "tf_motif_id"])["weighted_score"]
            .sum()
            .reset_index()
        )
        agg["day"] = day
        records.append(agg)
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


# %%
core_path = "/path/to/ipsc-ec-chrombpnet"
# Real scE2G output (per pipeline/dataset_config.sh's peaks_path).
scE2G_results_dir = "/path/to/10x_5_timepoints/scE2G/results"

finemo_dir = os.path.join(core_path, "results", "finemo_unified")
data_output_dir = os.path.join(core_path, "analysis", "1.0.gene_motif_linking")
os.makedirs(os.path.join(data_output_dir, "data"), exist_ok=True)
os.makedirs(os.path.join(data_output_dir, "plots"), exist_ok=True)

days = ["d0", "d1", "d2", "d3", "d4"]
peak_type = "all"

# Weight strategy: "e2g" | "abc" | "distance"
weight_strategy = "e2g"
decay_dist = 5000  # bp half-decay distance (distance strategy only)

# Hit score column used to weight motif contributions
# Options: "hit_importance" | "hit_coefficient_global"
hit_score_col = "hit_coefficient_global"

# %%
# Load cluster-level TF annotation
annot_df = pd.read_csv(
    os.path.join(
        core_path,
        "analysis",
        "0.0.motif_id_annotation",
        "tables",
        "cluster_to_tf_motif_id_mapping.tsv",
    ),
    sep="\t",
    usecols=["cluster_id", "tf_motif_id"],
)
annot_df["cluster_id"] = annot_df["cluster_id"].astype(str)
annot_df = annot_df.drop_duplicates("cluster_id").set_index("cluster_id")

# %%
# Load FiNeMo hits per day; attach cluster annotation, sign, and midpoint
hits_per_day = {}
for day in days:
    hits_path = os.path.join(
        core_path, "analysis", "0.2.finemo_hit_qc", f"hits_filtered_{day}.tsv"
    )
    hits = pd.read_csv(hits_path, sep="\t")
    print("-" * 10)
    print(f"  {day}: reading {os.path.basename(hits_path)}")
    hits["cluster_id"] = hits["motif_name"].str.split(".").str[-1]
    # pos_patterns: +1 (activating), neg_patterns: -1 (repressive)
    hits["sign"] = (
        hits["motif_name"].str.startswith("pos_patterns").map({True: 1, False: -1})
    )
    hits["midpoint"] = (hits["start"] + hits["end"]) // 2
    hits = hits.join(annot_df["tf_motif_id"], on="cluster_id")
    hits = hits.dropna(subset=["tf_motif_id"])
    hits_per_day[day] = hits
    print(f"  {day}: {len(hits):,} annotated hits loaded")
# %%
# Build gene to linked peaks map (union across all days)
e2g_rows = []
for day in days:
    f = os.path.join(
        scE2G_results_dir,
        day,
        "multiome_powerlaw_v3",
        "scE2G_predictions_threshold0.177.tsv.gz",
    )
    df = pd.read_csv(
        f,
        sep="\t",
        usecols=[
            "chr",
            "start",
            "end",
            "TargetGene",
            "TargetGeneTSS",
            "ABC.Score",
            "E2G.Score.qnorm",
            "isSelfPromoter",
        ],
    )
    e2g_rows.append(df)

e2g_all = (
    pd.concat(e2g_rows, ignore_index=True).drop_duplicates().reset_index(drop=True)
)

# %%
# Compute scores for promoter, enhancer, and all CREs
et_configs = {
    "promoter": e2g_all[e2g_all["isSelfPromoter"]].reset_index(drop=True),
    "enhancer": e2g_all[~e2g_all["isSelfPromoter"]].reset_index(drop=True),
    "all_cres": e2g_all,
}
# %%
all_scores = {}
for et, e2g_sub in et_configs.items():
    print(f"\n  [{weight_strategy}/{et}] {len(e2g_sub):,} linked peaks")
    scores = _compute_scores(
        hits_per_day, e2g_sub, weight_strategy, hit_score_col, decay_dist
    )
    if scores.empty:
        print(f"  [{weight_strategy}/{et}] No scores computed, skipping")
        continue
    scores["strategy"] = weight_strategy
    scores["et"] = et
    out_path = os.path.join(
        data_output_dir,
        "data",
        f"{weight_strategy}_{hit_score_col}_{et}_gene_motif_scores.tsv.gz",
    )
    scores.to_csv(out_path, sep="\t", index=False, compression="gzip")
    print(f"  [{weight_strategy}/{et}] Saved {len(scores):,} records: {out_path}")
    all_scores[et] = scores

# %%
