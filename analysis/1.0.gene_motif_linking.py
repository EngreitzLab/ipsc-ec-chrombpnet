"""
1.0.gene_motif_linking.py
Compute distance-weighted TF contribution scores per gene per day,
following the scE2G gene-motif linking approach.

Pipeline
--------
1. Load the scE2G links of each day (tagged with a day column).
2. For each element type (promoter / enhancer / all_cres) and each day:
   find that day's FiNeMo hits lying fully inside that day's linked peaks
   (hit_fully_inside), then compute a weighted score:
     weighted_score = hit_score_col * weight
   where weight depends on the chosen strategy (e2g / abc / distance).
   Only pos_patterns hits are used (pattern_prefixes): signed neg_patterns scores
   would cancel pos_patterns of the same TF and invert the direction of the
   one-sided enrichment tests in scripts 2 and 2.1.
   Optional (dedup_same_label, off for the current results): hits of different motif
   clusters sharing a tf_motif_id that overlap by more than label_overlap_bp are
   collapsed to the best hit_similarity (dedup_same_label_hits), so one site counts
   once per TF label (0.2 deduplicates only within a cluster).
3. Aggregate by (TargetGene, tf_motif_id, day): weighted_score (sum) and n_hits
   (number of hits in the gene's linked elements), and save.
   Also save motif-free per-gene covariates per day (number of linked elements,
   total element bp, summed E2G.Score.qnorm, TSS) for the adjusted test (2.1).
4. Plot mean TF score per day:
   - 3-panel heatmap (promoter | enhancer | all_cres) for top TFs
   - Per-day scatter of enhancer vs promoter mean score

Outputs (analysis/1.0.gene_motif_linking/)
------------------------------------------
  data/{strategy}_{hit_score_col}_{et}_gene_motif_scores.tsv.gz
  data/{et}_gene_covariates.tsv.gz
  tables/label_dedup_removed_hits.tsv   hits removed per day x tf_motif_id (dedup on)
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
def dedup_same_label_hits(hits, overlap_bp):
    """
    Collapse overlapping hits that share a tf_motif_id but come from different
    motif clusters: within each (chr, tf_motif_id), keep the hit with the highest
    hit_similarity and drop any hit overlapping an already-kept hit by more than
    overlap_bp (strand ignored). Same greedy rule as 0.2.finemo_hit_qc.py, which
    applies it within one motif cluster only.

    Hits are first split into chains: sorted by start, a hit starts a new chain when
    no earlier hit of the group overlaps it by more than overlap_bp. A hit can only
    be dropped by a hit it overlaps, which is always in the same chain, so running
    the greedy pass per chain gives the same result as running it over the group.

    Returns (kept hits, DataFrame of removed-hit counts per tf_motif_id).
    """
    h = hits.sort_values(["chr", "tf_motif_id", "start"], kind="stable").reset_index(
        drop=True
    )
    group = h["chr"].astype(str) + "|" + h["tf_motif_id"].astype(str)
    prev_max_end = h.groupby(group)["end"].cummax().groupby(group).shift()
    new_chain = prev_max_end.isna() | (h["start"] >= prev_max_end - overlap_bp)
    chain = new_chain.cumsum().to_numpy()

    starts = h["start"].to_numpy()
    ends = h["end"].to_numpy()
    similarity = h["hit_similarity"].to_numpy()
    keep = np.ones(len(h), dtype=bool)
    chain_sizes = np.bincount(chain)
    multi = np.flatnonzero(chain_sizes[chain] > 1)
    for idx in np.split(multi, np.flatnonzero(np.diff(chain[multi])) + 1):
        order = idx[np.argsort(-similarity[idx], kind="stable")]
        kept = []
        for i in order:
            if any(
                min(ends[i], ends[k]) - max(starts[i], starts[k]) > overlap_bp
                for k in kept
            ):
                keep[i] = False
            else:
                kept.append(i)
    removed = (
        h.loc[~keep, "tf_motif_id"].value_counts().rename("n_removed").reset_index()
    )
    return h[keep].reset_index(drop=True), removed


def gene_covariates(e2g_df, days):
    """Motif-free per-gene covariates per day from that day's scE2G links."""
    rows = []
    for day in days:
        d = e2g_df[e2g_df["day"] == day]
        cov = (
            d.assign(element_bp=d["end"] - d["start"])
            .groupby("TargetGene")
            .agg(
                chr=("chr", "first"),
                tss=("TargetGeneTSS", "first"),
                n_elements=("start", "size"),
                element_bp=("element_bp", "sum"),
                sum_e2g_qnorm=("E2G.Score.qnorm", "sum"),
            )
            .reset_index()
        )
        cov["day"] = day
        rows.append(cov)
    return pd.concat(rows, ignore_index=True)


def _compute_scores(
    hits_per_day, e2g_df, strategy, hit_score_col, decay_dist, hit_fully_inside
):
    """Compute weighted gene x TF x day scores for one element-type subset."""
    records = []
    for day, hits_df in hits_per_day.items():
        # Only this day's scE2G links: peaks are called per day, so links from other
        # days overlap the same enhancer and would count each hit once per copy
        e2g_day = e2g_df[e2g_df["day"] == day].drop(columns="day")
        if e2g_day.empty:
            raise ValueError(f"No scE2G links for {day}")
        overlapping = intersect_hits_peaks(
            hits_df, e2g_day, hit_fully_inside=hit_fully_inside
        )
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
            overlapping[hit_score_col] * overlapping["weight"]
        )
        agg = (
            overlapping.groupby(["TargetGene", "tf_motif_id"])
            .agg(
                weighted_score=("weighted_score", "sum"),
                n_hits=("weighted_score", "size"),
            )
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

# Count a hit only when it lies entirely inside a linked peak (as in PerturbNMF Stage 2);
# False counts any 1 bp overlap
hit_fully_inside = True

# FiNeMo pattern families kept (as in PerturbNMF Stage 2). neg_patterns are dropped:
# with signed scores they cancel pos_patterns of the same TF and invert the one-sided
# test of script 2; one d3 neg pattern (neg_patterns.21) also hits most elements
pattern_prefixes = ("pos_patterns",)

# Same-label dedup: hits of different clusters with the same tf_motif_id overlapping by
# more than label_overlap_bp count once (rule of 0.2.finemo_hit_qc.py within a cluster).
# Off for the current results (rerun 2026-10-20); validated against a brute-force pass
# (identical on chr21 + chr22) and removes 2.6% of d3 hits
dedup_same_label = False
label_overlap_bp = 3

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
# Load FiNeMo hits per day; keep pos_patterns, attach cluster annotation and midpoint
hits_per_day = {}
dedup_removed = []
for day in days:
    hits_path = os.path.join(
        core_path, "analysis", "0.2.finemo_hit_qc", f"hits_filtered_{day}.tsv"
    )
    hits = pd.read_csv(hits_path, sep="\t")
    print("-" * 10)
    print(f"  {day}: reading {os.path.basename(hits_path)}")
    n_all = len(hits)
    hits = hits[hits["motif_name"].str.startswith(pattern_prefixes)].copy()
    print(f"  {day}: kept {len(hits):,} / {n_all:,} hits from {pattern_prefixes}")
    hits["cluster_id"] = hits["motif_name"].str.split(".").str[-1]
    hits["midpoint"] = (hits["start"] + hits["end"]) // 2
    hits = hits.join(annot_df["tf_motif_id"], on="cluster_id")
    hits = hits.dropna(subset=["tf_motif_id"])
    if (hits[hit_score_col] < 0).any():
        raise ValueError(f"{day}: negative {hit_score_col} in {pattern_prefixes} hits")
    if dedup_same_label:
        n_before = len(hits)
        hits, removed = dedup_same_label_hits(hits, label_overlap_bp)
        removed["day"] = day
        dedup_removed.append(removed)
        print(
            f"  {day}: same-label dedup removed {n_before - len(hits):,} / {n_before:,} "
            f"hits ({100 * (n_before - len(hits)) / n_before:.1f}%)"
        )
    hits_per_day[day] = hits
    print(f"  {day}: {len(hits):,} annotated hits loaded")
# %%
# Load scE2G links per day; the day column keeps each day's hits matched to that day's links
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
    df["day"] = day
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
# Same-label dedup summary and motif-free gene covariates (inputs of 2.1)
os.makedirs(os.path.join(data_output_dir, "tables"), exist_ok=True)
if dedup_same_label:
    pd.concat(dedup_removed, ignore_index=True)[
        ["day", "tf_motif_id", "n_removed"]
    ].to_csv(
        os.path.join(data_output_dir, "tables", "label_dedup_removed_hits.tsv"),
        sep="\t",
        index=False,
    )
for et, e2g_sub in et_configs.items():
    cov = gene_covariates(e2g_sub, days)
    cov_path = os.path.join(data_output_dir, "data", f"{et}_gene_covariates.tsv.gz")
    cov.to_csv(cov_path, sep="\t", index=False, compression="gzip")
    print(f"  [{et}] covariates for {len(cov):,} gene-days: {cov_path}")

# %%
all_scores = {}
for et, e2g_sub in et_configs.items():
    print(f"\n  [{weight_strategy}/{et}] {len(e2g_sub):,} linked peaks")
    scores = _compute_scores(
        hits_per_day,
        e2g_sub,
        weight_strategy,
        hit_score_col,
        decay_dist,
        hit_fully_inside,
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
