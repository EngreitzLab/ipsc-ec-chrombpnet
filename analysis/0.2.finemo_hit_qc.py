#!/usr/bin/env python3
# Purpose: Explore per-motif CWM similarity from FiNeMo reports,
#          decide on a quality threshold, and produce filtered hit files.
#
# Reads:   results/finemo_unified/{day}_{peak_type}/finemo_report/motif_report.tsv
#          results/finemo_unified/{day}_{peak_type}/hits_unique.tsv
#          (report produced by pipeline step 11; hits_unique.tsv by step 10)
#
# Filter strategy: per-motif per-day cwm_similarity from motif_report.tsv.
#   cwm_similarity = cosine similarity between the AVERAGE contribution score
#   pattern across all instances of a motif in a day and the reference MODISCO CWM.
#   This is the only independent quality metric -- hit_similarity in hits_unique.tsv
#   is gated by the lambda threshold at calling time (finemo_alpha=0.8) and is
#   therefore >= lambda for every surviving hit by construction.
#
# Outputs (per day in results/finemo_unified/{day}_{peak_type}/):
#   hits_filtered.tsv       - hits passing cwm_similarity threshold
#   hits_filtered.bed.gz    - tabix-indexed BED for downstream use
#
# Analysis outputs:
#   analysis/0.2.finemo_hit_qc/plots/  - QC figures
#   analysis/0.2.finemo_hit_qc/tables/ - per-motif similarity summary
#
# Before running:
#   conda activate <your_analysis_env>
# %%
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["font.size"] = 14
mpl.rcParams["axes.labelsize"] = 14
mpl.rcParams["axes.titlesize"] = 14
mpl.rcParams["xtick.labelsize"] = 14
mpl.rcParams["ytick.labelsize"] = 14
mpl.rcParams["legend.fontsize"] = 14
mpl.rcParams["figure.dpi"] = 100
mpl.rcParams["savefig.dpi"] = 300
mpl.rcParams["savefig.bbox"] = "tight"
mpl.rcParams["savefig.transparent"] = True


# %%
def dedup_overlapping_hits(hits, overlap_bp=3):
    """
    Within each (chr, motif_name) group, keep only the hit with the highest
    hit_similarity when hits overlap by more than overlap_bp bp. Uses a
    best-first greedy pass: sort by hit_similarity descending, then suppress
    any lower-similarity hit that overlaps an already-kept hit.
    """
    if hits.empty:
        return hits
    parts = []
    for _, grp in hits.groupby(["chr", "motif_name"], sort=False):
        grp = grp.sort_values("hit_similarity", ascending=False)
        starts = grp["start"].to_numpy()
        ends = grp["end"].to_numpy()
        keep = np.ones(len(grp), dtype=bool)
        for i in range(len(grp)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(grp)):
                if (
                    keep[j]
                    and min(ends[i], ends[j]) - max(starts[i], starts[j]) > overlap_bp
                ):
                    keep[j] = False
        parts.append(grp[keep])
    return pd.concat(parts, ignore_index=True) if parts else hits.iloc[:0]


# %%
core_path = "/path/to/ipsc-ec-chrombpnet"
peak_type = "all"
days = ["d0", "d1", "d2", "d3", "d4"]

# Threshold for cwm_similarity: motifs below this in a given day are excluded
cwm_similarity_threshold = 0.8
# Hits of the same motif overlapping by more than this many bp are deduplicated
OVERLAP_BP = 3

save_plots = True
save_outputs = True

finemo_dir = os.path.join(core_path, "results", "finemo_unified")
output_path = os.path.join(core_path, "analysis", "0.2.finemo_hit_qc")
plots_dir = os.path.join(output_path, "plots")
tables_dir = os.path.join(output_path, "tables")
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(tables_dir, exist_ok=True)

# %%
# Load per-day motif_report.tsv files
report_frames = []
for day in days:
    report_path = os.path.join(
        finemo_dir, f"{day}_{peak_type}", "finemo_report", "motif_report.tsv"
    )
    if not os.path.exists(report_path):
        print(f"  [WARN] missing report for {day}: {report_path}")
        continue
    df = pd.read_csv(report_path, sep="\t")
    df["day"] = day
    report_frames.append(df)
    print(f"  {day}: {len(df)} motifs loaded")

reports = pd.concat(report_frames, ignore_index=True)

# %%
# Distribution of cwm_similarity per day
from scipy.stats import gaussian_kde

fig, ax = plt.subplots(1, 1, figsize=(5, 5))

data_by_day = [
    reports.loc[reports["day"] == d, "cwm_similarity"].dropna() for d in days
]

violin_half_width = 0.3
point_offset = 0.12

for i, (day, vals) in enumerate(zip(days, data_by_day)):
    if len(vals) < 2:
        continue

    # Half violin (left of center)
    kde = gaussian_kde(vals)
    y_grid = np.linspace(vals.min(), vals.max(), 300)
    kde_vals = kde(y_grid)
    kde_vals = kde_vals / kde_vals.max() * violin_half_width
    ax.fill_betweenx(y_grid, i - kde_vals, i, alpha=0.6, color="gray", lw=0)
    ax.plot(i - kde_vals, y_grid, color="black", lw=0.5)

    # IQR box + whiskers (centered)
    q25, q50, q75 = np.percentile(vals, [25, 50, 75])
    iqr = q75 - q25
    w_lo = max(vals.min(), q25 - 1.5 * iqr)
    w_hi = min(vals.max(), q75 + 1.5 * iqr)
    box_hw = 0.06
    ax.vlines(i, w_lo, w_hi, color="black", lw=0.8, zorder=4)
    ax.add_patch(
        plt.Rectangle(
            (i - box_hw, q25),
            2 * box_hw,
            iqr,
            facecolor="white",
            edgecolor="black",
            lw=1,
            zorder=5,
        )
    )
    ax.hlines(q50, i - box_hw, i + box_hw, color="black", lw=1.5, zorder=6)

    # Jittered points (right of center)
    rng = np.random.default_rng(i)
    jitter = rng.uniform(point_offset, point_offset + 0.12, size=len(vals))
    ax.scatter(i + jitter, vals, s=5, alpha=0.5, color="gray", zorder=3)
    ax.text(
        i,
        1.02,
        f"{q50:.2f}\nn={len(vals)}",
        ha="center",
        va="bottom",
        fontsize=8,
    )

ax.axhline(
    cwm_similarity_threshold,
    color="black",
    lw=1,
    ls="--",
    label=f"threshold = {cwm_similarity_threshold}",
)
ax.set_xticks(range(len(days)))
ax.set_xticklabels(days)
ax.set_ylim(bottom=0, top=1.15)
ax.set_xlabel("Day")
ax.set_ylabel("CWM similarity\n(motif average vs reference)")
ax.legend(fontsize=8)

if save_plots:
    fig.savefig(os.path.join(plots_dir, "cwm_similarity_per_day.pdf"))
    fig.savefig(os.path.join(plots_dir, "cwm_similarity_per_day.png"))
    reports[["day", "motif_name", "cwm_similarity"]].to_csv(
        os.path.join(plots_dir, "cwm_similarity_per_day_data.tsv"),
        sep="\t",
        index=False,
    )
# %%
# Per-motif cwm_similarity across days
# Useful to distinguish: motifs that are consistently bad (low everywhere, likely noise)
# vs motifs that are good in most days but low in one (potentially still real)
motif_summary = reports.pivot_table(
    index="motif_name", columns="day", values="cwm_similarity"
).reindex(columns=days)
# %%
motif_summary["min_similarity"] = motif_summary.min(axis=1)
motif_summary["n_days_passing"] = (motif_summary[days] >= cwm_similarity_threshold).sum(
    axis=1
)
motif_summary = motif_summary.sort_values("min_similarity")
# %%
if save_outputs:
    motif_summary.to_csv(
        os.path.join(tables_dir, "motif_cwm_similarity_summary.tsv"), sep="\t"
    )
# %%
# Apply threshold and write filtered hits
for day in days:
    day_dir = os.path.join(finemo_dir, f"{day}_{peak_type}")
    report_path = os.path.join(day_dir, "finemo_report", "motif_report.tsv")
    hits_path = os.path.join(day_dir, "hits_unique.tsv")
    filtered_tsv = os.path.join(output_path, f"hits_filtered_{day}.tsv")
    filtered_bed_tmp = os.path.join(output_path, f"hits_filtered_{day}.bed")
    filtered_bed = os.path.join(output_path, f"hits_filtered_{day}.bed.gz")

    if not os.path.exists(report_path) or not os.path.exists(hits_path):
        print(f"  [SKIP] {day}: report or hits_unique.tsv not found")
        continue

    report = pd.read_csv(report_path, sep="\t")
    hits = pd.read_csv(hits_path, sep="\t")
    bad_motifs = set(
        report.loc[report["cwm_similarity"] < cwm_similarity_threshold, "motif_name"]
    )
    mask = ~hits["motif_name"].isin(bad_motifs)
    hits_filtered = hits[mask].copy()

    n_after_cwm = len(hits_filtered)
    hits_filtered = dedup_overlapping_hits(hits_filtered, overlap_bp=OVERLAP_BP)

    print(
        f"  {day}: {len(hits):,} hits"
        f" -> {n_after_cwm:,} after cwm_similarity filter"
        f" ({len(bad_motifs)}/{len(report)} motifs flagged)"
        f" -> {len(hits_filtered):,} after overlap dedup (>{OVERLAP_BP} bp)"
    )

    if not save_outputs:
        continue

    hits_filtered.to_csv(filtered_tsv, sep="\t", index=False)

    bed_cols = ["chr", "start", "end", "motif_name", "hit_similarity", "strand"]
    hits_filtered[bed_cols].to_csv(
        filtered_bed_tmp, sep="\t", index=False, header=False
    )
# %%
