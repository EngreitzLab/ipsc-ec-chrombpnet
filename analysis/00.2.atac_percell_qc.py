#!/usr/bin/env python3
"""
00.2.atac_percell_qc.py
Per-cell scATAC QC across the 10x multiome 5 timepoints (d0-d4): fragments per
cell, TSS enrichment, and nucleosome signal.

Why this exists alongside 00.1.per_cell_qc.py
---------------------------------------------
00.1.per_cell_qc.py computes TSS enrichment with a home-rolled center-vs-flank
insertion density ratio, documented there as "a" reasonable metric rather than
"the" canonical one. Meanwhile the IGVF submission pipeline already produced
ArchR-computed per-cell metrics for the same barcodes:

  IGVF submission scripts: sce2g-wtc11-ec-5timepoints/scripts/compute_atac_percell_qc.R
      -> ArchR createArrowFiles + getCellColData (nFrags, TSSEnrichment)
      -> nucleosome signal from fragment lengths (mono 147-294bp / NFR <147bp)

This script plots those ArchR metrics instead, so the figures match what was
submitted to the IGVF portal. Barcode sets are identical to the per-day RNA
h5ad files (d0: 6174, d1: 5911, d2: 6289, d3: 3846, d4: 5038 cells), so these
are the RNA-called cells, not all barcodes.

Inputs (one per day, read only)
  {qc_dir}/{day}/WTC11-EC_10xmultiome_5timepoints_{day}_percell_qc_report.tsv.gz
  columns: barcode, atac_fragment_count, atac_tss_enrichment,
           atac_nucleosome_signal, rna_umi_count, rna_gene_count,
           rna_pct_mito, rna_scrublet_doublet_score, pass_strict_qc

Outputs (analysis/00.2.atac_percell_qc/)
  plots/atac_qc_by_day.{pdf,png}          + _data.tsv   3 violin panels
  plots/tss_vs_fragments.{pdf,png}        + _data.tsv   per day knee scatter
  plots/atac_qc_pass_fraction.{pdf,png}   + _data.tsv   fraction over guides
  tables/atac_qc_summary_by_day.tsv                     per day descriptives
  tables/atac_qc_stats_by_day.tsv                       KW + pairwise MWU, BH
  CONCLUSIONS.txt

Run
  ml biology py-scanpy/1.10.2_py312
  python3 analysis/00.2.atac_percell_qc.py
"""

# %%
import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests

plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["axes.titlesize"] = 10
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["savefig.transparent"] = True

# %%
core_path = "/path/to/ipsc-ec-chrombpnet"
qc_dir = "/path/to/scE2G_qc/generated_for_submission"
qc_file_template = "WTC11-EC_10xmultiome_5timepoints_{day}_percell_qc_report.tsv.gz"

days = ["d0", "d1", "d2", "d3", "d4"]
day_colors = {
    "d0": "#A6A6A6",
    "d1": "#D3C72F",
    "d2": "#D37739",
    "d3": "#9B3A48",
    "d4": "#CD7986",
}

save_plots = True
random_seed = 0
n_jitter_max = 1500  # points drawn per day; violin shape stays visible

# Display guides only. These are drawn as reference lines and used for the
# pass fraction panel; no cell is filtered out of any distribution below.
guide_min_fragments = 1000
guide_min_tss_enrichment = 4

# Metrics to plot: column, axis label, whether to log10 transform
metrics = [
    ("atac_fragment_count", "ATAC fragments per cell (log10)", True),
    ("atac_tss_enrichment", "TSS enrichment (ArchR)", False),
    ("atac_nucleosome_signal", "Nucleosome signal", False),
]

output_path = os.path.join(core_path, "analysis", "00.2.atac_percell_qc")
plots_dir = os.path.join(output_path, "plots")
tables_dir = os.path.join(output_path, "tables")
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(tables_dir, exist_ok=True)

rng = np.random.default_rng(random_seed)


# %%
def save_figure(fig, stem):
    """Save a figure as both .pdf and .png under plots_dir."""
    if not save_plots:
        return
    for ext in ("pdf", "png"):
        path = os.path.join(plots_dir, f"{stem}.{ext}")
        fig.savefig(path)
    print(f"saved plot {os.path.join(plots_dir, stem)}.pdf/.png")


def save_plot_data(df, stem):
    """Save the exact values behind a plot next to it, per lab convention."""
    if not save_plots:
        return
    path = os.path.join(plots_dir, f"{stem}_data.tsv")
    df.to_csv(path, sep="\t", index=False)
    print(f"saved plot data {path}")


def rank_biserial(x, y):
    """Rank biserial correlation, the effect size paired with Mann Whitney U.

    Returns a value in [-1, 1]; positive means x tends to exceed y.
    """
    u = stats.mannwhitneyu(x, y, alternative="two-sided").statistic
    return 2.0 * u / (len(x) * len(y)) - 1.0


def epsilon_squared(groups, h_statistic):
    """Epsilon squared effect size for Kruskal Wallis (0 to 1)."""
    n_total = sum(len(g) for g in groups)
    return h_statistic / (n_total - 1)


# %%
frames = []
for day in days:
    path = os.path.join(qc_dir, day, qc_file_template.format(day=day))
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing per cell QC report for {day}: {path}")
    df = pd.read_csv(path, sep="\t")

    expected = {
        "barcode",
        "atac_fragment_count",
        "atac_tss_enrichment",
        "atac_nucleosome_signal",
    }
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"{day}: per cell QC report missing columns {missing}")
    if df["barcode"].duplicated().any():
        raise ValueError(f"{day}: duplicate barcodes in per cell QC report")

    df["day"] = day
    frames.append(df)
    print(f"[{day}] loaded {len(df)} cells from {os.path.basename(path)}")

qc = pd.concat(frames, ignore_index=True)
qc["day"] = pd.Categorical(qc["day"], categories=days, ordered=True)
qc["log10_atac_fragment_count"] = np.log10(qc["atac_fragment_count"].clip(lower=1))

# Sanity checks on the loaded metrics
if (qc["atac_fragment_count"] < 0).any():
    raise ValueError("negative ATAC fragment counts found")
n_na = qc[[m for m, _, _ in metrics]].isna().sum()
if n_na.any():
    print("warning: missing values per metric (dropped test wise)")
    print(n_na[n_na > 0].to_string())

print(f"\ntotal cells across d0-d4: {len(qc)}")


# %%
summary_rows = []
for column, label, _ in metrics:
    for day in days:
        vals = qc.loc[qc["day"] == day, column].dropna()
        summary_rows.append(
            {
                "metric": column,
                "day": day,
                "n_cells": len(vals),
                "median": vals.median(),
                "mean": vals.mean(),
                "q25": vals.quantile(0.25),
                "q75": vals.quantile(0.75),
                "min": vals.min(),
                "max": vals.max(),
            }
        )

summary_df = pd.DataFrame(summary_rows)
summary_path = os.path.join(tables_dir, "atac_qc_summary_by_day.tsv")
summary_df.to_csv(summary_path, sep="\t", index=False)
print(f"saved table {summary_path}")
print(summary_df.pivot(index="day", columns="metric", values="median").to_string())


# %%
# Non parametric testing: Kruskal Wallis across all five days per metric, then
# pairwise Mann Whitney U. BH correction is applied within each metric across
# its ten pairwise tests, so the reported q values are per metric family.
stat_rows = []
for column, label, _ in metrics:
    groups = [qc.loc[qc["day"] == day, column].dropna().values for day in days]
    h_stat, kw_p = stats.kruskal(*groups)
    stat_rows.append(
        {
            "metric": column,
            "test": "kruskal_wallis",
            "group_1": "all_days",
            "group_2": "all_days",
            "n_1": sum(len(g) for g in groups),
            "n_2": np.nan,
            "statistic": h_stat,
            "p_raw": kw_p,
            "p_bh": np.nan,
            "effect_size": epsilon_squared(groups, h_stat),
            "effect_size_type": "epsilon_squared",
        }
    )

    pair_rows = []
    for day_a, day_b in itertools.combinations(days, 2):
        x = qc.loc[qc["day"] == day_a, column].dropna().values
        y = qc.loc[qc["day"] == day_b, column].dropna().values
        result = stats.mannwhitneyu(x, y, alternative="two-sided")
        pair_rows.append(
            {
                "metric": column,
                "test": "mann_whitney_u",
                "group_1": day_a,
                "group_2": day_b,
                "n_1": len(x),
                "n_2": len(y),
                "statistic": result.statistic,
                "p_raw": result.pvalue,
                "effect_size": rank_biserial(x, y),
                "effect_size_type": "rank_biserial",
            }
        )

    pair_df = pd.DataFrame(pair_rows)
    pair_df["p_bh"] = multipletests(pair_df["p_raw"], method="fdr_bh")[1]
    stat_rows.extend(pair_df.to_dict("records"))

stats_df = pd.DataFrame(stat_rows)[
    [
        "metric",
        "test",
        "group_1",
        "group_2",
        "n_1",
        "n_2",
        "statistic",
        "p_raw",
        "p_bh",
        "effect_size",
        "effect_size_type",
    ]
]
stats_path = os.path.join(tables_dir, "atac_qc_stats_by_day.tsv")
stats_df.to_csv(stats_path, sep="\t", index=False)
print(f"saved table {stats_path}")
print(
    stats_df[stats_df["test"] == "kruskal_wallis"][
        ["metric", "statistic", "p_raw", "effect_size"]
    ].to_string(index=False)
)


# %%
# Panel 1: distribution of each ATAC metric by day
fig, axes = plt.subplots(1, 3, figsize=(11, 5))

for ax, (column, label, use_log) in zip(axes, metrics):
    plot_column = f"log10_{column}" if use_log else column
    plot_df = qc[["day", plot_column]].dropna()

    sns.violinplot(
        data=plot_df,
        x="day",
        y=plot_column,
        order=days,
        hue="day",
        hue_order=days,
        palette=day_colors,
        legend=False,
        cut=0,
        inner=None,
        linewidth=0.8,
        saturation=1,
        ax=ax,
    )

    # Subsample the jitter so the violin outline stays readable
    jitter_parts = []
    for day in days:
        vals = plot_df.loc[plot_df["day"] == day, plot_column].values
        if len(vals) > n_jitter_max:
            vals = rng.choice(vals, size=n_jitter_max, replace=False)
        jitter_parts.append(pd.DataFrame({"day": day, plot_column: vals}))
    jitter_df = pd.concat(jitter_parts, ignore_index=True)

    sns.stripplot(
        data=jitter_df,
        x="day",
        y=plot_column,
        order=days,
        color="black",
        alpha=0.25,
        size=0.9,
        jitter=0.22,
        ax=ax,
    )

    # Median as a black line, plus median and n above each violin.
    # Headroom on top for the annotation, a little margin below so the
    # violins do not terminate flush against the axis.
    y_low, y_high = ax.get_ylim()
    y_range = y_high - y_low
    ax.set_ylim(y_low - 0.04 * y_range, y_high + 0.18 * y_range)
    for i, day in enumerate(days):
        vals = plot_df.loc[plot_df["day"] == day, plot_column]
        median = vals.median()
        ax.hlines(median, i - 0.35, i + 0.35, color="black", linewidth=1.6, zorder=5)
        display_median = 10**median if use_log else median
        ax.text(
            i,
            y_high + 0.02 * (y_high - y_low),
            f"{display_median:.3g}\nn={len(vals)}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    if column == "atac_fragment_count":
        ax.axhline(
            np.log10(guide_min_fragments), color="gray", linestyle=":", linewidth=1
        )
    if column == "atac_tss_enrichment":
        ax.axhline(guide_min_tss_enrichment, color="gray", linestyle=":", linewidth=1)

    ax.set_ylabel(label)
    ax.set_xlabel("")

fig.suptitle("Per cell scATAC quality metrics by differentiation day", y=1.02)
fig.tight_layout()
save_figure(fig, "atac_qc_by_day")

panel_data = qc[
    ["barcode", "day", "atac_fragment_count", "log10_atac_fragment_count"]
    + ["atac_tss_enrichment", "atac_nucleosome_signal"]
]
save_plot_data(panel_data, "atac_qc_by_day")


# %%
# Panel 2: the standard scATAC knee, TSS enrichment against fragment depth
fig, axes = plt.subplots(1, 5, figsize=(16, 3.4), sharex=True, sharey=True)

for ax, day in zip(axes, days):
    sub = qc[(qc["day"] == day)].dropna(
        subset=["log10_atac_fragment_count", "atac_tss_enrichment"]
    )
    ax.scatter(
        sub["log10_atac_fragment_count"],
        sub["atac_tss_enrichment"],
        s=2,
        alpha=0.25,
        color=day_colors[day],
        edgecolors="none",
        rasterized=True,
    )
    ax.axvline(np.log10(guide_min_fragments), color="gray", linestyle=":", linewidth=1)
    ax.axhline(guide_min_tss_enrichment, color="gray", linestyle=":", linewidth=1)

    n_pass = int(
        (
            (sub["atac_fragment_count"] >= guide_min_fragments)
            & (sub["atac_tss_enrichment"] >= guide_min_tss_enrichment)
        ).sum()
    )
    ax.set_title(f"{day}; n={len(sub)}")
    ax.text(
        0.97,
        0.95,
        f"over guides\n{100 * n_pass / len(sub):.1f}%",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
    )
    ax.set_xlabel("ATAC fragments per cell (log10)")

axes[0].set_ylabel("TSS enrichment (ArchR)")
fig.tight_layout()
save_figure(fig, "tss_vs_fragments")

save_plot_data(
    qc[
        [
            "barcode",
            "day",
            "atac_fragment_count",
            "log10_atac_fragment_count",
            "atac_tss_enrichment",
        ]
    ],
    "tss_vs_fragments",
)


# %%
# Panel 3: fraction of cells above both display guides, by day
pass_rows = []
for day in days:
    sub = qc[qc["day"] == day]
    over_frags = sub["atac_fragment_count"] >= guide_min_fragments
    over_tsse = sub["atac_tss_enrichment"] >= guide_min_tss_enrichment
    pass_rows.append(
        {
            "day": day,
            "n_cells": len(sub),
            "n_over_fragment_guide": int(over_frags.sum()),
            "n_over_tss_guide": int(over_tsse.sum()),
            "n_over_both_guides": int((over_frags & over_tsse).sum()),
            "frac_over_both_guides": float((over_frags & over_tsse).mean()),
        }
    )

pass_df = pd.DataFrame(pass_rows)

fig, ax = plt.subplots(figsize=(4.5, 4.5))
ax.bar(
    pass_df["day"],
    100 * pass_df["frac_over_both_guides"],
    color=[day_colors[d] for d in pass_df["day"]],
    edgecolor="black",
    linewidth=0.8,
)
for i, row in pass_df.iterrows():
    ax.text(
        i,
        100 * row["frac_over_both_guides"] + 1,
        f"{100 * row['frac_over_both_guides']:.1f}%\nn={row['n_cells']}",
        ha="center",
        va="bottom",
        fontsize=8,
    )
ax.set_ylim(0, 108)
ax.set_ylabel("Cells above both guides (%)")
ax.set_xlabel("")
ax.set_title(
    f"Cells with fragments >= {guide_min_fragments}\n"
    f"and TSS enrichment >= {guide_min_tss_enrichment}"
)
fig.tight_layout()
save_figure(fig, "atac_qc_pass_fraction")
save_plot_data(pass_df, "atac_qc_pass_fraction")


# %%
# Conclusions
kw = stats_df[stats_df["test"] == "kruskal_wallis"].set_index("metric")
pairwise = stats_df[stats_df["test"] == "mann_whitney_u"]
n_sig = int((pairwise["p_bh"] < 0.05).sum())

median_tbl = summary_df.pivot(index="day", columns="metric", values="median")
frag_medians = median_tbl["atac_fragment_count"]
tsse_medians = median_tbl["atac_tss_enrichment"]

conclusions = f"""CONCLUSIONS: per cell scATAC QC across d0-d4
Generated by analysis/00.2.atac_percell_qc.py

INPUT
  ArchR derived per cell metrics from the IGVF submission pipeline:
  {qc_dir}/<day>/{qc_file_template.format(day="<day>")}
  Produced by compute_atac_percell_qc.R (ArchR createArrowFiles with
  minTSS=0, minFrags=0, so no ATAC side cell calling was applied; the barcode
  set is the RNA called cells). Nucleosome signal is mono nucleosomal
  (147-294 bp) over nucleosome free (<147 bp) fragment counts.

SUMMARY OF FINDINGS
  Cells per day: {', '.join(f'{d}={int(summary_df[(summary_df.metric == "atac_fragment_count") & (summary_df.day == d)]["n_cells"].iloc[0])}' for d in days)}
  Total cells: {len(qc)}

  Median ATAC fragments per cell:
{frag_medians.round(1).to_string()}

  Median TSS enrichment (ArchR):
{tsse_medians.round(2).to_string()}

  Kruskal Wallis across the five days (raw p, epsilon squared effect size):
    fragments per cell : p={kw.loc["atac_fragment_count", "p_raw"]:.3g}, eps2={kw.loc["atac_fragment_count", "effect_size"]:.4f}
    TSS enrichment     : p={kw.loc["atac_tss_enrichment", "p_raw"]:.3g}, eps2={kw.loc["atac_tss_enrichment", "effect_size"]:.4f}
    nucleosome signal  : p={kw.loc["atac_nucleosome_signal", "p_raw"]:.3g}, eps2={kw.loc["atac_nucleosome_signal", "effect_size"]:.4f}

  Pairwise Mann Whitney U: {n_sig} of {len(pairwise)} day pairs have BH q < 0.05
  across the three metrics. Raw and BH corrected p values, plus rank biserial
  effect sizes, are in tables/atac_qc_stats_by_day.tsv.

  Fraction of cells above both display guides
  (fragments >= {guide_min_fragments} and TSS enrichment >= {guide_min_tss_enrichment}):
{pass_df.set_index("day")["frac_over_both_guides"].mul(100).round(1).to_string()}

LIMITATIONS
  - With 3800-6300 cells per day, Mann Whitney U detects differences far
    smaller than any QC relevant shift. Read the rank biserial effect sizes,
    not the p values: values near 0 mean the day distributions are nearly
    interchangeable regardless of how small q is.
  - No ATAC side cell calling was applied upstream, so low fragment barcodes
    that passed RNA cell calling remain in every distribution here. The guide
    lines are drawn for reference and were not used to filter.
  - The {guide_min_fragments} fragment and TSS enrichment {guide_min_tss_enrichment}
    guides are conventional scATAC values, not thresholds derived from this
    dataset or agreed for this project.
  - Nucleosome signal is computed from raw fragment lengths without duplicate
    or mitochondrial filtering, so it is comparable across days here but not
    directly comparable to Signac or 10x reported values.
  - Depth differs by day, and TSS enrichment is depth sensitive at low
    coverage; part of any TSS enrichment difference between days may reflect
    sequencing depth rather than chromatin quality.

SUGGESTED HYPOTHESES
  - If fragment depth declines across differentiation, it likely tracks
    library level differences between the two 10x channels or per timepoint
    CMO loading rather than biology; the channel column in the RNA h5ad
    obs can test this directly.
  - Days with a lower fraction above the guides deserve a look at whether
    their low quality cells concentrate in particular RNA leiden clusters.

POTENTIAL NEXT STEPS
  - Stratify each metric by pass_strict_qc and by channel to separate
    technical from timepoint effects.
  - Compare these ArchR TSS enrichment values against the home rolled metric
    in analysis/00.1.per_cell_qc/tables/tss_enrichment_per_cell.tsv to
    quantify how far that approximation sits from the canonical calculation.
  - If an ATAC side cell filter is wanted for downstream work, choose the
    fragment and TSS enrichment cutoffs from the knee in
    plots/tss_vs_fragments.pdf rather than the conventional defaults used here.
"""

conclusions_path = os.path.join(output_path, "CONCLUSIONS.txt")
with open(conclusions_path, "w") as handle:
    handle.write(conclusions)
print(f"saved {conclusions_path}")
print("\n" + conclusions)

# %%
