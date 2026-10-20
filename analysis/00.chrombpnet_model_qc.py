"""
00.chrombpnet_model_qc.py
Interactive QC for ChromBPNet full models across iPSC-EC differentiation timepoints.
Moved to the front of the analysis numbering (was 6.) since this is the data
overview / QC panel that should run before the motif/gene-linking analyses.

Sections:
  1.  ATAC depth  -- fragment count per day from the pseudobulk ATAC fragment files
  2.  Cell counts -- cells per day from the per-day scRNA-seq files
  3.  Plot 01     -- sequencing depth and cell counts (two barplots)
  4.  Scatter     -- load predicted vs observed log-counts (all folds per day)
  5.  Plot 02     -- density scatter, one panel per timepoint
  6.  Metrics     -- Pearson r and median JSD from model_metrics.tsv
  7.  Plot 03     -- performance barplots (bar = fold mean, points = per fold)
  8.  MoDISco     -- load pattern and seqlet counts from per-day h5 files
  9.  Plot 04     -- MoDISco patterns and seqlets per day
  10. Fi-NeMo     -- load QCed hits per day
  11. Plot 07     -- hits per peak distribution per day (boxplot)

Run inside the chrombpnet conda environment:
  conda activate chrombpnet
"""

# %%
import os
import subprocess

import anndata as ad
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyBigWig
from scipy.interpolate import interpn
from scipy.stats import pearsonr

mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["font.size"] = 14
mpl.rcParams["axes.labelsize"] = 14
mpl.rcParams["axes.titlesize"] = 14
mpl.rcParams["xtick.labelsize"] = 14
mpl.rcParams["ytick.labelsize"] = 14
mpl.rcParams["savefig.dpi"] = 300
mpl.rcParams["savefig.bbox"] = "tight"
mpl.rcParams["savefig.transparent"] = True

# %%
# params
core_path = "/path/to/ipsc-ec-chrombpnet"
# No BAM files exist in this project anymore; use fragment counts from the
# same pseudobulk ATAC files the ChromBPNet models were trained on instead
# (per pipeline/dataset_config.sh's frag_path) as the depth proxy.
frag_path = "/path/to/10x_5_timepoints/Pseudobulks/ATAC"
full_model_dir = os.path.join(core_path, "results", "full_models")
preprocessing_dir = os.path.join(core_path, "results", "preprocessing")
# Per-day scRNA-seq (upstream processing); the old seurat_rna.h5ad (single
# object, merged across days via a "cell.state" column) is retired.
gex_dir = "/path/to/10x_5_timepoints/Pseudobulks/RNA"
metrics_tsv = os.path.join(
    core_path, "results", "plots", "full_model_qc", "model_metrics.tsv"
)
annotation_tsv = os.path.join(
    core_path,
    "analysis",
    "0.0.motif_id_annotation",
    "tables",
    "cluster_to_tf_motif_id_mapping.tsv",
)
finemo_qc_dir = os.path.join(core_path, "analysis", "0.2.finemo_hit_qc")
output_dir = os.path.join(core_path, "analysis", "00.chrombpnet_model_qc")
plots_dir = os.path.join(output_dir, "plots")
tables_dir = os.path.join(output_dir, "tables")

os.makedirs(plots_dir, exist_ok=True)
os.makedirs(tables_dir, exist_ok=True)

days = ["d0", "d1", "d2", "d3", "d4"]
folds = ["0", "1", "2", "3", "4"]
peak_type = "all"
outputlen = 1000

day_colors = {
    "d0": "#A6A6A6",
    "d1": "#D3C72F",
    "d2": "#D37739",
    "d3": "#9B3A48",
    "d4": "#CD7986",
}
day_label = {d: d.upper() for d in ["d0", "d1", "d2", "d3", "d4"]}

# test chromosomes per fold (standard ChromBPNet splits)
TEST_CHROMS = {
    "0": ["chr1", "chr3", "chr6"],
    "1": ["chr2", "chr8", "chr9", "chr16"],
    "2": ["chr4", "chr11", "chr12", "chr15", "chrY"],
    "3": ["chr5", "chr10", "chr14", "chr18", "chr20", "chr22"],
    "4": ["chr7", "chr13", "chr17", "chr19", "chr21", "chrX"],
}
NARROWPEAK_COLS = [
    "chr",
    "start",
    "end",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "summit",
]

save_plots = True
rng = np.random.default_rng(42)

# %%
# helpers


def save_fig(fig, stem):
    if save_plots:
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(plots_dir, f"{stem}.{ext}"))


def save_data(df, stem):
    if save_plots:
        df.to_csv(os.path.join(plots_dir, f"{stem}_data.tsv"), sep="\t", index=False)


def density_scatter(ax, x, y, bins=50):
    """Density-coloured scatter using 2D histogram interpolation."""
    data, xe, ye = np.histogram2d(x, y, bins=bins, density=True)
    z = interpn(
        (0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1])),
        data,
        np.vstack([x, y]).T,
        method="splinef2d",
        bounds_error=False,
    )
    z = np.nan_to_num(z)
    idx = z.argsort()
    sc = ax.scatter(
        x[idx],
        y[idx],
        c=z[idx],
        s=5,
        alpha=0.7,
        linewidths=0,
        cmap="viridis",
        rasterized=True,
    )
    return sc


def get_cts(peaks_df, bw, outputlen=1000):
    """Fetch per-peak cut-site counts from a BigWig over summit-centred windows."""
    half = outputlen // 2
    bw_chroms = bw.chroms()
    result = []
    for _, row in peaks_df.iterrows():
        summit = int(row["start"]) + int(row["summit"])
        s, e = summit - half, summit + half
        if s < 0 or e > bw_chroms.get(row["chr"], 0):
            result.append(np.zeros(outputlen))
            continue
        vals = bw.values(row["chr"], s, e)
        result.append(np.nan_to_num(vals))
    return np.array(result)


def atac_fragment_count(frag_file):
    """Return the number of fragments in a bgzipped fragments.tsv.gz file."""
    try:
        result = subprocess.run(
            f"zcat {frag_file} | wc -l",
            shell=True,
            capture_output=True,
            text=True,
            check=True,
        )
        return int(result.stdout.strip())
    except Exception as err:
        print(f"  fragment count error for {frag_file}: {err}")
        return np.nan


def load_day_scatter(day):
    peaks_bed = os.path.join(
        preprocessing_dir,
        day,
        f"{day}_{peak_type}_peaks_no_blacklist.narrowPeak",
    )
    if not os.path.exists(peaks_bed):
        print(f"  peaks not found for {day}")
        return None, None

    peaks_df = pd.read_csv(peaks_bed, sep="\t", names=NARROWPEAK_COLS)
    obs_all, pred_all = [], []

    for fold in folds:
        tag = f"{day}_{peak_type}_fold_{fold}"
        h5_file = os.path.join(
            full_model_dir, tag, "evaluation", "chrombpnet_predictions.h5"
        )
        bw_file = os.path.join(full_model_dir, tag, "auxiliary", "data_unstranded.bw")

        if not (os.path.exists(h5_file) and os.path.exists(bw_file)):
            print(f"  missing files for {tag}, skipping fold")
            continue

        test_chroms = TEST_CHROMS[fold]

        with h5py.File(h5_file, "r") as h5:
            pred_logcts = h5["predictions"]["logcounts"][:].flatten()
            chroms = np.array(
                [
                    c.decode() if isinstance(c, bytes) else c
                    for c in h5["coords"]["coords_chrom"][:]
                ]
            )

        mask = np.isin(chroms, test_chroms)
        pred_logcts = pred_logcts[mask]
        peaks_sub = peaks_df[peaks_df["chr"].isin(test_chroms)].reset_index(drop=True)

        try:
            bw = pyBigWig.open(bw_file)
            obs_cts = get_cts(peaks_sub, bw, outputlen)
            bw.close()
        except Exception as exc:
            print(f"  BigWig error for {tag}: {exc}")
            continue

        obs_logcts = np.log(np.sum(obs_cts, axis=-1) + 1)
        n = min(len(obs_logcts), len(pred_logcts))
        obs_all.append(obs_logcts[:n])
        pred_all.append(pred_logcts[:n])

    if not obs_all:
        return None, None
    return np.concatenate(obs_all), np.concatenate(pred_all)


def bar_with_points(ax, days_list, metric_col, df, ylabel, title, ylim=None):
    x = np.arange(len(days_list))
    colors = [day_colors[d] for d in days_list]
    means = [df.loc[df["day"] == d, metric_col].mean() for d in days_list]
    ax.bar(x, means, color=colors, alpha=1, zorder=2)
    for xi, day in enumerate(days_list):
        vals = df.loc[df["day"] == day, metric_col].values
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(
            xi + jitter,
            vals,
            color="black",
            s=18,
            zorder=5,
            linewidths=0,
            alpha=1,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([day_label[d] for d in days_list])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)


def day_neg_tfs(day):
    """Return list of TF annotations for neg patterns present in a given day."""
    rows = annot_df[
        (annot_df["posneg"] == "neg")
        & annot_df["days_present"].apply(lambda s: day in str(s).split(","))
    ]
    return [str(t).strip() for t in rows["TFs"] if pd.notna(t)]


def neg_label(pattern_name):
    """Return TF annotation for a neg pattern name, or the raw name if not found."""
    try:
        cid = int(pattern_name.split(".")[-1])
        tfs = neg_annot.loc[cid, "TFs"] if cid in neg_annot.index else ""
        return tfs if pd.notna(tfs) and tfs else pattern_name
    except (ValueError, KeyError):
        return pattern_name


# %%
# 1. Sequencing depth: ATAC fragment count per day
frag_rows = []
for day in days:
    frag_file = os.path.join(frag_path, f"{day}.fragments.tsv.gz")
    n_frags = atac_fragment_count(frag_file)
    frag_rows.append({"day": day, "n_fragments": n_frags})
    print(f"  {day}: {n_frags:,}" if not np.isnan(n_frags) else f"  {day}: NA")

frag_df = pd.DataFrame(frag_rows)

# %%
# 2. Cell counts per day from the per-day scRNA-seq files (each file
# is already one day, fully QC'd/doublet-filtered -- no cell.state mapping needed)
cell_count_rows = []
for day in days:
    _rna = ad.read_h5ad(os.path.join(gex_dir, f"rna_{day}.h5ad"), backed="r")
    cell_count_rows.append({"day": day, "n_cells": _rna.n_obs})
    _rna.file.close()
cell_counts = pd.DataFrame(cell_count_rows)
print(cell_counts[["day", "n_cells"]].to_string(index=False))

stats_df = frag_df.merge(cell_counts[["day", "n_cells"]], on="day", how="left")
stats_df.to_csv(os.path.join(tables_dir, "frag_cell_stats.tsv"), sep="\t", index=False)

# %%
# 3. Plot 01: sequencing depth and cell counts
fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
colors = [day_colors[d] for d in stats_df["day"]]

xlabels = [day_label[d] for d in stats_df["day"]]

ax = axes[0]
bars = ax.bar(xlabels, stats_df["n_fragments"] / 1e6, color=colors)
ax.set_ylabel("# ATAC fragments, $10^6$")
ax.set_title("Sequencing depth\n")
for bar, val in zip(bars, stats_df["n_fragments"]):
    if not np.isnan(val):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.01,
            f"{val / 1e6:.0f}M",
            ha="center",
            va="bottom",
            fontsize=12,
        )

ax = axes[1]
bars = ax.bar(xlabels, stats_df["n_cells"], color=colors)
ax.set_ylabel("# cells")
for bar, val in zip(bars, stats_df["n_cells"]):
    if not np.isnan(val):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.01,
            f"{int(val)}",
            ha="center",
            va="bottom",
            fontsize=12,
        )

fig.tight_layout()
save_fig(fig, "01_frag_cell_stats")
save_data(stats_df, "01_frag_cell_stats")
plt.show()

# %%
# 4. Load predicted vs observed log-counts per day (all folds combined)
# each fold covers distinct test chromosomes, so combining gives genome-wide coverage

scatter_data = {}
for day in days:
    print(f"Loading {day}...")
    obs, pred = load_day_scatter(day)
    scatter_data[day] = (obs, pred)

# %%
# 5. Plot 02: predicted vs observed scatter, one panel per timepoint
fig, axes = plt.subplots(1, len(days), figsize=(3.5 * len(days), 3.5))
scatter_rows = []

for ax, day in zip(axes, days):
    obs, pred = scatter_data[day]
    if obs is None:
        ax.set_title(f"{day_label[day]}\n(no data)")
        continue

    ok = ~(np.isnan(obs) | np.isnan(pred))
    obs_ok, pred_ok = obs[ok], pred[ok]
    sc = density_scatter(ax, obs_ok, pred_ok)

    r = pearsonr(obs_ok, pred_ok)[0]
    ax.set_title(day_label[day], fontsize=16)
    ax.set_xlabel("Observed log(counts+1)")
    if ax is axes[0]:
        ax.set_ylabel("Predicted log-counts")
    ax.text(
        0.05,
        0.95,
        f"r = {r:.3f}",
        transform=ax.transAxes,
        fontsize=12,
        va="top",
    )
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Density", fontsize=10)
    cb.ax.tick_params(labelsize=9)

    scatter_rows.append(
        pd.DataFrame({"day": day, "obs_log_counts": obs_ok, "pred_log_counts": pred_ok})
    )

fig.tight_layout()
save_fig(fig, "02_scatter_pred_obs")
if scatter_rows:
    save_data(pd.concat(scatter_rows, ignore_index=True), "02_scatter_pred_obs")
plt.show()

# %%
# 6. Load model performance metrics (Pearson r, median JSD per day x fold)
metrics_df = pd.read_csv(metrics_tsv, sep="\t")
metrics_df["fold"] = metrics_df["fold"].astype(str)
print(metrics_df[["day", "fold", "pearsonr", "median_jsd"]].to_string(index=False))

# %%
# 7. Plot 03: performance barplots (bars = fold mean, points = per-fold values)
days_present = [d for d in days if d in metrics_df["day"].values]
fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))

bar_with_points(
    axes[0],
    days_present,
    "pearsonr",
    metrics_df,
    "Pearson's r",
    "Count prediction",
    ylim=(0, 1.0),
)
bar_with_points(
    axes[1],
    days_present,
    "median_jsd",
    metrics_df,
    "Median JSD",
    "Profile prediction",
    ylim=(0, 1.0),
)

fig.suptitle(
    "ChromBPNet model performance (peaks, test chromosomes)",
    fontsize=11,
    y=1.02,
)
fig.tight_layout()
save_fig(fig, "03_model_performance_barplot")
save_data(
    metrics_df[["day", "fold", "pearsonr", "median_jsd"]],
    "03_model_performance_barplot",
)
plt.show()

# %%
# 8. MoDISco: load pattern and seqlet counts from per-day h5 files
modisco_dir = os.path.join(core_path, "results", "contrib_scores")

modisco_rows = []
for day in days:
    h5_file = os.path.join(modisco_dir, day, "modisco", "modisco_counts_results.h5")
    if not os.path.exists(h5_file):
        print(f"  missing modisco h5 for {day}")
        continue
    with h5py.File(h5_file, "r") as f:
        n_pos = len(f["pos_patterns"]) if "pos_patterns" in f else 0
        n_neg = len(f["neg_patterns"]) if "neg_patterns" in f else 0
        seqlets_pos = sum(
            int(f["pos_patterns"][p]["seqlets"]["n_seqlets"][()])
            for p in f.get("pos_patterns", {})
        )
        seqlets_neg = (
            sum(
                int(f["neg_patterns"][p]["seqlets"]["n_seqlets"][()])
                for p in f["neg_patterns"]
            )
            if "neg_patterns" in f and n_neg > 0
            else 0
        )
    modisco_rows.append(
        {
            "day": day,
            "n_pos_patterns": n_pos,
            "n_neg_patterns": n_neg,
            "seqlets_pos": seqlets_pos,
            "seqlets_neg": seqlets_neg,
        }
    )

modisco_df = pd.DataFrame(modisco_rows)
modisco_df.to_csv(
    os.path.join(tables_dir, "modisco_pattern_counts.tsv"),
    sep="\t",
    index=False,
)
print(modisco_df.to_string(index=False))

# %%
# compute unique QC'd motif counts per day for plot 04
motif_sets_by_day = {}
for day in days:
    path = os.path.join(finemo_qc_dir, f"hits_filtered_{day}.tsv")
    if os.path.exists(path):
        _hits = pd.read_csv(path, sep="\t", usecols=["motif_name"])
        motif_sets_by_day[day] = set(_hits["motif_name"].unique())

all_motifs_union = (
    set().union(*motif_sets_by_day.values()) if motif_sets_by_day else set()
)
n_pos_union = sum(1 for m in all_motifs_union if m.startswith("pos_"))
n_neg_union = len(all_motifs_union) - n_pos_union

motif_counts_df = pd.DataFrame(
    [
        {
            "day": day,
            "n_pos_motifs": sum(
                1 for m in motif_sets_by_day[day] if m.startswith("pos_")
            ),
            "n_neg_motifs": sum(
                1 for m in motif_sets_by_day[day] if not m.startswith("pos_")
            ),
        }
        for day in days
        if day in motif_sets_by_day
    ]
)
print(
    f"QC'd motifs union: {len(all_motifs_union)} ({n_pos_union} pos, {n_neg_union} neg)"
)
print(motif_counts_df.to_string(index=False))

# %%
# 9. Plot 04: MoDISco patterns and seqlets per day (pos/neg stacked bars)
POS_COLOR = "#C0BFD9"  # red = positive
NEG_COLOR = "#454372"  # blue = negative

annot_df = pd.read_csv(annotation_tsv, sep="\t")
neg_annot = annot_df[annot_df["posneg"] == "neg"].set_index("cluster_id")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
x = np.arange(len(days))

ax = axes[0]
ax.bar(x, modisco_df["seqlets_pos"], color=POS_COLOR, label="positive")
ax.bar(
    x,
    modisco_df["seqlets_neg"],
    bottom=modisco_df["seqlets_pos"],
    color=NEG_COLOR,
    label="negative",
)
ax.set_xticks(x)
ax.set_xticklabels([day_label[d] for d in modisco_df["day"]], fontsize=18)
ax.set_ylabel("# seqlets", fontsize=18)
ax.set_title("MoDISco seqlets")
ax.legend(frameon=False, fontsize=14)

ax = axes[1]
ax.bar(x, modisco_df["n_pos_patterns"], color=POS_COLOR, label="positive")
ax.bar(
    x,
    modisco_df["n_neg_patterns"],
    bottom=modisco_df["n_pos_patterns"],
    color=NEG_COLOR,
    label="negative",
)
for xi, row in modisco_df.iterrows():
    if row["n_neg_patterns"] > 0:
        total = row["n_pos_patterns"] + row["n_neg_patterns"]
        tfs = day_neg_tfs(row["day"])
        label = "\n".join(tfs) if tfs else f"{int(row['n_neg_patterns'])} neg"
        ax.text(
            xi,
            total + 1.6,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
            color=NEG_COLOR,
            style="italic",
        )
ax.set_xticks(x)
ax.set_xticklabels([day_label[d] for d in modisco_df["day"]], fontsize=18)
ax.set_ylabel("# patterns", fontsize=18)
ax.set_title("MoDISco patterns")

ax = axes[2]
n_total_union = n_pos_union + n_neg_union
mc_days = motif_counts_df["day"].tolist()
xm = np.arange(len(mc_days))
ax.bar(xm, motif_counts_df["n_pos_motifs"], color=POS_COLOR)
ax.bar(
    xm,
    motif_counts_df["n_neg_motifs"],
    bottom=motif_counts_df["n_pos_motifs"],
    color=NEG_COLOR,
)
for xi, row in motif_counts_df.iterrows():
    total = row["n_pos_motifs"] + row["n_neg_motifs"]
    ax.text(xi, total, str(int(total)), ha="center", va="bottom", fontsize=14)
ax.set_xticks(xm)
ax.set_xticklabels([day_label[d] for d in mc_days], fontsize=18)
ax.set_ylabel("# motifs", fontsize=18)
ax.set_title(f"QC'd motifs per timepoint (union: {n_total_union})")

fig.tight_layout()
save_fig(fig, "04_modisco_patterns")
save_data(modisco_df, "04_modisco_patterns")
save_data(motif_counts_df, "04_motif_counts")
plt.show()

# %%
# 10. Fi-NeMo: load QCed hits (motif_name + peak_id only)

finemo_stats = []
hits_by_day = {}

for day in days:
    path = os.path.join(finemo_qc_dir, f"hits_filtered_{day}.tsv")
    peaks_path = os.path.join(
        core_path,
        "results",
        "finemo_unified",
        f"{day}_{peak_type}",
        "peaks_qc.tsv",
    )
    if not os.path.exists(path):
        print(f"  missing filtered hits for {day}")
        continue

    hits = pd.read_csv(path, sep="\t", usecols=["motif_name", "peak_id"])
    hits["is_pos"] = hits["motif_name"].str.startswith("pos_")
    hits_by_day[day] = hits

    total_peaks = (
        len(pd.read_csv(peaks_path, sep="\t")) if os.path.exists(peaks_path) else np.nan
    )
    peaks_with_pos = hits.loc[hits["is_pos"], "peak_id"].nunique()
    peaks_with_neg = hits.loc[~hits["is_pos"], "peak_id"].nunique()

    neg_patterns = sorted(hits.loc[~hits["is_pos"], "motif_name"].unique())

    finemo_stats.append(
        {
            "day": day,
            "total_hits_pos": int(hits["is_pos"].sum()),
            "total_hits_neg": int((~hits["is_pos"]).sum()),
            "n_neg_patterns": len(neg_patterns),
            "neg_patterns": ",".join(neg_patterns),
            "neg_tfs": ",".join(neg_label(p) for p in neg_patterns),
            "total_peaks": total_peaks,
            "peaks_with_pos": peaks_with_pos,
            "peaks_with_neg": peaks_with_neg,
            "frac_peaks_pos": peaks_with_pos / total_peaks if total_peaks else np.nan,
            "frac_peaks_neg": peaks_with_neg / total_peaks if total_peaks else np.nan,
        }
    )
    print(
        f"  {day}: {hits['is_pos'].sum():,} pos hits, "
        f"{(~hits['is_pos']).sum():,} neg hits "
        f"({len(neg_patterns)} neg patterns: {', '.join(neg_patterns)})"
    )

finemo_df = pd.DataFrame(finemo_stats)
finemo_df.to_csv(
    os.path.join(tables_dir, "finemo_hit_stats.tsv"), sep="\t", index=False
)

# %%
# 11. Plot 07: hits per peak distribution per day (violin, positive hits only)
hits_per_peak = {}
for day, hits in hits_by_day.items():
    counts = hits[hits["is_pos"]].groupby("peak_id").size()
    hits_per_peak[day] = counts.values

days_bx = [d for d in days if d in hits_per_peak]
data_bx = [hits_per_peak[d] for d in days_bx]

fig, ax = plt.subplots(figsize=(5, 5))
bp = ax.boxplot(
    data_bx,
    positions=np.arange(len(days_bx)),
    patch_artist=True,
    showfliers=False,
    medianprops=dict(color="black", linewidth=1.5),
    whiskerprops=dict(color="gray", linewidth=1),
    capprops=dict(color="gray", linewidth=1),
    boxprops=dict(linewidth=1),
    widths=0.5,
)
for patch, day in zip(bp["boxes"], days_bx):
    patch.set_facecolor(day_colors[day])
    # patch.set_alpha(0.7)

for xi, vals in enumerate(data_bx):
    ax.text(
        xi,
        1.0,
        f"n={len(vals)}",
        ha="center",
        va="top",
        fontsize=10,
        transform=ax.get_xaxis_transform(),
    )

ax.set_xticks(np.arange(len(days_bx)))
ax.set_xticklabels([day_label[d] for d in days_bx], fontsize=16)
ax.set_ylabel("# positive hits per peak", fontsize=16)
ax.set_title("Positive motif hits per peak")
fig.tight_layout()
save_fig(fig, "07_finemo_hits_per_peak")
save_data(
    pd.DataFrame(
        {
            "day": day,
            "hits_per_peak": hits_per_peak[day],
        }
        for day in days_bx
    ),
    "07_finemo_hits_per_peak",
)
plt.show()

# %%
