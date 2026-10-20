"""
3.chrombpnet_figures.py
Publication-ready ChromBPNet TF enrichment figures for programs of interest.
Adapts Plot 3 (per-day heatmaps) and Plot 4 (TF rank scatter) from
2.tf_enrichment_in_programs.py, restricted to the enhancer (distal enhancer)
element type, with motif CWM logos and improved color scaling.

Logo heatmap: all significant TFs, ward-clustered rows, CWM logos left.

Color scale: 95th percentile of all non-zero neg_log10_qval values across all
plotted (program, day) POI cells, shared across days for comparability.
"""

# %%
import os
import re
from collections import defaultdict

import h5py
import logomaker
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text
from scipy.cluster.hierarchy import leaves_list, linkage

mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["font.size"] = 14
mpl.rcParams["axes.labelsize"] = 14
mpl.rcParams["axes.titlesize"] = 14
mpl.rcParams["xtick.labelsize"] = 14
mpl.rcParams["ytick.labelsize"] = 14
mpl.rcParams["legend.fontsize"] = 14
mpl.rcParams["savefig.dpi"] = 300
mpl.rcParams["savefig.bbox"] = "tight"
mpl.rcParams["savefig.transparent"] = True

# %%
# Parameters
core_path = "/path/to/ipsc-ec-chrombpnet"
enrichment_dir = os.path.join(core_path, "analysis", "2.tf_enrichment_in_programs")
h5_path = os.path.join(
    core_path,
    "results",
    "compendium",
    "modisco_compiled",
    "modisco_compiled.h5",
)
cluster_map_path = os.path.join(
    core_path,
    "analysis",
    "0.0.motif_id_annotation",
    "tables",
    "cluster_to_tf_motif_id_mapping.tsv",
)
output_dir = os.path.join(core_path, "analysis", "3.chrombpnet_figures")
plots_dir = os.path.join(output_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)

et = "enhancer"
et_display = "distal enhancers"
days = ["d0", "d1", "d2", "d3"]
q_sig_threshold = 0.05
percentile_clip = 99  # color scale upper bound (of non-zero POI values)

# Programs of interest and the day each is most active
poi_prog_day = [
    (2, "d3"),
    (15, "d2"),
    (17, "d2"),
    (24, "d3"),
    (27, "d3"),
    (34, "d3"),
]

# Logo figure dimensions (inches)
logo_col_w = 1.2  # width of logo column
row_h_hm = 0.32  # height per TF row in logo heatmap
row_h_dot = 0.42  # height per TF row in dot plot

save_plots = True


# %%
# Helper functions


def save_figure(fig, path_stem):
    fig.savefig(path_stem + ".pdf")
    fig.savefig(path_stem + ".png")


def mark_significant(ax, qval_df, rows, cols, val_df=None, cmap=None, vmin=0, vmax=1):
    """
    Annotate heatmap cells with significance stars.
    Star color is adaptive: black on light cells, white on dark cells.
    When val_df/cmap/vmin/vmax are provided, luminance is computed from the
    cell value; otherwise black is used as a safe default.
    """
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            q = qval_df.loc[r, c]
            if q < 0.001:
                label = "***"
            elif q < 0.01:
                label = "**"
            elif q < 0.05:
                label = "*"
            else:
                label = ""
            if label:
                if val_df is not None and cmap is not None:
                    val = float(val_df.loc[r, c])
                    norm_val = np.clip((val - vmin) / (vmax - vmin), 0, 1)
                    r_c, g_c, b_c, _ = cmap(norm_val)
                    luminance = 0.299 * r_c + 0.587 * g_c + 0.114 * b_c
                    star_color = "black" if luminance > 0.45 else "white"
                else:
                    star_color = "black"
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    label,
                    ha="center",
                    va="center",
                    fontsize=6,
                    color=star_color,
                )


def strip_number_suffix(tf_list):
    """Remove underscore and everything after from all TF motif names."""
    return {tf: re.sub(r"_.*$", "", tf) for tf in tf_list}


def get_cluster_id(tf_motif_id, tf_to_cluster):
    """
    Map tf_motif_id to integer cluster_id.
    Handles both named IDs (lookup in tf_to_cluster) and cluster_N style.
    """
    m = re.match(r"^cluster_(\d+)$", tf_motif_id)
    if m:
        return int(m.group(1))
    return tf_to_cluster.get(tf_motif_id)


def trim_cwm(cwm, threshold=0.05):
    """Trim near-zero flanking positions from a CWM (L x 4 array)."""
    importance = np.abs(cwm).sum(axis=1)
    max_imp = importance.max()
    if max_imp == 0:
        return cwm
    above = importance > threshold * max_imp
    if not above.any():
        return cwm
    left = int(above.argmax())
    right = int(len(above) - above[::-1].argmax())
    return cwm[left:right]


def draw_cwm_logo(ax, cwm, trim_threshold=0.1):
    """Draw a CWM contribution weight matrix as a sequence logo via logomaker."""
    cwm = trim_cwm(cwm, threshold=trim_threshold)
    df = pd.DataFrame(cwm, columns=["A", "C", "G", "T"])
    try:
        logomaker.Logo(
            df,
            ax=ax,
            color_scheme="classic",
            baseline_width=0.5,
            stack_order="small_on_top",
        )
    except Exception:
        pass
    ax.set_xlim(-0.5, len(cwm) - 0.5)
    ax.set_yticks([])
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def ward_row_order(matrix):
    """Row indices reordered by ward hierarchical clustering."""
    if len(matrix) < 2:
        return list(range(len(matrix)))
    Z = linkage(matrix.fillna(0).values, method="ward", metric="euclidean")
    return leaves_list(Z).tolist()


def build_logo_figure(
    hm_data,
    display_names,
    cluster_ids,
    cwms,
    n_data_cols,
    row_h,
    extra_w=1.0,
    square=False,
    label_logos=False,
):
    """
    Create a figure with a logo column (left) and a content column (right).
    When label_logos=True, TF display names appear as ylabels to the left of logos.
    When square=True, hm content width is sized so cells are square (n_data_cols * row_h).
    Returns (fig, ax_content).
    """
    n_tfs = len(display_names)
    cell_w = row_h if square else 0.55
    hm_w = max(1.5, n_data_cols * cell_w + extra_w)
    fig_h = n_tfs * row_h + 1.5
    fig_w = logo_col_w + hm_w

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[logo_col_w, hm_w],
        wspace=0.02,
    )

    gs_logo = gs[0, 0].subgridspec(n_tfs, 1, hspace=0)
    for i, (cid, name) in enumerate(zip(cluster_ids, display_names)):
        ax_l = fig.add_subplot(gs_logo[i, 0])
        if cid is not None and cid in cwms:
            draw_cwm_logo(ax_l, cwms[cid])
        else:
            ax_l.axis("off")
        if label_logos:
            ax_l.set_ylabel(
                name,
                rotation=0,
                ha="right",
                va="center",
                fontsize=8,
                labelpad=4,
            )

    ax_content = fig.add_subplot(gs[0, 1])
    return fig, ax_content


# %%
# Load enhancer enrichment results
results_path = os.path.join(enrichment_dir, f"{et}_enrichment_results.tsv")
results = pd.read_csv(results_path, sep="\t")
results["significant"] = results["qval"] < q_sig_threshold
print(
    f"Loaded {len(results):,} rows; "
    f"{results['tf'].nunique()} TFs, {results['program'].nunique()} programs"
)

# %%
# Load cluster CWM data from compiled h5 (pos_patterns only)
cluster_cwms = {}
with h5py.File(h5_path, "r") as f:
    for cid_str in f["pos_patterns"].keys():
        cluster_cwms[int(cid_str)] = f["pos_patterns"][cid_str]["contrib_scores"][:]
print(f"Loaded CWMs for {len(cluster_cwms)} clusters")

# %%
# Build tf_motif_id -> cluster_id reverse mapping
cluster_map = pd.read_csv(cluster_map_path, sep="\t")
tf_to_cluster = dict(zip(cluster_map["tf_motif_id"], cluster_map["cluster_id"]))

# Compute display names for all TFs appearing in results
all_tfs = results["tf"].unique().tolist()
tf_display_map = strip_number_suffix(all_tfs)

# Sanity check: show a few renames
renamed = [(tf, d) for tf, d in tf_display_map.items() if tf != d]
print(f"Suffix stripped for {len(renamed)} TF names (examples: {renamed[:5]})")

# %%
# Group programs of interest by day
poi_by_day = defaultdict(list)
for prog, day in poi_prog_day:
    poi_by_day[day].append(prog)

# %%
# Compute shared color scale (95th pct of non-zero POI values)
poi_prog_day_df = pd.DataFrame(poi_prog_day, columns=["program", "day"])
poi_vals = (
    results.merge(poi_prog_day_df, on=["program", "day"])["neg_log10_qval"]
    .replace(0, np.nan)
    .dropna()
)
vmax_global = float(np.nanpercentile(poi_vals, percentile_clip))
vmax_global = max(vmax_global, 1.0)
print(
    f"Color vmax ({percentile_clip}th pct of non-zero POI values): {vmax_global:.2f} "
    f"(absolute max: {poi_vals.max():.2f})"
)

# %%
# Plot per-day logo heatmaps (all sig TFs, ward-clustered rows)
print("\nPlotting logo heatmaps (Plot 3A) ...")

for day in sorted(poi_by_day.keys()):
    progs_this_day = poi_by_day[day]
    col_labels = [f"p{p}" for p in progs_this_day]
    n_progs = len(progs_this_day)

    # TFs significant in any POI program on this day
    day_res = results[
        (results["program"].isin(progs_this_day)) & (results["day"] == day)
    ]
    poi_sig_tfs = sorted(day_res.loc[day_res["significant"], "tf"].unique())
    if not poi_sig_tfs:
        print(f"  {day}: no significant TFs, skipping")
        continue
    print(f"  {day}: {len(poi_sig_tfs)} significant TFs across {n_progs} programs")

    # Build heatmap matrices
    hm_data = pd.DataFrame(index=poi_sig_tfs, columns=col_labels, dtype=float)
    hm_qval = pd.DataFrame(index=poi_sig_tfs, columns=col_labels, dtype=float)
    hm_rbc = pd.DataFrame(index=poi_sig_tfs, columns=col_labels, dtype=float)
    for prog, col in zip(progs_this_day, col_labels):
        sub = results[(results["program"] == prog) & (results["day"] == day)].set_index(
            "tf"
        )
        hm_data[col] = sub.reindex(poi_sig_tfs)["neg_log10_qval"].values
        hm_qval[col] = sub.reindex(poi_sig_tfs)["qval"].values
        hm_rbc[col] = 2.0 * (sub.reindex(poi_sig_tfs)["AUC"].values - 0.5)
    hm_data = hm_data.astype(float).fillna(0.0)
    hm_qval = hm_qval.astype(float).fillna(1.0)
    hm_rbc = hm_rbc.astype(float).fillna(0.0)

    # Ward row ordering (shared across both heatmap versions)
    row_order = ward_row_order(hm_data)
    hm_data = hm_data.iloc[row_order]
    hm_qval = hm_qval.iloc[row_order]
    hm_rbc = hm_rbc.iloc[row_order]
    tfs_ordered = hm_data.index.tolist()
    display_names = [tf_display_map.get(tf, tf) for tf in tfs_ordered]
    cluster_ids = [get_cluster_id(tf, tf_to_cluster) for tf in tfs_ordered]

    fig, ax_hm = build_logo_figure(
        hm_data,
        display_names,
        cluster_ids,
        cluster_cwms,
        n_data_cols=n_progs,
        row_h=row_h_hm,
        extra_w=1.2,
        square=True,
        label_logos=True,
    )

    sns.heatmap(
        hm_data,
        ax=ax_hm,
        cmap="Blues",
        vmin=0,
        vmax=vmax_global,
        linewidths=0.3,
        linecolor="white",
        square=True,
        cbar=True,
        cbar_kws={"label": "-log10(q)", "shrink": 0.4},
        yticklabels=False,
    )
    mark_significant(
        ax_hm,
        hm_qval,
        tfs_ordered,
        col_labels,
        val_df=hm_data,
        cmap=plt.cm.Blues,
        vmin=0,
        vmax=vmax_global,
    )
    ax_hm.set_title(f"{et_display}, {day}\n*** q<0.001  ** q<0.01  * q<0.05", size=10)
    ax_hm.set_xlabel("")
    ax_hm.set_ylabel("")
    ax_hm.tick_params(axis="x", labelsize=9, rotation=0)
    ax_hm.tick_params(axis="y", length=0)

    plt.tight_layout()
    if save_plots:
        stem = os.path.join(plots_dir, f"logo_heatmap_{day}")
        save_figure(fig, stem)
        hm_data.assign(display_name=display_names).to_csv(f"{stem}_data.tsv", sep="\t")
        print(f"    saved {stem}")

    # Enrichment plot: same layout colored by rank biserial correlation
    # r = 2*(AUC - 0.5), ranges 0 (no enrichment) to 1 (perfect enrichment).
    # Non-significant cells masked to 0 (white); stars show significance.
    hm_rbc_masked = hm_rbc.where(hm_qval < q_sig_threshold, other=0.0)

    sig_rbc_vals = hm_rbc.values[hm_qval.values < q_sig_threshold]
    vmax_rbc = (
        float(np.nanpercentile(sig_rbc_vals, percentile_clip))
        if sig_rbc_vals.size > 0
        else 1.0
    )
    vmax_rbc = max(vmax_rbc, 0.1)

    fig_rbc, ax_rbc = build_logo_figure(
        hm_rbc_masked,
        display_names,
        cluster_ids,
        cluster_cwms,
        n_data_cols=n_progs,
        row_h=row_h_hm,
        extra_w=1.2,
        square=True,
        label_logos=True,
    )
    sns.heatmap(
        hm_rbc_masked,
        ax=ax_rbc,
        cmap="Blues",
        vmin=0,
        vmax=vmax_rbc,
        linewidths=0.3,
        linecolor="white",
        square=True,
        cbar=True,
        cbar_kws={"label": "rank biserial r", "shrink": 0.4},
        yticklabels=False,
    )
    mark_significant(
        ax_rbc,
        hm_qval,
        tfs_ordered,
        col_labels,
        val_df=hm_rbc_masked,
        cmap=plt.cm.Blues,
        vmin=0,
        vmax=vmax_rbc,
    )
    ax_rbc.set_title(
        f"{et_display}, {day} (effect size)\n*** q<0.001  ** q<0.01  * q<0.05",
        size=10,
    )
    ax_rbc.set_xlabel("")
    ax_rbc.set_ylabel("")
    ax_rbc.tick_params(axis="x", labelsize=9, rotation=0)
    ax_rbc.tick_params(axis="y", length=0)

    plt.tight_layout()
    if save_plots:
        stem_rbc = os.path.join(plots_dir, f"logo_heatmap_rbc_{day}")
        save_figure(fig_rbc, stem_rbc)
        hm_rbc_masked.assign(display_name=display_names).to_csv(
            f"{stem_rbc}_data.tsv", sep="\t"
        )
        print(f"    saved {stem_rbc}")
# %%
# TF rank scatter
print("\nPlotting TF rank scatter (Plot 4) ...")

for day, progs in sorted(poi_by_day.items()):
    n_progs = len(progs)
    fig, axes = plt.subplots(
        1, n_progs, figsize=(3 * n_progs, 5), sharey=True, sharex=True
    )
    if n_progs == 1:
        axes = [axes]

    prog_frames = []
    for prog in progs:
        sub = results[(results["program"] == prog) & (results["day"] == day)].copy()
        sub["rank"] = (
            sub["neg_log10_qval"].rank(ascending=False, method="min").astype(int)
        )
        sub["display_tf"] = sub["tf"].map(tf_display_map)
        sub = sub.sort_values("rank").reset_index(drop=True)
        prog_frames.append(sub)

    for i, (ax, prog, sub) in enumerate(zip(axes, progs, prog_frames)):
        ax.scatter(
            sub["rank"],
            sub["neg_log10_qval"],
            color="lightgray",
            s=20,
            zorder=2,
        )
        texts = [
            ax.text(
                row["rank"],
                row["neg_log10_qval"],
                row["display_tf"],
                fontsize=8,
            )
            for _, row in sub.head(7).iterrows()
        ]
        adjust_text(
            texts,
            ax=ax,
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.8),
        )
        ax.set_xlabel("TF motif rank")
        ax.set_title(f"program {prog}\nn={sub['tf'].nunique()}\n", size=10)
        if i == 0:
            ax.set_ylabel(f"-log10(q-value); {et_display}")

    fig.suptitle(f"TF enrichment rank, {day}, {et_display}", y=1.02, size=10)
    plt.tight_layout()
    if save_plots:
        for prog, sub in zip(progs, prog_frames):
            sub.to_csv(
                os.path.join(plots_dir, f"tf_rank_scatter_p{prog}_{day}_data.tsv"),
                sep="\t",
                index=False,
            )
        save_figure(fig, os.path.join(plots_dir, f"tf_rank_scatter_{day}"))
        print(f"  saved tf_rank_scatter_{day}")

# %%
print("\n Done ")
print(f"Plots in: {plots_dir}")

# %%
