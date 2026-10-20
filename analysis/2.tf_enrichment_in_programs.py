"""
2.tf_enrichment_in_programs.py
Test TF motif enrichment in cNMF programs using scE2G gene-motif scores.

For every (program, day, tf) combination, restricted to distal enhancer
elements only (single element_type, so BH correction below is over one test
family -- no separate per-element-type families to (incorrectly) union
afterward):
  - Foreground : top n_top_genes by cNMF program loading
  - Background : remaining E2G-linked genes in the cNMF universe for that TF/day
  - Test       : one-sided Mann-Whitney U (alternative='greater').
                 H0: TF importance scores for top-N genes are drawn from the
                 same distribution as TF importance scores of other genes.
                 Ha: TF importance scores are higher for top-N genes.
                 method='auto' lets scipy pick 'exact' for small/tie-free
                 samples rather than forcing the asymptotic approximation
                 (real per-TF universes here run as low as single digits).
  - Correction : Benjamini-Hochberg

Outputs
-------
  analysis/2.tf_enrichment_in_programs/
    {et}_enrichment_results.tsv
    plots/
      tf_program_count_heatmap.{pdf,png}       TFs x days: # programs enriched per TF
      program_tf_count_heatmap.{pdf,png}       programs x days: # TFs enriched per program
      per_program/{prog_name}.{pdf,png}        significant TFs x days per program
      poi_by_day_clustermap_{day}_{et}.{pdf,png}  clustered TFs x POI programs, per day
      tf_rank_scatter_{day}.{pdf,png}          TF rank scatter for programs of interest
"""

# %%
import os
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests

matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["axes.spines.right"] = False
matplotlib.rcParams["font.size"] = 14
matplotlib.rcParams["savefig.dpi"] = 300
matplotlib.rcParams["savefig.bbox"] = "tight"
matplotlib.rcParams["savefig.transparent"] = True

# %%
# --- Parameters ---
core_path = "/path/to/ipsc-ec-chrombpnet"
cnmf_h5mu = (
    "/path/to/cnmf_results/adata/cNMF_50_0_2.h5mu"  # K=50, density threshold 0.2
)
data_dir = os.path.join(core_path, "analysis", "1.0.gene_motif_linking", "data")
output_dir = os.path.join(core_path, "analysis", "2.tf_enrichment_in_programs")
plots_dir = os.path.join(output_dir, "plots")
os.makedirs(os.path.join(plots_dir, "per_program"), exist_ok=True)

# Must match values used in 1.0.gene_motif_linking.py
weight_strategy = "e2g"
hit_score_col = "hit_coefficient_global"  # "hit_importance", "hit_coefficient_global"
# Restricted to enhancers only. Also avoids a real bug the old
# ["promoter", "enhancer", "all_cres"] setup had: BH correction was applied
# separately per element type, then downstream reporting took the union of
# "significant" across all three -- a union of independently-corrected sets
# is not itself FDR-controlled at the nominal threshold.
element_types = ["enhancer"]

n_top_genes = 300  # foreground: top N genes per cNMF program
days = ["d0", "d1", "d2", "d3"]
q_sig_threshold = 0.05

# Programs of interest and the day each is most relevant
poi_prog_day = [
    (2, "d3"),
    (15, "d2"),
    (17, "d2"),
    (24, "d3"),
    (27, "d3"),
    (34, "d3"),
]
# Element type used for per-program clustermaps and TF rank scatter
primary_et = "enhancer"

save_plots = True

# Reuse the {et}_enrichment_results.tsv already on disk instead of recomputing
# the Mann-Whitney tests. Everything below the compute stage depends only on
# all_results, so a plot-only change does not need the cNMF object, the
# gene-motif score tables, or the full test sweep. Set to False to recompute
# from scratch (required if n_top_genes, days, element_types, weight_strategy
# or hit_score_col change).
reuse_enrichment = True

# Layout for the programs-of-interest clustermaps (Plot 3). seaborn's
# clustermap builds its own figure, so the figure is sized from the cell grid
# and the dendrogram share; the tick labels and colorbar overflow the axes and
# are picked up by bbox_inches="tight" on save.
poi_cell_in = 0.45  # side of one heatmap cell
poi_row_dendro_in = 0.9  # width of the TF (row) dendrogram
poi_col_dendro_in = 0.7  # height of the program (column) dendrogram
poi_label_pad_in = 0.25  # breathing room beyond the measured tick labels
poi_star_fontsize = 10  # significance stars, sized to the cells
# Clustering 2 columns is degenerate (a single merge in an arbitrary order),
# so columns are only clustered once there are at least this many programs.
poi_min_cols_to_cluster = 3


# %%
def save_figure(fig, path_stem):
    """Save figure as both PDF and PNG."""
    fig.savefig(path_stem + ".pdf")
    fig.savefig(path_stem + ".png")


def text_width_in(labels, fontsize):
    """Width in inches of the widest rendered label.

    seaborn's clustermap runs tight_layout internally, which pulls the tick
    labels inside the figure. If the figure is not wide enough to hold them the
    heatmap is squeezed to almost nothing, so the label extent has to be
    measured rather than guessed.
    """
    fig = plt.figure(figsize=(1, 1), dpi=100)
    renderer = fig.canvas.get_renderer()
    width = 0.0
    for label in labels:
        text = fig.text(0, 0, str(label), fontsize=fontsize)
        width = max(width, text.get_window_extent(renderer=renderer).width / fig.dpi)
        text.remove()
    plt.close(fig)
    return width


def mark_significant(ax, qval_matrix, rows, cols, fontsize=6):
    """Annotate heatmap cells with significance stars from a q-value matrix."""
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            q = qval_matrix.loc[r, c]
            if q < 0.001:
                label = "***"
            elif q < 0.01:
                label = "**"
            elif q < 0.05:
                label = "*"
            else:
                label = ""
            if label:
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    label,
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                    color="black",
                )


# %%
# The three compute stages below iterate over this list. When reusing on-disk
# results it is empty, which skips them without touching the plotting code.
compute_ets = [] if reuse_enrichment else list(element_types)

# %%
# Load cNMF program loadings (only needed to build the test foregrounds)
if compute_ets:
    import muon as mu

    cnmf = mu.read_h5mu(cnmf_h5mu)["cNMF"]
    program_loadings = pd.DataFrame(
        cnmf.varm["loadings"], columns=cnmf.uns["var_names"]
    )
    program_loadings.index = "program_" + program_loadings.index.astype(str)
    all_cnmf_genes = set(program_loadings.columns)
    n_programs = len(program_loadings)
    print(f"Loaded {n_programs} programs x {len(all_cnmf_genes)} genes")

# %%
# Load TF scores for each element type
all_scores = {}
for et in compute_ets:
    path = os.path.join(
        data_dir,
        f"{weight_strategy}_{hit_score_col}_{et}_gene_motif_scores.tsv.gz",
    )
    if not os.path.exists(path):
        print(f"[WARN] {et}: scores file not found, skipping: {path}")
        continue
    all_scores[et] = pd.read_csv(path, sep="\t")
    print(
        f"  {et}: {len(all_scores[et]):,} rows, "
        f"{all_scores[et]['tf_motif_id'].nunique()} TFs"
    )

# %%
# Build gene x day score matrix per (element type, TF)
# Universe: genes with E2G links for that TF on that day AND in the cNMF matrix
scores_by_tf = {}
for et in compute_ets:
    if et not in all_scores:
        continue
    scores_by_tf[et] = {}
    df = all_scores[et][all_scores[et]["day"].isin(days)]
    for tf, tf_group in df.groupby("tf_motif_id"):
        tf_df = tf_group.pivot_table(
            index="TargetGene",
            columns="day",
            values="weighted_score",
            aggfunc="sum",
        ).reindex(columns=days)
        tf_df = tf_df[tf_df.index.isin(all_cnmf_genes)]
        scores_by_tf[et][tf] = tf_df
    print(f"  {et}: {len(scores_by_tf[et])} TFs built")

# %%
# Run Mann-Whitney U enrichment for all (program, day, tf)
# H1 (alternative='greater'): top-N genes have higher TF scores than background
all_results = {}
for et in compute_ets:
    if et not in scores_by_tf:
        continue
    tfs = list(scores_by_tf[et].keys())
    n_tests = n_programs * len(days) * len(tfs)
    print(
        f"\n  [{et}] {n_programs} programs x {len(days)} days x {len(tfs)} TFs"
        f" = {n_tests:,} tests ..."
    )

    records = []
    for prog_idx, prog_name in enumerate(program_loadings.index):
        top_genes_set = set(program_loadings.loc[prog_name].nlargest(n_top_genes).index)
        for tf in tfs:
            for day in days:
                day_scores = scores_by_tf[et][tf][day].dropna()
                fg = day_scores[
                    [g for g in day_scores.index if g in top_genes_set]
                ].values
                bg = day_scores[
                    [g for g in day_scores.index if g not in top_genes_set]
                ].values
                if len(fg) == 0 or len(bg) == 0:
                    continue
                u_stat, pval = stats.mannwhitneyu(
                    fg, bg, alternative="greater", method="auto"
                )
                records.append(
                    {
                        "program": prog_idx,
                        "program_name": prog_name,
                        "tf": tf,
                        "day": day,
                        "n_top": len(fg),
                        "n_background": len(bg),
                        "U": u_stat,
                        "AUC": u_stat / (len(fg) * len(bg)),
                        "pval": pval,
                    }
                )

    results = pd.DataFrame(records)
    _, qvals, _, _ = multipletests(results["pval"], method="fdr_bh")
    results["qval"] = qvals
    results["neg_log10_qval"] = -np.log10(results["qval"].clip(lower=1e-300))
    results["significant"] = results["qval"] < q_sig_threshold
    results["et"] = et
    all_results[et] = results

    n_sig = results["significant"].sum()
    print(
        f"  [{et}] significant: {n_sig} / {len(results)} tests (q < {q_sig_threshold})"
    )

    out_path = os.path.join(output_dir, f"{et}_enrichment_results.tsv")
    results.to_csv(out_path, sep="\t", index=False)
    print(f"  [{et}] saved -> {out_path}")
# %%
# Reuse path: load the previously computed results so the plots below can be
# regenerated without redoing the test sweep.
if reuse_enrichment:
    for et in element_types:
        results_path = os.path.join(output_dir, f"{et}_enrichment_results.tsv")
        if not os.path.exists(results_path):
            raise FileNotFoundError(
                f"reuse_enrichment=True but {results_path} is missing; "
                "set reuse_enrichment = False to compute it"
            )
        all_results[et] = pd.read_csv(results_path, sep="\t")
        print(
            f"  [{et}] reused {len(all_results[et]):,} tests from "
            f"{os.path.basename(results_path)}"
        )
    n_programs = pd.concat(all_results.values())["program_name"].nunique()

# %%
# Quantify enrichment breadth across TFs and programs (union across element types)
sig_union = pd.concat(all_results.values())
sig_union = sig_union[sig_union["significant"]]

all_union = pd.concat(all_results.values())
n_total_tfs = all_union["tf"].nunique()
n_sig_tfs = sig_union["tf"].nunique()
n_sig_programs = sig_union["program"].nunique()
n_total_tests = sum(len(r) for r in all_results.values())
n_sig_tests = sum(r["significant"].sum() for r in all_results.values())

print("\n=== Enrichment breadth (union across element types) ===")
print(f"Significant TFs    : {n_sig_tfs} / {n_total_tfs} unique TFs tested")
print(f"Programs with any sig. TF: {n_sig_programs} / {n_programs}")
print(
    f"Significant tests  : {n_sig_tests:,} / {n_total_tests:,} ({100 * n_sig_tests / n_total_tests:.1f}%)"
)
# %%
sig_tfs_union = set(sig_union["tf"])
et_order = [et for et in element_types if et in all_results]
# %%
# Plot 1b: TF x day heatmap of number of significantly enriched programs
print("\nPlotting TF program-count heatmaps ...")

all_tfs_sorted = sorted(sig_tfs_union)

count_matrices = {}
for et in et_order:
    res = all_results[et]
    count_matrices[et] = (
        res[res["significant"]]
        .groupby(["tf", "day"])["program"]
        .nunique()
        .unstack(fill_value=0)
        .reindex(index=all_tfs_sorted, columns=days, fill_value=0)
    )

tf_total = sum(count_matrices[et].sum(axis=1) for et in et_order)
tf_order_counts = tf_total.sort_values(ascending=False).index.tolist()
count_vmax = max(m.values.max() for m in count_matrices.values())
# %%
fig, axes = plt.subplots(
    1,
    len(et_order),
    figsize=(2.5 * len(et_order) + 1, max(4, len(sig_tfs_union) * 0.25 + 2)),
)
if len(et_order) == 1:
    axes = [axes]

for ax, et in zip(axes, et_order):
    mat = count_matrices[et].reindex(tf_order_counts, fill_value=0)
    is_last = et == et_order[-1]
    sns.heatmap(
        mat,
        ax=ax,
        cmap="Blues",
        vmin=0,
        vmax=count_vmax,
        annot=True,
        fmt="d",
        linewidths=0.3,
        linecolor="white",
        cbar=is_last,
        cbar_kws={"label": "# programs", "shrink": 0.4},
        yticklabels=(et == et_order[0]),
    )
    ax.set_title(et)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10)

fig.suptitle(
    f"# programs with significantly enriched TF per day\n{weight_strategy}, {hit_score_col}",
    y=1.02,
)
plt.tight_layout()
if save_plots:
    stem = os.path.join(plots_dir, "tf_program_count_heatmap")
    save_figure(fig, stem)
    for et in et_order:
        count_matrices[et].reindex(tf_order_counts, fill_value=0).to_csv(
            f"{stem}_{et}_data.tsv", sep="\t"
        )
    print("Saved TF program count heatmap")

# %%
# Plot 1c: Program x day heatmap of number of significantly enriched TFs
print("\nPlotting program TF-count heatmaps ...")

all_programs = list(range(n_programs))

prog_count_matrices = {}
for et in et_order:
    res = all_results[et]
    prog_count_matrices[et] = (
        res[res["significant"]]
        .groupby(["program", "day"])["tf"]
        .nunique()
        .unstack(fill_value=0)
        .reindex(index=all_programs, columns=days, fill_value=0)
    )

prog_total = sum(prog_count_matrices[et].sum(axis=1) for et in et_order)
prog_order = prog_total.sort_values(ascending=False).index.tolist()
prog_count_vmax = max(m.values.max() for m in prog_count_matrices.values())

fig, axes = plt.subplots(
    1,
    len(et_order),
    figsize=(2.5 * len(et_order) + 1, max(4, len(all_programs) * 0.25 + 2)),
)
if len(et_order) == 1:
    axes = [axes]

for ax, et in zip(axes, et_order):
    mat = prog_count_matrices[et].reindex(prog_order, fill_value=0)
    is_last = et == et_order[-1]
    sns.heatmap(
        mat,
        ax=ax,
        cmap="Blues",
        annot=True,
        vmin=0,
        vmax=prog_count_vmax,
        linewidths=0.3,
        linecolor="white",
        cbar=is_last,
        cbar_kws={"label": "# TFs", "shrink": 0.4},
        yticklabels=(et == et_order[0]),
    )
    ax.set_title(et)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10, rotation=0)

fig.suptitle(
    f"# TFs significantly enriched per program per day\n{weight_strategy}, {hit_score_col}",
    y=1.02,
)
plt.tight_layout()
if save_plots:
    stem = os.path.join(plots_dir, "program_tf_count_heatmap")
    save_figure(fig, stem)
    for et in et_order:
        prog_count_matrices[et].reindex(prog_order, fill_value=0).to_csv(
            f"{stem}_{et}_data.tsv", sep="\t"
        )
    print("Saved program TF count heatmap")

# %%
# Plot 2: per-program clustermap (significant TFs x days), primary element type
print(f"\nPlotting per-program clustermaps ({primary_et}) ...")
for prog_idx in range(n_programs):
    prog_name = f"program_{prog_idx}"
    if primary_et not in all_results:
        break
    prog_res = all_results[primary_et][all_results[primary_et]["program"] == prog_idx]
    sig_tfs = sorted(prog_res.loc[prog_res["significant"], "tf"].unique())
    if len(sig_tfs) == 0:
        continue

    nlq_pivot = (
        prog_res[prog_res["tf"].isin(sig_tfs)]
        .pivot(index="tf", columns="day", values="neg_log10_qval")
        .reindex(columns=days, fill_value=0.0)
        .fillna(0.0)
    )
    qval_pivot = (
        prog_res[prog_res["tf"].isin(sig_tfs)]
        .pivot(index="tf", columns="day", values="qval")
        .reindex(columns=days, fill_value=1.0)
        .fillna(1.0)
    )

    row_cluster = len(sig_tfs) > 1
    cg = sns.clustermap(
        nlq_pivot,
        cmap="Blues",
        vmin=0,
        vmax=nlq_pivot.values.max(),
        method="ward",
        row_cluster=row_cluster,
        col_cluster=False,
        linewidths=0.3,
        linecolor="white",
        figsize=(4, max(2, len(sig_tfs) * 0.35 + 1)),
        cbar_pos=(1.02, 0.4, 0.02, 0.2),
        cbar_kws={"label": "-log10(q)"},
        yticklabels=True,
    )
    reordered_tfs = (
        nlq_pivot.index[cg.dendrogram_row.reordered_ind]
        if row_cluster
        else nlq_pivot.index
    )
    mark_significant(cg.ax_heatmap, qval_pivot, reordered_tfs, days)
    cg.ax_heatmap.yaxis.set_tick_params(rotation=0)
    cg.ax_heatmap.set_xlabel("")
    cg.ax_heatmap.set_ylabel("")
    cg.fig.suptitle(
        f"{prog_name}, {primary_et}\n*** q<0.001  ** q<0.01  * q<0.05\n",
        y=1.01,
    )
    if save_plots:
        stem = os.path.join(plots_dir, "per_program", prog_name)
        nlq_pivot.to_csv(f"{stem}_data.tsv", sep="\t")
        for ext in ["pdf", "png"]:
            cg.fig.savefig(f"{stem}.{ext}", bbox_inches="tight", transparent=True)

# %%
# Plot 3: Per-day clustermaps for programs of interest, one per element type
# TF rows = all TFs significant in at least one POI program on that day.
# Rows and columns are ward-clustered, so the TF ordering is driven by the
# enrichment pattern across programs rather than being alphabetical.
poi_progs = {prog for prog, day in poi_prog_day}

poi_by_day_3 = defaultdict(list)
for prog, day in poi_prog_day:
    poi_by_day_3[day].append(prog)

# Shared colour scale: max only over the (program, day) pairs actually plotted
poi_prog_day_df = pd.DataFrame(poi_prog_day, columns=["program", "day"])
poi_global_vmax = max(
    all_results[et]
    .merge(poi_prog_day_df, on=["program", "day"])["neg_log10_qval"]
    .max()
    for et in all_results
)
poi_global_vmax = max(poi_global_vmax, 1e-6)

for day in sorted(poi_by_day_3.keys()):
    progs_this_day = poi_by_day_3[day]
    col_labels = [f"p{prog}" for prog in progs_this_day]

    poi_sig_tfs = set()
    for et in all_results:
        res = all_results[et]
        mask = (
            res["program"].isin(set(progs_this_day))
            & (res["day"] == day)
            & res["significant"]
        )
        poi_sig_tfs |= set(res.loc[mask, "tf"])
    poi_sig_tfs_list = sorted(poi_sig_tfs)
    print(f"\n{day}: {len(poi_sig_tfs_list)} TFs significant in at least one program")
    if len(poi_sig_tfs_list) == 0:
        print(f"  {day}: no significant TFs, skipping")
        continue

    for et in et_order:
        hm_data = pd.DataFrame(index=poi_sig_tfs_list, columns=col_labels, dtype=float)
        hm_qval = pd.DataFrame(index=poi_sig_tfs_list, columns=col_labels, dtype=float)
        for prog, col in zip(progs_this_day, col_labels):
            sub = all_results[et][
                (all_results[et]["program"] == prog) & (all_results[et]["day"] == day)
            ].set_index("tf")
            hm_data[col] = sub.reindex(poi_sig_tfs_list)["neg_log10_qval"].values
            hm_qval[col] = sub.reindex(poi_sig_tfs_list)["qval"].values
        hm_data = hm_data.astype(float).fillna(0.0)
        hm_qval = hm_qval.astype(float).fillna(1.0)

        # Ward needs at least two observations on an axis to have anything to
        # merge; columns additionally need enough programs to be informative.
        row_cluster = len(poi_sig_tfs_list) > 1
        col_cluster = len(col_labels) >= poi_min_cols_to_cluster

        # clustermap runs tight_layout internally, so the tick labels are
        # pulled inside the figure: the figure must be wide enough to hold
        # them or the heatmap collapses. Measure them, then add the cell grid
        # and the dendrograms on top.
        label_fontsize = plt.rcParams["font.size"]
        ylabel_in = text_width_in(poi_sig_tfs_list, label_fontsize) + poi_label_pad_in
        # x tick labels are rotated 90 degrees, so their text width becomes
        # the height they occupy.
        xlabel_in = text_width_in(col_labels, label_fontsize) + poi_label_pad_in

        grid_w_in = len(col_labels) * poi_cell_in
        grid_h_in = len(poi_sig_tfs_list) * poi_cell_in
        fig_w_in = poi_row_dendro_in + grid_w_in + ylabel_in
        fig_h_in = poi_col_dendro_in + grid_h_in + xlabel_in
        # Ratios are relative to the dendrogram + grid area, since the label
        # allowance is consumed by tight_layout rather than the gridspec.
        dendro_ratio = (
            poi_row_dendro_in / (poi_row_dendro_in + grid_w_in),
            poi_col_dendro_in / (poi_col_dendro_in + grid_h_in),
        )

        # The corner above the row dendrogram is the only free area inside the
        # figure, so the colorbar goes there, horizontally.
        cbar_pos = (
            0.01,
            (xlabel_in + grid_h_in + 0.15) / fig_h_in,
            0.9 * poi_row_dendro_in / fig_w_in,
            0.25 * poi_col_dendro_in / fig_h_in,
        )

        cg = sns.clustermap(
            hm_data,
            cmap="Blues",
            vmin=0,
            vmax=poi_global_vmax,
            method="ward",
            row_cluster=row_cluster,
            col_cluster=col_cluster,
            linewidths=0.3,
            linecolor="white",
            figsize=(fig_w_in, fig_h_in),
            dendrogram_ratio=dendro_ratio,
            cbar_pos=cbar_pos,
            cbar_kws={"label": "-log10(q)", "orientation": "horizontal"},
            yticklabels=True,
            xticklabels=True,
        )

        # The stars are placed by position, so both orders have to come from
        # the dendrograms or they land on the wrong cells.
        reordered_tfs = (
            list(hm_data.index[cg.dendrogram_row.reordered_ind])
            if row_cluster
            else list(hm_data.index)
        )
        reordered_cols = (
            list(hm_data.columns[cg.dendrogram_col.reordered_ind])
            if col_cluster
            else list(hm_data.columns)
        )
        mark_significant(
            cg.ax_heatmap,
            hm_qval,
            reordered_tfs,
            reordered_cols,
            fontsize=poi_star_fontsize,
        )
        cg.ax_heatmap.yaxis.set_tick_params(rotation=0)
        cg.ax_heatmap.set_xlabel("")
        cg.ax_heatmap.set_ylabel("")
        # va="bottom" at y=1.0 puts the whole title above the axes area, so
        # it cannot overlap the column dendrogram; bbox_inches="tight" keeps it.
        cg.fig.suptitle(
            f"TF motif enrichment in programs of interest, {day}, {et}\n"
            f"{weight_strategy}, {hit_score_col}\n"
            f"*** q<0.001;  ** q<0.01;  * q<0.05",
            y=1.005,
            va="bottom",
        )

        if save_plots:
            stem = os.path.join(plots_dir, f"poi_by_day_clustermap_{day}_{et}")
            for ext in ("pdf", "png"):
                cg.fig.savefig(f"{stem}.{ext}", bbox_inches="tight", transparent=True)
            # Save in the plotted (clustered) order so the table matches the
            # figure row for row.
            hm_data.loc[reordered_tfs, reordered_cols].to_csv(
                f"{stem}_data.tsv", sep="\t"
            )
            print(f"  saved {os.path.basename(stem)}.pdf/.png")

# %%
# Plot 4: TF rank scatter for programs of interest (per day, primary element type)
# adjustText is only needed here, so it is imported lazily: a missing optional
# dependency should skip this one figure rather than abort the whole script.
try:
    from adjustText import adjust_text

    have_adjust_text = True
except ImportError:
    have_adjust_text = False
    print(
        "\n[WARN] adjustText not installed; skipping Plot 4 (TF rank scatter).\n"
        "       Existing tf_rank_scatter_*.pdf/.png are left untouched."
    )

poi_by_day = defaultdict(list)
for prog, day in poi_prog_day:
    poi_by_day[day].append(prog)
# %%
for day, progs in sorted(poi_by_day.items()) if have_adjust_text else []:
    n_progs = len(progs)
    fig, axes = plt.subplots(
        1, n_progs, figsize=(3 * n_progs, 5), sharey=True, sharex=True
    )
    if n_progs == 1:
        axes = [axes]

    prog_frames = []
    for prog in progs:
        sub = all_results[primary_et][
            (all_results[primary_et]["program"] == prog)
            & (all_results[primary_et]["day"] == day)
        ].copy()
        sub["rank"] = (
            sub["neg_log10_qval"].rank(ascending=False, method="min").astype(int)
        )
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
            ax.text(row["rank"], row["neg_log10_qval"], row["tf"], fontsize=8)
            for _, row in sub.head(7).iterrows()
        ]
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="gray", lw=0.8))
        ax.set_xlabel("TF motif rank")
        ax.set_title(f"program {prog}\nn={sub['tf'].nunique()}\n", size=10)
        if i == 0:
            ax.set_ylabel("-log10(q-value)")

    fig.suptitle(
        f"TF enrichment rank, {day}, {primary_et}, {weight_strategy}, {hit_score_col}",
        y=1.02,
        size=10,
    )
    plt.tight_layout()
    if save_plots:
        for prog, sub in zip(progs, prog_frames):
            sub.to_csv(
                os.path.join(plots_dir, f"tf_rank_scatter_p{prog}_{day}_data.tsv"),
                sep="\t",
                index=False,
            )
        save_figure(fig, os.path.join(plots_dir, f"tf_rank_scatter_{day}"))

# %%
print("\n=== Done ===")
print(f"Results in: {output_dir}")
# %%
