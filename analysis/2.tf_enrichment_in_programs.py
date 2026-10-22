"""
2.tf_enrichment_in_programs.py
Test TF motif enrichment in cNMF programs using scE2G gene-motif scores.

For every (program, day, tf) combination, restricted to distal enhancer
elements only (single element_type, so BH correction below is over one test
family -- no separate per-element-type families to (incorrectly) union
afterward):
  - Question   : among genes that carry the TF's motif in that day's linked
                 enhancers, do the program's top genes score higher?
  - Carriers   : cNMF genes with weighted_score > 0 for the TF on that day
                 (>= 1 pos_patterns hit in a linked enhancer); genes without a
                 hit are not part of the test
  - Foreground : carriers among the top n_top_genes by cNMF program loading
  - Background : all other carriers of the TF on that day
  - Filter     : a test runs only when the foreground and the background each have
                 at least min_carriers_per_group carriers (default 10, so the U
                 test is not based on a handful of genes). The filter uses only
                 who carries the motif, not the scores, so the conditional test
                 stays valid. Sensitivity (2026-10-21): >= 1 gave 12,817 tests and
                 465 significant, >= 5 8,588 and 493, >= 10 6,683 and 523, >= 20
                 5,027 and 553; the top motifs of the programs of interest are the
                 same at every cutoff. Requiring 10 or 20 background carriers
                 gives the same tests once the foreground has >= 10
  - Test       : one-sided Mann-Whitney U (alternative='greater').
                 H0: among carriers, scores of top genes and of other genes are
                 drawn from the same distribution.
                 Ha: among carriers, top genes score higher.
                 Scores are continuous and > 0 (no tied zeros), so scipy's
                 method='auto' asymptotic approximation is well calibrated
                 (checked on random 300-gene sets, d3, 2026-10-21)
  - Correction : Benjamini-Hochberg over all tests run
  Before 2026-10-21 the test was zero-filled (all genes with >= 1 hit of any TF,
  missing = 0). That mixes motif presence with score, its p-values were
  anti-conservative for rare motifs (~87% tied zeros), and presence tracks the
  number of linked enhancers.

Outputs
-------
  analysis/2.tf_enrichment_in_programs/
    {et}_enrichment_results_{run_tag}.tsv        run_tag = min{min_carriers_per_group}
    plots/
      poi_by_day_clustermap_{day}_{et}_{run_tag}.{pdf,png,_data.tsv}  clustered TFs x POI programs,
                                                            per day
  (TF rank scatter for the programs of interest: 3.chrombpnet_figures.py)
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
os.makedirs(plots_dir, exist_ok=True)

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
# Minimum motif carriers in the foreground and in the background for a test to run
# (default 10; 1 = every test with at least one carrier in each group). The environment variable
# MIN_CARRIERS_PER_GROUP overrides it, so several cutoffs can be run without editing
# (3.chrombpnet_figures.py reads the same variable). Outputs are tagged min{N}
min_carriers_per_group = int(os.environ.get("MIN_CARRIERS_PER_GROUP", 10))
run_tag = f"min{min_carriers_per_group}"
days = ["d0", "d1", "d2", "d3"]
# BH q cutoff for the one-sided (enrichment) test; the direction is fixed in
# advance, so q < 0.05 controls the FDR of enrichment calls at 5%
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

save_plots = True

# Reuse the {et}_enrichment_results.tsv already on disk instead of recomputing
# the Mann-Whitney tests. Everything below the compute stage depends only on
# all_results, so a plot-only change does not need the cNMF object, the
# gene-motif score tables, or the full test sweep. Set to False to recompute
# from scratch (required if n_top_genes, days, element_types, weight_strategy
# or hit_score_col change).
reuse_enrichment = False  # 1.0 scores changed (per-day links, hits fully inside)

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
            elif q < q_sig_threshold:
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
# Gene x TF score matrix per (element type, day); 0 marks genes without a hit of the
# TF (not carriers), which are left out of that TF's test below
score_matrices = {}
for et in compute_ets:
    if et not in all_scores:
        continue
    score_matrices[et] = {}
    df = all_scores[et]
    # Negative scores mean neg_patterns hits, which the one-sided test cannot handle
    if (df["weighted_score"] < 0).any():
        raise ValueError(
            f"{et}: negative weighted_score found; rerun 1.0.gene_motif_linking.py "
            "with pattern_prefixes = ('pos_patterns',)"
        )
    df = df[df["day"].isin(days) & df["TargetGene"].isin(all_cnmf_genes)]
    for day in days:
        mat = df[df["day"] == day].pivot_table(
            index="TargetGene",
            columns="tf_motif_id",
            values="weighted_score",
            aggfunc="sum",
            fill_value=0.0,
        )
        score_matrices[et][day] = mat
        n_carriers = (mat > 0).sum(axis=0)
        print(
            f"  {et} {day}: {mat.shape[1]} TFs; carriers per TF median "
            f"{int(n_carriers.median()):,} (range {n_carriers.min():,}-{n_carriers.max():,})"
        )

# %%
# Run Mann-Whitney U among motif carriers for all (program, day, tf)
# H1 (alternative='greater'): among carriers, top-N genes have higher TF scores
all_results = {}
for et in compute_ets:
    if et not in score_matrices:
        continue
    n_candidates = n_programs * sum(m.shape[1] for m in score_matrices[et].values())
    print(
        f"\n  [{et}] {n_programs} programs x {len(days)} days: {n_candidates:,} candidate tests ..."
    )

    records = []
    for prog_idx, prog_name in enumerate(program_loadings.index):
        top_genes_set = set(program_loadings.loc[prog_name].nlargest(n_top_genes).index)
        for day, mat in score_matrices[et].items():
            is_top = mat.index.isin(top_genes_set)
            for tf in mat.columns:
                values = mat[tf].to_numpy()
                carrier = values > 0
                fg = values[carrier & is_top]
                bg = values[carrier & ~is_top]
                if len(fg) < min_carriers_per_group or len(bg) < min_carriers_per_group:
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
                        "n_top_in_day_genes": int(is_top.sum()),
                        "median_score_top": float(np.median(fg)),
                        "median_score_background": float(np.median(bg)),
                        "U": u_stat,
                        "AUC": u_stat / (len(fg) * len(bg)),
                        "pval": pval,
                    }
                )
    records = [pd.DataFrame(records)]
    print(
        f"  [{et}] {len(records[0]):,} of {n_candidates:,} tests have >= "
        f"{min_carriers_per_group} carriers in both groups"
    )

    results = pd.concat(records, ignore_index=True)
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

    out_path = os.path.join(output_dir, f"{et}_enrichment_results_{run_tag}.tsv")
    results.to_csv(out_path, sep="\t", index=False)
    print(f"  [{et}] saved -> {out_path}")
# %%
# Reuse path: load the previously computed results so the plots below can be
# regenerated without redoing the test sweep.
if reuse_enrichment:
    for et in element_types:
        results_path = os.path.join(
            output_dir, f"{et}_enrichment_results_{run_tag}.tsv"
        )
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
et_order = [et for et in element_types if et in all_results]
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
            f"{weight_strategy}, {hit_score_col}; >= {min_carriers_per_group} carriers per group\n"
            f"*** q<0.001;  ** q<0.01;  * q<{q_sig_threshold}",
            y=1.005,
            va="bottom",
        )

        if save_plots:
            stem = os.path.join(
                plots_dir, f"poi_by_day_clustermap_{day}_{et}_{run_tag}"
            )
            for ext in ("pdf", "png"):
                cg.fig.savefig(f"{stem}.{ext}", bbox_inches="tight", transparent=True)
            # Save in the plotted (clustered) order so the table matches the
            # figure row for row.
            hm_data.loc[reordered_tfs, reordered_cols].to_csv(
                f"{stem}_data.tsv", sep="\t"
            )
            print(f"  saved {os.path.basename(stem)}.pdf/.png")

# %%
print("\n=== Done ===")
print(f"Results in: {output_dir}")
# %%
