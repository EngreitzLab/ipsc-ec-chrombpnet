"""
3.chrombpnet_figures.py
Publication figure of TF motif enrichment for the programs of interest, from the
enhancer results of 2.tf_enrichment_in_programs.py.

TF rank scatter per day, one panel per program of interest (motifs ranked by
-log10(BH-adjusted P), top motifs labeled).
The per-day clustermaps of the programs of interest are made by
2.tf_enrichment_in_programs.py (poi_by_day_clustermap_{day}_enhancer_{run_tag}).
"""

# %%
import os
import re
from collections import defaultdict

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from adjustText import adjust_text

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
output_dir = os.path.join(core_path, "analysis", "3.chrombpnet_figures")
plots_dir = os.path.join(output_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)

et = "enhancer"
et_display = "distal enhancers"
days = ["d0", "d1", "d2", "d3"]
# BH q cutoff for the one-sided (enrichment) test; the direction is fixed in
# advance, so q < 0.05 controls the FDR of enrichment calls at 5%
q_sig_threshold = 0.05
# Carrier cutoff of the 2.tf_enrichment_in_programs.py run to plot (same environment
# variable and min{N} tag as there)
min_carriers_per_group = int(os.environ.get("MIN_CARRIERS_PER_GROUP", 10))
run_tag = f"min{min_carriers_per_group}"

# Programs of interest and the day each is most active
poi_prog_day = [
    (2, "d3"),
    (15, "d2"),
    (17, "d2"),
    (24, "d3"),
    (27, "d3"),
    (34, "d3"),
]

save_plots = True


# %%
# Helper functions


def save_figure(fig, path_stem):
    fig.savefig(path_stem + ".pdf")
    fig.savefig(path_stem + ".png")


def strip_number_suffix(tf_list):
    """
    Display names: drop the "_<n>" suffix (FOS-JUN_0 -> FOS-JUN) unless another
    motif shares the same family name, in which case the suffix is kept so the
    labels stay unambiguous (cluster_121 stays cluster_121).
    """
    family = {tf: re.sub(r"_.*$", "", tf) for tf in tf_list}
    counts = pd.Series(list(family.values())).value_counts()
    return {tf: fam if counts[fam] == 1 else tf for tf, fam in family.items()}


# %%
# Load enhancer enrichment results
results_path = os.path.join(enrichment_dir, f"{et}_enrichment_results_{run_tag}.tsv")
results = pd.read_csv(results_path, sep="\t")
results["significant"] = results["qval"] < q_sig_threshold
print(
    f"Loaded {len(results):,} rows; "
    f"{results['tf'].nunique()} TFs, {results['program'].nunique()} programs"
)

# %%
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

    fig.suptitle(
        f"TF enrichment rank, {day}, {et_display}; >= {min_carriers_per_group} carriers per group",
        y=1.02,
        size=10,
    )
    plt.tight_layout()
    if save_plots:
        for prog, sub in zip(progs, prog_frames):
            sub.to_csv(
                os.path.join(
                    plots_dir, f"tf_rank_scatter_p{prog}_{day}_{run_tag}_data.tsv"
                ),
                sep="\t",
                index=False,
            )
        save_figure(fig, os.path.join(plots_dir, f"tf_rank_scatter_{day}_{run_tag}"))
        print(f"  saved tf_rank_scatter_{day}_{run_tag}")

# %%
print("\n Done ")
print(f"Plots in: {plots_dir}")

# %%
