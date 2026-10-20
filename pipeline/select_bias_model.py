"""
select_bias_model.py
Select the best ChromBPNet Tn5 bias model per fold.

Plots (PDF + PNG):
  bias_model_selection_heatmap    – traffic-light fold x bias heatmap
  bias_selection_overview         – line plot: metrics vs bias factor per fold
  bias_model_metrics_scatter      – scatter: peaks vs nonpeaks Pearson r, sized by norm JSD

Tables (TSV):
  all_bias_metrics.tsv            – all raw metrics per bias, fold
  selected_bias_per_fold.tsv      – final selection per fold

Text:
  bias_selection_explanation.txt  – verbal rationale and action items per fold

Selection criteria (ChromBPNet developer guidelines):
  Counts metrics:
    nonpeaks Pearson r > 0       (required; higher is better)
    peaks Pearson r > -0.3       (warn: -0.3 to -0.5; fail: < -0.5)
    peaks MSE will be high       (expected; not a selection criterion)
  Profile metrics:
    peaks median norm JSD        (higher = better; primary criterion)
    peaks median JSD             (lower = better; secondary tie-break)
  Tie-break (within NORM_JSD_SCORE_TIE_REL_EPS of the top norm-JSD score):
    1. Prefer peaks Pearson r <= nonpeaks Pearson r (peaks r shouldn't
       exceed the in-domain nonpeaks fit; if it does, the model likely
       learned real peak signal, not just Tn5 bias)
    2. Then peaks Pearson r closest to 0, unless the closest candidates are
       within PEAKS_PEARSONR_NEAR_ZERO_EPS of each other (both "close
       enough to 0"), in which case:
    3. ...prefer higher nonpeaks Pearson r instead

Usage:
  python select_bias_model.py \\
    --core-path /path/to/ipsc-ec-chrombpnet \\
    --biases 05 06 07 08 \\
    --folds 0 1 2 3 4 \\
    --day d0 \\
    --peak-type all
"""

# %%
import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.use("Agg")

mpl.rcParams.update(
    {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.transparent": True,
    }
)

# thresholds (ChromBPNet developer guidelines)
NONPEAKS_PEARSONR_PASS = 0.0
PEAKS_PEARSONR_WARN = -0.3
PEAKS_PEARSONR_FAIL = -0.5

# within a tier, candidates whose norm-JSD score is within this fraction of
# the best score are treated as a tie (chrombpnet's own docs only note norm
# JSD is read-depth sensitive, i.e. noisy - not how noisy) and broken by
# |peaks pearsonr| closest to 0, since chrombpnet's QC report explicitly
# flags peaks pearsonr as diagnostic of the bias model picking up AT-content
# bias (see chrombpnet/helpers/generate_reports/make_html_bias.py)
NORM_JSD_SCORE_TIE_REL_EPS = 0.02

# among norm-JSD-tied, "healthy" (pk_r <= np_r) candidates, |peaks pearsonr|
# values within this absolute margin of each other are themselves treated
# as tied (both are "close enough to 0") and broken instead by nonpeaks
# pearsonr (higher = better, chrombpnet's own primary counts metric) -
# e.g. fold 4: bias_09 (pk_r=+0.05) vs bias_1 (pk_r=-0.04) are both near 0,
# so np_r (0.21 vs 0.13) decides, favoring bias_09
PEAKS_PEARSONR_NEAR_ZERO_EPS = 0.05

BIAS_COLORS = {
    "05": "#4393C3",
    "06": "#4C72B0",
    "07": "#DD8452",
    "08": "#55A868",
    "09": "red",
    "1": "mediumslateblue",
}
STATUS_COLORS = {"pass": "#2ca02c", "warn": "#ff7f0e", "fail": "#d62728"}


# %%
def load_metrics(
    core_path: Path,
    biases: list[str],
    day: str,
    peak_type: str,
    folds: list[str],
) -> pd.DataFrame:
    rows = []
    for bias in biases:
        bias_dir = core_path / "results" / "bias_models" / f"bias_model_{bias}"
        for fold in folds:
            tag = f"{day}_{peak_type}_fold_{fold}"
            path = bias_dir / tag / "evaluation" / f"{tag}_bias_metrics.json"
            if not path.exists():
                print(f"  [MISSING] {path}")
                continue
            with open(path) as f:
                m = json.load(f)
            cm = m["counts_metrics"]
            pm = m["profile_metrics"]
            rows.append(
                {
                    "bias": bias,
                    "fold": fold,
                    "nonpeaks_pearsonr": cm["nonpeaks"]["pearsonr"],
                    "nonpeaks_spearmanr": cm["nonpeaks"]["spearmanr"],
                    "peaks_pearsonr": cm["peaks"]["pearsonr"],
                    "peaks_spearmanr": cm["peaks"]["spearmanr"],
                    "peaks_mse": cm["peaks"]["mse"],
                    "peaks_median_jsd": pm["peaks"]["median_jsd"],
                    "peaks_median_norm_jsd": pm["peaks"]["median_norm_jsd"],
                }
            )
    return pd.DataFrame(rows)


# %%
def classify_row(row) -> str:
    """Return 'pass', 'warn', or 'fail' for a single (bias, fold) row."""
    np_ok = row["nonpeaks_pearsonr"] > NONPEAKS_PEARSONR_PASS
    pk_ok = row["peaks_pearsonr"] > PEAKS_PEARSONR_WARN
    pk_survivable = row["peaks_pearsonr"] > PEAKS_PEARSONR_FAIL

    if np_ok and pk_ok:
        return "pass"
    elif pk_survivable:
        return "warn"
    else:
        return "fail"


def select_best(group: pd.DataFrame) -> str:
    """Given all bias models for one fold, return the best bias string.

    Ranks by norm-JSD score first. When candidates land within
    NORM_JSD_SCORE_TIE_REL_EPS of the top score (a tie, given norm JSD's
    known read-depth sensitivity), breaks the tie in three steps:
      1. Prefer candidates with peaks pearsonr <= nonpeaks pearsonr. The
         bias model is trained (and validated) on nonpeaks only, so peaks
         pearsonr is an out-of-domain generalization check; if it exceeds
         the in-domain nonpeaks pearsonr, the model is fitting something
         peak-specific (real accessibility signal), not just Tn5 bias -
         e.g. fold 0: bias_08 (np_r=0.18 > pk_r=0.16, healthy) vs bias_1
         (np_r=0.05 < pk_r=0.08, unhealthy) despite bias_1 having the
         smaller |pk_r| in isolation.
      2. Among the survivors, prefer |peaks pearsonr| closest to 0 - unless
         the closest candidates are themselves within
         PEAKS_PEARSONR_NEAR_ZERO_EPS of each other, in which case |pk_r|
         isn't discriminating (both are "close enough to 0") and step 3
         decides instead.
      3. Among candidates near-tied on |pk_r|, prefer higher nonpeaks
         pearsonr - e.g. fold 4: bias_09 (pk_r=+0.05, np_r=0.21) vs bias_1
         (pk_r=-0.04, np_r=0.13) have near-identical |pk_r|, so the model
         with the better in-domain (nonpeaks) fit wins.
    """
    group = group.copy()
    group["status"] = group.apply(classify_row, axis=1)
    group["norm_jsd_score"] = (
        group["peaks_median_norm_jsd"] - group["peaks_median_jsd"] / 100
    )

    for tier in ("pass", "warn", "fail"):
        candidates = group[group["status"] == tier]
        if candidates.empty:
            continue
        best_score = candidates["norm_jsd_score"].max()
        tied = candidates[
            candidates["norm_jsd_score"]
            >= best_score * (1 - NORM_JSD_SCORE_TIE_REL_EPS)
        ]
        healthy = tied[tied["peaks_pearsonr"] <= tied["nonpeaks_pearsonr"]]
        pool = healthy if not healthy.empty else tied

        abs_pk_r = pool["peaks_pearsonr"].abs()
        near_zero = pool[abs_pk_r <= abs_pk_r.min() + PEAKS_PEARSONR_NEAR_ZERO_EPS]
        idx = (
            near_zero["nonpeaks_pearsonr"].idxmax()
            if len(near_zero) > 1
            else abs_pk_r.idxmin()
        )
        return pool.loc[idx, "bias"]
    return group["bias"].iloc[0]


def build_selection_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per fold: the selected bias model and its metrics."""
    records = []
    for fold, grp in df.groupby("fold"):
        best = select_best(grp)
        row = grp.set_index("bias").loc[best].to_dict()
        row["fold"] = fold
        row["selected_bias"] = best
        row["status"] = classify_row(grp.set_index("bias").loc[best])
        records.append(row)
    return pd.DataFrame(records).set_index("fold").sort_index()


# %%
def generate_explanation(
    df: pd.DataFrame,
    selection: pd.DataFrame,
    day: str,
    biases: list[str],
    folds: list[str],
) -> str:
    """Human-readable report: thresholds, per-fold results, and actionable guidance."""
    df = df.copy()
    df["status"] = df.apply(classify_row, axis=1)

    lines = [
        "ChromBPNet Bias Model Selection Report",
        "=" * 60,
        f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Day       : {day}",
        f"Folds     : {', '.join(map(str, folds))}",
        f"Biases    : {', '.join(biases)}",
        "",
        "SELECTION CRITERIA (ChromBPNet developer guidelines)",
        "-" * 60,
        "Counts metrics:",
        "  Nonpeaks Pearson r > 0              required; higher is better",
        "  Peaks Pearson r > -0.3              warn if -0.3 to -0.5; fail if < -0.5",
        "  Peaks MSE will be high              expected; not a selection criterion",
        "Profile metrics:",
        "  Peaks median norm JSD               higher is better (primary criterion)",
        "  Peaks median JSD                    lower is better (secondary tie-break)",
        "",
        "SELECTION LOGIC",
        "-" * 60,
        "  1. Among PASS models (np_r > 0, pk_r > -0.3): select highest peaks norm JSD.",
        f"     Ties within {NORM_JSD_SCORE_TIE_REL_EPS:.0%} of the top norm-JSD score are broken by:",
        "       a) preferring pk_r <= np_r (peaks r beating the in-domain nonpeaks fit",
        "          suggests the model learned real peak signal, not just Tn5/AT-content",
        "          bias - see chrombpnet/helpers/generate_reports/make_html_bias.py),",
        "       b) then peaks Pearson r closest to 0,",
        f"       c) unless the closest candidates are within {PEAKS_PEARSONR_NEAR_ZERO_EPS}",
        "          of each other (both close enough to 0), in which case higher",
        "          nonpeaks Pearson r decides instead.",
        "  2. If no PASS model exists, relax to pk_r > -0.5 (WARN): repeat criterion.",
        "  3. If still none, select the least-bad model (highest peaks Pearson r).",
        "",
        "RESULTS BY FOLD",
        "-" * 60,
    ]

    warn_folds: list = []
    fail_folds: list = []

    for fold in sorted(selection.index):
        sel = selection.loc[fold]
        bias = sel["selected_bias"]
        status = sel["status"].upper()
        flag = {"PASS": "✓", "WARN": "⚠", "FAIL": "✗"}[status]

        sub = df[(df["fold"] == fold) & (df["bias"] == bias)]
        if sub.empty:
            continue
        r = sub.iloc[0]

        np_r = r["nonpeaks_pearsonr"]
        pk_r = r["peaks_pearsonr"]
        norm_jsd = r["peaks_median_norm_jsd"]
        jsd = r["peaks_median_jsd"]

        np_flag = "✓" if np_r > NONPEAKS_PEARSONR_PASS else "✗"
        if pk_r > PEAKS_PEARSONR_WARN:
            pk_flag = "✓"
        elif pk_r > PEAKS_PEARSONR_FAIL:
            pk_flag = "⚠"
        else:
            pk_flag = "✗"

        lines += [
            f"  Fold {fold}: bias_{bias}  [{status}]  {flag}",
            f"    Nonpeaks Pearson r    = {np_r:+.3f}  (threshold > 0)  {np_flag}",
            f"    Peaks Pearson r       = {pk_r:+.3f}  (warn < -0.3; fail < -0.5)  {pk_flag}",
            f"    Peaks median norm JSD = {norm_jsd:.3f}  (higher is better)",
            f"    Peaks median JSD      = {jsd:.4f}  (lower is better)",
        ]

        if status == "PASS":
            lines.append(
                "    Action: Model passes all criteria. Proceed with ChromBPNet training."
            )
        elif status == "WARN":
            warn_folds.append(fold)
            lines += [
                f"    Action: CAUTION — peaks Pearson r ({pk_r:+.3f}) is in the warning zone [-0.5, -0.3).",
                "      After running TFModisco, inspect the top-10 motifs:",
                "        If > 3 of the top-10 motifs are GC-rich: increase --bias_threshold_factor and retrain.",
                "        If <= 3 GC-rich motifs: the model is acceptable; proceed.",
            ]
        elif status == "FAIL":
            fail_folds.append(fold)
            lines += [
                f"    Action: FAIL — peaks Pearson r ({pk_r:+.3f}) < -0.5.",
                "      ChromBPNet training will automatically abort with this bias model.",
                "      Increase --bias_threshold_factor and retrain before proceeding.",
            ]
        lines.append("")

    winner_counts = selection["selected_bias"].value_counts()
    top_bias = winner_counts.index[0]
    top_n = winner_counts.iloc[0]

    lines += [
        "OVERALL RECOMMENDATION",
        "-" * 60,
        f"  bias_{top_bias} wins {top_n}/{len(selection)} folds.",
        f'  Set bias_suffix="_{top_bias}" in dataset_config.sh',
    ]

    if warn_folds:
        lines += [
            "",
            f"  WARNING: {len(warn_folds)} fold(s) in the warn zone: {warn_folds}",
            "    After TFModisco: check if > 3 of the top-10 motifs are GC-rich.",
            "    If so, increase --bias_threshold_factor and retrain.",
        ]
    if fail_folds:
        lines += [
            "",
            f"  FAIL: {len(fail_folds)} fold(s) failed threshold: {fail_folds}",
            "    These folds require retraining with a higher --bias_threshold_factor.",
        ]

    return "\n".join(lines)


# %%
def plot_selection_heatmap(
    df: pd.DataFrame, selection: pd.DataFrame, out_stem: Path
) -> None:
    """Traffic-light heatmap: rows = folds, columns = bias models."""
    folds = sorted(df["fold"].unique())
    biases = sorted(df["bias"].unique())

    status_mat = pd.DataFrame(index=folds, columns=biases, dtype=str)
    for _, row in df.iterrows():
        status_mat.loc[row["fold"], row["bias"]] = classify_row(row)

    color_map = {
        "pass": "#2ca02c",
        "warn": "#ff7f0e",
        "fail": "#d62728",
        "": "lightgray",
    }

    fig, ax = plt.subplots(figsize=(len(biases) * 2.8 + 1, len(folds) * 1.4 + 1.5))

    for ri, fold in enumerate(folds):
        for ci, bias in enumerate(biases):
            st = status_mat.loc[fold, bias]
            fc = mpl.colors.to_rgba(color_map.get(st, "lightgray"), alpha=0.35)
            rect = mpatches.FancyBboxPatch(
                (ci + 0.05, ri + 0.05),
                0.9,
                0.9,
                boxstyle="round,pad=0.05",
                linewidth=1,
                edgecolor="gray",
                facecolor=fc,
                zorder=2,
            )
            ax.add_patch(rect)

            sub = df[(df["bias"] == bias) & (df["fold"] == fold)]
            if not sub.empty:
                r = sub.iloc[0]
                ax.text(
                    ci + 0.5,
                    ri + 0.72,
                    f"np_r={r['nonpeaks_pearsonr']:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                )
                ax.text(
                    ci + 0.5,
                    ri + 0.52,
                    f"pk_r={r['peaks_pearsonr']:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                )
                ax.text(
                    ci + 0.5,
                    ri + 0.32,
                    f"nJSD={r['peaks_median_norm_jsd']:.3f}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                )

            if fold in selection.index and selection.loc[fold, "selected_bias"] == bias:
                outline = mpatches.FancyBboxPatch(
                    (ci + 0.02, ri + 0.02),
                    0.96,
                    0.96,
                    boxstyle="round,pad=0.05",
                    linewidth=3,
                    edgecolor="navy",
                    facecolor="none",
                    zorder=5,
                )
                ax.add_patch(outline)
                ax.text(
                    ci + 0.88,
                    ri + 0.88,
                    "✓",
                    ha="center",
                    va="center",
                    fontsize=11,
                    color="navy",
                    fontweight="bold",
                    zorder=6,
                )

    ax.set_xlim(0, len(biases))
    ax.set_ylim(0, len(folds))
    ax.set_xticks([c + 0.5 for c in range(len(biases))])
    ax.set_xticklabels(
        [f"bias_{b}\n(thresh {int(b) / 10:.1f})" for b in biases], fontsize=10
    )
    ax.set_yticks([r + 0.5 for r in range(len(folds))])
    ax.set_yticklabels([f"fold {f}" for f in folds], fontsize=10)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    legend_patches = [
        mpatches.Patch(
            color=color_map["pass"],
            alpha=0.5,
            label="PASS  (np_r > 0, pk_r > -0.3)",
        ),
        mpatches.Patch(
            color=color_map["warn"],
            alpha=0.5,
            label="WARN  (pk_r in [-0.5, -0.3])",
        ),
        mpatches.Patch(color=color_map["fail"], alpha=0.5, label="FAIL  (pk_r < -0.5)"),
        mpatches.Patch(
            facecolor="none",
            linewidth=2,
            edgecolor="navy",
            label="Selected (✓)",
        ),
    ]
    ax.legend(
        handles=legend_patches,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=4,
        fontsize=8,
        frameon=False,
    )
    fig.suptitle(
        "Bias model selection per fold\n"
        "np_r = nonpeaks Pearson r   pk_r = peaks Pearson r   nJSD = peaks median norm JSD",
        fontsize=10,
        y=1.02,
    )
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_stem.with_suffix(f".{ext}"))
    plt.close(fig)
    print(f"Saved: {out_stem}.pdf/.png")


# %%
def plot_bias_overview(df: pd.DataFrame, out_stem: Path) -> None:
    """
    Line plot: each metric vs bias factor, one line per fold.
    Points coloured by pass/warn/fail status; lines by fold.
    Threshold lines shown where applicable.
    """
    df = df.copy()
    df["status"] = df.apply(classify_row, axis=1)
    biases = sorted(df["bias"].unique())
    folds = sorted(df["fold"].unique())
    x_pos = {b: i for i, b in enumerate(biases)}
    x_labels = [f"{int(b) / 10:.1f}" for b in biases]
    fold_colors = plt.cm.tab10(np.linspace(0, 0.9, len(folds)))

    metrics = [
        (
            "nonpeaks_pearsonr",
            "Nonpeaks Pearson r\n(should be > 0)",
            NONPEAKS_PEARSONR_PASS,
        ),
        (
            "peaks_pearsonr",
            "Peaks Pearson r\n(should be > -0.3)",
            PEAKS_PEARSONR_WARN,
        ),
        ("peaks_median_jsd", "Peaks median JSD\n(lower = better)", None),
        (
            "peaks_median_norm_jsd",
            "Peaks median norm JSD\n(higher = better)",
            None,
        ),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))

    for ax, (metric, ylabel, thresh) in zip(axes, metrics):
        for fold, line_color in zip(folds, fold_colors):
            sub = df[df["fold"] == fold].sort_values("bias")
            xs = [x_pos[b] for b in sub["bias"]]
            ax.plot(
                xs,
                sub[metric],
                color=line_color,
                linewidth=1,
                label=f"fold {fold}",
                zorder=2,
            )
            for _, row in sub.iterrows():
                ax.scatter(
                    x_pos[row["bias"]],
                    row[metric],
                    color=STATUS_COLORS.get(row["status"], "#888888"),
                    s=55,
                    zorder=3,
                    linewidths=0,
                )

        if thresh is not None:
            ax.axhline(
                thresh,
                color="black",
                linestyle="--",
                linewidth=0.8,
                label=f"threshold = {thresh}",
            )

        ax.set_xlabel("Bias threshold factor")
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(len(biases)))
        ax.set_xticklabels(x_labels)
        ax.legend(fontsize=7, loc="best")

    status_handles = [
        mpatches.Patch(color=c, label=s) for s, c in STATUS_COLORS.items()
    ]
    fig.legend(
        handles=status_handles,
        title="Status",
        loc="lower center",
        ncol=3,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.06),
        frameon=False,
    )
    fig.suptitle(
        f"Bias model metrics by threshold factor (n={len(folds)} folds)\n"
        "Point color = pass/warn/fail; line color = fold",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_stem.with_suffix(f".{ext}"))
    plt.close(fig)

    df[["bias", "fold", "status"] + [m for m, *_ in metrics]].to_csv(
        out_stem.parent / (out_stem.name + "_data.tsv"), sep="\t", index=False
    )
    print(f"Saved: {out_stem}.pdf/.png")


# %%
def plot_metrics_scatter(
    df: pd.DataFrame, selection: pd.DataFrame, out_stem: Path
) -> None:
    """
    Single-glance summary: every (fold, bias) combo as one point.
    x = nonpeaks Pearson r (required > 0), y = peaks Pearson r
    (warn < -0.3, fail < -0.5), point size = peaks median norm JSD
    (higher = better; primary tie-break criterion). Color = bias model.
    Selected (fold, bias) pairs get a black outline and fold label.
    """
    selected_pairs = {
        (str(row["fold"]), row["selected_bias"])
        for _, row in selection.reset_index().iterrows()
    }

    # norm JSD -> marker area; sqrt scaling keeps area proportional to value
    jsd = df["peaks_median_norm_jsd"].to_numpy()
    jsd_min, jsd_max = np.nanmin(jsd), np.nanmax(jsd)
    size_min, size_max = 30, 300

    def jsd_to_size(v):
        if jsd_max == jsd_min:
            return (size_min + size_max) / 2
        frac = (v - jsd_min) / (jsd_max - jsd_min)
        return size_min + frac * (size_max - size_min)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    for _, row in df.iterrows():
        is_sel = (str(row["fold"]), row["bias"]) in selected_pairs
        ax.scatter(
            row["nonpeaks_pearsonr"],
            row["peaks_pearsonr"],
            s=jsd_to_size(row["peaks_median_norm_jsd"]),
            color=BIAS_COLORS.get(row["bias"], "gray"),
            edgecolors="black" if is_sel else "none",
            linewidth=1.5 if is_sel else 0,
            alpha=0.85,
            zorder=4 if is_sel else 3,
        )
        if is_sel:
            ax.annotate(
                f"fold {row['fold']}",
                (row["nonpeaks_pearsonr"], row["peaks_pearsonr"]),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=7,
            )

    ax.axvline(NONPEAKS_PEARSONR_PASS, color="black", lw=0.8, ls="--", zorder=1)
    ax.axhline(
        PEAKS_PEARSONR_WARN,
        color="black",
        lw=0.8,
        ls="--",
        label=f"warn = {PEAKS_PEARSONR_WARN}",
        zorder=1,
    )
    ax.axhline(
        PEAKS_PEARSONR_FAIL,
        color="red",
        lw=0.8,
        ls=":",
        label=f"fail = {PEAKS_PEARSONR_FAIL}",
        zorder=1,
    )
    ylims = ax.get_ylim()
    ax.axhspan(ylims[0], PEAKS_PEARSONR_FAIL, color="red", alpha=0.06, zorder=0)
    ax.axhspan(
        PEAKS_PEARSONR_FAIL,
        PEAKS_PEARSONR_WARN,
        color="tomato",
        alpha=0.06,
        zorder=0,
    )

    ax.set_xlabel("Nonpeaks Pearson r (required > 0)")
    ax.set_ylabel("Peaks Pearson r (warn < -0.3, fail < -0.5)")
    ax.set_title(
        "Bias model fit across the sweep\n"
        "point size = peaks median norm JSD (higher = better)"
    )

    color_handles = [
        mlines.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=BIAS_COLORS.get(b, "gray"),
            markersize=8,
            label=f"bias_{b}",
        )
        for b in sorted(df["bias"].unique())
    ]
    color_legend = ax.legend(
        handles=color_handles,
        title="Bias model",
        loc="upper left",
        fontsize=7,
        frameon=False,
    )
    ax.add_artist(color_legend)

    size_vals = np.linspace(jsd_min, jsd_max, 3)
    size_handles = [
        mlines.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="gray",
            markersize=(jsd_to_size(v) ** 0.5) / 2,
            label=f"{v:.2f}",
        )
        for v in size_vals
    ]
    ax.legend(
        handles=size_handles,
        title="Norm JSD",
        loc="lower right",
        fontsize=7,
        frameon=False,
    )

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_stem.with_suffix(f".{ext}"))
    plt.close(fig)

    out_data = df[
        ["bias", "fold", "nonpeaks_pearsonr", "peaks_pearsonr", "peaks_median_norm_jsd"]
    ].copy()
    out_data["selected"] = out_data.apply(
        lambda r: (str(r["fold"]), r["bias"]) in selected_pairs, axis=1
    )
    out_data.to_csv(
        out_stem.parent / (out_stem.name + "_data.tsv"), sep="\t", index=False
    )
    print(f"Saved: {out_stem}.pdf/.png")


# %%
def print_summary(selection: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("BIAS MODEL SELECTION SUMMARY")
    print("=" * 70)
    cols = [
        "selected_bias",
        "status",
        "nonpeaks_pearsonr",
        "peaks_pearsonr",
        "peaks_median_jsd",
        "peaks_median_norm_jsd",
    ]
    print(selection[cols].to_string(float_format="{:+.4f}".format))
    print("=" * 70)
    print("\nRecommended bias model per fold:")
    for fold, row in selection.iterrows():
        flag = {"pass": "✓", "warn": "⚠", "fail": "✗"}[row["status"]]
        print(
            f"  fold {fold}: bias_{row['selected_bias']}  {flag}  "
            f"[np_r={row['nonpeaks_pearsonr']:+.3f}  "
            f"pk_r={row['peaks_pearsonr']:+.3f}  "
            f"nJSD={row['peaks_median_norm_jsd']:.3f}]"
        )
    winner_counts = selection["selected_bias"].value_counts()
    top_bias = winner_counts.index[0]
    top_n = winner_counts.iloc[0]
    print(
        f"\nOverall: bias_{top_bias} wins {top_n}/{len(selection)} folds "
        f'set bias_suffix="_{top_bias}" in dataset_config.sh\n'
    )


# %%
def parse_args():
    p = argparse.ArgumentParser(description="Select ChromBPNet bias model per fold")
    p.add_argument("--core-path", required=True, help="Project root directory")
    p.add_argument("--biases", nargs="+", default=["05", "06", "07", "08"])
    p.add_argument("--folds", nargs="+", default=["0", "1", "2", "3", "4"])
    p.add_argument("--day", default="d0", help="Day used for bias model training")
    p.add_argument("--peak-type", default="all")
    p.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: <core-path>/results/plots/bias_model_selection)",
    )
    p.add_argument("--save-plots", action="store_true", default=True)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else Path(args.core_path) / "results" / "plots" / "bias_model_selection"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_metrics(
        core_path=Path(args.core_path),
        biases=args.biases,
        day=args.day,
        peak_type=args.peak_type,
        folds=args.folds,
    )

    if df.empty:
        print("No metrics found. Check that bias models have been evaluated.")
        return

    selection = build_selection_table(df)

    df.to_csv(out_dir / "all_bias_metrics.tsv", sep="\t", index=False)
    selection.to_csv(out_dir / "selected_bias_per_fold.tsv", sep="\t")
    print(f"Tables written to {out_dir}/")

    explanation = generate_explanation(df, selection, args.day, args.biases, args.folds)
    (out_dir / "bias_selection_explanation.txt").write_text(explanation)
    print(f"Explanation written to {out_dir}/bias_selection_explanation.txt")

    if args.save_plots:
        plot_selection_heatmap(df, selection, out_dir / "bias_model_selection_heatmap")
        plot_bias_overview(df, out_dir / "bias_selection_overview")
        plot_metrics_scatter(df, selection, out_dir / "bias_model_metrics_scatter")

    print_summary(selection)
    print(f"All outputs in: {out_dir}/")


if __name__ == "__main__":
    main()
