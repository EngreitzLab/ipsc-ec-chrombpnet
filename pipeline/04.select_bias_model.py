"""
select_bias_model.py
====================
Select the best ChromBPNet Tn5 bias model per fold by evaluating
bias_metrics.json files produced after 'chrombpnet bias pipeline'.

Selection criteria (from ChromBPNet developers):
  Counts metrics
    - nonpeaks pearsonr > 0              (higher = better)
    - peaks   pearsonr > -0.3            (below -0.3 is warning; below -0.5 fail)
    - peaks   MSE will be high (expected, not a selection criterion)
  Profile metrics
    - peaks median_jsd       lower = better
    - peaks median_norm_jsd  higher = better

Per-fold selection:
  1. Among bias models that PASS both count thresholds, pick the one with
     the highest peaks median_norm_jsd (tie-break: lowest median_jsd).
  2. If no bias passes both, relax to peaks pearsonr > -0.5 and repeat.
  3. If still no candidate, pick the least-bad (highest peaks pearsonr).

Usage (interactive or batch):
  python 04.select_bias_model.py \
      --core-path /path/to/multiome_ipsc_ec \
      --biases 06 07 08 \
      --folds  0 1 2 3 4 \
      --day    d0 --peak-type all \
      --out-dir qc/bias_model_selection
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── thresholds ──────────────────────────────────────────────────────────────
NONPEAKS_PEARSONR_PASS = 0.0  # must be > 0
PEAKS_PEARSONR_WARN = -0.3  # between -0.3 and -0.5 → warning
PEAKS_PEARSONR_FAIL = -0.5  # below -0.5 → fail


# ── data loading ─────────────────────────────────────────────────────────────


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


# ── selection logic ───────────────────────────────────────────────────────────


def classify_row(row) -> str:
    """Return 'pass', 'warn', or 'fail' for a single (bias, fold) row."""
    np_ok = row["nonpeaks_pearsonr"] > NONPEAKS_PEARSONR_PASS
    pk_ok = row["peaks_pearsonr"] > PEAKS_PEARSONR_WARN  # > -0.3
    pk_warn = row["peaks_pearsonr"] > PEAKS_PEARSONR_FAIL  # > -0.5

    if np_ok and pk_ok:
        return "pass"
    elif pk_warn:  # nonpeaks failing but peaks not too bad
        return "warn"
    else:
        return "fail"


def select_best(group: pd.DataFrame) -> str:
    """Given all bias models for one fold, return the best bias string."""
    group = group.copy()
    group["status"] = group.apply(classify_row, axis=1)

    for tier in ("pass", "warn", "fail"):
        candidates = group[group["status"] == tier]
        if not candidates.empty:
            # prefer highest norm_jsd; tie-break lowest jsd
            idx = (
                candidates["peaks_median_norm_jsd"]
                .sub(candidates["peaks_median_jsd"] / 100)  # small penalty
                .idxmax()
            )
            return candidates.loc[idx, "bias"]
    return group["bias"].iloc[0]


def build_selection_table(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for fold, grp in df.groupby("fold"):
        best = select_best(grp)
        row = grp.set_index("bias").loc[best].to_dict()
        row["fold"] = fold
        row["selected_bias"] = best
        row["status"] = classify_row(grp.set_index("bias").loc[best])
        records.append(row)
    return pd.DataFrame(records).set_index("fold").sort_index()


# ── plotting ─────────────────────────────────────────────────────────────────

BIAS_COLORS = {"06": "#4C72B0", "07": "#DD8452", "08": "#55A868"}
STATUS_COLORS = {"pass": "#2ca02c", "warn": "#ff7f0e", "fail": "#d62728"}

METRIC_CONFIG = [
    (
        "nonpeaks_pearsonr",
        "Nonpeaks Pearson r\n(should be > 0)",
        NONPEAKS_PEARSONR_PASS,
        "above",
        None,
    ),
    (
        "peaks_pearsonr",
        "Peaks Pearson r\n(should be > −0.3)",
        PEAKS_PEARSONR_WARN,
        "above",
        PEAKS_PEARSONR_FAIL,
    ),
    (
        "peaks_median_jsd",
        "Peaks median JSD\n(lower = better)",
        None,
        None,
        None,
    ),
    (
        "peaks_median_norm_jsd",
        "Peaks median norm JSD\n(higher = better)",
        None,
        None,
        None,
    ),
]


def plot_metrics(
    df: pd.DataFrame, selection: pd.DataFrame, out_path: Path
) -> None:
    """
    Plot 1 – 4-panel grouped bar chart showing each key metric across
    folds, with bars coloured by bias model.
    Threshold lines are drawn where applicable; pass/warn/fail regions shaded.
    Winning bias per fold is marked with a star.
    """
    folds = sorted(df["fold"].unique())
    biases = sorted(df["bias"].unique())
    n_fold = len(folds)
    n_bias = len(biases)
    w = 0.8 / n_bias
    x = np.arange(n_fold)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, (metric, ylabel, thresh1, direction, thresh2) in zip(
        axes, METRIC_CONFIG
    ):
        for i, bias in enumerate(biases):
            vals = [
                df[(df["bias"] == bias) & (df["fold"] == f)][metric].values
                for f in folds
            ]
            vals = [v[0] if len(v) else np.nan for v in vals]
            bars = ax.bar(
                x + (i - n_bias / 2 + 0.5) * w,
                vals,
                width=w * 0.9,
                color=BIAS_COLORS.get(bias, "gray"),
                label=f"bias_{bias}",
                alpha=0.85,
                zorder=3,
            )

        # threshold lines
        if thresh1 is not None:
            ax.axhline(
                thresh1,
                color="black",
                lw=1.2,
                ls="--",
                label=f"threshold = {thresh1}",
                zorder=4,
            )
        if thresh2 is not None:
            ax.axhline(
                thresh2,
                color="red",
                lw=1.0,
                ls=":",
                label=f"fail threshold = {thresh2}",
                zorder=4,
            )

        # shade region below pass threshold (for pearsonr plots)
        ylims = ax.get_ylim()
        if direction == "above" and thresh1 is not None:
            ax.axhspan(ylims[0], thresh1, color="tomato", alpha=0.08, zorder=1)
        if thresh2 is not None:
            ax.axhspan(ylims[0], thresh2, color="red", alpha=0.06, zorder=0)

        # stars for winner
        row_y = max(
            df[(df["fold"] == f)][metric].max()
            for f in folds
            if not df[df["fold"] == f][metric].empty
        )
        for fi, fold in enumerate(folds):
            best_bias = selection.loc[fold, "selected_bias"]
            val = df[(df["bias"] == best_bias) & (df["fold"] == fold)][
                metric
            ].values
            if len(val):
                bi = biases.index(best_bias)
                xpos = fi + (bi - n_bias / 2 + 0.5) * w
                ax.annotate(
                    "★",
                    xy=(xpos, val[0]),
                    xytext=(xpos, val[0] + 0.003),
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    color=BIAS_COLORS.get(best_bias, "black"),
                    zorder=5,
                )

        ax.set_xticks(x)
        ax.set_xticklabels([f"fold {f}" for f in folds])
        ax.set_ylabel(ylabel, fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=7, loc="best")

    fig.suptitle(
        "ChromBPNet bias model evaluation\n(★ = selected per fold)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_selection_heatmap(
    df: pd.DataFrame, selection: pd.DataFrame, out_path: Path
) -> None:
    """
    Plot 2 – Traffic-light heatmap.
    Rows = folds, columns = bias models.
    Each cell is coloured green/yellow/red by pass/warn/fail status.
    The selected model is outlined and annotated with its key metrics.
    """
    folds = sorted(df["fold"].unique())
    biases = sorted(df["bias"].unique())

    # status matrix
    status_mat = pd.DataFrame(index=folds, columns=biases, dtype=str)
    for _, row in df.iterrows():
        status_mat.loc[row["fold"], row["bias"]] = classify_row(row)

    fig, ax = plt.subplots(
        figsize=(len(biases) * 2.8 + 1, len(folds) * 1.4 + 1.5)
    )

    color_map = {
        "pass": "#2ca02c",
        "warn": "#ff7f0e",
        "fail": "#d62728",
        "": "lightgray",
    }

    for ri, fold in enumerate(folds):
        for ci, bias in enumerate(biases):
            st = status_mat.loc[fold, bias]
            color = color_map.get(st, "lightgray")
            fc = plt.matplotlib.colors.to_rgba(color, alpha=0.35)
            rect = mpatches.FancyBboxPatch(
                (ci + 0.05, ri + 0.05),
                0.9,
                0.9,
                boxstyle="round,pad=0.05",
                linewidth=1,
                edgecolor="gray",
                facecolor=fc,
                transform=ax.transData,
                zorder=2,
            )
            ax.add_patch(rect)

            # metric annotations
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
                    color="black",
                )
                ax.text(
                    ci + 0.5,
                    ri + 0.52,
                    f"pk_r={r['peaks_pearsonr']:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="black",
                )
                ax.text(
                    ci + 0.5,
                    ri + 0.32,
                    f"nJSD={r['peaks_median_norm_jsd']:.3f}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="black",
                )

            # outline winner
            if selection.loc[fold, "selected_bias"] == bias:
                outline = mpatches.FancyBboxPatch(
                    (ci + 0.02, ri + 0.02),
                    0.96,
                    0.96,
                    boxstyle="round,pad=0.05",
                    linewidth=3,
                    edgecolor="navy",
                    facecolor="none",
                    transform=ax.transData,
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

    # legend
    legend_patches = [
        mpatches.Patch(
            color=color_map["pass"],
            alpha=0.5,
            label="PASS  (np_r>0, pk_r>-0.3)",
        ),
        mpatches.Patch(
            color=color_map["warn"],
            alpha=0.5,
            label="WARN  (pk_r in [-0.5, -0.3])",
        ),
        mpatches.Patch(
            color=color_map["fail"], alpha=0.5, label="FAIL  (pk_r < -0.5)"
        ),
        mpatches.Patch(
            facecolor="none",
            linewidth=2,
            edgecolor="navy",
            label="Selected model (✓)",
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
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── summary printing ──────────────────────────────────────────────────────────


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
    # overall recommendation
    winner_counts = selection["selected_bias"].value_counts()
    top_bias = winner_counts.index[0]
    top_n = winner_counts.iloc[0]
    print(
        f"\nOverall: bias_{top_bias} wins {top_n}/{len(selection)} folds → "
        f'set bias_suffix="_{top_bias}" in config.sh\n'
    )


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description="Select ChromBPNet bias model per fold"
    )
    p.add_argument(
        "--core-path",
        default="/oak/stanford/groups/engreitz/Users/opushkar/multiome_ipsc_ec",
    )
    p.add_argument("--biases", nargs="+", default=["06", "07", "08"])
    p.add_argument("--folds", nargs="+", default=["0", "1", "2", "3", "4"])
    p.add_argument("--day", default="d0")
    p.add_argument("--peak-type", default="all")
    p.add_argument("--out-dir", default="../results/bias_models/selection")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
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

    # save tables
    df.to_csv(out_dir / "all_bias_metrics.tsv", sep="\t", index=False)
    selection.to_csv(out_dir / "selected_bias_per_fold.tsv", sep="\t")
    print(f"Tables written to {out_dir}/")

    # plots
    plot_metrics(df, selection, out_dir / "bias_model_metrics.png")
    plot_selection_heatmap(
        df, selection, out_dir / "bias_model_selection_heatmap.png"
    )

    print_summary(selection)
    print(f"All outputs in: {out_dir}/")


if __name__ == "__main__":
    main()
