"""
04.1.qc_bias_selection.py
Visualize Tn5 bias model evaluation metrics across different threshold factors
(-b parameter in 'chrombpnet bias pipeline') to complement 04.select_bias_model.py.

Reads *_bias_metrics.json produced by step 03 (train_bias_model.sh).
Run after step 03, before or alongside 04.select_bias.sh.

Both scripts work on the same files: this one shows aggregate cross-factor trends
across all folds; 04.0.select_bias_model.py performs per-fold selection.

Usage:
  python 04.1.qc_bias_selection.py \\
      --core-path  /path/to/multiome_ipsc_ec \\
      --biases     05 06 07 08 \\
      --folds      0 1 2 3 4 \\
      --bias-day   d0 \\
      --peak-type  all
"""

# %%
import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["font.size"] = 10
mpl.rcParams["axes.labelsize"] = 10
mpl.rcParams["axes.titlesize"] = 10
mpl.rcParams["xtick.labelsize"] = 10
mpl.rcParams["ytick.labelsize"] = 10
mpl.rcParams["legend.fontsize"] = 10
mpl.rcParams["figure.dpi"] = 100
mpl.rcParams["savefig.dpi"] = 300
mpl.rcParams["savefig.bbox"] = "tight"
mpl.rcParams["savefig.transparent"] = True

# Thresholds (consistent with 04.select_bias_model.py)
NONPEAKS_PEARSONR_PASS = 0.0
PEAKS_PEARSONR_WARN = -0.3
PEAKS_PEARSONR_FAIL = -0.5

METRIC_CONFIG = [
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
    ("peaks_median_norm_jsd", "Peaks median norm JSD\n(higher = better)", None),
]

STATUS_COLORS = {"pass": "#2ca02c", "warn": "#ff7f0e", "fail": "#d62728"}


# %%
def load_metrics(core_path, biases, bias_day, peak_type, folds):
    """Load *_bias_metrics.json for all (bias, fold) combinations."""
    rows = []
    for bias in biases:
        bias_dir = core_path / "results" / "bias_models" / f"bias_model_{bias}"
        for fold in folds:
            tag = f"{bias_day}_{peak_type}_fold_{fold}"
            path = bias_dir / tag / "evaluation" / f"{tag}_bias_metrics.json"
            if not path.exists():
                print(f"  [MISSING] {path}", file=sys.stderr)
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
                    "peaks_pearsonr": cm["peaks"]["pearsonr"],
                    "peaks_median_jsd": pm["peaks"]["median_jsd"],
                    "peaks_median_norm_jsd": pm["peaks"]["median_norm_jsd"],
                }
            )
    return pd.DataFrame(rows)


def classify_status(row):
    """Return pass/warn/fail for one (bias, fold) row (same criteria as 04.select_bias_model.py)."""
    np_ok = row["nonpeaks_pearsonr"] > NONPEAKS_PEARSONR_PASS
    pk_ok = row["peaks_pearsonr"] > PEAKS_PEARSONR_WARN
    pk_warn = row["peaks_pearsonr"] > PEAKS_PEARSONR_FAIL
    if np_ok and pk_ok:
        return "pass"
    elif pk_warn:
        return "warn"
    return "fail"


# %%
def plot_bias_metrics(df, out_dir, save_plots):
    """4-panel line plot: each metric vs bias factor, one line per fold.

    Points are colored by pass/warn/fail status (same criteria as 04.select_bias_model.py).
    Threshold lines shown where applicable.
    """
    df = df.copy()
    df["status"] = df.apply(classify_status, axis=1)
    biases = sorted(df["bias"].unique())
    folds = sorted(df["fold"].unique())
    x_pos = {b: i for i, b in enumerate(biases)}
    x_labels = [f"{int(b) / 10:.1f}" for b in biases]
    fold_colors = plt.cm.tab10(np.linspace(0, 0.9, len(folds)))

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))

    for ax, (metric, ylabel, thresh) in zip(axes, METRIC_CONFIG):
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
        "Point color = pass/warn/fail status; line color = fold",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()

    if save_plots:
        for ext in ("pdf", "png"):
            fig.savefig(out_dir / f"bias_selection_overview.{ext}")
        plt.close(fig)
        print("Saved bias_selection_overview.pdf, bias_selection_overview.png")


def recommend(df):
    """Print aggregated recommendation across folds."""
    df = df.copy()
    df["status"] = df.apply(classify_status, axis=1)

    agg = (
        df.groupby("bias")
        .agg(
            mean_np_pearsonr=("nonpeaks_pearsonr", "mean"),
            mean_pk_pearsonr=("peaks_pearsonr", "mean"),
            mean_jsd=("peaks_median_jsd", "mean"),
            mean_norm_jsd=("peaks_median_norm_jsd", "mean"),
            n_pass=("status", lambda x: (x == "pass").sum()),
            n_warn=("status", lambda x: (x == "warn").sum()),
            n_fail=("status", lambda x: (x == "fail").sum()),
        )
        .reset_index()
    )

    print("\nBias factor summary (averaged across folds):")
    print(agg.to_string(index=False))

    good = agg[agg["n_fail"] == 0]
    if good.empty:
        best = agg.loc[agg["mean_pk_pearsonr"].idxmax()]
        print("\n[WARN] All bias factors have at least one failing fold.")
        print(
            f"  Least-bad: bias_{best['bias']} "
            f"(mean pk_r={best['mean_pk_pearsonr']:+.3f})"
        )
    else:
        rec = good.loc[good["mean_norm_jsd"].idxmax()]
        print(
            f"\n[RECOMMENDATION] bias_{rec['bias']}  "
            f"(mean pk_r={rec['mean_pk_pearsonr']:+.3f}, "
            f"mean norm_jsd={rec['mean_norm_jsd']:.3f})"
        )
    print("Run 04.select_bias.sh for per-fold selection and config.sh update.")


# %%
def parse_args():
    p = argparse.ArgumentParser(
        description="Bias factor QC visualization for ChromBPNet"
    )
    p.add_argument("--core-path", required=True, help="Project root directory")
    p.add_argument(
        "--biases",
        nargs="+",
        default=["05", "06", "07", "08"],
        help="Bias factor suffixes to compare (e.g. 05 06 07 08)",
    )
    p.add_argument("--folds", nargs="+", default=["0", "1", "2", "3", "4"])
    p.add_argument(
        "--bias-day",
        default="d0",
        help="Day used for bias model training (bias_day in config.sh)",
    )
    p.add_argument("--peak-type", default="all")
    p.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: <core-path>/results/plots/bias_selection)",
    )
    p.add_argument("--save-plots", action="store_true", default=True)
    return p.parse_args()


def main():
    args = parse_args()
    core_path = Path(args.core_path)
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else core_path / "results" / "plots" / "bias_selection"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_metrics(
        core_path=core_path,
        biases=args.biases,
        bias_day=args.bias_day,
        peak_type=args.peak_type,
        folds=args.folds,
    )

    if df.empty:
        print(
            "No bias metrics found. "
            "Check that step 03 (train_bias_model.sh) has completed.",
            file=sys.stderr,
        )
        sys.exit(1)

    df.to_csv(out_dir / "bias_selection_metrics.tsv", sep="\t", index=False)
    print(f"Metrics table written to {out_dir}/bias_selection_metrics.tsv")

    plot_bias_metrics(df, out_dir, save_plots=args.save_plots)
    recommend(df)

    print(f"\nAll outputs in: {out_dir}/")


if __name__ == "__main__":
    main()
