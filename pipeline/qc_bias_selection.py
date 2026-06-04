"""
qc_bias_selection.py
Compare ChromBPNet models trained with different Tn5 bias threshold factors
(-b parameter in 'chrombpnet bias pipeline') and select the optimal value.

The bias factor controls how aggressively the Tn5 bias model captures sequence
preferences. A good bias factor achieves:
  (1) LOW Tn5 motif response in the final chrombpnet_nobias model
      (ideally < 0.003 for all 5 Tn5 motifs)
  (2) HIGH counts Spearman/Pearson correlation on the test set

Usage:
  python qc_bias_selection.py \
      --core-path  /path/to/multiome_ipsc_ec \
      --days       d0 d1 d2 d3 d4 \
      --folds      0 3 \
      --peak-type  all \
      --bias-suffixes _04 _05 _06 _07 _08 \
      --out-dir    qc/bias_selection

Requires:
  Trained full models in results/bias_models/bias_model<suffix>/
  The bias factor value is inferred from the directory name (e.g. _08 -> 0.8),
  but can be overridden with --bias-factor-map.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TEST_CHROMS = {
    "0": ["chr1",  "chr3",  "chr6"],
    "1": ["chr2",  "chr8",  "chr9",  "chr16"],
    "2": ["chr4",  "chr11", "chr12", "chr15", "chrY"],
    "3": ["chr5",  "chr10", "chr14", "chr18", "chr20", "chr22"],
    "4": ["chr7",  "chr13", "chr17", "chr19", "chr21", "chrX"],
}


def parse_bias_response(path: Path) -> dict:
    """Parse chrombpnet_nobias_max_bias_response.txt.

    Content example: corrected_0.001_0.001/0.001/0.001/0.001/0.001
    Returns {"tn5_1": float, ..., "tn5_5": float}.
    """
    text = path.read_text().strip()
    text = re.sub(r"^corrected_", "", text)
    parts = re.split(r"[_/]", text)
    result = {}
    for i, val in enumerate(parts[:5], start=1):
        try:
            result[f"tn5_{i}"] = float(val)
        except ValueError:
            result[f"tn5_{i}"] = float("nan")
    return result


def suffix_to_factor(suffix: str) -> str:
    """Best-effort: convert '_08' → '0.8', '_04' → '0.4', etc."""
    digits = re.sub(r"[^0-9]", "", suffix)
    if len(digits) == 2:
        return f"0.{digits[-1]}" if digits[0] == "0" else f"{digits[0]}.{digits[1]}"
    return suffix


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def collect_metrics(
    core_path: Path,
    bias_suffixes: list[str],
    days: list[str],
    folds: list[str],
    peak_type: str,
    factor_map: dict,
) -> pd.DataFrame:
    rows = []
    for suffix in bias_suffixes:
        factor_label = factor_map.get(suffix, suffix_to_factor(suffix))
        full_model_dir = core_path / "results" / "bias_models" / f"bias_model{suffix}"

        for day in days:
            for fold in folds:
                tag      = f"{day}_{peak_type}_fold_{fold}"
                eval_dir = full_model_dir / tag / "evaluation"

                metrics_path   = eval_dir / "chrombpnet_metrics.json"
                bias_resp_path = eval_dir / "chrombpnet_nobias_max_bias_response.txt"

                if not metrics_path.exists():
                    print(f"  [{suffix} {tag}] metrics JSON not found, skipping.",
                          file=sys.stderr)
                    continue

                with open(metrics_path) as f:
                    m = json.load(f)
                pr  = m["counts_metrics"]["peaks"]["pearsonr"]
                sr  = m["counts_metrics"]["peaks"]["spearmanr"]
                jsd = m["profile_metrics"]["peaks"]["median_jsd"]

                tn5 = {}
                if bias_resp_path.exists():
                    tn5 = parse_bias_response(bias_resp_path)
                else:
                    print(f"  [{suffix} {tag}] bias response file missing.",
                          file=sys.stderr)

                rows.append({
                    "bias_suffix":  suffix,
                    "bias_factor":  factor_label,
                    "day":          day,
                    "fold":         fold,
                    "pearsonr":     pr,
                    "spearmanr":    sr,
                    "median_jsd":   jsd,
                    "max_tn5":      max(tn5.values()) if tn5 else float("nan"),
                    **tn5,
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

TN5_THRESHOLD = 0.003   # empirical "good bias" threshold (from Greenleaf lab)
PERFORMANCE_THRESHOLD_SR = 0.5  # minimum acceptable Spearman R


def plot_selection(df: pd.DataFrame, out_dir: Path) -> None:
    """Two-panel plot:
      Left:   model performance (Spearman R) vs bias factor, one line per day
      Right:  max Tn5 motif response vs bias factor, same layout
    """
    if df.empty:
        print("No data to plot.", file=sys.stderr)
        return

    # Average across folds for cleaner plot
    agg = (
        df.groupby(["bias_factor", "day"], sort=False)
        .agg(
            spearmanr=("spearmanr", "mean"),
            pearsonr=("pearsonr",  "mean"),
            max_tn5= ("max_tn5",   "mean"),
        )
        .reset_index()
    )

    days = df["day"].unique()
    factors = sorted(agg["bias_factor"].unique())
    cmap = plt.cm.get_cmap("tab10", len(days))
    day_color = {d: cmap(i) for i, d in enumerate(days)}

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # --- Spearman R ---
    ax = axes[0]
    for day in days:
        sub = agg[agg.day == day].set_index("bias_factor").reindex(factors)
        ax.plot(factors, sub["spearmanr"], marker="o", label=day,
                color=day_color[day])
    ax.axhline(PERFORMANCE_THRESHOLD_SR, color="gray", linestyle="--",
               linewidth=0.8, label=f"threshold={PERFORMANCE_THRESHOLD_SR}")
    ax.set_xlabel("Bias threshold factor")
    ax.set_ylabel("Counts Spearman R")
    ax.set_title("Model performance vs bias factor")
    ax.legend(fontsize=7, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(bottom=0)

    # --- Max Tn5 response ---
    ax = axes[1]
    for day in days:
        sub = agg[agg.day == day].set_index("bias_factor").reindex(factors)
        ax.plot(factors, sub["max_tn5"], marker="o", label=day,
                color=day_color[day])
    ax.axhline(TN5_THRESHOLD, color="red", linestyle="--", linewidth=0.8,
               label=f"target max Tn5 < {TN5_THRESHOLD}")
    ax.set_xlabel("Bias threshold factor")
    ax.set_ylabel("Max Tn5 motif response")
    ax.set_title("Tn5 motif response vs bias factor\n(lower = better bias capture)")
    ax.legend(fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_dir / "bias_selection_overview.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved bias_selection_overview.pdf")


def plot_tn5_per_motif(df: pd.DataFrame, out_dir: Path) -> None:
    """Show each Tn5 motif's response separately, averaged across days."""
    tn5_cols = [c for c in df.columns if c.startswith("tn5_") and c != "max_tn5"]
    if not tn5_cols:
        return

    factors = sorted(df["bias_factor"].unique())
    agg = df.groupby("bias_factor")[tn5_cols].mean().reindex(factors)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    for col in tn5_cols:
        ax.plot(factors, agg[col], marker="o", label=col)
    ax.axhline(TN5_THRESHOLD, color="red", linestyle="--", linewidth=0.8,
               label=f"target < {TN5_THRESHOLD}")
    ax.set_xlabel("Bias threshold factor")
    ax.set_ylabel("Tn5 motif response (mean across days)")
    ax.set_title("Per-motif Tn5 response vs bias factor")
    ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "bias_selection_per_motif.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved bias_selection_per_motif.pdf")


def recommend(df: pd.DataFrame) -> None:
    """Print a simple recommendation based on the two criteria."""
    if df.empty:
        return

    agg = (
        df.groupby("bias_factor")
        .agg(mean_sr=("spearmanr", "mean"), mean_tn5=("max_tn5", "mean"))
        .reset_index()
    )

    # Candidates: Tn5 response below threshold AND best Spearman R
    good = agg[agg["mean_tn5"] < TN5_THRESHOLD]
    if good.empty:
        best = agg.loc[agg["mean_tn5"].idxmin()]
        print(
            f"\n[WARN] No bias factor achieves Tn5 response < {TN5_THRESHOLD}.\n"
            f"  Closest: bias_factor={best['bias_factor']}  "
            f"(max_tn5={best['mean_tn5']:.4f}, SpearmanR={best['mean_sr']:.3f})\n"
            "  Consider training with a lower bias_factor (e.g. -b 0.4)."
        )
    else:
        rec = good.loc[good["mean_sr"].idxmax()]
        print(
            f"\n[RECOMMENDATION] Best bias factor: {rec['bias_factor']}\n"
            f"  Spearman R = {rec['mean_sr']:.3f}  "
            f"max Tn5 response = {rec['mean_tn5']:.4f}"
        )

    print("\nFull summary (averaged across days and folds):")
    print(agg.to_string(index=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Bias factor selection for ChromBPNet")
    p.add_argument("--core-path",      required=True,
                   help="Root project directory (contains chrombpnet_full_model_XX)")
    p.add_argument("--days",           nargs="+", default=["d0","d1","d2","d3","d4"])
    p.add_argument("--folds",          nargs="+", default=["0"])
    p.add_argument("--peak-type",      default="all")
    p.add_argument("--bias-suffixes",  nargs="+", required=True,
                   help="E.g. _04 _05 _06 _07 _08 (matches chrombpnet_full_model<suffix>)")
    p.add_argument("--bias-factor-map", nargs="+", default=[],
                   metavar="SUFFIX=LABEL",
                   help="Override auto-inferred factor label, e.g. _08=0.8")
    p.add_argument("--out-dir",        default="qc/bias_selection")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    factor_map = {}
    for item in args.bias_factor_map:
        k, v = item.split("=", 1)
        factor_map[k] = v

    df = collect_metrics(
        core_path      = Path(args.core_path),
        bias_suffixes  = args.bias_suffixes,
        days           = args.days,
        folds          = args.folds,
        peak_type      = args.peak_type,
        factor_map     = factor_map,
    )

    if df.empty:
        print("No metrics found. Check that full models have been trained.", file=sys.stderr)
        sys.exit(1)

    df.to_csv(out_dir / "bias_selection_metrics.tsv", sep="\t", index=False)
    print(f"Metrics table written to bias_selection_metrics.tsv")

    plot_selection(df, out_dir)
    plot_tn5_per_motif(df, out_dir)
    recommend(df)

    print(f"\nAll outputs in: {out_dir}/")


if __name__ == "__main__":
    main()
