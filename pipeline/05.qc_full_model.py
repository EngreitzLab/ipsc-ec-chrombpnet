"""
05.qc_full_model.py
Visualize ChromBPNet full-model performance metrics across all days and folds.

Reads from <full_model_dir>/<day>_<peak_type>_fold_<fold>/:
  - evaluation/chrombpnet_metrics.json            (Pearson R, Spearman R, median JSD)
  - evaluation/chrombpnet_nobias_max_bias_response.txt (Tn5 motif response in final model)
  - evaluation/chrombpnet_predictions.h5          (predicted log-counts)
  - auxiliary/data_unstranded.bw                  (observed ATAC-seq signal)

Produces:
  <out-dir>/model_metrics.tsv                      table of all metrics
  <out-dir>/performance_boxplot.pdf/png            counts Pearson R, JSD
  <out-dir>/tn5_response.pdf/png                   Tn5 motif response per day x fold
  <out-dir>/<day>_fold<fold>_scatter.pdf/png       predicted vs observed log-count scatter
  <out-dir>/<day>_fold<fold>_scatter_data.tsv      scatter plot data

Run after step 05 (train_full_model.sh) via 05.qc_run_full_model.sh.

Usage:
  python 05.qc_full_model.py \\
      --full-model-dir ../results/full_models \\
      --data-path      ../results/preprocessing \\
      --days d0 d1 d2 d3 d4 \\
      --folds 0 1 2 3 4 \\
      --peak-type all \\
      --out-dir ../results/plots/full_model_qc
"""

# %%
import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["axes.spines.right"] = False
matplotlib.rcParams["font.size"] = 10
matplotlib.rcParams["axes.labelsize"] = 10
matplotlib.rcParams["axes.titlesize"] = 10
matplotlib.rcParams["xtick.labelsize"] = 10
matplotlib.rcParams["ytick.labelsize"] = 10
matplotlib.rcParams["legend.fontsize"] = 10
matplotlib.rcParams["figure.dpi"] = 100
matplotlib.rcParams["savefig.dpi"] = 300
matplotlib.rcParams["savefig.bbox"] = "tight"
matplotlib.rcParams["savefig.transparent"] = True

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

# Test chromosomes per fold (standard ChromBPNet splits)
TEST_CHROMS = {
    "0": ["chr1", "chr3", "chr6"],
    "1": ["chr2", "chr8", "chr9", "chr16"],
    "2": ["chr4", "chr11", "chr12", "chr15", "chrY"],
    "3": ["chr5", "chr10", "chr14", "chr18", "chr20", "chr22"],
    "4": ["chr7", "chr13", "chr17", "chr19", "chr21", "chrX"],
}
NARROWPEAK_SCHEMA = [
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

DAY_COLORS = {
    "d0": "#A6A6A6",
    "d1": "#D3C72F",
    "d2": "#D37739",
    "d3": "#9B3A48",
    "d4": "#CD7986",
}


# %%
def load_metrics_json(path):
    with open(path) as f:
        return json.load(f)


def parse_bias_response(path):
    """Parse chrombpnet_nobias_max_bias_response.txt.

    Example content:  corrected_0.001_0.001/0.001/0.001/0.001/0.001
    Returns dict with keys tn5_1 ... tn5_5 (floats).
    """
    text = path.read_text().strip()
    text = re.sub(r"^corrected_", "", text)
    parts = re.split(r"[_/]", text)
    responses = {}
    for i, val in enumerate(parts[:5], start=1):
        try:
            responses[f"tn5_{i}"] = float(val)
        except ValueError:
            responses[f"tn5_{i}"] = float("nan")
    return responses


def load_pred_obs_counts(
    pred_h5_path,
    filtered_peaks_bed,
    full_model_dir,
    day,
    fold,
    peak_type,
    outputlen=1000,
):
    """Load predicted log-counts and observed log-counts.

    Observed counts come from auxiliary/data_unstranded.bw, written by ChromBPNet
    during training. Returns (obs_log_counts, pred_log_counts) for test chromosomes.
    """
    import chrombpnet.training.utils.data_utils as data_utils
    import h5py
    import pyBigWig

    with h5py.File(pred_h5_path, "r") as h5:
        pred_logcts = h5["predictions"]["logcounts"][:]
        chroms = np.array(
            [
                c.decode() if isinstance(c, bytes) else c
                for c in h5["coords"]["coords_chrom"][:]
            ]
        )

    test_chroms = TEST_CHROMS.get(fold, [])
    mask = np.isin(chroms, test_chroms)
    pred_logcts = pred_logcts[mask]

    peaks_df = pd.read_csv(
        filtered_peaks_bed, sep="\t", names=NARROWPEAK_SCHEMA
    )
    peaks_df = peaks_df[peaks_df["chr"].isin(test_chroms)].reset_index(
        drop=True
    )

    bw_file = (
        Path(full_model_dir)
        / f"{day}_{peak_type}_fold_{fold}"
        / "auxiliary"
        / "data_unstranded.bw"
    )
    if not bw_file.exists():
        print(
            f"  Warning: observed BigWig not found: {bw_file}", file=sys.stderr
        )
        return None, pred_logcts.flatten()

    try:
        bw = pyBigWig.open(str(bw_file))
        obs_data = data_utils.get_cts(peaks_df, bw, outputlen)
        bw.close()
        obs_logcts = np.log(np.sum(obs_data, axis=-1) + 1)
        return obs_logcts, pred_logcts.flatten()
    except Exception as e:
        print(
            f"  Warning: could not load observed counts: {e}", file=sys.stderr
        )
        return None, pred_logcts.flatten()


# %%
def box_with_points(ax, labels, group_values, ylabel, title, ylim=None):
    """Boxplot with jittered per-fold points overlaid."""
    colors = [DAY_COLORS.get(lbl, "#888888") for lbl in labels]
    x = np.arange(1, len(labels) + 1)

    bp = ax.boxplot(
        group_values,
        positions=x,
        widths=0.5,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(linewidth=1.0),
        capprops=dict(linewidth=1.0),
        flierprops=dict(marker=""),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    for xi, vals in zip(x, group_values):
        jitter = np.random.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(
            xi + jitter,
            vals,
            color="black",
            s=20,
            zorder=5,
            alpha=1,
            linewidths=0,
            rasterized=True,
        )
        median_val = np.median(vals)
        ax.text(
            xi,
            max(vals) + 0.01,
            f"median={median_val:.2f}\nn={len(vals)}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=12)
    if ylim:
        ax.set_ylim(*ylim)


def density_scatter(ax, x, y, bins=30):
    """2D-density coloured scatter."""
    from scipy.interpolate import interpn

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
    ax.scatter(
        x[idx],
        y[idx],
        c=z[idx],
        s=10,
        alpha=0.8,
        linewidths=0,
        cmap="viridis",
        rasterized=True,
    )


# %%
def parse_args():
    p = argparse.ArgumentParser(description="ChromBPNet full model QC plots")
    p.add_argument(
        "--full-model-dir",
        required=True,
        help="Directory containing per-day/fold model subdirectories (full_model_dir in config.sh)",
    )
    p.add_argument(
        "--data-path",
        required=True,
        help="Preprocessing directory with per-day peaks (data_path in config.sh)",
    )
    p.add_argument("--days", nargs="+", default=["d0", "d1", "d2", "d3", "d4"])
    p.add_argument("--folds", nargs="+", default=["0"])
    p.add_argument("--peak-type", default="all")
    p.add_argument("--out-dir", default="qc")
    p.add_argument("--save-plots", action="store_true", default=True)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    pearsonr_vals = {d: [] for d in args.days}
    spearmanr_vals = {d: [] for d in args.days}
    jsd_vals = {d: [] for d in args.days}
    tn5_vals = {d: {f"tn5_{i}": [] for i in range(1, 6)} for d in args.days}

    for day in args.days:
        for fold in args.folds:
            tag = f"{day}_{args.peak_type}_fold_{fold}"
            eval_dir = Path(args.full_model_dir) / tag / "evaluation"

            metrics_path = eval_dir / "chrombpnet_metrics.json"
            bias_resp_path = (
                eval_dir / "chrombpnet_nobias_max_bias_response.txt"
            )
            pred_h5_path = eval_dir / "chrombpnet_predictions.h5"
            peaks_bed = (
                Path(args.data_path)
                / day
                / f"{day}_{args.peak_type}_peaks_no_blacklist.narrowPeak"
            )

            if not metrics_path.exists():
                print(
                    f"[{tag}] metrics JSON not found, skipping.",
                    file=sys.stderr,
                )
                continue

            m = load_metrics_json(metrics_path)
            pr = m["counts_metrics"]["peaks"]["pearsonr"]
            sr = m["counts_metrics"]["peaks"]["spearmanr"]
            jsd = m["profile_metrics"]["peaks"]["median_jsd"]

            pearsonr_vals[day].append(pr)
            spearmanr_vals[day].append(sr)
            jsd_vals[day].append(jsd)

            tn5 = {}
            if bias_resp_path.exists():
                tn5 = parse_bias_response(bias_resp_path)
                for k, v in tn5.items():
                    tn5_vals[day][k].append(v)

            rows.append(
                {
                    "day": day,
                    "fold": fold,
                    "pearsonr": pr,
                    "spearmanr": sr,
                    "median_jsd": jsd,
                    **tn5,
                }
            )

            # Predicted vs observed scatter
            if pred_h5_path.exists() and peaks_bed.exists():
                try:
                    obs, pred = load_pred_obs_counts(
                        pred_h5_path,
                        peaks_bed,
                        args.full_model_dir,
                        day,
                        fold,
                        args.peak_type,
                    )
                    if obs is not None:
                        ok = ~(np.isnan(obs) | np.isnan(pred))
                        obs_ok, pred_ok = obs[ok], pred[ok]
                        rp = pearsonr(obs_ok, pred_ok)[0]
                        rs = spearmanr(obs_ok, pred_ok)[0]

                        pd.DataFrame(
                            {
                                "obs_log_counts": obs_ok,
                                "pred_log_counts": pred_ok,
                            }
                        ).to_csv(
                            out_dir / f"{day}_fold{fold}_scatter_data.tsv",
                            sep="\t",
                            index=False,
                        )

                        if args.save_plots:
                            fig, ax = plt.subplots(figsize=(5, 5))
                            density_scatter(ax, obs_ok, pred_ok)
                            ax.set_xlabel(
                                "Observed log(counts + 1)", fontsize=12
                            )
                            ax.set_ylabel("Predicted log-counts", fontsize=12)
                            ax.set_title(f"{day} fold {fold}", fontsize=12)
                            ax.text(
                                0.05,
                                0.95,
                                f"Pearson={rp:.3f}\nSpearman={rs:.3f}",
                                transform=ax.transAxes,
                                fontsize=10,
                                va="top",
                            )
                            fig.tight_layout()
                            for ext in ("pdf", "png"):
                                fig.savefig(
                                    out_dir / f"{day}_fold{fold}_scatter.{ext}"
                                )
                            plt.close(fig)
                except Exception as e:
                    print(f"[{tag}] Scatter plot failed: {e}", file=sys.stderr)

    # Save metrics table
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(out_dir / "model_metrics.tsv", sep="\t", index=False)
    print(metrics_df.to_string())

    days_present = [d for d in args.days if pearsonr_vals[d]]
    if not days_present:
        print("No metrics found. Exiting.")
        return

    # Performance boxplots
    if args.save_plots:
        fig, axes = plt.subplots(1, 2, figsize=(6, 5))
        box_with_points(
            axes[0],
            days_present,
            [pearsonr_vals[d] for d in days_present],
            "Pearson's r",
            "Counts",
            ylim=(0.6, 1),
        )
        box_with_points(
            axes[1],
            days_present,
            [jsd_vals[d] for d in days_present],
            "Median JSD",
            "Profile",
            ylim=(0.6, 1),
        )
        fig.suptitle(
            "ChromBPNet model performance (peaks, test chromosomes)",
            fontsize=11,
            y=1.02,
        )
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(out_dir / f"performance_boxplot.{ext}")
        plt.close(fig)

    # Tn5 motif response plots
    n_motifs = 5
    tn5_rows = []
    for day in days_present:
        for fold in args.folds:
            row_match = metrics_df[
                (metrics_df.day == day) & (metrics_df.fold == fold)
            ]
            if row_match.empty:
                continue
            for i in range(1, n_motifs + 1):
                col = f"tn5_{i}"
                val = (
                    row_match[col].values[0]
                    if col in row_match.columns
                    else float("nan")
                )
                tn5_rows.append(
                    {"day": day, "fold": fold, "motif": col, "response": val}
                )

    if tn5_rows and args.save_plots:
        tn5_df = pd.DataFrame(tn5_rows)

        fig, axes = plt.subplots(
            1, n_motifs, figsize=(2.5 * n_motifs, 3.5), sharey=True
        )
        if n_motifs == 1:
            axes = [axes]

        for ax, motif in zip(
            axes, [f"tn5_{i}" for i in range(1, n_motifs + 1)]
        ):
            sub = tn5_df[tn5_df.motif == motif]
            vals_by_day = [
                sub[sub.day == d]["response"].values for d in days_present
            ]
            box_with_points(
                ax,
                days_present,
                vals_by_day,
                "Max response" if ax == axes[0] else "",
                motif,
                ylim=(0, None),
            )
            ax.axhline(
                0.003,
                color="red",
                linestyle="--",
                linewidth=0.8,
                label="threshold (0.003)",
            )
            ax.tick_params(axis="x", labelrotation=45)

        axes[0].legend(fontsize=7)
        fig.suptitle(
            "Tn5 motif response in chrombpnet_nobias\n(low = bias well factorized)",
            fontsize=10,
            y=1.02,
        )
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(out_dir / f"tn5_response.{ext}")
        plt.close(fig)

    print(f"\nAll outputs written to: {out_dir}/")


if __name__ == "__main__":
    main()
