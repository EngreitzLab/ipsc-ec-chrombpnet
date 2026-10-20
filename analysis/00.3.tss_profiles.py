#!/usr/bin/env python3
"""
00.3.tss_profiles.py
TSS enrichment profiles for all five timepoints (d0-d4) overlaid in one axes.

Reads the 1 bp insertion histograms produced by 00.3.tss_profiles.sh and
normalizes each to its own flanking background, following the ArchR / ENCODE
convention:

  background = mean insertions per bp over the outer flank_norm_bp of each
               side of the +/- 2 kb window
  profile    = insertions per bp / background

so the flanks sit at 1.0 and the value at the TSS is the fold enrichment, on
the same scale as the per cell TSS enrichment values plotted in
00.2.atac_percell_qc. A centered rolling mean is applied for display; both raw
and smoothed values are saved. All three parameters (flank 2000, norm 100,
smooth 11) are ArchR's plotTSSEnrichment defaults, chosen so this profile
follows the same convention as the ArchR per cell metric it sits next to.
Note the smoothing is not needed to suppress noise: with ~10,000 to 16,000
insertions per bp near the peak the Poisson CV per bin is only about 1 percent.
It costs roughly 10 percent of the raw single bin maximum by blunting a
genuinely sharp apex, so the peak values reported here are smoothed values.

Inputs
  analysis/00.3.tss_profiles/tables/tss_profile_{day}.tsv
  (distance_from_tss, insertions)

Outputs (analysis/00.3.tss_profiles/)
  plots/tss_enrichment_profiles.{pdf,png}   all five days, one axes
  plots/tss_enrichment_profiles_data.tsv    normalized profiles, long format
  tables/tss_profile_summary.tsv            peak height, peak position, totals
  CONCLUSIONS.txt

Run
  ml biology py-scanpy/1.10.2_py312
  python3 analysis/00.3.tss_profiles.py
"""

# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["font.size"] = 14
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
plt.rcParams["legend.fontsize"] = 14
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["savefig.transparent"] = True

# %%
core_path = "/path/to/ipsc-ec-chrombpnet"

days = ["d0", "d1", "d2", "d3", "d4"]
day_colors = {
    "d0": "#A6A6A6",
    "d1": "#D3C72F",
    "d2": "#D37739",
    "d3": "#9B3A48",
    "d4": "#CD7986",
}

save_plots = True
flank_bp = 2000  # half width of the profile window, must match the .sh
flank_norm_bp = 100  # outer bp per side used as the background, ArchR default
smooth_bp = 11  # centered rolling mean window for display, odd.
# 11 is ArchR's plotTSSEnrichment default, as are
# flank_bp=2000 and flank_norm_bp=100 above, so the
# profile matches the convention behind the per cell
# TSS enrichment values used in 00.2.atac_percell_qc.

output_path = os.path.join(core_path, "analysis", "00.3.tss_profiles")
plots_dir = os.path.join(output_path, "plots")
tables_dir = os.path.join(output_path, "tables")
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(tables_dir, exist_ok=True)


# %%
def load_profile(day):
    """Read one day's 1 bp insertion histogram and validate its extent."""
    path = os.path.join(tables_dir, f"tss_profile_{day}.tsv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"missing profile for {day}: {path}\n"
            "run analysis/00.3.tss_profiles.sh first"
        )
    df = pd.read_csv(path, sep="\t")
    if list(df.columns) != ["distance_from_tss", "insertions"]:
        raise ValueError(f"{day}: unexpected columns {list(df.columns)}")
    if len(df) != 2 * flank_bp + 1:
        raise ValueError(f"{day}: expected {2 * flank_bp + 1} bins, found {len(df)}")
    df = df.sort_values("distance_from_tss").reset_index(drop=True)
    if df["insertions"].sum() == 0:
        raise ValueError(f"{day}: profile is empty")
    return df


def normalize_profile(df):
    """Divide by the flanking background so the flanks sit at 1.0."""
    distance = df["distance_from_tss"].values
    insertions = df["insertions"].values.astype(float)

    in_flank = (distance <= -(flank_bp - flank_norm_bp + 1)) | (
        distance >= (flank_bp - flank_norm_bp + 1)
    )
    background = insertions[in_flank].mean()
    if background <= 0:
        raise ValueError("flanking background is zero; cannot normalize")
    return insertions / background, background, int(in_flank.sum())


def rolling_mean(values, window):
    """Centered rolling mean, edges shrink rather than turn into NaN."""
    return (
        pd.Series(values)
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .values
    )


# %%
profiles = {}
summary_rows = []
for day in days:
    df = load_profile(day)
    normalized, background, n_flank_bins = normalize_profile(df)
    smoothed = rolling_mean(normalized, smooth_bp)

    profiles[day] = pd.DataFrame(
        {
            "day": day,
            "distance_from_tss": df["distance_from_tss"].values,
            "insertions": df["insertions"].values,
            "enrichment": normalized,
            "enrichment_smoothed": smoothed,
        }
    )

    peak_idx = int(np.argmax(smoothed))
    summary_rows.append(
        {
            "day": day,
            "total_tss_proximal_insertions": int(df["insertions"].sum()),
            "background_insertions_per_bp": background,
            "n_background_bins": n_flank_bins,
            "peak_enrichment_smoothed": smoothed[peak_idx],
            "peak_position_bp": int(df["distance_from_tss"].values[peak_idx]),
            "enrichment_at_tss": normalized[df["distance_from_tss"] == 0][0],
            "enrichment_at_tss_smoothed": smoothed[df["distance_from_tss"].values == 0][
                0
            ],
        }
    )
    print(
        f"[{day}] {int(df['insertions'].sum()):>12,} insertions; "
        f"background {background:.2f}/bp; "
        f"peak {smoothed[peak_idx]:.2f} at {df['distance_from_tss'].values[peak_idx]:+d} bp"
    )

profile_df = pd.concat(profiles.values(), ignore_index=True)
summary_df = pd.DataFrame(summary_rows)

summary_path = os.path.join(tables_dir, "tss_profile_summary.tsv")
summary_df.to_csv(summary_path, sep="\t", index=False)
print(f"\nsaved table {summary_path}")


# %%
fig, ax = plt.subplots(figsize=(5.5, 5))

for day in days:
    sub = profiles[day]
    ax.plot(
        sub["distance_from_tss"],
        sub["enrichment_smoothed"],
        color=day_colors[day],
        linewidth=1.8,
        label=f"{day}; peak {sub['enrichment_smoothed'].max():.1f}",
    )

ax.axhline(1.0, color="gray", linestyle=":", linewidth=1)
ax.axvline(0, color="gray", linestyle=":", linewidth=1)

ax.set_xlim(-flank_bp, flank_bp)
ax.set_ylim(bottom=0)
ax.set_xlabel("Distance from TSS (bp)")
ax.set_ylabel("TSS enrichment (insertions over flanking background)")
ax.set_title(
    f"Aggregate TSS enrichment profile by day\n"
    f"{smooth_bp} bp rolling mean; background is the outer {flank_norm_bp} bp per side"
)
ax.legend(frameon=False, title="Timepoint")

fig.tight_layout()

if save_plots:
    for ext in ("pdf", "png"):
        path = os.path.join(plots_dir, f"tss_enrichment_profiles.{ext}")
        fig.savefig(path)
    print(f"saved plot {os.path.join(plots_dir, 'tss_enrichment_profiles')}.pdf/.png")

    data_path = os.path.join(plots_dir, "tss_enrichment_profiles_data.tsv")
    profile_df.to_csv(data_path, sep="\t", index=False)
    print(f"saved plot data {data_path}")


# %%
tss_windows_path = os.path.join(output_path, "work", f"tss_windows_{flank_bp}bp.bed")
if os.path.exists(tss_windows_path):
    with open(tss_windows_path) as handle:
        n_tss = sum(1 for _ in handle)
else:
    n_tss = "unknown"

percell_summary = os.path.join(
    core_path,
    "analysis",
    "00.2.atac_percell_qc",
    "tables",
    "atac_qc_summary_by_day.tsv",
)
percell_note = ""
if os.path.exists(percell_summary):
    percell = pd.read_csv(percell_summary, sep="\t")
    medians = percell[percell["metric"] == "atac_tss_enrichment"].set_index("day")[
        "median"
    ]
    compare = summary_df.set_index("day")[["peak_enrichment_smoothed"]].join(medians)
    compare.columns = ["aggregate_profile_peak", "per_cell_median_tsse"]
    percell_note = compare.round(2).to_string()

conclusions = f"""CONCLUSIONS: aggregate TSS enrichment profiles, d0-d4
Generated by analysis/00.3.tss_profiles.py from the histograms built by
analysis/00.3.tss_profiles.sh

METHOD
  Two 1 bp insertion sites per fragment (both ends), counted against every
  TSS within {flank_bp} bp, strand aware, in 1 bp bins. TSS set is scE2G's
  CollapsedGeneBounds hg38 GENCODE v43 deduplicated by symbol and restricted
  to chr1-22, X, Y ({n_tss} TSSs), the same annotation the ArchR per cell QC
  used, so these profiles are on the same footing as the per cell values in
  00.2.atac_percell_qc. Only RNA called cells are counted. Each profile is
  divided by its own flanking background, the mean insertions per bp over the
  outer {flank_norm_bp} bp of each side, so the flanks sit at 1.0 and the
  curve reads directly as fold enrichment. Display uses a {smooth_bp} bp
  centered rolling mean; raw bins are in the plot data table.

SUMMARY OF FINDINGS
{summary_df.set_index("day")[["peak_enrichment_smoothed", "peak_position_bp", "total_tss_proximal_insertions"]].round(2).to_string()}

  Aggregate profile peak against the per cell median from 00.2:
{percell_note if percell_note else "  (00.2 summary table not found; skipped)"}

  These two numbers answer different questions and are not expected to match
  exactly. The aggregate profile is depth weighted, so cells contributing more
  fragments dominate it, whereas the per cell median weights every cell
  equally. A profile peak above the per cell median is the normal consequence
  of higher depth cells also having higher signal to background.

LIMITATIONS
  - All five profiles are built from unfiltered called cells, including the
    low fragment, low enrichment barcodes visible in
    00.2.atac_percell_qc/plots/tss_vs_fragments.pdf. Those barcodes flatten
    every curve by a similar amount, so between day comparisons hold, but the
    absolute peak heights are lower than a filtered analysis would give.
  - Insertions near two TSSs are counted once per TSS, the standard ENCODE
    and ArchR behaviour; bidirectional promoters therefore contribute twice.
  - No duplicate, blacklist, or mitochondrial filtering was applied beyond
    what is already baked into the fragment files.
  - Depth differs by day (d4 is the shallowest, see 00.2), and signal to
    background is depth sensitive, so part of any difference in peak height
    between days reflects library depth rather than chromatin quality.

SUGGESTED HYPOTHESES
  - If the curves are near superimposable, chromatin accessibility data
    quality is stable across the differentiation and no timepoint needs to be
    down weighted in downstream ATAC analyses.
  - Any day whose curve is both lower and broader is the signature of
    increased background rather than lost promoter signal, which would point
    at the tagmentation or nuclei prep for that timepoint.

POTENTIAL NEXT STEPS
  - Recompute the profiles on cells passing an ATAC filter (for example
    fragments >= 1000 and per cell TSS enrichment >= 4) to separate the
    contribution of low quality barcodes from real per day differences.
  - Split each day by the channel column in the RNA h5ad obs to test whether
    any difference tracks the two 10x channels rather than the timepoint.
  - Add the fragment size distribution as a companion panel; ArchR already
    wrote per day versions under
    scE2G_qc/generated_for_submission/<day>/archr_work_<day>/
    QualityControl/<day>/<day>-Fragment_Size_Distribution.pdf
"""

conclusions_path = os.path.join(output_path, "CONCLUSIONS.txt")
with open(conclusions_path, "w") as handle:
    handle.write(conclusions)
print(f"saved {conclusions_path}")
print("\n" + conclusions)

# %%
