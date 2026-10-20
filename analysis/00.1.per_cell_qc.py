#!/usr/bin/env python3
"""
00.1.per_cell_qc.py
Per-cell 10x QC panels (FigS5 panel b): ATAC fragments/cell, RNA UMIs/cell,
TSS enrichment, and a joint RNA+ATAC UMAP, across d0-d4.

Barcodes match exactly between the per-day RNA h5ad
(Pseudobulks/RNA/rna_{day}.h5ad, obs/barcode) and the raw ATAC fragments
(Pseudobulks/ATAC/{day}.fragments.tsv.gz, column 4), confirmed on real data
(same {barcode}_{sample} format, d0 fragment file's unique-barcode count
matches the d0 RNA cell count exactly) -- this is genuine multiome data, so
a per-cell joint analysis is possible without extra inputs.

No scATAC-specific tool (snapatac2, episcanpy, ArchR, Signac) is installed,
and snapatac2 fails to build here (Rust/rust-lld toolchain incompatible with
this cluster's linker, confirmed with both the default and gcc/10.1.0
modules). Everything below is implemented directly with pybedtools/pandas/
sklearn/scanpy instead:

  - ATAC fragments/cell : plain per-barcode line count from the fragments file
  - TSS enrichment      : a standard center-window vs flank-window insertion
                           density ratio per cell (not a byte-for-byte
                           reproduction of ENCODE/ArchR's exact pipeline,
                           which has additional normalization steps -- this
                           is "a" reasonable TSS enrichment metric, not "the"
                           canonical one)
  - ATAC LSI            : per-day cell x peak count matrix (bedtools
                           intersect against that day's ChromBPNet peak set)
                           -> TF-IDF -> TruncatedSVD, dropping the first
                           component per standard scATAC-LSI convention
  - Joint UMAP          : RNA PCA (scanpy, on the pflog layer) concatenated
                           with ATAC LSI, z-scored, neighbors+UMAP -- a quick
                           joint embedding, not a true WNN

Outputs (analysis/00.1.per_cell_qc/):
  tables/atac_fragments_per_cell.tsv
  tables/rna_umis_per_cell.tsv
  tables/tss_enrichment_per_cell.tsv
  plots/frag_umi_tsse_per_day.{pdf,png}   3-panel violin, one row
  plots/joint_umap.{pdf,png}              pooled across days, colored by day
"""

# %%
import os
import subprocess
import tempfile

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pybedtools
import scanpy as sc
import seaborn as sns
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer

plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["font.size"] = 14
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["savefig.transparent"] = True

# %%
core_path = "/path/to/ipsc-ec-chrombpnet"
gex_dir = "/path/to/10x_5_timepoints/Pseudobulks/RNA"
frag_dir = "/path/to/10x_5_timepoints/Pseudobulks/ATAC"
preprocessing_dir = os.path.join(core_path, "results", "preprocessing")

days = ["d0", "d1", "d2", "d3", "d4"]
peak_type = "all"
day_colors = {
    "d0": "#A6A6A6",
    "d1": "#D3C72F",
    "d2": "#D37739",
    "d3": "#9B3A48",
    "d4": "#CD7986",
}

tss_center_bp = 50  # +/- around TSS for the "center" window
tss_flank_lo = 1000  # flank window starts this far from TSS
tss_flank_hi = 2000  # flank window ends this far from TSS
n_lsi_components = 30
n_pca_components = 30
main_chroms = {f"chr{i}" for i in range(1, 23)} | {"chrX", "chrY"}

output_path = os.path.join(core_path, "analysis", "00.1.per_cell_qc")
tables_dir = os.path.join(output_path, "tables")
plots_dir = os.path.join(output_path, "plots")
os.makedirs(tables_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)


# %%
def count_fragments_per_barcode(frag_file):
    """Per-barcode fragment count via a single-pass awk tally (column 4)."""
    result = subprocess.run(
        f"zcat {frag_file} | awk '{{c[$4]++}} END{{for (b in c) print b, c[b]}}'",
        shell=True,
        capture_output=True,
        text=True,
        check=True,
    )
    rows = [line.split() for line in result.stdout.splitlines()]
    return pd.DataFrame(rows, columns=["barcode", "n_fragments"]).astype(
        {"n_fragments": int}
    )


# %%
def build_tss_windows(var_df):
    """Center and flank BED windows per gene TSS, main chromosomes only."""
    var_df = var_df[var_df["chr"].isin(main_chroms)].copy()
    tss = np.where(var_df["strand"] == "+", var_df["start"], var_df["end"])
    var_df = var_df.assign(tss=tss)

    center = pd.DataFrame(
        {
            "chr": var_df["chr"],
            "start": (var_df["tss"] - tss_center_bp).clip(lower=0),
            "end": var_df["tss"] + tss_center_bp,
        }
    )
    flank_left = pd.DataFrame(
        {
            "chr": var_df["chr"],
            "start": (var_df["tss"] - tss_flank_hi).clip(lower=0),
            "end": (var_df["tss"] - tss_flank_lo).clip(lower=0),
        }
    )
    flank_right = pd.DataFrame(
        {
            "chr": var_df["chr"],
            "start": var_df["tss"] + tss_flank_lo,
            "end": var_df["tss"] + tss_flank_hi,
        }
    )
    flank = pd.concat([flank_left, flank_right], ignore_index=True)
    flank = flank[flank["end"] > flank["start"]]
    return center, flank


# %%
def insertion_sites_bed(frag_file, tmp_path):
    """Two 1bp insertion points (fragment start, fragment end) per line."""
    subprocess.run(
        f"zcat {frag_file} | "
        "awk 'BEGIN{OFS=\"\\t\"} {print $1,$2,$2+1,$4; print $1,$3-1,$3,$4}'"
        f" > {tmp_path}",
        shell=True,
        check=True,
    )
    return pybedtools.BedTool(tmp_path)


def tss_enrichment_for_day(day, frag_file, center_bed, flank_bed, tmp_dir):
    insertions_path = os.path.join(tmp_dir, f"{day}_insertions.bed")
    insertions = insertion_sites_bed(frag_file, insertions_path)

    center_hits = insertions.intersect(center_bed, wa=True)
    flank_hits = insertions.intersect(flank_bed, wa=True)

    center_counts = pd.Series([f.fields[3] for f in center_hits]).value_counts()
    flank_counts = pd.Series([f.fields[3] for f in flank_hits]).value_counts()

    center_bp = len(center_bed) * 2 * tss_center_bp
    flank_bp = len(center_bed) * 2 * (tss_flank_hi - tss_flank_lo)

    barcodes = sorted(set(center_counts.index) | set(flank_counts.index))
    center_density = center_counts.reindex(barcodes, fill_value=0) / center_bp
    flank_density = flank_counts.reindex(barcodes, fill_value=0) / flank_bp

    tsse = (center_density / flank_density.replace(0, np.nan)).rename("tsse")
    return pd.DataFrame({"barcode": barcodes, "day": day, "tsse": tsse.values})


# %%
frag_rows, umi_rows, tsse_rows = [], [], []
tmp_dir = tempfile.mkdtemp(prefix="per_cell_qc_")
try:
    var_df = None
    for day in days:
        print(f"[{day}] loading RNA h5ad ...")
        adata = ad.read_h5ad(os.path.join(gex_dir, f"rna_{day}.h5ad"))
        if var_df is None:
            var_df = adata.var[["chr", "start", "end", "strand"]].copy()
            center_bed_df, flank_bed_df = build_tss_windows(var_df)
            center_bed = pybedtools.BedTool.from_dataframe(center_bed_df).sort()
            flank_bed = pybedtools.BedTool.from_dataframe(flank_bed_df).sort()

        umi_rows.append(
            pd.DataFrame(
                {
                    "barcode": adata.obs_names,
                    "day": day,
                    "n_umis": adata.obs["total_counts"].values,
                }
            )
        )

        frag_file = os.path.join(frag_dir, f"{day}.fragments.tsv.gz")
        print(f"[{day}] counting ATAC fragments per barcode ...")
        frag_counts = count_fragments_per_barcode(frag_file)
        frag_counts["day"] = day
        frag_counts = frag_counts[frag_counts["barcode"].isin(set(adata.obs_names))]
        frag_rows.append(frag_counts)

        print(f"[{day}] computing TSS enrichment ...")
        tsse_df = tss_enrichment_for_day(day, frag_file, center_bed, flank_bed, tmp_dir)
        tsse_df = tsse_df[tsse_df["barcode"].isin(set(adata.obs_names))]
        tsse_rows.append(tsse_df)
finally:
    pybedtools.helpers.cleanup(remove_all=True)

frag_df = pd.concat(frag_rows, ignore_index=True)
umi_df = pd.concat(umi_rows, ignore_index=True)
tsse_df = pd.concat(tsse_rows, ignore_index=True)

frag_df.to_csv(
    os.path.join(tables_dir, "atac_fragments_per_cell.tsv"), sep="\t", index=False
)
umi_df.to_csv(os.path.join(tables_dir, "rna_umis_per_cell.tsv"), sep="\t", index=False)
tsse_df.to_csv(
    os.path.join(tables_dir, "tss_enrichment_per_cell.tsv"), sep="\t", index=False
)

print(frag_df.groupby("day")["n_fragments"].median())
print(umi_df.groupby("day")["n_umis"].median())
print(tsse_df.groupby("day")["tsse"].median())

# %%
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, df, col, title in [
    (axes[0], frag_df, "n_fragments", "ATAC fragments/cell"),
    (axes[1], umi_df, "n_umis", "RNA UMIs/cell"),
    (axes[2], tsse_df, "tsse", "TSS enrichment"),
]:
    order = days
    colors = [day_colors[d] for d in order]
    sns.violinplot(data=df, x="day", y=col, order=order, palette=colors, ax=ax, cut=0)
    sns.stripplot(
        data=df, x="day", y=col, order=order, color="black", alpha=0.15, size=1.5, ax=ax
    )
    for i, day in enumerate(order):
        vals = df.loc[df["day"] == day, col].dropna()
        ax.text(
            i,
            ax.get_ylim()[1],
            f"{vals.median():.3g}\nn={len(vals)}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    ax.set_title(title)
    ax.set_xlabel("")

fig.savefig(os.path.join(plots_dir, "frag_umi_tsse_per_day.pdf"))
fig.savefig(os.path.join(plots_dir, "frag_umi_tsse_per_day.png"))
print(f"saved -> {os.path.join(plots_dir, 'frag_umi_tsse_per_day.pdf')}")

# %%
lsi_parts = []
pca_parts = []
tmp_dir = tempfile.mkdtemp(prefix="per_cell_qc_lsi_")
try:
    for day in days:
        print(f"[{day}] building peak x cell matrix ...")
        peaks_path = os.path.join(
            preprocessing_dir, day, f"{day}_{peak_type}_peaks_no_blacklist.narrowPeak"
        )
        peaks_bed_df = pd.read_csv(peaks_path, sep="\t", header=None, usecols=[0, 1, 2])
        peaks_bed_df.columns = ["chr", "start", "end"]
        peaks_bed_df["peak_row"] = range(len(peaks_bed_df))
        peaks_bed = pybedtools.BedTool.from_dataframe(peaks_bed_df)

        frag_bed_path = os.path.join(tmp_dir, f"{day}_frag.bed")
        subprocess.run(
            f"zcat {os.path.join(frag_dir, f'{day}.fragments.tsv.gz')} > {frag_bed_path}",
            shell=True,
            check=True,
        )
        frags = pybedtools.BedTool(frag_bed_path)
        overlap = frags.intersect(peaks_bed, wa=True, wb=True)

        barcodes, peak_rows = [], []
        for f in overlap:
            barcodes.append(f.fields[3])
            peak_rows.append(int(f.fields[-1]))

        adata = ad.read_h5ad(os.path.join(gex_dir, f"rna_{day}.h5ad"))
        cell_barcodes = list(adata.obs_names)
        bc_to_idx = {b: i for i, b in enumerate(cell_barcodes)}

        rows, cols = [], []
        for b, p in zip(barcodes, peak_rows):
            idx = bc_to_idx.get(b)
            if idx is not None:
                rows.append(idx)
                cols.append(p)

        n_peaks = len(peaks_bed_df)
        mat = sparse.coo_matrix(
            (np.ones(len(rows)), (rows, cols)), shape=(len(cell_barcodes), n_peaks)
        ).tocsr()

        tfidf = TfidfTransformer(norm="l2", sublinear_tf=True)
        tfidf_mat = tfidf.fit_transform(mat)
        svd = TruncatedSVD(n_components=n_lsi_components, random_state=0)
        lsi = svd.fit_transform(tfidf_mat)[:, 1:]  # drop component 0 (depth-correlated)
        lsi_df = pd.DataFrame(lsi, index=cell_barcodes)
        lsi_df["day"] = day
        lsi_parts.append(lsi_df)

        print(f"[{day}] RNA PCA ...")
        sc.pp.highly_variable_genes(adata, layer="pflog", n_top_genes=2000)
        adata_hvg = adata[:, adata.var["highly_variable"]].copy()
        adata_hvg.X = adata_hvg.layers["pflog"]
        sc.pp.scale(adata_hvg, max_value=10)
        sc.tl.pca(adata_hvg, n_comps=n_pca_components, random_state=0)
        pca_df = pd.DataFrame(adata_hvg.obsm["X_pca"], index=cell_barcodes)
        pca_df["day"] = day
        pca_parts.append(pca_df)
finally:
    pybedtools.helpers.cleanup(remove_all=True)

lsi_all = pd.concat(lsi_parts)
pca_all = pd.concat(pca_parts)

common_idx = lsi_all.index.intersection(pca_all.index)
lsi_vals = lsi_all.loc[common_idx].drop(columns="day").values
pca_vals = pca_all.loc[common_idx].drop(columns="day").values

lsi_z = (lsi_vals - lsi_vals.mean(axis=0)) / lsi_vals.std(axis=0)
pca_z = (pca_vals - pca_vals.mean(axis=0)) / pca_vals.std(axis=0)
joint = np.concatenate([pca_z, lsi_z], axis=1)

joint_adata = ad.AnnData(joint)
joint_adata.obs["day"] = pca_all.loc[common_idx, "day"].values
sc.pp.neighbors(joint_adata, use_rep="X", random_state=0)
sc.tl.umap(joint_adata, random_state=0)

# %%
fig, ax = plt.subplots(figsize=(5, 5))
for day in days:
    mask = joint_adata.obs["day"] == day
    ax.scatter(
        joint_adata.obsm["X_umap"][mask, 0],
        joint_adata.obsm["X_umap"][mask, 1],
        s=3,
        alpha=0.6,
        color=day_colors[day],
        label=day,
    )
ax.set_xlabel("UMAP1")
ax.set_ylabel("UMAP2")
ax.set_title("Multiome UMAP (RNA + ATAC)")
ax.legend(markerscale=3, fontsize=8, frameon=False)

fig.savefig(os.path.join(plots_dir, "joint_umap.pdf"))
fig.savefig(os.path.join(plots_dir, "joint_umap.png"))
pd.DataFrame(
    joint_adata.obsm["X_umap"], columns=["UMAP1", "UMAP2"], index=common_idx
).assign(day=joint_adata.obs["day"].values).to_csv(
    os.path.join(plots_dir, "joint_umap_data.tsv"), sep="\t"
)
print(f"saved -> {os.path.join(plots_dir, 'joint_umap.pdf')}")
# %%
