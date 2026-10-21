"""Genomic interval utilities shared across analyses."""

import pandas as pd
import pybedtools


def intersect_hits_peaks(
    hits_df: pd.DataFrame, peaks_df: pd.DataFrame, hit_fully_inside: bool = False
) -> pd.DataFrame:
    """
    Intersect hits with peaks using pybedtools.

    Parameters
    ----------
    hits_df  : must have columns chr, start, end (plus any extra hit columns)
    peaks_df : must have columns chr, start, end (plus any peak metadata columns
               e.g. TargetGene, TargetGeneTSS, ABC.Score, E2G.Score.qnorm)
    hit_fully_inside : True keeps only hits lying entirely inside a peak
               (bedtools intersect -f 1.0); False (default) keeps any 1 bp overlap

    Returns
    -------
    DataFrame with all hit columns followed by all peak metadata columns,
    one row per (hit, overlapping peak) pair.
    """
    hit_cols = list(hits_df.columns)
    peak_cols = list(peaks_df.columns)

    hits_bed = pybedtools.BedTool.from_dataframe(
        hits_df[
            ["chr", "start", "end"]
            + [c for c in hit_cols if c not in ("chr", "start", "end")]
        ]
    )
    peaks_bed = pybedtools.BedTool.from_dataframe(
        peaks_df[
            ["chr", "start", "end"]
            + [c for c in peak_cols if c not in ("chr", "start", "end")]
        ]
    )

    overlap_kwargs = {"f": 1.0} if hit_fully_inside else {}
    intersected = hits_bed.intersect(peaks_bed, wa=True, wb=True, **overlap_kwargs)

    if len(intersected) == 0:
        return pd.DataFrame()

    # wa=True wb=True yields all hit fields then all peak fields
    peak_extra = [c for c in peak_cols if c not in ("chr", "start", "end")]
    result_cols = hit_cols + ["peak_chr", "peak_start", "peak_end"] + peak_extra

    df = intersected.to_dataframe(
        names=result_cols, dtype={"chr": str, "peak_chr": str}
    )
    df = df.drop(columns=["peak_chr", "peak_start", "peak_end"])

    # Restore integer dtypes for known numeric hit columns
    for col in ("start", "end", "cluster_id", "midpoint"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col]).astype("int64")

    return df.reset_index(drop=True)
