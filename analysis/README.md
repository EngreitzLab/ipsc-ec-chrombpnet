# Analysis

This folder contains scripts for gene-motif linking and downstream analyses of
multiome iPSC-EC differentiation data.

---

## Scripts

Run order: `00.*` (QC), `0.*` (motif annotation and hit QC), `1.0` (gene-motif
linking), then `2` and `3` (TF motif enrichment in cNMF programs). Edit `core_path`
and the external input paths in the parameter block at the top of each script.

| Script | Description |
|--------|-------------|
| `00.chrombpnet_model_qc.py` | ChromBPNet full-model QC: ATAC depth, cell counts, predicted vs observed counts, Pearson r and JSD per fold and day. |
| `00.1.per_cell_qc.py` | Per-cell 10x multiome QC: ATAC fragments and RNA UMIs per cell, TSS enrichment, joint RNA+ATAC UMAP. |
| `00.2.atac_percell_qc.py` | Per-cell scATAC QC from the ArchR metrics used for the IGVF submission: fragments, TSS enrichment, nucleosome signal. |
| `00.3.tss_profiles.sh` + `00.3.tss_profiles.py` | Aggregate +/- 2 kb TSS insertion profiles per day (shell, bedtools) and the overlaid, flank-normalized profile plot (Python). |
| `0.0.motif_id_annotation.py` | Annotates compendium motif clusters against MotifCompendium-Database-Human and records which matched TFs are expressed per day. |
| `0.2.finemo_hit_qc.py` | Chooses a per-motif CWM-similarity threshold from the Fi-NeMo reports and writes filtered hit tables. |
| `0.3.hits_to_bed.sh` + `hits_to_bed.py` | Converts Fi-NeMo hits to colored BED9 tracks for IGV. |
| `1.0.gene_motif_linking.py` | Computes distance-weighted TF contribution scores per gene per day using the scE2G gene-motif linking approach. |
| `2.tf_enrichment_in_programs.py` | Tests enrichment of TF motif instances (separately) in the top genes of each cNMF program across all 50 programs × 4 days. See [detailed methods](#2-tf-motif-enrichment-in-cnmf-programs) below. |
| `3.chrombpnet_figures.py` | Publication figures for programs of interest from the `2.` results (enhancer elements): per-day heatmaps with CWM logos, TF rank scatter. |
| `src/genomics.py` | Shared genomic interval helpers (pybedtools hit/peak intersection). |

---

## FiNeMo output fields: `hit_coefficient`, `hit_coefficient_global`, `hit_importance`

### Background: how FiNeMo works

ChromBPNet assigns every nucleotide in a peak a **contribution score**: how much
does this base contribute to the model's predicted accessibility? FiNeMo then
tries to explain that per-nucleotide contribution signal as a sparse sum of motif
CWMs placed at specific positions:

```
contributions ≈ c₁ × [TF3 CWM at pos 200]
              + c₂ × [AP1   CWM at pos 450]
              + c₃ × [GATA  CWM at pos 710]
              + ...
```

This is solved as a **non-negative LASSO** (proximal gradient descent). Most
coefficients are zero. A non-zero coefficient at a position means the CWM
pattern genuinely fits the contribution signal there, and no other motif
explains it better — motifs compete. That is a **hit**.

Before optimization, each peak's contributions are normalized by `global_scale`:

```
global_scale = sqrt( sum(contribs²) / L )
```

This is the RMS of contribution scores across the entire peak — a measure of
how much total attribution signal that peak carries. Normalization puts all
peaks on the same scale during solving.

---

### `hit_coefficient`

The raw LASSO coefficient `c` from the optimization, computed on
**normalized** (per-peak) contributions. It answers:

> *After all motifs competed to explain this peak's contributions, how much did
> this specific motif's CWM need to contribute at this position?*

- High value → the CWM pattern is a strong, necessary fit to the local
  contribution signal
- Comparable across peaks within the same model (because all peaks are
  normalized)
- Controlled for co-occurring motifs (competitive LASSO)

---

### `hit_coefficient_global`

```
hit_coefficient_global = hit_coefficient × global_scale²
```

The coefficient rescaled back to original (un-normalized) contribution units.
It combines:

1. **Motif fit quality** (from `hit_coefficient` — competitive, CWM-specific)
2. **Overall peak activity** (from `global_scale²` — how active is this peak
   genome-wide)

A hit in a highly accessible, high-signal peak will have a larger
`hit_coefficient_global` than the same motif match in a weaker peak, even if
`hit_coefficient` is identical.

---

### `hit_importance`

```
hit_importance = Σ |contribution_original[i]|   for i in the motif footprint window
```

Internally, FiNeMo computes this on the **normalized** contributions
(`contribs / global_scale`), then multiplies by `global_scale` when writing
the output TSV. The two factors cancel exactly:

```
internal:  Σ |contribs / global_scale|  =  (1 / global_scale) × Σ |contribs|
output:    × global_scale                =  Σ |contribs_original|
```

So `hit_importance` in `hits.tsv` is the plain sum of absolute original
contribution scores under the motif footprint. There is **no global_scale
weighting** in the final value.

It answers:

> *Given that FiNeMo found a hit here, how much total contribution signal lives
> under that motif footprint (in original contribution score units)?*

Key properties:
- **Not involved in calling the hit** — `hit_coefficient` decides that
- **No CWM pattern matching** — just a ruler held over the contribution bar
  chart at the already-identified hit location
- **Non-competitive** — does not account for whether another motif better
  explains the same signal; two overlapping motifs each get the full footprint
  sum, so `hit_importance` can be high simply because the region is generally
  active, not because this motif specifically drives it

---

### Comparison table

| Field | Calls the hit? | Uses CWM pattern? | Competitive? | Reflects peak activity? |
|-------|---------------|-------------------|--------------|-------------------------|
| `hit_coefficient` | yes (is the LASSO coef) | yes | yes | no (normalized out) |
| `hit_coefficient_global` | no (derived) | yes | yes | yes (via global_scale²) |
| `hit_importance` | no (post-hoc) | no | no | yes (raw contribution sum) |

**Concrete example:** two TF3 hits with the same `hit_coefficient` (equally
good CWM match):
- Hit A is in a highly active enhancer → large `global_scale` → large
  `hit_importance` and large `hit_coefficient_global`
- Hit B is in a weakly accessible peak → small `hit_importance` and small
  `hit_coefficient_global`

**Another example:** two hits with the same `hit_importance` (same total
contribution under the window):
- Hit A has a clean, specific TF3 CWM pattern → high `hit_coefficient`
- Hit B has generally high contributions that don't look like TF3 → low
  `hit_coefficient` (another motif won the competition there)

---

### Which score to use for scE2G gene-motif linking

**Use `hit_coefficient_global`.** (`1.0.gene_motif_linking.py`, parameter `hit_score_col`)

#### What scE2G captures

`E2G.Score.qnorm` is a peak-to-gene regulatory score built from six features:

1. ABC score: pseudobulk ATAC-seq activity × inverse-distance contact estimate
2. Chromatin accessibility at the enhancer and promoter
3. Genomic distance and gene density
4. Promoter class (ubiquitous vs. cell-type-specific)

Critically, features 1 and 2 are **chromatin-accessibility-based** — they already
encode how open and active a peak is.

#### Why `hit_coefficient_global`

The gene-motif linking formula is:

```
TF–gene score  =  Σ_peaks  E2G.Score.qnorm  ×  hit_score  ×  sign
```

The ideal `hit_score` should be **orthogonal** to `E2G.Score.qnorm`: it should
capture something the E2G score does not already encode.

`hit_coefficient_global = hit_coefficient × global_scale²`

- `hit_coefficient` answers: *among all motifs competing to explain this peak's
  contribution profile, how much did this TF's CWM specifically need to
  contribute?* This is shape-specificity — it is high only when the contribution
  profile actually looks like this motif, independent of whether the peak is
  generally open or closed.
- `global_scale²` rescales from the normalized regression space back to original
  contribution energy units, so the score is comparable across peaks.

The result captures: **does this TF's binding specifically explain the regulatory
activity at this peak, in competition with all other motifs, scaled by how
energetically important this peak is?** This is largely orthogonal to
accessibility and distance that scE2G already accounts for.

#### Why not `hit_importance`

`hit_importance = Σ|contributions_original|` within the footprint is a raw
accessibility-adjacent quantity — it is high wherever contribution scores are
high, which correlates directly with ATAC signal (features 1 and 2 in scE2G).
Multiplying `E2G.Score.qnorm × hit_importance` partially double-counts
accessibility: a simply open peak will inflate both the E2G score and
`hit_importance` for every motif that happens to land there, regardless of
whether any TF is specifically driving regulatory activity. `hit_importance` also
does not account for competing motifs — two different TFs at the same active locus
both receive the full footprint sum.

---

## 2 TF motif enrichment in cNMF programs

**Script:** `2.tf_enrichment_in_programs.py`

### Goal

Determine which of the 50 cNMF programs, at each of the 4 differentiation
days (d0–d3), have their top-loaded genes specifically enriched for TF
motif instances.

---

### Inputs

| Input | Description |
|-------|-------------|
| `cNMF_50_0_2.h5mu` | cNMF results: 50-program loadings (program × gene matrix) |
| `hit_importance_gene_motif_linking/data/e2g_gene_motif_scores.tsv.gz` | Long-format table of gene × TF × day scE2G-weighted motif scores (`TargetGene`, `tf_name`, `day`, `weighted_score`) |

---

### Key parameter

`n_top_genes = 300` — number of top-loading genes used to define each program's
gene set (ranked by cNMF loading weight).

Restricted to distal enhancer elements only (`element_types = ["enhancer"]`);
promoter/all_cres are no longer tested here.

---

### Step-by-step logic

#### Step 1 — Define the gene universe

**Universe = genes present in the scE2G score file AND in the cNMF loadings.**
Genes absent from the score file have no scE2G-linked peaks with any motif hit.
Including them as score-0 background would conflate "no regulatory peaks" with
"has peaks but no TF signal" — two biologically distinct situations. Since top
program genes tend to be more expressed and therefore more accessible, including
peak-less genes would inflate apparent enrichment artifactually. Using only
scE2G-scored genes ensures the background is a meaningful biological negative
control: genes with regulatory peak-gene links but low TF signal.

#### Step 2 — Per-TF scored gene set, per day

For each TF, `day_scores = scores_by_tf[et][tf][day]` holds one row per gene
that has **at least one actual FiNeMo motif hit for that specific TF**
somewhere in its scE2G-linked peaks (from `1.0.gene_motif_linking.py`'s
`groupby(["TargetGene", "tf_motif_id"])["weighted_score"].sum()`). A gene with
an scE2G link but *no* hit for that TF is not zero-filled — it is simply
absent from `day_scores`, per the same reasoning as Step 1: a missing TF link
is not the same claim as "TF importance score of exactly 0," and conflating
the two would bias the comparison. This means the per-TF universe
(`n1 + n2` below) varies substantially by TF — from single digits up to
~13,000 genes in practice (25th percentile ~123, median ~608) — since it
depends on how many genes that TF actually hits anywhere in this dataset, not
on a fixed genome-wide gene count.

`top_genes_set` (`n_top_genes = 300` highest-loading genes for the program)
splits `day_scores` into:
- `fg` — top-program genes that have a score for this TF
- `bg` — every other gene that has a score for this TF

H0: TF importance scores for the top 300 genes are drawn from the same
distribution as the TF importance scores of other (TF-linked) genes.
Ha: TF importance scores are higher for the top 300 genes.

If either `fg` or `bg` is empty for a given program × TF × day (e.g. a
low-hit-count TF with no top-program-gene overlap), that combination is
skipped rather than tested.

#### Step 3 — Enrichment test: `scipy.stats.mannwhitneyu`

For each program × day × TF combination, call directly (no precomputed
per-column ranks — each call does its own ranking internally):

```python
stat, pval = mannwhitneyu(
    fg,
    bg,
    alternative="greater",
    method="auto",
)
AUC = stat / (len(fg) * len(bg))  # 0.5 = null; >0.5 = top genes score higher
```

`method='auto'` lets scipy pick `'exact'` when a sample is small and tie-free,
falling back to the asymptotic normal approximation otherwise — real per-TF
universe sizes here range as low as single digits (see Step 2), so a hardcoded
`'asymptotic'` would have been inappropriate for a meaningful fraction of
tests; `'auto'` matches scipy's own guidance on when each method is valid.

**Why one-sided:** The hypothesis is directional by design (H0: top-N genes'
TF scores are drawn from the same distribution as other genes' TF scores;
Ha: higher for top-N genes) — depletion/repression is not the question this
test is asking. Note this means a TF that specifically represses top program
genes (motif contributions are signed: `pos_patterns` = +1 activating,
`neg_patterns` = -1 repressive) would not be flagged as significant here;
that would need a separate one-sided `alternative='less'` test if wanted.

**Why test all TFs:** Running enrichment for all ~104 TFs makes other motifs a
natural matched background. TF's rank among all TFs (1 = most enriched)
directly answers whether TF is specifically recruited to program genes beyond
what any other motif shows.

#### Step 4 — Multiple testing correction

BH FDR applied **globally** across all 50 programs × 4 days × ~104 TFs ≈ 20,800
tests.

#### Step 5 — Rank TF among all TFs

For each program × day, TFs are ranked by AUC (descending). TF2 and TF3
ranks indicate their relative enrichment specificity.

---

### Outputs

| File | Description |
|------|-------------|
| `results_primary.tsv` | All TF × program × day: AUC, p-value, q-value (BH), TF rank |
| `results_tf.tsv` | TF2 and TF3 rows only |
| `plots/t_auc_heatmap` | Two-panel heatmap (TF2, TF3): 50 programs × 4 days, AUC; stars = q<0.05/0.01 |
| `plots/TF_rank_heatmap` | Two-panel heatmap: TF2, TF3 rank among all TFs |
| `plots/TF_clustermap_poi` | Clustermap: rows = union of significant TFs in any POI × day; cols = programs of interest × days; AUC color (bwr, centered at 0.5); TF2/3 highlighted in red |

Programs of interest (13, 15, 17, 30, 34) are bolded on heatmap y-axes.

---

### Interpretation guide

A program × day is considered robustly TF-enriched if:
1. TF2 or TF3 AUC > 0.5, q < 0.05
2. TF rank is low (top 5–10 among ~104 TFs) — enrichment is TF-specific, not
   a general property of highly regulated genes

The clustermap reveals which TFs co-enrich with TF across programs and days,
and whether TFs cluster with other known co-regulators.