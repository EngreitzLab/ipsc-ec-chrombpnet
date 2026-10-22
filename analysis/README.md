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
| `1.0.gene_motif_linking.py` | Computes scE2G-weighted TF contribution scores per gene per day (each day's FiNeMo hits in that day's scE2G-linked elements). See [gene-motif linking](#1-gene-motif-linking) below. |
| `2.tf_enrichment_in_programs.py` | Among genes carrying a TF motif in their linked enhancers, tests whether each cNMF program's top 300 genes have higher gene-motif scores (50 programs × d0–d3 × TFs); per-day clustermaps of the programs of interest. See [detailed methods](#2-tf-motif-enrichment-in-cnmf-programs) below. |
| `3.chrombpnet_figures.py` | TF rank scatter per day for the programs of interest (`plots/tf_rank_scatter_{day}`), from the `2.` results. |
| `src/genomics.py` | Shared genomic interval helpers (pybedtools hit/peak intersection). |
| `run_analysis_script.sh` | Slurm wrapper that runs one analysis script in the analysis conda env, e.g. `sbatch --time=1:00:00 --mem=64G run_analysis_script.sh 2.tf_enrichment_in_programs.py`; set `analysis_env` and the partition for your cluster. |

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
TF–gene score  =  Σ_peaks  E2G.Score.qnorm  ×  hit_score   (pos_patterns hits only)
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

## 1 Gene-motif linking

**Script:** `1.0.gene_motif_linking.py`

For each element type (`promoter` = scE2G `isSelfPromoter` TRUE, `enhancer` =
FALSE, i.e. genic + intergenic, `all_cres` = both) and each day (d0–d4):

1. **Links of that day only.** Day X FiNeMo hits (`0.2.finemo_hit_qc/hits_filtered_{day}.tsv`)
   are matched to day X scE2G links (`{day}/multiome_powerlaw_v3/scE2G_predictions_threshold0.177.tsv.gz`).
   scE2G calls peaks per day, so the same enhancer appears in each day's file with
   slightly shifted coordinates; matching hits to the union of all days would count a
   hit once per overlapping copy (for d3 enhancers: 2.34 copies per hit-gene pair on
   average, 61.5% of pairs counted more than once) and would add links that scE2G did
   not predict on that day.
2. **Hits fully inside an element** (`hit_fully_inside = True`, bedtools `-f 1.0`),
   as in PerturbNMF Stage 2. Hits overhanging an element edge are not counted
   (d3 enhancers: 94,109 → 93,815 gene-TF pairs).
3. **Positive patterns only** (`pattern_prefixes = ("pos_patterns",)`), as in PerturbNMF
   Stage 2. Until 2026-10-20 `neg_patterns` hits were kept and scored −1: they cancelled
   `pos_patterns` of the same `tf_motif_id` (e.g. NFY-PBX_0) and inverted the meaning of
   the one-sided test in script 2. One of them, `neg_patterns.21` (PRDM13 label), hits
   most d3 elements.
4. **Score** per hit and linked gene: `hit_coefficient_global × E2G.Score.qnorm`
   (`weight_strategy = "e2g"`); all scores are >= 0.
5. **Aggregate** by gene × TF (`tf_motif_id`) × day: sum over all hits in all linked
   elements of that day (`weighted_score`), and the number of those hits (`n_hits`).
   Gene-TF pairs without a hit have no row; in script 2 these genes are not carriers of
   the motif and are left out of its test.

Overlapping hits are deduplicated only within one motif cluster
(`0.2.finemo_hit_qc.py`, > 3 bp, best `hit_similarity`), not across clusters sharing a
`tf_motif_id`, in the current results: two clusters with the same label calling the same
site both count (d3: 2.6% of positive-pattern hits, mostly ETV-ELF-NFAT-ELK_0, SOX_0 and
CTCF_0). Cross-cluster dedup is implemented in `1.0` (`dedup_same_label`, off).

Outputs:
- `1.0.gene_motif_linking/data/{strategy}_{hit_score_col}_{et}_gene_motif_scores.tsv.gz`
  (`TargetGene`, `tf_motif_id`, `weighted_score`, `n_hits`, `day`, `strategy`, `et`)
- `1.0.gene_motif_linking/data/{et}_gene_covariates.tsv.gz`: motif-free per-gene
  covariates per day (`n_elements`, `element_bp`, `sum_e2g_qnorm`, `chr`, `tss`)

---

## 2 TF motif enrichment in cNMF programs

**Script:** `2.tf_enrichment_in_programs.py` (figures: also `3.chrombpnet_figures.py`)

### Question

For each of the 50 cNMF programs and each day d0–d3: **among genes that carry a TF's
motif in their linked enhancers, do the program's top genes have higher gene-motif
scores than the other carriers?** This asks whether program genes have stronger or more
highly weighted motif instances. It does not ask whether program genes carry the motif
more often.

cNMF was fitted on cells from d0 to d3 (the cNMF object has no d4 cells), so d4 is not
tested.

### Inputs

| Input | Description |
|-------|-------------|
| `cNMF_50_0_2.h5mu` | cNMF results: 50-program gene loadings (`varm["loadings"]`; behaves like cNMF `gene_spectra_score`) |
| `1.0.gene_motif_linking/data/e2g_hit_coefficient_global_enhancer_gene_motif_scores.tsv.gz` | Gene × TF × day scE2G-weighted motif scores (`TargetGene`, `tf_motif_id`, `day`, `weighted_score`) |

### Parameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `element_types` | `["enhancer"]` | Distal enhancers only (`isSelfPromoter` False) |
| `n_top_genes` | 300 | Top genes per program by cNMF loading |
| `min_carriers_per_group` | 10 | Minimum carriers in both the top set and the background for a test to run (override with the environment variable `MIN_CARRIERS_PER_GROUP`; outputs are tagged `min{N}`) |
| `q_sig_threshold` | 0.05 | BH-adjusted P cutoff |
| `poi_prog_day` | 2 (d3), 15 (d2), 17 (d2), 24 (d3), 27 (d3), 34 (d3) | Programs of interest and their day (chosen independently of the motif results) |

### Steps

1. **Carriers.** For each TF motif and day, the carriers are the cNMF genes with
   `weighted_score > 0`, i.e. at least one positive-pattern hit in that day's linked
   enhancers. Median carriers per TF: 140 (d0), 340 (d1), 257 (d2), 310 (d3).
2. **Groups.** Foreground = carriers among the program's top 300 genes (`n_top`);
   background = all other carriers (`n_background`). Genes without a hit of the motif are
   not part of its test.
3. **Filter.** A test runs only if `n_top >= 10` and `n_background >= 10`: 6,683 of
   16,950 program × day × TF combinations. Ten genes per group keep the U test from
   resting on a handful of genes. The filter uses only which genes carry the motif, not
   their scores, so the test stays valid. Every test with >= 10 top-gene carriers also
   has >= 20 background carriers (median ~555), so a background cutoff of 10 or 20 gives
   the same tests.
4. **Test.** One-sided Mann-Whitney U (`scipy.stats.mannwhitneyu`,
   `alternative="greater"`, `method="auto"`). H0: among carriers, scores of top genes
   and of other genes come from the same distribution; Ha: top genes score higher.
   Effect size `AUC = U / (n_top * n_background)` (0.5 = no difference), plus the median
   score in each group. The direction is fixed in advance, so q < 0.05 controls the FDR
   of enrichment calls at 5%.
5. **Correction.** BH across all tests run (one family: programs × days × TFs).

**Calibration.** On 200 random 300-gene sets (d3), the test gave p < 0.05 in 4.8%,
p < 0.01 in 0.98% and p < 0.001 in 0.13% of tests, as expected under the null. Scores
are continuous and > 0, so there are no tied zeros. Among carriers, scores barely track
the number of linked enhancers (median Spearman 0.01 across TFs).

**History.** 2026-10-20 to 2026-10-21 the test was zero-filled: all genes with >= 1 hit
of any TF, missing = 0. That mixed motif presence with score; with ~87% tied zeros its
asymptotic p-values were badly anti-conservative for rare motifs (random gene sets gave
p < 0.001 in 2.5% of tests for motifs with < 10 carriers), and presence tracks the
number of linked enhancers.

### Results (run 2026-10-21, >= 10 carriers per group)

523 of 6,683 tests significant (q < 0.05); 40 of 70 tested TFs significant in at least
one program; 33 of 50 programs with at least one significant TF.

**Sensitivity to the carrier cutoff** (same code, `MIN_CARRIERS_PER_GROUP`):

| Cutoff per group | Tests | Significant |
|------------------|-------|-------------|
| >= 1 | 12,817 | 465 |
| >= 5 | 8,588 | 493 |
| **>= 10 (main)** | **6,683** | **523** |
| >= 20 | 5,027 | 553 |

The top motifs of the programs of interest are the same at every cutoff. More tests
pass at higher cutoffs because the removed tests have little power and only add to the
BH correction; 460 of the 465 hits at >= 1 are also significant at >= 10.

| Program, day | Significant / tested TFs | Strongest TFs |
|--------------|--------------------------|---------------|
| P2, d3 | 10 / 49 | ETV-ELF-NFAT-ELK, SOX, GATA, TCF-BHLH-SNAI |
| P15, d2 | 9 / 45 | ETV-ELF-NFAT-ELK, GATA, ZNF_9, ZNF813 |
| P17, d2 | 8 / 42 | GATA, ETV-ELF-NFAT-ELK, ZNF_9, TEAD |
| P24, d3 | 6 / 46 | ETV-ELF-NFAT-ELK, SOX, MYCN, FOS-JUN |
| P27, d3 | 4 / 45 | ETV-ELF-NFAT-ELK, SOX, MYCN, GATA |
| P34, d3 | 8 / 42 | ETV-ELF-NFAT-ELK, TEAD, FOS-JUN, SOX |

### Outputs

| File | Description |
|------|-------------|
| `2.tf_enrichment_in_programs/enhancer_enrichment_results_min10.tsv` | One row per test run: `program`, `day`, `tf`, `n_top`, `n_background` (carriers), `n_top_in_day_genes`, `median_score_top`, `median_score_background`, `U`, `AUC`, `pval`, `qval` (BH), `neg_log10_qval`, `significant` |
| `2.tf_enrichment_in_programs/plots/poi_by_day_clustermap_{day}_enhancer_min10` | Per day: TFs significant in at least one program of interest × those programs, −log10(q), ward-clustered, stars = q < 0.001 / 0.01 / 0.05 |
| `3.chrombpnet_figures/plots/tf_rank_scatter_{day}_min10` | Per day: one panel per program of interest, TFs ranked by −log10(q), top TFs labeled |

Every plot has `.pdf`, `.png` and a `_data.tsv` with the plotted values. The
`enhancer_enrichment_results_min1.tsv` and `_min20.tsv` tables are the sensitivity runs.

### Interpretation

- A significant TF means: among genes with that motif in linked enhancers, program genes
  have stronger or more highly weighted instances (higher `hit_coefficient_global` ×
  `E2G.Score.qnorm`, summed over their hits). It is not evidence that program genes carry
  the motif more often.
- Tests need at least 10 carriers among the top genes and in the background, so sparse
  motifs are often not tested (70 of 96 motifs are tested at least once); "not tested"
  (no row) is not "not enriched".
- Near-duplicate labels (e.g. TEAD_0 and TEAD1, ETS-like labels) are tested separately
  and can reflect one signal.
