# ChromBPNet iPSC-EC Pipeline

ChromBPNet (Chromatin BPNet) is a bias-factorised deep learning model that predicts
per-base ATAC-seq accessibility from DNA sequence. It decomposes the signal into a Tn5
bias model (sequence preferences of the Tn5 transposase) and a ChromBPNet model that
captures true biological accessibility. The bias model is subtracted so that the final
`chrombpnet_nobias` model only learns TF binding motifs and chromatin accessibility
signals, not Tn5 sequence artifacts.

This pipeline trains ChromBPNet across 5 differentiation timepoints (d0-d4) using
5-fold cross-validation, then discovers and annotates the regulatory motifs active at
each stage.

All paths and shared parameters live in `config.sh`. Source it at the top of any new
script.

---

## How motifs are identified

ChromBPNet is trained on ATAC-seq data and learns to predict chromatin accessibility
from DNA sequence. Once trained, DeepLIFT is applied to compute per-nucleotide
contribution scores — these highlight which bases the model relies on to make its
predictions. Contribution scores are averaged across cross-validation folds per
timepoint to cancel fold-specific noise, and TF-MoDISco is run on the averaged scores
to discover recurring high-importance sequence patterns (motifs). MotifCompendium then
merges similar motifs across all timepoints into a unified non-redundant set and
annotates them against the MotifCompendium reference database to assign TF names. Finally, Fi-NeMo searches this
unified motif set against the averaged contribution scores of each day to call the
genomic locations where each motif occurs. The result is a set of TF binding sites
grounded in model-derived sequence importance rather than ChIP-seq.

---

## Stage overview

```
Stage 1  Preprocess data              01, 02
Stage 2  Train models                 03.0 -> 03.1 -> 03.2 -> 04.0
  QC     Inspect model quality        04.1
  extra  Predictions (optional)       04.2
Stage 3  Contribution scores          05
Stage 3  Fold averaging               06, 07 (BigWig)
Stage 3  MoDISco on averaged scores   08
Stage 4  Unified motif compendium     09 -> 10 -> 11
```
Each Stage 4 step has a profile-head variant (`09.1`, `10.1`, `11.1`) that runs the
same logic on the profile contribution scores; the counts-head scripts (`09`, `10`,
`11`) are the main path.

---

## Stage 1 — Data preprocessing

### `01.preprocess_peaks.sh`
Removes ENCODE blacklisted regions from per-day ATAC-seq peak files and reformats them
to narrowPeak with the summit at the midpoint. The blacklist is expanded by 1057 bp on
each side (half of ChromBPNet's 2114 bp input window) to ensure the full training window
around every retained peak is free of low-mappability sequence.

Output: `results/preprocessing/{day}/{day}_{peak_type}_peaks_no_blacklist.narrowPeak`

### `02.preprocess_nonpeaks.sh`
Generates negative training examples (non-peak genomic regions) that are GC-content
matched to the peaks. ChromBPNet needs negative examples to learn that accessibility is
driven by sequence features at peaks specifically, not by generic sequence composition
present genome-wide.

Output: `results/preprocessing/{day}/output_{peak_type}_fold_{fold}_negatives.bed`

---

## Stage 2 — Bias model training and selection

### Why train multiple bias models?
The key free parameter is the bias threshold factor (`-b`), which controls how
aggressively the bias model captures Tn5 sequence preferences. Too low: the bias model
under-captures Tn5 artifacts, which then leak into the ChromBPNet model as spurious
motifs. Too high: the bias model over-captures and starts absorbing genuine TF signals.
The correct value varies per fold (because each fold uses a different subset of training
data) and must be validated empirically.

### `03.0.train_bias_model.sh`
Trains Tn5 bias models for each combination of fold and bias threshold factor (0.5, 0.6,
0.7, 0.8, 0.9), then computes fast QC metrics (counts/profile Pearson r, JSD) for each via
`predict_bias_metrics.py`. Each bias model learns only from non-peak regions, so it
captures Tn5 sequence preferences without contamination from real TF binding signal.
Deliberately skips the expensive DeepLIFT interpretation + TF-MoDISco step (that only
runs on the model selected per fold, in `03.2.qc_selected_bias.sh`) — running it on all
25 fold x bias-factor combos is what made this step slow.

Output: `results/bias_models/bias_model{suffix}/{bias_day}_{peak_type}_fold_{fold}/`

### `03.1.select_bias.sh` + `select_bias_model.py`
Evaluates each candidate bias model using two key metrics from the fast QC step:
- **Non-peaks Pearson R**: how well the bias model explains Tn5 sequence preferences on
  held-out non-peak regions. Should be positive (> 0).
- **Peaks Pearson R**: correlation between bias model predictions and observed counts
  at peak regions. Should not be strongly negative (< -0.3 is a warning; < -0.5 fails)
  because a good bias model should not anti-correlate with real accessibility signal.

Reads `*_bias_metrics.json` files, classifies each (fold, bias) pair as pass/warn/fail,
and selects the best bias threshold per fold. Also writes cross-fold comparison plots
(`bias_model_pearsonr_boxplot`, `bias_model_metrics`, `bias_model_selection_heatmap`,
`bias_selection_overview`) to support the selection. Update `fold_bias_suffix` in
`dataset_config.sh` with the output.

### `03.2.qc_selected_bias.sh` + `run_bias_qc.py`
Runs the expensive QC (marginal footprinting, DeepLIFT interpretation, TF-MoDISco) on
only the bias model selected per fold — the step that verifies the Tn5 signal has
actually been learned. Inspect `evaluation/*_bias_profile.pdf`: if the discovered motifs
are Tn5-like rather than GC-rich/TF-like, the bias model is behaving as intended.

### `04.0.train_full_model.sh`
Trains the final ChromBPNet models using the per-fold optimal bias model selected in
step 03. Each fold uses its own best bias threshold rather than a single shared value.

Output: `results/full_models/{day}_{peak_type}_fold_{fold}/`
- `models/chrombpnet_nobias.h5` — bias-free model used for contribution scores
- `models/chrombpnet.h5` — full model (bias included)
- `evaluation/` — per-fold QC metrics

---

## QC scripts (run after Stage 2)

### `04.1.qc_run_full_model.sh` + `qc_full_model.py`
Aggregates ChromBPNet evaluation metrics across all days and folds from the
`evaluation/` subdirectories: counts Pearson/Spearman R, median JSD, and Tn5 motif
response in the final bias-free model.

Output: `results/plots/full_model_qc/`

**What to look for**: Pearson R > 0.6 across most days/folds. Tn5 motif response < 0.003
in the nobias model (if higher, the bias subtraction was insufficient).

---

## Predictions (optional)

### `04.2.generate_predictions.sh` + `predict_and_avg.py`
Generates genome-wide predicted accessibility BigWig tracks from the trained models,
averaged across all available folds. Two tracks per day: bias-corrected
(`chrombpnet_nobias`) and uncorrected (`chrombpnet`). These BigWigs can be loaded in
IGV to visualise model predictions alongside observed ATAC-seq signal. Independent of
Stage 3 — can run any time after Stage 2 completes.

Output: `results/predictions/{day}_{peak_type}/`
- `{day}_avg_chrombpnet_nobias.bw`
- `{day}_avg_chrombpnet_uncorrected.bw`

---

## Stage 3 — Contribution scores

### What are contribution scores?
ChromBPNet uses DeepLIFT to assign an importance score to every nucleotide at every
peak, reflecting how much that base contributes to the model's accessibility prediction.
Summing these per-base scores across a short stretch of DNA reveals a TF binding motif
footprint: the positions where changing the sequence most changes the predicted
accessibility.

### `05.get_contrib_scores.sh`
Computes DeepLIFT contribution scores for each day x fold using the bias-free
`chrombpnet_nobias` model. Outputs an H5 file per day/fold with three arrays
(shape: n_peaks x 4 bases x 2114 bp):
- `raw/seq` — one-hot encoded DNA sequences
- `shap/seq` — hypothetical contribution scores (what each base would contribute)
- `projected_shap/seq` — projected scores (multiplied by the actual one-hot, used for
  motif discovery)

Output: `results/full_models/{day}_{peak_type}_fold_{fold}/interpretation/interpretation.counts_scores.h5`

### `06.average_contrib_scores.sh` + `average_contrib_scores.py`
Loads `interpretation.counts_scores.h5` from all 5 folds for a given day and computes
the element-wise mean of `projected_shap/seq` and `shap/seq`. The one-hot sequences
(`raw/seq`) are identical across folds and are taken from fold 0.

**Why average?** Each model was trained on a different 80/20 data split, so its
contribution scores carry fold-specific noise. Averaging cancels this noise while
preserving robustly important signal — following the Greenleaf lab ChromBPNet approach.

Output: `results/contrib_scores/{day}/{day}_average_shaps.counts.h5`

### `07.contribs_to_bigwig.sh` + `contribs_to_bigwig.py`
Converts the fold-averaged contribution H5 to a BigWig file for genome browser
inspection.

Output: `results/contrib_scores/{day}/{day}_average_shaps.counts.bw`

### `08.run_modisco.sh`
Runs TF-MoDISco on the fold-averaged contribution scores to discover the motifs active
at each differentiation stage. MoDISco operates on 1000 bp windows centred on each peak
summit, extracting ~50 bp seqlets around high-scoring positions and clustering them into
motifs (up to 500,000 seqlets per run, `-w 500`). Also generates an HTML report
annotated against JASPAR.

Note: the `modisco motifs` command in this script is commented out by default. Run it
once to generate `modisco_counts_results.h5`, then comment it out to rerun only the
report without repeating the expensive motif discovery step.

Output: `results/contrib_scores/{day}/modisco/modisco_counts_results.h5`

---

## Stage 4 — Unified motif compendium and hit calling

### Why build a unified motif set?
If each day discovers its own motif set independently, pattern IDs are not comparable:
"pattern_0" in d0 and "pattern_0" in d4 may refer to completely different TFs. To
compare motif usage across differentiation timepoints — the main scientific question —
all days need to share the same motif vocabulary. The compendium pipeline clusters
similar patterns from all days, merges them into consensus motifs, annotates them
against the MotifCompendium-Database-Human (JASPAR 2024 + HOCOMOCO v13 + CIS-BP +
Codebook + CAP-SELEX), and then calls hits in all days using this unified set. Because
all days use identical PWMs, hit counts, occupancy fractions, and co-occurrence
statistics are directly comparable across timepoints.

### `09.motif_compendium.sh` + `motif_compendium.py` (+ `09.1.motif_compendium_profile.sh`)
Builds the non-redundant motif compendium using
[MotifCompendium](https://github.com/kundajelab/MotifCompendium) (Kundaje lab).

Steps:
1. Reads `modisco_counts_results.h5` for each day from `results/contrib_scores/`.
2. Builds a pairwise similarity matrix across all motifs (all days, pos + neg strands).
3. Annotates each motif against MotifCompendium-Database-Human; TF name is the first
   entry in the comma-separated list (e.g. `CTCF,CTCFL` -> `CTCF`).
4. Clusters at similarity threshold 0.95 (Leiden algorithm).
5. Exports one averaged pattern per cluster -> `modisco_compiled.h5` for Fi-NeMo.

This is what gives a single, common TF/motif list shared across all five
timepoints (d0-d4) in this dataset — every day's hit calling in step 10 uses
the same PWMs, so hit counts and occupancy are directly comparable across days.

Output: `results/compendium/modisco_compiled/`
- `modisco_compiled.h5` — clustered motifs (input for step 10)
- `modisco_compendium.meme` — MEME format export
- `modisco_compendium.mc` — pickled MotifCompendium object for interactive inspection
- `modisco_compendium_meta.tsv` — per-motif TF annotations + cluster IDs
- `modisco_config.tsv` — day to MoDISco H5 path mapping

### `10.run_finemo_unified.sh` (+ `10.1.run_finemo_unified_profile.sh`)
Calls motif hits per day using `modisco_compiled.h5` from step 09 and the
fold-averaged contribution scores from step 06. Runs as a SLURM array over days
(one job per day).

**Why averaged scores?** Using the same averaged scores that drove motif discovery
(step 08) ensures consistency: the PWMs were learned from averaged signals, so hit
calling on the same averaged signals is most coherent.

**Note on missing hits**: If a locus shows high contribution signal in earlier days but
no Fi-NeMo hit in a later day, the most common reason is that the locus is not in that
day's called peak set — contribution scores are only computed within peaks, so Fi-NeMo
never sees that position. Check peak overlap before concluding TF binding is lost.

Output: `results/finemo_unified/{day}_{peak_type}/`
- `hits.bed.gz` + `hits.bed.gz.tbi` — tabix-indexed hit calls
- `hits.tsv` — full hit table with scores
- `intermediate_inputs.npz` — extracted regions (input to Fi-NeMo)

### `11.postprocess_finemo.sh` (+ `11.1.postprocess_finemo_profile.sh`)
Post-processes the per-day hit calls from step 10:
1. `finemo report` — computes per-motif instance-CWM vs input-CWM similarity
   (`cwm_similarity` in `motif_report.tsv`), used downstream to filter low-confidence
   hits.
2. `finemo collapse-hits` is intentionally **not** run: its selection logic demotes
   higher-similarity hits, so the `is_primary` column it would add is not a useful
   filter. Downstream filtering (`analysis/0.2.finemo_hit_qc.py`) uses `hits_unique.tsv`
   directly and applies a `cwm_similarity` threshold from `motif_report.tsv` instead.

Output: `results/finemo_unified/{day}_{peak_type}/finemo_report/motif_report.tsv`

### `analysis/0.3.hits_to_bed.sh` + `analysis/hits_to_bed.py`
Converts per-day `hits.bed.gz` to BED9 format for visualisation in the IGV web app.

**Output format**: BED9 with `itemRgb=On` track header. Each hit is rendered in IGV as
a labelled rectangle with strand arrows. Each unique TF name receives a distinct color
(evenly spaced HSV hues); unannotated patterns are gray.

**Name format**: `{TF}_{pattern_id}` (e.g. `CTCF_pos_patterns.5`). All patterns
sharing the same TF name share the same color regardless of cluster.

Output: `results/compendium/bed/{day}_{peak_type}_hits.bed`

---

## IGVF submission notes (added 2026-10-09)

What was submitted to the IGVF portal from this pipeline, and how it maps to the steps
here.

- **Submitted motif hits are the counts head:** `results/finemo_unified/{day}_all/hits.bed.gz`
  from `10.run_finemo_unified.sh` (Fi-NeMo on `{day}_average_shaps.counts.h5` against the
  unified `modisco_compiled.h5`). `10.1.run_finemo_unified_profile.sh` /
  `11.1.postprocess_finemo_profile.sh` are a profile-head variant (`{day}_all_profile/`), which
  is not submitted. The profile-head hits were the ones submitted on 2026-09-02 (the packaging
  step that chose them is not recorded); they were replaced with the counts-head hits on
  2026-10-09 (user decision).
- **Motif names in the hits** (`pos_patterns.N`) refer to the cross-day compendium
  `results/compendium/modisco_compiled/modisco_compiled.h5` (MotifCompendium 1.0.16), not to a
  single day's MoDISco output.
- **TF-MoDISco version:** `modisco motifs` in the `chrombpnet` conda env is tfmodisco-lite
  (PyPI `modisco-lite` **2.0.7**, installed 2025-11-07). TF-MoDISco v2.x (`kundajelab/tfmodisco`,
  PyPI `modisco`) is the same implementation at later versions; the old `modisco` 0.5.16 (v0)
  is also installed but unused.
- **Counts-head MoDISco provenance:** `08.run_modisco.sh` now runs only `profile`. The counts
  results (`modisco_counts_results.h5`, written 2026-08-22/23) came from an earlier version of
  the script; no logs survive, so their exact options are not recorded (the profile runs used
  `-n 500000 -w 500`).
- **bigWig headers:** output bigWigs for IGVF must list only the portal's GRCh38 contigs.
  `dataset_config.sh` now defines `chrom_sizes_bigwig` (`genome/hg38.portal.chrom.sizes`, 195
  contigs), used by `04.2` and `07`; training and preprocessing still use `chrom_sizes`.

## Key thresholds

| Threshold | Value | What it controls | Notes |
|---|---|---|---|
| `motif_compendium_threshold` | 0.95 | Similarity cutoff for Leiden clustering in step 09 | Higher -> more clusters (finer splitting); lower -> more merging. TF family members may appear as separate clusters at 0.95. |
| `finemo_alpha` | 0.8 | Fi-NeMo hit-calling significance level (step 10) | Lower -> more hits (higher FDR); higher -> fewer, more confident hits. ~1M hits/day at 0.8 is permissive — consider 0.7 or 0.6 to reduce noise. |
| `min_annotation_score` | 0.7 | Minimum similarity to reference database for TF name assignment (step 09) | Clusters below this appear unannotated (gray in IGV). May be real motifs absent from the reference database, or degenerate composite patterns. |
| MoDISco seqlet p-value | default | Which positions within peaks seed seqlets (step 08) | Not explicitly set — MoDISco defaults apply. Permissive settings allow weak/noisy patterns into the compendium. |
| MoDISco `-n` | 500,000 | Max seqlets used for motif discovery (step 08) | Limits runtime; if peaks x high-scoring positions exceed this, a random subset is used. |

---

## Directory structure (key outputs)

```
results/
  preprocessing/
    hg38.fa, hg38.chrom.sizes, blacklist.bed.gz   reference files
    {day}/
      {day}_{peak_type}_peaks_no_blacklist.narrowPeak   filtered peaks (step 01)
      output_{peak_type}_fold_{fold}_negatives.bed      GC-matched negatives (step 02)

  bias_models/
    bias_model{suffix}/
      {bias_day}_{peak_type}_fold_{fold}/
        models/        {bias_day}_{peak_type}_fold_{fold}_bias.h5
        evaluation/    bias_metrics.json

  full_models/
    {day}_{peak_type}_fold_{fold}/
      models/          chrombpnet_nobias.h5, chrombpnet.h5
      interpretation/  interpretation.counts_scores.h5
                       interpretation.interpreted_regions.bed
      evaluation/      counts_pearsonr, counts_spearmanr, etc.

  contrib_scores/
    {day}/
      {day}_average_shaps.counts.h5      fold-averaged contribution scores (step 06)
      {day}_average_shaps.counts.bw      BigWig for genome browser (step 07)
      modisco/
        modisco_counts_results.h5        MoDISco on averaged scores (step 08)
        counts_report/                   HTML report annotated against JASPAR

  predictions/
    {day}_{peak_type}/
      {day}_avg_chrombpnet_nobias.bw         bias-corrected accessibility (step 04.2)
      {day}_avg_chrombpnet_uncorrected.bw    uncorrected accessibility (step 04.2)

  compendium/
    modisco_compiled/
      modisco_config.tsv              day -> MoDISco H5 path (reference)
      modisco_compiled.h5             clustered motifs, input for Fi-NeMo (step 09)
      modisco_compendium.meme         MEME format export
      modisco_compendium.mc           pickled MotifCompendium object
      modisco_compendium_meta.tsv     per-motif TF annotations + cluster IDs
    bed/
      {day}_{peak_type}_hits.bed      BED9 with TF names + colors for IGV (analysis/0.3)

  finemo_unified/
    {day}_{peak_type}/
      hits.bed.gz + hits.bed.gz.tbi   tabix-indexed hit calls (step 10)
      hits.tsv                        full hit table
      intermediate_inputs.npz
      finemo_report/
        motif_report.tsv              per-motif cwm_similarity (step 11)

  plots/
    full_model_qc/                    QC plots from 04.1.qc_run_full_model.sh
```

---

## Step-by-step execution

All commands are run from the `pipeline/` directory. Steps within each block can run
in parallel; blocks must complete in order.

```bash
# Block 1: Data preparation (run from pipeline/ directory)
srun --mem 100G --time 6:00:00 --partition <partition> bash 01.preprocess_peaks.sh
srun --mem 100G --time 6:00:00 --partition <partition> bash 02.preprocess_nonpeaks.sh

# Block 2: Bias model training (all fold x bias-factor combos) + fast QC metrics
sbatch 03.0.train_bias_model.sh

# Block 3: Bias selection (after 03.0 completes)
bash 03.1.select_bias.sh           # writes QC plots; update fold_bias_suffix in dataset_config.sh

# Block 4: Full QC (interpretation + TF-MoDISco) on the selected bias model only
sbatch 03.2.qc_selected_bias.sh    # verify Tn5 signal learned; inspect *_bias_profile.pdf

# Block 5: Full model training
sbatch 04.0.train_full_model.sh

# Block 6: Model QC (optional, can run alongside Block 7)
sbatch 04.1.qc_run_full_model.sh   # produces results/plots/full_model_qc/

# Block 7: Predictions (optional, independent of blocks 8+)
sbatch 04.2.generate_predictions.sh

# Block 8: Contribution scores (after 04.0 completes)
sbatch 05.get_contrib_scores.sh

# Block 9: Fold averaging (after 05 completes)
sbatch 06.average_contrib_scores.sh   # one averaged H5 per day
sbatch 07.contribs_to_bigwig.sh       # BigWig tracks (can run alongside 06)

# Block 10: MoDISco (after 06 completes)
sbatch 08.run_modisco.sh              # one MoDISco result per day

# Block 11: Unified motif compendium (after 08 completes)
sbatch 09.motif_compendium.sh         # cluster + annotate -> modisco_compiled.h5

# Block 12: Unified hit calls (after 09 completes)
sbatch 10.run_finemo_unified.sh       # cross-day comparable motif hits

# Block 13: Post-process hit calls (after 10 completes)
sbatch 11.postprocess_finemo.sh       # per-motif cwm_similarity for downstream QC

# Block 14: IGV visualisation (after 11 completes; run from analysis/ directory)
sbatch analysis/0.3.hits_to_bed.sh    # BED9 files with TF colors for IGV
```

### Minimal run (no QC or optional steps)

```
01 -> 02 -> 03.0 -> 03.1 -> 04.0 -> 05 -> 06 -> 08 -> 09 -> 10 -> 11
```
