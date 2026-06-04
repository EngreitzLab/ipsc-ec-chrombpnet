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
Stage 2  Train models                 03 -> 04 -> 05
  QC     Inspect model quality        04.1.qc_bias_selection.py, 05.qc_run_full_model.sh
Stage 3  Contribution scores          06
Stage 3  Fold averaging               07, 08 (BigWig)
Stage 3  MoDISco on averaged scores   09
Stage 4  Predictions (optional)       10
  legacy Per-fold hit calls           _11 (not used for cross-day analysis)
Stage 5  Unified motif compendium     11 -> 12 -> 13
```

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

### `03.train_bias_model.sh`
Trains Tn5 bias models for each combination of fold and bias threshold factor (0.5, 0.6,
0.7, 0.8). Each bias model learns only from non-peak regions, so it captures Tn5
sequence preferences without contamination from real TF binding signal.

Output: `results/bias_models/bias_model{suffix}/{bias_day}_{peak_type}_fold_{fold}/`

### `04.select_bias.sh` + `04.0.select_bias_model.py` + `04.1.qc_bias_selection.py`
Evaluates each candidate bias model using two key metrics:
- **Non-peaks Pearson R**: how well the bias model explains Tn5 sequence preferences on
  held-out non-peak regions. Should be positive (> 0).
- **Peaks Pearson R**: correlation between bias model predictions and observed counts
  at peak regions. Should not be strongly negative (< -0.3 is a warning; < -0.5 fails)
  because a good bias model should not anti-correlate with real accessibility signal.

Reads `*_bias_metrics.json` files, classifies each (fold, bias) pair as pass/warn/fail,
and selects the best bias threshold per fold. Update `fold_bias_suffix` in `config.sh`
with the output.

### `05.train_full_model.sh`
Trains the final ChromBPNet models using the per-fold optimal bias model selected in
step 04. Each fold uses its own best bias threshold rather than a single shared value.

Output: `results/full_models/{day}_{peak_type}_fold_{fold}/`
- `models/chrombpnet_nobias.h5` — bias-free model used for contribution scores
- `models/chrombpnet.h5` — full model (bias included)
- `evaluation/` — per-fold QC metrics

---

## QC scripts (run after Stage 2)

### `04.1.qc_bias_selection.py`
Detailed cross-fold comparison of models trained at different bias threshold values.
Visualises bias metric distributions to support the selection made by `04.select_bias.sh`.

### `05.qc_run_full_model.sh` + `05.qc_full_model.py`
Aggregates ChromBPNet evaluation metrics across all days and folds from the
`evaluation/` subdirectories: counts Pearson/Spearman R, median JSD, and Tn5 motif
response in the final bias-free model.

Output: `results/plots/full_model_qc/`

**What to look for**: Pearson R > 0.6 across most days/folds. Tn5 motif response < 0.003
in the nobias model (if higher, the bias subtraction was insufficient).

---

## Stage 3 — Contribution scores

### What are contribution scores?
ChromBPNet uses DeepLIFT to assign an importance score to every nucleotide at every
peak, reflecting how much that base contributes to the model's accessibility prediction.
Summing these per-base scores across a short stretch of DNA reveals a TF binding motif
footprint: the positions where changing the sequence most changes the predicted
accessibility.

### `06.get_contrib_scores.sh`
Computes DeepLIFT contribution scores for each day x fold using the bias-free
`chrombpnet_nobias` model. Outputs an H5 file per day/fold with three arrays
(shape: n_peaks x 4 bases x 2114 bp):
- `raw/seq` — one-hot encoded DNA sequences
- `shap/seq` — hypothetical contribution scores (what each base would contribute)
- `projected_shap/seq` — projected scores (multiplied by the actual one-hot, used for
  motif discovery)

Output: `results/full_models/{day}_{peak_type}_fold_{fold}/interpretation/interpretation.counts_scores.h5`

### `07.average_contrib_scores.sh` + `07.average_contrib_scores.py`
Loads `interpretation.counts_scores.h5` from all 5 folds for a given day and computes
the element-wise mean of `projected_shap/seq` and `shap/seq`. The one-hot sequences
(`raw/seq`) are identical across folds and are taken from fold 0.

**Why average?** Each model was trained on a different 80/20 data split, so its
contribution scores carry fold-specific noise. Averaging cancels this noise while
preserving robustly important signal — following the Greenleaf lab ChromBPNet approach.

Output: `results/contrib_scores/{day}/{day}_average_shaps.counts.h5`

### `08.contribs_to_bigwig.sh` + `08.contribs_to_bigwig.py`
Converts the fold-averaged contribution H5 to a BigWig file for genome browser
inspection.

Output: `results/contrib_scores/{day}/{day}_average_shaps.counts.bw`

### `09.run_modisco.sh`
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

## Stage 4 — Predictions

### `10.generate_predictions.sh` + `10.predict_and_avg.py`
Generates genome-wide predicted accessibility BigWig tracks from the trained models,
averaged across all available folds. Two tracks per day: bias-corrected
(`chrombpnet_nobias`) and uncorrected (`chrombpnet`). These BigWigs can be loaded in
IGV to visualise model predictions alongside observed ATAC-seq signal.

Output: `results/predictions/{day}_{peak_type}/`
- `{day}_avg_chrombpnet_nobias.bw`
- `{day}_avg_chrombpnet_uncorrected.bw`

### `_11.run_finemo.sh` *(legacy — not part of main pipeline)*
Calls motif hits per day using per-day contribution scores and per-day MoDISco patterns.
Hit IDs are not comparable across days. Kept for reference only; use
`12.run_finemo_unified.sh` instead.

---

## Stage 5 — Unified motif compendium and hit calling

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

### `11.motif_compendium.sh` + `11.motif_compendium.py`
Builds the non-redundant motif compendium using
[MotifCompendium](https://github.com/kundajelab/MotifCompendium) (Kundaje lab).

Steps:
1. Reads `modisco_counts_results.h5` for each day from `results/contrib_scores/`.
2. Builds a pairwise similarity matrix across all motifs (all days, pos + neg strands).
3. Annotates each motif against MotifCompendium-Database-Human; TF name is the first
   entry in the comma-separated list (e.g. `CTCF,CTCFL` -> `CTCF`).
4. Clusters at similarity threshold 0.95 (Leiden algorithm).
5. Exports one averaged pattern per cluster -> `modisco_compiled.h5` for Fi-NeMo.

Output: `results/compendium/modisco_compiled/`
- `modisco_compiled.h5` — clustered motifs (input for step 12)
- `modisco_compendium.meme` — MEME format export
- `modisco_compendium.mc` — pickled MotifCompendium object for interactive inspection
- `modisco_compendium_meta.tsv` — per-motif TF annotations + cluster IDs
- `modisco_config.tsv` — day to MoDISco H5 path mapping

### `12.run_finemo_unified.sh`
Calls motif hits per day using `modisco_compiled.h5` from step 11 and the
fold-averaged contribution scores from step 07. Runs as a SLURM array over days
(one job per day).

**Why averaged scores?** Using the same averaged scores that drove motif discovery
(step 09) ensures consistency: the PWMs were learned from averaged signals, so hit
calling on the same averaged signals is most coherent.

**Note on missing hits**: If a locus shows high contribution signal in earlier days but
no Fi-NeMo hit in a later day, the most common reason is that the locus is not in that
day's called peak set — contribution scores are only computed within peaks, so Fi-NeMo
never sees that position. Check peak overlap before concluding TF binding is lost.

Output: `results/finemo_unified/{day}_{peak_type}/`
- `hits.bed.gz` + `hits.bed.gz.tbi` — tabix-indexed hit calls
- `hits.tsv` — full hit table with scores
- `intermediate_inputs.npz` — extracted regions (input to Fi-NeMo)

### `analysis/13.hits_to_bed.sh` + `analysis/13.hits_to_bed.py`
Converts per-day `hits.bed.gz` to BED9 format for visualisation in the IGV web app.

**Output format**: BED9 with `itemRgb=On` track header. Each hit is rendered in IGV as
a labelled rectangle with strand arrows. Each unique TF name receives a distinct color
(evenly spaced HSV hues); unannotated patterns are gray.

**Name format**: `{TF}_{pattern_id}` (e.g. `CTCF_pos_patterns.5`). All patterns
sharing the same TF name share the same color regardless of cluster.

Output: `results/compendium/bed/{day}_{peak_type}_hits.bed`

---

## Key thresholds

| Threshold | Value | What it controls | Notes |
|---|---|---|---|
| `motif_compendium_threshold` | 0.95 | Similarity cutoff for Leiden clustering in step 11 | Higher -> more clusters (finer splitting); lower -> more merging. TF family members may appear as separate clusters at 0.95. |
| `finemo_alpha` | 0.8 | Fi-NeMo hit-calling significance level (step 12) | Lower -> more hits (higher FDR); higher -> fewer, more confident hits. ~1M hits/day at 0.8 is permissive — consider 0.7 or 0.6 to reduce noise. |
| `min_annotation_score` | 0.7 | Minimum similarity to reference database for TF name assignment (step 11) | Clusters below this appear unannotated (gray in IGV). May be real motifs absent from the reference database, or degenerate composite patterns. |
| MoDISco seqlet p-value | default | Which positions within peaks seed seqlets (step 09) | Not explicitly set — MoDISco defaults apply. Permissive settings allow weak/noisy patterns into the compendium. |
| MoDISco `-n` | 500,000 | Max seqlets used for motif discovery (step 09) | Limits runtime; if peaks x high-scoring positions exceed this, a random subset is used. |

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
      {day}_average_shaps.counts.h5      fold-averaged contribution scores (step 07)
      {day}_average_shaps.counts.bw      BigWig for genome browser (step 08)
      modisco/
        modisco_counts_results.h5        MoDISco on averaged scores (step 09)
        counts_report/                   HTML report annotated against JASPAR

  predictions/
    {day}_{peak_type}/
      {day}_avg_chrombpnet_nobias.bw         bias-corrected accessibility (step 10)
      {day}_avg_chrombpnet_uncorrected.bw    uncorrected accessibility (step 10)

  compendium/
    modisco_compiled/
      modisco_config.tsv              day -> MoDISco H5 path (reference)
      modisco_compiled.h5             clustered motifs, input for Fi-NeMo (step 11)
      modisco_compendium.meme         MEME format export
      modisco_compendium.mc           pickled MotifCompendium object
      modisco_compendium_meta.tsv     per-motif TF annotations + cluster IDs
    bed/
      {day}_{peak_type}_hits.bed      BED9 with TF names + colors for IGV (step 13)

  finemo_unified/
    {day}_{peak_type}/
      hits.bed.gz + hits.bed.gz.tbi   tabix-indexed hit calls (step 12)
      hits.tsv                        full hit table
      intermediate_inputs.npz

  plots/
    full_model_qc/                    QC plots from 05.qc_run_full_model.sh
```

---

## Step-by-step execution

All commands are run from the `pipeline/` directory. Steps within each block can run
in parallel; blocks must complete in order.

```bash
# Block 1: Data preparation (run from pipeline/ directory)
srun --mem 100G --time 6:00:00 --partition engreitz bash 01.preprocess_peaks.sh
srun --mem 100G --time 6:00:00 --partition engreitz bash 02.preprocess_nonpeaks.sh

# Block 2: Bias model training
sbatch 03.train_bias_model.sh

# Block 3: Bias selection (after 03 completes)
bash 04.select_bias.sh             # writes QC plots; update fold_bias_suffix in config.sh

# Block 4: Full model training
sbatch 05.train_full_model.sh

# Block 5: Model QC (optional, can run alongside Block 6)
sbatch 05.qc_run_full_model.sh     # produces results/plots/full_model_qc/

# Block 6: Contribution scores (after 05 completes)
sbatch 06.get_contrib_scores.sh

# Block 7: Fold averaging (after 06 completes)
sbatch 07.average_contrib_scores.sh   # one averaged H5 per day
sbatch 08.contribs_to_bigwig.sh       # BigWig tracks (can run alongside 07)

# Block 8: MoDISco (after 07 completes)
sbatch 09.run_modisco.sh              # one MoDISco result per day

# Block 9: Predictions (optional, independent of blocks 8+)
sbatch 10.generate_predictions.sh

# Block 10: Unified motif compendium (after 09 completes)
sbatch 11.motif_compendium.sh         # cluster + annotate -> modisco_compiled.h5

# Block 11: Unified hit calls (after 11 completes)
sbatch 12.run_finemo_unified.sh       # cross-day comparable motif hits

# Block 12: IGV visualisation (after 12 completes; run from analysis/ directory)
sbatch 13.hits_to_bed.sh              # BED9 files with TF colors for IGV
```

### Minimal run (no QC or optional steps)

```
01 -> 02 -> 03 -> 04 -> 05 -> 06 -> 07 -> 09 -> 11 -> 12 -> 13
```
