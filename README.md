# ChromBPNet iPSC-EC Differentiation Pipeline

Deep learning-based chromatin accessibility analysis (ChromBPNet) across 5 iPSC-to-EC
differentiation timepoints (d0–d4), followed by motif discovery and TF activity analyses.

---

## Repository structure

```
pipeline/          ChromBPNet workflow: preprocessing -> training -> motif discovery
analysis/          Downstream scientific analyses: TF activity, gene-motif linking, TEAD enrichment
data/              Input data: BAMs, peak BED files, scE2G links
data/folds/        Cross-validation fold definitions (JSON) used by the pipeline
```

Large output directories (gitignored):
```
results/preprocessing/        Preprocessed peaks, non-peaks, genome reference
results/bias_models/          Trained Tn5 bias models
results/full_models/          Trained ChromBPNet models (5 days x 5 folds)
results/contrib_scores/       Fold-averaged contribution scores + MoDISco results
results/compendium/           Unified motif compendium (MotifCompendium clustering)
results/finemo_unified/       Cross-day comparable motif hit calls (Fi-NeMo)
results/predictions/          Predicted accessibility BigWig tracks
results/plots/                QC and analysis output plots
```

---

## Quick start

### 1. Configure paths

Edit `pipeline/config.sh`:
- `core_path` is auto-detected from the script location; override only if needed
- Place `hg38.fa`, `hg38.chrom.sizes`, and `blacklist.bed.gz` in `results/preprocessing/`
- Set `CONDA_ENV`, `finemo_conda`, `motif_compendium_conda` to your conda environment paths
- Set `ref_db_meme` to the path of `MotifCompendium-Database-Human.meme.txt`

### 2. Run the pipeline

All scripts are in `pipeline/`. Run from that directory:

```bash
cd pipeline/

# Stage 1: Preprocess peaks and non-peaks
bash 01.preprocess_peaks.sh
bash 02.preprocess_nonpeaks.sh

# Stage 2: Train bias model, select best, train full model
sbatch 03.train_bias_model.sh
bash 04.select_bias.sh          # after 03 completes; updates fold_bias_suffix in config.sh
sbatch 05.train_full_model.sh

# Stage 3: Contribution scores, averaging, MoDISco
sbatch 06.get_contrib_scores.sh
sbatch 07.average_contrib_scores.sh
sbatch 08.contribs_to_bigwig.sh
sbatch 09.run_modisco.sh

# Stage 4 (optional): Predicted accessibility BigWigs
sbatch 10.generate_predictions.sh

# Stage 5: Unified motif compendium and hit calling
sbatch 11.motif_compendium.sh
sbatch 12.run_finemo_unified.sh
sbatch 13.hits_to_bed.sh
```

See `pipeline/README.md` for full documentation.

### 3. Run downstream analyses

Scripts in `analysis/` are numbered sequentially and run independently:

```bash
cd analysis/
python 01.pattern_logos.py           # CWM logos for all motif clusters
python 02.gene_motif_linking.py      # TF scores linked to genes via scE2G
python 08.tf_activity.py             # TF activity heatmaps and dynamics across days
python 05.tead_enrichment.py         # TEAD motif enrichment in cNMF programs
```

See `analysis/README.md` for full documentation.

---

## Dependencies

| Tool | Purpose |
|------|---------|
| ChromBPNet | Bias-factorised accessibility model |
| TF-MoDISco (modiscolite) | Motif discovery from contribution scores |
| MotifCompendium | Cross-day motif clustering and JASPAR annotation |
| Fi-NeMo | Motif hit calling from contribution scores |
| scanpy / muon | Single-cell data handling |
| pybedtools | Genomic coordinate operations |

Conda environments used:
- `chrombpnet`: main pipeline (ChromBPNet, modiscolite)
- `finemo`: Fi-NeMo hit calling
- `motif_compendium`: MotifCompendium clustering

---

## References

Genome: hg38
Motif annotation: MotifCompendium-Database-Human (download from https://github.com/kundajelab/MotifCompendium)