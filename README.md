# ChromBPNet iPSC-EC Differentiation Pipeline

Deep learning-based chromatin accessibility analysis (ChromBPNet) across 5 iPSC-to-EC
differentiation timepoints (d0–d4), followed by motif discovery and TF activity analyses.

---

## Repository structure

```
pipeline/          ChromBPNet workflow: preprocessing -> training -> motif discovery
analysis/          Downstream analyses: QC, motif annotation, gene-motif linking, TF motif enrichment in cNMF programs
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
- Edit `pipeline/dataset_config.sh` for days, folds, and input peak/fragment paths
- Place `hg38.fa`, `hg38.chrom.sizes`, and `blacklist.bed.gz` in `genome/`
- `#SBATCH` partition/QOS lines are generic (`normal`, `gpu`); adjust them for your cluster
- Set `CONDA_INIT`, `CONDA_ENV`, `finemo_conda`, `motif_compendium_conda` to your conda paths (envs in `envs/`)
- Set `ref_db_meme` to the path of `MotifCompendium-Database-Human.meme.txt`

### 2. Run the pipeline

All scripts are in `pipeline/`. Run from that directory:

```bash
cd pipeline/

# Stage 1: Preprocess peaks and non-peaks
bash 01.preprocess_peaks.sh
bash 02.preprocess_nonpeaks.sh

# Stage 2: Train bias models, select per fold, QC, train full model
sbatch 03.0.train_bias_model.sh
bash 03.1.select_bias.sh            # after 03.0; update fold_bias_suffix in dataset_config.sh
sbatch 03.2.qc_selected_bias.sh
sbatch 04.0.train_full_model.sh
sbatch 04.1.qc_run_full_model.sh    # optional model QC
sbatch 04.2.generate_predictions.sh # optional predicted-accessibility BigWigs

# Stage 3: Contribution scores, fold averaging, MoDISco
sbatch 05.get_contrib_scores.sh
sbatch 06.average_contrib_scores.sh
sbatch 07.contribs_to_bigwig.sh
sbatch 08.run_modisco.sh

# Stage 4: Unified motif compendium and hit calling (counts head)
sbatch 09.motif_compendium.sh
sbatch 10.run_finemo_unified.sh
sbatch 11.postprocess_finemo.sh
# Profile-head variants: 09.1.motif_compendium_profile.sh,
# 10.1.run_finemo_unified_profile.sh, 11.1.postprocess_finemo_profile.sh
```

See `pipeline/README.md` for full documentation.

### 3. Run downstream analyses

Scripts in `analysis/` are numbered in run order:

```bash
cd analysis/
python 00.chrombpnet_model_qc.py      # model and data QC
python 0.0.motif_id_annotation.py     # TF annotation of compendium motifs
python 0.2.finemo_hit_qc.py           # CWM-similarity filtering of hits
sbatch 0.3.hits_to_bed.sh             # IGV tracks
python 1.0.gene_motif_linking.py      # TF scores linked to genes via scE2G
python 2.tf_enrichment_in_programs.py # TF motif enrichment in cNMF programs
python 3.chrombpnet_figures.py        # figures for programs of interest
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