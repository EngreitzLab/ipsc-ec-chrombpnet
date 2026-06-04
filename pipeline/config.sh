#!/bin/bash
# config.sh - Shared configuration for the iPSC-EC ChromBPNet pipeline
# Source this at the top of every pipeline script: source config.sh

# Core paths
# core_path is auto-detected from config.sh location; override if needed
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
core_path="$(dirname "${SCRIPT_DIR}")"

results_path="${core_path}/results"
bam_path="${core_path}/data/bams"
folds_dir="${core_path}/data/folds"
log_dir="${SCRIPT_DIR}/logs"

# Preprocessing directory: reference files + filtered peaks (written by steps 01-02)
# Named data_path for compatibility with downstream scripts
data_path="${results_path}/preprocessing"

# Reference files
# Place hg38.fa, hg38.chrom.sizes, and blacklist.bed.gz in ${data_path} before step 01
genome_fa="${data_path}/hg38.fa"
chrom_sizes="${data_path}/hg38.chrom.sizes"
blacklist="${data_path}/blacklist.bed.gz"

# MotifCompendium reference database in MEME format (used by step 11)
# Download from: https://github.com/kundajelab/MotifCompendium
ref_db_meme="/oak/stanford/groups/engreitz/Users/opushkar/MotifCompendium/pipeline/data/MotifCompendium-Database-Human.meme.txt"

# Experiment parameters
peak_type="all"
days=( "d0" "d1" "d2" "d3" "d4" )

# Use folds=( "0" ) for a quick single-fold test run
folds=( "0" "1" "2" "3" "4" )

# Bias model configuration (steps 03-04)
# bias_day: which day's ATAC-seq is used to train the Tn5 bias model
# bias_factors/bias_suffixes_sweep: Tn5 threshold values to sweep
# suffix derived automatically: 0.5 -> _05, 0.6 -> _06, etc.
bias_day="d0"
bias_factors=( "0.5" "0.6" "0.7" "0.8" )
bias_suffixes_sweep=( "_05" "_06" "_07" "_08" )

# Per-fold bias model selection (populated after running 04.select_bias.sh)
# Maps fold index to the best bias suffix for that fold
# Update after reviewing 04.select_bias_model.py output
declare -A fold_bias_suffix=(
    [0]="_08"   # WARN: pk_r=-0.326 (best available; watch TFModisco for GC motifs)
    [1]="_07"   # PASS: np_r=+0.095, pk_r=+0.201
    [2]="_08"   # PASS: np_r=+0.340, pk_r=+0.190
    [3]="_08"   # PASS: np_r=+0.185, pk_r=+0.071
    [4]="_06"   # PASS: np_r=+0.453, pk_r=+0.393
)

# Output directories
full_model_dir="${results_path}/full_models"
full_model_dir_selected="${full_model_dir}" # alias kept for script compatibility
predictions_dir="${results_path}/predictions"
averaged_dir="${results_path}/contrib_scores"
compendium_dir="${results_path}/compendium"
modisco_compiled_dir="${compendium_dir}/modisco_compiled"
finemo_unified_dir="${results_path}/finemo_unified"
finemo_dir="${results_path}/finemo" # used by legacy _11.run_finemo.sh

# Algorithm parameters
finemo_alpha="0.8" # Fi-NeMo hit-calling threshold (lower = more hits)
motif_compendium_threshold="0.95" # Leiden clustering similarity cutoff (step 11)

# Conda environments
CONDA_INIT="/home/groups/engreitz/Software/anaconda3/etc/profile.d/conda.sh"
CONDA_ENV="/home/groups/engreitz/Users/opushkar/.conda/envs/chrombpnet"
finemo_conda="/home/groups/engreitz/Users/opushkar/.conda/envs/finemo"
motif_compendium_conda="/home/groups/engreitz/Users/opushkar/.conda/envs/motif_compendium"
