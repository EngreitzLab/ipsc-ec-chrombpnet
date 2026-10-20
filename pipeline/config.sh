#!/bin/bash
# config.sh - Pipeline configuration for ChromBPNet
# Source this at the top of every pipeline script: source config.sh
#
# This file sets pipeline-generic parameters. Dataset-specific parameters
# (days, genome, BAM paths, bias selection) live in dataset_config.sh,
# which is sourced automatically below.

# Core paths (auto-detected from config.sh location; override if needed)
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
core_path="$(dirname "${SCRIPT_DIR}")"
log_dir="${SCRIPT_DIR}/logs"

# Preprocessing directory: reference files + filtered peaks (written by steps 01-02)
# Named data_path for compatibility with downstream scripts
genome_path="${core_path}/genome"
results_path="${core_path}/results"
data_path="${results_path}/preprocessing"
# Dataset-specific parameters (days, genome, BAMs, bias config)
# core_path and data_path are available to dataset_config.sh
source "${SCRIPT_DIR}/dataset_config.sh"

# Output directories
full_model_dir="${results_path}/full_models"
full_model_dir_selected="${full_model_dir}" # alias kept for script compatibility
predictions_dir="${results_path}/predictions"
averaged_dir="${results_path}/contrib_scores"
compendium_dir="${results_path}/compendium"
modisco_compiled_dir="${compendium_dir}/modisco_compiled"
modisco_compiled_profile_dir="${compendium_dir}/modisco_compiled_profile"
finemo_unified_dir="${results_path}/finemo_unified"
finemo_dir="${results_path}/finemo"

# MotifCompendium reference database in MEME format (used by step 09)
# Download from: https://github.com/kundajelab/MotifCompendium
ref_db_meme="/path/to/MotifCompendium/pipeline/data/MotifCompendium-Database-Human.meme.txt"

# Algorithm parameters
finemo_alpha="0.8" # Fi-NeMo hit-calling threshold (lower = more hits)
motif_compendium_threshold="0.95" # Leiden clustering similarity cutoff (step 09)

# Conda environments
CONDA_INIT="/path/to/anaconda3/etc/profile.d/conda.sh"
CONDA_ENV="/path/to/conda/envs/chrombpnet"
finemo_conda="/path/to/conda/envs/finemo"
motif_compendium_conda="/path/to/conda/envs/motif_compendium"
