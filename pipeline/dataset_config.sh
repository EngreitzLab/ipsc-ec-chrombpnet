#!/bin/bash
# dataset_config.sh - Dataset-specific parameters for the iPSC-EC ChromBPNet pipeline
#
# Sourced automatically by config.sh after core_path and data_path are set.
# To run the pipeline on a new dataset, copy this file into that project's pipeline/
# directory and update the values below.
#
# Variables available when this file is sourced:
#   core_path   - project root (parent of pipeline/)
#   genome_path - genome reference files

# Experiment parameters
peak_type="all"
days=( "d0" "d1" "d2" "d3" "d4" )

# Use folds=( "0" ) for a quick single-fold test run
folds=( "0" "1" "2" "3" "4" )

# Pseudobulk peaks and fragments, processed with the IGVF scE2G pipeline
peaks_path="/path/to/10x_5_timepoints/scE2G/results"
# Input data: deduplicated pseudobulk ATAC fragment files (same files used for scE2G)
frag_path="/path/to/10x_5_timepoints/Pseudobulks/ATAC"
folds_dir="${genome_path}/folds"

# Deduplicated d0 fragment file for bias model training
frag_d0="${frag_path}/d0.fragments.tsv.gz"

# Reference genome files
# Place hg38.fa, hg38.chrom.sizes, and blacklist.bed.gz in ${genome_path} before step 01
genome_fa="${genome_path}/hg38.fa"
chrom_sizes="${genome_path}/hg38.chrom.sizes"
# Header for output bigWigs (steps 04.2, 07): hg38.chrom.sizes restricted to the 195 contigs
# of IGVF portal GRCh38 (IGVFFI6788CPPS); no _alt contigs, which the portal rejects.
chrom_sizes_bigwig="${genome_path}/hg38.portal.chrom.sizes"
blacklist="${genome_path}/blacklist.bed.gz"

# Bias model configuration (steps 03-04)
# bias_day: which day's ATAC-seq is used to train the Tn5 bias model
# bias_factors/bias_suffixes_sweep: Tn5 threshold values to sweep
# suffix derived automatically: 0.5 -> _05, 0.6 -> _06, etc.
bias_day="d0"
bias_factors=( "0.5" "0.6" "0.7" "0.8" "0.9" "1" )
bias_suffixes_sweep=( "_05" "_06" "_07" "_08" "_09" "_1")

# Per-fold bias model selection (populated after running 03.1.select_bias.sh)
# Maps fold index to the best bias suffix for that fold
declare -A fold_bias_suffix=(
    [0]="_08"
    [1]="_08"
    [2]="_07"
    [3]="_08"
    [4]="_09"
)
