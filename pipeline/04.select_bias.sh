#!/bin/bash

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "${SCRIPT_DIR}/config.sh"

source "${CONDA_INIT}"
conda activate "${CONDA_ENV}"

python3 "${SCRIPT_DIR}/04.0.select_bias_model.py" \
    --core-path "${core_path}"

python3 "${SCRIPT_DIR}/04.1.qc_bias_selection.py" \
    --core-path "${core_path}"
