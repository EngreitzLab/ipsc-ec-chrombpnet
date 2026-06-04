#!/bin/bash

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "${SCRIPT_DIR}/config.sh"

python3 "${SCRIPT_DIR}/04.select_bias_model.py"
