#!/bin/bash
#SBATCH --job-name=motif_compendium_profile
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --partition=normal
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# 09.1.motif_compendium_profile.sh
# Purpose: Build a non-redundant motif compendium from per-day averaged
#          profile-score MoDISco H5 files (step 08), kept fully separate
#          from the counts compendium (step 09) so counts results/paths
#          are never touched.
#
# Input:  Per-day fold-averaged MoDISco H5s (step 08)
#           ${averaged_dir}/{day}/modisco/modisco_profile_results.h5
# Output (inside ${modisco_compiled_profile_dir}/):
#   modisco_compiled.h5          - clustered motifs for FiNeMo (step 10.1)
#   modisco_compendium.meme      - MEME format export
#   modisco_compendium.mc        - pickled MotifCompendium object
#   modisco_compendium_meta.tsv  - per-motif TF annotations + cluster IDs
#   modisco_config.tsv           - day -> H5 path mapping (for reference)
#
# Prerequisites: 08.run_modisco.sh must have completed for score_type=profile.
#   Requires the 'motif_compendium' conda environment (same as step 09):
#     mamba create -n motif_compendium python=3.10
#     pip install MotifCompendium
#
# Usage:
#   sbatch 09.1.motif_compendium_profile.sh

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "${SCRIPT_DIR}/config.sh"

mkdir -p "${modisco_compiled_profile_dir}" "${log_dir}"

# Build config TSV: day_name <TAB> modisco_h5_path
config_tsv="${modisco_compiled_profile_dir}/modisco_config.tsv"
echo "# day    modisco_h5" > "${config_tsv}"

for day in "${days[@]}"; do
    modisco_h5="${averaged_dir}/${day}/modisco/modisco_profile_results.h5"
    if [[ -f "${modisco_h5}" ]]; then
        echo -e "${day}\t${modisco_h5}" >> "${config_tsv}"
    else
        echo "  [WARN] missing averaged MoDISco H5 for ${day}: ${modisco_h5}" >&2
        echo "  Run 08.run_modisco.sh (score_type=profile) first." >&2
    fi
done

n_found=$(grep -c "^[^#]" "${config_tsv}" || true)
echo "[$(date)] Config TSV: ${config_tsv} (${n_found} days)"

if [[ "${n_found}" -eq 0 ]]; then
    echo "ERROR: no profile MoDISco H5s found. Run 08.run_modisco.sh first." >&2
    exit 1
fi

# Run MotifCompendium clustering + annotation

source "${CONDA_INIT}"
conda activate "${motif_compendium_conda}"

echo "[$(date)] Running MotifCompendium on profile scores (threshold=${motif_compendium_threshold})..."

python "${SCRIPT_DIR}/motif_compendium.py" \
    --config "${config_tsv}" \
    --out-dir "${modisco_compiled_profile_dir}" \
    --ref-db "${ref_db_meme}" \
    --threshold "${motif_compendium_threshold}" \
    --cpus "${SLURM_CPUS_PER_TASK:-16}"

if [[ ! -f "${modisco_compiled_profile_dir}/modisco_compiled.h5" ]]; then
    echo "ERROR: modisco_compiled.h5 not produced. Check the log above." >&2
    exit 1
fi

echo "[$(date)] 09.1.motif_compendium_profile complete."
echo "  Compiled H5  : ${modisco_compiled_profile_dir}/modisco_compiled.h5"
echo "  Annotations  : ${modisco_compiled_profile_dir}/modisco_compendium_meta.tsv"
echo ""
echo "  Next step: sbatch 10.1.run_finemo_unified_profile.sh"
