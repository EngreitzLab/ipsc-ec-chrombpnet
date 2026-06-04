# Pipeline issues to fix

## 1. README and config.sh reference non-existent script names
**Files:** `README.md` (lines 299–300, 27), `config.sh` (lines 110–116)

The docs reference `07.average_contrib_scores.sh` and `09.run_modisco.sh`
but the actual files are `07.average_contrib_scores.sh` and `09.run_modisco.sh`.
Update all references in README execution block, stage overview table, and config.sh comments.

---

## 2. `--output-bed` argparse bool bug
**File:** `10.predict_and_avg.py` (line 51)

```python
# Current (broken for string "False"):
p.add_argument("-ob", "--output-bed", type=bool, default=False, ...)

# Fix:
p.add_argument("-ob", "--output-bed", action="store_true", ...)
# and remove `--output-bed True` from 10.generate_predictions.sh, just pass the flag bare
```

`bool("False") == True` in Python, so passing `--output-bed False` would silently write the BED.
Currently always passing `True` so it works, but fragile.

---

## 3. Misleading echo in 3.0 contrib score scripts
**Files:** `06.get_contrib_scores.sh` (lines 52–53), and d1–d4 equivalents

The echo prints all days from config.sh, then the next line overrides `days` to a single day.
Move the `days=(...)` override to before the echo, or just remove the echo.

```bash
# Current:
echo "[$(date)] Fold ${fold}: computing contribution scores for days [${days[*]}]"
days=( "d0" );

# Fix:
days=( "d0" )
echo "[$(date)] Fold ${fold}: computing contribution scores for days [${days[*]}]"
```

---

## 4. `09.run_modisco.sh` requests GPUs unnecessarily
**File:** `09.run_modisco.sh` (lines 6–8)

`modisco motifs` (tfmodisco-lite) is CPU-dominated. Requesting 2 GPUs wastes resources
and inflates queue wait time. Match the resource spec from `07.average_contrib_scores.sh`.

```bash
# Remove:
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu
#SBATCH --constraint=[IB:HDR|IB:NDR]

# Replace with:
#SBATCH --partition=normal,owners
```

Also remove the `export CUDA_VISIBLE_DEVICES=0,1` and `export TF_FORCE_GPU_ALLOW_GROWTH=true` lines.

---

## 5. Shape assertion in `07.average_contrib_scores.py` checks wrong array pair
**File:** `07.average_contrib_scores.py` (line 116)

Currently compares `raw/seq` shape (fold 0) against `projected_shap/seq` shape (fold N).
Works incidentally since both are `(n_peaks, 4, seq_len)`, but the intent is to verify
cross-fold consistency for the same key.

```python
# Current:
assert raw.shape == avg_projected_shap.shape, ...

# Fix (check that fold N has same number of peaks as fold 0):
assert avg_projected_shap.shape == avg_projected.shape, \
    f"Shape mismatch at fold {fold}: {avg_projected_shap.shape} vs {avg_projected.shape}"
```
