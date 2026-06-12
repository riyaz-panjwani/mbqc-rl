#!/bin/bash
# Overnight experiment suite — statistical rigor + 5×5 forgetting fix.
#
# Runs sequentially (~6-7 hours total on Apple Silicon):
#   1. 5×5 mixed       (2.25M steps)        — does the forgetting fix scale?
#   2. 4×4 mixed       seeds 0,1,2 (1.5M ea) — reproducibility of the fix
#   3. 3×3 angled mixed seeds 0,1,2 (750k ea) — reproducibility of oracle-match
#
# Usage:  caffeinate -i bash scripts/run_seed_suite.sh
# Safe to re-run: completed runs are skipped (checks for final.pt).

set -e
cd "$(dirname "$0")/.."
PY=/usr/local/bin/python3.13

run_if_missing () {
    local dir="$1"; shift
    if [ -f "$dir/final.pt" ]; then
        echo ">>> SKIP $dir (final.pt exists)"
    else
        echo ">>> RUN  $dir"
        "$@"
    fi
}

echo "════════════════════════════════════════════════"
echo "  SEED SUITE — started $(date)"
echo "════════════════════════════════════════════════"

# ── 1. 5×5 mixed: does the forgetting fix scale? ──────────────────
run_if_missing results/5x5_mixed \
    $PY scripts/train_mixed.py --rows 5 --cols 5 \
        --total-timesteps 2250000 --hidden-dim 512 --n-steps 1024 \
        --save-dir results/5x5_mixed --log-interval 20

# ── 2. 4×4 mixed reproducibility (seeds 0,1,2; seed 42 exists) ────
for s in 0 1 2; do
    run_if_missing results/4x4_mixed_s$s \
        $PY scripts/train_mixed.py --rows 4 --cols 4 \
            --total-timesteps 1500000 --seed $s \
            --save-dir results/4x4_mixed_s$s --log-interval 20
done

# ── 3. 3×3 angled reproducibility (seeds 0,1,2; seed 42 exists) ───
for s in 0 1 2; do
    run_if_missing results/3x3_angled_mixed_s$s \
        $PY scripts/train_mixed.py --rows 3 --cols 3 --use-angles \
            --total-timesteps 750000 --seed $s \
            --save-dir results/3x3_angled_mixed_s$s --log-interval 20
done

echo "════════════════════════════════════════════════"
echo "  SEED SUITE — finished $(date)"
echo "════════════════════════════════════════════════"
