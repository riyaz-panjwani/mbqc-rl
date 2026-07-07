#!/bin/bash
# GNN 4×4 angled — 3-seed reproducibility for the headline result.
#
# The single existing run (results/gnn_4x4_angled, seed 42) gave 0.943 home /
# 0.990 zero-shot 3×3. This reruns seeds 0,1,2 so we can report mean±std with
# the same rigor as the 18-run MLP suite, then aggregates automatically.
#
# Usage:  caffeinate -i bash scripts/run_gnn_seeds.sh
# Safe to re-run: completed runs are skipped (checks for final.pt).
# Wall time: ~50 min/seed on Apple Silicon → ~2.5 h total.

set -e
cd "$(dirname "$0")/.."
PY=/usr/local/bin/python3.13
mkdir -p results/seeds

run_if_missing () {
    local dir="$1"; shift
    if [ -f "$dir/final.pt" ]; then echo ">>> SKIP $dir (final.pt exists)"
    else echo ">>> RUN  $dir"; "$@"; fi
}

echo "════════════════════════════════════════════════════════"
echo "  GNN 4×4 ANGLED — 3 SEEDS — started $(date)"
echo "════════════════════════════════════════════════════════"

for s in 0 1 2; do
    run_if_missing results/gnn_4x4_angled_s$s \
        $PY scripts/train_gnn.py --rows 4 --cols 4 --use-angles \
            --hidden-dim 128 --n-heads 4 --n-layers 3 \
            --total-timesteps 2250000 --seed $s \
            --save-dir results/gnn_4x4_angled_s$s --log-interval 20
done

echo ""
echo "════════════════════════════════════════════════════════"
echo "  AGGREGATING (500 trials @ 1% defect, grid topology)"
echo "════════════════════════════════════════════════════════"
$PY scripts/aggregate_seeds.py \
    --pattern "results/gnn_4x4_angled_s*/final.pt" \
    --n-trials 500 --defect-rate 0.01 \
    --save results/seeds/gnn_4x4_angled_seeds.json

echo ""
echo "  Done $(date). Tell Claude — it will run the cross-size + brickwork"
echo "  transfer tables across all 3 seeds."
echo "════════════════════════════════════════════════════════"
