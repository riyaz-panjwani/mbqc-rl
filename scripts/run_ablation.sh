#!/bin/bash
# GNN architecture ablation — what drives performance?
#
# All runs: 4×4 angled, mixed defect U[0,0.3], FIXED 1M-step budget, seed 0,
# so configs are directly comparable. Varies one axis at a time from the
# baseline (3 GAT layers, hidden 128, 4 heads).
#
#   depth  : n_layers ∈ {1, 2, 3}        (hidden 128)
#   width  : hidden  ∈ {64, 128, 256}    (3 layers)
#
# 5 distinct configs (3L/128 shared). Each benchmarked at 1% defect.
#
# Usage:  caffeinate -i bash scripts/run_ablation.sh
# Wall time: ~1.5–2 h (5 × 1M steps on 4×4).

set -e
cd "$(dirname "$0")/.."
PY=/usr/local/bin/python3.13
mkdir -p results/ablation

run_cfg () {
    local name="$1" layers="$2" hidden="$3"
    local dir="results/ablation/$name"
    if [ -f "$dir/final.pt" ]; then
        echo ">>> SKIP $dir (final.pt exists)"
    else
        echo ">>> RUN  $dir  (layers=$layers hidden=$hidden)"
        $PY scripts/train_gnn.py --rows 4 --cols 4 --use-angles \
            --hidden-dim "$hidden" --n-heads 4 --n-layers "$layers" \
            --total-timesteps 1000000 --seed 0 \
            --save-dir "$dir" --log-interval 25
    fi
}

echo "════════════════════════════════════════════════════════"
echo "  GNN ARCHITECTURE ABLATION (4×4 angled, 1M steps) — $(date)"
echo "════════════════════════════════════════════════════════"

# depth sweep (hidden=128)
run_cfg L1_h128 1 128
run_cfg L2_h128 2 128
run_cfg L3_h128 3 128      # baseline
# width sweep (layers=3)
run_cfg L3_h64  3 64
run_cfg L3_h256 3 256

echo ""
echo "──────────── ABLATION RESULTS (500 trials @ 1% defect) ────────────"
printf "  %-10s  %-8s  %-8s  %s\n" "config" "layers" "hidden" "mean reward"
for cfg in L1_h128:1:128 L2_h128:2:128 L3_h128:3:128 L3_h64:3:64 L3_h256:3:256; do
    name="${cfg%%:*}"; rest="${cfg#*:}"; layers="${rest%%:*}"; hidden="${rest##*:}"
    mean=$($PY scripts/benchmark.py \
        --checkpoint "results/ablation/$name/final.pt" \
        --n-trials 500 --defect-rate 0.01 --save-dir results/ablation/bench \
        2>/dev/null | grep "Mean reward" | awk '{print $3}')
    printf "  %-10s  %-8s  %-8s  %s\n" "$name" "$layers" "$hidden" "$mean"
done

echo "════════════════════════════════════════════════════════"
echo "  Done $(date)"
echo "════════════════════════════════════════════════════════"
