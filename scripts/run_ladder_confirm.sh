#!/bin/bash
# Confirm the ladder winner is real, not seed noise.
#
# The sweep's best config — virtual node + RWSE encodings on the foundation —
# edged the distance heuristic (0.890 vs 0.885) at seed 0, a narrow margin. This
# re-runs that exact config at seeds 1 and 2 so we can report mean ± std across 3
# seeds and state whether the "beat" is robust.
#
# Usage:  caffeinate -i bash scripts/run_ladder_confirm.sh
# Wall time: ~3 h (2 × 1.5M-step RL). Safe to re-run (skips finished).

set -e
cd "$(dirname "$0")/.."
PY=/usr/local/bin/python3.13

FOUND="--topology irregular --observe-original --reward-shaping \
--defect-lo 0 --defect-hi 0 --hidden-dim 128 --n-heads 4 --n-layers 5 \
--total-timesteps 1500000 --log-interval 25 --virtual-node --pos-dim 8"

echo "════════════════════════════════════════════════════════"
echo "  CONFIRM WINNER (virtual + RWSE), seeds 1,2 — $(date)"
echo "════════════════════════════════════════════════════════"

for s in 1 2; do
    dir="results/ladder/r13_virtual_rwse_s$s"
    if [ -f "$dir/final.pt" ]; then echo ">>> SKIP $dir"
    else echo ">>> RUN  $dir"; $PY scripts/train_gnn.py $FOUND --seed $s --save-dir "$dir"; fi
done

echo ""
echo "──────── EVAL all 3 seeds vs distance heuristic ────────"
for d in r13_virtual_rwse r13_virtual_rwse_s1 r13_virtual_rwse_s2; do
    echo ">>> $d"
    $PY scripts/compare_baselines.py \
        --checkpoint results/ladder/$d/final.pt \
        --rows 4 --cols 4 --topology irregular --human distance \
        --defect-rates 0.0 0.01 0.05 --n-trials 400 \
        --save-dir results/ladder/figs 2>/dev/null | grep -E "rate|0.00|0.01|0.05"
done

echo "════════════════════════════════════════════════════════"
echo "  DONE $(date) — tell Claude to aggregate the 3 seeds."
echo "  Seed 0 held-out: agent 0.890 vs heuristic 0.885."
echo "════════════════════════════════════════════════════════"
