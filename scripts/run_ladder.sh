#!/bin/bash
# Irregular-regime SOLUTION LADDER — systematic bottom→top sweep + combinations.
#
# All runs share the proven FOUNDATION: irregular topology, static observation
# (--observe-original), dense reward (--reward-shaping), depth 5. On top, we add
# one rung at a time, then combinations, to see which closes the gap to the
# distance heuristic (~0.89). Single seed per config to identify what helps;
# multi-seed the winner afterwards.
#
# Ladder (bottom→top):
#   0. foundation only           (baseline for the deltas)
#   1. + virtual node            (anti-over-squashing)
#   2. + weight-tied processor   (algorithmic alignment)
#   3. + RWSE pos-encoding (k=8) (structural/positional info)
#   combinations: 2+3, 1+3, 1+2+3
#   (rung 4, auxiliary gflow-layer supervision, is added separately once built.)
#
# Usage:  caffeinate -i bash scripts/run_ladder.sh
# Long: 7 × 1.5M-step RL runs. Run overnight; safe to re-run (skips finished).

set -e
cd "$(dirname "$0")/.."
PY=/usr/local/bin/python3.13

FOUND="--topology irregular --observe-original --reward-shaping \
--defect-lo 0 --defect-hi 0 --hidden-dim 128 --n-heads 4 --n-layers 5 \
--total-timesteps 1500000 --seed 0 --log-interval 25"

run () {  # name, extra-flags
    local dir="results/ladder/$1"; shift
    if [ -f "$dir/final.pt" ]; then
        echo ">>> SKIP $dir (final.pt exists)"
    else
        echo ">>> RUN  $dir   flags: $*"
        $PY scripts/train_gnn.py $FOUND --save-dir "$dir" "$@"
    fi
}

echo "════════════════════════════════════════════════════════"
echo "  SOLUTION LADDER — started $(date)"
echo "════════════════════════════════════════════════════════"

run r0_foundation
run r1_virtual            --virtual-node
run r2_tied               --weight-tied
run r3_rwse               --pos-dim 8
run r23_tied_rwse         --weight-tied --pos-dim 8
run r13_virtual_rwse      --virtual-node --pos-dim 8
run r123_all              --virtual-node --weight-tied --pos-dim 8

echo ""
echo "──────── EVAL: agent vs distance-heuristic on irregular ────────"
for d in r0_foundation r1_virtual r2_tied r3_rwse r23_tied_rwse r13_virtual_rwse r123_all; do
    echo ">>> $d"
    $PY scripts/compare_baselines.py \
        --checkpoint results/ladder/$d/final.pt \
        --rows 4 --cols 4 --topology irregular --human distance \
        --defect-rates 0.0 0.01 0.05 \
        --n-trials 400 --save-dir results/ladder/figs 2>/dev/null \
        | grep -E "rate|0.00|0.01|0.05"
done

echo "════════════════════════════════════════════════════════"
echo "  LADDER DONE $(date) — tell Claude to rank the rungs."
echo "  Bar to beat: distance heuristic ≈ 0.89 (oracle 1.0)."
echo "════════════════════════════════════════════════════════"
