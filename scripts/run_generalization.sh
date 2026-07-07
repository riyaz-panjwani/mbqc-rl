#!/bin/bash
# Generalisation phase — attack the train↔held-out gap on irregular graphs.
#
# The ladder localised the barrier: with the best config (virtual+RWSE) training
# reward hits ~0.97 but held-out is ~0.84 (heuristic 0.885). The network fits the
# training graphs but doesn't generalise across the per-episode-random
# distribution. This phase applies the standard tools for an overfitting gap, on
# top of the winning combo (foundation + virtual + RWSE):
#
#   r_wd        + weight decay 1e-4                  (L2 regularisation)
#   r_rand      + irregularity randomisation         (skip-prob U[0.1,0.8]:
#                                                      train on varied graphs,
#                                                      eval at fixed 0.45)
#   r_wd_rand   + both
#
# Eval is always at the standard distribution (fixed skip 0.45) vs the 0.885
# heuristic. Single seed to find what helps; multi-seed the winner afterwards.
#
# Usage:  caffeinate -i bash scripts/run_generalization.sh
# Long: 3 × 1.5M-step RL. Overnight; safe to re-run (skips finished).

set -e
cd "$(dirname "$0")/.."
PY=/usr/local/bin/python3.13

BASE="--topology irregular --observe-original --reward-shaping \
--defect-lo 0 --defect-hi 0 --hidden-dim 128 --n-heads 4 --n-layers 5 \
--virtual-node --pos-dim 8 --total-timesteps 1500000 --seed 0 --log-interval 25"

run () {  # dir, extra-flags
    local dir="results/generalization/$1"; shift
    if [ -f "$dir/final.pt" ]; then echo ">>> SKIP $dir"
    else echo ">>> RUN  $dir   $*"; $PY scripts/train_gnn.py $BASE --save-dir "$dir" "$@"; fi
}

echo "════════════════════════════════════════════════════════"
echo "  GENERALISATION PHASE — started $(date)"
echo "════════════════════════════════════════════════════════"

run r_wd        --weight-decay 1e-4
run r_rand      --skip-lo 0.1 --skip-hi 0.8
run r_wd_rand   --weight-decay 1e-4 --skip-lo 0.1 --skip-hi 0.8

echo ""
echo "──────── EVAL vs distance heuristic (≈0.885), standard dist. ────────"
for d in r_wd r_rand r_wd_rand; do
    echo ">>> $d"
    $PY scripts/compare_baselines.py \
        --checkpoint results/generalization/$d/final.pt \
        --rows 4 --cols 4 --topology irregular --human distance \
        --defect-rates 0.0 0.01 0.05 --n-trials 400 \
        --save-dir results/generalization/figs 2>/dev/null | grep -E "rate|0.00|0.01|0.05"
done

echo "════════════════════════════════════════════════════════"
echo "  DONE $(date) — tell Claude. Baselines: no-reg virtual+RWSE = 0.839±0.037,"
echo "  heuristic 0.885. If a variant clears ~0.86+, multi-seed it to confirm."
echo "════════════════════════════════════════════════════════"
