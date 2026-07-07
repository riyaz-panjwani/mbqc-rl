#!/bin/bash
# Rung 4 — auxiliary gflow-layer supervision (NAR "hints"), the top-ranked lever.
#
# The ladder confirmed virtual+RWSE reaches 0.839 ± 0.037 — near but not beating
# the 0.885 distance heuristic. Rung 4 adds a supervised head predicting each
# node's gflow layer alongside PPO, directly injecting the signal the GNN
# struggles to compute. Requires the static observation (--observe-original), so
# the layer target is well-posed.
#
# Runs (all on the foundation: static-obs + reward-shaping + depth 5):
#   r4_aux            aux only                    (isolate aux's contribution)
#   r34_rwse_aux      RWSE + aux                  (aux on the main lever)
#   r134_all_aux_s0/1/2  virtual + RWSE + aux     (best combo + aux, 3 seeds)
#
# Usage:  caffeinate -i bash scripts/run_ladder_rung4.sh
# Long: 5 × 1.5M-step RL. Overnight; safe to re-run (skips finished).

set -e
cd "$(dirname "$0")/.."
PY=/usr/local/bin/python3.13

FOUND="--topology irregular --observe-original --reward-shaping \
--defect-lo 0 --defect-hi 0 --hidden-dim 128 --n-heads 4 --n-layers 5 \
--total-timesteps 1500000 --log-interval 25 --aux-layer --aux-weight 1.0"

run () {  # dir, seed, extra-flags
    local dir="results/ladder/$1"; local seed="$2"; shift 2
    if [ -f "$dir/final.pt" ]; then echo ">>> SKIP $dir"
    else echo ">>> RUN  $dir   seed=$seed   $*"
         $PY scripts/train_gnn.py $FOUND --seed "$seed" --save-dir "$dir" "$@"; fi
}

echo "════════════════════════════════════════════════════════"
echo "  RUNG 4 (auxiliary gflow-layer supervision) — $(date)"
echo "════════════════════════════════════════════════════════"

run r4_aux            0
run r34_rwse_aux      0 --pos-dim 8
run r134_all_aux_s0   0 --virtual-node --pos-dim 8
run r134_all_aux_s1   1 --virtual-node --pos-dim 8
run r134_all_aux_s2   2 --virtual-node --pos-dim 8

echo ""
echo "──────── EVAL vs distance heuristic (≈0.885) on irregular ────────"
for d in r4_aux r34_rwse_aux r134_all_aux_s0 r134_all_aux_s1 r134_all_aux_s2; do
    echo ">>> $d"
    $PY scripts/compare_baselines.py \
        --checkpoint results/ladder/$d/final.pt \
        --rows 4 --cols 4 --topology irregular --human distance \
        --defect-rates 0.0 0.01 0.05 --n-trials 400 \
        --save-dir results/ladder/figs 2>/dev/null | grep -E "rate|0.00|0.01|0.05"
done

echo "════════════════════════════════════════════════════════"
echo "  DONE $(date) — tell Claude to aggregate r134_all_aux seeds 0-2"
echo "  vs the no-aux baseline (virtual+RWSE = 0.839 ± 0.037)."
echo "════════════════════════════════════════════════════════"
