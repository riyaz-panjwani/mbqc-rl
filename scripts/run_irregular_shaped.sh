#!/bin/bash
# Irregular-topology experiment WITH dense reward shaping — the fix for the
# learning failure seen with the sparse terminal reward (agent was stuck at
# random because every episode is a new graph and the terminal-only signal
# gives no gradient).
#
# Reward shaping decomposes the same gflow-consistency score into a per-step
# signal (undiscounted return is identical), giving usable credit assignment.
# Uses 2 GAT layers (the ablation's sweet spot).
#
# Trains 3 seeds, then runs the four-way comparison
# (oracle / agent / distance-heuristic / random) on irregular graphs.
# Target: agent > distance-heuristic (~0.89) — i.e. RL beats the heuristic in
# the regime where the heuristic provably fails.
#
# Usage:  caffeinate -i bash scripts/run_irregular_shaped.sh
# Wall time: ~3 h (3 × 1.5M steps; gflow verified per episode).

set -e
cd "$(dirname "$0")/.."
PY=/usr/local/bin/python3.13

run_if_missing () {
    local dir="$1"; shift
    if [ -f "$dir/final.pt" ]; then echo ">>> SKIP $dir (final.pt exists)"
    else echo ">>> RUN  $dir"; "$@"; fi
}

echo "════════════════════════════════════════════════════════"
echo "  IRREGULAR + REWARD SHAPING — started $(date)"
echo "════════════════════════════════════════════════════════"

for s in 0 1 2; do
    run_if_missing results/gnn_irregular_shaped_s$s \
        $PY scripts/train_gnn.py --rows 4 --cols 4 --topology irregular \
            --reward-shaping --defect-lo 0 --defect-hi 0 \
            --hidden-dim 128 --n-heads 4 --n-layers 2 \
            --total-timesteps 1500000 --seed $s \
            --save-dir results/gnn_irregular_shaped_s$s --log-interval 25
done

echo ""
echo "──────── FOUR-WAY COMPARISON (agent vs distance-heuristic) ────────"
for s in 0 1 2; do
    echo ">>> seed $s:"
    $PY scripts/compare_baselines.py \
        --checkpoint results/gnn_irregular_shaped_s$s/final.pt \
        --rows 4 --cols 4 --topology irregular --human distance \
        --defect-rates 0.0 0.01 0.05 0.10 \
        --n-trials 400 --save-dir results/figures_irregular_shaped
done

echo "════════════════════════════════════════════════════════"
echo "  DONE $(date) — tell Claude to aggregate."
echo "  Sparse-reward baseline failed (agent≈0.49). Shaped target: agent>0.89."
echo "════════════════════════════════════════════════════════"
