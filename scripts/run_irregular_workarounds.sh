#!/bin/bash
# Irregular regime — RL with the work-arounds the imitation study identified.
#
# Diagnosis (scripts/behavioral_clone.py): on irregular graphs the agent failed
# because (a) the adjacency observation is DEGRADED as qubits are measured,
# destroying the global gflow structure, and (b) gflow layer is a global property
# needing a deep receptive field. Behavioral cloning confirmed this — with the
# original (static) adjacency + a deeper GNN, supervised imitation rose from
# 0.48 (random) to ~0.76.
#
# This script tests whether those fixes also let RL learn:
#   --observe-original : show the full graph every step (history marks measured)
#   --reward-shaping   : dense per-step reward (rules out reward sparsity)
#   --n-layers 5       : deeper receptive field for the global gflow order
#
# Trains 3 seeds, then four-way comparison vs the distance heuristic (~0.89).
#
# Usage:  caffeinate -i bash scripts/run_irregular_workarounds.sh
# Wall time: ~3–4 h (3 × 1.5M steps; deeper GNN + gflow per episode).

set -e
cd "$(dirname "$0")/.."
PY=/usr/local/bin/python3.13

run_if_missing () {
    local dir="$1"; shift
    if [ -f "$dir/final.pt" ]; then echo ">>> SKIP $dir (final.pt exists)"
    else echo ">>> RUN  $dir"; "$@"; fi
}

echo "════════════════════════════════════════════════════════"
echo "  IRREGULAR + WORK-AROUNDS (static obs + shaping + depth) — $(date)"
echo "════════════════════════════════════════════════════════"

for s in 0 1 2; do
    run_if_missing results/gnn_irregular_fix_s$s \
        $PY scripts/train_gnn.py --rows 4 --cols 4 --topology irregular \
            --observe-original --reward-shaping --defect-lo 0 --defect-hi 0 \
            --hidden-dim 128 --n-heads 4 --n-layers 5 \
            --total-timesteps 1500000 --seed $s \
            --save-dir results/gnn_irregular_fix_s$s --log-interval 25
done

echo ""
echo "  NOTE: evaluate these with observe_original_graph=True. Use a small"
echo "  eval that builds the static observation (see behavioral_clone.py), or"
echo "  tell Claude to run the matched four-way comparison."
echo ""
echo "  Reference: distance heuristic ≈ 0.89, BC(static,5-layer) ≈ 0.76,"
echo "             sparse/degraded RL ≈ 0.49 (failed)."
echo "════════════════════════════════════════════════════════"
echo "  DONE $(date)"
echo "════════════════════════════════════════════════════════"
