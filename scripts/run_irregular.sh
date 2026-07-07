#!/bin/bash
# Irregular-topology experiment — the regime where RL can beat hand-designed
# heuristics. On these graphs the optimal order is NOT monotonic in
# distance-from-output, so the strong structural heuristic
# (DistanceScheduleBaseline) is provably suboptimal (~0.89 vs oracle 1.00),
# leaving room for a learned policy that reads the whole adjacency.
#
# Trains a GNN on irregular flow graphs (3 seeds), then runs the four-way
# comparison (oracle / agent / distance-heuristic / random).
#
# Usage:  caffeinate -i bash scripts/run_irregular.sh
# Wall time: ~2.5–3 h (3 × 1.5M steps; gflow is verified per episode).

set -e
cd "$(dirname "$0")/.."
PY=/usr/local/bin/python3.13

run_if_missing () {
    local dir="$1"; shift
    if [ -f "$dir/final.pt" ]; then echo ">>> SKIP $dir (final.pt exists)"
    else echo ">>> RUN  $dir"; "$@"; fi
}

echo "════════════════════════════════════════════════════════"
echo "  IRREGULAR-TOPOLOGY EXPERIMENT — started $(date)"
echo "════════════════════════════════════════════════════════"

# Train on clean irregular graphs (the regime where the heuristic fails).
for s in 0 1 2; do
    run_if_missing results/gnn_irregular_s$s \
        $PY scripts/train_gnn.py --rows 4 --cols 4 --topology irregular \
            --defect-lo 0 --defect-hi 0 \
            --hidden-dim 128 --n-heads 4 --n-layers 3 \
            --total-timesteps 1500000 --seed $s \
            --save-dir results/gnn_irregular_s$s --log-interval 25
done

echo ""
echo "──────── FOUR-WAY COMPARISON (agent vs distance-heuristic) ────────"
for s in 0 1 2; do
    echo ">>> seed $s:"
    $PY scripts/compare_baselines.py \
        --checkpoint results/gnn_irregular_s$s/final.pt \
        --rows 4 --cols 4 --topology irregular --human distance \
        --defect-rates 0.0 0.01 0.05 0.10 \
        --n-trials 400 --save-dir results/figures_irregular
done

echo "════════════════════════════════════════════════════════"
echo "  DONE $(date) — tell Claude to aggregate the 3 seeds."
echo "  Expected: agent > distance-heuristic (unlike on lattices)."
echo "════════════════════════════════════════════════════════"
