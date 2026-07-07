#!/bin/bash
# Does the irregular WIN survive the NON-CLIFFORD (universal-MBQC) reward?
#
# The 3-size irregular win (4x4/5x5/6x6) was established with the PLAIN
# gflow-consistency reward (use_angles=False): every non-output qubit imposes an
# ordering constraint. The *angled* reward (score_measurement_order_angled) is the
# universal-MBQC objective — only NON-CLIFFORD qubits are hard constraints
# (Clifford corrections are Pauli-frame-trackable), and the agent additionally
# OBSERVES each qubit's angle. This trains the SAME winning combo on irregular
# graphs WITH angles, 3 seeds, for the early-stopping + ensemble protocol.
#
# Usage:   caffeinate -i bash scripts/run_irregular_angled.sh [ROWS] [COLS]
#   default 4 4 (cheapest, primary size). Saves to results/irregular_angled/rRxC_s*.
#
# After it finishes, Claude runs (tooling now takes --use-angles):
#   python scripts/early_stopping_eval.py     --use-angles --rows R --cols C --runs results/irregular_angled/rRxC_s0 ...
#   python scripts/early_stopping_validate.py --use-angles --rows R --cols C --runs ...
#
# Long: 3 x 1.5M-step irregular RL. Run overnight. Safe to re-run (skips finished).

set -e
cd "$(dirname "$0")/.."
PY=/usr/local/bin/python3.13
ROWS=${1:-4}
COLS=${2:-4}

# Identical to the winning-combo COMBO in run_irregular_scale.sh, PLUS --use-angles.
COMBO="--topology irregular --observe-original --reward-shaping --use-angles \
--clifford-fraction 0.5 --defect-lo 0 --defect-hi 0 --hidden-dim 128 --n-heads 4 \
--n-layers 5 --virtual-node --pos-dim 8 --total-timesteps 1500000 --log-interval 25"

echo "════════════════════════════════════════════════════════"
echo "  IRREGULAR ANGLED (non-Clifford) — ${ROWS}x${COLS}, winning combo, 3 seeds — $(date)"
echo "════════════════════════════════════════════════════════"

for s in 0 1 2; do
    dir="results/irregular_angled/r${ROWS}x${COLS}_s$s"
    if [ -f "$dir/final.pt" ]; then echo ">>> SKIP $dir"
    else echo ">>> RUN  $dir"
         $PY scripts/train_gnn.py $COMBO --rows "$ROWS" --cols "$COLS" --seed $s --save-dir "$dir"; fi
done

echo "════════════════════════════════════════════════════════"
echo "  DONE $(date) — tell Claude to run the ANGLED early-stopping + ensemble"
echo "  eval at ${ROWS}x${COLS} (--use-angles) vs the distance heuristic."
echo "════════════════════════════════════════════════════════"
