#!/bin/bash
# Does the irregular WIN generalise across SIZE? (size-parameterised)
#
# At 4x4 AND 5x5 (trained at size), the early-stopped ensemble of the winning
# combo (virtual + RWSE on the static-obs + reward-shaping + depth foundation)
# beats the distance heuristic — 4x4: 0.92 vs 0.88 (p=4e-9); 5x5: 0.94 vs 0.92
# (p=4e-17), both robustly validated. This trains the SAME winning combo on an
# arbitrary size (3 seeds, clean), saving intermediate checkpoints for the
# early-stopping + ensemble protocol.
#
# Usage:   caffeinate -i bash scripts/run_irregular_scale.sh [ROWS] [COLS]
#   default 5 5;  for the 6x6 confirmation:  ... run_irregular_scale.sh 6 6
#
# After it finishes, Claude runs (tooling is size-general):
#   python scripts/early_stopping_eval.py     --rows R --cols C --runs results/irregular_scale/rRxC_s0 ...
#   python scripts/early_stopping_validate.py --rows R --cols C --runs results/irregular_scale/rRxC_s0 ...
#
# Long: 3 x 1.5M-step irregular RL (gflow verified per episode); 6x6 (36 qubits)
# is notably slower. Run overnight. Safe to re-run (skips finished).

set -e
cd "$(dirname "$0")/.."
PY=/usr/local/bin/python3.13
ROWS=${1:-5}
COLS=${2:-5}
LABEL=${3:-}   # optional label suffix, e.g. "fix" → r6x6fix_s0

COMBO="--topology irregular --observe-original --reward-shaping \
--defect-lo 0 --defect-hi 0 --hidden-dim 128 --n-heads 4 --n-layers 5 \
--virtual-node --pos-dim 8 --total-timesteps 1500000 --log-interval 25"

echo "════════════════════════════════════════════════════════"
echo "  IRREGULAR SCALE TEST — ${ROWS}x${COLS}${LABEL}, winning combo, 3 seeds — $(date)"
echo "════════════════════════════════════════════════════════"

for s in 0 1 2; do
    dir="results/irregular_scale/r${ROWS}x${COLS}${LABEL}_s$s"
    if [ -f "$dir/final.pt" ]; then echo ">>> SKIP $dir"
    else echo ">>> RUN  $dir"
         $PY scripts/train_gnn.py $COMBO --rows "$ROWS" --cols "$COLS" --seed $s --save-dir "$dir"; fi
done

echo "════════════════════════════════════════════════════════"
echo "  DONE $(date) — tell Claude to run the early-stopping + ensemble"
echo "  eval at ${ROWS}x${COLS} (validation/test split) vs the distance heuristic."
echo "════════════════════════════════════════════════════════"
