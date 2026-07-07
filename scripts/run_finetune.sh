#!/bin/bash
# Fine-tuning vs scratch — transfer-learning efficiency on 5×5.
#
# Question: can a 4×4-trained GNN reach good 5×5 performance with far fewer
# steps than training from scratch?
#
# Three points at a FIXED low budget (250k steps) on 5×5 angled:
#   1. GNN from scratch          (250k)            — low-budget baseline
#   2. GNN fine-tuned from 4×4   (250k, warm-start) — transfer learning
#   3. (reference) MLP from scratch needed 2.25M steps → 0.853
#
# If fine-tuning >> scratch at 250k, transfer learning gives a ~9× speed-up.
#
# Usage:  caffeinate -i bash scripts/run_finetune.sh
# Wall time: ~20 min total (2 × 250k steps on 5×5).

set -e
cd "$(dirname "$0")/.."
PY=/usr/local/bin/python3.13
mkdir -p results/seeds

run_if_missing () {
    local dir="$1"; shift
    if [ -f "$dir/final.pt" ]; then echo ">>> SKIP $dir (final.pt exists)"
    else echo ">>> RUN  $dir"; "$@"; fi
}

echo "════════════════════════════════════════════════════════"
echo "  FINE-TUNE vs SCRATCH (5×5 angled, 250k steps) — $(date)"
echo "════════════════════════════════════════════════════════"

# 1. From scratch
run_if_missing results/gnn_5x5_scratch250k \
    $PY scripts/train_gnn.py --rows 5 --cols 5 --use-angles \
        --hidden-dim 128 --n-heads 4 --n-layers 3 \
        --total-timesteps 250000 --seed 0 \
        --save-dir results/gnn_5x5_scratch250k --log-interval 20

# 2. Fine-tuned from the 4×4 angled checkpoint
run_if_missing results/gnn_5x5_finetune250k \
    $PY scripts/train_gnn.py --rows 5 --cols 5 --use-angles \
        --hidden-dim 128 --n-heads 4 --n-layers 3 \
        --init-from results/gnn_4x4_angled/final.pt \
        --total-timesteps 250000 --seed 0 \
        --save-dir results/gnn_5x5_finetune250k --log-interval 20

echo ""
echo "──────────── BENCHMARKS (500 trials @ 1% defect) ────────────"
echo ">>> GNN 5×5 from scratch (250k):"
$PY scripts/benchmark.py --checkpoint results/gnn_5x5_scratch250k/final.pt \
    --n-trials 500 --defect-rate 0.01 --save-dir results/benchmark_gnn \
    | grep -E "Mean reward|% Perfect"
echo ">>> GNN 5×5 fine-tuned from 4×4 (250k):"
$PY scripts/benchmark.py --checkpoint results/gnn_5x5_finetune250k/final.pt \
    --n-trials 500 --defect-rate 0.01 --save-dir results/benchmark_gnn \
    | grep -E "Mean reward|% Perfect"

echo ""
echo "  Reference: MLP 5×5 angled from scratch (2.25M steps) = 0.8526 ± 0.0122"
echo "  Reference: GNN 4×4→5×5 zero-shot (0 extra steps)     = 0.8371"
echo "════════════════════════════════════════════════════════"
echo "  Done $(date)"
echo "════════════════════════════════════════════════════════"
