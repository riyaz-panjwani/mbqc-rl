#!/bin/bash
# 6x6 irregular — early-stopping eval + full robustness validation, in one shot.
#
# Runs on the FIXED 6x6 runs (r6x6fix_s*), trained after the generator's
# _max_tries was raised to 400 so ~99.5% of 6x6 irregular graphs have a valid
# gflow (the old r6x6_s* runs had 31% gflow-less episodes -> heuristic/oracle
# were only 0.57/0.61; the fixed runs give 0.93/0.99, the healthy regime).
#
# Confirms whether the early-stopped 3-seed ensemble beats the distance
# heuristic at a THIRD size (after 4x4: 0.92 vs 0.88 and 5x5: 0.94 vs 0.92).
#
# Usage:   caffeinate -i bash scripts/run_6x6_eval.sh
#   (slower than 4x4/5x5 — 36 qubits, ~20 checkpoints/seed. Let it run.)

set -e
cd "$(dirname "$0")/.."
PY=/usr/local/bin/python3.13

RUNS="results/irregular_scale/r6x6fix_s0 \
results/irregular_scale/r6x6fix_s1 \
results/irregular_scale/r6x6fix_s2"

echo "════════════════════════════════════════════════════════"
echo "  6x6 EARLY-STOPPING EVAL  — $(date)"
echo "════════════════════════════════════════════════════════"
$PY scripts/early_stopping_eval.py \
    --rows 6 --cols 6 --val-trials 150 --test-trials 300 --stride 3 \
    --runs $RUNS

echo
echo "════════════════════════════════════════════════════════"
echo "  6x6 ROBUSTNESS VALIDATION  — $(date)"
echo "════════════════════════════════════════════════════════"
$PY scripts/early_stopping_validate.py \
    --rows 6 --cols 6 --val-trials 150 --test-trials 300 --stride 3 \
    --runs $RUNS

echo
echo "════════════════════════════════════════════════════════"
echo "  DONE $(date) — paste both blocks back to Claude."
echo "════════════════════════════════════════════════════════"
