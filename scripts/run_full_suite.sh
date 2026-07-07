#!/bin/bash
# Full 18-run experiment suite — 3 grids × 2 angle modes × 3 seeds.
#
# Grid   | plain (no angles) | angled (use_angles)
# -------+-------------------+--------------------
# 3×3    | s0 s1 s2          | s0 s1 s2  ← already done (symlinked)
# 4×4    | s0 s1 s2  ← done | s0 s1 s2
# 5×5    | s0 s1 s2          | s0 s1 s2
#
# 11 new runs, 7 skipped via symlinks.  Estimated wall time: 8–9 h.
#
# Usage:  caffeinate -i bash scripts/run_full_suite.sh
# Safe to re-run: completed runs are skipped (checks for final.pt).

set -e
cd "$(dirname "$0")/.."
PY=/usr/local/bin/python3.13
mkdir -p results/seeds

# ── Symlink already-completed runs so naming is consistent ────────────
# 3×3 angled: 3x3_angled_mixed_s* → 3x3_angled_s*
# 4×4 plain:  4x4_mixed_s*        → 4x4_plain_s*
(
  cd results
  for s in 0 1 2; do
    [ -e "3x3_angled_s$s" ] || ln -sfn "3x3_angled_mixed_s$s" "3x3_angled_s$s"
    [ -e "4x4_plain_s$s"  ] || ln -sfn "4x4_mixed_s$s"        "4x4_plain_s$s"
  done
)

run_if_missing () {
    local dir="$1"; shift
    if [ -f "$dir/final.pt" ]; then
        echo ">>> SKIP $dir (final.pt exists)"
    else
        echo ">>> RUN  $dir"
        "$@"
    fi
}

echo "════════════════════════════════════════════════════════"
echo "  FULL SUITE (18 runs) — started $(date)"
echo "════════════════════════════════════════════════════════"

# ── 3×3 plain (no angles) ─────────────────────────────────────────────
for s in 0 1 2; do
    run_if_missing results/3x3_plain_s$s \
        $PY scripts/train_mixed.py --rows 3 --cols 3 \
            --total-timesteps 750000 --seed $s \
            --save-dir results/3x3_plain_s$s --log-interval 20
done

# ── 3×3 angled — SKIPPED via symlinks (already done) ─────────────────
for s in 0 1 2; do
    run_if_missing results/3x3_angled_s$s \
        $PY scripts/train_mixed.py --rows 3 --cols 3 --use-angles \
            --total-timesteps 750000 --seed $s \
            --save-dir results/3x3_angled_s$s --log-interval 20
done

# ── 4×4 plain — SKIPPED via symlinks (already done) ──────────────────
for s in 0 1 2; do
    run_if_missing results/4x4_plain_s$s \
        $PY scripts/train_mixed.py --rows 4 --cols 4 \
            --total-timesteps 1500000 --seed $s \
            --save-dir results/4x4_plain_s$s --log-interval 20
done

# ── 4×4 angled ────────────────────────────────────────────────────────
for s in 0 1 2; do
    run_if_missing results/4x4_angled_s$s \
        $PY scripts/train_mixed.py --rows 4 --cols 4 --use-angles \
            --total-timesteps 1500000 --seed $s \
            --save-dir results/4x4_angled_s$s --log-interval 20
done

# ── 5×5 plain ─────────────────────────────────────────────────────────
for s in 0 1 2; do
    run_if_missing results/5x5_plain_s$s \
        $PY scripts/train_mixed.py --rows 5 --cols 5 \
            --total-timesteps 2250000 --hidden-dim 512 --n-steps 1024 \
            --seed $s --save-dir results/5x5_plain_s$s --log-interval 20
done

# ── 5×5 angled ────────────────────────────────────────────────────────
for s in 0 1 2; do
    run_if_missing results/5x5_angled_s$s \
        $PY scripts/train_mixed.py --rows 5 --cols 5 --use-angles \
            --total-timesteps 2250000 --hidden-dim 512 --n-steps 1024 \
            --seed $s --save-dir results/5x5_angled_s$s --log-interval 20
done

echo ""
echo "════════════════════════════════════════════════════════"
echo "  ALL TRAINING DONE — aggregating results"
echo "════════════════════════════════════════════════════════"

# ── Aggregate each group (500 trials at 1% defects) ──────────────────
$PY scripts/aggregate_seeds.py \
    --pattern "results/3x3_plain_s*/final.pt" \
    --n-trials 500 --defect-rate 0.01 \
    --save results/seeds/3x3_plain.json

$PY scripts/aggregate_seeds.py \
    --pattern "results/3x3_angled_s*/final.pt" \
    --n-trials 500 --defect-rate 0.01 \
    --save results/seeds/3x3_angled.json

$PY scripts/aggregate_seeds.py \
    --pattern "results/4x4_plain_s*/final.pt" \
    --n-trials 500 --defect-rate 0.01 \
    --save results/seeds/4x4_plain.json

$PY scripts/aggregate_seeds.py \
    --pattern "results/4x4_angled_s*/final.pt" \
    --n-trials 500 --defect-rate 0.01 \
    --save results/seeds/4x4_angled.json

$PY scripts/aggregate_seeds.py \
    --pattern "results/5x5_plain_s*/final.pt" \
    --n-trials 500 --defect-rate 0.01 \
    --save results/seeds/5x5_plain.json

$PY scripts/aggregate_seeds.py \
    --pattern "results/5x5_angled_s*/final.pt" \
    --n-trials 500 --defect-rate 0.01 \
    --save results/seeds/5x5_angled.json

echo ""
echo "════════════════════════════════════════════════════════"
echo "  FULL SUITE — finished $(date)"
echo "  Results in results/seeds/*.json"
echo "════════════════════════════════════════════════════════"
