"""
Robustness validation of the early-stopped-ensemble irregular win.

The headline: a 3-seed early-stopped GNN ensemble beats the distance heuristic on
irregular graphs (0.92 vs 0.89). This script stress-tests that claim:

  1. Checkpoints are selected on a VALIDATION seed range; we then evaluate the
     ensemble on SEVERAL INDEPENDENT TEST ranges (disjoint from each other and
     from validation). A real effect should hold on every test set.
  2. Bootstrap 95% CI on the per-graph (ensemble − heuristic) difference.
  3. Leave-one-out: all three 2-of-3 ensembles, to check the win is not driven by
     a single lucky seed.
  4. Selection stability: re-select on a second validation range.

All analysis on existing checkpoints — no training.

Usage
-----
python scripts/early_stopping_validate.py --val-trials 200 --test-trials 400 --stride 2
"""

from __future__ import annotations

import argparse
import glob
import itertools
import re
import os
import sys

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mbqc_rl.env.mbqc_env import MBQCEnv
from mbqc_rl.agent.gnn_policy import GNNActorCritic
from mbqc_rl.baselines.classical import DistanceScheduleBaseline

ROWS, COLS = 4, 4
USE_ANGLES = False          # set from --use-angles: non-Clifford (universal-MBQC) reward
CLIFFORD_FRAC = 0.5
RUNS = ["results/ladder/r13_virtual_rwse",
        "results/ladder/r13_virtual_rwse_s1",
        "results/ladder/r13_virtual_rwse_s2"]


def _arch():
    return dict(n=ROWS * COLS, hidden_dim=128, n_heads=4, n_layers=5,
                use_angles=USE_ANGLES, virtual_node=True, pos_dim=8)


def _policy(path):
    p = GNNActorCritic(**_arch())
    p.load_state_dict(torch.load(path, map_location="cpu",
                                 weights_only=False)["policy_state_dict"])
    p.eval()
    return p


def _rewards(policies, seed0, n):
    if not isinstance(policies, (list, tuple)):
        policies = [policies]
    env = MBQCEnv(rows=ROWS, cols=COLS, defect_rate=0.0, topology="irregular",
                  observe_original_graph=True,
                  use_angles=USE_ANGLES, clifford_fraction=CLIFFORD_FRAC)
    out = np.zeros(n)
    for i in range(n):
        obs, _ = env.reset(seed=seed0 + i)
        done = False
        while not done:
            o = torch.as_tensor(obs["observation"], dtype=torch.float32).unsqueeze(0)
            m = torch.as_tensor(obs["action_mask"],  dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                lg = sum(p(o, m)[0] for p in policies) / len(policies)
            obs, r, t, tr, _ = env.step(int(lg.argmax(-1)))
            out[i] += r; done = t or tr
    return out


def _heur_rewards(seed0, n):
    env = MBQCEnv(rows=ROWS, cols=COLS, defect_rate=0.0, topology="irregular",
                  use_angles=USE_ANGLES, clifford_fraction=CLIFFORD_FRAC)
    out = np.zeros(n)
    for i in range(n):
        obs, info = env.reset(seed=seed0 + i)
        b = DistanceScheduleBaseline(env); b.reset(obs, info)
        done = False
        while not done:
            obs, r, t, tr, _ = env.step(b.select_action(obs))
            out[i] += r; done = t or tr
    return out


def select(run, val_seed0, val_trials, stride):
    cs = sorted(glob.glob(f"{run}/checkpoints/*.pt"),
                key=lambda p: int(re.search(r"(\d+)", p.split("/")[-1]).group(1)))[::stride]
    best_v, best_c, best_u = -1, None, None
    for c in cs:
        v = _rewards(_policy(c), val_seed0, val_trials).mean()
        if v > best_v:
            best_v, best_c = v, c
            best_u = int(re.search(r"(\d+)", c.split("/")[-1]).group(1))
    return best_c, best_u


def main():
    global ROWS, COLS, RUNS, USE_ANGLES, CLIFFORD_FRAC
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-trials", type=int, default=200)
    ap.add_argument("--test-trials", type=int, default=400)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--use-angles", action="store_true",
                    help="Score on the non-Clifford (universal-MBQC) angled reward.")
    ap.add_argument("--clifford-fraction", type=float, default=0.5)
    ap.add_argument("--runs", nargs="+", default=RUNS)
    args = ap.parse_args()
    ROWS, COLS, RUNS = args.rows, args.cols, args.runs
    USE_ANGLES, CLIFFORD_FRAC = args.use_angles, args.clifford_fraction

    print(f"\n{'='*70}\n  ROBUSTNESS VALIDATION — early-stopped ensemble (irregular)\n{'='*70}")

    # 1. Select on validation seeds 50k
    sel = [select(r, 50_000, args.val_trials, args.stride) for r in RUNS]
    print("  validation-selected checkpoints (val seeds 50k):")
    for r, (c, u) in zip(RUNS, sel):
        print(f"    {r.split('/')[-1]:22s} @ update {u}")
    models = [_policy(c) for c, _ in sel]

    # 2. Multiple INDEPENDENT test sets
    print(f"\n  Ensemble vs heuristic on independent test sets ({args.test_trials} graphs):")
    print(f"  {'test seed0':>11} {'ensemble':>9} {'heuristic':>10} {'Δ':>8} {'p (MW)':>9}")
    deltas = []
    for ts0 in [90_000, 200_000, 300_000, 400_000, 500_000]:
        e = _rewards(models, ts0, args.test_trials)
        h = _heur_rewards(ts0, args.test_trials)
        _, p = stats.mannwhitneyu(e, h, alternative="greater")
        deltas.append(e.mean() - h.mean())
        flag = "✓" if (e.mean() > h.mean() and p < 0.05) else "✗"
        print(f"  {ts0:>11} {e.mean():>9.4f} {h.mean():>10.4f} "
              f"{e.mean()-h.mean():>+8.4f} {p:>9.1e} {flag}")
    print(f"  → ensemble beats heuristic on {sum(d>0 for d in deltas)}/5 "
          f"independent test sets; mean Δ = {np.mean(deltas):+.4f}")

    # 3. Bootstrap 95% CI on per-graph difference (pooled large test set)
    e = _rewards(models, 90_000, 1000)
    h = _heur_rewards(90_000, 1000)
    diff = e - h
    boot = [np.mean(diff[np.random.default_rng(b).integers(0, len(diff), len(diff))])
            for b in range(2000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"\n  Bootstrap (1000 graphs, 2000 resamples): "
          f"ensemble−heuristic = {diff.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"    {'CI excludes 0 → significant' if lo > 0 else 'CI includes 0'}")

    # 4. Leave-one-out: each pair of seeds, on the main test set
    heur90 = _heur_rewards(90_000, args.test_trials).mean()
    print(f"\n  Leave-one-out (2-of-3 ensembles, test 90k; heuristic={heur90:.4f}):")
    for combo in itertools.combinations(range(3), 2):
        sub = [models[i] for i in combo]
        em = _rewards(sub, 90_000, args.test_trials).mean()
        print(f"    seeds {combo}: {em:.4f}  ({'> ' if em>heur90 else '≤ '}heuristic)")

    # 5. Selection stability — re-select on a different validation range
    sel2 = [select(r, 60_000, args.val_trials, args.stride) for r in RUNS]
    same = sum(c1 == c2 for (c1, _), (c2, _) in zip(sel, sel2))
    print(f"\n  Selection stability (val 50k vs 60k): {same}/3 checkpoints identical; "
          f"updates {[u for _,u in sel]} vs {[u for _,u in sel2]}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
