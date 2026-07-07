"""
Early-stopping / checkpoint-selection for the irregular regime — done rigorously.

The winning ladder combo (virtual + RWSE) overfits: training reward → 0.98 while
held-out falls to ~0.82. Intermediate checkpoints peak much earlier. This script
tests whether EARLY STOPPING recovers a policy that beats the 0.885 distance
heuristic — without selection bias:

  • VALIDATION seeds  → pick the best checkpoint per training run.
  • TEST seeds (disjoint) → report that checkpoint's score.

Done for all 3 seeds; compared to the heuristic on the same TEST set. This is the
honest way to claim "early stopping beats the heuristic" (no peeking at test).

Usage
-----
python scripts/early_stopping_eval.py --val-trials 200 --test-trials 500 --stride 2
"""

from __future__ import annotations

import argparse
import glob
import re
import sys
import os

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mbqc_rl.env.mbqc_env import MBQCEnv
from mbqc_rl.agent.gnn_policy import GNNActorCritic
from mbqc_rl.baselines.classical import DistanceScheduleBaseline, GreedyGflowBaseline

# Winning-combo architecture; n is set per --rows/--cols in main(). Size-agnostic.
ROWS, COLS = 4, 4
VAL_SEED0, TEST_SEED0 = 50_000, 90_000
USE_ANGLES = False          # set from --use-angles: non-Clifford (universal-MBQC) reward
CLIFFORD_FRAC = 0.5


def _arch():
    return dict(n=ROWS * COLS, hidden_dim=128, n_heads=4, n_layers=5,
                use_angles=USE_ANGLES, virtual_node=True, pos_dim=8)


def _policy(state_dict):
    p = GNNActorCritic(**_arch())
    p.load_state_dict(state_dict)
    p.eval()
    return p


def _agent_rewards(policies, seed0, n):
    """Per-graph rewards for a single policy or an ENSEMBLE (averaged logits)."""
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


def _agent_score(policy, seed0, n):
    return _agent_rewards(policy, seed0, n).mean()


def _baseline_score(cls, seed0, n):
    env = MBQCEnv(rows=ROWS, cols=COLS, defect_rate=0.0, topology="irregular",
                  use_angles=USE_ANGLES, clifford_fraction=CLIFFORD_FRAC)
    tot = 0.0
    for i in range(n):
        obs, info = env.reset(seed=seed0 + i)
        b = cls(env); b.reset(obs, info)
        done = False
        while not done:
            obs, r, t, tr, _ = env.step(b.select_action(obs))
            tot += r; done = t or tr
    return tot / n


def _ckpts(run_dir, stride):
    cs = sorted(glob.glob(f"{run_dir}/checkpoints/*.pt"),
                key=lambda p: int(re.search(r"(\d+)", p.split("/")[-1]).group(1)))
    return cs[::stride] + ([cs[-1]] if cs and cs[-1] not in cs[::stride] else [])


def main():
    global ROWS, COLS, USE_ANGLES, CLIFFORD_FRAC
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-trials", type=int, default=200)
    ap.add_argument("--test-trials", type=int, default=500)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--use-angles", action="store_true",
                    help="Score on the non-Clifford (universal-MBQC) angled reward.")
    ap.add_argument("--clifford-fraction", type=float, default=0.5)
    ap.add_argument("--runs", nargs="+", default=[
        "results/ladder/r13_virtual_rwse",
        "results/ladder/r13_virtual_rwse_s1",
        "results/ladder/r13_virtual_rwse_s2"],
        help="Run dirs (each with checkpoints/ and final.pt) to ensemble.")
    args = ap.parse_args()
    ROWS, COLS = args.rows, args.cols
    USE_ANGLES, CLIFFORD_FRAC = args.use_angles, args.clifford_fraction
    runs = args.runs

    heur_test = _baseline_score(DistanceScheduleBaseline, TEST_SEED0, args.test_trials)
    orc_test  = _baseline_score(GreedyGflowBaseline,      TEST_SEED0, args.test_trials)
    print(f"\n{'='*70}")
    print(f"  EARLY-STOPPING (validation-selected) — irregular, TEST set")
    print(f"  distance heuristic (test) = {heur_test:.4f}   oracle = {orc_test:.4f}")
    print(f"{'='*70}")

    test_scores, final_scores, selected = [], [], []
    for run in runs:
        ckpts = _ckpts(run, args.stride)
        if not ckpts:
            print(f"  {run}: no checkpoints"); continue
        # pick best on VALIDATION
        best_val, best_ck, best_upd = -1.0, None, None
        for c in ckpts:
            sd = torch.load(c, map_location="cpu", weights_only=False)["policy_state_dict"]
            v = _agent_score(_policy(sd), VAL_SEED0, args.val_trials)
            if v > best_val:
                best_val, best_ck = v, c
                best_upd = int(re.search(r"(\d+)", c.split("/")[-1]).group(1))
        # report selected checkpoint on TEST
        sd = torch.load(best_ck, map_location="cpu", weights_only=False)["policy_state_dict"]
        t = _agent_score(_policy(sd), TEST_SEED0, args.test_trials)
        selected.append(_policy(sd))
        # also the final checkpoint on TEST (for the overfitting delta)
        fsd = torch.load(f"{run}/final.pt", map_location="cpu", weights_only=False)["policy_state_dict"]
        ft = _agent_score(_policy(fsd), TEST_SEED0, args.test_trials)
        test_scores.append(t); final_scores.append(ft)
        tag = " > heuristic" if t > heur_test else ""
        print(f"  {run.split('/')[-1]:22s} val-best@upd{best_upd:<5} "
              f"test={t:.4f}{tag}   (final={ft:.4f})")

    ts = np.array(test_scores); fs = np.array(final_scores)
    print(f"\n  {'-'*60}")
    print(f"  final-ckpt    (3 seeds): {fs.mean():.4f} ± {fs.std():.4f}")
    print(f"  early-stopped (3 seeds): {ts.mean():.4f} ± {ts.std():.4f}  "
          f"(beats heuristic in {int((ts>heur_test).sum())}/3)")

    # ENSEMBLE of the validation-selected checkpoints, evaluated on TEST.
    ens = _agent_rewards(selected, TEST_SEED0, args.test_trials)
    heur_r = np.array([_baseline_score(DistanceScheduleBaseline, TEST_SEED0 + i, 1)
                       for i in range(args.test_trials)])
    u, p = stats.mannwhitneyu(ens, heur_r, alternative="greater")
    print(f"  ENSEMBLE      (early-stopped): {ens.mean():.4f}  "
          f"Δ={ens.mean()-heur_test:+.4f}  Mann-Whitney p={p:.4g}")
    print(f"  distance heuristic           : {heur_test:.4f}")
    verdict = ("ENSEMBLE BEATS the heuristic (significant)" if p < 0.05
               else "ensemble ≈ heuristic")
    print(f"  => {verdict}")
    print(f"  {'-'*60}\n")


if __name__ == "__main__":
    main()
