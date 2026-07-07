"""
Multi-seed aggregation — statistical rigor for the paper.

Benchmarks every seed checkpoint matching a glob pattern at a given defect
rate, then reports mean ± std ACROSS seeds (the number reviewers want),
plus a per-seed breakdown.

Usage
-----
# After training results/4x4_mixed_s0/final.pt, ..._s1/final.pt, ...:
python scripts/aggregate_seeds.py \\
    --pattern "results/4x4_mixed_s*/final.pt" \\
    --n-trials 1000 --defect-rate 0.01 \\
    --save results/seeds/4x4_mixed_seeds.json

# Works with staged curriculum checkpoints too:
python scripts/aggregate_seeds.py \\
    --pattern "results/3x3_full_s*/stage3_final.pt" \\
    --n-trials 1000 --defect-rate 0.01
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mbqc_rl.env.mbqc_env        import MBQCEnv
from mbqc_rl.agent.policy        import MBQCActorCritic
from mbqc_rl.agent.gnn_policy    import GNNActorCritic
from mbqc_rl.baselines.classical import GreedyGflowBaseline, RandomBaseline


def _run_agent(env, policy, seed, device):
    obs_dict, _ = env.reset(seed=seed)
    done, total = False, 0.0
    while not done:
        obs_t  = torch.as_tensor(obs_dict["observation"], dtype=torch.float32,
                                 device=device).unsqueeze(0)
        mask_t = torch.as_tensor(obs_dict["action_mask"], dtype=torch.float32,
                                 device=device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = policy(obs_t, mask_t)
            action = int(logits.argmax(dim=-1).item())
        obs_dict, r, terminated, truncated, _ = env.step(action)
        total += r
        done = terminated or truncated
    return total


def benchmark_checkpoint(ckpt_path: str, n_trials: int, defect_rate: float,
                         base_seed: int) -> dict:
    """Benchmark one checkpoint; returns its stats."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg  = ckpt.get("config", {})

    rows       = cfg.get("rows", 3)
    cols       = cfg.get("cols", 3)
    use_angles = cfg.get("use_angles", False)
    cliff_frac = cfg.get("clifford_fraction", 0.5)
    device     = torch.device("cpu")

    if cfg.get("model_type") == "gnn":
        n       = cfg.get("n", rows * cols)
        policy  = GNNActorCritic(n=n, hidden_dim=cfg.get("hidden_dim", 64),
                                 n_heads=cfg.get("n_heads", 4),
                                 n_layers=cfg.get("n_layers", 3),
                                 use_angles=use_angles,
                                 virtual_node=cfg.get("virtual_node", False),
                                 weight_tied=cfg.get("weight_tied", False),
                                 pos_dim=cfg.get("pos_dim", 0),
                                 aux_layer_head=cfg.get("aux_layer_head", False))
    else:
        obs_dim    = cfg.get("obs_dim", rows * cols * (rows * cols + 1))
        n_actions  = cfg.get("n_actions", rows * cols)
        hidden_dim = cfg.get("hidden_dim", 256)
        policy = MBQCActorCritic(obs_dim=obs_dim, n_actions=n_actions,
                                 hidden_dim=hidden_dim)

    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()

    env = MBQCEnv(rows=rows, cols=cols, defect_rate=defect_rate,
                  use_angles=use_angles, clifford_fraction=cliff_frac)

    rewards = np.zeros(n_trials)
    for i in range(n_trials):
        rewards[i] = _run_agent(env, policy, base_seed + i, device)

    return {
        "checkpoint":  ckpt_path,
        "rows":        rows,
        "cols":        cols,
        "use_angles":  use_angles,
        "mean":        float(rewards.mean()),
        "std":         float(rewards.std()),
        "pct_perfect": float((rewards == 1.0).mean() * 100),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Aggregate benchmark across seeds")
    p.add_argument("--pattern",     required=True, type=str,
                   help='Glob for checkpoints, e.g. "results/4x4_mixed_s*/final.pt"')
    p.add_argument("--n-trials",    type=int,   default=1000)
    p.add_argument("--defect-rate", type=float, default=0.01)
    p.add_argument("--seed",        type=int,   default=1000)
    p.add_argument("--save",        type=str,   default=None)
    args = p.parse_args()

    paths = sorted(glob.glob(args.pattern))
    if not paths:
        print(f"No checkpoints match: {args.pattern}")
        sys.exit(1)

    print(f"\n{'═'*60}")
    print(f"  MULTI-SEED AGGREGATION  ({len(paths)} checkpoints)")
    print(f"  defect_rate={args.defect_rate}  trials/seed={args.n_trials}")
    print(f"{'═'*60}\n")

    per_seed = []
    for path in paths:
        stats = benchmark_checkpoint(path, args.n_trials, args.defect_rate,
                                     args.seed)
        per_seed.append(stats)
        print(f"  {path}")
        print(f"    mean={stats['mean']:.4f}  std={stats['std']:.4f}  "
              f"perfect={stats['pct_perfect']:.1f}%")

    means = np.array([s["mean"] for s in per_seed])
    pcts  = np.array([s["pct_perfect"] for s in per_seed])

    print(f"\n{'─'*60}")
    print(f"  ACROSS {len(paths)} SEEDS:")
    print(f"    Mean reward : {means.mean():.4f} ± {means.std():.4f}")
    print(f"    % Perfect   : {pcts.mean():.1f}% ± {pcts.std():.1f}%")
    print(f"    Min / Max   : {means.min():.4f} / {means.max():.4f}")
    print(f"{'─'*60}\n")

    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save, "w") as f:
            json.dump({
                "pattern":      args.pattern,
                "defect_rate":  args.defect_rate,
                "n_trials":     args.n_trials,
                "per_seed":     per_seed,
                "across_seeds": {
                    "mean_of_means": float(means.mean()),
                    "std_of_means":  float(means.std()),
                    "mean_pct_perfect": float(pcts.mean()),
                    "std_pct_perfect":  float(pcts.std()),
                },
            }, f, indent=2)
        print(f"  Saved → {args.save}")


if __name__ == "__main__":
    main()
