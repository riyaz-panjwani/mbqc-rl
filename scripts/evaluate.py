"""
Evaluate a saved policy against the environment.

Usage
-----
# Evaluate a checkpoint on its training grid
python scripts/evaluate.py --checkpoint checkpoints/policy_final.pt --n-episodes 200

# Evaluate on a different defect rate (transfer test)
python scripts/evaluate.py --checkpoint checkpoints/policy_final.pt \\
    --defect-rate 0.01 --n-episodes 1000

# Compare against a random baseline
python scripts/evaluate.py --checkpoint checkpoints/policy_final.pt \\
    --n-episodes 500 --baseline
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mbqc_rl.env.mbqc_env import MBQCEnv
from mbqc_rl.agent.policy import MBQCActorCritic


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained MBQC PPO policy")
    p.add_argument("--checkpoint",   required=True,  type=str)
    p.add_argument("--n-episodes",   type=int,   default=500)
    p.add_argument("--defect-rate",  type=float, default=None,
                   help="Override defect rate from checkpoint (default: use training rate)")
    p.add_argument("--seed",         type=int,   default=0)
    p.add_argument("--baseline",     action="store_true",
                   help="Also evaluate a random valid-action baseline")
    p.add_argument("--greedy",       action="store_true",
                   help="Use argmax policy (greedy) instead of sampling")
    return p.parse_args()


def run_episodes(
    env: MBQCEnv,
    policy: MBQCActorCritic | None,
    n_episodes: int,
    greedy: bool = False,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """
    Run n_episodes and return statistics.
    If policy is None, uses a uniform random valid-action baseline.
    """
    rewards = []
    gflow_exists_count = 0

    for _ in range(n_episodes):
        obs_dict, info = env.reset()
        if info["gflow_exists"]:
            gflow_exists_count += 1

        done = False
        ep_reward = 0.0

        while not done:
            mask = obs_dict["action_mask"]

            if policy is None:
                # Random baseline: pick uniformly from valid actions
                valid = np.where(mask == 1)[0]
                action = int(np.random.choice(valid))
            else:
                obs_t  = torch.as_tensor(obs_dict["observation"], dtype=torch.float32, device=device).unsqueeze(0)
                mask_t = torch.as_tensor(mask,                    dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    logits, _ = policy(obs_t, mask_t)
                    if greedy:
                        action = int(logits.argmax(dim=-1).item())
                    else:
                        from torch.distributions import Categorical
                        action = int(Categorical(logits=logits).sample().item())

            obs_dict, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated

        rewards.append(ep_reward)

    rewards = np.array(rewards)
    return {
        "mean":             float(rewards.mean()),
        "std":              float(rewards.std()),
        "median":           float(np.median(rewards)),
        "min":              float(rewards.min()),
        "max":              float(rewards.max()),
        "pct_perfect":      float((rewards == 1.0).mean() * 100),
        "gflow_rate":       float(gflow_exists_count / n_episodes * 100),
    }


def print_stats(label: str, stats: dict) -> None:
    print(f"\n  {label}")
    print(f"    Mean reward   : {stats['mean']:.4f} ± {stats['std']:.4f}")
    print(f"    Median        : {stats['median']:.4f}")
    print(f"    Min / Max     : {stats['min']:.4f} / {stats['max']:.4f}")
    print(f"    Perfect (1.0) : {stats['pct_perfect']:.1f}%")
    print(f"    Graphs w/ gflow: {stats['gflow_rate']:.1f}%")


def main() -> None:
    args = parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg  = ckpt.get("config", {})
    saved_args = ckpt.get("args", {})

    rows        = cfg.get("rows",        saved_args.get("rows",        3))
    cols        = cfg.get("cols",        saved_args.get("cols",        3))
    defect_rate = args.defect_rate if args.defect_rate is not None \
                  else cfg.get("defect_rate", saved_args.get("defect_rate", 0.0))
    obs_dim     = cfg.get("obs_dim",     rows * cols * (rows * cols + 1))
    n_actions   = cfg.get("n_actions",   rows * cols)
    hidden_dim  = cfg.get("hidden_dim",  saved_args.get("hidden_dim",  256))

    device = torch.device("cpu")   # evaluation on CPU for reproducibility

    policy = MBQCActorCritic(obs_dim=obs_dim, n_actions=n_actions, hidden_dim=hidden_dim)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.to(device)
    policy.eval()

    env = MBQCEnv(rows=rows, cols=cols, defect_rate=defect_rate, seed=args.seed)
    np.random.seed(args.seed)

    print(f"\n{'='*55}")
    print(f"  Evaluating: {args.checkpoint}")
    print(f"  Grid: {rows}×{cols}  defect_rate={defect_rate:.3f}")
    print(f"  Episodes: {args.n_episodes}  greedy={args.greedy}")
    print(f"{'='*55}")

    policy_stats = run_episodes(
        env, policy, args.n_episodes,
        greedy=args.greedy, device=device,
    )
    print_stats("Trained policy", policy_stats)

    if args.baseline:
        env2 = MBQCEnv(rows=rows, cols=cols, defect_rate=defect_rate, seed=args.seed)
        np.random.seed(args.seed)
        baseline_stats = run_episodes(env2, None, args.n_episodes)
        print_stats("Random baseline", baseline_stats)

        improvement = policy_stats["mean"] - baseline_stats["mean"]
        print(f"\n  Improvement over random: {improvement:+.4f}")

    print()


if __name__ == "__main__":
    main()
