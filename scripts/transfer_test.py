"""
Transfer test — evaluate one checkpoint across a sweep of defect rates.

Answers the dissertation's key empirical question:
  "Does training at 20–30% defects make the policy robust at realistic 1%?"

The test loads a single checkpoint (e.g. Stage 3, trained at 30% defects),
evaluates it at 0%, 1%, 5%, 10%, 20%, 30% defect rates, and compares it
against the GreedyGflowBaseline and RandomBaseline at each rate.

Outputs
-------
  results/transfer/transfer_<rows>x<cols>.json   — full per-rate stats
  results/transfer/transfer_<rows>x<cols>.png    — transfer curve plot

Usage
-----
# Full transfer test (500 episodes per rate, ~5 min)
python scripts/transfer_test.py \\
    --checkpoint results/3x3_curriculum/stage3_final.pt \\
    --n-episodes 500 \\
    --save-dir results/transfer

# Quick sanity check (50 episodes per rate)
python scripts/transfer_test.py \\
    --checkpoint results/3x3_curriculum/stage3_final.pt \\
    --n-episodes 50 \\
    --save-dir results/transfer_quick
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mbqc_rl.env.mbqc_env        import MBQCEnv
from mbqc_rl.agent.policy        import MBQCActorCritic
from mbqc_rl.agent.gnn_policy    import GNNActorCritic
from mbqc_rl.baselines.classical import GreedyGflowBaseline, RandomBaseline


def _load_policy(ckpt: dict, device: torch.device):
    """Instantiate and load the right policy class from a checkpoint config."""
    cfg = ckpt.get("config", {})
    if cfg.get("model_type") == "gnn":
        n = cfg.get("n", cfg.get("rows", 3) * cfg.get("cols", 3))
        policy = GNNActorCritic(n=n, hidden_dim=cfg.get("hidden_dim", 64),
                                n_heads=cfg.get("n_heads", 4),
                                n_layers=cfg.get("n_layers", 3),
                                use_angles=cfg.get("use_angles", False),
                                virtual_node=cfg.get("virtual_node", False),
                                weight_tied=cfg.get("weight_tied", False),
                                pos_dim=cfg.get("pos_dim", 0),
                                aux_layer_head=cfg.get("aux_layer_head", False))
    else:
        rows = cfg.get("rows", 3); cols = cfg.get("cols", 3)
        policy = MBQCActorCritic(obs_dim=cfg.get("obs_dim", rows*cols*(rows*cols+1)),
                                 n_actions=cfg.get("n_actions", rows*cols),
                                 hidden_dim=cfg.get("hidden_dim", 256))
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.to(device)
    policy.eval()
    return policy


# ── Default defect rate sweep ──────────────────────────────────────────────
DEFAULT_RATES = [0.00, 0.01, 0.05, 0.10, 0.20, 0.30]


# ---------------------------------------------------------------------------
# Episode runners (same as benchmark.py — small, self-contained)
# ---------------------------------------------------------------------------

def _run_agent(env, policy, seed, device):
    obs_dict, _ = env.reset(seed=seed)
    done = False
    total = 0.0
    while not done:
        obs_t  = torch.as_tensor(
            obs_dict["observation"], dtype=torch.float32, device=device
        ).unsqueeze(0)
        mask_t = torch.as_tensor(
            obs_dict["action_mask"], dtype=torch.float32, device=device
        ).unsqueeze(0)
        with torch.no_grad():
            logits, _ = policy(obs_t, mask_t)
            action = int(logits.argmax(dim=-1).item())
        obs_dict, reward, terminated, truncated, _ = env.step(action)
        total += reward
        done = terminated or truncated
    return total


def _run_greedy(env, seed):
    obs_dict, info = env.reset(seed=seed)
    baseline = GreedyGflowBaseline(env)
    baseline.reset(obs_dict, info)
    done = False
    total = 0.0
    while not done:
        action = baseline.select_action(obs_dict)
        obs_dict, reward, terminated, truncated, _ = env.step(action)
        total += reward
        done = terminated or truncated
    return total


def _run_random(env, seed, rng):
    obs_dict, info = env.reset(seed=seed)
    baseline = RandomBaseline(rng=rng)
    baseline.reset(obs_dict, info)
    done = False
    total = 0.0
    while not done:
        action = baseline.select_action(obs_dict)
        obs_dict, reward, terminated, truncated, _ = env.step(action)
        total += reward
        done = terminated or truncated
    return total


# ---------------------------------------------------------------------------
# One-rate evaluation
# ---------------------------------------------------------------------------

def evaluate_at_rate(
    defect_rate: float,
    rows: int, cols: int,
    policy: MBQCActorCritic,
    n_episodes: int,
    seed: int,
    device: torch.device,
    use_angles: bool = False,
    clifford_fraction: float = 0.5,
    topology: str = "grid",
) -> dict:
    """
    Run n_episodes at a fixed defect rate and return per-method statistics.
    """
    agent_r  = np.zeros(n_episodes)
    greedy_r = np.zeros(n_episodes)
    random_r = np.zeros(n_episodes)
    gflow_n  = 0

    env_kwargs = dict(rows=rows, cols=cols, defect_rate=defect_rate,
                      use_angles=use_angles, clifford_fraction=clifford_fraction,
                      topology=topology)
    env_a = MBQCEnv(**env_kwargs)
    env_g = MBQCEnv(**env_kwargs)
    env_r = MBQCEnv(**env_kwargs)
    rng   = np.random.default_rng(seed + 77777)

    for i in range(n_episodes):
        s = seed + i
        _, info = env_a.reset(seed=s)
        gflow_n += int(info["gflow_exists"])
        agent_r[i]  = _run_agent( env_a, policy, s, device)
        greedy_r[i] = _run_greedy(env_g, s)
        random_r[i] = _run_random(env_r, s, rng)

    def _s(arr, label):
        return {
            "label":       label,
            "mean":        float(arr.mean()),
            "std":         float(arr.std()),
            "pct_perfect": float((arr == 1.0).mean() * 100),
        }

    return {
        "defect_rate": defect_rate,
        "n_episodes":  n_episodes,
        "gflow_rate":  float(gflow_n / n_episodes * 100),
        "agent":       _s(agent_r,  "PPO agent"),
        "greedy":      _s(greedy_r, "Greedy"),
        "random":      _s(random_r, "Random"),
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _plot_transfer(all_results: list[dict], save_to: str, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not found — skipping plot")
        return

    rates  = [r["defect_rate"] for r in all_results]
    agent  = [r["agent"]["mean"]  for r in all_results]
    greedy = [r["greedy"]["mean"] for r in all_results]
    rand   = [r["random"]["mean"] for r in all_results]
    agent_err  = [r["agent"]["std"]  for r in all_results]
    greedy_err = [r["greedy"]["std"] for r in all_results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(rates, agent,  yerr=agent_err,  fmt="o-", color="steelblue",
                capsize=4, label="PPO agent (greedy)", lw=2)
    ax.errorbar(rates, greedy, yerr=greedy_err, fmt="s--", color="green",
                capsize=4, label="GreedyGflow (oracle)", lw=1.5, alpha=0.8)
    ax.plot(rates, rand, "^:", color="gray", label="Random baseline", lw=1.5, alpha=0.8)

    ax.set_xlabel("Defect rate", fontsize=12)
    ax.set_ylabel("Mean episode reward", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.axvline(0.01, ls=":", color="red", alpha=0.5, label="Realistic rate (1%)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_to, dpi=150, bbox_inches="tight")
    print(f"  Transfer curve saved → {save_to}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Transfer test across defect rates")
    p.add_argument("--checkpoint",  required=True, type=str)
    p.add_argument("--n-episodes",  type=int,   default=500,
                   help="Episodes per defect rate (default: 500)")
    p.add_argument("--defect-rates", nargs="+", type=float,
                   default=DEFAULT_RATES,
                   help="Space-separated list of defect rates to test")
    p.add_argument("--rows",        type=int,   default=None)
    p.add_argument("--cols",        type=int,   default=None)
    p.add_argument("--topology",    type=str,   default="grid",
                   choices=["grid", "brickwork"])
    p.add_argument("--seed",        type=int,   default=2000)
    p.add_argument("--save-dir",    type=str,   default="results/transfer")
    p.add_argument("--no-save",     action="store_true")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    t0     = time.time()
    device = torch.device("cpu")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg  = ckpt.get("config", {})

    rows       = args.rows or cfg.get("rows",       3)
    cols       = args.cols or cfg.get("cols",       3)
    use_angles = cfg.get("use_angles", False)
    cliff_frac = cfg.get("clifford_fraction", 0.5)

    policy = _load_policy(ckpt, device)

    print(f"\n{'═'*65}")
    print(f"  TRANSFER TEST: {rows}×{cols} {args.topology}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Rates: {args.defect_rates}")
    print(f"  Episodes per rate: {args.n_episodes}")
    print(f"{'═'*65}")
    print(f"\n  {'Rate':>6}  {'Agent':>8}  {'Greedy':>8}  {'Random':>8}  {'gflow%':>8}")
    print(f"  {'────':>6}  {'─────':>8}  {'──────':>8}  {'──────':>8}  {'──────':>8}")

    all_results = []
    for rate in args.defect_rates:
        res = evaluate_at_rate(
            defect_rate=rate,
            rows=rows, cols=cols,
            policy=policy,
            n_episodes=args.n_episodes,
            seed=args.seed,
            device=device,
            use_angles=use_angles,
            clifford_fraction=cliff_frac,
            topology=args.topology,
        )
        all_results.append(res)
        print(f"  {rate:>6.3f}  "
              f"{res['agent']['mean']:>8.4f}  "
              f"{res['greedy']['mean']:>8.4f}  "
              f"{res['random']['mean']:>8.4f}  "
              f"{res['gflow_rate']:>7.1f}%")

    print(f"\n  Completed in {time.time()-t0:.1f}s")

    if not args.no_save:
        os.makedirs(args.save_dir, exist_ok=True)

        suffix = f"{rows}x{cols}" + ("" if args.topology == "grid"
                                     else f"_{args.topology}")
        json_path = Path(args.save_dir) / f"transfer_{suffix}.json"
        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  Results saved → {json_path}")

        plot_path = Path(args.save_dir) / f"transfer_{suffix}.png"
        _plot_transfer(
            all_results,
            save_to=str(plot_path),
            title=f"Transfer Test — {rows}×{cols} {args.topology}",
        )


if __name__ == "__main__":
    main()
