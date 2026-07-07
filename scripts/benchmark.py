"""
1,000-grid benchmark — Phase 4 empirical evaluation.

For each trial the same random graph instance is shown to all three methods,
making the comparison paired (reduces variance caused by graph difficulty):

  1. PPO agent           — greedy decoding from checkpoint
  2. GreedyGflowBaseline — oracle; achieves reward=1.0 on every graph with gflow
  3. RandomBaseline      — uniform random from valid actions; lower bound

Statistical tests (scipy.stats.mannwhitneyu, two-sided):
  - Agent vs Greedy   → tests whether agent is significantly below/above the oracle
  - Agent vs Random   → tests whether agent meaningfully outperforms chance

Usage
-----
# Full benchmark (default: 1,000 trials, 1% defect, Stage-3 checkpoint)
python scripts/benchmark.py \\
    --checkpoint results/3x3_curriculum/stage3_final.pt \\
    --n-trials 1000 --defect-rate 0.01 \\
    --save-dir results/benchmark_1pct

# Multiple grid sizes in one run
python scripts/benchmark.py \\
    --checkpoint results/3x3_curriculum/stage3_final.pt \\
    --rows 3 --cols 3 --n-trials 1000 --defect-rate 0.01

# Quick sanity check
python scripts/benchmark.py \\
    --checkpoint results/3x3_curriculum/stage3_final.pt \\
    --n-trials 50 --defect-rate 0.01
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
from mbqc_rl.utils.metrics       import save_history


def _load_policy(ckpt: dict, device: torch.device):
    """Instantiate and load the right policy class from a checkpoint."""
    cfg        = ckpt.get("config", {})
    model_type = cfg.get("model_type", "mlp")

    if model_type == "gnn":
        n          = cfg.get("n", cfg.get("rows", 3) * cfg.get("cols", 3))
        hidden_dim = cfg.get("hidden_dim", 64)
        n_heads    = cfg.get("n_heads", 4)
        n_layers   = cfg.get("n_layers", 3)
        use_angles = cfg.get("use_angles", False)
        policy = GNNActorCritic(n=n, hidden_dim=hidden_dim, n_heads=n_heads,
                                n_layers=n_layers, use_angles=use_angles,
                                virtual_node=cfg.get("virtual_node", False),
                                weight_tied=cfg.get("weight_tied", False),
                                pos_dim=cfg.get("pos_dim", 0),
                                aux_layer_head=cfg.get("aux_layer_head", False))
    else:
        rows       = cfg.get("rows", 3)
        cols       = cfg.get("cols", 3)
        obs_dim    = cfg.get("obs_dim", rows * cols * (rows * cols + 1))
        n_actions  = cfg.get("n_actions", rows * cols)
        hidden_dim = cfg.get("hidden_dim", 256)
        policy = MBQCActorCritic(obs_dim=obs_dim, n_actions=n_actions,
                                 hidden_dim=hidden_dim)

    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.to(device)
    policy.eval()
    return policy


# ---------------------------------------------------------------------------
# Single-episode runners
# ---------------------------------------------------------------------------

def _run_agent(env: MBQCEnv, policy: MBQCActorCritic,
               seed: int, device: torch.device) -> float:
    """Run one episode with the trained policy (greedy/argmax decoding)."""
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
            action = int(logits.argmax(dim=-1).item())   # greedy
        obs_dict, reward, terminated, truncated, _ = env.step(action)
        total += reward
        done = terminated or truncated
    return total


def _run_greedy(env: MBQCEnv, seed: int) -> float:
    """Run one episode with the GreedyGflowBaseline."""
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


def _run_random(env: MBQCEnv, seed: int, rng: np.random.Generator) -> float:
    """Run one episode with the RandomBaseline."""
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
# Statistics
# ---------------------------------------------------------------------------

def _stats(rewards: np.ndarray, label: str) -> dict:
    return {
        "label":       label,
        "n":           int(len(rewards)),
        "mean":        float(rewards.mean()),
        "std":         float(rewards.std()),
        "median":      float(np.median(rewards)),
        "min":         float(rewards.min()),
        "max":         float(rewards.max()),
        "pct_perfect": float((rewards == 1.0).mean() * 100),
    }


def _mannwhitney(a: np.ndarray, b: np.ndarray) -> dict:
    """Two-sided Mann-Whitney U test (non-parametric; no normality assumed)."""
    from scipy.stats import mannwhitneyu
    stat, p = mannwhitneyu(a, b, alternative="two-sided")
    return {"U_statistic": float(stat), "p_value": float(p)}


def _print_table(results: dict) -> None:
    """Print a formatted comparison table."""
    agent   = results["agent"]
    greedy  = results["greedy"]
    random_ = results["random"]

    print(f"\n{'─'*65}")
    print(f"  {'Metric':<22}  {'PPO agent':>12}  {'Greedy':>12}  {'Random':>12}")
    print(f"  {'──────':<22}  {'─────────':>12}  {'──────':>12}  {'──────':>12}")

    def row(label, key, fmt=".4f"):
        print(f"  {label:<22}  "
              f"{agent[key]:>{12}{fmt}}  "
              f"{greedy[key]:>{12}{fmt}}  "
              f"{random_[key]:>{12}{fmt}}")

    row("Mean reward",    "mean")
    row("Std deviation",  "std")
    row("Median",         "median")
    row("Min",            "min")
    row("Max",            "max")
    row("% Perfect",      "pct_perfect", fmt=".1f")
    print(f"{'─'*65}")

    mw_ag = results["mann_whitney"]["agent_vs_greedy"]
    mw_ar = results["mann_whitney"]["agent_vs_random"]
    print(f"\n  Mann-Whitney U tests (two-sided):")
    print(f"    Agent vs Greedy : U={mw_ag['U_statistic']:.0f}  "
          f"p={mw_ag['p_value']:.4f}"
          + (" ***" if mw_ag['p_value'] < 0.001
             else " **" if mw_ag['p_value'] < 0.01
             else " *"  if mw_ag['p_value'] < 0.05
             else " (n.s.)"))
    print(f"    Agent vs Random : U={mw_ar['U_statistic']:.0f}  "
          f"p={mw_ar['p_value']:.4f}"
          + (" ***" if mw_ar['p_value'] < 0.001
             else " **" if mw_ar['p_value'] < 0.01
             else " *"  if mw_ar['p_value'] < 0.05
             else " (n.s.)"))
    print(f"\n  Graphs with valid gflow: {results['gflow_rate']:.1f}%")
    print(f"{'─'*65}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="1,000-grid benchmark — Phase 4")
    p.add_argument("--checkpoint",  required=True,  type=str)
    p.add_argument("--n-trials",    type=int,   default=1000)
    p.add_argument("--defect-rate", type=float, default=0.01,
                   help="Defect rate for evaluation (default: 0.01, i.e. realistic 1%%)")
    p.add_argument("--rows",        type=int,   default=None,
                   help="Override grid rows from checkpoint")
    p.add_argument("--cols",        type=int,   default=None,
                   help="Override grid cols from checkpoint")
    p.add_argument("--seed",        type=int,   default=1000,
                   help="Base seed — trial i uses seed+i")
    p.add_argument("--save-dir",    type=str,   default="results/benchmark",
                   help="Directory to save JSON results and table")
    p.add_argument("--no-save",     action="store_true",
                   help="Skip saving results to disk (just print)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0   = time.time()

    # ── Load checkpoint ────────────────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg  = ckpt.get("config", {})

    rows       = args.rows or cfg.get("rows",       3)
    cols       = args.cols or cfg.get("cols",       3)
    obs_dim    = cfg.get("obs_dim",    rows * cols * (rows * cols + 1))
    n_actions  = cfg.get("n_actions",  rows * cols)
    hidden_dim = cfg.get("hidden_dim", 256)
    use_angles = cfg.get("use_angles", False)
    cliff_frac = cfg.get("clifford_fraction", 0.5)

    device = torch.device("cpu")  # evaluation on CPU for reproducibility

    policy = _load_policy(ckpt, device)

    print(f"\n{'═'*65}")
    print(f"  BENCHMARK: {rows}×{cols} grid  defect_rate={args.defect_rate:.3f}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Trials: {args.n_trials}  |  seed offset: {args.seed}")
    print(f"{'═'*65}")

    # ── Run trials ─────────────────────────────────────────────────────────
    agent_rewards  = np.zeros(args.n_trials)
    greedy_rewards = np.zeros(args.n_trials)
    random_rewards = np.zeros(args.n_trials)
    gflow_exists   = np.zeros(args.n_trials, dtype=bool)

    # Three separate envs so each agent sees the same seed → same graph
    env_kwargs = dict(rows=rows, cols=cols, defect_rate=args.defect_rate,
                      use_angles=use_angles, clifford_fraction=cliff_frac,
                      observe_original_graph=cfg.get("observe_original", False))
    env_agent  = MBQCEnv(**env_kwargs)
    env_greedy = MBQCEnv(**env_kwargs)
    env_random = MBQCEnv(**env_kwargs)
    rng        = np.random.default_rng(args.seed + 9999)

    print_every = max(1, args.n_trials // 10)

    for i in range(args.n_trials):
        trial_seed = args.seed + i

        # Record whether this graph has a valid gflow
        _, info = env_agent.reset(seed=trial_seed)
        gflow_exists[i] = info["gflow_exists"]

        # Agent (re-reset with same seed)
        agent_rewards[i]  = _run_agent( env_agent,  policy, trial_seed, device)
        greedy_rewards[i] = _run_greedy(env_greedy, trial_seed)
        random_rewards[i] = _run_random(env_random, trial_seed, rng)

        if (i + 1) % print_every == 0:
            pct = (i + 1) / args.n_trials * 100
            print(f"  [{i+1:>5}/{args.n_trials}]  {pct:5.1f}%  "
                  f"agent={agent_rewards[:i+1].mean():.3f}  "
                  f"greedy={greedy_rewards[:i+1].mean():.3f}  "
                  f"random={random_rewards[:i+1].mean():.3f}")

    elapsed = time.time() - t0

    # ── Statistics ─────────────────────────────────────────────────────────
    results = {
        "config": {
            "checkpoint":  args.checkpoint,
            "rows":        rows,
            "cols":        cols,
            "defect_rate": args.defect_rate,
            "n_trials":    args.n_trials,
            "seed":        args.seed,
        },
        "agent":   _stats(agent_rewards,  "PPO agent (greedy)"),
        "greedy":  _stats(greedy_rewards, "GreedyGflowBaseline"),
        "random":  _stats(random_rewards, "RandomBaseline"),
        "mann_whitney": {
            "agent_vs_greedy": _mannwhitney(agent_rewards, greedy_rewards),
            "agent_vs_random": _mannwhitney(agent_rewards, random_rewards),
        },
        "gflow_rate": float(gflow_exists.mean() * 100),
        "elapsed_sec": round(elapsed, 1),
        "raw_rewards": {
            "agent":  agent_rewards.tolist(),
            "greedy": greedy_rewards.tolist(),
            "random": random_rewards.tolist(),
        },
    }

    print(f"\n  Completed in {elapsed:.1f}s")
    _print_table(results)

    # ── Save ───────────────────────────────────────────────────────────────
    if not args.no_save:
        os.makedirs(args.save_dir, exist_ok=True)
        fname = (f"benchmark_{rows}x{cols}_"
                 f"defect{int(args.defect_rate*100):02d}pct_"
                 f"n{args.n_trials}.json")
        out_path = Path(args.save_dir) / fname
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Results saved → {out_path}")

        # Also save a plain text summary table
        txt_path = out_path.with_suffix(".txt")
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _print_table(results)
        txt_path.write_text(buf.getvalue())


if __name__ == "__main__":
    main()
