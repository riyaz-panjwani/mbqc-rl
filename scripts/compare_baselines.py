"""
Four-way comparison — where does the agent sit between human and oracle?

The benchmark.py table reports the oracle (ceiling) and random (floor). This
script adds the missing middle reference: a non-adaptive *human / textbook*
schedule (StaticScheduleBaseline) — the fixed column-sweep order a person would
use on a grid resource state. The gap between that static baseline and the RL
agent is the *value of adaptation*: both are equal on perfect graphs, but only
the agent can react to edge defects.

Methods (per defect rate):
    oracle  — GreedyGflow, recomputes gflow per instance (upper bound)
    agent   — trained policy (GNN or MLP, auto-detected)
    human   — StaticSchedule, fixed geometric order, never sees the gflow
    random  — uniform valid action (lower bound)

Outputs a table + a curve figure (results/figures/R7_human_baseline.png).

Usage
-----
python scripts/compare_baselines.py \\
    --checkpoint results/gnn_4x4_angled/final.pt \\
    --rows 4 --cols 4 --n-trials 400
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mbqc_rl.env.mbqc_env     import MBQCEnv
from mbqc_rl.agent.policy     import MBQCActorCritic
from mbqc_rl.agent.gnn_policy import GNNActorCritic
from mbqc_rl.baselines.classical import (
    GreedyGflowBaseline, RandomBaseline,
    StaticScheduleBaseline, DistanceScheduleBaseline,
)

DEFAULT_RATES = [0.00, 0.01, 0.05, 0.10, 0.20]
C = {"oracle": "#2ca02c", "agent": "#1f77b4", "human": "#9467bd", "random": "#7f7f7f"}


def load_policy(ckpt, device):
    cfg = ckpt.get("config", {})
    if cfg.get("model_type") == "gnn":
        n = cfg.get("n", cfg.get("rows", 3) * cfg.get("cols", 3))
        p = GNNActorCritic(n=n, hidden_dim=cfg.get("hidden_dim", 64),
                           n_heads=cfg.get("n_heads", 4), n_layers=cfg.get("n_layers", 3),
                           use_angles=cfg.get("use_angles", False),
                           virtual_node=cfg.get("virtual_node", False),
                           weight_tied=cfg.get("weight_tied", False),
                           pos_dim=cfg.get("pos_dim", 0),
                           aux_layer_head=cfg.get("aux_layer_head", False))
    else:
        rows = cfg.get("rows", 3); cols = cfg.get("cols", 3)
        p = MBQCActorCritic(obs_dim=cfg.get("obs_dim", rows*cols*(rows*cols+1)),
                            n_actions=cfg.get("n_actions", rows*cols),
                            hidden_dim=cfg.get("hidden_dim", 256))
    p.load_state_dict(ckpt["policy_state_dict"]); p.to(device); p.eval()
    return p, cfg


def run_agent(env, policy, seed, device):
    obs, _ = env.reset(seed=seed); tot, done = 0.0, False
    while not done:
        o = torch.as_tensor(obs["observation"], dtype=torch.float32, device=device).unsqueeze(0)
        m = torch.as_tensor(obs["action_mask"],  dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = policy(o, m)
        obs, r, t, tr, _ = env.step(int(logits.argmax(-1))); tot += r; done = t or tr
    return tot


def run_baseline(env, make_baseline, seed):
    obs, info = env.reset(seed=seed)
    b = make_baseline(env); b.reset(obs, info)
    tot, done = 0.0, False
    while not done:
        obs, r, t, tr, _ = env.step(b.select_action(obs)); tot += r; done = t or tr
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--rows", type=int, default=None)
    ap.add_argument("--cols", type=int, default=None)
    ap.add_argument("--topology", default="grid",
                    choices=["grid", "brickwork", "irregular"])
    ap.add_argument("--human", default="distance", choices=["distance", "column"],
                    help="Which static baseline to use as the 'human' reference: "
                         "distance-from-output (general) or grid column-sweep")
    ap.add_argument("--n-trials", type=int, default=400)
    ap.add_argument("--defect-rates", nargs="+", type=float, default=DEFAULT_RATES)
    ap.add_argument("--save-dir", default="results/figures")
    args = ap.parse_args()

    device = torch.device("cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    policy, cfg = load_policy(ckpt, device)
    rows = args.rows or cfg.get("rows", 3)
    cols = args.cols or cfg.get("cols", 3)
    use_angles = cfg.get("use_angles", False)
    cliff = cfg.get("clifford_fraction", 0.5)
    N = args.n_trials

    print(f"\n{'═'*68}")
    print(f"  FOUR-WAY COMPARISON — {rows}×{cols} {args.topology}  ({N} trials/rate)")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"{'═'*68}")
    print(f"\n  {'rate':>6}  {'oracle':>8}  {'agent':>8}  {'human':>8}  {'random':>8}"
          f"  {'agent−human':>12}")
    print(f"  {'────':>6}  {'──────':>8}  {'─────':>8}  {'─────':>8}  {'──────':>8}"
          f"  {'───────────':>12}")

    observe_original = cfg.get("observe_original", False)
    series = {k: [] for k in ("oracle", "agent", "human", "random")}
    for rate in args.defect_rates:
        kw = dict(rows=rows, cols=cols, defect_rate=rate, use_angles=use_angles,
                  clifford_fraction=cliff, topology=args.topology,
                  observe_original_graph=observe_original)
        ea, eo, eh, er = (MBQCEnv(**kw) for _ in range(4))
        human_cls = (DistanceScheduleBaseline if args.human == "distance"
                     else StaticScheduleBaseline)
        a = np.mean([run_agent(ea, policy, 5000+i, device) for i in range(N)])
        o = np.mean([run_baseline(eo, GreedyGflowBaseline, 5000+i) for i in range(N)])
        h = np.mean([run_baseline(eh, human_cls, 5000+i) for i in range(N)])
        rd = np.mean([run_baseline(er, lambda e: RandomBaseline(
            rng=np.random.default_rng(123)), 5000+i) for i in range(N)])
        for k, v in zip(series, (o, a, h, rd)):
            series[k].append(v)
        print(f"  {rate:>6.2f}  {o:>8.4f}  {a:>8.4f}  {h:>8.4f}  {rd:>8.4f}"
              f"  {a-h:>+12.4f}")

    # ── Figure ──────────────────────────────────────────────────────────
    plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 160,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25})
    x = np.array(args.defect_rates) * 100
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(x, series["oracle"], "s--", color=C["oracle"], lw=1.8, label="gflow oracle (ceiling)")
    ax.plot(x, series["agent"],  "o-",  color=C["agent"],  lw=2.6, ms=8,
            label="RL agent (learned, adaptive)")
    ax.plot(x, series["human"],  "D-",  color=C["human"],  lw=2.2, ms=7,
            label="human / static schedule (non-adaptive)")
    ax.plot(x, series["random"], "^:",  color=C["random"], lw=1.8, label="random (floor)")
    # shade the (large) margin every structured method holds over random
    ax.fill_between(x, series["random"], series["agent"], color=C["agent"], alpha=0.10)
    ax.set_xlabel("Edge defect rate (%)")
    ax.set_ylabel(f"Mean reward ({rows}×{cols} {args.topology})")
    ax.set_title("RL agent reaches expert/oracle-level scheduling from reward alone\n"
                 "(a hand-designed static schedule is also strong on regular lattices)")
    ax.legend()
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    out = Path(args.save_dir) / "R7_human_baseline.png"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"\n  Figure saved → {out}\n{'═'*68}\n")


if __name__ == "__main__":
    main()
