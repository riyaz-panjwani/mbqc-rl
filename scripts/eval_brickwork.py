"""
Brickwork topology generalization — the key practical-relevance experiment.

Brickwork states (Broadbent, Fitzsimons & Kashefi 2009) are the universal
resource state for blind MBQC — what real protocols actually run on. A policy
that was trained ONLY on grid (cluster) states and still performs well on
brickwork has generalised across graph *topology*, not just graph *size*.

The MLP cannot even be evaluated on a different grid size (its input layer is
fixed to n²+…); but it CAN be run on a same-size brickwork graph, so we report
it at 4×4 as a topology-shift baseline. The GNN, being size-inductive, is
evaluated on brickwork at every size zero-shot.

Usage
-----
python scripts/eval_brickwork.py \\
    --gnn results/gnn_4x4_angled/final.pt \\
    --mlp results/4x4_angled_s0/final.pt \\
    --n-trials 300 --defect-rate 0.01
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mbqc_rl.env.mbqc_env        import MBQCEnv
from mbqc_rl.agent.policy        import MBQCActorCritic
from mbqc_rl.agent.gnn_policy    import GNNActorCritic
from mbqc_rl.baselines.classical import GreedyGflowBaseline


def load_policy(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg  = ckpt.get("config", {})
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
    policy.eval()
    return policy, cfg


def run_agent(env, policy, seed):
    obs, _ = env.reset(seed=seed)
    total, done = 0.0, False
    while not done:
        o = torch.as_tensor(obs["observation"], dtype=torch.float32).unsqueeze(0)
        m = torch.as_tensor(obs["action_mask"],  dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits, _ = policy(o, m)
        obs, r, term, trunc, _ = env.step(int(logits.argmax(-1)))
        total += r; done = term or trunc
    return total


def run_greedy(env, seed):
    obs, info = env.reset(seed=seed)
    b = GreedyGflowBaseline(env); b.reset(obs, info)
    total, done = 0.0, False
    while not done:
        obs, r, term, trunc, _ = env.step(b.select_action(obs))
        total += r; done = term or trunc
    return total


def run_random(env, seed):
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    total, done = 0.0, False
    while not done:
        valid = np.where(obs["action_mask"] == 1)[0]
        obs, r, term, trunc, _ = env.step(int(rng.choice(valid)))
        total += r; done = term or trunc
    return total


def evaluate(policy, cfg, rows, cols, topology, n_trials, defect_rate):
    use_angles = cfg.get("use_angles", False)
    cliff = cfg.get("clifford_fraction", 0.5)
    kw = dict(rows=rows, cols=cols, defect_rate=defect_rate,
              use_angles=use_angles, clifford_fraction=cliff, topology=topology)
    env_a, env_g, env_r = MBQCEnv(**kw), MBQCEnv(**kw), MBQCEnv(**kw)
    ar = np.array([run_agent(env_a, policy, 7000+i) for i in range(n_trials)])
    gr = np.array([run_greedy(env_g, 7000+i)         for i in range(n_trials)])
    rr = np.array([run_random(env_r, 7000+i)         for i in range(n_trials)])
    _, p = stats.mannwhitneyu(ar, gr, alternative="two-sided")
    return ar, gr, rr, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gnn", required=True, help="GNN checkpoint (size-inductive)")
    ap.add_argument("--mlp", default=None, help="MLP checkpoint (same-size baseline)")
    ap.add_argument("--n-trials", type=int, default=300)
    ap.add_argument("--defect-rate", type=float, default=0.01)
    args = ap.parse_args()

    gnn, gcfg = load_policy(args.gnn)
    trained = f"{gcfg.get('rows')}×{gcfg.get('cols')}"

    print(f"\n{'═'*70}")
    print(f"  BRICKWORK GENERALIZATION  —  GNN trained on {trained} GRID, angled")
    print(f"  defect_rate={args.defect_rate}  trials={args.n_trials}")
    print(f"{'═'*70}")

    def tag(p): return "n.s." if p>0.05 else ("*" if p>0.01 else ("**" if p>0.001 else "***"))

    print(f"\n  GNN zero-shot on BRICKWORK (never trained on brickwork OR these sizes):")
    print(f"  {'grid':>6}  {'agent':>8} {'perfect':>8}  {'greedy':>8}  {'random':>8}  {'vs greedy':>10}")
    for rows, cols in [(3,3), (4,4), (5,5)]:
        ar, gr, rr, p = evaluate(gnn, gcfg, rows, cols, "brickwork",
                                 args.n_trials, args.defect_rate)
        print(f"  {rows}×{cols:<4} {ar.mean():>8.4f} {(ar==1.0).mean()*100:>7.1f}% "
              f" {gr.mean():>8.4f}  {rr.mean():>8.4f}  {p:>8.4f}{tag(p)}")

    # Cross-reference: same GNN on GRID (its training topology) for the same sizes
    print(f"\n  Reference — same GNN on GRID topology:")
    for rows, cols in [(3,3), (4,4), (5,5)]:
        ar, gr, rr, p = evaluate(gnn, gcfg, rows, cols, "grid",
                                 args.n_trials, args.defect_rate)
        print(f"  {rows}×{cols:<4} {ar.mean():>8.4f} {(ar==1.0).mean()*100:>7.1f}% "
              f" {gr.mean():>8.4f}  {rr.mean():>8.4f}  {p:>8.4f}{tag(p)}")

    # MLP same-size topology-shift baseline (only valid at its training size)
    if args.mlp:
        mlp, mcfg = load_policy(args.mlp)
        mr, mc = mcfg.get("rows"), mcfg.get("cols")
        print(f"\n  MLP ({mr}×{mc}, trained on GRID) — topology shift to brickwork at same size:")
        ar_g, gr_g, _, p_g = evaluate(mlp, mcfg, mr, mc, "grid",
                                      args.n_trials, args.defect_rate)
        ar_b, gr_b, _, p_b = evaluate(mlp, mcfg, mr, mc, "brickwork",
                                      args.n_trials, args.defect_rate)
        print(f"    on grid     : {ar_g.mean():.4f}  ({(ar_g==1.0).mean()*100:.1f}% perfect)")
        print(f"    on brickwork: {ar_b.mean():.4f}  ({(ar_b==1.0).mean()*100:.1f}% perfect)  "
              f"Δ={ar_b.mean()-ar_g.mean():+.4f}")

    print(f"\n{'═'*70}\n")


if __name__ == "__main__":
    main()
