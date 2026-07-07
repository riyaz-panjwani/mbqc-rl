"""
Work-around attempt A — imitation learning (behavioral cloning) from the oracle.

The RL agent fails to learn on irregular graphs (sparse OR shaped reward: entropy
stays at ~1.67, reward stuck at random). This script tests a different question:
can the GNN even *represent* a good ordering policy on irregular graphs if it is
handed the answer? We roll out the gflow oracle (GreedyGflowBaseline) on many
irregular graphs, collect (state, oracle-action) pairs, and train the GNN by
supervised cross-entropy to imitate the oracle — no RL, no exploration.

Interpretation:
  • If BC succeeds (≫ random, approaching the heuristic/oracle) → the GNN is
    expressive enough; the barrier is purely RL credit-assignment/exploration.
    Imitation from the classical oracle is then a viable practical work-around.
  • If BC also fails → the limitation is representational (obs/architecture),
    a deeper finding.

Either outcome is a clean, reportable result for the "attempts" section.

Usage
-----
python scripts/behavioral_clone.py --topology irregular \\
    --n-train-graphs 1500 --n-eval 400 --epochs 40
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mbqc_rl.env.mbqc_env        import MBQCEnv
from mbqc_rl.agent.gnn_policy    import GNNActorCritic
from mbqc_rl.baselines.classical import (
    GreedyGflowBaseline, DistanceScheduleBaseline,
)


def _static_obs(obs_vec, orig_adj_flat, n):
    """Replace the (degraded) adjacency block with the original full adjacency,
    keeping the history (+angle) channels. Lets the GNN always see the full
    graph structure, with the history channel marking measured qubits."""
    return np.concatenate([orig_adj_flat, obs_vec[n * n:]]).astype(np.float32)


def collect_demos(rows, cols, topology, n_graphs, seed0, static_obs=False):
    """Roll out the gflow oracle; return arrays of (obs, mask, action)."""
    obs_list, mask_list, act_list = [], [], []
    env = MBQCEnv(rows=rows, cols=cols, defect_rate=0.0, topology=topology)
    n = rows * cols
    kept = 0
    for i in range(n_graphs):
        obs, info = env.reset(seed=seed0 + i)
        if not info["gflow_exists"]:
            continue
        kept += 1
        orig_adj = env._adj.copy().flatten() if static_obs else None
        oracle = GreedyGflowBaseline(env)
        oracle.reset(obs, info)
        done = False
        while not done:
            a = oracle.select_action(obs)
            ob_vec = obs["observation"].copy()
            if static_obs:
                ob_vec = _static_obs(ob_vec, orig_adj, n)
            obs_list.append(ob_vec)
            mask_list.append(obs["action_mask"].copy())
            act_list.append(a)
            obs, _, term, trunc, _ = env.step(a)
            done = term or trunc
    return (np.asarray(obs_list, dtype=np.float32),
            np.asarray(mask_list, dtype=np.float32),
            np.asarray(act_list, dtype=np.int64), kept)


def evaluate(policy, rows, cols, topology, n_eval, seed0, device, static_obs=False):
    """Greedy rollout of the BC policy + reference baselines on held-out graphs."""
    n = rows * cols

    def run_agent(env, seed):
        obs, _ = env.reset(seed=seed); tot, done = 0.0, False
        orig_adj = env._adj.copy().flatten() if static_obs else None
        while not done:
            ob_vec = obs["observation"]
            if static_obs:
                ob_vec = _static_obs(ob_vec, orig_adj, n)
            o = torch.as_tensor(ob_vec, dtype=torch.float32, device=device).unsqueeze(0)
            m = torch.as_tensor(obs["action_mask"],  dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                logits, _ = policy(o, m)
            obs, r, t, tr, _ = env.step(int(logits.argmax(-1))); tot += r; done = t or tr
        return tot

    def run_b(env, cls, seed):
        obs, info = env.reset(seed=seed); b = cls(env); b.reset(obs, info)
        tot, done = 0.0, False
        while not done:
            obs, r, t, tr, _ = env.step(b.select_action(obs)); tot += r; done = t or tr
        return tot

    def run_rand(env, seed):
        obs, _ = env.reset(seed=seed); rng = np.random.default_rng(seed)
        tot, done = 0.0, False
        while not done:
            v = np.where(obs["action_mask"] == 1)[0]
            obs, r, t, tr, _ = env.step(int(rng.choice(v))); tot += r; done = t or tr
        return tot

    kw = dict(rows=rows, cols=cols, defect_rate=0.0, topology=topology)
    ea, eo, ed, er = (MBQCEnv(**kw) for _ in range(4))
    bc  = np.mean([run_agent(ea, seed0 + i)                       for i in range(n_eval)])
    orc = np.mean([run_b(eo, GreedyGflowBaseline, seed0 + i)      for i in range(n_eval)])
    dis = np.mean([run_b(ed, DistanceScheduleBaseline, seed0 + i) for i in range(n_eval)])
    rnd = np.mean([run_rand(er, seed0 + i)                        for i in range(n_eval)])
    return bc, orc, dis, rnd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--topology", default="irregular",
                    choices=["grid", "brickwork", "irregular"])
    ap.add_argument("--n-train-graphs", type=int, default=1500)
    ap.add_argument("--n-eval", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--static-obs", action="store_true",
                    help="Feed the ORIGINAL (undegraded) adjacency every step so "
                         "the GNN always sees full graph structure (history channel "
                         "marks measured qubits). Tests whether the degraded "
                         "observation is what blocks irregular imitation.")
    ap.add_argument("--save-dir", default="results/bc_irregular")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cpu")
    n = args.rows * args.cols

    print(f"\n{'═'*64}\n  IMITATION LEARNING (BC) from gflow oracle — {args.rows}×{args.cols} "
          f"{args.topology}\n{'═'*64}")
    print(f"  Collecting demos from {args.n_train_graphs} graphs …")
    obs, mask, act, kept = collect_demos(args.rows, args.cols, args.topology,
                                         args.n_train_graphs, seed0=10_000,
                                         static_obs=args.static_obs)
    print(f"  {kept} graphs with gflow → {len(act)} (state, oracle-action) pairs")

    policy = GNNActorCritic(n=n, hidden_dim=args.hidden_dim, n_heads=4,
                            n_layers=args.n_layers, use_angles=False).to(device)
    # The actor head is initialised near-zero (gain 0.01) for RL exploration;
    # re-init it at unit gain so supervised BC can actually fit the logits.
    nn.init.orthogonal_(policy.actor_head.weight, gain=1.0)
    nn.init.zeros_(policy.actor_head.bias)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    ce  = nn.CrossEntropyLoss()

    obs_t  = torch.as_tensor(obs,  device=device)
    mask_t = torch.as_tensor(mask, device=device)
    act_t  = torch.as_tensor(act,  device=device)
    N = len(act)

    print(f"  Training BC for {args.epochs} epochs …")
    for ep in range(args.epochs):
        perm = torch.randperm(N)
        tot_loss, tot_acc, nb = 0.0, 0.0, 0
        policy.train()
        for s in range(0, N, args.batch_size):
            idx = perm[s:s + args.batch_size]
            logits, _ = policy(obs_t[idx], mask_t[idx])
            loss = ce(logits, act_t[idx])
            opt.zero_grad(); loss.backward()
            # Mild clip: tames the −1e9-mask logit blow-up without crippling the
            # fit (max_norm=1 under-fits; this keeps grid imitation ~0.9).
            nn.utils.clip_grad_norm_(policy.parameters(), max_norm=10.0)
            opt.step()
            tot_loss += loss.item(); nb += 1
            tot_acc += (logits.argmax(-1) == act_t[idx]).float().mean().item()
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"    epoch {ep+1:>3}: loss={tot_loss/nb:.4f}  "
                  f"next-action acc={tot_acc/nb:.3f}")

    policy.eval()
    print(f"\n  Evaluating on {args.n_eval} held-out graphs …")
    bc, orc, dis, rnd = evaluate(policy, args.rows, args.cols, args.topology,
                                 args.n_eval, seed0=90_000, device=device,
                                 static_obs=args.static_obs)
    print(f"\n{'─'*64}")
    print(f"  BC agent (imitation)   : {bc:.4f}")
    print(f"  distance heuristic     : {dis:.4f}   <- the bar")
    print(f"  gflow oracle           : {orc:.4f}")
    print(f"  random                 : {rnd:.4f}")
    print(f"  (sparse/shaped RL agent was ≈0.49 — stuck at random)")
    verdict = ("GNN CAN represent it → barrier is RL exploration"
               if bc > dis else
               "BC ≈ heuristic/oracle? see numbers" if bc > rnd + 0.1 else
               "BC also fails → representational limit")
    print(f"  VERDICT: {verdict}")
    print(f"{'─'*64}")

    os.makedirs(args.save_dir, exist_ok=True)
    torch.save({"policy_state_dict": policy.state_dict(),
                "config": dict(model_type="gnn", rows=args.rows, cols=args.cols,
                               n=n, obs_dim=policy.obs_dim, n_actions=n,
                               hidden_dim=args.hidden_dim, n_heads=4,
                               n_layers=args.n_layers, use_angles=False,
                               topology=args.topology, trained_by="behavioral_cloning")},
               os.path.join(args.save_dir, "final.pt"))
    print(f"  Saved → {args.save_dir}/final.pt\n")


if __name__ == "__main__":
    main()
