"""
Top-ranked work-around — auxiliary gflow-layer supervision (NAR "hints").

Builds on the two proven fixes (static observation + depth). On top, we add an
auxiliary head that predicts each qubit's gflow LAYER from its node embedding,
trained jointly with the action head. Idea (Neural Algorithmic Reasoning,
Veličković et al.): supervising a network on the algorithm's intermediate
quantities shapes its representation so the main task becomes learnable. The
gflow layer is exactly the quantity that determines the measurement order.

This is still a diagnostic (uses the oracle's layers + actions as targets), run
in the fast supervised setting to get a quick verdict before committing to RL.

Compares:
  • BC (action only)            — baseline (~0.76 with static obs + 5 layers)
  • BC + aux gflow-layer head   — does the hint help?
against the distance heuristic (~0.89) and oracle (1.0) on irregular graphs.

Usage
-----
python scripts/aux_supervision.py --aux-weight 1.0 --n-layers 5 --epochs 60
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
from mbqc_rl.baselines.classical import GreedyGflowBaseline, DistanceScheduleBaseline


def _static_obs(obs_vec, orig_adj_flat, n):
    return np.concatenate([orig_adj_flat, obs_vec[n * n:]]).astype(np.float32)


def collect(rows, cols, n_graphs, seed0):
    """Demos with static obs + per-node gflow-layer targets (normalised)."""
    n = rows * cols
    O, M, A, L = [], [], [], []
    env = MBQCEnv(rows=rows, cols=cols, defect_rate=0.0, topology="irregular")
    kept = 0
    for i in range(n_graphs):
        obs, info = env.reset(seed=seed0 + i)
        if not info["gflow_exists"]:
            continue
        kept += 1
        order = env.gflow_order                      # node -> layer (output=0)
        max_layer = max(order.values()) or 1
        layer_vec = np.array([order.get(v, 0) / max_layer for v in range(n)],
                             dtype=np.float32)
        orig = env._adj.copy().flatten()
        oracle = GreedyGflowBaseline(env); oracle.reset(obs, info)
        done = False
        while not done:
            a = oracle.select_action(obs)
            O.append(_static_obs(obs["observation"].copy(), orig, n))
            M.append(obs["action_mask"].copy())
            A.append(a)
            L.append(layer_vec.copy())
            obs, _, t, tr, _ = env.step(a); done = t or tr
    return (np.asarray(O, np.float32), np.asarray(M, np.float32),
            np.asarray(A, np.int64), np.asarray(L, np.float32), kept)


def evaluate(policy, rows, cols, n_eval, seed0):
    n = rows * cols
    def run_agent(env, seed):
        obs, _ = env.reset(seed=seed); orig = env._adj.copy().flatten()
        tot, done = 0.0, False
        while not done:
            ov = _static_obs(obs["observation"], orig, n)
            o = torch.as_tensor(ov, dtype=torch.float32).unsqueeze(0)
            m = torch.as_tensor(obs["action_mask"], dtype=torch.float32).unsqueeze(0)
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
    kw = dict(rows=rows, cols=cols, defect_rate=0.0, topology="irregular")
    ea, ed, eo = MBQCEnv(**kw), MBQCEnv(**kw), MBQCEnv(**kw)
    ag = np.mean([run_agent(ea, seed0 + i) for i in range(n_eval)])
    di = np.mean([run_b(ed, DistanceScheduleBaseline, seed0 + i) for i in range(n_eval)])
    orc = np.mean([run_b(eo, GreedyGflowBaseline, seed0 + i) for i in range(n_eval)])
    return ag, di, orc


def train(rows, cols, n_layers, hidden, epochs, bs, lr, aux_weight,
          O, M, A, L, device):
    n = rows * cols
    policy = GNNActorCritic(n=n, hidden_dim=hidden, n_heads=4,
                            n_layers=n_layers, use_angles=False).to(device)
    nn.init.orthogonal_(policy.actor_head.weight, gain=1.0)
    nn.init.zeros_(policy.actor_head.bias)
    aux_head = nn.Linear(hidden, 1).to(device) if aux_weight > 0 else None

    params = list(policy.parameters())
    if aux_head is not None:
        params += list(aux_head.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    ce = nn.CrossEntropyLoss(); mse = nn.MSELoss()

    Ot = torch.as_tensor(O, device=device); Mt = torch.as_tensor(M, device=device)
    At = torch.as_tensor(A, device=device); Lt = torch.as_tensor(L, device=device)
    N = len(A)
    for ep in range(epochs):
        perm = torch.randperm(N)
        tl, ta, nb = 0.0, 0.0, 0
        policy.train()
        for s in range(0, N, bs):
            idx = perm[s:s + bs]
            H = policy._embed(Ot[idx])                       # (b, n, hidden)
            logits = policy.actor_head(H).squeeze(-1)
            logits = logits + (1.0 - Mt[idx]) * (-1e9)
            loss = ce(logits, At[idx])
            if aux_head is not None:
                layer_pred = aux_head(H).squeeze(-1)         # (b, n)
                loss = loss + aux_weight * mse(layer_pred, Lt[idx])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(params, 10.0)
            opt.step()
            tl += loss.item(); ta += (logits.argmax(-1) == At[idx]).float().mean().item(); nb += 1
        if (ep + 1) % 15 == 0 or ep == 0:
            print(f"    epoch {ep+1:>3}: loss={tl/nb:.4f}  action-acc={ta/nb:.3f}")
    policy.eval()
    return policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--n-train-graphs", type=int, default=2000)
    ap.add_argument("--n-eval", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=5)
    ap.add_argument("--aux-weight", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cpu")

    print(f"\n{'═'*64}\n  AUX GFLOW-LAYER SUPERVISION — {args.rows}×{args.cols} irregular "
          f"(static obs, {args.n_layers} layers)\n{'═'*64}")
    O, M, A, L, kept = collect(args.rows, args.cols, args.n_train_graphs, 10_000)
    print(f"  {kept} graphs → {len(A)} samples\n")

    print("  [baseline] action-only BC:")
    p0 = train(args.rows, args.cols, args.n_layers, args.hidden_dim, args.epochs,
               args.batch_size, args.lr, 0.0, O, M, A, L, device)
    bc0, di, orc = evaluate(p0, args.rows, args.cols, args.n_eval, 90_000)

    print(f"\n  [+aux] action + gflow-layer head (weight={args.aux_weight}):")
    p1 = train(args.rows, args.cols, args.n_layers, args.hidden_dim, args.epochs,
               args.batch_size, args.lr, args.aux_weight, O, M, A, L, device)
    bc1, _, _ = evaluate(p1, args.rows, args.cols, args.n_eval, 90_000)

    print(f"\n{'─'*64}")
    print(f"  action-only BC (static obs)   : {bc0:.4f}")
    print(f"  + aux gflow-layer supervision : {bc1:.4f}   (Δ = {bc1-bc0:+.4f})")
    print(f"  distance heuristic            : {di:.4f}   <- the bar")
    print(f"  gflow oracle                  : {orc:.4f}")
    print(f"  {'AUX HELPS' if bc1 > bc0 + 0.01 else 'aux ~neutral'}; "
          f"{'BEATS heuristic!' if bc1 > di else 'still below heuristic'}")
    print(f"{'─'*64}\n")


if __name__ == "__main__":
    main()
