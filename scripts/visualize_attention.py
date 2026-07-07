"""
GAT attention interpretability — what did the GNN learn from reward alone?

Loads a trained GNN, runs it on one sample graph, and extracts the attention
coefficients of every GAT layer. Produces:

  1. A per-layer figure: the graph with each edge coloured by how much
     attention flows along it (averaged over heads).
  2. Quantitative metrics:
       • Attention focus — entropy of each node's attention vs the uniform
         (attend-equally-to-all-neighbours) baseline. Lower = more selective.
       • Flow alignment — fraction of attention mass directed *toward* the
         output column (down the gflow layers). If the GNN has discovered the
         measurement-flow structure, attention should be biased toward outputs.

The point: the only training signal was a scalar gflow-consistency reward.
If attention nonetheless aligns with gflow structure, the GNN has learned the
physics of MBQC measurement ordering implicitly.

Usage
-----
python scripts/visualize_attention.py \\
    --checkpoint results/gnn_4x4_angled/final.pt \\
    --rows 4 --cols 4 --seed 7 \\
    --save-dir results/attention
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mbqc_rl.env.mbqc_env     import MBQCEnv
from mbqc_rl.agent.gnn_policy import GNNActorCritic


def load_gnn(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg  = ckpt.get("config", {})
    if cfg.get("model_type") != "gnn":
        raise ValueError("visualize_attention.py expects a GNN checkpoint")
    n = cfg.get("n", cfg.get("rows", 3) * cfg.get("cols", 3))
    policy = GNNActorCritic(n=n, hidden_dim=cfg.get("hidden_dim", 64),
                            n_heads=cfg.get("n_heads", 4),
                            n_layers=cfg.get("n_layers", 3),
                            use_angles=cfg.get("use_angles", False),
                            virtual_node=cfg.get("virtual_node", False),
                            weight_tied=cfg.get("weight_tied", False),
                            pos_dim=cfg.get("pos_dim", 0),
                            aux_layer_head=cfg.get("aux_layer_head", False))
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()
    return policy, cfg


def attention_metrics(alpha: np.ndarray, adj: np.ndarray,
                      order: dict[int, int]) -> dict:
    """
    alpha: (n, n) head-averaged attention (row i attends to col j).
    adj:   (n, n) binary adjacency.
    order: gflow layer per node (output=0, higher=measured earlier).
    """
    n = alpha.shape[0]
    # Restrict to graph neighbours (+self) — same support the layer used
    eye = np.eye(n)
    support = ((adj + eye) > 0)

    # ── Focus: entropy of each node's attention vs uniform over its support ──
    focus_ratios = []
    for i in range(n):
        nbrs = np.where(support[i])[0]
        if len(nbrs) <= 1:
            continue
        p = alpha[i, nbrs]
        p = p / (p.sum() + 1e-12)
        ent      = -(p * np.log(p + 1e-12)).sum()
        ent_unif = np.log(len(nbrs))
        focus_ratios.append(ent / (ent_unif + 1e-12))   # 1=uniform, <1=focused
    mean_focus = float(np.mean(focus_ratios)) if focus_ratios else 1.0

    # ── Flow alignment: attention mass toward outputs (lower gflow layer) ──
    toward, away = 0.0, 0.0
    for i in range(n):
        for j in range(n):
            if i == j or not support[i, j]:
                continue
            if i not in order or j not in order:
                continue
            if order[j] < order[i]:      # j is closer to output → "toward"
                toward += alpha[i, j]
            elif order[j] > order[i]:
                away += alpha[i, j]
    total = toward + away
    flow_align = float(toward / total) if total > 0 else 0.5

    return {"mean_focus": mean_focus, "flow_alignment": flow_align}


def plot_attention(adj, attentions_mean, outputs, rows, cols, save_to):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        print("matplotlib not found — skipping figure")
        return

    n = rows * cols
    pos = {r * cols + c: (c, -r) for r in range(rows) for c in range(cols)}
    out_set = set(outputs)
    n_layers = len(attentions_mean)

    fig, axes = plt.subplots(1, n_layers, figsize=(5 * n_layers, 4.6))
    if n_layers == 1:
        axes = [axes]

    for L, (ax, alpha) in enumerate(zip(axes, attentions_mean)):
        # Edge weights = symmetric attention along each graph edge
        edges, weights = [], []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j] > 0:
                    edges.append((i, j))
                    weights.append((alpha[i, j] + alpha[j, i]) / 2.0)
        w = np.array(weights)
        w_norm = (w - w.min()) / (np.ptp(w) + 1e-12) if len(w) else w

        for (i, j), wn in zip(edges, w_norm):
            x = [pos[i][0], pos[j][0]]; y = [pos[i][1], pos[j][1]]
            ax.plot(x, y, "-", color=cm.viridis(wn),
                    lw=1.0 + 5.0 * wn, alpha=0.85, zorder=1)

        for node, (x, y) in pos.items():
            is_out = node in out_set
            ax.scatter([x], [y], s=420,
                       color="#d62728" if is_out else "#1f77b4",
                       edgecolors="black", zorder=2)
            ax.text(x, y, "O" if is_out else str(node), color="white",
                    ha="center", va="center", fontsize=9, fontweight="bold",
                    zorder=3)

        ax.set_title(f"GAT layer {L + 1}", fontsize=12, fontweight="bold")
        ax.set_aspect("equal"); ax.axis("off")

    fig.suptitle("GAT attention along graph edges  "
                 "(thicker/brighter = more attention; red = output qubits)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(save_to, dpi=150, bbox_inches="tight")
    print(f"  Figure saved → {save_to}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--topology", default="grid", choices=["grid", "brickwork"])
    ap.add_argument("--save-dir", default="results/attention")
    args = ap.parse_args()

    policy, cfg = load_gnn(args.checkpoint)

    # Sample a perfect graph that has gflow
    env = MBQCEnv(rows=args.rows, cols=args.cols, defect_rate=0.0,
                  use_angles=cfg.get("use_angles", False),
                  clifford_fraction=cfg.get("clifford_fraction", 0.5),
                  topology=args.topology)
    obs, info = env.reset(seed=args.seed)
    if not info["gflow_exists"]:
        print("Sampled graph has no gflow; try another --seed")
        return

    adj   = env._adj.copy()
    order = env.gflow_order
    obs_t = torch.as_tensor(obs["observation"], dtype=torch.float32).unsqueeze(0)

    attentions = policy.get_attention(obs_t)          # list of (1,n,n,nH)
    att_mean   = [a[0].mean(dim=-1).numpy() for a in attentions]  # per-layer (n,n)

    print(f"\n{'═'*64}")
    print(f"  GAT ATTENTION ANALYSIS  —  {args.rows}×{args.cols} {args.topology}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"{'═'*64}")
    print(f"\n  {'layer':>6}  {'focus (1=uniform)':>18}  {'flow→output':>14}")
    for L, alpha in enumerate(att_mean):
        m = attention_metrics(alpha, adj, order)
        print(f"  {L+1:>6}  {m['mean_focus']:>18.3f}  {m['flow_alignment']:>13.1%}")
    print("\n  focus < 1.0  → attention is selective, not uniform")
    print("  flow→output > 50% → attention is biased down the gflow layers")

    os.makedirs(args.save_dir, exist_ok=True)
    suffix = f"{args.rows}x{args.cols}" + ("" if args.topology == "grid"
                                           else f"_{args.topology}")
    plot_attention(adj, att_mean, env.output_qubits,
                   args.rows, args.cols,
                   str(Path(args.save_dir) / f"attention_{suffix}.png"))
    print(f"{'═'*64}\n")


if __name__ == "__main__":
    main()
