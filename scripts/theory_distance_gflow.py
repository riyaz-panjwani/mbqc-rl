"""
Theory: when does distance-from-output equal the gflow layer order?

This explains the central dichotomy of the project: the distance heuristic
("measure farthest-from-output first") is near-optimal on lattices but fails on
irregular graphs.

Claim
-----
Let (G, O) be an open graph with gflow layers L(v) (L=0 on outputs, higher =
measured earlier) and let d(v) be the graph distance from v to the output set O.
If L(v) is a strictly-increasing function of d(v) for all non-output v, then the
distance ordering (descending d) is a valid gflow order — so the distance
heuristic is optimal.

- **Defect-free m-column grid, outputs = last column:** the gflow layer of a node
  in column c is exactly (cols−1−c) = d(v). Hence L ≡ d and the column-sweep /
  distance heuristic is optimal (reward 1.0). [verified below]
- **Irregular graphs with skip edges:** a skip edge shortcuts an upstream node to
  near the output, so d(v) drops while L(v) stays high → L is NOT monotone in d →
  the distance ordering measures that node too late → gflow violated → suboptimal.

This script measures, per topology: Spearman ρ(L, d), the exact-match rate
L(v)=d(v), and whether the distance ordering is gflow-valid (score = 1.0). It
produces a scatter figure (L vs d) for grid vs irregular.

Usage
-----
python scripts/theory_distance_gflow.py --n-graphs 300 --save-dir results/theory
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import networkx as nx
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mbqc_rl.utils.generators import (
    make_grid_graph, make_brickwork_graph, make_irregular_flow_graph,
)
from mbqc_rl.utils.gflow import compute_gflow, score_measurement_order

MAKERS = {"grid": make_grid_graph, "brickwork": make_brickwork_graph,
          "irregular": make_irregular_flow_graph}


def dist_from_outputs(G, outs):
    H = G.copy(); s = "__src__"
    H.add_node(s)
    for o in outs:
        H.add_edge(s, o)
    d = nx.single_source_shortest_path_length(H, s)
    return {v: d.get(v, 10**6) - 1 for v in G.nodes()}


def analyse(topology, rows, cols, n_graphs, defect, seed0):
    make = MAKERS[topology]
    rhos, exacts, valids = [], [], []
    Ls, Ds = [], []
    for i in range(n_graphs):
        rng = np.random.default_rng(seed0 + i)
        G, outs = make(rows, cols, defect_rate=defect, rng=rng)
        gflow, order = compute_gflow(G, outs)
        if gflow is None:
            continue
        out_set = set(outs)
        non_out = [v for v in G.nodes() if v not in out_set]
        d = dist_from_outputs(G, outs)
        L = [order[v] for v in non_out]
        D = [d[v] for v in non_out]
        if len(set(D)) > 1 and len(set(L)) > 1:
            rhos.append(spearmanr(L, D).correlation)
        exacts.append(np.mean([order[v] == d[v] for v in non_out]))
        # is the distance ordering (farthest first) a valid gflow order?
        dist_order = sorted(non_out, key=lambda v: -d[v])
        steps = {v: k + 1 for k, v in enumerate(dist_order)}
        valids.append(score_measurement_order(gflow, steps, out_set))
        Ls += L; Ds += D
    return {
        "spearman": float(np.mean(rhos)) if rhos else float("nan"),
        "exact_match": float(np.mean(exacts)) * 100 if exacts else float("nan"),
        "dist_order_valid": float(np.mean(valids)) if valids else float("nan"),
        "pts": (np.array(Ds), np.array(Ls)),
    }


def plot(grid_pts, irr_pts, save_to):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True, sharey=True)
    for ax, (name, (D, L)) in zip(axes, [("grid", grid_pts), ("irregular", irr_pts)]):
        # jitter for visibility
        jx = D + np.random.default_rng(0).normal(0, 0.06, len(D))
        jy = L + np.random.default_rng(1).normal(0, 0.06, len(L))
        ax.scatter(jx, jy, s=10, alpha=0.25,
                   color="#1f77b4" if name == "grid" else "#d62728")
        lim = max(D.max(), L.max()) + 1
        ax.plot([0, lim], [0, lim], "k--", lw=1, label="L = d (heuristic optimal)")
        ax.set_xlabel("distance from output  d(v)")
        ax.set_title(name)
        ax.grid(alpha=0.3); ax.legend(fontsize=9)
    axes[0].set_ylabel("gflow layer  L(v)")
    fig.suptitle("Why the distance heuristic wins on lattices, fails on irregular:\n"
                 "L(v) ≡ d(v) on grids (on the diagonal); scattered on irregular",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_to, dpi=150, bbox_inches="tight")
    print(f"  figure → {save_to}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-graphs", type=int, default=300)
    ap.add_argument("--save-dir", default="results/theory")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"\n{'='*72}\n  DISTANCE-FROM-OUTPUT vs GFLOW LAYER  ({args.n_graphs} graphs)\n{'='*72}")
    print(f"  {'topology':18s} {'Spearman ρ(L,d)':>16} {'exact L=d':>11} "
          f"{'dist-order gflow-valid':>23}")
    print(f"  {'-'*18} {'-'*16} {'-'*11} {'-'*23}")
    pts = {}
    for topo in ["grid", "brickwork", "irregular"]:
        r = analyse(topo, 4, 4, args.n_graphs, 0.0, args.seed)
        pts[topo] = r["pts"]
        print(f"  {topo+' 4x4 (perfect)':18s} {r['spearman']:>16.3f} "
              f"{r['exact_match']:>10.1f}% {r['dist_order_valid']:>22.3f}")

    print(f"\n  Interpretation:")
    print(f"  • grid: ρ≈1, exact≈100%, distance order is gflow-valid (≈1.0) →")
    print(f"    the distance/column-sweep heuristic is provably optimal on lattices.")
    print(f"  • irregular: ρ<1, exact<100%, distance order NOT fully valid (<1.0) →")
    print(f"    L is not monotone in d (skip edges) → heuristic suboptimal; the")
    print(f"    regime where a learned, adjacency-reading policy could add value.")

    os.makedirs(args.save_dir, exist_ok=True)
    plot(pts["grid"], pts["irregular"],
         os.path.join(args.save_dir, "distance_vs_gflow.png"))
    print()


if __name__ == "__main__":
    main()
