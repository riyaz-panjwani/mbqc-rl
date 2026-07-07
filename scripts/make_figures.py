"""
Paper figure suite — conceptual + results figures for the dissertation.

Conceptual figures (demonstrate understanding of the problem):
  C1  defect_gflow      — what an edge defect does to a graph state & its gflow
  C2  gflow_layers      — gflow layer structure and the induced measurement order
  C3  gflow_vs_defect   — P(valid gflow) vs defect rate; why the task gets hard
  C4  angle_constraints — Clifford vs non-Clifford (hard-constraint) qubits
  C5  rollout           — an episode: the graph degrading measurement by measurement

Results figures (from our experiments):
  R1  scaling           — GNN zero-shot agent vs oracle vs random, 3×3 … 6×6
  R2  mlp_suite         — 18-run MLP suite, plain vs angled, mean ± std
  R3  defect_transfer   — GNN reward vs defect rate, home (4×4) and zero-shot (5×5)
  R4  reward_hist       — reward distributions: agent vs oracle vs random
  R5  forgetting        — staged curriculum (forgets) vs mixed-rate (fix)
  R6  topology          — grid vs brickwork structure + zero-shot performance

Usage
-----
python scripts/make_figures.py                 # all figures
python scripts/make_figures.py --only C3 R1     # selected figures
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import networkx as nx
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mbqc_rl.utils.generators import (
    make_grid_graph, make_brickwork_graph,
    assign_measurement_angles, is_clifford_angle,
)
from mbqc_rl.utils.gflow      import compute_gflow
from mbqc_rl.env.mbqc_env     import MBQCEnv
from mbqc_rl.agent.gnn_policy import GNNActorCritic

OUT = Path("results/figures")
GNN_CKPT = "results/gnn_4x4_angled/final.pt"

# Consistent palette
C_AGENT, C_GREEDY, C_RANDOM = "#1f77b4", "#2ca02c", "#7f7f7f"
C_NODE, C_OUT, C_DEFECT, C_NONCLIFF = "#1f77b4", "#d62728", "#ff7f0e", "#9467bd"


def setup_style():
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 160, "font.size": 11,
        "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25,
    })


# ─────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────

def grid_pos(rows, cols):
    return {r * cols + c: (c, -r) for r in range(rows) for c in range(cols)}


def draw_graph(ax, G, pos, outputs, node_colors=None, edge_colors=None,
               removed_edges=None, labels=None, node_size=420, title=None):
    out_set = set(outputs)
    # removed (defect) edges, dashed
    if removed_edges:
        for u, v in removed_edges:
            x = [pos[u][0], pos[v][0]]; y = [pos[u][1], pos[v][1]]
            ax.plot(x, y, ls=(0, (2, 2)), color=C_DEFECT, lw=1.8, alpha=0.8, zorder=1)
    # present edges
    for u, v in G.edges():
        x = [pos[u][0], pos[v][0]]; y = [pos[u][1], pos[v][1]]
        col = "0.45"
        if edge_colors and (u, v) in edge_colors:
            col = edge_colors[(u, v)]
        elif edge_colors and (v, u) in edge_colors:
            col = edge_colors[(v, u)]
        ax.plot(x, y, "-", color=col, lw=2.0, zorder=1, alpha=0.9)
    # nodes
    for node, (x, y) in pos.items():
        if node_colors and node in node_colors:
            c = node_colors[node]
        else:
            c = C_OUT if node in out_set else C_NODE
        ax.scatter([x], [y], s=node_size, color=c, edgecolors="black",
                   linewidths=1.2, zorder=2)
        lab = labels.get(node, "") if labels else ("O" if node in out_set else str(node))
        ax.text(x, y, lab, color="white", ha="center", va="center",
                fontsize=8.5, fontweight="bold", zorder=3)
    ax.set_aspect("equal"); ax.axis("off")
    if title:
        ax.set_title(title)


def load_gnn(ckpt_path=GNN_CKPT):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg  = ckpt["config"]
    policy = GNNActorCritic(n=cfg["n"], hidden_dim=cfg["hidden_dim"],
                            n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
                            use_angles=cfg["use_angles"],
                            virtual_node=cfg.get("virtual_node", False),
                            weight_tied=cfg.get("weight_tied", False),
                            pos_dim=cfg.get("pos_dim", 0),
                            aux_layer_head=cfg.get("aux_layer_head", False))
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
        obs, r, t, tr, _ = env.step(int(logits.argmax(-1)))
        total += r; done = t or tr
    return total


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")


# ─────────────────────────────────────────────────────────────────────────
# C1 — defect → gflow
# ─────────────────────────────────────────────────────────────────────────

def fig_defect_gflow():
    rows, cols = 4, 4
    pos = grid_pos(rows, cols)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    fig.subplots_adjust(wspace=0.05)

    # perfect
    Gp, outs = make_grid_graph(rows, cols, defect_rate=0.0,
                               rng=np.random.default_rng(0))
    gfp, _ = compute_gflow(Gp, outs)
    draw_graph(axes[0], Gp, pos, outs,
               title=f"Perfect cluster state\ngflow {'EXISTS ✓' if gfp else 'broken'} "
                     f"(deterministic)")

    # defective: drop a few specific edges that break gflow
    Gd = Gp.copy()
    full_edges = set(map(lambda e: tuple(sorted(e)), Gp.edges()))
    # remove edges to isolate a path to outputs
    drop = [(1, 2), (5, 6), (9, 10), (13, 14)]  # cut column feeding outputs
    removed = []
    for u, v in drop:
        if Gd.has_edge(u, v):
            Gd.remove_edge(u, v); removed.append((u, v))
    gfd, _ = compute_gflow(Gd, outs)
    draw_graph(axes[1], Gd, pos, outs, removed_edges=removed,
               title=f"After 4 edge defects (orange dashed)\n"
                     f"gflow {'EXISTS ✓' if gfd else 'BROKEN ✗'}")

    legend = [
        Line2D([0], [0], color="0.45", lw=2, label="intact edge (CZ entanglement)"),
        Line2D([0], [0], color=C_DEFECT, lw=1.8, ls=(0, (2, 2)), label="defect (deleted edge)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_NODE,
               markersize=11, label="measured qubit"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_OUT,
               markersize=11, label="output qubit"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("An edge defect can destroy the gflow that guarantees "
                 "deterministic MBQC", y=1.02)
    save(fig, "C1_defect_gflow")


# ─────────────────────────────────────────────────────────────────────────
# C2 — gflow layers & measurement order
# ─────────────────────────────────────────────────────────────────────────

def fig_gflow_layers():
    rows, cols = 4, 5
    pos = grid_pos(rows, cols)
    G, outs = make_grid_graph(rows, cols, defect_rate=0.0,
                              rng=np.random.default_rng(0))
    gf, order = compute_gflow(G, outs)
    max_layer = max(order.values())
    cmap = plt.cm.viridis_r
    node_colors = {v: cmap(order[v] / max(max_layer, 1)) for v in G.nodes()}
    labels = {v: f"L{order[v]}" for v in G.nodes()}

    fig, ax = plt.subplots(figsize=(8, 5.5))
    draw_graph(ax, G, pos, outs, node_colors=node_colors, labels=labels,
               node_size=520)
    # measurement-order arrow
    ax.annotate("", xy=(cols - 1.2, 0.9), xytext=(0.2, 0.9),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=2))
    ax.text((cols - 1) / 2, 1.35, "measurement order  (high layer → low layer → outputs)",
            ha="center", fontsize=10, style="italic")
    ax.set_title("gflow layer decomposition — the correct measurement order\n"
                 "Lk = gflow layer k  (L0 = outputs, never measured; "
                 "higher = measured earlier)")
    save(fig, "C2_gflow_layers")


# ─────────────────────────────────────────────────────────────────────────
# C3 — P(valid gflow) vs defect rate  (KEY explanatory figure)
# ─────────────────────────────────────────────────────────────────────────

def fig_gflow_vs_defect(n_samples=400):
    rates = np.linspace(0, 0.40, 17)
    sizes = [(3, 3), (4, 4), (5, 5), (6, 6)]
    colors = plt.cm.plasma(np.linspace(0.1, 0.8, len(sizes)))

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for (rows, cols), col in zip(sizes, colors):
        frac = []
        for rate in rates:
            cnt = 0
            for i in range(n_samples):
                G, outs = make_grid_graph(rows, cols, defect_rate=rate,
                                          rng=np.random.default_rng(10_000*rows + i + int(rate*1e4)))
                gf, _ = compute_gflow(G, outs)
                cnt += int(gf is not None)
            frac.append(cnt / n_samples)
        ax.plot(rates * 100, np.array(frac) * 100, "o-", color=col,
                lw=2, markersize=4, label=f"{rows}×{cols} ({rows*cols} qubits)")

    ax.axvline(1, ls=":", color="red", alpha=0.6)
    ax.text(1.4, 8, "realistic\nrate (1%)", color="red", fontsize=9)
    ax.set_xlabel("Edge defect rate (%)")
    ax.set_ylabel("P(a valid gflow exists)  (%)")
    ax.set_title("Why the task gets hard: defects destroy gflow\n"
                 "Above ~20% defects most instances are physically unsolvable "
                 "— reward 0 is unavoidable")
    ax.legend(title="grid size")
    ax.set_ylim(-3, 103)
    save(fig, "C3_gflow_vs_defect")


# ─────────────────────────────────────────────────────────────────────────
# C4 — Clifford vs non-Clifford angle constraints
# ─────────────────────────────────────────────────────────────────────────

def fig_angle_constraints():
    rows, cols = 4, 4
    pos = grid_pos(rows, cols)
    G, outs = make_grid_graph(rows, cols, defect_rate=0.0,
                              rng=np.random.default_rng(2))
    angles = assign_measurement_angles(G, outs, clifford_fraction=0.5,
                                       rng=np.random.default_rng(3))
    out_set = set(outs)
    node_colors, labels = {}, {}
    for v in G.nodes():
        k = angles[v]
        if v in out_set:
            node_colors[v] = C_OUT; labels[v] = "O"
        elif is_clifford_angle(k):
            node_colors[v] = C_NODE; labels[v] = f"{k}"
        else:
            node_colors[v] = C_NONCLIFF; labels[v] = f"{k}"

    fig, ax = plt.subplots(figsize=(7.5, 6))
    draw_graph(ax, G, pos, outs, node_colors=node_colors, labels=labels,
               node_size=560)
    legend = [
        Patch(facecolor=C_NODE,     edgecolor="k", label="Clifford angle (k even) — corrections\nPauli-frame trackable: ordering FREE"),
        Patch(facecolor=C_NONCLIFF, edgecolor="k", label="non-Clifford (k odd, T-type) — basis\nadaptation: ordering is a HARD constraint"),
        Patch(facecolor=C_OUT,      edgecolor="k", label="output qubit (not measured)"),
    ]
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.02),
              frameon=False, fontsize=9)
    ax.set_title("Universal MBQC: only non-Clifford qubits impose hard\n"
                 "ordering constraints  (label = angle index k, angle = kπ/4)")
    save(fig, "C4_angle_constraints")


# ─────────────────────────────────────────────────────────────────────────
# C5 — measurement rollout
# ─────────────────────────────────────────────────────────────────────────

def fig_rollout():
    rows, cols = 3, 4
    pos = grid_pos(rows, cols)
    env = MBQCEnv(rows=rows, cols=cols, defect_rate=0.0)
    obs, info = env.reset(seed=1)
    G0 = env._graph.copy()
    outs = env.output_qubits
    order = env.gflow_order
    # follow the correct (descending-layer) order
    non_out = [q for q in range(rows*cols) if q not in set(outs)]
    seq = sorted(non_out, key=lambda q: -order[q])

    n_panels = 5
    pick = np.linspace(0, len(seq), n_panels, dtype=int)
    fig, axes = plt.subplots(1, n_panels, figsize=(3.3*n_panels, 3.6))

    measured = set()
    G = G0.copy()
    step_idx = 0
    panel = 0
    snapshots = set(pick.tolist())
    # build snapshots
    states = []
    states.append((G0.copy(), set()))   # step 0
    Gc = G0.copy(); meas = set()
    for q in seq:
        Gc = Gc.copy(); meas = meas | {q}
        if Gc.has_node(q):
            Gc.remove_node(q)
        states.append((Gc.copy(), set(meas)))
    idxs = np.linspace(0, len(seq), n_panels, dtype=int)

    for ax, k in zip(axes, idxs):
        Gc, meas = states[k]
        node_colors = {}
        labels = {}
        for v in range(rows*cols):
            if v in set(outs):
                node_colors[v] = C_OUT; labels[v] = "O"
            elif v in meas:
                node_colors[v] = "0.8"; labels[v] = "✓"
            else:
                node_colors[v] = C_NODE; labels[v] = str(v)
        # draw with full node set positions but only remaining edges
        draw_graph(ax, Gc, pos, outs, node_colors=node_colors, labels=labels,
                   node_size=360)
        ax.set_title(f"after {k} measurements", fontsize=11)

    fig.suptitle("MDP rollout: each action measures one qubit and removes it "
                 "(grey ✓) — the graph degrades step by step", y=1.04)
    save(fig, "C5_rollout")


# ─────────────────────────────────────────────────────────────────────────
# R1 — zero-shot scaling curve
# ─────────────────────────────────────────────────────────────────────────

def fig_scaling(n_trials=300):
    policy, cfg = load_gnn()
    sizes = [(3, 3), (4, 4), (5, 5), (6, 6)]
    qubits, agent, greedy, rand = [], [], [], []
    from mbqc_rl.baselines.classical import GreedyGflowBaseline
    for rows, cols in sizes:
        kw = dict(rows=rows, cols=cols, defect_rate=0.01,
                  use_angles=cfg["use_angles"],
                  clifford_fraction=cfg["clifford_fraction"])
        ea, eg, er = MBQCEnv(**kw), MBQCEnv(**kw), MBQCEnv(**kw)
        a = np.mean([run_agent(ea, policy, 7000+i) for i in range(n_trials)])
        # greedy
        def rg(env, seed):
            obs, info = env.reset(seed=seed)
            b = GreedyGflowBaseline(env); b.reset(obs, info)
            tot, d = 0.0, False
            while not d:
                obs, r, t, tr, _ = env.step(b.select_action(obs)); tot += r; d = t or tr
            return tot
        g = np.mean([rg(eg, 7000+i) for i in range(n_trials)])
        def rr(env, seed):
            obs, _ = env.reset(seed=seed); rng = np.random.default_rng(seed)
            tot, d = 0.0, False
            while not d:
                valid = np.where(obs["action_mask"] == 1)[0]
                obs, r, t, tr, _ = env.step(int(rng.choice(valid))); tot += r; d = t or tr
            return tot
        rd = np.mean([rr(er, 7000+i) for i in range(n_trials)])
        qubits.append(rows*cols); agent.append(a); greedy.append(g); rand.append(rd)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = np.array(qubits)
    ax.plot(x, greedy, "s--", color=C_GREEDY, lw=2, label="gflow oracle (ceiling)")
    ax.plot(x, agent,  "o-",  color=C_AGENT,  lw=2.5, markersize=8,
            label="GNN agent (trained on 4×4 only)")
    ax.plot(x, rand,   "^:",  color=C_RANDOM, lw=2, label="random (floor)")
    ax.axvspan(15, 17, color=C_AGENT, alpha=0.08)
    ax.text(16, 0.55, "training\nsize (16)", ha="center", color=C_AGENT, fontsize=9)
    for xi, yi in zip(x, agent):
        ax.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=9, color=C_AGENT)
    ax.set_xlabel("Graph size (number of qubits)")
    ax.set_ylabel("Mean reward @ 1% defect")
    ax.set_xticks(x); ax.set_xticklabels([f"{int(q)}\n({int(q**0.5)}×{int(q**0.5)})" for q in x])
    ax.set_ylim(0.3, 1.03)
    ax.set_title("Zero-shot size generalization: one GNN, 9 → 36 qubits\n"
                 "(trained only on 4×4; 3×3/5×5/6×6 never seen)")
    ax.legend()
    save(fig, "R1_scaling")


# ─────────────────────────────────────────────────────────────────────────
# R2 — MLP 18-run suite bars
# ─────────────────────────────────────────────────────────────────────────

def fig_mlp_suite():
    grids = ["3x3", "4x4", "5x5"]
    plain_m, plain_s, ang_m, ang_s = [], [], [], []
    for g in grids:
        p = json.load(open(f"results/seeds/{g}_plain.json"))["across_seeds"]
        a = json.load(open(f"results/seeds/{g}_angled.json"))["across_seeds"]
        plain_m.append(p["mean_of_means"]); plain_s.append(p["std_of_means"])
        ang_m.append(a["mean_of_means"]);   ang_s.append(a["std_of_means"])

    x = np.arange(len(grids)); w = 0.36
    fig, ax = plt.subplots(figsize=(8, 5.5))
    b1 = ax.bar(x - w/2, plain_m, w, yerr=plain_s, capsize=5,
                color="#4c72b0", label="plain (Clifford only)")
    b2 = ax.bar(x + w/2, ang_m, w, yerr=ang_s, capsize=5,
                color="#dd8452", label="angled (universal, non-Clifford)")
    for b, m in zip(b1, plain_m):
        ax.text(b.get_x()+b.get_width()/2, m+0.01, f"{m:.3f}", ha="center", fontsize=9)
    for b, m in zip(b2, ang_m):
        ax.text(b.get_x()+b.get_width()/2, m+0.01, f"{m:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([g.replace("x", "×") for g in grids])
    ax.set_ylabel("Mean reward @ 1% defect (3 seeds, mean ± std)")
    ax.set_xlabel("Grid size")
    ax.set_ylim(0, 1.08)
    ax.set_title("MLP mixed-rate suite — 18 runs (3 grids × 2 modes × 3 seeds)\n"
                 "angled reward improves stability and mean at larger sizes")
    ax.legend()
    save(fig, "R2_mlp_suite")


# ─────────────────────────────────────────────────────────────────────────
# R3 — defect transfer curves
# ─────────────────────────────────────────────────────────────────────────

def fig_defect_transfer():
    home = json.load(open("results/transfer_gnn/transfer_4x4.json"))
    zs   = json.load(open("results/transfer_gnn/transfer_5x5.json"))

    def series(data, key):
        return ([d["defect_rate"]*100 for d in data],
                [d[key]["mean"] for d in data])

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    rh, ah = series(home, "agent"); _, gh = series(home, "greedy")
    rz, az = series(zs, "agent");   _, gz = series(zs, "greedy")
    ax.plot(rh, ah, "o-",  color=C_AGENT,  lw=2.5, label="GNN @ 4×4 (home)")
    ax.plot(rh, gh, "s--", color=C_AGENT,  lw=1.3, alpha=0.5, label="oracle @ 4×4")
    ax.plot(rz, az, "o-",  color="#d62728", lw=2.5, label="GNN @ 5×5 (zero-shot)")
    ax.plot(rz, gz, "s--", color="#d62728", lw=1.3, alpha=0.5, label="oracle @ 5×5")
    ax.set_xlabel("Edge defect rate (%)")
    ax.set_ylabel("Mean reward")
    ax.set_title("GNN tracks the oracle across all defect rates\n"
                 "high-rate fall-off is gflow vanishing (see C3), not agent failure")
    ax.legend()
    save(fig, "R3_defect_transfer")


# ─────────────────────────────────────────────────────────────────────────
# R4 — reward distribution histograms
# ─────────────────────────────────────────────────────────────────────────

def fig_reward_hist(n_trials=600):
    policy, cfg = load_gnn()
    from mbqc_rl.baselines.classical import GreedyGflowBaseline
    kw = dict(rows=4, cols=4, defect_rate=0.01,
              use_angles=cfg["use_angles"], clifford_fraction=cfg["clifford_fraction"])
    ea, eg, er = MBQCEnv(**kw), MBQCEnv(**kw), MBQCEnv(**kw)
    a = np.array([run_agent(ea, policy, 8000+i) for i in range(n_trials)])
    def rg(env, seed):
        obs, info = env.reset(seed=seed); b = GreedyGflowBaseline(env); b.reset(obs, info)
        tot, d = 0.0, False
        while not d:
            obs, r, t, tr, _ = env.step(b.select_action(obs)); tot += r; d = t or tr
        return tot
    def rr(env, seed):
        obs, _ = env.reset(seed=seed); rng = np.random.default_rng(seed)
        tot, d = 0.0, False
        while not d:
            v = np.where(obs["action_mask"] == 1)[0]
            obs, r, t, tr, _ = env.step(int(rng.choice(v))); tot += r; d = t or tr
        return tot
    g = np.array([rg(eg, 8000+i) for i in range(n_trials)])
    rd = np.array([rr(er, 8000+i) for i in range(n_trials)])

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    bins = np.linspace(0, 1.0, 21)
    ax.hist(rd, bins=bins, color=C_RANDOM, alpha=0.6, label=f"random (μ={rd.mean():.2f})")
    ax.hist(a,  bins=bins, color=C_AGENT,  alpha=0.7, label=f"GNN agent (μ={a.mean():.2f})")
    ax.hist(g,  bins=bins, color=C_GREEDY, alpha=0.5, label=f"oracle (μ={g.mean():.2f})")
    ax.axvline(a.mean(), color=C_AGENT, ls="--", lw=1.5)
    ax.set_xlabel("Episode reward (gflow-consistency)")
    ax.set_ylabel("Number of graph instances")
    ax.set_title("Reward distributions on 4×4 @ 1% defect (600 graphs)\n"
                 "the agent's mass sits at reward 1.0, near the oracle")
    ax.legend()
    save(fig, "R4_reward_hist")


# ─────────────────────────────────────────────────────────────────────────
# R5 — forgetting diagnosis
# ─────────────────────────────────────────────────────────────────────────

def fig_forgetting():
    # Documented, verified results (see DISSERTATION_LOG.md Sessions 5–6)
    grids = ["4×4", "5×5"]
    stage0 = [0.954, 0.931]   # learned the clean-graph skill
    stage3 = [0.747, 0.511]   # staged curriculum forgot it
    mixed  = [0.960, 0.863]   # mixed-rate fix

    x = np.arange(len(grids)); w = 0.26
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.bar(x - w, stage0, w, color="#8da0cb", label="staged Stage-0 (clean-graph skill)")
    ax.bar(x,     stage3, w, color="#fc8d62", label="staged Stage-3 (after 20–30% — forgot)")
    ax.bar(x + w, mixed,  w, color="#66c2a5", label="mixed-rate U[0,0.3] (our fix)")
    for i in range(len(grids)):
        for off, val in [(-w, stage0[i]), (0, stage3[i]), (w, mixed[i])]:
            ax.text(x[i]+off, val+0.012, f"{val:.2f}", ha="center", fontsize=8.5)
    # forgetting arrows
    for i in range(len(grids)):
        ax.annotate("", xy=(x[i], stage3[i]+0.02), xytext=(x[i]-w, stage0[i]-0.02),
                    arrowprops=dict(arrowstyle="->", color="firebrick", lw=1.5))
    ax.text(0.5, 0.40, "catastrophic\nforgetting", color="firebrick", fontsize=9, ha="center")
    ax.set_xticks(x); ax.set_xticklabels(grids)
    ax.set_ylabel("Mean reward @ 1% defect")
    ax.set_xlabel("Grid size")
    ax.set_ylim(0, 1.30)
    ax.set_title("Staged curriculum forgets at scale; mixed-rate training fixes it")
    ax.legend(fontsize=9, loc="upper center", ncol=3, frameon=True)
    save(fig, "R5_forgetting")


# ─────────────────────────────────────────────────────────────────────────
# R6 — grid vs brickwork
# ─────────────────────────────────────────────────────────────────────────

def fig_topology():
    fig = plt.figure(figsize=(12, 5.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.1])

    rows, cols = 4, 5
    pos = grid_pos(rows, cols)
    ax0 = fig.add_subplot(gs[0])
    Gg, outs = make_grid_graph(rows, cols, defect_rate=0.0, rng=np.random.default_rng(0))
    draw_graph(ax0, Gg, pos, outs, node_size=300, title=f"Grid cluster state\n({Gg.number_of_edges()} edges)")
    ax1 = fig.add_subplot(gs[1])
    Gb, outs = make_brickwork_graph(rows, cols, defect_rate=0.0, rng=np.random.default_rng(0))
    draw_graph(ax1, Gb, pos, outs, node_size=300, title=f"Brickwork state (BFK 2009)\n({Gb.number_of_edges()} edges, staggered)")

    # performance bars (documented zero-shot results, GNN 4×4 angled)
    ax2 = fig.add_subplot(gs[2])
    sizes = ["3×3", "4×4", "5×5"]
    grid_perf  = [0.989, 0.962, 0.856]
    brick_perf = [0.970, 0.897, 0.785]
    x = np.arange(len(sizes)); w = 0.38
    ax2.bar(x - w/2, grid_perf,  w, color="#4c72b0", label="grid (eval)")
    ax2.bar(x + w/2, brick_perf, w, color="#c44e52", label="brickwork (eval)")
    ax2.set_xticks(x); ax2.set_xticklabels(sizes)
    ax2.set_ylabel("GNN mean reward @ 1% defect")
    ax2.set_ylim(0, 1.08)
    ax2.set_title("Zero-shot to a new topology\n(GNN trained on 4×4 grid)")
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.25)

    fig.suptitle("Topology generalization: the GNN transfers from grid to the "
                 "brickwork resource state used in real blind MBQC", y=1.03)
    save(fig, "R6_topology")


# ─────────────────────────────────────────────────────────────────────────
# R8 — the irregular win: early-stopped ensemble beats the distance heuristic
#      at 3 sizes, under BOTH the plain and the non-Clifford (angled) reward.
#      Verified numbers from early_stopping_eval.py / _validate.py (Sessions
#      24–29; see RESULTS.md §7 and DISSERTATION_LOG.md).
# ─────────────────────────────────────────────────────────────────────────

def fig_irregular_win():
    sizes = ["4×4", "5×5", "6×6"]
    x = np.arange(len(sizes))
    # (heuristic, ensemble, oracle, Δ, p-string)
    plain = dict(heur=[0.885, 0.915, 0.929], ens=[0.920, 0.944, 0.952],
                 orc=[1.000, 1.000, 0.990], d=[0.035, 0.029, 0.023],
                 p=["4e-9", "4e-17", "8e-21"])
    angled = dict(heur=[0.893, 0.918, 0.928], ens=[0.958, 0.954, 0.961],
                  orc=[1.000, 1.000, 0.990], d=[0.065, 0.036, 0.033],
                  p=["5e-15", "7e-12", "1e-22"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5),
                             gridspec_kw={"width_ratios": [1, 1, 0.85]})
    w = 0.38
    ylo = 0.80

    def panel(ax, data, title):
        bh = ax.bar(x - w/2, data["heur"], w, color=C_RANDOM,
                    label="distance heuristic")
        be = ax.bar(x + w/2, data["ens"], w, color=C_AGENT,
                    label="early-stopped ensemble (ours)")
        # oracle reference tick per group
        for i, o in enumerate(data["orc"]):
            ax.plot([x[i] - w, x[i] + w], [o, o], color=C_GREEDY, lw=2,
                    zorder=5, label="greedy-gflow oracle" if i == 0 else None)
        # Δ + significance above each ensemble bar
        for i in range(len(sizes)):
            ax.annotate(f"+{data['d'][i]:.3f}\n(p={data['p'][i]})",
                        xy=(x[i] + w/2, data["ens"][i]),
                        xytext=(x[i] + w/2, data["ens"][i] + 0.012),
                        ha="center", va="bottom", fontsize=8.5,
                        fontweight="bold", color=C_AGENT)
        ax.set_xticks(x); ax.set_xticklabels(sizes)
        ax.set_ylim(ylo, 1.05)
        ax.set_xlabel("Grid size (qubits: 16 / 25 / 36)")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.text(0.02, 0.02, "random ≈ 0.47 (off-scale)", transform=ax.transAxes,
                fontsize=8, style="italic", color="#999999")
        return bh, be

    panel(axes[0], plain,  "Plain reward\n(gflow-consistency)")
    panel(axes[1], angled, "Angled reward\n(universal MBQC, non-Clifford)")
    axes[0].set_ylabel(f"Mean reward on irregular graphs  (y from {ylo})")
    # one shared legend
    h, l = axes[0].get_legend_handles_labels()
    axes[0].legend(h, l, fontsize=8.5, loc="upper right", framealpha=0.95)

    # Panel 3 — the punchline: Δ (ensemble − heuristic) vs size, plain vs angled
    ax2 = axes[2]
    ax2.plot(x, plain["d"],  "o-", color="#8c8c8c", lw=2, ms=8, label="plain reward")
    ax2.plot(x, angled["d"], "s-", color="#d62728", lw=2, ms=8, label="angled reward")
    for i in range(len(sizes)):
        ax2.annotate(f"+{angled['d'][i]:.3f}", (x[i], angled["d"][i]),
                     textcoords="offset points", xytext=(0, 8), ha="center",
                     fontsize=8.5, color="#d62728", fontweight="bold")
        ax2.annotate(f"+{plain['d'][i]:.3f}", (x[i], plain["d"][i]),
                     textcoords="offset points", xytext=(0, -14), ha="center",
                     fontsize=8.5, color="#8c8c8c", fontweight="bold")
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_xticks(x); ax2.set_xticklabels(sizes)
    ax2.set_ylim(-0.005, 0.085)
    ax2.set_xlabel("Grid size")
    ax2.set_ylabel("Δ = ensemble − heuristic")
    ax2.set_title("Win margin:\nangled ≥ plain at every size")
    ax2.legend(fontsize=9, loc="upper right")
    ax2.grid(axis="y", alpha=0.25)

    fig.subplots_adjust(wspace=0.28)
    fig.suptitle("On irregular MBQC graphs the learned ensemble beats the strong "
                 "distance heuristic — at 3 sizes, under both rewards\n"
                 "(all 6 wins robustness-validated: 5/5 test sets, bootstrap CI "
                 "excludes 0, leave-one-out; §7)", y=1.12, fontsize=12.5)
    save(fig, "R8_irregular_win")


# ─────────────────────────────────────────────────────────────────────────

ALL = {
    "C1": fig_defect_gflow,    "C2": fig_gflow_layers,
    "C3": fig_gflow_vs_defect, "C4": fig_angle_constraints,
    "C5": fig_rollout,
    "R1": fig_scaling,         "R2": fig_mlp_suite,
    "R3": fig_defect_transfer, "R4": fig_reward_hist,
    "R5": fig_forgetting,      "R6": fig_topology,
    "R8": fig_irregular_win,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", default=None,
                    help=f"Subset of figures to generate: {list(ALL)}")
    args = ap.parse_args()
    setup_style()

    keys = args.only or list(ALL)
    print(f"\nGenerating {len(keys)} figure(s) → {OUT}/\n")
    for k in keys:
        if k not in ALL:
            print(f"  ? unknown figure {k}"); continue
        try:
            ALL[k]()
        except Exception as e:
            print(f"  ✗ {k} failed: {e}")
    print("\nDone.\n")


if __name__ == "__main__":
    main()
