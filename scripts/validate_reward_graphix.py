"""
Validate the gflow-consistency REWARD against actual MBQC computation, using the
tested `graphix` library (not a hand-rolled simulator).

The reward `score_measurement_order` rewards orders that satisfy the gflow
ordering constraints; reward = 1 ⇔ a valid gflow order. This script verifies the
two facts that make that reward physically meaningful, with an independent,
tested engine:

  (A) gflow CROSS-VALIDATION — our `compute_gflow` agrees with graphix's
      `OpenGraph.find_gflow` on whether a gflow exists, across many random open
      graphs. This independently validates the algorithm the whole reward is
      built on.

  (B) gflow ⇒ DETERMINISM — for graphs with a gflow, the canonical pattern
      (correct order + corrections, built by graphix) yields an output that is
      IDENTICAL across random measurement-outcome branches. i.e. a valid gflow
      order (reward = 1) gives a genuinely deterministic, correct computation.
      This anchors the reward to real computational fidelity at its optimum.

Together: the reward's maximum corresponds to an actually-correct quantum
computation, and the gflow it is defined from is independently confirmed.

(Note: the order-RESOLVED graded correlation — fidelity of arbitrary invalid
orders — needs order-resolved pattern simulation, which is exactly the subtle,
error-prone part; we rely on graphix for the endpoints rather than hand-rolling
a wrong-order simulator that could mislead.)

Usage
-----
python scripts/validate_reward_graphix.py --n-graphs 200
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys

import numpy as np
import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graphix.opengraph import OpenGraph
from graphix.measurements import BlochMeasurement
from graphix.fundamentals import Plane

from mbqc_rl.utils.generators import make_grid_graph, make_brickwork_graph
from mbqc_rl.utils.gflow import compute_gflow


def _open_graph(G, outs, rows, cols, rng, clifford_fraction):
    """Build a graphix OpenGraph with random k·π/4 angles (mixed Clifford / non-
    Clifford). NO input nodes — matching our compute_gflow(G, outputs), which
    assumes I = ∅ (state-preparation pattern). gflow with inputs is stricter and
    would not match our convention. Outputs = rightmost column."""
    non_out = [v for v in G.nodes() if v not in set(outs)]
    meas = {}
    for v in non_out:
        if rng.random() < clifford_fraction:
            k = int(rng.choice([0, 2, 4, 6]))     # Clifford
        else:
            k = int(rng.choice([1, 3, 5, 7]))     # non-Clifford
        meas[v] = BlochMeasurement(k / 4.0, Plane.XY)   # angle in units of π
    return OpenGraph(G, input_nodes=[], output_nodes=outs, measurements=meas)


def _branch_determinism(pattern, k_branches, seed0):
    """Max output infidelity across k random outcome branches (0 = deterministic)."""
    states = []
    for s in range(k_branches):
        st = pattern.simulate_pattern("statevector",
                                      rng=np.random.default_rng(seed0 + s))
        v = np.asarray(st.flatten()).ravel()
        states.append(v / (np.linalg.norm(v) + 1e-12))
    worst = 0.0
    for a, b in itertools.combinations(states, 2):
        overlap = abs(np.vdot(a, b))               # phase-invariant
        worst = max(worst, 1.0 - overlap)
    return worst


def run(topology, rows, cols, n_graphs, defect_rate, clifford_fraction,
        k_branches, seed0):
    make = make_brickwork_graph if topology == "brickwork" else make_grid_graph
    agree = both_yes = both_no = disagree = 0
    det_spreads = []
    for i in range(n_graphs):
        rng = np.random.default_rng(seed0 + i)
        G, outs = make(rows, cols, defect_rate=defect_rate, rng=rng)
        og = _open_graph(G, outs, rows, cols, rng, clifford_fraction)
        ours = compute_gflow(G, outs)[0] is not None
        theirs = og.find_gflow() is not None
        agree += (ours == theirs)
        both_yes += (ours and theirs)
        both_no += ((not ours) and (not theirs))
        disagree += (ours != theirs)
        if theirs:
            det_spreads.append(_branch_determinism(og.to_pattern(),
                                                   k_branches, seed0 + 1000 * i))
    return {
        "topology": topology, "size": f"{rows}x{cols}", "defect": defect_rate,
        "n": n_graphs, "agree": agree, "both_gflow": both_yes,
        "both_nogflow": both_no, "disagree": disagree,
        "mean_det_infidelity": float(np.mean(det_spreads)) if det_spreads else None,
        "max_det_infidelity": float(np.max(det_spreads)) if det_spreads else None,
        "pct_deterministic": (float(np.mean([s < 1e-6 for s in det_spreads]) * 100)
                              if det_spreads else None),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-graphs", type=int, default=150)
    ap.add_argument("--k-branches", type=int, default=6)
    ap.add_argument("--clifford-fraction", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    configs = [
        ("grid", 3, 3, 0.0), ("grid", 3, 3, 0.1),
        ("grid", 4, 4, 0.0), ("grid", 4, 4, 0.1),
        ("brickwork", 3, 3, 0.0), ("brickwork", 4, 4, 0.0),
    ]
    print(f"\n{'='*78}")
    print("  REWARD VALIDATION via graphix  (non-Clifford angles, mixed)")
    print(f"{'='*78}")
    print(f"  {'config':16s} {'gflow agree':>12} {'both✓/✗/≠':>14} "
          f"{'det. infid.':>12} {'% determ.':>10}")
    print(f"  {'-'*16} {'-'*12} {'-'*14} {'-'*12} {'-'*10}")
    all_agree = True
    all_det = True
    for topo, r, c, d in configs:
        res = run(topo, r, c, args.n_graphs, d, args.clifford_fraction,
                  args.k_branches, args.seed)
        agree_pct = res["agree"] / res["n"] * 100
        all_agree &= (res["disagree"] == 0)
        if res["mean_det_infidelity"] is not None:
            all_det &= (res["max_det_infidelity"] < 1e-6)
        cfg = f"{topo} {res['size']} d={d}"
        det = ("%.2e" % res["mean_det_infidelity"]
               if res["mean_det_infidelity"] is not None else "n/a")
        pct = ("%.1f%%" % res["pct_deterministic"]
               if res["pct_deterministic"] is not None else "n/a")
        print(f"  {cfg:16s} {agree_pct:>11.1f}% "
              f"{res['both_gflow']:>4}/{res['both_nogflow']:>3}/{res['disagree']:>3}"
              f" {det:>12} {pct:>10}")

    print(f"\n  {'─'*60}")
    print(f"  (A) gflow cross-validation : "
          f"{'PASS — our gflow ≡ graphix on every graph' if all_agree else 'MISMATCH found'}")
    print(f"  (B) gflow ⇒ determinism    : "
          f"{'PASS — every gflow pattern deterministic (reward=1 ⇔ correct computation)' if all_det else 'non-deterministic case found'}")
    print(f"  {'─'*60}\n")


if __name__ == "__main__":
    main()
