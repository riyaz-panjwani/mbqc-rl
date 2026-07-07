"""
Build the MBQC defective-instance benchmark — a reusable, labelled dataset.

This materialises the "1,000 randomised defective grid instances" named in the
project proposal into a fixed, versioned, self-contained benchmark that future
work can evaluate against WITHOUT rerunning any of this code. Each instance is
generated through the exact same `MBQCEnv` path used in all experiments (seed →
`np.random.default_rng(seed)` → `make_grid_graph`/`make_irregular_flow_graph`),
so the dataset is bit-for-bit the distribution the agent/oracle/heuristic were
evaluated on — and fully reproducible from the seeds recorded in every record.

Each record carries:
  • the graph (edge list) + output qubits + defect rate + seed + topology/size
  • ground-truth GFLOW labels: per-node layer order (the oracle schedule) and the
    correction sets g(v)  — computed and cross-validated elsewhere against graphix
  • reference rewards on the gflow-consistency objective: oracle / distance
    heuristic / random — so a new method can be compared on a common yardstick.

Two files are shipped (composition below):
  grids_defective_n1000.jsonl — the proposal's 1,000 defective GRID instances
                                 (4 sizes × 5 defect rates × 50), the deliverable.
  irregular_n500.jsonl        — 500 irregular flow-graph instances (the regime
                                 where the learned policy beats the heuristic).

Usage
-----
python scripts/build_benchmark.py                 # build both + manifest + README
python scripts/build_benchmark.py --only grids    # just the grid set
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mbqc_rl.env.mbqc_env import MBQCEnv
from mbqc_rl.baselines.classical import (
    GreedyGflowBaseline, DistanceScheduleBaseline, RandomBaseline,
)

OUT = Path("benchmark")
SCHEMA_VERSION = "1.0"

# ── Canonical composition ───────────────────────────────────────────────────
# GRID benchmark: 4 sizes × 5 defect rates × 50 = 1,000 defective grid instances.
GRID_SIZES = [(3, 3), (4, 4), (5, 5), (6, 6)]
GRID_RATES = [0.01, 0.05, 0.10, 0.20, 0.30]
GRID_PER_CELL = 50
GRID_SEED_BASE = 1_000_000          # held-out: disjoint from training & from the
#                                     early-stopping val/test seeds (50k, 90k–500k)
# IRREGULAR benchmark: clean flow graphs (the regime where RL beats the heuristic).
IRR_SIZES = [(4, 4), (5, 5), (6, 6)]
IRR_PER_CELL = 167                  # ≈500 total (3 × 167 = 501)
IRR_SEED_BASE = 2_000_000


def _rollout(env, baseline, seed):
    obs, info = env.reset(seed=seed)
    baseline.reset(obs, info)
    done, total = False, 0.0
    while not done:
        obs, r, t, tr, _ = env.step(baseline.select_action(obs))
        total += r
        done = t or tr
    return float(total)


def _instance(env, rows, cols, defect_rate, topology, seed):
    """Generate one instance through the env and label it."""
    env.reset(seed=seed)
    G = env._graph
    outs = sorted(env._output_set)
    gmap, gorder = env._gflow, env._gflow_order
    exists = gmap is not None

    edges = sorted(sorted((int(u), int(v))) for u, v in G.edges())
    rec = {
        "id": f"{topology}_{rows}x{cols}_d{defect_rate:g}_s{seed}",
        "topology": topology,
        "rows": rows, "cols": cols, "n_qubits": rows * cols,
        "defect_rate": defect_rate,
        "seed": seed,
        "edges": edges,
        "output_qubits": [int(o) for o in outs],
        "gflow_exists": bool(exists),
        "gflow_order": ({str(k): int(v) for k, v in gorder.items()}
                        if exists else None),
        "gflow_correction_sets": (
            {str(k): sorted(int(x) for x in v) for k, v in gmap.items()}
            if exists else None),
    }
    # reference rewards on the gflow-consistency objective (same env, same seed)
    rec["reference_reward"] = {
        "oracle": _rollout(env, GreedyGflowBaseline(env), seed),
        "distance_heuristic": _rollout(env, DistanceScheduleBaseline(env), seed),
        "random": _rollout(env, RandomBaseline(np.random.default_rng(seed)), seed),
    }
    return rec


def _build_set(name, cells, topology):
    """cells: list of (rows, cols, defect_rate, seed0, count)."""
    records, t0 = [], time.time()
    for rows, cols, rate, seed0, count in cells:
        env = MBQCEnv(rows=rows, cols=cols, defect_rate=rate, topology=topology)
        for i in range(count):
            records.append(_instance(env, rows, cols, rate, topology, seed0 + i))
        print(f"  {topology} {rows}x{cols} d={rate:<4} ×{count}  "
              f"(gflow {sum(r['gflow_exists'] for r in records[-count:])}/{count})")
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.jsonl"
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"  → {path}  ({len(records)} instances, {time.time()-t0:.1f}s, "
          f"sha256 {sha[:12]}…)")
    return records, path.name, sha


def _cell_stats(records):
    """Per-(size, rate) aggregate for the manifest."""
    stats = {}
    for r in records:
        key = f"{r['rows']}x{r['cols']}_d{r['defect_rate']:g}"
        s = stats.setdefault(key, dict(n=0, gflow=0, orc=0.0, heur=0.0, rnd=0.0))
        s["n"] += 1
        s["gflow"] += int(r["gflow_exists"])
        s["orc"]  += r["reference_reward"]["oracle"]
        s["heur"] += r["reference_reward"]["distance_heuristic"]
        s["rnd"]  += r["reference_reward"]["random"]
    for s in stats.values():
        n = s["n"]
        s["gflow_exists_rate"] = round(s.pop("gflow") / n, 4)
        s["mean_oracle"]    = round(s.pop("orc")  / n, 4)
        s["mean_heuristic"] = round(s.pop("heur") / n, 4)
        s["mean_random"]    = round(s.pop("rnd")  / n, 4)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["grids", "irregular"], default=None)
    args = ap.parse_args()

    print(f"\n{'='*66}\n  BUILDING MBQC BENCHMARK (schema v{SCHEMA_VERSION})\n{'='*66}")
    manifest = {"schema_version": SCHEMA_VERSION,
                "generator": "MBQCEnv seed→graph; gflow via mbqc_rl.utils.gflow "
                             "(cross-validated vs graphix, RESULTS.md §8)",
                "objective": "gflow-consistency reward in [0,1]; 1.0 = fully "
                             "deterministic (valid) measurement order",
                "files": {}}

    if args.only in (None, "grids"):
        print("\nGRID set (proposal deliverable — 1,000 defective grid instances):")
        cells = [(r, c, rate, GRID_SEED_BASE + si * 10_000 + ri * 2_000, GRID_PER_CELL)
                 for si, (r, c) in enumerate(GRID_SIZES)
                 for ri, rate in enumerate(GRID_RATES)]
        recs, fname, sha = _build_set("grids_defective_n1000", cells, "grid")
        manifest["files"][fname] = {
            "n_instances": len(recs), "sha256": sha,
            "composition": f"{len(GRID_SIZES)} sizes × {len(GRID_RATES)} defect "
                           f"rates × {GRID_PER_CELL}",
            "sizes": [f"{r}x{c}" for r, c in GRID_SIZES],
            "defect_rates": GRID_RATES,
            "seed_base": GRID_SEED_BASE,
            "cells": _cell_stats(recs)}

    if args.only in (None, "irregular"):
        print("\nIRREGULAR set (headline regime — RL beats the heuristic):")
        cells = [(r, c, 0.0, IRR_SEED_BASE + si * 10_000, IRR_PER_CELL)
                 for si, (r, c) in enumerate(IRR_SIZES)]
        recs, fname, sha = _build_set("irregular_n500", cells, "irregular")
        manifest["files"][fname] = {
            "n_instances": len(recs), "sha256": sha,
            "composition": f"{len(IRR_SIZES)} sizes × {IRR_PER_CELL} (clean, "
                           "defect_rate=0)",
            "sizes": [f"{r}x{c}" for r, c in IRR_SIZES],
            "seed_base": IRR_SEED_BASE,
            "cells": _cell_stats(recs)}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n  → {OUT}/manifest.json")
    print(f"{'='*66}\n")


if __name__ == "__main__":
    main()
