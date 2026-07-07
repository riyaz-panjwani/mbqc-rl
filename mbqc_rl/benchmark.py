"""
Loader for the shipped MBQC defective-instance benchmark (see `benchmark/`).

The benchmark is a fixed, versioned dataset of defective graph-state instances
labelled with ground-truth gflow orderings and reference baseline rewards, built
by `scripts/build_benchmark.py`. This module lets any downstream code load an
instance and reconstruct its graph without depending on the generators or seeds.

Example
-------
>>> from mbqc_rl.benchmark import load_benchmark, instance_to_graph
>>> insts = load_benchmark("benchmark/grids_defective_n1000.jsonl")
>>> G, outputs = instance_to_graph(insts[0])
>>> insts[0]["reference_reward"]["distance_heuristic"]   # heuristic yardstick
"""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx


def load_benchmark(path: str | Path) -> list[dict]:
    """Load a benchmark `.jsonl` file into a list of instance records."""
    path = Path(path)
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def instance_to_graph(record: dict) -> tuple[nx.Graph, list[int]]:
    """Reconstruct (graph, output_qubits) from a benchmark record.

    The graph has nodes 0 … n_qubits-1 (isolated nodes included) and exactly the
    stored edges; `output_qubits` is the list of output node ids.
    """
    G = nx.Graph()
    G.add_nodes_from(range(record["n_qubits"]))
    G.add_edges_from((u, v) for u, v in record["edges"])
    return G, list(record["output_qubits"])


def gflow_order(record: dict) -> dict[int, int] | None:
    """Ground-truth gflow layer per node (output=0, higher=measured earlier), or
    None if the instance has no valid gflow."""
    go = record.get("gflow_order")
    return None if go is None else {int(k): int(v) for k, v in go.items()}
