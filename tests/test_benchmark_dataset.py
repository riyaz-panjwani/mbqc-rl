"""
Integrity tests for the shipped MBQC benchmark (benchmark/*.jsonl).

These guarantee the deliverable is self-consistent: every record loads, its graph
reconstructs, and its stored gflow labels reproduce a fresh recompute — so the
dataset can be trusted by downstream/future work without rerunning generation.
"""

import json
from pathlib import Path

import pytest

from mbqc_rl.benchmark import load_benchmark, instance_to_graph, gflow_order
from mbqc_rl.utils.gflow import compute_gflow

BENCH = Path(__file__).resolve().parent.parent / "benchmark"
FILES = ["grids_defective_n1000.jsonl", "irregular_n500.jsonl"]

pytestmark = pytest.mark.skipif(
    not (BENCH / FILES[0]).exists(),
    reason="benchmark not built (run scripts/build_benchmark.py)")


@pytest.fixture(scope="module")
def manifest():
    return json.loads((BENCH / "manifest.json").read_text())


@pytest.mark.parametrize("fname", FILES)
def test_counts_match_manifest(fname, manifest):
    insts = load_benchmark(BENCH / fname)
    assert len(insts) == manifest["files"][fname]["n_instances"]


def test_grid_set_is_1000():
    assert len(load_benchmark(BENCH / "grids_defective_n1000.jsonl")) == 1000


@pytest.mark.parametrize("fname", FILES)
def test_graph_reconstructs(fname):
    for r in load_benchmark(BENCH / fname):
        G, outs = instance_to_graph(r)
        assert G.number_of_nodes() == r["n_qubits"]
        assert G.number_of_edges() == len(r["edges"])
        assert set(outs) == set(r["output_qubits"])
        assert len(outs) == r["rows"]           # outputs = one per row (last column)


@pytest.mark.parametrize("fname", FILES)
def test_gflow_labels_reproduce(fname):
    """The stored gflow-existence flag and layer order must match a fresh
    recompute on the reconstructed graph — the core correctness guarantee."""
    for r in load_benchmark(BENCH / fname):
        G, outs = instance_to_graph(r)
        gmap, gorder = compute_gflow(G, outs)
        assert (gmap is not None) == r["gflow_exists"], r["id"]
        if r["gflow_exists"]:
            assert {int(k): v for k, v in gorder.items()} == gflow_order(r), r["id"]


@pytest.mark.parametrize("fname", FILES)
def test_reference_rewards_bounded_and_ordered(fname):
    insts = load_benchmark(BENCH / fname)
    for r in insts:
        rr = r["reference_reward"]
        for v in rr.values():
            assert 0.0 <= v <= 1.0 + 1e-9, r["id"]
        # an instance with a valid gflow: the oracle must be perfect
        if r["gflow_exists"]:
            assert rr["oracle"] >= 0.999, r["id"]
    # on aggregate the oracle dominates the heuristic, which dominates random
    mean = lambda k: sum(r["reference_reward"][k] for r in insts) / len(insts)
    assert mean("oracle") >= mean("distance_heuristic") >= mean("random")
