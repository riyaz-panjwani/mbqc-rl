"""
Tests for the PyZX integration bridge.

These verify that:
1. NetworkX → PyZX conversion produces a well-formed ZX-diagram.
2. PyZX's gflow finds a gflow wherever our own implementation does.
3. Both implementations satisfy the partial-order constraint independently.
4. The compare utility correctly identifies agreement / disagreement.
"""

import networkx as nx
import pytest

from mbqc_rl.utils.generators import make_grid_graph
from mbqc_rl.utils.gflow import compute_gflow
from mbqc_rl.utils.pyzx_bridge import (
    nx_to_zx_graph,
    pyzx_gflow,
    compare_gflow_implementations,
)


# ---------------------------------------------------------------------------
# nx_to_zx_graph
# ---------------------------------------------------------------------------

def test_zx_graph_vertex_count():
    G, output_qubits = make_grid_graph(3, 3, defect_rate=0.0)
    g, vertex_map = nx_to_zx_graph(G, output_qubits)
    n_qubits = G.number_of_nodes()
    n_outputs = len(output_qubits)
    # Each qubit gets a Z-spider + each output gets an extra BOUNDARY vertex
    assert g.num_vertices() == n_qubits + n_outputs


def test_zx_graph_outputs_are_set():
    G, output_qubits = make_grid_graph(3, 3, defect_rate=0.0)
    g, _ = nx_to_zx_graph(G, output_qubits)
    assert len(g.outputs()) == len(output_qubits)


def test_zx_graph_vertex_map_complete():
    G, output_qubits = make_grid_graph(3, 3, defect_rate=0.0)
    _, vertex_map = nx_to_zx_graph(G, output_qubits)
    for node in G.nodes():
        assert node in vertex_map


# ---------------------------------------------------------------------------
# pyzx_gflow — existence matches our implementation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rows,cols", [(2, 3), (3, 3), (3, 4), (4, 4)])
def test_pyzx_agrees_on_existence_perfect_grid(rows, cols):
    G, output_qubits = make_grid_graph(rows, cols, defect_rate=0.0)
    our_gf, our_ord = compute_gflow(G, output_qubits)
    pyzx_gf, pyzx_ord = pyzx_gflow(G, output_qubits)

    assert (our_gf is None) == (pyzx_gf is None), (
        f"Implementations disagree on gflow existence for {rows}×{cols} perfect grid."
    )


def test_pyzx_gflow_covers_all_non_output_qubits():
    G, output_qubits = make_grid_graph(3, 3, defect_rate=0.0)
    pyzx_gf, _ = pyzx_gflow(G, output_qubits)
    assert pyzx_gf is not None
    output_set = set(output_qubits)
    non_output = {n for n in G.nodes() if n not in output_set}
    assert set(pyzx_gf.keys()) == non_output


def test_pyzx_order_constraint_holds():
    """Every correction set member must have a lower layer than the qubit."""
    G, output_qubits = make_grid_graph(3, 3, defect_rate=0.0)
    pyzx_gf, pyzx_ord = pyzx_gflow(G, output_qubits)
    assert pyzx_gf is not None
    output_set = set(output_qubits)
    for v, g_v in pyzx_gf.items():
        for w in g_v:
            if w not in output_set:
                assert pyzx_ord[w] < pyzx_ord[v], (
                    f"pyzx: g({v}) contains {w} but order[{w}]={pyzx_ord[w]} "
                    f">= order[{v}]={pyzx_ord[v]}"
                )


# ---------------------------------------------------------------------------
# compare_gflow_implementations
# ---------------------------------------------------------------------------

def test_compare_perfect_grid_both_agree():
    G, output_qubits = make_grid_graph(3, 3, defect_rate=0.0)
    our_gf, our_ord = compute_gflow(G, output_qubits)
    result = compare_gflow_implementations(G, output_qubits, our_gf, our_ord)

    assert result["exist"] is True
    assert result["agree"] is True
    assert result["our_order_valid"] is True
    assert result["pyzx_order_valid"] is True


def test_compare_no_gflow_both_agree():
    """A graph where no gflow should exist: both should return None."""
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2])
    # qubit 0 is isolated from output 2 — no gflow possible for qubit 0
    G.add_edge(1, 2)
    output_qubits = [2]
    our_gf, our_ord = compute_gflow(G, output_qubits)
    result = compare_gflow_implementations(G, output_qubits, our_gf, our_ord)

    assert result["agree"] is True


def test_compare_4x4_grid():
    G, output_qubits = make_grid_graph(4, 4, defect_rate=0.0)
    our_gf, our_ord = compute_gflow(G, output_qubits)
    result = compare_gflow_implementations(G, output_qubits, our_gf, our_ord)
    assert result["agree"] is True
    assert result["our_order_valid"] is True
    assert result["pyzx_order_valid"] is True


def test_pyzx_gflow_defective_grid_does_not_crash():
    """Defective grids may or may not have a gflow; the bridge must not crash."""
    G, output_qubits = make_grid_graph(3, 3, defect_rate=0.3, rng=__import__('numpy').random.default_rng(99))
    pyzx_gf, pyzx_ord = pyzx_gflow(G, output_qubits)
    # Just verify it returns the right types
    assert pyzx_gf is None or isinstance(pyzx_gf, dict)
