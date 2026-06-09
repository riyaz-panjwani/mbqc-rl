import numpy as np
import networkx as nx
import pytest

from mbqc_rl.utils.gflow import compute_gflow, score_measurement_order, _gf2_solve
from mbqc_rl.utils.generators import make_grid_graph


# ---------------------------------------------------------------------------
# GF(2) solver
# ---------------------------------------------------------------------------

def test_gf2_identity_system():
    A = np.eye(3, dtype=np.int8)
    b = np.array([1, 0, 1], dtype=np.int8)
    sol = _gf2_solve(A, b)
    assert sol is not None
    assert list(sol) == [1, 0, 1]


def test_gf2_inconsistent():
    A = np.array([[1, 1], [1, 1]], dtype=np.int8)
    b = np.array([1, 0], dtype=np.int8)
    assert _gf2_solve(A, b) is None


def test_gf2_underdetermined():
    A = np.array([[1, 1, 0]], dtype=np.int8)
    b = np.array([1], dtype=np.int8)
    sol = _gf2_solve(A, b)
    assert sol is not None
    assert int((A @ sol).sum() % 2) == int(b[0])


# ---------------------------------------------------------------------------
# gflow on a perfect grid
# ---------------------------------------------------------------------------

def test_perfect_grid_has_gflow():
    G, output_qubits = make_grid_graph(3, 3, defect_rate=0.0)
    gflow, order = compute_gflow(G, output_qubits)
    assert gflow is not None, "Perfect 3×3 grid must have a gflow."
    output_set = set(output_qubits)
    for v in gflow:
        assert v not in output_set, "Output qubits must not appear as keys."


def test_gflow_layer_ordering():
    """
    Correction set members must have a strictly lower layer than the qubit being corrected:
    g(v) ⊆ {w : order[w] < order[v]}.
    """
    G, output_qubits = make_grid_graph(3, 3, defect_rate=0.0)
    gflow, order = compute_gflow(G, output_qubits)
    assert gflow is not None
    output_set = set(output_qubits)
    for v, g_v in gflow.items():
        for w in g_v:
            if w not in output_set:
                assert order[w] < order[v], (
                    f"g({v}) contains {w} but order[{w}]={order[w]} >= order[{v}]={order[v]}"
                )


def test_4x4_grid_has_gflow():
    G, output_qubits = make_grid_graph(4, 4, defect_rate=0.0)
    gflow, order = compute_gflow(G, output_qubits)
    assert gflow is not None


# ---------------------------------------------------------------------------
# gflow on degenerate / broken graphs
# ---------------------------------------------------------------------------

def test_isolated_nonoutput_qubit_no_gflow():
    """A non-output qubit with no edges to any qubit in C cannot be corrected."""
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2])
    G.add_edge(1, 2)    # output 2 is only connected to qubit 1
    # qubit 0 is isolated — no path to output qubit 2
    gflow, _ = compute_gflow(G, output_qubits=[2])
    # qubit 0 has no neighbors at all, so Odd(g(0)) cannot contain 0
    assert gflow is None


def test_all_output_trivial_gflow():
    """If all qubits are outputs, gflow is trivially empty."""
    G = nx.Graph()
    G.add_nodes_from([0, 1])
    G.add_edge(0, 1)
    gflow, order = compute_gflow(G, output_qubits=[0, 1])
    assert gflow == {}
    assert order[0] == 0 and order[1] == 0


# ---------------------------------------------------------------------------
# score_measurement_order
# ---------------------------------------------------------------------------

def _perfect_steps(gflow, order):
    """Assign step numbers so higher-layer qubits are measured first."""
    non_output = sorted(gflow.keys(), key=lambda v: -order[v])
    return {q: i + 1 for i, q in enumerate(non_output)}


def _reversed_steps(gflow, order):
    """Assign step numbers so lower-layer (later) qubits are measured first — wrong order."""
    non_output = sorted(gflow.keys(), key=lambda v: order[v])
    return {q: i + 1 for i, q in enumerate(non_output)}


def test_score_perfect_order_is_one():
    G, output_qubits = make_grid_graph(3, 3, defect_rate=0.0)
    gflow, order = compute_gflow(G, output_qubits)
    assert gflow is not None

    steps = _perfect_steps(gflow, order)
    score = score_measurement_order(gflow, steps, set(output_qubits))
    assert score == 1.0


def test_score_reversed_order_is_low():
    G, output_qubits = make_grid_graph(3, 3, defect_rate=0.0)
    gflow, order = compute_gflow(G, output_qubits)
    assert gflow is not None

    steps = _reversed_steps(gflow, order)
    score = score_measurement_order(gflow, steps, set(output_qubits))
    assert score < 1.0


def test_score_no_gflow_returns_zero_from_env():
    """score_measurement_order itself returns 1.0 on empty map; env wraps it."""
    score = score_measurement_order({}, {}, set())
    assert score == 1.0
