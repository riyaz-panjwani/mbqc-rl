"""
PyZX integration bridge.

Converts our NetworkX-based MBQC graphs into PyZX ZX-diagrams and runs
PyZX's own gflow algorithm as an independent cross-check against our GF(2)
implementation.

Why PyZX?
---------
PyZX represents quantum states in the ZX-calculus where graph states map
directly to ZX-diagrams: each qubit is a Z-spider, entangling CZ operations
become Hadamard edges, and measurement angles become spider phases. This is
the simulation and verification layer described in the dissertation proposal
(Kissinger and van de Wetering, 2020). The bridge is also the hook for future
phases when continuous measurement angles need to be tracked per node.

Convention differences
----------------------
PyZX gflow layers: 0 = measured first, max = output (measured last / never).
Our gflow layers:  0 = output, higher = measured earlier.
The bridge translates between them transparently.
"""

from __future__ import annotations

import pyzx
from pyzx.gflow import gflow as _pyzx_gflow
import networkx as nx


def nx_to_zx_graph(
    G_nx: nx.Graph,
    output_qubits: list[int],
) -> tuple[pyzx.Graph, dict[int, int]]:
    """
    Convert a NetworkX MBQC graph to a PyZX ZX-diagram (graph state).

    Mapping:
    - Each NetworkX node  → Z-spider (phase 0, representing |+⟩)
    - Each NetworkX edge  → Hadamard edge (CZ in graph-state prep)
    - Each output qubit   → Z-spider connected to a BOUNDARY vertex

    PyZX's gflow algorithm requires BOUNDARY vertices to identify outputs.
    Input boundaries are left empty (no specific input state is injected).

    Returns
    -------
    g : pyzx.Graph  — the ZX-diagram
    vertex_map : dict mapping NetworkX node index → PyZX vertex index
    """
    g = pyzx.Graph()
    vertex_map: dict[int, int] = {}
    output_set = set(output_qubits)

    for node in sorted(G_nx.nodes()):
        v = g.add_vertex(pyzx.VertexType.Z, qubit=node, row=0)
        vertex_map[node] = v

    for u, v in G_nx.edges():
        g.add_edge(g.edge(vertex_map[u], vertex_map[v]), pyzx.EdgeType.HADAMARD)

    boundary_vertices: list[int] = []
    for q in sorted(output_set):
        b = g.add_vertex(pyzx.VertexType.BOUNDARY, qubit=q, row=1)
        g.add_edge(g.edge(vertex_map[q], b), pyzx.EdgeType.SIMPLE)
        boundary_vertices.append(b)

    g.set_inputs([])
    g.set_outputs(boundary_vertices)

    return g, vertex_map


def pyzx_gflow(
    G_nx: nx.Graph,
    output_qubits: list[int],
) -> tuple[dict[int, set[int]], dict[int, int]] | tuple[None, None]:
    """
    Compute gflow using PyZX's implementation, translated back to our format.

    Returns
    -------
    (gflow_map, order) in the same convention as compute_gflow() in gflow.py:
        gflow_map : node → correction set (sets of node indices)
        order     : node → layer  (output = 0, higher layer = measured earlier)
    (None, None) if no gflow exists.
    """
    g, vertex_map = nx_to_zx_graph(G_nx, output_qubits)
    reverse_map = {v: n for n, v in vertex_map.items()}

    result = _pyzx_gflow(g)
    if result is None:
        return None, None

    pyzx_order_raw, pyzx_corrections = result

    # PyZX layer convention: 0 = first measured, max = output/never measured.
    # Our convention: 0 = output, higher = measured earlier.
    # Translation: our_layer(v) = max_pyzx_layer - pyzx_layer(v)
    max_layer = max(pyzx_order_raw.values()) if pyzx_order_raw else 0

    order: dict[int, int] = {}
    for pyzx_v, pyzx_layer in pyzx_order_raw.items():
        if pyzx_v in reverse_map:
            order[reverse_map[pyzx_v]] = max_layer - pyzx_layer

    gflow_map: dict[int, set[int]] = {}
    output_set = set(output_qubits)
    for pyzx_v, corr_set in pyzx_corrections.items():
        if pyzx_v in reverse_map:
            node = reverse_map[pyzx_v]
            gflow_map[node] = {
                reverse_map[cv] for cv in corr_set if cv in reverse_map
            }

    return gflow_map, order


def compare_gflow_implementations(
    G_nx: nx.Graph,
    output_qubits: list[int],
    our_gflow: dict[int, set[int]] | None,
    our_order: dict[int, int] | None,
) -> dict:
    """
    Run PyZX's gflow on the same graph and report whether both implementations
    agree on existence, covered qubits, and ordering-constraint validity.

    Both implementations may produce different (but equally valid) correction
    sets since gflow is not unique. Agreement is checked on:
      1. Whether a gflow exists at all.
      2. Whether both cover the same set of non-output qubits.
      3. Whether each gflow independently satisfies the partial-order constraint.

    Returns a dict with keys: exist, agree, our_order_valid, pyzx_order_valid.
    """
    pyzx_gflow_map, pyzx_order = pyzx_gflow(G_nx, output_qubits)
    output_set = set(output_qubits)

    if our_gflow is None and pyzx_gflow_map is None:
        return {"exist": False, "agree": True}
    if (our_gflow is None) != (pyzx_gflow_map is None):
        return {
            "exist": True,
            "agree": False,
            "reason": "implementations disagree on existence",
            "ours_found": our_gflow is not None,
            "pyzx_found": pyzx_gflow_map is not None,
        }

    our_keys = set(our_gflow) if our_gflow else set()
    pyzx_keys = set(pyzx_gflow_map) if pyzx_gflow_map else set()

    def order_constraint_holds(gf: dict, ord_: dict) -> bool:
        for v, g_v in gf.items():
            for w in g_v:
                if w not in output_set and ord_.get(w, 0) >= ord_.get(v, 0):
                    return False
        return True

    return {
        "exist": True,
        "agree": our_keys == pyzx_keys,
        "our_qubit_count": len(our_keys),
        "pyzx_qubit_count": len(pyzx_keys),
        "our_order_valid": order_constraint_holds(our_gflow, our_order),
        "pyzx_order_valid": order_constraint_holds(pyzx_gflow_map, pyzx_order),
    }
