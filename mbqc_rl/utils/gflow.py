"""
gflow computation for open graph states.

Implements Algorithm 1 from:
  Browne, Kashefi, Mhalla, Perdrix (2007). Generalized flow and determinism
  in measurement-based quantum computation. New Journal of Physics, 9:250.

Conventions
-----------
- layer 0  → output qubits (not measured)
- layer k  → qubits measured in the k-th round; higher k = measured earlier
- g(v) ⊆ {qubits with layer < layer(v)} ∪ output_set
  i.e., the correction set consists of qubits measured AFTER v (or not measured)
"""

from __future__ import annotations

import numpy as np
import networkx as nx


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_gflow(
    G: nx.Graph,
    output_qubits: list[int],
) -> tuple[dict[int, set[int]], dict[int, int]] | tuple[None, None]:
    """
    Compute a gflow for open graph (G, output_qubits).

    Returns
    -------
    (gflow_map, order)  if a gflow exists, where
        gflow_map : v → correction set g(v)  for each non-output qubit v
        order     : v → layer number          (output = 0, higher = earlier)
    (None, None) if no gflow exists on this graph.
    """
    all_qubits = set(G.nodes())
    output_set = set(output_qubits) & all_qubits
    non_output_sorted = sorted(all_qubits - output_set)

    if not non_output_sorted:
        return {}, {v: 0 for v in output_set}

    gflow_map: dict[int, set[int]] = {}
    order: dict[int, int] = {v: 0 for v in output_set}

    C = set(output_set)        # processed: their indices may appear in correction sets
    remaining = set(non_output_sorted)
    layer = 1

    while remaining:
        C_list = sorted(C)
        k = len(C_list)
        if k == 0:
            return None, None

        # Build GF(2) constraint matrix.
        # Row i = non_output_sorted[i], col j = C_list[j].
        # A[i,j] = 1 iff qubit i is adjacent to qubit j.
        m = len(non_output_sorted)
        A = np.zeros((m, k), dtype=np.int8)
        for i, u in enumerate(non_output_sorted):
            for j, w in enumerate(C_list):
                if G.has_edge(u, w):
                    A[i, j] = 1

        # For each remaining qubit v: solve A x = e_v over GF(2).
        # A solution x gives g(v) = {C_list[j] : x[j] = 1}.
        # The constraint Odd(g(v)) ∩ (V\O) = {v} translates to exactly this system.
        newly_assigned: dict[int, set[int]] = {}
        for v in sorted(remaining):
            v_idx = non_output_sorted.index(v)
            b = np.zeros(m, dtype=np.int8)
            b[v_idx] = 1
            sol = _gf2_solve(A, b)
            if sol is not None:
                g_v = {C_list[j] for j in range(k) if sol[j] == 1}
                newly_assigned[v] = g_v

        if not newly_assigned:
            return None, None

        for v, g_v in newly_assigned.items():
            gflow_map[v] = g_v
            order[v] = layer
            C.add(v)

        remaining -= set(newly_assigned.keys())
        layer += 1

    return gflow_map, order


def score_measurement_order(
    gflow_map: dict[int, set[int]],
    measurement_steps: dict[int, int],
    output_set: set[int],
) -> float:
    """
    Compute the fraction of gflow ordering constraints satisfied by the agent.

    A constraint for qubit v is: step_of[v] < step_of[w]  for every
    non-output w in g(v)  (v must be measured before its correction targets).

    Returns a float in [0, 1].  Returns 1.0 if gflow_map is empty.
    """
    total = 0
    satisfied = 0

    for v, g_v in gflow_map.items():
        if v not in measurement_steps:
            continue
        step_v = measurement_steps[v]
        for w in g_v:
            if w in output_set:
                continue
            total += 1
            if w in measurement_steps and step_v < measurement_steps[w]:
                satisfied += 1

    return 1.0 if total == 0 else satisfied / total


def score_measurement_order_angled(
    gflow_map: dict[int, set[int]],
    measurement_steps: dict[int, int],
    output_set: set[int],
    angles: dict[int, int],
) -> float:
    """
    Angle-aware ordering score for universal (non-Clifford) MBQC patterns.

    Physics: when qubit v is measured, corrections land on the qubits in
    g(v). If a correction target w has a CLIFFORD angle (k even), the
    correction is a Pauli that can be tracked classically — w's physical
    measurement basis never changes, so the constraint v ≺ w is free.
    If w has a NON-CLIFFORD angle (k odd), the correction changes w's
    measurement basis non-trivially, so w genuinely must be measured
    after v — a hard constraint.

    Only hard constraints are scored. Returns fraction satisfied ∈ [0, 1];
    1.0 if there are no hard constraints (fully Clifford pattern — any
    order works, consistent with Gottesman–Knill).
    """
    total = 0
    satisfied = 0

    for v, g_v in gflow_map.items():
        if v not in measurement_steps:
            continue
        step_v = measurement_steps[v]
        for w in g_v:
            if w in output_set:
                continue
            if angles.get(w, 0) % 2 == 0:      # Clifford target → free
                continue
            total += 1
            if w in measurement_steps and step_v < measurement_steps[w]:
                satisfied += 1

    return 1.0 if total == 0 else satisfied / total


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _gf2_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray | None:
    """
    Solve A x = b over GF(2) via Gaussian elimination.

    Returns a solution x (free variables set to 0) or None if inconsistent.
    """
    m, n = A.shape
    aug = np.hstack([A, b.reshape(-1, 1)]).astype(np.int8)

    pivot_info: list[tuple[int, int]] = []   # (pivot_row, pivot_col)
    row = 0

    for col in range(n):
        # Find a pivot in this column at or below 'row'
        pivot = next((r for r in range(row, m) if aug[r, col] == 1), None)
        if pivot is None:
            continue

        aug[[row, pivot]] = aug[[pivot, row]]
        pivot_info.append((row, col))

        # Eliminate all other rows
        for r in range(m):
            if r != row and aug[r, col] == 1:
                aug[r] = (aug[r] + aug[row]) % 2

        row += 1

    # Check consistency: any row 0…0 | 1 means no solution
    for r in range(row, m):
        if aug[r, n] == 1:
            return None

    x = np.zeros(n, dtype=np.int8)
    for pivot_row, pivot_col in pivot_info:
        x[pivot_col] = aug[pivot_row, n]
    return x
