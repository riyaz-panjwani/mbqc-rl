import numpy as np
import networkx as nx


def make_grid_graph(
    rows: int,
    cols: int,
    defect_rate: float = 0.0,
    rng: np.random.Generator = None,
) -> tuple[nx.Graph, list[int]]:
    """
    Create a rows × cols grid graph with Bernoulli edge deletions.

    Node index convention: node at grid position (r, c) → r * cols + c.
    Output qubits are the rightmost column (c = cols - 1).

    Args:
        rows: Number of grid rows. Must be >= 1.
        cols: Number of grid columns. Must be >= 2 (at least one non-output column).
        defect_rate: Probability that any given edge is deleted.
        rng: numpy random Generator; created from OS entropy if None.

    Returns:
        G: NetworkX graph with nodes 0 … rows*cols-1.
        output_qubits: List of node indices for the output (rightmost) column.
    """
    if cols < 2:
        raise ValueError("cols must be >= 2 so there is at least one non-output qubit.")
    if rows < 1:
        raise ValueError("rows must be >= 1.")

    if rng is None:
        rng = np.random.default_rng()

    n = rows * cols
    G = nx.Graph()
    G.add_nodes_from(range(n))

    # Horizontal edges (left–right within each row)
    for r in range(rows):
        for c in range(cols - 1):
            u = r * cols + c
            v = r * cols + c + 1
            if rng.random() >= defect_rate:
                G.add_edge(u, v)

    # Vertical edges (top–bottom within each column)
    for r in range(rows - 1):
        for c in range(cols):
            u = r * cols + c
            v = (r + 1) * cols + c
            if rng.random() >= defect_rate:
                G.add_edge(u, v)

    output_qubits = [r * cols + (cols - 1) for r in range(rows)]
    return G, output_qubits


def assign_measurement_angles(
    G: nx.Graph,
    output_qubits: list[int],
    clifford_fraction: float = 0.5,
    rng: np.random.Generator = None,
) -> dict[int, int]:
    """
    Assign each non-output qubit an XY-plane measurement angle k·π/4, k ∈ 0…7.

    Clifford angles  (k even: 0, π/2, π, 3π/2): Pauli corrections from earlier
    measurements can be tracked classically in the Pauli frame — the physical
    measurement basis never has to change, so ordering constraints on these
    qubits are free.

    Non-Clifford angles (k odd: π/4, 3π/4, 5π/4, 7π/4): corrections change the
    measurement basis non-trivially (a T-type rotation), so the qubit MUST be
    measured after the qubits that correct it — these are the hard ordering
    constraints. Non-Clifford angles are what make MBQC universal
    (Clifford-only circuits are classically simulable, Gottesman–Knill).

    Args:
        G: The graph (only its nodes are read).
        output_qubits: Outputs are not measured; they get angle k=0.
        clifford_fraction: Probability a qubit gets a Clifford angle.
        rng: numpy random Generator; created from OS entropy if None.

    Returns:
        angles: node → k  (angle = k·π/4). Stored on nodes as G.nodes[v]["angle_k"].
    """
    if rng is None:
        rng = np.random.default_rng()

    output_set = set(output_qubits)
    angles: dict[int, int] = {}

    for v in G.nodes():
        if v in output_set:
            angles[v] = 0
        elif rng.random() < clifford_fraction:
            angles[v] = int(rng.choice([0, 2, 4, 6]))   # Clifford: k even
        else:
            angles[v] = int(rng.choice([1, 3, 5, 7]))   # non-Clifford: k odd

    nx.set_node_attributes(G, angles, "angle_k")
    return angles


def is_clifford_angle(k: int) -> bool:
    """True if angle k·π/4 is a Clifford angle (k even)."""
    return k % 2 == 0
