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
