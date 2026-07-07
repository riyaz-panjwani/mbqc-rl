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


def make_brickwork_graph(
    rows: int,
    cols: int,
    defect_rate: float = 0.0,
    rng: np.random.Generator = None,
) -> tuple[nx.Graph, list[int]]:
    """
    Create a rows × cols brickwork graph state with Bernoulli edge deletions.

    Brickwork states (Broadbent, Fitzsimons & Kashefi, 2009, "Universal Blind
    Quantum Computation") are the canonical *universal* resource state for
    measurement-based quantum computing — every blind-MBQC protocol runs on a
    brickwork lattice rather than a plain cluster (grid) state. The defining
    feature is a *staggered* pattern of vertical couplings, so the graph tiles
    the plane like a brick wall.

    Construction (node (r, c) → r * cols + c, identical indexing to the grid):
      • Horizontal edges: every (r, c)–(r, c+1) is present — the logical wires
        carrying information left-to-right toward the output column.
      • Vertical "rungs": (r, c)–(r+1, c) is present iff (r + c) is odd. This
        places rungs at alternating columns on successive rows, giving 2-wide
        bricks offset by one cell per course — the standard brick-wall tiling.

    Compared with the grid (which has *every* vertical edge), the brickwork
    state keeps only half of them in a staggered layout. It is therefore a
    genuinely different topology, while remaining a valid MBQC resource with
    a well-defined gflow (verified empirically in the tests).

    Output qubits are the rightmost column (c = cols - 1), as for the grid, so
    the same gflow machinery, environment and policy apply unchanged — a policy
    trained on grids can be evaluated here zero-shot.

    Args:
        rows: Number of rows. Must be >= 1.
        cols: Number of columns. Must be >= 2 (at least one non-output column).
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

    # Horizontal edges — full row paths (the logical wires)
    for r in range(rows):
        for c in range(cols - 1):
            u = r * cols + c
            v = r * cols + c + 1
            if rng.random() >= defect_rate:
                G.add_edge(u, v)

    # Vertical rungs — staggered brick pattern: present iff (r + c) is odd
    for r in range(rows - 1):
        for c in range(cols):
            if (r + c) % 2 == 1:
                u = r * cols + c
                v = (r + 1) * cols + c
                if rng.random() >= defect_rate:
                    G.add_edge(u, v)

    output_qubits = [r * cols + (cols - 1) for r in range(rows)]
    return G, output_qubits


def make_irregular_flow_graph(
    rows: int,
    cols: int,
    defect_rate: float = 0.0,
    rng: np.random.Generator = None,
    cross_prob: float = 0.35,
    skip_prob: float = 0.45,
    _max_tries: int = 400,
) -> tuple[nx.Graph, list[int]]:
    """
    Create an *irregular* graph state that still has a valid gflow, but whose
    correct measurement order is NOT monotonic in distance-from-output.

    Motivation
    ----------
    On regular lattices the optimal order equals "measure farthest-from-output
    first", so a hand-designed static schedule (see StaticScheduleBaseline) is
    near-optimal and a learned policy has no edge. These irregular graphs are
    built specifically to break that assumption: a learned policy that reads the
    whole adjacency can win where a geometric/distance heuristic fails.

    Construction (n = rows*cols nodes, n_outputs = rows, to mirror the grid):
      • Outputs are the last `rows` node ids.
      • The non-output nodes are split into `rows` vertex-disjoint *flow chains*
        of random length, each ending at a distinct output. The chain edges are
        the causal flow (Danos & Kashefi 2006) — this guarantees a gflow exists.
      • Random *cross edges* join nodes at the same flow depth in different chains
        (like a grid's vertical edges).
      • Random *skip edges* shortcut from a node at depth d to depth d-2. These
        reduce a node's graph distance to the outputs WITHOUT changing that it is
        upstream in the flow — so distance-from-output no longer matches the gflow
        layer. This is what defeats the distance heuristic.
      • Bernoulli edge deletions at `defect_rate`, as for the grid.

    The result is verified with compute_gflow and regenerated if (at zero defect)
    gflow was accidentally broken by the extra edges.

    Returns:
        G: NetworkX graph with nodes 0 … rows*cols-1.
        output_qubits: the last `rows` node ids.
    """
    if cols < 2:
        raise ValueError("cols must be >= 2 so there is at least one non-output qubit.")
    if rows < 1:
        raise ValueError("rows must be >= 1.")
    if rng is None:
        rng = np.random.default_rng()

    from mbqc_rl.utils.gflow import compute_gflow  # local import avoids a cycle

    n = rows * cols
    n_outputs = rows
    outputs = list(range(n - n_outputs, n))
    non_output = list(range(n - n_outputs))

    for _ in range(_max_tries):
        G = nx.Graph()
        G.add_nodes_from(range(n))

        pool = non_output.copy()
        rng.shuffle(pool)

        # Split the non-output pool into `rows` chains of random length.
        n_chains = n_outputs
        cuts = sorted(rng.choice(range(1, len(pool)), size=min(n_chains - 1, len(pool) - 1),
                                 replace=False)) if len(pool) > n_chains else []
        chains: list[list[int]] = []
        prev = 0
        for c in cuts:
            chains.append(pool[prev:c]); prev = c
        chains.append(pool[prev:])
        while len(chains) < n_chains:
            chains.append([])

        # depth[node] = position from the output (output=0, deeper=measured earlier);
        # build each chain as  output — d1 — d2 — …  (flow edges toward the output)
        depth: dict[int, int] = {o: 0 for o in outputs}
        chain_of: dict[int, int] = {}
        for ci, chain in enumerate(chains):
            o = outputs[ci]
            prev_node = o
            for d, node in enumerate(chain, start=1):
                G.add_edge(prev_node, node)          # flow (causal-flow) edge
                depth[node] = d
                chain_of[node] = ci
                prev_node = node

        by_depth: dict[int, list[int]] = {}
        for node, d in depth.items():
            if node not in outputs:
                by_depth.setdefault(d, []).append(node)

        # Cross edges: same-depth nodes in different chains (grid-like)
        for d, nodes in by_depth.items():
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    if rng.random() < cross_prob:
                        G.add_edge(nodes[i], nodes[j])

        # Skip edges: depth d → depth d-2 (shortcut that breaks distance monotonicity)
        for node, d in depth.items():
            if node in outputs or d < 2:
                continue
            targets = by_depth.get(d - 2, []) + ([] if d - 2 != 0 else outputs)
            if d - 2 == 0:
                targets = outputs
            targets = [t for t in targets if t != node and not G.has_edge(node, t)]
            if targets and rng.random() < skip_prob:
                G.add_edge(node, int(rng.choice(targets)))

        # Bernoulli edge defects (after structure is built)
        if defect_rate > 0:
            for u, v in list(G.edges()):
                if rng.random() < defect_rate:
                    G.remove_edge(u, v)

        gf, _ = compute_gflow(G, outputs)
        if defect_rate > 0 or gf is not None:
            return G, outputs

    return G, outputs


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
