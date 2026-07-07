"""
Classical measurement-ordering baselines.

These establish the performance floor and ceiling that the PPO agent
is measured against in Phase 4.

RandomBaseline
--------------
At each step, picks uniformly at random from the currently valid qubits.
This is the weakest possible sensible policy — if the RL agent cannot
beat it, something is wrong.

GreedyGflowBaseline
-------------------
At the start of each episode, reads the gflow the environment already
computed (via env.gflow_order) and sorts qubits into a measurement plan:
highest gflow layer first (= measured earliest in a valid computation).
Then executes that pre-planned sequence.

On a graph that has a valid gflow this always achieves reward = 1.0.
On a graph with no gflow it falls back to a random order.

This is the "oracle classical" ceiling: it has perfect knowledge of the
gflow and executes it exactly. The RL agent cannot beat it on perfect
graphs, but on defective graphs where the gflow has to be inferred
dynamically it may match or approach it.

TopologicalBaseline
-------------------
Like GreedyGflowBaseline but breaks ties *randomly* (instead of
deterministically by index). This gives multiple valid orderings
consistent with the partial order and is useful for sampling
distributions of valid-order performance.

Why these three?
----------------
Together they span the performance range:
  Random < Topological ≤ Greedy ≤ 1.0 (on graphs with gflow)
The RL agent should land somewhere in this range and, with training on
defective graphs, should surpass Greedy on defective instances where
Greedy's pre-planned order becomes invalid mid-episode.
"""

from __future__ import annotations

import numpy as np
from mbqc_rl.env.mbqc_env import MBQCEnv


class RandomBaseline:
    """
    Picks uniformly at random from currently valid (unmasked) actions.
    Stateless — no reset() needed.
    """

    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self._rng = rng or np.random.default_rng()

    def reset(self, obs_dict: dict, info: dict) -> None:  # noqa: ARG002
        pass

    def select_action(self, obs_dict: dict) -> int:
        valid = np.where(obs_dict["action_mask"] == 1)[0]
        return int(self._rng.choice(valid))


class GreedyGflowBaseline:
    """
    Pre-computes a deterministic measurement plan at each episode start
    using the environment's gflow layer ordering.

    Measurement plan: sort valid qubits by descending gflow layer
    (highest layer = measured first, consistent with gflow partial order).

    On graphs with a valid gflow → reward 1.0 every episode.
    On graphs without gflow → random valid order.
    """

    def __init__(self, env: MBQCEnv, rng: np.random.Generator | None = None) -> None:
        self._env = env
        self._rng = rng or np.random.default_rng()
        self._plan: list[int] = []
        self._plan_idx: int = 0

    def reset(self, obs_dict: dict, info: dict) -> None:  # noqa: ARG002
        """Build the measurement plan for the new episode."""
        gflow_order = self._env.gflow_order
        non_output   = [q for q in range(self._env.n) if q not in self._env._output_set]

        if gflow_order is not None:
            # Descending layer: highest layer measured first
            self._plan = sorted(non_output, key=lambda q: -gflow_order.get(q, 0))
        else:
            self._plan = list(self._rng.permutation(non_output))

        self._plan_idx = 0

    def select_action(self, obs_dict: dict) -> int:
        mask = obs_dict["action_mask"]
        # Execute the plan in order, skipping any qubits already gone
        while self._plan_idx < len(self._plan):
            action = self._plan[self._plan_idx]
            self._plan_idx += 1
            if mask[action] == 1:
                return action
        # Fallback (should not occur in normal operation)
        valid = np.where(mask == 1)[0]
        return int(self._rng.choice(valid))


class StaticScheduleBaseline:
    """
    Non-adaptive 'textbook' schedule — the natural human / hand-designed baseline.

    A person who understands MBQC, asked to measure a grid resource state, would
    sweep column by column from the input side toward the outputs: measure the
    leftmost column first, then the next, ending at the (never-measured) output
    column on the right. On a *defect-free* grid this fixed order is exactly the
    causal-flow (Danos & Kashefi 2006) order and is optimal (reward 1.0).

    Crucially this baseline is computed ONCE from the grid geometry alone (the
    column index of each qubit) and is applied UNCHANGED to every instance — it
    never inspects the per-graph gflow. That is what makes it a fair proxy for a
    human / static compiler schedule rather than an oracle: when edge defects
    change the true gflow, the static order cannot adapt and its reward degrades.
    The gap between this baseline and the RL agent is therefore a direct measure
    of the *value of adaptation*, which is the whole point of learning a policy.

    Order: ascending column (leftmost = highest defect-free gflow layer = first),
    ties broken by row. Skips qubits already removed; falls back to any valid
    qubit only if the schedule is exhausted.
    """

    def __init__(self, env: MBQCEnv, rng: np.random.Generator | None = None) -> None:
        self._env = env
        self._rng = rng or np.random.default_rng()
        self._plan: list[int] = []
        self._plan_idx: int = 0

    def reset(self, obs_dict: dict, info: dict) -> None:  # noqa: ARG002
        cols = self._env.cols
        non_output = [q for q in range(self._env.n) if q not in self._env._output_set]
        # Fixed geometric order: leftmost column first (col = q % cols), then row.
        self._plan = sorted(non_output, key=lambda q: (q % cols, q // cols))
        self._plan_idx = 0

    def select_action(self, obs_dict: dict) -> int:
        mask = obs_dict["action_mask"]
        while self._plan_idx < len(self._plan):
            action = self._plan[self._plan_idx]
            self._plan_idx += 1
            if mask[action] == 1:
                return action
        valid = np.where(mask == 1)[0]
        return int(self._rng.choice(valid))


class DistanceScheduleBaseline:
    """
    Structural 'farthest-from-output first' schedule — the strongest hand-designed
    heuristic, and the fair baseline on *irregular* graphs.

    Unlike StaticScheduleBaseline (which uses the grid column index), this orders
    qubits by graph distance from the output set: measure the most-upstream
    (farthest) qubits first. It needs no geometric embedding — only the graph —
    so it is permutation-invariant and well-defined on any topology. On a grid it
    reduces exactly to the column sweep, and on lattices it is near-optimal.

    It is, however, still NON-ADAPTIVE and computed from distance alone, never the
    gflow. On irregular graphs whose gflow order is not monotonic in
    distance-from-output (e.g. skip edges that shortcut an upstream qubit close to
    an output) this heuristic is provably suboptimal — that is the regime where a
    learned policy that reads the full adjacency can beat it.
    """

    def __init__(self, env: MBQCEnv, rng: np.random.Generator | None = None) -> None:
        self._env = env
        self._rng = rng or np.random.default_rng()
        self._plan: list[int] = []
        self._plan_idx: int = 0

    def reset(self, obs_dict: dict, info: dict) -> None:  # noqa: ARG002
        import networkx as nx
        G = self._env._graph
        out_set = self._env._output_set
        # Multi-source shortest path from the output set via a temporary super-source.
        src = "__super_src__"
        H = G.copy()
        H.add_node(src)
        for o in out_set:
            H.add_edge(src, o)
        dist = nx.single_source_shortest_path_length(H, src)
        non_output = [q for q in range(self._env.n) if q not in out_set]
        # Farthest first; unreachable qubits (defect-isolated) treated as farthest.
        self._plan = sorted(
            non_output,
            key=lambda q: -(dist.get(q, 10 ** 6) - 1),
        )
        self._plan_idx = 0

    def select_action(self, obs_dict: dict) -> int:
        mask = obs_dict["action_mask"]
        while self._plan_idx < len(self._plan):
            action = self._plan[self._plan_idx]
            self._plan_idx += 1
            if mask[action] == 1:
                return action
        valid = np.where(mask == 1)[0]
        return int(self._rng.choice(valid))


class TopologicalBaseline:
    """
    At each step, samples uniformly from the subset of valid qubits whose
    gflow layer is the current maximum.  This is a randomised topological
    sort consistent with the gflow partial order.

    Compared to GreedyGflowBaseline it produces a distribution over all
    gflow-valid orderings rather than a single deterministic one.
    """

    def __init__(self, env: MBQCEnv, rng: np.random.Generator | None = None) -> None:
        self._env = env
        self._rng = rng or np.random.default_rng()

    def reset(self, obs_dict: dict, info: dict) -> None:  # noqa: ARG002
        pass

    def select_action(self, obs_dict: dict) -> int:
        mask        = obs_dict["action_mask"]
        gflow_order = self._env.gflow_order
        valid       = [q for q in range(self._env.n) if mask[q] == 1]

        if not valid:
            raise RuntimeError("No valid actions — episode should have ended.")

        if gflow_order is None:
            return int(self._rng.choice(valid))

        # Find the maximum layer among valid qubits, then pick randomly from that set
        max_layer = max(gflow_order.get(q, 0) for q in valid)
        frontier  = [q for q in valid if gflow_order.get(q, 0) == max_layer]
        return int(self._rng.choice(frontier))
