"""
Gymnasium environment for MBQC measurement-order optimisation.

State (MDP)
-----------
- Observation: flattened current adjacency matrix (n²) + measurement history (n)
  where history[q] = -1 (unmeasured), 0 or 1 (outcome).
- Action mask: binary vector of length n; 1 = valid (unmeasured, non-output) qubit.

Action
------
Discrete(n): index of the qubit to measure next.

Reward
------
Sparse. At episode end: fraction of gflow ordering constraints satisfied ∈ [0, 1].
During the episode: 0. Invalid action: -1 (action mask violations are penalised).

Episode
-------
Starts on a freshly sampled defective grid graph. Ends when all non-output qubits
have been measured.
"""

from __future__ import annotations

import numpy as np
import networkx as nx
import gymnasium as gym
from gymnasium import spaces

from mbqc_rl.utils.generators import make_grid_graph, assign_measurement_angles
from mbqc_rl.utils.gflow import (
    compute_gflow,
    score_measurement_order,
    score_measurement_order_angled,
)


class MBQCEnv(gym.Env):

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        rows: int = 3,
        cols: int = 3,
        defect_rate: float | tuple[float, float] = 0.0,
        use_angles: bool = False,
        clifford_fraction: float = 0.5,
        render_mode: str | None = None,
        seed: int | None = None,
    ) -> None:
        """
        Args:
            defect_rate: Either a fixed rate, or a (lo, hi) tuple — in which
                case each episode samples a rate uniformly from [lo, hi]
                (mixed-rate training, the anti-forgetting alternative to
                the staged curriculum).
            use_angles: If True, each non-output qubit gets an XY-plane
                measurement angle k·π/4 (universal, non-Clifford MBQC).
                The observation gains n angle entries and the reward counts
                only the hard (non-Clifford-target) ordering constraints.
            clifford_fraction: Probability a qubit's angle is Clifford
                (only used when use_angles=True).
        """
        super().__init__()

        if cols < 2:
            raise ValueError("cols must be >= 2.")
        if rows < 1:
            raise ValueError("rows must be >= 1.")

        self.rows = rows
        self.cols = cols
        self.n = rows * cols
        self.defect_rate = defect_rate
        self.use_angles = use_angles
        self.clifford_fraction = clifford_fraction
        self.render_mode = render_mode
        self._rng = np.random.default_rng(seed)

        n = self.n
        obs_dim = n * n + n + (n if use_angles else 0)
        self.action_space = spaces.Discrete(n)
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(
                low=-1.0, high=1.0,
                shape=(obs_dim,),
                dtype=np.float32,
            ),
            "action_mask": spaces.Box(
                low=0, high=1,
                shape=(n,),
                dtype=np.int8,
            ),
        })

        # Populated on reset()
        self._graph: nx.Graph | None = None
        self._output_set: set[int] = set()
        self._adj: np.ndarray | None = None          # (n, n) float32
        self._history: np.ndarray | None = None      # (n,) float32, -1/0/1
        self._measured: np.ndarray | None = None     # (n,) bool
        self._step_of: dict[int, int] = {}           # qubit → step number
        self._gflow: dict[int, set[int]] | None = None
        self._gflow_order: dict[int, int] | None = None
        self._angles: dict[int, int] = {}            # qubit → k (angle = k·π/4)
        self._episode_defect_rate: float = 0.0
        self._step_count: int = 0

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Mixed-rate training: sample a fresh rate per episode if given a range
        if isinstance(self.defect_rate, (tuple, list)):
            lo, hi = self.defect_rate
            self._episode_defect_rate = float(self._rng.uniform(lo, hi))
        else:
            self._episode_defect_rate = float(self.defect_rate)

        graph, output_qubits = make_grid_graph(
            self.rows, self.cols,
            defect_rate=self._episode_defect_rate,
            rng=self._rng,
        )

        self._graph = graph
        self._output_set = set(output_qubits)

        if self.use_angles:
            self._angles = assign_measurement_angles(
                graph, output_qubits,
                clifford_fraction=self.clifford_fraction,
                rng=self._rng,
            )
        else:
            self._angles = {}
        self._adj = nx.to_numpy_array(
            graph, nodelist=list(range(self.n)), dtype=np.float32
        )
        self._history = np.full(self.n, -1.0, dtype=np.float32)
        self._measured = np.zeros(self.n, dtype=bool)
        self._step_of = {}
        self._step_count = 0

        for q in self._output_set:
            self._measured[q] = True

        self._gflow, self._gflow_order = compute_gflow(graph, output_qubits)

        obs = self._build_obs()
        info = {
            "gflow_exists": self._gflow is not None,
            "output_qubits": list(self._output_set),
            "defect_rate": self._episode_defect_rate,
            "angles": dict(self._angles),
        }
        return obs, info

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        assert self._graph is not None, "Call reset() before step()."

        mask = self._action_mask()
        if not mask[action]:
            return self._build_obs(), -1.0, False, False, {"error": "invalid_action"}

        # Random Clifford measurement outcome
        outcome = int(self._rng.integers(0, 2))
        self._history[action] = float(outcome)
        self._measured[action] = True
        self._step_count += 1
        self._step_of[action] = self._step_count

        # Measuring destroys the qubit — remove it from the graph
        self._graph.remove_node(action)
        self._adj = np.zeros((self.n, self.n), dtype=np.float32)
        for u, v in self._graph.edges():
            self._adj[u, v] = 1.0
            self._adj[v, u] = 1.0

        non_output = [q for q in range(self.n) if q not in self._output_set]
        terminated = all(self._measured[q] for q in non_output)
        reward = self._compute_reward() if terminated else 0.0

        return self._build_obs(), reward, terminated, False, {"step": self._step_count}

    def render(self) -> None:
        if self.render_mode != "human":
            return
        print(f"\nStep {self._step_count}  defect_rate={self.defect_rate}")
        for r in range(self.rows):
            row_parts = []
            for c in range(self.cols):
                q = r * self.cols + c
                if q in self._output_set:
                    cell = " O"
                elif self._measured[q]:
                    cell = f"[{int(self._history[q])}]"
                else:
                    cell = f" {q:2d}"
                h_edge = ""
                if c < self.cols - 1:
                    h_edge = "—" if self._adj[q, q + 1] else " "
                row_parts.append(cell + h_edge)
            print("".join(row_parts))
            if r < self.rows - 1:
                for c in range(self.cols):
                    q = r * self.cols + c
                    below = (r + 1) * self.cols + c
                    print(" | " if self._adj[q, below] else "   ", end="")
                print()
        print()

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def output_qubits(self) -> list[int]:
        return sorted(self._output_set)

    @property
    def gflow(self) -> dict | None:
        """The gflow correction-set map computed at reset(), or None."""
        return self._gflow

    @property
    def gflow_order(self) -> dict | None:
        """Layer ordering from compute_gflow() — output=0, higher=measured earlier."""
        return self._gflow_order

    @property
    def angles(self) -> dict[int, int]:
        """Measurement angles: qubit → k where angle = k·π/4. Empty if use_angles=False."""
        return dict(self._angles)

    def valid_actions(self) -> list[int]:
        return [q for q in range(self.n)
                if not self._measured[q] and q not in self._output_set]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_obs(self) -> dict:
        parts = [self._adj.flatten(), self._history]
        if self.use_angles:
            # Angle channel: k/8 ∈ [0, 7/8] for non-output, -1 for outputs
            angle_vec = np.full(self.n, -1.0, dtype=np.float32)
            for q, k in self._angles.items():
                if q not in self._output_set:
                    angle_vec[q] = k / 8.0
            parts.append(angle_vec)
        obs_vec = np.concatenate(parts, dtype=np.float32)
        return {"observation": obs_vec, "action_mask": self._action_mask()}

    def _action_mask(self) -> np.ndarray:
        mask = np.zeros(self.n, dtype=np.int8)
        for q in range(self.n):
            if not self._measured[q] and q not in self._output_set:
                mask[q] = 1
        return mask

    def _compute_reward(self) -> float:
        """Gflow-consistency score in [0, 1]. Returns 0 if no gflow on this instance."""
        if self._gflow is None:
            return 0.0
        if self.use_angles:
            return score_measurement_order_angled(
                self._gflow, self._step_of, self._output_set, self._angles
            )
        return score_measurement_order(self._gflow, self._step_of, self._output_set)
