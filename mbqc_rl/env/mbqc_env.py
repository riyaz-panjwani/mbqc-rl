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

from mbqc_rl.utils.generators import (
    make_grid_graph,
    make_brickwork_graph,
    make_irregular_flow_graph,
    assign_measurement_angles,
)
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
        topology: str = "grid",
        reward_shaping: bool = False,
        observe_original_graph: bool = False,
        irregular_skip_prob: float | tuple[float, float] = 0.45,
        irregular_cross_prob: float | tuple[float, float] = 0.35,
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
            topology: "grid" (full cluster state) or "brickwork" (the
                universal blind-MBQC resource state, BFK 2009). Same node
                indexing and output column for both, so a policy trained on
                one topology can be evaluated on the other zero-shot.
            reward_shaping: if True, the terminal gflow-consistency score is
                decomposed into a dense per-step reward — each measurement is
                rewarded for the ordering constraints it satisfies at that step.
                The undiscounted episode return is identical to the sparse
                terminal score, so the objective is unchanged, but credit
                assignment is far easier. This is the fix for tasks where the
                graph structure changes every episode (e.g. irregular topology),
                where the sparse terminal reward gives no learnable gradient.
            observe_original_graph: if True, the adjacency block of the
                observation is the ORIGINAL graph (fixed for the episode) rather
                than the degraded one; the history channel still marks which
                qubits are measured. The dynamics and reward are unchanged. This
                preserves the global gflow structure in the observation, which
                imitation-learning experiments showed is destroyed by adjacency
                degradation on irregular graphs (BC: 0.48→0.70+ when enabled).
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
        if topology not in ("grid", "brickwork", "irregular"):
            raise ValueError(
                f"topology must be 'grid', 'brickwork' or 'irregular', got {topology!r}")
        self.topology = topology
        self.reward_shaping = reward_shaping
        self.observe_original_graph = observe_original_graph
        # Irregular-topology knobs; a (lo, hi) tuple is sampled per episode
        # (domain randomisation over graph irregularity → better generalisation).
        self.irregular_skip_prob = irregular_skip_prob
        self.irregular_cross_prob = irregular_cross_prob
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
        self._adj: np.ndarray | None = None          # (n, n) float32, degraded
        self._orig_adj: np.ndarray | None = None     # (n, n) float32, full graph
        self._history: np.ndarray | None = None      # (n,) float32, -1/0/1
        self._measured: np.ndarray | None = None     # (n,) bool
        self._step_of: dict[int, int] = {}           # qubit → step number
        self._gflow: dict[int, set[int]] | None = None
        self._gflow_order: dict[int, int] | None = None
        self._angles: dict[int, int] = {}            # qubit → k (angle = k·π/4)
        self._episode_defect_rate: float = 0.0
        self._step_count: int = 0
        # Reward-shaping bookkeeping (built in reset when reward_shaping=True)
        self._constraint_targets: dict[int, set[int]] = {}   # v → {w : (v before w)}
        self._total_constraints: int = 0

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

        if self.topology == "irregular":
            def _samp(x):
                return (float(self._rng.uniform(*x))
                        if isinstance(x, (tuple, list)) else float(x))
            graph, output_qubits = make_irregular_flow_graph(
                self.rows, self.cols,
                defect_rate=self._episode_defect_rate,
                rng=self._rng,
                cross_prob=_samp(self.irregular_cross_prob),
                skip_prob=_samp(self.irregular_skip_prob),
            )
        else:
            make_graph = (make_brickwork_graph if self.topology == "brickwork"
                          else make_grid_graph)
            graph, output_qubits = make_graph(
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
        self._orig_adj = self._adj.copy()            # full graph, for observe_original_graph
        self._history = np.full(self.n, -1.0, dtype=np.float32)
        self._measured = np.zeros(self.n, dtype=bool)
        self._step_of = {}
        self._step_count = 0

        for q in self._output_set:
            self._measured[q] = True

        self._gflow, self._gflow_order = compute_gflow(graph, output_qubits)
        self._build_constraints()

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

        # Dense per-step reward: credit the constraints this measurement satisfies.
        # A constraint (action ≺ w) is satisfied iff w is measured AFTER action,
        # i.e. w is still unmeasured at this moment (action is now marked measured,
        # so w == action is correctly excluded). Summed over the episode this
        # equals the terminal gflow-consistency score exactly.
        step_reward = 0.0
        if self.reward_shaping and self._total_constraints > 0:
            satisfied_now = sum(
                1 for w in self._constraint_targets.get(action, ())
                if not self._measured[w]
            )
            step_reward = satisfied_now / self._total_constraints

        # Measuring destroys the qubit — remove it from the graph
        self._graph.remove_node(action)
        self._adj = np.zeros((self.n, self.n), dtype=np.float32)
        for u, v in self._graph.edges():
            self._adj[u, v] = 1.0
            self._adj[v, u] = 1.0

        non_output = [q for q in range(self.n) if q not in self._output_set]
        terminated = all(self._measured[q] for q in non_output)

        if self.reward_shaping:
            if self._total_constraints > 0:
                reward = step_reward
            elif terminated and self._gflow is not None:
                # No hard constraints (e.g. fully-Clifford pattern): any order is
                # perfect — award 1.0 once, matching score_measurement_order's
                # "return 1.0 if total == 0" convention. (gflow None → stays 0.)
                reward = 1.0
            else:
                reward = 0.0
        else:
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

    @property
    def gflow_layer_vector(self) -> np.ndarray:
        """
        Per-node gflow layer, normalised to [0, 1], for auxiliary supervision.

        layer[v] = gflow_order[v] / max_layer  (outputs = 0). Returns zeros if
        the current instance has no gflow. Constant over an episode (the gflow
        is computed once at reset). Used as the target for the GNN's auxiliary
        layer-prediction head (neural-algorithmic-reasoning "hint").
        """
        vec = np.zeros(self.n, dtype=np.float32)
        if self._gflow_order is None:
            return vec
        max_layer = max(self._gflow_order.values()) or 1
        for v, layer in self._gflow_order.items():
            vec[v] = layer / max_layer
        return vec

    def valid_actions(self) -> list[int]:
        return [q for q in range(self.n)
                if not self._measured[q] and q not in self._output_set]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_obs(self) -> dict:
        adj = self._orig_adj if self.observe_original_graph else self._adj
        parts = [adj.flatten(), self._history]
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

    def _build_constraints(self) -> None:
        """
        Enumerate the gflow ordering constraints for dense reward shaping.

        Mirrors the (v, w) pairs counted by score_measurement_order[_angled]:
        for each non-output v and each w in g(v) that is a non-output (and, with
        angles, non-Clifford), record the constraint "v measured before w".
        `_total_constraints` is the denominator used by the per-step reward, so
        the shaped return sums exactly to the terminal score.
        """
        self._constraint_targets = {}
        self._total_constraints = 0
        if self._gflow is None:
            return
        for v, g_v in self._gflow.items():
            targets = set()
            for w in g_v:
                if w in self._output_set:
                    continue
                if self.use_angles and self._angles.get(w, 0) % 2 == 0:
                    continue                      # Clifford target → free
                targets.add(w)
            if targets:
                self._constraint_targets[v] = targets
                self._total_constraints += len(targets)
