"""
Tests for the two Session-5 extensions:
  1. Non-Clifford measurement angles (universal MBQC) + angle-aware reward
  2. Mixed per-episode defect-rate sampling (anti-forgetting training)
"""

import numpy as np
import pytest

from mbqc_rl.env.mbqc_env import MBQCEnv
from mbqc_rl.utils.generators import (
    make_grid_graph,
    assign_measurement_angles,
    is_clifford_angle,
)
from mbqc_rl.utils.gflow import (
    compute_gflow,
    score_measurement_order_angled,
)


# ---------------------------------------------------------------------------
# Angle assignment
# ---------------------------------------------------------------------------

class TestAngleAssignment:
    def test_all_qubits_get_angles(self):
        G, outputs = make_grid_graph(3, 3)
        angles = assign_measurement_angles(G, outputs, rng=np.random.default_rng(0))
        assert set(angles.keys()) == set(G.nodes())

    def test_outputs_get_zero(self):
        G, outputs = make_grid_graph(3, 3)
        angles = assign_measurement_angles(G, outputs, rng=np.random.default_rng(0))
        for q in outputs:
            assert angles[q] == 0

    def test_angles_in_range(self):
        G, outputs = make_grid_graph(3, 3)
        angles = assign_measurement_angles(G, outputs, rng=np.random.default_rng(1))
        assert all(0 <= k <= 7 for k in angles.values())

    def test_clifford_fraction_extremes(self):
        G, outputs = make_grid_graph(4, 4)
        rng = np.random.default_rng(2)
        all_cliff = assign_measurement_angles(G, outputs, clifford_fraction=1.0, rng=rng)
        non_out = [v for v in G.nodes() if v not in set(outputs)]
        assert all(all_cliff[v] % 2 == 0 for v in non_out)

        G2, outputs2 = make_grid_graph(4, 4)
        none_cliff = assign_measurement_angles(G2, outputs2, clifford_fraction=0.0,
                                               rng=np.random.default_rng(3))
        non_out2 = [v for v in G2.nodes() if v not in set(outputs2)]
        assert all(none_cliff[v] % 2 == 1 for v in non_out2)

    def test_stored_on_graph(self):
        G, outputs = make_grid_graph(3, 3)
        angles = assign_measurement_angles(G, outputs, rng=np.random.default_rng(0))
        for v in G.nodes():
            assert G.nodes[v]["angle_k"] == angles[v]

    def test_is_clifford_angle(self):
        assert is_clifford_angle(0)
        assert is_clifford_angle(2)
        assert not is_clifford_angle(1)
        assert not is_clifford_angle(7)


# ---------------------------------------------------------------------------
# Angle-aware scoring
# ---------------------------------------------------------------------------

class TestAngledScore:
    def _setup(self):
        G, outputs = make_grid_graph(3, 3)
        gflow, order = compute_gflow(G, outputs)
        assert gflow is not None
        return G, outputs, gflow, order

    def test_all_clifford_any_order_perfect(self):
        """Fully Clifford pattern → all constraints free → score 1.0 always."""
        G, outputs, gflow, order = self._setup()
        angles = {v: 0 for v in G.nodes()}          # all Clifford
        non_out = sorted(set(G.nodes()) - set(outputs))
        # Worst possible order: reversed gflow order
        steps = {v: i + 1 for i, v in
                 enumerate(sorted(non_out, key=lambda q: order[q]))}
        score = score_measurement_order_angled(gflow, steps, set(outputs), angles)
        assert score == 1.0

    def test_all_non_clifford_equals_standard_score(self):
        """All non-Clifford → every constraint is hard → same as unangled score."""
        from mbqc_rl.utils.gflow import score_measurement_order
        G, outputs, gflow, order = self._setup()
        angles = {v: 1 for v in G.nodes()}          # all non-Clifford
        non_out = sorted(set(G.nodes()) - set(outputs))
        # Correct order: descending layer
        steps = {v: i + 1 for i, v in
                 enumerate(sorted(non_out, key=lambda q: -order[q]))}
        angled   = score_measurement_order_angled(gflow, steps, set(outputs), angles)
        standard = score_measurement_order(gflow, steps, set(outputs))
        assert angled == standard == 1.0

    def test_correct_order_perfect_regardless_of_angles(self):
        """Following gflow order satisfies all constraints, hard or free."""
        G, outputs, gflow, order = self._setup()
        rng = np.random.default_rng(5)
        angles = {v: int(rng.integers(0, 8)) for v in G.nodes()}
        non_out = sorted(set(G.nodes()) - set(outputs))
        steps = {v: i + 1 for i, v in
                 enumerate(sorted(non_out, key=lambda q: -order[q]))}
        score = score_measurement_order_angled(gflow, steps, set(outputs), angles)
        assert score == 1.0


# ---------------------------------------------------------------------------
# Env with angles
# ---------------------------------------------------------------------------

class TestEnvAngles:
    def test_obs_dim_grows_by_n(self):
        env_plain  = MBQCEnv(rows=3, cols=3)
        env_angled = MBQCEnv(rows=3, cols=3, use_angles=True)
        n = 9
        assert env_plain.observation_space["observation"].shape  == (n*n + n,)
        assert env_angled.observation_space["observation"].shape == (n*n + 2*n,)

    def test_angle_channel_values(self):
        env = MBQCEnv(rows=3, cols=3, use_angles=True)
        obs, info = env.reset(seed=0)
        n = 9
        angle_channel = obs["observation"][n*n + n:]
        # Outputs = -1, others = k/8 ∈ [0, 7/8]
        for q in range(n):
            if q in set(env.output_qubits):
                assert angle_channel[q] == -1.0
            else:
                assert 0.0 <= angle_channel[q] <= 7/8
                assert angle_channel[q] == info["angles"][q] / 8.0

    def test_episode_completes_with_angles(self):
        env = MBQCEnv(rows=3, cols=3, use_angles=True, seed=0)
        obs, _ = env.reset(seed=42)
        done = False
        total = 0.0
        while not done:
            valid = np.where(obs["action_mask"] == 1)[0]
            obs, r, terminated, truncated, _ = env.step(int(valid[0]))
            total += r
            done = terminated or truncated
        assert 0.0 <= total <= 1.0

    def test_angles_property(self):
        env = MBQCEnv(rows=3, cols=3, use_angles=True)
        env.reset(seed=0)
        assert len(env.angles) == 9
        env_plain = MBQCEnv(rows=3, cols=3)
        env_plain.reset(seed=0)
        assert env_plain.angles == {}

    def test_greedy_baseline_scores_one_with_angles(self):
        """Following gflow order satisfies hard constraints too → reward 1.0."""
        from mbqc_rl.baselines.classical import GreedyGflowBaseline
        env = MBQCEnv(rows=3, cols=3, use_angles=True, defect_rate=0.0)
        obs, info = env.reset(seed=7)
        b = GreedyGflowBaseline(env)
        b.reset(obs, info)
        done = False
        total = 0.0
        while not done:
            a = b.select_action(obs)
            obs, r, terminated, truncated, _ = env.step(a)
            total += r
            done = terminated or truncated
        assert total == 1.0


# ---------------------------------------------------------------------------
# Mixed defect-rate sampling
# ---------------------------------------------------------------------------

class TestMixedDefectRate:
    def test_fixed_rate_unchanged(self):
        env = MBQCEnv(rows=3, cols=3, defect_rate=0.1)
        _, info = env.reset(seed=0)
        assert info["defect_rate"] == 0.1

    def test_range_samples_within_bounds(self):
        env = MBQCEnv(rows=3, cols=3, defect_rate=(0.0, 0.3), seed=0)
        rates = []
        for i in range(50):
            _, info = env.reset(seed=i)
            rates.append(info["defect_rate"])
        assert all(0.0 <= r <= 0.3 for r in rates)

    def test_range_actually_varies(self):
        env = MBQCEnv(rows=3, cols=3, defect_rate=(0.0, 0.3), seed=0)
        rates = set()
        for i in range(30):
            _, info = env.reset(seed=i)
            rates.add(round(info["defect_rate"], 6))
        assert len(rates) > 10   # near-continuous sampling → many distinct values

    def test_episode_completes_with_range(self):
        env = MBQCEnv(rows=3, cols=3, defect_rate=(0.1, 0.3), seed=0)
        obs, _ = env.reset(seed=3)
        done = False
        while not done:
            valid = np.where(obs["action_mask"] == 1)[0]
            obs, r, terminated, truncated, _ = env.step(int(valid[0]))
            done = terminated or truncated
        assert 0.0 <= r <= 1.0
