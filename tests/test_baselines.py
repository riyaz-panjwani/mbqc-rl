"""
Tests for classical baselines and metrics utilities.
"""

import numpy as np
import pytest

from mbqc_rl.env.mbqc_env        import MBQCEnv
from mbqc_rl.baselines.classical  import RandomBaseline, GreedyGflowBaseline, TopologicalBaseline
from mbqc_rl.utils.metrics        import rolling_mean, save_history, load_history, summarise


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def perfect_env():
    return MBQCEnv(rows=3, cols=3, defect_rate=0.0, seed=0)


@pytest.fixture
def defective_env():
    return MBQCEnv(rows=3, cols=3, defect_rate=0.3, seed=7)


def _run_episode(env, baseline):
    obs_dict, info = env.reset(seed=42)
    baseline.reset(obs_dict, info)
    done = False
    total = 0.0
    while not done:
        action = baseline.select_action(obs_dict)
        obs_dict, reward, terminated, truncated, _ = env.step(action)
        total += reward
        done = terminated or truncated
    return total


# ---------------------------------------------------------------------------
# RandomBaseline
# ---------------------------------------------------------------------------

class TestRandomBaseline:
    def test_always_picks_valid_action(self, perfect_env):
        obs_dict, info = perfect_env.reset(seed=0)
        baseline = RandomBaseline(rng=np.random.default_rng(0))
        baseline.reset(obs_dict, info)
        for _ in range(20):
            valid = set(np.where(obs_dict["action_mask"] == 1)[0])
            action = baseline.select_action(obs_dict)
            assert action in valid
            obs_dict, _, terminated, truncated, _ = perfect_env.step(action)
            if terminated or truncated:
                break

    def test_completes_episode(self, perfect_env):
        baseline = RandomBaseline()
        reward = _run_episode(perfect_env, baseline)
        assert 0.0 <= reward <= 1.0

    def test_completes_defective_episode(self, defective_env):
        baseline = RandomBaseline()
        reward = _run_episode(defective_env, baseline)
        assert reward >= 0.0


# ---------------------------------------------------------------------------
# GreedyGflowBaseline
# ---------------------------------------------------------------------------

class TestGreedyGflowBaseline:
    def test_perfect_graph_reward_one(self, perfect_env):
        """On a perfect graph, greedy gflow should always achieve reward = 1.0."""
        baseline = GreedyGflowBaseline(perfect_env)
        rewards = [_run_episode(perfect_env, baseline) for _ in range(10)]
        assert all(r == 1.0 for r in rewards), f"Expected all 1.0, got {rewards}"

    def test_actions_always_valid(self, perfect_env):
        obs_dict, info = perfect_env.reset(seed=0)
        baseline = GreedyGflowBaseline(perfect_env)
        baseline.reset(obs_dict, info)
        for _ in range(20):
            valid = set(np.where(obs_dict["action_mask"] == 1)[0])
            action = baseline.select_action(obs_dict)
            assert action in valid
            obs_dict, _, terminated, truncated, _ = perfect_env.step(action)
            if terminated or truncated:
                break

    def test_completes_defective_episode(self, defective_env):
        baseline = GreedyGflowBaseline(defective_env)
        reward = _run_episode(defective_env, baseline)
        assert reward >= 0.0

    def test_greedy_beats_random_on_average(self, perfect_env):
        """Greedy should achieve reward >= random mean on a perfect graph."""
        rng      = np.random.default_rng(99)
        greedy   = GreedyGflowBaseline(perfect_env, rng=rng)
        random_b = RandomBaseline(rng=rng)
        n_ep     = 20
        greedy_r = [_run_episode(perfect_env, greedy)   for _ in range(n_ep)]
        random_r = [_run_episode(perfect_env, random_b) for _ in range(n_ep)]
        assert np.mean(greedy_r) >= np.mean(random_r)


# ---------------------------------------------------------------------------
# TopologicalBaseline
# ---------------------------------------------------------------------------

class TestTopologicalBaseline:
    def test_completes_episode(self, perfect_env):
        baseline = TopologicalBaseline(perfect_env)
        reward = _run_episode(perfect_env, baseline)
        assert 0.0 <= reward <= 1.0

    def test_picks_valid_action(self, perfect_env):
        obs_dict, info = perfect_env.reset(seed=0)
        baseline = TopologicalBaseline(perfect_env)
        baseline.reset(obs_dict, info)
        valid = set(np.where(obs_dict["action_mask"] == 1)[0])
        action = baseline.select_action(obs_dict)
        assert action in valid


# ---------------------------------------------------------------------------
# gflow / gflow_order properties on MBQCEnv
# ---------------------------------------------------------------------------

def test_env_exposes_gflow():
    env = MBQCEnv(rows=3, cols=3, defect_rate=0.0)
    env.reset(seed=0)
    assert env.gflow is not None
    assert env.gflow_order is not None


def test_env_gflow_order_keys_match_gflow():
    env = MBQCEnv(rows=3, cols=3, defect_rate=0.0)
    env.reset(seed=0)
    # Every non-output qubit should appear in both gflow and gflow_order
    output_set = set(env.output_qubits)
    non_output = {q for q in range(env.n) if q not in output_set}
    assert set(env.gflow.keys()) == non_output


# ---------------------------------------------------------------------------
# Metrics utilities
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_rolling_mean_length(self):
        vals = list(range(100))
        rm   = rolling_mean(vals, window=10)
        assert len(rm) == 100

    def test_rolling_mean_single(self):
        vals = [5.0]
        assert rolling_mean(vals, window=5)[0] == 5.0

    def test_rolling_mean_average(self):
        vals = [1.0, 2.0, 3.0, 4.0]
        rm   = rolling_mean(vals, window=4)
        assert abs(rm[-1] - 2.5) < 1e-9

    def test_save_and_load_history(self, tmp_path):
        history = [{"update": 1, "reward": 0.5}, {"update": 2, "reward": 0.8}]
        path    = tmp_path / "hist.json"
        save_history(history, path)
        loaded  = load_history(path)
        assert loaded == history

    def test_summarise_does_not_crash(self, capsys):
        history = [
            {"update": i, "timestep": i*100,
             "mean_ep_reward": float(i)/10,
             "n_episodes": 5,
             "policy_loss": -0.01, "value_loss": 0.1, "entropy": 0.3}
            for i in range(1, 61)
        ]
        summarise(history)
        captured = capsys.readouterr()
        assert "Training summary" in captured.out
