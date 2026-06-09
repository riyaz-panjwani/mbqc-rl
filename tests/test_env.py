import numpy as np
import pytest

from mbqc_rl.env.mbqc_env import MBQCEnv


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------

def test_obs_shapes():
    env = MBQCEnv(rows=3, cols=3)
    obs, info = env.reset(seed=0)
    n = 9
    assert obs["observation"].shape == (n * n + n,)
    assert obs["action_mask"].shape == (n,)
    assert obs["observation"].dtype == np.float32
    assert obs["action_mask"].dtype == np.int8


def test_output_qubits_not_in_action_mask():
    env = MBQCEnv(rows=3, cols=3)
    obs, _ = env.reset(seed=0)
    for q in env.output_qubits:
        assert obs["action_mask"][q] == 0, f"Output qubit {q} must not be a valid action."


def test_info_contains_gflow_flag():
    env = MBQCEnv(rows=3, cols=3, defect_rate=0.0)
    _, info = env.reset(seed=0)
    assert "gflow_exists" in info
    assert info["gflow_exists"] is True


def test_history_all_minus_one_at_reset():
    env = MBQCEnv(rows=3, cols=3)
    obs, _ = env.reset(seed=42)
    history = obs["observation"][81:]   # last 9 values of a 3×3 env
    non_output_mask = obs["action_mask"]
    # All measurable qubits should start as -1
    for q in range(9):
        if non_output_mask[q]:
            assert history[q] == -1.0


# ---------------------------------------------------------------------------
# step()
# ---------------------------------------------------------------------------

def test_invalid_action_returns_penalty():
    env = MBQCEnv(rows=3, cols=3)
    obs, _ = env.reset(seed=0)
    invalid = int(np.where(obs["action_mask"] == 0)[0][0])
    _, reward, terminated, _, info = env.step(invalid)
    assert reward == -1.0
    assert not terminated
    assert info.get("error") == "invalid_action"


def test_graph_shrinks_after_step():
    env = MBQCEnv(rows=3, cols=3, defect_rate=0.0)
    obs, _ = env.reset(seed=0)
    n2 = env.n ** 2
    initial_edges = obs["observation"][:n2].sum()

    action = int(np.where(obs["action_mask"] == 1)[0][0])
    obs, _, _, _, _ = env.step(action)
    new_edges = obs["observation"][:n2].sum()
    assert new_edges <= initial_edges, "Adjacency matrix should shrink after measurement."


def test_outcome_recorded_in_history():
    env = MBQCEnv(rows=3, cols=3)
    obs, _ = env.reset(seed=0)
    action = int(np.where(obs["action_mask"] == 1)[0][0])
    obs, _, _, _, _ = env.step(action)
    history = obs["observation"][env.n ** 2:]
    assert history[action] in (0.0, 1.0)


# ---------------------------------------------------------------------------
# Full episode
# ---------------------------------------------------------------------------

def test_episode_completes_perfect_grid():
    env = MBQCEnv(rows=3, cols=3, defect_rate=0.0)
    obs, _ = env.reset(seed=0)
    done = False
    steps = 0
    while not done:
        valid = np.where(obs["action_mask"] == 1)[0]
        assert len(valid) > 0, "Valid actions exhausted before episode ended."
        obs, reward, terminated, truncated, _ = env.step(int(valid[0]))
        done = terminated or truncated
        steps += 1
    assert steps > 0
    assert 0.0 <= reward <= 1.0


def test_episode_completes_defective_grid():
    env = MBQCEnv(rows=3, cols=3, defect_rate=0.3)
    obs, _ = env.reset(seed=7)
    done = False
    while not done:
        valid = np.where(obs["action_mask"] == 1)[0]
        if len(valid) == 0:
            break
        obs, _, terminated, truncated, _ = env.step(int(valid[0]))
        done = terminated or truncated


def test_episode_reward_in_unit_interval():
    """Reward must always be in [0, 1] (never -1, which is for invalid actions only)."""
    env = MBQCEnv(rows=3, cols=3, defect_rate=0.0)
    for trial in range(5):
        obs, _ = env.reset(seed=trial)
        done = False
        last_reward = None
        while not done:
            valid = np.where(obs["action_mask"] == 1)[0]
            obs, reward, terminated, truncated, _ = env.step(int(valid[0]))
            done = terminated or truncated
            if done:
                last_reward = reward
        assert last_reward is not None
        assert 0.0 <= last_reward <= 1.0


# ---------------------------------------------------------------------------
# Different grid sizes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rows,cols", [(2, 2), (3, 3), (3, 4), (4, 4)])
def test_various_grid_sizes(rows, cols):
    env = MBQCEnv(rows=rows, cols=cols)
    obs, _ = env.reset(seed=0)
    n = rows * cols
    assert obs["observation"].shape == (n * n + n,)
    assert obs["action_mask"].shape == (n,)

    done = False
    while not done:
        valid = np.where(obs["action_mask"] == 1)[0]
        obs, _, terminated, truncated, _ = env.step(int(valid[0]))
        done = terminated or truncated


# ---------------------------------------------------------------------------
# valid_actions helper
# ---------------------------------------------------------------------------

def test_valid_actions_consistent_with_mask():
    env = MBQCEnv(rows=3, cols=3)
    obs, _ = env.reset(seed=0)
    mask_valid = set(np.where(obs["action_mask"] == 1)[0])
    helper_valid = set(env.valid_actions())
    assert mask_valid == helper_valid
