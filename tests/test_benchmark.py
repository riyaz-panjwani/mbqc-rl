"""
Tests for Phase 4 evaluation scripts: benchmark and transfer test.

We use a tiny 2×2 grid with 10–20 trials to keep tests fast (~5 s).
The tests verify correctness of outputs rather than quality of results.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mbqc_rl.env.mbqc_env        import MBQCEnv
from mbqc_rl.agent.policy        import MBQCActorCritic
from mbqc_rl.baselines.classical import GreedyGflowBaseline, RandomBaseline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_checkpoint(tmp_dir: Path, rows: int = 2, cols: int = 2) -> Path:
    """Create a tiny policy checkpoint for testing."""
    n        = rows * cols
    obs_dim  = n * n + n
    policy   = MBQCActorCritic(obs_dim=obs_dim, n_actions=n, hidden_dim=32)
    ckpt     = {
        "policy_state_dict": policy.state_dict(),
        "config": {
            "rows": rows, "cols": cols,
            "obs_dim": obs_dim, "n_actions": n, "hidden_dim": 32,
        },
    }
    path = tmp_dir / "test_ckpt.pt"
    torch.save(ckpt, path)
    return path


# ---------------------------------------------------------------------------
# Import the runner functions directly (avoids subprocess, faster)
# ---------------------------------------------------------------------------

import importlib.util, types

def _import_script(name: str, path: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"


# ---------------------------------------------------------------------------
# benchmark.py
# ---------------------------------------------------------------------------

class TestBenchmarkFunctions:

    def setup_method(self):
        self.tmp   = Path(tempfile.mkdtemp())
        self.ckpt  = _make_checkpoint(self.tmp)
        self.bmod  = _import_script("benchmark", str(_SCRIPTS_DIR / "benchmark.py"))

    def _make_policy(self):
        ckpt   = torch.load(self.ckpt, map_location="cpu", weights_only=False)
        cfg    = ckpt["config"]
        policy = MBQCActorCritic(obs_dim=cfg["obs_dim"], n_actions=cfg["n_actions"],
                                  hidden_dim=cfg["hidden_dim"])
        policy.load_state_dict(ckpt["policy_state_dict"])
        policy.eval()
        return policy

    def test_run_agent_returns_float_in_range(self):
        env    = MBQCEnv(rows=2, cols=2, defect_rate=0.0)
        policy = self._make_policy()
        reward = self.bmod._run_agent(env, policy, seed=0,
                                       device=torch.device("cpu"))
        assert isinstance(reward, float)
        assert 0.0 <= reward <= 1.0

    def test_run_greedy_perfect_graph(self):
        env    = MBQCEnv(rows=2, cols=2, defect_rate=0.0)
        reward = self.bmod._run_greedy(env, seed=42)
        assert reward == 1.0

    def test_run_random_returns_valid(self):
        env    = MBQCEnv(rows=2, cols=2, defect_rate=0.0)
        rng    = np.random.default_rng(0)
        reward = self.bmod._run_random(env, seed=7, rng=rng)
        assert 0.0 <= reward <= 1.0

    def test_stats_keys(self):
        rewards = np.array([0.5, 0.7, 1.0, 0.8])
        s = self.bmod._stats(rewards, "test")
        for key in ("mean", "std", "median", "min", "max", "pct_perfect"):
            assert key in s
        assert abs(s["mean"] - 0.75) < 1e-9
        assert s["pct_perfect"] == 25.0

    def test_mannwhitney_identical_arrays(self):
        a = np.array([0.5, 0.6, 0.7])
        result = self.bmod._mannwhitney(a, a)
        assert "U_statistic" in result
        assert "p_value" in result
        assert result["p_value"] == 1.0  # identical arrays → p=1

    def test_full_benchmark_small(self):
        """Run the full benchmark pipeline with 5 trials on a 2×2 grid."""
        policy = self._make_policy()
        env_a  = MBQCEnv(rows=2, cols=2, defect_rate=0.0)
        env_g  = MBQCEnv(rows=2, cols=2, defect_rate=0.0)
        env_r  = MBQCEnv(rows=2, cols=2, defect_rate=0.0)
        rng    = np.random.default_rng(42)
        n      = 5

        agent_r  = np.zeros(n)
        greedy_r = np.zeros(n)
        random_r = np.zeros(n)

        for i in range(n):
            agent_r[i]  = self.bmod._run_agent( env_a, policy, i, torch.device("cpu"))
            greedy_r[i] = self.bmod._run_greedy(env_g, i)
            random_r[i] = self.bmod._run_random(env_r, i, rng)

        assert all(0.0 <= r <= 1.0 for r in agent_r)
        assert all(r == 1.0 for r in greedy_r)  # greedy is oracle on perfect 2×2


# ---------------------------------------------------------------------------
# transfer_test.py
# ---------------------------------------------------------------------------

class TestTransferFunctions:

    def setup_method(self):
        self.tmp  = Path(tempfile.mkdtemp())
        self.ckpt = _make_checkpoint(self.tmp)
        self.tmod = _import_script(
            "transfer_test", str(_SCRIPTS_DIR / "transfer_test.py")
        )

    def _make_policy(self):
        ckpt   = torch.load(self.ckpt, map_location="cpu", weights_only=False)
        cfg    = ckpt["config"]
        policy = MBQCActorCritic(obs_dim=cfg["obs_dim"], n_actions=cfg["n_actions"],
                                  hidden_dim=cfg["hidden_dim"])
        policy.load_state_dict(ckpt["policy_state_dict"])
        policy.eval()
        return policy

    def test_evaluate_at_rate_structure(self):
        policy = self._make_policy()
        result = self.tmod.evaluate_at_rate(
            defect_rate=0.0, rows=2, cols=2,
            policy=policy, n_episodes=5,
            seed=0, device=torch.device("cpu"),
        )
        assert result["defect_rate"] == 0.0
        assert result["n_episodes"]  == 5
        for key in ("agent", "greedy", "random"):
            assert "mean" in result[key]
            assert "std"  in result[key]
            assert "pct_perfect" in result[key]

    def test_evaluate_at_rate_perfect_graph(self):
        """Greedy should achieve mean=1.0 on a perfect 2×2 graph."""
        policy = self._make_policy()
        result = self.tmod.evaluate_at_rate(
            defect_rate=0.0, rows=2, cols=2,
            policy=policy, n_episodes=10,
            seed=42, device=torch.device("cpu"),
        )
        assert result["greedy"]["mean"] == 1.0

    def test_evaluate_sweep_multiple_rates(self):
        policy = self._make_policy()
        rates  = [0.0, 0.1, 0.2]
        results = []
        for rate in rates:
            res = self.tmod.evaluate_at_rate(
                defect_rate=rate, rows=2, cols=2,
                policy=policy, n_episodes=5,
                seed=0, device=torch.device("cpu"),
            )
            results.append(res)
        assert len(results) == 3
        assert all(0.0 <= r["agent"]["mean"] <= 1.0 for r in results)

    def test_json_save(self):
        policy = self._make_policy()
        result = self.tmod.evaluate_at_rate(
            defect_rate=0.0, rows=2, cols=2,
            policy=policy, n_episodes=3,
            seed=0, device=torch.device("cpu"),
        )
        json_path = self.tmp / "test_transfer.json"
        with open(json_path, "w") as f:
            json.dump([result], f, indent=2)
        loaded = json.loads(json_path.read_text())
        assert loaded[0]["defect_rate"] == 0.0


# ---------------------------------------------------------------------------
# Integration: greedy baseline reward monotone in defect rate
# ---------------------------------------------------------------------------

def test_greedy_reward_monotone_in_defect_rate():
    """
    Greedy baseline mean reward should (weakly) decrease with defect rate.
    Tested over 50 episodes at each rate. May fail occasionally on tiny seeds;
    we allow one non-monotone step.
    """
    rates   = [0.0, 0.1, 0.2, 0.3]
    rewards = []
    for rate in rates:
        env  = MBQCEnv(rows=3, cols=3, defect_rate=rate)
        ep_r = []
        for i in range(50):
            obs_dict, info = env.reset(seed=i)
            b = GreedyGflowBaseline(env)
            b.reset(obs_dict, info)
            done = False
            total = 0.0
            while not done:
                a = b.select_action(obs_dict)
                obs_dict, r, terminated, truncated, _ = env.step(a)
                total += r
                done = terminated or truncated
            ep_r.append(total)
        rewards.append(np.mean(ep_r))

    # Allow at most one non-monotone adjacent pair
    violations = sum(
        rewards[i] < rewards[i+1] for i in range(len(rewards)-1)
    )
    assert violations <= 1, (
        f"Greedy reward not weakly decreasing with defect rate: {rewards}"
    )
