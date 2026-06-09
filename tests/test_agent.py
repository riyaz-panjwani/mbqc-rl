"""
Tests for Phase 2: Actor-Critic policy, rollout buffer, and PPO trainer.

These tests verify correctness and stability — they do NOT run a full
training loop (that would take minutes). The Phase-2 milestone (agent
learns on a 2×2 defect-free grid) is checked by a short training run
in test_short_training_non_negative_reward, which just verifies the
training loop completes and produces valid rewards.
"""

import numpy as np
import pytest
import torch
from torch.distributions import Categorical

from mbqc_rl.env.mbqc_env   import MBQCEnv
from mbqc_rl.agent.policy   import MBQCActorCritic
from mbqc_rl.agent.buffer   import RolloutBuffer
from mbqc_rl.agent.ppo      import PPOTrainer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_env():
    """2×2 grid, no defects — smallest non-trivial MBQC instance."""
    return MBQCEnv(rows=2, cols=2, defect_rate=0.0, seed=0)


@pytest.fixture
def env_3x3():
    return MBQCEnv(rows=3, cols=3, defect_rate=0.0, seed=0)


@pytest.fixture
def policy_2x2(small_env):
    n = small_env.n              # 4
    return MBQCActorCritic(obs_dim=n*n+n, n_actions=n, hidden_dim=64)


@pytest.fixture
def policy_3x3(env_3x3):
    n = env_3x3.n                # 9
    return MBQCActorCritic(obs_dim=n*n+n, n_actions=n, hidden_dim=64)


# ---------------------------------------------------------------------------
# MBQCActorCritic
# ---------------------------------------------------------------------------

class TestPolicy:
    def test_output_shapes(self, policy_2x2, small_env):
        n   = small_env.n
        obs = torch.zeros(1, n*n+n)
        msk = torch.ones(1, n)
        logits, value = policy_2x2(obs, msk)
        assert logits.shape == (1, n)
        assert value.shape  == (1,)

    def test_action_masking_zeroes_invalid(self, policy_2x2, small_env):
        """Masked actions must have (near-)zero probability after softmax."""
        n   = small_env.n
        obs = torch.randn(1, n*n+n)
        # Only action 0 is valid
        msk = torch.zeros(1, n)
        msk[0, 0] = 1.0
        logits, _ = policy_2x2(obs, msk)
        probs = torch.softmax(logits, dim=-1)
        assert probs[0, 0].detach().item() > 0.99, "Valid action must have ~1.0 probability"
        for j in range(1, n):
            assert probs[0, j].detach().item() < 1e-6, f"Invalid action {j} must have ~0 probability"

    def test_get_action_samples_valid(self, policy_2x2, small_env):
        """Sampled action must always be one of the valid actions."""
        obs_dict, _ = small_env.reset(seed=42)
        obs  = torch.as_tensor(obs_dict["observation"],  dtype=torch.float32).unsqueeze(0)
        mask = torch.as_tensor(obs_dict["action_mask"],  dtype=torch.float32).unsqueeze(0)
        for _ in range(50):
            action, log_prob, entropy, value = policy_2x2.get_action_and_value(obs, mask)
            assert mask[0, action.item()].item() == 1.0, "Sampled invalid action"
            assert log_prob.shape == (1,)
            assert value.shape   == (1,)

    def test_get_action_with_given_action(self, policy_2x2, small_env):
        """Providing an action should evaluate its log-prob, not sample."""
        obs_dict, _ = small_env.reset(seed=0)
        obs  = torch.as_tensor(obs_dict["observation"], dtype=torch.float32).unsqueeze(0)
        mask = torch.as_tensor(obs_dict["action_mask"], dtype=torch.float32).unsqueeze(0)
        action = torch.tensor([0], dtype=torch.long)
        _, log_prob, _, _ = policy_2x2.get_action_and_value(obs, mask, action=action)
        assert log_prob.shape == (1,)

    def test_batch_forward(self, policy_3x3):
        """Policy must handle batches of observations."""
        batch = 16
        n     = 9
        obs   = torch.randn(batch, n*n+n)
        mask  = torch.ones(batch, n)
        logits, values = policy_3x3(obs, mask)
        assert logits.shape == (batch, n)
        assert values.shape == (batch,)

    def test_get_value(self, policy_2x2, small_env):
        obs_dict, _ = small_env.reset(seed=0)
        obs = torch.as_tensor(obs_dict["observation"], dtype=torch.float32).unsqueeze(0)
        v = policy_2x2.get_value(obs)
        assert v.shape == (1,)

    def test_gradients_flow(self, policy_2x2, small_env):
        """A backward pass must produce non-None gradients on all parameters."""
        n   = small_env.n
        obs  = torch.randn(4, n*n+n)
        mask = torch.ones(4, n)
        _, lp, ent, val = policy_2x2.get_action_and_value(obs, mask)
        loss = -lp.mean() + val.mean() - ent.mean()
        loss.backward()
        for name, param in policy_2x2.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"


# ---------------------------------------------------------------------------
# RolloutBuffer
# ---------------------------------------------------------------------------

class TestRolloutBuffer:
    def _make_buf(self, n_steps=16, obs_dim=20, n_actions=4):
        return RolloutBuffer(n_steps, obs_dim, n_actions)

    def test_add_and_pos(self):
        buf = self._make_buf()
        obs = np.zeros(20, dtype=np.float32)
        msk = np.ones(4, dtype=np.float32)
        buf.add(obs, msk, action=0, reward=1.0, done=False, value=0.5, log_prob=-1.0)
        assert buf.pos == 1

    def test_full_buffer_raises(self):
        buf = self._make_buf(n_steps=2)
        obs = np.zeros(20, dtype=np.float32)
        msk = np.ones(4,  dtype=np.float32)
        buf.add(obs, msk, 0, 0.0, False, 0.0, 0.0)
        buf.add(obs, msk, 0, 0.0, False, 0.0, 0.0)
        with pytest.raises(RuntimeError):
            buf.add(obs, msk, 0, 0.0, False, 0.0, 0.0)

    def test_reset_clears_pos(self):
        buf = self._make_buf()
        obs = np.zeros(20, dtype=np.float32)
        msk = np.ones(4,  dtype=np.float32)
        buf.add(obs, msk, 0, 0.0, False, 0.0, 0.0)
        buf.reset()
        assert buf.pos == 0

    def test_gae_terminal_episode(self):
        """
        Single-step episode: reward=1, done=True.
        With V(s)=0 and no bootstrap, advantage should equal reward − value.
        """
        buf = RolloutBuffer(n_steps=1, obs_dim=4, n_actions=2, gamma=0.99, gae_lambda=0.95)
        obs = np.zeros(4, dtype=np.float32)
        msk = np.ones(2, dtype=np.float32)
        buf.add(obs, msk, action=0, reward=1.0, done=True, value=0.0, log_prob=0.0)
        buf.compute_returns_and_advantages(last_value=0.0, last_done=True)
        assert abs(buf.advantages[0] - 1.0) < 1e-5
        assert abs(buf.returns[0]    - 1.0) < 1e-5

    def test_gae_multi_step(self):
        """
        Two-step episode: r=[0, 1], done=[False, True], V=[0, 0].
        Advantage at t=0 should propagate the terminal reward.
        """
        buf = RolloutBuffer(n_steps=2, obs_dim=4, n_actions=2, gamma=0.99, gae_lambda=0.95)
        obs = np.zeros(4, dtype=np.float32)
        msk = np.ones(2, dtype=np.float32)
        buf.add(obs, msk, 0, reward=0.0, done=False, value=0.0, log_prob=0.0)
        buf.add(obs, msk, 0, reward=1.0, done=True,  value=0.0, log_prob=0.0)
        buf.compute_returns_and_advantages(last_value=0.0, last_done=True)
        # A[1] = 1.0 - 0.0 = 1.0
        assert abs(buf.advantages[1] - 1.0) < 1e-5
        # A[0] = 0 + 0.99*0.95 * A[1] = 0.9405
        expected_a0 = 0.99 * 0.95 * 1.0
        assert abs(buf.advantages[0] - expected_a0) < 1e-4

    def test_get_all_returns_tensors(self):
        buf = self._make_buf(n_steps=8)
        obs = np.zeros(20, dtype=np.float32)
        msk = np.ones(4,  dtype=np.float32)
        for _ in range(8):
            buf.add(obs, msk, 0, 0.0, False, 0.0, 0.0)
        buf.compute_returns_and_advantages(0.0, False)
        data = buf.get_all()
        assert isinstance(data["observations"],  torch.Tensor)
        assert isinstance(data["advantages"],    torch.Tensor)
        assert data["observations"].shape == (8, 20)


# ---------------------------------------------------------------------------
# PPOTrainer
# ---------------------------------------------------------------------------

class TestPPOTrainer:
    def _make_trainer(self, rows=2, cols=2, n_steps=16, n_epochs=2, batch_size=8):
        env    = MBQCEnv(rows=rows, cols=cols, defect_rate=0.0, seed=0)
        n      = env.n
        policy = MBQCActorCritic(obs_dim=n*n+n, n_actions=n, hidden_dim=32)
        opt    = torch.optim.Adam(policy.parameters(), lr=3e-4)
        return PPOTrainer(
            env=env, policy=policy, optimizer=opt,
            n_steps=n_steps, n_epochs=n_epochs, batch_size=batch_size,
            device=torch.device("cpu"),
        )

    def test_collect_rollout_fills_buffer(self):
        trainer = self._make_trainer()
        stats = trainer.collect_rollout()
        assert trainer.buffer.pos == trainer.n_steps
        assert "mean_ep_reward" in stats
        assert "n_episodes"     in stats

    def test_rollout_reward_in_range(self):
        """All terminal rewards must be ∈ [0, 1] (or −1 for invalid actions)."""
        trainer = self._make_trainer()
        trainer.collect_rollout()
        rewards = trainer.buffer.rewards[:trainer.buffer.pos]
        assert (rewards >= -1.0).all() and (rewards <= 1.0).all()

    def test_update_returns_loss_dict(self):
        trainer = self._make_trainer()
        trainer.collect_rollout()
        losses = trainer.update()
        assert "policy_loss" in losses
        assert "value_loss"  in losses
        assert "entropy"     in losses

    def test_update_does_not_crash(self):
        trainer = self._make_trainer(n_steps=32, n_epochs=2, batch_size=8)
        for _ in range(3):
            trainer.collect_rollout()
            trainer.update()   # must not raise

    def test_policy_changes_after_update(self):
        """Policy parameters must change after at least one gradient step."""
        trainer = self._make_trainer()
        params_before = [p.clone() for p in trainer.policy.parameters()]
        trainer.collect_rollout()
        trainer.update()
        params_after = [p.clone() for p in trainer.policy.parameters()]
        changed = any(not torch.allclose(b, a) for b, a in zip(params_before, params_after))
        assert changed, "Policy parameters did not change after update"

    def test_short_training_completes(self, tmp_path):
        """Full train() call on a 2×2 grid for 1 update — must complete without error."""
        trainer = self._make_trainer(n_steps=16)
        history = trainer.train(
            total_timesteps=16,
            save_dir=str(tmp_path),
            log_interval=1,
            save_interval=999,
        )
        assert len(history) == 1
        assert 0.0 <= history[0]["mean_ep_reward"] <= 1.0
