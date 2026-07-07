"""
Rollout buffer with Generalised Advantage Estimation (GAE).

What it stores
--------------
For each of the n_steps timesteps in a rollout:
  obs          — the observation seen before taking the action
  action_mask  — which actions were valid at that step
  action       — the action the policy chose
  reward       — the reward received
  done         — whether the episode ended after this step
  value        — V(obs) estimated by the critic at collection time
  log_prob     — log π(action | obs) under the policy at collection time

GAE (Generalised Advantage Estimation)
---------------------------------------
After a rollout is collected, compute_returns_and_advantages() fills:
  advantages[t] = δ_t + (γλ) δ_{t+1} + (γλ)² δ_{t+2} + …
  where δ_t = r_t + γ V(s_{t+1})(1 − done_t) − V(s_t)

  returns[t] = advantages[t] + values[t]   (used as regression targets for V)

The λ parameter interpolates between:
  λ=0 → one-step TD  (low variance, high bias)
  λ=1 → Monte-Carlo  (zero bias, high variance)
λ=0.95 is standard and works well for sparse rewards.

Why this matters for MBQC
--------------------------
The reward is completely sparse: 0 at every step except the last, where
it is the gflow-consistency score. GAE propagates this terminal signal
backwards through the episode, giving the critic useful gradient even at
early steps. Without GAE, the critic would see a 0 return for most
transitions and learn almost nothing.
"""

from __future__ import annotations

import numpy as np
import torch


class RolloutBuffer:
    """
    Fixed-length on-policy experience buffer with GAE.

    Args:
        n_steps:     Number of environment steps per rollout.
        obs_dim:     Dimension of the flattened observation vector.
        n_actions:   Number of possible actions (= n qubits).
        gamma:       Discount factor γ.
        gae_lambda:  GAE smoothing parameter λ.
    """

    def __init__(
        self,
        n_steps: int,
        obs_dim: int,
        n_actions: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        aux_dim: int = 0,
    ) -> None:
        self.n_steps = n_steps
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.aux_dim = aux_dim          # >0 → store per-step auxiliary targets
        self.pos = 0
        self._reset_arrays()

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all stored data. Call at the start of each rollout."""
        self.pos = 0
        self._reset_arrays()

    def add(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        value: float,
        log_prob: float,
        aux_target: np.ndarray | None = None,
    ) -> None:
        """Store one transition. Raises if the buffer is already full."""
        if self.pos >= self.n_steps:
            raise RuntimeError("RolloutBuffer is full — call reset() first.")
        self.observations[self.pos] = obs
        self.action_masks[self.pos] = action_mask
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = float(done)
        self.values[self.pos] = value
        self.log_probs[self.pos] = log_prob
        if self.aux_dim > 0 and aux_target is not None:
            self.aux_targets[self.pos] = aux_target
        self.pos += 1

    # ------------------------------------------------------------------
    # GAE
    # ------------------------------------------------------------------

    def compute_returns_and_advantages(
        self, last_value: float, last_done: bool
    ) -> None:
        """
        Compute GAE advantages and returns for all stored steps.

        Args:
            last_value: V(s) of the state after the last stored step,
                        estimated by the critic. Pass 0.0 if that step
                        ended an episode.
            last_done:  Whether the episode ended at the last stored step.
        """
        last_gae = 0.0
        n = self.pos

        for t in reversed(range(n)):
            # Bootstrap from the next stored value, or from last_value at the boundary
            if t == n - 1:
                next_value = last_value
            else:
                next_value = self.values[t + 1]

            # (1 − done_t) zeroes the bootstrap when the episode ended at step t
            non_terminal = 1.0 - self.dones[t]

            delta = self.rewards[t] + self.gamma * next_value * non_terminal - self.values[t]
            last_gae = delta + self.gamma * self.gae_lambda * non_terminal * last_gae
            self.advantages[t] = last_gae

        self.returns[:n] = self.advantages[:n] + self.values[:n]

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_all(self) -> dict[str, torch.Tensor]:
        """
        Return all stored data as a dict of CPU tensors, ready for mini-batching.
        Only returns data up to self.pos (may be less than n_steps).
        """
        n = self.pos
        data = {
            "observations":  torch.as_tensor(self.observations[:n],  dtype=torch.float32),
            "action_masks":  torch.as_tensor(self.action_masks[:n],  dtype=torch.float32),
            "actions":       torch.as_tensor(self.actions[:n],       dtype=torch.long),
            "log_probs":     torch.as_tensor(self.log_probs[:n],     dtype=torch.float32),
            "advantages":    torch.as_tensor(self.advantages[:n],    dtype=torch.float32),
            "returns":       torch.as_tensor(self.returns[:n],       dtype=torch.float32),
        }
        if self.aux_dim > 0:
            data["aux_targets"] = torch.as_tensor(self.aux_targets[:n], dtype=torch.float32)
        return data

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _reset_arrays(self) -> None:
        n, d, a = self.n_steps, self.obs_dim, self.n_actions
        self.observations = np.zeros((n, d), dtype=np.float32)
        self.action_masks = np.zeros((n, a), dtype=np.float32)
        self.actions      = np.zeros(n,      dtype=np.int64)
        self.rewards      = np.zeros(n,      dtype=np.float32)
        self.dones        = np.zeros(n,      dtype=np.float32)
        self.values       = np.zeros(n,      dtype=np.float32)
        self.log_probs    = np.zeros(n,      dtype=np.float32)
        self.advantages   = np.zeros(n,      dtype=np.float32)
        self.returns      = np.zeros(n,      dtype=np.float32)
        if self.aux_dim > 0:
            self.aux_targets = np.zeros((n, self.aux_dim), dtype=np.float32)
