"""
PPO trainer — wraps rollout collection and the PPO update step.

PPO at a glance
---------------
1. Collect n_steps transitions using the *current* policy π_θ_old.
2. Compute GAE advantages A_t for each step.
3. For n_epochs epochs, sample random mini-batches and compute:

   ratio        r_t = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
   actor loss   L_clip = -E[min(r_t A_t,  clip(r_t, 1-ε, 1+ε) A_t)]
   critic loss  L_vf   = ½ E[(V_θ(s_t) - returns_t)²]
   entropy      L_ent  = -E[H(π_θ(·|s_t))]
   total        L      = L_clip + c_v L_vf - c_e L_ent

4. Update θ with Adam, clipping gradient norms to 0.5.
5. Repeat from step 1.

Why each loss term
------------------
L_clip  — the clipped surrogate prevents overly large policy updates,
           which is the main stability contribution of PPO.
L_vf    — trains the critic to give accurate value estimates, which
           are needed for good GAE advantages.
L_ent   — entropy bonus encourages exploration; critical here because
           the reward signal is sparse and the agent could collapse to
           a deterministic but suboptimal ordering early in training.

Gradient clipping (max norm 0.5) is standard for PPO and prevents
exploding gradients, which can appear under high reward variance.

Device
------
Prefers MPS (Apple Silicon) > CUDA > CPU.  All policy computation runs
on-device; environment steps stay on CPU (numpy). Only the tensor copy
is done per-step, not per-element.
"""

from __future__ import annotations

import os
import logging
import numpy as np
import torch
import torch.nn as nn

from mbqc_rl.agent.policy import MBQCActorCritic
from mbqc_rl.agent.buffer import RolloutBuffer
from mbqc_rl.env.mbqc_env import MBQCEnv

log = logging.getLogger(__name__)


def _best_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class PPOTrainer:
    """
    PPO trainer for a single MBQCEnv instance.

    Args:
        env:         The Gymnasium environment (MBQCEnv).
        policy:      The Actor-Critic network (MBQCActorCritic).
        optimizer:   A PyTorch optimiser already constructed for `policy`.
        n_steps:     Steps collected per rollout before each update batch.
        n_epochs:    Number of passes over the rollout data per update.
        batch_size:  Mini-batch size for gradient steps.
        gamma:       Discount factor.
        gae_lambda:  GAE λ.
        clip_range:  PPO clipping ε.
        vf_coef:     Critic loss weight c_v.
        ent_coef:    Entropy bonus weight c_e.
        device:      Torch device; auto-detected if None.
    """

    def __init__(
        self,
        env: MBQCEnv,
        policy: MBQCActorCritic,
        optimizer: torch.optim.Optimizer,
        n_steps:    int   = 512,
        n_epochs:   int   = 10,
        batch_size: int   = 64,
        gamma:      float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        vf_coef:    float = 0.5,
        ent_coef:   float = 0.01,
        device:     torch.device | None = None,
    ) -> None:
        self.env        = env
        self.policy     = policy
        self.optimizer  = optimizer
        self.n_steps    = n_steps
        self.n_epochs   = n_epochs
        self.batch_size = batch_size
        self.gamma      = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.vf_coef    = vf_coef
        self.ent_coef   = ent_coef
        self.device     = device or _best_device()

        self.policy.to(self.device)

        n       = env.n
        obs_dim = n * n + n
        self.buffer = RolloutBuffer(n_steps, obs_dim, n, gamma, gae_lambda)

        # Running environment state — persists across collect_rollout() calls
        # so episodes that straddle rollout boundaries are handled correctly.
        self._obs:         np.ndarray | None = None
        self._action_mask: np.ndarray | None = None
        self._needs_reset: bool = True

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def collect_rollout(self) -> dict:
        """
        Run the current policy in the environment for n_steps steps.
        Fills the buffer, then computes GAE advantages.

        Returns a dict with rollout statistics (mean episode reward, etc.).
        """
        if self._needs_reset:
            obs_dict, _ = self.env.reset()
            self._obs         = obs_dict["observation"]
            self._action_mask = obs_dict["action_mask"]
            self._needs_reset = False

        self.buffer.reset()
        self.policy.eval()

        episode_rewards: list[float] = []
        ep_reward = 0.0

        for _ in range(self.n_steps):
            obs_t  = torch.as_tensor(self._obs,         dtype=torch.float32, device=self.device).unsqueeze(0)
            mask_t = torch.as_tensor(self._action_mask, dtype=torch.float32, device=self.device).unsqueeze(0)

            with torch.no_grad():
                action_t, log_prob_t, _, value_t = self.policy.get_action_and_value(obs_t, mask_t)

            action   = action_t.item()
            log_prob = log_prob_t.item()
            value    = value_t.item()

            next_obs_dict, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            self.buffer.add(
                self._obs, self._action_mask, action,
                reward, done, value, log_prob,
            )
            ep_reward += reward

            if done:
                episode_rewards.append(ep_reward)
                ep_reward = 0.0
                obs_dict, _ = self.env.reset()
                self._obs         = obs_dict["observation"]
                self._action_mask = obs_dict["action_mask"]
            else:
                self._obs         = next_obs_dict["observation"]
                self._action_mask = next_obs_dict["action_mask"]

        # Bootstrap value for the state we're currently in (may be mid-episode)
        last_done = bool(self.buffer.dones[self.buffer.pos - 1])
        if last_done:
            last_value = 0.0
        else:
            obs_t  = torch.as_tensor(self._obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            mask_t = torch.as_tensor(self._action_mask, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                last_value = self.policy.get_value(obs_t).item()

        self.buffer.compute_returns_and_advantages(last_value, last_done)

        return {
            "mean_ep_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
            "n_episodes":     len(episode_rewards),
        }

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def update(self) -> dict:
        """
        Run n_epochs of mini-batch PPO updates on the collected rollout.
        Returns a dict of mean losses.
        """
        data        = self.buffer.get_all()
        obs         = data["observations"].to(self.device)
        masks       = data["action_masks"].to(self.device)
        actions     = data["actions"].to(self.device)
        old_lps     = data["log_probs"].to(self.device)
        advantages  = data["advantages"].to(self.device)
        returns     = data["returns"].to(self.device)

        # Normalise advantages across the entire rollout
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n_samples = obs.shape[0]
        indices   = np.arange(n_samples)

        pg_losses, vf_losses, ent_losses = [], [], []

        self.policy.train()
        for _ in range(self.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, n_samples, self.batch_size):
                idx = indices[start : start + self.batch_size]
                idx_t = torch.as_tensor(idx, dtype=torch.long, device=self.device)

                b_obs      = obs[idx_t]
                b_masks    = masks[idx_t]
                b_actions  = actions[idx_t]
                b_old_lps  = old_lps[idx_t]
                b_adv      = advantages[idx_t]
                b_returns  = returns[idx_t]

                _, new_lps, entropy, new_values = self.policy.get_action_and_value(
                    b_obs, b_masks, b_actions
                )

                # Clipped surrogate objective
                log_ratio  = new_lps - b_old_lps
                ratio      = torch.exp(log_ratio)
                pg_loss1   = -b_adv * ratio
                pg_loss2   = -b_adv * torch.clamp(ratio, 1.0 - self.clip_range,
                                                          1.0 + self.clip_range)
                pg_loss    = torch.max(pg_loss1, pg_loss2).mean()

                vf_loss    = 0.5 * ((new_values - b_returns) ** 2).mean()
                ent_loss   = entropy.mean()

                loss = pg_loss + self.vf_coef * vf_loss - self.ent_coef * ent_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()

                pg_losses.append(pg_loss.item())
                vf_losses.append(vf_loss.item())
                ent_losses.append(ent_loss.item())

        return {
            "policy_loss": float(np.mean(pg_losses)),
            "value_loss":  float(np.mean(vf_losses)),
            "entropy":     float(np.mean(ent_losses)),
        }

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(
        self,
        total_timesteps: int,
        save_dir: str = "checkpoints",
        log_interval: int = 10,
        save_interval: int = 50,
    ) -> list[dict]:
        """
        Alternate rollout collection and PPO updates until total_timesteps.

        Returns the full training history as a list of per-update dicts.
        """
        os.makedirs(save_dir, exist_ok=True)
        n_updates = total_timesteps // self.n_steps
        history   = []

        print(
            f"\n{'='*60}\n"
            f"  PPO Training — {self.env.rows}×{self.env.cols} grid  "
            f"defect_rate={self.env.defect_rate}\n"
            f"  Device: {self.device}  |  Total timesteps: {total_timesteps:,}\n"
            f"  Updates: {n_updates}  |  Steps/update: {self.n_steps}\n"
            f"{'='*60}"
        )

        for update in range(1, n_updates + 1):
            rollout_stats = self.collect_rollout()
            loss_stats    = self.update()

            stats = {
                "update":    update,
                "timestep":  update * self.n_steps,
                **rollout_stats,
                **loss_stats,
            }
            history.append(stats)

            if update % log_interval == 0:
                print(
                    f"  [{update:>4}/{n_updates}]  ts={stats['timestep']:>7,}  "
                    f"reward={rollout_stats['mean_ep_reward']:.3f}  "
                    f"eps={rollout_stats['n_episodes']:>3}  "
                    f"pg={loss_stats['policy_loss']:+.4f}  "
                    f"vf={loss_stats['value_loss']:.4f}  "
                    f"ent={loss_stats['entropy']:.3f}"
                )

            if update % save_interval == 0:
                path = os.path.join(save_dir, f"ckpt_update{update:05d}.pt")
                torch.save({
                    "update":               update,
                    "policy_state_dict":    self.policy.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "stats":                stats,
                    "config": {
                        "rows":        self.env.rows,
                        "cols":        self.env.cols,
                        "defect_rate": self.env.defect_rate,
                        "obs_dim":     self.policy.obs_dim,
                        "n_actions":   self.policy.n_actions,
                        "hidden_dim":  self.policy.hidden_dim,
                    },
                }, path)
                print(f"          → saved {path}")

        print(f"\n  Training complete. Final mean reward: "
              f"{history[-1]['mean_ep_reward']:.3f}\n")
        return history
