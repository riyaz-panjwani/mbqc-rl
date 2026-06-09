"""
Actor-Critic policy network for MBQC measurement-order optimisation.

Architecture
------------
Shared trunk  →  Actor head  (logits over n qubits)
              →  Critic head (scalar state value)

The shared trunk lets the critic's value estimates inform the gradient
signal flowing back through the actor — standard in PPO. Both heads
have their own extra linear layer so they can specialise independently.

Action masking
--------------
Before converting logits to a probability distribution, we set the logit
for every invalid action (already-measured or output qubit) to −1e9.
This forces the policy to always pick a valid qubit, which is essential
because the environment penalises invalid actions with reward −1.

Initialisation
--------------
Orthogonal initialisation (Saxe et al., 2013) with gain √2 is standard
for Tanh activations in deep RL. The actor output layer is initialised
with gain 0.01 so the initial policy is nearly uniform — important when
we want the agent to explore before it has any signal about which qubit
order is better.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch.distributions import Categorical


class MBQCActorCritic(nn.Module):
    """
    Shared-trunk Actor-Critic for the MBQCEnv.

    Args:
        obs_dim:    Size of the flattened observation (n² + n for an n-qubit grid).
        n_actions:  Total number of qubits (= n). The action space is Discrete(n).
        hidden_dim: Width of all hidden layers (default 256).
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 256) -> None:
        super().__init__()

        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim

        # Shared feature extractor — both heads read from this
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        # Actor: maps trunk output → per-action logits
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_actions),
        )

        # Critic: maps trunk output → scalar value V(s)
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        self._init_weights()

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def forward(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs:         (batch, obs_dim)   float32
            action_mask: (batch, n_actions) binary — 1 = valid action
        Returns:
            logits: (batch, n_actions)  — invalid actions set to −1e9
            value:  (batch,)
        """
        features = self.trunk(obs)
        logits = self.actor_head(features)
        # Zero out invalid actions before softmax
        logits = logits + (1.0 - action_mask.float()) * (-1e9)
        value = self.critic_head(features).squeeze(-1)
        return logits, value

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample (or evaluate) an action and return all quantities needed for PPO.

        Args:
            obs:         (batch, obs_dim)
            action_mask: (batch, n_actions)
            action:      (batch,) long — if provided, evaluate this action
                         instead of sampling

        Returns:
            action:   (batch,) long
            log_prob: (batch,) — log π(action | obs)
            entropy:  (batch,) — H(π(· | obs))
            value:    (batch,) — V(obs)
        """
        logits, value = self.forward(obs, action_mask)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Return only the value estimate V(obs). Used for GAE bootstrapping."""
        features = self.trunk(obs)
        return self.critic_head(features).squeeze(-1)

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Orthogonal initialisation — standard for PPO (Schulman et al., 2017)."""
        sqrt2 = math.sqrt(2)

        def _ortho(layer: nn.Linear, gain: float) -> None:
            nn.init.orthogonal_(layer.weight, gain=gain)
            nn.init.constant_(layer.bias, 0.0)

        for layer in self.trunk:
            if isinstance(layer, nn.Linear):
                _ortho(layer, sqrt2)

        # Hidden layers of each head use √2; output layers use specialised gains
        actor_layers = [l for l in self.actor_head if isinstance(l, nn.Linear)]
        for i, layer in enumerate(actor_layers):
            _ortho(layer, gain=0.01 if i == len(actor_layers) - 1 else sqrt2)

        critic_layers = [l for l in self.critic_head if isinstance(l, nn.Linear)]
        for i, layer in enumerate(critic_layers):
            _ortho(layer, gain=1.0 if i == len(critic_layers) - 1 else sqrt2)
