"""
Graph-Attention Actor-Critic for MBQC.

Why a GNN?
----------
The flat MLP treats the n×n adjacency as a raw vector, losing the graph
structure.  A GNN processes the same adjacency as an actual graph, giving
two key properties:

  1. Permutation equivariance — qubit relabelling doesn't change the policy,
     so the agent generalises better across isomorphic graph instances.

  2. Inductive (size-agnostic) — GATConv layers are parameterised by
     (in_dim, out_dim, n_heads) only, independent of graph size n.
     A policy trained on 3×3 grids can be evaluated on 4×4 or 5×5 grids
     at inference time with zero retraining (zero-shot transfer).

Architecture
------------
  obs (flat)
    ├─ adj (n×n)    ← graph structure, passed as A to GATConv
    ├─ meas_hist (n) ┐ node features
    └─ angle (n)    ┘ (angle only if use_angles=True)

  input_proj: Linear(d_in → hidden_dim)           per node
  GATConv × n_layers                              message-passing
  actor_head: Linear(hidden_dim → 1)              per-node logit
  critic_mlp: mean-pool → Linear → Linear → 1    global value

GATConv (Veličković et al., 2018)
----------------------------------
  e_{ij} = LeakyReLU( a_src · z_i + a_dst · z_j )
  α_{ij} = softmax_j(e_{ij})   over neighbours + self
  h_i'   = ELU( concat_k  Σ_j α_{ij}^k  W^k h_j )

Additive (decomposed) attention: O(n·d) vs O(n²·d) for concat attention.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical


class GATConv(nn.Module):
    """Single multi-head graph-attention layer with residual connection."""

    def __init__(
        self,
        in_dim:      int,
        out_dim:     int,
        n_heads:     int   = 4,
        leaky_slope: float = 0.2,
    ) -> None:
        super().__init__()
        assert out_dim % n_heads == 0, "out_dim must be divisible by n_heads"
        self.n_heads  = n_heads
        self.head_dim = out_dim // n_heads

        self.W     = nn.Linear(in_dim, n_heads * self.head_dim, bias=False)
        self.a_src = nn.Parameter(torch.empty(1, 1, n_heads, self.head_dim))
        self.a_dst = nn.Parameter(torch.empty(1, 1, n_heads, self.head_dim))
        self.leaky = nn.LeakyReLU(leaky_slope)

        # Residual projection when dims differ
        self.res_proj = (
            nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity()
        )

        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src.view(n_heads, self.head_dim))
        nn.init.xavier_uniform_(self.a_dst.view(n_heads, self.head_dim))

    def forward(
        self,
        H: Tensor,
        A: Tensor,
        return_attention: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """
        Args:
            H: (B, n, in_dim)  — node features
            A: (B, n, n)       — binary adjacency (self-loops added internally)
            return_attention:  if True, also return the attention coefficients
                               α (B, n, n, n_heads) for interpretability.
        Returns:
            (B, n, out_dim)                       if return_attention is False
            ((B, n, out_dim), (B, n, n, n_heads)) if return_attention is True
        """
        B, n, _ = H.shape
        nH = self.n_heads
        d  = self.head_dim

        # Project to (B, n, nH, d)
        Z = self.W(H).view(B, n, nH, d)

        # Additive attention scores e[b,i,j,h] = a_src·z_i + a_dst·z_j
        e_src = (Z * self.a_src).sum(-1)           # (B, n, nH)
        e_dst = (Z * self.a_dst).sum(-1)           # (B, n, nH)
        e = self.leaky(
            e_src.unsqueeze(2) + e_dst.unsqueeze(1)  # (B, n, n, nH)
        )

        # Mask: attend only to neighbours + self
        eye   = torch.eye(n, device=A.device, dtype=A.dtype).unsqueeze(0)
        A_hat = (A + eye).clamp(0, 1)                        # (B, n, n)
        e = e.masked_fill(~A_hat.bool().unsqueeze(-1), float("-inf"))

        alpha = F.softmax(e, dim=2).nan_to_num(0.0)           # (B, n, n, nH)

        # Aggregate: out[b,i,h,d] = Σ_j α[b,i,j,h] · Z[b,j,h,d]
        # Reshape for batched matmul over (B·nH) pairs
        alpha_T = alpha.permute(0, 3, 1, 2).reshape(B * nH, n, n)  # (B·nH, n, n)
        Z_T     = Z.permute(0, 2, 1, 3).reshape(B * nH, n, d)      # (B·nH, n, d)
        out = torch.bmm(alpha_T, Z_T)                               # (B·nH, n, d)
        out = out.view(B, nH, n, d).permute(0, 2, 1, 3)            # (B, n, nH, d)
        out = F.elu(out.reshape(B, n, nH * d))                      # (B, n, out_dim)

        out = out + self.res_proj(H)
        if return_attention:
            return out, alpha
        return out


class GNNActorCritic(nn.Module):
    """
    Graph-Attention Actor-Critic — drop-in replacement for MBQCActorCritic.

    Args:
        n:          Number of qubits (rows × cols).
        hidden_dim: Width of all GATConv layers (must be divisible by n_heads).
        n_heads:    Attention heads per GATConv layer.
        n_layers:   Number of GATConv layers.
        use_angles: Whether the observation includes the angle channel.
        virtual_node: if True, after every GAT layer a global summary
            (mean over nodes, passed through an MLP) is broadcast back to every
            node. This is a virtual-node / global-readout channel that gives the
            network a shortcut for global information, mitigating over-squashing
            (Alon & Yahav 2021) — relevant where the target (gflow layer) is a
            global property. Ladder rung 1.
        weight_tied: if True, a SINGLE GATConv is applied n_layers times (shared
            weights) instead of n_layers distinct layers — an iterative
            "processor" that aligns with the layer-by-layer nature of gflow
            (encode-process-decode; neural algorithmic reasoning). Ladder rung 2.
        pos_dim: if > 0, append a k=pos_dim random-walk structural encoding to
            each node's features (return probabilities of length-1..k walks on
            the self-looped graph; Dwivedi et al.). Injects global positional/
            structural information the GNN can't easily compute itself. Computed
            internally from the adjacency, so the observation is unchanged.
            Ladder rung 3.
    """

    def __init__(
        self,
        n:          int,
        hidden_dim: int  = 64,
        n_heads:    int  = 4,
        n_layers:   int  = 3,
        use_angles: bool = False,
        virtual_node: bool = False,
        weight_tied:  bool = False,
        pos_dim:      int  = 0,
        aux_layer_head: bool = False,
    ) -> None:
        super().__init__()
        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads"

        self.n            = n
        self.hidden_dim   = hidden_dim
        self.n_heads      = n_heads
        self.n_layers     = n_layers
        self.use_angles   = use_angles
        self.virtual_node = virtual_node
        self.weight_tied  = weight_tied
        self.pos_dim      = pos_dim
        self.aux_layer_head = aux_layer_head

        # Compatibility attrs expected by PPOTrainer checkpoint save
        self.obs_dim   = n * n + n + (n if use_angles else 0)
        self.n_actions = n

        d_in = (2 if use_angles else 1) + pos_dim   # node feature dimension

        # Per-node input projection
        self.input_proj = nn.Linear(d_in, hidden_dim)

        # Message-passing layers — one shared conv if weight-tied, else n_layers
        n_convs = 1 if weight_tied else n_layers
        self.gat_layers = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim, n_heads=n_heads)
            for _ in range(n_convs)
        ])

        # Virtual-node / global-readout channel (one MLP reused after each layer)
        self.global_mlp = (
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
            if virtual_node else None
        )

        # Auxiliary head: predict each node's (normalised) gflow layer — a
        # neural-algorithmic-reasoning "hint" that shapes the representation.
        self.aux_head = nn.Linear(hidden_dim, 1) if aux_layer_head else None

        # Actor: per-node embedding → scalar logit
        self.actor_head = nn.Linear(hidden_dim, 1)

        # Critic: global mean-pool → MLP → scalar value
        self.critic_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._init_weights()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _unpack(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        """Split flat observation into node features and adjacency.

        Infers n from obs_dim at runtime so a checkpoint trained on 3×3 can
        be evaluated on 4×4 or 5×5 without any weight changes (zero-shot).
          No angles:  obs_dim = n²+n  →  n = floor((-1+√(1+4·obs_dim))/2)
          With angles: obs_dim = n²+2n →  n = floor((-2+√(4+4·obs_dim))/2)
        """
        B, obs_dim = obs.shape
        if self.use_angles:
            n = int((-2 + (4 + 4 * obs_dim) ** 0.5) / 2)
        else:
            n = int((-1 + (1 + 4 * obs_dim) ** 0.5) / 2)

        A    = obs[:, :n * n].view(B, n, n)                          # (B, n, n)
        meas = obs[:, n * n: n * n + n].unsqueeze(-1)                 # (B, n, 1)
        if self.use_angles:
            ang  = obs[:, n * n + n: n * n + 2 * n].unsqueeze(-1)    # (B, n, 1)
            feats = torch.cat([meas, ang], dim=-1)                    # (B, n, 2)
        else:
            feats = meas                                               # (B, n, 1)
        return feats, A

    def _rwse(self, A: Tensor) -> Tensor:
        """Random-walk structural encoding: per-node return probabilities of
        length-1..pos_dim walks on the self-looped graph. Shape (B, n, pos_dim)."""
        B, n, _ = A.shape
        eye = torch.eye(n, device=A.device, dtype=A.dtype).unsqueeze(0)
        A_hat = A + eye
        deg = A_hat.sum(-1, keepdim=True).clamp(min=1.0)
        P = A_hat / deg                                  # row-stochastic (B, n, n)
        feats, Pk = [], P
        for _ in range(self.pos_dim):
            feats.append(torch.diagonal(Pk, dim1=-2, dim2=-1))   # (B, n)
            Pk = torch.bmm(Pk, P)
        return torch.stack(feats, dim=-1)                # (B, n, pos_dim)

    def _node_features(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        """Node features (with optional RWSE appended) and adjacency."""
        feats, A = self._unpack(obs)
        if self.pos_dim > 0:
            feats = torch.cat([feats, self._rwse(A)], dim=-1)
        return feats, A

    def _embed(self, obs: Tensor) -> Tensor:
        """Run full GNN forward pass; returns node embeddings (B, n, hidden_dim)."""
        feats, A = self._node_features(obs)
        H = torch.tanh(self.input_proj(feats))   # (B, n, hidden_dim)
        for i in range(self.n_layers):
            layer = self.gat_layers[0] if self.weight_tied else self.gat_layers[i]
            H = layer(H, A)
            if self.global_mlp is not None:
                # Broadcast a global summary back to every node (virtual node)
                H = H + self.global_mlp(H.mean(dim=1, keepdim=True))
        return H

    # ------------------------------------------------------------------
    # Public interface (matches MBQCActorCritic exactly)
    # ------------------------------------------------------------------

    def forward(
        self,
        obs:         Tensor,
        action_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            obs:         (B, obs_dim)
            action_mask: (B, n) — 1 = valid, 0 = invalid
        Returns:
            logits: (B, n)  — invalid actions set to −1e9
            value:  (B,)
        """
        H      = self._embed(obs)                             # (B, n, hidden_dim)
        logits = self.actor_head(H).squeeze(-1)               # (B, n)
        logits = logits + (1.0 - action_mask.float()) * (-1e9)
        value  = self.critic_mlp(H.mean(dim=1)).squeeze(-1)  # (B,)
        return logits, value

    def get_action_and_value(
        self,
        obs:         Tensor,
        action_mask: Tensor,
        action:      Tensor | None = None,
        return_aux:  bool = False,
    ):
        """Single-embed action/value evaluation. If return_aux, also return the
        auxiliary per-node gflow-layer prediction (B, n) — used by the trainer's
        auxiliary supervised loss."""
        H      = self._embed(obs)
        logits = self.actor_head(H).squeeze(-1) + (1.0 - action_mask.float()) * (-1e9)
        value  = self.critic_mlp(H.mean(dim=1)).squeeze(-1)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        lp, ent = dist.log_prob(action), dist.entropy()
        if return_aux:
            aux = self.aux_head(H).squeeze(-1) if self.aux_head is not None else None
            return action, lp, ent, value, aux
        return action, lp, ent, value

    def get_value(self, obs: Tensor) -> Tensor:
        H = self._embed(obs)
        return self.critic_mlp(H.mean(dim=1)).squeeze(-1)

    @torch.no_grad()
    def get_attention(self, obs: Tensor) -> list[Tensor]:
        """
        Run the GNN and capture the attention coefficients of every GAT layer.

        Returns a list of length n_layers; entry ℓ is (B, n, n, n_heads), where
        [b, i, j, h] is how much node i attends to node j in head h at layer ℓ.
        Used by scripts/visualize_attention.py for the interpretability figure.
        """
        feats, A = self._node_features(obs)
        H = torch.tanh(self.input_proj(feats))
        attentions: list[Tensor] = []
        for i in range(self.n_layers):
            layer = self.gat_layers[0] if self.weight_tied else self.gat_layers[i]
            H, alpha = layer(H, A, return_attention=True)
            attentions.append(alpha)
            if self.global_mlp is not None:
                H = H + self.global_mlp(H.mean(dim=1, keepdim=True))
        return attentions

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        sqrt2 = math.sqrt(2)
        nn.init.orthogonal_(self.input_proj.weight, gain=sqrt2)
        nn.init.zeros_(self.input_proj.bias)
        # Actor output: near-uniform initial policy
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.zeros_(self.actor_head.bias)
        # Critic MLP
        for module in self.critic_mlp:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.zeros_(module.bias)
        # Virtual-node global MLP
        if self.global_mlp is not None:
            for module in self.global_mlp:
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, gain=1.0)
                    nn.init.zeros_(module.bias)
        # Auxiliary layer-prediction head
        if self.aux_head is not None:
            nn.init.orthogonal_(self.aux_head.weight, gain=1.0)
            nn.init.zeros_(self.aux_head.bias)
