"""
Tests for the configurable GNN options used in the irregular-regime solution
ladder: virtual node (rung 1), weight-tied processor (rung 2), random-walk
structural encodings (rung 3), and their combinations. Defaults must preserve
the original architecture (backward compatibility).
"""

import torch
import pytest

from mbqc_rl.agent.gnn_policy import GNNActorCritic

N = 16
OBS_DIM = N * N + N


def _forward_ok(policy, batch=6):
    obs = torch.randn(batch, OBS_DIM)
    mask = torch.ones(batch, N)
    mask[:, 0] = 0
    logits, value = policy(obs, mask)
    assert logits.shape == (batch, N)
    assert value.shape == (batch,)
    assert torch.isfinite(logits[:, 1:]).all()
    assert (logits[:, 0] < -1e8).all()           # masked action suppressed
    # gradients reach every parameter under a combined loss
    a, lp, ent, v = policy.get_action_and_value(obs, mask)
    (-lp.mean() + 0.5 * v.pow(2).mean() - 0.01 * ent.mean()).backward()
    missing = [n for n, p in policy.named_parameters()
               if p.requires_grad and p.grad is None]
    assert not missing, missing


class TestLadderOptions:
    def test_baseline_backward_compatible(self):
        p = GNNActorCritic(n=N, hidden_dim=64, n_layers=3)
        assert p.global_mlp is None
        assert p.weight_tied is False and p.pos_dim == 0
        assert len(p.gat_layers) == 3            # one conv per layer
        _forward_ok(p)

    def test_virtual_node(self):
        p = GNNActorCritic(n=N, hidden_dim=64, n_layers=3, virtual_node=True)
        assert p.global_mlp is not None
        _forward_ok(p)

    def test_weight_tied_shares_one_conv(self):
        p = GNNActorCritic(n=N, hidden_dim=64, n_layers=5, weight_tied=True)
        assert len(p.gat_layers) == 1            # shared across 5 applications
        _forward_ok(p)

    def test_pos_encoding_grows_input(self):
        k = 8
        p = GNNActorCritic(n=N, hidden_dim=64, n_layers=3, pos_dim=k)
        assert p.input_proj.in_features == 1 + k     # meas channel + RWSE
        _forward_ok(p)

    def test_pos_encoding_with_angles(self):
        k = 4
        p = GNNActorCritic(n=N, hidden_dim=64, n_layers=3, use_angles=True, pos_dim=k)
        assert p.input_proj.in_features == 2 + k     # meas + angle + RWSE
        obs = torch.randn(4, N * N + 2 * N)
        mask = torch.ones(4, N)
        logits, value = p(obs, mask)
        assert logits.shape == (4, N)

    def test_all_combined(self):
        p = GNNActorCritic(n=N, hidden_dim=64, n_layers=5,
                           virtual_node=True, weight_tied=True, pos_dim=8)
        assert p.global_mlp is not None
        assert len(p.gat_layers) == 1
        assert p.input_proj.in_features == 1 + 8
        _forward_ok(p)

    def test_rwse_is_size_inductive(self):
        """RWSE is computed from the adjacency, so a model still runs at a new n."""
        p = GNNActorCritic(n=9, hidden_dim=64, n_layers=3, pos_dim=6)
        obs = torch.randn(2, 25 * 25 + 25)       # feed a 5×5 obs to a 3×3 model
        mask = torch.ones(2, 25)
        logits, _ = p(obs, mask)
        assert logits.shape == (2, 25)

    def test_attention_path_with_options(self):
        p = GNNActorCritic(n=N, hidden_dim=64, n_layers=4,
                           virtual_node=True, weight_tied=True, pos_dim=4)
        att = p.get_attention(torch.randn(1, OBS_DIM))
        assert len(att) == 4
        assert att[0].shape == (1, N, N, 4)
