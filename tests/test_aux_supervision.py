"""
Tests for rung 4 — auxiliary gflow-layer supervision: the GNN aux head, the
buffer's auxiliary-target storage, the env's layer-target vector, and the PPO
trainer's auxiliary loss path. Non-aux paths must be unaffected.
"""

import numpy as np
import torch

from mbqc_rl.env.mbqc_env import MBQCEnv
from mbqc_rl.agent.gnn_policy import GNNActorCritic
from mbqc_rl.agent.buffer import RolloutBuffer
from mbqc_rl.agent.ppo import PPOTrainer

N = 16
OBS = N * N + N


class TestGNNAuxHead:
    def test_return_aux_shapes(self):
        p = GNNActorCritic(n=N, hidden_dim=64, n_layers=3, aux_layer_head=True)
        obs, mask = torch.randn(5, OBS), torch.ones(5, N)
        out = p.get_action_and_value(obs, mask, return_aux=True)
        assert len(out) == 5
        action, lp, ent, value, aux = out
        assert aux.shape == (5, N)

    def test_no_aux_head_returns_none(self):
        p = GNNActorCritic(n=N, hidden_dim=64, n_layers=3, aux_layer_head=False)
        *_, aux = p.get_action_and_value(torch.randn(3, OBS), torch.ones(3, N),
                                         return_aux=True)
        assert aux is None

    def test_default_call_still_four_tuple(self):
        p = GNNActorCritic(n=N, hidden_dim=64, n_layers=3, aux_layer_head=True)
        out = p.get_action_and_value(torch.randn(3, OBS), torch.ones(3, N))
        assert len(out) == 4               # backward-compatible default


class TestBufferAux:
    def test_aux_roundtrip(self):
        buf = RolloutBuffer(n_steps=4, obs_dim=OBS, n_actions=N, aux_dim=N)
        tgt = np.arange(N, dtype=np.float32) / N
        for _ in range(4):
            buf.add(np.zeros(OBS), np.ones(N), 0, 0.0, False, 0.0, 0.0, aux_target=tgt)
        buf.compute_returns_and_advantages(0.0, True)
        data = buf.get_all()
        assert "aux_targets" in data
        assert data["aux_targets"].shape == (4, N)
        assert torch.allclose(data["aux_targets"][0], torch.as_tensor(tgt))

    def test_no_aux_dim_omits_key(self):
        buf = RolloutBuffer(n_steps=4, obs_dim=OBS, n_actions=N)   # aux_dim=0
        buf.add(np.zeros(OBS), np.ones(N), 0, 0.0, False, 0.0, 0.0)
        buf.compute_returns_and_advantages(0.0, True)
        assert "aux_targets" not in buf.get_all()


class TestEnvLayerVector:
    def test_layer_vector_normalised(self):
        env = MBQCEnv(rows=4, cols=4, topology="irregular", defect_rate=0.0)
        env.reset(seed=0)
        vec = env.gflow_layer_vector
        assert vec.shape == (16,)
        assert vec.min() >= 0.0 and vec.max() <= 1.0
        # outputs are layer 0
        for o in env.output_qubits:
            assert vec[o] == 0.0


class TestPPOAuxIntegration:
    def test_one_update_trains_aux_head(self):
        env = MBQCEnv(rows=4, cols=4, topology="irregular",
                      observe_original_graph=True, reward_shaping=True,
                      defect_rate=(0.0, 0.0), seed=0)
        policy = GNNActorCritic(n=16, hidden_dim=64, n_heads=4, n_layers=3,
                                pos_dim=4, aux_layer_head=True)
        opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
        tr = PPOTrainer(env, policy, opt, n_steps=64, n_epochs=2,
                        batch_size=16, aux_weight=1.0)
        before = policy.aux_head.weight.detach().clone()
        tr.collect_rollout()
        assert "aux_targets" in tr.buffer.get_all()
        stats = tr.update()
        assert stats["aux_loss"] > 0 and np.isfinite(stats["aux_loss"])
        assert not torch.equal(before, policy.aux_head.weight.detach())

    def test_aux_weight_zero_no_aux(self):
        """aux_weight=0 ⇒ buffer has no aux targets and update reports aux_loss 0."""
        env = MBQCEnv(rows=3, cols=3, defect_rate=0.0, seed=0)
        policy = GNNActorCritic(n=9, hidden_dim=64, n_layers=2)
        opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
        tr = PPOTrainer(env, policy, opt, n_steps=48, n_epochs=1, batch_size=16)
        tr.collect_rollout()
        assert "aux_targets" not in tr.buffer.get_all()
        assert tr.update()["aux_loss"] == 0.0
