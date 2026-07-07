"""
Tests for the irregular flow-graph topology and the distance-schedule baseline.

The scientific point of this topology: it has a valid gflow but its optimal
measurement order is NOT monotonic in distance-from-output, so the strong
structural heuristic (DistanceScheduleBaseline) is provably suboptimal — the
regime where a learned policy can beat hand-designed heuristics.
"""

import numpy as np
import pytest

from mbqc_rl.env.mbqc_env import MBQCEnv
from mbqc_rl.utils.generators import make_irregular_flow_graph
from mbqc_rl.utils.gflow import compute_gflow, score_measurement_order
from mbqc_rl.baselines.classical import (
    GreedyGflowBaseline, DistanceScheduleBaseline,
)


class TestIrregularGraph:
    def test_node_and_output_count(self):
        G, outs = make_irregular_flow_graph(4, 4, rng=np.random.default_rng(0))
        assert G.number_of_nodes() == 16
        assert outs == [12, 13, 14, 15]          # last `rows` ids

    def test_perfect_graph_has_gflow(self):
        """Construction guarantees a gflow on every perfect graph."""
        for seed in range(20):
            G, outs = make_irregular_flow_graph(4, 4, defect_rate=0.0,
                                                rng=np.random.default_rng(seed))
            gf, _ = compute_gflow(G, outs)
            assert gf is not None

    def test_connected_to_outputs(self):
        """Every non-output reaches an output (flow chains guarantee this)."""
        import networkx as nx
        G, outs = make_irregular_flow_graph(4, 5, defect_rate=0.0,
                                            rng=np.random.default_rng(1))
        out_set = set(outs)
        for v in G.nodes():
            if v not in out_set:
                assert any(nx.has_path(G, v, o) for o in outs)

    def test_invalid_cols(self):
        with pytest.raises(ValueError):
            make_irregular_flow_graph(3, 1)


class TestRegimeExists:
    """The whole reason for this topology: the distance heuristic is suboptimal."""

    def _order_to_steps(self, order):
        return {node: i + 1 for i, node in enumerate(order)}

    def test_distance_heuristic_is_suboptimal(self):
        import networkx as nx
        oracle_scores, dist_scores = [], []
        for i in range(60):
            G, outs = make_irregular_flow_graph(4, 4, defect_rate=0.0,
                                                rng=np.random.default_rng(200 + i))
            gf, order = compute_gflow(G, outs)
            if gf is None:
                continue
            out_set = set(outs)
            non_out = [v for v in G.nodes() if v not in out_set]
            # oracle: descending gflow layer
            oo = sorted(non_out, key=lambda v: -order[v])
            oracle_scores.append(
                score_measurement_order(gf, self._order_to_steps(oo), out_set))
            # distance heuristic: farthest-from-output first (super-source BFS)
            H = G.copy(); H.add_node("s")
            for o in outs:
                H.add_edge("s", o)
            d = nx.single_source_shortest_path_length(H, "s")
            do = sorted(non_out, key=lambda v: -(d.get(v, 10**6) - 1))
            dist_scores.append(
                score_measurement_order(gf, self._order_to_steps(do), out_set))

        # Oracle is perfect by construction; the heuristic is clearly below it.
        assert np.mean(oracle_scores) == pytest.approx(1.0, abs=1e-9)
        assert np.mean(dist_scores) < 0.97   # effect is large (~0.89 observed)


class TestIrregularEnv:
    def test_env_runs(self):
        env = MBQCEnv(rows=4, cols=4, topology="irregular", defect_rate=0.0)
        obs, info = env.reset(seed=0)
        assert obs["observation"].shape == (16 * 16 + 16,)
        assert info["gflow_exists"]

    def test_oracle_beats_distance_on_irregular(self):
        """Within one env, the gflow oracle should not be worse than distance."""
        env = MBQCEnv(rows=4, cols=4, topology="irregular", defect_rate=0.0)

        def run(make_b, seed):
            obs, info = env.reset(seed=seed)
            b = make_b(env); b.reset(obs, info)
            tot, done = 0.0, False
            while not done:
                obs, r, t, tr, _ = env.step(b.select_action(obs)); tot += r; done = t or tr
            return tot

        oracle = np.mean([run(GreedyGflowBaseline, s) for s in range(40)])
        dist   = np.mean([run(DistanceScheduleBaseline, s) for s in range(40)])
        assert oracle >= dist


class TestDistanceBaselineOnGrid:
    def test_reduces_to_column_sweep_on_perfect_grid(self):
        """On a defect-free grid, distance order = column sweep = reward 1.0."""
        env = MBQCEnv(rows=3, cols=3, topology="grid", defect_rate=0.0)
        obs, info = env.reset(seed=0)
        b = DistanceScheduleBaseline(env); b.reset(obs, info)
        tot, done = 0.0, False
        while not done:
            obs, r, t, tr, _ = env.step(b.select_action(obs)); tot += r; done = t or tr
        assert tot == 1.0


class TestObserveOriginalGraph:
    """The static-observation work-around: adjacency stays full across the episode."""

    def test_original_adjacency_constant(self):
        n = 16
        env = MBQCEnv(rows=4, cols=4, topology="irregular",
                      observe_original_graph=True, defect_rate=0.0)
        obs, _ = env.reset(seed=1)
        a0 = obs["observation"][:n * n].copy()
        for _ in range(3):
            v = np.where(obs["action_mask"] == 1)[0]
            obs, _, t, tr, _ = env.step(int(v[0]))
            if t or tr:
                break
        assert np.array_equal(a0, obs["observation"][:n * n])

    def test_default_adjacency_degrades(self):
        n = 16
        env = MBQCEnv(rows=4, cols=4, topology="irregular", defect_rate=0.0)
        obs, _ = env.reset(seed=1)
        a0 = obs["observation"][:n * n].copy()
        v = np.where(obs["action_mask"] == 1)[0]
        obs, _, _, _, _ = env.step(int(v[0]))
        assert not np.array_equal(a0, obs["observation"][:n * n])

    def test_obs_dim_unchanged(self):
        env = MBQCEnv(rows=4, cols=4, topology="irregular",
                      observe_original_graph=True)
        obs, _ = env.reset(seed=0)
        assert obs["observation"].shape == (16 * 16 + 16,)
