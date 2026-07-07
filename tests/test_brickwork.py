"""
Tests for the brickwork topology (BFK 2009 universal MBQC resource state)
and its integration into MBQCEnv.
"""

import numpy as np
import pytest

from mbqc_rl.env.mbqc_env import MBQCEnv
from mbqc_rl.utils.generators import make_brickwork_graph, make_grid_graph
from mbqc_rl.utils.gflow import compute_gflow
from mbqc_rl.baselines.classical import GreedyGflowBaseline


class TestBrickworkGraph:
    def test_node_count(self):
        G, _ = make_brickwork_graph(4, 5, rng=np.random.default_rng(0))
        assert G.number_of_nodes() == 20

    def test_output_qubits_rightmost_column(self):
        G, outs = make_brickwork_graph(4, 5, rng=np.random.default_rng(0))
        assert outs == [4, 9, 14, 19]          # rightmost column, same as grid

    def test_full_horizontal_edges(self):
        """Every horizontal edge present on a perfect brickwork graph."""
        rows, cols = 4, 6
        G, _ = make_brickwork_graph(rows, cols, defect_rate=0.0,
                                    rng=np.random.default_rng(0))
        for r in range(rows):
            for c in range(cols - 1):
                assert G.has_edge(r*cols + c, r*cols + c + 1)

    def test_staggered_vertical_rungs(self):
        """Vertical rung (r,c)-(r+1,c) present iff (r+c) is odd."""
        rows, cols = 4, 6
        G, _ = make_brickwork_graph(rows, cols, defect_rate=0.0,
                                    rng=np.random.default_rng(0))
        for r in range(rows - 1):
            for c in range(cols):
                u, v = r*cols + c, (r+1)*cols + c
                expected = (r + c) % 2 == 1
                assert G.has_edge(u, v) == expected

    def test_fewer_edges_than_grid(self):
        """Brickwork keeps only ~half the vertical edges → sparser than grid."""
        rng_a = np.random.default_rng(0)
        rng_b = np.random.default_rng(0)
        Gb, _ = make_brickwork_graph(5, 5, defect_rate=0.0, rng=rng_a)
        Gg, _ = make_grid_graph(5, 5, defect_rate=0.0, rng=rng_b)
        assert Gb.number_of_edges() < Gg.number_of_edges()

    def test_defects_remove_edges(self):
        G_full, _ = make_brickwork_graph(5, 5, defect_rate=0.0,
                                         rng=np.random.default_rng(0))
        G_def, _  = make_brickwork_graph(5, 5, defect_rate=0.5,
                                         rng=np.random.default_rng(0))
        assert G_def.number_of_edges() < G_full.number_of_edges()

    def test_invalid_cols(self):
        with pytest.raises(ValueError):
            make_brickwork_graph(3, 1)


class TestBrickworkGflow:
    @pytest.mark.parametrize("rows,cols", [(3,3),(4,4),(5,5),(3,5),(4,6),(6,6)])
    def test_gflow_exists_perfect(self, rows, cols):
        """Brickwork is a valid MBQC resource → perfect graphs have gflow."""
        G, outs = make_brickwork_graph(rows, cols, defect_rate=0.0,
                                       rng=np.random.default_rng(1))
        gflow, _ = compute_gflow(G, outs)
        assert gflow is not None


class TestBrickworkEnv:
    def test_topology_selects_brickwork(self):
        env = MBQCEnv(rows=4, cols=4, topology="brickwork")
        env_grid = MBQCEnv(rows=4, cols=4, topology="grid")
        env.reset(seed=0); env_grid.reset(seed=0)
        # Brickwork is sparser, so its adjacency has fewer 1s
        bw_edges = env._adj.sum()
        gr_edges = env_grid._adj.sum()
        assert bw_edges < gr_edges

    def test_invalid_topology_raises(self):
        with pytest.raises(ValueError):
            MBQCEnv(rows=3, cols=3, topology="hexagonal")

    def test_obs_shape_unchanged(self):
        """Topology does not change observation dimension."""
        env = MBQCEnv(rows=3, cols=3, topology="brickwork")
        obs, _ = env.reset(seed=0)
        assert obs["observation"].shape == (9*9 + 9,)

    def test_episode_completes(self):
        env = MBQCEnv(rows=4, cols=4, topology="brickwork", defect_rate=0.0)
        obs, _ = env.reset(seed=3)
        done, total = False, 0.0
        while not done:
            valid = np.where(obs["action_mask"] == 1)[0]
            obs, r, term, trunc, _ = env.step(int(valid[0]))
            total += r; done = term or trunc
        assert 0.0 <= total <= 1.0

    def test_greedy_scores_one_on_perfect_brickwork(self):
        """The gflow oracle achieves reward 1.0 on a perfect brickwork graph."""
        env = MBQCEnv(rows=4, cols=4, topology="brickwork", defect_rate=0.0)
        obs, info = env.reset(seed=7)
        b = GreedyGflowBaseline(env); b.reset(obs, info)
        done, total = False, 0.0
        while not done:
            obs, r, term, trunc, _ = env.step(b.select_action(obs))
            total += r; done = term or trunc
        assert total == 1.0

    def test_brickwork_with_angles(self):
        env = MBQCEnv(rows=4, cols=4, topology="brickwork",
                      use_angles=True, defect_rate=0.0)
        obs, info = env.reset(seed=1)
        assert obs["observation"].shape == (16*16 + 2*16,)
        assert len(info["angles"]) == 16
