"""
Tests for the distance-from-output vs gflow-layer characterization (#2 theory).

On a defect-free grid, the gflow layer L(v) equals the graph distance d(v) from
the output column, so the distance heuristic is optimal. On irregular graphs
(skip edges) L(v) >= d(v) with strict inequality for some v, so it is not.
"""

import numpy as np
import networkx as nx
import pytest

from mbqc_rl.utils.generators import make_grid_graph, make_irregular_flow_graph
from mbqc_rl.utils.gflow import compute_gflow


def _dist(G, outs):
    H = G.copy(); H.add_node("s")
    for o in outs:
        H.add_edge("s", o)
    d = nx.single_source_shortest_path_length(H, "s")
    return {v: d[v] - 1 for v in G.nodes()}


class TestGridLayerEqualsDistance:
    @pytest.mark.parametrize("rows,cols", [(3, 3), (4, 4), (3, 5), (4, 6)])
    def test_layer_equals_distance_on_perfect_grid(self, rows, cols):
        G, outs = make_grid_graph(rows, cols, defect_rate=0.0,
                                  rng=np.random.default_rng(0))
        _, order = compute_gflow(G, outs)
        d = _dist(G, outs)
        for v in G.nodes():
            if v not in set(outs):
                assert order[v] == d[v], (rows, cols, v, order[v], d[v])


class TestIrregularBreaksIt:
    def test_layer_geq_distance_and_sometimes_strict(self):
        strict_any = False
        for i in range(40):
            rng = np.random.default_rng(100 + i)
            G, outs = make_irregular_flow_graph(4, 4, defect_rate=0.0, rng=rng)
            gflow, order = compute_gflow(G, outs)
            if gflow is None:
                continue
            d = _dist(G, outs)
            out_set = set(outs)
            for v in G.nodes():
                if v not in out_set:
                    assert order[v] >= d[v]              # L >= d always
                    if order[v] > d[v]:
                        strict_any = True
        assert strict_any   # skip edges make L > d for some nodes
