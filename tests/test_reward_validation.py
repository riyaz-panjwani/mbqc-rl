"""
Reward-validation regression tests (require the `graphix` MBQC library).

These pin the two facts that make the gflow-consistency reward physically
meaningful, cross-checked against graphix's independent, tested implementation:
  (A) our compute_gflow agrees with graphix.OpenGraph.find_gflow (no inputs);
  (B) a valid gflow pattern is deterministic (output independent of outcomes).

Skipped automatically if graphix is not installed.
"""

import itertools
import numpy as np
import pytest

graphix = pytest.importorskip("graphix")
from graphix.opengraph import OpenGraph
from graphix.measurements import BlochMeasurement
from graphix.fundamentals import Plane

from mbqc_rl.utils.generators import make_grid_graph, make_brickwork_graph
from mbqc_rl.utils.gflow import compute_gflow


def _open_graph(G, outs, rng, clifford_fraction=0.5):
    non_out = [v for v in G.nodes() if v not in set(outs)]
    meas = {}
    for v in non_out:
        k = int(rng.choice([0, 2, 4, 6] if rng.random() < clifford_fraction
                           else [1, 3, 5, 7]))
        meas[v] = BlochMeasurement(k / 4.0, Plane.XY)
    return OpenGraph(G, input_nodes=[], output_nodes=outs, measurements=meas)


class TestGflowCrossValidation:
    @pytest.mark.parametrize("rows,cols,defect", [
        (3, 3, 0.0), (3, 3, 0.15), (4, 4, 0.0), (4, 4, 0.15),
    ])
    def test_our_gflow_matches_graphix(self, rows, cols, defect):
        for i in range(25):
            rng = np.random.default_rng(1000 + i)
            G, outs = make_grid_graph(rows, cols, defect_rate=defect, rng=rng)
            ours = compute_gflow(G, outs)[0] is not None
            theirs = _open_graph(G, outs, rng).find_gflow() is not None
            assert ours == theirs, (rows, cols, defect, i, ours, theirs)

    def test_brickwork_matches(self):
        for i in range(20):
            rng = np.random.default_rng(7000 + i)
            G, outs = make_brickwork_graph(4, 4, defect_rate=0.0, rng=rng)
            ours = compute_gflow(G, outs)[0] is not None
            theirs = _open_graph(G, outs, rng).find_gflow() is not None
            assert ours == theirs


class TestGflowImpliesDeterminism:
    def test_gflow_pattern_is_deterministic(self):
        """A graph with gflow → canonical pattern output is outcome-independent."""
        checked = 0
        for i in range(15):
            rng = np.random.default_rng(2000 + i)
            G, outs = make_grid_graph(3, 3, defect_rate=0.0, rng=rng)
            og = _open_graph(G, outs, rng)
            if og.find_gflow() is None:
                continue
            checked += 1
            pat = og.to_pattern()
            states = []
            for s in range(4):
                st = pat.simulate_pattern("statevector",
                                          rng=np.random.default_rng(10 * i + s))
                v = np.asarray(st.flatten()).ravel()
                states.append(v / (np.linalg.norm(v) + 1e-12))
            for a, b in itertools.combinations(states, 2):
                assert 1.0 - abs(np.vdot(a, b)) < 1e-6     # deterministic
        assert checked > 0
