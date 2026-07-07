# MBQC defective-instance benchmark (v1.0)

A fixed, versioned, labelled dataset of defective MBQC graph-state instances — the
materialised form of the **"1,000 randomised defective grid instances"** named in
the project proposal, released as a reusable benchmark for future research.

Each instance ships with **ground-truth gflow labels** (the deterministic
measurement-order the agent must learn) and **reference baseline rewards**, so a
new method (RL, heuristic, or supervised) can be compared on a common yardstick
without regenerating anything or reimplementing gflow.

Build / rebuild: `python scripts/build_benchmark.py`
Load: `from mbqc_rl.benchmark import load_benchmark, instance_to_graph`
Integrity tests: `pytest tests/test_benchmark_dataset.py`

## Files

| file | instances | composition | sha256 |
|------|:---:|---|---|
| `grids_defective_n1000.jsonl` | **1000** | 4 sizes {3×3,4×4,5×5,6×6} × 5 defect rates {0.01,0.05,0.10,0.20,0.30} × 50 | `96dc0ec832391b85…` |
| `irregular_n500.jsonl` | 501 | 3 sizes {4×4,5×5,6×6} × 167, clean (defect 0) — the regime where the learned policy beats the distance heuristic | `2ae57c716dd1ac3a…` |
| `manifest.json` | — | schema, per-(size,rate) gflow-existence rate + mean oracle/heuristic/random reward | — |

## Record schema (one JSON object per line)

```jsonc
{
  "id": "grid_4x4_d0.05_s1012003",
  "topology": "grid",                  // "grid" | "irregular"
  "rows": 4, "cols": 4, "n_qubits": 16,
  "defect_rate": 0.05,
  "seed": 1012003,                     // reproduces the instance exactly (see below)
  "edges": [[0,1],[1,4], ...],         // undirected; nodes 0..n_qubits-1
  "output_qubits": [3,7,11,15],        // last column (one per row)
  "gflow_exists": true,
  "gflow_order": {"0":3,"1":2, ...},   // GROUND TRUTH: node → layer (output=0,
                                       //   higher = measured earlier). null if none.
  "gflow_correction_sets": {"0":[3,7], ...},   // g(v) per non-output node. null if none.
  "reference_reward": {                // gflow-consistency objective, in [0,1]
    "oracle": 1.0,                     //   greedy-gflow schedule (=1.0 iff gflow exists)
    "distance_heuristic": 0.87,        //   "measure farthest-from-output first"
    "random": 0.41                     //   uniform valid order
  }
}
```

## How to use it

- **Evaluate a method:** for each instance, reconstruct the graph
  (`instance_to_graph`), run your policy to produce a measurement order, score it
  with the gflow-consistency reward, and compare the mean to the stored
  `reference_reward` (oracle = ceiling, distance_heuristic = the strong classical
  baseline to beat, random = floor).
- **Supervised / imitation learning:** `gflow_order` is a ready-made per-node
  label; `gflow_correction_sets` gives the full gflow.
- **Study the gflow-existence boundary:** `gflow_exists` across the 5 defect rates
  traces how fast defects destroy determinism (see `manifest.json` `cells`), e.g.
  6×6 grids: 48/50 solvable at 1% defect → 0/50 at 30%.

## Provenance & reproducibility

- Instances are generated through the exact `MBQCEnv` seed→graph path used in all
  experiments: `np.random.default_rng(seed)` → `make_grid_graph` /
  `make_irregular_flow_graph`. Re-running the builder reproduces every record
  bit-for-bit (sha256 in the manifest).
- Seeds are **held out**: grid base `1_000_000`, irregular base `2_000_000` —
  disjoint from training (per-episode random) and from the early-stopping
  validation/test seeds (50k, 90k–500k) used for the headline results.
- gflow is computed by `mbqc_rl.utils.gflow.compute_gflow`, cross-validated 100%
  against the tested `graphix` library (RESULTS.md §8); on a valid gflow the
  computation is provably deterministic (infidelity ≈ 2e-12).

Reference means (full sets): grids — oracle 0.575 / heuristic 0.560 / random 0.287
(low because heavy-defect cells are often gflow-less → reward 0 by construction);
irregular — oracle 0.998 / heuristic 0.913 / random 0.504.
