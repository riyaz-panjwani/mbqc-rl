# Irregular scale-test runs

Winning combo (static-obs + reward-shaping + depth-5 GNN + virtual node + RWSE),
trained on clean irregular graphs, 3 seeds per size, for the early-stopping +
ensemble protocol (`scripts/early_stopping_eval.py`, `..._validate.py`).

| dir              | size | status  | notes |
|------------------|------|---------|-------|
| `r5x5_s{0,1,2}`  | 5×5  | VALID   | ensemble 0.944 vs heuristic 0.915, p=4e-17 |
| `r6x6_s{0,1,2}`  | 6×6  | **INVALID — DO NOT EVAL** | trained on the buggy generator (see below) |
| `r6x6fix_s{0,1,2}` | 6×6 | VALID | ensemble 0.952 vs heuristic 0.929, p=8e-21 |

## Why `r6x6_s*` is invalid (kept deliberately as debugging evidence)

`make_irregular_flow_graph` originally had `_max_tries = 80`. At 6×6, a single
graph-construction attempt yields a valid gflow only ~2% of the time, so 80 tries
found a valid gflow in only **69%** of graphs — the other **31% of episodes were
gflow-less**, with reward pinned at 0 regardless of the agent's actions. This
corrupted both training and the eval baselines: on `r6x6_s*` the oracle scored
only 0.61 (vs ~0.99 when healthy) and the ensemble 0.54 < heuristic 0.57, which
*looked* like a failure of the method.

The tell was the **oracle ceiling**: an oracle that can't reach ~1.0 on clean
irregular graphs means the graphs themselves are broken, not the agent. Raising
`_max_tries` to 400 restored 99.5% gflow success at 6×6 (4×4/5×5 always succeeded
on the first try and were never affected), and the win reproduced cleanly on
`r6x6fix_s*`.

Kept as evidence of the diagnosis — see DISSERTATION_LOG.md Session 28.
