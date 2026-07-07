# Figure suite

Regenerate all: `python scripts/make_figures.py`
Regenerate some: `python scripts/make_figures.py --only C3 R1`

## Conceptual figures — demonstrate understanding of the problem

| File | What it shows | Suggested paper use |
|------|---------------|---------------------|
| `C1_defect_gflow.png` | Perfect cluster state vs one with 4 edge defects; gflow exists → BROKEN. gflow status computed live. | §Background / Problem — what a defect *is* and why it matters |
| `C2_gflow_layers.png` | gflow layer decomposition; nodes labelled by layer; measurement-order arrow | §Background — the gflow ordering the agent must learn |
| `C3_gflow_vs_defect.png` | P(valid gflow exists) vs defect rate, for 3×3…6×6. **The key "why it's hard" figure.** | §Problem / §Results discussion — explains the reward ceiling and high-defect fall-off |
| `C4_angle_constraints.png` | Clifford (free) vs non-Clifford (hard-constraint) qubits on a graph | §Method — universal MBQC and the angle-aware reward |
| `C5_rollout.png` | An episode: graph degrading measurement by measurement | §Method — the MDP formulation |

## Results figures

| File | What it shows | Suggested paper use |
|------|---------------|---------------------|
| `R1_scaling.png` | GNN zero-shot agent vs oracle vs random, 9→36 qubits | §Results — size generalization (headline) |
| `R2_mlp_suite.png` | 18-run MLP suite, plain vs angled, mean±std | §Results — statistical rigor; angled improves stability |
| `R3_defect_transfer.png` | GNN reward vs defect rate, home 4×4 and zero-shot 5×5 | §Results — robustness; ties to C3 |
| `R4_reward_hist.png` | Reward distributions: agent vs oracle vs random | §Results — agent mass sits at reward 1.0 |
| `R5_forgetting.png` | Staged curriculum forgets; mixed-rate fix | §Results — the forgetting diagnosis & fix |
| `R6_topology.png` | Grid vs brickwork structure + zero-shot performance | §Results — topology generalization (brickwork) |
| `R7_human_baseline.png` | Four-way: oracle / agent / human static schedule / random vs defect rate | §Results / §Discussion — honest comparison vs a strong hand-designed baseline |
| `R8_irregular_win.png` | **The irregular win.** Early-stopped ensemble vs distance heuristic vs oracle, at 4×4/5×5/6×6, under both the plain and the non-Clifford (angled) reward; third panel = win-margin Δ (angled ≥ plain at every size) | §Results — **headline positive result**: RL beats the strong heuristic where it is not optimal, across 3 sizes and both rewards |

## Provenance / honesty notes
- C1, C2, C3, C4, C5, R1, R3, R4 are computed fresh from the code/checkpoints at run time.
- R2 reads `results/seeds/*.json` (the 18-run aggregates).
- R5 and R6's performance bars use the verified numbers documented in
  `DISSERTATION_LOG.md` (Sessions 5–6 for forgetting; Session 10 for brickwork).
  Source each in captions.
- R8 uses the verified early-stopping/ensemble numbers from
  `early_stopping_eval.py` + `early_stopping_validate.py` (Sessions 24–29;
  RESULTS.md §7). Regenerate: `python scripts/make_figures.py --only R8`.
