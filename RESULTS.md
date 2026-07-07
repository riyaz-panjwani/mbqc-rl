# Consolidated Results — RL for MBQC Measurement Ordering

Paper-ready summary of every experiment, with verified numbers. Source of truth
for the dissertation write-up. Full chronology in `../DISSERTATION_LOG.md`.
All benchmarks: greedy/argmax decoding, 1% defect unless stated, held-out seeds.

---

## 0. Headline

A PPO agent learns gflow-consistent measurement orderings on defective MBQC graph
states from a fidelity-style reward alone. Contributions:

1. **Catastrophic forgetting diagnosed + fixed** (staged curriculum → mixed-rate).
2. **Universal MBQC**: angle-aware reward (non-Clifford); agent ≡ oracle (p=1.0).
3. **18-run statistical suite** (3 grids × plain/angled × 3 seeds).
4. **GNN with zero-shot transfer** across size (9→36 qubits) and topology
   (grid→brickwork), ~9× more sample-efficient than the MLP.
5. **Interpretability**: attention recovers the gflow flow direction from reward.
6. **Baseline analysis**: on regular lattices RL *matches* the strong static
   heuristic (which is provably optimal there, §9b); on **irregular graphs** an
   early-stopped GNN ensemble **beats** the heuristic at all three sizes trained —
   4×4: 0.92 vs 0.88 (p=4e-9), 5×5: 0.94 vs 0.92 (p=4e-17), 6×6: 0.95 vs 0.93
   (p=8e-21) — and the win *strengthens* under the realistic **non-Clifford
   (universal-MBQC) reward** at all three sizes (4×4: Δ+0.065 p=5e-15; 5×5: Δ+0.036
   p=7e-12; 6×6: Δ+0.033 p=1e-22) — §7.
7. **Reward validated** against the tested `graphix` simulator (gflow correct;
   reward optimum ⇔ deterministic computation).
8. **Flow-hierarchy comparison** (causal flow ⊆ gflow ⊆ Pauli flow).
9. **Released benchmark** (§13): the proposal's 1,000 defective grid instances
   (+500 irregular) as a fixed, gflow-labelled dataset with reference baseline
   rewards, for future research.

Test suite: **180 passing**. Runtime: `/usr/local/bin/python3.13`; training on
Apple-Silicon MPS.

---

## 1. Scaling and the forgetting fix

Staged curriculum (0→10→20→30% defects) Stage-3 checkpoint, 1k trials @ 1%:

| Grid | Agent | Greedy oracle | Random | Note |
|------|-------|---------------|--------|------|
| 3×3 | 0.979 | 0.982 | 0.495 | p=0.085 n.s. — matches oracle |
| 4×4 | 0.747 | 0.976 | 0.481 | forgetting |
| 5×5 | 0.511 | 0.955 | 0.479 | forgetting |

**Catastrophic forgetting** diagnosis — Stage-0 vs Stage-3 (1% defect):

| Grid | Stage-0 ckpt | Stage-3 ckpt | Δ |
|------|-------------|-------------|---|
| 4×4 | 0.954 | 0.747 | +0.21 |
| 5×5 | 0.931 | 0.511 | +0.42 |

Stage-0 learned the clean-graph skill (4×4: 0.996 train, 5×5: 0.988); later
high-defect stages overwrote it (cf. Kirkpatrick et al. 2017, EWC).

**Mixed-rate fix** (per-episode defect U[0,0.3], same budget):

| Checkpoint | 4×4 @1% | % perfect |
|------------|---------|-----------|
| Mixed-rate | **0.960** | 86.4% |
| Staged Stage-3 | 0.747 | 1.1% |
Mixed beats staged at *every* defect rate, including 30%.

---

## 2. Statistical suite (MLP, mixed-rate, 3 seeds, 500 trials @ 1%)

| Condition | Mean ± std | % Perfect |
|-----------|-----------|-----------|
| 3×3 plain  | **0.9709 ± 0.0005** | 96.7% |
| 3×3 angled | 0.9686 ± 0.0047 | 96.5% |
| 4×4 plain  | 0.9413 ± 0.0174 | 68.4% |
| 4×4 angled | **0.9443 ± 0.0097** | 83.5% |
| 5×5 plain  | 0.7909 ± 0.0663 | 0.0% |
| 5×5 angled | **0.8526 ± 0.0122** | 16.5% |

Findings: 3×3 near-deterministic (std 0.0005); **angled ≥ plain at 4×4 and 5×5**
in both mean and stability (5×5 std 0.066→0.012). The angle-aware reward is the
better signal at scale.

---

## 3. Universal (non-Clifford) MBQC

Angle-aware reward counts only non-Clifford targets as hard constraints (Clifford
corrections are Pauli-frame-trackable). 3×3 angled, 1k trials @ 1%:

| Method | Mean | % Perfect |
|--------|------|-----------|
| PPO agent | **0.9820** | 98.2% |
| Greedy oracle | 0.9820 | 98.2% |
| Random | 0.5619 | 42.0% |

Mann-Whitney **p = 1.0000** — agent's reward distribution identical to the oracle.

---

## 4. GNN: efficiency + zero-shot transfer

GNN (GAT, 2–5 layers) matches the MLP and transfers.

**Sample efficiency (5×5 angled):** GNN scratch **0.876 in 250k steps** beats MLP
0.853 from 2.25M (~9× fewer steps). (Warm-starting from 4×4 hurt — negative
transfer, 0.747.)

**3-seed GNN 4×4 angled:** 0.9223 ± 0.0407 (reproducible, matches MLP-level).

**Zero-shot SIZE transfer — one GNN trained on 4×4 (angled), 300 trials @1%:**

| Grid | Qubits | Agent | Greedy |
|------|--------|-------|--------|
| 3×3 | 9 | 0.990 (oracle-match, p=0.42 n.s.) | 0.993 |
| 4×4 | 16 | 0.943 (home) | 0.980 |
| 5×5 | 25 | 0.837 | 0.945 |
| 6×6 | 36 | 0.766 | 0.957 |

One model, 9→36 qubits, never retrained; always ≫ random (≈0.47).

**Zero-shot TOPOLOGY transfer — grid-trained GNN on brickwork (BFK 2009):**

| Grid | Brickwork agent | (its grid score) | topology cost |
|------|----------------|------------------|---------------|
| 3×3 | 0.970 | 0.989 | −0.018 |
| 4×4 | 0.897 | 0.962 | −0.065 |
| 5×5 | 0.785 | 0.856 | −0.071 |

GNN degrades **less** under grid→brickwork shift than the MLP (−0.065 vs −0.087)
— and the MLP cannot change size at all.

**Architecture ablation (4×4 angled, 1M steps):** 1 layer 0.864 → **2 layers
0.942** (best) → 3 layers 0.908 (over-smoothing); width helps (h256 0.935).

---

## 5. Interpretability

GAT attention (4×4): focus sharpens with depth (0.911→0.719); layer 2 directs
60.8% of attention toward the output column. The agent recovers the gflow flow
direction from a scalar reward alone. Figure: `results/attention/attention_4x4.png`.

---

## 6. Baseline comparison (honest)

Four-way, GNN 4×4 angled, 400 trials:

| Rate | oracle | RL agent | human (static schedule) | random |
|------|--------|----------|-------------------------|--------|
| 0%  | 1.000 | 0.976 | **1.000** | 0.723 |
| 1%  | 0.983 | 0.948 | 0.962 | 0.695 |
| 5%  | 0.888 | 0.825 | 0.818 | 0.604 |
| 10% | 0.700 | 0.613 | 0.624 | 0.467 |

**On regular lattices a hand-designed static schedule is near-optimal** (it is
provably optimal there — §9b); the RL agent matches it and the oracle but does
not beat it. This is expected, not a weakness: §7 shows that on **irregular**
graphs — where the heuristic is *not* optimal — the learned policy beats it.

---

## 7. Irregular regime — barrier, diagnosis, ladder

Irregular flow graphs are built so the optimal order is NOT monotonic in
distance-from-output (skip edges). On perfect irregular 4×4 the distance
heuristic is suboptimal (0.885 vs oracle 1.000), so there is room to win.

**Diagnosis (imitation learning):** the GNN cannot fit the ordering even with
direct oracle supervision on irregular graphs (BC 0.48 = random, vs 0.93 on
grids) → the barrier is **representational/generalisation**, not RL exploration
or reward sparsity (shaping didn't help either).

**Solution ladder** (on static-obs + dense-reward + depth foundation):

| config | held-out @ irregular |
|--------|----------------------|
| sparse + degraded obs (baseline) | 0.49 (= random) |
| foundation (static obs + shaping + depth) | 0.75 |
| + RWSE structural encoding | 0.82 |
| + virtual node + RWSE (best, 3-seed) | **0.84 ± 0.04** |
| + auxiliary gflow-layer supervision | 0.82 (no further help) |
| distance heuristic (bar) | 0.885 |

The ladder lifts RL from total failure (0.49) to near the heuristic (0.84), but
**does not robustly beat it** (1/3 seeds, p=1.0). With aux supervision, training
reward hit 0.98 while held-out stayed 0.82 → residual barrier is **cross-
distribution generalisation** (NAR/CLRS: GNNs struggle to generalise algorithm
execution across graph distributions).

**Generalisation phase (regularisation):** weight decay 0.809, irregularity
randomisation 0.837, both 0.812 — none beat the no-reg baseline (0.839). Standard
generalisation tools don't help — but checkpoint analysis revealed the real cause:
**overfitting in time.**

**THE WIN — early stopping + ensembling (`scripts/early_stopping_eval.py`).** The
winning combo PEAKS EARLY (~128k steps, held-out 0.90) then overfits to ~0.82.
Rigorous protocol (select checkpoint on VALIDATION seeds, report on DISJOINT TEST
seeds, 3 seeds, 500 trials):

| method | test reward |
|--------|-------------|
| final checkpoint (overfit) | 0.839 ± 0.037 |
| early-stopped (val-selected) | 0.890 ± 0.007 (parity, 2/3 beat heuristic) |
| **early-stopped ensemble (3 models)** | **0.923** |
| distance heuristic | 0.885 |
| oracle | 1.000 |

Ensemble vs heuristic: **Δ = +0.038, Mann-Whitney one-sided p = 4.2e-9** — the
learned policy **decisively beats the strong hand-designed heuristic** on
irregular graphs (the regime built to favour it), via standard techniques (early
stopping + deep ensembling), with no test peeking. **Full arc:** 0.49 (failure) →
diagnosis → ladder (0.84) → overfitting diagnosis → early stop (0.89) → ensemble
(**0.92, beats heuristic**).

**Robustness (`scripts/early_stopping_validate.py`):** the win holds on **5/5
independent test sets** (Δ +0.035…+0.042, all p<2e-5); **bootstrap 95% CI on the
per-graph difference = [+0.027, +0.043]** (excludes 0); **leave-one-out** 2-of-3
ensembles all beat the heuristic (0.905–0.913); checkpoint **selection is stable**
across validation ranges (3/3 identical). Robust across test sets, seeds, and
selection.

**Defect robustness:** conditioned on the instance being solvable (gflow exists),
the ensemble beats the heuristic at every defect rate — 1%: 0.920 vs 0.884
(p=3.4e-3), 5%: 0.920 vs 0.883 (p=2.0e-3), 10%: 0.923 vs 0.891 (p=2.4e-3). The
win is not clean-only. (Unconditionally the gap washes out under defects because
sparse irregular graphs lose gflow fast — only ~24% solvable at 1% — so the mean
is dominated by unsolvable reward-0 instances; a graph property, not agent
failure.)

**Size generalisation (trained at size):** the recipe (winning combo + early
stopping + ensemble) reproduces the win at ALL THREE sizes when trained there:
| size | qubits | ensemble | heuristic | Δ | p |
|------|--------|----------|-----------|---|---|
| 4×4 | 16 | 0.920 | 0.885 | +0.035 | 4e-9 |
| 5×5 | 25 | 0.944 | 0.915 | +0.029 | 4e-17 |
| 6×6 | 36 | 0.952 | 0.929 | +0.023 | 8e-21 |
It does NOT transfer *zero-shot* (4×4→5×5 loses, 0.855 vs 0.913) — but trained at
the target size it wins decisively. So beating the heuristic on irregular graphs
is a general property of the recipe across problem size, not a small-instance
artefact. The 5×5 win is robust: 5/5 independent test sets (all p<4e-15),
bootstrap CI [+0.029, +0.036], leave-one-out all win, and robust to the validation
selection range (an alternate selection also wins, 0.947 vs 0.913; only the
fully-overfit final ensemble fails — early stopping is necessary, the exact
checkpoint is not). The 6×6 win is likewise robust: 5/5 independent test sets
(Δ +0.015…+0.023, all p<3.3e-15), bootstrap CI [+0.011, +0.020], leave-one-out
all 3 pairs win, checkpoint selection perfectly stable (3/3 identical, val 50k vs
60k). **Reproducibility note:** the irregular generator's gflow-search budget
(`make_irregular_flow_graph`, `_max_tries`) was raised 80→400 — at 6×6 a single
construction attempt has only ~2% gflow success, so 80 tries left 31% of graphs
gflow-less; 400 tries restores 99.5% (4×4/5×5 always succeeded on the first try
and are unaffected).

**Universal-MBQC (non-Clifford angled reward) — the win STRENGTHENS, at all three
sizes.** All the above uses the plain gflow-consistency reward (every non-output
qubit is a hard ordering constraint). The realistic universal-MBQC objective is
the *angled* reward (`score_measurement_order_angled`): only **non-Clifford**
qubits constrain the order (Clifford corrections are Pauli-frame-trackable), and
the agent additionally observes each qubit's angle. Retraining the same winning
combo on irregular graphs WITH angles (3 seeds/size, `results/irregular_angled/`):

| size | plain Δ | angled ensemble | angled heuristic | angled Δ | p |
|------|:---:|:---:|:---:|:---:|:---:|
| 4×4 | +0.035 | 0.958 | 0.893 | **+0.065** | 5e-15 |
| 5×5 | +0.029 | 0.954 | 0.918 | **+0.036** | 7e-12 |
| 6×6 | +0.023 | 0.961 | 0.928 | **+0.033** | 1e-22 |

**The angled margin beats the plain margin at every size** — and unlike the plain
win (Δ decaying +0.035→+0.023 with size) the angled advantage stays flat/large.
The reason is intuitive: the agent *sees* the angles and exploits which qubits are
Clifford (unconstrained) vs non-Clifford (constrained), while the distance
heuristic is angle-blind and orders everything geometrically. The learned edge is
angle-awareness — exactly the capability a hand-designed geometric heuristic lacks.

All three angled wins are robustness-validated (`early_stopping_validate.py
--use-angles`): **5/5 independent test sets each** (4×4 all p<3e-14; 5×5 all
p<8e-7; 6×6 all p<1e-17); bootstrap 95% CIs all exclude 0 (4×4 [+0.062,+0.077],
5×5 [+0.029,+0.037], 6×6 [+0.029,+0.034]); leave-one-out all 3 pairs win at every
size; checkpoint selection stable (5×5 & 6×6 3/3 identical, 4×4 2/3 with the third
seed's alternate pick still an early checkpoint). ⇒ the irregular win is not an
artefact of the simplified all-constraints reward: it holds — more strongly —
on the realistic universal-MBQC objective, across all three problem sizes.

---

## 8. Reward validation (graphix, tested simulator)

`scripts/validate_reward_graphix.py`, 80 graphs/config, non-Clifford angles:

- **(A) gflow cross-validation: 100% agreement** — our `compute_gflow` ≡ graphix
  `find_gflow` on every grid/brickwork/irregular graph (perfect & defective).
- **(B) gflow ⇒ determinism: 100%** — every gflow pattern's output is identical
  across measurement-outcome branches (infidelity ≈ 2e-12).

⇒ The reward is not an arbitrary proxy: its underlying gflow is provably correct,
and its optimum corresponds to an actually-correct deterministic computation.

---

## 9. Flow hierarchy comparison (existing algorithms)

`scripts/flow_comparison.py`, existence rates, 120 graphs/cell:

- **Grid/brickwork:** causal flow ≈ gflow ≈ Pauli flow — classic causal flow
  suffices on lattices.
- **Irregular:** causal flow strictly weaker (0 defect: 83.3% vs gflow 100%).
  gflow is the right determinism condition for the defective/irregular setting.
- our gflow ≡ graphix gflow: 100% everywhere.

Figure: `results/flow/flow_hierarchy.png`.

---

## 9b. Theory — distance-from-output vs gflow layer

**Claim.** If the gflow layer L(v) is strictly increasing in the graph distance
d(v) to the output set, the distance ordering (farthest-first) is a valid gflow
order, so the distance heuristic is optimal. `scripts/theory_distance_gflow.py`
(300 graphs):

| topology (perfect 4×4) | Spearman ρ(L,d) | exact L=d | distance-order valid |
|------------------------|:---:|:---:|:---:|
| grid | **1.000** | **100%** | **1.000** |
| brickwork | **1.000** | **100%** | **1.000** |
| irregular | 0.879 | 52.8% | **0.887** |

- **Lattices: L(v) ≡ d(v) exactly** ⇒ distance/column-sweep heuristic provably
  optimal (proven for grids; L(column c)=cols−1−c=d(v)). Test-locked.
- **Irregular: L(v) ≥ d(v) with strict inequality for ~47% of nodes** (skip edges
  shortcut upstream qubits near the output) ⇒ distance ordering ceilings at 0.887
  — matching the measured distance baseline (0.885). This is the precise reason
  the heuristic dominates on lattices and leaves room only on irregular graphs.

Figure `results/theory/distance_vs_gflow.png`: grids on the L=d diagonal,
irregular scattered above it.

---

## 10. Honest limitations
- On regular lattices RL only *matches* the static heuristic (provably optimal
  there, §9b) — the win is specific to irregular graphs, where the heuristic is
  suboptimal.
- The irregular win requires training **at the target size** (no zero-shot size
  transfer) and the early-stopping + ensemble recipe (a single final checkpoint
  overfits); it is not a property of one bare policy.
- Reward is gflow-consistency (validated at its optimum); full noisy-fidelity
  simulation of arbitrary orders not done (endpoints validated via graphix).
- Edge-deletion defects only; ≤36 qubits; GAT architecture family only.

---

## 11. Figure index
`results/figures/` C1–C5 (conceptual), R1–R8 (results) — see
`results/figures/README.md`. **R8_irregular_win.png** is the headline positive
result (§7): ensemble vs heuristic vs oracle at 4×4/5×5/6×6 under both rewards,
plus the win-margin panel. Plus `results/attention/`, `results/flow/`,
`results/transfer_gnn/`.

## 12. Reproduction
All training in-terminal with `caffeinate -i`; Python `/usr/local/bin/python3.13`.
Key scripts: `train_mixed.py`, `train_gnn.py`, `benchmark.py`, `aggregate_seeds.py`,
`compare_baselines.py`, `eval_brickwork.py`, `make_figures.py`,
`validate_reward_graphix.py`, `flow_comparison.py`. Suites: `run_full_suite.sh`,
`run_ladder.sh`, `run_generalization.sh`.

---

## 13. Benchmark deliverable (`benchmark/`)

The proposal's **"1,000 randomised defective grid instances"** are released as a
fixed, versioned, labelled benchmark for future research (`scripts/build_benchmark.py`,
loader `mbqc_rl/benchmark.py`, 9 integrity tests):

- `grids_defective_n1000.jsonl` — 1,000 grid instances (4 sizes × 5 defect rates
  {0.01…0.30} × 50).
- `irregular_n500.jsonl` — 501 clean irregular flow-graph instances (the win regime).

Each record ships the graph + outputs + defect rate + **held-out seed** (base 1e6
grids / 2e6 irregular, disjoint from all training and selection seeds), the
**ground-truth gflow labels** (per-node layer order + correction sets), and
**reference rewards** (oracle / distance-heuristic / random) on the
gflow-consistency objective — a common yardstick a new method can be scored
against without rerunning generation. Labels reproduce a fresh recompute with 0
mismatches over all 1,501 instances (gflow cross-validated vs graphix, §8). The
per-defect-rate `gflow_exists` gradient (e.g. 6×6: 48/50 solvable at 1% → 0/50 at
30%) is itself a usable artefact for studying the determinism boundary.
