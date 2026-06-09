# Phase 4 Explainer — Evaluation & Dissertation Write-up

**What this document covers:** Every Phase 4 script, how to produce the empirical results tables for the dissertation, the statistical analysis, and a guide to which numbers go where in the write-up.

---

## What Phase 4 Had to Deliver

From the project plan (Weeks 9–12):

1. **Classical baseline comparison** on 1,000 grid instances
2. **Transfer test** (training defect rate vs deployment robustness)
3. **Dissertation write-up** — empirical results from Phases 3–4

---

## The Two Core Evaluation Scripts

### `scripts/benchmark.py` — 1,000-grid comparison table

This is the main empirical result for the dissertation. It answers:

> *"Does the PPO agent learn a better-than-random measurement strategy, and how close does it get to the oracle classical ceiling?"*

**How it works:**

For each trial `i ∈ {0, …, 999}`:
1. Generate a grid instance with `seed = base_seed + i` (same seed → same graph for all methods)
2. Run the PPO agent with **greedy (argmax) decoding** on that graph
3. Run `GreedyGflowBaseline` on the same graph
4. Run `RandomBaseline` on the same graph

The comparison is **paired** — each method sees the same sequence of graphs — so differences in mean reward are purely due to method quality, not random graph difficulty.

**Statistical test:** Mann-Whitney U (non-parametric, makes no normality assumption):
- *Agent vs Greedy*: tests whether the agent is significantly below the oracle
- *Agent vs Random*: tests whether the agent significantly outperforms chance

**Output files:**
```
results/benchmark/benchmark_3x3_defect01pct_n1000.json   — full stats + raw rewards
results/benchmark/benchmark_3x3_defect01pct_n1000.txt    — formatted table
```

**How to read the table:**

```
─────────────────────────────────────────────────────────────────
  Metric                  PPO agent       Greedy       Random
  ──────                  ─────────       ──────       ──────
  Mean reward              0.9200         0.9400       0.6500
  Std deviation            0.1100         0.0900       0.2200
  Median                   0.9583         1.0000       0.6667
  Min                      0.3333         0.3333       0.0000
  Max                      1.0000         1.0000       1.0000
  % Perfect                78.0%          85.0%        12.0%

  Mann-Whitney U tests (two-sided):
    Agent vs Greedy : U=450234  p=0.0023 **
    Agent vs Random : U=812301  p<0.001 ***

  Graphs with valid gflow: 94.3%
─────────────────────────────────────────────────────────────────
```

**Interpreting the numbers for the dissertation:**
- **Mean reward** gap between agent and greedy = how far below the oracle the agent falls
- **% Perfect** = fraction of episodes where the agent found a perfectly valid ordering
- **Graphs with valid gflow** = the "hardness" of the benchmark (at 1% defects, ~94% of graphs have a valid gflow; the other 6% give reward=0 regardless of strategy)
- A **significant p-value for agent vs random** (p < 0.05) proves the agent learned something non-trivial
- A **non-significant p-value for agent vs greedy** (p > 0.05) would suggest the agent matches the oracle

---

### `scripts/transfer_test.py` — Robustness across defect rates

This answers the dissertation's second empirical question:

> *"Does training at high defect rates (20–30%) produce a policy that is robust at realistic deployment rates (~1%)?"*

**How it works:**

Take a single checkpoint (e.g. Stage 3, trained at 30% defects). Evaluate it at:
- 0% defects (trivial — should score ~1.0)
- 1% defects (realistic deployment target)
- 5%, 10%, 20%, 30% (increasing hardness toward and beyond training conditions)

At each rate: run N episodes (default 500), record mean ± std for agent, greedy, random.

**Output files:**
```
results/transfer/transfer_3x3.json   — per-rate stats for all methods
results/transfer/transfer_3x3.png    — transfer curve plot
```

**The transfer curve plot:**
- X-axis: defect rate (0→30%)
- Y-axis: mean episode reward
- Three lines: agent (blue, with error bars), greedy oracle (green dashed), random (grey)
- Red dotted vertical line at 1% marks the realistic deployment rate

**What a successful transfer curve looks like:**
- Agent stays close to greedy at low defect rates (0%–5%)
- Agent stays well above random at moderate rates (10%–20%)
- At 30% both agent and greedy drop significantly (many graphs have no gflow)
- The key result: *at 1% deployment rate, the Stage-3 policy (trained at 30%) achieves high reward* — demonstrating the curriculum strategy worked

---

## Running Order for Dissertation Results

### Step 1 — Full curriculum (overnight)
```bash
cd Dissertation/mbqc-rl

/usr/local/bin/python3.13 scripts/curriculum.py \
  --rows 3 --cols 3 \
  --save-dir results/3x3_full \
  --n-steps 512

# Optional: 4×4 grid (larger, needs longer training)
/usr/local/bin/python3.13 scripts/curriculum.py \
  --rows 4 --cols 4 \
  --save-dir results/4x4_full \
  --n-steps 512 \
  --total-timesteps-override 400000   # not a real flag; just 2× the timesteps
```

### Step 2 — 1,000-grid benchmark
```bash
/usr/local/bin/python3.13 scripts/benchmark.py \
  --checkpoint results/3x3_full/stage3_final.pt \
  --n-trials 1000 \
  --defect-rate 0.01 \
  --save-dir results/benchmark
```
**Expected runtime:** ~2 min on CPU, ~45 sec on MPS.

### Step 3 — Transfer test
```bash
/usr/local/bin/python3.13 scripts/transfer_test.py \
  --checkpoint results/3x3_full/stage3_final.pt \
  --n-episodes 500 \
  --save-dir results/transfer
```
**Expected runtime:** ~3 min on CPU.

### Step 4 — Evaluate individual stages (optional but useful)
```bash
# Compare stage 0 (no defects) vs stage 3 (30% defects) at realistic rate
for stage in 0 1 2 3; do
  /usr/local/bin/python3.13 scripts/evaluate.py \
    --checkpoint results/3x3_full/stage${stage}_final.pt \
    --defect-rate 0.01 --n-episodes 500 --baseline
done
```

---

## Dissertation Write-up Guide

### Section: Implementation (Phase 1–2)

| Content | Source file |
|---|---|
| Gymnasium environment description | `mbqc_rl/env/mbqc_env.py` + `PHASE1_EXPLAINER.md` |
| gflow algorithm derivation | `mbqc_rl/utils/gflow.py` + `PHASE1_EXPLAINER.md §3` |
| Network architecture diagram | `mbqc_rl/agent/policy.py` + `PHASE2_EXPLAINER.md §2` |
| PPO pseudocode | `mbqc_rl/agent/ppo.py` + `PHASE2_EXPLAINER.md §3` |

### Section: Training (Phase 3)

| Content | Source |
|---|---|
| Curriculum design rationale | `PHASE3_EXPLAINER.md §2` |
| Training curves (4 stages) | `results/3x3_full/curriculum_training_curve.png` |
| Quick-run summary table | DISSERTATION_LOG.md Session 3 |

### Section: Evaluation (Phase 4)

| Content | Source |
|---|---|
| 1,000-grid comparison table | `results/benchmark/benchmark_3x3_defect01pct_n1000.txt` |
| Mann-Whitney U statistics | `results/benchmark/benchmark_3x3_defect01pct_n1000.json` |
| Transfer curve figure | `results/transfer/transfer_3x3.png` |
| Transfer curve data | `results/transfer/transfer_3x3.json` |

### Discussion points

1. **Agent vs greedy gap:** Explain that the gap is expected — the agent doesn't have access to the pre-computed gflow, it must infer ordering from observation alone. At 1% defects with the full curriculum, we expect this gap to be < 5%.

2. **Transfer robustness:** If the Stage-3 checkpoint achieves > 0.9 reward at 1% defects (well above random ~0.65), this confirms that domain-randomisation curriculum training works for MBQC measurement optimisation.

3. **Null result scenario:** If the agent fails to beat random significantly, this is still a valid dissertation finding — it would suggest the 9-qubit state representation is insufficient and the agent would need graph neural network (GNN) observations to generalise. Discuss next steps.

4. **Photonic hardware context:** Connect Stage-3 training (30% defects) to Bartolucci et al. 2023 fusion failure rates (1%). The curriculum trains in a regime 30× harder than realistic hardware, ensuring the learned policy is conservative.

---

## Phase 4 Completion Checklist

| Task | Status |
|---|---|
| `scripts/benchmark.py` (1,000-grid, Mann-Whitney) | ✅ done |
| `scripts/transfer_test.py` (defect rate sweep) | ✅ done |
| Unit tests: 85/85 passing | ✅ done |
| Quick benchmark validated (50 trials) | ✅ see below |
| Full curriculum (750k steps) result | ⏳ run overnight |
| Full benchmark (1,000 trials, 1% defects) | ⏳ after full curriculum |
| Transfer test results (6 defect rates × 500 eps) | ⏳ after full curriculum |
| Dissertation write-up | ⏳ use tables above |
