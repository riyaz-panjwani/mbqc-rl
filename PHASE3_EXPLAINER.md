# Phase 3 Explainer — Training & Optimisation

**What this document covers:** Every file added in Phase 3, the design rationale behind the defect-rate curriculum, how the classical baselines work and why they exist, and what the training results mean for the dissertation's empirical contribution.

---

## What Phase 3 Had to Deliver

From the project plan, Phase 3 (Weeks 6–8) has three tasks:

1. **Training on progressively larger grid topologies** — 3×3 → 4×4 → 5×5
2. **20–30% defect rate curriculum** — domain randomisation strategy
3. **Hyperparameter tuning for sparse reward signal**

The central empirical question being answered is: *does training at elevated defect rates make the policy robust when deployed at the realistic 1% defect rate?*

---

## The Defect-Rate Curriculum

### Why train at high defect rates?

On a perfect graph (0% defects), every measurement ordering is valid — the agent gets reward 1.0 no matter what it does. Training here teaches the agent nothing interesting. It just learns "measure anything, everything works."

At realistic hardware defect rates (~1%), defects appear occasionally and the agent must route around them. But if the agent has only ever seen perfect graphs, the first time it encounters a defective instance it has no learned heuristics to fall back on.

The solution, borrowed from robot simulation-to-real transfer (Tobin et al., 2017): **train at artificially elevated defect rates (20–30%), then deploy at the realistic ~1% rate.** A policy trained on severely broken graphs has already developed heuristics for navigating missing edges. Deployment graphs (near-perfect) are easy by comparison.

### The four-stage curriculum

| Stage | Defect rate | Timesteps (full) | Timesteps (quick) | Goal |
|-------|------------|------|------|------|
| 0 | 0%  | 200k | 20k | Learn the basic gflow ordering on perfect graphs |
| 1 | 10% | 150k | 15k | Introduce mild defects; policy must handle occasional missing edges |
| 2 | 20% | 200k | 20k | Moderate defects; routing heuristics start mattering |
| 3 | 30% | 200k | 20k | Severe defects; curriculum peak — far worse than deployment |

Each stage starts from the checkpoint of the previous stage, so knowledge is accumulated progressively rather than learned from scratch.

### What the quick-run results showed

Running the quick curriculum (75k total steps, ~2 minutes on MPS):

| Stage | Defect rate | Mean reward (first 10 updates) | Mean reward (last 50 updates) |
|-------|------------|-------------------------------|-------------------------------|
| 0 | 0.0 | 0.83 | **0.99** |
| 1 | 0.1 | 0.74 | 0.80 |
| 2 | 0.2 | 0.55 | 0.60 |
| 3 | 0.3 | ~0.45 | ~0.55 |

Key observations:
- **Stage 0:** reward converges to ~0.99 within 4,000 steps — the agent quickly learns the correct gflow ordering on perfect 3×3 graphs
- **Stages 1–3:** reward drops as defects increase (some graphs have no valid gflow, making reward 0 regardless of what the agent does)
- **Full training** (750k steps across all stages) is expected to produce a policy that approaches 0.8–0.9 on defective instances — this will be tested in Phase 4

---

## File-by-File Breakdown

### `mbqc_rl/baselines/classical.py` — Three classical baselines

Three baselines define the performance range the RL agent is measured against.

#### `RandomBaseline`

At each step, picks uniformly at random from currently valid (unmasked) qubits. This is the floor — the absolute minimum a sensible policy should beat. Any training signal at all should produce a policy that beats this.

#### `GreedyGflowBaseline`

This is the **oracle classical ceiling**. It:
1. Reads the gflow layer ordering computed by the environment at reset
2. Sorts all non-output qubits by *descending* layer (highest layer = measured first)
3. Executes this pre-planned sequence step by step

**On graphs with a valid gflow → reward = 1.0 every episode.** It has perfect knowledge of the gflow and executes it exactly.

**On defective graphs without a gflow** (some edges missing break all valid flows) → falls back to a random valid order.

**Why this is the right ceiling:** The RL agent doesn't know the gflow in advance — it must infer a good ordering from the observation alone. On perfect graphs, the agent should approach the greedy baseline's 1.0 reward. On defective graphs, the agent may actually *beat* the greedy baseline, because the greedy baseline's pre-planned order can become invalid mid-episode when the graph degrades differently than expected.

#### `TopologicalBaseline`

A randomised variant of `GreedyGflowBaseline`. At each step, instead of following a pre-planned sequence, it finds the set of valid qubits at the *current maximum* gflow layer and picks randomly among them. This produces different valid orderings each run and is useful for sampling the distribution of valid-order rewards.

### `mbqc_rl/utils/metrics.py` — Training metrics

Four functions:

**`save_history(history, path)`** / **`load_history(path)`** — Save/load the training history list as JSON. Handles numpy types with a custom encoder.

**`rolling_mean(values, window=20)`** — Smooth a reward sequence with a rolling average. Used in plots to see the trend clearly despite per-update noise.

**`summarise(history)`** — Prints a formatted table comparing the first 10 updates (before learning) to the last 50 (converged behaviour). Shows mean reward, policy loss, value loss, and entropy.

**`plot_training(history, save_to, title, show)`** — Generates a 3-panel matplotlib figure:
- Top panel: raw reward + rolling mean + 1.0 reference line
- Bottom left: policy loss (should become less negative as policy improves)
- Bottom right: value loss (should decrease as critic learns) + entropy (should decrease as policy becomes more deterministic)

Saved as PNG to the results directory automatically by `curriculum.py`.

### `scripts/curriculum.py` — Curriculum training runner

Runs the four-stage defect-rate curriculum end-to-end. Each stage:
1. Creates a new `MBQCEnv` with the stage's defect rate
2. Wraps the same policy in a new `PPOTrainer` (policy weights carry over)
3. Trains for the stage's timestep budget
4. Saves a checkpoint (`stage{N}_final.pt`) and history JSON

Outputs at the end:
- `full_curriculum_history.json` — all four stages combined
- `curriculum_training_curve.png` — full training curve across all stages

**Key flags:**
```bash
--quick              # 20k/15k/20k/20k steps per stage (~2 min on MPS)
--resume-stage N     # skip the first N stages (useful if a stage crashes)
--resume-ckpt PATH   # load a checkpoint before the resume stage
```

### How the environment was extended

Two properties were added to `MBQCEnv` to support the baselines:

```python
env.gflow        → the correction-set map (dict or None)
env.gflow_order  → layer ordering (dict or None)
```

These expose the gflow the environment already computed at `reset()`, so baselines and analysis scripts can access it without re-computing it.

---

## How to Run Phase 3

### Full curriculum (recommended — runs overnight)
```bash
cd Dissertation/mbqc-rl

# 3×3 grid — full curriculum (~18 min on MPS)
/usr/local/bin/python3.13 scripts/curriculum.py \
  --rows 3 --cols 3 \
  --save-dir results/3x3_full

# 4×4 grid — train on stage 2–3 only (larger grid needs more steps)
/usr/local/bin/python3.13 scripts/curriculum.py \
  --rows 4 --cols 4 \
  --save-dir results/4x4_full
```

### Quick validation (~2 min)
```bash
/usr/local/bin/python3.13 scripts/curriculum.py \
  --rows 3 --cols 3 --quick \
  --save-dir results/3x3_quick
```

### Evaluate after training
```bash
# Compare trained policy vs greedy and random baselines
/usr/local/bin/python3.13 scripts/evaluate.py \
  --checkpoint results/3x3_full/stage3_final.pt \
  --defect-rate 0.01 \
  --n-episodes 1000 \
  --baseline
```

### View the training curve
```python
from mbqc_rl.utils.metrics import load_history, plot_training
history = load_history("results/3x3_full/full_curriculum_history.json")
plot_training(history, title="3×3 Curriculum", save_to="training_curve.png")
```

---

## Phase 3 Completion Checklist

| Task | Status |
|---|---|
| Defect-rate curriculum (4 stages, 0%→30%) | ✅ `scripts/curriculum.py` |
| Classical baselines (random, greedy-gflow, topological) | ✅ `mbqc_rl/baselines/classical.py` |
| Training metrics & plotting | ✅ `mbqc_rl/utils/metrics.py` |
| `gflow` / `gflow_order` properties on env | ✅ `mbqc_rl/env/mbqc_env.py` |
| Unit tests (16 new, 74 total passing) | ✅ `tests/test_baselines.py` |
| Quick curriculum validated on MPS | ✅ Stage 0 reward → 0.99 |
| Training curve saved to results/ | ✅ `curriculum_training_curve.png` |

---

## What Phase 4 Is

Phase 4 (Weeks 9–12): **Evaluation & Dissertation Write-up**

Three concrete deliverables:

### 4.1 — 1,000-grid benchmark (`scripts/benchmark.py`)

Generate 1,000 random defective grid instances (n ∈ {3,4,5}, defect_rate=0.01). For each instance, run:
- The trained PPO agent (Stage 3 checkpoint, greedy decoding)
- `GreedyGflowBaseline`
- `RandomBaseline`

Record the reward for each. Produce a results table with mean, std, and statistical significance (Mann-Whitney U test comparing agent vs greedy).

### 4.2 — Transfer test

Take the 3×3 Stage 3 checkpoint (trained at 30% defects) and evaluate at:
- 0% defects
- 1% defects (realistic deployment)
- 10% defects
- 20% defects
- 30% defects

This directly answers the dissertation's second open question: *how does training defect rate relate to deployment robustness?*

### 4.3 — Dissertation write-up

The code and results files provide the material for:
- Section on implementation (Phase 1–2)
- Training results and curves (Phase 3)
- Evaluation tables and statistical analysis (Phase 4)
- Discussion of the four open questions from the lit survey
