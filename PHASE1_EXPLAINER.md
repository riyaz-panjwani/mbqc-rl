# Phase 1 Explainer — Environment Construction

**What this document covers:** Every file built in Phase 1, why each design decision was made, how each piece works mathematically and in code, and how they all connect into a working simulation environment.

---

## What Phase 1 Had to Deliver

From the project plan, Phase 1 (Weeks 1–3) has three concrete deliverables:

1. **Gymnasium wrapper** — a standard RL environment interface the PPO agent will talk to
2. **PyZX integration** — PyZX as the quantum simulation and verification backend
3. **Dynamic adjacency matrix updating logic** — the graph must shrink as qubits are measured

By end of Phase 1, the environment must pass unit tests confirming graph degradation and reward generation behave correctly. All three are now complete.

---

## Background You Need to Follow the Code

### What MBQC actually is (very short version)

In normal quantum computing you apply gate after gate. MBQC works differently: you prepare a big entangled resource state (a *graph state*) all at once, then drive the computation purely by measuring individual qubits. The trick is that the outcome of each measurement is random (0 or 1), so every subsequent measurement angle must be adjusted based on all previous outcomes. This adjustment is called **feed-forward logic**.

A graph state |G⟩ is built from a graph G = (V, E):
- Start every qubit in |+⟩ = (|0⟩ + |1⟩)/√2
- Apply a controlled-Z (CZ) gate to every pair of qubits connected by an edge in G

The resulting state encodes all possible computation paths at once.

### What gflow is and why it matters

For a measurement sequence to be deterministic (despite random outcomes) it must satisfy the **gflow conditions** (Browne et al., 2007). A gflow is a pair (g, ≺) where:
- **g(v)** is a *correction set* for each measured qubit v — the set of qubits whose measurement angles must be adjusted to cancel the random byproduct error caused by measuring v
- **≺** is a partial order on the qubits — v ≺ w means v is measured before w

The conditions that make this work:
1. v ∉ g(v) — a qubit cannot correct itself
2. For all w ∈ g(v): v ≺ w — you can only correct a qubit that hasn't been measured yet
3. **Odd(g(v)) ∩ (V \ O) = {v}** — among non-output qubits, exactly v has an odd number of neighbours in its correction set

Condition 3 is the key constraint. It means that applying the corrections in g(v) will cancel v's byproduct error without introducing new errors anywhere else. It is a linear constraint over GF(2) (the field with elements {0, 1}).

### The optimisation problem

On perfect hardware, a valid gflow always exists for a grid graph and can be computed efficiently. On real hardware, edge-formation fails with probability ~1%. A missing edge can break the gflow entirely. The project asks: can a PPO agent *learn* to route around missing edges, choosing a measurement order that preserves output fidelity — faster and more robustly than any classical algorithm that recomputes from scratch each time?

### The MDP formulation

To use RL, we cast measurement ordering as a Markov Decision Process:
- **State:** the current graph topology (which edges still exist) + the history of measurement outcomes so far
- **Action:** choose which qubit to measure next
- **Reward:** at the end of the episode, a score reflecting how well the measurement sequence respected the gflow constraints
- **Episode:** one complete computation on one defective graph instance

---

## File-by-File Breakdown

### `pyproject.toml` — Project configuration

**What:** Defines the package, its version, its dependencies, and how to install it.

**Why this form:** `pyproject.toml` is the modern PEP 517/518 standard, replacing `setup.py`. It keeps configuration in one place. The editable install (`pip install -e .`) means any change to the source is immediately visible without reinstalling.

**Key dependencies and why each was chosen:**
| Package | Why |
|---|---|
| `gymnasium` | Standard RL environment API. PPO implementations (e.g. stable-baselines3) all expect the Gymnasium interface. |
| `numpy` | Matrix operations for adjacency matrix, GF(2) arithmetic, random sampling. |
| `networkx` | Graph data structure — efficient edge/node operations, simple to convert to adjacency matrix. |
| `torch` | PyTorch for the PPO agent in Phase 2. Included now so the environment is tested against the full stack. |
| `pyzx` | ZX-calculus simulator. Used as the quantum backend: graph-state representation, gflow verification, and (in later phases) measurement-angle simulation. |
| `pytest` | Test runner. All 39 tests run with one command. |

**How to install:**
```bash
pip install -e ".[dev]"   # use /usr/local/bin/pip3.13 on this machine
```

---

### `mbqc_rl/utils/generators.py` — Grid graph builder

**What:** A single function `make_grid_graph(rows, cols, defect_rate, rng)` that creates the MBQC resource graph used as the environment's starting state each episode.

**Why a grid?** The dissertation evaluates on n×n grid graphs (n ∈ {3, 4, 5}) because grids are the standard benchmark topology in MBQC literature (Corli et al., 2025; Bartolucci et al., 2023) and have a well-understood gflow structure.

**Node numbering convention:**
```
Grid position (row r, column c) → node index  r * cols + c

3×3 example:
  0 — 1 — 2
  |   |   |
  3 — 4 — 5
  |   |   |
  6 — 7 — 8

Output qubits (last column, never measured): 2, 5, 8
```

This is row-major order. The output column is always on the right, matching the standard MBQC convention where computation flows left-to-right.

**Why Bernoulli edge deletion?** Each edge is deleted independently with probability `defect_rate`. This follows Bartolucci et al. (2023), who model photonic fusion failures as independent Bernoulli events. The realistic rate is ~1%; the training curriculum uses 20–30%.

**How the function works:**
```python
G = nx.Graph()
G.add_nodes_from(range(rows * cols))   # all nodes, even isolated ones

# Horizontal edges — only between adjacent columns
for r in range(rows):
    for c in range(cols - 1):
        if rng.random() >= defect_rate:   # keep edge with prob 1 - defect_rate
            G.add_edge(r*cols + c, r*cols + c + 1)

# Vertical edges — only between adjacent rows
for r in range(rows - 1):
    for c in range(cols):
        if rng.random() >= defect_rate:
            G.add_edge(r*cols + c, (r+1)*cols + c)
```

The `rng` parameter is a `numpy.random.Generator`, not the legacy `np.random` functions. This ensures reproducibility: calling `make_grid_graph` with the same `rng` seed always produces the same graph, which is essential for reproducible training runs.

---

### `mbqc_rl/utils/gflow.py` — gflow algorithm

**What:** Three public functions:
1. `compute_gflow(G, output_qubits)` — finds a valid gflow via the Browne et al. (2007) layered algorithm
2. `score_measurement_order(gflow_map, measurement_steps, output_set)` — turns a gflow into the episode reward
3. `_gf2_solve(A, b)` — solves Ax = b over GF(2) (used internally by the algorithm)

**Why implement gflow from scratch (rather than only using PyZX)?** Two reasons. First, our gflow needs to operate on a NetworkX graph, not a ZX-diagram. Second, having an independent implementation lets us use PyZX as a cross-check rather than just trusting one library blindly. If both agree, we can be confident.

#### `_gf2_solve(A, b)` — GF(2) Gaussian elimination

The gflow condition 3 (Odd(g(v)) ∩ (V\O) = {v}) is a linear equation over GF(2). For a candidate qubit v with correction set from C (the already-processed qubits), the equation is:

```
A x = b   over GF(2)
```

where:
- Rows of A = all non-output qubits
- Columns of A = qubits in C (candidates for g(v))
- A[i,j] = 1 if non-output qubit i is adjacent to candidate j
- b[i] = 1 if i == v, else 0 (only v should be in the odd neighbourhood)

`_gf2_solve` uses Gaussian elimination with full row reduction (both above and below each pivot), augmenting A with b. After elimination, any remaining row of the form `[0 … 0 | 1]` means the system is inconsistent (no solution). Otherwise, free variables are set to 0 and pivot variables are read from the last column.

#### `compute_gflow` — layered algorithm

The algorithm builds gflow layer by layer:

```
Layer 0: output qubits O  (these are "already done", correction sets point here)

Layer 1: find all v ∉ O such that
           solving A x = e_v gives a solution
         Assign g(v) = {C[j] : x[j] = 1}, layer(v) = 1
         Add these v's to C

Layer 2: same, but C now includes layer-1 qubits too
         Higher layer → measured earlier in actual computation

...until all non-output qubits have been assigned.
If any round produces no new assignments: no gflow exists → return (None, None).
```

**Why this layer convention?** Layer 0 = output, higher layer = measured earlier. This matches the natural reading: the correction set g(v) contains qubits with lower layer numbers (measured later), so they can still be corrected when v is measured.

#### `score_measurement_order` — reward function

Once an episode ends, we know the actual step number at which each qubit was measured (`step_of[v]`). The gflow says: for each qubit v, every non-output member w of g(v) must be measured *after* v, i.e., `step_of[v] < step_of[w]`.

The score is the fraction of such constraints that were satisfied:
```
score = (# satisfied constraints) / (# total constraints)
```

This gives a continuous reward signal in [0, 1]. A score of 1.0 means the agent produced a perfectly gflow-consistent measurement order. A score of 0 means every constraint was violated.

**Why this reward and not fidelity directly?** Computing the true output-state fidelity would require a full quantum state simulation — exponentially expensive for large graphs. The gflow-consistency score is a polynomial-time proxy that is guaranteed to equal 1.0 when the order is valid. In Phase 3 (training), if this proxy proves insufficient, shaped intermediate rewards (penalising gflow-violating actions step-by-step) can be added.

---

### `mbqc_rl/utils/pyzx_bridge.py` — PyZX integration

**What:** Three functions that connect our NetworkX-based environment to PyZX's ZX-calculus engine:
1. `nx_to_zx_graph(G_nx, output_qubits)` — converts a NetworkX graph into a PyZX ZX-diagram
2. `pyzx_gflow(G_nx, output_qubits)` — runs PyZX's built-in gflow algorithm and translates results to our format
3. `compare_gflow_implementations(G_nx, output_qubits, our_gflow, our_order)` — checks both implementations agree

**Why PyZX?** PyZX represents quantum states in the ZX-calculus. For a graph state |G⟩, the mapping is exact:
- Each qubit → Z-spider with phase 0
- Each CZ entangling gate → Hadamard edge between two Z-spiders
- Output qubits → Z-spiders connected to BOUNDARY vertices

This mapping means PyZX can natively reason about graph states: it knows which vertices are measured (non-boundary Z-spiders) and which are outputs (adjacent to boundary vertices). PyZX's `gflow()` function uses the same Browne et al. algorithm we implemented, giving an independent check.

**In later phases:** PyZX's rewriting rules will be used to track measurement angle updates (feed-forward logic) and to verify that the agent's chosen measurement order remains causally consistent after each step.

**Layer convention difference:**
PyZX uses the *opposite* layer numbering from ours:
```
PyZX:  layer 0 = measured first,  higher = measured later / output
Ours:  layer 0 = output,          higher = measured earlier
```
The bridge translates: `our_layer(v) = max_pyzx_layer - pyzx_layer(v)`.

**How `nx_to_zx_graph` builds the diagram:**
```python
# 1. Z-spider for each qubit (phase 0 = resource state |+⟩)
for node in sorted(G_nx.nodes()):
    v = g.add_vertex(pyzx.VertexType.Z, qubit=node, row=0)

# 2. Hadamard edges for CZ gates
for u, v in G_nx.edges():
    g.add_edge(g.edge(u_vertex, v_vertex), pyzx.EdgeType.HADAMARD)

# 3. BOUNDARY vertices for outputs + connect them
for q in output_qubits:
    b = g.add_vertex(pyzx.VertexType.BOUNDARY, qubit=q, row=1)
    g.add_edge(g.edge(qubit_vertex, b), pyzx.EdgeType.SIMPLE)

g.set_inputs([])      # no specific input state injected
g.set_outputs(boundary_list)
```

---

### `mbqc_rl/env/mbqc_env.py` — The Gymnasium environment

**What:** The complete MDP environment. It wraps the graph generator, the gflow algorithm, and the reward function into a standard Gymnasium interface that any PPO implementation can talk to.

**Why Gymnasium?** Gymnasium (successor to OpenAI Gym) is the de facto standard for RL environments. Every major PPO library — Stable-Baselines3, CleanRL, RLlib — expects this interface. Implementing it means we can swap PPO implementations without changing the environment.

#### Observation space

```python
observation_space = spaces.Dict({
    "observation": spaces.Box(low=-1.0, high=1.0, shape=(n*n + n,), dtype=float32),
    "action_mask": spaces.Box(low=0, high=1, shape=(n,), dtype=int8),
})
```

The `"observation"` vector has two parts:
1. **Flattened adjacency matrix** (n² values, 0.0 or 1.0): the current connectivity of the graph. Starts fully populated; entries go to 0.0 as qubits are measured and removed.
2. **Measurement history** (n values, −1/0/1): −1 = not yet measured; 0 or 1 = outcome of that qubit's measurement.

Together these give the agent full information about the current state of the computation.

The `"action_mask"` is a binary vector where 1 means "this qubit is valid to measure right now." Qubits are invalid if they are already measured or are output qubits.

**Why a Dict observation?** Action masking is critical here — the agent must never try to measure an already-measured qubit or an output qubit. Returning the mask alongside the observation lets PPO implementations apply masking directly in the policy network's output layer (standard practice for environments with variable valid action sets).

#### Action space

```python
action_space = spaces.Discrete(n)
```

The agent selects an integer from 0 to n−1, representing which qubit to measure next.

**What happens on an invalid action:** Reward = −1, episode continues. This trains the agent to respect the mask rather than needing a hard reset.

#### Step logic — how the graph degrades

```python
# 1. Simulate measurement outcome (Clifford phase: uniformly random 0 or 1)
outcome = int(self._rng.integers(0, 2))
self._history[action] = float(outcome)

# 2. Remove the measured qubit from the graph
self._graph.remove_node(action)

# 3. Recompute adjacency matrix from the shrunk graph
self._adj = np.zeros((n, n), dtype=np.float32)
for u, v in self._graph.edges():
    self._adj[u, v] = 1.0
    self._adj[v, u] = 1.0
```

This mirrors the physical reality: measuring a qubit in MBQC destroys it. The agent sees a strictly smaller graph at each step. By the end of the episode, only output qubits remain.

Note that removed qubits stay in `_history` (their outcome is recorded) but have all zeros in `_adj`. The observation therefore always has shape (n² + n,) regardless of how many qubits have been measured — the zeros encode the missing nodes.

#### Episode termination and reward

The episode ends when all non-output qubits have been measured. The reward is then:
```python
reward = score_measurement_order(self._gflow, self._step_of, self._output_set)
```

where `self._gflow` is computed once at `reset()` on the initial (pre-measurement) graph, and `self._step_of` records at which step each qubit was measured.

If the initial graph had no valid gflow (some defective instances won't), the reward is 0.0 — the agent cannot do better than random on an uncomputable instance.

#### `reset()` — starting a new episode

Every call to `reset()` generates a fresh random defective graph. This means the agent never memorises a single graph; it must learn a *policy* that generalises across thousands of different defective instances. This is the core training challenge.

---

## How All the Pieces Connect

```
reset() call
    │
    ├── make_grid_graph()           → NetworkX graph G, output qubit list
    │       generators.py
    │
    ├── compute_gflow(G, outputs)   → gflow_map, order  (stored, used at reward time)
    │       gflow.py
    │
    └── returns initial observation (flat adj matrix + history of -1s, action mask)

step(action) call
    │
    ├── validate action against mask
    ├── sample random outcome (0 or 1)
    ├── record step number in step_of[action]
    ├── remove action from graph → rebuild adj matrix
    │
    ├── if all non-output qubits measured:
    │       score_measurement_order(gflow_map, step_of, output_set)  → reward ∈ [0,1]
    │       gflow.py
    │
    └── returns new observation

(verification, when needed)
    pyzx_bridge.pyzx_gflow(G, outputs)         → cross-check gflow result
    pyzx_bridge.compare_gflow_implementations() → confirm both agree
```

---

## Running Everything

```bash
# Install (once)
cd Dissertation/mbqc-rl
/usr/local/bin/pip3.13 install -e ".[dev]"

# Run all 39 tests
/usr/local/bin/python3.13 -m pytest tests/ -v

# Quick smoke-test of the environment
/usr/local/bin/python3.13 - <<'EOF'
from mbqc_rl.env.mbqc_env import MBQCEnv
import numpy as np

env = MBQCEnv(rows=3, cols=3, defect_rate=0.05, render_mode="human")
obs, info = env.reset(seed=42)
print(f"gflow exists: {info['gflow_exists']}")
done = False
while not done:
    valid = np.where(obs["action_mask"] == 1)[0]
    obs, reward, terminated, truncated, _ = env.step(int(valid[0]))
    done = terminated or truncated
    env.render()
print(f"Final reward: {reward:.3f}")
EOF
```

---

## Phase 1 Completion Checklist

| Task | Status | Where |
|---|---|---|
| Gymnasium wrapper | ✅ Complete | `mbqc_rl/env/mbqc_env.py` |
| PyZX integration | ✅ Complete | `mbqc_rl/utils/pyzx_bridge.py` |
| Dynamic adjacency matrix updating | ✅ Complete | `MBQCEnv.step()` in `mbqc_env.py` |
| Unit tests: graph degradation | ✅ Passing | `tests/test_env.py` |
| Unit tests: reward generation | ✅ Passing | `tests/test_gflow.py` |
| Unit tests: PyZX cross-check | ✅ Passing | `tests/test_pyzx_bridge.py` |
| **Total tests** | **39 / 39 passing** | |

**Phase 1 milestone met:** The simulation environment passes unit tests confirming that graph degradation and reward generation behave correctly.

---

## What Phase 2 Is

Phase 2 (Weeks 4–5): **Agent Development**

The goal is to build the PPO Actor-Critic network in PyTorch that will actually learn a measurement policy.

Specifically:

### 2.1 — Actor-Critic architecture (`mbqc_rl/agent/ppo.py`)

The network takes the observation vector (n² + n values) and outputs:
- **Actor head:** logits for each of the n qubits → softmax → policy distribution. The action mask is applied here (masking invalid actions to −∞ before softmax).
- **Critic head:** a single scalar — the estimated *value* of the current state (used to reduce variance in the gradient estimates).

Both heads share a set of early layers (the "trunk"), which learns a general representation of the graph state. This sharing means the critic's value estimates inform the actor's gradient updates.

### 2.2 — Training loop (`scripts/train.py`)

The PPO update rule:
1. Collect a batch of episodes by running the current policy in the environment.
2. Compute *advantages* (how much better/worse each action was than the critic expected) using Generalised Advantage Estimation (GAE).
3. Update the network using PPO's clipped surrogate objective — the clip prevents the policy from changing too drastically in a single update, which stabilises training under sparse rewards.
4. Repeat for thousands of episodes.

### 2.3 — Key design decisions to make in Phase 2

- **Observation encoding:** The flat (n² + n) vector is the baseline. A graph-aware encoder (e.g. a small GNN) could be added if the flat encoder fails to learn.
- **Action masking:** Applied in the actor head by setting logits for invalid actions to −1e9 before softmax. This forces the policy to always select a valid qubit.
- **Reward scaling:** The gflow-consistency score ∈ [0, 1] may need to be rescaled if gradients are too small.
- **First sanity check (end of Phase 2):** The agent should achieve above-random reward on a 4-qubit (2×2) defect-free grid. If it cannot learn even this trivial case, the state representation or reward signal needs adjustment before moving to larger grids.
