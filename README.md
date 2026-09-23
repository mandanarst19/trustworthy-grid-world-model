# Trustworthy World Models for Power Grid Simulation

A physics-informed heterogeneous GNN world model for three-phase distribution
grids, built so that constraint enforcement (Kirchhoff's laws, voltage and
SoC bounds, the AC power flow equations) is **structural** — encoded in the
architecture and training procedure — rather than an emergent, unverified
property of the training data.

## Motivation

An ordinary world model predicts the next state from the current one. A
world model for a power grid has to do more: it has to simulate the
consequence of an intervention (a line outage, a battery dispatched at 2x
its planned rate) under physical constraints that cannot be violated. A
model that hallucinates a physically impossible grid state is worse than no
model at all — it can suggest control actions that damage equipment or
destabilize the grid. This project treats that trustworthiness as a
prerequisite for deployment, not a nice-to-have.

The research plan is organized around four questions:

1. **Structural** — what is the minimal set of physical constraints that
   must be encoded in the *architecture* (as opposed to the loss function
   or the training data) to guarantee physical feasibility under
   distribution shift?
2. **Representational** — how should the heterogeneous graph structure of a
   three-phase distribution grid encode the physical coupling between
   phases, nodes, and components?
3. **Robustness** — under what perturbations (line outages, load spikes,
   phase imbalances, measurement noise) does constraint satisfaction
   degrade, and is the degradation predictable from the model's internal
   representations *before* the perturbation occurs?
4. **Evaluative** — what benchmark protocol is needed to measure
   trustworthiness specifically in the safety-critical scenarios where
   implicit constraint learning fails?

The full proposal (motivation, methodology, connection to physical
reservoir computing) lives in [`docs/PROPOSAL.md`](docs/PROPOSAL.md).

## Project status

**Phase 0 (in progress): data pipeline and graph representation.**

- [x] Load and solve the official IEEE 13-bus test feeder via OpenDSS
- [x] Extract a clean, ML-framework-agnostic topology (`src/topology.py`)
- [x] Build the heterogeneous per-phase graph (`src/graph_builder.py`)
- [ ] Scenario generator: load scaling, N-1 contingencies, phase-imbalance
      perturbations (`src/simulate.py`)
- [ ] Baseline `HeteroConv` GNN (plain MSE loss) — the reference point
      every later architectural/loss change is measured against
- [ ] Physics-informed loss terms (AC power flow equality constraints,
      voltage/SoC bounds)
- [ ] KCL-respecting message-passing layer (architectural prior)
- [ ] Robustness evaluation protocol, extending SafePowerGraph

## Graph representation

The core design decision (documented in full in `src/graph_builder.py`):
**one `bus` node per (physical bus, phase) pair that actually exists**, not
one node per bus with a zero-padded 3-phase feature vector. A single-phase
lateral like bus `611` (phase C only) contributes exactly one node. This
keeps Kirchhoff's Current Law a genuinely local statement at each node, and
gives the model an explicit structural signal for "this phase doesn't
exist here" instead of asking it to infer that from data.

| Edge type | Meaning |
|---|---|
| `(bus, series_same_phase, bus)` | Diagonal series-impedance term: current in phase *p* relates to voltage drop in the same phase *p* at the other end. |
| `(bus, series_mutual_phase, bus)` | Off-diagonal term: the actual physical mechanism by which imbalance on one phase propagates to the others. |
| `(bus, transformer, bus)` | 3-phase power transformers (same-phase-to-same-phase; delta/wye phase shift not yet modeled — known limitation). |
| `(bus, regulator, bus)` | Single-phase voltage regulators. |
| `(load, load_feeds, bus)` / `(capacitor, cap_feeds, bus)` | Load and shunt-capacitor injections, one edge per phase terminal the device is connected to. |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

```bash
cd src
python topology.py       # solves the circuit, prints a topology summary
python graph_builder.py  # builds the HeteroData graph, prints node/edge counts
```

A self-contained Kaggle-ready version of the same Phase-0 pipeline (circuit
data and code embedded, no external repo access needed) is in
[`notebooks/grid_world_model_phase0.ipynb`](notebooks/grid_world_model_phase0.ipynb).

## Data source

The IEEE 13-bus test feeder circuit definition
(`data/ieee13/IEEE13Nodeckt.dss`) is the official EPRI/IEEE test case,
obtained from the [`dss-extensions/electricdss-tst`](https://github.com/dss-extensions/electricdss-tst)
repository (`Version8/Distrib/IEEETestCases/13Bus/`). It is used here
unmodified.

## Repository structure

```
.
├── data/
│   └── ieee13/              # official IEEE13 OpenDSS circuit files
├── src/
│   ├── topology.py          # OpenDSS -> clean Python circuit representation
│   └── graph_builder.py     # circuit -> PyTorch Geometric HeteroData
├── notebooks/
│   └── grid_world_model_phase0.ipynb
├── tests/
├── docs/
│   └── PROPOSAL.md
├── requirements.txt
└── README.md
```
