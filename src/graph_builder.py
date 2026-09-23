"""
graph_builder.py

Converts a `topology.Circuit` into a PyTorch Geometric `HeteroData` graph.

=====================================================================
KEY DESIGN DECISION (this is where Research Question 2 gets answered
for the first, baseline architecture -- other representations will be
implemented and compared against this one later):
=====================================================================

Node type 'bus' = one node per (physical bus, phase) pair that actually
exists in the circuit -- NOT one node per bus with a 3-slot feature vector.

    e.g. bus 611 (which only has phase C) contributes exactly ONE 'bus'
    node ("611.3"), not three nodes with two of them zero-padded.

Why: padding missing phases with zeros is exactly the kind of "implicit"
handling the proposal argues against (Problem Statement). A zero-padded
phase slot looks, to a homogeneous GNN, like a phase that exists but
happens to carry no current -- the model has no structural signal that
distinguishes "phase B carries 0 A because there's no phase-B conductor
here" from "phase B carries 0 A because the network is perfectly balanced
right now". Making the phase-node's existence conditional on the physical
conductor's existence pushes that distinction into the graph topology
itself, where it can't be forgotten by training.

This also makes Kirchhoff's Current Law a genuinely *local* statement at
each node (sum of currents on incident edges + injection = 0), matching
the message-passing formulation described in the proposal's Methodology
section -- which is the property the custom KCL-respecting message-passing
layer (Phase 3 of the plan) will exploit.

Edge types (all directed bus1->bus2 for lines/transformers; a HeteroConv
using both a type and its reverse can be layered on top for bidirectional
message passing -- see `T.ToUndirected()` used in `build_hetero_data`):

  ('bus', 'series_same_phase', 'bus')
      Diagonal entries of a line's series impedance matrix: current
      flowing in phase p at bus1 relates to the voltage drop in the SAME
      phase p at bus2. edge_attr = [r_ohm_per_mile, x_ohm_per_mile,
      length_miles, r_total_ohm, x_total_ohm].

  ('bus', 'series_mutual_phase', 'bus')
      Off-diagonal entries of the same matrix: current in phase p at
      bus1 also induces a voltage drop in a DIFFERENT phase q at bus2,
      through mutual inductive/capacitive coupling. This edge type is
      the actual physical mechanism by which an imbalance on one phase
      propagates to the other phases -- it is the graph-level encoding
      of exactly the phase-asymmetry problem the proposal is about.
      Without this edge type, a heterogeneous GNN could only learn
      cross-phase effects indirectly through shared node embeddings,
      not through an explicit structural pathway.

  ('bus', 'transformer', 'bus')
      3-phase power transformers, connected same-phase-to-same-phase
      (a simplification: does not yet encode delta/wye phase-shift,
      flagged as a known limitation for the wye-delta case).

  ('bus', 'regulator', 'bus')
      Single-phase voltage regulators, one edge per regulated phase.
      edge_attr = [vreg_pu, band_pu].

  ('load', 'load_feeds', 'bus') and reverse ('bus', 'rev_load_feeds', 'load')
      A load node connects to every phase-bus-node it draws current
      from (1 edge for a wye single-phase load, 2 for a phase-to-phase
      delta load, 3 for a 3-phase delta load spanning all pairs).

  ('capacitor', 'cap_feeds', 'bus') and reverse
      Same pattern as loads, for shunt capacitor banks.

Node feature layout (static topology features only -- operating-point
quantities such as voltage/power at a given snapshot are attached
separately by `attach_snapshot`, so the same static graph can be reused
across many simulated operating points without rebuilding it):

  bus:        [phase_a, phase_b, phase_c (one-hot), base_kv, x, y, is_slack]
  load:       [is_delta, model_1, model_2, model_5, kw, kvar]
  capacitor:  [kvar, kv]
"""

from __future__ import annotations

import torch
from torch_geometric.data import HeteroData

from topology import Circuit


def _bus_node_id(bus: str, phase: int) -> str:
    return f"{bus.lower()}.{phase}"


def build_hetero_data(circuit: Circuit, slack_bus: str = "sourcebus") -> tuple[HeteroData, dict]:
    """Builds the static topology graph. Returns (HeteroData, index_maps) where
    index_maps lets later code (attach_snapshot, scenario generation) look up
    the row index of a given (bus, phase) / load name / capacitor name."""

    # ---- enumerate bus-phase nodes ----
    bus_phase_nodes: list[tuple[str, int]] = []
    bus_phase_index: dict[str, int] = {}
    for bus in circuit.buses.values():
        for phase in bus.phases:
            node_id = _bus_node_id(bus.name, phase)
            bus_phase_index[node_id] = len(bus_phase_nodes)
            bus_phase_nodes.append((bus.name, phase))

    n_bus = len(bus_phase_nodes)
    bus_feat = torch.zeros((n_bus, 7), dtype=torch.float32)
    for i, (bus_name, phase) in enumerate(bus_phase_nodes):
        bus = circuit.buses[bus_name]
        bus_feat[i, phase - 1] = 1.0            # one-hot phase (cols 0,1,2)
        bus_feat[i, 3] = bus.base_kv
        bus_feat[i, 4] = bus.x if bus.x is not None else 0.0
        bus_feat[i, 5] = bus.y if bus.y is not None else 0.0
        bus_feat[i, 6] = 1.0 if bus.name.lower() == slack_bus.lower() else 0.0

    # ---- series line edges (same-phase + mutual) ----
    same_src, same_dst, same_attr = [], [], []
    mut_src, mut_dst, mut_attr = [], [], []
    for line in circuit.lines.values():
        n = len(line.phases)
        has_matrix = bool(line.rmatrix) and bool(line.xmatrix)
        for a, pa in enumerate(line.phases):
            id1 = _bus_node_id(line.bus1, pa)
            if id1 not in bus_phase_index:
                continue
            for b, pb in enumerate(line.phases):
                id2 = _bus_node_id(line.bus2, pb)
                if id2 not in bus_phase_index:
                    continue
                r = line.rmatrix[a][b] if has_matrix else (1.0 if a == b else 0.0)
                x = line.xmatrix[a][b] if has_matrix else 0.0
                length = line.length
                if pa == pb:
                    same_src.append(bus_phase_index[id1])
                    same_dst.append(bus_phase_index[id2])
                    same_attr.append([r, x, length, r * length, x * length])
                else:
                    mut_src.append(bus_phase_index[id1])
                    mut_dst.append(bus_phase_index[id2])
                    mut_attr.append([r, x, length, r * length, x * length])

    # ---- transformer edges (same-phase, 3-phase units only) ----
    tr_src, tr_dst = [], []
    for tr in circuit.transformers.values():
        if len(tr.buses) < 2:
            continue
        bus1, bus2 = tr.buses[0], tr.buses[1]
        for phase in (1, 2, 3):
            id1, id2 = _bus_node_id(bus1, phase), _bus_node_id(bus2, phase)
            if id1 in bus_phase_index and id2 in bus_phase_index:
                tr_src.append(bus_phase_index[id1])
                tr_dst.append(bus_phase_index[id2])

    # ---- regulator edges (single-phase) ----
    reg_src, reg_dst, reg_attr = [], [], []
    for reg in circuit.regulators.values():
        id1 = _bus_node_id(reg.bus1, reg.phase)
        id2 = _bus_node_id(reg.bus2, reg.phase)
        if id1 in bus_phase_index and id2 in bus_phase_index:
            reg_src.append(bus_phase_index[id1])
            reg_dst.append(bus_phase_index[id2])
            reg_attr.append([reg.vreg, reg.band])

    # ---- load nodes + edges ----
    load_names = list(circuit.loads.keys())
    load_index = {name: i for i, name in enumerate(load_names)}
    load_feat = torch.zeros((len(load_names), 6), dtype=torch.float32)
    load_edge_src, load_edge_dst = [], []  # load -> bus
    for name, load in circuit.loads.items():
        i = load_index[name]
        load_feat[i, 0] = 1.0 if load.conn == "delta" else 0.0
        if load.model == 1:
            load_feat[i, 1] = 1.0
        elif load.model == 2:
            load_feat[i, 2] = 1.0
        elif load.model == 5:
            load_feat[i, 3] = 1.0
        load_feat[i, 4] = load.kw
        load_feat[i, 5] = load.kvar
        for phase in load.phases:
            bid = _bus_node_id(load.bus, phase)
            if bid in bus_phase_index:
                load_edge_src.append(i)
                load_edge_dst.append(bus_phase_index[bid])

    # ---- capacitor nodes + edges ----
    cap_names = list(circuit.capacitors.keys())
    cap_index = {name: i for i, name in enumerate(cap_names)}
    cap_feat = torch.zeros((len(cap_names), 2), dtype=torch.float32)
    cap_edge_src, cap_edge_dst = [], []
    for name, cap in circuit.capacitors.items():
        i = cap_index[name]
        cap_feat[i, 0] = cap.kvar
        cap_feat[i, 1] = cap.kv
        for phase in cap.phases:
            bid = _bus_node_id(cap.bus, phase)
            if bid in bus_phase_index:
                cap_edge_src.append(i)
                cap_edge_dst.append(bus_phase_index[bid])

    # ---- assemble HeteroData ----
    data = HeteroData()
    data["bus"].x = bus_feat
    data["load"].x = load_feat
    data["capacitor"].x = cap_feat

    def _ei(src, dst):
        if not src:
            return torch.zeros((2, 0), dtype=torch.long)
        return torch.tensor([src, dst], dtype=torch.long)

    data["bus", "series_same_phase", "bus"].edge_index = _ei(same_src, same_dst)
    data["bus", "series_same_phase", "bus"].edge_attr = (
        torch.tensor(same_attr, dtype=torch.float32) if same_attr else torch.zeros((0, 5))
    )

    data["bus", "series_mutual_phase", "bus"].edge_index = _ei(mut_src, mut_dst)
    data["bus", "series_mutual_phase", "bus"].edge_attr = (
        torch.tensor(mut_attr, dtype=torch.float32) if mut_attr else torch.zeros((0, 5))
    )

    data["bus", "transformer", "bus"].edge_index = _ei(tr_src, tr_dst)
    data["bus", "regulator", "bus"].edge_index = _ei(reg_src, reg_dst)
    data["bus", "regulator", "bus"].edge_attr = (
        torch.tensor(reg_attr, dtype=torch.float32) if reg_attr else torch.zeros((0, 2))
    )

    data["load", "load_feeds", "bus"].edge_index = _ei(load_edge_src, load_edge_dst)
    data["capacitor", "cap_feeds", "bus"].edge_index = _ei(cap_edge_src, cap_edge_dst)

    index_maps = {
        "bus_phase_index": bus_phase_index,   # "671.1" -> row index in data['bus'].x
        "bus_phase_nodes": bus_phase_nodes,   # row index -> (bus_name, phase)
        "load_index": load_index,
        "cap_index": cap_index,
    }
    return data, index_maps


def add_reverse_edges(data: HeteroData) -> HeteroData:
    """Adds reverse copies of every directed edge type so a HeteroConv can
    aggregate messages in both directions (current flows both ways along a
    line depending on operating conditions; a load also affects its bus and
    is affected by it during iterative solves)."""
    import torch_geometric.transforms as T
    return T.ToUndirected()(data)


def describe(data: HeteroData, index_maps: dict) -> str:
    lines = [f"Node types: {data.node_types}", f"Edge types:"]
    for et in data.edge_types:
        lines.append(f"  {et}: {data[et].edge_index.shape[1]} edges")
    lines.append(f"Node counts: " + ", ".join(f"{nt}={data[nt].num_nodes}" for nt in data.node_types))
    return "\n".join(lines)


if __name__ == "__main__":
    from topology import load_circuit

    circuit = load_circuit()
    data, maps = build_hetero_data(circuit)
    print(describe(data, maps))
    print()
    print("bus feature tensor shape:", data["bus"].x.shape)
    print("example bus-phase node index:", maps["bus_phase_index"].get("611.3"))
    print("example bus-phase node index:", maps["bus_phase_index"].get("671.1"))

    data_undirected = add_reverse_edges(data)
    print()
    print("After ToUndirected():")
    print(describe(data_undirected, maps))
