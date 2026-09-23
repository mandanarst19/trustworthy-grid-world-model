"""
topology.py

Loads the official IEEE 13-bus test feeder (EPRI/IEEE, via the
dss-extensions electricdss-tst repository) using OpenDSSDirect, solves it,
and extracts a clean, structured representation of the circuit that is
independent of OpenDSS's internal bookkeeping.

This is the ground-truth topology that src/graph_builder.py will later turn
into a PyTorch Geometric HeteroData object. Keeping this extraction step
separate from the graph-construction step means we can change the GNN's
graph representation (Phase 1/3 of the proposal) without re-deriving the
circuit each time, and we can unit-test the topology independently of any
ML code.

Design note (ties back to the proposal):
    Each Line here keeps its *actual* phase list (e.g. line 632-645 is only
    phases [2, 3]). This is exactly the source of the three-phase asymmetry
    that breaks homogeneous-GNN symmetry assumptions (Research Question 2).
    We do NOT pad missing phases with zeros at this layer -- that decision
    is deferred to the graph builder, so we can experiment with different
    ways of representing "this phase does not exist here" (e.g. a missing
    edge vs. a zero-weighted edge vs. a separate per-phase node).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import opendssdirect as dss

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DSS_MASTER = os.path.join(
    THIS_DIR, "..", "data", "ieee13", "IEEE13Nodeckt.dss"
)


@dataclass
class Bus:
    name: str
    phases: list[int]  # subset of [1, 2, 3], present physical phases at this bus
    base_kv: float
    x: float | None = None
    y: float | None = None


@dataclass
class Line:
    name: str
    bus1: str
    bus2: str
    phases: list[int]  # which of the 3 phases this line actually carries
    length: float
    length_units: str
    linecode: str | None
    is_switch: bool = False
    # Per-unit-length series impedance matrices (ohms/mile-equivalent),
    # sized [len(phases), len(phases)], in the *local* phase order given
    # by `phases`. Kept as nested lists (not numpy) so this module has no
    # hard numpy dependency at the data-extraction layer.
    rmatrix: list[list[float]] = field(default_factory=list)
    xmatrix: list[list[float]] = field(default_factory=list)


@dataclass
class Transformer:
    name: str
    buses: list[str]        # [primary_bus, secondary_bus, ...]
    phases: int
    windings: int
    kvs: list[float]
    kvas: list[float]
    conns: list[str]


@dataclass
class RegulatorTransformer:
    name: str
    bank: str
    bus1: str
    bus2: str
    phase: int               # single-phase regulators, phase this unit controls
    vreg: float
    band: float


@dataclass
class Load:
    name: str
    bus: str
    phases: list[int]
    conn: str                # "wye" or "delta"
    model: int                # ZIP/const-P-Q/const-Z/... model code used by OpenDSS
    kv: float
    kw: float
    kvar: float


@dataclass
class Capacitor:
    name: str
    bus: str
    phases: list[int]
    kvar: float
    kv: float


@dataclass
class Circuit:
    buses: dict[str, Bus]
    lines: dict[str, Line]
    loads: dict[str, Load]
    capacitors: dict[str, Capacitor]
    transformers: dict[str, Transformer]
    regulators: dict[str, RegulatorTransformer]
    base_frequency: float = 60.0


def _load_bus_coords(csv_path: str) -> dict[str, tuple[float, float]]:
    coords = {}
    if not os.path.exists(csv_path):
        return coords
    with open(csv_path) as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split(",")]
            if len(parts) < 3:
                continue
            name, x, y = parts[0], parts[1], parts[2]
            try:
                coords[name.lower()] = (float(x), float(y))
            except ValueError:
                continue
    return coords


def load_circuit(dss_master_path: str = DEFAULT_DSS_MASTER) -> Circuit:
    """Redirects OpenDSS to the master .dss file, solves the base case, and
    extracts a Circuit object. Assumes the .dss file ends with `Solve`."""

    dss_master_path = os.path.abspath(dss_master_path)
    workdir = os.path.dirname(dss_master_path)
    cwd = os.getcwd()
    try:
        os.chdir(workdir)
        dss.Command(f'Redirect "{os.path.basename(dss_master_path)}"')
        dss.Solution.Solve()
        if not dss.Solution.Converged():
            raise RuntimeError("OpenDSS power flow did not converge on base case")

        coords = _load_bus_coords("IEEE13Node_BusXY.csv")

        # ---- Buses ----
        buses: dict[str, Bus] = {}
        for bname in dss.Circuit.AllBusNames():
            dss.Circuit.SetActiveBus(bname)
            nodes = dss.Bus.Nodes()  # e.g. [1,2,3] or [3] or [1,2]
            phases = sorted(n for n in nodes if n in (1, 2, 3))
            kv = dss.Bus.kVBase()
            xy = coords.get(bname.lower())
            buses[bname] = Bus(
                name=bname,
                phases=phases,
                base_kv=kv,
                x=xy[0] if xy else None,
                y=xy[1] if xy else None,
            )

        # ---- Lines ----
        lines: dict[str, Line] = {}
        for lname in dss.Lines.AllNames():
            dss.Lines.Name(lname)
            bus1_full = dss.Lines.Bus1()   # e.g. "671.1.2.3" or "632.3.2"
            bus2_full = dss.Lines.Bus2()
            bus1 = bus1_full.split(".")[0]
            bus2 = bus2_full.split(".")[0]
            node_tags = bus1_full.split(".")[1:]
            phases = sorted(int(n) for n in node_tags if n in ("1", "2", "3"))
            if not phases:
                # phases not explicitly tagged on the bus name -> use the
                # element's own Phases property, default to 1..N
                nph = dss.Lines.Phases()
                phases = list(range(1, nph + 1))

            is_switch = dss.Lines.IsSwitch()
            rmatrix_flat = dss.Lines.RMatrix()
            xmatrix_flat = dss.Lines.XMatrix()
            n = len(phases)
            rmatrix = [rmatrix_flat[i * n:(i + 1) * n] for i in range(n)] if rmatrix_flat else []
            xmatrix = [xmatrix_flat[i * n:(i + 1) * n] for i in range(n)] if xmatrix_flat else []

            lines[lname] = Line(
                name=lname,
                bus1=bus1,
                bus2=bus2,
                phases=phases,
                length=dss.Lines.Length(),
                length_units=str(dss.Lines.Units()),
                linecode=dss.Lines.LineCode() or None,
                is_switch=is_switch,
                rmatrix=rmatrix,
                xmatrix=xmatrix,
            )

        # ---- Loads ----
        loads: dict[str, Load] = {}
        for i, lname in enumerate(dss.Loads.AllNames()):
            dss.Loads.Name(lname)
            full_name = dss.CktElement.Name()  # "Load.671" etc
            bus_full = dss.CktElement.BusNames()[0]
            bus = bus_full.split(".")[0]
            node_tags = bus_full.split(".")[1:]
            phases = sorted(int(n) for n in node_tags if n in ("1", "2", "3"))
            if not phases:
                phases = list(range(1, dss.Loads.Phases() + 1))
            is_delta = dss.Loads.IsDelta()
            loads[lname] = Load(
                name=lname,
                bus=bus,
                phases=phases,
                conn="delta" if is_delta else "wye",
                model=dss.Loads.Model(),
                kv=dss.Loads.kV(),
                kw=dss.Loads.kW(),
                kvar=dss.Loads.kvar(),
            )

        # ---- Capacitors ----
        capacitors: dict[str, Capacitor] = {}
        for cname in dss.Capacitors.AllNames():
            dss.Capacitors.Name(cname)
            bus_full = dss.CktElement.BusNames()[0]
            bus = bus_full.split(".")[0]
            node_tags = bus_full.split(".")[1:]
            phases = sorted(int(n) for n in node_tags if n in ("1", "2", "3"))
            if not phases:
                phases = list(range(1, dss.CktElement.NumPhases() + 1))
            kvar_total = sum(dss.Capacitors.kvar()) if isinstance(dss.Capacitors.kvar(), (list, tuple)) else dss.Capacitors.kvar()
            capacitors[cname] = Capacitor(
                name=cname,
                bus=bus,
                phases=phases,
                kvar=kvar_total,
                kv=dss.Capacitors.kV(),
            )

        # ---- Transformers (3-phase power transformers only; regulators separate) ----
        transformers: dict[str, Transformer] = {}
        regulators: dict[str, RegulatorTransformer] = {}
        reg_names = {n.lower() for n in dss.RegControls.AllNames()}
        for tname in dss.Transformers.AllNames():
            dss.Transformers.Name(tname)
            n_windings = dss.Transformers.NumWindings()
            phases = dss.CktElement.NumPhases()
            buses_full = dss.CktElement.BusNames()
            buses_ = [b.split(".")[0] for b in buses_full]

            if phases == 1 and tname.lower() in {r for r in reg_names} | {
                b.lower() for b in []
            }:
                pass  # handled below via regcontrol loop instead

            transformers[tname] = Transformer(
                name=tname,
                buses=buses_,
                phases=phases,
                windings=n_windings,
                kvs=[],   # left for a future pass if per-winding kv is needed
                kvas=[],
                conns=[],
            )

        for rname in dss.RegControls.AllNames():
            dss.RegControls.Name(rname)
            xfmr_name = dss.RegControls.Transformer()
            dss.Transformers.Name(xfmr_name)
            buses_full = dss.CktElement.BusNames()
            bus1 = buses_full[0].split(".")[0]
            bus2 = buses_full[1].split(".")[0]
            node_tags = buses_full[0].split(".")[1:]
            phase = int(node_tags[0]) if node_tags else 1
            regulators[rname] = RegulatorTransformer(
                name=rname,
                bank=xfmr_name,
                bus1=bus1,
                bus2=bus2,
                phase=phase,
                vreg=dss.RegControls.ForwardVreg(),
                band=dss.RegControls.ForwardBand(),
            )
            # remove the per-phase regulator transformer from the plain
            # transformer dict -- it's represented via `regulators` instead
            transformers.pop(xfmr_name, None)

        return Circuit(
            buses=buses,
            lines=lines,
            loads=loads,
            capacitors=capacitors,
            transformers=transformers,
            regulators=regulators,
            base_frequency=dss.Settings.DefaultBaseFrequency() if hasattr(dss.Settings, "DefaultBaseFrequency") else 60.0,
        )
    finally:
        os.chdir(cwd)


def summarize(circuit: Circuit) -> str:
    lines = []
    lines.append(f"Buses: {len(circuit.buses)}")
    for b in circuit.buses.values():
        lines.append(f"  {b.name:10s} phases={b.phases} base_kv={b.base_kv:.3f} xy={b.x, b.y}")
    lines.append(f"Lines: {len(circuit.lines)}")
    for l in circuit.lines.values():
        tag = " [SWITCH]" if l.is_switch else ""
        lines.append(f"  {l.name:10s} {l.bus1:>8s} -> {l.bus2:<8s} phases={l.phases} len={l.length}{l.length_units}{tag}")
    lines.append(f"Loads: {len(circuit.loads)}")
    for ld in circuit.loads.values():
        lines.append(f"  {ld.name:10s} bus={ld.bus:8s} phases={ld.phases} conn={ld.conn:5s} kW={ld.kw:7.1f} kvar={ld.kvar:6.1f}")
    lines.append(f"Capacitors: {len(circuit.capacitors)}")
    for c in circuit.capacitors.values():
        lines.append(f"  {c.name:10s} bus={c.bus:8s} phases={c.phases} kvar={c.kvar}")
    lines.append(f"Transformers: {len(circuit.transformers)}")
    for t in circuit.transformers.values():
        lines.append(f"  {t.name:10s} buses={t.buses} phases={t.phases} windings={t.windings}")
    lines.append(f"Regulators: {len(circuit.regulators)}")
    for r in circuit.regulators.values():
        lines.append(f"  {r.name:10s} bank={r.bank} {r.bus1}->{r.bus2} phase={r.phase} vreg={r.vreg} band={r.band}")
    return "\n".join(lines)


if __name__ == "__main__":
    circ = load_circuit()
    print(summarize(circ))
