"""Compiler pipeline: verify -> fuse -> partition -> place -> emit."""
from __future__ import annotations

from dataclasses import dataclass, field

from ..ir import Graph, Plasticity, Target, print_ir
from .cost import Cost, Ops, cut_cost, edge_traffic, estimated_ops, node_cost, units_needed  # noqa: F401
from .hardware import SOURCES, HardwareConfig  # noqa: F401
from .partition import TARGETS, Partitioner, PartitionReport, assign_all  # noqa: F401
from .passes import FusionReport, fuse  # noqa: F401


@dataclass
class CompiledProgram:
    source: Graph                 # the validated input
    graph: Graph                  # fused and mapped
    baseline: Graph               # same fused graph, every node on the accelerator
    hardware: HardwareConfig
    mapping: dict = field(default_factory=dict)
    report: dict = field(default_factory=dict)

    @property
    def ir_text(self) -> str:
        return print_ir(self.graph)

    def to_dict(self) -> dict:
        return {"name": self.graph.name, "ir": self.ir_text, "source_ir": print_ir(self.source),
                "graph": self.graph.to_dict(), "mapping": self.mapping, "report": self.report,
                "layout": layout(self.graph)}


def place(g: Graph, hw: HardwareConfig) -> dict:
    """Give every node its cores or tiles along a serpentine path over the mesh,
    in topological order so that connected nodes sit near each other.
    Slot (0, 0) is the bridge to the tensor accelerator."""
    w, h = hw.noc.mesh_w, hw.noc.mesh_h
    cursor = 1
    placement = {}
    for nid in g.topo_order():
        n = g.nodes[nid]
        units = units_needed(n, n.target, hw)
        slots = []
        for _ in range(units):
            idx = cursor % (w * h)
            y, x = divmod(idx, w)
            slots.append([x if y % 2 == 0 else w - 1 - x, y])
            cursor += 1
        placement[nid] = {"target": n.target.value, "plasticity": n.plasticity.value, "units": units,
                          "slots": slots or [[0, 0]]}
    return placement


def hops_between(mapping: dict, a: str, b: str) -> int:
    (ax, ay), (bx, by) = mapping["nodes"][a]["slots"][0], mapping["nodes"][b]["slots"][0]
    return abs(ax - bx) + abs(ay - by)


def layout(g: Graph) -> dict:
    """Layered coordinates for drawing the DAG: column = depth, row = order within depth."""
    depth = g.depths()
    rows: dict[int, int] = {}
    pos = {}
    for nid in g.topo_order():
        d = depth[nid]
        pos[nid] = {"col": d, "row": rows.get(d, 0)}
        rows[d] = rows.get(d, 0) + 1
    return {"nodes": pos, "cols": max(depth.values()) + 1 if depth else 0, "rows": max(rows.values()) if rows else 0}


def compile_graph(graph: Graph, hardware: HardwareConfig | None = None, fusion: bool = True,
                  w_latency: float = 0.0, respect_pins: bool = True) -> CompiledProgram:
    """Compile a graph: validate, fuse, partition, place. Raises IRError / ValueError on bad input."""
    hw = hardware or HardwareConfig()
    source = graph.copy().validate()
    fused, frep = fuse(source) if fusion else (source.copy(), FusionReport(nodes_before=len(source), nodes_after=len(source)))
    part = Partitioner(hw, w_latency)
    mapped, prep = part.run(fused, respect_pins=respect_pins)
    baseline = assign_all(fused)

    # What fusion bought: partition the unfused graph too and compare cut edges.
    unfused_mapped, _ = part.run(source, respect_pins=respect_pins)
    placement = place(mapped, hw)
    usage = {t.value: sum(p["units"] for p in placement.values() if p["target"] == t.value) for t in TARGETS}
    by_target = {t.value: [n for n, p in placement.items() if p["target"] == t.value] for t in TARGETS}
    mapping = {
        "module": mapped.name,
        "nodes": placement,
        "noc": {"mesh": [hw.noc.mesh_w, hw.noc.mesh_h], "bridge": [0, 0]},
        "usage": usage,
        "capacity": {Target.NEUROMORPHIC.value: hw.neuromorphic.cores, Target.PIM.value: hw.pim.tiles},
        "by_target": by_target,
        "plasticity": {p.value: [n.id for n in mapped.nodes.values() if n.plasticity == p] for p in Plasticity},
        "cut_edges": [e.to_dict() for e in mapped.cut_edges()],
    }
    report = {
        "fusion": frep.to_dict(),
        "partition": prep.to_dict(),
        "cut_edges": {"with_fusion": len(mapped.cut_edges()), "without_fusion": len(unfused_mapped.cut_edges())},
        "estimated": {
            "heterogeneous_pj_per_step": round(prep.objective, 3),
            "gpu_baseline_pj_per_step": round(prep.baseline_objective, 3),
            "ratio": round(prep.baseline_objective / prep.objective, 3) if prep.objective > 0 else None,
        },
        "note": "Energies are cost-model estimates from the hardware constants, not measurements.",
    }
    return CompiledProgram(source=source, graph=mapped, baseline=baseline, hardware=hw, mapping=mapping, report=report)
