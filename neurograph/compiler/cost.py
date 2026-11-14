"""The cost model shared by the partitioner (with estimated activity) and the
runtime (with measured operation counts)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..ir import Edge, EdgeKind, Graph, Node, Plasticity, Target
from .hardware import HardwareConfig

INF = float("inf")


@dataclass
class Cost:
    energy_pj: float = INF
    latency_ns: float = INF
    feasible: bool = False
    reason: str = ""
    parts: dict = field(default_factory=dict)   # energy by category

    def to_dict(self) -> dict:
        if not self.feasible:
            return {"feasible": False, "reason": self.reason}
        return {"feasible": True, "energy_pj": round(self.energy_pj, 3), "latency_ns": round(self.latency_ns, 3)}


@dataclass
class Ops:
    """Operation counts for one node over one step."""
    macs: float = 0.0            # dense multiply-accumulates
    synops: float = 0.0          # event-driven synaptic operations
    neuron_updates: float = 0.0
    stdp_updates: float = 0.0
    events: float = 0.0          # spikes emitted


def estimated_ops(node: Node) -> Ops:
    """Ops the partitioner expects, from the node's declared activity."""
    f = node.features()
    # pair-based STDP updates a synapse on its pre spike and on its post spike
    stdp = 2 * f.activity * f.synapses if node.plasticity == Plasticity.LOCAL_STDP else 0.0
    return Ops(macs=f.macs, synops=f.activity * f.synapses, neuron_updates=f.neurons,
               stdp_updates=stdp, events=f.activity * f.neurons)


def units_needed(node: Node, target: Target, hw: HardwareConfig) -> int:
    """Cores or tiles a node occupies on a target (0 on the accelerator)."""
    f = node.features()
    if target == Target.NEUROMORPHIC:
        nm = hw.neuromorphic
        return max(1, math.ceil(f.neurons / nm.neurons_per_core), math.ceil(f.synapses / nm.synapses_per_core))
    if target == Target.PIM:
        t = hw.pim.tile_dim
        dense = f.cells >= f.in_dim * f.out_dim
        if dense:
            return max(1, math.ceil(f.in_dim / t) * math.ceil(f.out_dim / t))
        return max(1, math.ceil(f.cells / (t * t)))
    return 0


def node_cost(node: Node, target: Target, hw: HardwareConfig, ops: Ops | None = None) -> Cost:
    """Energy and latency of one step of `node` on `target`."""
    f = node.features()
    ops = ops or estimated_ops(node)
    if target == Target.TENSOR:
        ta = hw.tensor
        macs = ops.macs
        if node.plasticity == Plasticity.LOCAL_STDP:
            macs += 2 * f.synapses      # pair-based STDP in software: potentiation and depression terms
        e = macs * ta.e_mac_pj
        # The memory wall: weights that do not fit in cache are streamed from DRAM each step.
        wbytes = f.weights * ta.bytes_per_weight
        e_mem = wbytes * ta.e_dram_pj_per_byte if wbytes > ta.cache_bytes else 0.0
        return Cost(e + e_mem, ta.t_launch_us * 1e3 + macs / ta.macs_per_s * 1e9, True, parts={"tensor": e, "memory": e_mem})

    if target == Target.NEUROMORPHIC:
        nm = hw.neuromorphic
        if not f.event_capable:
            return Cost(reason="not an event-time op")
        if node.bits > nm.weight_bits:
            return Cost(reason=f"needs {node.bits} bits, cores hold {nm.weight_bits}")
        units = units_needed(node, target, hw)
        if units > nm.cores:
            return Cost(reason=f"needs {units} cores, chip has {nm.cores}")
        e_stdp = ops.stdp_updates * nm.e_stdp_pj
        e = ops.synops * nm.e_synop_pj + ops.neuron_updates * nm.e_neuron_pj
        t = (ops.synops * nm.t_synop_ns + ops.neuron_updates * nm.t_neuron_ns + ops.stdp_updates * nm.t_stdp_ns) / units
        return Cost(e + e_stdp, t, True, parts={"neuromorphic": e, "plasticity": e_stdp})

    pim = hw.pim
    if not f.linear:
        return Cost(reason="only linear maps run on a crossbar")
    if node.bits > pim.bits:
        return Cost(reason=f"needs {node.bits} bits, crossbar gives {pim.bits}")
    if node.plasticity == Plasticity.LOCAL_STDP:
        return Cost(reason="local plasticity exceeds the crossbar write budget")
    units = units_needed(node, target, hw)
    if units > pim.tiles:
        return Cost(reason=f"needs {units} tiles, device has {pim.tiles}")
    dense = f.cells >= f.in_dim * f.out_dim
    in_tiles = math.ceil(f.in_dim / pim.tile_dim) if dense else 1
    e_adc = f.out_dim * in_tiles * pim.phases * pim.e_adc_pj
    e_dac = f.in_dim * pim.phases * pim.e_dac_pj
    e_cell = f.macs * pim.phases * pim.e_cell_pj
    return Cost(e_adc + e_dac + e_cell, f.reads * pim.t_mvm_ns, True,
                parts={"pim_adc": e_adc, "pim_dac": e_dac, "pim_array": e_cell})


def edge_traffic(g: Graph, e: Edge, events: float | None = None) -> tuple[float, float]:
    """(bytes, values) an edge carries per step."""
    src = g.nodes[e.src]
    if e.kind == EdgeKind.SPIKES:
        f = src.features()
        n = events if events is not None else f.activity * f.neurons
        return n * 4.0, n                                # one 32-bit address-event packet per spike
    w = src.width()
    return w * max(1, src.bits // 8), float(w)


def cut_cost(g: Graph, e: Edge, a: Target, b: Target, hw: HardwareConfig, hops: float | None = None,
             events: float | None = None) -> Cost:
    """Cost of an edge whose endpoints sit on targets a and b."""
    if a == b:
        return Cost(0.0, 0.0, True)
    noc = hw.noc
    nbytes, values = edge_traffic(g, e, events)
    flits = math.ceil(nbytes / noc.flit_bytes) if nbytes else 0
    h = noc.est_hops if hops is None else hops
    e_noc = flits * h * noc.e_hop_pj
    t = h * noc.t_hop_ns
    e_bridge = 0.0
    if Target.TENSOR in (a, b):
        e_bridge = nbytes * noc.e_bridge_pj_per_byte
        t += noc.t_bridge_us * 1e3
    # Crossing between the event world and the tensor world pays for transcoding.
    transcode = (a == Target.NEUROMORPHIC) != (b == Target.NEUROMORPHIC)
    e_codec = values * noc.e_codec_pj if transcode else 0.0
    return Cost(e_noc + e_bridge + e_codec, t, True, parts={"noc": e_noc + e_bridge, "codec": e_codec, "bytes": nbytes})
