"""Hybrid execution engine.

Each step is synchronous at the graph level (nodes run in topological order,
as between two barrier synchronizations) and discrete-event inside the mesh
(bursts on cut edges are routed and queue on shared links). Local plasticity
runs on the edge cores every step; weights are reconciled with the accelerator
side only at epoch barriers, and feedback edges are epoch-registered.

The engine is a simulator: functional behaviour is computed, while energy and
latency come from the cost model applied to the measured operation counts.
"""
from __future__ import annotations

import math

import numpy as np

from ..compiler import CompiledProgram, hops_between
from ..compiler.cost import cut_cost, node_cost
from ..ir import Plasticity, Target
from .kernels import make_kernel
from .noc import MeshNoC

MODES = ("heterogeneous", "gpu_baseline")
SERIES = ("energy_uj", "latency_us", "noc_bytes", "events", "sync_drift", "phase_error",
          "koopman_deviation", "symmetry_error", "bump_amplitude")
DISCLAIMER = ("Simulated. Functional behaviour is computed; energy and latency are cost-model estimates "
              "from the hardware constants, which mix published, derived and assumed values. Not a measurement.")


def simulate(program: CompiledProgram, steps: int = 400, seed: int = 0, mode: str = "heterogeneous",
             sync_every: int = 64, record_every: int = 4) -> dict:
    """Run `steps` 1 ms steps of the compiled program and return telemetry."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if steps < 1 or steps > 20000:
        raise ValueError("steps must be between 1 and 20000")
    if sync_every < 1:
        raise ValueError("sync_every must be >= 1")
    hetero = mode == "heterogeneous"
    g = program.graph if hetero else program.baseline
    hw = program.hardware
    order = g.topo_order()
    kernels = {nid: make_kernel(g.nodes[nid], hw, seed) for nid in order}
    noc = MeshNoC(hw.noc.mesh_w, hw.noc.mesh_h, hw.noc.t_hop_ns, hw.noc.flit_bytes)
    slots = {nid: tuple(program.mapping["nodes"][nid]["slots"][0]) for nid in order} if hetero else {}
    bridge = tuple(program.mapping["noc"]["bridge"])

    def where(nid):   # mesh position of a node; accelerator nodes sit behind the bridge
        return bridge if g.nodes[nid].target == Target.TENSOR else slots[nid]

    value: dict[str, np.ndarray] = {}
    registered: dict[str, np.ndarray] = {}        # epoch-registered copies for feedback edges
    energy = {k: 0.0 for k in ("neuromorphic", "pim_adc", "pim_dac", "pim_array", "tensor", "memory", "plasticity", "noc", "codec", "sync")}
    by_target = {t.value: 0.0 for t in Target}
    per_node = {nid: {"target": g.nodes[nid].target.value, "energy_pj": 0.0, "events": 0.0, "macs": 0.0, "synops": 0.0} for nid in order}
    series = {k: [] for k in SERIES}
    series["t"] = []
    lat_all, noc_bytes_all, peak_link = [], 0.0, 0
    weight_bytes = events_total = 0.0
    sync_count = modulator_bytes = 0
    has_global = any(n.plasticity == Plasticity.GLOBAL_SYNC for n in g.nodes.values())
    local_nodes = [nid for nid in order if g.nodes[nid].plasticity == Plasticity.LOCAL_STDP]
    stab = {k: [] for k in ("phase_error", "koopman_deviation", "symmetry_error", "sync_drift", "bump_amplitude")}

    for t in range(steps):
        step_e = 0.0
        lat = {tg: [] for tg in Target}
        step_metrics: dict[str, list[float]] = {}
        step_events = step_noc_bytes = 0.0
        ops_by_node = {}
        for nid in order:
            node = g.nodes[nid]
            ins = []
            for e in g.preds(nid):
                src = registered.get(e.src) if e.feedback else value.get(e.src)
                if src is not None:
                    ins.append(src)
            out, ops, m = kernels[nid].step(t, ins)
            value[nid] = out
            ops_by_node[nid] = ops
            c = node_cost(node, node.target, hw, ops)
            for k, v in c.parts.items():
                energy[k] += v
            by_target[node.target.value] += c.energy_pj
            pn = per_node[nid]
            pn["energy_pj"] += c.energy_pj; pn["events"] += ops.events; pn["macs"] += ops.macs; pn["synops"] += ops.synops
            step_e += c.energy_pj
            lat[node.target].append(c.latency_ns)
            step_events += ops.events
            # The memory wall: weights that must travel between memory and compute this step.
            if node.target == Target.TENSOR:
                f = kernels[nid].f
                moved = f.weights * hw.tensor.bytes_per_weight
                if node.plasticity != Plasticity.NONE:
                    moved *= 2                      # read, update, write back
                weight_bytes += moved
            for k, v in m.items():
                step_metrics.setdefault(k, []).append(v)

        # Cut edges become bursts on the mesh.
        if hetero:
            for e in g.edges:
                a, b = g.nodes[e.src].target, g.nodes[e.dst].target
                if a == b:
                    continue
                hops = abs(where(e.src)[0] - where(e.dst)[0]) + abs(where(e.src)[1] - where(e.dst)[1])
                c = cut_cost(g, e, a, b, hw, hops=hops, events=ops_by_node[e.src].events if e.kind.value == "spikes" else None)
                energy["noc"] += c.parts["noc"]; energy["codec"] += c.parts["codec"]
                step_e += c.energy_pj
                step_noc_bytes += c.parts["bytes"]
                noc.send(where(e.src), where(e.dst), c.parts["bytes"])
            # The third factor: one modulator value per plastic neuron, broadcast from the accelerator.
            if has_global:
                for nid in local_nodes:
                    if g.nodes[nid].target != Target.TENSOR:
                        nb = float(kernels[nid].f.neurons)
                        modulator_bytes += nb; step_noc_bytes += nb
                        e_mod = nb * hw.noc.e_bridge_pj_per_byte + math.ceil(nb / hw.noc.flit_bytes) * hw.noc.e_hop_pj * (slots[nid][0] + slots[nid][1])
                        energy["noc"] += e_mod; step_e += e_mod
                        noc.send(bridge, slots[nid], nb)

        # Epoch barrier: registered edges swap, local deltas are consolidated.
        if (t + 1) % sync_every == 0:
            registered = {k: v.copy() for k, v in value.items()}
            for nid in local_nodes:
                shipped = kernels[nid].consolidate()
                if hetero and g.nodes[nid].target != Target.TENSOR and shipped:
                    nb = float(shipped)             # one byte per changed synapse
                    e_sync = nb * hw.noc.e_bridge_pj_per_byte + shipped * hw.tensor.e_mac_pj
                    energy["sync"] += e_sync; step_e += e_sync
                    step_noc_bytes += nb
                    noc.send(slots[nid], bridge, nb)
            sync_count += 1

        d = noc.deliver()
        peak_link = max(peak_link, d["peak_link_flits"])
        noc_bytes_all += step_noc_bytes
        events_total += step_events
        if hetero:   # cores and tiles run in parallel; accelerator kernels launch one after another
            crossing = hw.noc.t_bridge_us * 1e3 if any(Target.TENSOR in (g.nodes[e.src].target, g.nodes[e.dst].target) for e in g.cut_edges()) else 0.0
            step_lat = max(max(lat[Target.NEUROMORPHIC], default=0.0), max(lat[Target.PIM], default=0.0),
                           sum(lat[Target.TENSOR])) + d["arrival_ns"] + crossing
        else:
            step_lat = sum(lat[Target.TENSOR])
        lat_all.append(step_lat)
        for k in stab:
            if k in step_metrics:
                stab[k].append(float(np.mean(step_metrics[k])))
        if t % record_every == 0:
            series["t"].append(t)
            series["energy_uj"].append(round(step_e * 1e-6, 6))
            series["latency_us"].append(round(step_lat * 1e-3, 4))
            series["noc_bytes"].append(round(step_noc_bytes, 2))
            series["events"].append(step_events)
            for k in ("sync_drift", "phase_error", "koopman_deviation", "symmetry_error", "bump_amplitude"):
                vals = step_metrics.get(k)
                last = series[k][-1] if series[k] else None
                series[k].append(round(float(np.mean(vals)), 6) if vals else last)

    total_pj = sum(energy.values())
    lat_arr = np.array(lat_all)
    info = {nid: k.info for nid, k in kernels.items() if k.info}

    def stat(k, fn):
        return round(float(fn(stab[k])), 6) if stab[k] else None

    return {
        "mode": mode, "module": g.name, "steps": steps, "seed": seed, "dt_ms": 1.0, "sync_every": sync_every,
        "totals": {
            "energy_mj": total_pj * 1e-9,
            "energy_uj_per_step": total_pj * 1e-6 / steps,
            "latency_ms_mean": float(lat_arr.mean()) * 1e-6,
            "latency_ms_p95": float(np.percentile(lat_arr, 95)) * 1e-6,
            "interconnect_bytes_per_step": noc_bytes_all / steps,
            "noc_peak_link_flits": int(peak_link),
            "noc_flit_hops": int(noc.total_flit_hops),
            "weight_bytes_moved_per_step": weight_bytes / steps,
            "events_per_step": events_total / steps,
            "modulator_bytes_per_step": modulator_bytes / steps,
            "sync_epochs": sync_count,
        },
        "energy_breakdown_uj": {k: round(v * 1e-6, 6) for k, v in energy.items()},
        "by_target_uj": {k: round(v * 1e-6, 6) for k, v in by_target.items()},
        "stability": {
            "attractor_phase_error_rad_rms": stat("phase_error", lambda v: math.sqrt(np.mean(np.square(v)))),
            "attractor_bump_amplitude_min": stat("bump_amplitude", np.min),
            "koopman_deviation_mean": stat("koopman_deviation", np.mean),
            "symmetry_error_mean": stat("symmetry_error", np.mean),
            "sync_drift_max": stat("sync_drift", np.max),
        },
        "series": series,
        "per_node": {nid: {**v, "energy_uj": round(v.pop("energy_pj") * 1e-6, 6)} for nid, v in per_node.items()},
        "noc": {"mesh": [hw.noc.mesh_w, hw.noc.mesh_h], "hottest_links": noc.hottest_links()},
        "info": info,
        "disclaimer": DISCLAIMER,
    }


def benchmark(program: CompiledProgram, steps: int = 400, seed: int = 0, sync_every: int = 64, record_every: int = 4) -> dict:
    """Run the heterogeneous mapping and the homogeneous GPU baseline on the same stimulus."""
    het = simulate(program, steps, seed, "heterogeneous", sync_every, record_every)
    gpu = simulate(program, steps, seed, "gpu_baseline", sync_every, record_every)
    h, b = het["totals"], gpu["totals"]

    def ratio(x, y):
        return round(x / y, 4) if y else None

    saved = 1.0 - (h["weight_bytes_moved_per_step"] / b["weight_bytes_moved_per_step"]) if b["weight_bytes_moved_per_step"] else 0.0
    return {
        "module": program.graph.name, "steps": steps, "seed": seed,
        "heterogeneous": het, "gpu_baseline": gpu,
        "comparison": {
            "energy_ratio_gpu_over_hetero": ratio(b["energy_mj"], h["energy_mj"]),
            "latency_ratio_gpu_over_hetero": ratio(b["latency_ms_mean"], h["latency_ms_mean"]),
            "energy_mj": {"heterogeneous": h["energy_mj"], "gpu_baseline": b["energy_mj"]},
            "latency_ms": {"heterogeneous": h["latency_ms_mean"], "gpu_baseline": b["latency_ms_mean"]},
            "interconnect_bytes_per_step": {"heterogeneous": h["interconnect_bytes_per_step"], "gpu_baseline": b["weight_bytes_moved_per_step"]},
            "memory_wall_savings_pct": round(100.0 * saved, 2),
            "stability": {"heterogeneous": het["stability"], "gpu_baseline": gpu["stability"]},
        },
        "disclaimer": DISCLAIMER,
    }
