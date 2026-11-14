"""Sparsity- and plasticity-aware target partitioning.

Minimizes  sum(node energy) + sum(cut-edge energy) + w_latency * latency
over assignments of nodes to targets, subject to feasibility and capacity.
Seed, then single-node refinement, then capacity legalization.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..ir import Graph, Plasticity, Target
from .cost import INF, Cost, cut_cost, node_cost, units_needed
from .hardware import HardwareConfig

TARGETS = (Target.NEUROMORPHIC, Target.PIM, Target.TENSOR)


@dataclass
class PartitionReport:
    objective: float = 0.0
    baseline_objective: float = 0.0
    moves: int = 0
    evictions: list[str] = field(default_factory=list)
    fell_back: bool = False
    contract_pins: list[str] = field(default_factory=list)
    candidates: dict = field(default_factory=dict)   # node -> target -> cost

    def to_dict(self) -> dict:
        return {"objective_pj": round(self.objective, 3), "baseline_objective_pj": round(self.baseline_objective, 3),
                "moves": self.moves, "evictions": self.evictions, "contract_pins": self.contract_pins, "fell_back_to_baseline": self.fell_back,
                "candidates": self.candidates}


class Partitioner:
    def __init__(self, hw: HardwareConfig, w_latency: float = 0.0):
        self.hw = hw
        self.w_latency = w_latency   # pJ charged per ns of latency; 0 optimizes energy alone

    def _scalar(self, c: Cost) -> float:
        return c.energy_pj + self.w_latency * c.latency_ns if c.feasible else INF

    def _unary(self, g: Graph, nid: str, t: Target) -> float:
        return self._scalar(node_cost(g.nodes[nid], t, self.hw))

    def _local(self, g: Graph, nid: str, t: Target, assign: dict[str, Target]) -> float:
        c = self._unary(g, nid, t)
        if c == INF:
            return c
        for e in g.edges:
            if e.src == nid or e.dst == nid:
                a = t if e.src == nid else assign[e.src]
                b = t if e.dst == nid else assign[e.dst]
                c += self._scalar(cut_cost(g, e, a, b, self.hw))
        return c

    def objective(self, g: Graph, assign: dict[str, Target]) -> float:
        total = sum(self._unary(g, n, assign[n]) for n in g.nodes)
        return total + sum(self._scalar(cut_cost(g, e, assign[e.src], assign[e.dst], self.hw)) for e in g.edges)

    def _refine(self, g: Graph, assign: dict[str, Target], pinned: set[str]) -> int:
        """Each accepted move strictly lowers a bounded objective, so this terminates."""
        moves, moved = 0, True
        while moved:
            moved = False
            for nid in g.nodes:
                if nid in pinned:
                    continue
                best_t, best = assign[nid], self._local(g, nid, assign[nid], assign)
                for t in TARGETS:
                    c = self._local(g, nid, t, assign)
                    if c < best - 1e-9:
                        best_t, best = t, c
                if best_t != assign[nid]:
                    assign[nid] = best_t
                    moves += 1
                    moved = True
        return moves

    def _usage(self, g: Graph, assign: dict[str, Target]) -> dict[Target, int]:
        use = {t: 0 for t in TARGETS}
        for nid, t in assign.items():
            use[t] += units_needed(g.nodes[nid], t, self.hw)
        return use

    def run(self, graph: Graph, respect_pins: bool = True) -> tuple[Graph, PartitionReport]:
        g = graph.copy()
        rep = PartitionReport()
        pinned = {n.id for n in g.nodes.values() if n.target is not None} if respect_pins else set()
        for n in g.nodes.values():
            if n.target is None or not respect_pins:
                n.target = None
            # `local_stdp` is an ownership contract, not a cost choice: the weight is
            # written by the learning engine of an edge core, so the node lives there.
            if n.plasticity == Plasticity.LOCAL_STDP and n.id not in pinned:
                n.target = Target.NEUROMORPHIC
                pinned.add(n.id)
                rep.contract_pins.append(n.id)
        assign: dict[str, Target] = {}
        # 1. Seed: cheapest feasible target per node, cuts ignored.
        for nid, n in g.nodes.items():
            costs = {t: node_cost(n, t, self.hw) for t in TARGETS}
            rep.candidates[nid] = {t.value: c.to_dict() for t, c in costs.items()}
            if nid in pinned:
                if not costs[n.target].feasible:
                    raise ValueError(f"%{nid} is pinned to {n.target.value}: {costs[n.target].reason}")
                assign[nid] = n.target
            else:
                assign[nid] = min(TARGETS, key=lambda t: self._scalar(costs[t]))
        # 2. Refine with cut costs.
        rep.moves = self._refine(g, assign, pinned)
        # 3. Legalize shared capacity: evict the node that loses least by moving.
        cap = {Target.NEUROMORPHIC: self.hw.neuromorphic.cores, Target.PIM: self.hw.pim.tiles, Target.TENSOR: 1 << 60}
        frozen = set(pinned)
        for _ in range(2 * len(g) + 1):
            use = self._usage(g, assign)
            over = next((t for t in TARGETS if use[t] > cap[t]), None)
            if over is None:
                break
            victim, dest, least = None, None, INF
            for nid in g.nodes:
                if assign[nid] != over or nid in pinned:
                    continue
                stay = self._local(g, nid, over, assign)
                for t in TARGETS:
                    if t == over:
                        continue
                    regret = self._local(g, nid, t, assign) - stay
                    if regret < least:
                        victim, dest, least = nid, t, regret
            if victim is None:
                raise ValueError(f"{over.value} is oversubscribed and no node can move")
            assign[victim] = dest
            frozen.add(victim)
            rep.evictions.append(victim)
            rep.moves += self._refine(g, assign, frozen)
        # 4. Never do worse than the homogeneous baseline.
        baseline = {n: Target.TENSOR for n in g.nodes}
        rep.baseline_objective = self.objective(g, baseline)
        rep.objective = self.objective(g, assign)
        if rep.objective > rep.baseline_objective and not pinned:
            assign, rep.objective, rep.fell_back = baseline, rep.baseline_objective, True
        for nid, t in assign.items():
            g.nodes[nid].target = t
        g.verify_mapping()
        return g, rep


def assign_all(graph: Graph, target: Target = Target.TENSOR) -> Graph:
    """The homogeneous baseline: every node on one target."""
    g = graph.copy()
    for n in g.nodes.values():
        n.target = target
    return g
