"""The graph: nodes, typed edges, validation and ordering."""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass

from .nodes import Node, node_from_dict
from .types import EdgeKind, Plasticity, Target

_ID = re.compile(r"^[A-Za-z_][\w.]*$")


class IRError(ValueError):
    """Raised when a graph or its textual form is not valid NeuroGraph IR."""


@dataclass
class Edge:
    src: str
    dst: str
    kind: EdgeKind
    feedback: bool = False   # epoch-registered: readers see the previous epoch's writes

    def to_dict(self) -> dict:
        return {"src": self.src, "dst": self.dst, "kind": self.kind.value, "feedback": self.feedback}


class Graph:
    def __init__(self, name: str = "module"):
        self.name = name
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []

    # ---- construction
    def add(self, node: Node) -> Node:
        if not _ID.match(node.id):
            raise IRError(f"invalid node id {node.id!r}")
        if node.id in self.nodes:
            raise IRError(f"duplicate node id {node.id!r}")
        self.nodes[node.id] = node
        return node

    def connect(self, src: str, dst: str, feedback: bool = False, kind: EdgeKind | str | None = None) -> Edge:
        for n in (src, dst):
            if n not in self.nodes:
                raise IRError(f"edge references unknown node {n!r}")
        k = EdgeKind(kind) if kind is not None else self.nodes[src].out_kind()
        e = Edge(src, dst, k, feedback)
        self.edges.append(e)
        return e

    # ---- queries
    def preds(self, nid: str, feedback: bool | None = None) -> list[Edge]:
        return [e for e in self.edges if e.dst == nid and (feedback is None or e.feedback == feedback)]

    def succs(self, nid: str, feedback: bool | None = None) -> list[Edge]:
        return [e for e in self.edges if e.src == nid and (feedback is None or e.feedback == feedback)]

    def topo_order(self) -> list[str]:
        """Kahn's algorithm over forward edges. Feedback edges are ignored."""
        indeg = {n: 0 for n in self.nodes}
        for e in self.edges:
            if not e.feedback:
                indeg[e.dst] += 1
        ready = [n for n in self.nodes if indeg[n] == 0]
        order = []
        while ready:
            n = ready.pop(0)
            order.append(n)
            for e in self.succs(n, feedback=False):
                indeg[e.dst] -= 1
                if indeg[e.dst] == 0:
                    ready.append(e.dst)
        if len(order) != len(self.nodes):
            cyc = sorted(n for n in self.nodes if n not in order)
            raise IRError(f"cycle through {cyc} has no feedback (epoch-registered) edge")
        return order

    def depths(self) -> dict[str, int]:
        d: dict[str, int] = {}
        for n in self.topo_order():
            d[n] = 1 + max((d[e.src] for e in self.preds(n, feedback=False)), default=-1)
        return d

    # ---- validation
    def validate(self) -> "Graph":
        errs = []
        if not self.nodes:
            errs.append("module has no nodes")
        for n in self.nodes.values():
            errs += [f"%{n.id}: {m}" for m in n.check()]
            if not (1 <= n.bits <= 64):
                errs.append(f"%{n.id}: bits must be in 1..64")
            if n.plasticity == Plasticity.LOCAL_STDP and not n.features().event_capable:
                errs.append(f"%{n.id}: local_stdp needs an event-time op (ral.evt.* or ral.dyn.attractor)")
        for e in self.edges:
            if e.src not in self.nodes or e.dst not in self.nodes:
                errs.append(f"edge {e.src}->{e.dst} references an unknown node")
                continue
            s, d = self.nodes[e.src], self.nodes[e.dst]
            if e.kind != s.out_kind():
                errs.append(f"edge %{e.src}->%{e.dst} carries {e.kind.value} but %{e.src} produces {s.out_kind().value}")
            if e.kind not in d.accepts():
                errs.append(f"%{e.dst} ({d.op()}) does not accept {e.kind.value}; insert an explicit ral.evt codec")
        if errs:
            raise IRError("; ".join(errs))
        self.topo_order()
        return self

    # ---- mapping checks (after partitioning)
    def verify_mapping(self) -> None:
        errs = []
        for n in self.nodes.values():
            if n.target is None:
                errs.append(f"%{n.id} has no target")
            elif n.target == Target.PIM and n.plasticity == Plasticity.LOCAL_STDP:
                errs.append(f"%{n.id}: a locally plastic weight cannot live on a crossbar")
            elif n.target == Target.PIM and not n.features().linear:
                errs.append(f"%{n.id}: only linear maps run on a crossbar")
            elif n.target == Target.NEUROMORPHIC and not n.features().event_capable:
                errs.append(f"%{n.id}: not an event-time op")
        if errs:
            raise IRError("; ".join(errs))

    def cut_edges(self) -> list[Edge]:
        return [e for e in self.edges if self.nodes[e.src].target != self.nodes[e.dst].target]

    # ---- (de)serialization
    def to_dict(self) -> dict:
        return {"name": self.name, "nodes": [n.to_dict() for n in self.nodes.values()], "edges": [e.to_dict() for e in self.edges]}

    @classmethod
    def from_dict(cls, d: dict) -> "Graph":
        if not isinstance(d, dict) or "nodes" not in d:
            raise IRError("graph JSON needs a 'nodes' list")
        g = cls(str(d.get("name", "module")))
        try:
            for nd in d["nodes"]:
                g.add(node_from_dict(nd))
            for ed in d.get("edges", []):
                g.connect(ed["src"], ed["dst"], bool(ed.get("feedback", False)), ed.get("kind"))
        except (KeyError, TypeError, ValueError) as e:
            raise IRError(str(e)) from e
        return g.validate()

    def copy(self) -> "Graph":
        return copy.deepcopy(self)

    def __len__(self) -> int:
        return len(self.nodes)
