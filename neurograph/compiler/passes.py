"""Structure-exploiting rewrites: the equivariant and manifold fusion pass."""
from __future__ import annotations

from dataclasses import dataclass, field

from ..ir import Graph, KoopmanOperator, ManifoldGroupAction, Node, Plasticity


@dataclass
class FusionReport:
    rewrites: list[dict] = field(default_factory=list)
    nodes_before: int = 0
    nodes_after: int = 0

    def to_dict(self) -> dict:
        return {"rewrites": self.rewrites, "nodes_before": self.nodes_before, "nodes_after": self.nodes_after}


def _chain(g: Graph, a: str, b: str) -> bool:
    """True if a feeds only b, b is fed only by a, and the edge is a forward edge."""
    out, inn = g.succs(a), g.preds(b)
    return len(out) == 1 and len(inn) == 1 and out[0].dst == b and not out[0].feedback


def _splice_out(g: Graph, keep: str, drop: str) -> None:
    """Remove `drop` (the successor of `keep`) and reattach its consumers to `keep`."""
    for e in list(g.edges):
        if e.src == drop:
            e.src = keep
    g.edges = [e for e in g.edges if e.dst != drop]
    del g.nodes[drop]


def _remove_pair(g: Graph, a: str, b: str) -> None:
    """Remove both a and b (a -> b), wiring a's producers to b's consumers."""
    producers = g.preds(a)
    consumers = g.succs(b)
    g.edges = [e for e in g.edges if e.src not in (a, b) and e.dst not in (a, b)]
    for p in producers:
        for c in consumers:
            g.connect(p.src, c.dst, feedback=p.feedback or c.feedback)
    del g.nodes[a], g.nodes[b]


def _geo(n: Node, *kinds: str) -> bool:
    return isinstance(n, ManifoldGroupAction) and n.kind in kinds and n.plasticity == Plasticity.NONE


def fuse(graph: Graph) -> tuple[Graph, FusionReport]:
    """Apply fusion rewrites to a fixed point. Returns a new graph and a report.

    Rewrites (all exact, so none spends error budget):
      project . project           -> project           (same manifold: idempotent)
      exp_map . log_map, and back -> removed           (inverse pair)
      group_act . group_act       -> group_act         (compose the group elements)
      tensor_product . equiv_linear -> fused_conv      (no materialized intermediate)
      koopman . koopman           -> koopman           (K^k folded into one operator)
      equiv_linear                -> Schur-shared      (one W_l per irrep type)
    """
    g = graph.copy()
    rep = FusionReport(nodes_before=len(g))
    changed = True
    while changed:
        changed = False
        for e in list(g.edges):
            if e.src not in g.nodes or e.dst not in g.nodes or not _chain(g, e.src, e.dst):
                continue
            a, b = g.nodes[e.src], g.nodes[e.dst]
            if _geo(a, "project") and _geo(b, "project") and (a.manifold, a.dim) == (b.manifold, b.dim):
                a.composed += b.composed
                _splice_out(g, a.id, b.id)
                rep.rewrites.append({"rule": "project_idempotent", "kept": a.id, "removed": [b.id]})
            elif ((_geo(a, "exp_map") and _geo(b, "log_map")) or (_geo(a, "log_map") and _geo(b, "exp_map"))) \
                    and (a.manifold, a.dim) == (b.manifold, b.dim) and g.preds(a.id) and g.succs(b.id):
                _remove_pair(g, a.id, b.id)
                rep.rewrites.append({"rule": "exp_log_cancel", "kept": None, "removed": [a.id, b.id]})
            elif _geo(a, "group_act") and _geo(b, "group_act") and a.group == b.group \
                    and (a.irreps_out or a.irreps_in) == (b.irreps_in or b.irreps_out):
                a.composed += b.composed
                _splice_out(g, a.id, b.id)
                rep.rewrites.append({"rule": "group_act_compose", "kept": a.id, "removed": [b.id]})
            elif _geo(a, "tensor_product") and _geo(b, "equiv_linear") and a.group == b.group:
                a.kind, a.irreps_out, a.composed = "fused_conv", b.irreps_out or b.irreps_in, a.composed + b.composed
                a.name = a.name or "fused_conv"
                _splice_out(g, a.id, b.id)
                rep.rewrites.append({"rule": "tensor_product_linear_fuse", "kept": a.id, "removed": [b.id]})
            elif isinstance(a, KoopmanOperator) and isinstance(b, KoopmanOperator) and a.observables == b.observables \
                    and a.plasticity == b.plasticity == Plasticity.NONE:
                a.steps += b.steps
                a.rho = max(a.rho, b.rho)
                _splice_out(g, a.id, b.id)
                rep.rewrites.append({"rule": "koopman_power_fold", "kept": a.id, "removed": [b.id]})
            else:
                continue
            changed = True
            break
    for n in g.nodes.values():
        if _geo(n, "equiv_linear") and not n.schur:
            n.schur = True
            rep.rewrites.append({"rule": "schur_share", "kept": n.id, "removed": []})
    rep.nodes_after = len(g)
    g.validate()
    return g, rep
