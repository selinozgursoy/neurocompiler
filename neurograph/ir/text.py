"""MLIR-style textual form of NeuroGraph IR.

    // NeuroGraph IR v1
    module @ring {
      %vel  = ral.evt.encoder {population = 64, rate = 0.08} : () -> spikes<64, 1>
      %ring = ral.dyn.attractor {neurons = 256} (%vel, %stim feedback) : (spikes, spikes) -> spikes<256, 8> @NeuromorphicCore
    }

One op per line. Operands name their producers; `feedback` marks an
epoch-registered edge. The declared types are checked against the ops.
"""
from __future__ import annotations

import re

from .graph import Graph, IRError
from .nodes import OP_REGISTRY, node_from_dict
from .types import Target

HEADER = "// NeuroGraph IR v1"
_MODULE = re.compile(r"^module\s+@([\w.\-]+)\s*\{$")
_OP = re.compile(
    r"^%(?P<id>[\w.]+)\s*=\s*(?P<op>[\w.]+)\s*\{(?P<attrs>[^}]*)\}\s*"
    r"(?:\((?P<operands>[^)]*)\)\s*)?:\s*\((?P<ins>[^)]*)\)\s*->\s*(?P<out>.+?)"
    r"(?:\s+@(?P<target>\w+))?$")
_ATTR = re.compile(r'([A-Za-z_]\w*)\s*=\s*("(?:[^"\\]|\\.)*"|[^,\s]+)')


def _fmt(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(v, float):
        return repr(v)
    return str(v)


def _parse_value(tok: str):
    if tok.startswith('"'):
        return tok[1:-1].replace('\\"', '"').replace("\\\\", "\\")
    if tok in ("true", "false"):
        return tok == "true"
    try:
        return int(tok)
    except ValueError:
        pass
    try:
        return float(tok)
    except ValueError:
        raise IRError(f"cannot parse attribute value {tok!r}") from None


def print_ir(g: Graph) -> str:
    """Serialize a graph. Nodes are printed in topological order."""
    lines = [HEADER, f"module @{g.name} {{"]
    for nid in g.topo_order():
        n = g.nodes[nid]
        attrs = ", ".join(f"{k} = {_fmt(v)}" for k, v in n.attrs().items() if v not in ("", None))
        ins = g.preds(nid)
        operands = ", ".join(f"%{e.src}" + (" feedback" if e.feedback else "") for e in ins)
        sig = ", ".join(e.kind.value for e in ins)
        line = f"  %{nid} = {n.op()} {{{attrs}}} "
        if ins:
            line += f"({operands}) "
        line += f": ({sig}) -> {n.out_type()}"
        if n.target is not None:
            line += f" @{n.target.value}"
        lines.append(line)
    lines.append("}")
    return "\n".join(lines) + "\n"


def parse_ir(text: str) -> Graph:
    """Parse the textual form. Raises IRError with a line number on bad input."""
    g: Graph | None = None
    pending: list[tuple[int, str, list[tuple[str, bool]], list[str], str]] = []
    closed = False
    for ln, raw in enumerate(text.splitlines(), 1):
        line = raw.split("//", 1)[0].strip()
        if not line:
            continue
        if g is None:
            m = _MODULE.match(line)
            if not m:
                raise IRError(f"line {ln}: expected 'module @name {{'")
            g = Graph(m.group(1))
            continue
        if line == "}":
            closed = True
            continue
        if closed:
            raise IRError(f"line {ln}: text after the closing brace")
        m = _OP.match(line)
        if not m:
            raise IRError(f"line {ln}: cannot parse op: {line[:80]}")
        op = m.group("op")
        if op not in OP_REGISTRY:
            raise IRError(f"line {ln}: unknown op {op!r}")
        attrs = {k: _parse_value(v) for k, v in _ATTR.findall(m.group("attrs"))}
        target = m.group("target")
        if target is not None and target not in {t.value for t in Target}:
            raise IRError(f"line {ln}: unknown target @{target}")
        try:
            node = node_from_dict({"id": m.group("id"), "op": op, "attrs": attrs, "target": target})
            g.add(node)
        except ValueError as e:
            raise IRError(f"line {ln}: {e}") from e
        operands = []
        for tok in filter(None, (t.strip() for t in (m.group("operands") or "").split(","))):
            parts = tok.split()
            if not parts[0].startswith("%") or (len(parts) > 1 and parts[1:] != ["feedback"]):
                raise IRError(f"line {ln}: bad operand {tok!r}")
            operands.append((parts[0][1:], len(parts) > 1))
        ins = [t.strip() for t in m.group("ins").split(",") if t.strip()]
        if len(ins) != len(operands):
            raise IRError(f"line {ln}: {len(operands)} operand(s) but {len(ins)} input type(s)")
        pending.append((ln, node.id, operands, ins, m.group("out").strip()))
    if g is None or not closed:
        raise IRError("module is missing or not closed with '}'")
    for ln, nid, operands, ins, out in pending:
        node = g.nodes[nid]
        if out != node.out_type():
            raise IRError(f"line {ln}: %{nid} declares {out} but {node.op()} produces {node.out_type()}")
        for (src, fb), kind in zip(operands, ins):
            if src not in g.nodes:
                raise IRError(f"line {ln}: %{nid} uses undefined value %{src}")
            if kind != g.nodes[src].out_kind().value:
                raise IRError(f"line {ln}: %{src} is {g.nodes[src].out_kind().value}, not {kind}")
            g.connect(src, nid, feedback=fb)
    try:
        return g.validate()
    except IRError as e:
        raise IRError(str(e)) from None
