"""Typed IR nodes. Each node knows its op name, its output type and the
cost features the partitioner needs."""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import ClassVar

from ..geometry import irreps_dim, parse_irreps, schur_cells
from .types import EdgeKind, Plasticity, Target


@dataclass
class Features:
    """What the partitioner knows about one node, per simulation step."""
    macs: float = 0.0            # multiply-accumulates if evaluated densely
    weights: int = 0             # stored parameters
    cells: int = 0               # crossbar cells needed (<= weights after Schur sharing)
    reads: int = 1               # MVM reads per step on a crossbar (2l+1 multiplexing)
    in_dim: int = 0
    out_dim: int = 0
    neurons: int = 0
    synapses: int = 0
    activity: float = 1.0        # expected fraction of units emitting an event per step
    linear: bool = False         # a pure matrix-vector product with fixed weights
    event_capable: bool = False  # has event-time semantics a neuromorphic core can run


@dataclass
class Node:
    id: str
    name: str = ""
    plasticity: Plasticity = Plasticity.NONE
    bits: int = 32
    target: Target | None = None

    DIALECT: ClassVar[str] = ""
    KINDS: ClassVar[tuple[str, ...]] = ()

    def __post_init__(self):
        if not isinstance(self.plasticity, Plasticity):
            self.plasticity = Plasticity(self.plasticity)
        if self.target is not None and not isinstance(self.target, Target):
            self.target = Target(self.target)

    # ---- overridden by subclasses
    def op(self) -> str: raise NotImplementedError
    def out_kind(self) -> EdgeKind: return EdgeKind.TENSOR
    def width(self) -> int: raise NotImplementedError
    def out_type(self) -> str: return f"tensor<{self.width()}x{'f32' if self.bits >= 32 else f'i{self.bits}'}>"
    def accepts(self) -> set[EdgeKind]: return {EdgeKind.TENSOR}
    def features(self) -> Features: raise NotImplementedError
    def check(self) -> list[str]: return []

    # ---- serialization
    def attrs(self) -> dict:
        skip = {"id", "target"}
        out = {}
        for f in fields(self):
            if f.name in skip:
                continue
            v = getattr(self, f.name)
            if f.name == "kind":
                continue  # carried by the op name
            v = v.value if isinstance(v, (Plasticity, Target)) else v
            default = f.default.value if isinstance(f.default, Plasticity) else f.default
            if v != default:
                out[f.name] = v
        return out

    def to_dict(self) -> dict:
        d = {"id": self.id, "op": self.op(), "attrs": self.attrs(), "type": self.out_type()}
        if self.target is not None:
            d["target"] = self.target.value
        return d


@dataclass
class ManifoldGroupAction(Node):
    """Group-equivariant operators and manifold maps (the `ral.geo` dialect)."""
    kind: str = "equiv_linear"
    group: str = "SO3"
    irreps_in: str = ""
    irreps_out: str = ""
    irreps_sh: str = ""       # second operand of a tensor product
    manifold: str = ""        # sphere | torus | so3 | stiefel
    dim: int = 0
    schur: bool = False       # store one W_l per irrep type (set by the fusion pass)
    composed: int = 1         # how many source ops were fused into this one

    DIALECT = "ral.geo"
    KINDS = ("sph_harm", "tensor_product", "equiv_linear", "fused_conv", "group_act", "project", "exp_map", "log_map")
    _MAPS = ("project", "exp_map", "log_map")

    def op(self): return f"{self.DIALECT}.{self.kind}"

    def width(self):
        if self.kind in self._MAPS:
            return self.dim
        return irreps_dim(self.irreps_out or self.irreps_in)

    def out_type(self):
        if self.kind == "log_map":
            return f"tangent<{self.manifold}, {self.dim}>"
        if self.kind in self._MAPS:
            return f"point<{self.manifold}, {self.dim}>"
        return f'rep<{self.group}, "{self.irreps_out or self.irreps_in}">'

    def check(self):
        errs = []
        if self.kind not in self.KINDS:
            return [f"unknown ral.geo op {self.kind!r}"]
        if self.kind in self._MAPS:
            if not self.manifold or self.dim <= 0:
                errs.append(f"{self.op()} needs a manifold and a positive dim")
            return errs
        try:
            for s in (self.irreps_in, self.irreps_out, self.irreps_sh):
                if s:
                    parse_irreps(s)
        except ValueError as e:
            return [str(e)]
        if not (self.irreps_out or self.irreps_in):
            errs.append(f"{self.op()} needs irreps")
        if self.kind in ("tensor_product", "fused_conv") and self.irreps_in and self.irreps_sh and self.irreps_out and self.kind == "tensor_product":
            from ..geometry import tensor_product_allowed
            bad = tensor_product_allowed(self.irreps_in, self.irreps_sh, self.irreps_out, check_parity=self.group in ("O3", "SE3"))
            if bad:
                errs.append(f"output irreps {bad} violate the selection rules")
        return errs

    def features(self):
        od = self.width()
        if self.kind in self._MAPS:
            return Features(macs=3.0 * self.dim, in_dim=self.dim, out_dim=od)
        idim = irreps_dim(self.irreps_in) if self.irreps_in else od
        if self.kind == "equiv_linear":
            dense, shared = schur_cells(self.irreps_in, self.irreps_out or self.irreps_in)
            if self.schur:
                macs = sum(i.mul * o.mul * (2 * i.l + 1) for i in parse_irreps(self.irreps_in)
                           for o in parse_irreps(self.irreps_out or self.irreps_in) if (i.l, i.parity) == (o.l, o.parity))
                reads = max(2 * i.l + 1 for i in parse_irreps(self.irreps_in))
                return Features(macs=macs, weights=shared, cells=shared, reads=reads, in_dim=idim, out_dim=od, linear=True)
            return Features(macs=dense, weights=dense, cells=dense, in_dim=idim, out_dim=od, linear=True)
        if self.kind == "group_act":
            macs = sum(i.mul * (2 * i.l + 1) ** 2 for i in parse_irreps(self.irreps_in or self.irreps_out))
            return Features(macs=macs, in_dim=idim, out_dim=od)
        if self.kind == "sph_harm":
            return Features(macs=4.0 * od, in_dim=3, out_dim=od)
        # tensor_product / fused_conv: bilinear, block sparse, data dependent
        macs = float(idim * od) * (2 if self.kind == "fused_conv" else 1)
        return Features(macs=macs, weights=idim * od if self.kind == "fused_conv" else 0, in_dim=idim, out_dim=od)


@dataclass
class KoopmanOperator(Node):
    """z' = K z + B u in observable space (`ral.dyn.koopman`). Linear and frozen."""
    observables: int = 64
    rho: float = 0.98
    steps: int = 1
    reencode_every: int = 8
    bits: int = 8

    DIALECT = "ral.dyn"

    def op(self): return "ral.dyn.koopman"
    def width(self): return self.observables

    def check(self):
        errs = []
        if self.observables <= 0:
            errs.append("koopman needs observables > 0")
        if not (0 < self.rho <= 1.0):
            errs.append(f"koopman spectral bound rho={self.rho} must be in (0, 1]")
        if self.steps < 1:
            errs.append("koopman steps must be >= 1")
        return errs

    def features(self):
        n = self.observables
        return Features(macs=float(n * n), weights=n * n, cells=n * n, in_dim=n, out_dim=n, linear=True)


@dataclass
class ContinuousAttractor(Node):
    """A continuous attractor network on a ring or torus (`ral.dyn.attractor`)."""
    neurons: int = 256
    topology: str = "ring"
    fan_out: int = 0           # 0 means all-to-all
    activity: float = 0.03     # expected events per neuron per step
    bits: int = 8

    DIALECT = "ral.dyn"

    def op(self): return "ral.dyn.attractor"
    def out_kind(self): return EdgeKind.SPIKES
    def width(self): return self.neurons
    def out_type(self): return f"spikes<{self.neurons}, {self.bits}>"
    def accepts(self): return {EdgeKind.SPIKES, EdgeKind.TENSOR}

    def check(self):
        errs = []
        if self.neurons <= 0:
            errs.append("attractor needs neurons > 0")
        if self.topology not in ("ring", "torus"):
            errs.append(f"unknown attractor topology {self.topology!r}")
        if not (0 <= self.activity <= 1):
            errs.append("attractor activity must be in [0, 1]")
        return errs

    def features(self):
        fo = self.fan_out or self.neurons
        syn = self.neurons * fo
        return Features(macs=float(syn), weights=syn, cells=syn, in_dim=self.neurons, out_dim=self.neurons,
                        neurons=self.neurons, synapses=syn, activity=self.activity, event_capable=True)


@dataclass
class SpikeEventNode(Node):
    """Event-driven populations and codecs (the `ral.evt` dialect)."""
    kind: str = "lif"
    population: int = 128
    fan_out: int = 16
    rate: float = 0.05         # expected events per neuron per step
    decode_dim: int = 2        # output size of a decoder
    bits: int = 8

    DIALECT = "ral.evt"
    KINDS = ("encoder", "lif", "decoder", "stim")

    def op(self): return f"{self.DIALECT}.{self.kind}"
    def out_kind(self): return EdgeKind.TENSOR if self.kind == "decoder" else EdgeKind.SPIKES
    def width(self): return self.decode_dim if self.kind == "decoder" else self.population

    def out_type(self):
        if self.kind == "decoder":
            return f"tensor<{self.decode_dim}xf32>"
        return f"spikes<{self.population}, {1 if self.kind in ('encoder', 'lif') else self.bits}>"

    def accepts(self):
        return {EdgeKind.TENSOR} if self.kind in ("encoder", "stim") else {EdgeKind.SPIKES}

    def check(self):
        errs = []
        if self.kind not in self.KINDS:
            errs.append(f"unknown ral.evt op {self.kind!r}")
        if self.population <= 0:
            errs.append("spike node needs population > 0")
        if not (0 <= self.rate <= 1):
            errs.append("spike rate must be in [0, 1]")
        return errs

    def features(self):
        syn = self.population * max(1, self.fan_out)
        return Features(macs=float(syn), weights=syn, cells=syn, in_dim=self.population, out_dim=self.width(),
                        neurons=self.population, synapses=syn, activity=self.rate, event_capable=True)


@dataclass
class DenseTensorOp(Node):
    """Ordinary dense tensor work (the `ral.tensor` dialect)."""
    kind: str = "matmul"
    in_dim: int = 64
    out_dim: int = 64
    layers: int = 1

    DIALECT = "ral.tensor"
    KINDS = ("matmul", "mlp", "nonlinearity", "reduce", "readout")

    def op(self): return f"{self.DIALECT}.{self.kind}"
    def width(self): return self.in_dim if self.kind == "nonlinearity" else self.out_dim

    def check(self):
        errs = []
        if self.kind not in self.KINDS:
            errs.append(f"unknown ral.tensor op {self.kind!r}")
        if self.in_dim <= 0 or self.out_dim <= 0:
            errs.append("dense op needs positive dims")
        return errs

    def features(self):
        i, o = self.in_dim, self.width()
        if self.kind in ("matmul", "readout"):
            return Features(macs=float(i * o), weights=i * o, cells=i * o, in_dim=i, out_dim=o, linear=True)
        if self.kind == "mlp":
            w = i * o * max(1, self.layers)
            return Features(macs=float(w), weights=w, in_dim=i, out_dim=o)
        return Features(macs=float(i), in_dim=i, out_dim=o)


NODE_CLASSES = (ManifoldGroupAction, KoopmanOperator, ContinuousAttractor, SpikeEventNode, DenseTensorOp)

OP_REGISTRY: dict[str, tuple[type, str | None]] = {"ral.dyn.koopman": (KoopmanOperator, None), "ral.dyn.attractor": (ContinuousAttractor, None)}
for _cls in (ManifoldGroupAction, SpikeEventNode, DenseTensorOp):
    for _k in _cls.KINDS:
        OP_REGISTRY[f"{_cls.DIALECT}.{_k}"] = (_cls, _k)


def node_from_dict(d: dict) -> Node:
    """Build a node from {'id', 'op', 'attrs', 'target'?}. Unknown ops or attrs raise ValueError."""
    op = d.get("op")
    if op not in OP_REGISTRY:
        raise ValueError(f"unknown op {op!r}; known ops: {sorted(OP_REGISTRY)}")
    cls, kind = OP_REGISTRY[op]
    attrs = dict(d.get("attrs") or {})
    allowed = {f.name for f in fields(cls)} - {"id", "target", "kind"}
    extra = set(attrs) - allowed
    if extra:
        raise ValueError(f"{op} does not take attribute(s) {sorted(extra)}")
    if kind is not None:
        attrs["kind"] = kind
    return cls(id=d["id"], target=d.get("target"), **attrs)
