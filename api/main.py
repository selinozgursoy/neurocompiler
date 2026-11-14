"""FastAPI service: compilation, IR inspection, simulation and benchmarking.
The same app serves the web studio from ../web, so one container serves everything."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import neurograph as ng
from neurograph.compiler import SOURCES, CompiledProgram, HardwareConfig, compile_graph
from neurograph.ir import ContinuousAttractor, Graph, IRError, parse_ir, print_ir
from neurograph.runtime import DISCLAIMER, benchmark, simulate
from neurograph.workloads import PRESETS, load_preset

_HERE = Path(__file__).resolve().parent
WEB = next((p for p in (_HERE.parent / "web", _HERE / "web") if (p / "index.html").exists()), _HERE.parent / "web")
MAX_NODES, MAX_UNITS = 200, 2048

app = FastAPI(title="NeuroGraph-HW", version=ng.__version__,
              description="Compile geometric neural circuits onto heterogeneous hardware. Energy and latency are simulated estimates.")


class Source(BaseModel):
    preset: str | None = Field(None, description="preset id; see GET /api/presets")
    ir_text: str | None = Field(None, description="MLIR-style NeuroGraph IR")
    graph: dict | None = Field(None, description="graph JSON: {name, nodes, edges}")
    fusion: bool = True
    hardware: dict | None = Field(None, description="overrides, e.g. {'pim': {'tiles': 4}}")
    w_latency: float = Field(0.0, ge=0, description="pJ charged per ns of latency in the partitioner")


class SimRequest(Source):
    steps: int = Field(400, ge=1, le=5000)
    seed: int = Field(0, ge=0)
    sync_every: int = Field(64, ge=1, le=5000)
    record_every: int = Field(4, ge=1, le=100)


class SimOne(SimRequest):
    mode: Literal["heterogeneous", "gpu_baseline"] = "heterogeneous"


class ParseRequest(BaseModel):
    ir_text: str


def _graph(src: Source) -> Graph:
    given = [x for x in (src.preset, src.ir_text, src.graph) if x]
    if len(given) != 1:
        raise HTTPException(422, "give exactly one of: preset, ir_text, graph")
    try:
        if src.preset:
            g = load_preset(src.preset)
        elif src.ir_text:
            g = parse_ir(src.ir_text)
        else:
            g = Graph.from_dict(src.graph)
    except KeyError as e:
        raise HTTPException(404, str(e).strip('"')) from e
    except IRError as e:
        raise HTTPException(422, f"invalid IR: {e}") from e
    if len(g) > MAX_NODES:
        raise HTTPException(422, f"graph has {len(g)} nodes; this service accepts at most {MAX_NODES}")
    return g


def _compile(src: Source) -> CompiledProgram:
    g = _graph(src)
    try:
        return compile_graph(g, HardwareConfig.from_overrides(src.hardware), fusion=src.fusion, w_latency=src.w_latency)
    except (IRError, ValueError) as e:
        raise HTTPException(422, f"cannot compile: {e}") from e


def _check_sim_size(prog: CompiledProgram) -> None:
    for n in prog.graph.nodes.values():
        f = n.features()
        size = n.neurons if isinstance(n, ContinuousAttractor) else max(f.neurons, f.in_dim, f.out_dim)
        if size > MAX_UNITS:
            raise HTTPException(422, f"%{n.id} has {size} units; this service simulates at most {MAX_UNITS} per node (compile still works)")


@app.get("/api/health")
def health():
    return {"status": "ok", "version": ng.__version__}


@app.get("/api/presets")
def presets():
    out = []
    for pid, p in PRESETS.items():
        g = p["build"]()
        out.append({"id": pid, "title": p["title"], "summary": p["summary"], "nodes": len(g), "ir": print_ir(g), "graph": g.to_dict()})
    return out


@app.get("/api/hardware")
def hardware():
    return {"defaults": HardwareConfig().to_dict(),
            "sources": {k: {"status": s, "note": n} for k, (s, n) in SOURCES.items()},
            "disclaimer": DISCLAIMER}


@app.post("/api/ir/parse")
def ir_parse(req: ParseRequest):
    try:
        g = parse_ir(req.ir_text)
    except IRError as e:
        raise HTTPException(422, f"invalid IR: {e}") from e
    return {"graph": g.to_dict(), "ir": print_ir(g)}


@app.post("/api/compile")
def compile_(src: Source):
    return _compile(src).to_dict()


@app.post("/api/simulate")
def simulate_(req: SimOne):
    prog = _compile(req)
    _check_sim_size(prog)
    try:
        return simulate(prog, req.steps, req.seed, req.mode, req.sync_every, req.record_every)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


@app.post("/api/benchmark")
def benchmark_(req: SimRequest):
    prog = _compile(req)
    _check_sim_size(prog)
    try:
        out = benchmark(prog, req.steps, req.seed, req.sync_every, req.record_every)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    out["compiled"] = prog.to_dict()
    return out


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(WEB / "index.html")


app.mount("/static", StaticFiles(directory=WEB), name="static")
