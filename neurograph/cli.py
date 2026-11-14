"""Command line interface:  neurograph compile | simulate | benchmark | presets | serve"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .compiler import HardwareConfig, compile_graph
from .ir import Graph, IRError, parse_ir
from .runtime import benchmark, simulate
from .workloads import PRESETS, load_preset


def load_source(src: str) -> Graph:
    """A preset name, a .json graph file, or a textual IR file."""
    if src in PRESETS:
        return load_preset(src)
    path = Path(src)
    if not path.exists():
        raise IRError(f"{src!r} is neither a preset ({', '.join(PRESETS)}) nor a file")
    text = path.read_text()
    return Graph.from_dict(json.loads(text)) if path.suffix == ".json" else parse_ir(text)


def _hardware(path: str | None) -> HardwareConfig:
    return HardwareConfig.from_overrides(json.loads(Path(path).read_text())) if path else HardwareConfig()


def _write(path: str | None, text: str) -> None:
    if path:
        Path(path).write_text(text)
    else:
        sys.stdout.write(text if text.endswith("\n") else text + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="neurograph", description="Compile geometric neural circuits onto heterogeneous hardware.")
    ap.add_argument("--version", action="version", version=f"neurograph {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("source", help="preset name, IR text file, or graph .json")
        p.add_argument("--hardware", help="JSON file of hardware overrides")
        p.add_argument("--no-fusion", action="store_true", help="skip the fusion pass")

    c = sub.add_parser("compile", help="compile and print MLIR-style IR")
    common(c)
    c.add_argument("-o", "--output", help="write the compiled IR here")
    c.add_argument("--mapping", help="write the hardware mapping JSON here")
    c.add_argument("--report", action="store_true", help="print the compile report as JSON instead of IR")

    for name, helptext in (("simulate", "run the heterogeneous mapping"), ("benchmark", "compare against the GPU-only baseline")):
        s = sub.add_parser(name, help=helptext)
        common(s)
        s.add_argument("--steps", type=int, default=400)
        s.add_argument("--seed", type=int, default=0)
        s.add_argument("--sync-every", type=int, default=64)
        s.add_argument("-o", "--output", help="write full telemetry JSON here")

    sub.add_parser("presets", help="list preset workloads")
    sv = sub.add_parser("serve", help="start the API and web studio")
    sv.add_argument("--host", default="0.0.0.0")
    sv.add_argument("--port", type=int, default=7860)

    args = ap.parse_args(argv)
    try:
        if args.cmd == "presets":
            for k, v in PRESETS.items():
                print(f"{k:18s} {v['title']}: {v['summary']}")
            return 0
        if args.cmd == "serve":
            import uvicorn
            uvicorn.run("api.main:app", host=args.host, port=args.port)
            return 0
        prog = compile_graph(load_source(args.source), _hardware(args.hardware), fusion=not args.no_fusion)
        if args.cmd == "compile":
            if args.mapping:
                Path(args.mapping).write_text(json.dumps(prog.mapping, indent=2))
            _write(args.output, json.dumps(prog.report, indent=2) if args.report else prog.ir_text)
            return 0
        if args.cmd == "simulate":
            out = simulate(prog, args.steps, args.seed, "heterogeneous", args.sync_every)
            summary = {"module": out["module"], "totals": out["totals"], "stability": out["stability"], "disclaimer": out["disclaimer"]}
        else:
            out = benchmark(prog, args.steps, args.seed, args.sync_every)
            summary = {"module": out["module"], "comparison": out["comparison"], "disclaimer": out["disclaimer"]}
        if args.output:
            Path(args.output).write_text(json.dumps(out))
        print(json.dumps(summary, indent=2))
        return 0
    except (IRError, ValueError, KeyError, json.JSONDecodeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
