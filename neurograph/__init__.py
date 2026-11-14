"""NeuroGraph-HW: compile geometric neural circuits onto heterogeneous hardware."""
from .ir import Graph, parse_ir, print_ir  # noqa: F401
from .compiler import compile_graph, HardwareConfig  # noqa: F401
from .runtime import simulate, benchmark  # noqa: F401
from .workloads import PRESETS, load_preset  # noqa: F401

__version__ = "0.1.0"
