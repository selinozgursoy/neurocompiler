"""Core enums of the Representation Abstraction Layer (RAL)."""
from enum import Enum


class EdgeKind(str, Enum):
    """What travels on an edge."""
    TENSOR = "tensor"   # continuous manifold / tensor data, dense
    SPIKES = "spikes"   # sparse address-event spike packets


class Target(str, Enum):
    NEUROMORPHIC = "NeuromorphicCore"
    PIM = "PIMCrossbar"
    TENSOR = "TensorAccelerator"


class Plasticity(str, Enum):
    NONE = "none"
    LOCAL_STDP = "local_stdp"    # written on the edge core, reconciled at epochs
    GLOBAL_SYNC = "global_sync"  # written by the accelerator-side optimizer
