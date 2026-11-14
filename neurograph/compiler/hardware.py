"""Hardware target descriptions and the constants of the cost model.

Every constant is a model parameter. SOURCES records where each default comes
from and whether it is published, derived from a published figure, or assumed.
The defaults mix chips and process nodes, so absolute energy ratios produced
with them are illustrative. Override them with measurements for your board.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields


@dataclass
class NeuromorphicCoreSpec:
    """Asynchronous event-driven cores (Loihi-class)."""
    cores: int = 128
    neurons_per_core: int = 8192
    synapses_per_core: int = 131072      # 128 KB of synapse memory at 8-bit weights
    weight_bits: int = 8
    e_synop_pj: float = 23.6             # energy per synaptic spike operation
    e_neuron_pj: float = 52.0            # energy per neuron update
    e_stdp_pj: float = 120.0             # energy per synaptic (STDP) update
    t_synop_ns: float = 3.5
    t_neuron_ns: float = 5.3
    t_stdp_ns: float = 6.1


@dataclass
class PIMCrossbarSpec:
    """Analog in-memory crossbar tiles (PCM / ReRAM)."""
    tiles: int = 64
    tile_dim: int = 256
    bits: int = 8
    phases: int = 4                      # read phases in high-precision mode
    e_mac_pj: float = 0.403              # all-in energy per MAC of an 8-bit MVM
    adc_share: float = 0.58              # share of that energy spent in ADCs
    dac_share: float = 0.12              # share spent driving inputs
    t_mvm_ns: float = 520.0              # one full-tile MVM, all phases
    e_write_pj: float = 1000.0           # programming one cell (iterative program-verify)
    max_write_hz: float = 1.0            # endurance budget per cell
    read_noise: float = 0.01             # per-read noise, fraction of the largest weight
    prog_noise: float = 0.01             # programming error, same units

    @property
    def e_adc_pj(self) -> float:         # per conversion
        return self.adc_share * self.e_mac_pj * self.tile_dim / self.phases

    @property
    def e_dac_pj(self) -> float:         # per driven input per phase
        return self.dac_share * self.e_mac_pj * self.tile_dim / self.phases

    @property
    def e_cell_pj(self) -> float:        # per cell read per phase
        return (1.0 - self.adc_share - self.dac_share) * self.e_mac_pj / self.phases


@dataclass
class TensorAcceleratorSpec:
    """A GPU / TPU class dense accelerator, at its peak efficiency."""
    e_mac_pj: float = 2.56               # 400 W / 312 TFLOP/s, two FLOPs per MAC
    macs_per_s: float = 1.56e14
    t_launch_us: float = 10.0            # fixed cost of launching one small kernel
    bytes_per_weight: int = 4
    cache_bytes: float = 40e6            # weights larger than this stream from DRAM every step
    e_dram_pj_per_byte: float = 40.0     # energy of that DRAM traffic: the memory wall


@dataclass
class NoCSpec:
    """Packet-switched mesh between cores and tiles, plus the bridge to the accelerator."""
    mesh_w: int = 16
    mesh_h: int = 12
    flit_bytes: int = 4
    e_hop_pj: float = 3.5                # per flit per hop
    t_hop_ns: float = 4.0
    e_bridge_pj_per_byte: float = 40.0   # off-chip link to the accelerator
    t_bridge_us: float = 2.0
    e_codec_pj: float = 1.0              # tensor <-> event transcoding, per value
    est_hops: float = 3.0                # used before placement


SOURCES = {
    "neuromorphic.e_synop_pj": ("published", "Loihi, 23.6 pJ per synaptic spike op (Davies et al., IEEE Micro 2018)"),
    "neuromorphic.e_stdp_pj": ("published", "Loihi, 120 pJ per synaptic update (Davies et al., 2018)"),
    "neuromorphic.e_neuron_pj": ("recalled", "Loihi, about 52 pJ per neuron update (Davies et al., 2018); check before relying on it"),
    "neuromorphic.t_synop_ns": ("published", "Loihi, 3.5 ns per synaptic spike op (Davies et al., 2018)"),
    "neuromorphic.t_stdp_ns": ("published", "Loihi, 6.1 ns per synaptic update (Davies et al., 2018)"),
    "neuromorphic.t_neuron_ns": ("recalled", "Loihi, about 5.3 ns per neuron update (Davies et al., 2018); check before relying on it"),
    "neuromorphic.cores": ("published", "Loihi 2, 128 neuron cores per chip (Pedersen et al., 2024)"),
    "neuromorphic.neurons_per_core": ("published", "Loihi 2, up to 8,192 neurons per core (Hajizada et al., 2025)"),
    "neuromorphic.synapses_per_core": ("derived", "Loihi 2, up to 128 KB synapse memory per core, at one byte per synapse"),
    "pim.tiles": ("published", "IBM HERMES, 64 cores of 256x256 PCM unit cells (Le Gallo et al., 2023)"),
    "pim.e_mac_pj": ("derived", "HERMES high-precision mode, 2.48 TOPS/W, so 0.403 pJ per operation"),
    "pim.adc_share": ("published", "ISAAC, ADCs take 58% of power (Shafiee et al., as reported by Chou et al., 2019); a different chip from HERMES"),
    "pim.dac_share": ("assumed", "split of the remaining 42% between input drivers and the array"),
    "pim.t_mvm_ns": ("derived", "HERMES, 16.1 TOPS over 64 cores gives about 0.52 us per 256x256 MVM"),
    "pim.e_write_pj": ("assumed", "order of magnitude for iterative PCM programming"),
    "pim.read_noise": ("assumed", "1% of the largest weight per read"),
    "tensor.e_mac_pj": ("derived", "NVIDIA A100, 312 TFLOP/s FP16 at 400 W, so 1.28 pJ per FLOP at peak utilization"),
    "tensor.t_launch_us": ("assumed", "fixed latency of one small kernel launch"),
    "tensor.cache_bytes": ("published", "NVIDIA A100, 40 MB of L2 cache"),
    "tensor.e_dram_pj_per_byte": ("published", "about 5 pJ per bit for a DRAM access (survey figure), so 40 pJ per byte"),
    "noc.e_hop_pj": ("recalled", "Loihi, a few pJ per tile hop (Davies et al., 2018); check before relying on it"),
    "noc.e_bridge_pj_per_byte": ("derived", "1 to 10 pJ per bit communicated off chip; 5 pJ per bit taken"),
    "noc.e_codec_pj": ("assumed", "tensor to event transcoding energy per value"),
    "noc.t_hop_ns": ("assumed", "per-hop latency of an asynchronous mesh router"),
    "noc.t_bridge_us": ("assumed", "latency of one crossing between the mesh and the accelerator"),
}


@dataclass
class HardwareConfig:
    neuromorphic: NeuromorphicCoreSpec = field(default_factory=NeuromorphicCoreSpec)
    pim: PIMCrossbarSpec = field(default_factory=PIMCrossbarSpec)
    tensor: TensorAcceleratorSpec = field(default_factory=TensorAcceleratorSpec)
    noc: NoCSpec = field(default_factory=NoCSpec)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["pim"].update(e_adc_pj=self.pim.e_adc_pj, e_dac_pj=self.pim.e_dac_pj, e_cell_pj=self.pim.e_cell_pj)
        return d

    @classmethod
    def from_overrides(cls, overrides: dict | None) -> "HardwareConfig":
        """Build a config from {'pim': {'tiles': 4}, ...}. Unknown keys raise ValueError."""
        hw = cls()
        for section, values in (overrides or {}).items():
            if not hasattr(hw, section):
                raise ValueError(f"unknown hardware section {section!r}")
            spec = getattr(hw, section)
            names = {f.name: f.type for f in fields(spec)}
            for k, v in (values or {}).items():
                if k not in names:
                    if k in ("e_adc_pj", "e_dac_pj", "e_cell_pj"):
                        continue  # derived, read-only
                    raise ValueError(f"unknown hardware parameter {section}.{k}")
                if not isinstance(v, (int, float)) or isinstance(v, bool) or v < 0:
                    raise ValueError(f"{section}.{k} must be a non-negative number")
                setattr(spec, k, type(getattr(spec, k))(v))
        return hw
