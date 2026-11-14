"""Preset neural circuit workloads."""
from __future__ import annotations

from .ir import (ContinuousAttractor, DenseTensorOp, Graph, KoopmanOperator, ManifoldGroupAction, SpikeEventNode)

_F = "32x0e+16x1o+8x2e"      # a 120-dimensional equivariant feature field
_SH = "1x0e+1x1o+1x2e"


def se3_spatial() -> Graph:
    """Equivariant point features, a pose on SO(3) and a Koopman predictor."""
    g = Graph("se3_equivariant_spatial_circuit")
    g.add(ManifoldGroupAction("embed", name="spherical harmonic embedding", kind="sph_harm", group="SE3", irreps_out=_F))
    g.add(ManifoldGroupAction("align_a", name="frame alignment", kind="group_act", group="SE3", irreps_in=_F))
    g.add(ManifoldGroupAction("align_b", name="camera to body", kind="group_act", group="SE3", irreps_in=_F))
    g.add(ManifoldGroupAction("tp", name="tensor product", kind="tensor_product", group="SE3", irreps_in=_F, irreps_sh=_SH, irreps_out=_F))
    g.add(ManifoldGroupAction("mix", name="equivariant linear", kind="equiv_linear", group="SE3", irreps_in=_F, irreps_out=_F))
    g.add(DenseTensorOp("gate", name="gate nonlinearity", kind="nonlinearity", in_dim=120))
    g.add(ManifoldGroupAction("head", name="equivariant head", kind="equiv_linear", group="SE3", irreps_in=_F, irreps_out=_F, bits=8))
    g.add(ManifoldGroupAction("pose", name="project to SO(3)", kind="project", manifold="so3", dim=9))
    g.add(ManifoldGroupAction("pose_fix", name="re-orthonormalize", kind="project", manifold="so3", dim=9))
    g.add(KoopmanOperator("predict", name="Koopman predictor", observables=128))
    g.add(KoopmanOperator("predict_2", name="second step", observables=128))
    g.add(DenseTensorOp("twist", name="twist readout", kind="reduce", in_dim=128, out_dim=6))
    for a, b in [("embed", "align_a"), ("align_a", "align_b"), ("align_b", "tp"), ("tp", "mix"), ("mix", "gate"),
                 ("gate", "head"), ("head", "pose"), ("pose", "pose_fix"), ("pose_fix", "predict"),
                 ("predict", "predict_2"), ("predict_2", "twist")]:
        g.connect(a, b)
    return g.validate()


def hippocampal_ring() -> Graph:
    """Path integration on a ring attractor with local plasticity."""
    g = Graph("hippocampal_continuous_attractor_ring")
    g.add(SpikeEventNode("velocity", name="angular velocity cells", kind="encoder", population=64, rate=0.08, fan_out=256))
    g.add(ContinuousAttractor("ring", name="head-direction ring", neurons=256, activity=0.03, plasticity="local_stdp"))
    g.add(SpikeEventNode("heading", name="population vector", kind="decoder", population=256, decode_dim=2))
    g.add(ManifoldGroupAction("wrap", name="project to circle", kind="project", manifold="torus", dim=2))
    g.add(ManifoldGroupAction("wrap_fix", name="re-project", kind="project", manifold="torus", dim=2))
    g.add(KoopmanOperator("path", name="Koopman path predictor", observables=128))
    g.add(DenseTensorOp("place", name="place readout", kind="readout", in_dim=128, out_dim=16, plasticity="global_sync"))
    for a, b in [("velocity", "ring"), ("ring", "heading"), ("heading", "wrap"), ("wrap", "wrap_fix"),
                 ("wrap_fix", "path"), ("path", "place")]:
        g.connect(a, b)
    return g.validate()


def bci_decoder() -> Graph:
    """Closed loop: spikes in, a manifold decode, a controller, stimulation back."""
    g = Graph("closed_loop_spiking_bci_manifold_decoder")
    g.add(SpikeEventNode("electrodes", name="recorded units", kind="encoder", population=256, rate=0.06, fan_out=128))
    g.add(SpikeEventNode("population", name="adaptive population", kind="lif", population=512, fan_out=128, rate=0.04, plasticity="local_stdp"))
    g.add(SpikeEventNode("latent", name="latent decoder", kind="decoder", population=512, decode_dim=16))
    g.add(ManifoldGroupAction("manifold", name="project to sphere", kind="project", manifold="sphere", dim=16))
    g.add(KoopmanOperator("dynamics", name="Koopman latent dynamics", observables=96))
    g.add(DenseTensorOp("controller", name="controller", kind="mlp", in_dim=96, out_dim=32, layers=2, plasticity="global_sync"))
    g.add(SpikeEventNode("stimulation", name="stimulation encoder", kind="stim", population=128, rate=0.03, fan_out=64))
    for a, b in [("electrodes", "population"), ("population", "latent"), ("latent", "manifold"),
                 ("manifold", "dynamics"), ("dynamics", "controller"), ("controller", "stimulation")]:
        g.connect(a, b)
    g.connect("stimulation", "population", feedback=True)      # the loop closes one epoch later
    return g.validate()


PRESETS = {
    "se3_spatial": {"title": "SE(3) Equivariant Spatial Circuit", "build": se3_spatial,
                    "summary": "Equivariant features, a pose on SO(3) and a Koopman predictor."},
    "hippocampal_ring": {"title": "Hippocampal Continuous Attractor Ring", "build": hippocampal_ring,
                         "summary": "Path integration on a plastic ring attractor."},
    "bci_decoder": {"title": "Closed-Loop Spiking BCI Manifold Decoder", "build": bci_decoder,
                    "summary": "Spikes in, a manifold decode, a controller and stimulation back."},
}


def load_preset(name: str) -> Graph:
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
    return PRESETS[name]["build"]()
