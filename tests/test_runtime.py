import math

import numpy as np
import pytest

import neurograph as ng
from neurograph.compiler import HardwareConfig, compile_graph
from neurograph.ir import ContinuousAttractor, DenseTensorOp, Graph, KoopmanOperator, ManifoldGroupAction, Target
from neurograph.runtime import MeshNoC, benchmark, simulate
from neurograph.runtime.kernels import AttractorKernel, KoopmanKernel

F = "8x0e+4x1o+2x2e"


@pytest.fixture(scope="module")
def ring_bench(programs):
    return benchmark(programs["hippocampal_ring"], steps=200, seed=1)


def test_telemetry_shape(ring_bench):
    het = ring_bench["heterogeneous"]
    assert {"totals", "energy_breakdown_uj", "by_target_uj", "stability", "series", "per_node", "noc", "disclaimer"} <= set(het)
    n = len(het["series"]["t"])
    assert n == 50 and all(len(v) == n for v in het["series"].values())
    assert het["totals"]["energy_mj"] > 0 and het["totals"]["latency_ms_p95"] >= het["totals"]["latency_ms_mean"] * 0.5
    assert "Not a measurement" in het["disclaimer"]


def test_energy_accounting_adds_up(ring_bench):
    for mode in ("heterogeneous", "gpu_baseline"):
        r = ring_bench[mode]
        total_uj = r["totals"]["energy_mj"] * 1e3
        assert sum(r["energy_breakdown_uj"].values()) == pytest.approx(total_uj, rel=1e-4)
        node_uj = sum(v["energy_uj"] for v in r["per_node"].values())
        extra = sum(r["energy_breakdown_uj"][k] for k in ("noc", "codec", "sync"))
        assert node_uj + extra == pytest.approx(total_uj, rel=1e-4)
        assert sum(r["by_target_uj"].values()) == pytest.approx(node_uj, rel=1e-4)


def test_runs_are_deterministic(programs):
    a = simulate(programs["bci_decoder"], steps=60, seed=3)
    b = simulate(programs["bci_decoder"], steps=60, seed=3)
    c = simulate(programs["bci_decoder"], steps=60, seed=4)
    assert a["totals"] == b["totals"] and a["series"] == b["series"]
    assert a["totals"]["events_per_step"] != c["totals"]["events_per_step"]


def test_both_modes_see_the_same_stimulus(ring_bench):
    het, gpu = ring_bench["heterogeneous"], ring_bench["gpu_baseline"]
    assert het["per_node"]["velocity"]["events"] == gpu["per_node"]["velocity"]["events"] > 0


def test_baseline_uses_no_mesh_and_only_the_accelerator(ring_bench):
    gpu, het = ring_bench["gpu_baseline"], ring_bench["heterogeneous"]
    assert gpu["totals"]["noc_flit_hops"] == 0 and gpu["energy_breakdown_uj"]["noc"] == 0
    assert gpu["by_target_uj"]["NeuromorphicCore"] == 0 and gpu["by_target_uj"]["PIMCrossbar"] == 0
    assert het["totals"]["noc_flit_hops"] > 0 and het["by_target_uj"]["NeuromorphicCore"] > 0


def test_event_driven_ring_costs_less_than_dense(ring_bench):
    het, gpu = ring_bench["heterogeneous"]["per_node"]["ring"], ring_bench["gpu_baseline"]["per_node"]["ring"]
    assert het["synops"] < 0.2 * gpu["macs"]            # far fewer operations than dense evaluation
    assert het["energy_uj"] < gpu["energy_uj"]


def test_memory_wall_metric(ring_bench):
    c = ring_bench["comparison"]
    assert 0 <= c["memory_wall_savings_pct"] <= 100
    assert ring_bench["heterogeneous"]["totals"]["weight_bytes_moved_per_step"] < ring_bench["gpu_baseline"]["totals"]["weight_bytes_moved_per_step"]


def test_accuracy_cost_of_event_coding_is_reported(ring_bench):
    s = ring_bench["comparison"]["stability"]
    assert s["gpu_baseline"]["attractor_phase_error_rad_rms"] < s["heterogeneous"]["attractor_phase_error_rad_rms"] < 0.5
    assert s["heterogeneous"]["attractor_bump_amplitude_min"] > 0.9
    assert s["gpu_baseline"]["koopman_deviation_mean"] == 0 < s["heterogeneous"]["koopman_deviation_mean"]


def test_consolidation_bounds_synchronization_drift(programs):
    often = simulate(programs["hippocampal_ring"], steps=160, sync_every=16)
    never = simulate(programs["hippocampal_ring"], steps=160, sync_every=10_000)
    assert often["stability"]["sync_drift_max"] < 0.25 * never["stability"]["sync_drift_max"]
    assert often["totals"]["sync_epochs"] == 10 and never["totals"]["sync_epochs"] == 0
    assert often["energy_breakdown_uj"]["sync"] > 0 == never["energy_breakdown_uj"]["sync"]


def test_feedback_edges_are_epoch_registered(programs):
    """Until the first barrier the loop is open, so removing the feedback edge changes nothing."""
    g = ng.load_preset("bci_decoder")
    open_loop = g.copy()
    open_loop.edges = [e for e in open_loop.edges if not e.feedback]
    a = simulate(compile_graph(g), steps=40, sync_every=10_000)
    b = simulate(compile_graph(open_loop), steps=40, sync_every=10_000)
    assert a["per_node"]["population"]["events"] == b["per_node"]["population"]["events"]
    closed = simulate(compile_graph(g), steps=40, sync_every=8)
    assert closed["per_node"]["population"]["synops"] != a["per_node"]["population"]["synops"]


def test_modulators_flow_only_with_a_global_objective(programs):
    assert simulate(programs["hippocampal_ring"], steps=20)["totals"]["modulator_bytes_per_step"] == 256
    assert simulate(programs["se3_spatial"], steps=20)["totals"]["modulator_bytes_per_step"] == 0


def test_shared_cells_keep_symmetry_end_to_end():
    """Programming error alone leaves a Schur-shared map exactly equivariant; read noise does not."""
    g = Graph("sym")
    g.add(ManifoldGroupAction("emb", kind="sph_harm", irreps_out=F))
    g.add(ManifoldGroupAction("lin", kind="equiv_linear", irreps_in=F, irreps_out=F, bits=8, target="PIMCrossbar"))
    g.connect("emb", "lin")

    def err(read, prog):
        hw = HardwareConfig.from_overrides({"pim": {"read_noise": read, "prog_noise": prog}})
        return simulate(compile_graph(g, hw), steps=64)["stability"]["symmetry_error_mean"]

    assert err(0.0, 0.05) < 1e-9
    assert err(0.02, 0.0) > 1e-3


def test_koopman_kernel_reports_device_spectrum_and_reencodes():
    hw = HardwareConfig()
    node = KoopmanOperator("k", observables=64, reencode_every=0, target=Target.PIM)
    free = KoopmanKernel(node, hw, 0)
    assert free.info["rho_float"] == pytest.approx(0.98, abs=0.02) and free.info["rho_device"] > 0
    often = KoopmanKernel(KoopmanOperator("k", observables=64, reencode_every=2, target=Target.PIM), hw, 0)
    d_free = np.mean([free.step(t, [])[2]["koopman_deviation"] for t in range(120)])
    d_often = np.mean([often.step(t, [])[2]["koopman_deviation"] for t in range(120)])
    assert d_often < d_free


def test_stationary_bump_emits_no_events():
    k = AttractorKernel(ContinuousAttractor("ring", neurons=128, target=Target.NEUROMORPHIC), HardwareConfig(), 0)
    events = sum(k.step(t, [])[1].events for t in range(50))
    assert events == 0 and k.step(50, [])[2]["bump_amplitude"] == pytest.approx(1.0, abs=1e-3)


def test_mesh_routes_x_then_y_and_queues_on_shared_links():
    noc = MeshNoC(8, 8, t_hop_ns=4.0)
    path = noc.route((1, 1), (4, 3))
    assert len(path) == 5 and path[0] == ((1, 1), (2, 1)) and path[-1] == ((4, 2), (4, 3))
    noc.send((0, 0), (3, 0), 40)                       # 10 flits over 3 hops
    alone = noc.deliver()
    assert alone["flit_hops"] == 30 and alone["arrival_ns"] == pytest.approx(3 * 10 * 4.0)
    noc.send((0, 0), (3, 0), 40); noc.send((0, 0), (3, 0), 40)
    both = noc.deliver()
    assert both["arrival_ns"] > alone["arrival_ns"] and both["peak_link_flits"] == 20
    assert noc.total_flit_hops == 90 and noc.hottest_links(1)[0]["flits"] == 30


def test_argument_validation(programs):
    p = programs["se3_spatial"]
    for kwargs in ({"mode": "quantum"}, {"steps": 0}, {"sync_every": 0}):
        with pytest.raises(ValueError):
            simulate(p, **kwargs)


def test_high_order_irreps_compile_but_do_not_simulate():
    g = Graph("l3")
    g.add(ManifoldGroupAction("emb", kind="sph_harm", irreps_out="4x0e+2x3o"))
    prog = compile_graph(g)
    with pytest.raises(ValueError, match="l <= 2"):
        simulate(prog, steps=2)


def test_all_presets_benchmark(programs):
    for name, prog in programs.items():
        b = benchmark(prog, steps=48)
        c = b["comparison"]
        assert c["energy_mj"]["heterogeneous"] > 0 and c["energy_mj"]["gpu_baseline"] > 0, name
        assert math.isfinite(c["energy_ratio_gpu_over_hetero"]) and math.isfinite(c["latency_ratio_gpu_over_hetero"])
