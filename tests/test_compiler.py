import pytest

import neurograph as ng
from neurograph.compiler import HardwareConfig, Partitioner, assign_all, compile_graph, fuse, node_cost, units_needed
from neurograph.compiler.cost import cut_cost
from neurograph.ir import (ContinuousAttractor, DenseTensorOp, Graph, KoopmanOperator, ManifoldGroupAction,
                           Plasticity, SpikeEventNode, Target)

HW = HardwareConfig()
F = "8x0e+4x1o+2x2e"


def chain(*nodes):
    g = Graph("c")
    for n in nodes:
        g.add(n)
    for a, b in zip(nodes, nodes[1:]):
        g.connect(a.id, b.id)
    return g.validate()


# ---------------------------------------------------------------- fusion pass
def test_projection_is_idempotent():
    g = chain(DenseTensorOp("src", kind="nonlinearity", in_dim=9),
              ManifoldGroupAction("p1", kind="project", manifold="so3", dim=9),
              ManifoldGroupAction("p2", kind="project", manifold="so3", dim=9),
              DenseTensorOp("out", kind="reduce", in_dim=9, out_dim=3))
    f, rep = fuse(g)
    assert list(f.nodes) == ["src", "p1", "out"] and rep.rewrites[0]["rule"] == "project_idempotent"
    assert [(e.src, e.dst) for e in f.edges] == [("src", "p1"), ("p1", "out")]


def test_projections_onto_different_manifolds_are_kept():
    g = chain(ManifoldGroupAction("p1", kind="project", manifold="sphere", dim=9),
              ManifoldGroupAction("p2", kind="project", manifold="so3", dim=9))
    assert len(fuse(g)[0]) == 2


def test_exp_log_cancel():
    g = chain(DenseTensorOp("src", kind="nonlinearity", in_dim=8),
              ManifoldGroupAction("e", kind="exp_map", manifold="sphere", dim=8),
              ManifoldGroupAction("l", kind="log_map", manifold="sphere", dim=8),
              DenseTensorOp("out", kind="reduce", in_dim=8, out_dim=2))
    f, rep = fuse(g)
    assert list(f.nodes) == ["src", "out"] and [(e.src, e.dst) for e in f.edges] == [("src", "out")]
    assert rep.rewrites[0]["rule"] == "exp_log_cancel"


def test_tensor_product_and_linear_fuse_into_one_op():
    g = chain(ManifoldGroupAction("tp", kind="tensor_product", irreps_in=F, irreps_sh="1x0e+1x1o", irreps_out=F),
              ManifoldGroupAction("lin", kind="equiv_linear", irreps_in=F, irreps_out=F))
    f, _ = fuse(g)
    assert list(f.nodes) == ["tp"] and f.nodes["tp"].kind == "fused_conv" and f.nodes["tp"].composed == 2


def test_koopman_steps_fold_into_a_power():
    g = chain(KoopmanOperator("a", observables=32), KoopmanOperator("b", observables=32), KoopmanOperator("c", observables=32))
    f, _ = fuse(g)
    assert list(f.nodes) == ["a"] and f.nodes["a"].steps == 3


def test_schur_sharing_shrinks_cells():
    g = chain(ManifoldGroupAction("lin", kind="equiv_linear", irreps_in=F, irreps_out=F))
    before = g.nodes["lin"].features().cells
    f, rep = fuse(g)
    after = f.nodes["lin"].features()
    assert before == 30 * 30 and after.cells == 64 + 16 + 4 and after.reads == 5
    assert rep.rewrites[-1]["rule"] == "schur_share"


def test_fusion_never_crosses_a_fan_out_or_a_plastic_node():
    g = Graph("fan")
    for n in (ManifoldGroupAction("p1", kind="project", manifold="so3", dim=9),
              ManifoldGroupAction("p2", kind="project", manifold="so3", dim=9),
              DenseTensorOp("side", kind="reduce", in_dim=9, out_dim=3)):
        g.add(n)
    g.connect("p1", "p2"); g.connect("p1", "side")
    assert len(fuse(g.validate())[0]) == 3


def test_fusion_leaves_the_input_untouched_and_is_idempotent():
    g = ng.load_preset("se3_spatial")
    before = g.to_dict()
    f, rep = fuse(g)
    assert g.to_dict() == before and rep.nodes_before == 12 and rep.nodes_after == 8
    assert fuse(f)[1].rewrites == []


def test_fusion_reduces_cut_edges_on_the_se3_preset(programs):
    cuts = programs["se3_spatial"].report["cut_edges"]
    assert cuts["with_fusion"] < cuts["without_fusion"]


# ------------------------------------------------------------------ cost model
def test_feasibility_rules():
    koop = KoopmanOperator("k", observables=64)
    assert node_cost(koop, Target.PIM, HW).feasible and not node_cost(koop, Target.NEUROMORPHIC, HW).feasible
    hi = DenseTensorOp("r", kind="readout", in_dim=64, out_dim=8, bits=32)
    assert not node_cost(hi, Target.PIM, HW).feasible                       # needs more bits than a crossbar gives
    assert not node_cost(DenseTensorOp("n", kind="nonlinearity", bits=8), Target.PIM, HW).feasible
    ring = ContinuousAttractor("c", neurons=256, plasticity="local_stdp")
    assert node_cost(ring, Target.NEUROMORPHIC, HW).feasible and not node_cost(ring, Target.PIM, HW).feasible
    for n in (koop, hi, ring):
        assert node_cost(n, Target.TENSOR, HW).feasible                      # the baseline can always run


def test_neuromorphic_cost_scales_with_activity_not_size():
    quiet = node_cost(ContinuousAttractor("a", neurons=256, activity=0.01), Target.NEUROMORPHIC, HW).energy_pj
    busy = node_cost(ContinuousAttractor("a", neurons=256, activity=0.5), Target.NEUROMORPHIC, HW).energy_pj
    dense_q = node_cost(ContinuousAttractor("a", neurons=256, activity=0.01), Target.TENSOR, HW).energy_pj
    dense_b = node_cost(ContinuousAttractor("a", neurons=256, activity=0.5), Target.TENSOR, HW).energy_pj
    assert busy > 10 * quiet and dense_q == dense_b


def test_pim_energy_is_dominated_by_conversions():
    c = node_cost(KoopmanOperator("k", observables=256), Target.PIM, HW)
    assert c.parts["pim_adc"] > c.parts["pim_dac"] + c.parts["pim_array"]
    assert c.energy_pj == pytest.approx(256 * 256 * HW.pim.e_mac_pj, rel=1e-6)   # calibrated to the full tile


def test_memory_wall_applies_only_beyond_the_cache():
    small = node_cost(DenseTensorOp("m", kind="matmul", in_dim=512, out_dim=512), Target.TENSOR, HW)
    big = node_cost(DenseTensorOp("m", kind="matmul", in_dim=4096, out_dim=4096), Target.TENSOR, HW)
    assert small.parts["memory"] == 0 and big.parts["memory"] > big.parts["tensor"]


def test_units_needed():
    assert units_needed(KoopmanOperator("k", observables=256), Target.PIM, HW) == 1
    assert units_needed(KoopmanOperator("k", observables=600), Target.PIM, HW) == 9
    assert units_needed(ContinuousAttractor("c", neurons=1024), Target.NEUROMORPHIC, HW) == 8   # synapse memory binds


def test_cut_cost_is_zero_inside_a_target_and_pays_for_the_bridge():
    g = chain(SpikeEventNode("e", kind="encoder", population=64), SpikeEventNode("l", kind="lif", population=64))
    e = g.edges[0]
    assert cut_cost(g, e, Target.NEUROMORPHIC, Target.NEUROMORPHIC, HW).energy_pj == 0
    to_acc = cut_cost(g, e, Target.NEUROMORPHIC, Target.TENSOR, HW)
    assert to_acc.parts["codec"] > 0 and to_acc.latency_ns >= HW.noc.t_bridge_us * 1e3


# ------------------------------------------------------------------ partitioner
def test_routing_invariants_on_presets(programs):
    for prog in programs.values():
        g = prog.graph
        g.verify_mapping()
        for n in g.nodes.values():
            f = n.features()
            if n.target == Target.PIM:
                assert f.linear and n.bits <= HW.pim.bits and n.plasticity != Plasticity.LOCAL_STDP
            if n.target == Target.NEUROMORPHIC:
                assert f.event_capable
            if n.plasticity == Plasticity.LOCAL_STDP:
                assert n.target == Target.NEUROMORPHIC                       # ownership contract
            if not f.linear and not f.event_capable:
                assert n.target == Target.TENSOR


def test_expected_targets(programs):
    t = {n.id: n.target for n in programs["hippocampal_ring"].graph.nodes.values()}
    assert t["ring"] == Target.NEUROMORPHIC and t["path"] == Target.PIM and t["place"] == Target.TENSOR
    t = {n.id: n.target for n in programs["se3_spatial"].graph.nodes.values()}
    assert t["predict"] == Target.PIM and t["tp"] == Target.TENSOR and t["twist"] == Target.TENSOR


def test_activity_decides_between_cores_and_accelerator():
    def target(activity):
        g = chain(ContinuousAttractor("ring", neurons=256, activity=activity))
        return Partitioner(HW).run(g)[0].nodes["ring"].target
    assert target(0.01) == Target.NEUROMORPHIC and target(0.9) == Target.TENSOR


def test_capacity_is_legalized():
    hw = HardwareConfig.from_overrides({"pim": {"tiles": 2}})
    g = Graph("many")
    for i in range(5):
        g.add(KoopmanOperator(f"k{i}", observables=256))
    mapped, rep = Partitioner(hw).run(g.validate())
    on_pim = [n for n in mapped.nodes.values() if n.target == Target.PIM]
    assert len(on_pim) == 2 and len(rep.evictions) == 3


def test_never_worse_than_the_baseline_without_contracts(programs):
    for name in ("se3_spatial",):
        r = programs[name].report["partition"]
        assert r["objective_pj"] <= r["baseline_objective_pj"]
    g = chain(DenseTensorOp("a", kind="mlp", in_dim=32, out_dim=32), DenseTensorOp("b", kind="reduce", in_dim=32, out_dim=4))
    mapped, rep = Partitioner(HW).run(g)
    assert all(n.target == Target.TENSOR for n in mapped.nodes.values()) and rep.objective == rep.baseline_objective


def test_pins_are_respected_and_checked():
    g = chain(KoopmanOperator("k", observables=64))
    g.nodes["k"].target = Target.TENSOR
    assert Partitioner(HW).run(g)[0].nodes["k"].target == Target.TENSOR
    g.nodes["k"].target = Target.NEUROMORPHIC
    with pytest.raises(ValueError, match="pinned"):
        Partitioner(HW).run(g)


def test_baseline_is_all_accelerator(programs):
    assert all(n.target == Target.TENSOR for n in programs["bci_decoder"].baseline.nodes.values())
    assert all(n.target == Target.TENSOR for n in assign_all(ng.load_preset("hippocampal_ring")).nodes.values())


def test_mapping_document(programs):
    m = programs["hippocampal_ring"].mapping
    assert set(m["nodes"]) == set(programs["hippocampal_ring"].graph.nodes)
    assert m["usage"]["NeuromorphicCore"] <= m["capacity"]["NeuromorphicCore"]
    w, h = m["noc"]["mesh"]
    for p in m["nodes"].values():
        assert all(0 <= x < w and 0 <= y < h for x, y in p["slots"])
    assert "ring" in m["plasticity"]["local_stdp"] and "place" in m["plasticity"]["global_sync"]


def test_hardware_overrides_are_validated():
    assert HardwareConfig.from_overrides({"pim": {"tiles": 4}}).pim.tiles == 4
    for bad in ({"quantum": {}}, {"pim": {"qubits": 1}}, {"pim": {"tiles": -1}}, {"pim": {"tiles": "many"}}):
        with pytest.raises(ValueError):
            HardwareConfig.from_overrides(bad)


def test_compile_report_shape(programs):
    r = programs["se3_spatial"].report
    assert {"fusion", "partition", "cut_edges", "estimated", "note"} <= set(r)
    assert r["estimated"]["ratio"] > 1 and "not measurements" in r["note"]
    assert compile_graph(ng.load_preset("se3_spatial"), fusion=False).report["fusion"]["rewrites"] == []
