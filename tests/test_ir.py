import pytest

from neurograph.geometry import irreps_dim, parse_irreps, schur_cells, tensor_product_allowed
from neurograph.ir import (ContinuousAttractor, DenseTensorOp, EdgeKind, Graph, IRError, KoopmanOperator,
                           ManifoldGroupAction, SpikeEventNode, node_from_dict, parse_ir, print_ir)
from neurograph.workloads import PRESETS, load_preset


def small():
    g = Graph("t")
    g.add(SpikeEventNode("enc", kind="encoder", population=64, rate=0.1))
    g.add(ContinuousAttractor("ring", neurons=128, plasticity="local_stdp"))
    g.add(SpikeEventNode("dec", kind="decoder", population=128))
    g.add(KoopmanOperator("k", observables=32))
    for a, b in [("enc", "ring"), ("ring", "dec"), ("dec", "k")]:
        g.connect(a, b)
    return g.validate()


def test_irreps_parse_and_dim():
    assert irreps_dim("16x0e+8x1o+4x2e") == 16 + 24 + 20
    assert [str(i) for i in parse_irreps("2x0e + 1x1o")] == ["2x0e", "1x1o"]
    with pytest.raises(ValueError):
        parse_irreps("8y1o")


def test_schur_cells_match_hand_count():
    dense, shared = schur_cells("16x0e+8x1o+4x2e", "16x0e+8x1o+4x2e")
    assert dense == 3600 and shared == 16 * 16 + 8 * 8 + 4 * 4


def test_selection_rules():
    assert tensor_product_allowed("1x1o", "1x1o", "1x0e+1x1e+1x2e") == []
    assert tensor_product_allowed("1x1o", "1x1o", "1x3e") == ["3e"]
    assert tensor_product_allowed("1x1o", "1x1o", "1x1o") == ["1o"]            # parity
    assert tensor_product_allowed("1x1o", "1x1o", "1x1o", check_parity=False) == []


def test_edge_kinds_are_inferred_and_typed():
    g = small()
    kinds = {(e.src, e.dst): e.kind for e in g.edges}
    assert kinds[("enc", "ring")] == EdgeKind.SPIKES and kinds[("dec", "k")] == EdgeKind.TENSOR
    assert g.nodes["ring"].out_type() == "spikes<128, 8>"
    assert g.nodes["k"].out_type() == "tensor<32xi8>"


def test_spikes_into_a_tensor_op_need_an_explicit_codec():
    g = Graph("bad")
    g.add(ContinuousAttractor("ring", neurons=64))
    g.add(KoopmanOperator("k", observables=16))
    g.connect("ring", "k")
    with pytest.raises(IRError, match="explicit ral.evt codec"):
        g.validate()


def test_cycle_needs_a_feedback_edge():
    g = Graph("loop")
    g.add(SpikeEventNode("a", kind="lif", population=8))
    g.add(SpikeEventNode("b", kind="lif", population=8))
    g.connect("a", "b"); g.connect("b", "a")
    with pytest.raises(IRError, match="feedback"):
        g.validate()
    g.edges[-1].feedback = True
    assert g.validate().topo_order() == ["a", "b"]


def test_validation_rejects_bad_nodes():
    for node, msg in [(KoopmanOperator("k", rho=1.2), "rho"), (ContinuousAttractor("c", topology="cube"), "topology"),
                      (ManifoldGroupAction("p", kind="project"), "manifold"),
                      (DenseTensorOp("d", kind="matmul", plasticity="local_stdp"), "event-time"),
                      (ManifoldGroupAction("t", kind="tensor_product", group="O3", irreps_in="1x1o", irreps_sh="1x1o", irreps_out="1x3e"), "selection")]:
        g = Graph("g"); g.add(node)
        with pytest.raises(IRError, match=msg):
            g.validate()


def test_duplicate_and_unknown_ids():
    g = Graph("g"); g.add(KoopmanOperator("k"))
    with pytest.raises(IRError):
        g.add(KoopmanOperator("k"))
    with pytest.raises(IRError):
        g.connect("k", "missing")
    with pytest.raises(IRError):
        g.add(KoopmanOperator("1bad"))


def test_node_from_dict_rejects_unknown_ops_and_attrs():
    with pytest.raises(ValueError, match="unknown op"):
        node_from_dict({"id": "x", "op": "ral.nope"})
    with pytest.raises(ValueError, match="does not take"):
        node_from_dict({"id": "x", "op": "ral.dyn.koopman", "attrs": {"neurons": 3}})


@pytest.mark.parametrize("name", list(PRESETS))
def test_text_round_trip(name):
    g = load_preset(name)
    text = print_ir(g)
    g2 = parse_ir(text)
    assert print_ir(g2) == text
    assert {n.id: n.to_dict() for n in g2.nodes.values()} == {n.id: n.to_dict() for n in g.nodes.values()}
    assert sorted((e.src, e.dst, e.kind, e.feedback) for e in g2.edges) == sorted((e.src, e.dst, e.kind, e.feedback) for e in g.edges)


@pytest.mark.parametrize("name", list(PRESETS))
def test_json_round_trip(name):
    g = load_preset(name)
    assert Graph.from_dict(g.to_dict()).to_dict() == g.to_dict()


def test_text_keeps_targets_feedback_and_comments(programs):
    text = programs["bci_decoder"].ir_text
    assert "@NeuromorphicCore" in text and "feedback" in text
    g = parse_ir("// a comment\n" + text)
    assert g.nodes["population"].target.value == "NeuromorphicCore"
    assert any(e.feedback for e in g.edges)


@pytest.mark.parametrize("bad, msg", [
    ("", "missing"),
    ("module @x {\n", "not closed"),
    ("module @x {\n  %a = ral.dyn.koopman {} : () -> tensor<3xi8>\n}", "declares"),
    ("module @x {\n  %a = ral.dyn.koopman {} (%b) : (tensor) -> tensor<64xi8>\n}", "undefined"),
    ("module @x {\n  %a = ral.dyn.koopman {} : () -> tensor<64xi8> @Quantum\n}", "unknown target"),
    ("module @x {\n  nonsense\n}", "cannot parse"),
    ("module @x {\n  %a = ral.dyn.koopman {observables = many} : () -> tensor<64xi8>\n}", "attribute value"),
])
def test_parser_reports_errors(bad, msg):
    with pytest.raises(IRError, match=msg):
        parse_ir(bad)
