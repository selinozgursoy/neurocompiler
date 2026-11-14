import json

from neurograph.cli import main
from neurograph.ir import parse_ir, print_ir
from neurograph.workloads import load_preset


def test_presets_listing(capsys):
    assert main(["presets"]) == 0
    out = capsys.readouterr().out
    assert "hippocampal_ring" in out and "SE(3)" in out


def test_compile_prints_ir_that_parses(capsys):
    assert main(["compile", "bci_decoder"]) == 0
    g = parse_ir(capsys.readouterr().out)
    assert all(n.target is not None for n in g.nodes.values())


def test_compile_writes_ir_and_mapping(tmp_path, capsys):
    ir, mp = tmp_path / "out.mlir", tmp_path / "map.json"
    assert main(["compile", "se3_spatial", "-o", str(ir), "--mapping", str(mp)]) == 0
    assert parse_ir(ir.read_text()).name == "se3_equivariant_spatial_circuit"
    assert json.loads(mp.read_text())["by_target"]["PIMCrossbar"] == ["predict"]
    assert main(["compile", "se3_spatial", "--report", "--no-fusion"]) == 0
    assert json.loads(capsys.readouterr().out)["fusion"]["rewrites"] == []


def test_compile_from_files(tmp_path, capsys):
    g = load_preset("hippocampal_ring")
    (tmp_path / "ring.mlir").write_text(print_ir(g))
    (tmp_path / "ring.json").write_text(json.dumps(g.to_dict()))
    for f in ("ring.mlir", "ring.json"):
        assert main(["compile", str(tmp_path / f)]) == 0
        assert "@NeuromorphicCore" in capsys.readouterr().out


def test_hardware_file(tmp_path, capsys):
    hw = tmp_path / "hw.json"
    hw.write_text(json.dumps({"pim": {"tiles": 0}}))
    assert main(["compile", "se3_spatial", "--hardware", str(hw)]) == 0
    assert "@PIMCrossbar" not in capsys.readouterr().out


def test_simulate_and_benchmark(tmp_path, capsys):
    assert main(["simulate", "hippocampal_ring", "--steps", "30"]) == 0
    assert json.loads(capsys.readouterr().out)["totals"]["energy_mj"] > 0
    out = tmp_path / "bench.json"
    assert main(["benchmark", "hippocampal_ring", "--steps", "30", "-o", str(out)]) == 0
    assert "comparison" in json.loads(capsys.readouterr().out)
    assert {"heterogeneous", "gpu_baseline"} <= set(json.loads(out.read_text()))


def test_errors_exit_with_code_2(tmp_path, capsys):
    assert main(["compile", "no_such_thing"]) == 2
    assert "error:" in capsys.readouterr().err
    bad = tmp_path / "bad.mlir"
    bad.write_text("module @x {\n  garbage\n}")
    assert main(["compile", str(bad)]) == 2
