import pytest
from fastapi.testclient import TestClient

from api.main import app
from neurograph.ir import print_ir
from neurograph.workloads import load_preset

client = TestClient(app)


def test_health_and_docs():
    assert client.get("/api/health").json()["status"] == "ok"
    assert client.get("/docs").status_code == 200


def test_studio_is_served_from_the_same_app():
    r = client.get("/")
    assert r.status_code == 200 and "NeuroGraph-HW Studio" in r.text
    for f in ("app.js", "styles.css"):
        assert client.get(f"/static/{f}").status_code == 200


def test_presets():
    ps = client.get("/api/presets").json()
    assert [p["id"] for p in ps] == ["se3_spatial", "hippocampal_ring", "bci_decoder"]
    assert all(p["ir"].startswith("// NeuroGraph IR v1") and p["graph"]["nodes"] for p in ps)


def test_hardware_lists_defaults_and_their_provenance():
    h = client.get("/api/hardware").json()
    assert h["defaults"]["pim"]["tile_dim"] == 256 and h["defaults"]["neuromorphic"]["e_synop_pj"] == 23.6
    assert {v["status"] for v in h["sources"].values()} <= {"published", "derived", "assumed", "recalled"}


def test_parse_endpoint():
    text = print_ir(load_preset("hippocampal_ring"))
    r = client.post("/api/ir/parse", json={"ir_text": text})
    assert r.status_code == 200 and r.json()["ir"] == text
    bad = client.post("/api/ir/parse", json={"ir_text": "module @x {\n  junk\n}"})
    assert bad.status_code == 422 and "line 2" in bad.json()["detail"]


@pytest.mark.parametrize("source", ["preset", "ir_text", "graph"])
def test_compile_accepts_three_source_forms(source):
    g = load_preset("se3_spatial")
    body = {"preset": "se3_spatial"} if source == "preset" else {"ir_text": print_ir(g)} if source == "ir_text" else {"graph": g.to_dict()}
    r = client.post("/api/compile", json=body)
    assert r.status_code == 200
    d = r.json()
    assert {"ir", "graph", "mapping", "report", "layout"} <= set(d)
    assert "@PIMCrossbar" in d["ir"] and d["mapping"]["by_target"]["PIMCrossbar"] == ["predict"]
    assert all(n["target"] for n in d["graph"]["nodes"])


def test_compile_errors_are_reported_as_422_or_404():
    assert client.post("/api/compile", json={}).status_code == 422
    assert client.post("/api/compile", json={"preset": "se3_spatial", "ir_text": "x"}).status_code == 422
    assert client.post("/api/compile", json={"preset": "nope"}).status_code == 404
    r = client.post("/api/compile", json={"preset": "se3_spatial", "hardware": {"pim": {"qubits": 3}}})
    assert r.status_code == 422 and "qubits" in r.json()["detail"]
    r = client.post("/api/compile", json={"graph": {"nodes": [{"id": "a", "op": "ral.nope"}]}})
    assert r.status_code == 422 and "unknown op" in r.json()["detail"]


def test_fusion_flag_and_hardware_overrides_change_the_result():
    on = client.post("/api/compile", json={"preset": "se3_spatial"}).json()
    off = client.post("/api/compile", json={"preset": "se3_spatial", "fusion": False}).json()
    assert len(on["graph"]["nodes"]) < len(off["graph"]["nodes"])
    none = client.post("/api/compile", json={"preset": "se3_spatial", "hardware": {"pim": {"tiles": 0}}}).json()
    assert none["mapping"]["by_target"]["PIMCrossbar"] == []


def test_simulate_modes():
    het = client.post("/api/simulate", json={"preset": "hippocampal_ring", "steps": 40}).json()
    gpu = client.post("/api/simulate", json={"preset": "hippocampal_ring", "steps": 40, "mode": "gpu_baseline"}).json()
    assert het["mode"] == "heterogeneous" and gpu["mode"] == "gpu_baseline"
    assert het["totals"]["noc_flit_hops"] > 0 == gpu["totals"]["noc_flit_hops"]
    assert client.post("/api/simulate", json={"preset": "hippocampal_ring", "mode": "quantum"}).status_code == 422
    assert client.post("/api/simulate", json={"preset": "hippocampal_ring", "steps": 999999}).status_code == 422


def test_benchmark_compares_against_the_gpu_baseline():
    r = client.post("/api/benchmark", json={"preset": "hippocampal_ring", "steps": 60, "seed": 2})
    assert r.status_code == 200
    d = r.json()
    c = d["comparison"]
    assert c["energy_ratio_gpu_over_hetero"] > 1 and 0 <= c["memory_wall_savings_pct"] <= 100
    assert set(c["energy_mj"]) == {"heterogeneous", "gpu_baseline"}
    assert d["compiled"]["ir"] and "Not a measurement" in d["disclaimer"]


def test_oversized_simulations_are_refused_but_still_compile():
    ir = 'module @big {\n  %ring = ral.dyn.attractor {neurons = 4096} : () -> spikes<4096, 8>\n}'
    assert client.post("/api/compile", json={"ir_text": ir}).status_code == 200
    r = client.post("/api/benchmark", json={"ir_text": ir, "steps": 10})
    assert r.status_code == 422 and "at most" in r.json()["detail"]
