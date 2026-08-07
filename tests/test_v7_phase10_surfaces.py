"""GAIRA V7 — Phase 10 tests: FastAPI, MCP, SDK, CLI and the plugin contracts."""
from __future__ import annotations

import base64
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from gaira.v7 import GAIRA
from gaira.v7.api import app
from gaira.v7.contracts import InferenceResult, Modality, SampleType
from gaira.v7.io import frozen_root
from gaira.v7.mcp import TOOL_NAMES, TOOLS, call as mcp_call
from gaira.v7.plugins import NotImplementedAdapter, context, modality


@pytest.fixture(scope="module")
def corpus():
    z = np.load(frozen_root() / "phase01/artifacts/balanced_references_v1.npz",
                allow_pickle=True)
    return np.asarray(z["X"], float), np.asarray(z["grid"], float)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def sdk():
    return GAIRA.shared()


def body(g, y, **opts):
    return {"spectrum": {"wavenumber": g.tolist(), "intensity": y.tolist()},
            "options": {"already_preprocessed": True, **opts}}


# ── API ──────────────────────────────────────────────────────────────────────
def test_startup_and_health(client):
    h = client.get("/v1/health").json()
    assert h["status"] == "ok" and h["engine_loaded"] and h["frozen_assets_verified"]


def test_engine_route_carries_fingerprints(client):
    e = client.get("/v1/engine").json()
    assert e["fingerprints"]["atlas"] == "09ed804a40836f4a05a91ba10900cded"
    assert e["fingerprints"]["csm"] == "0b4aa550ccefed3edabdbde5bae11c8d"
    assert e["n_csms"] == 49 and e["n_molecules"] == 154


def test_root_lists_the_routes(client):
    assert "/v1/infer" in client.get("/").json()["routes"]


def test_openapi_documents_every_route(client):
    paths = client.get("/openapi.json").json()["paths"]
    for r in ("/v1/health", "/v1/engine", "/v1/infer", "/v1/compare", "/v1/report",
              "/v1/validate-spectrum"):
        assert r in paths


def test_infer_and_determinism(client, corpus):
    X, g = corpus
    a = client.post("/v1/infer", json=body(g, X[0])).json()
    b = client.post("/v1/infer", json=body(g, X[0])).json()
    assert a["result_digest"] == b["result_digest"]
    assert a["chemistry"]["predicted_class"] == b["chemistry"]["predicted_class"]


def test_infer_matches_direct_python(client, sdk, corpus):
    X, g = corpus
    http = client.post("/v1/infer", json=body(g, X[17])).json()
    local = sdk.infer(g.tolist(), X[17].tolist(), None, {"already_preprocessed": True})
    assert http["result_digest"] == local.result_digest


@pytest.mark.parametrize("payload,status", [
    ({"spectrum": {"wavenumber": [1.0], "intensity": [1.0]}}, 422),
    ({"spectrum": {"wavenumber": [1.0, 2.0], "intensity": [1.0]}}, 422),
    ({"spectrum": {"wavenumber": [1.0, 2.0], "intensity": [1.0, 2.0]}}, 422),
    ({"nonsense": 1}, 422),
    ({}, 422),
])
def test_malformed_requests_are_rejected(client, payload, status):
    assert client.post("/v1/infer", json=payload).status_code == status


def test_unknown_field_is_rejected_not_ignored(client, corpus):
    X, g = corpus
    b = body(g, X[0]); b["surprise"] = True
    assert client.post("/v1/infer", json=b).status_code == 422


def test_unsupported_modality_is_blocked(client, corpus):
    X, g = corpus
    b = body(g, X[0]); b["metadata"] = {"modality": "ag_sers"}
    r = client.post("/v1/infer", json=b)
    assert r.status_code == 422
    assert r.json()["detail"]["code"] == "spectrum_rejected"
    codes = [d["code"] for d in r.json()["detail"]["validation"]["diagnostics"]]
    assert "scope.modality_unsupported" in codes


def test_unvalidated_sample_type_runs_and_warns(client, corpus):
    X, g = corpus
    b = body(g, X[0]); b["metadata"] = {"sample_type": "serum"}
    r = client.post("/v1/infer", json=b)
    assert r.status_code == 200
    assert any(d["code"] == "scope.sample_type_unvalidated" for d in r.json()["diagnostics"])


def test_validate_route(client, corpus):
    X, g = corpus
    r = client.post("/v1/validate-spectrum",
                    json={"spectrum": {"wavenumber": g.tolist(),
                                       "intensity": X[0].tolist()}}).json()
    assert r["can_run"] and r["n_points"] == 676


def test_compare_route(client, corpus):
    X, g = corpus
    r = client.post("/v1/compare", json={"a": body(g, X[0]), "b": body(g, X[100]),
                                         "label_a": "x", "label_b": "y"})
    assert r.status_code == 200
    d = r.json()
    assert len(d["chemistry_delta"]) == 16
    assert "biological state change" in d["scope_note"]


@pytest.mark.parametrize("fmt", ["json", "html", "pdf"])
def test_report_route(client, corpus, fmt):
    X, g = corpus
    res = client.post("/v1/infer", json=body(g, X[3], include_reconstruction=True)).json()
    r = client.post("/v1/report", json={"format": fmt, "inference": res})
    assert r.status_code == 200
    if fmt == "pdf":
        assert base64.b64decode(r.json()["content"])[:5] == b"%PDF-"
    elif fmt == "html":
        assert "<html" in r.text
    else:
        assert json.loads(r.text)["result"]["result_digest"] == res["result_digest"]


def test_report_route_needs_an_input(client):
    assert client.post("/v1/report", json={"format": "json"}).status_code == 400


def test_report_route_takes_no_filesystem_path(client):
    """A report endpoint that writes where the caller says is a file-write primitive."""
    schema = client.get("/openapi.json").json()["components"]["schemas"]["ReportRequest"]
    for field in schema["properties"]:
        assert field not in ("path", "output_path", "filename", "destination", "dest")


def test_oversized_body_is_refused(client):
    r = client.post("/v1/infer", json={"spectrum": {"wavenumber": [1.0], "intensity": [1.0]}},
                    headers={"content-length": str(64 * 1024 * 1024)})
    assert r.status_code in (413, 422)


def test_concurrent_requests_agree_with_serial(client, corpus):
    from concurrent.futures import ThreadPoolExecutor
    X, g = corpus
    payloads = [body(g, X[i]) for i in range(8)]
    serial = [client.post("/v1/infer", json=p).json()["result_digest"] for p in payloads]
    with ThreadPoolExecutor(max_workers=8) as ex:
        par = list(ex.map(lambda p: client.post("/v1/infer", json=p).json()["result_digest"],
                          payloads))
    assert serial == par


# ── MCP ──────────────────────────────────────────────────────────────────────
def test_eight_tools_declared():
    assert len(TOOLS) == 8
    assert set(TOOL_NAMES) == {
        "gaira_engine_info", "gaira_validate_spectrum", "gaira_infer_spectrum",
        "gaira_compare_spectra", "gaira_get_molecular_evidence",
        "gaira_get_chemistry_evidence", "gaira_explain_result", "gaira_generate_report"}


def test_every_tool_has_a_schema_and_a_scope_aware_description():
    for t in TOOLS:
        assert t["inputSchema"]["type"] == "object"
        assert len(t["description"]) > 80
    joined = " ".join(t["description"] for t in TOOLS)
    assert "RELATIVE" in joined and "analogues" in joined


def test_no_low_level_numerical_tool_is_exposed():
    """Coarse by design: an agent must not be able to assemble its own inference path."""
    for n in TOOL_NAMES:
        for banned in ("nnls", "project", "matrix", "raw_activation", "dictionary", "eval"):
            assert banned not in n


def test_tools_route_through_the_runtime(corpus, sdk):
    X, g = corpus
    spec = {"wavenumber": g.tolist(), "intensity": X[50].tolist()}
    r = mcp_call("gaira_infer_spectrum", {"spectrum": spec})
    local = sdk.infer(g.tolist(), X[50].tolist(), None, {})
    assert r["result_digest"] == local.result_digest


def test_narrow_tools_agree_with_the_full_one(corpus):
    X, g = corpus
    spec = {"wavenumber": g.tolist(), "intensity": X[60].tolist()}
    full = mcp_call("gaira_infer_spectrum", {"spectrum": spec})
    chem = mcp_call("gaira_get_chemistry_evidence", {"spectrum": spec})
    mol = mcp_call("gaira_get_molecular_evidence", {"spectrum": spec})
    exp = mcp_call("gaira_explain_result", {"spectrum": spec})
    assert chem["chemistry"] == full["chemistry"]
    assert mol["retrieval"] == full["retrieval"]
    assert exp["interpretation"] == full["interpretation"]
    assert {chem["result_digest"], mol["result_digest"], exp["result_digest"]} == \
        {full["result_digest"]}


def test_engine_info_tool_states_the_limitations():
    i = mcp_call("gaira_engine_info", {})
    assert i["supported_modalities"] == ["raman"]
    assert any("open-set" in lim for lim in i["known_limitations"])


def test_tool_accepts_inline_text_through_the_same_adapters():
    text = "\n".join(["wavenumber,intensity"] + [f"{450 + 2 * i},{abs(np.sin(i / 7)) + 0.1:.5f}"
                                                 for i in range(676)])
    r = mcp_call("gaira_validate_spectrum", {"spectrum": {"text": text}})
    assert r["can_run"]


def test_tool_rejects_bad_arguments():
    for args in ({}, {"spectrum": {}}, {"spectrum": {"wavenumber": "x"}}):
        with pytest.raises(ValueError):
            mcp_call("gaira_get_chemistry_evidence", args)
    with pytest.raises(ValueError):
        mcp_call("gaira_not_a_tool", {})


def test_tool_blocks_unsupported_modality(corpus):
    X, g = corpus
    r = mcp_call("gaira_validate_spectrum", {
        "spectrum": {"wavenumber": g.tolist(), "intensity": X[0].tolist()},
        "metadata": {"modality": "au_sers"}})
    assert not r["can_run"]


def test_mcp_report_refuses_pdf_with_a_pointer(corpus):
    X, g = corpus
    with pytest.raises(ValueError, match="json.*html|PDF"):
        mcp_call("gaira_generate_report", {
            "spectrum": {"wavenumber": g.tolist(), "intensity": X[0].tolist()},
            "format": "pdf"})


def test_mcp_server_module_imports_without_a_model_or_network():
    import gaira.v7.mcp.server as srv
    src = open(srv.__file__).read()
    for banned in ("openai", "anthropic", "google.gener", "vertexai", "requests.post",
                   "httpx.post", "urllib.request"):
        assert banned not in src


# ── SDK ──────────────────────────────────────────────────────────────────────
def test_sdk_repr_and_info(sdk):
    assert "atlas=" in repr(sdk)
    assert sdk.engine_info().n_molecules == 154


def test_sdk_read_and_infer_file(tmp_path, corpus):
    X, g = corpus
    p = tmp_path / "s.csv"
    p.write_text("wavenumber,intensity\n" + "\n".join(f"{w},{v}" for w, v in zip(g, X[5])))
    sdk = GAIRA.shared()
    x, y, diags = sdk.read(p)
    assert len(x) == 676
    r = sdk.infer_file(p)
    assert r.request_metadata.source_name == "s.csv"


def test_sdk_compare(sdk, corpus):
    X, g = corpus
    c = sdk.compare((g.tolist(), X[0].tolist()), (g.tolist(), X[80].tolist()))
    assert len(c.chemistry_delta) == 16


# ── CLI ──────────────────────────────────────────────────────────────────────
def test_cli_info_json(capsys):
    from gaira.v7.cli import main
    assert main(["info", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["n_csms"] == 49


def test_cli_validate_and_infer(tmp_path, capsys, corpus):
    from gaira.v7.cli import main
    X, g = corpus
    p = tmp_path / "s.csv"
    p.write_text("wavenumber,intensity\n" + "\n".join(f"{w},{v}" for w, v in zip(g, X[6])))
    assert main(["validate", str(p)]) == 0
    assert main(["infer", str(p)]) == 0
    out = capsys.readouterr().out
    assert "Chemistry Evidence" in out and "reference analogues" in out


def test_cli_rejects_an_unsupported_modality(tmp_path, corpus):
    from gaira.v7.cli import main
    X, g = corpus
    p = tmp_path / "s.csv"
    p.write_text("\n".join(f"{w},{v}" for w, v in zip(g, X[6])))
    assert main(["validate", str(p), "--modality", "ag_sers"]) == 2
    assert main(["infer", str(p), "--modality", "ag_sers"]) == 2


def test_cli_writes_a_report(tmp_path, corpus):
    from gaira.v7.cli import main
    X, g = corpus
    p = tmp_path / "s.csv"
    p.write_text("\n".join(f"{w},{v}" for w, v in zip(g, X[6])))
    out = tmp_path / "r.pdf"
    assert main(["infer", str(p), "--report", str(out)]) == 0
    assert out.read_bytes()[:5] == b"%PDF-"


# ── plugin contracts ─────────────────────────────────────────────────────────
def test_raman_adapter_is_the_only_implemented_modality():
    assert modality.get(Modality.RAMAN).implemented
    for m in (Modality.AG_SERS, Modality.AU_SERS, Modality.SERS, Modality.DART):
        assert not modality.get(m).implemented


def test_pure_context_is_the_only_implemented_context():
    assert context.get(SampleType.PURE).implemented
    for s in (SampleType.MIXTURE, SampleType.SERUM, SampleType.PLASMA, SampleType.EV,
              SampleType.BACTERIA, SampleType.TISSUE):
        assert not context.get(s).implemented


def test_no_stub_performs_fake_inference():
    """A stub returning plausible numbers is worse than no stub. Every one must raise."""
    x, y = np.linspace(450, 1800, 676), np.ones(676)
    for m in (Modality.AG_SERS, Modality.AU_SERS, Modality.SERS, Modality.DART):
        with pytest.raises(NotImplementedAdapter):
            modality.get(m).admit(x, y, {})
    for s in (SampleType.MIXTURE, SampleType.SERUM, SampleType.PLASMA, SampleType.EV,
              SampleType.BACTERIA, SampleType.TISSUE):
        with pytest.raises(NotImplementedAdapter):
            context.get(s).frame(None)


def test_unimplemented_adapters_state_what_they_need():
    for m in (Modality.AG_SERS, Modality.DART):
        with pytest.raises(NotImplementedAdapter) as e:
            modality.get(m).admit(np.ones(10), np.ones(10), {})
        assert len(str(e.value)) > 150 and "must first supply" in str(e.value)


def test_raman_adapter_passes_through_unchanged():
    x, y = np.linspace(450, 1800, 676), np.abs(np.sin(np.arange(676) / 9))
    d = modality.get(Modality.RAMAN).admit(x, y, {})
    assert d.admissible and d.transfer_applied == "none"
    assert np.array_equal(d.intensity, y) and np.array_equal(d.wavenumber, x)


def test_adapters_satisfy_their_protocols():
    from gaira.v7.plugins import ModalityAdapter, SampleContextAdapter
    assert isinstance(modality.get(Modality.RAMAN), ModalityAdapter)
    assert isinstance(context.get(SampleType.PURE), SampleContextAdapter)
    assert isinstance(modality.get(Modality.AG_SERS), ModalityAdapter)


def test_context_adapter_has_no_way_to_change_a_number():
    """The protocol returns framing only. If it could return evidence, it could rewrite it."""
    from gaira.v7.plugins.protocols import ContextFraming
    fields = set(ContextFraming.__dataclass_fields__)
    assert fields == {"caveats", "diagnostics", "framing", "evidence_weighting"}
    f = context.get(SampleType.PURE).frame(None)
    assert f.evidence_weighting is None


# ── security and file handling (Step 15) ─────────────────────────────────────
def test_upload_size_is_bounded():
    from gaira.v7.adapters.text import MAX_BYTES
    from gaira.v7.api.dependencies import MAX_UPLOAD_BYTES
    from gaira.v7.contracts import MAX_POINTS
    assert MAX_BYTES <= 64 * 1024 * 1024
    assert MAX_UPLOAD_BYTES <= 64 * 1024 * 1024
    assert MAX_POINTS <= 1_000_000


def test_a_huge_spectrum_is_refused_by_the_contract():
    from gaira.v7.contracts import SpectrumInput
    with pytest.raises(Exception):
        SpectrumInput(wavenumber=[0.0] * 300_000, intensity=[0.0] * 300_000)


def test_csv_headers_are_not_trusted():
    """A header naming a column 'intensity' must not override a non-numeric body."""
    from gaira.v7.adapters import load as load_spectrum
    p = load_spectrum(b"wavenumber,intensity\nnot,numbers\nalso,bad\nstill,bad\n", "e.csv")
    assert not p.ok


def test_no_pickle_or_eval_in_any_phase_10_module():
    from gaira.v7.io import repo_root
    roots = [repo_root() / "src/gaira/v7" / p for p in
             ("api", "mcp", "sdk", "runtime", "contracts", "adapters", "validation",
              "reporting", "plugins")]
    roots.append(repo_root() / "src/gaira/v7/cli.py")
    for root in roots:
        files = sorted(root.rglob("*.py")) if root.is_dir() else [root]
        for f in files:
            src = f.read_text()
            for banned in ("pickle.load", "eval(", "exec(", "os.system", "yaml.load("):
                assert banned not in src, f"{f.name} contains {banned}"


def test_np_load_never_trusts_user_data():
    """allow_pickle is used only on frozen repository artefacts, never on an upload."""
    from gaira.v7.io import repo_root
    for f in sorted((repo_root() / "src/gaira/v7/adapters").rglob("*.py")):
        assert "np.load" not in f.read_text()


def test_mcp_and_api_expose_no_filesystem_path(client):
    schemas = client.get("/openapi.json").json()["components"]["schemas"]
    for name, schema in schemas.items():
        for field in schema.get("properties", {}):
            assert field not in ("path", "file", "filepath", "output_path", "directory")
    for t in TOOLS:
        for field in t["inputSchema"].get("properties", {}):
            assert field not in ("path", "file", "filepath", "output_path", "directory")


def test_report_filename_is_derived_not_supplied(client, corpus):
    X, g = corpus
    res = client.post("/v1/infer", json=body(g, X[0])).json()
    r = client.post("/v1/report", json={"format": "pdf", "inference": res}).json()
    assert r["filename"] == f"gaira_v7_report_{res['result_digest'][:12]}.pdf"
    assert "/" not in r["filename"] and ".." not in r["filename"]


def test_a_title_cannot_inject_markup(client, corpus):
    X, g = corpus
    res = client.post("/v1/infer", json=body(g, X[0])).json()
    r = client.post("/v1/report", json={"format": "html", "inference": res,
                                        "title": "<script>alert(1)</script>"})
    assert "<script>alert(1)</script>" not in r.text
    assert "&lt;script&gt;" in r.text


def test_metadata_strings_are_length_bounded():
    from gaira.v7.contracts import SpectrumMetadata
    with pytest.raises(Exception):
        SpectrumMetadata(sample_id="x" * 500)
    with pytest.raises(Exception):
        SpectrumMetadata(notes="x" * 10_000)
