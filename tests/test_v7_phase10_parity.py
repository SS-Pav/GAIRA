"""GAIRA V7 — Phase 10: cross-surface parity and the static architecture rules.

The static tests here are the ones that keep Phase 10 honest over time. They parse each surface
with `ast` rather than grepping text, because Phase 09 learned that a module docstring listing
what it excludes will fail a naive substring search — and Phase 10's own freeze audit reproduced
the failure mode it was written to prevent within twenty minutes of starting.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from gaira.v7 import GAIRA
from gaira.v7.api import app
from gaira.v7.canonical import GAIRAEngine
from gaira.v7.contracts import (InferenceOptions, InferenceRequest, SpectrumInput,
                                SpectrumMetadata)
from gaira.v7.io import frozen_root, repo_root
from gaira.v7.mcp import call as mcp_call
from gaira.v7.runtime.service import GAIRAService

TOL = 1e-12
PHASE10 = frozen_root() / "phase10"

# Names that mean a module is doing science. `nnls` and `savgol_filter` are the projection and
# the smoother; `NMF` and `PCA` are decompositions; `cosine_similarity` is retrieval.
SCIENTIFIC_NAMES = {
    "nnls", "lsq_linear", "nmf", "pca", "truncatedsvd", "savgol_filter", "find_peaks",
    "gaussian_filter1d", "cosine_similarity", "pairwise_distances", "linear_model",
    "logisticregression", "kmeans", "umap", "spsolve", "cholesky", "lstsq", "svd", "eigh",
}
SCIENTIFIC_MODULES = {
    "scipy.optimize", "scipy.signal", "scipy.linalg", "scipy.ndimage", "sklearn",
    "sklearn.decomposition", "sklearn.linear_model", "sklearn.metrics", "sklearn.cluster",
    "gaira.v7.lsm", "gaira.v7.csm", "gaira.v7.chemistry", "gaira.v7.retrieval",
    "gaira.v7.atlas_decomposition", "gaira.v7.themes", "gaira.v7.programs", "gaira.v7.latent",
    "gaira.v7.geometry", "gaira.v7.meta", "gaira.v7.inference",
}

SURFACES = {
    "api": repo_root() / "src/gaira/v7/api",
    "mcp": repo_root() / "src/gaira/v7/mcp",
    "sdk": repo_root() / "src/gaira/v7/sdk",
    "cli": repo_root() / "src/gaira/v7/cli.py",
    "streamlit": repo_root() / "streamlit_apps/gaira_v7_console.py",
}


def _identifiers(path: Path) -> tuple[set[str], set[str]]:
    """(referenced names, imported modules), from the AST — docstrings and comments ignored."""
    tree = ast.parse(path.read_text())
    names, modules = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id.lower())
        elif isinstance(node, ast.Attribute):
            names.add(node.attr.lower())
        elif isinstance(node, ast.Import):
            for a in node.names:
                modules.add(a.name.lower()); names.add(a.name.split(".")[-1].lower())
        elif isinstance(node, ast.ImportFrom):
            mod = (node.module or "").lower()
            modules.add(mod)
            for a in node.names:
                names.add(a.name.lower()); modules.add(f"{mod}.{a.name}".lower())
    return names, modules


def _files(target: Path) -> list[Path]:
    return sorted(target.rglob("*.py")) if target.is_dir() else [target]


# ── static: the science lives in exactly one place ───────────────────────────
@pytest.mark.parametrize("surface", sorted(SURFACES))
def test_surface_imports_no_scientific_module(surface):
    for f in _files(SURFACES[surface]):
        names, modules = _identifiers(f)
        bad = {m for m in modules if any(m == s or m.startswith(s + ".")
                                         for s in SCIENTIFIC_MODULES)}
        assert not bad, f"{f.relative_to(repo_root())} imports {sorted(bad)}"


@pytest.mark.parametrize("surface", sorted(SURFACES))
def test_surface_references_no_scientific_primitive(surface):
    for f in _files(SURFACES[surface]):
        names, _ = _identifiers(f)
        bad = names & SCIENTIFIC_NAMES
        assert not bad, f"{f.relative_to(repo_root())} references {sorted(bad)}"


def test_streamlit_computes_nothing_scientific():
    """The UI may reshape and plot. It may not project, score, calibrate or aggregate."""
    f = SURFACES["streamlit"]
    names, modules = _identifiers(f)
    assert not (names & SCIENTIFIC_NAMES)
    assert "gaira.v7.canonical" not in modules, "the UI must go through the runtime, not the engine"
    src = f.read_text()
    for banned in ("def project", "def score", "def calibrate", "def preprocess",
                   "def retrieve"):
        assert banned not in src


def test_only_the_engine_defines_the_scientific_path():
    """One implementation of preprocessing, projection, retrieval and chemistry."""
    engine = repo_root() / "src/gaira/v7/canonical/engine.py"
    tree = ast.parse(engine.read_text())
    defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    for m in ("preprocess", "prepare", "project_lsm", "project_csm", "retrieve", "chemistry",
              "infer"):
        assert m in defined
    for surface in SURFACES.values():
        for f in _files(surface):
            src = f.read_text()
            for m in ("def preprocess", "def project_csm", "def project_lsm", "def retrieve("):
                assert m not in src, f"{f.name} redefines {m}"


def test_runtime_service_orchestrates_and_does_not_compute():
    """The service may translate and compare; it may not project, fit or calibrate."""
    f = repo_root() / "src/gaira/v7/runtime/service.py"
    names, modules = _identifiers(f)
    assert not (names & SCIENTIFIC_NAMES)
    assert not any(m.startswith("sklearn") or m.startswith("scipy") for m in modules)
    # It reaches the engine and nothing lower.
    assert "gaira.v7.canonical" in modules


def test_no_surface_reaches_the_filesystem_or_the_network():
    for surface in ("api", "mcp"):
        for f in _files(SURFACES[surface]):
            src = f.read_text()
            for banned in ("open(", "Path(", "os.system", "subprocess", "eval(", "exec(",
                           "pickle", "__import__"):
                assert banned not in src, f"{f.name} contains {banned}"


def test_no_llm_or_cloud_dependency_anywhere_in_phase_10():
    banned = ("openai", "anthropic", "google.generativeai", "vertexai", "google.cloud",
              "cohere", "transformers", "litellm", "langchain", "boto3")
    roots = [repo_root() / "src/gaira/v7" / p for p in
             ("api", "mcp", "sdk", "runtime", "contracts", "adapters", "validation",
              "reporting", "plugins")]
    roots += [repo_root() / "src/gaira/v7/cli.py", SURFACES["streamlit"]]
    for root in roots:
        for f in _files(root):
            _, modules = _identifiers(f)
            hits = {m for m in modules for b in banned if m == b or m.startswith(b + ".")}
            assert not hits, f"{f.name} imports {sorted(hits)}"


def test_interpretation_is_template_driven():
    f = repo_root() / "src/gaira/v7/runtime/interpret.py"
    _, modules = _identifiers(f)
    real = {m for m in modules if m and not m.startswith("__future__")}
    assert not real, f"interpretation imports {sorted(real)}; it must be pure templating"


# ── runtime parity ───────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def corpus():
    z = np.load(frozen_root() / "phase01/artifacts/balanced_references_v1.npz",
                allow_pickle=True)
    return np.asarray(z["X"], float), np.asarray(z["grid"], float)


@pytest.fixture(scope="module")
def surfaces():
    with TestClient(app) as http:
        yield {"engine": GAIRAEngine.load(), "service": GAIRAService.instance(),
               "sdk": GAIRA.shared(), "http": http}


@pytest.mark.parametrize("idx", [0, 37, 100, 210, 300, 374])
def test_six_surfaces_agree_exactly(surfaces, corpus, idx):
    X, g = corpus
    opts = {"already_preprocessed": True, "top_k_molecules": 10}

    a = surfaces["engine"].infer(X[idx], g, top_k=10, already_preprocessed=True)
    b = surfaces["service"].infer(InferenceRequest(
        spectrum=SpectrumInput(wavenumber=g.tolist(), intensity=X[idx].tolist()),
        metadata=SpectrumMetadata(), options=InferenceOptions(**opts)))
    c = surfaces["sdk"].infer(g.tolist(), X[idx].tolist(), None, opts)
    d = surfaces["http"].post("/v1/infer", json={
        "spectrum": {"wavenumber": g.tolist(), "intensity": X[idx].tolist()},
        "options": opts}).json()
    f = surfaces["sdk"].infer(g.tolist(), X[idx].tolist(), {"sample_type": "pure"},
                              {**opts, "include_reconstruction": True})

    assert b.result_digest == c.result_digest == d["result_digest"] == f.result_digest
    for got, want in zip(b.csm.activation, a.csm["activation"]):
        assert abs(got - float(want)) <= TOL
    for got, want in zip(b.chemistry.evidence, a.chemistry["evidence"]):
        assert abs(got - float(want)) <= TOL
    assert [h.molecule for h in b.retrieval.top] == [t["molecule"] for t in a.retrieval["top"]]
    assert abs(b.confidence.overall - a.confidence["overall"]) <= TOL


def test_mcp_agrees_with_the_sdk_on_the_same_path(surfaces, corpus):
    X, g = corpus
    spec = {"wavenumber": g.tolist(), "intensity": X[44].tolist()}
    m = mcp_call("gaira_infer_spectrum", {"spectrum": spec})
    s = surfaces["sdk"].infer(g.tolist(), X[44].tolist(), None, {"top_k_molecules": 10})
    assert m["result_digest"] == s.result_digest


def test_committed_parity_run_found_no_divergence():
    d = json.loads((PHASE10 / "artifacts/parity_and_performance_v1.json").read_text())
    assert d["parity"]["n_divergent"] == 0
    assert d["parity"]["max_abs_diff"] == 0.0
    assert len(d["parity"]["surfaces"]) == 6
    assert d["gates"]["failed"] == 0


def test_committed_run_reproduced_phase09_science():
    d = json.loads((PHASE10 / "artifacts/parity_and_performance_v1.json").read_text())
    sci = d["scientific_validation"]
    assert sci["max_deviation"] == 0.0
    assert abs(sci["measured"]["molecule_top1"] - 0.6053333333333333) < 1e-12
    assert abs(sci["measured"]["molecule_top5"] - 0.7946666666666666) < 1e-12
    assert abs(sci["measured"]["molecule_mrr"] - 0.6870030418103813) < 1e-12


def test_committed_freeze_audit_passed():
    d = json.loads((PHASE10 / "artifacts/engine_freeze_audit_v1.json").read_text())
    assert d["gates"]["failed"] == 0
    assert d["declared_fingerprints"]["match"]
    assert d["phase09_reproduction"]["max_deviation"] == 0.0
    assert d["ontology_order_match"] and d["deterministic"]


def test_performance_is_interactive():
    d = json.loads((PHASE10 / "artifacts/parity_and_performance_v1.json").read_text())
    p = d["performance"]
    assert p["single_inference_ms_median"] < 250
    assert p["api_overhead_ms"] < 100
    assert p["engine_load_seconds"] < 30


def test_local_inference_needs_no_external_volume():
    """Every asset the engine reads is committed inside the repository."""
    from gaira.v7.runtime.freeze import FROZEN_ASSETS
    root = frozen_root()
    for rel in FROZEN_ASSETS:
        p = (root / rel).resolve()
        assert p.exists()
        assert str(p).startswith(str(repo_root())), f"{rel} resolves outside the repository"
        assert "/Volumes/" not in str(p)
