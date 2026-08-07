"""GAIRA V7 — Phase 11 tests: the interactive demo changes no science.

Phase 11 is presentation. These tests exist to prove that, and to keep it true.

The static checks parse each demo module with `ast` rather than grepping it — a docstring that
lists what a module excludes will fail a substring search, which Phase 09 learned and Phase 10
inherited.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from gaira.v7 import GAIRA
from gaira.v7.io import frozen_root, repo_root

APP = repo_root() / "streamlit_apps/gaira_v7_demo.py"
PKG = repo_root() / "streamlit_apps/gaira_v7_demo"
PHASE11 = frozen_root() / "phase11"
HAS_RUN = (PHASE11 / "artifacts/phase11_validation_v1.json").exists()
needs_run = pytest.mark.skipif(not HAS_RUN, reason="Phase 11 validation has not been run")

# Names that mean a module is doing science rather than drawing it.
SCIENTIFIC_NAMES = {
    "nnls", "lsq_linear", "nmf", "pca", "truncatedsvd", "savgol_filter", "find_peaks",
    "gaussian_filter1d", "cosine_similarity", "pairwise_distances", "logisticregression",
    "kmeans", "umap", "spsolve", "cholesky", "lstsq", "svd", "eigh", "softmax",
}
SCIENTIFIC_MODULES = {
    "scipy", "scipy.optimize", "scipy.signal", "scipy.linalg", "scipy.ndimage",
    "sklearn", "sklearn.decomposition", "sklearn.linear_model", "sklearn.metrics",
    "gaira.v7.lsm", "gaira.v7.csm", "gaira.v7.chemistry", "gaira.v7.retrieval",
    "gaira.v7.atlas_decomposition", "gaira.v7.themes", "gaira.v7.programs",
    "gaira.v7.latent", "gaira.v7.geometry", "gaira.v7.meta", "gaira.v7.inference",
    "gaira.v7.canonical",           # the demo must go through the runtime, not past it
}


def _identifiers(path: Path) -> tuple[set[str], set[str]]:
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


def _files() -> list[Path]:
    return [APP] + sorted(PKG.glob("*.py"))


# ── static: the demo computes nothing ────────────────────────────────────────
@pytest.mark.parametrize("path", _files(), ids=lambda p: p.name)
def test_demo_module_references_no_scientific_primitive(path):
    names, _ = _identifiers(path)
    bad = names & SCIENTIFIC_NAMES
    assert not bad, f"{path.name} references {sorted(bad)}"


@pytest.mark.parametrize("path", _files(), ids=lambda p: p.name)
def test_demo_module_imports_no_scientific_module(path):
    _, modules = _identifiers(path)
    bad = {m for m in modules
           if any(m == s or m.startswith(s + ".") for s in SCIENTIFIC_MODULES)}
    # data.py legitimately reads the frozen dictionaries for DISPLAY, via the freeze ledger.
    if path.name == "data.py":
        bad -= {"gaira.v7.runtime", "gaira.v7.runtime.freeze"}
    assert not bad, f"{path.name} imports {sorted(bad)}"


def test_demo_defines_no_scientific_function():
    """Exact function names from the AST, not substrings.

    A substring check for "def preprocess" flags `preprocessing_stages`, which is a FIGURE. The
    same over-broad matching has produced a false positive in three phases now; matching the
    parsed name is the fix each time.
    """
    banned = {"preprocess", "prepare", "project_csm", "project_lsm", "project", "retrieve",
              "chemistry", "calibrate", "score", "similarity", "infer"}
    for f in _files():
        tree = ast.parse(f.read_text())
        defined = {n.name for n in ast.walk(tree)
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        clash = defined & banned
        assert not clash, f"{f.name} defines {sorted(clash)}"


def test_demo_makes_exactly_one_inference_call():
    """All science enters through GAIRA.infer. More than one entry point is more than one
    opportunity to drift."""
    src = APP.read_text()
    assert src.count("client().infer(") == 1
    assert "GAIRAEngine" not in src


def test_demo_has_no_llm_or_cloud_dependency():
    banned = ("openai", "anthropic", "google.generativeai", "vertexai", "google.cloud",
              "cohere", "transformers", "litellm", "langchain", "boto3", "requests")
    for f in _files():
        _, modules = _identifiers(f)
        hits = {m for m in modules for b in banned if m == b or m.startswith(b + ".")}
        assert not hits, f"{f.name} imports {sorted(hits)}"


def test_demo_uses_plotly_and_not_matplotlib():
    for f in _files():
        _, modules = _identifiers(f)
        assert not any(m.startswith("matplotlib") for m in modules), f.name
    _, modules = _identifiers(PKG / "figures.py")
    assert any(m.startswith("plotly") for m in modules)


def test_theme_module_is_pure_presentation():
    _, modules = _identifiers(PKG / "theme.py")
    real = {m for m in modules if m and not m.startswith("__future__")}
    assert not real, f"theme.py imports {sorted(real)}"


def test_display_data_is_verified_against_the_freeze_ledger():
    """The motif spectra the demo draws must provably be the ones the engine used."""
    src = (PKG / "data.py").read_text()
    assert "FREEZE.verify(strict=True)" in src


def test_scope_language_is_present_and_overclaiming_language_is_not():
    src = APP.read_text().lower()
    for required in ("reference analogue", "not a concentration", "open-set",
                     "relative"):
        assert required in src, f"missing required caveat: {required}"
    for banned in ("ai identifies your molecule", "detects disease", "clinical diagnosis",
                   "diagnose", "concentration of the"):
        assert banned not in src, f"overclaiming language: {banned}"


def test_unsupported_modality_is_blocked_in_the_ui():
    src = APP.read_text()
    assert 'if modality != "raman":' in src
    assert "Analysis is blocked" in src


# ── the demo path returns exactly what every other surface returns ───────────
@pytest.fixture(scope="module")
def corpus():
    z = np.load(frozen_root() / "phase01/artifacts/balanced_references_v1.npz",
                allow_pickle=True)
    return (np.asarray(z["X"], float), np.asarray(z["grid"], float),
            [str(s) for s in z["canonical_id"]])


DEMO_OPTIONS = {"include_reconstruction": True, "top_k_molecules": 10,
                "already_preprocessed": True}


def test_demo_options_match_the_app():
    """If the app changes what it asks for, this test must change with it."""
    src = APP.read_text()
    assert '"include_reconstruction": True, "top_k_molecules": 10' in src


@pytest.mark.parametrize("idx", [0, 40, 120, 300])
def test_demo_call_equals_plain_call(corpus, idx):
    X, grid, _ = corpus
    g = GAIRA.shared()
    demo = g.infer(grid.tolist(), X[idx].tolist(),
                   {"modality": "raman", "sample_type": "pure"}, DEMO_OPTIONS)
    plain = g.infer(grid.tolist(), X[idx].tolist(), None,
                    {"already_preprocessed": True, "top_k_molecules": 10})
    assert demo.result_digest == plain.result_digest
    assert demo.chemistry.evidence == plain.chemistry.evidence
    assert [h.molecule for h in demo.retrieval.top] == \
           [h.molecule for h in plain.retrieval.top]


def test_reconstruction_option_adds_display_data_only(corpus):
    X, grid, _ = corpus
    g = GAIRA.shared()
    with_recon = g.infer(grid.tolist(), X[5].tolist(), None, DEMO_OPTIONS)
    without = g.infer(grid.tolist(), X[5].tolist(), None,
                      {"already_preprocessed": True, "top_k_molecules": 10})
    assert with_recon.result_digest == without.result_digest
    assert with_recon.csm.reconstruction is not None
    assert without.csm.reconstruction is None


# ── display-data integrity ───────────────────────────────────────────────────
def test_reference_motifs_have_the_frozen_shapes():
    import sys
    sys.path.insert(0, str(repo_root()))
    from streamlit_apps.gaira_v7_demo.data import load_reference_motifs
    m = load_reference_motifs()
    assert m["H_lsm"].shape == (50, 676)
    assert m["CSM"].shape == (49, 676)
    assert len(m["csm_ids"]) == 49 and len(m["lsm_ids"]) == 50
    assert len(m["csm_records"]) == 49
    assert m["grid"].shape == (676,)


def test_every_demo_example_exists_in_the_corpus():
    import sys
    sys.path.insert(0, str(repo_root()))
    from streamlit_apps.gaira_v7_demo.data import DEMO_SPECTRA, load_reference_spectra
    refs = load_reference_spectra()
    assert len(refs) == 154
    missing = [m for m in DEMO_SPECTRA.values() if m and m not in refs]
    assert not missing, f"demo offers molecules that are not in the corpus: {missing}"


def test_figures_build_from_a_result_without_touching_the_engine(corpus):
    import sys
    sys.path.insert(0, str(repo_root()))
    from streamlit_apps.gaira_v7_demo import figures as FIG
    X, grid, _ = corpus
    r = GAIRA.shared().infer(grid.tolist(), X[9].tolist(), None, DEMO_OPTIONS)
    d = r.model_dump(mode="json")
    for fig in (FIG.raw_spectrum(grid, X[9]),
                FIG.processed_spectrum(d["preprocessing"]["grid"],
                                       d["preprocessing"]["processed_intensity"]),
                FIG.chemistry_radar(d["chemistry"]), FIG.chemistry_bars(d["chemistry"]),
                FIG.chemistry_polar_bars(d["chemistry"]),
                FIG.retrieval_bars(d["retrieval"]["top"]),
                FIG.csm_contribution_waterfall(d["retrieval"]["top"][0]),
                FIG.confidence_gauge(d["confidence"]),
                FIG.confidence_factors(d["confidence"]),
                FIG.reconstruction(d["preprocessing"]["grid"],
                                   d["preprocessing"]["processed_intensity"],
                                   d["csm"]["reconstruction"]),
                FIG.provenance_sankey(d["provenance"], d["chemistry"]),
                FIG.architecture_flow(2)):
        assert fig.data or fig.layout.shapes, "figure produced nothing"


def test_every_figure_title_avoids_the_undefined_bug():
    """A Plotly title dict without `text` renders the literal string 'undefined'."""
    import sys
    sys.path.insert(0, str(repo_root()))
    from streamlit_apps.gaira_v7_demo import theme as T
    assert T.plotly_layout()["title"]["text"] == ""


def test_all_sixteen_axes_have_a_plain_english_description():
    import sys
    sys.path.insert(0, str(repo_root()))
    from gaira.v7.chemistry.registry import CLASS_ORDER
    from streamlit_apps.gaira_v7_demo import figures as FIG
    for axis in CLASS_ORDER:
        assert FIG.AXIS_NOTE.get(axis), f"no description for {axis}"


def test_architecture_stages_match_the_engine():
    import sys
    sys.path.insert(0, str(repo_root()))
    from streamlit_apps.gaira_v7_demo import figures as FIG
    names = " ".join(n for n, _, _ in FIG.ARCH_STAGES)
    assert "50 Local Spectral Motifs" in names
    assert "49 Consensus Spectral Motifs" in names
    assert len(FIG.ARCH_STAGES) == 7


# ── committed validation artifacts ───────────────────────────────────────────
@needs_run
def test_committed_validation_found_no_divergence():
    d = json.loads((PHASE11 / "artifacts/phase11_validation_v1.json").read_text())
    assert d["parity"]["n_divergent"] == 0
    assert d["parity"]["max_abs_diff"] == 0.0
    assert len(d["parity"]["surfaces"]) == 7
    assert d["gates"]["failed"] == 0
    assert all(x["all_surfaces_equal"] for x in d["digests"])
    assert d["cli"]["ok"]


@needs_run
def test_committed_validation_used_the_apps_real_options():
    d = json.loads((PHASE11 / "artifacts/phase11_validation_v1.json").read_text())
    assert d["demo_options"] == DEMO_OPTIONS


@needs_run
def test_demo_analysis_is_interactive():
    d = json.loads((PHASE11 / "artifacts/phase11_validation_v1.json").read_text())
    assert d["performance"]["demo_inference_ms_median"] < 200
    assert d["inference_plus_figures_ms"] < 1000


@needs_run
def test_screenshot_gallery_is_committed():
    shots = sorted((PHASE11 / "gallery").glob("*.png"))
    assert len(shots) >= 12, f"only {len(shots)} screenshots"
    assert all(p.stat().st_size > 20_000 for p in shots)
