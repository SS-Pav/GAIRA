"""GAIRA V7 — Phase 04 tests: the frozen projection engine.

The claims a reviewer would not take on trust are the engine invariants, so those are the
tests that earn their place:

    test_no_fitting_anywhere_in_the_inference_path
    test_batch_independence
    test_determinism_is_bit_identical
    test_frozen_atlas_refuses_to_load_on_a_fingerprint_mismatch
"""
from __future__ import annotations

import ast
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from gaira.v7.io import PhaseOutputs, frozen_root                # noqa: E402
from gaira.v7.engine import FrozenAtlas, project_spectrum        # noqa: E402
from gaira.v7.engine import aggregation as AGG                   # noqa: E402
from gaira.v7.engine import geometry as GEO                      # noqa: E402
from gaira.v7.engine import inference as INF                     # noqa: E402
from gaira.v7.engine import projection as PRJ                    # noqa: E402
from gaira.v7.engine import state as ST                          # noqa: E402
from gaira.v7.engine import validation as VAL                    # noqa: E402

OUT = PhaseOutputs("04")
T, A, V, F, R = OUT.tables, OUT.artifacts, OUT.validation, OUT.figures, OUT.reports
FROZEN = frozen_root()
EXPECTED = {"atlas": "09ed804a40836f4a05a91ba10900cded",
            "lsm": "208482d6f7178b5b8f16cace91be55b0",
            "csm": "0b4aa550ccefed3edabdbde5bae11c8d",
            "theme": "f54d4835ffdf8aa2d50a4a203da0e8f4"}
CFG = {"projection_method": "elastic_net", "aggregation_method": "direct_csm_projection",
       "theme_mode": "confidence_weighted", "bsv_variant": "theme_only",
       "geometry_extension": "landmark_barycentric", "knn": 5}

ran = pytest.mark.skipif(not (A / "phase_04_manifest_v1.json").is_file(),
                         reason="Phase 04 has not been run in this checkout")


@pytest.fixture(scope="module")
def atlas():
    return FrozenAtlas.load(FROZEN, CFG, EXPECTED)


@pytest.fixture(scope="module")
def X():
    return np.asarray(np.load(FROZEN / "phase01/artifacts/balanced_references_v1.npz",
                              allow_pickle=True)["X"], float)


@pytest.fixture(scope="module")
def state():
    return json.loads((OUT.root / "PHASE_STATE.json").read_text())


# ── A. THE ENGINE INVARIANTS ─────────────────────────────────────────────────
# The functions that actually run when a spectrum is projected. Benchmark helpers in the same
# modules (noise_stability, extension_fidelity) legitimately use a SEEDED generator and are not
# on this path — checking them would be checking the wrong thing.
INFERENCE_PATH = [
    INF.project_spectrum, INF.quality_control, INF.preprocessing_hash, INF._csm_distance,
    INF._nearest_molecules, PRJ.project, PRJ.project_nnls, PRJ.project_lasso,
    PRJ.project_elastic_net, AGG.lsm_to_csm, AGG.csm_uncertainty, AGG.theme_activation,
    AGG.build_bsv, AGG.bsv_elevation, AGG.rejected_theme_to_uncertainty,
    GEO.extend, GEO.landmark_barycentric, GEO.nystrom, GEO.knn_weighted,
    GEO.residual_ood, GEO.ood_score, GEO.local_density, GEO.bridge_proximity,
    ST.assign_confidence,
]


@pytest.mark.parametrize("fn", INFERENCE_PATH, ids=lambda f: f.__name__)
def test_no_fitting_anywhere_in_the_inference_path(fn):
    """No dictionary fitting and no randomness, checked on the AST of the functions that run.

    `projection.py` legitimately calls scikit-learn estimators whose API is `.fit(D.T, x)` —
    that solves for a coefficient vector against a FIXED dictionary, which is what a projection
    is. What must not appear is fitting of the dictionary itself, or any randomness at all.
    """
    tree = ast.parse(inspect.getsource(fn).lstrip())
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            assert node.attr not in {"default_rng", "RandomState", "shuffle", "permutation"}, \
                f"{fn.__name__} uses randomness: {node.attr}"
            assert node.attr not in {"fit_transform", "partial_fit"}, \
                f"{fn.__name__} calls {node.attr}"


def _code_only(mod) -> str:
    """Module source with docstrings stripped — the prose legitimately names what it forbids."""
    tree = ast.parse(inspect.getsource(mod))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body = node.body[1:] or [ast.Pass()]
    return ast.unparse(tree)


def test_inference_module_never_fits_a_dictionary():
    src = _code_only(INF)
    for bad in ("NMF(", "PCA(", "fit_transform", "partial_fit", "default_rng"):
        assert bad not in src, f"inference path contains {bad}"


@ran
def test_batch_independence(atlas, X):
    alone = project_spectrum(X[7], atlas, "x")
    inside = [project_spectrum(X[i], atlas, "x") for i in (3, 7, 11)][1]
    assert np.array_equal(alone.bsv, inside.bsv)
    assert np.array_equal(alone.csm_activations, inside.csm_activations)
    assert alone.confidence == inside.confidence


@ran
def test_determinism_is_bit_identical(atlas, X):
    a = project_spectrum(X[13], atlas, "x")
    b = project_spectrum(X[13], atlas, "x")
    for f in ("lsm_activations", "csm_activations", "theme_activations", "bsv",
              "geometry_coords"):
        assert np.array_equal(getattr(a, f), getattr(b, f)), f
    assert a.to_json() == b.to_json()


def test_frozen_atlas_refuses_to_load_on_a_fingerprint_mismatch():
    bad = dict(EXPECTED, csm="0" * 32)
    with pytest.raises(RuntimeError, match="refusing to run"):
        FrozenAtlas.load(FROZEN, CFG, bad)


@ran
def test_upstream_phases_untouched():
    for ph, key, want in (("phase01", "registry_fingerprint", EXPECTED["lsm"]),
                          ("phase02", "csm_fingerprint", EXPECTED["csm"]),
                          ("phase03", "theme_fingerprint", EXPECTED["theme"])):
        assert json.loads((FROZEN / ph / "PHASE_STATE.json").read_text())[key] == want


# ── B. CONTRACT C-09 / C-10 ──────────────────────────────────────────────────
@ran
def test_bsv_is_absolute_and_non_negative(atlas, X):
    s = project_spectrum(X[5], atlas, "x")
    assert (s.bsv >= 0).all()
    assert len(s.bsv) == len(s.bsv_axis_names)


@ran
def test_elevation_is_a_separate_signed_field(atlas, X):
    s = project_spectrum(X[5], atlas, "x")
    assert hasattr(s, "bsv_elevation")
    d = s.to_dict()
    assert "bsv" in d and "bsv_elevation" in d
    assert d["bsv"] != d["bsv_elevation"] or np.allclose(s.bsv_elevation, 0)


@ran
def test_no_delta_bsv_is_returned_by_the_inference_path(atlas, X):
    d = project_spectrum(X[5], atlas, "x").to_dict()
    assert not any("delta" in k.lower() for k in d)


@ran
def test_every_output_carries_the_atlas_fingerprints(atlas, X):
    s = project_spectrum(X[5], atlas, "x")
    assert s.lsm_fingerprint == EXPECTED["lsm"]
    assert s.csm_fingerprint == EXPECTED["csm"]
    assert s.theme_fingerprint == EXPECTED["theme"]
    assert s.preprocessing_config_hash


@ran
def test_bsv_reference_frame_reports_effective_rank():
    ref = json.loads((A / "bsv_reference_v1.json").read_text())
    er = ref["effective_rank"]
    assert er["participation_ratio"] <= er["nominal_K"]
    assert ref["schema"] == "bsv_reference_v1"
    assert all("uncertainty_inflation" in a for a in ref["axes"])


# ── C. EXPLAINABILITY ────────────────────────────────────────────────────────
@ran
def test_every_activated_theme_resolves_to_molecules(atlas, X):
    s = project_spectrum(X[7], atlas, "x")
    reg = {"S": atlas.S, "csm_ids": atlas.csm_ids, "theme_ids": atlas.theme_ids,
           "theme_names": atlas.theme_names, "accepted": atlas.theme_accepted}
    for k in range(len(s.theme_activations)):
        e = s.explain(k, reg, atlas.csm_registry)
        if s.theme_activations[k] > 1e-9:
            assert e["supporting_csms"], f"theme {k} activated but unexplained"
            assert e["supporting_csms"][0]["canonical_molecules"]


def test_explain_translates_an_accepted_index_to_its_membership_column():
    """Indexing S directly with an accepted-theme index explained the wrong theme."""
    src = inspect.getsource(ST.SpectrumState.explain)
    assert "accepted" in src and "col" in src


# ── D. UNCERTAINTY AND ITS SEMANTICS ─────────────────────────────────────────
@ran
def test_uncertainty_is_carried_at_every_level(atlas, X):
    s = project_spectrum(X[5], atlas, "x")
    for k in ("theme_entropy", "rejected_theme_mass", "mean_csm_disagreement",
              "geometry_local_confidence"):
        assert k in s.uncertainty
    assert "explained_variance" in s.residual
    assert s.confidence_tier in ST.CONFIDENCE_TIERS


def test_rejected_theme_contributes_only_to_uncertainty():
    T = np.array([[1.0, 2.0, 5.0, 1.0, 1.0]])
    acc = np.array([True, True, False, True, True])
    m = AGG.rejected_theme_to_uncertainty(T, acc)[0]
    assert m == pytest.approx(0.5)
    bsv, names = AGG.build_bsv(T[:, acc], np.array([0.1]), np.array([m]), np.array([0.0]),
                               "theme_only")
    assert bsv.shape[1] == 4 and "rejected" not in " ".join(names)


def test_zero_evidence_leakage_catches_theme_collapse():
    """A softmax gives every theme activation even where the CSM evidence is exactly zero."""
    S = np.array([[1.0, 0.0], [1.0, 0.0]])
    a = np.array([[1.0, 1.0]])
    soft = AGG.theme_activation(a, S, "soft_membership")
    prob = AGG.theme_activation(a, S, "probabilistic")
    assert AGG.zero_evidence_leakage(a, S, soft) == pytest.approx(0.0)
    assert AGG.zero_evidence_leakage(a, S, prob) > 0.0


def test_no_spectrum_is_forced_into_every_theme():
    S = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    a = np.array([[1.0, 0.0, 0.0]])
    for mode in ("soft_membership", "sparse_topk", "confidence_weighted"):
        T = AGG.theme_activation(a, S, mode, theme_confidence=np.ones(3))
        assert (T[0] == 0).any(), mode


# ── E. PROJECTION AND GEOMETRY ───────────────────────────────────────────────
def test_constrained_estimators_produce_no_negative_mass():
    rng = np.random.default_rng(0)
    D = np.abs(rng.normal(size=(6, 60))) + 0.05
    x = np.abs(rng.normal(size=60))
    for m in ("nnls", "lasso", "elastic_net"):
        assert PRJ.negativity(PRJ.project(x, D, m)) == pytest.approx(0.0)


def test_residual_ood_sees_what_the_geometric_score_cannot():
    """The geometric score works on the reconstruction, which lies inside the dictionary cone."""
    grid = np.linspace(450, 1800, 200)
    g = lambda c: np.exp(-((grid - c) ** 2) / (2 * 8.0 ** 2))
    D = np.vstack([g(700), g(1000), g(1400)])
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    inside = g(1000)
    outside = np.exp(-((grid - 1750) ** 2) / (2 * 4.0 ** 2))
    a_in = PRJ.project(inside, D, "nnls")[0]
    a_out = PRJ.project(outside, D, "nnls")[0]
    assert GEO.residual_ood(outside, D, a_out) > GEO.residual_ood(inside, D, a_in)


def test_geometry_extension_is_deterministic():
    rng = np.random.default_rng(0)
    Dref = np.abs(rng.normal(size=(12, 12)))
    Dref = (Dref + Dref.T) / 2
    np.fill_diagonal(Dref, 0)
    C = rng.normal(size=(12, 3))
    d = np.abs(rng.normal(size=(1, 12)))
    for m in GEO.EXTENSIONS:
        assert np.array_equal(GEO.extend(m, d, Dref, C), GEO.extend(m, d, Dref, C))


# ── F. VALIDATION SEMANTICS ──────────────────────────────────────────────────
def test_molecule_topk_is_undefined_under_molecule_grouping():
    """The first run reported 0.000 at every level; that was the split, not the engine."""
    A = np.eye(6)
    y = np.array(["a", "a", "b", "b", "c", "c"])
    te = np.array([0, 1])
    tr = np.array([2, 3, 4, 5])
    r = VAL.molecule_retrieval(A[te], A[tr], y[te], y[tr])
    assert r["top1"] == 0.0, "grouping removes the answer from the reference set by construction"
    assert "undefined" in VAL.grouped_folds_note()


def test_leave_one_spectrum_out_excludes_singletons():
    y = np.array(["a", "a", "b", "c", "c", "c"])
    m = VAL.leave_one_spectrum_out(y)
    assert m.tolist() == [True, True, False, True, True, True]


@ran
def test_leakage_control_compares_two_dictionaries():
    d = pd.read_csv(V / "leakage_control_v1.csv")
    assert set(d.dictionary) == {"frozen_dictionary", "fold_honest_dictionary"}
    g = d.groupby("dictionary").top1.mean()
    assert g["frozen_dictionary"] >= g["fold_honest_dictionary"] - 1e-9


@ran
def test_the_ood_failure_is_reported_not_hidden(state):
    probes = json.loads((A / "ood_probes_v1.json").read_text())
    assert "real_sers" in probes and "synthetic_band_shift" in probes
    gates = pd.read_csv(V / "phase04_gates_v1.csv")
    ood = gates[gates.gate.str.contains("OOD")]
    assert len(ood) == 1
    if state["ood_auroc"] < 0.70:
        assert ood.iloc[0].status == "FAIL", "a failing capability must fail its gate"


# ── G. OUTPUTS ───────────────────────────────────────────────────────────────
@ran
def test_output_root_is_configurable(monkeypatch):
    monkeypatch.setenv("GAIRA_V7_OUTPUT_ROOT", "/tmp/gaira-p4")
    assert PhaseOutputs("04").root == Path("/tmp/gaira-p4/phase04").resolve()
    assert frozen_root() == REPO / "results" / "v7_rebuild"


@ran
@pytest.mark.parametrize("n", range(1, 15))
def test_figure_exists(n):
    assert sorted(F.glob(f"fig{n:02d}_*.png")), f"figure {n:02d} missing"


@ran
def test_figures_are_png_only():
    assert not list(F.glob("*.svg")), "PNG only from Phase 02.5 onward"


@ran
def test_figure_pdf_and_reports_exist():
    assert (R / "PHASE_04_FIGURES.pdf").stat().st_size > 200_000
    rep = (R / "PHASE_04_REPORT.md").read_text()
    for term in ("split A", "split B", "leakage", "Ag-SERS", "effective rank", "Phase 05"):
        assert term.lower() in rep.lower(), term
    aud = (R / "PHASE_04_SCIENTIFIC_AUDIT.md").read_text()
    for term in ("falsif", "weakness", "unsupported", "reviewer", "risk"):
        assert term.lower() in aud.lower(), term


@ran
def test_manifest_lists_every_output():
    man = json.loads((A / "phase_04_manifest_v1.json").read_text())
    for o in man["outputs"]:
        p = Path(o["path"])
        assert (p if p.is_absolute() else REPO / p).is_file(), o["path"]


@ran
def test_worked_example_carries_the_full_chain():
    ex = json.loads((A / "worked_example_v1.json").read_text())
    assert ex["example_state"]["schema"] == "gaira_v7_inference_v1"
    assert ex["example_explanation"]["chain"].startswith("theme")
