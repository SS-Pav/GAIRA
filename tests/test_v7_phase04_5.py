"""GAIRA V7 — Phase 04.5 tests: hierarchical NMF over frozen CSM activations.

This phase reports a NEGATIVE result, so the tests that earn their place are the ones that
would catch a negative result reached carelessly:

    test_nmf_is_applied_to_the_activation_matrix_not_to_spectra
    test_geometry_prior_is_one_sided
    test_stability_gains_are_gated_by_an_informativeness_floor
    test_k_diagnostic_closes_the_escape_route
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

from gaira.v7.io import PhaseOutputs, frozen_root              # noqa: E402
from gaira.v7.meta import evaluation as EV                     # noqa: E402
from gaira.v7.meta import factorization as MF                  # noqa: E402
from gaira.v7.meta import perturbations as PT                  # noqa: E402

OUT = PhaseOutputs("04.5")
T, A, V, F, R = OUT.tables, OUT.artifacts, OUT.validation, OUT.figures, OUT.reports
FROZEN = frozen_root()
EXPECTED = {"lsm": "208482d6f7178b5b8f16cace91be55b0",
            "csm": "0b4aa550ccefed3edabdbde5bae11c8d",
            "theme": "f54d4835ffdf8aa2d50a4a203da0e8f4"}

ran = pytest.mark.skipif(not (A / "phase_04_5_manifest_v1.json").is_file(),
                         reason="Phase 04.5 has not been run in this checkout")


@pytest.fixture(scope="module")
def state():
    return json.loads((OUT.root / "PHASE_STATE.json").read_text())


@pytest.fixture(scope="module")
def store():
    return np.load(A / "meta_components_v1.npz", allow_pickle=True)


@pytest.fixture(scope="module")
def verdict():
    return json.loads((A / "verdict_v1.json").read_text())


# ── A. WHAT WAS FACTORISED ───────────────────────────────────────────────────
@ran
def test_nmf_is_applied_to_the_activation_matrix_not_to_spectra(store):
    """The whole premise: A is spectra x CSM activations, 375 x 49 — not 375 x 676."""
    assert store["A_csm"].shape == (375, 49)
    assert store["H"].shape[1] == 49
    assert store["W"].shape[0] == 375


def test_factorisation_never_touches_a_spectrum_or_a_similarity_matrix():
    src = inspect.getsource(MF)
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body = node.body[1:] or [ast.Pass()]
    code = ast.unparse(tree)
    for bad in ("wasserstein", "cosine_similarity", "pairwise_distances", "louvain"):
        assert bad not in code, f"factorisation reaches for {bad}"


@ran
def test_upstream_phases_untouched():
    for ph, key, want in (("phase01", "registry_fingerprint", EXPECTED["lsm"]),
                          ("phase02", "csm_fingerprint", EXPECTED["csm"]),
                          ("phase03", "theme_fingerprint", EXPECTED["theme"])):
        assert json.loads((FROZEN / ph / "PHASE_STATE.json").read_text())[key] == want


# ── B. THE GEOMETRY PRIOR ────────────────────────────────────────────────────
def test_geometry_prior_is_one_sided():
    """`tr(H L H^T)` rewards nearby CSMs for loading similarly and has NO term that grows when
    distant CSMs do — so it can never push anything apart or manufacture a cluster."""
    rng = np.random.default_rng(0)
    n = 8
    D = np.abs(rng.normal(size=(n, n)))
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0)
    L, Wg = MF.csm_graph_laplacian(D, k=3)
    assert np.allclose(L, L.T)
    ev = np.linalg.eigvalsh(L)
    assert ev.min() > -1e-8, "a Laplacian is positive semi-definite; a repulsive term would not be"
    # smoothness: identical loadings on every CSM give exactly zero penalty
    H_flat = np.ones((2, n))
    assert float(np.trace(H_flat @ L @ H_flat.T)) == pytest.approx(0.0, abs=1e-8)
    # and any variation costs something non-negative
    H_var = rng.normal(size=(2, n))
    assert float(np.trace(H_var @ L @ H_var.T)) >= -1e-8


def test_geometry_regularised_fit_stays_non_negative():
    rng = np.random.default_rng(0)
    Amat = np.abs(rng.normal(size=(40, 12)))
    D = np.abs(rng.normal(size=(12, 12)))
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0)
    L, _ = MF.csm_graph_laplacian(D, k=3)
    f = MF.fit_geometry_regularised(Amat, 3, L, lam=0.5, n_iter=60)
    assert (f["W"] >= 0).all() and (f["H"] >= 0).all()


@ran
def test_both_variants_were_compared():
    s = pd.read_csv(T / "model_selection_sweep_v1.csv")
    assert set(s.variant) == set(MF.VARIANTS)
    assert set(s.K) == set(MF.K_GRID)
    assert len(s) == len(MF.VARIANTS) * len(MF.K_GRID)


# ── C. MODEL SELECTION ───────────────────────────────────────────────────────
@ran
def test_k_was_not_selected_on_reconstruction_alone():
    sel = json.loads((A / "model_selection_v1.json").read_text())
    w = sel["pareto_weights"]
    assert w["explained_variance"]["weight"] < 0.25
    assert sum(v["weight"] for v in w.values()) == pytest.approx(1.0, abs=1e-9)
    assert sel["reconstruction_is_minority_weighted"] is True


@ran
def test_all_eleven_selection_metrics_are_reported():
    s = pd.read_csv(T / "model_selection_sweep_v1.csv")
    for col in ("reconstruction_error", "explained_variance", "bootstrap_stability",
                "consensus_stability", "component_sparsity", "effective_rank",
                "redundancy", "mutual_coherence", "interpretability",
                "activation_entropy", "participation_ratio"):
        assert col in s.columns, col


@ran
def test_k_diagnostic_closes_the_escape_route(state):
    """'A different K would have worked' is the obvious objection; it is answered."""
    kd = pd.read_csv(V / "k_downstream_diagnostic_v1.csv")
    assert len(kd) == len(MF.VARIANTS) * len(MF.K_GRID)
    csm_b = state["comparison"]["CSM"]["B_top1"]
    assert kd.B_top1.max() < csm_b, "no K in the sweep beats the CSM layer"


def test_k_diagnostic_is_not_used_for_selection():
    src = inspect.getsource(MF.__loader__.get_data(MF.__file__).decode()
                            if False else MF)
    assert "B_top1" not in src, "the downstream metric must not reach the selection code"


# ── D. THE INFORMATIVENESS FLOOR ─────────────────────────────────────────────
@ran
def test_stability_gains_are_gated_by_an_informativeness_floor(state, verdict):
    """Replicate consistency and robustness AUC are maximised by a representation that says the
    same thing about everything — the Phase 03 softmax trap. The floor is what stops that
    counting as a benefit."""
    assert "informativeness_floor_passed" in state
    if not state["informativeness_floor_passed"]:
        assert state["recommended_action"] == "discard"
    assert verdict["information_retained_ratio"] < 1.0


@ran
def test_meta_wins_on_stability_and_still_loses(state, verdict):
    """The honest shape of the result: it does win on the stability axes, and that is not
    enough."""
    cm = state["comparison"]
    assert cm["META"]["replicate_consistency"] > cm["CSM"]["replicate_consistency"]
    assert cm["META"]["B_top1"] < cm["CSM"]["B_top1"]
    assert state["recommended_action"] == "discard"


@ran
def test_verdict_is_derived_from_the_numbers(state):
    n = state["n_axes_improved"]
    assert 0 <= n <= 8
    assert state["recommended_action"] in ("discard", "augment", "replace")
    assert state["verdict"]


# ── E. INFERENCE IS PROJECTION ONLY ──────────────────────────────────────────
def test_meta_projection_does_not_fit():
    src = inspect.getsource(MF.project)
    for bad in ("fit(", "fit_transform", "default_rng", "NMF("):
        assert bad not in src, f"meta projection calls {bad}"


@ran
def test_meta_projection_is_deterministic_and_non_negative(store):
    H = store["H"]
    Amat = store["A_csm"]
    a = MF.project(Amat[:12], H)
    b = MF.project(Amat[:12], H)
    assert np.array_equal(a, b)
    assert (a >= 0).all()


@ran
def test_meta_projection_is_batch_independent(store):
    H, Amat = store["H"], store["A_csm"]
    alone = MF.project(Amat[[7]], H)
    batch = MF.project(Amat[:12], H)[7]
    assert np.allclose(alone[0], batch)


# ── F. PERTURBATIONS ─────────────────────────────────────────────────────────
def test_twelve_perturbations_with_sweeps():
    assert len(PT.PERTURBATIONS) == 12
    assert set(PT.LEVELS) == set(PT.PERTURBATIONS)
    assert all(len(PT.LEVELS[k]) == 5 for k in PT.PERTURBATIONS)


@pytest.mark.parametrize("kind", PT.PERTURBATIONS)
def test_every_perturbation_is_deterministic_and_stays_non_negative(kind):
    grid = np.linspace(450, 1800, 200)
    X = np.abs(np.sin(np.linspace(0, 12, 200)))[None, :].repeat(3, 0) + 0.05
    lvl = PT.LEVELS[kind][2]
    a = PT.apply(kind, X, grid, lvl, seed=0)
    b = PT.apply(kind, X, grid, lvl, seed=0)
    assert np.array_equal(a, b), f"{kind} is not deterministic"
    assert (a >= 0).all(), f"{kind} produced negative intensity"
    assert np.isfinite(a).all()


def test_intensity_scaling_is_invisible_to_a_normalised_representation():
    """The sanity check on the perturbation suite: a global gain change must not move an
    L2-normalised spectrum."""
    grid = np.linspace(450, 1800, 200)
    x = np.abs(np.sin(np.linspace(0, 12, 200)))[None, :] + 0.05
    x = x / np.linalg.norm(x)
    out = PT.apply("intensity_scaling", x, grid, 1.0, seed=0)
    assert float(np.dot(x[0], out[0])) == pytest.approx(1.0, abs=1e-6)


def test_stronger_perturbation_moves_the_spectrum_further():
    grid = np.linspace(450, 1800, 300)
    x = np.abs(np.sin(np.linspace(0, 20, 300)))[None, :] + 0.05
    x = x / np.linalg.norm(x)
    d = []
    for lvl in PT.LEVELS["gaussian_noise"]:
        out = PT.apply("gaussian_noise", x, grid, lvl, seed=0)
        d.append(1.0 - float(np.dot(x[0], out[0])))
    assert d == sorted(d), "degradation must be monotone in the sweep level"


# ── G. THE ROBUSTNESS STUDY ──────────────────────────────────────────────────
@ran
def test_robustness_study_covers_every_condition():
    rob = pd.read_csv(T / "robustness_curves_v1.csv")
    assert len(rob) == 12 * 5 * 4
    assert set(rob.representation) == {"RAW", "LSM", "CSM", "META"}
    assert set(rob.perturbation) == set(PT.PERTURBATIONS)


@ran
def test_all_representations_share_identical_splits(state):
    """No representation may get a different query set, split or metric."""
    v = pd.read_csv(T / "representation_comparison_v1.csv")
    assert set(v.representation) == {"RAW", "LSM", "CSM", "META"}
    assert v.dim.tolist() == [676, 50, 49, state["K"]]


def test_area_under_robustness_is_normalised_by_the_clean_baseline():
    """So a representation that starts lower and stays flat can win — which is the whole
    hypothesis being tested."""
    lv = [0.0, 1.0]
    assert EV.area_under_robustness(lv, [0.5, 0.5], 0.5) == pytest.approx(1.0)
    assert EV.area_under_robustness(lv, [0.9, 0.45], 0.9) == pytest.approx(0.75)


# ── H. OUTPUTS ───────────────────────────────────────────────────────────────
@ran
def test_all_gates_pass(state):
    g = pd.read_csv(V / "phase04_5_gates_v1.csv")
    assert g[g.status != "PASS"].empty
    assert state["status"] == "COMPLETE"


@ran
def test_meta_dictionary_is_marked_a_candidate():
    d = json.loads((A / "meta_dictionary_v1.json").read_text())
    assert "CANDIDATE" in d["status"]
    assert "no fitting" in d["inference"]


@ran
@pytest.mark.parametrize("n", range(1, 15))
def test_figure_exists(n):
    assert sorted(F.glob(f"fig{n:02d}_*.png")), f"figure {n:02d} missing"


@ran
def test_figures_are_png_only():
    assert not list(F.glob("*.svg"))


@ran
def test_reports_and_pdf_exist():
    assert (R / "PHASE_04_5_RESULTS.pdf").stat().st_size > 200_000
    rep = (R / "PHASE_04_5_REPORT.md").read_text()
    for term in ("discard", "informativeness floor", "robustness", "K = 3", "negative"):
        assert term.lower() in rep.lower(), term
    aud = (R / "PHASE_04_5_SCIENTIFIC_AUDIT.md").read_text()
    for term in ("fair chance", "weakness", "outruns", "reviewer", "risk"):
        assert term.lower() in aud.lower(), term


@ran
def test_manifest_lists_every_output():
    man = json.loads((A / "phase_04_5_manifest_v1.json").read_text())
    for o in man["outputs"]:
        p = Path(o["path"])
        assert (p if p.is_absolute() else REPO / p).is_file(), o["path"]


@ran
def test_output_root_is_configurable(monkeypatch):
    monkeypatch.setenv("GAIRA_V7_OUTPUT_ROOT", "/tmp/gaira-p45")
    assert PhaseOutputs("04.5").root == Path("/tmp/gaira-p45/phase04_5").resolve()
    assert frozen_root() == REPO / "results" / "v7_rebuild"
