"""GAIRA V7 — Phase 09 regression tests: the canonical inference engine.

Three kinds of test. Contract tests hold the engine to its stated guarantees — frozen, stateless,
deterministic, reconciling. Artifact tests hold the committed run to the numbers it reported.
Adversarial tests encode the four defects found during the phase, each written so that it fails
if the defect returns.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from gaira.v7.canonical import engine as ENG
from gaira.v7.canonical.engine import FrozenArtifactError, GAIRAEngine
from gaira.v7.io import PhaseOutputs

OUT = PhaseOutputs("09", extra=("reports_examples",))
T, A_, F, R = OUT.tables, OUT.artifacts, OUT.figures, OUT.reports
HAS_RUN = (OUT.root / "PHASE_STATE.json").exists()
needs_run = pytest.mark.skipif(not HAS_RUN, reason="Phase 09 has not been run")


@pytest.fixture(scope="module")
def engine():
    return GAIRAEngine.load()


@pytest.fixture(scope="module")
def summary():
    return json.loads((A_ / "phase09_summary_v1.json").read_text())


@pytest.fixture(scope="module")
def spectrum(engine):
    """One real corpus spectrum, already on the canonical grid."""
    z = np.load(A_ / "engine_activations_v1.npz", allow_pickle=True)
    a = np.asarray(z["A"], float)[0]
    return (a @ engine._CSM), engine.grid


# ── contract: the engine is frozen ───────────────────────────────────────────
def test_expected_fingerprints_are_the_phase_05_set():
    assert ENG.EXPECTED_FINGERPRINTS == {
        "atlas": "09ed804a40836f4a05a91ba10900cded",
        "lsm": "208482d6f7178b5b8f16cace91be55b0",
        "csm": "0b4aa550ccefed3edabdbde5bae11c8d",
        "engine": "20d8bd99ce71f45a125c6a2b1d719e51",
    }


def test_load_verifies_fingerprints_and_would_refuse_a_changed_atlas(monkeypatch):
    """The refusal must be real, not a comment. Perturb one expected value and require a raise."""
    bad = dict(ENG.EXPECTED_FINGERPRINTS, atlas="0" * 32)
    monkeypatch.setattr(ENG, "EXPECTED_FINGERPRINTS", bad)
    with pytest.raises(FrozenArtifactError):
        GAIRAEngine.load()


def test_canonical_grid_constants_are_unchanged_since_v5():
    assert (ENG.GRID_LO, ENG.GRID_HI, ENG.GRID_STEP, ENG.N_BINS) == (450.0, 1800.0, 2.0, 676)


def test_engine_shape(engine):
    assert len(engine._lsm_ids) == 50
    assert len(engine._csm_ids) == 49
    assert len(engine.reference_molecules) == 154
    assert len(engine.chemistry_axes) == 16
    assert engine.grid.shape == (676,)


def test_chemistry_axes_are_the_frozen_sixteen(engine):
    from gaira.v7.chemistry.registry import CLASS_ORDER
    assert tuple(engine.chemistry_axes) == tuple(CLASS_ORDER)


# ── contract: stateless and deterministic ────────────────────────────────────
def test_reports_are_immutable(engine, spectrum):
    x, w = spectrum
    r = engine.infer(x, w, already_preprocessed=True)
    with pytest.raises(Exception):
        r.chemistry = {}
    with pytest.raises(Exception):
        r.preprocessing.n_peaks = 0


def test_inference_is_deterministic(engine, spectrum):
    x, w = spectrum
    a = engine.infer(x, w, already_preprocessed=True)
    b = engine.infer(x, w, already_preprocessed=True)
    assert a.to_dict() == b.to_dict()


def test_batch_position_does_not_change_a_result(engine):
    """A spectrum's answer must not depend on what was inferred before it."""
    z = np.load(A_ / "engine_activations_v1.npz", allow_pickle=True)
    A = np.asarray(z["A"], float)[:3]
    X = A @ engine._CSM
    alone = engine.infer(X[2], already_preprocessed=True).to_dict()
    for x in X:
        last = engine.infer(x, already_preprocessed=True).to_dict()
    assert last == alone


def test_grid_property_returns_a_copy(engine):
    g = engine.grid
    g[0] = -1.0
    assert engine.grid[0] == 450.0


# ── contract: scores reconcile ───────────────────────────────────────────────
def test_every_retrieval_score_reconciles(engine, spectrum):
    x, w = spectrum
    ret = engine.infer(x, w, already_preprocessed=True).retrieval
    for cand in ret["top"]:
        assert cand["reconciles"]
        assert abs(cand["contribution_sum"] - cand["similarity"]) < 1e-9


def test_similarity_decomposes_into_the_listed_csm_contributions(engine, spectrum):
    """The displayed supporting CSMs must be a subset of a decomposition that is exact."""
    x, w = spectrum
    ret = engine.infer(x, w, already_preprocessed=True).retrieval
    top = ret["top"][0]
    shown = sum(c["contribution"] for c in top["supporting_csms"])
    assert 0.0 <= shown <= top["similarity"] + 1e-9


def test_ranking_is_monotone(engine, spectrum):
    x, w = spectrum
    top = engine.infer(x, w, already_preprocessed=True).retrieval["top"]
    sims = [t["similarity"] for t in top]
    assert sims == sorted(sims, reverse=True)
    assert [t["rank"] for t in top] == list(range(1, len(top) + 1))


# ── contract: the inference path ─────────────────────────────────────────────
def test_projections_are_non_negative(engine, spectrum):
    """P-02. A Raman mixture cannot contain a negative component."""
    x, w = spectrum
    r = engine.infer(x, w, already_preprocessed=True)
    assert min(r.lsm["activation"]) >= 0.0
    assert min(r.csm["activation"]) >= 0.0
    assert min(r.chemistry["evidence"]) >= 0.0


def test_preprocessing_produces_a_unit_vector_on_the_canonical_grid(engine):
    rng = np.random.default_rng(0)
    w = np.linspace(400.0, 1900.0, 900)
    v = np.abs(rng.normal(1.0, 0.2, 900))
    x, pre = engine.preprocess(w, v)
    assert x.shape == (676,)
    assert abs(np.linalg.norm(x) - 1.0) < 1e-6
    assert pre.baseline_method == "asymmetric least squares"
    assert pre.smoothing == "Savitzky-Golay (9, 3)"


def test_a_short_spectrum_is_zero_filled_with_a_warning_never_extrapolated(engine):
    """Outside the measured range the engine must not invent signal.

    Not exactly zero after the fact. Two later stages bleed across the boundary: the
    Savitzky-Golay window spans nine bins, and asLS fits a slightly negative baseline under the
    empty region whose subtraction leaves a positive residue. Both are local — within 50 cm-1 of
    the edge the residue reaches 35% of the covered maximum, and beyond that it falls below 9%.
    What must hold is that the far region carries no trace of the measured level, which it would
    if the interpolation extrapolated instead of zero-filling.
    """
    w = np.linspace(900.0, 1200.0, 300)
    v = np.abs(np.random.default_rng(1).normal(1.0, 0.1, 300))
    x, pre = engine.preprocess(w, v)
    assert any("zero-filled" in s for s in pre.warnings)
    g = engine.grid
    covered = (g >= 900.0) & (g <= 1200.0)
    far = (g < 850.0) | (g > 1250.0)
    assert x[far].max() < 0.10 * x[covered].max()


def test_length_mismatch_raises(engine):
    with pytest.raises(ValueError):
        engine.infer(np.ones(100), np.ones(50))


def test_already_preprocessed_requires_the_canonical_bin_count(engine):
    with pytest.raises(ValueError):
        engine.infer(np.ones(100), np.ones(100), already_preprocessed=True)


def test_provenance_tree_reaches_wavenumbers(engine, spectrum):
    x, w = spectrum
    prov = engine.infer(x, w, already_preprocessed=True).provenance
    assert prov["root"] == "spectrum"
    assert {"lsm_layer", "csm_layer", "chemistry_layer", "molecule_layer"} <= set(prov)
    assert any(c["bands"] for c in prov["csm_layer"])
    assert prov["atlas_fingerprint"] == engine.atlas_fingerprint


def test_confidence_is_reconstruction_times_similarity(engine, spectrum):
    """Deliberately multiplicative: an unexplained spectrum must not be confident."""
    x, w = spectrum
    r = engine.infer(x, w, already_preprocessed=True)
    expected = np.clip(r.csm["explained_variance"], 0, 1) * r.retrieval["top"][0]["similarity"]
    assert abs(r.confidence["overall"] - expected) < 1e-9


def test_white_noise_is_far_less_confident_than_a_real_spectrum(engine):
    """Structureless input must not produce a confident answer.

    Measured, not assumed: over 20 seeds the highest confidence white noise achieves is 0.495,
    against a corpus mean of 0.803. It is separated, but note that the `unknown` warning fires on
    only 1 of 20 — white noise reconstructs at EV ~0.61, above the 0.50 floor. That is a real
    limitation of the warning and it is recorded in the audit rather than papered over here.
    """
    confs = []
    for seed in range(20):
        rng = np.random.default_rng(seed)
        x = np.abs(rng.normal(0, 1, 676))
        r = engine.infer(x / np.linalg.norm(x), already_preprocessed=True)
        confs.append(r.confidence["overall"])
    assert max(confs) < 0.60, f"white noise reached confidence {max(confs):.3f}"


# ── contract: what must be absent ────────────────────────────────────────────
def _engine_identifiers() -> set[str]:
    """Every name the engine module actually references, ignoring prose.

    Parsed rather than grepped: the module's own docstring lists the excluded machinery by name,
    so a text search would flag the very documentation that states the exclusion.
    """
    import ast
    tree = ast.parse(open(ENG.__file__.replace(".pyc", ".py")).read())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id.lower())
        elif isinstance(node, ast.Attribute):
            names.add(node.attr.lower())
        elif isinstance(node, ast.Import):
            names.update(a.name.lower() for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add((node.module or "").lower())
            names.update(a.name.lower() for a in node.names)
    return names


def test_no_bsv2_geometry_umap_pca_or_clustering_on_the_inference_path():
    names = _engine_identifiers()
    for banned in ("umap", "pca", "kmeans", "bsv2", "dbscan", "tsne", "louvain",
                   "agglomerativeclustering", "sklearn.decomposition", "sklearn.cluster",
                   "sklearn.manifold", "gaira.v7.programs", "gaira.v7.latent"):
        assert not any(banned in n for n in names), f"{banned} is referenced by the engine"


def test_engine_draws_no_random_numbers_at_inference():
    text = open(ENG.__file__.replace(".pyc", ".py")).read()
    for banned in ("np.random", "default_rng", "random.random", "shuffle"):
        assert banned not in text


def test_engine_writes_nothing():
    text = open(ENG.__file__.replace(".pyc", ".py")).read()
    for banned in ("open(", ".write(", "to_csv", "savez", "mkdir"):
        assert banned not in text, f"the engine appears to write via {banned}"


# ── artifacts: the committed run ─────────────────────────────────────────────
@needs_run
def test_all_gates_pass():
    g = pd.read_csv(T / "phase09_gates_v1.csv")
    assert len(g) == 16
    assert (g.status == "PASS").all(), g[g.status != "PASS"].to_string()


@needs_run
def test_every_spectrum_was_processed():
    e = pd.read_csv(T / "engine_outputs_all_spectra_v1.csv")
    assert len(e) == 375
    assert e.all_scores_reconcile.all()


@needs_run
def test_retrieval_reproduces_the_frozen_baseline(summary):
    v3 = summary["validation_3_retrieval"]
    assert summary["baseline_match"] is True
    for k, want in (("top1", 0.6053), ("top3", 0.7627), ("top5", 0.7947),
                    ("top10", 0.8107), ("mrr", 0.6870), ("ndcg5", 0.7112)):
        assert abs(v3[k] - want) < 5e-4, f"{k}: {v3[k]:.4f} != {want}"


@needs_run
def test_representative_reports_cover_every_chemistry_family():
    r = pd.read_csv(T / "representative_analytes_v1.csv")
    assert r.family.nunique() == 16
    assert len(r) == 48
    assert set(r.kind) == {"best", "median", "worst"}


@needs_run
def test_noise_robustness_ordering_radar_then_chemistry_then_molecule(summary):
    """The architecture's central promise: the general answer degrades after the specific one."""
    n = summary["noise_robustness"]
    assert n["mean_radar_cosine"] > n["mean_chemistry_top1"] > n["mean_retrieval_top1"]


@needs_run
def test_sixteen_figures_exist():
    pngs = sorted(F.glob("F*.png"))
    assert len(pngs) == 16
    assert all(p.stat().st_size > 20_000 for p in pngs)


@needs_run
def test_all_six_documents_exist():
    for name in ("PHASE_09_REPORT.md", "PHASE_09_DECISION_GATE.md", "PHASE_09_ENGINE_SPEC.md",
                 "PHASE_09_MATHEMATICAL_APPENDIX.md", "PHASE_09_SCIENTIFIC_AUDIT.md",
                 "PHASE_09_FIGURES.pdf"):
        assert (R / name).exists(), name


# ── adversarial: the four defects found during the phase ─────────────────────
@needs_run
def test_chemistry_accuracy_is_reported_held_out_not_only_in_sample(summary):
    """Defect 1. Validation 4 was in-sample and would have claimed 0.955 as the headline."""
    v4 = summary["validation_4_chemistry"]
    assert "fine_top1_heldout" in v4
    assert "IN_SAMPLE_WARNING" in v4
    assert v4["fine_top1_heldout"] < v4["fine_top1_in_sample"], (
        "held-out above in-sample would mean the split is not doing its job")
    assert abs(v4["fine_top1_heldout"] - 0.8507) < 5e-4


@needs_run
def test_the_in_sample_warning_names_the_number_to_quote(summary):
    w = summary["validation_4_chemistry"]["IN_SAMPLE_WARNING"].lower()
    assert "not a performance claim" in w
    assert "fine_top1_heldout" in w


@needs_run
def test_retrieval_calibration_temperature_was_fitted_not_pinned(summary):
    """Defect 2. A pinned T = 0.02 reported ECE 0.2529, twice the fitted value."""
    v3 = summary["validation_3_retrieval"]
    assert v3["ece"] < 0.20, f"ECE {v3['ece']:.4f} suggests the temperature is pinned again"


@needs_run
def test_retrieval_discrimination_is_not_self_referential(summary):
    """Defect 4. Confidence derived from 1/rank, scored against rank <= 1, gave exactly 1.000.

    A discrimination of 1.0 here is not excellence; it is the metric being computed against
    itself. The confidence must come from a quantity available without knowing the answer.
    """
    d = summary["validation_3_retrieval"]["discrimination"]
    assert 0.5 < d < 0.99, f"discrimination {d:.4f} — is confidence derived from the rank again?"


def test_load_does_not_shadow_the_classmethod_parameter():
    """Defect 3. A local `cls` inside load() shadowed the classmethod's own `cls`."""
    import inspect
    src = inspect.getsource(GAIRAEngine.load)
    body = src[src.index(":") :]
    assert "\n        cls =" not in body and "\n    cls =" not in body


def test_engine_module_hardcodes_no_output_path():
    text = open(ENG.__file__.replace(".pyc", ".py")).read()
    assert "/Volumes/" not in text
    assert "/Users/" not in text
    assert "results/v7_rebuild" not in text
