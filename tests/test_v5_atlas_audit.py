"""GAIRA Raman Reference Atlas v0.1 — audit integrity tests.

The central guarantee is that the audit is READ-ONLY: the frozen atlas must be
byte-identical before and after any analysis, and no audit step may refit it.
"""
import json
import sys
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "results/v5_rebuild/reference_atlas_audit/code"))

from gaira.foundation import serialization as SER

FROZEN = REPO / "results/v5_rebuild/foundation/artifacts"
AUD = REPO / "results/v5_rebuild/reference_atlas_audit"
VOL = Path("/Volumes/SSD_Rad/GAIRA_DATA")
needs_data = pytest.mark.skipif(not VOL.exists(), reason="data volume not mounted")
needs_frozen = pytest.mark.skipif(not (FROZEN / "manifold.json").exists(),
                                  reason="frozen atlas not present")


def _synth(n=40, d=676, k=24, seed=0):
    rng = np.random.default_rng(seed)
    return np.clip(rng.normal(1.0, 0.3, (n, d)), 0, None)


# ── frozen-atlas contract ──
@needs_frozen
def test_atlas_fingerprint_verified_on_load():
    atlas = SER.load_frozen_manifold(FROZEN)
    fp = hashlib.sha256(np.ascontiguousarray(atlas.components).tobytes()).hexdigest()[:32]
    assert fp == atlas.meta["fingerprint"]
    assert atlas.k == 24 and atlas.name == "NMF"


@needs_frozen
def test_load_rejects_tampered_atlas(tmp_path):
    npz = np.load(FROZEN / "manifold_components.npz")
    meta = json.loads((FROZEN / "manifold.json").read_text())
    comps = npz["components"].copy()
    comps[0, 0] += 1.0                                    # tamper
    np.savez(tmp_path / "manifold_components.npz", components=comps,
             grid=npz["grid"], mean=npz["mean"])
    (tmp_path / "manifold.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        SER.load_frozen_manifold(tmp_path)


@needs_frozen
def test_projection_does_not_mutate_atlas():
    atlas = SER.load_frozen_manifold(FROZEN)
    before = atlas.components.copy()
    X = _synth()
    atlas.project(X); atlas.coordinates(X); atlas.reconstruct(X)
    assert np.array_equal(before, atlas.components), "projection mutated the frozen atlas"


@needs_frozen
def test_coordinates_are_nonnegative_shares():
    atlas = SER.load_frozen_manifold(FROZEN)
    Z = atlas.coordinates(_synth())
    assert (Z >= -1e-9).all()
    assert np.allclose(Z.sum(axis=1), 1.0, atol=1e-6)


@needs_frozen
def test_projection_is_deterministic():
    atlas = SER.load_frozen_manifold(FROZEN)
    X = _synth()
    assert np.allclose(atlas.project(X), atlas.project(X))


# ── audit engine ──
def test_family_refinement_is_audit_local():
    """The audit refines chemistry locally; the foundation module must be untouched."""
    import atlas_audit as AA
    from gaira.foundation.families_raman import family_of as base
    assert base("elaidic acid") == "organic_acid"        # foundation unchanged
    assert AA.family_of("elaidic acid") == "fatty_acid"  # audit corrects it
    assert AA.molecular_class("elaidic acid") == "lipid"
    assert AA.family_of("glucose") == base("glucose")    # unaffected elsewhere


def test_metadata_absence_is_reported_not_invented():
    import atlas_audit as AA
    assert AA.molecular_weight("glucose") == "unavailable"
    assert AA.biochemical_role("glucose") == "unavailable"
    assert AA.subfamily("citrate") == "unavailable"
    assert AA.subfamily("(+)-sucrose") == "disaccharide"


def test_entropy_and_gini_bounds():
    import atlas_audit as AA
    assert AA.norm_entropy([1, 0, 0, 0]) == pytest.approx(0.0, abs=1e-9)
    assert AA.norm_entropy([1, 1, 1, 1]) == pytest.approx(1.0, abs=1e-6)
    assert 0.0 <= AA.gini(np.array([1.0, 1.0, 1.0])) <= 1.0
    assert AA.gini(np.array([0, 0, 0, 5.0])) > AA.gini(np.ones(4))


def test_component_bands_and_uniqueness():
    import atlas_audit as AA
    grid = np.linspace(450, 1800, 676)
    W = np.zeros((3, 676))
    for j, c in enumerate((600, 1000, 1400)):
        W[j] += np.exp(-0.5 * ((grid - c) / 8) ** 2)
    W[2] += np.exp(-0.5 * ((grid - 1000) / 8) ** 2)      # shares a band with comp 1
    bands = AA.component_bands(W, grid)
    assert all(len(b) >= 1 for b in bands)
    u = AA.band_uniqueness(bands)
    assert u[0] > u[2]                                    # c0 unique, c2 shares with c1


def test_coherence_detects_pure_vs_mixed():
    import atlas_audit as AA
    idx = ["tripalmitin", "tristearin", "trimyristin", "glucose", "albumin", "adenine"]
    A = pd.DataFrame(np.array([
        [1.0, 0.1], [0.9, 0.1], [0.95, 0.1],              # lipids load comp 0
        [0.05, 1.0], [0.05, 0.9], [0.05, 0.95]]), index=idx)
    Xa = np.random.default_rng(0).normal(size=(6, 50))
    pure = AA.coherence(A, 0, Xa, top_n=3)
    mixed = AA.coherence(A, 1, Xa, top_n=3)
    assert pure["class_purity"] > mixed["class_purity"]
    assert pure["dominant_class"] == "lipid"
    assert pure["avg_molecular_similarity"] == "unavailable"


def test_grouping_study_does_not_optimise_silhouette_alone():
    import atlas_audit as AA
    rng = np.random.default_rng(0)
    k = 12
    W = np.abs(rng.normal(size=(k, 200)))
    A = pd.DataFrame(np.abs(rng.normal(size=(30, k))),
                     index=[f"an{i}" for i in range(30)])
    rel = AA.relationships(A, W, AA.component_bands(W, np.linspace(450, 1800, 200)))
    df, assign, Zl, D = AA.grouping_study(A, W, rel, np.abs(rng.normal(size=(30, 200))),
                                          ks=(3, 4), n_boot=3)
    for c in ("silhouette", "bootstrap_reproducibility", "chemical_coherence",
              "interpretable_group_fraction", "composite"):
        assert c in df.columns
    assert set(assign) == {3, 4}


def test_mss_readiness_shape_and_confidence():
    import atlas_audit as AA
    rng = np.random.default_rng(0)
    A = pd.DataFrame(np.abs(rng.normal(size=(12, 8))), index=[f"an{i}" for i in range(12)])
    m = AA.mss_readiness(A)
    assert len(m) == 12
    assert set(m.assignment_confidence).issubset({"high", "moderate", "low"})
    assert (m.signature_uniqueness >= 0).all()


def test_confusability_classifier():
    import run_confusability as RC
    assert RC.classify("fructose", "(-)-fructose") == "duplicate"
    assert RC.classify("(+)-dextrose", "glucose") == "duplicate"
    assert RC.classify("tristearin", "tripalmitin") == "homolog"
    assert RC.classify("tristearin", "elaidic acid") in ("family", "same_class")
    assert RC.classify("glucose", "albumin") == "distinct"


# ── produced artifacts ──
@needs_data
@needs_frozen
def test_audit_outputs_present_and_consistent():
    man = AUD / "artifacts/audit_manifest.json"
    if not man.exists():
        pytest.skip("audit not yet run")
    m = json.loads(man.read_text())
    assert m["atlas"]["verified_unchanged"] is True
    assert m["atlas"]["k"] == 24
    inv = pd.read_csv(AUD / "tables/p1_component_inventory.csv")
    assert len(inv) == 24
    assert inv.component.tolist() == list(range(24))
    comp = pd.read_csv(AUD / "tables/p2_full_analyte_composition.csv")
    assert set(comp.component.unique()) == set(range(24))
    # every component's contributions sum to ~100%
    s = comp.groupby("component").contribution_pct.sum()
    assert np.allclose(s.values, 100.0, atol=1.0)


@needs_data
@needs_frozen
def test_ood_sets_are_labelled_out_of_domain():
    p = AUD / "tables/p13_out_of_domain_stress_test.csv"
    if not p.exists():
        pytest.skip("stress test not run")
    d = pd.read_csv(p)
    assert set(d.dataset.unique()) <= {"adenine_series", "ergothioneine_calibration",
                                       "uricase_depletion"}
    assert (d.ood_distance >= 0).all()


@needs_data
@needs_frozen
def test_nothing_frozen_by_the_audit():
    """The audit must not emit a frozen ontology or BSV."""
    onto = json.loads((AUD / "tables/p10_ontology_v0_1.json").read_text())
    bsv = json.loads((AUD / "tables/p11_bsv_design_study.json").read_text())
    assert "NOT frozen" in onto["version"]
    assert bsv["not_frozen"] is True
    assert bsv["recommendation"] == "C_hierarchical"
