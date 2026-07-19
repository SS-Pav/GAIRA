"""GAIRA V5 — Perturbation Response Audit tests.

Guarantees: the frozen atlas is never touched, the identity test is
label-independent, and response statistics behave correctly.
"""
import json
import sys
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "results/v5_rebuild/perturbation_response/code"))

FROZEN = REPO / "results/v5_rebuild/foundation/artifacts"
PR = REPO / "results/v5_rebuild/perturbation_response"
VOL = Path("/Volumes/SSD_Rad/GAIRA_DATA")
needs_data = pytest.mark.skipif(not VOL.exists(), reason="data volume not mounted")
needs_prior = pytest.mark.skipif(
    not (REPO / "results/v5_rebuild/reference_atlas_audit/tables/p1_component_inventory.csv").exists(),
    reason="component audit not present")


# ── response fingerprint + trajectory statistics (pure logic) ──
def test_response_fingerprint_is_delta_vs_control():
    import response_lib as RL
    ctrl = np.tile(np.array([1.0] + [0.0] * 23), (4, 1))
    treat = np.tile(np.array([1.0, 2.0] + [0.0] * 22), (5, 1))
    fp = RL.response_fingerprint(treat, ctrl)
    assert fp["delta"][1] == pytest.approx(2.0)
    assert fp["delta"][0] == pytest.approx(0.0)
    assert 1 in fp["top_up"]
    assert fp["n"] == 5


def test_fingerprint_bootstrap_flags_significant_component():
    import response_lib as RL
    rng = np.random.default_rng(0)
    ctrl = rng.normal(0, 0.01, (5, 24))
    treat = ctrl.mean(0) + np.eye(24)[10] * 0.5 + rng.normal(0, 0.01, (6, 24))
    fp = RL.response_fingerprint(treat, ctrl)
    assert fp["significant"][10]


def test_component_dose_response_monotone_and_saturating():
    import response_lib as RL
    concs = np.repeat([0, 1, 2, 4, 8, 16], 4).astype(float)
    Z = np.zeros((len(concs), 24))
    Z[:, 7] = 5 * concs / (2 + concs) + np.random.default_rng(0).normal(0, 0.01, len(concs))
    r = RL.component_dose_response(Z, concs, 7)
    assert r["spearman_rho"] > 0.95
    assert r["direction"] == "up"
    assert r.get("saturating_r2", 0) >= r.get("linear_r2", 0) - 0.05


def test_trajectory_fingerprint_fields():
    import response_lib as RL
    concs = np.repeat([0, 1, 2, 3], 3).astype(float)
    d = np.eye(24)[5]
    Z = np.vstack([c * d + np.random.default_rng(0).normal(0, 0.001, 24) for c in concs])
    tf = RL.trajectory_fingerprint(Z, concs)
    for k in ("path_length", "straightness", "mean_curvature_deg", "component_turnover",
              "response_entropy", "dominant_components"):
        assert k in tf
    assert tf["straightness"] == pytest.approx(1.0, abs=0.05)


# ── the label-independent identity test (the study's key methodological choice) ──
@needs_prior
def test_component_encodes_is_label_independent():
    import response_lib as RL
    cl = RL.load_component_reference_loadings()
    # c3's top reference analyte is adenine even though its audit label is 'sterol'
    assert RL.component_encodes("adenine", cl, 3)
    # a saccharide does not load the adenine component
    assert not RL.component_encodes("glucose", cl, 3)


@needs_prior
def test_component_encodes_handles_synonyms():
    import response_lib as RL
    cl = RL.load_component_reference_loadings()
    # if a component is loaded by 'dextrose', glucose should match via canonicalisation
    hits = [j for j in range(24) if RL.component_encodes("glucose", cl, j)]
    hits_dex = [j for j in range(24) if RL.component_encodes("dextrose", cl, j)]
    assert set(hits) == set(hits_dex)


# ── frozen atlas contract ──
@needs_data
def test_atlas_never_mutated_by_audit():
    p = PR / "artifacts/response_audit_manifest.json"
    if not p.exists():
        pytest.skip("audit not run")
    m = json.loads(p.read_text())
    assert m["verified_unchanged"] is True
    W = np.load(FROZEN / "manifold_components.npz")["components"]
    fp = hashlib.sha256(np.ascontiguousarray(W).tobytes()).hexdigest()[:32]
    assert fp == m["atlas_fingerprint"]


# ── produced results are self-consistent ──
@needs_data
def test_purine_substructure_two_pairs():
    p = PR / "tables/part6_purine_similarity.csv"
    if not p.exists():
        pytest.skip("purine case not run")
    m = pd.read_csv(p, index_col=0)
    assert m.loc["adenine", "hypoxanthine"] > 0.5     # one pair coheres
    assert m.loc["xanthine", "guanine"] > 0.5         # the other pair coheres
    assert m.loc["adenine", "xanthine"] < 0.3         # the pairs are distinct


@needs_data
def test_uricase_depletion_is_selective():
    p = PR / "tables/part8_uricase.json"
    if not p.exists():
        pytest.skip("uricase not run")
    u = json.loads(p.read_text())
    assert u["purine_component_c15_change"] < 0        # purine component decreases
    assert u["selective"] is True


@needs_data
def test_pure_vs_spike_consistency_reported():
    p = PR / "tables/part5_analyte_consistency.csv"
    if not p.exists():
        pytest.skip("consistency not run")
    c = pd.read_csv(p)
    assert "consistency_cosine" in c
    assert c.consistency_cosine.between(-1, 1).all()


@needs_data
def test_response_families_compared_against_raw_spectra():
    p = PR / "tables/part10_response_families.json"
    if not p.exists():
        pytest.skip("families not run")
    f = json.loads(p.read_text())
    for k in ("response_fingerprint_best_ari", "raw_spectrum_best_ari"):
        assert k in f


@needs_data
def test_theme_match_records_identity_column():
    p = PR / "tables/part3_theme_match.csv"
    if not p.exists():
        pytest.skip("theme match not run")
    t = pd.read_csv(p)
    assert "analyte_loads_this_component" in t
    # adenine's strongest responder must be a component it actually encodes
    ad = t[(t.analyte == "adenine") & (t.effect_rank == 0)]
    assert ad.analyte_loads_this_component.any()
