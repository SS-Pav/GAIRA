"""Regression tests for the V6 detection gate + restricted hierarchy + Explorer V6.

Validates: physically-anchored detection (anchors pass/fail), deterministic score, tier logic,
restricted-hierarchy gains, transfer/roadmap scoping, V1–V5 untouched, frozen fingerprint, and
Explorer V6 rendering.
"""
import sys
import json
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
V6 = REPO / "results/v5_rebuild/detection_gate_v6"
CANON = "09ed804a40836f4a05a91ba10900cded"
V6APP = REPO / "gaira_foundation_explorer_v6"
HAS_ST = importlib.util.find_spec("streamlit") is not None


@pytest.fixture(scope="module")
def det():
    return pd.read_csv(V6 / "tables/detection_metrics.csv").set_index("analyte")


@pytest.fixture(scope="module")
def dsum():
    return json.loads((V6 / "artifacts/detection_summary.json").read_text())


@pytest.fixture(scope="module")
def rsum():
    return json.loads((V6 / "artifacts/restricted_hierarchy_summary.json").read_text())


# 1 · 51 analytes gated
def test_51_analytes(det):
    assert len(det) == 51


# 2 · the physically-required anchors (the validation contract)
def test_anchors_pass(det):
    for a in ["adenine", "ergothioneine", "urate", "xanthine"]:
        assert det.loc[a, "detection_pass"], a
    for a in ["ergothioneine", "urate", "xanthine"]:
        assert det.loc[a, "detection_tier"] == "GOOD"


def test_anchors_fail(det):
    for a in ["glucose", "tyrosine", "oleate"]:
        assert not det.loc[a, "detection_pass"], a
    assert det.loc["oleate", "detection_tier"] == "UNDETECTABLE"


# 3 · replicate cosine rejected — Pearson discriminates, not cosine
def test_pearson_discriminates(det):
    strong = det.loc[["xanthine", "urate", "ergothioneine"], "rep_pearson"].min()
    weak = det.loc[["glucose", "tyrosine", "oleate"], "rep_pearson"].max()
    assert strong > weak + 0.3


# 4 · tier logic matches thresholds
def test_tier_logic(det, dsum):
    th = dsum["tier_thresholds"]
    for a, r in det.iterrows():
        dc = r.detection_confidence
        exp = "GOOD" if dc >= th["GOOD"] else "MODERATE" if dc >= th["MODERATE"] else "POOR" if dc >= th["POOR"] else "UNDETECTABLE"
        assert r.detection_tier == exp, a
        assert bool(r.detection_pass) == (dc >= dsum["detection_pass_threshold"])


# 5 · deterministic score in [0,1]
def test_score_range(det):
    assert (det.detection_confidence >= 0).all() and (det.detection_confidence <= 1).all()


# 6 · counts consistent
def test_counts(det, dsum):
    assert dsum["n_pass"] == int(det.detection_pass.sum())
    assert dsum["n_pass"] + dsum["n_fail"] == 51


# 7 · restricted hierarchy: detectable-only >= all at every level (gain >= 0)
def test_recovery_gain_nonnegative(rsum):
    for row in rsum["recovery_ladder_all_vs_detectable"]:
        assert row["detectable_frac"] >= row["all_frac"] - 1e-9


# 8 · abstraction improves but specific stays low (the key finding)
def test_abstraction_partial(rsum):
    ab = rsum["abstraction_improves_after_gate"]
    assert ab["exact_detectable"] > ab["exact_all"]           # exact ~doubles
    assert ab["mss_specific_detectable"] < 0.2                 # specific still low
    assert ab["theme_specific_detectable"] < 0.15


# 9 · transfer cases + roadmap partition all 51
def test_transfer_partition(rsum):
    assert sum(rsum["transfer_cases"].values()) == 51
    assert sum(rsum["roadmap_groups"].values()) == 51


# 10 · roadmap: a defined non-empty transfer-worth-trying set
def test_transfer_target_set(rsum):
    n = rsum["roadmap_groups"].get("potentially recoverable (transfer worth trying)", 0)
    assert 5 <= n <= 20


# 11 · undetectable analytes are measurement-limited (Case A) in transfer decision
def test_undetectable_are_case_A():
    td = pd.read_csv(V6 / "tables/per_analyte_transfer_decision.csv")
    fail = td[~td.detection_pass]
    assert (fail.transfer_case.str.startswith("A")).all()


# 12 · edge cases documented (identity recovered despite failing detection)
def test_edge_cases(rsum):
    assert set(rsum["identity_recovered_but_detection_fail"]) <= {"creatinine", "thymine"}


# 13 · fingerprint unchanged
def test_fingerprint(dsum):
    assert dsum["atlas_fingerprint"] == CANON
    sys.path.insert(0, str(REPO / "src"))
    from gaira.engine import GAIRAEngine
    assert GAIRAEngine().atlas.meta["fingerprint"] == CANON


# 14 · reports + validation notebook + figures exist
def test_artifacts_exist():
    assert (V6 / "GAIRA_Pure_AgSERS_Evaluation_V6.pdf").exists()
    assert (V6 / "code/validate_detection.ipynb").exists()
    for i in range(1, 11):
        assert list((V6 / "figures").glob(f"fig{i:02d}_*.png")), f"figure {i} missing"
    md = (V6 / "GAIRA_Pure_AgSERS_Evaluation_V6.md").read_text().lower()
    for s in ["detection gate", "representative spectra", "transfer", "roadmap", "conclusions"]:
        assert s in md


# 15-19 · V1-V5 importable (untouched)
@pytest.mark.parametrize("app,mod", [
    ("gaira_foundation_explorer", "explorer_core.data"),
    ("gaira_foundation_explorer_v2", "v2_core.data"),
    ("gaira_foundation_explorer_v3", "v3_core.data"),
    ("gaira_foundation_explorer_v4", "v4_core.data"),
    ("gaira_foundation_explorer_v5", "v5_core.data")])
def test_prior_explorers_import(app, mod):
    sys.path.insert(0, str(REPO / app))
    import importlib; importlib.import_module(mod)


# 20 · reuses V5 recovery flags unchanged (no re-derivation)
def test_reuses_v5_flags():
    v5 = pd.read_csv(REPO / "results/v5_rebuild/abstraction_recovery_v5/tables/per_analyte_abstraction_recovery.csv").set_index("analyte")
    td = pd.read_csv(V6 / "tables/per_analyte_transfer_decision.csv").set_index("analyte")
    assert (td.latent_identity_recovered == v5.latent_identity_recovered.reindex(td.index)).all()


# 21 · Explorer V6 renders
@pytest.mark.skipif(not HAS_ST, reason="streamlit not installed")
@pytest.mark.parametrize("page", [
    "1 · Overview", "2 · Detection Gate", "4 · Representative Spectra", "5 · Detection Confidence",
    "6 · Detectable vs Undetectable", "7 · Recovery Hierarchy", "9 · Transfer Function Assessment",
    "10 · Roadmap", "11 · Individual Analytes", "13 · Final Conclusions"])
def test_v6_page_renders(page):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(V6APP / "app.py"), default_timeout=300).run()
    assert not at.exception, [str(e.value) for e in at.exception]
    at.radio[0].set_value(page).run()
    assert not at.exception, (page, [str(e.value) for e in at.exception])
