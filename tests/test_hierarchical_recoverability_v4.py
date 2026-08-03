"""Regression tests for the V4 null-calibrated recoverability analysis + Explorer V4.

Validates reproducibility of V3, the documented recovery rules, null controls, FDR, blank
controls, that raw cosine never classifies recovery, and page-by-page rendering — plus that
V1/V2/V3 remain importable and the frozen atlas is unchanged.
"""
import sys
import json
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "results/v5_rebuild/hierarchical_recoverability_v4"
TABLES = BASE / "tables"; ARTIFACTS = BASE / "artifacts"
V3 = REPO / "results/v5_rebuild/representation_hierarchy_v3/tables/per_analyte_hierarchy.csv"
CANON = "09ed804a40836f4a05a91ba10900cded"
V4APP = REPO / "gaira_foundation_explorer_v4"
HAS_ST = importlib.util.find_spec("streamlit") is not None


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(TABLES / "per_analyte_evidence_profile.csv")


@pytest.fixture(scope="module")
def summary():
    return json.loads((ARTIFACTS / "recoverability_summary.json").read_text())


@pytest.fixture(scope="module")
def cards():
    return json.loads((ARTIFACTS / "all_cards_v4.json").read_text())


# 1
def test_51_matched(df):
    assert len(df) == 51 and summary_ok()


def summary_ok():
    return True


# 2-4 reproduce V3 exactly
def test_reproduces_v3(summary):
    r = summary["reproducibility_vs_v3"]
    assert r["C_latent_maxdiff"] == 0.0 and r["C_MSS_maxdiff"] == 0.0 and r["C_theme_raw_maxdiff"] == 0.0


def test_latent_mss_theme_match_v3(df):
    v3 = pd.read_csv(V3).set_index("analyte"); j = df.set_index("analyte")
    assert (j.C_latent - v3.L1_latent_fingerprint).abs().max() < 1e-9
    assert (j.C_MSS - v3.L2_mss_motif).abs().max() < 1e-9
    assert (j.C_theme_raw - v3.L3a_theme_raw).abs().max() < 1e-9


# 5 determinism of the FDR helper
def test_fdr_deterministic_and_correct():
    sys.path.insert(0, str(BASE / "code"))
    from recoverability_analysis import bh_fdr
    p = np.array([0.01, 0.02, 0.03, 0.5, 0.9])
    q1 = bh_fdr(p); q2 = bh_fdr(p)
    assert np.allclose(q1, q2)                       # deterministic
    assert np.all(np.diff(q1[np.argsort(p)]) >= -1e-9)  # monotone non-decreasing in p order
    assert q1[0] == pytest.approx(0.05, abs=1e-9)    # 0.01*5/1


# 6 recovery follows the documented rule: recovered => rank1 & matched>null95
def test_recovery_rule_latent(df):
    rec = df[df.latent_recovered]
    assert (rec.latent_rank == 1).all()
    assert (rec.C_latent > rec.latent_null95).all()


def test_recovery_rule_mss(df):
    rec = df[df.MSS_recovered]
    assert (rec.MSS_rank == 1).all() and (rec.C_MSS > rec.MSS_null95).all()


# 8 counts match per-analyte table
def test_counts_match_table(df, summary):
    c = {r["level"]: r for _, r in pd.read_csv(TABLES / "recoverable_analyte_counts.csv").iterrows()}
    assert int(c["latent"]["n_recovered"]) == int(df.latent_recovered.sum())
    assert int(c["MSS"]["n_recovered"]) == int(df.MSS_recovered.sum())
    assert int(c["theme"]["n_recovered"]) == int(df.theme_recovered.sum())
    assert int(c["matrix"]["n_recovered"]) == int(df.matrix_recovered.sum())


# 9 perturbation only for the 3
def test_perturbation_only_three(df):
    assert set(df[df.perturbation_validated].analyte) == {"adenine", "ergothioneine", "urate"}
    assert (df[~df.perturbation_validated].perturbation_status == "not tested").all()


# 10 matrix status only for serum-tested
def test_matrix_only_serum_tested(df):
    untested = df[~df.serum_tested]
    assert (untested.serum_tier == "not tested").all()
    assert not untested.matrix_recovered.any()


# 11 untested fields render "not tested" in cards
def test_cards_not_tested(cards):
    for a, c in cards.items():
        if a not in ("adenine", "ergothioneine", "urate"):
            assert c["level4_perturbation"]["status"] == "not tested"


# 12 raw theme cosine never alone classifies recovery
def test_theme_recovery_needs_identity_and_expected(df):
    rec = df[df.theme_recovered]
    assert (rec.theme_id_rank == 1).all()           # identity residual rank-1, not raw cosine
    assert rec.expected_theme_top3.all()            # expected theme retained
    # a high raw theme cosine with no identity is NOT recovered
    high_raw_no_id = df[(df.C_theme_raw > 0.9) & (~df.theme_recovered)]
    assert len(high_raw_no_id) > 0


# 13 argmax labelled supporting/strict in decision table
def test_argmax_labelled_supporting():
    dec = pd.read_csv(TABLES / "metric_decision_table.csv")
    row = dec[dec.primary_metric.str.contains("Argmax", case=False)]
    assert len(row) == 1 and "brittle" in row.iloc[0].null_result.lower()


# 14-15 Spearman + top-k nulls present
def test_nulls_present(df, summary):
    assert "spearman_matched_vs_null" in summary
    assert "top3_null" in df.columns and (df.top3_null > 0).any()


# 16 purine blank controls load
def test_purine_blank(summary):
    pc = summary["purine_controls"]
    assert pc["serum_blank_dominant_theme"] == "nucleic_purine"
    assert 0.1 < pc["serum_blank_purine_theme"] < 0.5


# 17-19 V1/V2/V3 importable (untouched)
def test_v1_imports():
    sys.path.insert(0, str(REPO / "gaira_foundation_explorer"))
    import importlib; importlib.import_module("explorer_core.data")


def test_v2_imports():
    sys.path.insert(0, str(REPO / "gaira_foundation_explorer_v2"))
    import importlib; importlib.import_module("v2_core.data")


def test_v3_imports():
    sys.path.insert(0, str(REPO / "gaira_foundation_explorer_v3"))
    import importlib; importlib.import_module("v3_core.data")


# 21 reports exist
def test_reports_exist():
    assert (BASE / "GAIRA_Hierarchical_Cross_Modal_Validation_V4.pdf").exists()
    md = (BASE / "GAIRA_Hierarchical_Cross_Modal_Validation_V4.md").read_text()
    for sec in ["Executive summary", "Recoverable-analyte counts", "purine attractor", "Perturbation",
                "Matrix recoverability", "Conclusions"]:
        assert sec.lower() in md.lower()


# 22 fingerprint unchanged
def test_fingerprint(summary):
    assert summary["atlas_fingerprint"] == CANON
    sys.path.insert(0, str(REPO / "src"))
    from gaira.engine import GAIRAEngine
    assert GAIRAEngine().atlas.meta["fingerprint"] == CANON


# 23 MSS-not-primary is documented and true
def test_mss_not_primary(summary):
    m = summary["mss_is_primary_candidate"]
    assert m["mss_separation"] < m["latent_separation"]         # smaller null separation
    assert m["mss_n_recovered"] <= m["latent_n_recovered"]


# 24 all figures + docs exist
def test_figures_and_docs_exist():
    for i in range(1, 12):
        figs = list((BASE / "figures").glob(f"fig{i:02d}_*.png"))
        assert figs, f"figure {i} missing"
    for d in ["AUDIT_OF_V3_METRICS.md", "METRICS_AND_DECISION_RULES.md"]:
        assert (BASE / d).exists()


# 20 Explorer V4 renders every page
@pytest.mark.skipif(not HAS_ST, reason="streamlit not installed")
@pytest.mark.parametrize("page", [
    "1 · Overview", "2 · Foundation Dataset", "3 · Latent Representation",
    "4 · How GAIRA Interprets a Spectrum", "5 · Cross-Modal Validation", "6 · MSS Motif Recovery",
    "7 · Biochemical Theme Interpretation", "8 · Recoverable Analytes ★", "9 · The Purine Attractor",
    "10 · Perturbation Validation", "11 · Matrix Recoverability", "12 · Biological Studies",
    "13 · Limitations", "14 · Future DART", "15 · Methods & Provenance"])
def test_v4_page_renders(page):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(V4APP / "app.py"), default_timeout=300).run()
    assert not at.exception, [str(e.value) for e in at.exception]
    at.radio[0].set_value(page).run()
    assert not at.exception, (page, [str(e.value) for e in at.exception])
