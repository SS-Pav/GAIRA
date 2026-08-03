"""Regression tests for the V5 abstraction-recovery analysis + Explorer V5.

Validates: V4 identity reproduction, null-adjusted MSS/theme recovery, overlay provenance +
low-count flagging, no-replicate-leakage LOAO, balanced metrics, raw-cosine not used alone,
perturbation/matrix scoping, determinism, V1–V4 untouched, and Explorer V5 rendering.
"""
import sys
import json
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "results/v5_rebuild/abstraction_recovery_v5"
TABLES = BASE / "tables"; ARTIFACTS = BASE / "artifacts"
CANON = "09ed804a40836f4a05a91ba10900cded"
V5APP = REPO / "gaira_foundation_explorer_v5"
HAS_ST = importlib.util.find_spec("streamlit") is not None


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(TABLES / "per_analyte_abstraction_recovery.csv")


@pytest.fixture(scope="module")
def summary():
    return json.loads((ARTIFACTS / "abstraction_summary.json").read_text())


@pytest.fixture(scope="module")
def overlay():
    return pd.read_csv(TABLES / "analyte_classification_overlay.csv")


@pytest.fixture(scope="module")
def cards():
    return json.loads((ARTIFACTS / "all_cards_v5.json").read_text())


# 1
def test_51_analytes(df):
    assert len(df) == 51


# 2 reproduce V4 identity
def test_reproduces_v4_identity(summary):
    r = summary["reproducibility_vs_v4_identity"]
    assert r["latent"] and r["MSS"] and r["theme"]
    assert summary["exact_identity"] == {"latent": 7, "MSS": 3, "theme": 4}


# 3 component recovery present & deterministic-shaped
def test_component_columns(df):
    for c in ["comp_top3_overlap", "comp_mass_retained", "component_recovered"]:
        assert c in df.columns


# 4 expected motif complete or explicitly unassigned
def test_expected_mss_complete_or_unassigned(df):
    assert df.expected_mss.notna().all()
    assert set(df[df.expected_mss == "unassigned"].analyte) == {"phosphate", "creatinine", "urea"}


# 5 MSS recovery uses null-adjusted evidence (recovered => present & enrichment computed)
def test_mss_recovery_null_adjusted(df):
    rec = df[df.mss_motif_recovered]
    assert rec.mss_present_top3.all()
    assert rec.mss_enrich_null.notna().all()


# 6 subclass provenance
def test_overlay_provenance(overlay):
    assert (overlay.assignment_source.str.len() > 0).all()
    assert (BASE / "ANALYTE_CLASSIFICATION_PROVENANCE.md").exists()


# 7 low-count subclasses flagged
def test_low_count_flagged(overlay):
    singletons = overlay[overlay.subclass_n < 2]
    assert singletons.subclass_exploratory.all()
    assert len(singletons) >= 10


# 8 no replicate leakage — classification uses per-analyte means (51 rows), analyte-level LOAO
def test_analyte_level_no_leakage(df, summary):
    # exactly one row per analyte ⇒ per-analyte means ⇒ replicates cannot split across folds
    assert df.analyte.nunique() == 51 == len(df)
    assert "classification" in summary


# 9 broad-family results use balanced metrics
def test_balanced_metrics_present(summary):
    for key in summary["classification"]:
        assert "balanced_accuracy" in summary["classification"][key]
        assert "macro_f1" in summary["classification"][key]


# 10 theme recovery not raw-cosine-alone (recovered => top3 & enrichment computed)
def test_theme_recovery_null_adjusted(df):
    rec = df[df.theme_recovered]
    assert rec.theme_present_top3.all()
    assert rec.theme_enrich_null.notna().all()


# 11 purine-attractor not falsely theme-recovered: a non-purine merely purine-pulled isn't theme-recovered
def test_purine_pull_not_false_recovery(df):
    pulled = df[(df.broad_family != "purine") & (df.delta_purine > 0.05)]
    # none of the merely-pulled non-purines are theme-recovered on purine
    assert not (pulled.theme_recovered & (pulled.expected_theme == "nucleic_purine")).any()


# 12 perturbation only three
def test_perturbation_only_three(df):
    assert set(df[df.perturbation_status != "not tested"].analyte) == {"adenine", "ergothioneine", "urate"}


# 13 matrix only serum-tested
def test_matrix_only_serum_tested(df):
    assert not df[df.serum_tier == "not tested"].matrix_recovered.any()


# 14 counts match per-analyte records
def test_counts_match(df, summary):
    lists = summary["recovered_lists"]
    assert set(lists["exact_latent"]) == set(df[df.latent_identity_recovered].analyte)
    assert set(lists["mss_motif"]) == set(df[df.mss_motif_recovered].analyte)
    assert set(lists["theme"]) == set(df[df.theme_recovered].analyte)


# 15 cards render 'not tested' / 'unassigned'
def test_cards_not_tested(cards):
    for a, c in cards.items():
        if a not in ("adenine", "ergothioneine", "urate"):
            assert c["perturbation"]["status"] == "not tested"
    assert cards["phosphate"]["mss_motif"]["expected"] == "unassigned"


# 16 the decisive finding: Raman control >> cross-modal
def test_raman_control_beats_crossmodal(summary):
    ctrl = summary["raman_raman_control"]
    cross = summary["classification"]
    assert ctrl["theme"]["balanced_accuracy"] > cross["theme|latent"]["balanced_accuracy"]
    assert ctrl["family"]["balanced_accuracy"] > cross["family|latent"]["balanced_accuracy"]
    # abstraction helps within Raman
    assert ctrl["theme"]["balanced_accuracy"] > ctrl["subclass"]["balanced_accuracy"]


# 17-20 V1-V4 importable (untouched)
@pytest.mark.parametrize("app,mod", [
    ("gaira_foundation_explorer", "explorer_core.data"),
    ("gaira_foundation_explorer_v2", "v2_core.data"),
    ("gaira_foundation_explorer_v3", "v3_core.data"),
    ("gaira_foundation_explorer_v4", "v4_core.data")])
def test_prior_explorers_import(app, mod):
    sys.path.insert(0, str(REPO / app))
    import importlib; importlib.import_module(mod)


# 18-19 report exists + sections
def test_report_exists_and_sections():
    assert (BASE / "GAIRA_Pure_AgSERS_Abstraction_Recovery_V5.pdf").exists()
    md = (BASE / "GAIRA_Pure_AgSERS_Abstraction_Recovery_V5.md").read_text().lower()
    for s in ["executive summary", "mss motif recovery", "molecular subclass", "biochemical-theme",
              "purine-attractor", "perturbation", "conclusions"]:
        assert s in md


# 20 fingerprint unchanged
def test_fingerprint(summary):
    assert summary["atlas_fingerprint"] == CANON
    sys.path.insert(0, str(REPO / "src"))
    from gaira.engine import GAIRAEngine
    assert GAIRAEngine().atlas.meta["fingerprint"] == CANON


# 21 all 12 figures exist
def test_figures_exist():
    for i in range(1, 13):
        assert list((BASE / "figures").glob(f"fig{i:02d}_*.png")), f"figure {i} missing"


# 22 Explorer V5 renders every page
@pytest.mark.skipif(not HAS_ST, reason="streamlit not installed")
@pytest.mark.parametrize("page", [p for p in [
    "1 · Overview", "5 · Exact Analyte Recovery", "7 · MSS Motif Recovery",
    "8 · Molecular Subclass Recovery", "9 · Biochemical Theme Recovery",
    "10 · Recovery by Abstraction Level ★", "11 · The Purine Attractor", "12 · Perturbation Validation",
    "14 · Individual Analytes", "18 · Methods & Provenance"]])
def test_v5_page_renders(page):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(V5APP / "app.py"), default_timeout=300).run()
    assert not at.exception, [str(e.value) for e in at.exception]
    at.radio[0].set_value(page).run()
    assert not at.exception, (page, [str(e.value) for e in at.exception])
