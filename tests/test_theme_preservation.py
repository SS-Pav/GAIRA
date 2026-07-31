"""Regression tests for the multi-level cross-modal transfer analysis + Foundation Explorer V2.

Covers the science (the four levels), the honesty guarantees (raw theme cosine is a baseline
artifact; perturbation is never fabricated; the purine attractor is real), cross-consistency
with the pre-existing committed artifact, and page-by-page rendering of Explorer V2.

CI-safe: reads committed artifacts + the frozen engine; guards streamlit/engine imports.
"""
import sys
import json
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "results/v5_rebuild/pure_ag_sers_theme_preservation"
TABLES = BASE / "tables"
FIGS = BASE / "figures"
ARTIFACTS = BASE / "artifacts"
CANON_FP = "09ed804a40836f4a05a91ba10900cded"
V2 = REPO / "gaira_foundation_explorer_v2"

HAS_ST = importlib.util.find_spec("streamlit") is not None


@pytest.fixture(scope="module")
def metrics():
    p = TABLES / "per_analyte_transfer_metrics.csv"
    assert p.exists(), "run code/theme_preservation.py"
    return pd.read_csv(p)


@pytest.fixture(scope="module")
def summary():
    return json.loads((ARTIFACTS / "theme_preservation_summary.json").read_text())


@pytest.fixture(scope="module")
def cards():
    return json.loads((ARTIFACTS / "all_cards.json").read_text())


# ── 1. shape + provenance ──
def test_metrics_shape(metrics):
    assert len(metrics) == 51
    for col in ("component_cosine", "theme_cosine", "theme_cosine_distinct", "theme_null_mean",
                "theme_separation", "self_theme_rank", "mss_cosine", "dominant_theme_match",
                "raman_dominant", "sers_dominant", "quadrant"):
        assert col in metrics.columns


def test_summary_records_frozen_fingerprint(summary):
    assert summary["atlas_fingerprint"] == CANON_FP


def test_frozen_atlas_unchanged():
    sys.path.insert(0, str(REPO / "src"))
    from gaira.engine import GAIRAEngine
    assert GAIRAEngine().atlas.meta["fingerprint"] == CANON_FP


# ── 2. the four levels are DISTINCT metrics ──
def test_theme_cosine_exceeds_component_for_all(metrics):
    # the core methodological claim: theme and fingerprint are different, theme is higher
    assert (metrics.theme_cosine > metrics.component_cosine).all()


def test_preservation_rises_latent_to_mss_to_theme(summary):
    assert (summary["component_cosine"]["median"]
            < summary["mss_cosine"]["median"]
            < summary["theme_cosine_raw"]["median"])


# ── 3. the honesty guarantee: raw theme cosine is a BASELINE ARTIFACT ──
def test_raw_theme_cosine_is_baseline_inflated(summary):
    # raw ~0.9 but distinctive collapses, and self-rank is ~chance -> not real preservation
    assert summary["theme_cosine_raw"]["median"] > 0.85
    assert summary["theme_cosine_distinct"]["median"] < 0.30
    assert summary["self_theme_rank"]["median"] >= summary["self_theme_rank"]["chance_median"] - 5


def test_distinctive_preservation_is_selective(summary):
    # only a handful are self-nearest; a slim majority beat their own null
    assert summary["self_is_nearest_theme"] <= 10
    assert 0 < summary["theme_separation"]["positive_count"] <= 40


# ── 4. the purine attractor ──
def test_purine_attractor(metrics):
    assert (metrics.sers_dominant == "nucleic_purine").sum() == 50


def test_dominant_match_are_all_already_purine(metrics):
    # every 'match' is an analyte already purine-dominant in Raman (not per-analyte survival)
    matched = metrics[metrics.dominant_theme_match]
    assert len(matched) == int(metrics.dominant_theme_match.sum())
    assert (matched.raman_dominant == "nucleic_purine").all()


def test_agrees_with_existing_committed_artifact():
    old = json.loads((REPO / "results/v5_rebuild/foundation_audit/tables"
                      / "pure_ag_sers_validation.json").read_text())
    st = pd.Series([p["sers_theme"] for p in old["per_analyte"]]).value_counts()
    assert st.get("nucleic_purine", 0) == 50  # same attractor, independently
    assert sum(1 for p in old["per_analyte"] if p.get("theme_preserved")) == 18


# ── 5. perturbation layer — measured only, never fabricated ──
def test_perturbation_only_three_analytes():
    p = pd.read_csv(TABLES / "perturbation_sensitivity.csv")
    assert set(p.analyte) == {"adenine", "ergothioneine", "urate"}


def test_uricase_is_directional_not_dose():
    p = pd.read_csv(TABLES / "perturbation_sensitivity.csv").set_index("analyte")
    assert "directional" in p.loc["urate", "perturbation_kind"]
    assert "dose-response" in p.loc["adenine", "perturbation_kind"]


def test_untested_analytes_say_not_tested(cards):
    for a, c in cards.items():
        L3 = c["level3_perturbation_sensitivity"]
        if a not in ("adenine", "ergothioneine", "urate"):
            assert L3.get("status") == "Not tested" or L3.get("tested") is False


def test_adenine_is_the_hypothesis_case(cards):
    a = cards["adenine"]
    assert a["level1_latent_fingerprint"]["component_cosine"] < 0.55       # weak latent
    assert a["level2_biochemical_theme"]["dominant_theme_match"] is True   # theme dominant kept
    assert "dose-response" in a["level3_perturbation_sensitivity"]["perturbation_kind"]
    assert a["quadrant"].startswith("Q2")


# ── 6. matrix recoverability linkage ──
def test_matrix_linkage_has_serum_tiers():
    m = pd.read_csv(TABLES / "matrix_recoverability_linkage.csv")
    tested = m[m.serum_tested == True]
    assert set(tested.serum_recovery_tier.dropna()) <= {"strong", "moderate", "weak"}
    strong = set(tested[tested.serum_recovery_tier == "strong"].analyte)
    assert {"hypoxanthine", "xanthine"} <= strong  # oxopurines survive the matrix


# ── 7. figures + cards exist ──
def test_all_nine_figures_exist():
    for i, name in enumerate([
        "fig1_component_vs_theme", "fig2_theme_ranking", "fig3_theme_heatmap",
        "fig4_family_comparison", "fig5_redistribution_waterfalls",
        "fig6_dominant_theme_confusion", "fig7_preservation_vs_ood",
        "fig8_perturbation_sensitivity", "fig9_matrix_recoverability"], 1):
        assert (FIGS / f"{name}.png").exists(), name


def test_cards_cover_all_analytes_with_four_levels(cards):
    assert len(cards) == 51
    for c in cards.values():
        assert "level1_latent_fingerprint" in c
        assert "level2_biochemical_theme" in c
        assert "level3_perturbation_sensitivity" in c
        assert "level4_matrix_recoverability" in c


# ── 8. docs ──
def test_docs_exist_and_state_the_verdict():
    spec = (BASE / "METRICS_SPECIFICATION.md").read_text()
    assert "distinctive" in spec.lower() and "baseline" in spec.lower()
    assess = (BASE / "THEME_PRESERVATION_ASSESSMENT.md").read_text()
    assert "Partially supported" in assess
    fw = (REPO / "GAIRA_MULTI_LEVEL_VALIDATION_FRAMEWORK.md").read_text()
    assert "four levels" in fw.lower() or "four-level" in fw.lower()


# ── 9. Explorer V2 ──
def test_v2_data_layer_present():
    sys.path.insert(0, str(V2))
    from v2_core import data as D
    assert D.present()
    assert len(D.metrics()) == 51
    assert D.CANON_FINGERPRINT == CANON_FP


@pytest.mark.skipif(not HAS_ST, reason="streamlit not installed")
@pytest.mark.parametrize("page_label", [
    "1 · Overview", "2 · The Metric Problem", "3 · Cross-Modal Validation ★",
    "4 · The Purine Attractor", "5 · Theme Redistribution", "6 · MSS Motif Preservation",
    "7 · Perturbation Validation", "8 · Matrix Recoverability", "9 · Per-Analyte Cards",
    "10 · Framework & Methods", "11 · Verdict"])
def test_v2_page_renders(page_label):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(V2 / "app.py"), default_timeout=240).run()
    assert not at.exception, [str(e.value) for e in at.exception]
    at.radio[0].set_value(page_label).run()
    assert not at.exception, (page_label, [str(e.value) for e in at.exception])
