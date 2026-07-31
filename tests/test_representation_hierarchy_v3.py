"""Regression tests for the V3 representation-hierarchy analysis + Foundation Explorer V3.

Validates the new metrics (Spearman rank + top-k), the honesty guarantees (raw rank is
baseline-inflated; matrix prediction is weak; perturbation never fabricated), exact
reproducibility of V2, and page-by-page rendering of Explorer V3.

CI-safe: reads committed artifacts + the frozen engine; guards streamlit/engine imports.
"""
import sys
import json
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "results/v5_rebuild/representation_hierarchy_v3"
TABLES = BASE / "tables"
FIGS = BASE / "figures"
ARTIFACTS = BASE / "artifacts"
CANON_FP = "09ed804a40836f4a05a91ba10900cded"
V3 = REPO / "gaira_foundation_explorer_v3"
HAS_ST = importlib.util.find_spec("streamlit") is not None


@pytest.fixture(scope="module")
def metrics():
    p = TABLES / "per_analyte_hierarchy.csv"
    assert p.exists(), "run code/hierarchy_analysis.py"
    return pd.read_csv(p)


@pytest.fixture(scope="module")
def summary():
    return json.loads((ARTIFACTS / "hierarchy_summary.json").read_text())


@pytest.fixture(scope="module")
def cards():
    return json.loads((ARTIFACTS / "all_cards_v3.json").read_text())


# ── reproducibility + provenance ──
def test_reproduces_v2_exactly(summary):
    repro = summary["reproducibility_vs_v2"]
    assert all(abs(v) < 1e-6 for v in repro.values()), repro


def test_frozen_atlas_unchanged(summary):
    assert summary["atlas_fingerprint"] == CANON_FP
    sys.path.insert(0, str(REPO / "src"))
    from gaira.engine import GAIRAEngine
    assert GAIRAEngine().atlas.meta["fingerprint"] == CANON_FP


def test_metrics_have_all_layers(metrics):
    for c in ("L1_latent_fingerprint", "L2_mss_motif", "L3a_theme_raw", "L3b_theme_identity",
              "L4_theme_rank_rho", "L4_rank_null", "L4_rank_separation", "L5_top2_overlap",
              "L5_top3_overlap", "L6_argmax_agreement", "delta_purine"):
        assert c in metrics.columns
    assert len(metrics) == 51


# ── the hierarchy ──
def test_raw_agreement_rises_up_hierarchy(summary):
    L = summary["layers"]
    assert (L["L1_latent_fingerprint"]["median"] < L["L2_mss_motif"]["median"]
            < L["L3a_theme_raw"]["median"])


# ── NEW: theme rank (Spearman) is baseline-inflated too ──
def test_rank_is_baseline_inflated(metrics, summary):
    L = summary["layers"]
    # raw rank high...
    assert L["L4_theme_rank_rho"]["median"] > 0.8
    # ...but its separation from the null is tiny
    assert abs(L["L4_rank_separation"]["median"]) < 0.05
    # rank separation carries a slim identity edge (positive majority)
    assert 25 <= summary["rank_positive_separation"] <= 45


def test_rank_null_close_to_raw(metrics):
    # per-analyte the raw rho and its null are close (baseline)
    assert (metrics.L4_theme_rank_rho - metrics.L4_rank_null).abs().median() < 0.06


# ── NEW: top-k overlap ──
def test_topk_medians(summary):
    L = summary["layers"]
    assert 0.4 <= L["L5_top2_overlap"]["median"] <= 0.6
    assert 0.55 <= L["L5_top3_overlap"]["median"] <= 0.75   # ~2 of 3 retained


# ── purine attractor, quantified ──
def test_purine_attractor_and_delta(metrics, summary):
    assert (metrics.sers_dominant == "nucleic_purine").sum() == 50
    dp = summary["delta_purine"]
    assert dp["n_increase"] == int((metrics.delta_purine > 0).sum())
    assert dp["n_increase"] >= 30                      # majority gain purine


def test_delta_purine_anticorrelates_with_latent(metrics):
    from scipy.stats import pearsonr
    r, p = pearsonr(metrics.delta_purine, metrics.L1_latent_fingerprint)
    assert r < 0 and p < 0.05                           # significant negative


# ── matrix robustness: honest weak predictor ──
def test_matrix_prediction_is_weak(summary):
    reg = summary["matrix_regression"]["predictor_latent_fingerprint"]
    assert abs(reg["r"]) < 0.35 and reg["p_value"] > 0.05   # weak, not significant


# ── perturbation never fabricated ──
def test_only_three_perturbation_analytes(cards):
    validated = [a for a, c in cards.items()
                 if "perturbation validation" in c["layer8_interpretation"].lower()
                 and "no dynamic perturbation" not in " ".join(c["layer9_limitations"]).lower()]
    assert set(validated) == {"adenine", "ergothioneine", "urate"}


def test_untested_cards_state_not_measured(cards):
    for a, c in cards.items():
        if a not in ("adenine", "ergothioneine", "urate"):
            joined = " ".join(c["layer9_limitations"]).lower()
            assert "no dynamic perturbation" in joined


# ── cards: 9 layers, physics-aware language ──
def test_cards_nine_layers(cards):
    assert len(cards) == 51
    for c in cards.values():
        for k in ("layer1_latent_fingerprint", "layer2_mss_motif", "layer3_theme_cosine",
                  "layer4_theme_rank_correlation", "layer5_top3_overlap", "layer6_argmax_agreement",
                  "layer7_family", "layer8_interpretation", "layer9_limitations"):
            assert k in c


def test_language_is_not_binary(cards):
    # no card uses the forbidden binary "theme preserved/failed" framing
    for c in cards.values():
        txt = c["layer8_interpretation"].lower()
        assert "theme preserved" not in txt and "theme failed" not in txt


def test_adenine_card(cards):
    a = cards["adenine"]
    assert a["layer1_latent_fingerprint"]["component_cosine"] < 0.55
    assert "latent redistribution" in a["layer8_interpretation"].lower()
    assert "functional perturbation" in a["layer8_interpretation"].lower()


# ── figures + docs ──
def test_all_eight_figures_exist():
    for name in ["fig_h1_representation_hierarchy", "fig_h2_metric_comparison",
                 "fig_h3_family_heatmap", "fig_h4_topk_and_rank_null", "fig_h5_delta_purine",
                 "fig_h6_delta_purine_vs_component", "fig_h7_matrix_regression",
                 "fig_h8_perturbation_summary"]:
        assert (FIGS / f"{name}.png").exists(), name


def test_docs_exist():
    for name in ["HIERARCHY_METRICS_SPECIFICATION.md", "REPRESENTATION_HIERARCHY.md",
                 "INTERPRETATION_GUIDE.md", "CHANGELOG.md"]:
        assert (BASE / name).exists() and len((BASE / name).read_text()) > 500


# ── V1/V2 remain present (additive guarantee) ──
def test_v1_and_v2_apps_still_present():
    assert (REPO / "gaira_foundation_explorer/app.py").exists()
    assert (REPO / "gaira_foundation_explorer_v2/app.py").exists()


# ── Explorer V3 ──
def test_v3_data_layer():
    sys.path.insert(0, str(V3))
    from v3_core import data as D
    assert D.present() and len(D.metrics()) == 51 and D.CANON_FINGERPRINT == CANON_FP


@pytest.mark.skipif(not HAS_ST, reason="streamlit not installed")
@pytest.mark.parametrize("page_label", [
    "1 · Overview", "2 · Representation Hierarchy ★", "3 · L1 · Latent fingerprint",
    "4 · L2 · MSS motifs", "5 · L3 · Theme (raw + identity)", "6 · L3 · Theme rank ρ (NEW)",
    "7 · L3 · Top-k overlap (NEW)", "8 · L3 · Argmax agreement", "9 · The Purine Attractor",
    "10 · Cross-Modal Transfer", "11 · L4 · Perturbation", "12 · L5 · Matrix Robustness",
    "13 · Per-Analyte Cards", "14 · Framework & Methods", "15 · Verdict"])
def test_v3_page_renders(page_label):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(V3 / "app.py"), default_timeout=300).run()
    assert not at.exception, [str(e.value) for e in at.exception]
    at.radio[0].set_value(page_label).run()
    assert not at.exception, (page_label, [str(e.value) for e in at.exception])
