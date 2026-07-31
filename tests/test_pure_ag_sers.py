"""Regression tests: the pure Ag-SERS validation stage is wired into the Foundation
Explorer and reproduces from the frozen atlas. CI-safe (no raw data / SSD needed) — uses
the committed artifact + the frozen engine.
"""
import sys
import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
ART = REPO / "results/v5_rebuild/foundation_audit/tables/pure_ag_sers_validation.json"
EXPLORER = REPO / "gaira_foundation_explorer"
CANON_FP = "09ed804a40836f4a05a91ba10900cded"


@pytest.fixture(scope="module")
def artifact():
    assert ART.exists(), "pure_ag_sers_validation.json missing — run pure_ag_sers_validation.py"
    return json.loads(ART.read_text())


# ── the dataset / projection artifact ──
def test_pure_ag_sers_artifact_shape(artifact):
    s = artifact["summary"]
    assert s["n_matched_to_raman"] == 51
    assert s["n_sers_analytes"] == 53 and s["n_sers_spectra"] == 265
    assert 0.3 < s["median_coord_cosine"] < 0.6            # partial, adsorption-limited
    assert set(s["tier_counts"]) <= {"Excellent", "Good", "Moderate", "Weak", "Poor"}
    assert len(artifact["per_analyte"]) == 51


def test_pure_ag_sers_science_is_stable(artifact):
    per = {p["analyte"]: p for p in artifact["per_analyte"]}
    # strong Ag chemisorbers transfer well; weak physisorbers do not
    assert per["hypoxanthine"]["recoverability_tier"] in ("Excellent", "Good")
    assert per["xanthine"]["coord_cosine"] > 0.7
    assert per["uracil"]["coord_cosine"] < 0.2             # pyrimidine, poor adsorber
    assert per["hypoxanthine"]["theme_preserved"] is True
    # every per-analyte record carries the reasoning fields the demo renders
    for p in artifact["per_analyte"]:
        assert len(p["raman_coord"]) == 24 and len(p["sers_coord"]) == 24
        assert p["recoverability_tier"] in ("Excellent", "Good", "Moderate", "Weak", "Poor")
        assert "raman_theme" in p and "sers_theme" in p and "top_mss" in p


def test_ranking_figures_exist():
    figs = REPO / "results/v5_rebuild/foundation_audit/figures"
    assert (figs / "pure_ag_sers_ranking.png").exists()
    assert (figs / "pure_ag_sers_by_family.png").exists()


# ── the frozen engine actually generates components / MSS / themes / radars ──
def test_projection_generates_components_mss_themes(artifact):
    sys.path.insert(0, str(REPO / "gaira_demo_reasoning_v4"))
    sys.path.insert(0, str(REPO / "src"))
    from demo_core.engine_bridge import get_bridge
    from demo_core import figures as F
    b = get_bridge()
    assert b.eng.atlas.meta["fingerprint"] == CANON_FP     # frozen, unchanged
    p = next(x for x in artifact["per_analyte"] if x["analyte"] == "hypoxanthine")
    rc, sc = np.array(p["raman_coord"]), np.array(p["sers_coord"])
    rb, sb = b.infer(rc, "buffer").bsv, b.infer(sc, "buffer").bsv
    # themes generated
    assert len(b.bio_themes) == 11
    comp = {t: sb.composition[t] for t in b.bio_themes}
    assert abs(sum(comp.values())) > 0
    # MSS activations generated
    acts = b.mss.activate(sb)
    assert len(acts) >= 10
    # radars generate without error (absolute overlay + delta)
    before = [{"theme": t, "score": float(rb.composition[t])} for t in b.bio_themes]
    after = [{"theme": t, "score": float(sb.composition[t])} for t in b.bio_themes]
    dax = [{"theme": t, "delta": float(sb.composition[t] - rb.composition[t])} for t in b.bio_themes]
    assert F.radar(after, ref_axes=before) is not None
    assert F.delta_radar(dax, max(abs(x["delta"]) for x in dax) or 1e-3) is not None


# ── it is wired into the demo, with no regression of the calibration tabs ──
@pytest.mark.skipif(__import__("importlib").util.find_spec("streamlit") is None,
                    reason="streamlit not installed")
def test_section5_renders_with_pure_ag_sers_tab():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(EXPLORER / "app.py"), default_timeout=180).run()
    at.radio[0].set_value("5 · Calibration & Validation").run()
    assert not at.exception, [str(e.value) for e in at.exception]
    # the calibration sub-experiments still exist (no regression)
    from demo_core.pages import p4_calibration as CAL
    assert all(hasattr(CAL, f) for f in ("_adenine", "_ergothioneine", "_uricase"))


def test_data_loader_exposes_pure_ag_sers():
    sys.path.insert(0, str(EXPLORER))
    from explorer_core import data as D
    assert hasattr(D, "pure_ag_sers")
