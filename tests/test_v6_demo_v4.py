"""GAIRA V6 demo (gaira_demo_reasoning_v4) — guard tests.

The demo is presentation only: it must load the FROZEN engine, render its figures
from real engine outputs, and never mutate the atlas. Streamlit render() functions
need a script context and are exercised by selfcheck.py, not here; these tests cover
the engine bridge, the figures, and that all page modules import cleanly.
"""
import sys
import hashlib
from pathlib import Path
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
DEMO = REPO / "gaira_demo_reasoning_v4"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(DEMO))

FROZEN = REPO / "results/v5_rebuild/foundation/artifacts"
needs_art = pytest.mark.skipif(not (FROZEN / "manifold.json").exists(), reason="frozen atlas absent")
needs_st = pytest.mark.skipif(__import__("importlib").util.find_spec("streamlit") is None,
                              reason="streamlit not installed")


@needs_art
def test_bridge_loads_frozen_and_reports_stats():
    from demo_core.engine_bridge import Bridge
    from gaira.engine.versioning import VERSIONS
    b = Bridge()
    s = b.platform_stats()
    assert b.eng.atlas.meta["fingerprint"] == VERSIONS.atlas_fingerprint
    assert s["n_components"] == 24 and s["n_biochemical_themes"] == 11
    assert s["n_mss_motifs"] == 13 and s["n_reference_spectra"] == 375


@needs_art
def test_figures_render_from_real_engine_output():
    import matplotlib
    matplotlib.use("Agg")
    from demo_core.engine_bridge import Bridge
    from demo_core import figures as F, data as D
    b = Bridge()
    Z, meta = D.load_projection("ils_adenine")
    hi = int(np.asarray(meta["conc_uM"], float).argmax())
    out, acts = b.bsv_and_mss(Z[hi], domain="buffer")
    # centerpiece: adenine max-dose must put the purine motif on top by elevation
    bio = [a for a in acts if not a.non_biochemical]
    assert max(bio, key=lambda a: a.elevation).id == "purine_ring_breathing"
    for fig in (F.architecture_diagram(), F.radar(out.radar["axes"]), F.mss_hierarchy(acts),
                F.component_fingerprint(out.bsv.component_coord),
                F.band_collision_map(b.mss.motifs)):
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


@needs_art
def test_demo_does_not_mutate_atlas():
    from demo_core.engine_bridge import Bridge
    b = Bridge()
    W = np.load(FROZEN / "manifold_components.npz")["components"]
    disk_fp = hashlib.sha256(np.ascontiguousarray(W).tobytes()).hexdigest()[:32]
    b.bsv_and_mss(np.full(24, 1.0 / 24))
    assert b.eng.atlas.meta["fingerprint"] == disk_fp


@needs_art
@needs_st
def test_all_pages_import_and_expose_render():
    from demo_core.pages import (p1_overview, p2_reference_atlas, p3_reasoning, p4_calibration,
                                 p5_serum, p6_biological, p7_dart, p8_methods)
    for m in (p1_overview, p2_reference_atlas, p3_reasoning, p4_calibration,
              p5_serum, p6_biological, p7_dart, p8_methods):
        assert callable(getattr(m, "render", None)), m.__name__


# ── Page 4 (Calibration) — the strongest page ──
@needs_art
def test_calibration_adenine_monotonic_purine():
    """Adenine dose (recoverable cAg@785) must drive the purine theme monotonically up."""
    from demo_core.engine_bridge import Bridge
    from demo_core import calibration as CAL, data as D
    b = Bridge()
    s = CAL.build_dose_series(D.calibration("adenine"), method=CAL.ADENINE_METHOD)
    mean, rl, rs = CAL.theme_series(b, s, "nucleic_purine")
    assert CAL.spearman(rl, rs) > 0.5              # monotone increasing
    assert mean[-1] > mean[0]
    fit = CAL.langmuir_fit(rl, rs)
    assert fit is not None and fit[3] > 0.3        # a saturating fit exists


@needs_art
def test_calibration_ergothioneine_cleanest():
    """Ergothioneine → sulfur is the cleanest dose-response (near-monotone Langmuir)."""
    from demo_core.engine_bridge import Bridge
    from demo_core import calibration as CAL, data as D
    b = Bridge()
    s = CAL.build_dose_series(D.calibration("ergothioneine"))
    _, rl, rs = CAL.theme_series(b, s, "sulfur_antioxidant")
    assert CAL.spearman(rl, rs) > 0.85


@needs_art
def test_calibration_adenine_mss_evolution_purine_on_top():
    from demo_core.engine_bridge import Bridge
    from demo_core import calibration as CAL, data as D
    b = Bridge()
    s = CAL.build_dose_series(D.calibration("adenine"), method=CAL.ADENINE_METHOD)
    evo = CAL.motif_evolution(b, s, [m.id for m in b.mss.biochemical()])
    # purine motif has the largest elevation at the top dose of any biochemical motif
    top_dose = {m: v[-1] for m, v in evo.items()}
    assert max(top_dose, key=top_dose.get) == "purine_ring_breathing"


@needs_art
def test_uricase_depletes_oxopurine_specifically():
    """Uricase (urate knock-out) must drop the oxopurine motif more than the adenine
    purine-ring motif — the MSS layer resolves the specific depletion."""
    from demo_core.engine_bridge import Bridge
    from demo_core import calibration as CAL
    b = Bridge()
    cond = CAL.uricase_conditions(b)
    before, after = cond["spiked"], cond["spiked+uricase"]
    mb = {a.id: a.composition for a in b.bsv_and_mss(before)[1]}
    ma = {a.id: a.composition for a in b.bsv_and_mss(after)[1]}
    d_oxo = ma["oxopurine_carbonyl"] - mb["oxopurine_carbonyl"]
    d_pur = ma["purine_ring_breathing"] - mb["purine_ring_breathing"]
    assert d_oxo < 0                       # oxopurine falls on urate removal
    assert d_oxo < d_pur                   # and falls more than the adenine-type motif


@needs_art
def test_joint_trajectories_three_classes_render():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from demo_core.engine_bridge import Bridge
    from demo_core import calibration as CAL, figures as F, data as D
    b = Bridge()
    J = CAL.joint_trajectories(b)
    assert {v["class"] for v in J.values()} == {"redistribution", "scaling", "depletion"}
    s = CAL.build_dose_series(D.calibration("adenine"), method=CAL.ADENINE_METHOD)
    fig = F.reasoning_cascade(b, s.mean_coord[-1], "1.8 µM")   # the signature figure renders
    assert fig is not None
    plt.close(fig)


# ── Part 2/3: calibration radar + mechanism corrections ──
@needs_art
def test_calibration_radars_differ_across_doses():
    """Regression for the 'identical radars' report: different doses MUST produce
    different radar arrays and a non-zero, baseline-zero delta radar."""
    from demo_core.engine_bridge import Bridge
    from demo_core import calibration as CAL, data as D
    b = Bridge()
    s = CAL.build_dose_series(D.calibration("adenine"), method=CAL.ADENINE_METHOD)
    lo = np.array([a["score"] for a in b.infer(s.mean_coord[0]).radar["axes"]])
    hi = np.array([a["score"] for a in b.infer(s.mean_coord[-1]).radar["axes"]])
    assert not np.allclose(lo, hi)                       # not stale / cached
    assert np.abs(hi - lo).max() > 0.02
    # delta radar: exactly zero at baseline, non-zero at top dose
    d0 = CAL.radar_delta_axes(b, s.mean_coord[0], s.mean_coord[0])
    dN = CAL.radar_delta_axes(b, s.mean_coord[-1], s.mean_coord[0])
    assert all(abs(a["delta"]) < 1e-9 for a in d0)
    assert max(abs(a["delta"]) for a in dN) > 0.02


@needs_art
def test_calibration_delta_direction_and_mechanism():
    from demo_core.engine_bridge import Bridge
    from demo_core import calibration as CAL, data as D
    b = Bridge()
    ad = CAL.build_dose_series(D.calibration("adenine"), method=CAL.ADENINE_METHOD)
    er = CAL.build_dose_series(D.calibration("ergothioneine"))
    # target theme moves in the correct (positive) direction at top dose
    dpur = {a["theme"]: a["delta"] for a in CAL.radar_delta_axes(b, ad.mean_coord[-1], ad.mean_coord[0])}
    assert dpur["nucleic_purine"] > 0
    dsulf = {a["theme"]: a["delta"] for a in CAL.radar_delta_axes(b, er.mean_coord[-1], er.mean_coord[0])}
    assert dsulf["sulfur_antioxidant"] > 0
    # adenine REDISTRIBUTES more than ergothioneine SCALES (mechanism metric)
    Rad = CAL.redistribution_index(b, ad); Rer = CAL.redistribution_index(b, er)
    assert Rad[-1] > Rer[-1]
    assert CAL.scaling_metrics(b, er)["cos_to_baseline"][-1] > CAL.scaling_metrics(b, ad)["cos_to_baseline"][-1]


@needs_art
def test_uricase_delta_radar_purine_decreases():
    from demo_core.engine_bridge import Bridge
    from demo_core import calibration as CAL
    b = Bridge()
    c = CAL.uricase_conditions(b)
    d = {a["theme"]: a["delta"] for a in CAL.radar_delta_axes(b, c["spiked+uricase"], c["spiked"], "serum")}
    assert d["nucleic_purine"] <= 0                      # urate removal lowers purine


# ── Page 2 (Reference Atlas) ──
@needs_art
def test_page2_reference_map_and_sankey_from_frozen():
    from demo_core.engine_bridge import Bridge
    b = Bridge()
    rm = b.reference_map()
    assert rm["coords"].shape == (167, 24)
    assert abs(rm["coords"].sum(1).mean() - 1.0) < 1e-6            # L1 (engine convention)
    assert len(set(rm["families"])) > 5
    sk = b.sankey_links()
    assert len(sk["comp_nodes"]) == 24 and len(sk["theme_nodes"]) == 11
    # many-to-many: at least one component feeds >1 MSS motif (no false one-to-one)
    assert any(len(b.component_linked_motifs(j)) > 1 for j in range(24))
    # c3 educational case: purine is its TOP theme despite the old "sterol" label
    tw = b.component_theme_weights(3)
    assert tw[0]["theme"] == "nucleic_purine"


# ── Page 5 (Serum Spike) ──
@needs_art
def test_page5_tiers_from_validated_table():
    from demo_core import serum as S
    summ = S.recoverability_summary()
    assert summ["n_analytes"] == 53
    assert summ["n_strong"] + summ["n_partial"] + summ["n_poor"] == 53
    # strong tier is dominated by strong Ag adsorbers (oxopurines + ergothioneine)
    assert {"hypoxanthine", "xanthine", "ergothioneine"} <= set(summ["strong_analytes"])


@needs_art
def test_page5_failure_is_not_absence_and_confidence_limitation():
    from demo_core.engine_bridge import Bridge
    from demo_core import serum as S
    b = Bridge()
    df = S.load_recoverability()
    # phenylalanine is a failure (poor tier) — but it IS present (spike moves the state)
    phe = df[df.analyte == "phenylalanine"].iloc[0]
    assert phe.tier == "poor"
    ba = S.before_after(b, "phenylalanine")
    assert ba is not None                                          # present, just not recovered
    # the confidence limitation: strong-tier ≈ poor-tier confidence (does NOT track recovery)
    cdf = S.confidence_recoverability(b, df)
    gap = abs(cdf[cdf.tier == "strong"].confidence.mean() - cdf[cdf.tier == "poor"].confidence.mean())
    assert gap < 0.05


# ── Page 6 (Biological) ──
@needs_art
def test_page6_artifacts_are_genuine_v6_not_legacy():
    """Each stored theme vector must equal the live engine's BSV of the stored coord —
    proving the artifact is real V6 output, not a relabelled legacy radar."""
    from demo_core.engine_bridge import Bridge
    from demo_core import biological as B
    if not B.available():
        pytest.skip("biological artifacts not built")
    b = Bridge()
    art = B.load("diabetes_plasma_ev_sers")
    r = art["records"][0]
    bsv = b.infer(np.array(r["coord"]), domain=art["domain"]).bsv
    live = [round(float(bsv.composition[t]), 5) for t in art["theme_ids"]]
    assert np.allclose(live, r["themes"], atol=1e-4)


@needs_art
def test_page6_patient_level_and_absolute_vs_delta_separate():
    from demo_core import biological as B
    if not B.available():
        pytest.skip("biological artifacts not built")
    art = B.load("diabetes_plasma_ev_sers")
    assert art["aggregation"] == "patient"                        # no per-scan pseudoreplication
    assert art["n_units"] == 63 and art["n_by_group"] == {"Impact": 39, "Strong-D": 24}
    _, means = B.group_theme_means(art)                           # absolute atlas position
    gc = B.group_contrast(art)                                    # signed ΔBSV — a separate object
    assert set(means) == {"Impact", "Strong-D"}
    assert "delta" in gc["rows"][0] and "ci_lo" in gc["rows"][0]
    # the real patient-level finding: purine differs with FDR significance + large effect
    pur = next(r for r in gc["rows"] if r["theme"] == "nucleic_purine")
    assert pur["sig"] and abs(pur["cliffs_delta"]) > 0.5


@needs_art
def test_page6_no_demographics_leak():
    import json
    from demo_core.biological import ART, CONTRAST
    import re
    for key in CONTRAST:
        p = ART / f"{key}.json"
        if not p.exists():
            continue
        txt = p.read_text().lower()
        for tok in [r"hba1c", r"\bbmi\b", r"race", r"ethnic", r"gender", r"\bage\b",
                    r"weight_kg", r"height_cm", r"waist", r"patient_data", r"2151-"]:
            assert not re.search(tok, txt), f"{key} leaks {tok}"


@needs_art
def test_page6_unavailable_studies_fabricate_nothing():
    from demo_core.pages.p6_biological import DEFERRED
    from demo_core import biological as B
    for key in DEFERRED:
        assert B.load(key) is None                                # no artifact => no output


# ── Page 8 (Methods) & cross-page ──
@needs_art
def test_page8_fingerprint_resolves():
    from demo_core.engine_bridge import Bridge
    from gaira.engine.versioning import VERSIONS
    b = Bridge()
    s = b.platform_stats()
    assert s["fingerprint"] == VERSIONS.atlas_fingerprint == b.eng.atlas.meta["fingerprint"]


@needs_art
def test_related_links_are_valid_page_labels():
    import re
    from pathlib import Path
    app = (DEMO / "app.py").read_text()
    labels = set(re.findall(r'"(\d · [^"]+)"', app))
    for p in (DEMO / "demo_core/pages").glob("p*.py"):
        for call in re.findall(r"C\.related\(\[([^\]]+)\]\)", p.read_text()):
            for lab in re.findall(r'"([^"]+)"', call):
                assert lab in labels, f"{p.name}: invalid related link {lab}"
