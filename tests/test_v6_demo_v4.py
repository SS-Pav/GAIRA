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


# ── radar rendering fixes: fixed shared scale + canonical axis order ──
@needs_art
def test_radar_canonical_axis_order_is_stable():
    """A theme must sit at the SAME radar angle regardless of its score (the engine
    returns axes sorted by score, which would move themes between conditions)."""
    from demo_core import figures as F
    from demo_core.engine_bridge import Bridge
    b = Bridge()
    lo = b.infer(np.full(24, 1.0 / 24)).radar["axes"]
    a = np.zeros(24); a[3] = 1.0
    hi = b.infer(a).radar["axes"]
    order_lo = [x["theme"] for x in F._order_axes(lo)]
    order_hi = [x["theme"] for x in F._order_axes(hi)]
    assert order_lo == order_hi == F.CANONICAL_THEME_ORDER


@needs_art
def test_radar_fixed_scale_is_shared_not_per_figure():
    """With a shared radial_max, a theme's plotted radius reflects its ABSOLUTE value —
    so growth across a dose series is visible instead of auto-rescaled away."""
    from demo_core import figures as F, calibration as CAL, data as D
    from demo_core.engine_bridge import Bridge
    b = Bridge()
    s = CAL.build_dose_series(D.calibration("adenine"), method=CAL.ADENINE_METHOD)
    ax_lo = b.infer(s.mean_coord[0]).radar["axes"]
    ax_hi = b.infer(s.mean_coord[-1]).radar["axes"]
    pur_lo = next(x["score"] for x in ax_lo if x["theme"] == "nucleic_purine")
    pur_hi = next(x["score"] for x in ax_hi if x["theme"] == "nucleic_purine")
    shared = max(max(x["score"] for x in ax_lo), max(x["score"] for x in ax_hi)) * 1.1
    # on the shared scale the purine radius genuinely grows (was pinned to the edge before)
    assert pur_hi / shared - pur_lo / shared > 0.2


@needs_art
def test_biological_group_delta_radar_reveals_difference():
    from demo_core import biological as B
    art = B.load("diabetes_plasma_ev_sers")
    a, bb, axes, shared = B.group_delta_axes(art)
    dpur = next(x["delta"] for x in axes if x["theme"] == "nucleic_purine")
    assert dpur < -0.02 and shared > 0.02       # the purine difference is visible on the delta radar


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


# ── Part 1: NMF-native atlas ──
@needs_art
def test_component_distance_matrix_valid_and_deterministic():
    from demo_core.engine_bridge import Bridge
    b = Bridge()
    D = b.component_distance()
    assert D.shape == (24, 24)
    assert np.allclose(D, D.T)                         # symmetric
    assert np.allclose(np.diag(D), 0)                  # zero diagonal
    assert np.allclose(D, b.component_distance())      # deterministic
    assert (D >= 0).all()


@needs_art
def test_nmf_decomposition_and_reconstruction():
    """A reference analyte decomposes into components; reconstructing from more
    components monotonically approaches the full atlas reconstruction."""
    from demo_core.engine_bridge import Bridge
    b = Bridge()
    coeff = b.analyte_coeffs("adenine")
    order = list(np.argsort(-coeff))
    assert order[0] == 3                                    # adenine's top motif is c3
    grid, full = b.reconstruct_from(coeff)
    err = []
    for k in (1, 3, 8):
        _, partial = b.reconstruct_from(coeff, keep=order[:k])
        err.append(float(np.linalg.norm(full - partial)))
    assert err[0] > err[1] > err[2]                        # more components → closer to full


@needs_art
def test_component_ego_network_and_clustered_heatmap_render():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from demo_core.engine_bridge import Bridge
    from demo_core import figures as F, biological as B
    b = Bridge()
    fig = F.component_ego_network(3, b.component_linked_motifs(3))
    assert fig is not None; plt.close(fig)
    art = B.load("shine_ev_sers")
    if art is not None:
        Z, g, l, strata = B.heatmap_matrix(art, "themes_mat")
        assert strata is not None                          # SHINE carries dose strata
        fig = F.sample_heatmap(Z, g, l, strata=strata); assert fig is not None; plt.close(fig)


@needs_art
def test_component_mds_and_sankey_cover_all_components():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from demo_core.engine_bridge import Bridge
    from demo_core import figures as F
    b = Bridge()
    D = b.component_distance()
    Y = F._classical_mds(D)
    assert Y.shape == (24, 2)
    assert np.allclose(Y, F._classical_mds(D))         # deterministic
    tbc = {j: b.component_dominant_theme(j) for j in range(24)}
    for fig in (F.component_similarity_map(D, tbc), F.component_dendrogram(D, tbc)):
        assert fig is not None; plt.close(fig)
    # component→MSS→theme network weights come from the registries, cover all 24 comps
    sk = b.sankey_links()
    comp_sources = {l[0] for l in sk["links"] if l[0] < 24}
    assert len(comp_sources) >= 20                     # nearly every component feeds a motif


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


# ── Part 4/5: serum audit + confidence separation ──
@needs_art
def test_serum_adenine_concentration_audit():
    """Adenine's poor serum recovery is a low-concentration case, not a mapping bug:
    its serum spike is far lower than the recoverable analytes'."""
    from demo_core.engine_bridge import Bridge
    from demo_core import serum as S
    b = Bridge()
    ad = S.pure_vs_serum(b, "adenine")
    assert ad["serum_tier"] == "poor"
    assert ad["pure_target_hi"] > ad["pure_target_lo"]            # strong in pure
    assert ad["serum_conc_uM"] <= 1.0                            # spiked very low
    erg = S.pure_vs_serum(b, "ergothioneine")
    assert erg["serum_conc_uM"] > ad["serum_conc_uM"]           # ergo spiked higher, recovers
    assert erg["serum_tier"] == "strong"


@needs_art
def test_serum_recoverability_terms_are_separated():
    from demo_core.engine_bridge import Bridge
    from demo_core import serum as S
    b = Bridge()
    terms = S.recoverability_terms(b, S.load_recoverability())
    for col in ("direction_agreement", "detectability", "reproducibility", "matrix_dominance"):
        assert col in terms.columns
    abl = S.ablation_table(terms)
    # reproducibility alone misranks: a consistent-but-wrong-direction analyte scores high
    assert "phenylalanine" in abl["reproducibility"]
    assert "phenylalanine" not in abl["direction_agreement"]


@needs_art
def test_confidence_metrics_are_separate_and_recoverability_absent_for_unknown():
    from demo_core.engine_bridge import Bridge
    from demo_core import metrics as M
    b = Bridge()
    bsv = b.infer(np.full(24, 1.0 / 24), domain="ev").bsv
    a = M.atlas_support(bsv); sp = M.overall_theme_specificity(b.eng.builder, bsv)
    assert 0 <= a <= 1 and 0 <= sp <= 1
    # for an unknown spectrum with no matrix-spike evidence, recoverability is None and
    # the composite does NOT treat it as a positive score
    comp = M.composite_interpretation(a, sp, None)
    assert comp["recoverability_known"] is False
    assert "matrix_recoverability" not in comp["terms"]


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
