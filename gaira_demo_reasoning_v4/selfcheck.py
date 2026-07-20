"""Headless self-check for the V6 demo: import every page, run the engine, and
render every publication figure to PNG so visualisations can be audited without a
browser. Also verifies the frozen atlas fingerprint is untouched.

    python selfcheck.py            # renders to ./_selfcheck/
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT = HERE / "_selfcheck"
OUT.mkdir(exist_ok=True)


def main():
    from demo_core.engine_bridge import Bridge
    from demo_core import figures as F, data as D
    import matplotlib.pyplot as plt

    b = Bridge()
    s = b.platform_stats()
    fp = b.eng.atlas.meta["fingerprint"]
    from gaira.engine.versioning import VERSIONS
    assert fp == VERSIONS.atlas_fingerprint, "atlas fingerprint drift!"
    print(f"engine OK · atlas {fp[:12]}… · {s['n_reference_spectra']} spectra · "
          f"{s['n_mss_motifs']} MSS motifs")

    # a real reasoning trace
    Z, meta = D.load_projection("ils_adenine")
    hi = int(np.asarray(meta["conc_uM"], float).argmax())
    out, acts = b.bsv_and_mss(Z[hi], domain="buffer")
    print(f"adenine max-dose top motif: {[a.name for a in acts if not a.non_biochemical][0]}")

    figs = {
        "architecture": F.architecture_diagram(),
        "radar": F.radar(out.radar["axes"]),
        "mss_hierarchy": F.mss_hierarchy(acts),
        "fingerprint": F.component_fingerprint(out.bsv.component_coord, highlight=[3, 15]),
        "basis_c3": F.basis_spectrum(*b.basis_spectrum(3), bands=b.motif_by_id("purine_ring_breathing").bands_cm),
        "collisions": F.band_collision_map(b.mss.motifs),
        "dose": F.dose_response(np.asarray(meta["conc_uM"], float),
                                [b.infer(Z[i]).bsv.composition["nucleic_purine"] for i in range(len(Z))],
                                xlabel="adenine (uM)", ylabel="purine share"),
    }
    # ── Page 4 (Calibration) figures ──
    from demo_core import calibration as CAL
    names = {m.id: m.name for m in b.mss.motifs}
    s_ade = CAL.build_dose_series(D.calibration("adenine"), method=CAL.ADENINE_METHOD)
    evo = CAL.motif_evolution(b, s_ade, [m.id for m in b.mss.biochemical()])
    mean, rl, rs = CAL.theme_series(b, s_ade, "nucleic_purine")
    comp = CAL.component_series(b, s_ade)
    topj = list(int(x) for x in np.argsort(-(comp.max(0) - comp.min(0)))[:6])
    vecs = CAL.bsv_theme_vectors(b, s_ade.mean_coord); proj, var = CAL.trajectory_2d(vecs)
    J = CAL.joint_trajectories(b)
    U = CAL.uricase_conditions(b)
    ids = [m.id for m in b.mss.biochemical()]
    mb = {a.id: a.composition for a in b.bsv_and_mss(U["spiked"])[1]}
    ma = {a.id: a.composition for a in b.bsv_and_mss(U["spiked+uricase"])[1]}
    print(f"adenine purine ρ={CAL.spearman(rl, rs):.2f} · joint traj classes: "
          f"{[v['class'] for v in J.values()]}")
    figs.update({
        "cal_cascade": F.reasoning_cascade(b, s_ade.mean_coord[-1], f"{s_ade.levels[-1]:.1f} µM"),
        "cal_schematic": F.experimental_schematic("adenine", "cAg", "785"),
        "cal_mss_evolution": F.mss_evolution(s_ade.levels, evo, "purine_ring_breathing", names),
        "cal_dose_langmuir": F.dose_response_langmuir(s_ade.levels, mean, rl, rs,
                                                      CAL.langmuir_fit(rl, rs), "adenine",
                                                      "nucleic_purine", CAL.spearman(rl, rs)),
        "cal_component_evo": F.component_evolution(s_ade.levels, comp, topj),
        "cal_trajectory": F.trajectory_2d(proj, s_ade.levels, var),
        "cal_uricase_diff_mss": F.difference_bars([names[i] for i in ids],
                                                  [mb[i] for i in ids], [ma[i] for i in ids]),
        "cal_compare": F.compare_trajectories([
            {"name": "adenine", "proj": J["adenine"]["proj"], "color": F.T.PRIMARY, "marker": "o"},
            {"name": "ergothioneine", "proj": J["ergothioneine"]["proj"], "color": F.T.GOOD, "marker": "s"},
            {"name": "uricase", "proj": J["uricase"]["proj"], "color": F.T.UP, "marker": "^"}]),
    })

    # ── Page 5 (Serum Spike) figures ──
    from demo_core import serum as SER
    rdf = SER.load_recoverability()
    cdf = SER.confidence_recoverability(b, rdf)
    ex = ["hypoxanthine", "xanthine", "guanine", "ergothioneine", "adenine",
          "phenylalanine", "lactate", "glucose"]
    hmat, hthemes = SER.theme_delta_matrix(b, ex)
    print(f"serum tiers: {int((rdf.tier=='strong').sum())} strong / "
          f"{int((rdf.tier=='partial').sum())} partial / {int((rdf.tier=='poor').sum())} poor")
    figs.update({
        "p5_reco_cascade": F.recoverability_cascade(),
        "p5_reco_scatter": F.recoverability_scatter(rdf, annotate=["hypoxanthine", "phenylalanine", "lactate"]),
        "p5_reco_heatmap": F.recoverability_heatmap(ex, hmat, hthemes),
        "p5_confidence_limitation": F.confidence_limitation(cdf),
    })

    # ── Page 6 (Biological) figures ──
    from demo_core import biological as BIO
    if BIO.available():
        art = BIO.load("diabetes_plasma_ev_sers")
        if art is not None:
            gc = BIO.group_contrast(art)
            proj, var = BIO.pca_2d(art["themes_mat"])
            print(f"biological: {list(BIO.available().keys())} · diabetes top Δtheme "
                  f"{gc['rows'][0]['theme']} δ={gc['rows'][0]['cliffs_delta']:+.2f}")
            figs.update({
                "p6_forest": F.forest_plot(gc["rows"], gc["a"], gc["b"]),
                "p6_radar": F.multi_radar([{"name": g, "axes": BIO.group_radar_axes(art, g)}
                                           for g in art["groups"]]),
                "p6_pca": F.bio_pca(proj, art["group"], var),
                "p6_quality": F.group_quality(art),
                "p6_centroids": F.study_centroid_map(BIO.study_centroids()),
            })

    for name, fig in figs.items():
        p = OUT / f"{name}.png"
        fig.savefig(p, bbox_inches="tight"); plt.close(fig)
        print(f"  rendered {p.relative_to(HERE)}")

    # every page module imports and exposes render()
    from demo_core.pages import (p1_overview, p2_reference_atlas, p3_reasoning, p4_calibration,
                                  p5_serum, p6_biological, p7_dart, p8_methods)
    for m in (p1_overview, p2_reference_atlas, p3_reasoning, p4_calibration,
              p5_serum, p6_biological, p7_dart, p8_methods):
        assert hasattr(m, "render"), m.__name__
    print(f"all 8 pages import · figures in {OUT.relative_to(HERE)} · atlas untouched")


if __name__ == "__main__":
    main()
