"""Section 5 — Calibration & Validation, restructured as the validation LADDER:

    Reference Raman → Pure Ag-SERS → Adenine → Ergothioneine → Uricase → Serum spike-ins

Each tab answers one scientific question. The Pure Ag-SERS tab is the new bridge — can a
Raman-trained atlas recognise pure Ag-SERS analytes before serum matrix effects? — and
reuses the frozen engine + committed pure-Ag-SERS validation artifact. The perturbation
tabs reuse the tested v4 calibration/serum reasoning.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import streamlit as st

_V4 = Path(__file__).resolve().parents[3] / "gaira_demo_reasoning_v4"
if str(_V4) not in sys.path:
    sys.path.insert(0, str(_V4))
from demo_core.engine_bridge import get_bridge
from demo_core import figures as F
from demo_core.pages import p4_calibration as CAL, p5_serum as SERUM

from .. import data as D, ui, theme as T

TIER_ORDER = ["Excellent", "Good", "Moderate", "Weak", "Poor"]


def render():
    ui.page_header(
        "The validation ladder", "Calibration & Validation",
        "Five rungs, from the cleanest test to the hardest. Each isolates one failure mode, "
        "so a result at rung 4 (serum) is the expected continuation of rung 2 (pure Ag-SERS), "
        "not a surprise.")
    ui.flow([("Reference Raman", "what it learns"), ("Pure Ag-SERS", "modality gap"),
             ("Perturbation", "concentration"), ("Serum matrix", "competition"),
             ("Biology", "next section")], highlight={1})

    tabs = st.tabs(["1 · Reference Raman", "2 · Pure Ag-SERS Validation", "3 · Adenine",
                    "4 · Ergothioneine", "5 · Uricase", "6 · Serum spike-ins"])
    b = get_bridge()
    with tabs[0]:
        _reference_raman()
    with tabs[1]:
        _pure_ag_sers(b)
    with tabs[2]:
        st.markdown("### Adenine — can the atlas recover concentration once transfer works?")
        CAL._adenine(b)
    with tabs[3]:
        st.markdown("### Ergothioneine — single-motif scaling")
        CAL._ergothioneine(b)
    with tabs[4]:
        st.markdown("### Uricase — selective enzymatic depletion")
        CAL._uricase(b)
    with tabs[5]:
        SERUM.render(b)


# ── tab 1 ──
def _reference_raman():
    r = D.validation().get("1_gobbato_raman", {})
    ui.question("What does the atlas learn?")
    ui.stat_row([(r.get("n_spectra"), "pure Raman spectra"), (r.get("n_analytes"), "analytes"),
                 (ui.fmt(r.get("mean_ood"), 3), "mean OOD (in-domain)"),
                 (ui.fmt(r.get("median_ood"), 3), "median OOD")])
    st.markdown(
        "The atlas is built from pure-compound **Raman** fingerprints. Projected back through "
        f"itself they sit in-distribution (mean OOD **{ui.fmt(r.get('mean_ood'),3)}**) with "
        "chemically-correct dominant themes (adenine→purine, glucose→saccharide, "
        "albumin→protein). This is the reference frame every rung below is measured against.")
    ui.takehome("The atlas represents its own pure-Raman chemistry faithfully — the necessary "
                "floor before asking whether it transfers to silver.")


# ── tab 2 · the new bridge ──
def _pure_ag_sers(b):
    v = D.pure_ag_sers()
    if not v:
        ui.note("caveat", "Pure Ag-SERS validation artifact not found. Run "
                          "<code>foundation_audit/code/pure_ag_sers_validation.py</code>.")
        return
    s = v["summary"]
    per = {p["analyte"]: p for p in v["per_analyte"]}
    ui.question("Can the Raman biochemical representation recognize pure Ag-SERS measurements "
                "— before any serum matrix effects?")
    ui.stat_row([
        (s["n_matched_to_raman"], "matched analytes"),
        (ui.fmt(s["median_coord_cosine"], 2), "median transfer cosine"),
        (f"{s['n_theme_preserved']}/{s['n_matched_to_raman']}", "theme preserved"),
        (ui.fmt(s["mean_sers_ood"], 2), "mean SERS OOD"),
        (f"{s['tier_counts'].get('Excellent',0)+s['tier_counts'].get('Good',0)}", "Excellent+Good"),
    ])

    ui.section("5.2a", "The headline — every analyte, ranked and tiered")
    ui.figure_card(
        "pure_ag_sers_ranking.png",
        question="Which pure analytes does the Raman atlas still recognise on silver?",
        method="Project each pure Ag-SERS analyte through the FROZEN atlas (no retraining, no "
               "modality correction); score the cosine between its Ag-SERS and Raman component "
               "coordinates; tier by that cosine.",
        result=f"Excellent {s['tier_counts'].get('Excellent',0)} · Good "
               f"{s['tier_counts'].get('Good',0)} · Moderate {s['tier_counts'].get('Moderate',0)} · "
               f"Weak {s['tier_counts'].get('Weak',0)} · Poor {s['tier_counts'].get('Poor',0)}. "
               f"Oxopurines and thiols top; pyrimidines and small amino acids bottom.",
        interpretation="Transfer is strongly chemistry-dependent — a property of Ag adsorption, "
                       "not of the model.",
        takehome_text="The Excellent/Good set is exactly the strong Ag chemisorbers; these are "
                      "the analytes that also survive serum (tab 6).")

    ui.section("5.2b", "Why — adsorption affinity by chemical family")
    ui.figure_card(
        "pure_ag_sers_by_family.png",
        question="Does chemical family predict transfer?",
        method="Mean Raman↔Ag-SERS coordinate cosine per chemical family.",
        result="Purines 0.66, sulfur cofactors 0.53, creatinine/urea 0.68 — strong. Amino acids "
               "0.39, pyrimidines 0.16 — weak/poor.",
        interpretation="Strong chemisorbers (ring-N purines, thiol/thione sulfur) preserve their "
                       "signature; weak physisorbers (sugars, most amino acids, pyrimidines) are "
                       "reshaped by surface selection rules.",
        takehome_text="Adsorption affinity — not the representation — sets the transfer ceiling.")

    ui.section("5.2c", "Per-analyte reasoning — Reference Raman → pure Ag-SERS")
    order = sorted(per, key=lambda a: -per[a]["coord_cosine"])
    a = st.selectbox("Analyte", order,
                     format_func=lambda x: f"{x} · {per[x]['recoverability_tier']} "
                                           f"(cos {per[x]['coord_cosine']:.2f})")
    p = per[a]
    rc = np.array(p["raman_coord"]); sc = np.array(p["sers_coord"])
    rb = b.infer(rc, "buffer").bsv
    sb = b.infer(sc, "buffer").bsv
    before = [{"theme": t, "score": float(rb.composition[t])} for t in b.bio_themes]
    after = [{"theme": t, "score": float(sb.composition[t])} for t in b.bio_themes]
    dax = [{"theme": t, "delta": float(sb.composition[t] - rb.composition[t])} for t in b.bio_themes]
    dmax = max(abs(x["delta"]) for x in dax) or 1e-3

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("**Component coordinates — Raman (dashed) vs Ag-SERS (solid)**")
        rmax = max(max(rc), max(sc)) * 1.1
        fig = F.difference_bars([f"c{j}" for j in np.argsort(-(rc + sc))[:8]],
                                [rc[j] for j in np.argsort(-(rc + sc))[:8]],
                                [sc[j] for j in np.argsort(-(rc + sc))[:8]],
                                title="Top components: Raman → Ag-SERS")
        st.pyplot(fig, use_container_width=True)
        st.markdown('<div class="small">Which latent motifs the Ag-SERS gains (red) or loses '
                    '(blue) vs the pure Raman.</div>', unsafe_allow_html=True)
    with c2:
        st.markdown("**Biochemical themes — absolute (Raman dashed, Ag-SERS solid)**")
        st.pyplot(F.radar(after, title=f"{a} BSV", ref_axes=before), use_container_width=True)

    c3, c4 = st.columns(2, gap="large")
    with c3:
        st.markdown("**ΔBSV radar (Ag-SERS − Raman)** — the shift on silver")
        st.pyplot(F.delta_radar(dax, dmax, f"{a}: Ag-SERS − Raman",
                                f"shared scale ±{dmax:.3f}"), use_container_width=True)
    with c4:
        st.markdown("**MSS motifs (Ag-SERS)** + interpretation")
        mss_txt = ", ".join(f"{m} ({w:.2f})" for m, w in p["top_mss"])
        near = ", ".join(f"{n} ({c:.2f})" for n, c in p["nearest_raman"][:3])
        st.markdown(
            f"- **Tier:** {p['recoverability_tier']} · transfer cosine **{p['coord_cosine']:.2f}**\n"
            f"- **Dominant theme:** Raman `{p['raman_theme']}` → SERS `{p['sers_theme']}` "
            f"({'preserved ✓' if p['theme_preserved'] else 'shifted ✗'})\n"
            f"- **Top Ag-SERS MSS:** {mss_txt}\n"
            f"- **Nearest Raman references:** {near}\n"
            f"- **OOD** {p['sers_ood']:.2f} · **confidence** {p['sers_confidence']:.2f}")
        tier = p["recoverability_tier"]
        msg = {
            "Excellent": "A strong Ag chemisorber — its Raman signature survives the surface almost "
                         "intact. Recoverable in serum too.",
            "Good": "Transfers well; the dominant biochemistry is preserved on silver.",
            "Moderate": "Partial transfer — the theme is roughly right but the coordinate has moved; "
                        "read with caution.",
            "Weak": "Weak adsorber — the Ag-SERS is substantially reshaped; identity is not reliably "
                    "recovered even without serum.",
            "Poor": "Physisorbs weakly; almost nothing of the Raman signature survives on silver. "
                    "Not recoverable — a measurement limit, not a model error.",
        }[tier]
        (ui.takehome if tier in ("Excellent", "Good") else lambda t: ui.note("caveat", t))(msg)

    ui.note("take",
            "Pure Ag-SERS transfer is partial and <b>adsorption-selective before any serum is "
            "added</b> — median cosine ~0.42, theme preserved for 18/51. This isolates the "
            "<b>modality</b> gap (Raman→silver) from the <b>matrix</b> gap (serum competition, tab "
            "6), so the serum results are the expected continuation, not a new failure.")
    ui.report_expander("PURE_AG_SERS_VALIDATION.md", "Read the full pure Ag-SERS validation")
