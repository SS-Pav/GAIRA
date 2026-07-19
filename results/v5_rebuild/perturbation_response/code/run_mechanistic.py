"""Part 15 mechanistic assessment + Part 16 DART implications + ontology confidence."""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path("/Users/surajpg/projects/GAIRA"); sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).parent))
import response_lib as RL

OUT = REPO / "results/v5_rebuild/perturbation_response"
TAB, ART = OUT / "tables", OUT / "artifacts"


def main():
    ctx = RL.load_atlas_context()
    tm = pd.read_csv(TAB / "part3_theme_match.csv")
    cons = pd.read_csv(TAB / "part5_analyte_consistency.csv")
    fam = json.loads((TAB / "part10_response_families.json").read_text())
    uric = json.loads((TAB / "part8_uricase.json").read_text())
    pur = pd.read_csv(TAB / "part6_purine_similarity.csv", index_col=0)
    rob = pd.read_csv(TAB / "part11_component_robustness.csv")
    p9 = pd.read_csv(TAB / "part9_serum_responders.csv")
    man = json.loads((ART / "response_audit_manifest.json").read_text())
    comp_load = RL.load_component_reference_loadings()

    # identity-match rate among strongest responders
    # per-analyte: in how many arms is the analyte's OWN encoding component the single
    # strongest responder?
    own0 = tm[(tm.effect_rank == 0) & tm.analyte_loads_this_component]
    adenine_arms = tm[tm.analyte == "adenine"].experiment.nunique()
    adenine_own0 = own0[own0.analyte == "adenine"].experiment.nunique()

    # ── ontology confidence updates ──
    # a component GAINS confidence if a perturbation of an analyte it encodes drives it;
    # a component's THEME LABEL loses confidence if the response contradicts it.
    gains, label_issues = [], []
    for exp in tm.experiment.unique():
        analyte = tm[tm.experiment == exp].analyte.iloc[0]
        strongest = tm[(tm.experiment == exp)].nsmallest(3, "effect_rank")
        for _, r in strongest.iterrows():
            if r.analyte_loads_this_component:
                gains.append({"component": int(r.component), "driven_by": analyte,
                              "effect_rank": int(r.effect_rank), "theme_label": r.theme,
                              "note": f"{analyte} drives c{int(r.component)} and loads it in the atlas — "
                                      f"identity confirmed"})
                if not r.theme_matches_analyte:
                    label_issues.append({"component": int(r.component), "current_label": r.theme,
                                         "driven_by": analyte,
                                         "reference_top_analytes": [n for n, _ in comp_load[int(r.component)][:4]],
                                         "issue": f"c{int(r.component)} responds to {analyte} and encodes it, "
                                                  f"but its audit theme is '{r.theme}' — the LABEL is unreliable "
                                                  f"(low purity), not the component."})

    assessment = {
        "1_loop_closed": (
            f"Partially, and at the level of component IDENTITY rather than theme label. For adenine, the "
            f"single strongest-responding component is c3 — whose top reference analyte IS adenine — in "
            f"{adenine_own0} of {adenine_arms} substrate x laser arms, a reproducible closure across "
            f"substrates and lasers. For ergothioneine the own-encoding component (c19) responds but is "
            f"not the strongest, and c19 is a low-purity generic component, so that closure is weak. "
            f"Overall, the analyte "
            f"loads its responding component in {int(tm.analyte_loads_this_component.sum())}/{len(tm)} "
            f"responsive-component instances, concentrated in the strongest responders."),
        "2_audit_predicts_response": (
            "The component AUDIT predicts response identity well but its THEME LABELS predict it poorly. "
            f"Theme-label agreement is only {int(tm.theme_matches_analyte.sum())}/{len(tm)}, because "
            "several low-purity components carry misleading coarse labels (c3 labelled 'sterol' actually "
            "encodes adenine; c13 labelled 'pyrimidine' is thymine-dominated). The underlying components "
            "are chemically real; the labels are the weak link."),
        "3_response_supports_ontology": (
            "It supports the ontology's HIGH-CONFIDENCE, chemically-focused entries and challenges the "
            "coarse labels on low-purity entries. The purine sub-structure recovered from perturbation "
            "(two anti-correlated pairs) is finer than the single 'purine' ontology axis."),
        "4_ontology_gains_confidence": sorted({g["component"] for g in gains}),
        "5_ontology_loses_confidence": sorted({li["component"] for li in label_issues}),
        "6_components_needing_reinterpretation": [
            {"component": li["current_label"] and li["component"], "current_label": li["current_label"],
             "driven_by": li["driven_by"], "reference_top": li["reference_top_analytes"]}
            for li in {li["component"]: li for li in label_issues}.values()],
        "purine_substructure": {
            "adenine_hypoxanthine_cos": float(pur.loc["adenine", "hypoxanthine"]),
            "xanthine_guanine_cos": float(pur.loc["xanthine", "guanine"]),
            "across_pair_cos": float(pur.loc["adenine", "xanthine"]),
            "reading": ("perturbation fingerprints split the purines into {adenine, hypoxanthine} "
                        "(6-amino/oxo) and {xanthine, guanine} (2,6-dioxo/oxo-amino), anti-correlated "
                        "across the divide — a chemically correct sub-classification the atlas was never "
                        "told about"),
        },
        "uricase_selectivity": {
            "purine_c15_change": uric.get("purine_component_c15_change"),
            "selective": uric.get("selective"),
            "reading": ("enzymatic urate depletion produces a SELECTIVE loss in a purine-encoding "
                        "component rather than a global change — mechanistically consistent with removing "
                        "a purine")
            if uric.get("selective") else "global change (not selective)",
        },
        "response_families_vs_spectra": (
            f"Response fingerprints recover chemical family better than raw spike spectra "
            f"(ARI {fam['response_fingerprint_best_ari']:.3f} vs {fam['raw_spectrum_best_ari']:.3f}), "
            "though both are modest — the response representation adds chemical structure that the raw "
            "background-dominated Ag-SERS spectra obscure."),
        "candidate_bsv_anchor_components": man["candidate_anchor_components"],
        "consistency_caveat": (
            f"Pure-vs-spike consistency is near zero (median cos {cons.consistency_cosine.median():+.3f}): "
            "an analyte does NOT in general activate the same components in serum as in buffer, so these "
            "fingerprints are matrix-specific and must not be treated as matrix-invariant signatures."),
    }
    (TAB / "part15_mechanistic_assessment.json").write_text(json.dumps(assessment, indent=2, default=float))

    # ── Part 16: DART implications (grounded) ──
    dart = {
        "premise": ("A DART experiment applies a controlled electrochemical perturbation and records a "
                    "time/potential series of spectra. Projected into the frozen atlas, each such series "
                    "is a TRAJECTORY of the same type this study built for chemical dose series."),
        "what_this_study_provides": [
            "A trajectory-fingerprint schema (path length, straightness, curvature, component turnover, "
            "response entropy, OOD evolution) already computed for 7 chemical dose trajectories — the "
            "first reference library a DART trajectory could be compared against.",
            "Evidence that dose trajectories are monotonic and saturating and that the strongest "
            "responding component often matches analyte identity — so a DART trajectory that redox-cycles "
            "a known species has a concrete, testable expectation (motion along that species' component).",
        ],
        "grounded_limitations": [
            "Every dataset here is Ag/Au-SERS projected into a Raman atlas and is out of domain; a DART "
            "measurement in yet another modality would be further out of domain still.",
            "Pure-vs-serum consistency is ~0, so a component response is matrix-specific; a DART "
            "trajectory in an electrochemical cell cannot be assumed to reuse buffer-derived fingerprints.",
            "Theme labels on low-purity components are unreliable; DART interpretation must anchor on "
            "component IDENTITY (reference loadings), not on the coarse theme.",
        ],
        "recommended_first_dart_test": (
            "Redox-cycle a single strong Ag adsorber with a known atlas component (e.g. a purine) and "
            "check whether the projected trajectory moves predominantly along that component and returns "
            "toward baseline on the reverse sweep — a direct falsifiable prediction."),
        "no_overclaim": ("This study does not demonstrate DART compatibility; it provides the trajectory "
                         "vocabulary and one falsifiable prediction for a future DART experiment."),
    }
    (TAB / "part16_dart_implications.json").write_text(json.dumps(dart, indent=2))

    print("=== MECHANISTIC ASSESSMENT ===")
    print("loop closed:", assessment["1_loop_closed"][:200], "…")
    print("gains confidence:", assessment["4_ontology_gains_confidence"])
    print("loses (label) confidence:", assessment["5_ontology_loses_confidence"])
    print("purine substructure:", assessment["purine_substructure"]["reading"][:120], "…")
    print("uricase:", assessment["uricase_selectivity"]["reading"][:100])


if __name__ == "__main__":
    main()
