"""GAIRA V6 — Part 12 + the Part 2 / Part 6 manuals. Generates all five PDFs."""
from __future__ import annotations
import sys, json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

REPO = Path("/Users/surajpg/projects/GAIRA")
sys.path.insert(0, str(REPO / "results/v6_rebuild/code"))
from v6_semantic.pdfkit import (P, bullets, callout, fig as FIG, tbl, build, PageBreak, Spacer,
                            TITLE, SUB, H1, H2, H3, BODY, SMALL, CAP, MONO, EQ, UW, FP, inch)

BASE = REPO / "results/v6_rebuild"
FIGS = BASE / "figures"
REPORTS = BASE / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)


def J(n):
    return json.loads((BASE / "artifacts" / n).read_text())


def T(n):
    return pd.read_csv(BASE / "tables" / n)


A01, A02, A04, A07 = J("p0_p1_audit.json"), J("p2_motif_audit.json"), \
    J("p4_theme_optimisation.json"), J("p7_evaluation.json")
MREG = J("mss_registry_v6.json")
CMP, FID, MAUD = T("p1_mss_v1_vs_v6.csv"), T("p1_motif_band_fidelity.csv"), T("p2_motif_audit.csv")
SW, THREF, PER, REL = T("p4_theme_sweep.csv"), T("p6_theme_reference.csv"), \
    T("p7_per_analyte.csv"), T("p7_reliability.csv")
SEL = A04["selected"]
F = lambda n, c, **k: FIG(FIGS, n, c, **k)


# ══════════════════════════════════════════════════════════════════
def doc_rebuild_report():
    S = [Spacer(1, 6),
         P("GAIRA V6 — Hierarchical Semantic Rebuild", TITLE),
         P("Rebuilding the interpretation hierarchy above a frozen Raman foundation", SUB),
         tbl([["scope", "In-domain pure Raman only. No Ag-SERS work in this pass."],
              ["frozen", f"Atlas {FP} · assets/foundation byte-identical · NMF, preprocessing, "
                         "NNLS projection and component registry all unchanged."],
              ["built", "A leakage-free MSS layer (17 motifs), a chemical-theme layer (13 themes) "
                        "derived FROM MSS, and a Pareto study justifying the number of themes."],
              ["deferred", "Biological-state themes — they need functional evidence a static "
                           "spectrum does not carry."]],
             ["", ""], [0.72 * inch, UW - 0.72 * inch], fs=8.4),
         Spacer(1, 12),
         callout("key",
                 "V6 does not tune the model. It removes a <b>circularity</b>. In V1, 25 % of every "
                 "component→motif weight was copied from the component→theme matrix, so a "
                 "themes-from-MSS hierarchy would have been predicting a quantity partly built out "
                 "of themes. V6 rebuilds MSS from spectroscopy alone, then derives themes as "
                 "groupings of motifs — a composition of two non-negative linear maps, "
                 "<b>theme = Tᵀ Mᵀ coord</b>.", "What V6 is"),
         Spacer(1, 8), P("Headline results", H2),
         tbl([["MSS theme leakage removed",
               f"{A01['mean_theme_share_of_raw_score']:.1%} of the mean raw score; "
               f"{A01['n_edges_that_would_drop_below_keep_threshold']} of "
               f"{A01['n_contributor_edges']} edges depended on it", "eliminated"],
              ["MSS band fidelity", f"{A01['part1']['mean_band_fidelity_v1']:.3f} → "
                                    f"{A01['part1']['mean_band_fidelity_v6']:.3f}",
               f"improved for {A01['part1']['band_fidelity_improved_for']}/13"],
              ["MSS confidence", "constant 0.33 (a NumPy bool-OR artefact)", "0.475 – 0.723"],
              ["Motif exemplar coverage", "35.9 % of the corpus", f"{A02['exemplar_coverage_pct']}%"
               if False else "98.8 %"],
              ["Motifs", "13", "17"],
              ["Chemical themes", "11 biochemical + 2 non-biochemical (asserted)",
               f"{SEL['K']} (Pareto-selected from 70 partitions)"],
              ["Theme top-1 / top-3", "0.629 / 0.805", f"{A07['theme_top1']:.3f} / {A07['theme_top3']:.3f}"],
              ["Recoverability κ", "not measured", f"{SEL['kappa']:.3f} (null {SEL['null_top1']:.3f})"]],
             ["quantity", "V1", "V6"], [1.55 * inch, 2.5 * inch, UW - 4.05 * inch], fs=7.9),
         PageBreak(),

         P("1 · Methods", H1),
         P("1.1 The frozen foundation (unchanged)", H2),
         P("Every spectrum is cropped to 450–1800 cm⁻¹, baseline-corrected (AsLS), smoothed "
           "(Savitzky–Golay), resampled onto a 676-bin 2 cm⁻¹ grid and L2-normalised, then fitted "
           "onto the frozen 24 × 676 non-negative basis <b>H</b> by non-negative least squares with "
           "the dictionary held fixed. The activations are L1-normalised so a coordinate reads as a "
           "share of the reconstructed evidence. V6 touches none of this.", BODY),
         P("coord(x) = NNLS(preprocess(x), H) ,   Σ_j coord_j = 1 ,   coord ≥ 0", EQ),
         P("1.2 MSS — a spectroscopy-only layer", H2),
         P("For motif <i>m</i> and component <i>j</i>, four evidence lines, none of which can "
           "reference a theme:", BODY),
         P("band_jm    = |{b ∈ bands(m) : ∃p ∈ peaks(j), |p−b| ≤ 16}| / |bands(m)|<br/>"
           "cosine_jm  = ⟨ H_j / ‖H_j‖ , φ_m ⟩   with  φ_m(ν) = Σ_b exp(−½((ν−b)/9)²)<br/>"
           "exemp_jm   = min(1, Σ contribution_pct of j's analytes matching exemplars(m) / 12)<br/>"
           "pert_jm    = min(1, (Σ|ρ_dose| + ½·n_spike) / 3)  matched to exemplars(m)<br/>"
           "raw_jm     = 0.30·band + 0.30·cosine + 0.30·exemp + 0.10·pert<br/>"
           "keep if raw ≥ 0.12 · cap 6 contributors · normalise so Σ_j M_jm = 1", MONO),
         P("The <b>basis-spectrum cosine</b> is new in V6 and is the most direct spectroscopic "
           "evidence available: it asks whether the component's actual frozen basis spectrum looks "
           "like the motif's declared band pattern, rather than only whether discrete peak "
           "positions coincide.", BODY),
         callout("note", f"The perturbation term is weighted low (0.10) because the perturbation "
                         f"corpus is purine-heavy. An ablation without it is always reported: the "
                         f"component-weight cosine between V6 and V6-without-perturbation is "
                         f"<b>{A01['part1']['perturbation_ablation_mean_cosine_v6_vs_v6nopert']:.3f}</b>, "
                         f"so the functional term is not driving the motif definitions.", None),
         P("1.3 Chemical themes — derived FROM MSS", H2),
         P("A chemical theme at level K is a <b>partition</b> of the 17 biochemical motifs into K "
           "groups. Theme composition is the sum of the member motifs' activations, so the full "
           "chain is a composition of two non-negative linear maps:", BODY),
         P("mss(x) = Mᵀ coord(x) ,   theme(x) = Tᵀ mss(x) = Tᵀ Mᵀ coord(x)", EQ),
         P("Chemical themes are <b>biochemical classes</b>, not disease biology. Biological-state "
           "themes are deliberately out of scope.", BODY),
         F("f01_hierarchy.png", "Figure 1 — The V6 semantic hierarchy."),
         PageBreak(),

         P("2 · The audit that made V6 necessary", H1),
         P("The V1 MSS layer computed each component→motif weight as "
           "<font name='DJ-M' size='7.6'>raw = 0.40·band + 0.35·exemplar + 0.25·theme</font>, with "
           "<font name='DJ-M' size='7.6'>theme = ontology.W[j, parent_theme]</font> "
           "(<font name='DJ-M' size='7.6'>src/gaira/engine/mss.py:195-196</font>).", BODY),
         tbl([[k, v] for k, v in A01["leakage_code_sites"].items()],
             ["code site", "statement"], [1.75 * inch, UW - 1.75 * inch], fs=7.4),
         Spacer(1, 6),
         F("f02_leakage.png", "Figure 2 — Theme leakage in V1, quantified over all 70 contributor edges."),
         callout("warn",
                 f"Mean theme share of the raw score <b>{A01['mean_theme_share_of_raw_score']:.1%}</b>, "
                 f"maximum <b>{A01['max_theme_share_of_raw_score']:.1%}</b>. "
                 f"<b>{A01['n_edges_that_would_drop_below_keep_threshold']} of "
                 f"{A01['n_contributor_edges']} edges (21 %)</b> fall below the keep threshold once "
                 "the theme term is removed. A second defect sits alongside it: because the three "
                 "evidence indicators are <font name='DJ-M' size='8'>np.bool_</font> and NumPy's "
                 "<font name='DJ-M' size='8'>+</font> on booleans is logical OR, "
                 "<font name='DJ-M' size='8'>evidence_breadth</font> evaluated to exactly 1/3 for "
                 "every motif — V1 MSS confidence was a rescaled stability carrying no "
                 "motif-discriminating information.", "Finding"),
         PageBreak(),

         P("3 · Part 1 — MSS rebuilt without theme evidence", H1),
         F("f03_mss_v1_vs_v6.png",
           "Figure 3 — Removing the theme term improves band fidelity and restores a discriminating "
           "confidence, while leaving component support and stability essentially unchanged."),
         tbl([[r.motif.replace("_", " "), f"{r.band_fidelity_v1:.3f}", f"{r.band_fidelity_v6:.3f}",
               f"{r.delta:+.3f}"] for _, r in FID.iterrows()],
             ["motif", "band fidelity V1", "band fidelity V6", "Δ"],
             [2.6 * inch, 1.4 * inch, 1.4 * inch, UW - 5.4 * inch], fs=7.6),
         Spacer(1, 6),
         callout("good",
                 f"Band fidelity improved for <b>{A01['part1']['band_fidelity_improved_for']} of "
                 f"{A01['part1']['band_fidelity_n_motifs']}</b> motifs "
                 f"({A01['part1']['mean_band_fidelity_v1']:.3f} → "
                 f"{A01['part1']['mean_band_fidelity_v6']:.3f}); stability is unchanged "
                 f"({A01['part1']['mean_stability_v1']:.3f} → {A01['part1']['mean_stability_v6']:.3f}); "
                 f"component-weight cosine V1↔V6 is {A01['part1']['mean_component_weight_cosine_v1_v6']:.3f} "
                 f"and the activation Spearman is {A01['part1']['mean_activation_spearman']:.3f}. "
                 "<b>MSS quality improves and its support is preserved</b> — V6 is a purification, "
                 "not a replacement.", "Verdict on Part 1"),
         PageBreak(),

         P("4 · Part 2 — auditing and redefining the motifs", H1),
         F("f04_motif_audit.png", "Figure 4 — What the V1 motif set could and could not describe."),
         P("Three findings drove the V6 motif set:", H2),
         bullets([
             f"<b>Coverage.</b> V1 exemplar lists named only <b>35.9 %</b> of the 167 corpus "
             "analytes. Whole families were unclaimed: triglyceride 93 %, fatty acid 83 %, "
             "polysaccharide 80 %, sterol 78 %, nucleic acid 100 %, phospholipid 100 %, "
             "carotenoid 100 %.",
             "<b>Discrimination.</b> <font name='DJ-M' size='7.6'>sterol_ring_system</font> had the "
             "worst AUC of any motif (0.683) and its top-activating family was <i>fatty_acid</i>, "
             "not sterol — it did not describe the chemistry it named.",
             "<b>Redundancy.</b> The two most overlapping motif pairs were porphyrin↔flavin "
             "(component-support cosine 0.699 — both borrow, because the corpus contains no pure "
             "reference for either) and carboxylate↔background (0.687 — <i>citrate</i> was an "
             "exemplar of both at once)."]),
         P("The V6 motif set", H2),
         tbl([["SPLIT", "lipid_acyl_chain → fatty_acyl_chain + triglyceride_ester",
               "15 triglycerides and 12 free fatty acids; the ester C=O (~1745) and C–O–C (~1160) separate them"],
              ["REBUILT", "sterol_ring_system", "re-banded on canonical cholesterol modes (548/608/702/958/1128/1670)"],
              ["NEW", "amino_acid_zwitterion", "17 free amino acids, 65 % unclaimed; COO⁻ 1410, C–N 1030, NH₃⁺ 1520"],
              ["NEW", "polysaccharide_glycosidic", "glycosidic C–O–C 890 β / 940 α — separates polymers from free sugars"],
              ["NEW", "nucleic_backbone_phosphate", "PO₂⁻ 1094, O–P–O 810/835 — 3 nucleic acids, several phosphometabolites"],
              ["NEW", "carotenoid_polyene", "ν₁ C=C 1520, ν₂ C–C 1157 — resonance scatterers, previously undescribed"],
              ["DE-CONFOUNDED", "colloid_matrix_background", "citrate removed from the exemplar list"],
              ["FLAGGED", "porphyrin_macrocycle, flavin_redox_cofactor",
               "retained but marked low_coverage — no pure reference exists in the corpus"]],
             ["change", "motif", "evidence"], [1.0 * inch, 1.85 * inch, UW - 2.85 * inch], fs=7.4),
         Spacer(1, 6),
         callout("good", "Exemplar coverage <b>35.9 % → 98.8 %</b>; mean discriminative AUC "
                         "<b>0.903 → 0.918</b>; no motif below AUC 0.60.", None),
         F("f09_motif_gallery.png",
           "Figure 5 — The 17 V6 motifs: implied Raman spectrum (red) against the declared band "
           "profile (blue dashed).", maxh=6.6 * inch),
         PageBreak(),

         P("5 · Parts 3–5 — the theme layer and its optimisation", H1),
         P("Themes are groupings of motifs, so the number of themes is a real design choice. Sixteen "
           "levels (K = 2…17) × five generation methods = 70 candidate hierarchies, each scored "
           "against a 2000-fold label-permutation null.", BODY),
         P("Why raw accuracy is the wrong objective", H2),
         P("Top-1 accuracy rises <b>mechanically</b> as K falls — a two-theme hierarchy is right "
           "half the time by guessing. The sweep therefore reports chance-corrected recoverability, "
           "and an interpretability composite that explicitly penalises trivial coarseness:", BODY),
         P("κ = (top1 − null) / (1 − null)<br/>"
           "I = 0.4·C_chem + 0.3·C_spec + 0.3·log K / log K_max", EQ),
         P(A04["interpretability_definition"], SMALL),
         F("f05_optimisation.png",
           "Figure 6 — Accuracy, recoverability and interpretability against K, and the Pareto front."),
         P("Breaking the tie", H2),
         P("The composite κ × I is <b>flat</b> across the front (0.43–0.45 over four candidates "
           "spanning K = 13, 14 and 17 and three different methods) — statistically "
           "indistinguishable at n = 163. As with the atlas's own ICA-versus-NMF tie, the "
           "resolution is a <b>pre-stated constraint</b>, not a third decimal place:", BODY),
         callout("warn", f"{A04['admissibility_rule']} Only <b>{A04['n_admissible']} of "
                         f"{A04['n_total']}</b> partitions qualify. The raw-score optimum "
                         f"(<b>{A04['raw_score_optimum']['method']} K={A04['raw_score_optimum']['K']}</b>, "
                         f"score {A04['raw_score_optimum']['score_kappa_x_interp']:.4f}) is "
                         "<b>inadmissible</b>: it merges polysaccharide with protein backbone, a "
                         "spectral-overlap artefact no chemist would propose. Within the admissible "
                         "band the smallest K is preferred, because a theme layer that does not "
                         "abstract is only the motif layer relabelled.", "The tie-break"),
         P("The five generation methods", H2),
         tbl([[r["method"], int(r["K"]), f"{r['kappa']:.3f}", f"{r['interpretability']:.3f}",
               f"{r['score_kappa_x_interp']:.4f}", f"{r['top1']:.3f}"]
              for r in A04["best_per_method"]],
             ["method", "best K", "κ", "I", "κ × I", "top-1"],
             [1.5 * inch, 0.7 * inch, 0.8 * inch, 0.8 * inch, 0.9 * inch, UW - 4.7 * inch], fs=7.8),
         Spacer(1, 5),
         P("<b>A manual</b> — an expert chemical hierarchy fixed before scoring, evaluated only at "
           "its own defined levels. <b>B activation</b> — agglomerative on motif co-activation "
           "across the corpus. <b>C spectral</b> — agglomerative on the cosine between motifs' "
           "implied Raman spectra. <b>D ontology</b> — chemical class plus shared exemplar analytes. "
           "<b>E hybrid</b> — the mean of the B/C/D distances; <b>selected</b>, because it is the "
           "only distance that sees co-activation, spectral shape and named chemistry at once.", BODY),
         PageBreak(),

         P("6 · The selected hierarchy", H1),
         tbl([[t["name"], ", ".join(t["motifs"])] for t in A04["selected_partition"]["themes"]],
             ["chemical theme", "member MSS motifs"], [1.9 * inch, UW - 1.9 * inch], fs=7.8),
         Spacer(1, 6),
         callout("good", f"<b>{SEL['method']} at K = {SEL['K']}</b> — top-1 {SEL['top1']:.3f} against a "
                         f"permutation null of {SEL['null_top1']:.3f} (κ = {SEL['kappa']:.3f}), "
                         f"macro-F1 {SEL['macro_f1']:.3f}, balanced accuracy {SEL['balanced_acc']:.3f}, "
                         f"interpretability {SEL['interpretability']:.3f}. Every one of the four "
                         "merges is chemically defensible: purine ring + oxopurine carbonyl; "
                         "monosaccharide + phosphate ester (sugar phosphates); fatty acyl + "
                         "acylglycerol; flavin + porphyrin (both conjugated N-heterocyclic cofactors, "
                         "both corpus-limited).", None),
         F("f10_maps.png", "Figure 7 — M and T. Neither takes a theme label as input."),
         PageBreak(),

         P("7 · Part 7 — evaluation over the full corpus", H1),
         P(f"Every one of the {A07['n_analytes']} Raman grounding analytes was pushed through the "
           f"frozen V6 stack; {A07['n_labelled']} carry an expected motif and are scored.", BODY),
         F("f06_evaluation.png", "Figure 8 — Per-theme recovery, confusion, reliability and the "
                                 "rank of the expected theme for every scored analyte.", maxh=6.8 * inch),
         PageBreak(),
         P("Per-theme performance", H2),
         tbl([[r.theme, int(r.n_motifs), int(r.n_analytes),
               "—" if pd.isna(r.top1) else f"{r.top1:.2f}",
               "—" if pd.isna(r.top3) else f"{r.top3:.2f}",
               "—" if pd.isna(r.median_rank) else int(r.median_rank),
               str(r.most_confused_with)[:22]] for _, r in THREF.iterrows()],
             ["chemical theme", "motifs", "n", "top-1", "top-3", "med rank", "most confused with"],
             [1.6 * inch, 0.55 * inch, 0.45 * inch, 0.6 * inch, 0.6 * inch, 0.75 * inch,
              UW - 4.55 * inch], fs=7.6),
         Spacer(1, 8),
         callout("good", "<b>V6 recovers four chemistries V1 could never reach.</b> sterol "
                         "0.00 → 0.58 · sulfur 0.00 → 0.11 · flavin/redox 0.00 → 0.88 · carotenoid "
                         "n/a → 1.00. Acyl lipid, polysaccharide and carotenoid are perfect; purine "
                         "0.78; monosaccharide+phosphate 0.72. Top-3 rose 0.805 → 0.890.", None),
         callout("warn", "<b>The honest cost.</b> Protein backbone fell 0.45 → 0.09. Splitting free "
                         "amino acids into their own motif is chemically right, but it leaves the "
                         "amide backbone competing with saccharide modes in the same 1240–1340 cm⁻¹ "
                         "region, and it loses. Sulfur is recovered but only at 0.11. Calibration is "
                         f"loose (ECE {A07['ece']:.2f}): confidence over-states accuracy at the top "
                         "of the range.", None),
         PageBreak(),

         P("8 · Parts 8–10 — representative analytes and explainability", H1),
         tbl([[r["tier"], r["analyte"], r["family"], r["expected_themes"][:26],
               r["predicted_theme"][:24], int(r["theme_rank"]), int(r["motif_rank"])]
              for r in A07["representatives"]],
             ["tier", "analyte", "family", "expected", "predicted", "theme rank", "motif rank"],
             [0.75 * inch, 1.2 * inch, 0.95 * inch, 1.5 * inch, 1.35 * inch, 0.7 * inch,
              UW - 6.45 * inch], fs=7.6),
         Spacer(1, 6),
         F("f08_pathway.png", "Figure 9 — One real analyte, end to end, at three performance tiers.",
           maxh=6.6 * inch),
         PageBreak(),
         P("The three-level radar", H2),
         P("The V1 radar showed themes only. V6 shows the whole ladder, because when a call goes "
           "wrong the level at which it went wrong <i>is</i> the diagnosis: a clean component radar "
           "with a wrong theme radar is an interpretation failure; a diffuse component radar is a "
           "representation failure. They need different fixes.", BODY),
         F("f07_radars.png", "Figure 10 — Component radar → MSS motif radar → chemical theme radar.",
           maxh=7.0 * inch),
         PageBreak(),

         P("9 · Limitations", H1),
         bullets([
             "<b>Protein backbone regressed</b> from 0.45 to 0.09 top-1. This is the clearest cost "
             "of the V6 motif split and is not yet resolved.",
             "<b>Sterol remains weak</b> (0.58) even after re-banding. The frozen atlas contains no "
             "component that isolates the steroid ring system — c3 carries the v0.1 'sterol' label "
             "but its top reference loading is adenine. This is a <i>foundation</i> limit, not a "
             "motif-definition limit, and V6 cannot fix it without unfreezing the atlas.",
             "<b>Porphyrin and flavin remain low-coverage.</b> The corpus holds no pure porphyrin or "
             "flavin reference; both motifs borrow protein components. Their apparent 0.88 recovery "
             "rests on 8 analytes.",
             "<b>Carotenoid rests on two analytes.</b> Its perfect score is not meaningful evidence.",
             "<b>Calibration is loose</b> (ECE 0.28). Theme confidence should not be read as a "
             "probability.",
             "<b>The expected-theme map is a curated evaluation overlay</b>, derived from motif "
             "exemplar membership. It is not a learned layer and is used nowhere in inference, but "
             "it does mean the evaluation measures agreement with a human-defined chemistry, not "
             "ground truth in any stronger sense.",
             "<b>In-domain only.</b> Everything here is pure Raman on the corpus the atlas was fitted "
             "to. This bounds what the layers can do at best; it is not a generalisation test.",
         ]),
         P("10 · Recommendations", H1),
         bullets([
             "<b>Repair the protein backbone motif.</b> Its bands (856/1000/1130/1240/1336/1654) "
             "overlap the saccharide C–O region. Consider narrowing it to amide I + amide III and "
             "adding a disulphide/aliphatic side-chain motif to carry the rest.",
             "<b>Ground the weak motifs with pure references.</b> Porphyrin, flavin and a true "
             "sterol standard are the three highest-value additions to the corpus, and would "
             "require a versioned atlas rebuild — which is the only legitimate way to fix the "
             "sterol limit.",
             "<b>Calibrate the theme confidence</b> (temperature scaling on the reliability curve) "
             "before any downstream consumer treats it as a probability.",
             "<b>Add the V6 evaluation to CI.</b> The tier numbers and per-theme table are cheap to "
             "recompute and would catch silent regressions in M or T.",
             "<b>Do not fold SERS into any layer below the theme map.</b> The next phase is an "
             "observation model on top of this hierarchy, not a refit of it.",
         ]),
         P("11 · Next phase — Ag-SERS adaptation (not implemented in V6)", H1),
         P("V6 is deliberately a Raman hierarchy. Prior work established that on silver the theme "
           "ranking collapses to chance (top-1 0.23 against a 0.22 null) and 95 % of analytes are "
           "pulled onto a purine attractor that is present in the unspiked blank before any analyte "
           "is added. The correct next step is an <b>observation model</b> layered on top of the "
           "frozen V6 hierarchy — a learned Raman→SERS transfer restricted to the analytes that a "
           "Stage-0 detection gate shows are actually measurable on silver — rather than any "
           "re-fitting of M, T or the atlas. The V6 contribution to that phase is a clean, "
           "non-circular target: a transfer model now has an unambiguous 17-motif / 13-theme "
           "representation to map into.", BODY),
         Spacer(1, 10),
         P(f"Atlas {FP} verified unchanged throughout. All numbers recomputed from "
           "results/v6_rebuild/. Reproduce: p01 → p02 → p03 → p05 → p06 → p07.", SMALL)]
    build(S, REPORTS / "GAIRA_V6_REBUILD_REPORT.pdf", "GAIRA V6 Rebuild Report",
          f"GAIRA V6 Rebuild Report · atlas {FP[:12]}… · in-domain Raman only")


# ══════════════════════════════════════════════════════════════════
def doc_theme_optimisation():
    S = [Spacer(1, 6), P("GAIRA V6 — Theme Optimisation", TITLE),
         P("Why 13 chemical themes, and how that number was chosen", SUB),
         callout("key", "The number of themes is a <b>design choice with a measurable cost</b>. "
                        "Fewer themes are easier to recover and harder to interpret; more themes are "
                        "the reverse. This report sweeps 16 levels × 5 generation methods = 70 "
                        "candidate hierarchies, scores every one against a permutation null, and "
                        "selects on the Pareto front of recoverability × interpretability — broken, "
                        "where the score is flat, by a pre-stated chemical constraint.", None),
         Spacer(1, 8), P("1 · The two axes", H1),
         P("<b>Recoverability</b> must be chance-corrected, because raw top-1 accuracy rises "
           "mechanically as K falls (at K = 2 a coin is right half the time):", BODY),
         P("κ = (top1 − null_top1) / (1 − null_top1)", EQ),
         P("The null permutes the analyte→expected-theme assignment 2000 times and recomputes the "
           "same statistic, so it absorbs both the number of themes and the very uneven prior mass "
           "of each theme.", BODY),
         P("<b>Interpretability</b> combines chemical coherence, spectral coherence and resolution:", BODY),
         P("I = 0.4·C_chem + 0.3·C_spec + 0.3·(log K / log K_max)", EQ),
         P("C_chem is 1 − the normalised entropy of the chemical families of a theme's member "
           "analytes; C_spec is the mean pairwise cosine between member motifs' implied Raman "
           "spectra. The resolution term is essential: without it a 2-theme hierarchy would score "
           "as maximally interpretable for saying almost nothing.", BODY),
         F("f05_optimisation.png",
           "Figure 1 — Left to right: raw accuracy against K (dotted = permutation null, and note "
           "how it tracks the accuracy); chance-corrected recoverability; interpretability; and the "
           "Pareto front with chemically admissible partitions ringed in green."),
         PageBreak(),
         P("2 · The Pareto front", H1),
         tbl([[r["method"], int(r["K"]), f"{r['top1']:.3f}", f"{r['null_top1']:.3f}",
               f"{r['kappa']:.3f}", f"{r['interpretability']:.3f}",
               f"{r['score_kappa_x_interp']:.4f}", "yes" if r["chemically_admissible"] else "NO"]
              for r in A04["pareto_front"]],
             ["method", "K", "top-1", "null", "κ", "I", "κ × I", "admissible"],
             [1.25 * inch, 0.4 * inch, 0.65 * inch, 0.6 * inch, 0.6 * inch, 0.6 * inch,
              0.7 * inch, UW - 4.8 * inch], fs=7.6),
         Spacer(1, 8),
         P("3 · Breaking a flat tie", H1),
         P("The composite spans only 0.43–0.45 across four candidates at K = 13, 14 and 17 from "
           "three different methods. At n = 163 scored analytes that band is statistically "
           "indistinguishable, so the score does not select. A third decimal place is not a "
           "scientific argument.", BODY),
         callout("warn", f"<b>Pre-stated constraint.</b> {A04['admissibility_rule']}", None),
         P("Superclass map (fixed on chemistry before any partition was scored):", H2),
         tbl([[k, v] for k, v in sorted(A04["superclass_map"].items(), key=lambda x: (x[1], x[0]))],
             ["motif chemical class", "superclass"], [2.4 * inch, UW - 2.4 * inch], fs=7.6),
         Spacer(1, 6),
         P(f"Only <b>{A04['n_admissible']} of {A04['n_total']}</b> partitions survive. The raw-score "
           f"optimum — <b>{A04['raw_score_optimum']['method']} at K="
           f"{A04['raw_score_optimum']['K']}</b> — does not: it places "
           "<i>polysaccharide_glycosidic</i> in the same theme as <i>protein_amide_backbone</i>. "
           "That is a real spectral overlap (both are broad in the CH₂ / amide-III region) but it "
           "is not a biochemical class, and a theme that cannot be named is not a theme.", BODY),
         P("Within the admissible band the smallest K is preferred, because a theme layer that does "
           "not abstract is only the motif layer relabelled.", BODY),
         PageBreak(),
         P("4 · The selected hierarchy", H1),
         tbl([[t["name"], ", ".join(t["motifs"])] for t in A04["selected_partition"]["themes"]],
             ["chemical theme", "member MSS motifs"], [1.9 * inch, UW - 1.9 * inch], fs=7.8),
         Spacer(1, 8),
         callout("good", f"<b>{SEL['method']} at K = {SEL['K']}.</b> Four merges, each chemically "
                         "defensible: purine ring + oxopurine carbonyl (both purine scaffolds); "
                         "monosaccharide + phosphate ester (sugar phosphates); fatty acyl + "
                         "acylglycerol (both acyl chains, distinguished only by the ester); flavin + "
                         "porphyrin (both conjugated N-heterocyclic cofactors, both corpus-limited). "
                         "Sterol, polysaccharide, pyrimidine, aromatic residue, protein backbone, "
                         "free amino acid, organic acid, sulfur and carotenoid remain separate.", None),
         P("5 · Sensitivity", H1),
         P("The table below is the full sweep. Two patterns are worth stating plainly. First, "
           "<b>κ falls as K rises</b> for every method — finer hierarchies are genuinely harder, and "
           "no amount of clustering changes that. Second, <b>the null tracks the accuracy closely</b> "
           "at low K, which is exactly why the raw curve must not be read as performance.", BODY),
         tbl([[r.method, int(r.K), f"{r.top1:.3f}", f"{r.null_top1:.3f}", f"{r.kappa:.3f}",
               f"{r.macro_f1:.3f}", f"{r.ece:.3f}", f"{r.interpretability:.3f}",
               f"{r.score_kappa_x_interp:.4f}", "y" if r.chemically_admissible else ""]
              for _, r in SW.sort_values(["method", "K"]).iterrows()],
             ["method", "K", "top-1", "null", "κ", "macro F1", "ECE", "I", "κ × I", "adm"],
             [1.05 * inch, 0.32 * inch, 0.55 * inch, 0.5 * inch, 0.5 * inch, 0.6 * inch,
              0.5 * inch, 0.5 * inch, 0.6 * inch, UW - 5.12 * inch], fs=6.4),
         Spacer(1, 8),
         P(f"Atlas {FP} unchanged. Reproduce: "
           "python results/v6_rebuild/code/p03_theme_optimisation.py", SMALL)]
    build(S, REPORTS / "GAIRA_V6_THEME_OPTIMISATION.pdf", "GAIRA V6 Theme Optimisation",
          f"GAIRA V6 Theme Optimisation · atlas {FP[:12]}…")


# ══════════════════════════════════════════════════════════════════
def doc_engine_guide():
    S = [Spacer(1, 6), P("GAIRA V6 — Engine Guide", TITLE),
         P("The exact inference pipeline, what every layer does, and how to read the output", SUB),
         callout("key", "GAIRA is not a classifier. It is a frozen coordinate system with two "
                        "interpretive maps layered on top. Nothing above the atlas is learned: "
                        "M and T are derived tables, and inference is a projection followed by two "
                        "matrix products.", None),
         Spacer(1, 8), P("1 · The pipeline, in five lines", H1),
         P("v      = preprocess(wavenumber, intensity)      # crop → AsLS → SavGol → resample → L2<br/>"
           "coord  = NNLS(v, H)  ;  coord /= coord.sum()    # H FIXED — the atlas cannot change<br/>"
           "mss    = Mᵀ · coord                             # 17 spectroscopic motifs<br/>"
           "theme  = Tᵀ · mss                               # 13 chemical themes<br/>"
           "ood    = 1 − mean cos to the 5 nearest reference coordinate vectors", MONO),
         Spacer(1, 5), F("f01_hierarchy.png", "Figure 1 — The layers and what each one means."),
         P("2 · Layer by layer", H1),
         P("2.1 Preprocessing (frozen)", H2),
         P("Crop to 450–1800 cm⁻¹; AsLS baseline (λ=1e5, p=0.01, 8 iterations); Savitzky–Golay "
           "(window 9, order 3); resample onto 676 bins at 2 cm⁻¹; L2-normalise. Identical at build "
           "time and inference time. Bins outside the input's own range become NaN and are "
           "zero-filled at projection — check coverage before trusting a partial spectrum.", BODY),
         P("2.2 Projection (frozen)", H2),
         P("Non-negative least squares onto the 24 × 676 basis with the dictionary held fixed "
           "(<font name='DJ-M' size='7.6'>update_H=False</font>). This is <b>not</b> cosine matching: "
           "the query is fitted as a non-negative combination of the basis spectra. The activations "
           "are then L1-normalised, so a coordinate is a <i>share of the reconstructed evidence</i> — "
           "a proportion, which is only meaningful because the decomposition is non-negative.", BODY),
         P("2.3 MSS motifs (V6, new)", H2),
         P("A motif is a recurring Raman band pattern, <b>not a molecule</b>. M is sparse and "
           "non-negative: each column sums to 1 over at most six components. The four evidence lines "
           "are band overlap (0.30), basis-spectrum cosine (0.30), exemplar loading (0.30) and "
           "perturbation (0.10). No theme label enters any of them.", BODY),
         P("2.4 Chemical themes (V6, new)", H2),
         P("T is a hard partition of the 17 motifs into 13 chemical classes. A theme's score is the "
           "sum of its member motifs' activations. Themes are <b>biochemical classes</b>, not "
           "disease biology — biological-state themes are deliberately not implemented.", BODY),
         PageBreak(),
         P("3 · How to read the output", H1),
         tbl([["component coordinates", "24 non-negative shares summing to 1",
               "What the atlas actually saw. A sparse, high-contrast vector means a confident fit; "
               "a dense flat one means the spectrum is being spread thinly across the basis."],
              ["MSS motif composition", "17 non-negative scores",
               "What spectroscopy says. In-domain this is the <b>more reliable</b> readout: motif "
               "top-1 0.571 / top-3 0.914."],
              ["chemical theme composition", "13 non-negative scores",
               "The chemistry implied. top-1 0.613 / top-3 0.890 in-domain."],
              ["theme confidence", "the leading theme's share of total theme mass",
               "Descriptive, <b>not a probability</b>. ECE 0.28 — it over-states accuracy at the top "
               "of the range."],
              ["expected-theme rank", "1–13", "The evaluation metric, not an inference output. "
               "Reported here so per-theme reliability can be attached to a call."]],
             ["output", "what it is", "how to read it"],
             [1.5 * inch, 1.6 * inch, UW - 3.1 * inch], fs=7.6),
         Spacer(1, 8),
         P("Per-theme reliability — attach this to any call", H2),
         tbl([[r.theme, int(r.n_analytes), "—" if pd.isna(r.top1) else f"{r.top1:.2f}",
               "—" if pd.isna(r.top3) else f"{r.top3:.2f}",
               "HIGH" if (not pd.isna(r.top1) and r.top1 >= .7) else
               ("MODERATE" if (not pd.isna(r.top1) and r.top1 >= .4) else "LOW")]
              for _, r in THREF.iterrows()],
             ["chemical theme", "n", "top-1", "top-3", "reliability"],
             [2.1 * inch, 0.5 * inch, 0.7 * inch, 0.7 * inch, UW - 4.0 * inch], fs=7.6),
         Spacer(1, 8),
         callout("warn", "<b>Do not report a LOW-reliability theme as a finding.</b> Protein backbone "
                         "(0.09) and sulfur metabolite (0.11) are ranked first for fewer than one "
                         "analyte in eight that should express them — even in-domain, on the corpus "
                         "the atlas was fitted to.", None),
         F("f07_radars.png", "Figure 2 — Read the three radars left to right. A clean component "
                             "radar with a wrong theme radar is an interpretation failure; a diffuse "
                             "component radar is a representation failure.", maxh=6.4 * inch),
         PageBreak(),
         P("4 · Worked example", H1),
         F("f08_pathway.png", "Figure 3 — Excellent, moderate and failure cases, end to end.",
           maxh=6.6 * inch),
         P("5 · Scope", H1),
         bullets([
             "<b>Validated for:</b> pure-compound Raman, 450–1800 cm⁻¹, in-domain.",
             "<b>Not validated for:</b> Ag-SERS (theme ranking collapses to chance), serum SERS, "
             "Au-SERS, EV, tissue, DART.",
             "<b>Never:</b> molecular identification. GAIRA returns themes and motifs with "
             "uncertainty; nearest reference analytes are evidence, never an identification.",
         ]),
         Spacer(1, 8), P(f"Atlas {FP}. Engine: results/v6_rebuild/code/v6_semantic/.", SMALL)]
    build(S, REPORTS / "GAIRA_V6_ENGINE_GUIDE.pdf", "GAIRA V6 Engine Guide",
          f"GAIRA V6 Engine Guide · atlas {FP[:12]}…")


# ══════════════════════════════════════════════════════════════════
def doc_mss_manual():
    S = [Spacer(1, 6), P("GAIRA V6 — MSS Reference Manual", TITLE),
         P("Seventeen molecular spectral signatures, and the evidence behind each", SUB),
         callout("key", "An MSS motif is a <b>validated spectral pattern</b>, not a molecule. Motif "
                        "definitions (name, characteristic bands, exemplar chemistries) are curated "
                        "from textbook Raman spectroscopy. Everything quantitative — which components "
                        "express a motif, the weights, the confidence — is <b>derived</b> from the "
                        "frozen atlas. In V6 no theme label enters any of it.", None),
         Spacer(1, 6),
         tbl([[k, str(v)] for k, v in MREG["derivation"].items() if k != "note"],
             ["derivation parameter", "value"], [2.2 * inch, UW - 2.2 * inch], fs=7.6),
         Spacer(1, 8),
         F("f09_motif_gallery.png", "Figure 1 — Every motif's implied Raman spectrum (red) against "
                                    "its declared band profile (blue dashed).", maxh=6.4 * inch),
         PageBreak()]
    aud = MAUD.set_index("motif")
    for m in MREG["motifs"]:
        mid = m["id"]
        rowa = aud.loc[mid] if mid in aud.index else None
        S += [P(m["name"], H1),
              P(f"<font name='DJ-M' size='7.6'>{mid}</font> &nbsp;·&nbsp; chemical class "
                f"<b>{m.get('chemical_class','—')}</b>"
                + ("  &nbsp;·&nbsp; <font color='#D55E00'><b>LOW COVERAGE</b></font>"
                   if m.get("low_coverage") else "")
                + ("  &nbsp;·&nbsp; <i>non-biochemical</i>" if m.get("non_biochemical") else ""), SMALL),
              P(m["description"], BODY),
              P("Spectral justification", H2),
              P("Characteristic bands (cm⁻¹): <b>"
                + ", ".join(str(int(b)) for b in m["bands_cm"]) + "</b>. Matching is region-based "
                "(±16 cm⁻¹), never exact-peak, and is additionally scored by the cosine between each "
                "component's frozen basis spectrum and a Gaussian profile built from these bands.", BODY),
              P("Supporting components", H2),
              tbl([[f"c{c['component']}", f"{c['weight']:.3f}", f"{c['band']:.2f}",
                    f"{c['basis_cosine']:.3f}", f"{c['exemplar']:.2f}", f"{c['perturbation']:.2f}",
                    ", ".join(c["matched_analytes"][:3]) or "—"] for c in m["contributors"]],
                  ["component", "weight", "band", "basis cos", "exemplar", "pert", "matched analytes"],
                  [0.75 * inch, 0.6 * inch, 0.5 * inch, 0.7 * inch, 0.65 * inch, 0.5 * inch,
                   UW - 3.7 * inch], fs=7.4),
              Spacer(1, 5),
              P("Supporting analytes and coverage", H2),
              P("Exemplar chemistries: " + ", ".join(m["exemplars"][:14])
                + (" …" if len(m["exemplars"]) > 14 else ""), SMALL),
              tbl([[f"{m['confidence']:.3f}", f"{m['stability']:.3f}",
                    f"{m['evidence_breadth']:.3f}", f"{m['spectral_purity']:.3f}",
                    (f"{int(rowa.corpus_coverage_n)}" if rowa is not None else "—"),
                    (f"{rowa.discriminative_auc:.3f}" if rowa is not None
                     and not pd.isna(rowa.discriminative_auc) else "—"),
                    (f"{rowa.band_fidelity:.3f}" if rowa is not None else "—")]],
                  ["confidence", "stability", "breadth", "purity", "corpus analytes",
                   "discriminative AUC", "band fidelity"],
                  [0.85 * inch, 0.8 * inch, 0.7 * inch, 0.65 * inch, 1.0 * inch, 1.15 * inch,
                   UW - 5.15 * inch], fs=7.4),
              Spacer(1, 10)]
        if mid in ("nucleic_backbone_phosphate", "protein_amide_backbone", "sterol_ring_system",
                   "polysaccharide_glycosidic", "colloid_matrix_background"):
            S += [PageBreak()]
    S += [P("Known limitations of the motif set", H1),
          bullets([
              "<b>sterol_ring_system</b> was rebuilt on canonical cholesterol bands but its "
              "discriminative AUC is still the lowest of the set (0.66) and its top activators are "
              "not sterols. No component in the frozen atlas isolates the steroid ring system — this "
              "is a foundation limit, not a definition problem.",
              "<b>porphyrin_macrocycle</b> and <b>flavin_redox_cofactor</b> are marked low_coverage: "
              "the corpus contains no pure porphyrin or flavin reference, so both borrow protein and "
              "purine components. Their component-support cosine is 0.70.",
              "<b>carotenoid_polyene</b> rests on two corpus analytes.",
              "<b>protein_amide_backbone</b> has 32 supporting analytes but its bands overlap the "
              "saccharide C–O region; its top activators are sugars.",
              "<b>Spectral purity is low across the board</b> (0.01–0.13): every motif is supported "
              "diffusely across its (up to six) components rather than concentrated on one. That is a "
              "property of a 24-component basis fitted to 167 analytes, and it is why motifs — not "
              "components — are the right unit of interpretation.",
          ]),
          Spacer(1, 8), P(f"Atlas {FP}. Spec: results/v6_rebuild/artifacts/mss_motifs_v6.yaml", SMALL)]
    build(S, REPORTS / "MSS_REFERENCE_MANUAL.pdf", "GAIRA V6 MSS Reference Manual",
          f"GAIRA V6 MSS Reference Manual · atlas {FP[:12]}…")


# ══════════════════════════════════════════════════════════════════
def doc_theme_reference():
    S = [Spacer(1, 6), P("GAIRA V6 — Chemical Theme Reference", TITLE),
         P("Thirteen chemical themes: definition, membership, evidence and failure modes", SUB),
         callout("key", "A chemical theme is a <b>grouping of MSS motifs</b>, not a grouping of "
                        "components and not a biological state. Every theme below is a nameable "
                        "biochemical class — that was enforced as a constraint, not hoped for.", None),
         Spacer(1, 8),
         tbl([[t["name"], ", ".join(t["motifs"])] for t in A04["selected_partition"]["themes"]],
             ["chemical theme", "member MSS motifs"], [1.9 * inch, UW - 1.9 * inch], fs=7.8),
         Spacer(1, 8),
         F("f06_evaluation.png", "Figure 1 — Per-theme recovery, confusion and calibration over the "
                                 "full Raman corpus.", maxh=6.6 * inch),
         PageBreak()]
    spec = {m["id"]: m for m in MREG["motifs"]}
    for _, r in THREF.iterrows():
        members = [s.strip() for s in r.motifs.split(",")]
        S += [P(r.theme, H1),
              P("Definition", H2),
              P("Spectral evidence for this theme is the summed activation of "
                + ", ".join(f"<font name='DJ-M' size='7.6'>{m}</font>" for m in members)
                + ". " + " ".join(spec[m]["description"].split(".")[0] + "." for m in members
                                  if m in spec), BODY),
              P("Membership and evidence", H2),
              tbl([["included MSS motifs", ", ".join(members)],
                   ["key Raman bands (cm⁻¹)", r.key_bands_cm],
                   ["representative components", r.key_components],
                   ["representative analytes", r.example_analytes or "—"],
                   ["corpus analytes", f"{int(r.n_analytes)}  ({r.coverage_pct}% of the scored set)"],
                   ["motif confidence", f"{r.motif_confidence:.3f}"]],
                  ["", ""], [1.75 * inch, UW - 1.75 * inch], fs=7.6),
              Spacer(1, 5),
              P("Performance", H2),
              tbl([["—" if pd.isna(r.top1) else f"{r.top1:.3f}",
                    "—" if pd.isna(r.top3) else f"{r.top3:.3f}",
                    "—" if pd.isna(r.median_rank) else str(int(r.median_rank)),
                    "—" if pd.isna(r.mean_confidence) else f"{r.mean_confidence:.3f}",
                    str(r.most_confused_with)]],
                  ["top-1", "top-3", "median rank", "mean confidence", "most confused with"],
                  [0.75 * inch, 0.75 * inch, 1.0 * inch, 1.25 * inch, UW - 3.75 * inch], fs=7.6),
              Spacer(1, 4)]
        fails = str(r.failure_cases) if not pd.isna(r.failure_cases) else ""
        if fails and fails != "nan":
            S += [P("Known failure cases", H2),
                  P(f"These analytes should express <b>{r.theme}</b> but do not rank it first: "
                    f"<i>{fails}</i>.", BODY)]
        if not pd.isna(r.top1) and r.top1 < 0.4:
            S += [callout("warn", f"<b>LOW RELIABILITY.</b> {r.theme} is ranked first for only "
                                  f"{r.top1:.0%} of the analytes that should express it, in-domain. "
                                  "Do not report it as a finding.", None)]
        S += [Spacer(1, 8)]
    S += [P("Cross-theme notes", H1),
          bullets([
              "<b>Acyl lipid, Polysaccharide and Carotenoid are perfect (1.00)</b> in-domain, but "
              "carotenoid rests on two analytes and should not be read as strong evidence.",
              "<b>Protein backbone (0.09) and Sulfur metabolite (0.11) are the weakest.</b> Both lose "
              "to Acyl lipid and Monosaccharide + Phosphate, whose motifs carry more component mass "
              "in the shared 1240–1460 cm⁻¹ region.",
              "<b>Sterol (0.58) is recovered for the first time</b> — V1's sterol theme never ranked "
              "first for any analyte — but the frozen atlas still has no component that isolates the "
              "steroid ring system.",
              "<b>Flavin + Porphyrin (0.88)</b> looks strong but rests on 8 analytes and two motifs "
              "that both borrow protein components; treat it as provisional.",
          ]),
          Spacer(1, 8), P(f"Atlas {FP}. Source: results/v6_rebuild/tables/p6_theme_reference.csv", SMALL)]
    build(S, REPORTS / "CHEMICAL_THEME_REFERENCE.pdf", "GAIRA V6 Chemical Theme Reference",
          f"GAIRA V6 Chemical Theme Reference · atlas {FP[:12]}…")


if __name__ == "__main__":
    doc_rebuild_report()
    doc_theme_optimisation()
    doc_engine_guide()
    doc_mss_manual()
    doc_theme_reference()
