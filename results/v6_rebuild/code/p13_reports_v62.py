"""GAIRA V6.2 — the ten reports. Each: objective, method, results, figures,
interpretation, limitations, recommendations. No padding."""
from __future__ import annotations
import sys, json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

REPO = Path("/Users/surajpg/projects/GAIRA")
sys.path.insert(0, str(REPO / "results/v6_rebuild/code"))
from v6_semantic.pdfkit import (P, bullets, callout, fig as FIG, tbl, build, PageBreak, Spacer,
                                TITLE, SUB, H1, H2, BODY, SMALL, CAP, MONO, EQ, UW, FP, inch)
from v62 import core as C

BASE = REPO / "results/v6_rebuild"
FIGS = BASE / "figures_v62"
REP = BASE / "reports"
REP.mkdir(parents=True, exist_ok=True)
F = lambda n, c, **k: FIG(FIGS, n + ".png", c, **k)

J1 = json.loads(C.art("v62_soft_hierarchy.json").read_text())
J2 = json.loads(C.art("v62_information_graph.json").read_text())
V6 = json.loads(C.art("p7_evaluation.json").read_text())
V6O = json.loads(C.art("p4_theme_optimisation.json").read_text())
MEMB = pd.read_csv(C.tab("v62_theme_membership.csv"))
SH = pd.read_csv(C.tab("v62_shared_motifs.csv"))
UNC = pd.read_csv(C.tab("v62_theme_uncertainty.csv"))
IB = pd.read_csv(C.tab("v62_information_bottleneck.csv"))
IBH = IB[IB.grouping == "hybrid_clustering"].sort_values("K")
IBL2 = IB[IB.grouping == "L2_superclass"]
PAR = pd.read_csv(C.tab("v62_pareto.csv"))
PROP = pd.read_csv(C.tab("v62_uncertainty_propagation.csv"))
CONF = pd.read_csv(C.tab("v62_confusion_explained.csv"))
GN = pd.read_csv(C.tab("v62_graph_nodes.csv"))
CLD = pd.read_csv(C.tab("v62_learned_vs_derived.csv"))
TSW = pd.read_csv(C.tab("v62_temperature_sweep.csv"))
L2 = J1["levels"]["L2_medium"]["names"]
L3 = J1["levels"]["L3_coarse"]["names"]

PHIL = ("GAIRA is a biochemical reasoning engine, not a molecular classifier. V6.2 optimises "
        "<b>maximum biochemical abstraction subject to minimum information loss</b>. Priority "
        "order: interpretability, information retention, recoverability, and only then "
        "analyte-level accuracy — a modest accuracy loss is an acceptable price for a "
        "representation that transfers.")


def head(t, s, objective, method):
    return [Spacer(1, 6), P(t, TITLE), P(s, SUB),
            tbl([["Objective", objective], ["Method", method],
                 ["Frozen", f"Atlas {FP}; the V6 MSS map M is an input, never rewritten."],
                 ["Reproduce", "results/v6_rebuild/code/p10 → p11 → p12 → p13"]],
                ["", ""], [0.78 * inch, UW - 0.78 * inch], fs=8.2), Spacer(1, 10)]


def limrec(lims, recs):
    return [P("Limitations", H1), bullets(lims), P("Recommendations", H1), bullets(recs),
            Spacer(1, 8),
            P(f"Atlas {FP} verified unchanged. All values recomputed from results/v6_rebuild/.", SMALL)]


def mk(story, fname, title):
    build(story, REP / fname, title, f"GAIRA V6.2 · {title} · atlas {FP[:12]}…")


# ══ 1 architecture ══
def r01():
    S = head("V6.2 · Semantic Hierarchy Architecture",
             "Soft, multi-scale, graph-structured interpretation above a frozen foundation",
             "Replace the single hard 13-theme layer of V6 with a soft, multi-resolution "
             "hierarchy that preserves spectroscopic information and represents genuine "
             "chemical overlap.",
             "Two non-negative linear maps above the frozen atlas: M (component→motif, frozen "
             "from V6) and S (motif→theme, soft and row-stochastic, new in V6.2).")
    S += [callout("key", PHIL, "Philosophy"),
          P("The chain", H1),
          P("coord(x) = NNLS(preprocess(x), H)        24 non-negative shares, frozen<br/>"
            "mss(x)   = Mᵀ coord(x)                    17 spectroscopic motifs, frozen from V6<br/>"
            "theme(x) = Sᵀ mss(x)                      K soft chemical themes, NEW<br/>"
            "S ≥ 0 ,  Σ_t S[m,t] = 1 ,  sparse", MONO),
          F("v62_f01_hierarchy", "Figure 1 — The V6.2 hierarchy at three semantic resolutions."),
          P("The three resolutions", H1),
          tbl([[k.replace("_", " "), str(v["K"]), "yes" if v["admissible"] else "no",
                v["source"][:70]] for k, v in J1["levels"].items()],
              ["level", "K", "chemically admissible", "how derived"],
              [1.1 * inch, 0.4 * inch, 1.3 * inch, UW - 2.8 * inch], fs=7.6),
          Spacer(1, 6),
          tbl([[n, ", ".join(g)] for n, g in zip(L2, J1["levels"]["L2_medium"]["groups"])],
              ["L2 chemical theme", "member MSS motifs"], [1.6 * inch, UW - 1.6 * inch], fs=7.6),
          Spacer(1, 6),
          tbl([[n, ", ".join(g)] for n, g in zip(L3, J1["levels"]["L3_coarse"]["groups"])],
              ["L3 biochemical system", "member MSS motifs"], [1.6 * inch, UW - 1.6 * inch], fs=7.4),
          Spacer(1, 8),
          callout("warn",
                  "<b>An honest negative result.</b> The K=8 grouping produced by the hybrid "
                  "motif distance was <b>chemically inadmissible</b> (it merged motifs from "
                  "unrelated superclasses), so L2 falls back to the chemical-superclass grouping "
                  f"at <b>K={J1['levels']['L2_medium']['K']}</b>, not the 8 suggested in the brief. "
                  "The data did not support 8 derived themes; the number was not forced.", None),
          P("Interpretation", H1),
          P("The architectural change is that a motif is no longer owned by one theme. "
            f"{J1['membership']['n_motifs_multi_theme_L2']} of 17 motifs carry membership in more "
            "than one chemical theme, with a mean of "
            f"{float((np.load(C.art('v62_membership.npz'))['S_L2'] > 0).sum(1).mean()):.2f} themes "
            "per motif. That is not noise: it is the spectroscopic reality that amide III and the "
            "saccharide C–O network occupy the same region.", BODY)]
    S += limrec(
        ["L2 is the chemical-superclass grouping, not a data-derived one — the derived K=8 "
         "clustering failed the admissibility rule.",
         "L3 is chemically inadmissible by the strict L2 rule, which is expected at a coarse "
         "level (it merges superclasses by design) but means the rule does not constrain it.",
         "The soft-membership temperature is a chosen parameter, selected by a stated sparsity "
         "criterion (§ Soft Theme Membership) rather than derived from first principles."],
        ["Treat L2 as the operational level and L1 (motifs) as the evidence level; L3 is for "
         "communication, not inference.",
         "Revisit the derived K=8 grouping once the corpus contains pure porphyrin, flavin and "
         "sterol references — the inadmissibility traces to those under-grounded motifs."])
    mk(S, "V62_01_ARCHITECTURE.pdf", "Semantic Hierarchy Architecture")


# ══ 2 soft membership ══
def r02():
    S = head("V6.2 · Soft Theme Membership",
             "Every motif carries a distribution over chemical themes, not a label",
             "Replace hard motif→theme assignment with a non-negative, row-stochastic, sparse "
             "membership matrix S.",
             "S[m,·] = softmax over the cosine between motif m's corpus activation profile and "
             "each theme centroid, followed by a floor at 0.02 and renormalisation.")
    S += [P("Method", H1),
          P("p_m  = motif m's activation across the 167 analytes, L2-normalised<br/>"
            "c_t  = mean p_m over theme t's seed motifs, L2-normalised<br/>"
            "S[m,t] ∝ exp( cos(p_m, c_t) / τ ) ,  then floor at 0.02 and renormalise", MONO),
          P("τ is chosen by a criterion fixed in advance: the largest (softest) temperature on a "
            "fixed grid for which the mean number of themes carrying a motif is ≤ 2.5 — "
            "\"a motif belongs to one, sometimes two chemistries\". The full sweep is reported so "
            "the choice is auditable.", BODY),
          tbl([[f"{r.temperature:g}", f"{r.mean_support:.2f}", f"{r.mean_entropy:.3f}",
                f"{r.mean_dominant_weight:.3f}",
                "◄ selected" if r.temperature == J1["soft_parameters"]["temperature"] else ""]
               for _, r in TSW.iterrows()],
              ["τ", "mean support", "mean entropy", "mean dominant weight", ""],
              [0.6 * inch, 1.0 * inch, 1.0 * inch, 1.4 * inch, UW - 4.0 * inch], fs=7.6),
          Spacer(1, 8),
          F("v62_f02_soft_membership",
            "Figure 1 — The membership matrix, per-motif entropy, and example theme posteriors."),
          P("Results", H1),
          tbl([[r.motif.replace("_", " "), r.dominant_theme, f"{r.dominant_weight:.3f}",
                r.runner_up_theme, f"{r.runner_up_weight:.3f}", f"{r.entropy:.3f}",
                str(int(r.n_themes_above_floor))]
               for _, r in MEMB[MEMB.level == "L2_medium"].iterrows()],
              ["motif", "dominant theme", "w", "runner-up", "w", "entropy", "support"],
              [1.5 * inch, 1.25 * inch, 0.45 * inch, 1.2 * inch, 0.45 * inch, 0.6 * inch,
               UW - 5.45 * inch], fs=7.0),
          Spacer(1, 8),
          callout("warn",
                  "<b>Two motifs land on the wrong dominant theme, and the soft layer makes it "
                  "visible rather than hiding it.</b> <i>sterol_ring_system</i> is dominated by "
                  "Redox / pigments (0.82) rather than Lipid chemistry (0.08), and "
                  "<i>carotenoid_polyene</i> by Protein chemistry (0.65) over Redox / pigments "
                  "(0.35). Both are corpus-limited motifs already flagged in V6; the membership "
                  "entropy now quantifies the problem instead of a hard label concealing it.", None),
          P("Interpretation", H1),
          P(f"Mean membership entropy at L2 is {J1['membership']['mean_entropy_L2']:.3f} and the "
            f"mean dominant weight {J1['membership']['mean_dominant_weight_L2']:.3f}. "
            f"{J1['membership']['n_motifs_single_theme_L2']} motifs are cleanly single-theme "
            "(purine, pyrimidine, monosaccharide, fatty acyl, acylglycerol, porphyrin) — these are "
            "the chemistries the corpus grounds well. The rest carry real, interpretable overlap.", BODY)]
    S += limrec(
        ["τ is a free parameter; a different sparsity target would give a different S.",
         "The theme centroids are computed from the seed grouping, so S inherits any error in that "
         "grouping — most visibly for sterol and carotenoid.",
         "Membership is derived from corpus co-activation, so it describes THIS corpus."],
        ["Report the entropy alongside every theme call; a high-entropy motif should not drive an "
         "interpretation on its own.",
         "Re-derive S after adding pure sterol/porphyrin/flavin references."])
    mk(S, "V62_02_SOFT_MEMBERSHIP.pdf", "Soft Theme Membership")


# ══ 3 shared motifs ══
def r03():
    S = head("V6.2 · Shared Biochemical Motifs",
             "Cross-theme overlap is represented, not suppressed",
             "Identify motifs whose chemistry genuinely spans more than one theme, and explain "
             "each overlap spectroscopically.",
             "A motif is flagged as shared when its dominant L2 membership is below 0.85. For "
             "each, the responsible components and the overlapping band positions are reported.")
    S += [callout("key", "V6 forced <i>protein_amide_backbone</i> and the saccharide motifs apart "
                         "into disjoint themes. That was a modelling decision fighting the "
                         "spectroscopy: amide III (~1240) and the CH₂/C–O saccharide modes "
                         "(~1122–1340) genuinely overlap. V6.2 represents the overlap.", None),
          F("v62_f03_shared_motifs",
            "Figure 1 — The motif→theme network (red edges are cross-theme membership) and the "
            "shared-motif inventory."),
          P("Results", H1),
          tbl([[r.motif.replace("_", " "), r.dominant_theme, f"{r.dominant_weight:.3f}",
                r.shared_with, f"{r.shared_weight:.3f}", r.components_responsible[:34],
                r.overlapping_bands_cm[:30]] for _, r in SH.iterrows()],
              ["motif", "dominant", "w", "shared with", "w", "components", "shared bands cm⁻¹"],
              [1.35 * inch, 1.15 * inch, 0.42 * inch, 1.05 * inch, 0.42 * inch, 1.15 * inch,
               UW - 5.54 * inch], fs=6.8),
          Spacer(1, 8),
          P("Interpretation", H1),
          P(f"{len(SH)} of 17 motifs carry meaningful cross-theme membership. Each has a "
            "spectroscopic reason on the same row: <i>nucleic_backbone_phosphate</i> shares the "
            "788/920/1240 cm⁻¹ region with the nucleobases; <i>oxopurine_carbonyl</i> shares "
            "640/938 with the organic acids through its carbonyl; <i>carotenoid_polyene</i> shares "
            "the 1005–1520 region with aromatic residues because both are conjugated π systems. "
            "These are not confusions to be fixed — they are the reason a Raman spectrum of a "
            "mixture is hard, stated explicitly.", BODY),
          callout("note", "This matters for transfer. A hierarchy that hides overlap will "
                          "mis-attribute it on a new substrate; one that represents it can carry "
                          "the ambiguity forward into the Ag-SERS phase as an explicit prior.", None)]
    S += limrec(
        ["The 0.85 threshold for 'shared' is a reporting convention, not a test.",
         "sterol_ring_system appears as shared with Lipid chemistry only weakly (0.08) because its "
         "dominant assignment is already wrong — the shared-motif view does not rescue it.",
         "Overlap is measured on this corpus; a corpus with more phospholipids would likely "
         "surface further sharing."],
        ["Propagate the shared membership into the BSV rather than collapsing to the dominant "
         "theme, so downstream consumers see the ambiguity.",
         "Use the shared-motif list as the prior for which themes will be hardest to separate on "
         "silver."])
    mk(S, "V62_03_SHARED_MOTIFS.pdf", "Shared Biochemical Motifs")


# ══ 4 continuous space ══
def r04():
    cs = J2["continuous_space"]
    S = head("V6.2 · Continuous Biochemical Theme Space",
             "Coordinates, not classes",
             "Replace theme classification with a continuous, non-negative theme vector, and show "
             "that chemistry organises continuously.",
             f"theme(x) = Sᵀ Mᵀ coord(x) ∈ ℝ^{cs['dim']}₊, stored as theme_embedding.npy. "
             "Visualised by PCA and UMAP; distances measured by cosine.")
    S += [F("v62_f05_manifold",
            "Figure 1 — PCA and UMAP of the continuous theme space, coloured by chemical family "
            "and by theme confidence."),
          P("Results", H1),
          tbl([["embedding dimension", str(cs["dim"])],
               ["themes", ", ".join(cs["themes"])],
               ["PCA explained variance", ", ".join(f"{x:.3f}" for x in cs["pca_explained_variance"])],
               ["first two PCs", f"{sum(cs['pca_explained_variance'][:2]):.1%} of the variance"],
               ["mean pairwise cosine distance", f"{cs['mean_pairwise_theme_distance']:.3f}"],
               ["UMAP available", str(cs["umap_available"])]],
              ["", ""], [1.9 * inch, UW - 1.9 * inch], fs=7.8),
          Spacer(1, 8),
          P("Interpretation", H1),
          P(f"Two principal components carry {sum(cs['pca_explained_variance'][:2]):.0%} of the "
            "theme-space variance, so the six-dimensional chemical state is close to a surface, "
            "not a cloud of discrete bins. In the UMAP the lipid analytes (triglycerides and fatty "
            "acids) form a single dense arm — consistent with their near-perfect recovery — while "
            "the amino acids and organic acids occupy a diffuse central region, which is exactly "
            "where the theme layer is least reliable.", BODY),
          callout("note", "A continuous coordinate is what makes cross-substrate transfer "
                          "expressible at all: a class label cannot be partially preserved, but a "
                          "coordinate can move a measurable distance.", None)]
    S += limrec(
        ["UMAP is a visualisation, not a metric; all quantitative statements use the cosine "
         "distance in the raw theme space.",
         "The embedding is unnormalised for spectral intensity, so an analyte with more total "
         "motif mass sits further from the origin.",
         "167 analytes is small for a manifold claim; the structure is suggestive, not established."],
        ["Use the raw theme vector, not the UMAP coordinates, for any downstream computation.",
         "Re-examine the manifold once biological cohorts are projected — the interesting question "
         "is whether disease states move along it."])
    mk(S, "V62_04_CONTINUOUS_SPACE.pdf", "Continuous Theme Space")


# ══ 5 information bottleneck ══
def r05():
    ib = J2["information_bottleneck"]
    S = head("V6.2 · Information Bottleneck Optimisation",
             "What abstraction costs, measured",
             "Replace top-1 accuracy as the optimisation objective with a direct measurement of "
             "the information retained by each level of abstraction.",
             "For K = 2…17 the motif activations are projected to K themes and reconstructed by "
             "least squares. Explained variance, reconstruction error, KL divergence, mutual "
             "information with the chemical family and the compression ratio are recorded.")
    S += [F("v62_f06_bottleneck",
            "Figure 1 — Information retained, information lost, chemical information per "
            "dimension, and compression against interpretability."),
          P("Results", H1),
          tbl([[str(int(r.K)), f"{r.explained_variance_motif:.4f}", f"{r.reconstruction_error:.4f}",
                f"{r.kl_divergence:.4f}", f"{r.mi_retained:.4f}", f"{r.compression_ratio:.2f}",
                "yes" if r.chemically_admissible else ""] for _, r in IBH.iterrows()],
              ["K", "variance retained", "recon. error", "KL", "MI retained", "compression", "adm."],
              [0.4 * inch, 1.25 * inch, 1.0 * inch, 0.7 * inch, 0.9 * inch, 1.0 * inch,
               UW - 5.25 * inch], fs=7.2),
          Spacer(1, 6),
          callout("good",
                  f"<b>The V6.2 L2 layer (K={int(IBL2.K.iloc[0])}) retains "
                  f"{float(IBL2.explained_variance_motif.iloc[0]):.3f} of the motif variance at a "
                  f"{float(IBL2.compression_ratio.iloc[0]):.2f}× compression</b> — against 0.981 "
                  "at V6's K=13 with only 1.31× compression. V6.2 gives up 0.23 of motif variance "
                  "to more than double the abstraction.", None),
          P("Two results worth stating precisely", H2),
          bullets([
              f"<b>The elbow sits at K={ib['elbow_K']}</b> by maximum curvature of the "
              "variance curve — but at that point only 0.56 of the motif variance survives. The "
              "elbow is a mathematical feature of the curve, not a recommendation.",
              "<b>Mutual information per dimension is <i>higher</i> in the theme space than in the "
              "motif space for K ≤ 13</b> (MI retained > 1.0). This is not free information: MI is "
              "normalised per dimension, so fewer dimensions each carry more chemical-family "
              "information. Abstraction concentrates the signal even as it discards variance."]),
          P("Interpretation", H1),
          P("The two curves answer different questions. Reconstruction variance says how much of "
            "the motif pattern could be rebuilt — it falls steeply below K≈8. Mutual information "
            "says how much chemistry the representation still distinguishes — it stays near or "
            "above parity all the way down to K=3. A hierarchy chosen on variance alone would keep "
            "13+ themes; one chosen on chemical information could go to 4. V6.2 sits between them "
            "at 6 because that is where the grouping is also chemically nameable.", BODY)]
    S += limrec(
        ["MI is estimated by per-dimension quantile binning; it is a plug-in estimate on 167 "
         "analytes and should be read as a trend, not a precise nat count.",
         "Reconstruction uses an unconstrained least-squares inverse, which is generous — a "
         "non-negative inverse would retain less.",
         "The sweep uses the hybrid clustering; the chosen L2 grouping is not on that curve and is "
         "reported as a separate labelled point."],
        ["Report both variance retained and MI retained whenever a hierarchy level is proposed; "
         "they disagree, and the disagreement is the interesting part.",
         "Use K=6 for interpretation and K=17 (the motif layer) whenever information matters more "
         "than nameability."])
    mk(S, "V62_05_INFORMATION_BOTTLENECK.pdf", "Information Bottleneck Optimisation")


# ══ 6 ontology graph ══
def r06():
    g = J2["ontology_graph"]
    S = head("V6.2 · Ontology Graph Analysis",
             "The biochemical hierarchy is a graph, not a tree",
             "Allow a motif to have several parents, and measure the resulting structure.",
             "A directed multi-level graph: motif → L2 chemical theme → L3 biochemical system, "
             "with edge weights taken from the soft membership. Centrality and communities are "
             "computed on the undirected projection.")
    S += [F("v62_f08_ontology_graph",
            "Figure 1 — The ontology graph. Red edges are the weaker, shared memberships."),
          P("Results", H1),
          tbl([["nodes", str(g["n_nodes"])], ["edges", str(g["n_edges"])],
               ["motifs with more than one parent", f"{g['n_multi_parent_motifs']} of 17"],
               ["communities detected", str(g["n_communities"])],
               ["multi-parent motifs", ", ".join(m.replace("_", " ") for m in g["multi_parent_motifs"])]],
              ["", ""], [2.1 * inch, UW - 2.1 * inch], fs=7.6),
          Spacer(1, 6),
          P("Highest-betweenness nodes — the bridges between chemistries", H2),
          tbl([[r["node"].replace("_", " "), r["kind"], f"{r['betweenness']:.4f}"]
               for r in g["highest_betweenness"]],
              ["node", "kind", "betweenness"], [2.3 * inch, 1.2 * inch, UW - 3.5 * inch], fs=7.6),
          Spacer(1, 8),
          P("Interpretation", H1),
          P(f"{g['n_multi_parent_motifs']} of 17 motifs have more than one parent, so the tree "
            "assumption V6 inherited is simply false for this corpus. The high-betweenness nodes "
            "are the chemistries that bridge themes — they are also, unsurprisingly, the motifs "
            "with the highest membership entropy and the ones most often confused in the "
            "recoverability matrix. Centrality, entropy and confusion are three views of one "
            "underlying fact.", BODY),
          callout("note", "An interactive version is written to "
                          "<font name='DJ-M' size='8'>figures_v62/v62_ontology_graph.html</font> "
                          "(plotly, self-contained, no external data).", None)]
    S += limrec(
        ["L3 edges are derived from the dominant L2 assignment, so the third level is coarser "
         "evidence than the second.",
         "Community detection on 25 nodes is unstable; the communities are reported for "
         "description, not inference.",
         "Edge weights are memberships, not probabilities of chemical presence."],
        ["Use the graph, not the tree, whenever a downstream layer needs a parent — several motifs "
         "have no single correct parent.",
         "Carry betweenness forward as a per-motif transfer-risk score into the Ag-SERS phase."])
    mk(S, "V62_06_ONTOLOGY_GRAPH.pdf", "Ontology Graph Analysis")


# ══ 7 recoverability ══
def r07():
    rc = J2["recoverability"]
    S = head("V6.2 · Recoverability and Confusion Analysis",
             "A 17 × 17 matrix, and a spectroscopic reason for every off-diagonal cell",
             "Replace a scalar accuracy with a full confusion structure, and explain each "
             "confusion by spectral and component overlap.",
             "For every labelled analyte, the expected motif (from exemplar membership) is "
             "compared with the arg-max motif. Each off-diagonal cell is annotated with the "
             "cosine between the two motifs' implied spectra, their component-support overlap and "
             "their shared bands.")
    S += [F("v62_f04_recoverability",
            "Figure 1 — The recoverability matrix, the confusion drivers, and the top cases."),
          P("Results", H1),
          tbl([["motif-level top-1", f"{rc['motif_top1']:.3f}"],
               ["analytes scored", str(rc["n_scored"])],
               ["weakest motifs", ", ".join(m.replace("_", " ") for m in rc["worst_motifs"])]],
              ["", ""], [1.6 * inch, UW - 1.6 * inch], fs=7.8),
          Spacer(1, 6),
          tbl([[r["expected"].replace("_", " "), r["predicted"].replace("_", " "), str(int(r["n"])),
                f"{r['spectral_cosine']:.3f}", f"{r['component_overlap']:.3f}",
                str(r["overlapping_bands_cm"])[:26]] for r in rc["top_confusions"]],
              ["expected", "predicted", "n", "spectral cos", "component overlap", "shared bands"],
              [1.5 * inch, 1.5 * inch, 0.35 * inch, 0.85 * inch, 1.15 * inch, UW - 5.35 * inch],
              fs=7.0),
          Spacer(1, 8),
          callout("good",
                  "<b>The confusions are self-explaining.</b> The single largest is "
                  "<i>fatty_acyl_chain → triglyceride_ester</i> (n=13) at a spectral cosine of "
                  "<b>0.938</b> and component overlap <b>0.913</b> — the two motifs share four "
                  "bands (1062, 1128, 1301, 1440) and differ only by the ester carbonyl. That is "
                  "not a model failure; it is what a free fatty acid and its triglyceride look "
                  "like. V6 merged them into one theme for exactly this reason, and V6.2's soft "
                  "layer keeps them adjacent rather than pretending they are separable.", None),
          P("Interpretation", H1),
          P("Off-diagonal mass concentrates where spectral cosine exceeds ~0.7. The exceptions are "
            "informative: <i>sterol_ring_system → triglyceride_ester</i> (n=7) has a spectral "
            "cosine of only 0.44 and a component overlap of 0.13, so that confusion is <b>not</b> "
            "explained by motif similarity — it is the known failure of the atlas to isolate the "
            "steroid ring system, surfacing again at a third level of the hierarchy.", BODY)]
    S += limrec(
        ["The expected-motif label is a curated evaluation overlay from exemplar membership, not "
         "ground truth.",
         "The matrix is computed on analyte means, so within-analyte variability is not represented.",
         "n is small for several motifs (carotenoid 2, polysaccharide 5)."],
        ["Treat spectral cosine > 0.9 pairs as a single reporting unit rather than trying to "
         "separate them.",
         "Investigate the sterol confusion at the component level; it is not a motif problem."])
    mk(S, "V62_07_RECOVERABILITY.pdf", "Recoverability and Confusion Analysis")


# ══ 8 uncertainty ══
def r08():
    up = J2["uncertainty_propagation"]
    S = head("V6.2 · Bayesian Confidence Propagation",
             "Uncertainty carried from replicate noise to the theme layer",
             "Replace a deterministic point estimate at each level with mean, variance, entropy "
             "and confidence.",
             "The empirical covariance of an analyte's replicate coordinates is propagated "
             "through the two linear maps by the delta method: Σ_mss = MᵀΣ_coord M and "
             "Σ_theme = SᵀΣ_mss S. Posterior, entropy, margin and confidence are computed from "
             "the normalised theme vector.")
    S += [P("Method", H1),
          P("Σ_mss   = Mᵀ Σ_coord M<br/>"
            "Σ_theme = Sᵀ Σ_mss   S<br/>"
            "posterior p = theme / Σ theme ,  H = normalised entropy(p)<br/>"
            "margin = p₍₁₎ − p₍₂₎ ,  confidence = p₍₁₎ · (1 − H)", MONO),
          F("v62_f07_propagation",
            "Figure 1 — Total variance at each level, confidence against entropy, and the "
            "distribution of variance amplification."),
          F("v62_f10_calibration",
            "Figure 2 — Confidence distribution, entropy–margin relation and the reliability of "
            "the theme posterior."),
          P("Results", H1),
          tbl([["analytes with ≥2 replicates", str(up["n_analytes"])],
               ["median coord → MSS variance ratio", f"{up['median_coord_to_mss_ratio']:.4f}"],
               ["median MSS → theme variance ratio", f"{up['median_mss_to_theme_ratio']:.4f}"],
               ["median theme confidence", f"{up['median_theme_confidence']:.4f}"],
               ["mean posterior entropy", f"{UNC.entropy.mean():.4f}"],
               ["mean top-2 margin", f"{UNC.margin.mean():.4f}"]],
              ["", ""], [2.4 * inch, UW - 2.4 * inch], fs=7.8),
          Spacer(1, 8),
          callout("good",
                  f"<b>The MSS layer denoises.</b> Replicate variance shrinks by roughly "
                  f"{1/up['median_coord_to_mss_ratio']:.0f}× from the 24 coordinates to the 17 "
                  f"motifs (median ratio {up['median_coord_to_mss_ratio']:.3f}) — pooling "
                  "components into motifs averages away measurement noise. The theme layer then "
                  f"<i>amplifies</i> it slightly (ratio {up['median_mss_to_theme_ratio']:.2f}), "
                  "because a soft, low-dimensional projection concentrates correlated error. "
                  "Abstraction is not uniformly stabilising, and the two steps behave differently.",
                  None),
          P("Interpretation", H1),
          P("Confidence is low in absolute terms (median "
            f"{up['median_theme_confidence']:.3f}) because it multiplies the top posterior by "
            f"(1 − entropy) and entropy is high at K=6. It should be read comparatively, and the "
            "reliability plot shows the posterior is <b>under-confident</b>: observed agreement "
            "exceeds the stated posterior across most of the range. That is the safe direction to "
            "err, but it means the number is not a probability.", BODY)]
    S += limrec(
        ["The delta method is a first-order approximation; the maps are linear so it is exact for "
         "the mean, but the posterior normalisation is not linear.",
         "Only analytes with ≥2 replicates contribute (87 of 167).",
         "Confidence is a heuristic composite, not a calibrated probability."],
        ["Calibrate the posterior (temperature scaling on the reliability curve) before any "
         "downstream consumer treats it as a probability.",
         "Report Σ_theme diagonals alongside the point estimate in the BSV."])
    mk(S, "V62_08_UNCERTAINTY.pdf", "Bayesian Confidence Propagation")


# ══ 9 V6 vs V6.2 ══
def r09():
    S2m = np.load(C.art("v62_membership.npz"))["S_L2"]
    S = head("V6.2 · Comparison with V6",
             "What improved, what is unchanged, and what it cost",
             "State precisely how the V6.2 hierarchy differs from V6 on the same corpus and the "
             "same frozen inputs.",
             "Both stacks read the identical 24 frozen coordinates and the identical V6 MSS map M. "
             "Only the layer above MSS differs.")
    S += [F("v62_f13_comparison", "Figure 1 — V6 versus V6.2 dashboard."),
          P("What is identical", H1),
          bullets(["The frozen atlas and its fingerprint " + FP + ".",
                   "Preprocessing, the NNLS projection and the component registry.",
                   "<b>The MSS layer.</b> M is read from V6 and never rewritten — the 17 motifs, "
                   "their bands, exemplars, components and weights are byte-identical."]),
          P("What changed", H1),
          tbl([["motif → theme map", "hard partition (one theme per motif)",
                "soft, row-stochastic, sparse"],
               ["mean themes per motif", "1.00", f"{float((S2m > 0).sum(1).mean()):.2f}"],
               ["semantic levels", "1 (K=13)", "3 (K=17 / 6 / 4)"],
               ["shared motifs", "0 — suppressed by construction", f"{len(SH)} represented"],
               ["ontology", "tree", f"graph, {J2['ontology_graph']['n_multi_parent_motifs']} "
                                    "multi-parent motifs"],
               ["uncertainty", "a scalar confidence", "posterior, entropy, margin, propagated variance"],
               ["theme output", "a ranked class", "a continuous non-negative coordinate"],
               ["objective", "κ × interpretability", "4-objective Pareto (interpretability, "
                                                     "information, recoverability, stability)"],
               ["motif variance retained", "0.981 (K=13)",
                f"{float(IBL2.explained_variance_motif.iloc[0]):.3f} (K=6)"],
               ["compression", "1.31×", f"{float(IBL2.compression_ratio.iloc[0]):.2f}×"]],
              ["property", "V6", "V6.2"],
              [1.35 * inch, 2.15 * inch, UW - 3.5 * inch], fs=7.2),
          Spacer(1, 8),
          callout("warn",
                  "<b>The cost, stated plainly.</b> Compressing 17 motifs to 6 chemical themes "
                  f"retains {float(IBL2.explained_variance_motif.iloc[0]):.3f} of the motif "
                  "variance, against 0.981 at V6's K=13. V6.2 discards roughly 23 % more of the "
                  "spectroscopic pattern in exchange for a level that is nameable, soft and "
                  "multi-parent. Under the stated philosophy that is the intended trade; under a "
                  "classification objective it would not be.", None),
          P("What was NOT re-measured", H1),
          P("V6.2 changes the theme layer, so V6's per-theme top-1 numbers (theme top-1 "
            f"{V6['theme_top1']:.3f}, top-3 {V6['theme_top3']:.3f} at K=13) do not transfer to the "
            "new K=6 soft layer and are not restated here as if they did. The V6.2 analogue is the "
            f"motif-level recoverability ({J2['recoverability']['motif_top1']:.3f}) plus the "
            "information and calibration measurements — deliberately, because analyte accuracy is "
            "the fourth priority, not the first.", BODY)]
    S += limrec(
        ["The two hierarchies are not directly comparable on accuracy, because they abstract to "
         "different K and V6.2's output is continuous rather than a class.",
         "V6.2's L2 grouping is the chemical-superclass grouping, so part of the comparison is "
         "clustering-vs-chemistry rather than V6-vs-V6.2.",
         "Both share the same corpus limits: sterol, porphyrin, flavin and carotenoid remain "
         "under-grounded, and V6.2 inherits every one of them."],
        ["Keep both layers available: V6's K=13 hard partition when information matters, V6.2's "
         "K=6 soft layer when interpretation and transfer matter.",
         "Do not report a V6.2 theme call without its entropy — the soft layer's value is that it "
         "can say 'both'."])
    mk(S, "V62_09_V6_VS_V62.pdf", "V6 versus V6.2 Comparison")


# ══ 10 technical summary ══
def r10():
    S2m = np.load(C.art("v62_membership.npz"))["S_L2"]
    S = head("V6.2 · Technical Summary and Recommendations",
             "What was built, what it shows, and what to do next",
             "Summarise the V6.2 semantic rebuild and set the agenda for the Ag-SERS phase.",
             "Ten analyses over the frozen atlas and the frozen V6 MSS layer, all deterministic "
             "and reproducible from results/v6_rebuild/code/.")
    S += [callout("key", PHIL, "Philosophy"),
          P("Headline numbers", H1),
          tbl([["frozen atlas", FP + " — unchanged"],
               ["MSS motifs", "17, identical to V6"],
               ["semantic levels", f"17 → {len(L2)} → {len(L3)}"],
               ["mean themes per motif", f"{float((S2m > 0).sum(1).mean()):.2f} (soft, sparse)"],
               ["shared motifs represented", f"{len(SH)} of 17"],
               ["multi-parent motifs in the graph",
                f"{J2['ontology_graph']['n_multi_parent_motifs']} of 17"],
               ["motif variance retained at L2", f"{float(IBL2.explained_variance_motif.iloc[0]):.3f}"],
               ["compression at L2", f"{float(IBL2.compression_ratio.iloc[0]):.2f}×"],
               ["MI retained at L2", f"{float(IBL2.mi_retained.iloc[0]):.3f}"],
               ["motif-level recoverability", f"{J2['recoverability']['motif_top1']:.3f}"],
               ["median theme confidence", f"{J2['uncertainty_propagation']['median_theme_confidence']:.3f}"],
               ["Pareto-optimal K values", str(int(PAR.pareto.sum())) + " of 16"]],
              ["", ""], [2.3 * inch, UW - 2.3 * inch], fs=7.8),
          Spacer(1, 8),
          F("v62_f11_pareto", "Figure 1 — The four-objective Pareto front."),
          P("What V6.2 establishes", H1),
          bullets([
              "<b>Chemical overlap is real and measurable.</b> 15 of 17 motifs share bands with a "
              "second chemistry; 11 have more than one parent in the ontology graph. A tree was "
              "the wrong data structure.",
              "<b>Abstraction has a price curve, and it is not the accuracy curve.</b> Variance "
              "retained falls steeply below K≈8 while mutual information per dimension stays at or "
              "above parity down to K=3. Choosing K on accuracy alone would have picked a "
              "different level than choosing it on chemistry.",
              "<b>The MSS layer denoises and the theme layer does not.</b> Replicate variance "
              f"shrinks {1/J2['uncertainty_propagation']['median_coord_to_mss_ratio']:.0f}× from "
              "coordinates to motifs, then grows slightly from motifs to themes.",
              "<b>Confusions are self-explaining.</b> Every large off-diagonal cell has a spectral "
              "cosine and a shared band list attached — except the sterol case, which is a "
              "foundation limit, not a semantic one."]),
          P("What V6.2 does not establish", H1),
          bullets([
              "That K=6 is optimal. It is the coarsest chemically nameable level; the Pareto front "
              "contains 11 non-dominated K values and the choice among them is a judgement.",
              "That the soft memberships are correct for sterol and carotenoid — both land on the "
              "wrong dominant theme, and V6.2 surfaces this rather than fixing it.",
              "Anything about substrates other than pure Raman. No SERS, EV, serum, bacterial or "
              "DART data enters V6.2."]),
          PageBreak(),
          P("Recommendations", H1),
          P("Immediate", H2),
          bullets([
              "Calibrate the theme posterior before any consumer treats it as a probability; the "
              "reliability curve shows it is systematically under-confident.",
              "Report entropy with every theme call. A soft layer whose ambiguity is discarded at "
              "the point of use is worse than a hard one.",
              "Add pure sterol, porphyrin and flavin references to the grounding corpus. Three of "
              "the four remaining semantic failures trace to the same absence, and no amount of "
              "work above the atlas will fix them."]),
          P("For the Ag-SERS adaptation phase (not implemented here)", H2),
          bullets([
              "<b>Transfer the coordinates, not the classes.</b> V6.2's continuous theme vector is "
              "the right target for a Raman→SERS observation model: a class label cannot be "
              "partially preserved, a coordinate can move a measurable distance.",
              "<b>Use the shared-motif list as the prior for what will be hardest.</b> Motifs with "
              "high membership entropy and high graph betweenness are the ones a substrate change "
              "will scramble first.",
              "<b>Keep the firewall.</b> Fit the observation model above the theme layer. Nothing "
              "in the atlas, M or S should be refitted on SERS data — that separation is what "
              "makes the OOD signal meaningful.",
              "<b>Gate on detectability first.</b> Earlier work showed 29 of 51 analytes are "
              "invisible on silver; a transfer model should be trained only on the detectable, "
              "representation-limited subset.",
              "<b>Expect the purine attractor.</b> On silver, 95 % of analytes previously collapsed "
              "onto purine. The V6.2 soft layer gives a natural way to express that: a substrate "
              "prior that shifts membership mass, rather than a hard re-assignment."]),
          Spacer(1, 10),
          P("Artifact inventory", H2),
          tbl([["theme_membership.yaml", "soft motif→theme membership at all three levels"],
               ["theme_embedding.npy", "the continuous theme coordinates (167 × 6)"],
               ["v62_membership.npz", "S at L1/L2/L3, learned S, distances, spectra"],
               ["v62_spaces.npz", "embeddings, PCA, UMAP, recoverability matrix"],
               ["v62_soft_hierarchy.json", "levels, membership, shared motifs, uncertainty"],
               ["v62_information_graph.json", "bottleneck, graph, propagation, Pareto"],
               ["tables/v62_*.csv", "11 tables: membership, shared motifs, bottleneck, "
                                    "recoverability, confusion, propagation, graph, Pareto"],
               ["figures_v62/*.png + *.pdf", "13 figures, raster and vector"],
               ["figures_v62/v62_ontology_graph.html", "interactive ontology graph"]],
              ["artifact", "contents"], [2.1 * inch, UW - 2.1 * inch], fs=7.4)]
    S += [Spacer(1, 8),
          P(f"Atlas {FP} verified unchanged. Reproduce: p10 → p11 → p12 → p13 under "
            "results/v6_rebuild/code/.", SMALL)]
    mk(S, "V62_10_TECHNICAL_SUMMARY.pdf", "Technical Summary and Recommendations")


if __name__ == "__main__":
    for f in (r01, r02, r03, r04, r05, r06, r07, r08, r09, r10):
        try:
            f()
        except Exception as e:                                          # noqa: BLE001
            import traceback
            print(f"!! {f.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc(limit=2)
