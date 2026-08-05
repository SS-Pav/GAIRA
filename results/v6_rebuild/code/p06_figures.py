"""GAIRA V6 — figures. Parts 1, 2, 4, 7, 8, 9, 10. Publication style."""
from __future__ import annotations
import sys, json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

REPO = Path("/Users/surajpg/projects/GAIRA")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "results/v6_rebuild/code"))
BASE = REPO / "results/v6_rebuild"
OUT = BASE / "figures"
OUT.mkdir(parents=True, exist_ok=True)

INK, MUTED, GRIDC = "#1b2430", "#5b6472", "#dde2e8"
BLUE, VERM, GREEN, ORANGE, PINK, SKY, PURP = ("#0072B2", "#D55E00", "#009E73", "#E69F00",
                                              "#CC79A7", "#56B4E9", "#7B52AB")
mpl.rcParams.update({
    "font.size": 8.4, "axes.edgecolor": GRIDC, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 9.4, "axes.titleweight": "bold",
    "axes.grid": True, "grid.color": GRIDC, "grid.linewidth": 0.5, "legend.frameon": False,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
})


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", name)


SW = pd.read_csv(BASE / "tables/p4_theme_sweep.csv")
CMP = pd.read_csv(BASE / "tables/p1_mss_v1_vs_v6.csv")
FID = pd.read_csv(BASE / "tables/p1_motif_band_fidelity.csv")
MAUD = pd.read_csv(BASE / "tables/p2_motif_audit.csv")
PER = pd.read_csv(BASE / "tables/p7_per_analyte.csv")
THREF = pd.read_csv(BASE / "tables/p6_theme_reference.csv")
REL = pd.read_csv(BASE / "tables/p7_reliability.csv")
P01 = json.loads((BASE / "artifacts/p0_p1_audit.json").read_text())
P04 = json.loads((BASE / "artifacts/p4_theme_optimisation.json").read_text())
P07 = json.loads((BASE / "artifacts/p7_evaluation.json").read_text())
V = np.load(BASE / "artifacts/p7_vectors.npz", allow_pickle=True)
THEMES = list(V["theme_names"]); MOTIFS = list(V["motif_ids"])
ANAL = list(V["analytes"]); Th = V["Th"]; A = V["A_bio"]; zA = V["zA"]
GRID = V["grid"]; SPEC = V["corpusX"]; C = V["confusion"]


# ═══ F1 · the V6 hierarchy ═══
def f01():
    fig, ax = plt.subplots(figsize=(11, 4.6)); ax.set_xlim(0, 100); ax.set_ylim(0, 42); ax.axis("off")

    def band(x, w, title, sub, col, items):
        ax.add_patch(FancyBboxPatch((x, 6), w, 30, boxstyle="round,pad=0.5,rounding_size=1.2",
                                    fc="white", ec=col, lw=1.6))
        ax.add_patch(FancyBboxPatch((x, 30.5), w, 5.5, boxstyle="round,pad=0.5,rounding_size=1.2",
                                    fc=col, ec=col))
        ax.text(x + w / 2, 33.2, title, ha="center", va="center", fontsize=9.2,
                weight="bold", color="white")
        ax.text(x + w / 2, 28.0, sub, ha="center", va="center", fontsize=7.0, color=MUTED)
        for i, it in enumerate(items):
            ax.text(x + w / 2, 25.0 - i * 2.05, it, ha="center", va="center", fontsize=6.4, color=INK)

    band(1, 20, "24 components", "frozen NMF basis · UNCHANGED",
         BLUE, ["c0 … c23", "", "learned once from", "375 pure-Raman spectra",
                "fingerprint 09ed804a…", "", "NNLS, H held fixed"])
    band(27, 27, "17 MSS motifs", "spectroscopy only — no theme input", VERM,
         [m.replace("_", " ") for m in MOTIFS[:11]] + ["…"])
    band(61, 24, "13 chemical themes", "derived FROM MSS", GREEN, THEMES[:11] + ["…"])
    band(90, 9, "biological\nstate", "future", "#9aa4b0", ["deferred", "", "needs", "functional", "evidence"])

    for x1, x2 in ((21, 27), (54, 61), (85, 90)):
        ax.add_patch(FancyArrowPatch((x1, 21), (x2, 21), arrowstyle="-|>", mutation_scale=13,
                                     color=MUTED, lw=1.5))
    ax.text(24, 23.6, "M", ha="center", fontsize=8.5, weight="bold", color=VERM)
    ax.text(57.5, 23.6, "T", ha="center", fontsize=8.5, weight="bold", color=GREEN)
    ax.text(50, 2.6, "theme(x) = Tᵀ · Mᵀ · coord(x)     — a composition of two non-negative linear maps",
            ha="center", fontsize=9.0, family="monospace", color=INK)
    ax.text(50, 0.2, "V1 fed the component→theme matrix into every MSS weight (25 % of the raw score), so "
                     "deriving themes from MSS would have been circular. V6 removes that term.",
            ha="center", fontsize=7.3, color=MUTED, style="italic")
    ax.set_title("The V6 semantic hierarchy", fontsize=11.5, weight="bold", pad=8)
    save(fig, "f01_hierarchy.png")


# ═══ F2 · theme leakage in V1 ═══
def f02():
    leak = pd.read_csv(BASE / "tables/p0_mss_theme_leakage.csv")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), gridspec_kw={"width_ratios": [1.1, 1.2, 1]})
    a = axes[0]
    a.hist(leak.theme_share_of_raw, bins=22, color=VERM, alpha=0.85)
    a.axvline(leak.theme_share_of_raw.mean(), color=INK, ls="--", lw=1.2)
    a.text(leak.theme_share_of_raw.mean() + .01, a.get_ylim()[1] * .9,
           f"mean {leak.theme_share_of_raw.mean():.1%}", fontsize=7.2, color=INK)
    a.set_xlabel("share of the V1 MSS score coming from the theme matrix")
    a.set_ylabel("contributor edges"); a.set_title("Every MSS weight is part theme")

    a = axes[1]
    g = leak.groupby("motif").theme_share_of_raw.mean().sort_values()
    a.barh(range(len(g)), g.values, color=[VERM if v > .15 else ORANGE for v in g.values])
    a.set_yticks(range(len(g))); a.set_yticklabels([m.replace("_", " ") for m in g.index], fontsize=6.4)
    a.set_xlabel("mean theme share"); a.set_title("Leakage per motif (V1)")
    a.grid(axis="y", visible=False)

    a = axes[2]; a.axis("off")
    lines = [("Where it enters", INK, True),
             ("src/gaira/engine/mss.py:195-196\n"
              "  theme = ontology.W[j, parent_theme]\n"
              "  raw = 0.40·band + 0.35·exemplar + 0.25·theme", MUTED, False),
             ("", INK, False),
             (f"{P01['n_edges_that_would_drop_below_keep_threshold']} of "
              f"{P01['n_contributor_edges']} contributor edges "
              f"({100*P01['n_edges_that_would_drop_below_keep_threshold']/P01['n_contributor_edges']:.0f} %) "
              "fall below the\nkeep threshold once the theme term is removed —\nthey exist only because of theme information.", VERM, True),
             ("", INK, False),
             ("Consequence: a themes-from-MSS hierarchy built on\nthe V1 layer would be circular. This is the single\nreason V6 rebuilds MSS first.", INK, True)]
    yy = 1.0
    for t, c, b in lines:
        a.text(0, yy, t, fontsize=7.3, color=c, weight="bold" if b else "normal",
               va="top", linespacing=1.55, family="monospace" if "mss.py" in t else "sans-serif")
        yy -= 0.07 + 0.06 * t.count("\n")
    fig.suptitle("Part 0 — theme leakage in the V1 MSS layer", fontsize=10.8, weight="bold", y=1.04)
    fig.tight_layout(); save(fig, "f02_leakage.png")


# ═══ F3 · V1 vs V6 MSS ═══
def f03():
    fig, axes = plt.subplots(1, 4, figsize=(11.4, 3.3))
    a = axes[0]
    x = np.arange(len(FID))
    a.barh(x - .2, FID.band_fidelity_v1, .4, color=MUTED, label="V1")
    a.barh(x + .2, FID.band_fidelity_v6, .4, color=GREEN, label="V6")
    a.set_yticks(x); a.set_yticklabels([m.replace("_", " ")[:20] for m in FID.motif], fontsize=5.8)
    a.set_xlabel("band fidelity"); a.legend(fontsize=7)
    a.set_title("Motif spectrum vs declared bands", fontsize=8.6); a.grid(axis="y", visible=False)

    a = axes[1]
    a.scatter(CMP.confidence_v1, CMP.confidence_v6, s=42, color=BLUE, edgecolor="white", lw=.6)
    lim = [0, max(CMP.confidence_v6.max(), CMP.confidence_v1.max()) * 1.1]
    a.plot(lim, lim, ls="--", color=MUTED, lw=1)
    a.set_xlabel("V1 confidence"); a.set_ylabel("V6 confidence")
    a.set_title("Confidence is no longer degenerate", fontsize=8.6)
    a.text(.05, .92, "V1 breadth ≡ 1/3 for every motif\n(a NumPy bool-OR artefact)",
           transform=a.transAxes, fontsize=6.6, color=VERM, va="top")

    a = axes[2]
    a.hist(CMP.component_weight_cosine, bins=12, color=SKY, alpha=.9)
    a.axvline(CMP.component_weight_cosine.mean(), color=INK, ls="--", lw=1.1)
    a.set_xlabel("cos(M_V1[:,m], M_V6[:,m])")
    a.set_title(f"Support preserved (mean {CMP.component_weight_cosine.mean():.2f})", fontsize=8.6)

    a = axes[3]
    a.scatter(CMP.stability_v1, CMP.stability_v6, s=42, color=GREEN, edgecolor="white", lw=.6)
    lim = [.7, .95]; a.plot(lim, lim, ls="--", color=MUTED, lw=1)
    a.set_xlabel("V1 stability"); a.set_ylabel("V6 stability")
    a.set_title("Stability unchanged", fontsize=8.6)
    fig.suptitle("Part 1 — MSS rebuilt without theme evidence: quality improves, support is preserved",
                 fontsize=10.6, weight="bold", y=1.05)
    fig.tight_layout(); save(fig, "f03_mss_v1_vs_v6.png")


# ═══ F4 · motif audit ═══
def f04():
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.3, 1, 1]})
    d = MAUD.sort_values("discriminative_auc")
    a = axes[0]
    a.barh(range(len(d)), d.discriminative_auc,
           color=[VERM if v < .75 else (ORANGE if v < .9 else GREEN) for v in d.discriminative_auc])
    a.axvline(.5, color=INK, ls="--", lw=1); a.text(.51, .3, "chance", fontsize=6.4, color=MUTED)
    a.set_yticks(range(len(d))); a.set_yticklabels([m.replace("_", " ") for m in d.motif], fontsize=6.2)
    a.set_xlim(.4, 1.02); a.set_xlabel("discriminative AUC (V1 motif set)")
    a.set_title("Does a motif activate on its own chemistry?", fontsize=8.8)
    a.grid(axis="y", visible=False)

    a = axes[1]
    cen = pd.read_csv(BASE / "tables/p2_family_census.csv", index_col=0).head(12)
    y = np.arange(len(cen))
    a.barh(y, cen.n_analytes, .62, color=GRIDC, label="corpus analytes")
    a.barh(y, cen.n_uncovered, .62, color=VERM, label="unclaimed by any V1 motif")
    a.set_yticks(y); a.set_yticklabels(cen.index, fontsize=6.4)
    a.invert_yaxis(); a.legend(fontsize=6.8)
    a.set_title("V1 exemplar coverage gap", fontsize=8.8); a.grid(axis="y", visible=False)

    a = axes[2]; a.axis("off")
    lines = [("V6 motif set — evidence-driven changes", INK, True), ("", INK, False),
             ("coverage 35.9 % → 98.8 %  ·  mean AUC 0.903 → 0.918", GREEN, True), ("", INK, False),
             ("SPLIT  lipid_acyl → fatty_acyl + triglyceride_ester\n       (15 triglycerides, 93 % unclaimed)", MUTED, False),
             ("REBUILT sterol_ring_system\n       (worst AUC 0.68; top-activated on fatty acids)", MUTED, False),
             ("NEW    amino_acid_zwitterion · polysaccharide_glycosidic\n       nucleic_backbone_phosphate · carotenoid_polyene", MUTED, False),
             ("DE-CONFOUNDED  citrate removed from the background motif\n       (it was an exemplar of two motifs at once)", MUTED, False),
             ("FLAGGED  porphyrin + flavin remain low-coverage:\n       no pure reference exists in the corpus", VERM, False)]
    yy = 1.0
    for t, c, b in lines:
        a.text(0, yy, t, fontsize=7.0, color=c, weight="bold" if b else "normal",
               va="top", linespacing=1.5)
        yy -= 0.06 + 0.055 * t.count("\n")
    fig.suptitle("Part 2 — auditing the motifs, and what V6 changes", fontsize=10.8, weight="bold", y=1.04)
    fig.tight_layout(); save(fig, "f04_motif_audit.png")


# ═══ F5 · theme optimisation curves + Pareto ═══
def f05():
    fig, axes = plt.subplots(1, 4, figsize=(11.6, 3.3))
    cols = {"A_manual": PURP, "B_activation": BLUE, "C_spectral": VERM,
            "D_ontology": GREEN, "E_hybrid": ORANGE}
    for a, (col, ttl, ylab) in zip(axes[:3],
                                   [("top1", "Raw accuracy vs K", "theme top-1"),
                                    ("kappa", "Recoverability vs K (chance-corrected)", "κ"),
                                    ("interpretability", "Interpretability vs K", "I")]):
        for m, g in SW.groupby("method"):
            g = g.sort_values("K")
            a.plot(g.K, g[col], "o-", color=cols[m], lw=1.4, ms=3.4, label=m.split("_")[1])
        if col == "top1":
            for m, g in SW.groupby("method"):
                g = g.sort_values("K")
                a.plot(g.K, g.null_top1, ":", color=cols[m], lw=1.0, alpha=.65)
            a.text(.97, .60, "dotted = permutation null", transform=a.transAxes, ha="right",
                   fontsize=6.6, color=MUTED)
        a.set_xlabel("number of chemical themes K"); a.set_ylabel(ylab); a.set_title(ttl, fontsize=8.8)
    axes[0].legend(fontsize=6.2, ncol=2, loc="lower left")

    a = axes[3]
    for m, g in SW.groupby("method"):
        a.scatter(g.kappa, g.interpretability, s=26, color=cols[m], alpha=.55, label=m.split("_")[1])
    p = SW[SW.pareto].sort_values("kappa")
    a.plot(p.kappa, p.interpretability, "-", color=INK, lw=1.2, alpha=.6)
    adm = SW[SW.chemically_admissible]
    a.scatter(adm.kappa, adm.interpretability, s=70, facecolor="none", edgecolor=GREEN, lw=1.1)
    sel = P04["selected"]
    a.scatter([sel["kappa"]], [sel["interpretability"]], s=190, marker="*", color=VERM,
              edgecolor="white", lw=.8, zorder=5)
    a.annotate(f"selected\n{sel['method']} K={sel['K']}", (sel["kappa"], sel["interpretability"]),
               fontsize=6.8, color=VERM, weight="bold", xytext=(10, -16), textcoords="offset points")
    raw = P04["raw_score_optimum"]
    a.annotate("raw optimum\n(inadmissible)", (raw["kappa"], raw["interpretability"]),
               fontsize=6.4, color=MUTED, xytext=(-14, -24), textcoords="offset points", ha="right")
    a.set_xlabel("recoverability  κ"); a.set_ylabel("interpretability  I")
    a.set_title("Pareto front — green ring = chemically admissible", fontsize=8.8)
    fig.suptitle("Part 4 — theme optimisation: accuracy alone is the wrong objective",
                 fontsize=10.8, weight="bold", y=1.05)
    fig.tight_layout(); save(fig, "f05_optimisation.png")


# ═══ F6 · per-theme performance + confusion + reliability ═══
def f06():
    fig = plt.figure(figsize=(11.4, 6.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[1.25, 1.15, 1],
                          hspace=.55, wspace=.35)
    d = THREF.dropna(subset=["top1"]).sort_values("top1")
    a = fig.add_subplot(gs[0, 0])
    y = np.arange(len(d))
    a.barh(y - .2, d.top1, .4, color=[GREEN if v >= .7 else (ORANGE if v >= .4 else VERM)
                                      for v in d.top1], label="top-1")
    a.barh(y + .2, d.top3, .4, color=SKY, alpha=.85, label="top-3")
    a.set_yticks(y); a.set_yticklabels([f"{t}  (n={n})" for t, n in zip(d.theme, d.n_analytes)], fontsize=6.4)
    a.set_xlim(0, 1.05); a.legend(fontsize=6.8, loc="lower right")
    a.set_title("Per-theme recovery", fontsize=8.8); a.grid(axis="y", visible=False)

    a = fig.add_subplot(gs[0, 1])
    Cn = C / np.maximum(C.sum(1, keepdims=True), 1)
    im = a.imshow(Cn, cmap="Blues", vmin=0, vmax=1)
    a.set_xticks(range(len(THEMES))); a.set_yticks(range(len(THEMES)))
    a.set_xticklabels(THEMES, rotation=60, ha="right", fontsize=5.4)
    a.set_yticklabels(THEMES, fontsize=5.4)
    a.set_xlabel("predicted"); a.set_ylabel("expected")
    a.set_title("Confusion (row-normalised)", fontsize=8.8); a.grid(False)
    fig.colorbar(im, ax=a, fraction=.045, pad=.03).ax.tick_params(labelsize=6)

    a = fig.add_subplot(gs[0, 2])
    a.plot([0, 1], [0, 1], ls="--", color=MUTED, lw=1)
    a.plot(REL.mean_confidence, REL.accuracy, "o-", color=VERM, lw=1.5, ms=5)
    for _, r in REL.iterrows():
        a.annotate(f"{int(r.n)}", (r.mean_confidence, r.accuracy), fontsize=6,
                   color=MUTED, xytext=(0, 6), textcoords="offset points", ha="center")
    a.set_xlabel("theme confidence"); a.set_ylabel("observed top-1 accuracy")
    a.set_title(f"Reliability  (ECE {P07['ece']:.3f})", fontsize=8.8)
    a.set_xlim(0, 1); a.set_ylim(0, 1.05)

    a = fig.add_subplot(gs[1, :2])
    lab = PER[PER.labelled]
    order = lab.sort_values(["theme_rank", "analyte"])
    a.scatter(range(len(order)), order.theme_rank, s=12,
              c=[GREEN if r == 1 else (ORANGE if r <= 3 else VERM) for r in order.theme_rank])
    a.set_yticks(range(1, 14)); a.set_ylabel("rank of the expected theme")
    a.set_xlabel("analyte (sorted)"); a.invert_yaxis()
    a.set_title(f"Every scored analyte  (n={len(order)})  ·  top-1 {P07['theme_top1']:.3f} · "
                f"top-3 {P07['theme_top3']:.3f}", fontsize=8.8)
    a.axhline(1.5, color=GREEN, lw=.8, ls="--"); a.axhline(3.5, color=ORANGE, lw=.8, ls="--")

    a = fig.add_subplot(gs[1, 2]); a.axis("off")
    lines = [("What the theme layer fixed", GREEN, True),
             ("sterol      0.00 → 0.58\nsulfur      0.00 → 0.11\nflavin/redox 0.00 → 0.88\ncarotenoid   n/a → 1.00",
              MUTED, False),
             ("", INK, False),
             ("What it cost", VERM, True),
             ("protein backbone 0.45 → 0.09\nFree amino acids were split out of\nprotein, and the amide-III / CH2 region\noverlaps the saccharide modes.", MUTED, False),
             ("", INK, False),
             ("top-3 improved 0.805 → 0.890", GREEN, True)]
    yy = 1.0
    for t, c, b in lines:
        a.text(0, yy, t, fontsize=7.0, color=c, weight="bold" if b else "normal",
               va="top", linespacing=1.5, family="monospace" if "→" in t and not b else "sans-serif")
        yy -= 0.07 + 0.06 * t.count("\n")
    fig.suptitle("Part 7 — full evaluation over every Raman grounding analyte",
                 fontsize=11.0, weight="bold", y=.99)
    save(fig, "f06_evaluation.png")


# ═══ F7 · the three-level radar (Part 9) ═══
def f07():
    reps = P07["representatives"]
    pick = [r for r in reps if r["tier"] in ("excellent", "moderate", "failure")]
    fig, axes = plt.subplots(len(pick), 3, figsize=(10.5, 3.3 * len(pick)),
                             subplot_kw=dict(projection="polar"))
    if len(pick) == 1:
        axes = axes[None, :]
    for r, rec in enumerate(pick):
        i = ANAL.index(rec["analyte"])
        for c, (vals, labels, title, col) in enumerate([
                (zA[i], [f"c{j}" for j in range(24)], "component radar", BLUE),
                (A[i], [m.replace("_", " ")[:14] for m in MOTIFS], "MSS motif radar", VERM),
                (Th[i], THEMES, "chemical theme radar", GREEN)]):
            a = axes[r, c]
            v = np.asarray(vals, float)
            ang = np.linspace(0, 2 * np.pi, len(v), endpoint=False)
            vv = np.concatenate([v, v[:1]]); aa = np.concatenate([ang, ang[:1]])
            a.plot(aa, vv, color=col, lw=1.5); a.fill(aa, vv, color=col, alpha=.22)
            a.set_xticks(ang)
            a.set_xticklabels(labels, fontsize=4.6 if len(v) > 15 else 5.4)
            a.set_yticklabels([]); a.grid(color=GRIDC, lw=.5)
            if r == 0:
                a.set_title(title, fontsize=9.0, weight="bold", color=col, pad=16)
            if c == 0:
                a.text(-0.32, .5, f"{rec['tier'].upper()}\n{rec['analyte']}",
                       transform=a.transAxes, fontsize=8.4, weight="bold",
                       color=INK, rotation=90, va="center", ha="center")
    fig.suptitle("Part 9 — the radar is now three nested radars, one per abstraction level\n"
                 "component (what the atlas saw) → motif (what spectroscopy says) → theme (what chemistry it implies)",
                 fontsize=10.4, weight="bold", y=1.005)
    fig.tight_layout(); save(fig, "f07_radars.png")


# ═══ F8 · end-to-end pathway (Part 10) ═══
def f08():
    reps = {r["tier"]: r for r in P07["representatives"]}
    pick = [reps[t] for t in ("excellent", "moderate", "failure") if t in reps]
    fig, axes = plt.subplots(len(pick), 4, figsize=(11.6, 2.7 * len(pick)),
                             gridspec_kw={"width_ratios": [1.35, 1.05, 1.2, 1.2]})
    for r, rec in enumerate(pick):
        i = ANAL.index(rec["analyte"])
        exp = rec["expected_themes"].split("|")[0]

        a = axes[r, 0]
        a.plot(GRID, SPEC[i], color=INK, lw=.8)
        a.fill_between(GRID, SPEC[i], color=INK, alpha=.10)
        a.set_yticks([]); a.tick_params(labelsize=6)
        a.set_title(f"{rec['tier'].upper()} · {rec['analyte']}", fontsize=8.4,
                    color=GREEN if rec["tier"] == "excellent" else
                    (ORANGE if rec["tier"] == "moderate" else VERM))
        if r == len(pick) - 1:
            a.set_xlabel("Raman shift (cm⁻¹)", fontsize=7)
        a.set_ylabel("1 · spectrum", fontsize=7)

        a = axes[r, 1]
        a.bar(range(24), zA[i], .78, color=BLUE)
        j = int(np.argmax(zA[i]))
        a.annotate(f"c{j}", (j, zA[i][j]), fontsize=6.6, color=VERM, weight="bold",
                   xytext=(0, 3), textcoords="offset points", ha="center")
        a.set_xticks(range(0, 24, 4)); a.tick_params(labelsize=6)
        a.set_ylabel("2 · components", fontsize=7)

        a = axes[r, 2]
        cols = [GREEN if MOTIFS[k] in rec["expected_themes"] or
                MOTIFS[k] == rec["predicted_motif"] else "#c8d0d8" for k in range(len(MOTIFS))]
        cols[int(np.argmax(A[i]))] = VERM if rec["motif_rank"] != 1 else GREEN
        a.barh(range(len(MOTIFS)), A[i], .74, color=cols)
        a.set_yticks(range(len(MOTIFS)))
        a.set_yticklabels([m.replace("_", " ")[:17] for m in MOTIFS], fontsize=4.8)
        a.tick_params(axis="x", labelsize=6); a.grid(axis="y", visible=False)
        a.set_ylabel("3 · MSS motifs", fontsize=7)

        a = axes[r, 3]
        cols = [GREEN if THEMES[k] == exp else "#c8d0d8" for k in range(len(THEMES))]
        if rec["theme_rank"] != 1:
            cols[int(np.argmax(Th[i]))] = VERM
        a.barh(range(len(THEMES)), Th[i], .74, color=cols)
        a.set_yticks(range(len(THEMES)))
        a.set_yticklabels(THEMES, fontsize=4.8)
        a.tick_params(axis="x", labelsize=6); a.grid(axis="y", visible=False)
        a.set_ylabel("4 · chemical themes", fontsize=7)
        a.text(.98, .04, f"expected rank {rec['theme_rank']}", transform=a.transAxes,
               ha="right", fontsize=6.6, weight="bold",
               color=GREEN if rec["theme_rank"] == 1 else VERM)
    fig.suptitle("Part 10 — one real analyte, end to end, at three performance tiers\n"
                 "green = expected · red = what actually won",
                 fontsize=10.4, weight="bold", y=1.01)
    fig.tight_layout(); save(fig, "f08_pathway.png")


# ═══ F9 · motif gallery: implied spectrum vs declared bands ═══
def f09():
    from v6_semantic.mss_v6 import motif_profile
    M = V["M_bio"]
    n = len(MOTIFS)
    ncol = 4; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(11.4, 1.75 * nrow))
    for k in range(nrow * ncol):
        a = axes.flat[k]
        if k >= n:
            a.axis("off"); continue
        s = M[:, k] @ np.load(REPO / "assets/foundation/manifold_components.npz")["components"]
        s = s / (s.max() + 1e-12)
        prof = motif_profile(MAUD.set_index("motif").loc[MOTIFS[k], "bands_cm"]
                             if isinstance(MAUD.set_index("motif").loc[MOTIFS[k], "bands_cm"], list)
                             else eval(str(MAUD.set_index("motif").loc[MOTIFS[k], "bands_cm"]))
                             if MOTIFS[k] in set(MAUD.motif) else [], GRID) \
            if MOTIFS[k] in set(MAUD.motif) else np.zeros_like(GRID)
        a.plot(GRID, s, color=VERM, lw=.9)
        a.fill_between(GRID, s, color=VERM, alpha=.16)
        if prof.max() > 0:
            a.plot(GRID, prof / prof.max(), color=BLUE, lw=.7, ls="--", alpha=.8)
        a.set_title(MOTIFS[k].replace("_", " "), fontsize=6.8)
        a.set_yticks([]); a.tick_params(labelsize=5.6)
    fig.suptitle("The 17 V6 MSS motifs — implied Raman spectrum (red, M[:,m]ᵀH) against the "
                 "declared band profile (blue dashed)", fontsize=10.2, weight="bold", y=1.005)
    fig.tight_layout(); save(fig, "f09_motif_gallery.png")


# ═══ F10 · component → MSS → theme flow ═══
def f10():
    M = V["M_bio"]; T = V["T"]
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6))
    a = axes[0]
    im = a.imshow(M.T, cmap="Oranges", aspect="auto", vmin=0)
    a.set_yticks(range(len(MOTIFS)))
    a.set_yticklabels([m.replace("_", " ") for m in MOTIFS], fontsize=6.4)
    a.set_xticks(range(0, 24, 2)); a.tick_params(labelsize=6.2)
    a.set_xlabel("NMF component")
    a.set_title("M · component → MSS motif  (columns sum to 1)", fontsize=9.0); a.grid(False)
    fig.colorbar(im, ax=a, fraction=.03, pad=.02).ax.tick_params(labelsize=6)

    a = axes[1]
    im = a.imshow(T.T, cmap="Greens", aspect="auto", vmin=0, vmax=1)
    a.set_yticks(range(len(THEMES))); a.set_yticklabels(THEMES, fontsize=6.4)
    a.set_xticks(range(len(MOTIFS)))
    a.set_xticklabels([m.replace("_", " ") for m in MOTIFS], rotation=70, ha="right", fontsize=5.6)
    a.set_title("T · MSS motif → chemical theme  (a hard partition)", fontsize=9.0); a.grid(False)
    fig.suptitle("The two interpretive maps of V6 — neither uses a theme label as input",
                 fontsize=10.6, weight="bold", y=1.0)
    fig.tight_layout(); save(fig, "f10_maps.png")


if __name__ == "__main__":
    print("figures →", OUT)
    for f in (f01, f02, f03, f04, f05, f06, f07, f08, f09, f10):
        try:
            f()
        except Exception as e:
            print(f"  !! {f.__name__} failed: {type(e).__name__}: {e}")
    print("done")
