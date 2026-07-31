"""Render the 7 global cross-modal-transfer figures (matplotlib, static PNG so each can be
audited headless). Reads only the theme-preservation tables/artifacts + the frozen engine.
Palette: Okabe-Ito (colorblind-safe). Additive; frozen atlas unchanged.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO = Path("/Users/surajpg/projects/GAIRA")
sys.path.insert(0, str(REPO / "src"))
from gaira.engine import GAIRAEngine
from gaira.engine.mss import MSSLayer

BASE = REPO / "results/v5_rebuild/pure_ag_sers_theme_preservation"
FIG = BASE / "figures"; FIG.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(BASE / "tables/per_analyte_transfer_metrics.csv")
summary = json.loads((BASE / "artifacts/theme_preservation_summary.json").read_text())

# Okabe-Ito colorblind-safe palette
OI = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
      "yellow": "#F0E442", "blue": "#0072B2", "verm": "#D55E00", "purple": "#CC79A7",
      "grey": "#999999"}
QCOL = {"Q1 identity preserved (both)": OI["blue"],
        "Q2 latent redistribution, theme survives": OI["green"],
        "Q3 superficial coord match, theme changes": OI["orange"],
        "Q4 poor transfer (both)": OI["verm"]}
plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.axisbelow": True,
                     "axes.spines.top": False, "axes.spines.right": False})


def _label_points(ax, x, y, names, which):
    for xi, yi, n in zip(x, y, names):
        if n in which:
            ax.annotate(n, (xi, yi), fontsize=7, xytext=(3, 3),
                        textcoords="offset points", color=OI["black"])


# ── FIG 1 (CENTERPIECE): component vs theme — naive (raw) vs honest (distinctive) ──
def fig1():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    hi = {"hypoxanthine", "xanthine", "adenine", "uracil", "guanine", "glucose",
          "riboflavin", "ergothioneine", "thymine", "creatinine", "albumin", "urate"}
    for ax, ycol, ylab, thr, title in [
        (axes[0], "theme_cosine", "RAW theme cosine  (baseline-inflated)", None,
         "A · Naive reading: theme cosine > component cosine for ALL 51"),
        (axes[1], "theme_cosine_distinct", "DISTINCTIVE theme cosine  (baseline-subtracted)",
         0.50, "B · Honest reading: theme preservation is selective")]:
        for q, sub in df.groupby("quadrant"):
            ax.scatter(sub.component_cosine, sub[ycol], s=48, alpha=0.85,
                       color=QCOL[q], edgecolor="white", linewidth=0.6, label=q, zorder=3)
        ax.plot([-1, 1], [-1, 1], ls="--", color=OI["grey"], lw=1, zorder=1)
        _label_points(ax, df.component_cosine, df[ycol], df.analyte, hi)
        ax.axvline(0.55, color=OI["grey"], ls=":", lw=0.8)
        if thr is not None:
            ax.axhline(thr, color=OI["grey"], ls=":", lw=0.8)
        ax.set_xlabel("Latent fingerprint preservation  (component cosine)")
        ax.set_ylabel(ylab); ax.set_title(title, fontsize=9.5, loc="left")
        ax.set_xlim(-0.02, 1.0)
    axes[0].set_ylim(0.3, 1.02); axes[1].set_ylim(-1.02, 1.02)
    handles = [Patch(color=QCOL[q], label=q.split(" ", 1)[1]) for q in QCOL]
    axes[1].legend(handles=handles, fontsize=7.2, loc="lower right", framealpha=0.9)
    fig.suptitle("Cross-modal transfer: latent fingerprint vs biochemical theme "
                 "(Raman → Ag-SERS, 51 pure analytes)", fontsize=12, y=0.99)
    fig.text(0.5, 0.005, "Left: raw theme cosine sits above the diagonal for every analyte — "
             "but this is compositional-baseline inflation, not preservation.  "
             "Right: baseline-subtracted, only strong chemisorbers keep an identity-specific theme.",
             ha="center", fontsize=7.6, color=OI["black"])
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(FIG / "fig1_component_vs_theme.png", bbox_inches="tight"); plt.close(fig)


# ── FIG 2: theme-preservation ranking (raw vs distinctive, per analyte) ──
def fig2():
    d = df.sort_values("theme_cosine_distinct", ascending=True)
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(9, 12))
    ax.barh(y, d.theme_cosine_distinct, color=[QCOL[q] for q in d.quadrant],
            edgecolor="white", linewidth=0.4, zorder=3, label="distinctive")
    ax.scatter(d.theme_null_mean, y, color=OI["black"], s=14, zorder=4,
               marker="|", label="null floor (other analytes)")
    ax.axvline(0, color=OI["grey"], lw=1)
    ax.set_yticks(y); ax.set_yticklabels(d.analyte, fontsize=7.5)
    ax.set_xlabel("Distinctive (baseline-subtracted) theme cosine   |   marker = null floor")
    ax.set_title("Identity-specific theme preservation, ranked\n"
                 "bar right of its null marker = preservation above background", fontsize=10)
    handles = [Patch(color=QCOL[q], label=q.split(" ", 1)[1]) for q in QCOL]
    handles.append(plt.Line2D([0], [0], marker="|", color=OI["black"], ls="",
                              label="null floor"))
    ax.legend(handles=handles, fontsize=7.4, loc="lower right")
    fig.tight_layout(); fig.savefig(FIG / "fig2_theme_ranking.png", bbox_inches="tight")
    plt.close(fig)


# ── FIG 3: paired theme-composition heatmap (Raman | Ag-SERS) ──
def fig3():
    eng = GAIRAEngine(); THEMES = eng.builder.onto.biochemical_theme_ids
    order = df.sort_values("theme_cosine_distinct", ascending=False).analyte.tolist()
    # recompute theme vectors directly from the frozen corpus + pure sers (same as analysis)
    sys.path.insert(0, str(REPO / "results/v5_rebuild/spike_validation/code"))
    import spike_lib as SL
    from gaira.foundation import dataset as DS
    from gaira.data.synonyms import canonical
    atlas = eng.atlas
    def coords(V): return atlas.coordinates(np.atleast_2d(np.nan_to_num(V)))
    def tvec(c):
        b = eng.infer(coordinates=np.asarray(c, float), domain="buffer").bsv
        return np.array([b.composition[t] for t in THEMES])
    corpus = DS.load_reference_corpus(); Zr = coords(corpus.X)
    Xs, rs = SL.load_pure_sers(); Zs = coords(Xs)
    R, S = {}, {}
    for a in pd.unique(corpus.meta.analyte):
        m = corpus.meta.analyte.values == a; R[a] = tvec(Zr[m].mean(0))
    for a in pd.unique(rs.analyte):
        m = rs.analyte.values == a; S[canonical(a)] = tvec(Zs[m].mean(0))
    order = [a for a in order if a in R and a in S]
    MR = np.array([R[a] for a in order]).T   # themes × analytes
    MS = np.array([S[a] for a in order]).T
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4), sharey=True)
    for ax, M, ttl in [(axes[0], MR, "Raman"), (axes[1], MS, "Ag-SERS")]:
        im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0, vmax=float(max(MR.max(), MS.max())))
        ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=90, fontsize=6)
        ax.set_yticks(range(len(THEMES))); ax.set_yticklabels(THEMES, fontsize=8)
        ax.set_title(f"{ttl} theme composition", fontsize=11); ax.grid(False)
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01, label="theme share")
    fig.suptitle("Theme composition, Raman vs Ag-SERS (analytes ordered by distinctive preservation)",
                 fontsize=12)
    fig.savefig(FIG / "fig3_theme_heatmap.png", bbox_inches="tight"); plt.close(fig)


# ── FIG 4: family comparison (component vs theme(raw) vs mss) ──
def fig4():
    fam = pd.read_csv(BASE / "tables/theme_preservation_by_family.csv")
    fam = fam.sort_values("component_cosine_mean", ascending=False)
    x = np.arange(len(fam)); w = 0.26
    fig, ax = plt.subplots(figsize=(12, 5.4))
    ax.bar(x - w, fam.component_cosine_mean, w, color=OI["verm"], label="latent fingerprint (component)")
    ax.bar(x, fam.mss_cosine_mean, w, color=OI["orange"], label="MSS motif")
    ax.bar(x + w, fam.theme_cosine_mean, w, color=OI["blue"], label="theme (raw)")
    for i, r in enumerate(fam.itertuples()):
        ax.annotate(f"n={r.n}", (i, 0.02), ha="center", fontsize=7, color="white")
    ax.set_xticks(x); ax.set_xticklabels(fam.family, rotation=35, ha="right")
    ax.set_ylabel("mean cosine"); ax.set_ylim(0, 1.05)
    ax.set_title("Transfer by biochemical family: preservation rises from latent → motif → theme\n"
                 "(raw theme is baseline-inflated; gap latent→theme is largest for weak adsorbers)",
                 fontsize=10)
    ax.legend(fontsize=8.5, loc="upper right")
    fig.tight_layout(); fig.savefig(FIG / "fig4_family_comparison.png", bbox_inches="tight")
    plt.close(fig)


# ── FIG 5: redistribution waterfalls for exemplar analytes ──
def fig5():
    eng = GAIRAEngine(); THEMES = eng.builder.onto.biochemical_theme_ids
    sys.path.insert(0, str(REPO / "results/v5_rebuild/spike_validation/code"))
    import spike_lib as SL
    from gaira.foundation import dataset as DS
    from gaira.data.synonyms import canonical
    atlas = eng.atlas
    def coords(V): return atlas.coordinates(np.atleast_2d(np.nan_to_num(V)))
    def tvec(c):
        b = eng.infer(coordinates=np.asarray(c, float), domain="buffer").bsv
        return np.array([b.composition[t] for t in THEMES])
    corpus = DS.load_reference_corpus(); Zr = coords(corpus.X)
    Xs, rs = SL.load_pure_sers(); Zs = coords(Xs)
    def get(a, Z, meta):
        m = meta.values == a; return tvec(Z[m].mean(0))
    picks = ["adenine", "hypoxanthine", "riboflavin", "glucose", "uracil", "cholesterol"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, a in zip(axes.ravel(), picks):
        tr = get(a, Zr, corpus.meta.analyte)
        sa = canonical(a); ms = rs.analyte.values == sa
        ts = tvec(Zs[ms].mean(0))
        d = ts - tr
        colors = [OI["green"] if v >= 0 else OI["verm"] for v in d]
        yy = np.arange(len(THEMES))
        ax.barh(yy, d, color=colors, edgecolor="white", linewidth=0.4)
        ax.axvline(0, color=OI["grey"], lw=1)
        ax.set_yticks(yy); ax.set_yticklabels(THEMES, fontsize=6.5)
        row = df[df.analyte == a].iloc[0]
        ax.set_title(f"{a}  (comp {row.component_cosine:.2f}, theme* {row.theme_cosine_distinct:.2f})",
                     fontsize=9)
        ax.set_xlabel("Δ theme share (SERS − Raman)", fontsize=8)
    fig.suptitle("Theme redistribution on Ag-SERS: where each analyte's composition moves\n"
                 "(green = gained on silver, red = lost)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG / "fig5_redistribution_waterfalls.png", bbox_inches="tight"); plt.close(fig)


# ── FIG 6: dominant-theme confusion matrix ──
def fig6():
    conf = pd.read_csv(BASE / "tables/dominant_theme_confusion.csv")
    themes = sorted(set(conf.raman_dominant) | set(conf.sers_dominant))
    M = np.zeros((len(themes), len(themes)))
    idx = {t: i for i, t in enumerate(themes)}
    for r in conf.itertuples():
        M[idx[r.raman_dominant], idx[r.sers_dominant]] = r.n
    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(M, cmap="magma", aspect="auto")
    ax.set_xticks(range(len(themes))); ax.set_xticklabels(themes, rotation=90, fontsize=8)
    ax.set_yticks(range(len(themes))); ax.set_yticklabels(themes, fontsize=8)
    ax.set_xlabel("Ag-SERS dominant theme"); ax.set_ylabel("Raman dominant theme")
    for i in range(len(themes)):
        for j in range(len(themes)):
            if M[i, j] > 0:
                ax.text(j, i, int(M[i, j]), ha="center", va="center",
                        color="white" if M[i, j] < M.max() * 0.6 else "black", fontsize=9)
    ax.grid(False)
    ax.set_title("Dominant-theme confusion (Raman → Ag-SERS)\n"
                 "off-diagonal mass on nucleic_purine column = the Ag 'purine attractor'", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="# analytes")
    fig.tight_layout(); fig.savefig(FIG / "fig6_dominant_theme_confusion.png", bbox_inches="tight")
    plt.close(fig)


# ── FIG 7: preservation vs OOD / confidence ──
def fig7():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    for ax, xcol, xlab in [(axes[0], "ood_sers", "Ag-SERS OOD score"),
                           (axes[1], "confidence_sers", "Ag-SERS confidence")]:
        for q, sub in df.groupby("quadrant"):
            ax.scatter(sub[xcol], sub.theme_cosine_distinct, s=46, color=QCOL[q],
                       edgecolor="white", linewidth=0.6, alpha=0.85, label=q.split(" ", 1)[1])
        ax.set_xlabel(xlab); ax.set_ylabel("distinctive theme cosine")
        ax.axhline(0, color=OI["grey"], lw=0.8, ls=":")
    axes[1].legend(fontsize=7.3, loc="lower right")
    fig.suptitle("Does the model KNOW when theme transfer fails? "
                 "Distinctive theme preservation vs OOD / confidence", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG / "fig7_preservation_vs_ood.png", bbox_inches="tight"); plt.close(fig)


# ── FIG 8: perturbation-sensitivity layer (dose curves + uricase directional) ──
def fig8():
    val = json.loads((REPO / "results/v5_rebuild/foundation_audit/tables/validation_results.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, key, col, ttl in [
        (axes[0], "3_adenine_dose", OI["blue"], "Adenine → purine theme"),
        (axes[1], "4_ergothioneine_dose", OI["green"], "Ergothioneine → sulfur theme")]:
        d = val[key]
        x = np.array(d["levels_uM"], float); y = np.array(d["theme_series"], float)
        o = np.argsort(x)
        ax.plot(x[o], y[o], "o-", color=col, lw=1.8, ms=5)
        ax.set_xlabel("concentration (µM)"); ax.set_ylabel(f"{d['theme']} share")
        ax.set_title(f"{ttl}\nρ={d['monotonicity_rho']}, {d['best_dose_model']} "
                     f"(K={d['saturating_K_uM']} µM, R²={d['saturating_r2']})", fontsize=9)
    # uricase directional depletion — motif vs theme
    u = val["6_uricase_depletion"]
    labels = ["oxopurine\ncarbonyl motif", "purine-ring\nbreathing motif", "purine\nTHEME (diffuse)"]
    vals = [u["delta_oxopurine_motif"], u["delta_purine_ring_motif"], u["purine_delta"]]
    axes[2].bar(labels, vals, color=[OI["verm"], OI["grey"], OI["grey"]], edgecolor="white")
    axes[2].axhline(0, color=OI["black"], lw=0.8)
    axes[2].set_ylabel("Δ on uricase (depletion)")
    axes[2].set_title("Uricase urate depletion\n(DIRECTIONAL: motif drops, theme diffuse — not a dose score)",
                      fontsize=9)
    fig.suptitle("Level 3 — Perturbation sensitivity (measured ONLY for adenine, ergothioneine, uricase)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIG / "fig8_perturbation_sensitivity.png", bbox_inches="tight"); plt.close(fig)


# ── FIG 9: matrix recoverability — pure transfer vs serum recovery ──
def fig9():
    link = pd.read_csv(BASE / "tables/matrix_recoverability_linkage.csv")
    link = link[link.serum_tested == True]
    TCOL = {"strong": OI["blue"], "moderate": OI["orange"], "weak": OI["verm"]}
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for t, sub in link.groupby("serum_recovery_tier"):
        ax.scatter(sub.component_cosine, sub.serum_spike_displacement, s=55,
                   color=TCOL.get(t, OI["grey"]), edgecolor="white", linewidth=0.6,
                   label=f"serum {t}", alpha=0.85)
    hi = {"hypoxanthine", "xanthine", "guanine", "adenine", "ergothioneine", "creatinine",
          "ascorbate", "uracil", "glucose"}
    for r in link.itertuples():
        if r.analyte in hi:
            ax.annotate(r.analyte, (r.component_cosine, r.serum_spike_displacement),
                        fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Pure Ag-SERS latent fingerprint preservation (component cosine)")
    ax.set_ylabel("Serum spike displacement (matrix recoverability)")
    ax.set_title("Level 4 — Matrix recoverability vs pure transfer\n"
                 "the same strong Ag chemisorbers survive both the modality and the matrix gap",
                 fontsize=10)
    ax.legend(fontsize=8.5, loc="upper left")
    fig.tight_layout(); fig.savefig(FIG / "fig9_matrix_recoverability.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig1(); print("fig1 ok")
    fig2(); print("fig2 ok")
    fig3(); print("fig3 ok")
    fig4(); print("fig4 ok")
    fig5(); print("fig5 ok")
    fig6(); print("fig6 ok")
    fig7(); print("fig7 ok")
    fig8(); print("fig8 ok")
    fig9(); print("fig9 ok")
    print("all figures ->", FIG)
