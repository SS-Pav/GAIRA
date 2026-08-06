#!/usr/bin/env python3
"""GAIRA V7 — Phase 01 figures (SVG vector + PNG preview).

Reads the Phase-01 tables and the serialised motif registry; performs no science of its
own. Deterministic: no RNG, no timestamps.

    python results/v7_rebuild/phase01/code/make_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
PHASE01 = HERE.parent
REPO = PHASE01.parent.parent.parent
sys.path.insert(0, str(REPO / "src"))

from gaira.v7.lsm import serialization as SER          # noqa: E402
from gaira.v7.lsm import visualization as VZ           # noqa: E402

T, F, A = PHASE01 / "tables", PHASE01 / "figures", PHASE01 / "artifacts"
INK, MUTED, LINE = VZ.INK, VZ.MUTED, VZ.LINE
BLUE, GREEN, AMBER, RED, GREY = VZ.BLUE, VZ.GREEN, VZ.AMBER, VZ.RED, VZ.GREY
VZ.apply_style()


def _save(fig, name):
    VZ.save(fig, F, name)
    print(f"  {name}.svg + {name}.png")


def load():
    df, spectra, ids, man = SER.load_registry(A)
    motifs = SER.motifs_from_table(df, spectra, ids)
    return df, motifs, man


def f01(comp):
    fig, ax = plt.subplots(figsize=(9.2, 8.4))
    VZ.component_motif_tree(comp, ax=ax)
    n_dec = int((comp.status == "DECOMPOSED").sum())
    n_irr = int((comp.status == "IRREDUCIBLE").sum())
    ax.set_title("1 — Component → motif tree\n"
                 f"{n_dec} of {len(comp)} frozen atlas components decompose; "
                 f"{n_irr} remain a single substructure",
                 fontsize=11, weight="bold", color=INK, loc="left", pad=14)
    _save(fig, "fig01_component_motif_tree")


def f02(motifs):
    """Motif spectra for the four components that decompose most strongly."""
    amb = pd.read_csv(T / "ambiguity_resolution_v1.csv")
    top = amb.sort_values("purity_gain", ascending=False).head(4).component.tolist()
    grid = np.asarray(np.load(REPO / "assets/foundation/manifold_components.npz")["grid"], float)
    H = np.asarray(np.load(REPO / "assets/foundation/manifold_components.npz")["components"], float)
    fig, axes = plt.subplots(len(top), 1, figsize=(9.4, 2.5 * len(top)), sharex=True)
    for ax, k in zip(np.atleast_1d(axes), top):
        ms = sorted([m for m in motifs if m.parent_component == k],
                    key=lambda m: -m.n_analytes)
        VZ.motif_spectra(ms, grid, parent=H[k], ax=ax)
        r = amb[amb.component == k].iloc[0]
        ax.set_title(f"component c{k:02d} — {int(r.n_retained_motifs)} motifs · purity "
                     f"{r.component_dominant_share:.2f} → {r.weighted_motif_purity:.2f}",
                     fontsize=8.6, loc="left", color=INK)
        ax.set_xlabel("")
    np.atleast_1d(axes)[-1].set_xlabel("Raman shift (cm$^{-1}$)")
    fig.suptitle("2 — Motif spectra: substructures of the frozen components\n"
                 "grey = parent atlas component (unchanged); coloured = its motifs",
                 fontsize=11, weight="bold", color=INK, x=0.005, ha="left", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    _save(fig, "fig02_motif_spectra")


def f03(motifs):
    C = pd.read_csv(T / "motif_overlap_matrix_v1.csv", index_col=0)
    red = json.loads((A / "redundancy_summary_v1.json").read_text())
    fig, ax = plt.subplots(figsize=(8.2, 8.6))
    VZ.overlap_graph(C, motifs, ax=ax, threshold=0.5)
    ax.set_title("3 — Motif overlap graph (cosine ≥ 0.5)\n"
                 f"max off-diagonal cosine {red['max_offdiag_cosine']} — "
                 f"grey = same parent component, red = cross-component",
                 fontsize=11, weight="bold", color=INK, loc="left", pad=12)
    _save(fig, "fig03_motif_overlap_graph")


def f04(df):
    kept = df[df.retained]
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.2))
    VZ.score_distribution(kept, "purity", ax=axes[0], color=BLUE, label="chemical purity")
    VZ.score_distribution(kept, "stability", ax=axes[1], color=GREEN, threshold=0.50,
                          label="jackknife stability")
    VZ.score_distribution(kept, "coverage_analytes", ax=axes[2], color=AMBER,
                          label="coverage (share of component participants)")
    for ax, t in zip(axes, ("purity", "stability", "coverage")):
        ax.set_title(t, fontsize=9, color=INK, loc="left")
    fig.suptitle("4 — Motif quality scores across the retained layer",
                 fontsize=11, weight="bold", color=INK, x=0.005, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    _save(fig, "fig04_motif_quality")


def f05():
    amb = pd.read_csv(T / "ambiguity_resolution_v1.csv")
    pn = pd.read_csv(T / "purity_null_v1.csv")
    fig, axes = plt.subplots(2, 1, figsize=(10.2, 7.2))
    VZ.ambiguity_waterfall(amb, ax=axes[0])
    axes[0].set_title("whole component vs its motif layer", fontsize=9, loc="left", color=INK)

    d = pn.sort_values("gain_beyond_mechanical", ascending=False)
    x = np.arange(len(d))
    cols = [GREEN if s else GREY for s in d.significant]
    axes[1].bar(x, d.gain_beyond_mechanical, .72, color=cols)
    axes[1].axhline(0, color=INK, lw=.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"c{int(c):02d}" for c in d.component], fontsize=6.4, rotation=90)
    axes[1].set_ylabel("purity beyond a size-matched\nrandom partition")
    axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].set_title(f"gain beyond mechanical — green = p < 0.05 "
                      f"({int(d.significant.sum())} of {len(d)})",
                      fontsize=9, loc="left", color=INK)
    fig.suptitle("5 — Does the motif layer resolve chemical ambiguity?\n"
                 "Top: raw purity. Bottom: the part that is NOT the mechanical effect of "
                 "cutting a set into more pieces.",
                 fontsize=11, weight="bold", color=INK, x=0.005, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    _save(fig, "fig05_ambiguity_resolution")


def f06(df):
    """Coverage: motifs per component and per analyte."""
    comp = pd.read_csv(T / "lsm_components_v1.csv")
    cov = json.loads((A / "coverage_report_v1.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.4))
    d = comp.sort_values("n_retained_motifs", ascending=False)
    cols = [BLUE if s == "DECOMPOSED" else GREY for s in d.status]
    axes[0].bar(np.arange(len(d)), d.n_retained_motifs, .74, color=cols)
    axes[0].set_xticks(np.arange(len(d)))
    axes[0].set_xticklabels([f"c{int(c):02d}" for c in d.component], fontsize=6, rotation=90)
    axes[0].set_ylabel("retained motifs")
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[0].set_title("motifs per atlas component", fontsize=9, loc="left", color=INK)

    kept = df[df.retained]
    axes[1].hist(kept.n_analytes, bins=range(1, int(kept.n_analytes.max()) + 2),
                 color=AMBER, edgecolor="white", linewidth=.5)
    axes[1].axvline(3, color=RED, lw=1.1, ls="--")
    axes[1].text(3, axes[1].get_ylim()[1] * .95, " reject < 3", fontsize=6.4, color=RED,
                 va="top")
    axes[1].set_xlabel("participating molecules per motif")
    axes[1].set_ylabel("motifs")
    axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].set_title(f"motif support — {cov['analyte_coverage']:.0%} of molecules covered, "
                      f"{cov['motifs_per_analyte_mean']:.1f} motifs each",
                      fontsize=9, loc="left", color=INK)
    fig.suptitle("6 — Coverage of the motif layer", fontsize=11, weight="bold",
                 color=INK, x=0.005, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    _save(fig, "fig06_coverage")


def f07(motifs):
    part = pd.read_csv(REPO / "results/v7_rebuild/phase00/tables/chemical_partition_v1.csv")
    class_of = dict(zip(part.canonical_id, part.fine_class))
    ids = sorted(class_of)
    M = pd.DataFrame(0, index=ids, columns=[m.motif_id for m in motifs])
    for m in motifs:
        for a in m.analytes:
            if a in M.index:
                M.loc[a, m.motif_id] = 1
    fig, ax = plt.subplots(figsize=(11.5, 9))
    VZ.participation_heatmap(M, class_of, ax=ax, max_analytes=90)
    ax.set_title("7 — Analyte × motif participation, ordered by chemical class\n"
                 "horizontal banding within a class = motifs that track chemistry",
                 fontsize=11, weight="bold", color=INK, loc="left", pad=12)
    _save(fig, "fig07_participation_heatmap")


def f08(df):
    """Representative analytes of the purest well-supported motifs."""
    kept = df[df.retained & (df.n_analytes >= 4)].sort_values(
        ["purity", "n_analytes"], ascending=[False, False]).head(12)
    fig, ax = plt.subplots(figsize=(10.6, 5.4))
    ax.axis("off")
    y = len(kept)
    for _, r in kept.iterrows():
        ax.text(0, y, r.motif_id, fontsize=7.6, weight="bold", color=INK, va="center")
        ax.text(7, y, r.dominant_class.replace("_", " "), fontsize=7.2, color=BLUE,
                va="center")
        ax.text(24, y, f"n={int(r.n_analytes)}  purity {r.purity:.2f}  "
                       f"stability {r.stability:.2f}", fontsize=7.0, color=MUTED, va="center")
        ax.text(45, y, str(r.band_centers_cm).replace(";", ", ") + " cm⁻¹",
                fontsize=6.6, color=INK, va="center")
        ex = [a for a in str(r.analytes).split(";")][:4]
        ax.text(72, y, ", ".join(a[:16] for a in ex), fontsize=6.4, color=MUTED, va="center")
        y -= 1
    ax.set_xlim(-1, 100)
    ax.set_ylim(0, len(kept) + 1.5)
    ax.text(0, len(kept) + 1.1, "motif", fontsize=7, color=MUTED)
    ax.text(7, len(kept) + 1.1, "dominant class", fontsize=7, color=MUTED)
    ax.text(24, len(kept) + 1.1, "support", fontsize=7, color=MUTED)
    ax.text(45, len(kept) + 1.1, "bands", fontsize=7, color=MUTED)
    ax.text(72, len(kept) + 1.1, "representative molecules", fontsize=7, color=MUTED)
    ax.set_title("8 — Representative motifs: the purest well-supported substructures",
                 fontsize=11, weight="bold", color=INK, loc="left", pad=14)
    _save(fig, "fig08_representative_motifs")


def f09():
    """Motif hierarchy: how the 24 components and their motifs map onto chemistry."""
    comp = pd.read_csv(T / "lsm_components_v1.csv")
    reg = pd.read_csv(T / "lsm_registry_v1.csv")
    kept = reg[reg.retained]
    fig, ax = plt.subplots(figsize=(10.6, 6.6))
    ax.axis("off")
    classes = (kept.groupby("dominant_class").size().sort_values(ascending=False))
    ax.text(2, 96, "FROZEN ATLAS", fontsize=9.5, weight="bold", color=GREY)
    ax.text(2, 92, f"{len(comp)} components — unchanged, fingerprint intact",
            fontsize=7.4, color=MUTED)
    ax.add_patch(Rectangle((2, 82), 96, 7, fc="#f3f4f6", ec=GREY, lw=1.1))
    for i in range(24):
        ax.add_patch(Rectangle((3 + i * 3.95, 83.2), 3.3, 4.6, fc="#d1d5db", ec="none"))
        ax.text(4.65 + i * 3.95, 85.5, f"{i:02d}", fontsize=5.2, ha="center", color=INK)

    ax.annotate("", xy=(50, 74), xytext=(50, 81),
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.4))
    ax.text(52, 77.5, "deterministic decomposition (no fitting)", fontsize=7.2, color=BLUE)

    ax.text(2, 71, "MOTIF LAYER", fontsize=9.5, weight="bold", color=BLUE)
    ax.text(2, 67.5, f"{len(kept)} retained Local Spectral Motifs "
                     f"({len(reg) - len(kept)} rejected, reasons recorded)",
            fontsize=7.4, color=MUTED)
    x = 2.0
    for cls, n in classes.items():
        w = 96 * n / classes.sum()
        ax.add_patch(Rectangle((x, 56), max(w - 0.4, 0.6), 8, fc=BLUE, ec="white",
                               lw=.6, alpha=.85))
        if w > 5.5:
            ax.text(x + w / 2, 60, f"{cls.replace('_', ' ')[:18]}\n{n}", fontsize=5.6,
                    ha="center", va="center", color="white")
        x += w
    ax.text(2, 52, "motifs grouped by their dominant chemical class "
                   "(class labels are EVALUATION ONLY — never used to build motifs)",
            fontsize=6.8, color=MUTED)

    dec = int((comp.status == "DECOMPOSED").sum())
    irr = int((comp.status == "IRREDUCIBLE").sum())
    al = pd.read_csv(T / "chemical_alignment_v1.csv")
    pn = pd.read_csv(T / "purity_null_v1.csv")
    ax.add_patch(Rectangle((2, 8), 96, 38, fc="#f9fafb", ec=LINE, lw=1.0))
    lines = [
        "WHAT THE LAYER DELIVERS",
        "",
        f"components decomposed              {dec} of {len(comp)}",
        f"components irreducible             {irr}",
        f"retained motifs                    {len(kept)}   "
        f"(mean {len(kept)/len(comp):.1f} per component)",
        f"aligned with chemistry (p<0.05)    {int(al.significant.sum())} components",
        f"purity above a SIZE-MATCHED null   {int(pn.significant.sum())} of {len(pn)} components, "
        f"median +{pn.gain_beyond_mechanical.median():.3f}",
        "",
        "The atlas, its projection and its fingerprint are unchanged. The motif layer only",
        "redistributes an activation the frozen atlas already produced — attributed evidence",
        "equals atlas activation to machine precision.",
    ]
    for i, t in enumerate(lines):
        ax.text(5, 42 - i * 3.1, t, fontsize=7.2 if i else 8.4,
                weight="bold" if i == 0 else "normal", color=INK, family="DejaVu Sans")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_title("9 — Motif hierarchy: frozen atlas → interpretation layer",
                 fontsize=11, weight="bold", color=INK, loc="left", pad=12)
    _save(fig, "fig09_motif_hierarchy")


if __name__ == "__main__":
    print(f"writing Phase 01 figures to {F}")
    df, motifs, man = load()
    comp = pd.read_csv(T / "lsm_components_v1.csv")
    f01(comp)
    f02(motifs)
    f03(motifs)
    f04(df)
    f05()
    f06(df)
    f07(motifs)
    f08(df)
    f09()
    print("done — 9 figures (SVG vector + PNG preview)")
