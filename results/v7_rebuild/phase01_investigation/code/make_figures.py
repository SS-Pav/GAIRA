#!/usr/bin/env python3
"""GAIRA V7 — Phase 01 investigation figures (SVG vector + PNG preview)."""
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
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.optimize import nnls
from scipy.spatial.distance import squareform

HERE = Path(__file__).resolve().parent
INV = HERE.parent
REPO = INV.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "results/v7_rebuild/phase00/code"))
sys.path.insert(0, str(REPO / "src"))

T, F, A = INV / "tables", INV / "figures", INV / "artifacts"
INK, MUTED, LINE = "#1a1a1a", "#6b7280", "#9ca3af"
BLUE, GREEN, AMBER, RED, GREY = "#2563eb", "#15803d", "#b45309", "#b91c1c", "#4b5563"
PAL = ["#2563eb", "#15803d", "#b45309", "#7c3aed", "#0891b2", "#be123c", "#4d7c0f"]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
                     "figure.facecolor": "white", "savefig.facecolor": "white",
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.18,
                     "svg.fonttype": "none"})


def save(fig, n):
    F.mkdir(parents=True, exist_ok=True)
    fig.savefig(F / f"{n}.svg", format="svg")
    fig.savefig(F / f"{n}.png", dpi=200)
    plt.close(fig)
    print(f"  {n}.svg + {n}.png")


def f01_before_after():
    """The k_c fix: before vs after, on held-out evidence."""
    d = pd.DataFrame([
        ("peptide_protein", 30, 2, 10, 0.180, 0.938, 0.645, 0.718),
        ("acylglycerol", 17, 1, 3, 0.288, 0.872, 0.580, 0.800),
        ("fatty_acid", 17, 2, 5, 0.595, 0.902, 0.579, 0.674),
        ("free_amino_acid", 18, 5, 7, 0.488, 0.633, 0.266, 0.281),
        ("sterol_steroid", 10, 2, 3, 0.575, 0.816, 0.508, 0.585),
        ("mono_oligosaccharide", 20, 5, 6, 0.583, 0.614, 0.340, 0.346),
        ("polysaccharide", 5, 1, 2, 0.585, 0.869, 0.520, 0.560),
        ("carboxylic_acid_metab.", 8, 3, 2, 0.322, 0.307, 0.144, 0.133),
    ], columns=["cls", "n", "k_old", "k_new", "worst_old", "worst_new", "ho_old", "ho_new"])
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.9))
    x = np.arange(len(d))
    axes[0].bar(x - .2, d.k_old, .38, color=GREY, label="before fix")
    axes[0].bar(x + .2, d.k_new, .38, color=BLUE, label="after fix")
    axes[0].set_ylabel("selected $k_c$")
    axes[1].bar(x - .2, d.worst_old, .38, color=GREY)
    axes[1].bar(x + .2, d.worst_new, .38, color=GREEN)
    axes[1].axhline(0.5, color=RED, ls="--", lw=1)
    axes[1].set_ylabel("WORST molecule explained variance")
    axes[2].bar(x - .2, d.ho_old, .38, color=GREY)
    axes[2].bar(x + .2, d.ho_new, .38, color=AMBER)
    axes[2].set_ylabel("HELD-OUT explained variance")
    for ax, t in zip(axes, ("$k_c$ selected", "in-sample worst case", "generalisation")):
        ax.set_xticks(x)
        ax.set_xticklabels(d.cls, rotation=90, fontsize=6.2)
        ax.set_title(t, fontsize=9, loc="left", color=INK)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle("1 — The duplicate-fraction fix: before vs after\n"
                 "held-out EV rises wherever $k_c$ rose, so the extra components generalise "
                 "rather than memorise",
                 fontsize=11, weight="bold", color=INK, x=.005, ha="left")
    fig.tight_layout(rect=(0, 0, 1, .86))
    save(fig, "fig01_kc_fix_before_after")


def f02_uniqueness(ctx):
    u = pd.read_csv(T / "inv1_uniqueness_v1.csv")
    big = u[u.k >= 3].sort_values("k", ascending=False).chemical_class.tolist()[:6]
    fig, axes = plt.subplots(2, 3, figsize=(11.6, 7.2))
    for ax, c in zip(axes.ravel(), big):
        Hc = ctx.Hc[c]
        N = Hc / (np.linalg.norm(Hc, axis=1, keepdims=True) + 1e-12)
        C = N @ N.T
        im = ax.imshow(C, cmap="RdYlBu_r", vmin=0, vmax=1)
        ax.set_title(f"{c.replace('_', ' ')} (k={Hc.shape[0]})\nmax off-diag "
                     f"{C[np.triu_indices(len(C), 1)].max():.3f}",
                     fontsize=7.6, loc="left", color=INK)
        ax.set_xticks(range(len(C)))
        ax.set_yticks(range(len(C)))
        ax.set_xticklabels([f"m{i:02d}" for i in range(len(C))], fontsize=5, rotation=90)
        ax.set_yticklabels([f"m{i:02d}" for i in range(len(C))], fontsize=5)
        for i in range(len(C)):
            for j in range(len(C)):
                if i != j and C[i, j] >= 0.8:
                    ax.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center", fontsize=4.6,
                            color="white")
    for ax in axes.ravel()[len(big):]:
        ax.axis("off")
    fig.colorbar(im, ax=axes, fraction=.02, label="cosine")
    fig.suptitle("2 — LSM uniqueness within each class\n"
                 "no pair reaches the 0.95 duplication threshold; the layer contains no "
                 "duplicate motifs",
                 fontsize=11, weight="bold", color=INK, x=.005, ha="left")
    save(fig, "fig02_uniqueness_heatmaps")


def f03_dendrograms(ctx):
    u = pd.read_csv(T / "inv1_uniqueness_v1.csv")
    big = u[u.k >= 4].sort_values("k", ascending=False).chemical_class.tolist()[:4]
    fig, axes = plt.subplots(1, len(big), figsize=(3.1 * len(big), 3.6))
    for ax, c in zip(np.atleast_1d(axes), big):
        Hc = ctx.Hc[c]
        N = Hc / (np.linalg.norm(Hc, axis=1, keepdims=True) + 1e-12)
        D = np.clip(1 - N @ N.T, 0, None)
        np.fill_diagonal(D, 0)
        Z = linkage(squareform(D, checks=False), "average")
        dendrogram(Z, ax=ax, labels=[f"m{i:02d}" for i in range(len(D))],
                   color_threshold=0.35, leaf_font_size=6)
        ax.axhline(0.05, color=RED, ls="--", lw=1)
        ax.set_title(f"{c.replace('_', ' ')}", fontsize=8, loc="left", color=INK)
        ax.set_ylabel("cosine distance")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("3 — Motif dendrograms (red line = the 0.95-cosine duplication threshold)\n"
                 "no branch merges below it: nothing should be collapsed to a lower $k_c$",
                 fontsize=11, weight="bold", color=INK, x=.005, ha="left")
    fig.tight_layout(rect=(0, 0, 1, .85))
    save(fig, "fig03_motif_dendrograms")


def f04_reconstruction():
    d = pd.read_csv(T / "inv2_per_molecule_reconstruction_v1.csv")
    s = pd.read_csv(T / "inv2_class_reconstruction_summary_v1.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6),
                             gridspec_kw={"width_ratios": [1.6, 1]})
    order = s.sort_values("ev_mean").chemical_class.tolist()
    data = [d[d.chemical_class == c].ev.values for c in order]
    bp = axes[0].boxplot(data, vert=False, patch_artist=True, widths=.62)
    for p, c in zip(bp["boxes"], order):
        n = int(s[s.chemical_class == c].n.iloc[0])
        p.set_facecolor(RED if n <= 5 else (AMBER if n <= 10 else BLUE))
        p.set_alpha(.75)
    axes[0].axvline(0.5, color=RED, ls="--", lw=1)
    axes[0].axvline(0.7, color=AMBER, ls="--", lw=1)
    axes[0].set_yticklabels([f"{c.replace('_', ' ')}  (n={int(s[s.chemical_class == c].n.iloc[0])}, "
                             f"k={int(s[s.chemical_class == c].k_c.iloc[0])})" for c in order],
                            fontsize=6.4)
    axes[0].set_xlabel("per-molecule explained variance")
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[0].set_title("distribution per chemistry class (red = n≤5)", fontsize=9, loc="left")

    w = d.nsmallest(12, "ev")
    axes[1].barh(np.arange(len(w)), w.ev, .68,
                 color=[RED if v < .5 else AMBER for v in w.ev])
    axes[1].set_yticks(np.arange(len(w)))
    axes[1].set_yticklabels([f"{m}  [{c.replace('_', ' ')[:14]}]"
                             for m, c in zip(w.molecule, w.chemical_class)], fontsize=6)
    axes[1].invert_yaxis()
    axes[1].axvline(0.5, color=RED, ls="--", lw=1)
    axes[1].set_xlabel("explained variance")
    axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].set_title("the 12 worst-reconstructed molecules", fontsize=9, loc="left")
    fig.suptitle(f"4 — Per-molecule reconstruction: corpus mean EV {d.ev.mean():.3f}, "
                 f"{int((d.ev < 0.5).sum())} of {len(d)} molecules below 0.5\n"
                 "class averages conceal the tail — every weak molecule sits in a class of "
                 "≤8 molecules",
                 fontsize=11, weight="bold", color=INK, x=.005, ha="left")
    fig.tight_layout(rect=(0, 0, 1, .88))
    save(fig, "fig04_per_molecule_reconstruction")


def f05_residuals(ctx):
    """Representative, reconstructed and residual for the named chemistries."""
    picks = [("peptide_protein", None), ("acylglycerol", None), ("fatty_acid", None),
             ("purine", None), ("pyrimidine", "thymine"), ("small_nitrogenous", "urea")]
    fig, axes = plt.subplots(len(picks), 1, figsize=(9.8, 2.15 * len(picks)), sharex=True)
    for ax, (c, want) in zip(axes, picks):
        Hc, mols, Xb = ctx.Hc[c], ctx.mol[c], ctx.Xc[c]
        i = mols.index(want) if (want and want in mols) else 0
        x = Xb[i]
        a, _ = nnls(Hc.T, np.maximum(x, 0))
        r = a @ Hc
        ss = float(np.sum(x ** 2)) or 1
        ev = 1 - float(np.sum((x - r) ** 2)) / ss
        ax.plot(ctx.grid, x, lw=1.3, color=INK, label=f"{mols[i]} (observed)")
        ax.plot(ctx.grid, r, lw=1.1, color=BLUE, ls="--",
                label=f"reconstruction, k$_c$={Hc.shape[0]}")
        ax.fill_between(ctx.grid, 0, np.abs(x - r), color=RED, alpha=.3, label="|residual|")
        ax.set_title(f"{c.replace('_', ' ')} — EV {ev:.3f}", fontsize=8.4, loc="left",
                     color=(RED if ev < .5 else INK))
        ax.legend(fontsize=5.8, frameon=False, ncol=3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("Raman shift (cm$^{-1}$)")
    fig.suptitle("5 — Observed, reconstructed and residual for the named chemistries\n"
                 "thymine and urea sit in 3- and 2-molecule classes where $k_c$ is "
                 "ceiling-bound at 1",
                 fontsize=11, weight="bold", color=INK, x=.005, ha="left")
    fig.tight_layout(rect=(0, 0, 1, .95))
    save(fig, "fig05_reconstruction_residuals")


def f06_kc_robustness():
    d = pd.read_csv(T / "inv3_kc_robustness_v1.csv")
    k = pd.read_csv(T / "inv3_knife_edge_v1.csv")
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8))
    for c, g in d.groupby("chemical_class"):
        g = g.sort_values("delta_k")
        axes[0].plot(g.delta_k, g.heldout_ev, "-o", ms=3, lw=.9, alpha=.75)
        axes[1].plot(g.delta_k, g.stability, "-o", ms=3, lw=.9, alpha=.75)
        axes[2].plot(g.delta_k, g.basis_match_to_selected, "-o", ms=3, lw=.9, alpha=.75)
    for ax, t, yl in zip(axes,
                         ("held-out reconstruction", "recurrence stability",
                          "basis match to the selected $k_c$"),
                         ("held-out EV", "stability", "mean Hungarian cosine")):
        ax.axvline(0, color=GREEN, lw=1.4)
        ax.set_xticks([-1, 0, 1])
        ax.set_xticklabels(["$k_c$−1", "$k_c$", "$k_c$+1"])
        ax.set_ylabel(yl)
        ax.set_title(t, fontsize=9, loc="left", color=INK)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"6 — $k_c$ robustness: {int(k.knife_edge.sum())} of {len(k)} classes on a "
                 f"knife edge\nno neighbouring $k$ gains more than 0.05 held-out EV in any class",
                 fontsize=11, weight="bold", color=INK, x=.005, ha="left")
    fig.tight_layout(rect=(0, 0, 1, .86))
    save(fig, "fig06_kc_robustness")


def f07_source(ctx):
    src = pd.read_csv(T / "inv4_source_consistency_v1.csv")
    per = pd.read_csv(T / "inv4_source_per_motif_v1.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.2))
    t = src[src.testable]
    nt = src[~src.testable]
    axes[0].barh(np.arange(len(t)), t.frac_motifs_differing, .6,
                 color=[RED if v > 0 else GREEN for v in t.frac_motifs_differing])
    axes[0].set_yticks(np.arange(len(t)))
    axes[0].set_yticklabels([f"{c.replace('_', ' ')} (n={int(n)})"
                             for c, n in zip(t.chemical_class, t.n_spectra)], fontsize=7)
    axes[0].set_xlabel("fraction of motifs with source-dependent activation\n"
                       "(Mann–Whitney, Bonferroni-corrected)")
    axes[0].set_xlim(0, 1.05)
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[0].set_title(f"{len(t)} testable classes", fontsize=9, loc="left")

    axes[1].axis("off")
    axes[1].text(0, 1, "NOT TESTABLE — single dominant source", fontsize=9, weight="bold",
                 color=AMBER, va="top")
    axes[1].text(0, .93, f"{len(nt)} of {len(src)} classes cannot be tested for source\n"
                         f"confounding, because one library supplies essentially all\n"
                         f"their molecules.", fontsize=7.6, color=INK, va="top")
    y = .78
    for _, r in nt.iterrows():
        flag = r.chemical_class in ("peptide_protein", "acylglycerol", "sterol_steroid",
                                    "nucleic_acid_polymer")
        axes[1].text(.02, y, ("⚠ " if flag else "· ") + r.chemical_class.replace("_", " "),
                     fontsize=7, color=(RED if flag else MUTED), va="top")
        y -= .055
    axes[1].text(0, y - .02,
                 "⚠ = flagged source-confounded in Phase 00/01.\n"
                 "These are exactly the classes the test cannot reach:\n"
                 "source confounding there is UNTESTED, not disproven.",
                 fontsize=7.4, color=RED, va="top")
    fig.suptitle("7 — Source consistency of class-local activations",
                 fontsize=11, weight="bold", color=INK, x=.005, ha="left")
    fig.tight_layout(rect=(0, 0, 1, .92))
    save(fig, "fig07_source_consistency")


def f08_interpretability(ctx):
    i5 = pd.read_csv(T / "inv5_spectroscopic_interpretation_v1.csv")
    picks = ["acylglycerol.m00", "fatty_acid.m00", "sterol_steroid.m00",
             "peptide_protein.m00", "purine.m00", "chromophore_pigment.m00"]
    picks = [p for p in picks if p in set(i5.motif_id)]
    fig, axes = plt.subplots(len(picks), 1, figsize=(10.2, 1.75 * len(picks)), sharex=True)
    for ax, mid in zip(axes, picks):
        h = ctx.H[ctx.ids.index(mid)]
        r = i5[i5.motif_id == mid].iloc[0]
        ax.fill_between(ctx.grid, h / (h.max() + 1e-12), color=BLUE, alpha=.35, lw=0)
        ax.plot(ctx.grid, h / (h.max() + 1e-12), color=BLUE, lw=1.0)
        for part in str(r.assignments).split(" | "):
            cm = float(part.split(":")[0])
            txt = part.split(": ", 1)[1]
            generic = "(generic)" in txt
            ax.axvline(cm, color=(LINE if generic else RED), lw=.8, ls=":" if generic else "-")
            if not generic:
                ax.text(cm, 1.04, txt.split(";")[0][:34], fontsize=5.2, rotation=90,
                        ha="center", va="bottom", color=RED)
        ax.set_ylim(0, 1.9)
        ax.set_ylabel("norm.")
        ax.text(.005, .84, f"{mid}  [{r.lsm_type}, n={r.n_analytes}]",
                transform=ax.transAxes, fontsize=7.6, weight="bold", color=INK)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("Raman shift (cm$^{-1}$)")
    fig.suptitle("8 — Spectroscopic interpretation: class-conditioned band assignments\n"
                 "red = class-specific diagnostic assignment · dotted = generic mode",
                 fontsize=11, weight="bold", color=INK, x=.005, ha="left")
    fig.tight_layout(rect=(0, 0, 1, .955))
    save(fig, "fig08_band_assignments")


def f09_cross_class(ctx):
    pairs = pd.read_csv(T / "inv7_class_pair_hypotheses_v1.csv")
    kept = ctx.kept
    mids = kept.motif_id.tolist()
    Hm = np.vstack([ctx.H[ctx.ids.index(m)] for m in mids])
    N = Hm / (np.linalg.norm(Hm, axis=1, keepdims=True) + 1e-12)
    C = N @ N.T
    order = np.argsort(kept.chemical_class.values, kind="stable")
    C = C[np.ix_(order, order)]
    cls = kept.chemical_class.values[order]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 6.0),
                             gridspec_kw={"width_ratios": [1.25, 1]})
    im = axes[0].imshow(C, cmap="RdYlBu_r", vmin=0, vmax=1)
    b, prev = [], None
    for i, c in enumerate(cls):
        if c != prev:
            b.append(i - .5)
            prev = c
    for x in b[1:]:
        axes[0].axhline(x, color="k", lw=.6)
        axes[0].axvline(x, color="k", lw=.6)
    ticks = [(b[i] + (b[i + 1] if i + 1 < len(b) else len(cls))) / 2 for i in range(len(b))]
    axes[0].set_xticks(ticks)
    axes[0].set_yticks(ticks)
    lab = [c.replace("_", " ")[:16] for c in dict.fromkeys(cls)]
    axes[0].set_xticklabels(lab, rotation=90, fontsize=5.6)
    axes[0].set_yticklabels(lab, fontsize=5.6)
    fig.colorbar(im, ax=axes[0], fraction=.046, label="cosine")
    axes[0].set_title("all 50 LSMs, blocked by class", fontsize=9, loc="left")

    d = pairs.head(14)
    axes[1].barh(np.arange(len(d)), d.max_cosine, .66,
                 color=[RED if v >= .95 else (AMBER if v >= .9 else BLUE)
                        for v in d.max_cosine])
    axes[1].set_yticks(np.arange(len(d)))
    axes[1].set_yticklabels([f"{a.replace('_', ' ')[:17]} ~ {b.replace('_', ' ')[:17]}"
                             for a, b in zip(d.class_a, d.class_b)], fontsize=6.4)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0.6, 1.0)
    axes[1].set_xlabel("max cross-class motif cosine")
    axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].set_title("Phase 02 consensus hypotheses (NOT merged here)", fontsize=9, loc="left")
    fig.suptitle("9 — Hidden cross-class redundancy\n"
                 "the lipid superfamily co-clusters as expected; "
                 "peptide_protein ~ polysaccharide at 0.970 needs scrutiny",
                 fontsize=11, weight="bold", color=INK, x=.005, ha="left")
    fig.tight_layout(rect=(0, 0, 1, .89))
    save(fig, "fig09_cross_class_redundancy")


def f10_sensitivity():
    s = pd.read_csv(T / "inv8_sensitivity_v1.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.0))
    order = ["seed", "noise_1pct", "noise_5pct", "molecule_bootstrap"]
    data = [s[s.perturbation == p].mean_basis_similarity.values for p in order]
    bp = axes[0].boxplot(data, patch_artist=True, widths=.55)
    for p, c in zip(bp["boxes"], [GREEN, GREEN, AMBER, BLUE]):
        p.set_facecolor(c)
        p.set_alpha(.75)
    axes[0].set_xticklabels(["seed", "noise 1%", "noise 5%", "molecule\nbootstrap"],
                            fontsize=7.5)
    axes[0].set_ylabel("basis similarity to the reference fit")
    axes[0].axhline(0.9, color=RED, ls="--", lw=1)
    axes[0].set_ylim(0.6, 1.02)
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[0].set_title("across all classes", fontsize=9, loc="left")

    mb = s[s.perturbation == "molecule_bootstrap"].sort_values("mean_basis_similarity")
    axes[1].barh(np.arange(len(mb)), mb.mean_basis_similarity, .66,
                 xerr=[mb.mean_basis_similarity - mb.ci95_low,
                       mb.ci95_high - mb.mean_basis_similarity],
                 color=[RED if v < .85 else BLUE for v in mb.mean_basis_similarity],
                 error_kw={"lw": .7, "ecolor": INK})
    axes[1].set_yticks(np.arange(len(mb)))
    axes[1].set_yticklabels([f"{c.replace('_', ' ')[:22]} (k={int(k)})"
                             for c, k in zip(mb.chemical_class, mb.k_c)], fontsize=6.2)
    axes[1].set_xlim(0.6, 1.02)
    axes[1].axvline(0.9, color=RED, ls="--", lw=1)
    axes[1].set_xlabel("basis similarity under molecule bootstrap (95% CI)")
    axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].set_title("the hardest perturbation, per class", fontsize=9, loc="left")
    fig.suptitle("10 — Sensitivity analysis: the dictionary is stable under seed, noise and "
                 "molecule resampling",
                 fontsize=11, weight="bold", color=INK, x=.005, ha="left")
    fig.tight_layout(rect=(0, 0, 1, .90))
    save(fig, "fig10_sensitivity")


def f11_limitations():
    lim = pd.read_csv(T / "inv9_limitations_v1.csv")
    fig, ax = plt.subplots(figsize=(10.6, 4.6))
    d = lim.sort_values("worst_molecule_ev")
    x = np.arange(len(d))
    cols = [RED if l.startswith("CORPUS") else (AMBER if l.startswith("ALGORITHM") else GREEN)
            for l in d.limitation]
    ax.bar(x, d.worst_molecule_ev, .68, color=cols)
    ax.axhline(0.5, color=RED, ls="--", lw=1)
    ax.axhline(0.7, color=AMBER, ls="--", lw=1)
    for i, (_, r) in enumerate(d.iterrows()):
        ax.text(i, r.worst_molecule_ev + .02, f"n={int(r.n_molecules)}\nk={int(r.k_c)}",
                ha="center", fontsize=5.4, color=MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", " ") for c in d.chemical_class], rotation=90,
                       fontsize=6.4)
    ax.set_ylabel("worst-molecule explained variance")
    ax.set_ylim(0, 1.08)
    ax.spines[["top", "right"]].set_visible(False)
    for col, lab in ((RED, "corpus-limited"), (AMBER, "algorithm-limited"),
                     (GREEN, "adequate")):
        ax.bar([0], [0], color=col, label=lab)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax.set_title("11 — Corpus versus algorithm: every remaining weakness is corpus-driven\n"
                 "all weak classes hold ≤8 molecules; none is limited by the selection "
                 "criterion",
                 fontsize=11, weight="bold", color=INK, loc="left", pad=12)
    save(fig, "fig11_corpus_vs_algorithm")


if __name__ == "__main__":
    from investigate import Ctx
    print(f"writing investigation figures to {F}")
    ctx = Ctx()
    f01_before_after()
    f02_uniqueness(ctx)
    f03_dendrograms(ctx)
    f04_reconstruction()
    f05_residuals(ctx)
    f06_kc_robustness()
    f07_source(ctx)
    f08_interpretability(ctx)
    f09_cross_class(ctx)
    f10_sensitivity()
    f11_limitations()
    print("done — 11 figures (SVG vector + PNG preview)")
