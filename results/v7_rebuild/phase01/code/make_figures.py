#!/usr/bin/env python3
"""GAIRA V7 — Phase 01 figures (SVG vector + PNG preview). Deterministic; no RNG."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

HERE = Path(__file__).resolve().parent
P01 = HERE.parent
REPO = P01.parents[2]
sys.path.insert(0, str(REPO / "src"))
from gaira.v7.lsm import serialization as SER          # noqa: E402

T, F, A = P01 / "tables", P01 / "figures", P01 / "artifacts"
INK, MUTED, LINE = "#1a1a1a", "#6b7280", "#9ca3af"
BLUE, GREEN, AMBER, RED, GREY = "#2563eb", "#15803d", "#b45309", "#b91c1c", "#4b5563"
PAL = ["#2563eb", "#15803d", "#b45309", "#7c3aed", "#0891b2", "#be123c"]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
                     "figure.facecolor": "white", "savefig.facecolor": "white",
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.18,
                     "svg.fonttype": "none"})


def save(fig, name):
    F.mkdir(parents=True, exist_ok=True)
    fig.savefig(F / f"{name}.svg", format="svg")
    fig.savefig(F / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"  {name}.svg + {name}.png")


def box(ax, x, y, w, h, t, fc="white", ec=GREY, fs=7.8, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.02",
                                facecolor=fc, edgecolor=ec, linewidth=1.1, zorder=2))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, color=INK,
            zorder=3, linespacing=1.45, weight=weight)


def arrow(ax, p0, p1, color=BLUE):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=8, color=color,
                                 linewidth=1.3, shrinkA=2, shrinkB=2, zorder=1))


# ── 1 the complete pipeline ───────────────────────────────────────────────────
def f01(cls_tab, summ):
    fig, ax = plt.subplots(figsize=(10.4, 8.2))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    ax.text(0, 99, "1 — Canonical V7 Phase 01 pipeline", fontsize=12, weight="bold",
            color=INK, va="top")
    ax.text(0, 94.4, "Balanced references → split by chemistry class → independent "
                     "class-local NMF → Local Spectral Motifs", fontsize=8, color=MUTED,
            va="top")
    steps = [
        ("Raman grounding corpus\n375 spectra · 154 canonical molecules", "white", GREY, "Phase 00"),
        ("BALANCED REFERENCE CONSTRUCTION\n8 arms compared · one molecule = one unit",
         "#dbeafe", BLUE, "Stage 1"),
        (f"split by chemistry class\n{len(cls_tab)} independent per-class datasets",
         "#dbeafe", BLUE, ""),
        ("INDEPENDENT CLASS-LOCAL NMF\nX_c ≈ W_c H_c · adaptive k_c · no global competition",
         "#dbeafe", BLUE, "Stage 2"),
        (f"LOCAL SPECTRAL MOTIFS\n{summ['n_lsms_retained']} retained · rows of H_c",
         "#dcfce7", GREEN, ""),
    ]
    y = 84
    for i, (t, fc, ec, note) in enumerate(steps):
        box(ax, 8, y, 66, 8.6, t, fc=fc, ec=ec, weight="bold" if i in (1, 3, 4) else "normal")
        if note:
            ax.text(76, y + 4.3, note, fontsize=7, color=MUTED, va="center")
        if i < len(steps) - 1:
            arrow(ax, (41, y), (41, y - 4.2))
        y -= 12.8
    box(ax, 8, 16, 66, 8.6, "Phase 02 — Consensus Spectral Motifs\nNOT STARTED",
        fc="#f9fafb", ec=MUTED, fs=7.6)
    ax.add_patch(FancyArrowPatch((41, 32.2), (41, 25), arrowstyle="-|>", mutation_scale=8,
                                 color=MUTED, linewidth=1.1, linestyle=(0, (3, 2))))
    box(ax, 8, 3, 90, 10,
        "THE FROZEN V5 ATLAS IS NOT AN INPUT (principle P-15)\n"
        "It is loaded only to verify its fingerprint and to serve as a baseline comparator.\n"
        "Its 24 components appear nowhere in the construction of any LSM.",
        fc="#fef3c7", ec=AMBER, fs=7.4)
    save(fig, "fig01_pipeline")


# ── 2 balanced reference arms ─────────────────────────────────────────────────
def f02(arms, sel):
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.6))
    d = arms.sort_values("effective_class_gini")
    cols = [GREEN if a == sel["selected_arm"] else (AMBER if a == sel["control_arm"] else GREY)
            for a in d.arm]
    axes[0].barh(np.arange(len(d)), d.effective_class_gini, .7, color=cols)
    axes[0].set_yticks(np.arange(len(d)))
    axes[0].set_yticklabels(d.arm, fontsize=6.4)
    axes[0].set_xlabel("effective class Gini  (lower = better balance)")
    axes[0].invert_yaxis()

    axes[1].barh(np.arange(len(d)), d.molecule_weight_ratio, .7, color=cols)
    axes[1].axvline(1.0, color=GREEN, lw=1.0, ls="--")
    axes[1].set_yticks([]); axes[1].set_xlabel("molecule weight ratio  (1.0 = perfectly balanced)")
    axes[1].invert_yaxis()

    axes[2].barh(np.arange(len(d)), d.band_fidelity, .7, color=cols)
    ctrl = float(arms[arms.arm == sel["control_arm"]].band_fidelity.iloc[0])
    axes[2].axvline(ctrl - 0.02, color=RED, lw=1.0, ls="--")
    axes[2].set_yticks([]); axes[2].set_xlim(0.94, 1.0)
    axes[2].set_xlabel("band fidelity  (red = tolerance floor)")
    axes[2].invert_yaxis()
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"2 — Balanced reference construction: 8 arms\n"
                 f"selected: {sel['selected_arm']}  ·  control (V5 behaviour) in amber",
                 fontsize=11, weight="bold", color=INK, x=0.005, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    save(fig, "fig02_reference_arms")


# ── 3 class-wise NMF and capacity ─────────────────────────────────────────────
def f03(cap):
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.2))
    d = cap.sort_values("n_analytes", ascending=False)
    x = np.arange(len(d))
    axes[0].bar(x - .2, d.v5_capacity_per_molecule, .38, color=GREY,
                label="V5 global fit (expected)")
    axes[0].bar(x + .2, d.capacity_per_molecule, .38, color=BLUE, label="V7 class-local")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([c.replace("_", " ")[:18] for c in d.chemical_class],
                            rotation=90, fontsize=6)
    axes[0].set_ylabel("decomposition capacity per molecule")
    axes[0].legend(fontsize=7, frameon=False)
    axes[0].spines[["top", "right"]].set_visible(False)

    axes[1].scatter(d.n_analytes, d.capacity_per_molecule, s=42, color=BLUE, zorder=3,
                    label="V7 class-local")
    axes[1].scatter(d.n_analytes, d.v5_capacity_per_molecule, s=28, color=GREY, zorder=2,
                    label="V5 global (flat by construction)")
    for _, r in d.iterrows():
        if r.capacity_per_molecule > 0.3 or r.n_analytes >= 17:
            axes[1].annotate(r.chemical_class.replace("_", " ")[:16],
                             (r.n_analytes, r.capacity_per_molecule), fontsize=5.6,
                             color=MUTED, xytext=(3, 3), textcoords="offset points")
    axes[1].set_xlabel("molecules in the class")
    axes[1].set_ylabel("capacity per molecule")
    axes[1].legend(fontsize=7, frameon=False)
    axes[1].spines[["top", "right"]].set_visible(False)
    fig.suptitle("3 — Capacity reallocation: rare chemistry gets its own decomposition\n"
                 "V5 allocated 24 components globally; V7 fits every class alone",
                 fontsize=11, weight="bold", color=INK, x=0.005, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    save(fig, "fig03_capacity_allocation")


# ── 4 k_c optimisation curves ─────────────────────────────────────────────────
def f04(sweep, ksel):
    classes = ksel.sort_values("k", ascending=False).chemical_class.tolist()[:9]
    fig, axes = plt.subplots(3, 3, figsize=(11.2, 8.2), sharex=False)
    for ax, cls in zip(axes.ravel(), classes):
        d = sweep[sweep.chemical_class == cls].sort_values("k")
        s = ksel[ksel.chemical_class == cls].iloc[0]
        ax.plot(d.k, d.composite, "-o", ms=3, color=BLUE, lw=1.2)
        lo, hi = float(s.plateau_start), float(s.plateau_end)
        ax.axvspan(lo - .3, hi + .3, color="#dbeafe", zorder=0, label="plateau")
        ax.axvline(s.k, color=GREEN, lw=1.4)
        ax.axvline(s.best_k, color=AMBER, lw=1.0, ls="--")
        ax.set_title(f"{cls.replace('_', ' ')}  (n={int(s.get('n', 0)) or ''})\n"
                     f"k_c = {int(s.k)}   argmax {int(s.best_k)}",
                     fontsize=7.4, color=INK, loc="left")
        ax.set_xlabel("k"); ax.tick_params(labelsize=6.5)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes.ravel()[len(classes):]:
        ax.axis("off")
    axes[0, 0].set_ylabel("composite")
    fig.suptitle("4 — Adaptive k_c: the composite sweep per class\n"
                 "green = selected (smallest on the contiguous plateau) · amber dashed = argmax",
                 fontsize=11, weight="bold", color=INK, x=0.005, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "fig04_kc_optimisation")


# ── 5 basis spectra ───────────────────────────────────────────────────────────
def f05(lsms, grid):
    by = {}
    for m in lsms:
        by.setdefault(m.chemical_class, []).append(m)
    classes = sorted(by, key=lambda c: -len(by[c]))[:6]
    fig, axes = plt.subplots(len(classes), 1, figsize=(9.6, 2.1 * len(classes)), sharex=True)
    for ax, cls in zip(np.atleast_1d(axes), classes):
        for j, m in enumerate(sorted(by[cls], key=lambda m: m.index_in_class)):
            v = m.spectrum / (m.spectrum.max() + 1e-12)
            ax.plot(grid, v, lw=1.1, color=PAL[j % len(PAL)],
                    label=f"{m.motif_id} · {m.lsm_type} (n={m.n_analytes})")
        ax.set_title(f"{cls.replace('_', ' ')}  —  k_c = {by[cls][0].k_c}",
                     fontsize=8.4, loc="left", color=INK)
        ax.set_ylabel("norm.")
        ax.legend(fontsize=5.8, frameon=False, loc="upper right", ncol=2)
        ax.spines[["top", "right"]].set_visible(False)
    np.atleast_1d(axes)[-1].set_xlabel("Raman shift (cm$^{-1}$)")
    fig.suptitle("5 — Local Spectral Motif basis spectra (rows of the class-local H_c)\n"
                 "newly fitted basis vectors — not restrictions of any existing component",
                 fontsize=11, weight="bold", color=INK, x=0.005, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    save(fig, "fig05_basis_spectra")


# ── 6 activation heatmap ──────────────────────────────────────────────────────
def f06(part_tab, reg_tab):
    kept = reg_tab[reg_tab.retained]
    mol = sorted(part_tab.canonical_id.unique())
    M = pd.DataFrame(0, index=mol, columns=kept.motif_id.tolist())
    for _, r in part_tab.iterrows():
        if r.motif_id in M.columns:
            M.loc[r.canonical_id, r.motif_id] = 1
    cls_of = dict(zip(reg_tab.motif_id, reg_tab.chemical_class))
    order = sorted(M.index, key=lambda a: (part_tab[part_tab.canonical_id == a]
                                           .chemical_class.iloc[0], a))
    D = M.loc[order]
    fig, ax = plt.subplots(figsize=(11.2, 8.4))
    ax.imshow(D.values, aspect="auto", cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=4.2)
    ax.set_xticks(range(D.shape[1]))
    ax.set_xticklabels(D.columns, rotation=90, fontsize=5)
    prev, bounds = None, []
    for i, c in enumerate(D.columns):
        if cls_of.get(c) != prev:
            bounds.append(i - .5); prev = cls_of.get(c)
    for b in bounds[1:]:
        ax.axvline(b, color=RED, lw=.8)
    ax.set_title("6 — Molecule × LSM participation, blocked by chemistry class\n"
                 "red lines separate independent class-local fits — no motif spans a boundary",
                 fontsize=11, weight="bold", color=INK, loc="left", pad=12)
    save(fig, "fig06_activation_heatmap")


# ── 7 stability and quality ───────────────────────────────────────────────────
def f07(reg_tab, cls_tab):
    kept = reg_tab[reg_tab.retained]
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.3))
    axes[0].hist(kept.stability, bins=12, color=GREEN, edgecolor="white", lw=.5)
    axes[0].axvline(0.60, color=RED, lw=1.1, ls="--")
    axes[0].text(0.60, axes[0].get_ylim()[1] * .95, " reject < 0.60", fontsize=6.4,
                 color=RED, va="top")
    axes[0].set_xlabel("recurrence stability")
    axes[0].set_ylabel("LSMs")

    axes[1].hist(kept.activation_sparsity, bins=12, color=BLUE, edgecolor="white", lw=.5)
    axes[1].set_xlabel("activation sparsity  (selectivity)")

    d = cls_tab.sort_values("explained_variance")
    axes[2].barh(np.arange(len(d)), d.explained_variance, .7, color=AMBER)
    axes[2].set_yticks(np.arange(len(d)))
    axes[2].set_yticklabels([c.replace("_", " ")[:18] for c in d.chemical_class], fontsize=5.6)
    axes[2].set_xlabel("class-local explained variance")
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle("7 — LSM stability, selectivity and per-class reconstruction",
                 fontsize=11, weight="bold", color=INK, x=0.005, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    save(fig, "fig07_stability_quality")


# ── 8 LSM typing ──────────────────────────────────────────────────────────────
def f08(cls_tab):
    d = cls_tab.sort_values("n_analytes", ascending=False)
    fig, ax = plt.subplots(figsize=(10.4, 4.4))
    x = np.arange(len(d))
    b = np.zeros(len(d))
    for col, colr, lab in (("n_class_shared", BLUE, "class-shared"),
                           ("n_subfamily", GREEN, "subfamily"),
                           ("n_discriminating", AMBER, "molecule-discriminating")):
        ax.bar(x, d[col], .72, bottom=b, color=colr, label=lab)
        b = b + d[col].values
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c.replace('_', ' ')[:20]}  (n={n}, k={k})"
                        for c, n, k in zip(d.chemical_class, d.n_analytes, d.k_c)],
                       rotation=90, fontsize=6)
    ax.set_ylabel("retained LSMs")
    ax.legend(fontsize=7, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("8 — LSM typing per class\n"
                 "typing is required by Phase 02: class-shared motifs may be merged across "
                 "classes, molecule-discriminating ones must not",
                 fontsize=11, weight="bold", color=INK, loc="left", pad=12)
    save(fig, "fig08_lsm_typing")


# ── 9 reconstruction examples ─────────────────────────────────────────────────
def f09(lsms, grid):
    Z = np.load(A / "balanced_references_v1.npz", allow_pickle=True)
    X = np.asarray(Z["X"], float)
    cid = np.array([str(c) for c in Z["canonical_id"]])
    from scipy.optimize import nnls
    by = {}
    for m in lsms:
        by.setdefault(m.chemical_class, []).append(m)
    picks = sorted(by, key=lambda c: -len(by[c]))[:4]
    fig, axes = plt.subplots(len(picks), 1, figsize=(9.6, 2.3 * len(picks)), sharex=True)
    for ax, cls in zip(np.atleast_1d(axes), picks):
        H = np.vstack([m.spectrum for m in by[cls]])
        mols = sorted({a for m in by[cls] for a in m.analytes})
        if not mols:
            continue
        mol = mols[0]
        rowsel = np.where(cid == mol)[0]
        x = X[rowsel].mean(axis=0)
        w, _ = nnls(H.T, np.maximum(x, 0))
        r = w @ H
        ax.plot(grid, x, lw=1.2, color=INK, label=f"{mol}  (balanced reference)")
        ax.plot(grid, r, lw=1.1, color=BLUE, ls="--",
                label=f"class-local reconstruction (k_c={len(by[cls])})")
        ax.fill_between(grid, 0, np.abs(x - r), color=RED, alpha=.25, label="residual")
        ax.set_title(cls.replace("_", " "), fontsize=8.4, loc="left", color=INK)
        ax.legend(fontsize=6, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
    np.atleast_1d(axes)[-1].set_xlabel("Raman shift (cm$^{-1}$)")
    fig.suptitle("9 — Reconstruction from the class-local basis alone",
                 fontsize=11, weight="bold", color=INK, x=0.005, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "fig09_reconstruction")


# ── 10 architecture compliance ────────────────────────────────────────────────
def f10(comp):
    fig, ax = plt.subplots(figsize=(11.2, 6.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    ax.text(0, 99, "10 — Architecture compliance", fontsize=12, weight="bold", color=INK,
            va="top")
    n_pass = int((comp.status == "PASS").sum())
    ax.text(0, 94.4, f"{n_pass} of {len(comp)} specification items PASS — the gate opens only "
                     f"if every row passes", fontsize=8, color=MUTED, va="top")
    y = 88
    for _, r in comp.iterrows():
        col = GREEN if r.status == "PASS" else RED
        ax.text(1, y, "✓" if r.status == "PASS" else "✗", fontsize=9, color=col,
                weight="bold", va="center")
        ax.text(4, y, r.specification_item[:74], fontsize=7.0, color=INK, va="center")
        ax.text(62, y, r.evidence[:60], fontsize=6.2, color=MUTED, va="center")
        y -= 4.6
    save(fig, "fig10_architecture_compliance")


if __name__ == "__main__":
    print(f"writing Phase 01 figures to {F}")
    reg_tab, H, ids, man = SER.load_registry(A)
    lsms = SER.lsms_from_table(reg_tab, H, ids)
    grid = np.asarray(np.load(A / "balanced_references_v1.npz", allow_pickle=True)["grid"], float)
    cls_tab = pd.read_csv(T / "lsm_classes_v1.csv")
    arms = pd.read_csv(T / "reference_arm_comparison_v1.csv")
    sel = json.loads((A / "reference_arm_selection_v1.json").read_text())
    cap = pd.read_csv(T / "capacity_allocation_v1.csv")
    sweep = pd.read_csv(T / "kc_sweep_v1.csv")
    ksel = pd.read_csv(T / "kc_selection_v1.csv")
    part_tab = pd.read_csv(T / "lsm_participation_v1.csv")
    comp = pd.read_csv(T / "architecture_compliance_v1.csv")
    f01(cls_tab, man["summary"])
    f02(arms, sel)
    f03(cap)
    f04(sweep, ksel)
    f05(lsms, grid)
    f06(part_tab, reg_tab)
    f07(reg_tab, cls_tab)
    f08(cls_tab)
    f09(lsms, grid)
    f10(comp)
    print("done — 10 figures (SVG vector + PNG preview)")
