#!/usr/bin/env python3
"""GAIRA V7 — Phase 04.5 figures (PNG, 200 dpi). Deterministic."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
from gaira.v7.io import PhaseOutputs, frozen_root      # noqa: E402

OUT = PhaseOutputs("04.5")
T, A, V, F = OUT.tables, OUT.artifacts, OUT.validation, OUT.figures
FROZEN = frozen_root()
INK, MUTED, LINE = "#1a1a1a", "#6b7280", "#9ca3af"
BLUE, GREEN, AMBER, RED, GREY = "#2563eb", "#15803d", "#b45309", "#b91c1c", "#4b5563"
REPCOL = {"RAW": GREY, "LSM": AMBER, "CSM": GREEN, "META": RED}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
                     "figure.facecolor": "white", "savefig.facecolor": "white",
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.18})


def save(fig, name):
    F.mkdir(parents=True, exist_ok=True)
    fig.savefig(F / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"  {name}")


class Ctx:
    def __init__(self):
        z = np.load(A / "meta_components_v1.npz", allow_pickle=True)
        self.H, self.W, self.A_meta = z["H"], z["W"], z["A_meta"]
        self.A_csm = z["A_csm"]
        self.csm_ids = [str(s) for s in z["csm_ids"]]
        self.y = np.array([str(s) for s in z["y"]])
        self.cls = np.array([str(s) for s in z["cls"]])
        self.coords = z["coords_csm"]
        self.sweep = pd.read_csv(T / "model_selection_sweep_v1.csv")
        self.evid = pd.read_csv(T / "meta_component_evidence_v1.csv")
        self.valid = pd.read_csv(T / "representation_comparison_v1.csv")
        self.rob = pd.read_csv(T / "robustness_curves_v1.csv")
        self.auc = pd.read_csv(T / "robustness_auc_v1.csv")
        self.kdiag = pd.read_csv(V / "k_downstream_diagnostic_v1.csv")
        self.state = json.loads((OUT.root / "PHASE_STATE.json").read_text())
        self.verdict = json.loads((A / "verdict_v1.json").read_text())
        treg = json.loads((FROZEN / "phase03/artifacts/theme_registry_v1.json").read_text())
        self.bridges = set(treg["bridge_csms"])
        self.K = self.H.shape[0]


def f01_architecture(c):
    fig, ax = plt.subplots(figsize=(10.4, 3.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3.4); ax.set_axis_off()
    steps = [("154 canonical\nmolecules", GREY), ("preprocessing", GREY),
             ("50 LSMs", GREEN), ("49 CSMs", GREEN), ("projection\nengine", GREEN),
             ("49-dim CSM\nactivations", GREEN), (f"{c.K} Meta\nComponents", RED)]
    for k, (lab, col) in enumerate(steps):
        x = 0.2 + k * 1.38
        ax.add_patch(FancyBboxPatch((x, 1.5), 1.15, 1.05, boxstyle="round,pad=0.05",
                                    fc="#fef2f2" if col == RED else
                                    ("#f0fdf4" if col == GREEN else "white"),
                                    ec=col, lw=1.6 if col == RED else 1.1))
        ax.text(x + 0.575, 2.02, lab, ha="center", va="center", fontsize=7, color=INK)
        if k:
            ax.add_patch(FancyArrowPatch((x - 0.2, 2.02), (x - 0.03, 2.02),
                                         arrowstyle="-|>", mutation_scale=8, color=MUTED,
                                         lw=0.9))
    ax.text(0.2, 1.05, "Everything left of the red box is frozen and was neither refitted nor "
                       "recomputed.", fontsize=7.8, color=INK)
    ax.text(0.2, 0.55, f"VERDICT: {c.state['recommended_action'].upper()} — "
                       f"{c.state['verdict'][:96]}", fontsize=7.8, color=RED)
    ax.set_title("Where Meta Components sit, and what happened to them", fontsize=10.5,
                 loc="left", color=INK)
    save(fig, "fig01_architecture")


def f02_workflow(c):
    fig, ax = plt.subplots(figsize=(9.6, 3.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3.4); ax.set_axis_off()
    ax.add_patch(FancyBboxPatch((0.3, 1.9), 2.2, 1.0, boxstyle="round,pad=0.06",
                                fc="#f0fdf4", ec=GREEN, lw=1.2))
    ax.text(1.4, 2.4, "A = 375 x 49\nspectra x CSM\nactivations", ha="center", va="center",
            fontsize=7.5)
    ax.add_patch(FancyArrowPatch((2.6, 2.4), (3.5, 2.4), arrowstyle="-|>",
                                 mutation_scale=10, color=MUTED))
    ax.add_patch(FancyBboxPatch((3.6, 1.9), 2.4, 1.0, boxstyle="round,pad=0.06",
                                fc="white", ec=RED, lw=1.4))
    ax.text(4.8, 2.4, "NMF on A\nA ≈ W H", ha="center", va="center", fontsize=8)
    ax.add_patch(FancyArrowPatch((6.1, 2.4), (7.0, 2.4), arrowstyle="-|>",
                                 mutation_scale=10, color=MUTED))
    ax.add_patch(FancyBboxPatch((7.1, 1.9), 2.5, 1.0, boxstyle="round,pad=0.06",
                                fc="#fef2f2", ec=RED, lw=1.2))
    ax.text(8.35, 2.4, f"W: 375 x {c.K}\nH: {c.K} x 49", ha="center", va="center", fontsize=7.5)
    ax.text(0.3, 1.35, "H rows = which frozen CSMs a programme uses  ·  "
                       "W rows = which programmes a spectrum runs", fontsize=7.5, color=INK)
    ax.text(0.3, 0.95, "Geometry prior (optional):  + λ · tr(H L Hᵀ)  over the frozen CSM "
                       "k-NN graph — a one-sided smoothness reward", fontsize=7.5, color=MUTED)
    ax.text(0.3, 0.5, "Inference:  spectrum → frozen CSM projection → 49 activations → "
                      "NNLS onto frozen H → Meta vector.  No fitting.", fontsize=7.5,
            color=INK)
    ax.set_title("Hierarchical NMF workflow", fontsize=10.5, loc="left", color=INK)
    save(fig, "fig02_workflow")


def f03_pareto(c):
    s = c.sweep
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.2), gridspec_kw={"wspace": 0.32})
    for v_, mark in (("plain", "o"), ("geometry_regularised", "s")):
        sub = s[s.variant == v_].sort_values("K")
        axes[0].plot(sub.K, sub.explained_variance, mark + "-", ms=5, label=v_)
        axes[1].plot(sub.K, sub.bootstrap_stability, mark + "-", ms=5, color=GREEN)
        axes[1].plot(sub.K, sub.consensus_stability, mark + "--", ms=5, color=AMBER)
        axes[2].plot(sub.K, sub.pareto if "pareto" in sub else np.nan, mark + "-", ms=5)
    axes[0].set_xlabel("K"); axes[0].set_ylabel("explained variance")
    axes[0].set_title("Reconstruction — weighted 0.14 of 1.0", fontsize=9, loc="left",
                      color=INK)
    axes[0].legend(frameon=False, fontsize=6.5)
    axes[1].set_xlabel("K"); axes[1].set_ylabel("stability")
    axes[1].legend(handles=[Line2D([], [], color=GREEN, label="bootstrap"),
                            Line2D([], [], color=AMBER, ls="--", label="consensus")],
                   frameon=False, fontsize=6.5)
    axes[1].set_title("Stability", fontsize=9, loc="left", color=INK)
    axes[2].axvline(c.K, color=RED, ls="--", lw=1.0)
    axes[2].set_xlabel("K"); axes[2].set_ylabel("Pareto composite")
    axes[2].set_title(f"Selected K = {c.K}", fontsize=9, loc="left", color=INK)
    for a in axes:
        for sp in ("top", "right"): a.spines[sp].set_visible(False)
    fig.suptitle("Model selection over eight K and two variants. Stability carries 0.40 of the "
                 "composite and reconstruction 0.14 —\nwhich is what pulls K down to 3.",
                 fontsize=9.5, x=0.005, ha="left", y=1.08, color=INK)
    save(fig, "fig03_pareto_frontier")


def f04_heatmap(c):
    order = np.argsort(-c.H.max(axis=0))
    fig, ax = plt.subplots(figsize=(10.0, 2.4 + 0.28 * c.K))
    im = ax.imshow(c.H[:, order], aspect="auto", cmap="magma")
    ax.set_yticks(range(c.K))
    ax.set_yticklabels([f"MC-{k+1:02d}" for k in range(c.K)], fontsize=8)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([c.csm_ids[i] for i in order], rotation=90, fontsize=4.6)
    for lbl, i in zip(ax.get_xmajorticklabels(), order):
        if c.csm_ids[i] in c.bridges:
            lbl.set_color(AMBER)
    fig.colorbar(im, ax=ax, shrink=0.7, label="loading")
    ax.set_title("Meta Component loadings over the 49 frozen CSMs "
                 "(amber tick = bridge CSM)", fontsize=9.5, loc="left", color=INK)
    save(fig, "fig04_meta_heatmap")


def f05_composition(c):
    fig, axes = plt.subplots(1, c.K, figsize=(3.5 * c.K, 3.2), gridspec_kw={"wspace": 0.42})
    for k, ax in enumerate(np.atleast_1d(axes)):
        h = c.H[k]
        nb = np.argsort(-h)[:6]
        nb = [i for i in nb if h[i] > 0]
        ax.barh(np.arange(len(nb))[::-1], h[nb],
                color=[AMBER if c.csm_ids[i] in c.bridges else BLUE for i in nb], height=0.6)
        ax.set_yticks(np.arange(len(nb))[::-1])
        ax.set_yticklabels([c.csm_ids[i] for i in nb], fontsize=7)
        ax.set_xlabel("loading")
        r = c.evid.iloc[k]
        ax.set_title(f"MC-{k+1:02d} · {r.n_spectra_dominant} spectra\n"
                     f"{str(r.dominant_classes)[:36]}", fontsize=7.8, loc="left", color=INK)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fig.suptitle("Component composition — top contributing CSMs, with bridge CSMs in amber",
                 fontsize=9.5, x=0.005, ha="left", y=1.06, color=INK)
    save(fig, "fig05_composition")


def f06_activation_map(c):
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.6), gridspec_kw={"wspace": 0.24})
    order = np.argsort(c.W.argmax(axis=1))
    im = axes[0].imshow(c.W[order] / (c.W[order].sum(1, keepdims=True) + 1e-12),
                        aspect="auto", cmap="viridis")
    axes[0].set_xticks(range(c.K))
    axes[0].set_xticklabels([f"MC-{k+1:02d}" for k in range(c.K)], fontsize=7)
    axes[0].set_ylabel("spectra (sorted by dominant programme)")
    axes[0].set_yticks([])
    fig.colorbar(im, ax=axes[0], shrink=0.75, label="share of activation")
    axes[0].set_title("Programme usage per spectrum", fontsize=9, loc="left", color=INK)
    ax = axes[1]
    fams = sorted(set(c.cls))
    dom = c.W.argmax(axis=1)
    M = np.zeros((len(fams), c.K))
    for i, f_ in enumerate(fams):
        for k in range(c.K):
            M[i, k] = float(((c.cls == f_) & (dom == k)).sum())
    M = M / (M.sum(axis=1, keepdims=True) + 1e-12)
    im = ax.imshow(M, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(c.K))
    ax.set_xticklabels([f"MC-{k+1:02d}" for k in range(c.K)], fontsize=7)
    ax.set_yticks(range(len(fams))); ax.set_yticklabels(fams, fontsize=6)
    fig.colorbar(im, ax=ax, shrink=0.75, label="share of class")
    ax.set_title("Chemistry class by dominant programme (revealed after fitting)",
                 fontsize=9, loc="left", color=INK)
    save(fig, "fig06_activation_maps")


def f07_comparison(c):
    v = c.valid.set_index("representation").loc[["RAW", "LSM", "CSM", "META"]]
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.2), gridspec_kw={"wspace": 0.34})
    x = np.arange(len(v))
    cols = [REPCOL[i] for i in v.index]
    axes[0].bar(x - 0.2, v.A_top1, width=0.4, color=cols, label="split A")
    axes[0].bar(x + 0.2, v.B_top1, width=0.4, color=cols, alpha=0.5, label="split B")
    axes[0].set_xticks(x); axes[0].set_xticklabels(v.index)
    axes[0].set_ylabel("top-1"); axes[0].set_ylim(0, 1)
    axes[0].legend(frameon=False, fontsize=7)
    axes[0].set_title("Retrieval", fontsize=9, loc="left", color=INK)
    axes[1].bar(x, v.macro_f1 if "macro_f1" in v else v.B_macro_f1, color=cols)
    axes[1].set_xticks(x); axes[1].set_xticklabels(v.index)
    axes[1].set_ylabel("macro F1"); axes[1].set_ylim(0, 1)
    axes[1].set_title("Class macro F1", fontsize=9, loc="left", color=INK)
    axes[2].bar(x, v.information_retained_vs_csm, color=cols)
    axes[2].axhline(0.5, color=RED, ls="--", lw=1.0)
    axes[2].text(0.05, 0.52, "informativeness floor", fontsize=6.5, color=RED)
    axes[2].set_xticks(x); axes[2].set_xticklabels(v.index)
    axes[2].set_ylabel("information retained vs CSM"); axes[2].set_ylim(0, 1.05)
    axes[2].set_title("Information", fontsize=9, loc="left", color=INK)
    for a in axes:
        for sp in ("top", "right"): a.spines[sp].set_visible(False)
    fig.suptitle("Four representations, identical frozen splits. Meta Components retain 0.185 "
                 "of the CSM layer's information\nand 0.458 of its class retrieval.",
                 fontsize=9.5, x=0.005, ha="left", y=1.08, color=INK)
    save(fig, "fig07_representation_comparison")


def f08_robustness_grid(c):
    ps = list(dict.fromkeys(c.rob.perturbation))
    n = len(ps)
    fig, axes = plt.subplots(3, 4, figsize=(12.4, 8.0),
                             gridspec_kw={"hspace": 0.62, "wspace": 0.3})
    for ax, p in zip(axes.ravel(), ps):
        sub = c.rob[c.rob.perturbation == p]
        for rep in ("RAW", "LSM", "CSM", "META"):
            s = sub[sub.representation == rep].sort_values("level")
            base = c.valid.set_index("representation").loc[rep, "A_top1"]
            ax.plot(s.level, s.A_top1 / (base + 1e-12), "o-", ms=3, lw=1.1,
                    color=REPCOL[rep], label=rep)
        ax.set_title(p.replace("_", " "), fontsize=8, loc="left", color=INK)
        ax.set_ylim(0, 1.15); ax.tick_params(labelsize=6.5)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    axes.ravel()[0].legend(frameon=False, fontsize=6)
    for ax in axes[-1]:
        ax.set_xlabel("perturbation level", fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel("fraction of clean top-1", fontsize=7)
    fig.suptitle("Molecule-retrieval degradation under twelve physically-motivated "
                 "perturbations, as a fraction of each representation's own clean "
                 "performance.\nMeta Components (red) collapse fastest on molecule identity.",
                 fontsize=9.5, x=0.005, ha="left", y=0.98, color=INK)
    save(fig, "fig08_robustness_curves")


def f09_robustness_auc(c):
    piv = c.auc.pivot_table(index="representation",
                            values=["aurc_A_top1", "aurc_B_top1",
                                    "aurc_activation_stability"]).loc[
        ["RAW", "LSM", "CSM", "META"]]
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.2), gridspec_kw={"wspace": 0.34})
    for ax, col, title in zip(axes, piv.columns,
                              ["molecule retrieval (split A)", "class retrieval (split B)",
                               "activation stability"]):
        ax.bar(np.arange(len(piv)), piv[col], color=[REPCOL[i] for i in piv.index])
        ax.set_xticks(np.arange(len(piv))); ax.set_xticklabels(piv.index)
        ax.set_ylabel("area under robustness curve"); ax.set_ylim(0, 1.05)
        ax.set_title(title, fontsize=9, loc="left", color=INK)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fig.suptitle("Mean area under the robustness curve. Meta Components win on activation "
                 "stability and class retrieval\nand lose catastrophically on molecule "
                 "identity — the profile of a low-information representation.",
                 fontsize=9.5, x=0.005, ha="left", y=1.08, color=INK)
    save(fig, "fig09_robustness_auc")


def f10_k_diagnostic(c):
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    for v_, mark in (("plain", "o"), ("geometry_regularised", "s")):
        s = c.kdiag[c.kdiag.variant == v_].sort_values("K")
        ax.plot(s.K, s.B_top1, mark + "-", ms=5, label=f"Meta ({v_})")
    csm = c.valid.set_index("representation").loc["CSM", "B_top1"]
    ax.axhline(csm, color=GREEN, ls="--", lw=1.3)
    ax.text(2.2, csm + 0.015, f"CSM layer = {csm:.3f}", fontsize=7.5, color=GREEN)
    ax.axvline(c.K, color=RED, ls=":", lw=1.0)
    ax.text(c.K + 0.15, 0.05, f"selected K = {c.K}", fontsize=7, color=RED)
    ax.set_xlabel("K"); ax.set_ylabel("class retrieval top-1 (split B)")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("Would a different K have saved it? No.\n"
                 "Reported as a diagnostic — selecting K on this metric would be circular.",
                 fontsize=9.5, loc="left", color=INK)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    save(fig, "fig10_k_diagnostic")


def f11_stability(c):
    s = c.sweep
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.2), gridspec_kw={"wspace": 0.3})
    for v_, mark in (("plain", "o"), ("geometry_regularised", "s")):
        sub = s[s.variant == v_].sort_values("K")
        axes[0].plot(sub.K, sub.bootstrap_stability, mark + "-", ms=5, label=v_)
        axes[0].fill_between(sub.K, sub.bootstrap_min, sub.bootstrap_stability, alpha=0.15)
        axes[1].plot(sub.K, sub.redundancy, mark + "-", ms=5, label=v_)
    axes[0].set_xlabel("K"); axes[0].set_ylabel("bootstrap component recovery")
    axes[0].legend(frameon=False, fontsize=7)
    axes[0].set_title("Stability is high everywhere — and says little", fontsize=9,
                      loc="left", color=INK)
    axes[1].set_xlabel("K"); axes[1].set_ylabel("max pairwise component cosine")
    axes[1].axhline(0.9, color=RED, ls="--", lw=1.0)
    axes[1].set_title("Redundancy rises with K", fontsize=9, loc="left", color=INK)
    for a in axes:
        for sp in ("top", "right"): a.spines[sp].set_visible(False)
    save(fig, "fig11_bootstrap_stability")


def f12_geometry_overlay(c):
    fig, axes = plt.subplots(1, c.K, figsize=(3.3 * c.K, 3.2), gridspec_kw={"wspace": 0.2})
    for k, ax in enumerate(np.atleast_1d(axes)):
        w = c.H[k] / (c.H[k].max() + 1e-12)
        ax.scatter(c.coords[:, 0], c.coords[:, 1], s=18 + 200 * w, c=w, cmap="magma",
                   vmin=0, vmax=1, edgecolor="white", linewidth=0.3)
        for i, cid in enumerate(c.csm_ids):
            if cid in c.bridges and w[i] > 0.3:
                ax.scatter(c.coords[i, 0], c.coords[i, 1], s=90, facecolors="none",
                           edgecolors=AMBER, linewidth=1.2)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"MC-{k+1:02d}", fontsize=9, loc="left", color=INK)
    fig.suptitle("Meta Component occupancy over the frozen Phase 02.5 CSM geometry "
                 "(amber ring = bridge CSM with loading > 0.3)",
                 fontsize=9.5, x=0.005, ha="left", y=1.06, color=INK)
    save(fig, "fig13_geometry_overlay")


def f13_bridges(c):
    fig, ax = plt.subplots(figsize=(8.0, 3.2))
    x = np.arange(c.K)
    tot = c.H.sum(axis=1) + 1e-12
    bidx = [i for i, cid in enumerate(c.csm_ids) if cid in c.bridges]
    share = c.H[:, bidx].sum(axis=1) / tot
    base = len(bidx) / len(c.csm_ids)
    ax.bar(x, share, color=[RED if s > base else BLUE for s in share], width=0.55)
    ax.axhline(base, color=INK, ls="--", lw=1.0)
    ax.text(-0.4, base + 0.01, f"base rate ({len(bidx)}/{len(c.csm_ids)} CSMs are bridges)",
            fontsize=7, color=INK)
    ax.set_xticks(x); ax.set_xticklabels([f"MC-{k+1:02d}" for k in range(c.K)])
    ax.set_ylabel("share of loading on bridge CSMs")
    ax.set_title("Bridge CSM occupancy per programme", fontsize=9.5, loc="left", color=INK)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    save(fig, "fig12_bridge_occupancy")


def f14_summary(c):
    v = c.valid.set_index("representation")
    piv = c.auc.pivot_table(index="representation", values=["aurc_A_top1", "aurc_B_top1"])
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    for rep in ("RAW", "LSM", "CSM", "META"):
        ax.scatter(v.loc[rep, "B_top1"], piv.loc[rep, "aurc_B_top1"], s=190,
                   color=REPCOL[rep], edgecolor=INK, linewidth=0.8, zorder=3)
        ax.annotate(rep, (v.loc[rep, "B_top1"], piv.loc[rep, "aurc_B_top1"]),
                    fontsize=9, xytext=(9, 5), textcoords="offset points", color=INK)
    ax.axvline(v.loc["CSM", "B_top1"] * 0.5, color=RED, ls="--", lw=1.0)
    ax.text(v.loc["CSM", "B_top1"] * 0.5 + 0.01, 0.86,
            "informativeness floor", fontsize=7.5, color=RED, rotation=90, va="bottom")
    ax.set_xlabel("clean class retrieval (top-1)")
    ax.set_ylabel("robustness — area under the degradation curve")
    ax.set_xlim(0, 1)
    ax.set_title("The whole result in one plot.\nMeta Components buy robustness that a "
                 "low-information representation gets for free,\nand pay for it with more than "
                 "half the clean accuracy.", fontsize=9.5, loc="left", color=INK)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    save(fig, "fig14_summary")


def main():
    c = Ctx()
    print("[phase04.5] figures")
    for fn in (f01_architecture, f02_workflow, f03_pareto, f04_heatmap, f05_composition,
               f06_activation_map, f07_comparison, f08_robustness_grid, f09_robustness_auc,
               f10_k_diagnostic, f11_stability, f13_bridges, f12_geometry_overlay, f14_summary):
        fn(c)


if __name__ == "__main__":
    main()
