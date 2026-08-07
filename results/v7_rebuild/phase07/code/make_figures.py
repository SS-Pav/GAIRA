#!/usr/bin/env python3
"""GAIRA V7 — Phase 07 figures. PNG only, 200 dpi, deterministic."""
from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
from gaira.v7.io import PhaseOutputs, frozen_root         # noqa: E402

OUT = PhaseOutputs("07", extra=("interactive", "manifests"))
T, A_, F = OUT.tables, OUT.artifacts, OUT.figures
FROZEN = frozen_root()
INK, MUTED, LINE = "#1a1a1a", "#6b7280", "#9ca3af"
BLUE, GREEN, AMBER, RED, GREY = "#2563eb", "#15803d", "#b45309", "#b91c1c", "#4b5563"
PURPLE, TEAL = "#7c3aed", "#0f766e"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
                     "figure.facecolor": "white", "savefig.facecolor": "white",
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.18})


def SH(c):
    return (c.replace("_", " ").replace("carboxylic acid metabolite", "carboxylic acid")
            .replace("phospholipid sphingolipid", "phospholipid")
            .replace("mono oligosaccharide", "mono/oligosacch")
            .replace("sulfur thiol cofactor", "sulfur/thiol")
            .replace("nucleic acid polymer", "nucleic polymer")
            .replace("chromophore pigment", "chromophore").replace("small nitrogenous", "small N"))


def save(fig, name):
    F.mkdir(parents=True, exist_ok=True)
    fig.savefig(F / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"  {name}")


def box(ax, x, y, w, h, t, fc="#eef2ff", ec=BLUE, fs=8.0, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008", fc=fc, ec=ec, lw=1.1))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, color=INK,
            weight=weight, linespacing=1.35)


def arrow(ax, p0, p1, col=LINE, lw=1.2):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=11, color=col, lw=lw,
                                 shrinkA=2, shrinkB=2))


class C:
    def __init__(self):
        self.s = json.loads((A_ / "phase07_summary_v1.json").read_text())
        self.state = json.loads((OUT.root / "PHASE_STATE.json").read_text())
        z = np.load(A_ / "bsv2_programmes_v1.npz", allow_pickle=True)
        self.W, self.P, self.Ev, self.R = z["W"], z["P"], z["Ev"], z["reconstruction"]
        self.y = np.array([str(v) for v in z["y"]])
        self.cls = np.array([str(v) for v in z["cls"]])
        self.axes = [str(v) for v in z["axis_names"]]
        self.folds = z["folds"]
        self.sweep = pd.read_csv(T / "programme_selection_v1.csv")
        self.per_axis = pd.read_csv(T / "reconstruction_per_axis_v1.csv")
        self.stab = pd.read_csv(T / "programme_stability_v1.csv")
        self.cmp = pd.read_csv(T / "layer_comparison_v1.csv")
        self.gen = pd.read_csv(T / "generalisation_v1.csv")
        self.rob = pd.read_csv(T / "noise_robustness_v1.csv")
        self.coh = pd.read_csv(T / "programme_coherence_v1.csv")
        self.gates = pd.read_csv(T / "phase07_gates_v1.csv")
        self.prog = self.s["programmes"]
        self.K = self.s["model"]["K"]
        b = np.load(FROZEN / "phase01/artifacts/balanced_references_v1.npz", allow_pickle=True)
        self.X, self.grid = np.asarray(b["X"], float), np.asarray(b["grid"], float)


def f01(c):
    fig, ax = plt.subplots(figsize=(11.2, 4.4))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    steps = [(0.01, 0.15, "Raman spectrum", "#f8fafc", GREY),
             (0.18, 0.15, "CSM activation\n49-d (frozen)", "#f1f5f9", GREY),
             (0.35, 0.17, "Chemistry Evidence\n16-d (frozen, Phase 06)", "#ecfdf5", GREEN),
             (0.54, 0.17, f"BSV2 programmes\n{c.K}-d (THIS PHASE)", "#f5f3ff", PURPLE),
             (0.73, 0.26, "interpretation\nautomatic descriptions · programme\nprovenance to "
                          "CSMs, LSMs, bands", "#fffbeb", AMBER)]
    for i, (x, w, t, fc, ec) in enumerate(steps):
        box(ax, x, 0.58, w, 0.20, t, fc, ec, 8.2, "bold" if i in (2, 3) else "normal")
        if i:
            arrow(ax, (steps[i - 1][0] + steps[i - 1][1], 0.68), (x, 0.68))
    ax.text(0.54, 0.53, "the ONLY input", ha="left", fontsize=8, color=PURPLE, weight="bold")
    arrow(ax, (0.435, 0.58), (0.60, 0.52), PURPLE, 1.4)
    s = c.s
    box(ax, 0.01, 0.10, 0.46, 0.34,
        "NOT read as model input\n\nraw spectra · CSM activations · geometry · UMAP · PCA\n"
        "cluster ids · continuous coordinates · theme layer · legacy BSV\n\n"
        "Frozen CSM/LSM artefacts are read for EXPLANATION only,\nafter fitting.",
        "#fef2f2", RED, 7.6)
    box(ax, 0.51, 0.10, 0.48, 0.34,
        f"RESULT — {s['model']['family']} at K = {s['model']['K']}\n\n"
        f"reconstruction EV {s['reconstruction']['explained_variance']:.3f} · cosine "
        f"{s['reconstruction']['mean_cosine']:.3f}\n"
        f"bootstrap stability {s['stability']['bootstrap']:.3f} · held-out EV "
        f"{np.mean([g['explained_variance'] for g in s['generalisation']]):.3f}\n"
        f"compression {16 / s['model']['K']:.1f}x · information retained "
        f"{s['information_retained']:.3f}", "#ecfdf5", GREEN, 7.8)
    ax.set_title("Figure 1 · Phase 07 architecture — Chemistry Evidence → BSV2 → interpretation",
                 loc="left", fontsize=11.5, weight="bold", color=INK)
    save(fig, "F01_architecture")


def f02(c):
    d = c.sweep[c.sweep.usable].copy()
    fig, axs = plt.subplots(2, 3, figsize=(12.6, 6.4))
    fams = sorted(set(d.family))
    cmap = {f: plt.get_cmap("tab10")(i % 10) for i, f in enumerate(fams)}
    panels = [("explained_variance", "reconstruction EV", axs[0, 0]),
              ("heldout_chemistry_retention", "held-out chemistry retention", axs[0, 1]),
              ("bootstrap_stability", "bootstrap programme recovery", axs[0, 2]),
              ("sparsity", "programme sparsity (Hoyer)", axs[1, 0]),
              ("max_pairwise_overlap", "max programme overlap", axs[1, 1]),
              ("max_single_axis_share", "max single-axis share", axs[1, 2])]
    for col, lab, ax in panels:
        for f in fams:
            g = d[d.family == f].sort_values("K")
            ax.plot(g.K, g[col], "o-", ms=3.2, lw=1.1, color=cmap[f], alpha=0.85, label=f)
        ax.set_xlabel("K"); ax.set_ylabel(lab, fontsize=7.6); ax.tick_params(labelsize=7)
        ax.axvline(c.K, color=RED, ls=":", lw=1.3)
    for col, lab, ax, floor in (("heldout_chemistry_retention", "", axs[0, 1], 0.50),
                                ("explained_variance", "", axs[0, 0], 0.50),
                                ("max_pairwise_overlap", "", axs[1, 1], 0.90),
                                ("max_single_axis_share", "", axs[1, 2], 0.90)):
        ax.axhline(floor, color=RED, ls="--", lw=1.0)
    axs[1, 2].axvspan(12.5, 16.5, color=RED, alpha=0.10)
    axs[1, 2].text(13.0, 0.05, "K > 12\nrejected", fontsize=6.4, color=RED)
    axs[0, 0].legend(frameon=False, fontsize=5.8, ncol=2)
    dec = c.s["decision"]
    fig.suptitle(f"Figure 2 · Programme-number sweep — 6 families × K = 2…16, with the "
                 f"pre-registered floors\ndecision: {dec['family']} at K = {dec['K']} · "
                 f"{dec['n_eligible']} of {len(d)} candidates eligible · red dotted line = the "
                 f"selected K",
                 x=0.035, ha="left", fontsize=10.5, weight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    save(fig, "F02_k_sweep")


def f03(c):
    P = c.P
    order = np.argsort(-np.abs(P).sum(axis=1))
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    v = np.abs(P).max()
    im = ax.imshow(P[order], aspect="auto", cmap="RdBu_r" if P.min() < 0 else "Purples",
                   vmin=-v if P.min() < 0 else 0, vmax=v)
    ax.set_xticks(range(len(c.axes)))
    ax.set_xticklabels([SH(a) for a in c.axes], rotation=55, ha="right", fontsize=7)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"P{int(i)} — {c.prog[int(i)]['auto_description'][:34]}"
                        for i in order], fontsize=7)
    for r, i in enumerate(order):
        for j in range(len(c.axes)):
            sh = abs(P[i, j]) / (np.abs(P[i]).sum() + 1e-12)
            if sh > 0.14:
                ax.text(j, r, f"{sh:.0%}", ha="center", va="center", fontsize=5.6,
                        color="white" if abs(P[i, j]) > 0.55 * v else INK)
    fig.colorbar(im, ax=ax, shrink=0.8, label="programme loading")
    ax.set_title("Figure 3 · Programme loadings over the 16 chemistry evidence axes\n"
                 "percentages mark axes carrying >14% of a programme's loading — the basis of "
                 "its automatic description", loc="left", fontsize=10.5, weight="bold",
                 color=INK)
    save(fig, "F03_programme_loadings")


def f04(c):
    O = np.array(c.s["overlap_matrix"])
    fig, axs = plt.subplots(1, 2, figsize=(11.0, 4.6),
                            gridspec_kw={"width_ratios": [1.0, 1.15]})
    ax = axs[0]
    im = ax.imshow(O, cmap="Oranges", vmin=0, vmax=1)
    for i in range(len(O)):
        for j in range(len(O)):
            if i != j and O[i, j] > 0.05:
                ax.text(j, i, f"{O[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if O[i, j] > 0.6 else INK)
    ax.set_xticks(range(len(O))); ax.set_xticklabels([f"P{i}" for i in range(len(O))],
                                                     fontsize=7)
    ax.set_yticks(range(len(O))); ax.set_yticklabels([f"P{i}" for i in range(len(O))],
                                                     fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8, label="cosine of loadings")
    iu = np.triu_indices(len(O), 1)
    ax.set_title(f"a · overlap matrix — max {O[iu].max():.3f}, mean {O[iu].mean():.3f}",
                 fontsize=9, loc="left")
    ax = axs[1]
    d = c.coh.sort_values("usage_share")
    ax.barh(range(len(d)), d.usage_share, color=PURPLE, alpha=0.9)
    ax.axvline(0.60, color=RED, ls="--", lw=1.1)
    ax.text(0.605, 0.2, "dominance floor", fontsize=6.8, color=RED, rotation=90)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([f"P{int(p)}  {c.prog[int(p)]['auto_description'][:28]}"
                        for p in d.programme], fontsize=6.6)
    for i, v in enumerate(d.usage_share):
        ax.text(v + 0.006, i, f"{v:.2f}", va="center", fontsize=7)
    ax.set_xlabel("share of spectra for which this programme is the top one")
    ax.set_title("b · usage — no programme dominates", fontsize=9, loc="left")
    fig.suptitle("Figure 4 · Programme overlap and usage", x=0.03, ha="left", fontsize=11.5,
                 weight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    save(fig, "F04_overlap")


def f05(c):
    order = np.argsort([sorted(set(c.cls)).index(x) for x in c.cls])
    fig, ax = plt.subplots(figsize=(11.6, 4.6))
    W = np.clip(c.W, 0, None)
    Wn = W / (W.sum(axis=1, keepdims=True) + 1e-12)
    im = ax.imshow(Wn[order].T, aspect="auto", cmap="magma", interpolation="nearest")
    ax.set_yticks(range(c.K))
    ax.set_yticklabels([f"P{i} {c.prog[i]['auto_description'][:30]}" for i in range(c.K)],
                       fontsize=6.6)
    bounds, prev = [], None
    for k, i in enumerate(order):
        if c.cls[i] != prev:
            bounds.append((k, c.cls[i])); prev = c.cls[i]
    for b, _ in bounds[1:]:
        ax.axvline(b, color="white", lw=0.6, alpha=0.75)
    ax.set_xticks([b for b, _ in bounds])
    ax.set_xticklabels([SH(l) for _, l in bounds], rotation=60, ha="right", fontsize=6)
    ax.set_xlabel("spectra, grouped by curated chemistry class (revealed for display only)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="programme activation share")
    ax.set_title("Figure 5 · Programme activation map — 375 spectra × "
                 f"{c.K} programmes", loc="left", fontsize=11, weight="bold", color=INK)
    save(fig, "F05_activation_map")


def f06(c):
    fig, axs = plt.subplots(3, 3, figsize=(12.4, 7.4))
    for ax, p in zip(axs.ravel(), c.prog):
        mols = p["representative_molecules"][:7][::-1]
        ax.barh(range(len(mols)), [m["mean_activation"] for m in mols], color=PURPLE, alpha=0.9)
        ax.set_yticks(range(len(mols)))
        ax.set_yticklabels([m["molecule"][:24] for m in mols], fontsize=6.2)
        ax.set_title(f"P{p['programme']} · {p['auto_description'][:40]}", fontsize=7.4,
                     loc="left")
        ax.tick_params(labelsize=6.4); ax.set_xlabel("mean activation", fontsize=6.6)
    for ax in axs.ravel()[len(c.prog):]:
        ax.axis("off")
    fig.suptitle("Figure 6 · Representative molecules per programme — the molecules that "
                 "activate it most", x=0.03, ha="left", fontsize=11, weight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, "F06_representative_molecules")


def f07(c):
    fig, axs = plt.subplots(3, 3, figsize=(12.4, 7.0))
    for ax, p in zip(axs.ravel(), c.prog):
        names = [m["molecule"] for m in p["representative_molecules"][:5]]
        sel = np.isin(c.y, names)
        if sel.sum() == 0:
            ax.axis("off"); continue
        Xs = c.X[sel]
        mu, sd = Xs.mean(axis=0), Xs.std(axis=0)
        ax.fill_between(c.grid, mu - sd, mu + sd, color=PURPLE, alpha=0.16)
        ax.plot(c.grid, mu, color=PURPLE, lw=1.0)
        for b in p.get("dominant_bands", [])[:6]:
            ax.axvline(b, color=AMBER, ls=":", lw=0.8, alpha=0.9)
        ax.set_title(f"P{p['programme']} · {p['auto_description'][:38]}", fontsize=7.2,
                     loc="left")
        ax.set_xlabel("cm$^{-1}$", fontsize=6.6); ax.tick_params(labelsize=6.4)
    for ax in axs.ravel()[len(c.prog):]:
        ax.axis("off")
    fig.suptitle("Figure 7 · Mean Raman spectrum ± 1 sd of each programme's top molecules "
                 "(amber = the programme's dominant CSM bands)\n"
                 "spectra are shown for interpretation only — they were never an input to the "
                 "factorisation", x=0.03, ha="left", fontsize=10.5, weight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, "F07_programme_spectra")


def f08(c):
    fig, axs = plt.subplots(1, 3, figsize=(12.2, 4.0))
    d = c.stab.sort_values("bootstrap_recovery")
    ax = axs[0]
    cols = [GREEN if v >= 0.70 else RED for v in d.bootstrap_recovery]
    ax.barh(range(len(d)), d.bootstrap_recovery, color=cols, alpha=0.9)
    ax.axvline(0.70, color=RED, ls="--", lw=1.1)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([f"P{int(p)}" for p in d.programme], fontsize=7)
    for i, v in enumerate(d.bootstrap_recovery):
        ax.text(v + 0.008, i, f"{v:.3f}", va="center", fontsize=7)
    ax.set_xlim(0, 1.08); ax.set_xlabel("Hungarian-matched bootstrap recovery")
    ax.set_title("a · per-programme recovery", fontsize=9, loc="left")
    ax = axs[1]
    st = c.s["stability"]
    ks = ["bootstrap", "seed", "fold"]
    ax.bar(range(3), [st[k] for k in ks], color=[PURPLE, BLUE, TEAL], alpha=0.9, width=0.55)
    for i, k in enumerate(ks):
        ax.text(i, st[k] + 0.012, f"{st[k]:.3f}", ha="center", fontsize=9, weight="bold")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["bootstrap\n(resample spectra)", "seed\n(refit)",
                        "fold\n(withhold molecules)"], fontsize=7.4)
    ax.set_ylim(0, 1.1); ax.set_ylabel("programme recovery")
    ax.set_title("b · three kinds of stability", fontsize=9, loc="left")
    ax = axs[2]
    g = c.gen
    x = np.arange(len(g))
    ax.bar(x - 0.2, g.explained_variance, 0.38, color=GREEN, alpha=0.9, label="held-out EV")
    ax.bar(x + 0.2, g.mean_cosine, 0.38, color=BLUE, alpha=0.9, label="held-out cosine")
    ax.axhline(c.s["reconstruction"]["explained_variance"], color=RED, ls="--", lw=1.1)
    ax.text(0.05, c.s["reconstruction"]["explained_variance"] + 0.012, "in-sample EV",
            fontsize=6.8, color=RED)
    ax.set_xticks(x); ax.set_xticklabels([f"fold {int(f)}" for f in g.fold], fontsize=7.4)
    ax.set_ylim(0, 1.08); ax.legend(frameon=False, fontsize=7)
    ax.set_title("c · generalisation to held-out molecules", fontsize=9, loc="left")
    fig.suptitle("Figure 8 · Programme stability and generalisation", x=0.03, ha="left",
                 fontsize=11.5, weight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    save(fig, "F08_stability")


def f09(c):
    kinds = list(dict.fromkeys(c.rob.perturbation))
    fig, axs = plt.subplots(1, len(kinds), figsize=(2.5 * len(kinds), 3.6), squeeze=False)
    for ax, k in zip(axs[0], kinds):
        d = c.rob[c.rob.perturbation == k].sort_values("level")
        ax.plot(range(len(d)), d.programme_cosine, "o-", color=PURPLE, lw=1.4, ms=4,
                label="programme cosine")
        ax.plot(range(len(d)), d.argmax_stability, "s--", color=GREY, lw=1.0, ms=3.2,
                label="argmax stability")
        ax.set_xticks(range(len(d)))
        ax.set_xticklabels([f"{v:g}" for v in d.level], fontsize=6)
        ax.set_ylim(0, 1.05); ax.set_title(k.replace("_", " "), fontsize=7.6, loc="left")
        ax.tick_params(labelsize=6.4)
    axs[0, 0].legend(frameon=False, fontsize=6.2)
    axs[0, 0].set_ylabel("retention", fontsize=7.4)
    nr = c.s["noise_robustness"]
    fig.suptitle(f"Figure 9 · Noise robustness propagated through the whole frozen chain "
                 f"(spectrum → CSM → Chemistry Evidence → BSV2)\n"
                 f"mean programme cosine {nr['mean_programme_cosine']:.3f}, argmax stability "
                 f"{nr['mean_argmax_stability']:.3f}", x=0.03, ha="left", fontsize=10,
                 weight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    save(fig, "F09_noise_robustness")


def f10(c):
    fig = plt.figure(figsize=(12.4, 5.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.0, 1.0], wspace=0.30)
    ax = fig.add_subplot(gs[0])
    d = c.per_axis.sort_values("explained_variance")
    cols = [GREEN if v >= 0.5 else (AMBER if v >= 0.2 else RED) for v in d.explained_variance]
    ax.barh(range(len(d)), d.explained_variance, color=cols, alpha=0.9)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([f"{SH(a)}  (mean {m:.2f})" for a, m in
                        zip(d.chemistry_axis, d.mean_evidence)], fontsize=6.8)
    ax.axvline(0, color=INK, lw=0.9)
    for i, v in enumerate(d.explained_variance):
        ax.text(v + 0.015 if v >= 0 else v - 0.015, i, f"{v:+.2f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=6.4)
    ax.set_xlabel("explained variance"); ax.set_xlim(min(-0.2, d.explained_variance.min() * 1.2),
                                                     1.12)
    ax.set_title("a · reconstruction per chemistry axis", fontsize=9, loc="left")
    ax = fig.add_subplot(gs[1])
    n = min(200, len(c.Ev))
    ax.scatter(c.Ev[:n].ravel(), np.clip(c.R, 0, None)[:n].ravel(), s=3, color=PURPLE,
               alpha=0.25)
    lim = max(c.Ev.max(), np.clip(c.R, 0, None).max())
    ax.plot([0, lim], [0, lim], color=RED, ls="--", lw=1.1)
    ax.set_xlabel("original chemistry evidence"); ax.set_ylabel("reconstructed")
    r = c.s["reconstruction"]
    ax.set_title(f"b · EV {r['explained_variance']:.3f} · RMSE {r['rmse']:.3f}", fontsize=9,
                 loc="left")
    ax = fig.add_subplot(gs[2])
    i = int(np.argmax(c.Ev.max(axis=1)))
    x = np.arange(16)
    ax.bar(x - 0.2, c.Ev[i], 0.38, color=GREEN, alpha=0.9, label="original")
    ax.bar(x + 0.2, np.clip(c.R, 0, None)[i], 0.38, color=PURPLE, alpha=0.9,
           label="reconstructed")
    ax.set_xticks(x); ax.set_xticklabels([SH(a) for a in c.axes], rotation=70, ha="right",
                                         fontsize=5.6)
    ax.legend(frameon=False, fontsize=7)
    ax.set_title(f"c · one spectrum: {c.y[i][:24]}", fontsize=9, loc="left")
    fig.suptitle("Figure 10 · Can BSV2 reconstruct the Chemistry Evidence vector?", x=0.03,
                 ha="left", fontsize=11.5, weight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    save(fig, "F10_reconstruction")


def f11(c):
    import networkx as nx
    O = np.array(c.s["overlap_matrix"])
    G = nx.Graph()
    for i in range(len(O)):
        G.add_node(i)
    for i in range(len(O)):
        for j in range(i + 1, len(O)):
            if O[i, j] > 0.05:
                G.add_edge(i, j, weight=float(O[i, j]))
    pos = nx.spring_layout(G, seed=0, weight="weight", iterations=300)
    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    ew = [G[u][v]["weight"] for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, ax=ax, width=[3.5 * w for w in ew], alpha=0.35,
                           edge_color=ew, edge_cmap=plt.get_cmap("Oranges"))
    sizes = [2600 * c.coh.set_index("programme").loc[i, "usage_share"] + 320
             for i in G.nodes()]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=sizes, node_color=PURPLE, alpha=0.85)
    for i in G.nodes():
        ax.annotate(f"P{i}\n{c.prog[i]['auto_description'][:22]}", pos[i], fontsize=6.2,
                    ha="center", va="center", color="white", weight="bold")
    ax.axis("off")
    iu = np.triu_indices(len(O), 1)
    ax.set_title("Figure 11 · Programme similarity graph — edge width is loading cosine, node "
                 f"size is usage share\nedges shown above 0.05; max overlap {O[iu].max():.3f}",
                 loc="left", fontsize=11, weight="bold", color=INK)
    save(fig, "F11_programme_graph")


def f12(c):
    fig = plt.figure(figsize=(12.4, 6.8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0], hspace=0.42, wspace=0.30)
    d = c.cmp.set_index("representation")
    order = ["chemistry_evidence_16", "BSV2_programmes", "PCA_control"]
    cols = [GREEN, PURPLE, GREY]
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(3)
    ax.bar(x - 0.2, [d.loc[o, "heldout_chemistry_top1"] for o in order], 0.38, color=cols,
           alpha=0.95, label="top-1")
    ax.bar(x + 0.2, [d.loc[o, "heldout_chemistry_top3"] for o in order], 0.38, color=cols,
           alpha=0.55, label="top-3")
    for i, o in enumerate(order):
        ax.text(i - 0.2, d.loc[o, "heldout_chemistry_top1"] + 0.014,
                f"{d.loc[o, 'heldout_chemistry_top1']:.3f}", ha="center", fontsize=7,
                weight="bold")
    ax.set_xticks(x); ax.set_xticklabels(["Chemistry\nEvidence 16", f"BSV2\n{c.K}",
                                          f"PCA control\n{c.K}"], fontsize=7.4)
    ax.set_ylim(0, 1.05); ax.legend(frameon=False, fontsize=7)
    ax.set_title("a · held-out chemistry prediction", fontsize=9, loc="left")
    ax = fig.add_subplot(gs[0, 1])
    ax.bar(x, [d.loc[o, "mutual_information_norm"] for o in order], color=cols, alpha=0.9,
           width=0.55)
    for i, o in enumerate(order):
        ax.text(i, d.loc[o, "mutual_information_norm"] + 0.01,
                f"{d.loc[o, 'mutual_information_norm']:.2f}", ha="center", fontsize=8,
                weight="bold")
    ax.set_xticks(x); ax.set_xticklabels(["Chem Ev", "BSV2", "PCA"], fontsize=8)
    ax.set_ylabel("normalised MI with chemistry class")
    ax.set_title("b · information about chemistry", fontsize=9, loc="left")
    ax = fig.add_subplot(gs[0, 2])
    ax.bar(x, [d.loc[o, "effective_rank"] for o in order], color=cols, alpha=0.9, width=0.55)
    ax.axhline(c.K, color=RED, ls="--", lw=1.0)
    for i, o in enumerate(order):
        ax.text(i, d.loc[o, "effective_rank"] + 0.16, f"{d.loc[o, 'effective_rank']:.2f}",
                ha="center", fontsize=8, weight="bold")
    ax.set_xticks(x); ax.set_xticklabels(["Chem Ev", "BSV2", "PCA"], fontsize=8)
    ax.set_ylabel("effective rank")
    ax.set_title(f"c · nominal K = {c.K} (red)", fontsize=9, loc="left")
    ax = fig.add_subplot(gs[1, :]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    g = c.gates
    ncol = 6
    nrow = int(np.ceil(len(g) / ncol))
    h = min(0.26, (0.92 - 0.10) / nrow - 0.030)
    for i, (_, r) in enumerate(g.iterrows()):
        gx = 0.008 + (i % ncol) * 0.166
        gy = 0.84 - (i // ncol) * (h + 0.032)
        ok = r.status == "PASS"
        box(ax, gx, gy, 0.156, h, "\n".join(textwrap.wrap(r.gate, 26)),
            "#ecfdf5" if ok else "#fef2f2", GREEN if ok else RED, 5.6)
    p02 = c.s.get("p02_compliant_alternative")
    ax.text(0.008, 0.03,
            f"{int((g.status == 'PASS').sum())} of {len(g)} gates pass · adopted "
            f"{c.s['model']['family']} K={c.K} · compression {16 / c.K:.1f}x" +
            (f" · P-02-compliant alternative {p02['family']} K={p02['K']} "
             f"(objective {p02['objective_cost_vs_rule_winner']:+.3f})" if p02 else ""),
            fontsize=7.4, color=MUTED)
    fig.suptitle("Figure 12 · Chemistry Evidence versus BSV2 versus a PCA control, and the "
                 "decision gate", x=0.03, ha="left", fontsize=11.5, weight="bold", color=INK)
    save(fig, "F12_comparison")


def main():
    c = C()
    print("[figures]")
    for fn in (f01, f02, f03, f04, f05, f06, f07, f08, f09, f10, f11, f12):
        fn(c)
    assert not list(F.glob("*.svg")), "PNG only"
    print(f"[figures] {len(list(F.glob('*.png')))} PNG written to {F}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
