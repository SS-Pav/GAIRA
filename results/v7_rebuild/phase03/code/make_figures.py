#!/usr/bin/env python3
"""GAIRA V7 — Phase 03 figures (PNG, 200 dpi). Deterministic; seeds fixed."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
from gaira.v7.io import PhaseOutputs, frozen_root      # noqa: E402

OUT = PhaseOutputs("03")
T, A, V, F = OUT.tables, OUT.artifacts, OUT.validation, OUT.figures
FROZEN = frozen_root()
INK, MUTED, LINE = "#1a1a1a", "#6b7280", "#9ca3af"
BLUE, GREEN, AMBER, RED, GREY = "#2563eb", "#15803d", "#b45309", "#b91c1c", "#4b5563"
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
        z = np.load(A / "theme_membership_v1.npz", allow_pickle=True)
        self.S, self.TH = z["S"], z["THEMES"]
        self.csm_ids = [str(s) for s in z["csm_ids"]]
        self.theme_ids = [str(s) for s in z["theme_ids"]]
        self.grid, self.D, self.coords = z["grid"], z["D_csm"], z["coords"]
        self.reg = json.loads((A / "theme_registry_v1.json").read_text())
        self.state = json.loads((OUT.root / "PHASE_STATE.json").read_text())
        self.roles = pd.read_csv(T / "membership_roles_v1.csv")
        self.sweep = pd.read_csv(T / "model_k_sweep_v1.csv")
        self.grad = pd.read_csv(V / "theme_gradients_v1.csv")
        self.rob = pd.read_csv(V / "robustness_v1.csv")
        self.hier = json.loads((A / "hierarchy_v1.json").read_text())
        cr = json.loads((FROZEN / "phase02/artifacts/csm_registry_v1.json").read_text())
        by = {c["csm_id"]: c for c in cr["csms"]}
        self.csm_class = [(by[c]["supporting_classes"][0]
                           if len(by[c]["supporting_classes"]) == 1 else "multi")
                          for c in self.csm_ids]
        self.X = np.asarray(np.load(FROZEN / "phase02/artifacts/csm_dictionary_v1.npz",
                                    allow_pickle=True)["CSM"], float)
        self.names = [t["name"] for t in self.reg["themes"]]
        self.short = [f"{t['theme_id']}\n{t['name'][:26]}" for t in self.reg["themes"]]
        cm = plt.get_cmap("tab10")
        self.tcol = [cm(i % 10) for i in range(len(self.theme_ids))]
        fams = sorted(set(self.csm_class))
        cm2 = plt.get_cmap("tab20")
        self.ccol = {c: cm2(i % 20) for i, c in enumerate(fams)}
        self.fams = fams


# ── 1. K and model selection ─────────────────────────────────────────────────
def f01_selection(c):
    s = c.sweep
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.2), gridspec_kw={"wspace": 0.32})
    ax = axes[0]
    for m, sub in s.groupby("model"):
        ax.plot(sub.K, sub.information_retained, "o-", ms=3, lw=1.0, label=m)
    ax.set_xlabel("K"); ax.set_ylabel("information retained")
    ax.set_title("Reconstruction of the CSM spectra", fontsize=9, loc="left", color=INK)
    ax.legend(frameon=False, fontsize=6)
    ax = axes[1]
    for m, sub in s.groupby("model"):
        ax.plot(sub.K, sub.stability, "o-", ms=3, lw=1.0)
    ax.set_xlabel("K"); ax.set_ylabel("bootstrap stability")
    ax.set_title("Stability — high when degenerate, so read with the left panel",
                 fontsize=9, loc="left", color=INK)
    ax = axes[2]
    ok = s[(~s.degenerate) & s.chemically_admissible & s.themes_distinct]
    bad = s[s.degenerate]
    ax.scatter(bad.K, bad.effective_K, s=26, color=RED, label="degenerate membership")
    ax.scatter(s[~s.degenerate].K, s[~s.degenerate].effective_K, s=22, color=LINE,
               label="viable")
    ax.scatter(ok.K, ok.effective_K, s=52, color=GREEN, edgecolor=INK, linewidth=0.6,
               label="admissible + distinct", zorder=3)
    ax.axhline(2.0, color=RED, ls="--", lw=0.9)
    ax.set_xlabel("K"); ax.set_ylabel("effective number of themes used")
    ax.set_title(f"Selected: {c.state['selected_model']}, K = {c.state['K']}",
                 fontsize=9, loc="left", color=INK)
    ax.legend(frameon=False, fontsize=6)
    for a in axes:
        for sp in ("top", "right"): a.spines[sp].set_visible(False)
    fig.suptitle("Model and K selection on label-free criteria — no chemistry label is "
                 "visible at this stage", fontsize=9.5, x=0.005, ha="left", y=1.06, color=INK)
    save(fig, "fig01_model_and_k_selection")


# ── 2. theme spectra with band annotation ────────────────────────────────────
def f02_spectra(c):
    K = len(c.theme_ids)
    fig, axes = plt.subplots(K, 1, figsize=(9.2, 1.55 * K), sharex=True,
                             gridspec_kw={"hspace": 0.62})
    for k, ax in enumerate(np.atleast_1d(axes)):
        t = c.reg["themes"][k]
        th = c.TH[k]
        ax.fill_between(c.grid, 0, th / th.max(), color=c.tcol[k], alpha=0.75)
        for b, lab in zip(t["dominant_bands_cm1"], t["band_assignments"]):
            ax.axvline(b, color=LINE, lw=0.6, ls=":")
            ax.text(b, 1.04, f"{int(b)}", fontsize=5.8, ha="center", color=MUTED)
        ax.set_ylim(0, 1.35); ax.set_yticks([])
        ax.set_title(f"{t['theme_id']} — {t['name']}   ·   {t['n_supporting_csms']} CSMs · "
                     f"confidence {t['confidence']:.2f} · concentration "
                     f"{t['family_concentration']:.2f}",
                     fontsize=8.2, loc="left", color=INK)
        for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
    np.atleast_1d(axes)[-1].set_xlabel("Raman shift (cm⁻¹)")
    fig.suptitle("Theme spectra with their dominant bands. Names come from the bands, and only "
                 "after validation.", fontsize=9.5, x=0.005, ha="left", y=0.995, color=INK)
    save(fig, "fig02_theme_spectra")


# ── 3. membership heatmap ────────────────────────────────────────────────────
def f03_membership(c):
    order = np.lexsort((-c.S.max(1), c.S.argmax(1)))
    fig, ax = plt.subplots(figsize=(6.4, 8.2))
    im = ax.imshow(c.S[order], aspect="auto", cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(len(c.theme_ids)))
    ax.set_xticklabels(c.short, fontsize=6, rotation=35, ha="right")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([c.csm_ids[i] for i in order], fontsize=5)
    for y, i in enumerate(order):
        r = c.roles.iloc[i]
        if r.role != "member":
            ax.text(-0.9, y, "◆" if r.role == "bridge" else "○",
                    color=AMBER if r.role == "bridge" else RED, fontsize=6, va="center")
    fig.colorbar(im, ax=ax, shrink=0.55, label="membership")
    ax.set_title("Soft membership S (49 CSMs × 6 themes), rows sum to 1.\n"
                 "◆ bridge (several themes claim it)   ○ poorly explained (its theme does not reconstruct it)",
                 fontsize=9, loc="left", color=INK)
    save(fig, "fig03_membership_matrix")


# ── 4. theme overlap network ─────────────────────────────────────────────────
def f04_overlap(c):
    K = len(c.theme_ids)
    Ov = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            if i != j:
                Ov[i, j] = float(np.minimum(c.S[:, i], c.S[:, j]).sum())
    G = nx.Graph()
    for k in range(K):
        G.add_node(k)
    for i in range(K):
        for j in range(i + 1, K):
            if Ov[i, j] > 0.15:
                G.add_edge(i, j, weight=float(Ov[i, j]))
    pos = nx.circular_layout(G)
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for u, v, d in G.edges(data=True):
        ax.plot(*zip(pos[u], pos[v]), lw=0.6 + 4 * d["weight"] / (Ov.max() + 1e-9),
                color=LINE, alpha=0.6, zorder=1)
        m = (np.array(pos[u]) + np.array(pos[v])) / 2
        ax.text(*m, f"{d['weight']:.1f}", fontsize=6.5, color=MUTED, ha="center")
    for k in range(K):
        sz = 260 + 900 * c.S[:, k].sum() / c.S.sum()
        ax.scatter(*pos[k], s=sz, color=c.tcol[k], edgecolor=INK, linewidth=0.8, zorder=3)
        ax.annotate(c.short[k], pos[k], fontsize=6.4, ha="center", va="center", zorder=4)
    ax.set_axis_off()
    ax.set_title("Theme overlap network. Edge weight is shared membership mass — themes are "
                 "allowed\nto overlap, and the CSMs on those edges are the bridges.",
                 fontsize=9.5, loc="left", color=INK)
    save(fig, "fig04_theme_overlap_network")


# ── 5. hierarchy ─────────────────────────────────────────────────────────────
def f05_hierarchy(c):
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import squareform
    N = c.TH / (np.linalg.norm(c.TH, axis=1, keepdims=True) + 1e-12)
    Dm = 1.0 - np.clip(N @ N.T, -1, 1)
    np.fill_diagonal(Dm, 0)
    Z = linkage(squareform(Dm, checks=False), method="average")
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.6),
                             gridspec_kw={"wspace": 0.3, "width_ratios": [1.2, 1]})
    dendrogram(Z, labels=[t.replace("\n", " ") for t in c.short], ax=axes[0],
               leaf_font_size=6.5, color_threshold=0.6 * Z[:, 2].max())
    axes[0].set_ylabel("1 − theme spectral cosine")
    axes[0].set_title(f"Inferred hierarchy — {c.hier['n_levels']} levels, not assumed",
                      fontsize=9, loc="left", color=INK)
    ax = axes[1]
    lv = pd.DataFrame([{"level": l["level"], "n_groups": l["n_groups"],
                        "within_correlation": l["mean_within_correlation"]}
                       for l in c.hier["levels"]])
    ax.plot(lv.n_groups, lv.within_correlation, "o-", color=BLUE, ms=5)
    for r in lv.itertuples():
        ax.annotate(f"L{r.level}", (r.n_groups, r.within_correlation), fontsize=7,
                    xytext=(4, 4), textcoords="offset points", color=MUTED)
    ax.set_xlabel("groups at this level"); ax.set_ylabel("mean within-group theme correlation")
    ax.set_title("Coherence of each inferred level", fontsize=9, loc="left", color=INK)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    save(fig, "fig05_hierarchy")


# ── 6. stability ─────────────────────────────────────────────────────────────
def f06_stability(c):
    ths = c.reg["themes"]
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.2), gridspec_kw={"wspace": 0.34})
    y = np.arange(len(ths))[::-1]
    ax = axes[0]
    ax.barh(y - 0.18, [t["bootstrap_stability"] for t in ths], height=0.34,
            color=BLUE, label="bootstrap")
    ax.barh(y + 0.18, [t["loo_stability"] for t in ths], height=0.34, color=GREEN,
            label="leave-one-out")
    ax.axvline(0.60, color=RED, ls="--", lw=0.9)
    ax.set_yticks(y); ax.set_yticklabels([t["theme_id"] for t in ths], fontsize=7)
    ax.set_xlabel("theme recovery"); ax.set_xlim(0, 1.05)
    ax.legend(frameon=False, fontsize=6.5)
    ax.set_title("Per-theme stability (rejection floor 0.60)", fontsize=9, loc="left", color=INK)
    ax = axes[1]
    rr = c.rob[c.rob.testable]
    ax.barh(np.arange(len(rr))[::-1], rr.theme_recovery, color=AMBER, height=0.6)
    ax.set_yticks(np.arange(len(rr))[::-1])
    ax.set_yticklabels(rr.held_out, fontsize=6)
    ax.axvline(0.60, color=RED, ls="--", lw=0.9)
    ax.set_xlabel("theme recovery"); ax.set_xlim(0, 1.05)
    ax.set_title("Leave-one-source / one-excitation out", fontsize=9, loc="left", color=INK)
    ax = axes[2]
    ax.hist(c.roles.membership_entropy, bins=18, color=GREY, alpha=0.8)
    ax.axvline(np.quantile(c.roles.membership_entropy, 0.70), color=AMBER, lw=1.2)
    ax.text(np.quantile(c.roles.membership_entropy, 0.70), ax.get_ylim()[1] * 0.9,
            " bridge threshold", fontsize=6.5, color=AMBER)
    ax.set_xlabel("membership entropy"); ax.set_ylabel("CSMs")
    ax.set_title("How split membership actually is", fontsize=9, loc="left", color=INK)
    for a in axes:
        for sp in ("top", "right"): a.spines[sp].set_visible(False)
    save(fig, "fig06_theme_stability")


# ── 7. bridges ───────────────────────────────────────────────────────────────
def f07_bridges(c):
    br = c.roles[c.roles.role == "bridge"].sort_values("membership_entropy", ascending=False)
    if br.empty:
        br = c.roles.nlargest(6, "membership_entropy")
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0),
                             gridspec_kw={"wspace": 0.32, "width_ratios": [1.15, 1]})
    ax = axes[0]
    idx = [c.csm_ids.index(x) for x in br.csm_id]
    bottom = np.zeros(len(idx))
    for k in range(len(c.theme_ids)):
        vals = c.S[idx, k]
        ax.barh(np.arange(len(idx)), vals, left=bottom, color=c.tcol[k], height=0.68,
                label=c.theme_ids[k] if k < 10 else None)
        bottom += vals
    ax.set_yticks(np.arange(len(idx)))
    ax.set_yticklabels([f"{r.csm_id}" for r in br.itertuples()], fontsize=6.5)
    ax.set_xlabel("membership"); ax.set_xlim(0, 1)
    ax.legend(frameon=False, fontsize=6, ncol=3, loc="lower right")
    ax.set_title(f"Bridge CSMs ({len(br)}) — membership genuinely split, and left that way",
                 fontsize=9, loc="left", color=INK)
    ax = axes[1]
    ax.scatter(c.roles.membership_entropy, c.roles.best_theme_fit, s=30,
               color=[AMBER if r == "bridge" else RED if r == "poorly_explained" else LINE
                      for r in c.roles.role], edgecolor="white", linewidth=0.4)
    ax.axhline(0.35, color=RED, ls="--", lw=0.9)
    ax.text(c.roles.membership_entropy.min(), 0.355, " poorly explained below this fit",
            fontsize=6.5, color=RED)
    ax.set_xlabel("membership entropy"); ax.set_ylabel("fit of the best theme")
    ax.set_title("Bridges and unassigned CSMs are opposite findings\n"
                 "— explained by several themes, versus explained by none",
                 fontsize=9, loc="left", color=INK)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    save(fig, "fig07_bridge_csms")


# ── 8. unassigned / isolated ─────────────────────────────────────────────────
def f08_unassigned(c):
    un = c.roles[c.roles.role == "poorly_explained"]
    fig, ax = plt.subplots(figsize=(9.0, 1.0 + 1.1 * max(len(un), 1)))
    if un.empty:
        ax.text(0.5, 0.5, "no poorly-explained CSMs — every CSM is reconstructed by its theme",
                ha="center", fontsize=10, color=MUTED)
        ax.set_axis_off()
    else:
        for k, r in enumerate(un.itertuples()):
            i = c.csm_ids.index(r.csm_id)
            x = c.X[i]
            ax.plot(c.grid, x / x.max() + k * 0.6, lw=1.0, color=RED, alpha=0.85)
            ax.text(470, k * 0.6 + 0.35,
                    f"{r.csm_id}  [{c.csm_class[i]}]  best-theme fit {r.best_theme_fit:.2f}",
                    fontsize=7, color=INK)
        ax.set_xlim(450, 1800); ax.set_yticks([])
        ax.set_xlabel("Raman shift (cm⁻¹)")
        for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
    ax.set_title("Poorly-explained CSMs — a theme claims them, but does not reconstruct them.\n"
                 "Recorded rather than absorbed: inventing a theme for an isolate is a motif "
                 "borrowing foreign mass (L-03).", fontsize=9.5, loc="left", color=INK)
    save(fig, "fig08_unassigned_csms")


# ── 9. gradients ─────────────────────────────────────────────────────────────
def f09_gradients(c):
    g = c.grad[c.grad.is_gradient].sort_values("abs_spearman", ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.4), gridspec_kw={"wspace": 0.3})
    ax = axes[0]
    piv = c.grad.pivot(index="theme", columns="diffusion_coord", values="spearman")
    im = ax.imshow(piv.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels([f"DC{i}" for i in piv.columns])
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels([c.theme_ids[i] for i in piv.index], fontsize=7)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            sig = c.grad[(c.grad.theme == piv.index[i])
                         & (c.grad.diffusion_coord == piv.columns[j])].iloc[0].is_gradient
            ax.text(j, i, f"{piv.values[i, j]:.2f}" + ("*" if sig else ""),
                    ha="center", va="center", fontsize=6.5,
                    color="white" if abs(piv.values[i, j]) > 0.5 else INK)
    fig.colorbar(im, ax=ax, shrink=0.75, label="Spearman ρ")
    ax.set_title("Membership against diffusion coordinates (* = p < 0.05 vs permutation)",
                 fontsize=9, loc="left", color=INK)
    ax = axes[1]
    if len(g):
        r = g.iloc[0]
        k, d = int(r.theme), int(r.diffusion_coord) - 1
        ax.scatter(c.coords[:, d], c.S[:, k], s=34, color=c.tcol[k], edgecolor="white",
                   linewidth=0.4)
        ax.set_xlabel(f"diffusion coordinate {d + 1}")
        ax.set_ylabel(f"membership in {c.theme_ids[k]}")
        ax.set_title(f"Strongest gradient: {c.theme_ids[k]} along DC{d + 1} "
                     f"(ρ = {r.spearman:.2f})", fontsize=9, loc="left", color=INK)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fig.suptitle("Phase 02.5 found a continuum; a theme layer that respects it shows "
                 "membership varying smoothly rather than switching",
                 fontsize=9.5, x=0.005, ha="left", y=1.05, color=INK)
    save(fig, "fig09_theme_gradients")


# ── 10. theme map with chemistry revealed ────────────────────────────────────
def f10_map(c):
    e = np.load(FROZEN / "phase02_5/artifacts/embeddings_v1.npz", allow_pickle=True)
    cr = json.loads((FROZEN / "phase02/artifacts/csm_registry_v1.json").read_text())
    lids = [str(s) for s in np.load(FROZEN / "phase02_5/artifacts/geometry_v1.npz",
                                    allow_pickle=True)["motif_ids"]]
    by = {x["csm_id"]: x for x in cr["csms"]}
    E = np.array([e["umap"][[lids.index(l["lsm_id"]) for l in by[cid]["contributing_lsms"]]].mean(0)
                  for cid in c.csm_ids])
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4), gridspec_kw={"wspace": 0.22})
    ax = axes[0]
    top = c.S.argmax(1)
    for i in range(len(c.csm_ids)):
        ax.scatter(E[i, 0], E[i, 1], s=40 + 160 * c.S[i].max(), color=c.tcol[top[i]],
                   edgecolor="white", linewidth=0.5, alpha=0.9)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Coloured by dominant theme; size = membership confidence",
                 fontsize=9, loc="left", color=INK)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=c.tcol[k],
                              label=f"{c.theme_ids[k]} {c.names[k][:22]}", markersize=5)
                       for k in range(len(c.theme_ids))],
              frameon=False, fontsize=6, loc="upper left")
    ax = axes[1]
    for i in range(len(c.csm_ids)):
        ax.scatter(E[i, 0], E[i, 1], s=44, color=c.ccol[c.csm_class[i]],
                   edgecolor="white", linewidth=0.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Coloured by curated chemistry — revealed only after the themes were fixed",
                 fontsize=9, loc="left", color=INK)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=c.ccol[f], label=f,
                              markersize=4.5) for f in c.fams],
              loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=5.8)
    save(fig, "fig10_theme_map")


# ── 11. reconstruction ───────────────────────────────────────────────────────
def f11_reconstruction(c):
    from scipy.optimize import nnls
    ev_c, ev_t = [], []
    for x in c.X:
        a = nnls(c.X.T, x)[0]
        ev_c.append(max(0.0, 1 - ((x - a @ c.X) ** 2).sum() / ((x ** 2).sum() + 1e-12)))
        b = nnls(c.TH.T, x)[0]
        ev_t.append(max(0.0, 1 - ((x - b @ c.TH) ** 2).sum() / ((x ** 2).sum() + 1e-12)))
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.6), gridspec_kw={"wspace": 0.3})
    ax = axes[0]
    order = np.argsort(ev_t)
    ax.barh(np.arange(len(order)), np.array(ev_t)[order], color=BLUE, height=0.75)
    ax.set_yticks(np.arange(len(order))[::4])
    ax.set_yticklabels([c.csm_ids[i] for i in order][::4], fontsize=5.5)
    ax.axvline(np.mean(ev_t), color=RED, lw=1.0)
    ax.set_xlabel("explained variance from the 6-theme basis")
    ax.set_title(f"Per CSM — mean {np.mean(ev_t):.3f}", fontsize=9, loc="left", color=INK)
    ax = axes[1]
    worst = order[:3]
    for k, i in enumerate(worst):
        x = c.X[i]
        b = nnls(c.TH.T, x)[0]
        ax.plot(c.grid, x / x.max() + k * 1.1, lw=1.1, color=INK)
        ax.plot(c.grid, (b @ c.TH) / x.max() + k * 1.1, lw=1.0, color=RED, ls="--")
        ax.text(470, k * 1.1 + 0.75, f"{c.csm_ids[i]}  EV {ev_t[i]:.2f}", fontsize=7,
                color=INK)
    ax.set_xlim(450, 1800); ax.set_yticks([]); ax.set_xlabel("Raman shift (cm⁻¹)")
    ax.legend(handles=[Line2D([], [], color=INK, label="CSM"),
                       Line2D([], [], color=RED, ls="--", label="theme reconstruction")],
              frameon=False, fontsize=7)
    ax.set_title("The three worst reconstructions — what the theme layer throws away",
                 fontsize=9, loc="left", color=INK)
    for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
    for sp in ("top", "right"): axes[0].spines[sp].set_visible(False)
    save(fig, "fig11_reconstruction")


# ── 12. evidence summary ─────────────────────────────────────────────────────
def f12_evidence(c):
    ths = c.reg["themes"]
    cols = ["bootstrap", "leave-one-out", "family\nconcentration", "naming\nconfidence",
            "source\nrobust", "gradient"]
    Z = np.array([[t["bootstrap_stability"], t["loo_stability"], t["family_concentration"],
                   t["name_confidence"], 1.0 if t["source_robust"] else 0.0,
                   min(1.0, t["gradient"]["n_gradient_coords"] / 3)] for t in ths])
    fig, ax = plt.subplots(figsize=(7.6, 0.55 * len(ths) + 2.2))
    im = ax.imshow(Z, cmap="YlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, fontsize=7)
    ax.set_yticks(range(len(ths)))
    ax.set_yticklabels([f"{t['theme_id']} {t['name'][:30]}" for t in ths], fontsize=7)
    for i in range(len(ths)):
        for j in range(len(cols)):
            ax.text(j, i, f"{Z[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if Z[i, j] > 0.65 else INK)
    fig.colorbar(im, ax=ax, shrink=0.6)
    ax.set_title("Evidence summary. Every accepted theme also carries recorded "
                 "counter-evidence\nand alternative explanations in the catalogue — a theme "
                 "with neither has not been examined.",
                 fontsize=9, loc="left", color=INK)
    save(fig, "fig12_evidence_summary")


# ── 13. architecture ─────────────────────────────────────────────────────────
def f13_architecture(c):
    fig, ax = plt.subplots(figsize=(10.2, 3.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3.6); ax.set_axis_off()
    steps = [("00\nbenchmark\nlock", GREEN), ("01\n50 LSMs", GREEN), ("02\n49 CSMs", GREEN),
             ("02.5\nlatent\ngeometry", GREEN), ("03\n6 themes", BLUE),
             ("04\ncontinuous\nBSV", MUTED), ("05\ninference\nengine", MUTED),
             ("06\nRaman\nvalidation", MUTED)]
    for k, (lab, col) in enumerate(steps):
        x = 0.15 + k * 1.23
        ax.add_patch(FancyBboxPatch((x, 1.5), 1.05, 1.05, boxstyle="round,pad=0.05",
                                    fc="#eff6ff" if col == BLUE else
                                    ("#f0fdf4" if col == GREEN else "white"),
                                    ec=col, lw=1.7 if col == BLUE else 1.1))
        ax.text(x + 0.525, 2.02, lab, ha="center", va="center", fontsize=7, color=INK)
        if k:
            ax.add_patch(FancyArrowPatch((x - 0.15, 2.02), (x - 0.02, 2.02),
                                         arrowstyle="-|>", mutation_scale=8, color=MUTED,
                                         lw=0.9))
    ax.text(0.15, 1.05, "Phase 04 consumes: the membership matrix S (49 × 6), the 6-theme "
                        "basis, the theme registry with\nits confidences and bridge "
                        "annotations, and the unassigned list. BSV dimension = K = 6.",
            fontsize=7.8, color=INK)
    ax.text(0.15, 0.35, f"theme fingerprint {c.state['theme_fingerprint']}  ·  "
                        f"bootstrap {c.state['bootstrap_mean']:.3f}  ·  "
                        f"AMI {c.state['ontology_ami']:.3f}  ·  "
                        f"{c.state['themes']['n_bridge_csms']} bridges, "
                        f"{c.state['themes']['n_unassigned_csms']} poorly explained",
            fontsize=7.2, color=MUTED)
    ax.set_title("GAIRA V7 architecture after Phase 03", fontsize=10.5, loc="left", color=INK)
    save(fig, "fig13_architecture")


def main():
    c = Ctx()
    print("[phase03] figures")
    for fn in (f01_selection, f02_spectra, f03_membership, f04_overlap, f05_hierarchy,
               f06_stability, f07_bridges, f08_unassigned, f09_gradients, f10_map,
               f11_reconstruction, f12_evidence, f13_architecture):
        fn(c)


if __name__ == "__main__":
    main()
