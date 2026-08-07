#!/usr/bin/env python3
"""GAIRA V7 — Phase 06.5 figures. PNG only, 200 dpi, deterministic."""
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
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
from gaira.v7.io import PhaseOutputs, frozen_root                 # noqa: E402
from gaira.v7.latent import clustering as CLU                     # noqa: E402

OUT = PhaseOutputs("06_5", extra=("interactive", "manifests"))
T, A_, F = OUT.tables, OUT.artifacts, OUT.figures
FROZEN = frozen_root()
INK, MUTED, LINE = "#1a1a1a", "#6b7280", "#9ca3af"
BLUE, GREEN, AMBER, RED, GREY = "#2563eb", "#15803d", "#b45309", "#b91c1c", "#4b5563"
PURPLE, TEAL = "#7c3aed", "#0f766e"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
                     "figure.facecolor": "white", "savefig.facecolor": "white",
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.18})


def save(fig, name):
    F.mkdir(parents=True, exist_ok=True)
    fig.savefig(F / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"  {name}")


def box(ax, x, y, w, h, t, fc="#eef2ff", ec=BLUE, fs=8.0, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008", fc=fc, ec=ec, lw=1.1))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, color=INK,
            weight=weight, linespacing=1.35)


class C:
    def __init__(self):
        self.s = json.loads((A_ / "phase06_5_summary_v1.json").read_text())
        self.state = json.loads((OUT.root / "PHASE_STATE.json").read_text())
        self.comp = json.loads((A_ / "cluster_composition_v1.json").read_text())["clusters"]
        z = np.load(A_ / "continuous_coordinates_v1.npz", allow_pickle=True)
        self.U, self.Um, self.P = z["U"], z["U_molecule"], z["prototypes"]
        self.lab = z["labels"]
        self.mols = np.array([str(v) for v in z["molecules"]])
        self.y = np.array([str(v) for v in z["y"]])
        self.cls = np.array([str(v) for v in z["cls"]])
        e = np.load(A_ / "embeddings_v1.npz", allow_pickle=True)
        self.emb = {k: e[k] for k in e.files if k not in ("_provenance",)}
        self.fine_m = np.array([str(v) for v in e["fine_class"]])
        self.broad_m = np.array([str(v) for v in e["broad_class"]])
        self.src_m = np.array([str(v) for v in e["source"]])
        z6 = np.load(FROZEN / "phase06/artifacts/chemistry_evidence_predictions_v1.npz",
                     allow_pickle=True)
        self.A = z6["A_csm"]
        self.M = np.vstack([self.A[self.y == m].mean(axis=0) for m in self.mols])
        b = np.load(FROZEN / "phase01/artifacts/balanced_references_v1.npz", allow_pickle=True)
        self.X, self.grid = np.asarray(b["X"], float), np.asarray(b["grid"], float)
        self.CSM = np.load(FROZEN / "phase02/artifacts/csm_dictionary_v1.npz")["CSM"]
        self.recs = json.loads(
            (FROZEN / "phase02/artifacts/csm_registry_v1.json").read_text())["csms"]
        self.sweep = pd.read_csv(T / "cluster_stability_verdicts_v1.csv")
        self.byk = pd.read_csv(T / "cluster_stability_by_k_v1.csv")
        self.mono = pd.read_csv(T / "k_selection_monotonicity_v1.csv")
        self.vp = pd.read_csv(T / "permanova_variance_partition_v1.csv")
        self.ami = pd.read_csv(T / "cluster_vs_factor_ami_v1.csv")
        self.csw = pd.read_csv(T / "coordinate_kernel_sweep_v1.csv")
        self.crob = pd.read_csv(T / "coordinate_robustness_v1.csv")
        self.arms = pd.read_csv(T / "retrieval_benchmark_v1.csv")
        self.wsw = pd.read_csv(T / "fusion_weight_sweep_v1.csv")
        self.branch = pd.read_csv(T / "dendrogram_branch_points_v1.csv")
        self.free = pd.read_csv(T / "free_k_algorithms_v1.csv")
        self.gates = pd.read_csv(T / "phase06_5_gates_v1.csv")
        self.sig = json.loads((A_ / "retrieval_significance_v1.json").read_text())


def SH(c):
    return c.replace("_", " ").replace("carboxylic acid metabolite", "carboxylic acid")\
            .replace("phospholipid sphingolipid", "phospholipid")\
            .replace("mono oligosaccharide", "mono/oligosacch")\
            .replace("sulfur thiol cofactor", "sulfur/thiol")\
            .replace("nucleic acid polymer", "nucleic polymer")\
            .replace("chromophore pigment", "chromophore").replace("small nitrogenous", "small N")


# ── 1 stability curves ───────────────────────────────────────────────────────
def f01(c):
    fig, axs = plt.subplots(2, 3, figsize=(12.6, 6.6))
    d = c.sweep[c.sweep.usable]
    panels = [("silhouette", "silhouette (higher = tighter)", GREEN, axs[0, 0]),
              ("davies_bouldin", "Davies–Bouldin (lower = tighter)", RED, axs[0, 1]),
              ("neighbour_preservation", "k-NN preservation", BLUE, axs[0, 2]),
              ("bootstrap_ari_mean", "bootstrap ARI", PURPLE, axs[1, 0]),
              ("mean_cluster_survival", "mean cluster survival (Jaccard)", TEAL, axs[1, 1]),
              ("membership_entropy", "membership entropy", AMBER, axs[1, 2])]
    for col, lab, cc, ax in panels:
        for algo, g in d.groupby("algorithm"):
            g = g.sort_values("K")
            ax.plot(g.K, g[col], "o-", ms=3.4, lw=1.1, alpha=0.65, label=algo)
        m = d.groupby("K")[col].mean()
        ax.plot(m.index, m.values, "s-", color=cc, lw=2.2, ms=5, label="mean", zorder=5)
        ax.axvline(16, color=RED, ls=":", lw=1.2)
        ax.set_xlabel("K"); ax.set_ylabel(lab, fontsize=7.6)
        r = c.mono[c.mono["index"] == col]
        if len(r):
            ax.set_title(f"Spearman vs K {float(r.spearman_rho_vs_K.iloc[0]):+.2f} · interior "
                         f"optimum {bool(r.has_interior_optimum.iloc[0])}", fontsize=7.6,
                         loc="left")
        ax.tick_params(labelsize=7)
    axs[0, 0].legend(frameon=False, fontsize=6.2, ncol=2)
    axs[0, 2].text(16.6, 0.95, "K=16\n(ontology size)", fontsize=6.6, color=RED,
                   transform=axs[0, 2].get_xaxis_transform(), va="top")
    fig.suptitle("Figure 1 · Cluster stability across 14 K and 4 fixed-K algorithms — "
                 "NO index has an interior optimum\n"
                 "every curve is monotone or flat in K: the geometry has no preferred "
                 "cluster count, which is what a continuum looks like",
                 x=0.035, ha="left", fontsize=11, weight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    save(fig, "F01_stability_curves")


# ── 2 K=16 detail + free-K ───────────────────────────────────────────────────
def f02(c):
    fig, axs = plt.subplots(1, 3, figsize=(12.2, 4.0),
                            gridspec_kw={"width_ratios": [1.1, 1.0, 1.1]})
    d = c.sweep[(c.sweep.usable) & (c.sweep.K == 16)]
    ax = axs[0]
    x = np.arange(len(d))
    ax.bar(x - 0.2, d.bootstrap_ari_mean, 0.38, color=PURPLE, alpha=0.9, label="bootstrap ARI")
    ax.bar(x + 0.2, d.mean_cluster_survival, 0.38, color=TEAL, alpha=0.9,
           label="mean cluster survival")
    ax.axhline(0.60, color=RED, ls="--", lw=1.0)
    ax.set_xticks(x); ax.set_xticklabels(d.algorithm, rotation=25, ha="right", fontsize=7.4)
    ax.set_ylim(0, 1.05); ax.legend(frameon=False, fontsize=7.2)
    ax.set_title(f"a · K=16 by algorithm — {int((d.verdict=='stable').sum())}/{len(d)} stable",
                 fontsize=9, loc="left")
    ax = axs[1]
    fr = c.free[c.free.usable]
    ax.scatter(fr.n_clusters, fr.bootstrap_ari_mean, s=70,
               c=[BLUE if a == "hdbscan" else GREEN for a in fr.algorithm], alpha=0.85)
    for _, r in fr.iterrows():
        ax.annotate(f"{r.algorithm[:4]} p={r.param:g}\n{int(r.n_unassigned)} unassigned",
                    (r.n_clusters, r.bootstrap_ari_mean), textcoords="offset points",
                    xytext=(6, -4), fontsize=5.8)
    ax.axvline(16, color=RED, ls=":", lw=1.2)
    ax.set_xlabel("K chosen by the algorithm"); ax.set_ylabel("bootstrap ARI")
    ax.set_title("b · algorithms that choose their own K", fontsize=9, loc="left")
    ax = axs[2]
    ax.axis("off")
    ss = c.s["k_selection"]
    txt = (f"NO PREFERRED CLUSTER COUNT EXISTS\n\n"
           f"{ss['n_indices_with_interior_optimum']} of "
           f"{len(ss['monotonicity'])} internal indices have an interior optimum.\n\n"
           "Silhouette rises monotonically to K=30 (Spearman +1.00), neighbour preservation "
           "falls monotonically to K=30 (−1.00), membership entropy rises monotonically "
           "(+1.00). Every index is tracking granularity, not structure.\n\n"
           "K=16 is therefore adopted as a REPORTING CONVENTION — it matches the curated "
           "ontology so Section 7's comparison is like-for-like — and NOT as a discovered "
           "optimum. An earlier rule that maximised bootstrap ARI over all K chose K=4, "
           "because a coarse partition is trivially reproducible: the same stability-without-"
           "informativeness trap that principle P-18 exists to catch.")
    yy = 0.98
    for line in txt.split("\n"):
        for w in textwrap.wrap(line, 54) or [""]:
            ax.text(0.0, yy, w, fontsize=7.2, color=INK,
                    weight="bold" if "NO PREFERRED" in w else "normal")
            yy -= 0.052
    fig.suptitle("Figure 2 · Is K=16 genuinely stable, or merely convenient?", x=0.035,
                 ha="left", fontsize=11.5, weight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    save(fig, "F02_k16_detail")


# ── 3 composition ────────────────────────────────────────────────────────────
def f03(c):
    comp = sorted(c.comp, key=lambda r: -r["n_molecules"])
    fig, axs = plt.subplots(1, 3, figsize=(13.0, 6.0),
                            gridspec_kw={"width_ratios": [1.5, 1.0, 1.0]})
    classes = sorted({k for r in comp for k in r["fine_classes"]})
    cmap = plt.get_cmap("tab20")
    ax = axs[0]
    for i, r in enumerate(comp):
        left = 0
        for j, cl in enumerate(classes):
            v = r["fine_classes"].get(cl, 0)
            if v:
                ax.barh(i, v, left=left, color=cmap(j % 20), alpha=0.92)
                if v >= 3:
                    ax.text(left + v / 2, i, SH(cl)[:11], ha="center", va="center",
                            fontsize=5.4, color="white")
                left += v
    ax.set_yticks(range(len(comp)))
    ax.set_yticklabels([f"C{r['cluster']}  n={r['n_molecules']}  [{r['kind'][:12]}]"
                        for r in comp], fontsize=6.6)
    ax.set_xlabel("molecules"); ax.invert_yaxis()
    ax.set_title("a · chemistry composition of every emergent cluster", fontsize=9, loc="left")
    ax = axs[1]
    x = np.arange(len(comp))
    ax.barh(x - 0.22, [r["fine_purity"] for r in comp], 0.4, color=GREEN, alpha=0.9,
            label="chemistry purity")
    ax.barh(x + 0.22, [r["source_purity"] for r in comp], 0.4, color=RED, alpha=0.8,
            label="source purity")
    ax.set_yticks(x); ax.set_yticklabels([f"C{r['cluster']}" for r in comp], fontsize=6.6)
    ax.invert_yaxis(); ax.set_xlim(0, 1.05); ax.legend(frameon=False, fontsize=7)
    ax.set_title("b · chemistry vs acquisition purity", fontsize=9, loc="left")
    ax = axs[2]
    kinds = pd.Series([r["kind"] for r in comp]).value_counts()
    cols = {"chemically_coherent": GREEN, "acquisition_confounded": RED,
            "unresolved": GREY, "hierarchical_subfamily": BLUE, "bridge": AMBER,
            "spectroscopically_coherent": TEAL, "mixed": PURPLE}
    ax.barh(range(len(kinds)), kinds.values, color=[cols.get(k, GREY) for k in kinds.index],
            alpha=0.9)
    ax.set_yticks(range(len(kinds)))
    ax.set_yticklabels([k.replace("_", "\n") for k in kinds.index], fontsize=7)
    for i, v in enumerate(kinds.values):
        ax.text(v + 0.1, i, str(v), va="center", fontsize=8, weight="bold")
    ax.set_xlabel("clusters"); ax.invert_yaxis()
    ax.set_title("c · cluster kinds (rule-based, declared before inspection)", fontsize=9,
                 loc="left")
    fig.suptitle("Figure 3 · What is inside each emergent cluster", x=0.03, ha="left",
                 fontsize=11.5, weight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, "F03_cluster_composition")


# ── 4 cluster spectra + CSM maps + band fingerprints ─────────────────────────
def f04(c):
    big = sorted([r for r in c.comp if r["n_molecules"] >= 5],
                 key=lambda r: -r["n_molecules"])[:6]
    fig, axs = plt.subplots(2, 3, figsize=(13.0, 6.2))
    for ax, r in zip(axs.ravel(), big):
        sel = np.isin(c.y, r["members"])
        Xs = c.X[sel]
        mu, sd = Xs.mean(axis=0), Xs.std(axis=0)
        ax.fill_between(c.grid, mu - sd, mu + sd, color=BLUE, alpha=0.18)
        ax.plot(c.grid, mu, color=BLUE, lw=1.1)
        for b in r["dominant_bands"][:6]:
            ax.axvline(b, color=AMBER, ls=":", lw=0.8, alpha=0.8)
        ax.set_title(f"C{r['cluster']} · n={r['n_molecules']} · {r['kind'][:20]}\n"
                     f"{SH(r['dominant_fine_class'])} {r['fine_purity']:.0%} · "
                     f"CSMs {', '.join(r['dominant_csms'][:3])}", fontsize=7.2, loc="left")
        ax.set_xlabel("cm$^{-1}$", fontsize=7); ax.tick_params(labelsize=6.5)
    fig.suptitle("Figure 4 · Mean spectrum ± 1 sd of the six largest emergent clusters "
                 "(amber = the cluster's dominant CSM bands)", x=0.03, ha="left",
                 fontsize=11, weight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, "F04_cluster_spectra")


def f05(c):
    order = np.argsort(c.lab)
    fig, axs = plt.subplots(1, 2, figsize=(12.6, 5.0),
                            gridspec_kw={"width_ratios": [1.6, 1.0]})
    ax = axs[0]
    im = ax.imshow(c.M[order].T, aspect="auto", cmap="magma", interpolation="nearest")
    ax.set_ylabel("frozen CSM"); ax.set_xlabel("molecules, grouped by emergent cluster")
    bounds, prev = [], None
    for k, i in enumerate(order):
        if c.lab[i] != prev:
            bounds.append((k, int(c.lab[i]))); prev = c.lab[i]
    for b, _ in bounds[1:]:
        ax.axvline(b, color="white", lw=0.6, alpha=0.75)
    ax.set_xticks([b for b, _ in bounds])
    ax.set_xticklabels([f"C{l}" for _, l in bounds], fontsize=6)
    fig.colorbar(im, ax=ax, shrink=0.8, label="CSM activation")
    ax.set_title("a · CSM activation map", fontsize=9, loc="left")
    ax = axs[1]
    D = CLU.cosine_distance(c.M)
    im2 = ax.imshow(D[np.ix_(order, order)], cmap="viridis_r")
    for b, _ in bounds[1:]:
        ax.axvline(b, color="white", lw=0.5, alpha=0.6)
        ax.axhline(b, color="white", lw=0.5, alpha=0.6)
    fig.colorbar(im2, ax=ax, shrink=0.8, label="cosine distance")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("b · molecule × molecule distance, cluster-ordered", fontsize=9, loc="left")
    fig.suptitle("Figure 5 · The CSM manifold, ordered by the emergent partition", x=0.03,
                 ha="left", fontsize=11.5, weight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, "F05_activation_heatmap")


# ── 6 embeddings ─────────────────────────────────────────────────────────────
def f06(c):
    keys = [k for k in ("pca", "umap", "diffusion") if k in c.emb]
    fig, axs = plt.subplots(2, len(keys), figsize=(4.4 * len(keys), 8.0), squeeze=False)
    cmap = plt.get_cmap("tab20")
    fine = sorted(set(c.fine_m))
    for j, k in enumerate(keys):
        Y = np.asarray(c.emb[k])[:, :2]
        ax = axs[0, j]
        for i, cl in enumerate(fine):
            m = c.fine_m == cl
            ax.scatter(Y[m, 0], Y[m, 1], s=22, color=cmap(i % 20), alpha=0.85,
                       label=SH(cl) if j == 0 else None)
        ax.set_title(f"{k.upper()} — coloured by CURATED chemistry", fontsize=8.4, loc="left")
        ax.set_xticks([]); ax.set_yticks([])
        ax = axs[1, j]
        for cl in sorted(set(c.lab.tolist())):
            m = c.lab == cl
            ax.scatter(Y[m, 0], Y[m, 1], s=22, alpha=0.85)
            if m.sum() >= 4:
                ax.annotate(f"C{cl}", Y[m].mean(axis=0), fontsize=7.4, weight="bold")
        ax.set_title(f"{k.upper()} — coloured by EMERGENT cluster", fontsize=8.4, loc="left")
        ax.set_xticks([]); ax.set_yticks([])
    axs[0, 0].legend(frameon=False, fontsize=5.4, ncol=2, loc="upper left")
    var = np.asarray(c.emb.get("pca_var", [0, 0]))
    fig.suptitle("Figure 6 · Embeddings for interpretation only — never inference\n"
                 f"PCA components 1–2 explain {100 * var[:2].sum():.1f}% of variance; "
                 "UMAP distances must not be read quantitatively",
                 x=0.03, ha="left", fontsize=11, weight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, "F06_embeddings")


# ── 7 nearest-neighbour / force-directed graph ───────────────────────────────
def f07(c):
    import networkx as nx
    D = CLU.cosine_distance(c.M)
    np.fill_diagonal(D, np.inf)
    G = nx.Graph()
    G.add_nodes_from(range(len(c.M)))
    for i in range(len(c.M)):
        for j in np.argsort(D[i])[:5]:
            G.add_edge(i, int(j), weight=float(np.exp(-D[i, j])))
    pos = nx.spring_layout(G, seed=0, weight="weight", iterations=250)
    fig, axs = plt.subplots(1, 2, figsize=(12.4, 6.0))
    cmap = plt.get_cmap("tab20")
    fine = sorted(set(c.fine_m))
    for ax, colour_by, title in ((axs[0], c.fine_m, "curated chemistry"),
                                 (axs[1], np.array([f"C{v}" for v in c.lab]),
                                  "emergent cluster")):
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.13, width=0.6)
        cats = sorted(set(colour_by.tolist()))
        for i, cat in enumerate(cats):
            m = colour_by == cat
            ax.scatter([pos[k][0] for k in np.where(m)[0]],
                       [pos[k][1] for k in np.where(m)[0]], s=26, color=cmap(i % 20),
                       alpha=0.9, label=SH(cat) if len(cats) <= 20 else None)
        ax.set_title(f"coloured by {title}", fontsize=9, loc="left")
        ax.axis("off")
    axs[0].legend(frameon=False, fontsize=5.6, ncol=2, loc="upper left")
    m = c.s["hierarchy"]["modularity"]
    fig.suptitle("Figure 7 · Force-directed 5-NN graph of the 154 molecules in CSM space\n"
                 f"modularity {m['modularity']:.3f} against a degree-preserving null of "
                 f"{m['null_mean']:.3f} (z = {m['z_score']:.0f}) — the graph is genuinely "
                 f"modular, in {m['n_communities']} communities",
                 x=0.03, ha="left", fontsize=11, weight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    save(fig, "F07_knn_graph")


# ── 8 confounding ────────────────────────────────────────────────────────────
def f08(c):
    fig, axs = plt.subplots(1, 3, figsize=(12.6, 4.2))
    d = c.vp.sort_values("marginal_R2")
    cols = [GREEN if "chemistry" in f else RED for f in d.factor]
    axs[0].barh(range(len(d)), d.marginal_R2, color=cols, alpha=0.9)
    axs[0].set_yticks(range(len(d)))
    axs[0].set_yticklabels([f.replace("_", " ") for f in d.factor], fontsize=7.2)
    for i, (v, p) in enumerate(zip(d.marginal_R2, d.p_value)):
        axs[0].text(v + 0.006, i, f"{v:.3f}  p={p:.3f}", va="center", fontsize=6.8,
                    color=MUTED)
    axs[0].set_xlim(0, d.marginal_R2.max() * 1.45)
    axs[0].set_xlabel("PERMANOVA marginal R²")
    axs[0].set_title("a · what explains the distances?", fontsize=9, loc="left")
    d2 = c.ami.sort_values("AMI")
    cols2 = [GREEN if "chemistry" in f else RED for f in d2.factor]
    axs[1].barh(range(len(d2)), d2.AMI, color=cols2, alpha=0.9)
    axs[1].set_yticks(range(len(d2)))
    axs[1].set_yticklabels([f.replace("_", " ") for f in d2.factor], fontsize=7.2)
    for i, v in enumerate(d2.AMI):
        axs[1].text(v + 0.008, i, f"{v:.3f}", va="center", fontsize=7)
    axs[1].set_xlim(0, d2.AMI.max() * 1.3)
    axs[1].set_xlabel("adjusted mutual information with the partition")
    axs[1].set_title("b · what does the partition track?", fontsize=9, loc="left")
    ax = axs[2]; ax.axis("off")
    cv = c.s["confounding"]
    r2 = {r["factor"]: r["marginal_R2"] for r in cv["variance_partition"]}
    txt = (f"CHEMISTRY DOMINATES: {cv['chemistry_dominates']}\n\n"
           f"fine chemistry R² {r2['fine_chemistry']:.3f}\n"
           f"excitation R²     {r2['excitation']:.3f}\n"
           f"source R²         {r2['source']:.3f}\n\n"
           f"Chemistry explains {r2['fine_chemistry']/r2['excitation']:.1f}× more of the "
           f"pairwise distance structure than excitation and "
           f"{r2['fine_chemistry']/r2['source']:.0f}× more than source library. All factors are individually significant at "
           "p = 0.001, which at n = 154 means detectable, not dominant.\n\n"
           "The caveat that matters: globally chemistry wins, but FOUR of sixteen clusters are "
           "individually acquisition-confounded by the pre-declared rule — their source or "
           "excitation purity exceeds their chemistry purity. A global verdict does not "
           "license a per-cluster claim.")
    yy = 0.98
    for line in txt.split("\n"):
        for w in textwrap.wrap(line, 50) or [""]:
            ax.text(0.0, yy, w, fontsize=7.2, color=INK,
                    family="DejaVu Sans Mono" if "R²" in w else "DejaVu Sans",
                    weight="bold" if "DOMINATES" in w else "normal")
            yy -= 0.052
    fig.suptitle("Figure 8 · Is the geometry chemistry, or the instrument?", x=0.03,
                 ha="left", fontsize=11.5, weight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    save(fig, "F08_confounding")


# ── 9 hierarchy ──────────────────────────────────────────────────────────────
def f09(c):
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import squareform
    fig = plt.figure(figsize=(13.0, 6.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1.0], hspace=0.45, wspace=0.28)
    ax = fig.add_subplot(gs[0, :])
    D = CLU.cosine_distance(c.M)
    Z = linkage(squareform(D, checks=False), method="average")
    cmap = plt.get_cmap("tab20")
    dn = dendrogram(Z, ax=ax, no_labels=True, color_threshold=0.72 * Z[:, 2].max(),
                    above_threshold_color=GREY)
    ax.set_ylabel("cosine distance")
    h = c.s["hierarchy"]
    ax.set_title(f"a · average-linkage dendrogram — cophenetic correlation "
                 f"{h['cophenetic']['best_correlation']:.3f} (tree-like: "
                 f"{h['cophenetic']['tree_like']})", fontsize=9, loc="left")
    ax = fig.add_subplot(gs[1, 0])
    d = CLU.cosine_distance(c.M)[np.triu_indices(len(c.M), 1)]
    ax.hist(d, bins=60, color=BLUE, alpha=0.85)
    ax.set_xlabel("pairwise cosine distance"); ax.set_ylabel("pairs")
    ax.set_title(f"b · valley depth {h['gap']['valley_depth']:.2f} → bimodal "
                 f"{h['gap']['bimodal']}", fontsize=8.4, loc="left")
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(c.branch.K, c.branch.silhouette, "o-", color=GREEN, lw=1.4)
    ax2 = ax.twinx()
    ax2.bar(c.branch.K, c.branch.split_gain, color=AMBER, alpha=0.45, width=0.6)
    ax.set_xlabel("K"); ax.set_ylabel("silhouette", color=GREEN)
    ax2.set_ylabel("gain from the split", color=AMBER, fontsize=7.4)
    ax.set_title("c · every split adds a little; none adds a lot", fontsize=8.4, loc="left")
    ax = fig.add_subplot(gs[1, 2]); ax.axis("off")
    idim = h["intrinsic_dimension"]
    txt = (f"SHAPE VERDICT: {h['shape'].upper()}\n\n"
           f"modularity z = {h['modularity']['z_score']:.0f} (p "
           f"{h['modularity']['p_value']:.3f})\n"
           f"cophenetic  = {h['cophenetic']['best_correlation']:.3f}\n"
           f"valley depth= {h['gap']['valley_depth']:.2f}\n"
           f"bridging    = {h['continuity']['fraction_bridging']:.0%} of molecules\n\n"
           f"intrinsic dimension: Levina–Bickel {idim['levina_bickel_mle']:.1f} vs "
           f"correlation {idim['correlation_dimension']:.1f} of ambient 49 — the two "
           f"estimators DISAGREE, so neither is quoted as the answer.\n\n"
           "The space is modular and tree-like at the same time as having no preferred cut "
           "height. That is not a contradiction: it is a hierarchy with continuous branch "
           "lengths.")
    yy = 0.98
    for line in txt.split("\n"):
        for w in textwrap.wrap(line, 46) or [""]:
            ax.text(0.0, yy, w, fontsize=6.9, color=INK,
                    weight="bold" if "SHAPE VERDICT" in w else "normal")
            yy -= 0.056
    fig.suptitle("Figure 9 · Is Raman molecule space tree-like, graph-like, continuous or "
                 "modular?", x=0.03, ha="left", fontsize=11.5, weight="bold", color=INK)
    save(fig, "F09_hierarchy")


# ── 10 coordinates ───────────────────────────────────────────────────────────
def f10(c):
    fig, axs = plt.subplots(1, 3, figsize=(12.6, 4.2))
    d = c.csw[c.csw.usable]
    ax = axs[0]
    for kern, g in d.groupby("kernel"):
        g = g.sort_values("temperature")
        ax.plot(g.temperature, g.neighbour_preservation_k10, "o-", ms=3.6, lw=1.2, label=kern)
    ax.set_xscale("log"); ax.set_xlabel("temperature")
    ax.set_ylabel("k-NN preservation vs 49-d CSM space")
    ax.legend(frameon=False, fontsize=6.4)
    ax.set_title("a · kernel × temperature sweep (label-free)", fontsize=9, loc="left")
    ax = axs[1]
    ax.scatter(d.mean_entropy, d.neighbour_preservation_k10, s=34,
               c=[{"softmax_cosine": BLUE, "gaussian": GREEN, "cosine_power": PURPLE,
                   "inverse_distance": AMBER, "wasserstein": RED}.get(k, GREY)
                  for k in d.kernel], alpha=0.85)
    cp = c.s["coordinates"]
    ax.scatter([cp["mean_entropy"]], [cp["neighbour_preservation_k10"]], s=170,
               facecolors="none", edgecolors=RED, lw=2)
    ax.axvspan(0, 0.10, color=RED, alpha=0.08); ax.axvspan(0.90, 1.0, color=RED, alpha=0.08)
    ax.set_xlabel("mean coordinate entropy"); ax.set_ylabel("k-NN preservation")
    ax.set_title(f"b · selected: {cp['kernel']} T={cp['temperature']:g}", fontsize=9,
                 loc="left")
    ax = axs[2]
    keys = [("neighbour_preservation_k10", "neighbour_preservation_hard_ids", "k-NN preservation"),
            ("effective_rank", "effective_rank_hard_ids", "effective rank ÷ 16")]
    x = np.arange(2)
    cont = [cp["neighbour_preservation_k10"], cp["effective_rank"] / 16]
    hard = [cp["neighbour_preservation_hard_ids"], cp["effective_rank_hard_ids"] / 16]
    ax.bar(x - 0.2, cont, 0.38, color=PURPLE, alpha=0.9, label="continuous coordinates")
    ax.bar(x + 0.2, hard, 0.38, color=GREY, alpha=0.9, label="hard cluster ids")
    for i, (a, b) in enumerate(zip(cont, hard)):
        ax.text(i - 0.2, a + 0.012, f"{a:.2f}", ha="center", fontsize=7.6, weight="bold")
        ax.text(i + 0.2, b + 0.012, f"{b:.2f}", ha="center", fontsize=7.6)
    ax.set_xticks(x); ax.set_xticklabels([k[2] for k in keys], fontsize=7.6)
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("c · do continuous coordinates beat hard ids?", fontsize=9, loc="left")
    fig.suptitle("Figure 10 · Continuous Spectral Coordinates — construction and properties",
                 x=0.03, ha="left", fontsize=11.5, weight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    save(fig, "F10_coordinates")


# ── 11 coordinate robustness ─────────────────────────────────────────────────
def f11(c):
    kinds = list(dict.fromkeys(c.crob.perturbation))
    fig, axs = plt.subplots(2, 3, figsize=(12.2, 5.4))
    for ax, k in zip(axs.ravel(), kinds):
        d = c.crob[c.crob.perturbation == k].sort_values("level")
        ax.plot(range(len(d)), d.coordinate_cosine, "o-", color=PURPLE, lw=1.4, ms=4,
                label="coordinate cosine")
        ax.plot(range(len(d)), d.argmax_stability, "s--", color=GREY, lw=1.1, ms=3.4,
                label="argmax stability")
        ax.set_xticks(range(len(d))); ax.set_xticklabels([f"{v:g}" for v in d.level],
                                                         fontsize=6.4)
        ax.set_ylim(0, 1.05); ax.set_title(k.replace("_", " "), fontsize=8.4, loc="left")
        ax.tick_params(labelsize=6.8)
    axs[0, 0].legend(frameon=False, fontsize=6.6)
    fig.suptitle(f"Figure 11 · Coordinate robustness under Raman perturbation — mean cosine "
                 f"{c.crob.coordinate_cosine.mean():.3f}", x=0.03, ha="left", fontsize=11,
                 weight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, "F11_coordinate_robustness")


# ── 12 retrieval ─────────────────────────────────────────────────────────────
def f12(c):
    fig, axs = plt.subplots(1, 3, figsize=(12.6, 4.2))
    d = c.arms.set_index("arm")
    order = ["A_csm_only", "B_csm_plus_coordinate_prior", "C_coordinates_only"]
    cols = [GREEN, PURPLE, GREY]
    ax = axs[0]
    x = np.arange(3)
    for k, (m, lab) in enumerate((("molecule_top1", "top-1"), ("molecule_top3", "top-3"),
                                  ("molecule_top5", "top-5"))):
        ax.bar(x + (k - 1) * 0.27, [d.loc[o, m] for o in order], 0.25,
               alpha=0.55 + 0.2 * k, color=cols, label=lab)
    ax.set_xticks(x); ax.set_xticklabels(["A\nCSM only", "B\nCSM+coord", "C\ncoord only"],
                                         fontsize=7.4)
    ax.set_ylim(0, 1); ax.set_ylabel("molecule retrieval (Split A)")
    ax.legend(frameon=False, fontsize=7, ncol=3)
    ax.set_title("a · molecule identity", fontsize=9, loc="left")
    ax = axs[1]
    for k, (m, lab) in enumerate((("chem_top1", "top-1"), ("chem_top3", "top-3"),
                                  ("chem_macro_f1", "macro-F1"))):
        ax.bar(x + (k - 1) * 0.27, [d.loc[o, m] for o in order], 0.25,
               alpha=0.55 + 0.2 * k, color=cols, label=lab)
    ax.set_xticks(x); ax.set_xticklabels(["A", "B", "C"], fontsize=8)
    ax.set_ylim(0, 1.05); ax.set_ylabel("chemistry class (Split B, unseen molecule)")
    ax.legend(frameon=False, fontsize=7, ncol=3)
    ax.set_title("b · chemistry class", fontsize=9, loc="left")
    ax = axs[2]
    ax.plot(c.wsw.coordinate_weight, c.wsw.chem_top1, "o-", color=PURPLE, lw=1.6, ms=5)
    ax.axhline(float(d.loc["A_csm_only", "chem_top1"]), color=GREEN, ls="--", lw=1.2)
    ax.text(0.55, float(d.loc["A_csm_only", "chem_top1"]) + 0.004, "CSM alone", fontsize=7,
            color=GREEN)
    ax.set_xlabel("weight on the coordinate similarity")
    ax.set_ylabel("chemistry top-1")
    ax.set_title("c · fusion-weight sweep", fontsize=9, loc="left")
    sg = c.sig
    fig.suptitle("Figure 12 · Retrieval benchmark — molecule-grouped, clustering refitted "
                 "inside every training fold\n"
                 f"A→B molecule Δ{sg['molecule']['delta']:+.3f} "
                 f"CI[{sg['molecule']['ci95'][0]:+.3f},{sg['molecule']['ci95'][1]:+.3f}] "
                 f"McNemar p={sg['molecule']['p_value']:.3f}  ·  chemistry "
                 f"Δ{sg['chemistry']['delta']:+.3f} p={sg['chemistry']['p_value']:.3f}  "
                 "— NEITHER IS SIGNIFICANT",
                 x=0.03, ha="left", fontsize=10.5, weight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    save(fig, "F12_retrieval")


# ── 13 agreement ─────────────────────────────────────────────────────────────
def f13(c):
    fig, axs = plt.subplots(1, 3, figsize=(12.8, 4.6),
                            gridspec_kw={"width_ratios": [1.3, 1.0, 1.0]})
    ax = axs[0]
    fine = sorted(set(c.fine_m))
    cl = sorted(set(c.lab.tolist()))
    M = np.zeros((len(fine), len(cl)))
    for i, f in enumerate(fine):
        for j, k in enumerate(cl):
            M[i, j] = int(((c.fine_m == f) & (c.lab == k)).sum())
    im = ax.imshow(M / (M.sum(axis=1, keepdims=True) + 1e-12), cmap="Purples", vmin=0, vmax=1)
    ax.set_yticks(range(len(fine))); ax.set_yticklabels([SH(f) for f in fine], fontsize=6.6)
    ax.set_xticks(range(len(cl))); ax.set_xticklabels([f"C{k}" for k in cl], fontsize=6.4)
    ax.set_xlabel("emergent cluster"); ax.set_ylabel("curated chemistry class")
    fig.colorbar(im, ax=ax, shrink=0.8, label="row share")
    ax.set_title("a · where each curated class went", fontsize=9, loc="left")
    ax = axs[1]
    ag = c.s["agreement"]
    ks = ["ARI", "AMI", "NMI", "homogeneity", "completeness"]
    ax.barh(range(len(ks)), [ag[k] for k in ks], color=PURPLE, alpha=0.9)
    ax.set_yticks(range(len(ks))); ax.set_yticklabels(ks, fontsize=8)
    for i, k in enumerate(ks):
        ax.text(ag[k] + 0.012, i, f"{ag[k]:.3f}", va="center", fontsize=8, weight="bold")
    ax.set_xlim(0, 1.05); ax.invert_yaxis()
    ax.set_title("b · agreement with the curated ontology", fontsize=9, loc="left")
    ax = axs[2]; ax.axis("off")
    txt = ("WHY THEY DISAGREE — explained, not scored\n\n"
           f"AMI {ag['AMI']:.3f}, completeness {ag['completeness']:.3f} > homogeneity "
           f"{ag['homogeneity']:.3f}.\n\n"
           "Completeness above homogeneity has a specific meaning: curated classes tend to stay "
           "TOGETHER, but emergent clusters MERGE several of them. The geometry is coarser than "
           "the ontology in places and finer in others.\n\n"
           "peptide_protein splits across 4 clusters; mono/oligosaccharide and sulfur/thiol "
           "across 3. Proteins split by acquisition and by size; saccharides split into a tight "
           "sugar core and a scattered tail.\n\n"
           "Neither partition is wrong. They answer different questions: the ontology asks what "
           "a molecule IS, the geometry asks what its spectrum RESEMBLES.")
    yy = 0.98
    for line in txt.split("\n"):
        for w in textwrap.wrap(line, 48) or [""]:
            ax.text(0.0, yy, w, fontsize=7.0, color=INK,
                    weight="bold" if "WHY THEY" in w else "normal")
            yy -= 0.053
    fig.suptitle("Figure 13 · Emergent geometry versus the curated 16-class ontology",
                 x=0.03, ha="left", fontsize=11.5, weight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    save(fig, "F13_agreement")


# ── 14 decision ──────────────────────────────────────────────────────────────
def f14(c):
    fig = plt.figure(figsize=(12.4, 7.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], hspace=0.32, wspace=0.20)
    ax = fig.add_subplot(gs[0, 0])
    crit = c.s["section9_criteria"]
    names = list(crit)
    ok = [crit[k]["pass"] for k in names]
    ax.barh(range(len(names)), [1] * len(names),
            color=[GREEN if o else RED for o in ok], alpha=0.85)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n.replace("_", " ") for n in names], fontsize=7.6)
    for i, k in enumerate(names):
        ax.text(0.02, i, ("PASS  " if crit[k]["pass"] else "FAIL  ") + crit[k]["evidence"][:58],
                va="center", fontsize=6.2, color="white" if ok[i] else "white")
    ax.set_xticks([]); ax.invert_yaxis()
    ax.set_title("a · the seven Section 9 criteria", fontsize=9, loc="left")
    ax = fig.add_subplot(gs[0, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    opts = [("A", "CSM → Chemistry Evidence → BSV2", True),
            ("B", "CSM → Coordinates → Chemistry Evidence → BSV2", False),
            ("C", "CSM → {Chemistry Evidence ∥ Coordinates} → fusion → BSV2", False),
            ("D", "another architecture", False)]
    for i, (k, t, sel) in enumerate(opts):
        box(ax, 0.02, 0.76 - i * 0.20, 0.96, 0.16,
            f"Option {k} — {t}" + ("     ★ RECOMMENDED" if sel else ""),
            "#ecfdf5" if sel else "#f8fafc", GREEN if sel else "#cbd5e1", 8.0,
            "bold" if sel else "normal")
    ax.text(0.02, 0.06, "Recommended because the coordinates are reproducible, robust and\n"
            "chemically meaningful — but do NOT significantly improve retrieval.",
            fontsize=7.6, color=INK, style="italic")
    ax.set_title("b · architecture recommendation", fontsize=9, loc="left")
    ax = fig.add_subplot(gs[1, :]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    g = c.gates
    ncol = 5
    nrow = int(np.ceil(len(g) / ncol))
    h = min(0.22, (0.92 - 0.10) / nrow - 0.030)
    for i, (_, r) in enumerate(g.iterrows()):
        x = 0.012 + (i % ncol) * 0.197
        yy = 0.86 - (i // ncol) * (h + 0.030)
        okg = r.status == "PASS"
        box(ax, x, yy, 0.185, h, "\n".join(textwrap.wrap(r.gate, 30)),
            "#ecfdf5" if okg else "#fef2f2", GREEN if okg else RED, 6.0)
    ax.text(0.012, 0.02, f"{int((g.status == 'PASS').sum())} of {len(g)} gates pass · "
            f"canonical partition {c.state['canonical_partition']} (a reporting convention) · "
            f"audit only, no architecture changed, Phase 07 not begun",
            fontsize=7.6, color=MUTED)
    fig.suptitle("Figure 14 · Decision gate and architecture recommendation", x=0.03,
                 ha="left", fontsize=11.5, weight="bold", color=INK)
    save(fig, "F14_decision")


def main():
    c = C()
    print("[figures]")
    for fn in (f01, f02, f03, f04, f05, f06, f07, f08, f09, f10, f11, f12, f13, f14):
        fn(c)
    assert not list(F.glob("*.svg")), "PNG only in this phase"
    print(f"[figures] {len(list(F.glob('*.png')))} PNG written to {F}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
