#!/usr/bin/env python3
"""GAIRA V7 — Phase 02.5 figures (SVG vector + PNG). Deterministic; seeds fixed."""
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
from scipy.cluster.hierarchy import dendrogram

HERE = Path(__file__).resolve().parent
PH = HERE.parent
REPO = PH.parents[2]
sys.path.insert(0, str(REPO / "src"))
from gaira.v7.geometry import structure as STR      # noqa: E402

T, F, A, V = PH / "tables", PH / "figures", PH / "artifacts", PH / "validation"
INT = PH / "interactive"
INK, MUTED, LINE = "#1a1a1a", "#6b7280", "#9ca3af"
BLUE, GREEN, AMBER, RED, GREY = "#2563eb", "#15803d", "#b45309", "#b91c1c", "#4b5563"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
                     "figure.facecolor": "white", "savefig.facecolor": "white",
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.18,
                     "svg.fonttype": "none"})


def save(fig, name):
    """PNG only. The vector copy lived alongside every figure until Phase 02.5; it is dropped
    from here on because each phase now ships a PDF report that carries the figures, which is
    the artefact anyone actually circulates."""
    F.mkdir(parents=True, exist_ok=True)
    fig.savefig(F / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"  {name}")


class Ctx:
    def __init__(self):
        g = np.load(A / "geometry_v1.npz", allow_pickle=True)
        self.ids = [str(s) for s in g["motif_ids"]]
        self.D = g["D_primary_metric"]          # neighbourhoods are computed on this
        self.Dfused = g["D_primary_geometry"]
        self.Ds = {k[2:]: g[k] for k in g.files
                   if k.startswith("D_") and not k.startswith("D_primary")}
        self.fused = {k[6:]: g[k] for k in g.files if k.startswith("fused_")}
        e = np.load(A / "embeddings_v1.npz", allow_pickle=True)
        self.umap, self.diff, self.spec = e["umap"], e["diffusion"], e["spectral"]
        self.eig = e["eigenvalues"]
        p = np.load(A / "pca_v1.npz", allow_pickle=True)
        self.pca_scores = p["spectral_profile_scores"]
        self.pca_load = p["spectral_profile_loadings"]
        self.grid = p["grid"]
        reg = pd.read_csv(REPO / "results/v7_rebuild/phase01/artifacts/lsm_registry_v1.csv")
        reg = reg.set_index("motif_id").loc[self.ids]
        self.cls = reg.chemical_class.tolist()
        self.H = np.asarray(np.load(
            REPO / "results/v7_rebuild/phase01/artifacts/lsm_dictionary_v1.npz",
            allow_pickle=True)["H"], float)
        self.fams = sorted(set(self.cls))
        cm = plt.get_cmap("tab20")
        self.col = {c: cm(i % 20) for i, c in enumerate(self.fams)}
        self.roles = pd.read_csv(T / "graph_roles_v1.csv")
        self.priors = json.loads((A / "phase03_geometry_priors.json").read_text())
        self.regions = pd.read_csv(T / "geometry_regions_v1.csv")
        self.prop = pd.read_csv(T / "rejected_proposal_geometry_v1.csv")
        self.cards = pd.read_csv(T / "nearest_neighbour_cards_v1.csv")
        man = json.loads((A / "phase_02_5_manifest_v1.json").read_text())
        self.primary_metric = man["primary_spectral_metric"]
        self.primary_geom = man["primary_geometry"]
        canon = pd.read_csv(REPO / "results/v7_rebuild/phase00/tables/canonical_analytes_v1.csv")
        so = {r.canonical_id: str(r.sources).split(";") for r in canon.itertuples()}
        eo = {r.canonical_id: str(r.excitations).split(";") for r in canon.itertuples()}
        an = [str(a).split(";") if pd.notna(a) else [] for a in reg.analytes]

        def dom(alist, tbl):
            c = {}
            for a in alist:
                for v in tbl.get(a, []):
                    c[v] = c.get(v, 0) + 1
            return sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[0][0] if c else "?"
        self.src = [dom(a, so) for a in an]
        self.exc = [dom(a, eo) for a in an]
        self.nmol = [len([x for x in a if x]) for a in an]

    def scatter(self, ax, E, colours, title, legend=None, sizes=None):
        for i in range(len(self.ids)):
            ax.scatter(E[i, 0], E[i, 1], s=(sizes[i] if sizes is not None else 42),
                       color=colours[i], edgecolor="white", linewidth=0.5, zorder=3)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(LINE)
        ax.set_title(title, fontsize=9, loc="left", color=INK)
        if legend:
            ax.legend(handles=legend, loc="center left", bbox_to_anchor=(1.01, 0.5),
                      frameon=False, fontsize=6.5)


def f01_schematic(c):
    fig, ax = plt.subplots(figsize=(10.0, 3.1))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3); ax.set_axis_off()
    steps = [("50 frozen LSMs\n49 frozen CSMs", GREY), ("7 representations", BLUE),
             ("10 metrics\nbenchmarked", BLUE), ("6 null\ngeometries", RED),
             ("linear · nonlinear\ngraph · hierarchy", GREEN),
             ("discrete vs\ncontinuous", GREEN), ("neighbourhoods\n+ tiers", AMBER),
             ("Phase 03 priors\n(provisional)", AMBER)]
    for k, (lab, col) in enumerate(steps):
        x = 0.1 + k * 1.235
        ax.add_patch(FancyBboxPatch((x, 1.15), 1.05, 0.95, boxstyle="round,pad=0.05",
                                    fc="white", ec=col, lw=1.3))
        ax.text(x + 0.525, 1.62, lab, ha="center", va="center", fontsize=6.8, color=INK)
        if k:
            ax.add_patch(FancyArrowPatch((x - 0.17, 1.62), (x - 0.02, 1.62),
                                         arrowstyle="-|>", mutation_scale=8,
                                         color=MUTED, lw=0.9))
    ax.text(0.1, 0.62, "chemistry and source labels are excluded from every representation "
                       "and every distance — revealed only at the neighbourhood step",
            fontsize=7.5, color=RED)
    ax.text(0.1, 0.25, "no dictionary is refitted · no themes are created", fontsize=7.5,
            color=MUTED)
    ax.set_title("Phase 02.5 — latent geometry of spectral motif space", fontsize=10.5,
                 loc="left", color=INK)
    save(fig, "fig01_pipeline_schematic")


def f02_metric_comparison(c):
    m = pd.read_csv(T / "metric_comparison_v1.csv")
    m = m[m.probeable].copy()
    cols = ["amplitude_invariance", "peak_shift_tolerance", "background_separation",
            "knn_chemical_coherence", "null_separation", "bootstrap_stability"]
    Z = m[cols].copy()
    for col in cols:
        v = Z[col].astype(float)
        Z[col] = (v - v.min()) / (v.max() - v.min() + 1e-12)
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    im = ax.imshow(Z.values, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([x.replace("_", "\n") for x in cols], fontsize=7)
    ax.set_yticks(range(len(m))); ax.set_yticklabels(m.metric, fontsize=8)
    for i in range(len(m)):
        for j, col in enumerate(cols):
            ax.text(j, i, f"{m.iloc[i][col]:.3g}", ha="center", va="center", fontsize=6.3,
                    color="white" if Z.values[i, j] > 0.6 else INK)
        if m.iloc[i].metric == c.primary_metric:
            ax.add_patch(plt.Rectangle((-0.5, i - 0.5), len(cols), 1, fill=False,
                                       ec=RED, lw=2.0))
    fig.colorbar(im, ax=ax, shrink=0.8, label="rank within column")
    ax.set_title(f"Metric benchmark — probes are scale-free (each divided by that metric's own\n"
                 f"median observed distance). Selected: {c.primary_metric}, the only metric "
                 f"that knows\nthe wavenumber axis is ordered.",
                 fontsize=9, loc="left", color=INK)
    save(fig, "fig02_metric_comparison")


def f03_null_vs_observed(c):
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.3), gridspec_kw={"wspace": 0.28})
    off = c.D[~np.eye(len(c.ids), dtype=bool)]
    ax = axes[0]
    ax.hist(off, bins=45, color=BLUE, alpha=0.75, density=True)
    ax.axvline(np.median(off), color=RED, lw=1.2)
    ax.set_xlabel(f"{c.primary_metric} distance between motifs")
    ax.set_ylabel("density")
    ax.set_title("Observed pairwise distances — unimodal, no density valley",
                 fontsize=9, loc="left", color=INK)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax = axes[1]
    cont = json.loads((A / "continuum_analysis_v1.json").read_text())
    lid = np.array(list(cont["local_intrinsic_dimension"]["per_motif"].values()), float)
    ax.hist(lid[np.isfinite(lid)], bins=22, color=GREEN, alpha=0.8)
    ax.axvline(np.nanmean(lid), color=RED, lw=1.2)
    ax.text(np.nanmean(lid), ax.get_ylim()[1] * 0.9, f"  mean {np.nanmean(lid):.2f}",
            color=RED, fontsize=7.5)
    ax.set_xlabel("local intrinsic dimension (Levina–Bickel, k = 10)")
    ax.set_ylabel("motifs")
    ax.set_title("Low-dimensional locally — consistent with a continuum, not islands",
                 fontsize=9, loc="left", color=INK)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    save(fig, "fig03_null_vs_observed_distances")


def f04_pca(c):
    ev = pd.read_csv(T / "pca_components_v1.csv")
    ev = ev[ev.representation == "spectral_profile"]
    fig = plt.figure(figsize=(10.0, 3.4))
    gs = fig.add_gridspec(1, 3, wspace=0.32, width_ratios=[1, 1.1, 1.1])
    ax = fig.add_subplot(gs[0])
    ax.bar(ev.pc, ev.explained_variance_ratio, color=BLUE, alpha=0.85)
    ax2 = ax.twinx()
    ax2.plot(ev.pc, ev.loading_stability, "o-", color=RED, ms=4)
    ax2.set_ylabel("loading stability", color=RED); ax2.set_ylim(0, 1)
    ax.set_xlabel("principal component"); ax.set_ylabel("explained variance")
    ax.set_title("Only PC1 is reproducible", fontsize=9, loc="left", color=INK)
    for s in ("top",): ax.spines[s].set_visible(False)
    leg = [Line2D([], [], marker="o", ls="", color=c.col[f], label=f, markersize=5)
           for f in c.fams]
    ax = fig.add_subplot(gs[1])
    c.scatter(ax, c.pca_scores[:, :2], [c.col[x] for x in c.cls],
              "PC1 vs PC2 — chemistry revealed after fitting")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax = fig.add_subplot(gs[2])
    c.scatter(ax, c.pca_scores[:, 1:3], [c.col[x] for x in c.cls], "PC2 vs PC3", legend=leg)
    save(fig, "fig04_pca_scores")


def f05_pca_loadings(c):
    fig, axes = plt.subplots(3, 1, figsize=(9.0, 5.0), sharex=True,
                             gridspec_kw={"hspace": 0.35})
    ev = pd.read_csv(T / "pca_components_v1.csv")
    ev = ev[ev.representation == "spectral_profile"]
    for j, ax in enumerate(axes):
        L = c.pca_load[j]
        ax.fill_between(c.grid, 0, L, where=L >= 0, color=BLUE, alpha=0.7)
        ax.fill_between(c.grid, 0, L, where=L < 0, color=RED, alpha=0.7)
        top = ev.iloc[j].top_bands_cm1.split(";")[:3]
        ax.set_title(f"PC{j + 1} — {ev.iloc[j].explained_variance_ratio:.1%} of variance, "
                     f"stability {ev.iloc[j].loading_stability:.2f}; driven by "
                     f"{', '.join(top)} cm⁻¹",
                     fontsize=8.5, loc="left", color=INK)
        ax.axhline(0, color=LINE, lw=0.6)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
    axes[-1].set_xlabel("Raman shift (cm⁻¹)")
    fig.suptitle("PCA loadings — what each axis is made of, spectroscopically",
                 fontsize=10, x=0.005, ha="left", y=1.01, color=INK)
    save(fig, "fig05_pca_loadings")


def f06_umap_grid(c):
    u = pd.read_csv(T / "umap_stability_sweep_v1.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.2), gridspec_kw={"wspace": 0.3})
    piv = u.pivot(index="n_neighbors", columns="min_dist", values="knn_jaccard_vs_highdim")
    im = axes[0].imshow(piv.values, cmap="YlGnBu", vmin=0.4, vmax=0.8)
    axes[0].set_xticks(range(piv.shape[1])); axes[0].set_xticklabels(piv.columns)
    axes[0].set_yticks(range(piv.shape[0])); axes[0].set_yticklabels(piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            axes[0].text(j, i, f"{piv.values[i, j]:.2f}", ha="center", va="center",
                         fontsize=7.5, color=INK)
    axes[0].set_xlabel("min_dist"); axes[0].set_ylabel("n_neighbors")
    axes[0].set_title("k-NN Jaccard vs the high-dimensional geometry", fontsize=9,
                      loc="left", color=INK)
    fig.colorbar(im, ax=axes[0], shrink=0.8)
    piv2 = u.pivot(index="n_neighbors", columns="min_dist", values="knn_jaccard_across_seeds")
    im2 = axes[1].imshow(piv2.values, cmap="YlOrRd", vmin=0.5, vmax=0.85)
    axes[1].set_xticks(range(piv2.shape[1])); axes[1].set_xticklabels(piv2.columns)
    axes[1].set_yticks(range(piv2.shape[0])); axes[1].set_yticklabels(piv2.index)
    for i in range(piv2.shape[0]):
        for j in range(piv2.shape[1]):
            axes[1].text(j, i, f"{piv2.values[i, j]:.2f}", ha="center", va="center",
                         fontsize=7.5, color=INK)
    axes[1].set_xlabel("min_dist"); axes[1].set_ylabel("n_neighbors")
    axes[1].set_title("k-NN Jaccard across three seeds", fontsize=9, loc="left", color=INK)
    fig.colorbar(im2, ax=axes[1], shrink=0.8)
    fig.suptitle("UMAP stability sweep — one layout is never evidence; the question is "
                 "whether neighbours survive the parameters",
                 fontsize=9.5, x=0.005, ha="left", y=1.06, color=INK)
    save(fig, "fig06_umap_stability_grid")


def f07_diffusion(c):
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.2), gridspec_kw={"wspace": 0.3})
    leg = [Line2D([], [], marker="o", ls="", color=c.col[f], label=f, markersize=5)
           for f in c.fams]
    c.scatter(axes[0], c.diff[:, :2], [c.col[x] for x in c.cls], "DC1 vs DC2")
    axes[0].set_xlabel("diffusion coordinate 1"); axes[0].set_ylabel("DC2")
    c.scatter(axes[1], c.diff[:, 1:3], [c.col[x] for x in c.cls], "DC2 vs DC3", legend=leg)
    ax = axes[2]
    ax.plot(range(1, len(c.eig) + 1), c.eig, "o-", color=BLUE, ms=4)
    ax.set_xlabel("index"); ax.set_ylabel("eigenvalue")
    ax.set_title("Diffusion spectrum — no sharp gap", fontsize=9, loc="left", color=INK)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    save(fig, "fig07_diffusion_map")


def f08_dendrogram(c):
    Z = STR.hierarchical(c.D, "average")
    fig, ax = plt.subplots(figsize=(10.4, 3.8))
    dendrogram(Z, labels=c.ids, ax=ax, color_threshold=0.7 * Z[:, 2].max(),
               leaf_font_size=5.6)
    for lbl in ax.get_xmajorticklabels():
        lbl.set_color(c.col[c.cls[c.ids.index(lbl.get_text())]])
    ax.set_ylabel(f"{c.primary_metric} distance")
    ax.set_title("Hierarchical clustering (average linkage) — leaf colour is chemistry class,\n"
                 "attached after the tree was built", fontsize=9.5, loc="left", color=INK)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    save(fig, "fig08_dendrogram")


def _heat(c, M, title, name, cmap="viridis"):
    Z = STR.hierarchical(c.D, "average")
    order = dendrogram(Z, no_plot=True)["leaves"]
    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    im = ax.imshow(M[np.ix_(order, order)], cmap=cmap)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([c.ids[i] for i in order], rotation=90, fontsize=4.6)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([c.ids[i] for i in order], fontsize=4.6)
    for lbl, i in zip(ax.get_ymajorticklabels(), order):
        lbl.set_color(c.col[c.cls[i]])
    for lbl, i in zip(ax.get_xmajorticklabels(), order):
        lbl.set_color(c.col[c.cls[i]])
    fig.colorbar(im, ax=ax, shrink=0.7)
    ax.set_title(title, fontsize=9, loc="left", color=INK)
    save(fig, name)


def f09_cosine_heatmap(c):
    _heat(c, 1.0 - c.Ds["spectral_cosine"], "Spectral cosine similarity, ordered by the "
          "primary geometry's dendrogram", "fig09_cosine_heatmap")


def f10_multiview_heatmap(c):
    _heat(c, 1.0 - c.fused[c.primary_geom],
          f"Primary multi-view geometry ({c.primary_geom}) similarity",
          "fig10_multiview_heatmap", cmap="magma")


def _graph_fig(c, G, pos, title, name, node_col=None, sizes=None):
    fig, ax = plt.subplots(figsize=(9.6, 6.6))
    for u, v, d in G.edges(data=True):
        ax.plot(*zip(pos[u], pos[v]), lw=0.6, color=LINE, alpha=0.45, zorder=1)
    for i, m in enumerate(c.ids):
        ax.scatter(*pos[m], s=(sizes[i] if sizes is not None else 60),
                   color=(node_col[i] if node_col else c.col[c.cls[i]]),
                   edgecolor="white", linewidth=0.6, zorder=3)
    for r in c.roles.itertuples():
        if r.is_bridge or r.is_isolated:
            ax.annotate(r.motif, pos[r.motif], fontsize=5.6, ha="center", va="bottom",
                        xytext=(0, 7), textcoords="offset points",
                        color=RED if r.is_bridge else GREY)
    ax.set_axis_off()
    ax.set_title(title, fontsize=9.5, loc="left", color=INK)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=c.col[f], label=f,
                              markersize=5) for f in c.fams],
              loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=6.5)
    save(fig, name)


def f11_knn_graph(c):
    G = STR.knn_graph(c.D, c.ids, k=5)
    pos = nx.spring_layout(G, seed=5, k=0.55, iterations=500)
    _graph_fig(c, G, pos, "k-nearest-neighbour graph (k = 5). Labelled: bridge motifs (red) "
               "and isolated motifs (grey).", "fig11_knn_graph")


def f12_force_directed(c):
    G = STR.knn_graph(c.D, c.ids, k=5)
    pos = nx.kamada_kawai_layout(G)
    _graph_fig(c, G, pos, "Force-directed motif graph (Kamada–Kawai)", "fig12_force_directed")


def f13_mst(c):
    G = STR.minimum_spanning_tree(c.D, c.ids)
    pos = nx.kamada_kawai_layout(G)
    _graph_fig(c, G, pos, "Minimum spanning tree — the backbone of motif space",
               "fig13_minimum_spanning_tree")


def f14_community_stability(c):
    s = pd.read_csv(T / "cluster_sweep_v1.csv")
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.0), gridspec_kw={"wspace": 0.33})
    for ax, metric, lab in zip(axes, ["silhouette", "davies_bouldin", "calinski_harabasz"],
                               ["silhouette (↑)", "Davies–Bouldin (↓)",
                                "Calinski–Harabasz (↑)"]):
        for meth, sub in s.groupby("method"):
            ax.plot(sub.k, sub[metric], "o-", ms=3, label=meth, lw=1.0)
        ax.set_xlabel("K"); ax.set_ylabel(lab)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    axes[0].legend(frameon=False, fontsize=6.5)
    fig.suptitle("Cluster quality across K and linkage — silhouette and Calinski–Harabasz "
                 "both pick K = 2 (a hydrophobic/polar split, bootstrap ARI 0.879);\n"
                 "Davies–Bouldin runs to the largest K, which is what it does when there is no "
                 "further structure. One defensible cut, at the top.",
                 fontsize=9.5, x=0.005, ha="left", y=1.09, color=INK)
    save(fig, "fig14_community_stability")


def f15_neighbour_cards(c):
    picks = (["acylglycerol.m01"] + list(c.roles[c.roles.is_bridge].motif[:2])
             + list(c.roles[c.roles.is_isolated].motif[:2])
             + list(c.roles[c.roles.is_hub].motif[:1]))
    picks = list(dict.fromkeys(picks))[:6]
    fig, axes = plt.subplots(2, 3, figsize=(10.4, 5.2),
                             gridspec_kw={"hspace": 0.75, "wspace": 0.25})
    for ax, m in zip(axes.ravel(), picks):
        sub = c.cards[c.cards.motif == m].sort_values("rank")
        y = np.arange(len(sub))[::-1]
        cols = [GREEN if t == "exact_equivalence" else
                BLUE if t == "shared_substructure" else
                AMBER if t == "broad_superfamily" else GREY
                for t in sub.relationship_tier]
        ax.barh(y, sub.similarity, color=cols, height=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{r.neighbour}\n{r.neighbour_class}" for r in sub.itertuples()],
                           fontsize=5.8)
        ax.set_xlim(0, 1); ax.set_xlabel("similarity", fontsize=7)
        role = c.roles[c.roles.motif == m].iloc[0]
        tag = ("bridge" if role.is_bridge else "isolated" if role.is_isolated
               else "hub" if role.is_hub else "typical")
        ax.set_title(f"{m}\n[{tag}] {c.cls[c.ids.index(m)]}", fontsize=7.5, loc="left",
                     color=INK)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.legend(handles=[Line2D([], [], color=k, lw=6, label=v) for k, v in
                        [(GREEN, "exact equivalence"), (BLUE, "shared substructure"),
                         (AMBER, "broad superfamily"), (GREY, "generic Raman overlap")]],
               loc="lower center", ncol=4, frameon=False, fontsize=7.5,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Nearest-neighbour cards — five neighbours per motif, each classified by "
                 "what kind of relationship it is",
                 fontsize=9.5, x=0.005, ha="left", y=1.0, color=INK)
    save(fig, "fig15_neighbour_cards")


def _colour_fig(c, key, values, title, name):
    uniq = sorted(set(values))
    cm = plt.get_cmap("tab10" if len(uniq) <= 10 else "tab20")
    cmap = {u: cm(i % 20) for i, u in enumerate(uniq)}
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.2), gridspec_kw={"wspace": 0.25})
    for ax, E, lab in zip(axes, [c.umap, c.diff[:, :2], c.pca_scores[:, :2]],
                          ["UMAP", "diffusion map", "PCA"]):
        c.scatter(ax, E, [cmap[v] for v in values], lab)
    axes[-1].legend(handles=[Line2D([], [], marker="o", ls="", color=cmap[u], label=str(u),
                                    markersize=5) for u in uniq],
                    loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=6.5)
    fig.suptitle(title, fontsize=9.5, x=0.005, ha="left", y=1.04, color=INK)
    save(fig, name)


def f16_by_class(c):
    conf = pd.read_csv(V / "confounding_v1.csv")
    r = conf[conf.label == "chemistry_class"].iloc[0]
    _colour_fig(c, "class", c.cls,
                f"Geometry coloured by chemistry class — PERMANOVA F = {r.permanova_F:.2f}, "
                f"p = {r.permanova_p:.3f}, R² = {r.permanova_R2:.3f}; "
                f"kNN {r.knn_accuracy:.3f} vs chance {r.chance:.3f}",
                "fig16_by_chemistry_class")


def f17_by_source(c):
    conf = pd.read_csv(V / "confounding_v1.csv")
    r = conf[conf.label == "source"].iloc[0]
    _colour_fig(c, "source", c.src,
                f"Geometry coloured by SOURCE — PERMANOVA F = {r.permanova_F:.2f}, "
                f"p = {r.permanova_p:.3f}, R² = {r.permanova_R2:.3f}. Significant but far "
                f"weaker than chemistry: a caution, not a veto.",
                "fig17_by_source")


def f18_by_excitation(c):
    conf = pd.read_csv(V / "confounding_v1.csv")
    r = conf[conf.label == "excitation"].iloc[0]
    _colour_fig(c, "excitation", c.exc,
                f"Geometry coloured by EXCITATION — PERMANOVA F = {r.permanova_F:.2f}, "
                f"p = {r.permanova_p:.3f}, R² = {r.permanova_R2:.3f}",
                "fig18_by_excitation")


def _neighbourhood_fig(c, prop_id, title, name):
    row = c.prop[c.prop.proposal == prop_id].iloc[0]
    rej = pd.read_csv(REPO / "results/v7_rebuild/phase02/tables/"
                      "rejected_consensus_motifs_v1.csv")
    members = rej[rej.proposed_group == prop_id].iloc[0].contributing_lsms.split(";")
    idx = [c.ids.index(m) for m in members]
    grad = pd.read_csv(T / "proposal_gradients_v1.csv")
    g = grad[grad.proposal == prop_id].sort_values("rank")

    fig = plt.figure(figsize=(11.6, 4.6))
    gs = fig.add_gridspec(1, 3, wspace=0.42, width_ratios=[0.9, 1.5, 0.75])
    ax = fig.add_subplot(gs[0])
    other = [i for i in range(len(c.ids)) if i not in idx]
    ax.scatter(c.umap[other, 0], c.umap[other, 1], s=22, color="#e5e7eb", zorder=1)
    for i in idx:
        ax.scatter(c.umap[i, 0], c.umap[i, 1], s=70, color=c.col[c.cls[i]],
                   edgecolor=INK, linewidth=0.8, zorder=3)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("position in UMAP", fontsize=8.5, loc="left", color=INK)

    ax = fig.add_subplot(gs[1])
    for rank, r in enumerate(g.itertuples()):
        i = c.ids.index(r.motif)
        h = c.H[i]
        ax.plot(c.grid, h / h.max() + rank * 0.55, lw=0.9, color=c.col[c.cls[i]])
        ax.text(470, rank * 0.55 + 0.30, f"{r.motif}", fontsize=6.0,
                color=c.col[c.cls[i]], va="bottom", ha="left")
    ax.set_xlim(450, 1800); ax.set_yticks([])
    ax.set_xlabel("Raman shift (cm⁻¹)")
    ax.set_title("ordered along diffusion coordinate 1 — is there a gradient?",
                 fontsize=8.5, loc="left", color=INK)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)

    ax = fig.add_subplot(gs[2])
    stats = [("separation ratio", row.separation_ratio, 2.0),
             ("conductance", row.conductance, 1.0),
             ("local dimension", row.mean_local_dimension, 6.0),
             ("internal stability", row.internal_stability, 1.0)]
    y = np.arange(len(stats))[::-1]
    ax.barh(y, [min(v / m, 1.0) for _, v, m in stats], color=BLUE, alpha=0.8, height=0.5)
    for yy, (lab, v, m) in zip(y, stats):
        ax.text(0.0, yy + 0.42, lab, va="bottom", fontsize=7.0, color=MUTED)
        ax.text(min(v / m, 1.0) + 0.03, yy, f"{v:.2f}", va="center", fontsize=7.5, color=INK)
    ax.set_yticks([]); ax.set_xlim(0, 1.35); ax.set_xticks([])
    ax.set_title(f"geometry: {row.geometry_type}", fontsize=8.5, loc="left", color=INK)
    for s in ax.spines.values(): s.set_visible(False)

    fig.suptitle(title, fontsize=9.5, x=0.005, ha="left", y=1.02, color=INK)
    save(fig, name)


def f19_lipid(c):
    _neighbourhood_fig(c, "proposal00",
                       "Lipid neighbourhood (Phase 02 proposal00) — rejected as a merge, "
                       "examined here as geometry", "fig19_lipid_neighbourhood")


def f20_polar(c):
    _neighbourhood_fig(c, "proposal03",
                       "Polar skeletal neighbourhood (proposal03) — protein, saccharide, "
                       "carboxylate, phosphate", "fig20_polar_skeletal_neighbourhood")


def f21_ring(c):
    _neighbourhood_fig(c, "proposal16",
                       "Heterocyclic / ring-system neighbourhood (proposal16) — nucleic acid, "
                       "purine, thiol cofactor", "fig21_ring_system_neighbourhood")


def f22_bridges(c):
    r = c.roles.sort_values("betweenness", ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.6), gridspec_kw={"wspace": 0.3})
    ax = axes[0]
    ax.scatter(c.roles.clustering, c.roles.betweenness, s=42,
               color=[RED if b else GREY for b in c.roles.is_bridge],
               edgecolor="white", linewidth=0.5)
    for x in c.roles[c.roles.is_bridge].itertuples():
        ax.annotate(x.motif, (x.clustering, x.betweenness), fontsize=6,
                    xytext=(4, 3), textcoords="offset points", color=RED)
    ax.set_xlabel("local clustering"); ax.set_ylabel("betweenness centrality")
    ax.set_title("Bridges: high betweenness, low clustering", fontsize=9, loc="left",
                 color=INK)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax = axes[1]
    top = r.head(10)
    y = np.arange(len(top))[::-1]
    ax.barh(y, top.betweenness, color=[RED if b else GREY for b in top.is_bridge],
            height=0.6)
    ax.set_yticks(y); ax.set_yticklabels(top.motif, fontsize=6.5)
    ax.set_xlabel("betweenness centrality")
    ax.set_title("The ten motifs most on the paths between neighbourhoods",
                 fontsize=9, loc="left", color=INK)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    save(fig, "fig22_bridge_motifs")


def f23_outliers(c):
    r = c.roles.sort_values("isolation", ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.6), gridspec_kw={"wspace": 0.3,
                                                                    "width_ratios": [1, 1.3]})
    ax = axes[0]
    top = r.head(10)
    y = np.arange(len(top))[::-1]
    ax.barh(y, top.isolation, color=[RED if b else GREY for b in top.is_isolated], height=0.6)
    ax.set_yticks(y); ax.set_yticklabels(top.motif, fontsize=6.5)
    ax.set_xlabel("isolation (mean kNN distance / median distance)")
    ax.set_title("Most isolated motifs", fontsize=9, loc="left", color=INK)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax = axes[1]
    for k, m in enumerate(c.roles[c.roles.is_isolated].motif):
        i = c.ids.index(m)
        h = c.H[i]
        ax.plot(c.grid, h / h.max() + k * 0.5, lw=0.9, color=c.col[c.cls[i]])
        ax.text(1810, k * 0.5 + 0.05, f"{m}", fontsize=5.8, color=c.col[c.cls[i]])
    ax.set_xlim(450, 1800); ax.set_yticks([]); ax.set_xlabel("Raman shift (cm⁻¹)")
    ax.set_title("Their spectra — candidates for singleton themes, or for exclusion",
                 fontsize=9, loc="left", color=INK)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    save(fig, "fig23_isolated_motifs")


def f24_prior_map(c):
    pr = c.priors["priors"]
    fig, ax = plt.subplots(figsize=(9.8, 6.4))
    ax.scatter(c.umap[:, 0], c.umap[:, 1], s=26, color="#e5e7eb", zorder=1)
    cm = plt.get_cmap("tab10")
    for k, p in enumerate(pr):
        idx = [c.ids.index(m) for m in p["supporting_lsms"] if m in c.ids]
        if not idx:
            continue
        pts = c.umap[idx]
        col = cm(k % 10)
        ax.scatter(pts[:, 0], pts[:, 1], s=64, color=col, edgecolor=INK, linewidth=0.5,
                   zorder=3, label=f"{p['provisional_name']} ({p['geometry_type']}, "
                                   f"conf {p['confidence']:.2f})")
        ctr = pts.mean(0)
        rad = max(np.linalg.norm(pts - ctr, axis=1).max() * 1.15, 0.4)
        ax.add_patch(plt.Circle(ctr, rad, fill=False, ec=col, lw=1.1, ls="--", alpha=0.7))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Provisional Phase 03 priors on the primary geometry.\n"
                 "These constrain Phase 03; they do not decide it. No theme is created here.",
                 fontsize=9.5, loc="left", color=INK)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=6.5)
    save(fig, "fig24_phase03_prior_map")


def f25_architecture(c):
    fig, ax = plt.subplots(figsize=(10.0, 3.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3.4); ax.set_axis_off()
    steps = [("00\nbenchmark\nlock", GREEN), ("01\nbalanced refs\n50 LSMs", GREEN),
             ("02\nCSMs\n49", GREEN), ("02.5\nlatent\ngeometry", BLUE),
             ("03\nbiochemical\nthemes", MUTED), ("04\ncontinuous\nBSV", MUTED),
             ("05\ninference\nengine", MUTED), ("06\nRaman\nvalidation", MUTED)]
    for k, (lab, col) in enumerate(steps):
        x = 0.15 + k * 1.23
        done = col in (GREEN, BLUE)
        ax.add_patch(FancyBboxPatch((x, 1.35), 1.05, 1.05, boxstyle="round,pad=0.05",
                                    fc="#eff6ff" if col == BLUE else
                                    ("#f0fdf4" if done else "white"), ec=col,
                                    lw=1.6 if col == BLUE else 1.1))
        ax.text(x + 0.525, 1.87, lab, ha="center", va="center", fontsize=7, color=INK)
        if k:
            ax.add_patch(FancyArrowPatch((x - 0.15, 1.87), (x - 0.02, 1.87),
                                         arrowstyle="-|>", mutation_scale=8, color=MUTED,
                                         lw=0.9))
    ax.text(0.15, 0.85, "Phase 02.5 is an inserted ANALYSIS phase. It produces geometry and "
                        "priors, not fitted objects;\nPhase 03 consumes the priors as "
                        "constraints and remains free to reject them.",
            fontsize=7.6, color=INK)
    ax.text(0.15, 0.25, f"primary metric: {c.primary_metric} · primary geometry: "
                        f"{c.primary_geom} · {len(c.priors['priors'])} provisional priors",
            fontsize=7.2, color=MUTED)
    ax.set_title("GAIRA V7 architecture after Phase 02.5", fontsize=10.5, loc="left",
                 color=INK)
    save(fig, "fig25_architecture_schematic")


def interactive(c):
    """Standalone HTML: hover any motif to read its class, source, excitation and role."""
    INT.mkdir(parents=True, exist_ok=True)
    pts = [{"x": float(c.umap[i, 0]), "y": float(c.umap[i, 1]), "id": c.ids[i],
            "cls": c.cls[i], "src": c.src[i], "exc": c.exc[i], "nmol": c.nmol[i],
            "role": ("bridge" if c.roles.iloc[i].is_bridge else
                     "isolated" if c.roles.iloc[i].is_isolated else
                     "hub" if c.roles.iloc[i].is_hub else "typical")}
           for i in range(len(c.ids))]
    colours = {f: matplotlib.colors.to_hex(c.col[f]) for f in c.fams}
    html = """<!doctype html><meta charset="utf-8"><title>GAIRA V7 Phase 02.5 — motif geometry</title>
<style>body{font:13px/1.5 -apple-system,system-ui,sans-serif;margin:24px;color:#1a1a1a}
svg{border:1px solid #e5e7eb;border-radius:6px}
#tip{position:fixed;background:#fff;border:1px solid #9ca3af;border-radius:5px;padding:7px 9px;
font-size:12px;pointer-events:none;opacity:0;box-shadow:0 2px 8px rgba(0,0,0,.12)}
button{margin:0 4px 8px 0;padding:4px 9px;border:1px solid #9ca3af;background:#fff;
border-radius:4px;cursor:pointer;font-size:12px}button.on{background:#1a1a1a;color:#fff}</style>
<h2>GAIRA V7 Phase 02.5 — latent geometry of 50 Local Spectral Motifs</h2>
<p>UMAP of the primary geometry. Colour by: <span id="btns"></span></p>
<svg id="p" width="900" height="620"></svg><div id="tip"></div>
<script>
const PTS=__PTS__, COL=__COL__;
const svg=document.getElementById('p'),tip=document.getElementById('tip');
const xs=PTS.map(p=>p.x),ys=PTS.map(p=>p.y);
const sx=v=>60+(v-Math.min(...xs))/(Math.max(...xs)-Math.min(...xs))*780;
const sy=v=>560-(v-Math.min(...ys))/(Math.max(...ys)-Math.min(...ys))*500;
const pal=['#2563eb','#15803d','#b45309','#7c3aed','#0891b2','#be123c','#ca8a04','#0f766e',
'#9333ea','#0369a1'];
let mode='cls';
function colour(p){if(mode==='cls')return COL[p.cls]||'#999';
const keys=[...new Set(PTS.map(q=>q[mode]))].sort();return pal[keys.indexOf(p[mode])%10];}
function draw(){svg.innerHTML='';PTS.forEach(p=>{
const c=document.createElementNS('http://www.w3.org/2000/svg','circle');
c.setAttribute('cx',sx(p.x));c.setAttribute('cy',sy(p.y));
c.setAttribute('r',p.role==='typical'?6:8);c.setAttribute('fill',colour(p));
c.setAttribute('stroke',p.role==='typical'?'#fff':'#1a1a1a');c.setAttribute('stroke-width',1.2);
c.onmousemove=e=>{tip.style.opacity=1;tip.style.left=(e.clientX+12)+'px';
tip.style.top=(e.clientY+12)+'px';
tip.innerHTML='<b>'+p.id+'</b><br>class: '+p.cls+'<br>source: '+p.src+
'<br>excitation: '+p.exc+' nm<br>molecules: '+p.nmol+'<br>role: '+p.role;};
c.onmouseleave=()=>tip.style.opacity=0;svg.appendChild(c);});}
const b=document.getElementById('btns');
[['cls','chemistry class'],['src','source'],['exc','excitation'],['role','graph role']]
.forEach(([k,lab])=>{const el=document.createElement('button');el.textContent=lab;
el.className=k==='cls'?'on':'';el.onclick=()=>{mode=k;
[...b.children].forEach(x=>x.className='');el.className='on';draw();};b.appendChild(el);});
draw();
</script>"""
    html = html.replace("__PTS__", json.dumps(pts)).replace("__COL__", json.dumps(colours))
    (INT / "motif_geometry.html").write_text(html)
    print("  interactive/motif_geometry.html")


def main():
    c = Ctx()
    print("[phase02.5] figures")
    for fn in (f01_schematic, f02_metric_comparison, f03_null_vs_observed, f04_pca,
               f05_pca_loadings, f06_umap_grid, f07_diffusion, f08_dendrogram,
               f09_cosine_heatmap, f10_multiview_heatmap, f11_knn_graph, f12_force_directed,
               f13_mst, f14_community_stability, f15_neighbour_cards, f16_by_class,
               f17_by_source, f18_by_excitation, f19_lipid, f20_polar, f21_ring,
               f22_bridges, f23_outliers, f24_prior_map, f25_architecture):
        fn(c)
    interactive(c)


if __name__ == "__main__":
    main()
