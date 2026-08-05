"""GAIRA V6.2 — publication figures. Vector (PDF) + raster (PNG) export."""
from __future__ import annotations
import sys, json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch

REPO = Path("/Users/surajpg/projects/GAIRA")
sys.path.insert(0, str(REPO / "results/v6_rebuild/code"))
from v62 import core as C

BASE = REPO / "results/v6_rebuild"
OUT = BASE / "figures_v62"
OUT.mkdir(parents=True, exist_ok=True)

INK, MUTED, GRIDC = "#1b2430", "#5b6472", "#dfe4ea"
PAL = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9", "#7B52AB", "#8C8C8C"]
BLUE, VERM, GREEN, ORANGE, PINK, SKY, PURP = PAL[:7]
mpl.rcParams.update({
    "font.size": 8.2, "axes.edgecolor": GRIDC, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 9.2, "axes.titleweight": "bold",
    "axes.grid": True, "grid.color": GRIDC, "grid.linewidth": 0.45, "legend.frameon": False,
    "figure.facecolor": "white", "savefig.facecolor": "white", "axes.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42,
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  wrote", name)


# ── data ──
SP = np.load(C.art("v62_spaces.npz"), allow_pickle=True)
MB = np.load(C.art("v62_membership.npz"), allow_pickle=True)
IB = pd.read_csv(C.tab("v62_information_bottleneck.csv"))
PAR = pd.read_csv(C.tab("v62_pareto.csv"))
SH = pd.read_csv(C.tab("v62_shared_motifs.csv"))
UNC = pd.read_csv(C.tab("v62_theme_uncertainty.csv"))
PROP = pd.read_csv(C.tab("v62_uncertainty_propagation.csv"))
CONF = pd.read_csv(C.tab("v62_confusion_explained.csv"))
GN = pd.read_csv(C.tab("v62_graph_nodes.csv"))
GE = pd.read_csv(C.tab("v62_graph_edges.csv"))
CMPLD = pd.read_csv(C.tab("v62_learned_vs_derived.csv"))
J1 = json.loads(C.art("v62_soft_hierarchy.json").read_text())
J2 = json.loads(C.art("v62_information_graph.json").read_text())
V6EV = json.loads(C.art("p7_evaluation.json").read_text())

IDS = [str(x) for x in SP["motif_ids"]]
L2 = [str(x) for x in SP["L2_names"]]
L3 = [str(x) for x in SP["L3_names"]]
S2, S3, A = SP["S2"], SP["S3"], SP["A"]
E2, P2, U2, R, Rn = SP["E2"], SP["P2"], SP["U2"], SP["R"], SP["Rn"]
ANA = [str(x) for x in SP["analytes"]]
FAM = [str(x) for x in SP["families"]]
SHORT = [m.replace("_", " ") for m in IDS]
IBH = IB[IB.grouping == "hybrid_clustering"].sort_values("K")
IBL2 = IB[IB.grouping == "L2_superclass"]


# ═══ F1 semantic hierarchy overview + multi-scale ═══
def f01():
    fig, ax = plt.subplots(figsize=(11.5, 5.4)); ax.set_xlim(0, 100); ax.set_ylim(0, 46); ax.axis("off")

    def col(x, w, title, sub, c, items, fs=6.0):
        ax.add_patch(FancyBboxPatch((x, 5), w, 32, boxstyle="round,pad=0.5,rounding_size=1.2",
                                    fc="white", ec=c, lw=1.5))
        ax.add_patch(FancyBboxPatch((x, 31.5), w, 5.5, boxstyle="round,pad=0.5,rounding_size=1.2",
                                    fc=c, ec=c))
        ax.text(x + w / 2, 34.2, title, ha="center", va="center", fontsize=8.8, weight="bold", color="white")
        ax.text(x + w / 2, 29.2, sub, ha="center", va="center", fontsize=6.6, color=MUTED)
        for i, it in enumerate(items):
            ax.text(x + w / 2, 26.6 - i * 1.42, it, ha="center", va="center", fontsize=fs, color=INK)

    col(1, 17, "24 components", "FROZEN atlas", "#8C8C8C", ["c0 … c23", "", "NNLS, H fixed",
                                                            "09ed804a…"], 6.4)
    col(21, 24, "17 MSS motifs", "FROZEN from V6", VERM, SHORT[:15] + ["…"], 5.4)
    col(49, 20, "L2 · 6 chemical themes", "soft, row-stochastic", GREEN, L2, 6.4)
    col(73, 14, "L3 · 4 systems", "coarse abstraction", BLUE, L3, 5.8)
    col(90, 9, "BSV +\ndomain", "future", "#b8c0c8", ["biological", "state", "", "deferred"], 6.0)
    for a, b in ((18, 21), (45, 49), (69, 73), (87, 90)):
        ax.add_patch(FancyArrowPatch((a, 20), (b, 20), arrowstyle="-|>", mutation_scale=12,
                                     color=MUTED, lw=1.3))
    ax.text(19.5, 22.4, "M", ha="center", fontsize=8, weight="bold", color=VERM)
    ax.text(47, 22.4, "S", ha="center", fontsize=8, weight="bold", color=GREEN)
    ax.text(71, 22.4, "S₃", ha="center", fontsize=8, weight="bold", color=BLUE)
    ax.text(50, 2.4, "theme(x) = Sᵀ Mᵀ coord(x)      S ≥ 0,  rows sum to 1,  mean support 2.06 themes / motif",
            ha="center", fontsize=8.6, family="monospace", color=INK)
    ax.text(50, 0.2, "V6 used a HARD partition. V6.2 uses SOFT membership, so a motif that genuinely "
                     "belongs to two chemistries is represented as belonging to two chemistries.",
            ha="center", fontsize=7.0, color=MUTED, style="italic")
    ax.set_title("The V6.2 semantic hierarchy — multi-scale, soft, non-circular",
                 fontsize=11.4, weight="bold", pad=6)
    save(fig, "v62_f01_hierarchy")


# ═══ F2 soft theme heatmap + probability distributions ═══
def f02():
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 4.4), gridspec_kw={"width_ratios": [1.3, 1, 1]})
    a = axes[0]
    im = a.imshow(S2, cmap="Greens", aspect="auto", vmin=0, vmax=1)
    a.set_yticks(range(len(IDS))); a.set_yticklabels(SHORT, fontsize=6.0)
    a.set_xticks(range(len(L2))); a.set_xticklabels(L2, rotation=40, ha="right", fontsize=6.4)
    for i in range(len(IDS)):
        for j in range(len(L2)):
            if S2[i, j] > 0.02:
                a.text(j, i, f"{S2[i,j]:.2f}", ha="center", va="center", fontsize=4.9,
                       color="white" if S2[i, j] > .55 else INK)
    a.set_title("S · soft motif → theme membership (L2)", fontsize=9.0); a.grid(False)
    fig.colorbar(im, ax=a, fraction=.035, pad=.02).ax.tick_params(labelsize=6)

    a = axes[1]
    ent = C.norm_entropy(S2, axis=1)
    o = np.argsort(ent)
    a.barh(range(len(IDS)), ent[o], color=[GREEN if e < .2 else (ORANGE if e < .4 else VERM)
                                           for e in ent[o]])
    a.set_yticks(range(len(IDS))); a.set_yticklabels([SHORT[i] for i in o], fontsize=6.0)
    a.set_xlabel("membership entropy (0 = one theme)")
    a.set_title("How shared is each motif?", fontsize=9.0); a.grid(axis="y", visible=False)

    a = axes[2]
    top = UNC.sort_values("entropy").iloc[[0, 1, len(UNC) // 2, -2, -1]]
    x = np.arange(len(L2))
    for k, (_, r) in enumerate(top.iterrows()):
        i = ANA.index(r.analyte)
        p = (A[i] @ S2); p = p / (p.sum() + C.EPS)
        a.plot(x, p, "o-", lw=1.3, ms=3.6, color=PAL[k % len(PAL)],
               label=f"{r.analyte[:16]}  H={r.entropy:.2f}")
    a.set_xticks(x); a.set_xticklabels(L2, rotation=40, ha="right", fontsize=6.2)
    a.set_ylabel("theme posterior"); a.legend(fontsize=6.0)
    a.set_title("Theme probability distributions", fontsize=9.0)
    fig.suptitle("Parts 1–2 — soft membership and theme uncertainty", fontsize=10.8, weight="bold", y=1.02)
    fig.tight_layout(); save(fig, "v62_f02_soft_membership")


# ═══ F3 shared motif network ═══
def f03():
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6), gridspec_kw={"width_ratios": [1.25, 1]})
    a = axes[0]
    G = nx.Graph()
    for t in L2:
        G.add_node(t, kind="theme")
    for i, m in enumerate(IDS):
        G.add_node(m, kind="motif")
        for j in range(len(L2)):
            if S2[i, j] > 0:
                G.add_edge(m, L2[j], weight=float(S2[i, j]))
    pos = nx.spring_layout(G, seed=3, k=0.55, weight="weight", iterations=250)
    for u, v, d in G.edges(data=True):
        a.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
               color=VERM if d["weight"] < .5 else GRIDC,
               lw=0.5 + 3.2 * d["weight"], alpha=.8 if d["weight"] < .5 else .55, zorder=1)
    tn = [n for n in G if G.nodes[n]["kind"] == "theme"]
    mn = [n for n in G if G.nodes[n]["kind"] == "motif"]
    a.scatter([pos[n][0] for n in mn], [pos[n][1] for n in mn], s=48, color=BLUE,
              edgecolor="white", lw=.7, zorder=2)
    a.scatter([pos[n][0] for n in tn], [pos[n][1] for n in tn], s=280, color=GREEN,
              edgecolor="white", lw=1.1, zorder=3)
    for n in tn:
        a.annotate(n, pos[n], fontsize=6.6, weight="bold", ha="center", va="center",
                   color="white", zorder=4)
    for n in mn:
        a.annotate(n.replace("_", " "), pos[n], fontsize=5.0, color=INK, zorder=4,
                   xytext=(0, 7), textcoords="offset points", ha="center")
    a.axis("off")
    a.set_title("Shared biochemical motifs — red edges are cross-theme membership", fontsize=9.0)

    a = axes[1]
    d = SH.sort_values("shared_weight")
    y = np.arange(len(d))
    a.barh(y, d.dominant_weight, .55, color=GREEN, label="dominant theme")
    a.barh(y, d.shared_weight, .55, left=d.dominant_weight, color=VERM, label="shared with")
    a.set_yticks(y); a.set_yticklabels([m.replace("_", " ") for m in d.motif], fontsize=6.6)
    a.set_xlabel("membership weight"); a.legend(fontsize=6.8)
    a.set_title(f"{len(SH)} motifs carry cross-theme membership", fontsize=9.0)
    a.grid(axis="y", visible=False)
    for i, (_, r) in enumerate(d.iterrows()):
        a.text(1.01, i, f"{r.dominant_theme.split()[0]} / {r.shared_with.split()[0]}",
               fontsize=5.6, color=MUTED, va="center")
    fig.suptitle("Part 3 — shared motifs are a feature, not a defect", fontsize=10.8,
                 weight="bold", y=1.02)
    fig.tight_layout(); save(fig, "v62_f03_shared_motifs")


# ═══ F4 recoverability + confusion ═══
def f04():
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 4.2), gridspec_kw={"width_ratios": [1.2, 1.2, 1]})
    a = axes[0]
    im = a.imshow(Rn, cmap="Blues", vmin=0, vmax=1)
    a.set_xticks(range(len(IDS))); a.set_yticks(range(len(IDS)))
    a.set_xticklabels(SHORT, rotation=70, ha="right", fontsize=5.0)
    a.set_yticklabels(SHORT, fontsize=5.0)
    a.set_xlabel("predicted motif"); a.set_ylabel("expected motif")
    a.set_title("17 × 17 recoverability matrix", fontsize=9.0); a.grid(False)
    fig.colorbar(im, ax=a, fraction=.045, pad=.02).ax.tick_params(labelsize=6)

    a = axes[1]
    sc = a.scatter(CONF.spectral_cosine, CONF.component_overlap, s=18 + 9 * CONF.n,
                   c=CONF.rate, cmap="Oranges", vmin=0, vmax=1, edgecolor=INK, lw=.4)
    for _, r in CONF.nlargest(5, "n").iterrows():
        a.annotate(f"{r.expected.split('_')[0]}→{r.predicted.split('_')[0]}",
                   (r.spectral_cosine, r.component_overlap), fontsize=5.6, color=INK,
                   xytext=(4, 4), textcoords="offset points")
    a.set_xlabel("spectral cosine between motifs"); a.set_ylabel("component-support overlap")
    a.set_title("Why motifs confuse", fontsize=9.0)
    fig.colorbar(sc, ax=a, fraction=.045, pad=.02).ax.tick_params(labelsize=6)

    a = axes[2]; a.axis("off")
    lines = [("Confusion is explained, not mysterious", INK, True), ("", INK, False)]
    for _, r in CONF.nlargest(5, "n").iterrows():
        lines.append((f"{r.expected.replace('_',' ')}\n  → {r.predicted.replace('_',' ')}  (n={int(r.n)})\n"
                      f"  spectral cos {r.spectral_cosine:.2f} · component overlap {r.component_overlap:.2f}\n"
                      f"  shared bands {r.overlapping_bands_cm}", MUTED, False))
    yy = 1.0
    for t, c, b in lines:
        a.text(0, yy, t, fontsize=6.4, color=c, weight="bold" if b else "normal",
               va="top", linespacing=1.5)
        yy -= 0.05 + 0.048 * t.count("\n")
    fig.suptitle("Part 7 — recoverability matrix and its spectroscopic explanation",
                 fontsize=10.8, weight="bold", y=1.03)
    fig.tight_layout(); save(fig, "v62_f04_recoverability")


# ═══ F5 theme manifold (PCA + UMAP) ═══
def f05():
    fams = np.array(FAM)
    keep = pd.Series(fams).value_counts()
    big = list(keep[keep >= 5].index)[:8]
    cmap = {f: PAL[i % len(PAL)] for i, f in enumerate(big)}
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.8))
    for a, XY, ttl in ((axes[0], P2[:, :2], "PCA of the continuous theme space"),
                       (axes[1], U2, "UMAP of the continuous theme space")):
        for f in big:
            m = fams == f
            a.scatter(XY[m, 0], XY[m, 1], s=22, color=cmap[f], label=f, edgecolor="white", lw=.4)
        oth = ~np.isin(fams, big)
        a.scatter(XY[oth, 0], XY[oth, 1], s=12, color="#c8d0d8", label="other", edgecolor="none")
        a.set_title(ttl, fontsize=9.0); a.set_xticks([]); a.set_yticks([])
    axes[0].legend(fontsize=5.8, ncol=2, loc="best")
    a = axes[2]
    conf = UNC.set_index("analyte").loc[ANA].confidence.values
    s = a.scatter(U2[:, 0], U2[:, 1], s=24, c=conf, cmap="viridis", edgecolor="white", lw=.4)
    a.set_title("UMAP coloured by theme confidence", fontsize=9.0)
    a.set_xticks([]); a.set_yticks([])
    fig.colorbar(s, ax=a, fraction=.045, pad=.02).ax.tick_params(labelsize=6)
    fig.suptitle("Parts 4 & 8 — chemistry organises continuously, not into discrete bins",
                 fontsize=10.8, weight="bold", y=1.04)
    fig.tight_layout(); save(fig, "v62_f05_manifold")


# ═══ F6 information bottleneck ═══
def f06():
    fig, axes = plt.subplots(1, 4, figsize=(11.8, 3.3))
    a = axes[0]
    a.plot(IBH.K, IBH.explained_variance_motif, "o-", color=BLUE, lw=1.6, ms=4)
    a.axvline(J2["information_bottleneck"]["elbow_K"], color=VERM, ls="--", lw=1.1)
    a.text(J2["information_bottleneck"]["elbow_K"] + .3, .45,
           f"elbow K={J2['information_bottleneck']['elbow_K']}", fontsize=6.6, color=VERM)
    if len(IBL2):
        a.scatter(IBL2.K, IBL2.explained_variance_motif, s=140, marker="*", color=GREEN,
                  edgecolor="white", zorder=5)
        a.annotate("L2 (chemical)", (IBL2.K.iloc[0], IBL2.explained_variance_motif.iloc[0]),
                   fontsize=6.4, color=GREEN, xytext=(6, -12), textcoords="offset points")
    a.set_xlabel("themes K"); a.set_ylabel("motif variance retained")
    a.set_title("Information retained", fontsize=8.8)

    a = axes[1]
    a.plot(IBH.K, IBH.reconstruction_error, "o-", color=VERM, lw=1.5, ms=3.6, label="recon. error")
    a.plot(IBH.K, IBH.kl_divergence, "s--", color=ORANGE, lw=1.3, ms=3.2, label="KL divergence")
    a.set_xlabel("themes K"); a.legend(fontsize=6.6)
    a.set_title("Information lost", fontsize=8.8)

    a = axes[2]
    a.plot(IBH.K, IBH.mi_retained, "o-", color=GREEN, lw=1.6, ms=4)
    a.axhline(1.0, color=MUTED, ls="--", lw=1)
    a.text(10, 1.02, "parity with the motif layer", fontsize=6.2, color=MUTED)
    a.set_xlabel("themes K"); a.set_ylabel("MI(family; theme) / MI(family; motif)")
    a.set_title("Chemical information per dimension", fontsize=8.8)

    a = axes[3]
    pk = PAR.drop_duplicates("K").set_index("K")
    ks = [k for k in IBH.K if k in pk.index]
    a.plot([float(IBH[IBH.K == k].compression_ratio.iloc[0]) for k in ks],
           [float(pk.loc[k, "interpretability"]) for k in ks], "o-", color=PURP, lw=1.6, ms=4)
    for _, r in IBH.iterrows():
        if int(r.K) in (2, 6, 13, 17):
            a.annotate(f"K={int(r.K)}", (r.compression_ratio,
                                         float(PAR[PAR.K == r.K].interpretability.iloc[0])),
                       fontsize=6.2, color=INK, xytext=(4, 4), textcoords="offset points")
    a.set_xlabel("compression ratio (17 / K)"); a.set_ylabel("interpretability")
    a.set_title("Compression vs interpretability", fontsize=8.8)
    fig.suptitle("Parts 5 & 9 — the information bottleneck: abstraction has a measurable price",
                 fontsize=10.8, weight="bold", y=1.05)
    fig.tight_layout(); save(fig, "v62_f06_bottleneck")


# ═══ F7 confidence propagation ═══
def f07():
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.6), gridspec_kw={"width_ratios": [1.15, 1, 1]})
    a = axes[0]
    lv = ["coordinates\n(24)", "MSS motifs\n(17)", "chemical themes\n(6)"]
    v = [PROP.coord_total_var.median(), PROP.mss_total_var.median(), PROP.theme_total_var.median()]
    a.bar(range(3), v, .55, color=["#8C8C8C", VERM, GREEN])
    for i, x in enumerate(v):
        a.text(i, x * 1.08, f"{x:.2e}", ha="center", fontsize=6.6, weight="bold")
    a.set_yscale("log"); a.set_xticks(range(3)); a.set_xticklabels(lv, fontsize=7)
    a.set_ylabel("median total variance (trace Σ)")
    a.set_title("Uncertainty through the hierarchy", fontsize=8.8)
    a.grid(axis="x", visible=False)

    a = axes[1]
    a.scatter(PROP.theme_entropy, PROP.theme_confidence, s=20, c=PROP.mss_to_theme_ratio,
              cmap="magma_r", edgecolor="white", lw=.4)
    a.set_xlabel("theme entropy"); a.set_ylabel("theme confidence")
    a.set_title("Confidence vs entropy", fontsize=8.8)

    a = axes[2]
    a.hist(PROP.coord_to_mss_ratio, bins=24, color=VERM, alpha=.85, label="coord → MSS")
    a.hist(PROP.mss_to_theme_ratio, bins=24, color=GREEN, alpha=.7, label="MSS → theme")
    a.axvline(1.0, color=INK, ls="--", lw=1)
    a.set_xscale("log"); a.set_xlabel("variance amplification factor"); a.legend(fontsize=6.8)
    a.set_title("Where variance shrinks and grows", fontsize=8.8)
    fig.suptitle("Part 12 — Bayesian (delta-method) uncertainty propagation from replicate noise",
                 fontsize=10.8, weight="bold", y=1.04)
    fig.tight_layout(); save(fig, "v62_f07_propagation")


# ═══ F8 ontology graph ═══
def f08():
    G = nx.DiGraph()
    for _, r in GE.iterrows():
        G.add_edge(r.source, r.target, weight=r.weight)
    lvl = GN.set_index("node")
    LAB = dict(zip(GN.node, GN.label))
    fig, ax = plt.subplots(figsize=(11.4, 6.2))
    pos = {}
    for L, ys in ((1, IDS), (2, [f"L2·{t}" for t in L2]), (3, [f"L3·{t}" for t in L3])):
        n = len(ys)
        for i, node in enumerate(ys):
            pos[node] = (L * 3.0, (i - (n - 1) / 2) * (12.0 / max(n, 1)))
    for u, v, d in G.edges(data=True):
        if u not in pos or v not in pos:
            continue
        x0, y0 = pos[u]; x1, y1 = pos[v]
        verts = [(x0, y0), ((x0 + x1) / 2, y0), ((x0 + x1) / 2, y1), (x1, y1)]
        ax.add_patch(PathPatch(MPath(verts, [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4,
                                             MPath.CURVE4]),
                               fc="none", ec=VERM if d["weight"] < .5 else "#b9c2cb",
                               lw=.4 + 3.0 * d["weight"], alpha=.75, zorder=1))
    for node, (x, y) in pos.items():
        L = int(lvl.loc[node, "level"]) if node in lvl.index else 1
        c = {1: BLUE, 2: GREEN, 3: PURP}[L]
        s = {1: 60, 2: 300, 3: 500}[L]
        ax.scatter([x], [y], s=s, color=c, edgecolor="white", lw=1.0, zorder=3)
        ax.annotate(LAB.get(node, node).replace("_", " "), (x, y), fontsize=5.4 if L == 1 else 6.8,
                    color=INK, weight="normal" if L == 1 else "bold",
                    ha="right" if L == 1 else "left", va="center", zorder=5,
                    xytext=(-9, 0) if L == 1 else (14, 0), textcoords="offset points")
    ax.set_xlim(-0.6, 13.6)
    ax.axis("off")
    ax.set_title("Part 11 — the biochemical ontology as a GRAPH, not a tree\n"
                 f"{J2['ontology_graph']['n_multi_parent_motifs']} of {len(IDS)} motifs have "
                 "more than one parent; red edges are the weaker, shared memberships",
                 fontsize=10.4, weight="bold")
    ax.text(0.5, -0.04, "motif  →  L2 chemical theme  →  L3 biochemical system",
            transform=ax.transAxes, ha="center", fontsize=7.4, color=MUTED, style="italic")
    save(fig, "v62_f08_ontology_graph")


# ═══ F9 information-flow Sankey ═══
def f09():
    fig, ax = plt.subplots(figsize=(11.4, 5.2)); ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    mass_m = A.sum(0); mass_m = mass_m / mass_m.sum()
    T = A @ S2; mass_t = T.sum(0); mass_t = mass_t / mass_t.sum()
    T3 = A @ S3; mass_3 = T3.sum(0); mass_3 = mass_3 / mass_3.sum()
    cols = {}
    for name, xs, mass, c in (("m", 1.4, mass_m, BLUE), ("t", 5.0, mass_t, GREEN),
                              ("s", 8.4, mass_3, PURP)):
        y = 0.4; d = {}
        for i, mm in enumerate(mass):
            h = mm * 8.6
            ax.add_patch(Rectangle((xs, y), 0.42, h, fc=c, ec="white", lw=.6))
            d[i] = (y, h)
            y += h + 0.10
        cols[name] = (xs, d)
    for i, (y, h) in cols["m"][1].items():
        ax.text(cols["m"][0] - 0.12, y + h / 2, SHORT[i], fontsize=5.2, ha="right", va="center")
    for i, (y, h) in cols["t"][1].items():
        ax.text(cols["t"][0] + 0.52, y + h / 2, L2[i], fontsize=6.6, ha="left", va="center")
    for i, (y, h) in cols["s"][1].items():
        ax.text(cols["s"][0] + 0.52, y + h / 2, L3[i], fontsize=6.8, ha="left", va="center")
    off_m = {i: cols["m"][1][i][0] for i in cols["m"][1]}
    off_t = {i: cols["t"][1][i][0] for i in cols["t"][1]}
    for i in range(len(IDS)):
        for j in range(len(L2)):
            w = S2[i, j] * mass_m[i] * 8.6
            if w < 1e-3:
                continue
            y0, y1 = off_m[i], off_t[j]
            verts = [(cols["m"][0] + .42, y0), (3.2, y0), (3.2, y1), (cols["t"][0], y1),
                     (cols["t"][0], y1 + w), (3.2, y1 + w), (3.2, y0 + w),
                     (cols["m"][0] + .42, y0 + w)]
            codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
                     MPath.LINETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4]
            ax.add_patch(PathPatch(MPath(verts, codes), fc=GREEN, ec="none", alpha=.30))
            off_m[i] += w; off_t[j] += w
    ax.text(5, 9.7, "Information flow: MSS motif mass → chemical theme → biochemical system",
            ha="center", fontsize=10.4, weight="bold")
    ax.text(5, 0.05, f"motif variance retained at L2: {float(IBL2.explained_variance_motif.iloc[0]):.3f}  ·  "
                     f"compression {float(IBL2.compression_ratio.iloc[0]):.2f}×  ·  "
                     f"KL {float(IBL2.kl_divergence.iloc[0]):.3f}",
            ha="center", fontsize=7.4, color=MUTED, style="italic")
    save(fig, "v62_f09_sankey")


# ═══ F10 calibration ═══
def f10():
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.5))
    conf = UNC.confidence.values
    a = axes[0]
    a.hist(conf, bins=26, color=GREEN, alpha=.85)
    a.axvline(np.median(conf), color=INK, ls="--", lw=1)
    a.set_xlabel("theme confidence"); a.set_ylabel("analytes")
    a.set_title(f"Confidence distribution (median {np.median(conf):.3f})", fontsize=8.8)
    a = axes[1]
    a.scatter(UNC.entropy, UNC.margin, s=18, color=BLUE, edgecolor="white", lw=.4)
    a.set_xlabel("posterior entropy"); a.set_ylabel("top-2 margin")
    a.set_title("Entropy vs margin", fontsize=8.8)
    a = axes[2]
    bins = np.linspace(0, UNC.p_1.max(), 9)
    xs, ys, ns = [], [], []
    prim = SP["primary"]; lab = SP["labelled"]
    dom = (A @ S2).argmax(1)
    exp = np.array([int(np.argmax(S2[p])) if p >= 0 else -1 for p in prim])
    correct = (dom == exp).astype(float)
    for b in range(len(bins) - 1):
        m = (UNC.p_1.values > bins[b]) & (UNC.p_1.values <= bins[b + 1]) & lab
        if m.sum() >= 3:
            xs.append(UNC.p_1.values[m].mean()); ys.append(correct[m].mean()); ns.append(int(m.sum()))
    a.plot([0, 1], [0, 1], ls="--", color=MUTED, lw=1)
    a.plot(xs, ys, "o-", color=VERM, lw=1.5, ms=5)
    for x, y, n in zip(xs, ys, ns):
        a.annotate(str(n), (x, y), fontsize=6, color=MUTED, xytext=(0, 6),
                   textcoords="offset points", ha="center")
    a.set_xlabel("top theme posterior"); a.set_ylabel("observed agreement")
    a.set_title("Reliability of the theme posterior", fontsize=8.8)
    a.set_xlim(0, 1); a.set_ylim(0, 1.05)
    fig.suptitle("Calibration of the V6.2 soft theme layer", fontsize=10.8, weight="bold", y=1.04)
    fig.tight_layout(); save(fig, "v62_f10_calibration")


# ═══ F11 Pareto ═══
def f11():
    fig, axes = plt.subplots(1, 4, figsize=(11.8, 3.3))
    pairs = [("information_retained", "interpretability"), ("recoverability", "interpretability"),
             ("compression", "information_retained"), ("stability", "recoverability")]
    for a, (xk, yk) in zip(axes, pairs):
        a.scatter(PAR[~PAR.pareto][xk], PAR[~PAR.pareto][yk], s=26, color="#c3ccd6")
        a.scatter(PAR[PAR.pareto][xk], PAR[PAR.pareto][yk], s=42, color=BLUE,
                  edgecolor="white", lw=.5)
        adm = PAR[PAR.chemically_admissible]
        a.scatter(adm[xk], adm[yk], s=95, facecolor="none", edgecolor=GREEN, lw=1.2)
        for _, r in PAR.iterrows():
            if int(r.K) in (2, 6, 13, 17):
                a.annotate(f"K={int(r.K)}", (r[xk], r[yk]), fontsize=6.0, color=INK,
                           xytext=(4, 4), textcoords="offset points")
        a.set_xlabel(xk.replace("_", " ")); a.set_ylabel(yk.replace("_", " "))
    axes[0].set_title("blue = Pareto · green ring = chemically admissible", fontsize=8.4)
    fig.suptitle("Part 13 — multi-objective optimisation: interpretability, information, "
                 "recoverability, stability", fontsize=10.6, weight="bold", y=1.05)
    fig.tight_layout(); save(fig, "v62_f11_pareto")


# ═══ F12 learned vs derived ═══
def f12():
    S_l = MB["S_learned"]
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.9), gridspec_kw={"width_ratios": [1.2, 1.2, .9]})
    for a, S, t in ((axes[0], S2, "derived (profile similarity)"),
                    (axes[1], S_l, "learned (constrained NNLS)")):
        im = a.imshow(S, cmap="Greens", aspect="auto", vmin=0, vmax=1)
        a.set_yticks(range(len(IDS))); a.set_yticklabels(SHORT, fontsize=5.6)
        a.set_xticks(range(len(L2))); a.set_xticklabels(L2, rotation=40, ha="right", fontsize=6.0)
        a.set_title(t, fontsize=8.8); a.grid(False)
    a = axes[2]
    d = CMPLD.sort_values("cosine")
    a.barh(range(len(d)), d.cosine, color=[GREEN if x else VERM for x in d.agree])
    a.set_yticks(range(len(d))); a.set_yticklabels([m.replace("_", " ") for m in d.motif], fontsize=5.8)
    a.set_xlabel("cosine(derived, learned)"); a.set_xlim(0, 1.02)
    a.set_title(f"agreement {CMPLD.agree.mean():.0%} · mean cos {CMPLD.cosine.mean():.2f}",
                fontsize=8.8)
    a.grid(axis="y", visible=False)
    fig.suptitle("Part 6 — learned vs derived theme weights", fontsize=10.8, weight="bold", y=1.03)
    fig.tight_layout(); save(fig, "v62_f12_learned_vs_derived")


# ═══ F13 V6 vs V6.2 dashboard ═══
def f13():
    fig = plt.figure(figsize=(11.6, 6.0))
    gs = fig.add_gridspec(2, 3, hspace=.5, wspace=.32)
    a = fig.add_subplot(gs[0, 0]); a.axis("off")
    rows = [("MSS motifs", "17", "17 (identical)"),
            ("Motif→theme map", "hard partition", "soft, row-stochastic"),
            ("Themes", "13 (one level)", "17 / 6 / 4 (three levels)"),
            ("Mean support", "1.00", f"{float((S2>0).sum(1).mean()):.2f}"),
            ("Shared motifs", "0 (suppressed)", f"{len(SH)} represented"),
            ("Uncertainty", "confidence only", "posterior, entropy, margin, variance"),
            ("Ontology", "tree", f"graph ({J2['ontology_graph']['n_multi_parent_motifs']} multi-parent)"),
            ("Objective", "κ × interpretability", "4-objective Pareto")]
    a.text(0, 1.0, "V6 → V6.2", fontsize=9.4, weight="bold", va="top")
    for i, (k, v1, v2) in enumerate(rows):
        y = .90 - i * .112
        a.text(0, y, k, fontsize=6.6, weight="bold", va="top")
        a.text(.42, y, v1, fontsize=6.4, color=MUTED, va="top")
        a.text(.72, y, v2, fontsize=6.4, color=GREEN, va="top")

    a = fig.add_subplot(gs[0, 1])
    a.bar([0, 1], [1.0, float((S2 > 0).sum(1).mean())], .5, color=[MUTED, GREEN])
    a.set_xticks([0, 1]); a.set_xticklabels(["V6 hard", "V6.2 soft"], fontsize=7.4)
    a.set_ylabel("themes per motif"); a.set_title("Membership support", fontsize=8.6)
    a.grid(axis="x", visible=False)

    a = fig.add_subplot(gs[0, 2])
    a.plot(IBH.K, IBH.explained_variance_motif, "-", color=BLUE, lw=1.5)
    a.scatter([13], [float(IBH[IBH.K == 13].explained_variance_motif.iloc[0])], s=110,
              color=MUTED, zorder=4, label="V6 K=13")
    a.scatter(IBL2.K, IBL2.explained_variance_motif, s=150, marker="*", color=GREEN,
              zorder=5, label="V6.2 L2 K=6")
    a.set_xlabel("themes K"); a.set_ylabel("variance retained")
    a.legend(fontsize=6.4); a.set_title("Information vs abstraction", fontsize=8.6)

    a = fig.add_subplot(gs[1, 0])
    ent = C.norm_entropy(S2, axis=1)
    a.hist(ent, bins=14, color=GREEN, alpha=.85)
    a.set_xlabel("motif membership entropy"); a.set_ylabel("motifs")
    a.set_title("V6.2 represents overlap", fontsize=8.6)

    a = fig.add_subplot(gs[1, 1])
    a.bar(range(len(L2)), (A @ S2).sum(0) / (A @ S2).sum(), .6,
          color=[PAL[i % len(PAL)] for i in range(len(L2))])
    a.set_xticks(range(len(L2))); a.set_xticklabels(L2, rotation=40, ha="right", fontsize=6.0)
    a.set_ylabel("share of theme mass"); a.set_title("L2 theme mass across the corpus", fontsize=8.6)
    a.grid(axis="x", visible=False)

    a = fig.add_subplot(gs[1, 2]); a.axis("off")
    a.text(0, 1.0, "What improved", fontsize=8.4, weight="bold", color=GREEN, va="top")
    a.text(0, .90, f"• overlap represented ({len(SH)} shared motifs)\n"
                   f"• three resolutions instead of one\n"
                   f"• full posterior + variance propagation\n"
                   f"• graph ontology, {J2['ontology_graph']['n_multi_parent_motifs']} multi-parent motifs\n"
                   f"• continuous embedding, not bins",
           fontsize=6.6, color=MUTED, va="top", linespacing=1.7)
    a.text(0, .44, "What is unchanged", fontsize=8.4, weight="bold", color=INK, va="top")
    a.text(0, .35, "• the frozen atlas and its fingerprint\n• the V6 MSS layer (M is identical)\n"
                   "• preprocessing and NNLS projection",
           fontsize=6.6, color=MUTED, va="top", linespacing=1.7)
    a.text(0, .13, "What it costs", fontsize=8.4, weight="bold", color=VERM, va="top")
    a.text(0, .045, f"• L2 retains {float(IBL2.explained_variance_motif.iloc[0]):.2f} of motif variance\n"
                    "  vs 0.98 at V6's K=13",
           fontsize=6.6, color=MUTED, va="top", linespacing=1.7)
    fig.suptitle("V6 vs V6.2 — a richer representation at a measured information cost",
                 fontsize=11.0, weight="bold", y=.99)
    save(fig, "v62_f13_comparison")


# ═══ interactive ontology graph (HTML, plotly, no new deps) ═══
def f14_html():
    import plotly.graph_objects as go
    G = nx.DiGraph()
    for _, r in GE.iterrows():
        G.add_edge(r.source, r.target, weight=r.weight)
    lvl = GN.set_index("node")
    pos = {}
    for L, ys in ((1, IDS), (2, [f"L2·{t}" for t in L2]), (3, [f"L3·{t}" for t in L3])):
        for i, n in enumerate(ys):
            pos[n] = (L * 3.0, (i - (len(ys) - 1) / 2) * (12.0 / max(len(ys), 1)))
    ex, ey = [], []
    for u, v, d in G.edges(data=True):
        if u in pos and v in pos:
            ex += [pos[u][0], pos[v][0], None]; ey += [pos[u][1], pos[v][1], None]
    nodes = [n for n in G.nodes if n in pos]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="#c9d2db", width=1),
                             hoverinfo="none", showlegend=False))
    fig.add_trace(go.Scatter(
        x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes], mode="markers+text",
        text=[dict(zip(GN.node, GN.label)).get(n, n).replace("_", " ") for n in nodes],
        textposition="top center",
        textfont=dict(size=9),
        marker=dict(size=[10 if lvl.loc[n, "level"] == 1 else 24 for n in nodes],
                    color=[{1: "#0072B2", 2: "#009E73", 3: "#7B52AB"}[int(lvl.loc[n, "level"])]
                           for n in nodes], line=dict(color="white", width=1)),
        hovertext=[f"{n}<br>level {int(lvl.loc[n,'level'])}"
                   f"<br>betweenness {lvl.loc[n,'betweenness']:.3f}"
                   f"<br>eigenvector {lvl.loc[n,'eigenvector']:.3f}" for n in nodes],
        hoverinfo="text", showlegend=False))
    fig.update_layout(title="GAIRA V6.2 — biochemical ontology graph (motif → theme → system)",
                      template="simple_white", height=760,
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    p = OUT / "v62_ontology_graph.html"
    fig.write_html(str(p), include_plotlyjs="cdn")
    print("  wrote v62_ontology_graph.html")


if __name__ == "__main__":
    print("figures →", OUT)
    for f in (f01, f02, f03, f04, f05, f06, f07, f08, f09, f10, f11, f12, f13, f14_html):
        try:
            f()
        except Exception as e:                                          # noqa: BLE001
            import traceback
            print(f"  !! {f.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc(limit=2)
    print("done")
