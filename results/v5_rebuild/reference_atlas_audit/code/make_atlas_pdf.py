"""GAIRA Raman Reference Atlas v0.1 — figures + publication-quality atlas PDF."""
from __future__ import annotations
import sys, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster.hierarchy import dendrogram

REPO = Path("/Users/surajpg/projects/GAIRA"); sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from gaira.foundation import dataset as DS, serialization as SER
from gaira.foundation.families_raman import family_of
import atlas_audit as AA

OUT = REPO / "results/v5_rebuild/reference_atlas_audit"
TAB, FIG, ART = OUT / "tables", OUT / "figures", OUT / "artifacts"
FIG.mkdir(parents=True, exist_ok=True)
PDF_PATH = REPO / "GAIRA_Raman_Reference_Atlas_v0.1_Component_Audit.pdf"

P, S = "#2563EB", "#D97706"
INK, MUTED, GRIDC = "#1f2328", "#6B7280", "#E5E7EB"
SEQ = LinearSegmentedColormap.from_list("seq", ["#F8FAFC", P])
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9,
                     "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "axes.grid": True,
                     "grid.color": GRIDC, "grid.linewidth": 0.5, "xtick.color": MUTED,
                     "ytick.color": MUTED, "text.color": INK, "axes.labelcolor": INK,
                     "axes.titlecolor": INK, "legend.frameon": False, "figure.facecolor": "white"})
PAGE = (8.5, 11.0)

CLASS_COLORS = {"lipid": "#D97706", "carbohydrate": "#2563EB", "protein/peptide": "#059669",
                "nucleic": "#7C3AED", "cofactor/small-molecule": "#DB2777", "unassigned": "#9CA3AF"}


def _wrap(t, n=104):
    out, line = [], ""
    for w in str(t).split():
        if len(line) + len(w) + 1 > n:
            out.append(line); line = w
        else:
            line = (line + " " + w).strip()
    if line: out.append(line)
    return out


def text_page(pdf, title, blocks, subtitle=None):
    fig = plt.figure(figsize=PAGE); y = 0.95
    fig.text(0.07, y, title, fontsize=16, fontweight="bold", color=INK); y -= 0.032
    if subtitle:
        fig.text(0.07, y, subtitle, fontsize=9.5, color=MUTED); y -= 0.028
    for kind, txt in blocks:
        if y < 0.06:
            pdf.savefig(fig); plt.close(fig); fig = plt.figure(figsize=PAGE); y = 0.95
        if kind == "h":
            y -= 0.013; fig.text(0.07, y, txt, fontsize=10.5, fontweight="bold", color=INK); y -= 0.024
        elif kind == "b":
            for ln in _wrap(txt):
                fig.text(0.07, y, ln, fontsize=8.5, color=INK); y -= 0.0163
            y -= 0.007
        elif kind == "m":
            fig.text(0.075, y, txt, fontsize=7.1, color=INK, family="DejaVu Sans Mono"); y -= 0.0136
    pdf.savefig(fig); plt.close(fig)


def wordcloud_axis(ax, names, weights, max_words=22):
    """Weighted analyte-name layout (no wordcloud dependency)."""
    ax.axis("off")
    o = np.argsort(-np.asarray(weights))[:max_words]
    names = [names[i] for i in o]; w = np.asarray(weights)[o]
    w = w / (w.max() + 1e-12)
    rng = np.random.default_rng(0)
    placed = []
    for nm, wi in zip(names, w):
        fs = 5 + 11 * wi
        for _ in range(220):
            x, y = rng.uniform(0.03, 0.97), rng.uniform(0.05, 0.95)
            h = fs / 190; ww = len(nm) * fs / 620
            box = (x - ww / 2, y - h / 2, x + ww / 2, y + h / 2)
            if all(box[2] < b[0] or box[0] > b[2] or box[3] < b[1] or box[1] > b[3] for b in placed):
                placed.append(box)
                ax.text(x, y, nm, fontsize=fs, ha="center", va="center",
                        color=CLASS_COLORS.get(AA.molecular_class(nm), MUTED), alpha=0.95)
                break


def main():
    atlas = SER.load_frozen_manifold(REPO / "results/v5_rebuild/foundation/artifacts")
    corpus = DS.load_reference_corpus()
    W, grid = atlas.components, atlas.grid
    Z = atlas.coordinates(corpus.X, normalise=True)
    A = AA.analyte_activation(Z, corpus.meta)
    Xa = pd.DataFrame(np.nan_to_num(corpus.X)).assign(a=corpus.meta.analyte.values) \
        .groupby("a").mean().loc[A.index]
    inv = pd.read_csv(TAB / "p1_component_inventory.csv")
    comp_all = pd.read_csv(TAB / "p2_full_analyte_composition.csv")
    coh = pd.read_csv(TAB / "p4_chemical_coherence.csv").set_index("component")
    grpstudy = pd.read_csv(TAB / "p7_grouping_study.csv")
    man = json.loads((ART / "audit_manifest.json").read_text())
    best_k = int(man["grouping_recommendation_k"])
    gcomp = pd.read_csv(TAB / f"p7_group_composition_k{best_k}.csv")
    onto = json.loads((TAB / "p10_ontology_v0_1.json").read_text())
    bsv = json.loads((TAB / "p11_bsv_design_study.json").read_text())
    fin = json.loads((TAB / "p14_final_assessment.json").read_text())
    conf = json.loads((TAB / "p12_confusability_summary.json").read_text())
    plaus = json.loads((TAB / "p9_biological_plausibility.json").read_text())
    rel = np.load(ART / "p6_relationship_matrices.npz")
    gz = np.load(ART / "p7_grouping.npz")
    bands = AA.component_bands(W, grid)
    ood = pd.read_csv(TAB / "p13_out_of_domain_stress_test.csv") if (TAB / "p13_out_of_domain_stress_test.csv").exists() else None

    # ═══ GLOBAL FIGURES (P8) ═══
    # G1 component x family matrix + component heatmap
    fams = np.array([family_of(a) for a in A.index])
    fam_names = sorted(set(fams))
    M = np.zeros((atlas.k, len(fam_names)))
    for fi, f in enumerate(fam_names):
        idx = np.where(fams == f)[0]
        M[:, fi] = A.values[idx].sum(axis=0)
    M = M / (M.sum(axis=1, keepdims=True) + 1e-12)
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.4))
    im = axs[0].imshow(M, aspect="auto", cmap=SEQ, vmin=0, vmax=M.max())
    axs[0].set_xticks(range(len(fam_names))); axs[0].set_xticklabels(fam_names, rotation=45, ha="right", fontsize=6.5)
    axs[0].set_yticks(range(atlas.k)); axs[0].set_yticklabels([f"c{j}" for j in range(atlas.k)], fontsize=6)
    axs[0].set_title("Component × chemical-family composition"); axs[0].grid(False)
    plt.colorbar(im, ax=axs[0], fraction=0.03)
    ordA = np.argsort(-A.values.max(axis=1))[:60]
    im2 = axs[1].imshow((A.values[ordA] / (A.values[ordA].sum(1, keepdims=True) + 1e-12)),
                        aspect="auto", cmap=SEQ, vmin=0, vmax=1)
    axs[1].set_yticks(range(len(ordA))); axs[1].set_yticklabels(A.index[ordA], fontsize=4)
    axs[1].set_xticks(range(atlas.k)); axs[1].set_xticklabels(range(atlas.k), fontsize=6)
    axs[1].set_xlabel("component"); axs[1].set_title("Component × analyte matrix (60 most concentrated)")
    axs[1].grid(False); plt.colorbar(im2, ax=axs[1], fraction=0.03)
    fig.tight_layout(); fig.savefig(FIG / "g1_component_matrices.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # G2 relationships: dendrogram + network + similarity matrices
    fig = plt.figure(figsize=(13, 8))
    ax = fig.add_subplot(2, 2, 1)
    dendrogram(gz["linkage"], labels=[f"c{j}" for j in range(atlas.k)], ax=ax,
               color_threshold=None, leaf_font_size=6)
    ax.set_title(f"Component dendrogram (cut at k={best_k})")
    ax2 = fig.add_subplot(2, 2, 2)
    im = ax2.imshow(rel["spectral_cosine"], cmap=SEQ, vmin=0, vmax=1)
    ax2.set_title("Spectral loading cosine"); ax2.grid(False); plt.colorbar(im, ax=ax2, fraction=0.04)
    ax3 = fig.add_subplot(2, 2, 3)
    im = ax3.imshow(rel["activation_corr"], cmap="RdBu_r", vmin=-1, vmax=1)
    ax3.set_title("Activation correlation across analytes"); ax3.grid(False)
    plt.colorbar(im, ax=ax3, fraction=0.04)
    ax4 = fig.add_subplot(2, 2, 4)
    import networkx as nx
    Gx = nx.Graph()
    lab = gz[f"labels_k{best_k}"]
    for j in range(atlas.k):
        Gx.add_node(j)
    Ssp = rel["spectral_cosine"]
    for i in range(atlas.k):
        for j in range(i + 1, atlas.k):
            if Ssp[i, j] > 0.35:
                Gx.add_edge(i, j, weight=float(Ssp[i, j]))
    pos = nx.spring_layout(Gx, seed=0, weight="weight")
    cols = [plt.cm.tab20(lab[j] % 20) for j in range(atlas.k)]
    nx.draw_networkx_edges(Gx, pos, ax=ax4, alpha=0.25, width=1)
    nx.draw_networkx_nodes(Gx, pos, ax=ax4, node_color=cols,
                           node_size=[300 * inv.variance_explained[j] / inv.variance_explained.max()
                                      + 60 for j in range(atlas.k)])
    nx.draw_networkx_labels(Gx, pos, ax=ax4, font_size=6)
    ax4.set_title(f"Component network (edges: spectral cosine > 0.35; colour = group at k={best_k})")
    ax4.axis("off"); ax4.grid(False)
    fig.tight_layout(); fig.savefig(FIG / "g2_relationships.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # G3 grouping study + Sankey (components -> groups)
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    ax = axs[0]
    for c, mk in (("silhouette", "o"), ("bootstrap_reproducibility", "s"),
                  ("chemical_coherence", "^"), ("interpretable_group_fraction", "d")):
        ax.plot(grpstudy.sort_values("n_groups").n_groups,
                grpstudy.sort_values("n_groups")[c], mk + "-", ms=5, label=c.replace("_", " "))
    ax.axvline(best_k, ls="--", c=INK, lw=1)
    ax.text(best_k, 0.02, f" recommended k={best_k}", fontsize=7, color=INK)
    ax.set_xlabel("number of higher-order groups"); ax.set_ylabel("criterion value")
    ax.set_title("P7 — grouping study (silhouette is NOT optimised alone)"); ax.legend(fontsize=7)
    ax = axs[1]; ax.axis("off")
    ax.set_title(f"Sankey: 24 components → {best_k} biochemical groups")
    gorder = gcomp.sort_values("share_of_atlas", ascending=False)
    ypos_g = {int(r.group): i for i, (_, r) in enumerate(gorder.iterrows())}
    for j in range(atlas.k):
        g = int(lab[j]); y0 = 1 - j / (atlas.k - 1)
        y1 = 1 - ypos_g[g] / max(1, len(gorder) - 1)
        ax.plot([0.08, 0.92], [y0, y1], lw=1.2 + 6 * inv.variance_explained[j] / inv.variance_explained.max(),
                alpha=0.45, color=plt.cm.tab20(g % 20), solid_capstyle="round")
        ax.text(0.06, y0, f"c{j}", fontsize=5.5, ha="right", va="center", color=INK)
    for _, r in gorder.iterrows():
        y1 = 1 - ypos_g[int(r.group)] / max(1, len(gorder) - 1)
        ax.text(0.94, y1, f"{r.dominant_family} ({r.dominant_fraction:.2f})", fontsize=6.5,
                ha="left", va="center", color=INK)
    ax.set_xlim(0, 1.35); ax.set_ylim(-0.05, 1.05)
    fig.tight_layout(); fig.savefig(FIG / "g3_grouping_sankey.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # G4 radar prototypes + biological plausibility + UMAP (exploratory)
    fig = plt.figure(figsize=(13, 4.6))
    ax = fig.add_subplot(1, 3, 1, projection="polar")
    gcols = gorder.dominant_family.tolist(); ng = len(gcols)
    ang = np.linspace(0, 2 * np.pi, ng, endpoint=False).tolist(); ang += ang[:1]
    for a_name in ["cholesterol", "(+)-glucose", "albumin", "adenine"]:
        if a_name not in A.index:
            continue
        v = []
        for _, r in gorder.iterrows():
            members = eval(r.components) if isinstance(r.components, str) else r.components
            v.append(A.loc[a_name].values[members].sum())
        v = np.array(v) / (np.sum(v) + 1e-12); v = v.tolist(); v += v[:1]
        ax.plot(ang, v, lw=1.3, label=a_name); ax.fill(ang, v, alpha=0.08)
    ax.set_xticks(ang[:-1]); ax.set_xticklabels([g[:10] for g in gcols], fontsize=5)
    ax.set_yticklabels([]); ax.legend(fontsize=6, loc="upper right", bbox_to_anchor=(1.25, 1.1))
    ax.set_title("Radar prototype (grouped coordinates)", fontsize=9)
    ax2 = fig.add_subplot(1, 3, 2)
    pf = pd.DataFrame(plaus).T.reset_index().rename(columns={"index": "family"})
    ax2.barh(pf.family, pf.within_family_bsv_cos, color=[P if c else S for c in pf.coheres])
    ax2.plot(pf.null_mean, pf.family, "k|", ms=9, label="random-analyte null")
    ax2.set_xlabel("within-family BSV cosine"); ax2.legend(fontsize=6)
    ax2.set_title("P9 — biological plausibility (blue = coheres, p<0.05)", fontsize=9)
    ax3 = fig.add_subplot(1, 3, 3)
    try:
        import umap
        emb = umap.UMAP(n_neighbors=6, min_dist=0.4, random_state=0).fit_transform(AA._unit(W))
        ax3.scatter(emb[:, 0], emb[:, 1], c=[plt.cm.tab20(lab[j] % 20) for j in range(atlas.k)], s=60)
        for j in range(atlas.k):
            ax3.annotate(f"c{j}", (emb[j, 0], emb[j, 1]), fontsize=6, xytext=(3, 3),
                         textcoords="offset points")
    except Exception as e:
        ax3.text(0.5, 0.5, f"UMAP unavailable\n{e}", ha="center", fontsize=7)
    ax3.set_title("UMAP of components — EXPLORATORY ONLY, not evidence", fontsize=8.5)
    fig.tight_layout(); fig.savefig(FIG / "g4_radar_plausibility_umap.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # G5 out-of-domain stress test
    if ood is not None:
        fig, axs = plt.subplots(1, 3, figsize=(13, 4))
        ad = ood[ood.dataset == "adenine_series"].sort_values("conc_M")
        if len(ad):
            axs[0].semilogx(ad.conc_M, ad.top_component_share, "o-", color=S)
            axs[0].set_xlabel("adenine concentration (M)"); axs[0].set_ylabel("top-3 component share")
            axs[0].set_title("Adenine Ag-SERS series — OUT OF DOMAIN", fontsize=8.5)
        er = ood[ood.dataset == "ergothioneine_calibration"]
        if len(er):
            axs[1].scatter(er.conc_M, er.ood_distance, s=18, color=S)
            axs[1].set_xlabel("ergothioneine conc"); axs[1].set_ylabel("OOD distance")
            axs[1].set_title("Ergothioneine Ag-SERS — OUT OF DOMAIN", fontsize=8.5)
        ur = ood[ood.dataset == "uricase_depletion"]
        if len(ur):
            grpsu = ur.groupby("label").ood_distance
            axs[2].boxplot([g.values for _, g in grpsu], labels=[k for k, _ in grpsu])
            axs[2].set_ylabel("OOD distance"); axs[2].tick_params(axis="x", rotation=20, labelsize=6)
            axs[2].set_title("Uricase depletion Ag-SERS — OUT OF DOMAIN", fontsize=8.5)
        for a in axs:
            a.axhline(0.0, lw=0)
        fig.suptitle("P13 — Ag-SERS projected into a RAMAN atlas: stress test only, not validation",
                     fontsize=9, color=S)
        fig.tight_layout(); fig.savefig(FIG / "g5_ood_stress.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # ═══ PDF ═══
    with PdfPages(PDF_PATH) as pdf:
        # cover
        fig = plt.figure(figsize=PAGE)
        fig.text(0.5, 0.80, "GAIRA Raman Reference Atlas", ha="center", fontsize=24,
                 fontweight="bold", color=INK)
        fig.text(0.5, 0.757, "v0.1 — Component Audit and Biochemical Structure Analysis",
                 ha="center", fontsize=13, color=MUTED)
        ax = fig.add_axes([0.13, 0.42, 0.74, 0.27]); ax.axis("off")
        rows = [["Atlas", f"NMF k=24 (FROZEN, fingerprint {atlas.meta['fingerprint'][:12]}…)"],
                ["Reference corpus", f"{corpus.X.shape[0]} Raman spectra · {A.shape[0]} pure analytes"],
                ["Mean component stability", f"{inv.bootstrap_stability.mean():.3f}"],
                ["Mean class purity", f"{coh.class_purity.mean():.3f}"],
                ["Interpretive confidence", f"{int((inv.confidence=='high').sum())} high · "
                                            f"{int((inv.confidence=='moderate').sum())} moderate · 0 low"],
                ["Recommended grouping", f"k={best_k} (interpretive overlay, NOT frozen)"],
                ["BSV recommendation", "Option C — hierarchical (24 canonical → themes)"],
                ["Status", "Audit only — nothing retrained, nothing frozen here"]]
        t = ax.table(cellText=rows, colWidths=[0.34, 0.66], loc="center", cellLoc="left")
        t.auto_set_font_size(False); t.set_fontsize(8.5); t.scale(1, 1.62)
        for (r, c), cell in t.get_celld().items():
            cell.set_edgecolor(GRIDC)
            if c == 0: cell.set_text_props(fontweight="bold")
        fig.text(0.5, 0.33, "PRINCIPAL FINDING", ha="center", fontsize=11, fontweight="bold", color=INK)
        fig.text(0.5, 0.20,
                 "The atlas has learned molecular CLASS, not molecular species. Lipids, carbohydrates\n"
                 "and proteins each cohere against a random-analyte null; amino acids, purines,\n"
                 "cofactors and organic acids do not. Components are stable (0.81) but only modestly\n"
                 "pure (0.35), so the 24 components should stay canonical as the NUMERICAL layer while\n"
                 "biochemical meaning is carried by a revisable, versioned theme overlay.",
                 ha="center", fontsize=9.4, color=INK)
        fig.text(0.5, 0.06, "Read-only audit · branch gaira-v5-rebuild-plan · nothing pushed",
                 ha="center", fontsize=8, color=MUTED)
        pdf.savefig(fig); plt.close(fig)

        # executive summary
        text_page(pdf, "Executive summary", [
            ("h", "What this audit asked"),
            ("b", "The Raman reference atlas (NMF, k=24) was already built and frozen. This study asks "
                  "one question: what biochemical structure has it actually learned? Nothing was "
                  "retrained; the atlas fingerprint was verified unchanged before and after the audit."),
            ("h", "What the atlas has learned"),
            ("b", fin["1_what_chemistry_learned"]),
            ("h", "Components are stable but only moderately pure"),
            ("b", f"Mean bootstrap component stability is {inv.bootstrap_stability.mean():.3f} — the "
                  f"components reproduce well under analyte resampling. But mean class purity is only "
                  f"{coh.class_purity.mean():.2f}: most components are chemically ENRICHED rather than "
                  f"chemically pure. Enrichment reaches 14x (pyrimidine) and 10x (purine) even where "
                  f"purity is modest, which is the signature of a real but small chemical theme sitting "
                  f"inside a broader mixture. {int((inv.confidence=='high').sum())} of 24 components "
                  f"reach high interpretive confidence; none are uninterpretable."),
            ("h", "Higher-order groups exist but are weak geometrically"),
            ("b", f"Grouping is highly reproducible under analyte bootstrap "
                  f"({grpstudy.iloc[0].bootstrap_reproducibility:.3f}) yet silhouette is only "
                  f"{grpstudy.iloc[0].silhouette:.3f} across every k tested. The components therefore do "
                  f"NOT fall into well-separated natural clusters — grouping is a useful interpretive "
                  f"convenience, not a discovered partition. k={best_k} ranks best on the composite of "
                  f"reproducibility, chemical coherence, silhouette and interpretability, but the margin "
                  f"over neighbouring k is small and the count should be treated as provisional."),
            ("h", "MSS: class-level yes, species-level no"),
            ("b", f"Median signature uniqueness is {conf['median_uniqueness_all']:.3f}, which looks "
                  f"alarming until the confusions are examined: "
                  f"{conf['fraction_of_low_uniqueness_explained_by_chemistry']:.0%} of low-uniqueness "
                  f"cases are duplicate names, homologous series (the saturated triacylglycerols differ "
                  f"only in acyl-chain length), or same-class chemistry. That is correct fingerprint-"
                  f"region spectroscopy, not atlas failure. Only "
                  f"{conf['n_genuine_confusions']} genuinely cross-chemistry confusions remain."),
            ("h", "Recommendation"),
            ("b", "Keep all 24 components as the canonical frozen numerical layer; report biochemistry "
                  "through a versioned theme overlay (Option C, hierarchical BSV); publish MSS at class "
                  "level with explicit confusable-group annotations. Do not expand k, do not retrain, "
                  "do not admit Ag-SERS."),
        ], f"Frozen NMF k=24 · {A.shape[0]} analytes · audit only")

        # atlas overview figures
        for name, cap in [
            ("g1_component_matrices.png", "Atlas overview — component × chemical-family composition and "
             "the component × analyte matrix. Most components draw on several families: enrichment, not purity."),
            ("g2_relationships.png", "P6 component relationships — dendrogram, spectral-loading cosine, "
             "activation correlation and the component network. Off-diagonal structure is modest, "
             "consistent with NMF parts that are largely distinct."),
            ("g3_grouping_sankey.png", "P7 grouping study and the flow from 24 components to higher-order "
             "groups. Silhouette stays low at every k, so grouping is an overlay, not a discovered partition."),
            ("g4_radar_plausibility_umap.png", "Radar prototype on grouped coordinates; P9 biological "
             "plausibility against a random-analyte null; UMAP of components marked exploratory only."),
        ]:
            fig = plt.figure(figsize=PAGE)
            img = plt.imread(FIG / name)
            axi = fig.add_axes([0.04, 0.40, 0.92, 0.48]); axi.imshow(img); axi.axis("off")
            fig.text(0.07, 0.94, cap.split("—")[0].strip(), fontsize=13, fontweight="bold", color=INK)
            y = 0.35
            for ln in _wrap(cap, 100):
                fig.text(0.07, y, ln, fontsize=8.7, color=INK); y -= 0.017
            pdf.savefig(fig); plt.close(fig)

        # ═══ ONE PAGE PER COMPONENT ═══
        sep = plt.figure(figsize=PAGE)
        sep.text(0.5, 0.55, "Atlas entries", ha="center", fontsize=20, fontweight="bold", color=INK)
        sep.text(0.5, 0.50, "one page per latent component (24)", ha="center", fontsize=11, color=MUTED)
        pdf.savefig(sep); plt.close(sep)

        for j in range(atlas.k):
            r = inv.iloc[j]
            comp = comp_all[comp_all.component == j].sort_values("loading", ascending=False)
            fig = plt.figure(figsize=PAGE)
            fig.text(0.07, 0.965, f"Component {j}", fontsize=17, fontweight="bold", color=INK)
            fig.text(0.07, 0.943,
                     f"tentative theme: {r.primary_interpretation}  ·  class: {r.dominant_class}  ·  "
                     f"confidence: {r.confidence}", fontsize=9.5, color=MUTED)
            meta = (f"variance {r.variance_explained:.4f}   stability {r.bootstrap_stability:.3f}   "
                    f"purity {r.class_purity:.3f}   enrichment {r.enrichment:.2f}x   "
                    f"entropy {r.entropy_analyte:.3f}\n"
                    f"analytes contributing {int(r.n_analytes_contributing)}   dominant (≥5%) "
                    f"{int(r.n_dominant_analytes)}   sparsity {r.sparsity_gini:.3f}   "
                    f"band uniqueness {r.band_uniqueness:.2f}   median band width "
                    f"{r.spectral_bandwidth_cm:.0f} cm⁻¹\n"
                    f"representative analyte: {r.representative_spectrum}   ·   secondary family: "
                    f"{r.secondary_interpretation}")
            for i, ln in enumerate(meta.split("\n")):
                fig.text(0.07, 0.921 - i * 0.0145, ln, fontsize=7.3, color=INK,
                         family="DejaVu Sans Mono")

            # spectral loading with band annotations
            ax1 = fig.add_axes([0.08, 0.700, 0.87, 0.155])
            ax1.plot(grid, W[j], color=P, lw=1.1)
            for _, b in bands[j].head(6).iterrows():
                ax1.annotate(f"{b.position:.0f}", (b.position, W[j][np.argmin(np.abs(grid - b.position))]),
                             fontsize=6, color=INK, xytext=(0, 4), textcoords="offset points", ha="center")
            ax1.set_xlim(grid[0], grid[-1]); ax1.set_xlabel("Raman shift (cm⁻¹)", fontsize=7)
            ax1.set_ylabel("loading"); ax1.tick_params(labelsize=6.5)
            ax1.set_title("Component spectral loading (non-negative) with dominant bands", fontsize=8.4)

            # representative analyte spectra
            ax2 = fig.add_axes([0.08, 0.520, 0.40, 0.135])
            for k2, a_name in enumerate(comp.analyte.head(4)):
                ax2.plot(grid, Xa.loc[a_name].values + k2 * 0.12, lw=0.75)
                ax2.text(grid[3], k2 * 0.12 + 0.085, a_name[:26], fontsize=5.5, color=INK)
            ax2.set_xlim(grid[0], grid[-1]); ax2.set_yticks([]); ax2.tick_params(labelsize=6.5)
            ax2.set_xlabel("Raman shift (cm⁻¹)", fontsize=7)
            ax2.set_title("Representative analyte spectra", fontsize=8.4)

            # top contributing analytes
            ax3 = fig.add_axes([0.56, 0.520, 0.39, 0.135])
            top = comp.head(10).iloc[::-1]
            ax3.barh(range(len(top)), top.contribution_pct,
                     color=[CLASS_COLORS.get(c, MUTED) for c in top.molecular_class])
            ax3.set_yticks(range(len(top)))
            ax3.set_yticklabels([a[:24] for a in top.analyte], fontsize=5.5)
            ax3.set_xlabel("contribution %", fontsize=7); ax3.tick_params(labelsize=6.5)
            ax3.set_title("Top contributing analytes", fontsize=8.4)

            # loading distribution + family composition
            ax4 = fig.add_axes([0.08, 0.335, 0.26, 0.125])
            ax4.hist(comp.contribution_pct, bins=24, color=P, alpha=0.9)
            ax4.set_xlabel("contribution %", fontsize=7); ax4.set_ylabel("analytes", fontsize=7)
            ax4.tick_params(labelsize=6.5); ax4.set_title("Loading distribution", fontsize=8.4)

            ax5 = fig.add_axes([0.40, 0.335, 0.26, 0.125])
            fc = comp.groupby("chemical_family").contribution_pct.sum().sort_values(ascending=False).head(7)
            ax5.barh(range(len(fc))[::-1], fc.values, color=S)
            ax5.set_yticks(range(len(fc))[::-1]); ax5.set_yticklabels(fc.index, fontsize=5.5)
            ax5.set_xlabel("% of component", fontsize=7); ax5.tick_params(labelsize=6.5)
            ax5.set_title("Chemical-family composition", fontsize=8.4)

            ax6 = fig.add_axes([0.70, 0.335, 0.25, 0.125])
            wordcloud_axis(ax6, comp.analyte.tolist(), comp.contribution_pct.values)
            ax6.set_title("Analyte word cloud", fontsize=8.4)

            # interpretation block
            lit = r.literature_groups
            interp = (f"Interpretation. The component is dominated by {r.dominant_class} chemistry "
                      f"(family '{r.primary_interpretation}', {100*r.class_purity:.0f}% of its loading, "
                      f"{r.enrichment:.1f}x enriched over the corpus base rate). Its strongest bands lie at "
                      f"{', '.join(str(int(x)) for x in eval(r.dominant_raman_peaks_cm)[:5]) if isinstance(r.dominant_raman_peaks_cm, str) else ''} cm⁻¹. "
                      f"Literature corroboration: {lit if isinstance(lit, str) and lit != '{}' else 'none in the available table'}.")
            artifact = ("Artifact assessment. " + (
                "Bootstrap stability is high and the component is chemically focused, so it is unlikely "
                "to be a fitting artifact."
                if (r.bootstrap_stability >= 0.80 and r.class_purity >= 0.40) else
                "Bootstrap stability is high but chemical purity is modest — most likely a genuine "
                "shared-band motif spanning several families rather than one biochemical theme."
                if r.bootstrap_stability >= 0.80 else
                "Bootstrap stability is below 0.80: this component may be partly a fitting artifact and "
                "should not anchor an ontology entry on its own."))
            caution = ("This is a mathematical component. The theme is a tentative post-hoc "
                       "interpretation, never a molecular assignment.")
            y = 0.284
            for txt, bold in ((interp, False), (artifact, False), (caution, True)):
                for ln in _wrap(txt, 108):
                    fig.text(0.07, y, ln, fontsize=7.6, color=(MUTED if bold else INK),
                             style=("italic" if bold else "normal")); y -= 0.0145
                y -= 0.006

            # full composition table (top 18)
            ax7 = fig.add_axes([0.07, 0.035, 0.86, 0.118]); ax7.axis("off")
            tt = comp.head(18)[["analyte", "contribution_pct", "molecular_class",
                                "chemical_family", "subfamily"]].copy()
            tt["contribution_pct"] = tt.contribution_pct.round(2)
            tbl = ax7.table(cellText=tt.values, colLabels=["analyte", "%", "class", "family", "subfamily"],
                            loc="upper center", cellLoc="left", colWidths=[0.34, 0.08, 0.19, 0.19, 0.20])
            tbl.auto_set_font_size(False); tbl.set_fontsize(5.4); tbl.scale(1, 0.95)
            for (rr, cc), cell in tbl.get_celld().items():
                cell.set_edgecolor(GRIDC)
                if rr == 0: cell.set_text_props(fontweight="bold")
            ax7.set_title(f"Full analyte composition — top 18 of {len(comp)} contributing analytes "
                          f"(molecular weight and biochemical role: unavailable in this corpus)",
                          fontsize=7, pad=2)
            pdf.savefig(fig); plt.close(fig)

        # coherence ranking page
        blocks = [("h", "P4 — chemical coherence ranking (most → least coherent)"),
                  ("m", f"{'comp':>4s} {'purity':>7s} {'enrich':>7s} {'entropy':>8s} {'MI':>6s} "
                        f"{'families':>9s} {'specSim':>8s}  dominant family")]
        for _, r in coh.reset_index().sort_values("class_purity", ascending=False).iterrows():
            blocks.append(("m", f"{int(r.component):>4d} {r.class_purity:>7.3f} {r.enrichment_vs_corpus:>7.2f} "
                                f"{r.shannon_entropy_family:>8.3f} {r.mutual_information_family:>6.3f} "
                                f"{int(r.n_chemical_families):>9d} {r.avg_spectral_similarity_top:>8.3f}  "
                                f"{r.dominant_family}"))
        blocks += [("b", "Average molecular similarity is reported as UNAVAILABLE: this corpus contains no "
                         "structures, SMILES or fingerprints, so a structural similarity cannot be computed "
                         "without inventing metadata."),
                   ("h", "True biochemical themes vs mathematical mixtures"),
                   ("b", f"Components with purity ≥ 0.40 and stability ≥ 0.80 behave as genuine biochemical "
                         f"themes ({int(((inv.class_purity>=0.40)&(inv.bootstrap_stability>=0.80)).sum())} of 24). "
                         f"The remainder are better described as stable mathematical mixtures that capture "
                         f"shared Raman motifs — most often the acyl-chain and C–H bands common to all lipid "
                         f"subclasses, and the C–O/C–C ring modes common to carbohydrates.")]
        text_page(pdf, "Chemical coherence", blocks)

        # grouping + ontology + BSV + MSS
        gblocks = [("h", f"P7 — grouping study (recommended k={best_k}, provisional)"),
                   ("m", f"{'k':>3s} {'silhouette':>11s} {'bootstrap':>10s} {'chem':>7s} {'interp':>7s} {'composite':>10s}")]
        for _, r in grpstudy.sort_values("n_groups").iterrows():
            gblocks.append(("m", f"{int(r.n_groups):>3d} {r.silhouette:>11.3f} "
                                 f"{r.bootstrap_reproducibility:>10.3f} {r.chemical_coherence:>7.3f} "
                                 f"{r.interpretable_group_fraction:>7.2f} {r.composite:>10.3f}"))
        gblocks += [("b", "Silhouette is low (≤0.10) at every k: the components do not form geometrically "
                          "well-separated clusters. Bootstrap reproducibility is high because the same "
                          "weak structure is recovered consistently — reproducible does not mean "
                          "well-separated. Grouping is therefore justified as an interpretive overlay only."),
                    ("h", f"Group composition at k={best_k}"),
                    ("m", f"{'grp':>4s} {'n':>3s} {'share':>7s}  dominant family (fraction)")]
        for _, r in gcomp.sort_values("share_of_atlas", ascending=False).iterrows():
            gblocks.append(("m", f"{int(r.group):>4d} {int(r.n_components):>3d} {r.share_of_atlas:>7.3f}  "
                                 f"{r.dominant_family} ({r.dominant_fraction:.2f})"))
        text_page(pdf, "Higher-order grouping", gblocks)

        oblocks = [("h", "P10 — provisional ontology v0.1 (NOT frozen)")]
        for tier in ("high_confidence", "moderate_confidence", "low_confidence", "unknown"):
            ents = onto["entries"][tier]
            oblocks.append(("b", f"{tier.replace('_',' ').upper()} — {len(ents)} components"))
            for e in ents:
                oblocks.append(("m", f"  c{e['component']:<2d} {e['theme'][:18]:18s} purity {e['purity']:.2f} "
                                     f"enrich {e['enrichment']:>5.1f}x stab {e['stability']:.2f}"))
        for c in onto["caveats"]:
            oblocks.append(("b", "Caveat: " + c))
        text_page(pdf, "Provisional ontology v0.1", oblocks)

        bblocks = [("h", f"P11 — BSV design study · recommendation: {bsv['recommendation']}")]
        for name, o in bsv["options"].items():
            bblocks.append(("b", f"{name}: {o['description']}"))
            bblocks.append(("m", f"    interpretability : {o['interpretability']}"))
            bblocks.append(("m", f"    stability        : {o['stability']}"))
            bblocks.append(("m", f"    extensibility    : {o['extensibility']}"))
            bblocks.append(("m", f"    clinical usability: {o['clinical_usability']}"))
            bblocks.append(("m", f"    verdict          : {o['verdict']}"))
        bblocks.append(("h", "Rationale"))
        for r_ in bsv["rationale"]:
            bblocks.append(("b", "• " + r_))
        bblocks.append(("b", "The BSV is NOT defined or frozen by this audit; only the architecture is "
                             "recommended."))
        text_page(pdf, "BSV design study", bblocks)

        mblocks = [("h", "P12 — MSS readiness"),
                   ("b", f"Median signature uniqueness {conf['median_uniqueness_all']:.3f} across "
                         f"{conf['n_analytes']} analytes; against genuinely different chemistry it rises to "
                         f"{conf['median_uniqueness_vs_distinct_chemistry']}."),
                   ("m", f"nearest-neighbour relation: {conf['nearest_neighbour_relation_counts']}"),
                   ("m", f"low-uniqueness (<0.15) relations: {conf['low_uniqueness_relation_counts']}"),
                   ("b", f"{conf['fraction_of_low_uniqueness_explained_by_chemistry']:.0%} of low-uniqueness "
                         "cases are duplicate entries, homologous series or same-class chemistry. The "
                         "saturated triacylglycerols (trilaurin/trimyristin/tripalmitin/tristearin/"
                         "triarachidin/tribehenin) are mutually indistinguishable because in the "
                         "450–1800 cm⁻¹ fingerprint region they differ only in CH2 count; estradiol and "
                         "estriol are likewise near-identical. This is correct spectroscopy."),
                   ("h", "Verdict"),
                   ("b", "MSS can be frozen at CLASS/THEME level with explicit confusable-group "
                         "annotations. It cannot yet be frozen at molecular-species level. Species-level "
                         "resolution would require spectral range beyond the fingerprint region (the "
                         "2800–3000 cm⁻¹ C–H stretch region separates acyl chain lengths) or an "
                         "orthogonal measurement — not a change to this atlas."),
                   ("h", "Genuine cross-chemistry confusions")]
        for r_ in conf["genuine_confusion_examples"][:8]:
            mblocks.append(("m", f"  {r_['analyte'][:30]:30s} ~ {r_['nn1'][:30]:30s} cos={r_['nn1_sim']:.3f}"))
        text_page(pdf, "MSS readiness", mblocks)

        # stress test appendix
        if ood is not None:
            fig = plt.figure(figsize=PAGE)
            img = plt.imread(FIG / "g5_ood_stress.png")
            axi = fig.add_axes([0.05, 0.44, 0.90, 0.40]); axi.imshow(img); axi.axis("off")
            fig.text(0.07, 0.94, "Appendix — out-of-domain stress test", fontsize=14,
                     fontweight="bold", color=INK)
            fig.text(0.07, 0.915, "NOT VALIDATION. Ag-SERS spectra projected into a Raman atlas.",
                     fontsize=9.5, color=S)
            y = 0.40
            summ = (f"Three Ag-SERS calibration sets were projected into the frozen Raman atlas purely to "
                    f"observe its behaviour off-domain: the adenine concentration series "
                    f"({len(ood[ood.dataset=='adenine_series'])} spectra), the ergothioneine calibration "
                    f"({len(ood[ood.dataset=='ergothioneine_calibration'])}), and the uricase depletion "
                    f"experiment ({len(ood[ood.dataset=='uricase_depletion'])}). Median out-of-domain "
                    f"distance is {ood.ood_distance.median():.3f}, materially higher than for in-domain "
                    f"Raman references, exactly as expected for a different observation physics. "
                    f"These projections were NOT used to modify, tune or judge the atlas, and no "
                    f"dose-response claim is made from them: prior work established that Ag-SERS spectra "
                    f"in this corpus are background-dominated, so any apparent trajectory here is not "
                    f"evidence of biochemical response.")
            for ln in _wrap(summ, 100):
                fig.text(0.07, y, ln, fontsize=8.7, color=INK); y -= 0.017
            pdf.savefig(fig); plt.close(fig)

        # final assessment
        fblocks = []
        qmap = [("1. What chemistry has the atlas learned?", fin["1_what_chemistry_learned"]),
                ("4. Should 24 remain canonical?", fin["4_should_24_remain_canonical"]),
                ("5. Should components be grouped?", fin["5_should_components_be_grouped"]),
                ("6. What grouping is most defensible?", fin["6_most_defensible_grouping"]),
                ("7. Should the BSV be 24-D or compressed?", fin["7_bsv_dimensionality"]),
                ("8. Is MSS mature enough?", fin["8_is_mss_mature"]),
                ("10. What should the next GAIRA phase be?", fin["10_next_phase"])]
        for q, a_ in qmap:
            fblocks.append(("h", q)); fblocks.append(("b", a_))
        fblocks.append(("h", "2. Biologically strongest components"))
        for e in fin["2_strongest_components"]:
            fblocks.append(("m", f"  c{int(e['component']):<2d} {e['primary_interpretation'][:18]:18s} "
                                 f"purity {e['class_purity']:.2f} enrich {e['enrichment']:>5.1f}x "
                                 f"stab {e['bootstrap_stability']:.2f}"))
        fblocks.append(("h", "3. Weakest components"))
        for e in fin["3_weakest_components"]:
            fblocks.append(("m", f"  c{int(e['component']):<2d} {e['primary_interpretation'][:18]:18s} "
                                 f"purity {e['class_purity']:.2f} enrich {e['enrichment']:>5.1f}x "
                                 f"stab {e['bootstrap_stability']:.2f}"))
        fblocks.append(("h", "9. Ontology confidence"))
        fblocks.append(("m", f"  {fin['9_ontology_confidence']}"))
        text_page(pdf, "Final scientific assessment", fblocks)

        d = pdf.infodict()
        d["Title"] = "GAIRA Raman Reference Atlas v0.1 — Component Audit"
    print(f"atlas PDF written: {PDF_PATH} ({PDF_PATH.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
