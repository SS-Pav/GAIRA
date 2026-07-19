"""Perturbation Response Audit — figures + response-atlas PDF (Part 13 + report)."""
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
import response_lib as RL

OUT = REPO / "results/v5_rebuild/perturbation_response"
TAB, FIG, ART = OUT / "tables", OUT / "figures", OUT / "artifacts"
FIG.mkdir(parents=True, exist_ok=True)
PDF_PATH = REPO / "GAIRA_Raman_Reference_Atlas_v0.1_Perturbation_Response_Audit.pdf"
SPIKE = RL.SPIKE
K = 24

P, S = "#2563EB", "#D97706"
INK, MUTED, GRIDC = "#1f2328", "#6B7280", "#E5E7EB"
DIV = LinearSegmentedColormap.from_list("div", [P, "#F7F7F7", S])
SEQ = LinearSegmentedColormap.from_list("seq", ["#F8FAFC", P])
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9,
                     "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "axes.grid": True,
                     "grid.color": GRIDC, "grid.linewidth": 0.5, "xtick.color": MUTED,
                     "ytick.color": MUTED, "text.color": INK, "axes.labelcolor": INK,
                     "axes.titlecolor": INK, "legend.frameon": False, "figure.facecolor": "white"})
PAGE = (8.5, 11.0)
FAMCOL = {"purine": "#7C3AED", "pyrimidine": "#9333EA", "cofactor": "#DB2777",
          "amino_acid": "#059669", "saccharide": P, "lipid": S, "organic_acid": "#0891B2",
          "protein": "#065F46", "nucleic_acid": "#6D28D9"}


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
            fig.text(0.075, y, txt, fontsize=7.0, color=INK, family="DejaVu Sans Mono"); y -= 0.0134
    pdf.savefig(fig); plt.close(fig)


def main():
    ctx = RL.load_atlas_context()
    themes = ctx["themes"]
    comp_load = RL.load_component_reference_loadings()
    fp = pd.read_csv(TAB / "part2_response_fingerprints.csv")
    dcols = [f"d{j}" for j in range(K)]
    dose = pd.read_csv(TAB / "part1_component_dose_response.csv")
    spec = pd.read_csv(TAB / "part4_component_specificity.csv")
    cons = pd.read_csv(TAB / "part5_analyte_consistency.csv").set_index("analyte")
    pur = pd.read_csv(TAB / "part6_purine_similarity.csv", index_col=0)
    uric = json.loads((TAB / "part8_uricase.json").read_text())
    p9 = pd.read_csv(TAB / "part9_serum_responders.csv")
    rob = pd.read_csv(TAB / "part11_component_robustness.csv")
    clus = pd.read_csv(TAB / "part10_clusters.csv")
    hubs = json.loads((TAB / "part14_hubs.json").read_text())
    mech = json.loads((TAB / "part15_mechanistic_assessment.json").read_text())
    dart = json.loads((TAB / "part16_dart_implications.json").read_text())
    trajlib = json.loads((ART / "trajectory_library.json").read_text())
    fam = json.loads((TAB / "part10_response_families.json").read_text())
    Zg = np.load(ART / "fingerprint_linkage.npz", allow_pickle=True)

    # projections + processed spectra for the atlas pages
    proj = {d: RL.load_projection(d) for d in RL.DATASETS}
    blob = np.load(SPIKE / "artifacts/processed_spectra.npz")
    Xs = blob["X_spiked_serum"]; _, ms, _ = proj["spiked_serum"]
    Xb = blob["X_serum_baseline"]
    from gaira.foundation import dataset as DS
    grid = DS.GRID

    F = fp[dcols].values
    U = RL._unit(F)
    analytes = fp.analyte.tolist()
    fams = fp.family.values

    # ── FIG1 fingerprint heatmap (clustered) ──
    order = list(map(int, dendrogram(Zg["linkage"], no_plot=True)["ivl"]))
    fig, ax = plt.subplots(figsize=(12, 8))
    Fo = F[order]
    v = np.percentile(np.abs(Fo), 98)
    im = ax.imshow(Fo, aspect="auto", cmap=DIV, vmin=-v, vmax=v)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([analytes[i] for i in order], fontsize=4.5)
    ax.set_xticks(range(K)); ax.set_xticklabels([f"c{j}" for j in range(K)], fontsize=6)
    for t, i in zip(ax.get_yticklabels(), order):
        t.set_color(FAMCOL.get(fams[i], INK))
    ax.set_xlabel("latent component"); ax.set_title("Part 2 — serum-spike response fingerprints "
                                                    "(Δ activation), analytes clustered by response")
    ax.grid(False); plt.colorbar(im, ax=ax, fraction=0.02)
    fig.tight_layout(); fig.savefig(FIG / "f1_fingerprints.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # ── FIG2 component dose-response + specificity ──
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
    ax = axs[0]
    Z, m, _ = proj["ils_adenine"]
    sub = m[(m.substrate == "cAg") & (m.laser_nm == 785)]
    pos = [m.index.get_loc(i) for i in sub.index]
    for j in [3, 13, 15, 2]:
        r = RL.component_dose_response(Z[pos], sub.conc_uM.values, j)
        ax.plot(r["levels"], np.array(r["mean"]) - r["mean"][0], "-o", ms=3,
                label=f"c{j} ({themes.get(j)})")
    ax.set_xlabel("adenine conc (µM)"); ax.set_ylabel("Δ component activation")
    ax.set_title("Part 1 — adenine cAg@785: selective response"); ax.legend(fontsize=6.5)
    ax = axs[1]
    sp = spec.sort_values("activation_gini")
    ax.barh(range(len(sp)), sp.activation_gini,
            color=[{"specific": P, "generic": S, "intermediate": MUTED}[x] for x in sp.specificity])
    ax.set_yticks(range(len(sp))); ax.set_yticklabels([f"c{int(c)}" for c in sp.component], fontsize=5.5)
    ax.set_xlabel("activation Gini (higher = more specific)")
    ax.set_title("Part 4 — component specificity")
    ax = axs[2]
    ax.scatter(rob.math_stability, rob.responsiveness_ratio, s=40,
               c=[P if c == "high" else MUTED for c in rob.confidence])
    for _, r in rob.nlargest(6, "anchor_score").iterrows():
        ax.annotate(f"c{int(r.component)}", (r.math_stability, r.responsiveness_ratio),
                    fontsize=6.5, xytext=(3, 2), textcoords="offset points")
    ax.set_xlabel("mathematical stability (audit)"); ax.set_ylabel("responsiveness ratio (spike/control)")
    ax.set_title("Part 11 — robustness: anchor candidates (blue = high-confidence theme)")
    fig.tight_layout(); fig.savefig(FIG / "f2_dose_specificity.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # ── FIG3 case studies ──
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
    ax = axs[0]
    im = ax.imshow(pur.values, cmap=DIV, vmin=-1, vmax=1)
    ax.set_xticks(range(len(pur))); ax.set_xticklabels(pur.columns, rotation=40, ha="right", fontsize=7)
    ax.set_yticks(range(len(pur))); ax.set_yticklabels(pur.index, fontsize=7)
    for i in range(len(pur)):
        for k in range(len(pur)):
            ax.text(k, i, f"{pur.values[i,k]:.2f}", ha="center", va="center", fontsize=6,
                    color=INK if abs(pur.values[i,k]) < 0.6 else "white")
    ax.set_title("Part 6 — purine response similarity\n(two anti-correlated pairs)"); ax.grid(False)
    ax = axs[1]
    Ze, me, _ = proj["ergothioneine"]
    for j in [15, 17, 19, 2]:
        r = RL.component_dose_response(Ze, me.conc_uM.values, j)
        ax.plot(r["levels"], np.array(r["mean"]) - r["mean"][0], "-o", ms=3, label=f"c{j}")
    ax.set_xlabel("ergothioneine conc"); ax.set_ylabel("Δ activation")
    ax.set_title("Part 7 — ergothioneine dose-response"); ax.legend(fontsize=6.5)
    ax = axs[2]
    d = np.array([uric["delta_components"][str(j)] for j in range(K)])
    ax.bar(range(K), d, color=[S if x < 0 else P for x in d])
    ax.axhline(0, color=INK, lw=0.6)
    ax.annotate("c15\n(purine)", (15, d[15]), fontsize=6.5, ha="center",
                va="top" if d[15] < 0 else "bottom")
    ax.set_xlabel("component"); ax.set_ylabel("Δ (spiked+uricase − spiked)")
    ax.set_title(f"Part 8 — uricase depletion (selective={uric['selective']})")
    fig.tight_layout(); fig.savefig(FIG / "f3_case_studies.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # ── FIG4 responders, families, network ──
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.4))
    ax = axs[0]
    for r, sub in p9.groupby("responder"):
        ax.scatter(sub.activation_norm, sub.response_entropy, s=34,
                   color=P if r else MUTED, label="responder" if r else "non-responder", alpha=0.85)
    for _, rr in p9[p9.responder].iterrows():
        ax.annotate(rr.analyte, (rr.activation_norm, rr.response_entropy), fontsize=6,
                    xytext=(3, 2), textcoords="offset points")
    ax.set_xlabel("response magnitude"); ax.set_ylabel("response entropy")
    ax.set_title("Part 9 — responders move MORE, not more focused"); ax.legend(fontsize=7)
    ax = axs[1]
    ax.bar(["response\nfingerprints", "raw spike\nspectra"],
           [fam["response_fingerprint_best_ari"], fam["raw_spectrum_best_ari"]], color=[P, S])
    ax.set_ylabel("family-recovery ARI")
    ax.set_title("Part 10 — response recovers chemistry\nbetter than raw spectra")
    ax = axs[2]
    import networkx as nx
    ed = pd.read_csv(TAB / "part14_bipartite_edges.csv")
    ed = ed[ed.weight.abs() > ed.weight.abs().quantile(0.6)]
    G = nx.Graph()
    for _, r in ed.iterrows():
        G.add_node(f"c{int(r.component)}", kind="comp")
        G.add_node(r.analyte, kind="analyte")
        G.add_edge(f"c{int(r.component)}", r.analyte, weight=abs(r.weight))
    posn = nx.spring_layout(G, seed=0, k=0.4)
    comp_nodes = [n for n in G if str(n).startswith("c") and n[1:].isdigit()]
    an_nodes = [n for n in G if n not in comp_nodes]
    nx.draw_networkx_edges(G, posn, ax=ax, alpha=0.15)
    nx.draw_networkx_nodes(G, posn, nodelist=comp_nodes, node_color=S, node_size=90, ax=ax)
    nx.draw_networkx_nodes(G, posn, nodelist=an_nodes, node_color=P, node_size=12, ax=ax)
    nx.draw_networkx_labels(G, posn, labels={n: n for n in comp_nodes}, font_size=5, ax=ax)
    ax.set_title(f"Part 14 — component–analyte network\n(hubs: {list(hubs['component_hubs'])[:4]})")
    ax.axis("off"); ax.grid(False)
    fig.tight_layout(); fig.savefig(FIG / "f4_families_network.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # ═══ PDF ═══
    with PdfPages(PDF_PATH) as pdf:
        fig = plt.figure(figsize=PAGE)
        fig.text(0.5, 0.80, "Perturbation Response Audit", ha="center", fontsize=22,
                 fontweight="bold", color=INK)
        fig.text(0.5, 0.757, "GAIRA Raman Reference Atlas v0.1 — linking perturbations to latent motifs",
                 ha="center", fontsize=12, color=MUTED)
        ax = fig.add_axes([0.12, 0.42, 0.76, 0.27]); ax.axis("off")
        rows = [["Atlas", f"NMF k=24, FROZEN ({ctx['fingerprint'][:12]}…, verified unchanged)"],
                ["Response fingerprints", "53 serum-spike analytes, 24-D Δ-activation each"],
                ["Dose experiments", "6 ILS adenine arms + ergothioneine (component-resolved)"],
                ["Loop closure", "adenine → c3 (its own encoding component) in 4/6 arms"],
                ["Purine sub-structure", "recovered: {adenine,hypoxanthine} vs {xanthine,guanine}"],
                ["Uricase depletion", "selective loss in a purine-encoding component"],
                ["Key caveat", "theme LABELS unreliable on low-purity components; identities are not"],
                ["Atlas modified?", "No"]]
        t = ax.table(cellText=rows, colWidths=[0.30, 0.70], loc="center", cellLoc="left")
        t.auto_set_font_size(False); t.set_fontsize(8.2); t.scale(1, 1.6)
        for (rr, cc), cell in t.get_celld().items():
            cell.set_edgecolor(GRIDC)
            if cc == 0: cell.set_text_props(fontweight="bold")
        fig.text(0.5, 0.33, "PRINCIPAL FINDING", ha="center", fontsize=11, fontweight="bold", color=INK)
        fig.text(0.5, 0.185,
                 "The loop closes at the level of component IDENTITY, not theme label. When adenine is\n"
                 "perturbed, the single strongest-responding latent component is the one that actually\n"
                 "encodes adenine in the reference atlas (c3) — reproducibly across substrates and lasers —\n"
                 "even though the audit's coarse label for c3 is 'sterol'. Perturbation response is a\n"
                 "sharper probe of component identity than the static audit labels, and it recovers a\n"
                 "purine sub-classification the atlas was never given.",
                 ha="center", fontsize=9.3, color=INK)
        fig.text(0.5, 0.06, "Mechanistic audit · atlas frozen · nothing pushed", ha="center",
                 fontsize=8, color=MUTED)
        pdf.savefig(fig); plt.close(fig)

        text_page(pdf, "Executive summary", [
            ("h", "Question"),
            ("b", "The Component Audit said what each latent Raman motif is; the Spike Validation said "
                  "perturbations move through the atlas. This study connects them: when a known analyte "
                  "is perturbed, which latent motifs respond, and does the response match the motif's "
                  "chemistry? The atlas was frozen and verified unchanged."),
            ("h", "Observed results"),
            ("b", "1. The response is selective, not a global amplitude shift. In the strongest adenine "
                  "arm only 7 of 24 components rise while 17 fall; the two largest movers are c3 and c13, "
                  "and c3's top reference analyte is adenine itself."),
            ("b", "2. Loop closure at component identity. For adenine the single strongest-responding "
                  "component is its own encoding component c3 in 4 of 6 substrate×laser arms — a "
                  "reproducible closure across acquisition conditions."),
            ("b", "3. Theme labels are the weak link, not the components. Theme-label agreement is only "
                  "15/128 responsive-component instances, because low-purity components carry misleading "
                  "coarse labels (c3='sterol' but encodes adenine; c13='pyrimidine' but is thymine-"
                  "dominated). The components are chemically real; the labels are not reliable."),
            ("b", f"4. Purine sub-structure recovered. Response fingerprints split the five purines into "
                  f"{{adenine, hypoxanthine}} (cos {pur.loc['adenine','hypoxanthine']:.2f}) and "
                  f"{{xanthine, guanine}} (cos {pur.loc['xanthine','guanine']:.2f}), anti-correlated "
                  f"across the divide — the 6-oxo/amino versus 2,6-dioxo distinction, which the atlas was "
                  f"never told about."),
            ("b", f"5. Uricase depletion is selective. Enzymatic urate removal produces a targeted change "
                  f"in a purine-encoding component (c15 Δ {uric['purine_component_c15_change']:+.3f}) "
                  f"rather than a global shift — mechanistically consistent with removing a purine."),
            ("b", f"6. Response fingerprints recover chemical family better than raw spectra "
                  f"(ARI {fam['response_fingerprint_best_ari']:.3f} vs {fam['raw_spectrum_best_ari']:.3f}), "
                  f"though both remain modest."),
            ("h", "What the data do NOT support"),
            ("b", f"7. Matrix-invariant signatures. Pure-analyte and serum-spike fingerprints for the same "
                  f"analyte agree only at median cosine {cons.consistency_cosine.median():+.3f} ≈ 0: a "
                  f"component response is matrix-specific and must not be reused across matrices."),
            ("b", "8. Identity recovery for weak adsorbers. Serum responders move more but not more "
                  "focused (entropy unchanged); most non-purine spikes produce weak, non-specific "
                  "activation, consistent with the prior finding that these Ag-SERS spectra are "
                  "background-dominated."),
            ("h", "Interpretation (separated from observation)"),
            ("b", "The atlas behaves as a coordinate system whose AXES ARE CHEMICALLY REAL but whose "
                  "NAMES were assigned too coarsely by a static purity metric. Perturbation is the "
                  "sharper naming tool: driving an analyte reveals which axis carries it. The immediate "
                  "consequence for the ontology is to re-anchor low-purity component labels on their "
                  "reference loadings and on perturbation identity, not on the dominant chemical family."),
        ], f"53 fingerprints · adenine→c3 in 4/6 arms · atlas frozen")

        for name, cap in [
            ("f1_fingerprints.png", "Part 2 — response fingerprints. Each row is one serum-spiked "
             "analyte's 24-component Δ-activation; analytes are clustered by their response and coloured "
             "by chemical family."),
            ("f2_dose_specificity.png", "Part 1/4/11 — component dose-response for adenine (selective), "
             "per-component specificity, and the robustness map identifying BSV-anchor candidates."),
            ("f3_case_studies.png", "Part 6/7/8 — purine response similarity (two anti-correlated pairs), "
             "ergothioneine dose-response, and the selective uricase depletion."),
            ("f4_families_network.png", "Part 9/10/14 — responders vs non-responders, family recovery "
             "vs raw spectra, and the component–analyte bipartite network with its hubs."),
        ]:
            fig = plt.figure(figsize=PAGE)
            img = plt.imread(FIG / name)
            axi = fig.add_axes([0.04, 0.40, 0.92, 0.48]); axi.imshow(img); axi.axis("off")
            fig.text(0.07, 0.94, cap.split("—")[0].strip(), fontsize=13, fontweight="bold", color=INK)
            y = 0.35
            for ln in _wrap(cap, 100):
                fig.text(0.07, y, ln, fontsize=8.7, color=INK); y -= 0.017
            pdf.savefig(fig); plt.close(fig)

        # methodology
        text_page(pdf, "Methodology", [
            ("h", "Vocabulary discipline"),
            ("b", "A COMPONENT is a mathematical latent Raman motif (an NMF basis vector). Its THEME is a "
                  "tentative post-hoc label from the Component Audit. A RESPONSE is the measured change in "
                  "component activation under perturbation. A response that matches a theme corroborates "
                  "it; a mismatch is reported, never hidden."),
            ("h", "Projections"),
            ("b", "All 24-component coordinates are the frozen-atlas projections computed in the Spike "
                  "Validation study (NMF dictionary held fixed). No spectrum was reprocessed and the "
                  "atlas fingerprint was verified byte-identical before and after."),
            ("h", "Identity test (the key methodological choice)"),
            ("b", "Because theme labels proved unreliable, the primary link test asks whether the "
                  "perturbed analyte is itself among the top reference analytes that LOAD the responding "
                  "component (from the audit's composition table) — a label-independent identity test. "
                  "Synonyms and stereo-descriptors are canonicalised before matching."),
            ("h", "Nulls and statistics"),
            ("b", "Component dose-response uses Spearman ρ with a saturating (Langmuir) vs linear model "
                  "comparison; response fingerprints carry a 500-sample replicate bootstrap CI per "
                  "component; family recovery is ARI against chemical family, compared head-to-head "
                  "against clustering the raw spike spectra."),
            ("h", "Datasets (all Ag/Au-SERS, out of domain for a Raman atlas)"),
            ("m", "ILS adenine 3381 · pure Ag-SERS 265 · serum spikes 265 (53 analytes) · serum baseline "
                  "15 · uricase 20 · isotopic 73 · ergothioneine 55"),
        ])

        # ═══ PART 13 — RESPONSE ATLAS: one page per analyte ═══
        sep = plt.figure(figsize=PAGE)
        sep.text(0.5, 0.55, "Response atlas", ha="center", fontsize=20, fontweight="bold", color=INK)
        sep.text(0.5, 0.50, "one page per serum-spiked analyte (53) — Part 13", ha="center",
                 fontsize=11, color=MUTED)
        pdf.savefig(sep); plt.close(sep)

        base = np.nan_to_num(Xb).mean(0)
        base_coord = np.nan_to_num(proj["serum_baseline"][0]).mean(0)
        order_by_resp = p9.sort_values("activation_norm", ascending=False).analyte.tolist()
        for a in order_by_resp:
            row = fp[fp.analyte == a].iloc[0]
            d = row[dcols].values.astype(float)
            pos = [ms.index.get_loc(i) for i in ms.index[ms.analyte == a]]
            Xmean = np.nan_to_num(Xs[pos]).mean(0)
            fam_a = row.family
            resp = a in set(p9[p9.responder].analyte)
            loads = [f"c{j}" for j in range(K) if RL.component_encodes(a, comp_load, j)]
            fig = plt.figure(figsize=PAGE)
            fig.text(0.07, 0.965, a, fontsize=17, fontweight="bold", color=INK)
            fig.text(0.07, 0.943, f"family: {fam_a}   ·   {'RESPONDER' if resp else 'weak / non-responder'}"
                                  f"   ·   spike encodes: {', '.join(loads) if loads else 'no atlas component'}",
                     fontsize=9.3, color=FAMCOL.get(fam_a, MUTED))
            # spectra
            ax1 = fig.add_axes([0.08, 0.70, 0.40, 0.17])
            ax1.plot(grid, base, color=MUTED, lw=0.7, label="serum baseline")
            ax1.plot(grid, Xmean, color=P, lw=0.9, label="spiked")
            ax1.set_xlim(grid[0], grid[-1]); ax1.legend(fontsize=6); ax1.set_yticks([])
            ax1.set_xlabel("cm⁻¹", fontsize=7); ax1.set_title("mean spectra", fontsize=8.4)
            ax2 = fig.add_axes([0.56, 0.70, 0.40, 0.17])
            ax2.plot(grid, Xmean - base, color=S, lw=0.8)
            ax2.axhline(0, color=GRIDC, lw=0.6); ax2.set_xlim(grid[0], grid[-1]); ax2.set_yticks([])
            ax2.set_xlabel("cm⁻¹", fontsize=7); ax2.set_title("difference spectrum (spiked − serum)", fontsize=8.4)
            # 24-component fingerprint
            ax3 = fig.add_axes([0.08, 0.50, 0.88, 0.135])
            ax3.bar(range(K), d, color=[S if x < 0 else P for x in d])
            for j in range(K):
                if RL.component_encodes(a, comp_load, j):
                    ax3.annotate("↑encodes", (j, d[j]), fontsize=6, ha="center",
                                 color=INK, va="bottom" if d[j] >= 0 else "top")
            ax3.axhline(0, color=INK, lw=0.6); ax3.set_xticks(range(K))
            ax3.set_xticklabels([f"c{j}" for j in range(K)], fontsize=5.5, rotation=90)
            ax3.set_ylabel("Δ activation"); ax3.set_title("24-component response fingerprint", fontsize=8.4)
            # activated themes + trajectory-ish (magnitude bars of top components)
            ax4 = fig.add_axes([0.08, 0.29, 0.40, 0.145])
            topj = np.argsort(-np.abs(d))[:6][::-1]
            ax4.barh(range(6), [d[j] for j in topj], color=[S if d[j] < 0 else P for j in topj])
            ax4.set_yticks(range(6))
            ax4.set_yticklabels([f"c{j}: {themes.get(int(j))}" for j in topj], fontsize=6)
            ax4.axvline(0, color=INK, lw=0.6); ax4.set_xlabel("Δ", fontsize=7)
            ax4.set_title("most responsive components + themes", fontsize=8.4)
            # nearest neighbours by fingerprint
            ax5 = fig.add_axes([0.56, 0.29, 0.40, 0.145]); ax5.axis("off")
            sims = (U @ RL._unit(d[None, :]).T).ravel()
            nn = [(analytes[i], sims[i]) for i in np.argsort(-sims) if analytes[i] != a][:6]
            ax5.text(0, 1.0, "nearest neighbours (response cosine)", fontsize=7.5, fontweight="bold",
                     transform=ax5.transAxes)
            for k, (nm, sc) in enumerate(nn):
                ax5.text(0.02, 0.85 - k * 0.14, f"{nm}", fontsize=7,
                         color=FAMCOL.get(RL.analyte_family(nm), INK), transform=ax5.transAxes)
                ax5.text(0.75, 0.85 - k * 0.14, f"{sc:+.2f}", fontsize=7, transform=ax5.transAxes)
            # interpretation
            cons_a = float(cons.consistency_cosine.get(a, np.nan))
            interp = (f"Interpretation. The spike drives components "
                      f"{', '.join('c'+str(int(j)) for j in np.argsort(-np.abs(d))[:3])} most strongly "
                      f"(themes: {', '.join(str(themes.get(int(j))) for j in np.argsort(-np.abs(d))[:3])}). "
                      + (f"Its own encoding component(s) {', '.join(loads)} "
                         + ("ARE among the strongest responders — an identity match."
                            if loads and any(int(l[1:]) in np.argsort(-np.abs(d))[:5] for l in loads)
                            else "are present but not the strongest responders.")
                         if loads else
                         "This analyte does not correspond to any single atlas component, so no identity "
                         "match is possible. ")
                      + f" Pure-vs-serum consistency for this analyte is cos {cons_a:+.2f}"
                      + (" (matrix-specific)." if np.isfinite(cons_a) and cons_a < 0.3 else "."))
            y = 0.255
            for ln in _wrap(interp, 108):
                fig.text(0.07, y, ln, fontsize=7.6, color=INK); y -= 0.0145
            fig.text(0.07, y - 0.004,
                     "A component is a mathematical Raman motif; themes are tentative and never molecular "
                     "assignments.", fontsize=6.6, color=MUTED, style="italic")
            # confidence
            fig.text(0.07, 0.045,
                     f"Response magnitude {np.linalg.norm(d):.3f} · significant components "
                     f"{int(row.n_significant_components)}/24 · entropy {row.response_entropy:.2f} · "
                     f"confidence: {'moderate (identity match)' if (loads and resp) else ('low' if not resp else 'moderate')}",
                     fontsize=7.2, color=INK, family="DejaVu Sans Mono")
            pdf.savefig(fig); plt.close(fig)

        # mechanistic + DART
        mblocks = [("h", "Part 15 — has the loop closed?"),
                   ("b", mech["1_loop_closed"]),
                   ("b", mech["2_audit_predicts_response"]),
                   ("b", mech["3_response_supports_ontology"]),
                   ("m", f"components that GAIN identity confidence: {mech['4_ontology_gains_confidence']}"),
                   ("m", f"components whose LABEL loses confidence: {mech['5_ontology_loses_confidence']}"),
                   ("h", "Components needing reinterpretation")]
        for r in mech["6_components_needing_reinterpretation"]:
            if r["component"] is not None:
                mblocks.append(("m", f"  c{r['component']}: labelled '{r['current_label']}', driven by "
                                     f"{r['driven_by']}, top reference {r['reference_top']}"))
        mblocks += [("b", "Reading: " + mech["purine_substructure"]["reading"]),
                    ("b", "Reading: " + mech["uricase_selectivity"]["reading"]),
                    ("b", mech["consistency_caveat"]),
                    ("h", "Part 16 — implications for DART (grounded)"),
                    ("b", dart["premise"])]
        for x in dart["what_this_study_provides"]:
            mblocks.append(("b", "• " + x))
        mblocks.append(("b", "Recommended first DART test: " + dart["recommended_first_dart_test"]))
        for x in dart["grounded_limitations"]:
            mblocks.append(("m", "limitation: " + x))
        mblocks.append(("b", dart["no_overclaim"]))
        text_page(pdf, "Mechanistic interpretation and DART", mblocks)

        d = pdf.infodict()
        d["Title"] = "GAIRA Raman Reference Atlas v0.1 — Perturbation Response Audit"
    print(f"PDF written: {PDF_PATH} ({PDF_PATH.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
