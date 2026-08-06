#!/usr/bin/env python3
"""GAIRA V7 — Phase 02 figures (SVG vector + PNG preview). Deterministic; no RNG."""
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
P02 = HERE.parent
REPO = P02.parents[2]
sys.path.insert(0, str(REPO / "src"))

T, F, A, V = P02 / "tables", P02 / "figures", P02 / "artifacts", P02 / "validation"
INK, MUTED, LINE = "#1a1a1a", "#6b7280", "#9ca3af"
BLUE, GREEN, AMBER, RED, GREY = "#2563eb", "#15803d", "#b45309", "#b91c1c", "#4b5563"
PAL = ["#2563eb", "#15803d", "#b45309", "#7c3aed", "#0891b2", "#be123c", "#ca8a04", "#0f766e"]

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


def load():
    z = np.load(A / "edge_features_v1.npz", allow_pickle=True)
    reg = json.loads((A / "csm_registry_v1.json").read_text())
    d = np.load(A / "csm_dictionary_v1.npz", allow_pickle=True)
    lsm = np.load(REPO / "results/v7_rebuild/phase01/artifacts/lsm_dictionary_v1.npz",
                  allow_pickle=True)
    meta = pd.read_csv(REPO / "results/v7_rebuild/phase01/artifacts/lsm_registry_v1.csv")
    return z, reg, d, lsm, meta


# ── 1. the Consensus Spectral Graph ──────────────────────────────────────────
def fig_graph(z, reg, meta):
    ids = [str(s) for s in z["motif_ids"]]
    W, C = z["W"], z["coassign"]
    n = len(ids)
    cls = meta.set_index("motif_id").loc[ids].chemical_class.tolist()
    fams = sorted(set(cls))
    cmap = plt.get_cmap("tab20")
    colour = {c: cmap(i % 20) for i, c in enumerate(fams)}

    rej = pd.read_csv(T / "rejected_consensus_motifs_v1.csv")
    proposals = {r.proposed_group: [ids.index(x) for x in r.contributing_lsms.split(";")]
                 for r in rej.itertuples()}
    accepted = {c["csm_id"]: [ids.index(l["lsm_id"]) for l in c["contributing_lsms"]]
                for c in reg["csms"] if c["n_lsms"] > 1}
    in_proposal = {i for v in list(proposals.values()) + list(accepted.values()) for i in v}

    # Core = the 20 motifs that belong to a merge proposal; ring = the 30 that do not. Laying
    # out all 50 together lets the unconnected majority squash the structure into a blob.
    G = nx.Graph()
    G.add_nodes_from(sorted(in_proposal))
    for i in sorted(in_proposal):
        for j in sorted(in_proposal):
            if i < j and W[i, j] > 0:
                G.add_edge(i, j, weight=float(W[i, j]) ** 4)
    pos = nx.spring_layout(G, weight="weight", seed=11, k=2.6, iterations=1500)
    pts = np.array(list(pos.values()))
    pos = {k: (v - pts.mean(0)) / (np.abs(pts - pts.mean(0)).max() + 1e-9) * 0.66
           for k, v in pos.items()}
    outer = [i for i in range(n) if i not in pos]
    for k, i in enumerate(outer):
        th = 2 * np.pi * k / max(len(outer), 1) + 0.15
        pos[i] = np.array([1.13 * np.cos(th), 1.13 * np.sin(th)])

    fig, ax = plt.subplots(figsize=(10.2, 7.4))
    for i in range(n):
        for j in range(i + 1, n):
            unan = C[i, j] >= 1.0 - 1e-9
            if not unan and W[i, j] < 0.45:
                continue
            ax.plot(*zip(pos[i], pos[j]), lw=0.5 + 3.6 * max(W[i, j] - 0.3, 0),
                    color=RED if unan else LINE, alpha=0.85 if unan else 0.30,
                    zorder=2 if unan else 1, solid_capstyle="round")
    # hulls around the four proposals
    hulls = ([(k, v, True) for k, v in accepted.items()]
             + [(k, v, False) for k, v in proposals.items()])
    for hidx, (label, members, ok) in enumerate(hulls):
        pts = np.array([pos[i] for i in members])
        ctr = pts.mean(axis=0)
        rad = max(np.linalg.norm(pts - ctr, axis=1).max() + 0.075, 0.09)
        ax.add_patch(plt.Circle(ctr, rad, fill=False, lw=1.4,
                                ls="-" if ok else (0, (4, 2)),
                                color=GREEN if ok else RED, zorder=4, alpha=0.85))
        th = [np.pi / 2, -np.pi / 2, np.pi, 0.0][hidx % 4]
        lab_pos = ctr + (rad + 0.06) * np.array([np.cos(th), np.sin(th)])
        ax.annotate(f"{label} — {'ACCEPTED' if ok else 'rejected'}", lab_pos,
                    ha="center" if abs(np.cos(th)) < 0.5 else ("left" if np.cos(th) > 0 else "right"),
                    va="bottom" if np.sin(th) > 0 else ("top" if np.sin(th) < 0 else "center"),
                    fontsize=7.5, color=GREEN if ok else RED,
                    weight="bold" if ok else "normal", zorder=5)
    for i in range(n):
        ax.scatter(*pos[i], s=105 if i in in_proposal else 46,
                   color=colour[cls[i]], edgecolor=INK if i in in_proposal else "white",
                   linewidth=1.1 if i in in_proposal else 0.5, zorder=3)
    ax.set_xlim(-1.45, 1.45); ax.set_ylim(-1.35, 1.42)
    ax.set_aspect("equal"); ax.set_axis_off()
    ax.set_title("Consensus Spectral Graph — 50 Local Spectral Motifs, edges weighted by "
                 "seven-feature merge confidence\n"
                 "red edges are pairs co-assigned at EVERY viable significance level (the "
                 "merge proposals);\nthe outer ring holds the 30 motifs that never join one. "
                 "Grey edges are the strongest non-unanimous evidence.",
                 fontsize=9.5, color=INK, loc="left", pad=12)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=colour[c], label=c,
                              markersize=6) for c in fams],
              loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=7)
    save(fig, "fig01_consensus_spectral_graph")


# ── 2. communities and what happened to them ─────────────────────────────────
def fig_communities(reg, meta):
    rej = pd.read_csv(T / "rejected_consensus_motifs_v1.csv")
    acc = [c for c in reg["csms"] if c["n_lsms"] > 1]
    rows = ([{"label": c["csm_id"], "n": c["n_lsms"],
              "classes": len(c["supporting_classes"]), "status": "accepted",
              "cost": c["ev_delta_vs_lsms"], "cohesion": c["cohesion"]} for c in acc]
            + [{"label": r.proposed_group, "n": r.n_lsms,
                "classes": len(r.supporting_classes.split(";")), "status": "rejected",
                "cost": r.isolated_ev_cost, "cohesion": r.cohesion}
               for r in rej.itertuples()])
    df = pd.DataFrame(rows).sort_values("n", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4), gridspec_kw={"wspace": 0.32})
    ax = axes[0]
    y = np.arange(len(df))
    col = [GREEN if s == "accepted" else RED for s in df.status]
    ax.barh(y, df.n, color=col, alpha=0.85, height=0.62)
    for k, (yy, r) in enumerate(zip(y, df.itertuples())):
        ax.text(r.n + 0.12, yy, f"{r.classes} classes", va="center", fontsize=7, color=MUTED)
    ax.set_yticks(y); ax.set_yticklabels(df.label, fontsize=7.5)
    ax.set_xlabel("contributing Local Spectral Motifs")
    ax.set_title("Merge proposals and their fate", fontsize=9, loc="left", color=INK)
    ax.set_xlim(0, df.n.max() * 1.35)
    for s in ("top", "right"): ax.spines[s].set_visible(False)

    ax = axes[1]
    ax.axvline(-0.05, color=RED, ls="--", lw=1.0)
    ax.text(-0.05, len(df) - 0.4, " tolerance", color=RED, fontsize=7, va="top")
    ax.barh(y, df.cost, color=col, alpha=0.85, height=0.62)
    ax.set_yticks(y); ax.set_yticklabels([])
    ax.set_xlabel("isolated reconstruction cost (Δ explained variance)")
    ax.set_title("Every rejection is a measured cost, not a judgement",
                 fontsize=9, loc="left", color=INK)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    save(fig, "fig02_graph_communities")


# ── 3. merge confidence against the null ─────────────────────────────────────
def fig_confidence(z):
    W, null = z["W"], z["null_weights"]
    iu = np.triu_indices(W.shape[0], 1)
    obs = W[iu]
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    bins = np.linspace(0, 0.95, 60)
    ax.hist(null, bins=bins, density=True, color=GREY, alpha=0.42,
            label=f"band-permutation null ({len(null):,} draws)")
    ax.hist(obs, bins=bins, density=True, histtype="step", lw=1.7, color=BLUE,
            label=f"observed pairs (n = {len(obs)})")
    q99 = np.quantile(null, 0.99)
    ax.axvline(q99, color=RED, ls="--", lw=1.1)
    ax.text(q99 + 0.01, ax.get_ylim()[1] * 0.82, f"null p99 = {q99:.2f}", color=RED, fontsize=7.5)
    for w, lab in [(obs.max(), "strongest observed pair")]:
        ax.annotate(lab, (w, 0.15), xytext=(w - 0.28, 1.9), fontsize=7.5, color=INK,
                    arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8))
    ax.set_xlabel("edge weight — confidence that two LSMs describe one phenomenon")
    ax.set_ylabel("density")
    ax.set_title("Merge confidence distribution: most apparent LSM similarity is what "
                 "generic\nRaman band statistics already produce",
                 fontsize=9.5, loc="left", color=INK)
    ax.legend(frameon=False, fontsize=7.5)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    save(fig, "fig03_merge_confidence_vs_null")


# ── 4. provenance of the accepted CSM ────────────────────────────────────────
def fig_provenance(reg, meta):
    c = next(x for x in reg["csms"] if x["n_lsms"] > 1)
    lsms = [l["lsm_id"] for l in c["contributing_lsms"]]
    m = meta.set_index("motif_id")
    fig, ax = plt.subplots(figsize=(9.4, 4.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_axis_off()

    def box(x, y, w, h, text, fc, ec, fs=7.5, weight="normal"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                    fc=fc, ec=ec, lw=1.0))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                color=INK, weight=weight)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=9, color=MUTED, lw=0.9))

    box(3.4, 8.4, 3.2, 1.1, f"{c['csm_id']}\ncis-unsaturation", "#eff6ff", BLUE, 8.5, "bold")
    for k, lid in enumerate(lsms):
        x = 1.1 + k * 4.4
        box(x, 6.1, 3.4, 1.0, f"{lid}\n{m.loc[lid, 'lsm_type']}", "#f0fdf4", GREEN)
        arrow(x + 1.7, 7.1, 4.6 + (k - 0.5) * 0.7, 8.4)
        cls = m.loc[lid, "chemical_class"]
        box(x + 0.4, 4.4, 2.6, 0.8, cls, "#fffbeb", AMBER, 7)
        arrow(x + 1.7, 4.4 + 0.8, x + 1.7, 6.1)
        an = str(m.loc[lid, "analytes"]).split(";")
        txt = "\n".join(an[:5]) + (f"\n… +{len(an) - 5} more" if len(an) > 5 else "")
        box(x + 0.1, 1.2, 3.2, 2.7, txt, "white", LINE, 6.8)
        arrow(x + 1.7, 3.9, x + 1.7, 4.4)
    box(0.3, 0.1, 9.4, 0.75,
        f"→ {c['n_analytes']} canonical molecules → original Raman spectra "
        f"(phase00 registry → spectrum_id)", "#f9fafb", LINE, 7.5)
    ax.set_title("Provenance chain — CSM → LSM → chemistry class → canonical molecule → "
                 "spectrum.\nNo level is recomputed; each is stored, so the chain cannot "
                 "silently break.", fontsize=9.5, loc="left", color=INK)
    save(fig, "fig04_provenance_diagram")


# ── 5. reconstruction comparison ─────────────────────────────────────────────
def fig_reconstruction():
    r = pd.read_csv(V / "reconstruction_comparison_v1.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), gridspec_kw={"wspace": 0.28})
    ax = axes[0]
    ax.scatter(r.ev_lsm, r.ev_csm, s=16, color=BLUE, alpha=0.6, edgecolor="none")
    lim = [min(r.ev_lsm.min(), r.ev_csm.min()) - 0.03, 1.005]
    ax.plot(lim, lim, color=LINE, lw=0.9, ls="--")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("explained variance — 50 LSMs")
    ax.set_ylabel("explained variance — 49 CSMs")
    ax.set_title(f"Per molecule (n = {len(r)}), not averaged", fontsize=9, loc="left", color=INK)
    worst = r.nsmallest(3, "delta")
    for w in worst.itertuples():
        ax.annotate(w.canonical_id, (w.ev_lsm, w.ev_csm), fontsize=6.5, color=MUTED,
                    xytext=(4, -6), textcoords="offset points")
    for s in ("top", "right"): ax.spines[s].set_visible(False)

    ax = axes[1]
    ax.hist(r.delta, bins=40, color=BLUE, alpha=0.75)
    ax.axvline(-0.05, color=RED, ls="--", lw=1.0)
    ax.text(-0.049, ax.get_ylim()[1] * 0.9, " tolerance", color=RED, fontsize=7)
    ax.axvline(float(r.delta.mean()), color=GREEN, lw=1.2)
    ax.set_xlabel("Δ explained variance (CSM − LSM)")
    ax.set_ylabel("canonical molecules")
    ax.set_title(f"mean Δ = {r.delta.mean():+.4f}; "
                 f"{int((r.delta < -0.05).sum())} molecules beyond tolerance",
                 fontsize=9, loc="left", color=INK)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    save(fig, "fig05_reconstruction_comparison")


# ── 6. bootstrap / threshold stability ───────────────────────────────────────
def fig_stability():
    raw = pd.read_csv(T / "threshold_sweep_raw_v1.csv")
    sig = pd.read_csv(T / "significance_sweep_v1.csv")
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.2), gridspec_kw={"wspace": 0.34})

    ax = axes[0]
    ax.plot(raw.threshold, raw.n_communities, "o-", ms=3, color=BLUE, label="communities")
    ax.plot(raw.threshold, raw.n_singletons, "s-", ms=3, color=AMBER, label="singletons")
    ax.set_xlabel("raw edge-weight cut τ"); ax.set_ylabel("count")
    ax.set_title("The pre-registered sweep", fontsize=9, loc="left", color=INK)
    ax.legend(frameon=False, fontsize=7)
    for s in ("top", "right"): ax.spines[s].set_visible(False)

    ax = axes[1]
    ax.semilogx(sig.alpha, sig.n_nontrivial, "o-", ms=3, color=GREEN)
    ax.semilogx(sig.alpha, sig.largest_community, "s--", ms=3, color=MUTED)
    ax.invert_xaxis()
    ax.set_xlabel("significance level α vs the null")
    ax.set_ylabel("count")
    ax.set_title("The significance sweep", fontsize=9, loc="left", color=INK)
    ax.legend(["non-trivial groups", "largest group"], frameon=False, fontsize=7)
    for s in ("top", "right"): ax.spines[s].set_visible(False)

    ax = axes[2]
    sens = pd.read_csv(T / "coassignment_rule_sensitivity_v1.csv")
    ax.plot(sens.coassignment_rule, sens.lsms_merged, "o-", ms=4, color=RED)
    ax.set_xlabel("co-assignment rule (1.0 = unanimous)")
    ax.set_ylabel("LSMs absorbed into merges")
    ax.set_title("Why unanimity", fontsize=9, loc="left", color=INK)
    ax.invert_xaxis()
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.suptitle("No single cut yields an invariant partition — the structure is a continuum, "
                 "so the estimator is the consensus across the sweep",
                 fontsize=9.5, x=0.005, ha="left", y=1.06, color=INK)
    save(fig, "fig06_threshold_and_bootstrap_stability")


# ── 7. before / after hierarchy ──────────────────────────────────────────────
def fig_hierarchy(reg, meta):
    fig, ax = plt.subplots(figsize=(8.6, 3.0))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.set_axis_off()
    steps = [("375 spectra", GREY), ("154 canonical\nmolecules", GREY),
             ("16 chemistry\nclasses", AMBER), ("50 Local\nSpectral Motifs", GREEN),
             ("49 Consensus\nSpectral Motifs", BLUE)]
    for k, (lab, col) in enumerate(steps):
        x = 0.25 + k * 1.98
        ax.add_patch(FancyBboxPatch((x, 1.3), 1.6, 1.3, boxstyle="round,pad=0.07",
                                    fc="white", ec=col, lw=1.4))
        ax.text(x + 0.8, 1.95, lab, ha="center", va="center", fontsize=8, color=INK)
        if k:
            ax.add_patch(FancyArrowPatch((x - 0.34, 1.95), (x - 0.03, 1.95),
                                         arrowstyle="-|>", mutation_scale=10,
                                         color=MUTED, lw=1.0))
    ax.text(9.0, 0.75, "1 merge accepted\n3 merges rejected and undone",
            ha="right", fontsize=7.5, color=RED)
    ax.set_title("Representation hierarchy before and after Phase 02 — the layer narrows by "
                 "one,\nbecause exactly one cross-class merge survived falsification",
                 fontsize=9.5, loc="left", color=INK)
    save(fig, "fig07_before_after_hierarchy")


# ── 8. representative consensus spectrum with band annotation ────────────────
def fig_consensus_spectrum(reg, d, lsm):
    c = next(x for x in reg["csms"] if x["n_lsms"] > 1)
    grid = np.asarray(d["grid"], float)
    ids = [str(s) for s in d["csm_ids"]]
    csm = np.asarray(d["CSM"], float)[ids.index(c["csm_id"])]
    lids = [str(s) for s in lsm["motif_ids"]]
    H = np.asarray(lsm["H"], float)

    fig, ax = plt.subplots(figsize=(9.4, 4.0))
    for k, l in enumerate(c["contributing_lsms"]):
        h = H[lids.index(l["lsm_id"])]
        ax.plot(grid, h / h.max(), lw=1.0, color=PAL[k], alpha=0.65, label=l["lsm_id"])
    ax.plot(grid, csm / csm.max(), lw=1.9, color=INK, label=f"{c['csm_id']} (consensus)")
    ymax = 1.46          # headroom so the band labels sit inside the axes, not over the title
    for b, lab in zip(c["dominant_bands"], c["band_assignment"].split(" | ")):
        ax.axvline(b, color=LINE, lw=0.6, ls=":", zorder=0)
        ax.text(b, 1.05, lab.split(": ", 1)[-1], rotation=90, fontsize=6.4, color=MUTED,
                va="bottom", ha="center")
    ax.set_xlim(grid.min(), grid.max()); ax.set_ylim(0, ymax)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Raman shift (cm⁻¹)"); ax.set_ylabel("normalised intensity")
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper left",
              bbox_to_anchor=(0.0, 0.72))
    ax.set_title(f"{c['csm_id']} — the one accepted consensus motif. Contributors agree at "
                 f"cosine {1 - c['uncertainty']:.3f};\nsupport is {', '.join(c['projected_support'])}"
                 f" — every one polyunsaturated",
                 fontsize=9.5, loc="left", color=INK)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    save(fig, "fig08_representative_consensus_spectrum")


# ── 9. the four named suspects ───────────────────────────────────────────────
def fig_suspects():
    s = pd.read_csv(V / "named_suspect_pairs_v1.csv")
    feats = ["spectral_cosine", "band_overlap", "peak_agreement", "bootstrap_cooccurrence",
             "activation_cooccurrence", "provenance_overlap", "substitutability"]
    fig, axes = plt.subplots(1, 4, figsize=(11.0, 3.3), subplot_kw=dict(polar=True),
                             gridspec_kw={"wspace": 0.42})
    ang = np.linspace(0, 2 * np.pi, len(feats), endpoint=False).tolist()
    ang += ang[:1]
    for ax, r in zip(axes, s.itertuples()):
        v = [getattr(r, f) for f in feats]
        v += v[:1]
        ok = bool(r.merged_final)
        ax.plot(ang, v, lw=1.5, color=GREEN if ok else RED)
        ax.fill(ang, v, color=GREEN if ok else RED, alpha=0.18)
        ax.set_xticks(ang[:-1])
        ax.set_xticklabels([f.replace("_", "\n") for f in feats], fontsize=5.6)
        ax.set_ylim(0, 1); ax.set_yticks([0.5, 1.0]); ax.set_yticklabels(["", ""])
        ax.set_title(f"{r.class_a}\n↔ {r.class_b}\n"
                     f"w = {r.edge_weight:.3f} · "
                     f"{'MERGED' if ok else 'NOT MERGED'}",
                     fontsize=7.6, color=GREEN if ok else RED, pad=26)
    fig.suptitle("The four pre-declared false-merge suspects. All four clear the null "
                 "(p < 0.001) and all four\nare proposed by the graph — only one survives "
                 "reconstruction falsification.",
                 fontsize=9.5, x=0.005, ha="left", y=1.12, color=INK)
    save(fig, "fig09_named_suspect_pairs")


# ── 10. feature independence ─────────────────────────────────────────────────
def fig_feature_correlation():
    c = pd.read_csv(T / "feature_correlation_v1.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(c.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(c))); ax.set_yticks(range(len(c)))
    ax.set_xticklabels(c.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(c.index, fontsize=7)
    for i in range(len(c)):
        for j in range(len(c)):
            ax.text(j, i, f"{c.values[i, j]:.2f}", ha="center", va="center", fontsize=6.2,
                    color="white" if abs(c.values[i, j]) > 0.55 else INK)
    fig.colorbar(im, ax=ax, shrink=0.75, label="Spearman ρ over 1225 pairs")
    ax.set_title("\"Seven independent lines of evidence\" is a claim about this matrix,\n"
                 "so it is measured — not asserted", fontsize=9, loc="left", color=INK)
    save(fig, "fig10_feature_independence")


# ── 11. integration-method comparison ────────────────────────────────────────
def fig_methods():
    m = pd.read_csv(T / "integration_method_comparison_v1.csv").sort_values("composite")
    crit = ["consensus_stability", "within_cohesion", "between_separation",
            "chemical_coherence", "retained_lsm_information", "heldout_reconstruction"]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.4), gridspec_kw={"wspace": 0.3,
                                                                    "width_ratios": [1, 1.5]})
    ax = axes[0]
    y = np.arange(len(m))
    ax.barh(y, m.composite, color=[GREEN if i == len(m) - 1 else GREY for i in range(len(m))],
            alpha=0.85, height=0.6)
    ax.set_yticks(y); ax.set_yticklabels(m.method, fontsize=8)
    ax.set_xlabel("pre-registered composite")
    ax.set_title("Winner by composite", fontsize=9, loc="left", color=INK)
    for s in ("top", "right"): ax.spines[s].set_visible(False)

    ax = axes[1]
    w = 0.13
    for k, cr in enumerate(crit):
        ax.bar(np.arange(len(m)) + k * w, m[cr], width=w, label=cr.replace("_", " "),
               color=PAL[k % len(PAL)], alpha=0.9)
    ax.set_xticks(np.arange(len(m)) + 2.5 * w)
    ax.set_xticklabels(m.method, fontsize=7.5, rotation=12)
    ax.set_ylabel("criterion value")
    ax.legend(frameon=False, fontsize=6.2, ncol=3, loc="upper left")
    ax.set_title("Published regardless of the winner — the table is the deliverable",
                 fontsize=9, loc="left", color=INK)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    save(fig, "fig11_integration_method_comparison")


# ── 12. spectroscopic band annotation across the rejected groups ─────────────
def fig_rejected_bands(z, lsm, meta):
    rej = pd.read_csv(T / "rejected_consensus_motifs_v1.csv")
    lids = [str(s) for s in lsm["motif_ids"]]
    H = np.asarray(lsm["H"], float)
    grid = np.linspace(450, 1800, H.shape[1])
    fig, axes = plt.subplots(len(rej), 1, figsize=(9.4, 2.3 * len(rej)),
                             gridspec_kw={"hspace": 0.55})
    for ax, r in zip(np.atleast_1d(axes), rej.itertuples()):
        for k, lid in enumerate(r.contributing_lsms.split(";")):
            h = H[lids.index(lid)]
            ax.plot(grid, h / h.max() + k * 0.22, lw=0.95, color=PAL[k % len(PAL)],
                    label=lid)
        ax.set_xlim(450, 1800)
        ax.set_yticks([])
        ax.set_title(f"{r.proposed_group} — {r.supporting_classes.replace(';', ', ')}\n"
                     f"REJECTED: {r.rejection_reason}",
                     fontsize=8, loc="left", color=RED)
        ax.legend(frameon=False, fontsize=5.8, ncol=4, loc="upper right")
        for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    np.atleast_1d(axes)[-1].set_xlabel("Raman shift (cm⁻¹)")
    fig.suptitle("Rejected consensus motifs — the contributing LSMs, shown so the reader can "
                 "see what was\nnot merged and judge the decision independently",
                 fontsize=9.5, x=0.005, ha="left", y=0.995, color=INK)
    save(fig, "fig12_rejected_group_spectra")


def main():
    z, reg, d, lsm, meta = load()
    print("[phase02] figures")
    fig_graph(z, reg, meta)
    fig_communities(reg, meta)
    fig_confidence(z)
    fig_provenance(reg, meta)
    fig_reconstruction()
    fig_stability()
    fig_hierarchy(reg, meta)
    fig_consensus_spectrum(reg, d, lsm)
    fig_suspects()
    fig_feature_correlation()
    fig_methods()
    fig_rejected_bands(z, lsm, meta)


if __name__ == "__main__":
    main()
