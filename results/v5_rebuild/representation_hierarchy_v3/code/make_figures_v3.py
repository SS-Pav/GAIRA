"""V3 publication-quality figures. Each answers ONE scientific question. Static matplotlib PNGs
(auditable headless); the Sankey lives interactively in Explorer V3. Okabe-Ito palette.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import linregress, pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

REPO = Path("/Users/surajpg/projects/GAIRA")
BASE = REPO / "results/v5_rebuild/representation_hierarchy_v3"
FIG = BASE / "figures"; FIG.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(BASE / "tables/per_analyte_hierarchy.csv")
fam = pd.read_csv(BASE / "tables/rank_by_family.csv")
mat = pd.read_csv(BASE / "tables/matrix_robustness.csv")
summary = json.loads((BASE / "artifacts/hierarchy_summary.json").read_text())

OI = {"black": "#111418", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
      "yellow": "#F0E442", "blue": "#0072B2", "verm": "#D55E00", "purple": "#CC79A7", "grey": "#8A929C"}
# distinct family colors — NONE pure black (black is reserved for the median line in fig_h2)
FAMCOL = {f: c for f, c in zip(sorted(df.family.unique()),
          [OI["blue"], OI["orange"], OI["green"], OI["verm"], OI["purple"], OI["sky"],
           OI["yellow"], "#B07AA1", "#9C755F", "#666666", "#17BECF"])}
plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True, "grid.alpha": 0.22,
                     "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False})


# ── FIG 1 (CENTRAL): the Representation Hierarchy ──
def fig_hierarchy():
    fig = plt.figure(figsize=(15, 7.2))
    gsL = fig.add_axes([0.03, 0.06, 0.32, 0.88]); gsL.axis("off")
    levels = [
        ("Level 1", "Latent fingerprint", "24 NMF coordinates", OI["verm"], summary["layers"]["L1_latent_fingerprint"]["median"]),
        ("Level 2", "MSS motif", "12 biochemical motifs", OI["orange"], summary["layers"]["L2_mss_motif"]["median"]),
        ("Level 3", "Biochemical theme", "11 themes · raw→identity→rank→argmax", OI["blue"], summary["layers"]["L3a_theme_raw"]["median"]),
        ("Level 4", "Perturbation validation", "dose / directional (3 analytes)", OI["green"], None),
        ("Level 5", "Matrix robustness", "serum competition", OI["purple"], None)]
    y = 0.92
    for i, (lv, name, sub, col, med) in enumerate(levels):
        box = FancyBboxPatch((0.05, y - 0.13), 0.9, 0.12, boxstyle="round,pad=0.01",
                             linewidth=0, facecolor=col, alpha=0.92, transform=gsL.transAxes)
        gsL.add_patch(box)
        gsL.text(0.10, y - 0.045, f"{lv} · {name}", fontsize=12.5, fontweight="bold",
                 color="white", transform=gsL.transAxes, va="center")
        gsL.text(0.10, y - 0.095, sub + (f"   ·   median {med}" if med is not None else ""),
                 fontsize=8.7, color="white", transform=gsL.transAxes, va="center")
        if i < len(levels) - 1:
            gsL.add_patch(FancyArrowPatch((0.5, y - 0.13), (0.5, y - 0.17), transform=gsL.transAxes,
                          arrowstyle="-|>", mutation_scale=16, color=OI["black"], lw=1.6))
        y -= 0.185
    gsL.text(0.5, 0.995, "The Representation Hierarchy", fontsize=14, fontweight="bold",
             ha="center", transform=gsL.transAxes)
    gsL.text(0.5, -0.02, "abstraction increases downward → surface physics gives way to biochemistry",
             fontsize=8.5, ha="center", color=OI["grey"], transform=gsL.transAxes, style="italic")

    ax = fig.add_axes([0.44, 0.12, 0.53, 0.78])
    cols = [("L1_latent_fingerprint", "L1 latent", OI["verm"]),
            ("L2_mss_motif", "L2 MSS motif", OI["orange"]),
            ("L3a_theme_raw", "L3 theme (raw)", OI["blue"]),
            ("L4_theme_rank_rho", "L3 theme (rank ρ)", OI["sky"]),
            ("L5_top3_overlap", "L3 theme (top-3)", OI["green"]),
            ("L3b_theme_identity", "L3 theme (identity)", OI["purple"])]
    data = [df[c].values for c, _, _ in cols]
    parts = ax.violinplot(data, vert=False, showmedians=True, widths=0.85)
    for pc, (_, _, c) in zip(parts["bodies"], cols):
        pc.set_facecolor(c); pc.set_alpha(0.55); pc.set_edgecolor(c)
    for key in ("cmedians", "cbars", "cmins", "cmaxes"):
        parts[key].set_color(OI["black"]); parts[key].set_linewidth(1.1)
    ax.set_yticks(range(1, len(cols) + 1)); ax.set_yticklabels([l for _, l, _ in cols])
    ax.axvline(0, color=OI["grey"], ls=":", lw=0.8)
    ax.set_xlabel("preservation (cosine / ρ / overlap)  —  1.0 = identical")
    ax.set_title("Per-level distributions across 51 analytes\n"
                 "raw theme & raw rank are high but baseline-inflated; identity is the honest, selective signal",
                 fontsize=10)
    ax.set_xlim(-1.02, 1.05)
    fig.savefig(FIG / "fig_h1_representation_hierarchy.png", bbox_inches="tight"); plt.close(fig)


# ── FIG 2: metric comparison — raw vs rank vs identity vs top3 per analyte (parallel coords) ──
def fig_metric_comparison():
    d = df.sort_values("L4_theme_rank_rho", ascending=False)
    axes_cols = ["L1_latent_fingerprint", "L2_mss_motif", "L3a_theme_raw",
                 "L4_theme_rank_rho", "L5_top3_overlap", "L3b_theme_identity"]
    labels = ["latent", "MSS", "theme raw", "rank ρ", "top-3", "theme identity"]
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(axes_cols))
    for r in d.itertuples():
        ys = [getattr(r, c) for c in axes_cols]
        ax.plot(x, ys, color=FAMCOL.get(r.family, OI["grey"]), alpha=0.5, lw=1)
    med = [d[c].median() for c in axes_cols]
    ax.plot(x, med, color=OI["black"], lw=3, marker="o", label="median", zorder=5)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.axhline(0, color=OI["grey"], ls=":", lw=0.8)
    ax.set_ylabel("preservation")
    ax.set_title("The same analytes, six metrics: preservation is not one number\n"
                 "raw theme/rank cluster near the top; identity spreads wide (many negative)", fontsize=10)
    handles = [plt.Line2D([0], [0], color=FAMCOL[f], lw=2, label=f) for f in sorted(df.family.unique())]
    handles.append(plt.Line2D([0], [0], color=OI["black"], lw=3, marker="o", label="median"))
    ax.legend(handles=handles, fontsize=7.6, ncol=2, loc="lower left")
    fig.tight_layout(); fig.savefig(FIG / "fig_h2_metric_comparison.png", bbox_inches="tight"); plt.close(fig)


# ── FIG 3: family heatmap across all layers ──
def fig_family_heatmap():
    cols = ["L1_latent", "L2_mss", "L3a_theme_raw", "L3b_theme_identity", "L4_rank_rho",
            "L4_rank_separation", "L5_top3", "L6_argmax", "delta_purine"]
    M = fam.set_index("family")[cols]
    fig, ax = plt.subplots(figsize=(11, 6.5))
    im = ax.imshow(M.values, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=40, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(M))); ax.set_yticklabels(M.index, fontsize=9)
    for i in range(len(M)):
        for j in range(len(cols)):
            v = M.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if abs(v) > 0.55 else OI["black"])
    ax.grid(False); ax.set_title("Preservation by biochemical family, every layer "
                                 "(red = high/positive, blue = low/negative)", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout(); fig.savefig(FIG / "fig_h3_family_heatmap.png", bbox_inches="tight"); plt.close(fig)


# ── FIG 4: top-k overlap + rank null (baseline inflation of rank) ──
def fig_topk_and_rank_null():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    parts = axes[0].violinplot([df.L5_top2_overlap, df.L5_top3_overlap], showmedians=True, widths=0.8)
    for pc, c in zip(parts["bodies"], [OI["blue"], OI["green"]]):
        pc.set_facecolor(c); pc.set_alpha(0.6)
    axes[0].set_xticks([1, 2]); axes[0].set_xticklabels(["top-2 overlap", "top-3 overlap"])
    axes[0].set_ylabel("fraction of top themes shared")
    axes[0].set_title("Top-k theme overlap\n(avoids argmax instability; median top-3 = 0.67 → 2 of 3 kept)", fontsize=9.5)
    # rank: raw vs null vs separation
    parts2 = axes[1].violinplot([df.L4_theme_rank_rho, df.L4_rank_null, df.L4_rank_separation],
                                showmedians=True, widths=0.8)
    for pc, c in zip(parts2["bodies"], [OI["sky"], OI["grey"], OI["green"]]):
        pc.set_facecolor(c); pc.set_alpha(0.6)
    axes[1].set_xticks([1, 2, 3]); axes[1].set_xticklabels(["rank ρ (raw)", "rank null\n(other analytes)", "rank separation\n(raw − null)"])
    axes[1].axhline(0, color=OI["grey"], ls=":", lw=0.8)
    axes[1].set_title("Theme RANK preservation is baseline-inflated too\n"
                      "raw ρ≈0.87 ≈ its null; the honest identity signal is the small positive separation", fontsize=9.5)
    fig.tight_layout(); fig.savefig(FIG / "fig_h4_topk_and_rank_null.png", bbox_inches="tight"); plt.close(fig)


# ── FIG 5: ΔPurine per analyte ──
def fig_delta_purine():
    d = df.sort_values("delta_purine")
    fig, ax = plt.subplots(figsize=(9, 12))
    y = np.arange(len(d))
    ax.barh(y, d.delta_purine, color=[OI["verm"] if v > 0 else OI["blue"] for v in d.delta_purine],
            edgecolor="white", linewidth=0.4)
    ax.axvline(0, color=OI["black"], lw=1)
    ax.set_yticks(y); ax.set_yticklabels(d.analyte, fontsize=7.4)
    ax.set_xlabel("ΔPurine share (Ag-SERS − Raman)")
    ax.set_title("Per-analyte pull into the purine attractor\n"
                 "red = gains purine share on silver (weak adsorbers) · blue = loses it (already purine-rich)",
                 fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "fig_h5_delta_purine.png", bbox_inches="tight"); plt.close(fig)


# ── FIG 6: ΔPurine vs latent fingerprint (the attractor signature) ──
def fig_delta_vs_component():
    x, yv = df.L1_latent_fingerprint.values, df.delta_purine.values
    lr = linregress(x, yv); r, p = pearsonr(x, yv)
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for fname, sub in df.groupby("family"):
        ax.scatter(sub.L1_latent_fingerprint, sub.delta_purine, s=52, color=FAMCOL.get(fname, OI["grey"]),
                   edgecolor="white", linewidth=0.6, label=fname, alpha=0.9)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, lr.intercept + lr.slope * xs, color=OI["black"], lw=2,
            label=f"fit r={r:.2f}, p={p:.3f}")
    ax.axhline(0, color=OI["grey"], ls=":", lw=0.8)
    ax.set_xlabel("Latent fingerprint preservation (component cosine)")
    ax.set_ylabel("ΔPurine share (Ag − Raman)")
    ax.set_title("Weaker adsorption fidelity → stronger pull into the purine attractor\n"
                 f"significant negative relationship (r={r:.2f}, p={p:.3f})", fontsize=10)
    ax.legend(fontsize=7.2, ncol=2, loc="upper right")
    fig.tight_layout(); fig.savefig(FIG / "fig_h6_delta_purine_vs_component.png", bbox_inches="tight"); plt.close(fig)


# ── FIG 7: matrix robustness regression (weak per-analyte predictor) ──
def fig_matrix_regression():
    x, yv = mat.L1_latent_fingerprint.values, mat.serum_spike_displacement.values
    lr = linregress(x, yv); r, p = pearsonr(x, yv)
    n = len(x); xs = np.linspace(x.min(), x.max(), 60)
    yhat = lr.intercept + lr.slope * xs
    resid = yv - (lr.intercept + lr.slope * x); s_err = np.sqrt(np.sum(resid**2) / (n - 2))
    se_line = s_err * np.sqrt(1/n + (xs - x.mean())**2 / np.sum((x - x.mean())**2))
    fig, ax = plt.subplots(figsize=(8.6, 6))
    ax.scatter(x, yv, s=52, color=OI["purple"], edgecolor="white", linewidth=0.6, alpha=0.85)
    ax.plot(xs, yhat, color=OI["black"], lw=2)
    ax.fill_between(xs, yhat - 1.98*se_line, yhat + 1.98*se_line, color=OI["grey"], alpha=0.18,
                    label="95% CI (mean)")
    hi = {"hypoxanthine", "xanthine", "adenine", "ergothioneine", "guanine", "uracil"}
    for r_ in mat.itertuples():
        if r_.analyte in hi:
            ax.annotate(r_.analyte, (r_.L1_latent_fingerprint, r_.serum_spike_displacement),
                        fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Pure Ag-SERS latent fingerprint preservation")
    ax.set_ylabel("Serum spike displacement (matrix recoverability)")
    ax.set_title("Does pure transfer PREDICT serum recovery? Only weakly.\n"
                 f"r={r:.2f}, R²={r**2:.3f}, p={p:.2f} (n.s.) — the top oxopurines survive both, "
                 "but there is no tight per-analyte law", fontsize=9.5)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(FIG / "fig_h7_matrix_regression.png", bbox_inches="tight"); plt.close(fig)


# ── FIG 8: perturbation summary (3 analytes) ──
def fig_perturbation_summary():
    val = json.loads((REPO / "results/v5_rebuild/foundation_audit/tables/validation_results.json").read_text())
    fig, ax = plt.subplots(figsize=(14, 3.2)); ax.axis("off")
    rows = [["adenine", "concentration dose-response", "nucleic_purine",
             f"ρ={val['3_adenine_dose']['monotonicity_rho']}, Langmuir K={val['3_adenine_dose']['saturating_K_uM']} µM"],
            ["ergothioneine", "concentration dose-response", "sulfur_antioxidant",
             f"ρ={val['4_ergothioneine_dose']['monotonicity_rho']}, saturating K={val['4_ergothioneine_dose']['saturating_K_uM']} µM"],
            ["uricase (urate)", "directional depletion (NOT a dose)", "oxopurine motif",
             f"Δoxopurine motif = {val['6_uricase_depletion']['delta_oxopurine_motif']}"]]
    tbl = ax.table(cellText=rows,
                   colLabels=["analyte", "perturbation type", "target", "key metric"],
                   loc="center", cellLoc="left", colColours=[OI["green"]]*4,
                   colWidths=[0.16, 0.30, 0.20, 0.34])
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 2.0)
    for (r_, c_), cell in tbl.get_celld().items():
        if r_ == 0:
            cell.set_text_props(color="white", fontweight="bold")
    ax.set_title("Level 4 — Perturbation validation exists for EXACTLY three analytes\n"
                 "dynamic response is stronger evidence than any static similarity — but it is rare",
                 fontsize=11, pad=18)
    fig.savefig(FIG / "fig_h8_perturbation_summary.png", bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    fig_hierarchy(); print("h1 hierarchy ok")
    fig_metric_comparison(); print("h2 metric comparison ok")
    fig_family_heatmap(); print("h3 family heatmap ok")
    fig_topk_and_rank_null(); print("h4 topk + rank null ok")
    fig_delta_purine(); print("h5 delta purine ok")
    fig_delta_vs_component(); print("h6 delta vs component ok")
    fig_matrix_regression(); print("h7 matrix regression ok")
    fig_perturbation_summary(); print("h8 perturbation summary ok")
    print("all V3 figures ->", FIG)
