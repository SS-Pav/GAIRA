"""V4 — 11 publication-quality figures for the null-calibrated recoverability analysis.
Static matplotlib PNGs (auditable). Okabe-Ito palette. Reads committed tables + vectors.npz.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch

REPO = Path("/Users/surajpg/projects/GAIRA")
BASE = REPO / "results/v5_rebuild/hierarchical_recoverability_v4"
FIG = BASE / "figures"; FIG.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(BASE / "tables/per_analyte_evidence_profile.csv")
S = json.loads((BASE / "artifacts/recoverability_summary.json").read_text())
CNT = pd.read_csv(BASE / "tables/recoverable_analyte_counts.csv")
LVL = pd.read_csv(BASE / "tables/level_null_summary.csv")
MP = pd.read_csv(BASE / "tables/matrix_prediction.csv")
MSR = pd.read_csv(BASE / "tables/mss_specificity_ranking.csv")
V = np.load(BASE / "artifacts/vectors.npz", allow_pickle=True)
AN = list(V["analytes"]); THEMES = list(V["themes"]); MOTIFS = list(V["motifs"])
GRID = V["grid"]; PUR = int(V["purine_idx"])
idx = {a: i for i, a in enumerate(AN)}

OI = {"black": "#111418", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
      "yellow": "#F0E442", "blue": "#0072B2", "verm": "#D55E00", "purple": "#CC79A7", "grey": "#8A929C"}
LCOL = {"latent": OI["verm"], "MSS": OI["orange"], "theme": OI["blue"], "rank": OI["sky"],
        "perturbation": OI["green"], "matrix": OI["purple"]}
plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True, "grid.alpha": 0.22,
                     "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False})


def sub(ax, q, res, lim):
    ax.text(0.0, -0.24, f"Q: {q}\nResult: {res}\nLimitation: {lim}", transform=ax.transAxes,
            fontsize=7.4, color=OI["grey"], va="top")


# ── FIG 1 — representation hierarchy with null separation + recovered counts ──
def fig1():
    fig = plt.figure(figsize=(15, 7.4))
    axL = fig.add_axes([0.02, 0.05, 0.40, 0.9]); axL.axis("off")
    levels = [("Level 1", "Latent fingerprint", OI["verm"], LVL.iloc[0]),
              ("Level 2", "MSS motif", OI["orange"], LVL.iloc[1]),
              ("Level 3", "Biochemical theme", OI["blue"], LVL.iloc[3]),
              ("Level 4", "Perturbation (functional)", OI["green"], None),
              ("Level 5", "Matrix (serum)", OI["purple"], None)]
    y = 0.9
    cnt = {r.level: r for _, r in CNT.iterrows()}
    for i, (lv, nm, col, row) in enumerate(levels):
        axL.add_patch(FancyBboxPatch((0.04, y - 0.135), 0.92, 0.125, boxstyle="round,pad=0.01",
                      linewidth=0, facecolor=col, alpha=0.93, transform=axL.transAxes))
        axL.text(0.08, y - 0.05, f"{lv} · {nm}", fontsize=12, fontweight="bold", color="white",
                 transform=axL.transAxes, va="center")
        if row is not None:
            axL.text(0.08, y - 0.1, f"matched {row.matched_median} · sep {row.separation} · "
                     f"recovered {row.n_recovered}/51", fontsize=8.2, color="white",
                     transform=axL.transAxes, va="center")
        else:
            key = "perturbation" if i == 3 else "matrix"
            c = cnt[key]
            axL.text(0.08, y - 0.1, f"recovered {int(c.n_recovered)}/{int(c.denominator)} "
                     f"({'functional' if i==3 else 'serum-tested denom'})", fontsize=8.2,
                     color="white", transform=axL.transAxes, va="center")
        if i < 4:
            axL.add_patch(FancyArrowPatch((0.5, y - 0.135), (0.5, y - 0.17), transform=axL.transAxes,
                          arrowstyle="-|>", mutation_scale=15, color=OI["black"], lw=1.5))
        y -= 0.185
    axL.text(0.5, 0.98, "The GAIRA representation hierarchy (null-calibrated)", fontsize=13.5,
             fontweight="bold", ha="center", transform=axL.transAxes)
    axL.text(0.5, 0.0, "abstraction ↑ downward · analyte identity does NOT increase with abstraction",
             fontsize=8.5, style="italic", ha="center", color=OI["grey"], transform=axL.transAxes)

    ax = fig.add_axes([0.5, 0.14, 0.47, 0.76])
    labels = ["L1 latent", "L2 MSS", "L3a theme raw", "L3b identity"]
    sep = [LVL.iloc[k].separation for k in range(4)]           # per-analyte median(matched−null)
    matched = [LVL.iloc[k].matched_median for k in range(4)]
    nullm = [LVL.iloc[k].null_median for k in range(4)]
    yy = np.arange(4)
    bars = ax.barh(yy, sep, 0.55, color=[OI["verm"], OI["orange"], OI["sky"], OI["blue"]],
                   edgecolor="white")
    for k in range(4):
        ax.text(sep[k] + 0.004, yy[k], f"sep {sep[k]:+.4f}\n(matched {matched[k]:.2f} vs null {nullm[k]:.2f})",
                va="center", fontsize=7.8, color=OI["black"])
    ax.set_yticks(yy); ax.set_yticklabels(labels); ax.set_xlim(0, 0.06)
    ax.axvline(0, color=OI["grey"], lw=0.6)
    ax.set_xlabel("per-analyte identity signal  =  median(matched − mismatched-null)")
    ax.set_title("The identity signal is tiny at EVERY level\n"
                 "MSS (0.008) is smaller than latent (0.024) — MSS is not more analyte-specific",
                 fontsize=10)
    fig.savefig(FIG / "fig01_representation_hierarchy.png", bbox_inches="tight"); plt.close(fig)


# ── FIG 2 — recoverable analytes by level, with CI + denominators ──
def fig2():
    order = ["latent", "MSS", "theme", "top3_over_null", "argmax_robust", "perturbation", "matrix"]
    lab = {"latent": "Latent-specific", "MSS": "MSS-specific", "theme": "Theme-specific",
           "top3_over_null": "Top-3 > null", "argmax_robust": "Argmax robust",
           "perturbation": "Perturbation", "matrix": "Matrix (serum)"}
    c = {r.level: r for _, r in CNT.iterrows()}
    fig, ax = plt.subplots(figsize=(11, 5.4))
    xs = np.arange(len(order))
    fracs = [c[k].fraction for k in order]
    lo = [c[k].fraction - c[k].ci95_low for k in order]
    hi = [c[k].ci95_high - c[k].fraction for k in order]
    cols = [LCOL.get(k, OI["sky"]) for k in order]
    ax.bar(xs, fracs, color=cols, edgecolor="white", yerr=[lo, hi], capsize=4)
    for i, k in enumerate(order):
        ax.text(i, fracs[i] + hi[i] + 0.02, f"{int(c[k].n_recovered)}/{int(c[k].denominator)}",
                ha="center", fontsize=8.5, fontweight="bold")
    ax.set_xticks(xs); ax.set_xticklabels([lab[k] for k in order], rotation=20, ha="right")
    ax.set_ylabel("fraction recovered (95% CI)"); ax.set_ylim(0, 0.75)
    ax.set_title("How many of 51 analytes are recoverable at each level?\n"
                 "matrix denominator = serum-tested; all others = 51 matched", fontsize=10.5)
    fig.tight_layout(); fig.savefig(FIG / "fig02_recoverable_by_level.png", bbox_inches="tight"); plt.close(fig)


# ── FIG 3 — per-analyte recovery matrix ──
def fig3():
    cols = [("latent_recovered", "latent"), ("MSS_recovered", "MSS"), ("theme_recovered", "theme"),
            ("argmax_robust", "rank/argmax"), ("top3_over_null", "top-3>null"),
            ("perturbation_validated", "perturb"), ("matrix_recovered", "matrix")]
    d = df.sort_values(["latent_recovered", "MSS_recovered", "theme_recovered", "C_latent"],
                       ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9.5, 13))
    for ci, (col, _) in enumerate(cols):
        for ri, a in enumerate(d.analyte):
            r = d.iloc[ri]
            tested = True
            if col == "perturbation_validated":
                tested = r.perturbation_status != "not tested"
                val = r[col]
            elif col == "matrix_recovered":
                tested = r.serum_tested; val = r[col]
            else:
                val = r[col]
            if not tested:
                ax.text(ci, ri, "·", ha="center", va="center", color=OI["grey"], fontsize=11)
            elif val:
                ax.add_patch(plt.Rectangle((ci - 0.45, ri - 0.45), 0.9, 0.9,
                             color=LCOL.get(_ if _ in LCOL else "theme", OI["green"]) if False else OI["green"]))
                ax.text(ci, ri, "✓", ha="center", va="center", color="white", fontsize=8)
            else:
                ax.add_patch(plt.Rectangle((ci - 0.45, ri - 0.45), 0.9, 0.9, color="#f0f2f5"))
    ax.set_xlim(-0.5, len(cols) - 0.5); ax.set_ylim(len(d) - 0.5, -0.5)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([c[1] for c in cols], rotation=35, ha="left", fontsize=8.5)
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks(range(len(d))); ax.set_yticklabels(d.analyte, fontsize=6.6)
    ax.grid(False)
    ax.set_title("Per-analyte recovery matrix (✓ recovered · blank not · · not tested)\n"
                 "green = specifically recovered at that level", fontsize=10, pad=28)
    fig.tight_layout(); fig.savefig(FIG / "fig03_recovery_matrix.png", bbox_inches="tight"); plt.close(fig)


# ── FIG 4 — matched vs mismatched distributions ──
def fig4():
    tR, tS, zR, zS, mR, mS = V["tR"], V["tS"], V["zR"], V["zS"], V["mR"], V["mS"]
    muR = tR.mean(0)
    def cos_rows(A, B):
        An = A/(np.linalg.norm(A,axis=1,keepdims=True)+1e-12); Bn=B/(np.linalg.norm(B,axis=1,keepdims=True)+1e-12)
        return An@Bn.T
    def md(A, B):
        s = cos_rows(A, B); n = len(A)
        m = np.diag(s); off = s[~np.eye(n, dtype=bool)]
        return m, off
    panels = [("latent cosine", md(zR, zS)), ("MSS cosine", md(mR, mS)),
              ("raw theme cosine", md(tR, tS)),
              ("identity residual", md(tR-muR, tS-muR)),
              ("Spearman rank", (np.array([spearmanr(tR[i],tS[i]).correlation for i in range(len(tR))]),
                                 np.array([spearmanr(tR[i],tS[j]).correlation for i in range(len(tR)) for j in range(len(tR)) if j!=i]))),
              ("top-3 overlap", (df.top3.values, df.top3_null.values))]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.4))
    for ax, (name, (m, off)) in zip(axes.ravel(), panels):
        lo = min(off.min(), m.min()); hi = max(off.max(), m.max())
        bins = np.linspace(lo, hi, 26)
        ax.hist(off, bins=bins, color=OI["grey"], alpha=0.6, density=True, label="mismatched null")
        ax.hist(m, bins=bins, color=OI["blue"], alpha=0.6, density=True, label="matched")
        ax.axvline(np.median(off), color=OI["grey"], ls="--", lw=1)
        ax.axvline(np.median(m), color=OI["blue"], ls="--", lw=1)
        ax.set_title(f"{name}  (Δmedian={np.median(m)-np.median(off):+.3f})", fontsize=9.5)
        ax.set_yticks([])
    axes[0, 0].legend(fontsize=7.5)
    fig.suptitle("Matched vs mismatched distributions — overlap = the metric does NOT carry analyte identity",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG / "fig04_matched_vs_null.png", bbox_inches="tight"); plt.close(fig)


# ── FIG 5 — component vs MSS vs theme trajectory ──
def fig5():
    hi = ["adenine", "ergothioneine", "urate", "hypoxanthine", "xanthine", "guanine", "glucose", "tyrosine", "uracil"]
    d = df.set_index("analyte")
    cols = ["C_latent", "C_MSS", "C_theme_raw", "theme_identity"]
    labels = ["latent", "MSS", "theme raw", "identity"]
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(cols))
    for a in df.analyte:
        ys = [d.loc[a, c] for c in cols]
        ax.plot(x, ys, color="#dfe4ea", lw=0.8, zorder=1)
    palette = [OI["verm"], OI["green"], OI["purple"], OI["blue"], OI["sky"], OI["orange"],
               OI["black"], OI["grey"], "#9C755F"]
    for a, col in zip(hi, palette):
        ys = [d.loc[a, c] for c in cols]
        ax.plot(x, ys, marker="o", lw=2, color=col, label=a, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.axhline(0, color=OI["grey"], ls=":", lw=0.8)
    ax.set_ylabel("value"); ax.set_title("Hierarchical trajectory per analyte (grey = all 51)\n"
                 "raw theme high for all; latent & identity separate the strong chemisorbers", fontsize=10)
    ax.legend(fontsize=7.8, ncol=3, loc="lower left")
    fig.tight_layout(); fig.savefig(FIG / "fig05_trajectory.png", bbox_inches="tight"); plt.close(fig)


# ── FIG 6 — MSS-specific recoverability ranking (null-adjusted) ──
def fig6():
    d = MSR.sort_values("mss_specificity", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 12))
    yy = np.arange(len(d))
    cols = [OI["green"] if r else "#c9ced6" for r in d.MSS_recovered]
    ax.barh(yy, d.mss_specificity, color=cols, edgecolor="white", linewidth=0.4)
    ax.axvline(0, color=OI["black"], lw=1)
    ax.set_yticks(yy); ax.set_yticklabels(d.analyte, fontsize=7)
    ax.set_xlabel("MSS specificity  (matched cosine − null95)")
    ax.set_title("MSS ranked by NULL-ADJUSTED specificity, not raw cosine\n"
                 "green = MSS-recovered (rank-1 + stable); only 3/51 clear the null", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "fig06_mss_specificity.png", bbox_inches="tight"); plt.close(fig)


# ── FIG 7 — broad theme vs identity-specific theme ──
def fig7():
    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    sc = ax.scatter(df.C_theme_raw, df.theme_identity, c=df.C_latent, cmap="viridis", s=48,
                    edgecolor="white", linewidth=0.5)
    for a in ["hypoxanthine", "xanthine", "guanine", "adenine", "glucose", "uracil", "creatinine", "urea"]:
        r = df[df.analyte == a].iloc[0]
        ax.annotate(a, (r.C_theme_raw, r.theme_identity), fontsize=7, xytext=(3, 3),
                    textcoords="offset points")
    ax.axhline(0, color=OI["grey"], ls=":", lw=0.8)
    ax.set_xlabel("raw theme cosine (broad interpretation — high for all)")
    ax.set_ylabel("identity residual (analyte-specific — mostly weak)")
    ax.set_title("Why a high raw theme cosine is NOT analyte identity\n"
                 "raw cosine clusters near 0.9 regardless; identity separates only strong adsorbers", fontsize=10)
    fig.colorbar(sc, ax=ax, label="latent cosine")
    fig.tight_layout(); fig.savefig(FIG / "fig07_broad_vs_identity.png", bbox_inches="tight"); plt.close(fig)


# ── FIG 8 — purine attractor controls ──
def fig8():
    pc = pd.read_csv(BASE / "tables/purine_blank_controls.csv").iloc[0]
    tb = V["t_blank"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.6))
    ax0, ax1, ax2, ax3 = axes.ravel()
    ax0.bar(range(len(THEMES)), tb, color=[OI["verm"] if i == PUR else OI["sky"] for i in range(len(THEMES))])
    ax0.set_xticks(range(len(THEMES))); ax0.set_xticklabels(THEMES, rotation=90, fontsize=7)
    ax0.set_ylabel("theme share"); ax0.set_title(
        f"CONTROL: unspiked-serum-on-Ag BLANK is already purine-dominant\n"
        f"(purine {pc.serum_blank_purine_theme}, dominant = {pc.serum_blank_dominant_theme}) — before any analyte",
        fontsize=9.5)
    d = df.sort_values("delta_purine")
    ax1.barh(range(len(d)), d.delta_purine, color=[OI["verm"] if v > 0 else OI["blue"] for v in d.delta_purine])
    ax1.set_yticks([]); ax1.set_xlabel("Δpurine (Ag − Raman)"); ax1.axvline(0, color=OI["black"], lw=1)
    ax1.set_title("Δpurine per analyte (36/51 gain purine on Ag)", fontsize=9.5)
    for f, sub_ in df.groupby("family"):
        ax2.scatter(sub_.C_latent, sub_.delta_purine, s=32, label=f, alpha=0.8)
    ax2.axhline(0, color=OI["grey"], ls=":"); ax2.set_xlabel("latent cosine"); ax2.set_ylabel("Δpurine")
    ax2.set_title("Δpurine vs latent (r=−0.38, p=0.006): weaker preservation → more pull", fontsize=9.5)
    ax2.legend(fontsize=6, ncol=2)
    ax3.scatter(df.C_MSS, df.delta_purine, s=32, color=OI["orange"])
    ax3.axhline(0, color=OI["grey"], ls=":"); ax3.set_xlabel("MSS cosine"); ax3.set_ylabel("Δpurine")
    ax3.set_title("Δpurine vs MSS (r=−0.40, p=0.003)", fontsize=9.5)
    fig.suptitle("The purine attractor — phenomenological, present in the background BEFORE analyte addition",
                 fontsize=12.5, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIG / "fig08_purine_controls.png", bbox_inches="tight"); plt.close(fig)


# ── FIG 9 — perturbation summary ──
def fig9():
    val = json.loads((REPO / "results/v5_rebuild/foundation_audit/tables/validation_results.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, key, col, ttl in [(axes[0], "3_adenine_dose", OI["blue"], "Adenine → purine"),
                              (axes[1], "4_ergothioneine_dose", OI["green"], "Ergothioneine → sulfur")]:
        d = val[key]; x = np.array(d["levels_uM"], float); y = np.array(d["theme_series"], float); o = np.argsort(x)
        ax.plot(x[o], y[o], "o-", color=col, lw=1.8)
        ax.set_xlabel("concentration (µM)"); ax.set_ylabel(f"{d['theme']} share")
        ax.set_title(f"{ttl}\nρ={d['monotonicity_rho']}, K={d['saturating_K_uM']} µM, R²={d['saturating_r2']}", fontsize=9)
    u = val["6_uricase_depletion"]
    axes[2].bar(["oxopurine\nmotif", "purine-ring\nmotif", "purine\ntheme"],
                [u["delta_oxopurine_motif"], u["delta_purine_ring_motif"], u["purine_delta"]],
                color=[OI["verm"], OI["grey"], OI["grey"]])
    axes[2].axhline(0, color=OI["black"], lw=0.8); axes[2].set_ylabel("Δ on uricase")
    axes[2].set_title("Uricase urate depletion (DIRECTIONAL)\noxopurine motif drops; theme diffuse", fontsize=9)
    fig.suptitle("Level 4 — functional perturbation validation (ONLY adenine, ergothioneine, uricase)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig(FIG / "fig09_perturbation.png", bbox_inches="tight"); plt.close(fig)


# ── FIG 10 — pure transfer vs matrix recovery (effect sizes + CI) ──
def fig10():
    d = MP.sort_values("pearson_r")
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    yy = np.arange(len(d))
    lo = d.pearson_r - d.ci95_low; hi = d.ci95_high - d.pearson_r
    cols = [OI["green"] if p < 0.05 else OI["grey"] for p in d.p]
    ax.barh(yy, d.pearson_r, xerr=[lo, hi], color=cols, capsize=3, edgecolor="white")
    ax.axvline(0, color=OI["black"], lw=1)
    ax.set_yticks(yy); ax.set_yticklabels(d.predictor)
    for i, (_, r) in enumerate(d.iterrows()):
        ax.text(r.pearson_r + (0.02 if r.pearson_r >= 0 else -0.02), i,
                f"p={r.p:.2g}", va="center", ha="left" if r.pearson_r >= 0 else "right", fontsize=7.5)
    ax.set_xlabel("Pearson r vs serum spike displacement (95% CI)")
    ax.set_title("Does any PURE metric predict serum recoverability?\n"
                 "green = p<0.05. Only confidence is significant (r=0.71) — likely signal-strength, not identity",
                 fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "fig10_matrix_prediction.png", bbox_inches="tight"); plt.close(fig)


# ── FIG 11 — representative analyte panels ──
def fig11():
    reps = ["adenine", "ergothioneine", "urate", "hypoxanthine", "glucose", "tyrosine", "uracil"]
    tR, tS, zR, zS = V["tR"], V["tS"], V["zR"], V["zS"]
    ram, sers = V["ram_spec"], V["sers_spec"]; muR = tR.mean(0)
    fig, axes = plt.subplots(len(reps), 4, figsize=(15, 2.1 * len(reps)))
    for row, a in enumerate(reps):
        i = idx[a]; r = df[df.analyte == a].iloc[0]
        a0, a1, a2, a3 = axes[row]
        # spectra
        a0.plot(GRID, ram[i] / (ram[i].max() + 1e-9), color=OI["blue"], lw=0.7, label="Raman")
        a0.plot(GRID, sers[i] / (sers[i].max() + 1e-9) - 1.1, color=OI["verm"], lw=0.7, label="Ag-SERS")
        a0.set_yticks([]); a0.set_xlim(450, 1800)
        if row == 0: a0.set_title("spectra (Raman / Ag-SERS)", fontsize=8.5)
        a0.set_ylabel(a, fontsize=9, fontweight="bold")
        # latent coords
        a1.bar(range(24), zR[i], color=OI["blue"], alpha=0.6, width=0.8, label="R")
        a1.bar(range(24), -zS[i], color=OI["verm"], alpha=0.6, width=0.8, label="S")
        a1.set_yticks([])
        if row == 0: a1.set_title(f"latent 24  (cos {r.C_latent:.2f})", fontsize=8.5)
        # themes
        a2.bar(range(11), tR[i], color=OI["blue"], alpha=0.6, width=0.8)
        a2.bar(range(11), -tS[i], color=OI["verm"], alpha=0.6, width=0.8)
        a2.set_yticks([])
        if row == 0: a2.set_title(f"themes  (raw {r.C_theme_raw:.2f})", fontsize=8.5)
        # identity residual
        a3.bar(range(11), tR[i] - muR, color=OI["blue"], alpha=0.6, width=0.8)
        a3.bar(range(11), -(tS[i] - muR), color=OI["verm"], alpha=0.6, width=0.8)
        a3.set_yticks([])
        ev = []
        if r.latent_recovered: ev.append("latent✓")
        if r.MSS_recovered: ev.append("MSS✓")
        if r.theme_recovered: ev.append("theme✓")
        if r.perturbation_validated: ev.append("perturb✓")
        if r.matrix_recovered: ev.append("matrix✓")
        if row == 0: a3.set_title(f"identity residual  ({r.theme_identity:.2f})", fontsize=8.5)
        a3.text(1.02, 0.5, "\n".join(ev) or "broad/none", transform=a3.transAxes, fontsize=7,
                va="center", color=OI["green"] if ev else OI["grey"])
    axes[0, 0].legend(fontsize=6.5, loc="upper right")
    fig.suptitle("Representative analytes: spectrum → latent → themes → identity residual → evidence "
                 "(blue = Raman, red = Ag-SERS)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 0.97, 0.97]); fig.savefig(FIG / "fig11_representative_panels.png", bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    for f in [fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10, fig11]:
        f(); print(f.__name__, "ok")
    print("figures ->", FIG)
