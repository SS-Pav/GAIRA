"""V5 — 12 publication-quality figures for the abstraction-recovery analysis. Static PNGs
(auditable). Okabe-Ito. Reads committed V5 tables + vectors_v5.npz."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

REPO = Path("/Users/surajpg/projects/GAIRA")
BASE = REPO / "results/v5_rebuild/abstraction_recovery_v5"
FIG = BASE / "figures"; FIG.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(BASE / "tables/per_analyte_abstraction_recovery.csv")
S = json.loads((BASE / "artifacts/abstraction_summary.json").read_text())
LAD = pd.read_csv(BASE / "tables/recovery_by_abstraction_level.csv")
CLS = pd.read_csv(BASE / "tables/subclass_classification_results.csv")
NN = pd.read_csv(BASE / "tables/nearest_neighbor_retrieval.csv")
FAMB = pd.read_csv(BASE / "tables/family_abstraction_breakdown.csv")
V = np.load(BASE / "artifacts/vectors_v5.npz", allow_pickle=True)
AN = list(V["analytes"]); THEMES = list(V["themes"]); MOTIFS = list(V["motifs"]); GRID = V["grid"]
idx = {a: i for i, a in enumerate(AN)}
OI = {"black": "#111418", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
      "yellow": "#F0E442", "blue": "#0072B2", "verm": "#D55E00", "purple": "#CC79A7", "grey": "#8A929C"}
plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True, "grid.alpha": 0.22,
                     "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False})


def fig1():  # evaluation hierarchy
    fig, ax = plt.subplots(figsize=(11, 8)); ax.axis("off")
    levels = [("L0", "Exact analyte identity", OI["black"], "7/51 · specific · highest resolution"),
              ("L1", "NMF component evidence", OI["verm"], "2/51 · emergent basis overlap"),
              ("L2", "MSS motif", OI["orange"], "present 19/48 · specific 2/48"),
              ("L3", "Molecular subclass (overlay)", OI["blue"], "LOAO ~chance · NN below chance"),
              ("L4", "Broad biochemical theme", OI["sky"], "present 25/51 · specific 1/51"),
              ("L5", "Perturbation (functional)", OI["green"], "3/51 · strongest beyond identity"),
              ("L6", "Matrix (serum)", OI["purple"], "9/51 · separate property")]
    y = 0.95
    for i, (lv, nm, col, res) in enumerate(levels):
        ax.add_patch(FancyBboxPatch((0.06, y - 0.105), 0.88, 0.095, boxstyle="round,pad=0.008",
                     linewidth=0, facecolor=col, alpha=0.93, transform=ax.transAxes))
        ax.text(0.09, y - 0.04, f"{lv} · {nm}", fontsize=12, fontweight="bold", color="white",
                transform=ax.transAxes, va="center")
        ax.text(0.62, y - 0.04, res, fontsize=8.4, color="white", transform=ax.transAxes, va="center")
        if i < len(levels) - 1:
            ax.add_patch(FancyArrowPatch((0.5, y - 0.105), (0.5, y - 0.135), transform=ax.transAxes,
                         arrowstyle="-|>", mutation_scale=14, color=OI["black"], lw=1.4))
        y -= 0.132
    ax.text(0.5, 0.99, "V5 evaluation hierarchy — identity is strictest; abstraction increases downward",
            fontsize=13, fontweight="bold", ha="center", transform=ax.transAxes)
    ax.text(0.5, 0.01, "abstraction raises apparent PRESENCE, not analyte-SPECIFIC recovery",
            fontsize=9, style="italic", ha="center", color=OI["grey"], transform=ax.transAxes)
    fig.savefig(FIG / "fig01_evaluation_hierarchy.png", bbox_inches="tight"); plt.close(fig)


def fig2():  # recovery by abstraction level (graded)
    d = LAD.copy(); d["label"] = d.level + "\n" + d.tier
    fig, ax = plt.subplots(figsize=(13, 6))
    colmap = {"specific": OI["blue"], "present": OI["sky"], "top-3": OI["verm"], "NN": OI["grey"],
              "LOAO": OI["grey"], "functional": OI["green"], "strong": OI["purple"]}
    cols = []
    for t in d.tier:
        c = OI["orange"]
        for k, v in colmap.items():
            if k in t: c = v; break
        cols.append(c)
    x = np.arange(len(d))
    ax.bar(x, d.fraction, color=cols, edgecolor="white")
    for i, r in d.iterrows():
        ax.text(i, r.fraction + 0.008, f"{int(r.n_recovered)}/{int(r.denominator)}", ha="center", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels(d.label, rotation=40, ha="right", fontsize=6.8)
    ax.set_ylabel("fraction recovered"); ax.set_ylim(0, 0.58)
    ax.set_title("Recovery by abstraction level (graded; denominators explicit)\n"
                 "PRESENCE (light) rises with abstraction; SPECIFIC recovery (dark) stays rare", fontsize=11)
    fig.tight_layout(); fig.savefig(FIG / "fig02_recovery_by_level.png", bbox_inches="tight"); plt.close(fig)


def fig3():  # per-analyte recovery ladder
    cols = [("latent_identity_recovered", "exact"), ("component_recovered", "component"),
            ("mss_present_top3", "MSS present"), ("mss_motif_recovered", "MSS specific"),
            ("subclass_loao_recovered", "subclass"), ("family_loao_recovered", "family"),
            ("theme_present_top3", "theme present"), ("theme_recovered", "theme specific"),
            ("perturbation_status", "perturb"), ("matrix_recovered", "matrix")]
    d = df.sort_values(["broad_family", "latent_identity_recovered", "mss_present_top3"], ascending=[True, False, False]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(11, 13))
    for ci, (col, _) in enumerate(cols):
        for ri in range(len(d)):
            r = d.iloc[ri]
            if col == "perturbation_status":
                tested = r[col] != "not tested"; val = tested
                if not tested: ax.text(ci, ri, "·", ha="center", va="center", color=OI["grey"]); continue
            elif col == "matrix_recovered":
                tested = r.serum_tier != "not tested"; val = r[col]
                if not tested: ax.text(ci, ri, "·", ha="center", va="center", color=OI["grey"]); continue
            elif col in ("mss_motif_recovered",) and r.expected_mss == "unassigned":
                ax.text(ci, ri, "u", ha="center", va="center", color=OI["grey"], fontsize=7); continue
            else:
                val = bool(r[col])
            spec = "specific" in _
            color = (OI["blue"] if spec else OI["sky"]) if val else "#eef0f3"
            ax.add_patch(plt.Rectangle((ci - 0.46, ri - 0.46), 0.92, 0.92, color=color))
            if val: ax.text(ci, ri, "✓", ha="center", va="center", color="white", fontsize=7)
    ax.set_xlim(-0.5, len(cols) - 0.5); ax.set_ylim(len(d) - 0.5, -0.5)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([c[1] for c in cols], rotation=35, ha="left", fontsize=8)
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks(range(len(d))); ax.set_yticklabels(d.analyte, fontsize=6.4); ax.grid(False)
    ax.set_title("Per-analyte recovery ladder — dark ✓ = specific, light ✓ = present-only, · not tested, u unassigned\n"
                 "sorted by family; presence-only cells dominate", fontsize=9.5, pad=26)
    fig.tight_layout(); fig.savefig(FIG / "fig03_recovery_ladder.png", bbox_inches="tight"); plt.close(fig)


def fig4():  # highest recovered level
    hc = S["highest_level_counts"]
    order = ["exact analyte", "component/motif (specific)", "theme (specific)", "perturbation-only",
             "broad presence only (non-specific)", "none"]
    vals = [hc.get(k, 0) for k in order]
    cols = [OI["black"], OI["orange"], OI["sky"], OI["green"], OI["grey"], "#c9ced6"]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.barh(range(len(order)), vals, color=cols, edgecolor="white")
    for i, v in enumerate(vals): ax.text(v + 0.3, i, str(v), va="center", fontsize=10)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order); ax.invert_yaxis()
    ax.set_xlabel("analytes"); ax.set_title("Highest STATISTICALLY-DEFENSIBLE recovery level per analyte\n"
                 "22 reach only non-specific broad presence; 19 nothing; classification (at chance) not counted", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "fig04_highest_level.png", bbox_inches="tight"); plt.close(fig)


def fig5():  # latent vs MSS vs theme + Raman control — THE key figure
    fig, ax = plt.subplots(figsize=(11, 6))
    grans = ["subclass", "family", "theme"]; spaces = ["latent", "MSS", "theme"]
    x = np.arange(len(grans)); w = 0.2
    cm = {"latent": OI["verm"], "MSS": OI["orange"], "theme": OI["blue"]}
    for k, sp in enumerate(spaces):
        vals = [CLS[(CLS.granularity == g) & (CLS.space == sp)].balanced_accuracy.iloc[0] for g in grans]
        ax.bar(x + (k - 1) * w, vals, w, color=cm[sp], label=f"Ag-SERS: {sp} space")
    ctrl = [CLS[(CLS.granularity == g) & (CLS.space.str.contains("control"))].balanced_accuracy.iloc[0] for g in grans]
    ax.plot(x, ctrl, "o-", color=OI["green"], lw=2.4, ms=9, label="Raman→Raman CONTROL (latent)", zorder=5)
    chance = {"subclass": 0.1, "family": 0.09, "theme": 0.09}
    ax.plot(x, [chance[g] for g in grans], "--", color=OI["grey"], label="~chance (balanced)")
    ax.set_xticks(x); ax.set_xticklabels(grans); ax.set_ylabel("balanced accuracy (LOAO)")
    ax.set_title("Abstraction helps WITHIN Raman (green control rises 0.23→0.42) but the Ag-SERS\n"
                 "modality gap collapses it to ~chance (bars). The failure is the modality, not the taxonomy.", fontsize=10)
    ax.legend(fontsize=8.5, loc="upper left"); ax.set_ylim(0, 0.5)
    fig.tight_layout(); fig.savefig(FIG / "fig05_classification_control.png", bbox_inches="tight"); plt.close(fig)


def fig6():  # MSS expected-motif recovery ranking
    d = df[df.expected_mss != "unassigned"].copy()
    d["mss_score"] = d.mss_enrich_null.fillna(-1)
    d = d.sort_values("mss_score")
    fig, ax = plt.subplots(figsize=(9, 12))
    cols = [OI["blue"] if r else (OI["sky"] if p else "#d3d8de") for r, p in zip(d.mss_motif_recovered, d.mss_present_top3)]
    ax.barh(range(len(d)), d.mss_score, color=cols, edgecolor="white")
    ax.axvline(0, color=OI["black"], lw=1)
    ax.set_yticks(range(len(d))); ax.set_yticklabels([f"{a} [{m}]" for a, m in zip(d.analyte, d.expected_mss)], fontsize=6.4)
    ax.set_xlabel("expected-motif enrichment over out-family null95")
    ax.set_title("MSS expected-motif recovery, null-adjusted\n"
                 "dark = specific (top-3 & >null & >bg, n=2) · light = present top-3 · grey = not present", fontsize=9.5)
    fig.tight_layout(); fig.savefig(FIG / "fig06_mss_motif_ranking.png", bbox_inches="tight"); plt.close(fig)


def _confusion(space_label, gran, ax):
    # cross-modal confusion via nearest-centroid predictions recomputed here (latent space)
    y = V["subclass"] if gran == "subclass" else V["families"]
    ZR, ZS = V["ZR"], V["ZS"]; N = len(AN)
    def cos(a, b): return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    classes = sorted(set(y)); pred = []
    for i in range(N):
        cls = {c: ZR[[j for j in range(N) if j != i and y[j] == c]].mean(0) for c in classes
               if [j for j in range(N) if j != i and y[j] == c]}
        pred.append(max(cls, key=lambda c: cos(ZS[i], cls[c])) if cls else None)
    labs = sorted(set(y))
    M = np.zeros((len(labs), len(labs)))
    li = {l: k for k, l in enumerate(labs)}
    for i in range(N):
        if pred[i] is not None: M[li[y[i]], li[pred[i]]] += 1
    im = ax.imshow(M, cmap="magma", aspect="auto")
    ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, rotation=90, fontsize=6)
    ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs, fontsize=6); ax.grid(False)
    ax.set_title(space_label, fontsize=9); ax.set_xlabel("predicted"); ax.set_ylabel("true")
    return im


def fig7():  # subclass/family confusion (cross-modal latent) + note
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    _confusion("Family confusion (Ag-SERS → Raman centroid, latent)", "family", axes[0])
    _confusion("Subclass confusion (Ag-SERS → Raman centroid, latent)", "subclass", axes[1])
    fig.suptitle("Cross-modal classification confusion — off-diagonal dominates (predictions collapse "
                 "toward attractor-adjacent classes)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(FIG / "fig07_confusion.png", bbox_inches="tight"); plt.close(fig)


def fig8():  # broad theme recovery
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    d = df[df.expected_theme != "unassigned"]
    axes[0].hist(d.expected_theme_rank_S.dropna(), bins=np.arange(0.5, 12.5, 1), color=OI["sky"], edgecolor="white")
    axes[0].axvline(3.5, color=OI["verm"], ls="--", label="top-3 cut")
    axes[0].set_xlabel("expected-theme rank in Ag-SERS"); axes[0].set_ylabel("analytes")
    axes[0].set_title("Expected-theme rank: 25/51 in top-3 (present)\nbut presence ≠ specific", fontsize=9.5); axes[0].legend(fontsize=8)
    axes[1].scatter(d.theme_enrich_null, d.delta_purine, s=36, c=[OI["verm"] if r else OI["grey"] for r in d.theme_recovered])
    axes[1].axvline(0, color=OI["black"], lw=0.8); axes[1].axhline(0, color=OI["grey"], ls=":")
    axes[1].set_xlabel("expected-theme enrichment over family-mismatched null95")
    axes[1].set_ylabel("Δpurine"); axes[1].set_title("Enrichment vs purine pull\nred = specifically recovered (n=1)", fontsize=9.5)
    fig.tight_layout(); fig.savefig(FIG / "fig08_theme_recovery.png", bbox_inches="tight"); plt.close(fig)


def fig9():  # purine attractor correction
    d = df.copy()
    non_purine = d[d.broad_family != "purine"]
    purine_dom = non_purine[non_purine.delta_purine > 0.05]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.scatter(d.delta_purine, d.theme_present_top3.astype(int) + np.random.default_rng(0).normal(0, 0.02, len(d)),
               s=30, alpha=0.5, color=OI["grey"], label="all")
    # highlight non-purines with purine pull that STILL retain expected MSS present
    keep = non_purine[(non_purine.delta_purine > 0.03) & (non_purine.mss_present_top3)]
    ax.scatter(keep.delta_purine, [1.15] * len(keep), s=60, color=OI["green"], label="non-purine, expected MSS present despite purine pull")
    fig.text(0.5, 0.02, "Non-purines pulled toward purine that still show their expected MSS in top-3: "
             f"{len(keep)} — genuine motif presence separable from attractor", ha="center", fontsize=8.5, color=OI["black"])
    ax.set_xlabel("Δpurine (Ag − Raman)"); ax.set_yticks([0, 1]); ax.set_yticklabels(["theme not top-3", "theme top-3"])
    ax.set_title("Purine-attractor correction: which non-purines retain expected chemistry despite purine pull", fontsize=10)
    ax.legend(fontsize=8, loc="center right")
    fig.tight_layout(rect=[0, 0.04, 1, 1]); fig.savefig(FIG / "fig09_purine_correction.png", bbox_inches="tight"); plt.close(fig)


def fig10():  # representative analyte panels
    reps = ["adenine", "ergothioneine", "urate", "xanthine", "hypoxanthine", "creatinine",
            "glucose", "tyrosine", "uracil", "oleate", "n-acetylglucosamine", "albumin"]
    reps = [r for r in reps if r in idx]
    ZR, ZS, TR, TS = V["ZR"], V["ZS"], V["TR"], V["TS"]; ram, sers = V["ram_spec"], V["sers_spec"]
    fig, axes = plt.subplots(len(reps), 3, figsize=(13, 1.9 * len(reps)))
    for row, a in enumerate(reps):
        i = idx[a]; r = df[df.analyte == a].iloc[0]; a0, a1, a2 = axes[row]
        a0.plot(GRID[:len(ram[i])] if len(GRID) == len(ram[i]) else np.arange(len(ram[i])), ram[i] / (ram[i].max() + 1e-9), color=OI["blue"], lw=0.6)
        a0.plot(GRID[:len(sers[i])] if len(GRID) == len(sers[i]) else np.arange(len(sers[i])), sers[i] / (sers[i].max() + 1e-9) - 1.1, color=OI["verm"], lw=0.6)
        a0.set_yticks([]); a0.set_ylabel(a, fontsize=8, fontweight="bold")
        if row == 0: a0.set_title("spectra (R blue / SERS red)", fontsize=8)
        a1.bar(range(11), TR[i], color=OI["blue"], alpha=0.6, width=0.8); a1.bar(range(11), -TS[i], color=OI["verm"], alpha=0.6, width=0.8)
        a1.set_yticks([])
        if row == 0: a1.set_title("themes (R up / SERS down)", fontsize=8)
        ev = []
        if r.latent_identity_recovered: ev.append("exact✓")
        if r.mss_motif_recovered: ev.append("MSS-spec✓")
        elif r.mss_present_top3: ev.append("MSS present")
        if r.theme_recovered: ev.append("theme-spec✓")
        elif r.theme_present_top3: ev.append("theme present")
        if r.perturbation_status != "not tested": ev.append("perturb✓")
        a2.axis("off")
        a2.text(0.0, 0.5, f"subclass: {r.subclass}\nexp MSS: {r.expected_mss} (rank {r.mss_rank_S})\n"
                f"exp theme: {r.expected_theme} (rank {r.expected_theme_rank_S})\n" + " · ".join(ev or ["broad/none"]),
                fontsize=7, va="center", transform=a2.transAxes)
        if row == 0: a2.set_title("evidence", fontsize=8)
    fig.suptitle("Representative analytes: spectrum → themes → evidence profile", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98]); fig.savefig(FIG / "fig10_representative.png", bbox_inches="tight"); plt.close(fig)


def fig11():  # perturbation overlay
    val = json.loads((REPO / "results/v5_rebuild/foundation_audit/tables/validation_results.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, key, col, ttl in [(axes[0], "3_adenine_dose", OI["blue"], "adenine (exact weak → purine dose strong)"),
                              (axes[1], "4_ergothioneine_dose", OI["green"], "ergothioneine (identity weak → sulfur dose)")]:
        d = val[key]; x = np.array(d["levels_uM"], float); yv = np.array(d["theme_series"], float); o = np.argsort(x)
        ax.plot(x[o], yv[o], "o-", color=col, lw=1.8); ax.set_xlabel("conc (µM)"); ax.set_ylabel(f"{d['theme']} share")
        ax.set_title(f"{ttl}\nρ={d['monotonicity_rho']}", fontsize=8.5)
    u = val["6_uricase_depletion"]
    axes[2].bar(["oxopurine\nmotif", "purine\ntheme"], [u["delta_oxopurine_motif"], u["purine_delta"]], color=[OI["verm"], OI["grey"]])
    axes[2].axhline(0, color=OI["black"], lw=0.8); axes[2].set_title("urate (motif depletion > broad-theme change)", fontsize=8.5)
    fig.suptitle("Perturbation overlay — functional response recovers class chemistry that static Ag-SERS cannot", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig(FIG / "fig11_perturbation_overlay.png", bbox_inches="tight"); plt.close(fig)


def fig12():  # pure abstraction vs serum recovery
    d = df[df.serum_tier != "not tested"].copy()
    tier_num = {"strong": 2, "moderate": 1, "weak": 0}
    d["snum"] = d.serum_tier.map(tier_num)
    fig, ax = plt.subplots(figsize=(9, 5.6))
    cats = [("MSS present", d.mss_present_top3), ("MSS specific", d.mss_motif_recovered),
            ("theme present", d.theme_present_top3), ("exact", d.latent_identity_recovered)]
    for k, (nm, mask) in enumerate(cats):
        strong_rate = d[mask].matrix_recovered.mean() if mask.sum() else 0
        ax.bar(k, strong_rate, color=OI["purple"])
        ax.text(k, strong_rate + 0.01, f"{d[mask].matrix_recovered.sum()}/{mask.sum()}", ha="center", fontsize=8)
    ax.set_xticks(range(len(cats))); ax.set_xticklabels([c[0] for c in cats])
    ax.set_ylabel("fraction serum-strong"); ax.set_ylim(0, 1)
    ax.set_title("Does pure abstraction recovery predict serum (matrix) recovery?\n"
                 "no monotonic relationship — matrix is a separate property", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "fig12_abstraction_vs_serum.png", bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    for f in [fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10, fig11, fig12]:
        f(); print(f.__name__, "ok")
    print("figures ->", FIG)
