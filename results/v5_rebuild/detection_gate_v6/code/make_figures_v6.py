"""V6 — 10 publication-quality figures for the detection-gate analysis. Static PNGs (auditable).
Okabe-Ito. Reads committed V6 tables + detection_spectra.npz."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.signal import find_peaks
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

REPO = Path("/Users/surajpg/projects/GAIRA")
BASE = REPO / "results/v5_rebuild/detection_gate_v6"
FIG = BASE / "figures"; FIG.mkdir(parents=True, exist_ok=True)
DET = pd.read_csv(BASE / "tables/detection_metrics.csv")
LAD = pd.read_csv(BASE / "tables/recovery_detectable_vs_all.csv")
TD = pd.read_csv(BASE / "tables/per_analyte_transfer_decision.csv")
S = json.loads((BASE / "artifacts/detection_summary.json").read_text())
R = json.loads((BASE / "artifacts/restricted_hierarchy_summary.json").read_text())
V = np.load(BASE / "artifacts/detection_spectra.npz", allow_pickle=True)
AN = list(V["analytes"]); GRID = V["grid"]; BLANK = V["blank_mean"]; SP = V["spectra"]
idx = {a: i for i, a in enumerate(AN)}
OI = {"black": "#111418", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
      "yellow": "#F0E442", "blue": "#0072B2", "verm": "#D55E00", "purple": "#CC79A7", "grey": "#8A929C"}
TIERCOL = {"GOOD": OI["green"], "MODERATE": OI["sky"], "POOR": OI["orange"], "UNDETECTABLE": OI["verm"]}
plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True, "grid.alpha": 0.22,
                     "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False})


def fig1():  # detection → recovery hierarchy
    fig, ax = plt.subplots(figsize=(10.5, 7.5)); ax.axis("off")
    stages = [("Stage 0", "Can Ag-SERS OBSERVE it?", OI["purple"], f"{S['n_pass']}/51 pass · {S['n_fail']} undetectable"),
              ("Stage 1", "Exact latent identity", OI["black"], "7/51 all · 5/22 detectable"),
              ("Stage 2", "MSS motif", OI["orange"], "present 55% / specific 10% (detectable)"),
              ("Stage 3", "Biochemical theme", OI["blue"], "present 59% / specific 4.5% (detectable)"),
              ("Stage 4", "Perturbation", OI["green"], "3 analytes (functional)"),
              ("Stage 5", "Matrix (serum)", OI["sky"], "separate property")]
    y = 0.93
    for i, (st, q, col, res) in enumerate(stages):
        ax.add_patch(FancyBboxPatch((0.06, y - 0.115), 0.88, 0.1, boxstyle="round,pad=0.008",
                     linewidth=0, facecolor=col, alpha=0.93, transform=ax.transAxes))
        ax.text(0.09, y - 0.045, f"{st} · {q}", fontsize=11.5, fontweight="bold", color="white", transform=ax.transAxes, va="center")
        ax.text(0.62, y - 0.045, res, fontsize=8.2, color="white", transform=ax.transAxes, va="center")
        if i < len(stages) - 1:
            ax.add_patch(FancyArrowPatch((0.5, y - 0.115), (0.5, y - 0.15), transform=ax.transAxes,
                         arrowstyle="-|>", mutation_scale=14, color=OI["black"], lw=1.4))
        y -= 0.148
    ax.text(0.5, 0.99, "V6 hierarchy — Stage 0 detection gates every later stage", fontsize=13, fontweight="bold", ha="center", transform=ax.transAxes)
    ax.text(0.5, 0.0, "only DETECTABLE analytes are eligible for identity / motif / theme evaluation",
            fontsize=9, style="italic", ha="center", color=OI["grey"], transform=ax.transAxes)
    fig.savefig(FIG / "fig01_detection_hierarchy.png", bbox_inches="tight"); plt.close(fig)


def fig2():  # detection score distribution
    d = DET.sort_values("detection_confidence", ascending=False)
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(range(len(d)), d.detection_confidence, color=[TIERCOL[t] for t in d.detection_tier], edgecolor="white")
    for name, t in S["tier_thresholds"].items():
        if t > 0:
            ax.axhline(t, ls="--", color=OI["grey"], lw=0.8); ax.text(len(d) - 0.5, t + 0.005, name, fontsize=7.5, ha="right")
    ax.set_xticks(range(len(d))); ax.set_xticklabels(d.analyte, rotation=90, fontsize=5.5)
    ax.set_ylabel("detection confidence (0–1)")
    ax.set_title(f"Stage-0 detection confidence — {S['n_pass']}/51 pass (GOOD+MODERATE), {S['n_fail']} fail\n"
                 "deterministic weighted score: replicate reproducibility + peak SNR + variance concentration", fontsize=10.5)
    handles = [plt.Rectangle((0, 0), 1, 1, color=TIERCOL[t]) for t in TIERCOL]
    ax.legend(handles, list(TIERCOL), fontsize=8.5, loc="upper right")
    fig.tight_layout(); fig.savefig(FIG / "fig02_detection_distribution.png", bbox_inches="tight"); plt.close(fig)


def _spec_panel(ax, a):
    s = SP[idx[a]]; sn = s / (s.max() + 1e-9); bn = BLANK / (BLANK.max() + 1e-9)
    diff = sn - bn
    ax.plot(GRID, sn, color=OI["blue"], lw=0.7, label="Ag-SERS")
    ax.plot(GRID, bn - 1.15, color=OI["grey"], lw=0.6, label="blank")
    ax.plot(GRID, diff - 2.4, color=OI["verm"], lw=0.6, label="difference")
    mn = s - np.median(s); pk, _ = find_peaks(mn / (mn.max() + 1e-12), prominence=0.08, distance=5)
    ax.plot(GRID[pk], sn[pk] + 0.05, "v", color=OI["green"], ms=3)
    r = DET.set_index("analyte").loc[a]
    ax.set_title(f"{a}  DC={r.detection_confidence:.2f} · {r.detection_tier}", fontsize=8.5)
    ax.set_yticks([]); ax.set_xlim(450, 1800)


def fig3():  # representative spectra pass/fail
    picks = ["xanthine", "ergothioneine", "urate", "guanine", "hypoxanthine", "adenine",
             "uracil", "glycine", "glucose", "tyrosine", "oleate", "methionine"]
    picks = [p for p in picks if p in idx]
    fig, axes = plt.subplots(3, 4, figsize=(16, 9))
    for ax, a in zip(axes.ravel(), picks):
        _spec_panel(ax, a)
    axes[0, 0].legend(fontsize=6.5, loc="upper right")
    fig.suptitle("Representative Ag-SERS spectra: Ag-SERS (blue) · blank (grey) · difference (red) · peaks (▼)\n"
                 "top row PASS (sharp reproducible peaks); bottom rows FAIL (blank-like, structureless)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(FIG / "fig03_representative_spectra.png", bbox_inches="tight"); plt.close(fig)


def fig4():  # blank vs analyte overlays (a few clean examples)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, a in zip(axes, ["xanthine", "adenine", "glucose"]):
        s = SP[idx[a]]; ax.plot(GRID, s / (s.max() + 1e-9), color=OI["blue"], lw=0.9, label=f"{a} Ag-SERS")
        ax.plot(GRID, BLANK / (BLANK.max() + 1e-9), color=OI["grey"], lw=0.9, label="Ag blank")
        r = DET.set_index("analyte").loc[a]
        ax.set_title(f"{a}  DC={r.detection_confidence:.2f} {r.detection_tier}", fontsize=9.5)
        ax.set_xlim(450, 1800); ax.set_yticks([]); ax.legend(fontsize=8); ax.set_xlabel("cm⁻¹")
    fig.suptitle("Analyte vs blank — a detectable analyte rises clearly above the Ag background; an "
                 "undetectable one tracks the blank", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(FIG / "fig04_blank_overlays.png", bbox_inches="tight"); plt.close(fig)


def fig5():  # recovery hierarchy detectable vs all
    d = LAD.copy(); x = np.arange(len(d)); w = 0.38
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x - w / 2, d.all_frac, w, color=OI["grey"], label="all 51")
    ax.bar(x + w / 2, d.detectable_frac, w, color=OI["green"], label="detectable-only")
    for i, r in d.iterrows():
        ax.text(i - w / 2, r.all_frac + 0.008, f"{int(r.all_n)}/{int(r.all_denom)}", ha="center", fontsize=6.5)
        ax.text(i + w / 2, r.detectable_frac + 0.008, f"{int(r.detectable_n)}/{int(r.detectable_denom)}", ha="center", fontsize=6.5)
    ax.set_xticks(x); ax.set_xticklabels(d.level, rotation=25, ha="right", fontsize=8.5)
    ax.set_ylabel("fraction recovered"); ax.set_ylim(0, 0.72)
    ax.set_title("Recovery: all 51 vs DETECTABLE-only — removing measurement failure lifts every level,\n"
                 "but analyte-SPECIFIC recovery (MSS/theme specific) stays low → residual is representational", fontsize=10.5)
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "fig05_recovery_detectable.png", bbox_inches="tight"); plt.close(fig)


def fig6():  # abstraction recovery gain
    d = LAD.copy()
    fig, ax = plt.subplots(figsize=(11, 5.4))
    cols = [OI["green"] if g > 0 else OI["grey"] for g in d.gain]
    ax.bar(range(len(d)), d.gain, color=cols, edgecolor="white")
    for i, g in enumerate(d.gain): ax.text(i, g + 0.003, f"+{g:.02f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(d))); ax.set_xticklabels(d.level, rotation=25, ha="right", fontsize=8.5)
    ax.set_ylabel("recovery gain (detectable − all)")
    ax.set_title("Recovery gain once measurement failure is removed\n"
                 "presence gains most; specific-recovery gain is small — the representation gap is real", fontsize=10.5)
    fig.tight_layout(); fig.savefig(FIG / "fig06_abstraction_gain.png", bbox_inches="tight"); plt.close(fig)


def fig7():  # transfer decision tree
    fig, ax = plt.subplots(figsize=(11, 7)); ax.axis("off")
    def box(x, y, w, h, text, col):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01", facecolor=col, alpha=0.92, linewidth=0, transform=ax.transAxes))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9, color="white", transform=ax.transAxes, wrap=True)
    tc = R["transfer_cases"]
    box(0.35, 0.86, 0.3, 0.1, "Ag-SERS spectrum", OI["black"])
    box(0.06, 0.6, 0.32, 0.13, f"Stage 0: detectable?\nNO → CASE A\nmeasurement-limited ({tc.get('A · measurement-limited',0)})\nneed better substrate", OI["verm"])
    box(0.62, 0.6, 0.32, 0.13, "YES → detectable\ngo to Stage 1", OI["green"])
    box(0.4, 0.36, 0.5, 0.13, f"Exact identity recovered?\nYES → CASE C already recoverable ({tc.get('C · already recoverable',0)})\ntransfer unnecessary", OI["blue"])
    box(0.1, 0.1, 0.5, 0.14, f"NO but chemistry present → CASE B\nrepresentation-limited, PROMISING ({tc.get('B · representation-limited (promising)',0)})\na learned Raman→SERS transfer model MAY help", OI["orange"])
    box(0.64, 0.1, 0.3, 0.14, f"CASE B hard ({tc.get('B · representation-limited (hard)',0)})\nno chemistry present", OI["grey"])
    for a, b in [((0.5, 0.86), (0.5, 0.73)), ((0.5, 0.73), (0.22, 0.73)), ((0.5, 0.73), (0.78, 0.73)),
                 ((0.78, 0.6), (0.65, 0.49)), ((0.5, 0.36), (0.35, 0.24)), ((0.6, 0.36), (0.79, 0.24))]:
        ax.add_patch(FancyArrowPatch(a, b, transform=ax.transAxes, arrowstyle="-|>", mutation_scale=12, color=OI["black"], lw=1.2))
    ax.text(0.5, 0.99, "Transfer-function decision tree", fontsize=13, fontweight="bold", ha="center", transform=ax.transAxes)
    fig.savefig(FIG / "fig07_transfer_decision.png", bbox_inches="tight"); plt.close(fig)


def fig8():  # roadmap
    road = R["roadmap_groups"]
    order = ["already recoverable", "potentially recoverable (transfer worth trying)",
             "probably impossible (no chemistry present)", "probably impossible (weak signal)",
             "impossible (measurement-limited)"]
    cols = [OI["green"], OI["orange"], OI["grey"], OI["sky"], OI["verm"]]
    vals = [road.get(k, 0) for k in order]
    fig, ax = plt.subplots(figsize=(11, 5.4))
    ax.barh(range(len(order)), vals, color=cols, edgecolor="white")
    for i, v in enumerate(vals): ax.text(v + 0.2, i, str(v), va="center", fontsize=10)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=9); ax.invert_yaxis()
    ax.set_xlabel("analytes"); ax.set_title("Learned Raman→SERS transfer roadmap\n"
                 "~11 'potentially recoverable' are the sweet spot; ~33 measurement/signal-limited need a better substrate", fontsize=10.5)
    fig.tight_layout(); fig.savefig(FIG / "fig08_transfer_roadmap.png", bbox_inches="tight"); plt.close(fig)


def fig9():  # per-analyte ladder (detection + recovery)
    d = TD.sort_values(["detection_pass", "detection_confidence"], ascending=[False, False]).reset_index(drop=True)
    cols = [("detection_pass", "detected"), ("latent_identity_recovered", "exact"),
            ("mss_present_top3", "MSS present"), ("mss_motif_recovered", "MSS spec"),
            ("theme_present_top3", "theme present"), ("theme_recovered", "theme spec"),
            ("matrix_recovered", "matrix")]
    fig, ax = plt.subplots(figsize=(10, 13))
    for ci, (col, _) in enumerate(cols):
        for ri in range(len(d)):
            val = bool(d.iloc[ri][col])
            spec = "spec" in _ or col == "latent_identity_recovered"
            color = (OI["blue"] if spec else OI["sky"]) if val else "#eef0f3"
            if col == "detection_pass": color = OI["green"] if val else OI["verm"]
            ax.add_patch(plt.Rectangle((ci - 0.46, ri - 0.46), 0.92, 0.92, color=color))
            if val: ax.text(ci, ri, "✓", ha="center", va="center", color="white", fontsize=6.5)
    ax.set_xlim(-0.5, len(cols) - 0.5); ax.set_ylim(len(d) - 0.5, -0.5)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([c[1] for c in cols], rotation=35, ha="left", fontsize=8)
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks(range(len(d))); ax.set_yticklabels(d.analyte, fontsize=6.2); ax.grid(False)
    ax.set_title("Per-analyte: detection (green) then recovery — sorted by detection confidence\n"
                 "recovery lives almost entirely in the detected (green) block", fontsize=9.5, pad=24)
    fig.tight_layout(); fig.savefig(FIG / "fig09_per_analyte_ladder.png", bbox_inches="tight"); plt.close(fig)


def fig10():  # updated summary
    fig, ax = plt.subplots(figsize=(11, 6)); ax.axis("off")
    ab = R["abstraction_improves_after_gate"]
    lines = [
        ("Detection gate", f"{S['n_pass']}/51 detectable · {S['n_fail']} undetectable "
         f"(GOOD {S['tier_counts'].get('GOOD',0)}, MODERATE {S['tier_counts'].get('MODERATE',0)}, "
         f"POOR {S['tier_counts'].get('POOR',0)}, UNDETECTABLE {S['tier_counts'].get('UNDETECTABLE',0)})"),
        ("Exact identity", f"all {ab['exact_all']:.0%} → detectable {ab['exact_detectable']:.0%} (≈ doubled)"),
        ("MSS present", f"all {ab['mss_present_all']:.0%} → detectable {ab['mss_present_detectable']:.0%}"),
        ("MSS / theme specific (detectable)", f"{ab['mss_specific_detectable']:.0%} / {ab['theme_specific_detectable']:.0%} — still low"),
        ("Transfer roadmap", f"{R['roadmap_groups'].get('already recoverable',0)} already · "
         f"{R['roadmap_groups'].get('potentially recoverable (transfer worth trying)',0)} worth trying · "
         f"rest measurement/signal-limited"),
    ]
    y = 0.85
    for k, v in lines:
        ax.text(0.04, y, k, fontsize=12, fontweight="bold", color=OI["black"], transform=ax.transAxes)
        ax.text(0.04, y - 0.055, v, fontsize=10.5, color=OI["grey"], transform=ax.transAxes)
        y -= 0.16
    ax.text(0.5, 0.97, "V6 summary — measurement failure separated from representation failure", fontsize=13, fontweight="bold", ha="center", transform=ax.transAxes)
    ax.text(0.04, 0.03, "Verdict: a learned Raman→SERS transfer model is justified for the ~11 detectable, "
            "representation-limited analytes; the rest need a better substrate, not a model.",
            fontsize=9.5, style="italic", color=OI["black"], transform=ax.transAxes, wrap=True)
    fig.savefig(FIG / "fig10_summary.png", bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    for f in [fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10]:
        f(); print(f.__name__, "ok")
    print("figures ->", FIG)
