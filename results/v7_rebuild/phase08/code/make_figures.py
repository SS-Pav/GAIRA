#!/usr/bin/env python3
"""GAIRA V7 — Phase 08 figures. PNG only, 200 dpi, deterministic."""
from __future__ import annotations

import json, sys, textwrap
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
from gaira.v7.io import PhaseOutputs, frozen_root      # noqa: E402

OUT = PhaseOutputs("08", extra=("interactive", "manifests"))
T, A_, F = OUT.tables, OUT.artifacts, OUT.figures
FROZEN = frozen_root()
INK, MUTED, LINE = "#1a1a1a", "#6b7280", "#9ca3af"
BLUE, GREEN, AMBER, RED, GREY = "#2563eb", "#15803d", "#b45309", "#b91c1c", "#4b5563"
PURPLE = "#7c3aed"
COL = {"A_raw_spectrum": GREY, "B_csm": GREEN, "C_chemistry_rerank": PURPLE,
       "D_probabilistic": RED, "E_bayesian_fusion": BLUE}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
                     "figure.facecolor": "white", "savefig.facecolor": "white",
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.18})


def SH(c):
    return (c.replace("_", " ").replace("carboxylic acid metabolite", "carboxylic acid")
            .replace("phospholipid sphingolipid", "phospholipid")
            .replace("mono oligosaccharide", "mono/oligosacch")
            .replace("sulfur thiol cofactor", "sulfur/thiol")
            .replace("nucleic acid polymer", "nucleic polymer")
            .replace("chromophore pigment", "chromophore").replace("small nitrogenous", "small N"))


def save(fig, name):
    F.mkdir(parents=True, exist_ok=True)
    fig.savefig(F / f"{name}.png", dpi=200); plt.close(fig); print(f"  {name}")


def box(ax, x, y, w, h, t, fc="#eef2ff", ec=BLUE, fs=8.0, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008", fc=fc, ec=ec, lw=1.1))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, color=INK,
            weight=weight, linespacing=1.35)


def arrow(ax, p0, p1, col=LINE, lw=1.2):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=11, color=col, lw=lw,
                                 shrinkA=2, shrinkB=2))


class C:
    def __init__(self):
        self.s = json.loads((A_ / "phase08_summary_v1.json").read_text())
        self.state = json.loads((OUT.root / "PHASE_STATE.json").read_text())
        z = np.load(A_ / "retrieval_ranks_v1.npz", allow_pickle=True)
        self.rk = {k: z[k] for k in z.files if k.startswith(("A_", "B_", "C_", "D_", "E_"))}
        self.y = np.array([str(v) for v in z["y"]])
        self.cls = np.array([str(v) for v in z["cls"]])
        self.a = pd.read_csv(T / "split_a_metrics_v1.csv")
        self.b = pd.read_csv(T / "split_b_metrics_v1.csv")
        self.rd = pd.read_csv(T / "rank_distribution_v1.csv")
        self.cal = pd.read_csv(T / "calibration_v1.csv")
        self.rc = pd.read_csv(T / "risk_coverage_v1.csv")
        self.rob = pd.read_csv(T / "noise_robustness_v1.csv")
        self.robs = pd.read_csv(T / "noise_robustness_summary_v1.csv")
        self.fc = pd.read_csv(T / "failure_by_class_v1.csv")
        self.imp = pd.read_csv(T / "chemistry_axis_importance_v1.csv")
        self.ev = pd.read_csv(T / "rank_evolution_examples_v1.csv")
        self.gates = pd.read_csv(T / "phase08_gates_v1.csv")
        self.dec = json.loads((A_ / "evidence_decomposition_v1.json").read_text())
        self.sig = self.s["significance_vs_csm"]
        self.rel = {m: pd.read_csv(T / f"reliability_{m}_v1.csv")
                    for m in ("B_csm", "C_chemistry_rerank")}


def f01(c):
    fig, ax = plt.subplots(figsize=(11.6, 4.6)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    steps = [(0.01, 0.145, "unknown Raman\nspectrum", "#f8fafc", GREY),
             (0.175, 0.145, "canonical\npreprocessing", "#f8fafc", GREY),
             (0.34, 0.145, "CSM projection\n49-d (frozen)", "#ecfdf5", GREEN),
             (0.505, 0.16, "Chemistry Evidence\n16-d (frozen)", "#ecfdf5", GREEN),
             (0.685, 0.30, "hierarchical molecular retrieval\nCSM shortlist → chemistry rerank\n"
                           "no hard filtering", "#f5f3ff", PURPLE)]
    for i, (x, w, t, fc, ec) in enumerate(steps):
        box(ax, x, 0.60, w, 0.20, t, fc, ec, 8.0, "bold" if i >= 2 else "normal")
        if i:
            arrow(ax, (steps[i-1][0] + steps[i-1][1], 0.70), (x, 0.70))
    box(ax, 0.01, 0.12, 0.47, 0.36,
        "NOT on this path\n\nBSV2 · latent geometry · UMAP · PCA · clustering\n"
        "SERS · Ag-SERS · serum · EV · DART-Met\n\nPure Raman only.", "#fef2f2", RED, 8.0)
    d = c.s["decision"]
    box(ax, 0.51, 0.12, 0.48, 0.36,
        f"RESULT — outcome {d['outcome']}\n\n"
        f"Δtop-1 vs CSM  {d['delta_top1']:+.4f}\n"
        f"95% CI [{d['delta_top1_ci'][0]:+.4f}, {d['delta_top1_ci'][1]:+.4f}]  p = "
        f"{d['delta_top1_p']:.3f}\n\n{d['action']}",
        "#ecfdf5" if d["outcome"] != "A" else "#fffbeb", GREEN if d["outcome"] != "A" else AMBER,
        8.2, "bold")
    ax.set_title("Figure 1 · Phase 08 architecture and canonical inference path", loc="left",
                 fontsize=11.5, weight="bold", color=INK)
    save(fig, "F01_architecture")


def f02(c):
    fig, axs = plt.subplots(1, 3, figsize=(12.6, 4.2))
    order = ["A_raw_spectrum", "B_csm", "C_chemistry_rerank", "D_probabilistic",
             "E_bayesian_fusion"]
    d = c.a.set_index("model")
    ax = axs[0]
    x = np.arange(len(order))
    for k, (m, lab) in enumerate((("top1", "top-1"), ("top5", "top-5"), ("top10", "top-10"))):
        ax.bar(x + (k - 1) * 0.27, [d.loc[o, m] for o in order], 0.25,
               color=[COL[o] for o in order], alpha=0.45 + 0.25 * k, label=lab)
    for i, o in enumerate(order):
        ax.text(i - 0.27, d.loc[o, "top1"] + 0.014, f"{d.loc[o,'top1']:.3f}", ha="center",
                fontsize=6.6, weight="bold")
    ax.set_xticks(x); ax.set_xticklabels([o.split("_")[0] for o in order], fontsize=8)
    ax.set_ylim(0, 1.02); ax.set_ylabel("Split A molecule retrieval")
    ax.legend(frameon=False, fontsize=7, ncol=3)
    ax.set_title("a · top-k", fontsize=9, loc="left")
    ax = axs[1]
    ax.bar(x, [d.loc[o, "mrr"] for o in order], color=[COL[o] for o in order], alpha=0.9,
           width=0.55)
    for i, o in enumerate(order):
        ax.text(i, d.loc[o, "mrr"] + 0.012, f"{d.loc[o,'mrr']:.3f}", ha="center", fontsize=7.6,
                weight="bold")
    ax.set_xticks(x); ax.set_xticklabels([o.split("_")[0] for o in order], fontsize=8)
    ax.set_ylim(0, 0.85); ax.set_ylabel("MRR")
    ax.set_title("b · mean reciprocal rank", fontsize=9, loc="left")
    ax = axs[2]; ax.axis("off")
    ax.text(0.0, 0.98, "paired tests against Model B (CSM)", fontsize=8.6, weight="bold")
    yy = 0.86
    for m in ("A_raw_spectrum", "C_chemistry_rerank", "D_probabilistic", "E_bayesian_fusion"):
        s = c.sig[m]["top1"]
        col = GREEN if s["significant"] else RED
        ax.text(0.0, yy, m.split("_")[0], fontsize=7.6, weight="bold", color=COL[m])
        ax.text(0.13, yy, f"Δ{s['delta']:+.4f}  CI[{s['ci95'][0]:+.3f},{s['ci95'][1]:+.3f}]",
                fontsize=7.0, family="DejaVu Sans Mono")
        ax.text(0.13, yy - 0.055, f"McNemar p={s['p_value']:.4f}   "
                f"{'SIGNIFICANT' if s['significant'] else 'not significant'}",
                fontsize=7.0, color=col)
        yy -= 0.145
    ax.text(0.0, 0.14, "Model C selected beta = gamma = delta = 0 in\nEVERY fold: the inner "
            "cross-validation chose to\nuse no chemistry at all, so C is identical to B.",
            fontsize=7.4, color=INK, style="italic")
    fig.suptitle("Figure 2 · Retrieval performance and significance — Split A, molecule present",
                 x=0.03, ha="left", fontsize=11.5, weight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.90]); save(fig, "F02_performance")


def f03(c):
    fig, axs = plt.subplots(1, 2, figsize=(11.4, 4.2))
    ax = axs[0]
    for m in ("A_raw_spectrum", "B_csm", "C_chemistry_rerank"):
        d = c.rd[c.rd.model == m]
        ax.plot(range(len(d)), np.cumsum(d.share), "o-", color=COL[m], lw=1.5, ms=4, label=m)
    d0 = c.rd[c.rd.model == "B_csm"]
    ax.set_xticks(range(len(d0)))
    ax.set_xticklabels([f"≤{int(v)}" if v < 1e5 else "miss" for v in d0.rank_upper], fontsize=7)
    ax.set_xlabel("rank of the true molecule"); ax.set_ylabel("cumulative share")
    ax.set_ylim(0, 1.02); ax.legend(frameon=False, fontsize=7)
    ax.set_title("a · rank distribution", fontsize=9, loc="left")
    ax = axs[1]
    d = c.b.set_index("model")
    x = np.arange(2)
    for k, (m, lab) in enumerate((("chem_top1", "chemistry top-1"),
                                  ("chem_top3", "chemistry top-3"),
                                  ("analogue_class_correct", "nearest analogue class"))):
        ax.bar(x + (k - 1) * 0.27, [d.loc[o, m] for o in ("B_csm", "C_chemistry_rerank")], 0.25,
               alpha=0.5 + 0.2 * k, color=[GREEN, PURPLE], label=lab)
    ax.set_xticks(x); ax.set_xticklabels(["B — CSM", "C — chemistry rerank"], fontsize=8)
    ax.set_ylim(0, 1.05); ax.legend(frameon=False, fontsize=7)
    ax.set_title("b · Split B — molecule absent; molecule top-1 is UNDEFINED", fontsize=9,
                 loc="left")
    fig.suptitle("Figure 3 · Rank distribution and the held-out-molecule split", x=0.03,
                 ha="left", fontsize=11.5, weight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.90]); save(fig, "F03_ranks_splitb")


def f04(c):
    fig, axs = plt.subplots(1, 3, figsize=(12.4, 4.0))
    ax = axs[0]
    for m in ("B_csm", "C_chemistry_rerank"):
        r = c.rel[m].dropna(subset=["empirical_accuracy"])
        ax.plot(r.bin_center, r.empirical_accuracy, "o-", color=COL[m], lw=1.5, ms=4, label=m)
    ax.plot([0, 1], [0, 1], color=LINE, ls="--", lw=1.0)
    ax.set_xlabel("confidence (score margin, calibrated in-fold)")
    ax.set_ylabel("empirical top-1 accuracy")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(frameon=False, fontsize=7)
    cal = c.cal.set_index("model")
    ax.set_title(f"a · reliability · ECE {cal.loc['B_csm','ece']:.3f}", fontsize=9, loc="left")
    ax = axs[1]
    for m in ("B_csm", "C_chemistry_rerank"):
        d = c.rc[c.rc.model == m]
        ax.plot(d.coverage, d.accuracy, "o-", color=COL[m], lw=1.5, ms=3.5, label=m)
    ax.axhline(0.90, color=RED, ls="--", lw=1.0)
    ax.text(0.05, 0.905, "accuracy 0.90", fontsize=6.8, color=RED)
    ax.set_xlabel("coverage"); ax.set_ylabel("accuracy among answered")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("b · risk–coverage: when to say 'I don't know'", fontsize=9, loc="left")
    ax = axs[2]
    d = c.cal.set_index("model")
    ks = ["ece", "brier", "sharpness", "discrimination"]
    x = np.arange(len(ks))
    ax.bar(x - 0.2, [d.loc["B_csm", k] for k in ks], 0.38, color=GREEN, alpha=0.9, label="B")
    ax.bar(x + 0.2, [d.loc["C_chemistry_rerank", k] for k in ks], 0.38, color=PURPLE, alpha=0.9,
           label="C")
    ax.set_xticks(x); ax.set_xticklabels(ks, fontsize=7.4, rotation=20)
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("c · calibration quality", fontsize=9, loc="left")
    fig.suptitle("Figure 4 · Calibration, selective prediction and abstention", x=0.03,
                 ha="left", fontsize=11.5, weight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.90]); save(fig, "F04_calibration")


def f05(c):
    kinds = list(dict.fromkeys(c.rob.perturbation))
    fig, axs = plt.subplots(2, 4, figsize=(12.8, 5.6))
    for ax, k in zip(axs.ravel(), kinds):
        d = c.rob[c.rob.perturbation == k]
        for m in ("A_raw_spectrum", "B_csm", "C_chemistry_rerank"):
            g = d[d.model == m].sort_values("level")
            ax.plot(range(len(g)), g.top1, "o-", color=COL[m], lw=1.2, ms=3, label=m)
        lv = d[d.model == "B_csm"].sort_values("level").level
        ax.set_xticks(range(len(lv))); ax.set_xticklabels([f"{v:g}" for v in lv], fontsize=6)
        ax.set_ylim(0, 1.05); ax.set_title(k.replace("_", " "), fontsize=8, loc="left")
        ax.tick_params(labelsize=6.5)
    ax = axs.ravel()[-1]; ax.axis("off")
    d = c.robs.set_index("model")
    ax.text(0.0, 0.95, "mean perturbed top-1", fontsize=8.6, weight="bold")
    for i, m in enumerate(("A_raw_spectrum", "B_csm", "C_chemistry_rerank")):
        ax.text(0.0, 0.78 - i * 0.14, f"{m.split('_')[0]:3s}  {d.loc[m,'top1']:.4f}",
                fontsize=8, color=COL[m], family="DejaVu Sans Mono")
    ax.text(0.0, 0.28, "Raw spectrum is MORE robust here because\nthe bank is in-sample: a "
            "perturbed spectrum still\nmatches its own unperturbed reference.\nThis is not a "
            "held-out number.", fontsize=6.8, color=MUTED, style="italic")
    axs.ravel()[0].legend(frameon=False, fontsize=6)
    axs.ravel()[0].set_ylabel("top-1", fontsize=7.4); axs.ravel()[4].set_ylabel("top-1", fontsize=7.4)
    fig.suptitle("Figure 5 · Noise robustness — 7 perturbations × 5 levels", x=0.03, ha="left",
                 fontsize=11.5, weight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.92]); save(fig, "F05_noise")


def f06(c):
    fig, axs = plt.subplots(1, 2, figsize=(11.6, 4.6),
                            gridspec_kw={"width_ratios": [1.3, 1.0]})
    ax = axs[0]
    d = c.fc.sort_values("delta_top1")
    cols = [GREEN if v > 0 else (RED if v < 0 else GREY) for v in d.delta_top1]
    ax.barh(range(len(d)), d.delta_top1, color=cols, alpha=0.9)
    ax.axvline(0, color=INK, lw=0.9)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([f"{SH(a)}  (n={int(n)})" for a, n in zip(d.true_class, d.n)],
                       fontsize=6.8)
    ax.set_xlabel("Δ top-1, chemistry rerank − CSM")
    ax.set_xlim(-0.05, 0.05)
    ax.set_title("a · where chemistry helps or hurts — all zero", fontsize=9, loc="left")
    ax = axs[1]
    d2 = c.imp.sort_values("delta_mrr")
    ax.barh(range(len(d2)), d2.delta_mrr, color=PURPLE, alpha=0.9)
    ax.set_yticks(range(len(d2)))
    ax.set_yticklabels([SH(a) for a in d2.axis], fontsize=6.6)
    ax.axvline(0, color=INK, lw=0.9); ax.set_xlim(-0.02, 0.02)
    ax.set_xlabel("ΔMRR when the axis is permuted")
    ax.set_title("b · chemistry-axis importance — exactly zero for all 16", fontsize=9,
                 loc="left")
    fig.suptitle("Figure 6 · Failure analysis and chemistry-axis contribution\n"
                 "with β = γ = δ = 0 selected in every fold, chemistry contributes nothing and "
                 "both panels are flat by construction",
                 x=0.03, ha="left", fontsize=10.5, weight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.88]); save(fig, "F06_failure_importance")


def f07(c):
    ex = c.dec["examples"][:3]
    fig, axs = plt.subplots(1, 3, figsize=(12.6, 4.6))
    for ax, e in zip(axs, ex):
        terms = e["terms"]
        vals = [t["contribution"] for t in terms]
        names = [t["term"].replace("_", "\n") for t in terms]
        cols = [GREEN if v >= 0 else RED for v in vals]
        ax.bar(range(len(vals)), vals, color=cols, alpha=0.9)
        ax.axhline(0, color=INK, lw=0.9)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.006 if v >= 0 else v - 0.012, f"{v:+.3f}", ha="center",
                    fontsize=6.8)
        ax.set_xticks(range(len(vals))); ax.set_xticklabels(names, fontsize=6.2)
        ax.set_title(f"{e['candidate'][:24]}\nsubtotal {e['terms_subtotal']:.4f} · total "
                     f"{e['score_total']:.4f} · reconciles {e['reconciles']}",
                     fontsize=7.4, loc="left")
    fig.suptitle("Figure 7 · Evidence decomposition — every score sums exactly, no hidden term\n"
                 f"{c.dec['n_non_reconciling']} of {len(c.dec['examples'])} shown decompositions "
                 "fail to reconcile", x=0.03, ha="left", fontsize=10.5, weight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.88]); save(fig, "F07_decomposition")


def f08(c):
    fig, axs = plt.subplots(1, 2, figsize=(11.6, 4.6))
    q = sorted(set(c.ev["query"]))[:2]
    for ax, qq in zip(axs, q):
        d = c.ev[c.ev["query"] == qq].head(8)
        ax.scatter(d.csm_rank, range(len(d)), s=60, color=GREEN, label="CSM rank", zorder=3)
        ax.scatter(d.final_rank, range(len(d)), s=60, color=PURPLE, marker="s",
                   label="final rank", zorder=3)
        for i, r in enumerate(d.itertuples()):
            ax.plot([r.csm_rank, r.final_rank], [i, i], color=LINE, lw=1.0, zorder=1)
        ax.set_yticks(range(len(d)))
        ax.set_yticklabels([m[:26] for m in d.molecule], fontsize=6.6)
        ax.invert_yaxis(); ax.set_xlabel("rank")
        ax.legend(frameon=False, fontsize=7)
        ax.set_title(f"query {qq} · truth {d.truth.iloc[0][:26]}", fontsize=8, loc="left")
    fig.suptitle("Figure 8 · Candidate evolution — CSM ranking versus final ranking\n"
                 "the two coincide exactly, because reranking selected zero chemistry weight",
                 x=0.03, ha="left", fontsize=10.5, weight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.88]); save(fig, "F08_rank_evolution")


def f09(c):
    fig = plt.figure(figsize=(12.0, 6.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.40, wspace=0.26)
    ax = fig.add_subplot(gs[0, 0])
    d = c.a.set_index("model")
    order = ["A_raw_spectrum", "B_csm", "C_chemistry_rerank", "D_probabilistic",
             "E_bayesian_fusion"]
    ax.scatter([d.loc[o, "top1"] for o in order], [d.loc[o, "mrr"] for o in order],
               s=140, c=[COL[o] for o in order])
    for o in order:
        ax.annotate(o.split("_")[0], (d.loc[o, "top1"], d.loc[o, "mrr"]),
                    textcoords="offset points", xytext=(9, -3), fontsize=8)
    ax.set_xlabel("Split A top-1"); ax.set_ylabel("MRR")
    ax.set_title("a · all five models", fontsize=9, loc="left")
    ax = fig.add_subplot(gs[0, 1]); ax.axis("off")
    s, dd = c.s, c.s["decision"]
    rows = [("baselines reproduced", "EXACT" if s["baselines_reproduced"] else "NO"),
            ("Model B (CSM) top-1", f"{d.loc['B_csm','top1']:.4f}"),
            ("Model C top-1", f"{d.loc['C_chemistry_rerank','top1']:.4f}"),
            ("Δ top-1", f"{dd['delta_top1']:+.4f}"),
            ("95% CI", f"[{dd['delta_top1_ci'][0]:+.4f}, {dd['delta_top1_ci'][1]:+.4f}]"),
            ("McNemar p", f"{dd['delta_top1_p']:.4f}"),
            ("any metric significant", str(dd["any_significant"])),
            ("decompositions non-reconciling", str(s["explainability"]["non_reconciling"])),
            ("OUTCOME", dd["outcome"])]
    ax.text(0.0, 1.0, "decision", fontsize=9.4, weight="bold", va="top")
    for i, (k, v) in enumerate(rows):
        ax.text(0.0, 0.87 - i * 0.098, k, fontsize=8, color=INK)
        ax.text(1.0, 0.87 - i * 0.098, v, fontsize=8, ha="right", weight="bold",
                color=AMBER if k == "OUTCOME" else GREEN, family="DejaVu Sans Mono")
    ax = fig.add_subplot(gs[1, :]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    g = c.gates
    ncol = 6; nrow = int(np.ceil(len(g) / ncol))
    h = min(0.26, (0.92 - 0.10) / nrow - 0.030)
    for i, (_, r) in enumerate(g.iterrows()):
        gx = 0.008 + (i % ncol) * 0.166
        gy = 0.84 - (i // ncol) * (h + 0.032)
        ok = r.status == "PASS"
        box(ax, gx, gy, 0.156, h, "\n".join(textwrap.wrap(r.gate, 26)),
            "#ecfdf5" if ok else "#fef2f2", GREEN if ok else RED, 5.6)
    ax.text(0.008, 0.03, f"{int((g.status=='PASS').sum())} of {len(g)} gates pass · "
            f"outcome {dd['outcome']} — {dd['action']} · BSV2 not on the inference path",
            fontsize=7.4, color=MUTED)
    fig.suptitle("Figure 9 · Summary and decision gate", x=0.03, ha="left", fontsize=11.5,
                 weight="bold", color=INK)
    save(fig, "F09_summary")


def main():
    c = C(); print("[figures]")
    for fn in (f01, f02, f03, f04, f05, f06, f07, f08, f09):
        fn(c)
    assert not list(F.glob("*.svg")), "PNG only"
    print(f"[figures] {len(list(F.glob('*.png')))} PNG written to {F}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
