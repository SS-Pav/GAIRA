#!/usr/bin/env python3
"""GAIRA V7 — Phase 06 figures (PNG 200 dpi + SVG vector source). Deterministic."""
from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
from gaira.v7.io import PhaseOutputs, frozen_root            # noqa: E402
from gaira.v7.chemistry import registry as REG               # noqa: E402

OUT = PhaseOutputs("06", extra=("interactive", "manifests"))
T, A_, F, V = OUT.tables, OUT.artifacts, OUT.figures, OUT.validation
FROZEN = frozen_root()
INK, MUTED, LINE = "#1a1a1a", "#6b7280", "#9ca3af"
BLUE, GREEN, AMBER, RED, GREY = "#2563eb", "#15803d", "#b45309", "#b91c1c", "#4b5563"
PURPLE, TEAL = "#7c3aed", "#0f766e"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
                     "figure.facecolor": "white", "savefig.facecolor": "white",
                     "svg.fonttype": "path", "savefig.bbox": "tight",
                     "savefig.pad_inches": 0.18})
CL = list(REG.CLASS_ORDER)
SHORT = {c: c.replace("_", " ").replace("carboxylic acid metabolite", "carboxylic acid")
          .replace("phospholipid sphingolipid", "phospholipid").replace("mono oligosaccharide",
          "mono/oligosacch.").replace("sulfur thiol cofactor", "sulfur/thiol")
          .replace("chromophore pigment", "chromophore").replace("nucleic acid polymer",
          "nucleic polymer").replace("small nitrogenous", "small N") for c in CL}


def save(fig, name):
    F.mkdir(parents=True, exist_ok=True)
    fig.savefig(F / f"{name}.png", dpi=200)
    fig.savefig(F / f"{name}.svg", format="svg")
    plt.close(fig)
    print(f"  {name}")


def box(ax, x, y, w, h, text, fc="#eef2ff", ec=BLUE, fs=8.0, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008", fc=fc, ec=ec, lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
            weight=weight, linespacing=1.35)


def arrow(ax, p0, p1, col=LINE, lw=1.2, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=11, color=col,
                                 lw=lw, linestyle=ls, shrinkA=2, shrinkB=2))


class Ctx:
    def __init__(self):
        self.s = json.loads((A_ / "phase06_summary_v1.json").read_text())
        self.state = json.loads((OUT.root / "PHASE_STATE.json").read_text())
        self.reg = json.loads((A_ / "chemistry_class_registry_v1.json").read_text())["classes"]
        z = np.load(A_ / "chemistry_evidence_predictions_v1.npz", allow_pickle=True)
        self.E, self.P, self.E05 = z["E"], z["P"], z["E_phase05"]
        self.y = np.array([str(v) for v in z["y"]])
        self.cls = np.array([str(v) for v in z["cls"]])
        self.folds, self.A = z["folds"], z["A_csm"]
        self.src = np.array([str(v) for v in z["source"]])
        self.ev = z["explained_variance"]
        self.spec_id = np.array([str(v) for v in z["spectrum_id"]])
        b = np.load(FROZEN / "phase01/artifacts/balanced_references_v1.npz", allow_pickle=True)
        self.X, self.grid = np.asarray(b["X"], float), np.asarray(b["grid"], float)
        self.recs = json.loads(
            (FROZEN / "phase02/artifacts/csm_registry_v1.json").read_text())["csms"]
        self.bench = pd.read_csv(T / "evidence_model_benchmark_v1.csv")
        self.nested = pd.read_csv(T / "nested_cv_folds_v1.csv")
        self.pc = pd.read_csv(T / "chemistry_per_class_v1.csv")
        self.cm = pd.read_csv(T / "chemistry_confusion_matrix_v1.csv")
        self.cal = pd.read_csv(T / "calibration_summary_v1.csv")
        self.rel = pd.read_csv(T / "reliability_v1.csv")
        self.selacc = pd.read_csv(T / "selective_accuracy_v1.csv")
        self.norm = pd.read_csv(T / "normalisation_comparison_v1.csv")
        self.cmp = pd.read_csv(T / "layer_comparison_v1.csv")
        self.sem = pd.read_csv(T / "semantic_comparator_v1.csv")
        self.rob = pd.read_csv(T / "robustness_v1.csv")
        self.robs = pd.read_csv(T / "robustness_summary_v1.csv")
        self.nov = pd.read_csv(T / "holdout_chemistry_novelty_v1.csv")
        self.fail = pd.read_csv(T / "failure_analysis_v1.csv")
        self.gates = pd.read_csv(T / "phase06_gates_v1.csv")
        self.prov = json.loads((A_ / "chemistry_evidence_provenance_v1.json").read_text())
        self.pred = np.array([CL[int(i)] for i in np.argmax(self.E, axis=1)])


# ── 1 ────────────────────────────────────────────────────────────────────────
def f01_pipeline(c):
    fig, ax = plt.subplots(figsize=(11.4, 4.4))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    steps = [(0.005, "Raman spectrum\n676 bins", "#f8fafc", GREY),
             (0.145, "canonical\npreprocessing", "#f8fafc", GREY),
             (0.285, "NNLS onto 49\nfrozen CSMs", "#ecfdf5", GREEN),
             (0.425, "CSM activation\na(x) ∈ ℝ₊⁴⁹", "#ecfdf5", GREEN),
             (0.565, "Chemistry Evidence\ne(x) ∈ ℝ₊¹⁶", "#f5f3ff", PURPLE),
             (0.705, "calibrated\nprobabilities", "#f5f3ff", PURPLE),
             (0.845, "16-axis radar\n+ provenance", "#fffbeb", AMBER)]
    for i, (x, t, fc, ec) in enumerate(steps):
        w = 0.13
        box(ax, x, 0.55, w, 0.20, t, fc, ec, 8.0, "bold" if i in (3, 4) else "normal")
        if i:
            arrow(ax, (steps[i - 1][0] + w, 0.65), (x, 0.65))
    ax.text(0.49, 0.50, "PHASE 06 — this phase", ha="center", fontsize=8.6, color=PURPLE,
            weight="bold")
    ax.plot([0.565, 0.975], [0.53, 0.53], color=PURPLE, lw=1.4)
    ax.plot([0.285, 0.555], [0.53, 0.53], color=GREEN, lw=1.4)
    ax.text(0.42, 0.50, "frozen, Phases 01–02", ha="center", fontsize=8.6, color=GREEN)
    box(ax, 0.005, 0.13, 0.47, 0.28,
        "NOT IMPLEMENTED HERE\n\nBSV2 — biochemical programmes → Phase 07\n"
        "hierarchical molecular retrieval → Phase 08", "#f1f5f9", "#94a3b8", 8.0)
    s = c.s
    box(ax, 0.52, 0.13, 0.455, 0.28,
        f"PHASE 06 RESULT\n\nfine-class top-1 {s['performance']['top1']['value']:.3f}  ·  "
        f"top-3 {s['performance']['top3']['value']:.3f}\n"
        f"macro-F1 {s['performance']['macro_f1']['value']:.3f}  ·  "
        f"balanced {s['performance']['balanced_accuracy']['value']:.3f}  ·  "
        f"ECE {s['calibration']['ece']:.3f}", "#ecfdf5", GREEN, 8.0)
    ax.set_title("Figure 1 · The GAIRA V7 pipeline through Phase 06", loc="left",
                 fontsize=11.5, weight="bold", color=INK)
    save(fig, "F01_pipeline")


# ── 2 ────────────────────────────────────────────────────────────────────────
def f02_math(c):
    fig, ax = plt.subplots(figsize=(11.0, 5.6))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.02, 0.93, "What Phase 05 actually computed  (traced to source, not to prose)",
            fontsize=9.6, weight="bold", color=INK)
    box(ax, 0.02, 0.62, 0.45, 0.25,
        "engine.py:86-89  ·  run_phase05.py::topk_class\n\n"
        r"$e_c(x) = \max_{i \in c}\ \cos(a(x),\ r_i)$" "\n\n"
        "one nearest molecule decides the class\nno class-size correction · no calibration\n"
        "no probability · not a vector", "#fef2f2", RED, 8.4)
    ax.text(0.02, 0.555, "reproduced bit-for-bit: top-1 0.845310, top-3 0.970667, "
            "macro-F1 0.806783, balanced 0.796612", fontsize=8.0, color=MUTED)
    ax.text(0.53, 0.93, "What Phase 06 computes", fontsize=9.6, weight="bold", color=INK)
    sel = c.s["selected_model"]
    box(ax, 0.53, 0.62, 0.45, 0.25,
        f"selected by nested CV: {sel['candidate']}\n\n"
        r"$e_c(x) = w_c \cdot \max_{i \in c}\cos(a(x), r_i) \cdot b_{\beta(c)}(x)^{\lambda}$" "\n\n"
        r"$w_c$ inverse-frequency size correction" "\n"
        r"$b_{\beta(c)}$ SOFT broad-superclass routing, $\lambda$=0.5"
        "\nevidence for all 16 classes, then calibrated", "#f5f3ff", PURPLE, 8.4)
    ax.text(0.02, 0.44, "Why the difference matters", fontsize=9.6, weight="bold", color=INK)
    pts = [
        "Phase 05 returned a LABEL. Phase 06 returns a 16-dimensional continuous vector: every "
        "class carries evidence, ambiguity is visible, and the second-best chemistry is a "
        "number rather than an absence.",
        "The size correction stops a 80-spectrum class outscoring a 7-spectrum class by having "
        "more chances to contain a near neighbour.",
        "Broad routing is SOFT — multiplicative and strictly positive — so a fine class stays "
        "reachable when its superclass is not top-1. A hard filter would make a broad error "
        "unrecoverable.",
        "The evidence is exactly decomposable: e_c is an explicit function of named "
        "molecule similarities, each an inner product of named CSM activations.",
    ]
    yy = 0.375
    for p in pts:
        for k, w in enumerate(textwrap.wrap(p, 118)):
            ax.text(0.035 if k else 0.02, yy, ("• " if k == 0 else "") + w, fontsize=8.2,
                    color=INK)
            yy -= 0.042
        yy -= 0.012
    ax.set_title("Figure 2 · The exact chemistry-class inference mathematics", loc="left",
                 fontsize=11.5, weight="bold", color=INK)
    save(fig, "F02_inference_mathematics")


# ── 3 ────────────────────────────────────────────────────────────────────────
def f03_corpus(c):
    fig, axs = plt.subplots(1, 3, figsize=(12.4, 5.0),
                            gridspec_kw={"width_ratios": [1.5, 1.0, 1.0]})
    r = pd.DataFrame(c.reg).sort_values("n_spectra")
    ax = axs[0]
    ax.barh(range(len(r)), r.n_spectra, color=BLUE, alpha=0.85, label="spectra")
    ax.barh(range(len(r)), r.n_molecules, color=GREEN, alpha=0.9, height=0.45,
            label="molecules")
    ax.set_yticks(range(len(r)))
    ax.set_yticklabels([f"[{int(i)}] {SHORT[cc]}" for i, cc in zip(r.class_index, r.class_id)],
                       fontsize=7.4)
    for k, (sp, mo) in enumerate(zip(r.n_spectra, r.n_molecules)):
        ax.text(sp + 1.5, k, f"{int(sp)}s / {int(mo)}m", va="center", fontsize=6.6, color=MUTED)
    ax.set_xlabel("count"); ax.legend(frameon=False, fontsize=7.6, loc="lower right")
    ax.set_title("a · all 16 frozen chemistry classes", fontsize=9, loc="left")
    ax.set_xlim(0, r.n_spectra.max() * 1.28)
    ax = axs[1]
    srcs = sorted(set(c.src))
    Mx = np.array([[int(((c.cls == cc) & (c.src == s)).sum()) for s in srcs] for cc in r.class_id])
    left = np.zeros(len(r))
    for j, s in enumerate(srcs):
        ax.barh(range(len(r)), Mx[:, j], left=left, label=s, alpha=0.9)
        left += Mx[:, j]
    ax.set_yticks(range(len(r))); ax.set_yticklabels([])
    ax.legend(frameon=False, fontsize=6.6, loc="lower right")
    ax.set_xlabel("spectra"); ax.set_title("b · source distribution", fontsize=9, loc="left")
    ax = axs[2]
    ax.barh(range(len(r)), r.imbalance_vs_uniform, color=AMBER, alpha=0.9)
    ax.axvline(1.0, color=INK, ls="--", lw=1.0)
    ax.set_yticks(range(len(r))); ax.set_yticklabels([])
    ax.set_xlabel("spectra ÷ uniform share")
    ax.set_title(f"c · imbalance ({int(r.n_spectra.max())}:{int(r.n_spectra.min())} = "
                 f"{r.n_spectra.max()/r.n_spectra.min():.1f}×)", fontsize=9, loc="left")
    fig.suptitle("Figure 3 · The frozen 16-class corpus — 375 spectra, 154 molecules",
                 x=0.04, ha="left", fontsize=11.5, weight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, "F03_corpus_composition")


# ── 4 ────────────────────────────────────────────────────────────────────────
def f04_benchmark(c):
    b = c.bench[c.bench.usable].copy().sort_values("macro_f1")
    fig, ax = plt.subplots(figsize=(9.6, 8.2))
    fam_col = {"A_similarity_evidence": BLUE, "B_class_prototype": GREEN,
               "C_probabilistic": PURPLE, "D_hierarchical": AMBER}
    cols = [fam_col.get(f, GREY) for f in b.family]
    ax.barh(range(len(b)), b.macro_f1, color=cols, alpha=0.9)
    ax.set_yticks(range(len(b))); ax.set_yticklabels(b.candidate, fontsize=6.4)
    for k, (m, t) in enumerate(zip(b.macro_f1, b.top1)):
        ax.text(m + 0.004, k, f"{m:.3f}  (top-1 {t:.3f})", va="center", fontsize=6.2,
                color=MUTED)
    sel = c.s["selected_model"]["candidate"]
    if sel in list(b.candidate):
        k = list(b.candidate).index(sel)
        ax.barh([k], [b.macro_f1.iloc[k]], color=RED, alpha=0.35)
        ax.text(0.02, k, "SELECTED", va="center", fontsize=6.6, color=RED, weight="bold")
    ax.set_xlim(0, b.macro_f1.max() * 1.25); ax.set_xlabel("macro-F1 (5-fold grouped CV)")
    ax.legend(handles=[Line2D([], [], marker="s", ls="", mfc=v, mec=v, ms=7, label=k)
                       for k, v in fam_col.items()], frameon=False, fontsize=7.2,
              loc="lower right")
    ax.set_title(f"Figure 4 · All {len(b)} candidate Chemistry Evidence models\n"
                 "flat grouped CV shown; selection was made by NESTED CV (Figure 5)",
                 loc="left", fontsize=11, weight="bold", color=INK)
    save(fig, "F04_model_benchmark")


# ── 5 ────────────────────────────────────────────────────────────────────────
def f05_nested(c):
    fig, axs = plt.subplots(1, 3, figsize=(12.0, 4.0),
                            gridspec_kw={"width_ratios": [1.2, 1.0, 1.0]})
    n = c.nested
    ax = axs[0]
    ax.bar(n.fold - 0.19, n.inner_score, 0.36, color=BLUE, alpha=0.85, label="inner (selection)")
    ax.bar(n.fold + 0.19, n.outer_macro_f1, 0.36, color=GREEN, alpha=0.9, label="outer (report)")
    for _, r in n.iterrows():
        ax.text(r.fold, 0.02, r.selected, rotation=90, fontsize=6.0, ha="center", color=INK)
    ax.set_xticks(n.fold); ax.set_xlabel("outer fold"); ax.set_ylabel("macro-F1")
    ax.legend(frameon=False, fontsize=7.4); ax.set_ylim(0, 1)
    ax.set_title("a · per-fold selection and outcome", fontsize=9, loc="left")
    ax = axs[1]
    best_flat = c.bench[c.bench.usable].macro_f1.max()
    vals = [best_flat, c.s["performance"]["macro_f1"]["value"]]
    ax.bar([0, 1], vals, color=[AMBER, GREEN], alpha=0.9, width=0.55)
    lo, hi = c.s["performance"]["macro_f1"]["ci95"]
    ax.errorbar([1], [vals[1]], yerr=[[vals[1] - lo], [hi - vals[1]]], color=INK, capsize=5,
                lw=1.2, fmt="none")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["best flat candidate\n(selected on the\nfolds it is scored on)",
                        "nested CV\n(honest)"], fontsize=7.4)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=8.4, weight="bold")
    ax.set_ylim(0, 1); ax.set_ylabel("macro-F1")
    ax.set_title(f"b · selection bias = {best_flat - vals[1]:+.3f}", fontsize=9, loc="left")
    ax = axs[2]
    ax.axis("off")
    ax.text(0.0, 0.95, "outer-fold metrics, 95% CI\n(molecule-level bootstrap, 2000 draws)",
            fontsize=8.4, weight="bold", va="top")
    yy = 0.74
    for k in ("top1", "top3", "macro_f1", "balanced_accuracy", "mrr"):
        v = c.s["performance"][k]
        ax.text(0.0, yy, k.replace("_", " "), fontsize=8.0)
        ax.text(0.62, yy, f"{v['value']:.3f}", fontsize=8.0, weight="bold", color=GREEN)
        ax.text(0.78, yy, f"[{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]", fontsize=7.2,
                color=MUTED)
        yy -= 0.115
    ax.text(0.0, 0.10, "Three different models won across five folds.\nThe modal choice is "
            "reported as canonical; the\nspread is a real finding, not a nuisance.",
            fontsize=7.4, color=MUTED, style="italic")
    fig.suptitle("Figure 5 · Nested molecule-grouped cross-validation", x=0.04, ha="left",
                 fontsize=11.5, weight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, "F05_nested_cv")


# ── 6 ────────────────────────────────────────────────────────────────────────
def f06_confusion(c):
    M = c.cm.set_index("true_class").reindex(index=CL, columns=CL).values.astype(float)
    N = M / (M.sum(axis=1, keepdims=True) + 1e-12)
    fig, ax = plt.subplots(figsize=(8.6, 7.6))
    im = ax.imshow(N, cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(range(16)); ax.set_xticklabels([SHORT[x] for x in CL], rotation=55,
                                                 ha="right", fontsize=7)
    ax.set_yticks(range(16)); ax.set_yticklabels(
        [f"{SHORT[x]}  (n={int(M[i].sum())})" for i, x in enumerate(CL)], fontsize=7)
    adjset = {(REG.CLASS_ORDER.index(a), REG.CLASS_ORDER.index(b)) for a, b in REG.ADJACENT}
    for i in range(16):
        for j in range(16):
            if N[i, j] > 0.015:
                ax.text(j, i, f"{N[i, j]:.2f}", ha="center", va="center", fontsize=5.8,
                        color="white" if N[i, j] > 0.55 else INK)
            if i != j and ((i, j) in adjset or (j, i) in adjset):
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec=AMBER, lw=1.0))
    fig.colorbar(im, ax=ax, shrink=0.75, label="row-normalised rate")
    ax.set_xlabel("predicted chemistry class"); ax.set_ylabel("true chemistry class")
    a = c.s["soft_evidence"]["adjacency"]
    ax.set_title("Figure 6 · 16-class confusion, molecule-grouped outer folds\n"
                 f"amber cells = pre-declared chemically adjacent pairs · "
                 f"{a['adjacent_fraction']:.0%} of errors are adjacent vs "
                 f"{a['chance_adjacent']:.0%} chance ({a['lift']:.1f}× lift)",
                 loc="left", fontsize=10.5, weight="bold", color=INK)
    save(fig, "F06_confusion_matrix")


# ── 7 ────────────────────────────────────────────────────────────────────────
def f07_per_class(c):
    p = c.pc.set_index("class_id").reindex(CL).reset_index()
    fig, ax = plt.subplots(figsize=(10.2, 5.0))
    x = np.arange(16)
    for k, (col, cc, lab) in enumerate((("precision", BLUE, "precision"),
                                        ("recall", GREEN, "recall"), ("f1", PURPLE, "F1"))):
        ax.bar(x + (k - 1) * 0.27, p[col], 0.26, color=cc, alpha=0.9, label=lab)
    ax.set_xticks(x); ax.set_xticklabels([f"{SHORT[cc]}\nn={int(nn)}"
                                          for cc, nn in zip(p.class_id, p.n)],
                                         rotation=52, ha="right", fontsize=6.8)
    ax.axhline(c.s["performance"]["macro_f1"]["value"], color=RED, ls="--", lw=1.1,
               label=f"macro-F1 {c.s['performance']['macro_f1']['value']:.3f}")
    ax.set_ylim(0, 1.05); ax.set_ylabel("score"); ax.legend(frameon=False, fontsize=7.6, ncol=4)
    ax.set_title("Figure 7 · Per-class precision, recall and F1\n"
                 "the four weakest classes are the four smallest — imbalance, not chemistry",
                 loc="left", fontsize=11, weight="bold", color=INK)
    save(fig, "F07_per_class")


# ── 8 ────────────────────────────────────────────────────────────────────────
def f08_calibration(c):
    d = c.cal.sort_values("log_loss")
    fig, axs = plt.subplots(1, 3, figsize=(12.0, 4.0))
    sel = c.s["calibration"]["method"]
    for ax, col, lab, inv in ((axs[0], "log_loss", "log loss (selection metric)", False),
                              (axs[1], "ece", "expected calibration error", False),
                              (axs[2], "sharpness", "sharpness (non-degeneracy)", True)):
        dd = d.sort_values(col, ascending=not inv)
        cols = [GREEN if m == sel else GREY for m in dd.method]
        ax.barh(range(len(dd)), dd[col], color=cols, alpha=0.9)
        ax.set_yticks(range(len(dd))); ax.set_yticklabels(dd.method, fontsize=7.4)
        ax.invert_yaxis()
        for k, v in enumerate(dd[col]):
            ax.text(v + dd[col].max() * 0.02, k, f"{v:.3f}", va="center", fontsize=7,
                    color=MUTED)
        ax.set_xlim(0, dd[col].max() * 1.3); ax.set_xlabel(lab)
    axs[2].axvline(0.05, color=RED, ls="--", lw=1.1)
    axs[2].text(0.052, len(d) - 0.6, "floor", fontsize=6.8, color=RED)
    fig.suptitle("Figure 8 · Calibration benchmark — selection on log loss, subject to "
                 "pre-registered non-degeneracy floors\n"
                 "isotonic wins ECE (0.116) and loses log loss (3.604): per-class "
                 "calibration then renormalised destroys the joint likelihood",
                 x=0.045, ha="left", fontsize=10, weight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    save(fig, "F08_calibration_benchmark")


# ── 9 ────────────────────────────────────────────────────────────────────────
def f09_reliability(c):
    fig, axs = plt.subplots(1, 2, figsize=(9.4, 4.2))
    r = c.rel.dropna(subset=["empirical_accuracy"])
    ax = axs[0]
    ax.plot([0, 1], [0, 1], color=LINE, ls="--", lw=1.0, label="perfect")
    ax.plot(r.bin_center, r.empirical_accuracy, "o-", color=GREEN, lw=1.5, ms=5,
            label=f"{c.s['calibration']['method']}")
    for _, row in r.iterrows():
        ax.annotate(f"n={int(row['count'])}", (row.bin_center, row.empirical_accuracy),
                    textcoords="offset points", xytext=(4, -10), fontsize=6, color=MUTED)
    ax.set_xlabel("reported confidence"); ax.set_ylabel("empirical accuracy")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(frameon=False, fontsize=8, loc="upper left")
    cal = c.s["calibration"]
    ax.set_title(f"a · reliability · ECE {cal['ece']:.3f} · classwise ECE "
                 f"{cal['classwise_ece']:.3f}", fontsize=9, loc="left")
    ax = axs[1]
    gap = r.empirical_accuracy - r.bin_center
    ax.bar(r.bin_center, gap, width=0.085, color=[GREEN if g >= 0 else RED for g in gap],
           alpha=0.85)
    ax.axhline(0, color=INK, lw=0.9)
    ax.set_xlabel("confidence bin"); ax.set_ylabel("accuracy − confidence")
    ax.set_title("b · signed gap (positive = under-confident)", fontsize=9, loc="left")
    fig.suptitle("Figure 9 · Reliability of the calibrated Chemistry Evidence", x=0.05,
                 ha="left", fontsize=11.5, weight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    save(fig, "F09_reliability")


# ── 10 ───────────────────────────────────────────────────────────────────────
def f10_selective(c):
    fig, axs = plt.subplots(1, 2, figsize=(9.4, 4.0))
    s = c.selacc
    axs[0].plot(s.coverage, s.accuracy, "o-", color=BLUE, lw=1.5, ms=4)
    axs[0].axhline(c.s["performance"]["top1"]["value"], color=RED, ls="--", lw=1.0)
    axs[0].text(0.05, c.s["performance"]["top1"]["value"] + 0.008,
                f"full coverage {c.s['performance']['top1']['value']:.3f}", fontsize=7,
                color=RED)
    axs[0].set_xlabel("coverage (fraction answered)"); axs[0].set_ylabel("accuracy among answered")
    axs[0].set_title("a · abstention buys accuracy", fontsize=9, loc="left")
    conf = c.P.max(axis=1)
    ok = c.pred == c.cls
    bins = np.linspace(0, 1, 26)
    axs[1].hist(conf[ok], bins=bins, color=GREEN, alpha=0.75, label=f"correct (n={ok.sum()})")
    axs[1].hist(conf[~ok], bins=bins, color=RED, alpha=0.6,
                label=f"wrong (n={(~ok).sum()})")
    axs[1].set_xlabel("calibrated confidence"); axs[1].set_ylabel("spectra")
    axs[1].legend(frameon=False, fontsize=7.8)
    axs[1].set_title(f"b · discrimination {c.s['calibration']['discrimination']:.3f}",
                     fontsize=9, loc="left")
    fig.suptitle("Figure 10 · Selective accuracy versus coverage", x=0.05, ha="left",
                 fontsize=11.5, weight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    save(fig, "F10_selective_accuracy")


# ── 11 ───────────────────────────────────────────────────────────────────────
def f11_heatmap(c):
    order = np.argsort([CL.index(x) for x in c.cls])
    fig, ax = plt.subplots(figsize=(11.4, 4.6))
    im = ax.imshow(c.E[order].T, aspect="auto", cmap="magma", interpolation="nearest")
    ax.set_yticks(range(16)); ax.set_yticklabels([SHORT[x] for x in CL], fontsize=7)
    bounds, prev = [], None
    for k, i in enumerate(order):
        if c.cls[i] != prev:
            bounds.append((k, c.cls[i])); prev = c.cls[i]
    for b, _ in bounds[1:]:
        ax.axvline(b, color="white", lw=0.6, alpha=0.8)
    ax.set_xticks([b for b, _ in bounds])
    ax.set_xticklabels([SHORT[l] for _, l in bounds], rotation=60, ha="right", fontsize=6)
    ax.set_xlabel("spectra, grouped by TRUE chemistry class")
    fig.colorbar(im, ax=ax, shrink=0.8, label="chemistry evidence")
    ax.set_title("Figure 11 · The Chemistry Evidence matrix — 375 spectra × 16 classes\n"
                 "the bright diagonal is the signal; the off-diagonal texture is the "
                 "ambiguity a hard label would discard",
                 loc="left", fontsize=11, weight="bold", color=INK)
    save(fig, "F11_evidence_heatmap")


# ── 12 ───────────────────────────────────────────────────────────────────────
def f12_rank_entropy(c):
    from gaira.v7.chemistry import validation as VAL
    fig, axs = plt.subplots(1, 3, figsize=(11.6, 3.9))
    ent = VAL.entropy(c.E)
    axs[0].hist(ent, bins=30, color=PURPLE, alpha=0.85)
    axs[0].axvline(ent.mean(), color=RED, ls="--")
    axs[0].set_xlabel("normalised entropy of e(x)"); axs[0].set_ylabel("spectra")
    axs[0].set_title(f"a · mean {ent.mean():.3f} — not one-hot, not flat", fontsize=9,
                     loc="left")
    rk = VAL.rank_of_true(c.E, c.cls)
    v, n = np.unique(rk, return_counts=True)
    axs[1].bar(v, n, color=GREEN, alpha=0.9)
    axs[1].set_xlabel("rank of the TRUE class"); axs[1].set_ylabel("spectra")
    axs[1].set_xlim(0.5, 8.5)
    axs[1].set_title(f"b · rank ≤3 for {np.mean(rk <= 3):.1%} of spectra", fontsize=9,
                     loc="left")
    ax = axs[2]
    layers = {"raw 676": None, "CSM 49": None}
    ers = [("chemistry evidence 16", VAL.effective_rank(c.E), PURPLE),
           ("calibrated 16", VAL.effective_rank(c.P), BLUE),
           ("phase05 evidence 16", VAL.effective_rank(c.E05), GREY)]
    ax.barh([e[0] for e in ers], [e[1] for e in ers], color=[e[2] for e in ers], alpha=0.9)
    ax.axvline(16, color=INK, ls="--", lw=1.0)
    ax.text(15.6, -0.42, "nominal 16", fontsize=7, color=INK, ha="right")
    for k, e in enumerate(ers):
        ax.text(e[1] + 0.2, k, f"{e[1]:.2f}", va="center", fontsize=8)
    ax.set_xlim(0, 17.5); ax.set_xlabel("effective rank")
    ax.set_title("c · the 16 axes are not 16 independent directions", fontsize=9, loc="left")
    fig.suptitle("Figure 12 · Effective rank, entropy and true-class rank", x=0.04, ha="left",
                 fontsize=11.5, weight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    save(fig, "F12_rank_entropy")


# ── 13 ───────────────────────────────────────────────────────────────────────
def f13_replicate(c):
    from gaira.v7.chemistry import validation as VAL
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 4.0))
    N = c.E / (np.linalg.norm(c.E, axis=1, keepdims=True) + 1e-12)
    vals, labs = [], []
    for m in sorted(set(c.y.tolist())):
        idx = np.where(c.y == m)[0]
        if len(idx) < 2:
            continue
        C = N[idx] @ N[idx].T
        iu = np.triu_indices(len(idx), 1)
        vals.append(float(C[iu].mean())); labs.append(m)
    axs[0].hist(vals, bins=28, color=GREEN, alpha=0.85)
    axs[0].axvline(np.mean(vals), color=RED, ls="--", lw=1.2)
    axs[0].set_xlabel("mean within-molecule cosine of e(x)")
    axs[0].set_ylabel("molecules")
    axs[0].set_title(f"a · replicate consistency {np.mean(vals):.3f} over {len(vals)} molecules",
                     fontsize=9, loc="left")
    wb = c.s["soft_evidence"]
    axs[1].bar([0, 1], [wb["within_class_cosine"], wb["between_class_cosine"]],
               color=[GREEN, GREY], alpha=0.9, width=0.5)
    axs[1].set_xticks([0, 1]); axs[1].set_xticklabels(["within class", "between class"])
    for i, v in enumerate([wb["within_class_cosine"], wb["between_class_cosine"]]):
        axs[1].text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=9, weight="bold")
    axs[1].set_ylim(0, 1); axs[1].set_ylabel("mean cosine of e(x)")
    axs[1].set_title(f"b · separation {wb['separation']:.3f}", fontsize=9, loc="left")
    fig.suptitle("Figure 13 · Replicate and within-class consistency of the evidence vector",
                 x=0.04, ha="left", fontsize=11.5, weight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    save(fig, "F13_replicate_consistency")


# ── 14 ───────────────────────────────────────────────────────────────────────
def f14_robustness(c):
    kinds = list(dict.fromkeys(c.rob.perturbation))
    reps = list(dict.fromkeys(c.rob.representation))
    col = {"raw_spectrum": GREY, "csm_49": GREEN, "chemistry_evidence_16": PURPLE,
           "legacy_theme_bsv": "#f59e0b", "legacy_11_axis": BLUE}
    fig, axs = plt.subplots(3, 4, figsize=(12.6, 7.6))
    for ax, k in zip(axs.ravel(), kinds):
        d = c.rob[c.rob.perturbation == k]
        for r in reps:
            s = d[d.representation == r].sort_values("level")
            ax.plot(range(len(s)), s.class_top1, "o-", color=col.get(r, GREY), lw=1.2, ms=3,
                    label=r)
        lv = d[d.representation == reps[0]].sort_values("level").level
        ax.set_xticks(range(len(lv))); ax.set_xticklabels([f"{v:g}" for v in lv], fontsize=6)
        ax.set_ylim(0, 1); ax.tick_params(labelsize=6.5)
        ax.set_title(k.replace("_", " "), fontsize=8, loc="left")
    ax = axs.ravel()[-1]
    ax.axis("off")
    d = c.robs.set_index("representation")
    ax.text(0.0, 0.95, "top-1 retention", fontsize=8.6, weight="bold")
    for i, r in enumerate(reps):
        ax.text(0.0, 0.80 - i * 0.13,
                f"{r:22s} {d.loc[r,'clean_top1']:.3f}→{d.loc[r,'mean_perturbed_top1']:.3f}  "
                f"({d.loc[r,'top1_retention']:.3f})",
                fontsize=6.6, color=col.get(r, GREY), family="DejaVu Sans Mono")
    ax.text(0.0, 0.14, "The 16-d layer tracks the CSM layer it is\ncomputed from and clearly "
            "beats the raw\nspectrum on both accuracy and retention.",
            fontsize=7, color=INK, style="italic")
    axs.ravel()[0].legend(frameon=False, fontsize=6, loc="lower left")
    axs.ravel()[0].set_ylabel("class top-1", fontsize=7.4)
    axs.ravel()[4].set_ylabel("class top-1", fontsize=7.4)
    axs.ravel()[8].set_ylabel("class top-1", fontsize=7.4)
    fig.suptitle("Figure 14 · Raman perturbation robustness — 11 perturbations × 5 levels",
                 x=0.04, ha="left", fontsize=11.5, weight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, "F14_robustness")


# ── 15 ───────────────────────────────────────────────────────────────────────
def f15_novelty(c):
    n = c.nov.sort_values("joint_auroc")
    fig, axs = plt.subplots(1, 3, figsize=(12.2, 4.2),
                            gridspec_kw={"width_ratios": [1.1, 1.1, 1.0]})
    ax = axs[0]
    cols = [GREEN if v >= 0.80 else (AMBER if v >= 0.65 else RED) for v in n.joint_auroc]
    ax.barh(range(len(n)), n.joint_auroc, color=cols, alpha=0.9)
    ax.axvline(0.5, color=INK, ls="--", lw=1.0)
    ax.set_yticks(range(len(n)))
    ax.set_yticklabels([f"{SHORT[x]}\n(n={int(m)})" for x, m in
                        zip(n.held_class, n.n_novel_spectra)], fontsize=7)
    for k, v in enumerate(n.joint_auroc):
        ax.text(v + 0.015, k, f"{v:.3f}", va="center", fontsize=7.4)
    ax.set_xlim(0, 1.12); ax.set_xlabel("novelty AUROC")
    ax.set_title("a · can the engine tell the class is missing?", fontsize=9, loc="left")
    ax = axs[1]
    x = np.arange(len(n))
    ax.bar(x - 0.2, n.mean_max_evidence_in_domain, 0.38, color=GREEN, alpha=0.85,
           label="in-domain")
    ax.bar(x + 0.2, n.mean_max_evidence_novel, 0.38, color=RED, alpha=0.85, label="withheld")
    ax.set_xticks(x); ax.set_xticklabels([SHORT[v] for v in n.held_class], rotation=50,
                                         ha="right", fontsize=6.8)
    ax.set_ylabel("mean max evidence"); ax.legend(frameon=False, fontsize=7.4)
    ax.set_title("b · evidence collapses — except for one class", fontsize=9, loc="left")
    ax = axs[2]
    ax.axis("off")
    ax.text(0.0, 0.98, "THE FAILURE, STATED PLAINLY", fontsize=8.8, weight="bold", color=RED)
    txt = ("acylglycerol scores AUROC 0.489 — no better than chance — and abstains on 0% of its "
           "spectra. Its mean max evidence barely falls (0.614 vs 0.665 in-domain).\n\n"
           "The reason is chemical, not computational: with acylglycerols withheld, fatty acids "
           "remain, and a triacylglycerol's Raman spectrum is dominated by the same acyl-chain "
           "motifs. The engine is not wrong that the chemistry is present — it is wrong that "
           "the chemistry is REPRESENTED.\n\n"
           "Novelty detection works when the withheld chemistry has no close represented "
           "neighbour, and fails when it does. That is the honest boundary of this capability.")
    yy = 0.88
    for line in txt.split("\n"):
        for w in textwrap.wrap(line, 52) or [""]:
            ax.text(0.0, yy, w, fontsize=7.2, color=INK)
            yy -= 0.055
    fig.suptitle("Figure 15 · Held-out chemistry novelty — an entire class withheld from the "
                 "atlas (Raman only)", x=0.035, ha="left", fontsize=11.5, weight="bold",
                 color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    save(fig, "F15_holdout_novelty")


def main():
    c = Ctx()
    print("[figures]")
    for fn in (f01_pipeline, f02_math, f03_corpus, f04_benchmark, f05_nested, f06_confusion,
               f07_per_class, f08_calibration, f09_reliability, f10_selective, f11_heatmap,
               f12_rank_entropy, f13_replicate, f14_robustness, f15_novelty):
        fn(c)
    import make_figures_b
    make_figures_b.main(c, save, box, arrow, SHORT, CL)
    print(f"[figures] {len(list(F.glob('*.png')))} PNG + "
          f"{len(list(F.glob('*.svg')))} SVG written to {F}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
