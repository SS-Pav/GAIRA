#!/usr/bin/env python3
"""GAIRA V7 — Phase 05 figures (PNG, 200 dpi). Deterministic; seeds fixed."""
from __future__ import annotations

import json
import sys
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
from gaira.v7.io import PhaseOutputs, frozen_root          # noqa: E402
from gaira.v7.inference import evidence as EV              # noqa: E402

OUT = PhaseOutputs("05")
T, A, F = OUT.tables, OUT.artifacts, OUT.figures
FROZEN = frozen_root()
INK, MUTED, LINE = "#1a1a1a", "#6b7280", "#9ca3af"
BLUE, GREEN, AMBER, RED, GREY = "#2563eb", "#15803d", "#b45309", "#b91c1c", "#4b5563"
PURPLE, TEAL = "#7c3aed", "#0f766e"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
                     "figure.facecolor": "white", "savefig.facecolor": "white",
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.18})
REP_ORDER = ["raw", "lsm", "csm", "evidence"]
REP_COL = {"raw": GREY, "lsm": BLUE, "csm": GREEN, "evidence": PURPLE}


def save(fig, name):
    F.mkdir(parents=True, exist_ok=True)
    fig.savefig(F / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"  {name}")


def box(ax, x, y, w, h, text, fc="#eef2ff", ec=BLUE, fs=8.0, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                fc=fc, ec=ec, lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=INK, weight=weight, linespacing=1.35)


def arrow(ax, p0, p1, col=LINE, style="-|>", lw=1.2):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=11,
                                 color=col, lw=lw, shrinkA=2, shrinkB=2))


class Ctx:
    def __init__(self):
        self.state = json.loads((OUT.root / "PHASE_STATE.json").read_text())
        self.summ = json.loads((A / "phase05_summary_v1.json").read_text())
        z = np.load(A / "csm_activations_v1.npz", allow_pickle=True)
        self.Acsm = z["A"]
        self.y = np.array([str(s) for s in z["y"]])
        self.cls = np.array([str(s) for s in z["cls"]])
        self.folds, self.grid = z["folds"], z["grid"]
        e = np.load(A / "evidence_profiles_v1.npz", allow_pickle=True)
        self.E, self.Econf = e["magnitude"], e["confidence"]
        self.Ecov, self.Esup = e["coverage"], e["support"]
        self.axes = [str(s) for s in e["axes"]]
        m = np.load(A / "evidence_axis_map_v1.npz", allow_pickle=True)
        self.M, self.unassigned, self.spec = m["M"], m["unassigned"], m["specificity"]
        o = np.load(A / "openset_scores_v1.npz", allow_pickle=True)
        self.j_in, self.j_out = o["joint_in"], o["joint_out"]
        self.neg_kind = np.array([str(s) for s in o["negative_kind"]])
        self.Xneg = o["Xneg"]
        b = np.load(FROZEN / "phase01/artifacts/balanced_references_v1.npz", allow_pickle=True)
        self.X = np.asarray(b["X"], float)
        self.CSM = np.load(FROZEN / "phase02/artifacts/csm_dictionary_v1.npz")["CSM"]
        self.recs = json.loads(
            (FROZEN / "phase02/artifacts/csm_registry_v1.json").read_text())["csms"]
        self.diag = pd.read_csv(T / "csm_projection_diagnostics_v1.csv")
        self.met = pd.read_csv(T / "similarity_metric_benchmark_v1.csv")
        self.cal = pd.read_csv(T / "calibration_summary_v1.csv")
        self.rel = pd.read_csv(T / "reliability_splitA_v1.csv")
        self.pa = pd.read_csv(T / "splitA_predictions_v1.csv")
        self.pb = pd.read_csv(T / "splitB_predictions_v1.csv")
        self.curve = pd.read_csv(T / "splitA_topk_curve_v1.csv")
        self.cm = pd.read_csv(T / "class_confusion_matrix_v1.csv")
        self.prf = pd.read_csv(T / "class_precision_recall_v1.csv")
        self.os_ch = pd.read_csv(T / "openset_channel_auroc_v1.csv")
        self.os_kind = pd.read_csv(T / "openset_by_negative_kind_v1.csv")
        self.roc = pd.read_csv(T / "openset_roc_v1.csv")
        self.axsum = pd.read_csv(T / "evidence_axis_summary_v1.csv")
        self.axval = pd.read_csv(T / "evidence_axis_validation_v1.csv")
        self.rob = pd.read_csv(T / "noise_robustness_v1.csv")
        self.robs = pd.read_csv(T / "robustness_summary_v1.csv")
        self.gates = pd.read_csv(T / "phase05_gates_v1.csv")
        self.reports = json.loads((A / "representative_reports_v1.json").read_text())
        self.ground = json.loads((A / "evidence_axis_grounding_v1.json").read_text())


# ── F01 ──────────────────────────────────────────────────────────────────────
def f01_pipeline(c):
    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    box(ax, 0.02, 0.60, 0.15, 0.13, "unknown\nRaman spectrum", "#f8fafc", GREY, 8.5, "bold")
    box(ax, 0.20, 0.60, 0.16, 0.13,
        "canonical preprocessing\n450–1800 cm$^{-1}$ · 2.0 step\nasLS → SG → L2", "#f8fafc", GREY)
    box(ax, 0.39, 0.60, 0.17, 0.13,
        "non-negative projection\nonto 49 frozen CSMs\n(NNLS, nothing fitted)", "#ecfdf5", GREEN,
        8.0, "bold")
    box(ax, 0.59, 0.60, 0.14, 0.13, "49-d CSM\nactivation vector", "#ecfdf5", GREEN, 8.5, "bold")
    arrow(ax, (0.17, 0.665), (0.20, 0.665)); arrow(ax, (0.36, 0.665), (0.39, 0.665))
    arrow(ax, (0.56, 0.665), (0.59, 0.665))
    branches = [("1. analyte retrieval", "154 reference vectors\ncosine · calibrated", "#eff6ff",
                 BLUE),
                ("2. chemistry class", "grouped CV\nmacro-F1 %.2f" % c.summ["split_b"]["macro_f1"],
                 "#eff6ff", BLUE),
                ("3. evidence profile", "11 declared axes\nno factorisation", "#f5f3ff", PURPLE),
                ("4. provenance", "axis → CSM → LSM\n→ molecule → spectra", "#f5f3ff", PURPLE),
                ("5. uncertainty", "residual · margin\nentropy · rejection", "#fffbeb", AMBER)]
    for i, (t, sub, fc, ec) in enumerate(branches):
        x = 0.03 + i * 0.192
        box(ax, x, 0.20, 0.17, 0.17, f"{t}\n\n{sub}", fc, ec, 8.0)
        arrow(ax, (0.66, 0.60), (x + 0.085, 0.37), LINE, "-|>", 0.9)
    box(ax, 0.78, 0.60, 0.20, 0.13,
        "open-set rejection\nrejects %.0f%% of synthetic\nnegatives @95%% acceptance"
        % (100 * c.summ["openset"]["operating_point"]["ood_reject"]), "#fffbeb", AMBER, 8.0)
    arrow(ax, (0.73, 0.665), (0.78, 0.665), AMBER)
    ax.text(0.5, 0.055, "Geometry is visualisation only — no inference path passes through it. "
            "Everything above the branches is frozen; the engine fits nothing at inference.",
            ha="center", fontsize=8.2, color=MUTED, style="italic")
    ax.set_title("Figure 1 · The canonical GAIRA inference pipeline", loc="left",
                 fontsize=11, weight="bold", color=INK)
    save(fig, "F01_canonical_pipeline")


# ── F02 ──────────────────────────────────────────────────────────────────────
def f02_projection(c):
    fig = plt.figure(figsize=(9.4, 5.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1.0], hspace=0.42, wspace=0.30)
    i = int(np.argsort(-c.Acsm.sum(axis=1))[3])
    rec = c.Acsm[i] @ c.CSM
    ax = fig.add_subplot(gs[0, :2])
    ax.plot(c.grid, c.X[i], color=INK, lw=1.0, label=f"observed · {c.y[i]}")
    ax.plot(c.grid, rec, color=GREEN, lw=1.0, label="CSM reconstruction")
    ax.fill_between(c.grid, c.X[i] - rec, 0, color=RED, alpha=0.25, lw=0, label="residual")
    ax.set_xlabel("Raman shift (cm$^{-1}$)"); ax.set_ylabel("intensity (L2)")
    ax.legend(frameon=False, fontsize=7.5)
    ax.set_title("a · one spectrum, explained by frozen motifs", fontsize=9, loc="left")
    ax = fig.add_subplot(gs[0, 2])
    act = c.Acsm[i]
    o = np.argsort(-act)[:8][::-1]
    ax.barh(range(len(o)), act[o], color=GREEN, alpha=0.85)
    ax.set_yticks(range(len(o)))
    ax.set_yticklabels([c.recs[j]["csm_id"] for j in o], fontsize=7)
    ax.set_xlabel("activation")
    ax.set_title("b · active CSMs", fontsize=9, loc="left")
    for ax, col, lab, cc in ((fig.add_subplot(gs[1, 0]), "explained_variance",
                              "explained variance", GREEN),
                             (fig.add_subplot(gs[1, 1]), "n_active_csms",
                              "active CSMs (of 49)", BLUE),
                             (fig.add_subplot(gs[1, 2]), "component_sparsity",
                              "Hoyer sparsity", PURPLE)):
        ax.hist(c.diag[col], bins=26, color=cc, alpha=0.8)
        ax.axvline(c.diag[col].mean(), color=RED, lw=1.1, ls="--")
        ax.set_xlabel(lab); ax.set_ylabel("spectra")
        ax.set_title(f"mean {c.diag[col].mean():.2f}", fontsize=8, loc="left", color=MUTED)
    fig.suptitle("Figure 2 · Direct projection onto the frozen CSM dictionary", x=0.09,
                 ha="left", fontsize=11, weight="bold", color=INK)
    save(fig, "F02_csm_projection")


# ── F03 ──────────────────────────────────────────────────────────────────────
def f03_confusion(c):
    labs = [x for x in c.cm.true_class.tolist()]
    Mx = c.cm.drop(columns=["true_class"]).values.astype(float)
    keep = Mx.sum(axis=1) > 0
    Mx, labs = Mx[keep], [l for l, k in zip(labs, keep) if k]
    cols = [x for x in c.cm.columns[1:]]
    idx = [cols.index(l) for l in labs]
    Mx = Mx[:, idx]
    N = Mx / (Mx.sum(axis=1, keepdims=True) + 1e-12)
    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    im = ax.imshow(N, cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, rotation=55, ha="right", fontsize=7)
    ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs, fontsize=7)
    for i in range(len(labs)):
        for j in range(len(labs)):
            if N[i, j] > 0.02:
                ax.text(j, i, f"{N[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if N[i, j] > 0.55 else INK)
    fig.colorbar(im, ax=ax, shrink=0.75, label="row-normalised rate")
    ax.set_xlabel("predicted chemistry class"); ax.set_ylabel("true chemistry class")
    ax.set_title("Figure 3 · Chemistry-class confusion, Split B (molecule unseen)\n"
                 f"top-1 {c.summ['split_b']['class_top1']:.3f} · macro-F1 "
                 f"{c.summ['split_b']['macro_f1']:.3f} · balanced accuracy "
                 f"{c.summ['split_b']['balanced_accuracy']:.3f}",
                 loc="left", fontsize=10.5, weight="bold", color=INK)
    save(fig, "F03_class_confusion_matrix")


# ── F04 ──────────────────────────────────────────────────────────────────────
def f04_calibration(c):
    fig, axs = plt.subplots(1, 2, figsize=(9.4, 4.0))
    for ax, split, title in ((axs[0], "splitA_molecule", "Split A · molecule identity"),
                             (axs[1], "splitB_class", "Split B · chemistry class")):
        d = c.cal[c.cal.split == split].sort_values("ece")
        cols = [GREEN if m == c.summ["split_a"]["calibration"] else GREY for m in d.method]
        ax.barh(range(len(d)), d.ece, color=cols, alpha=0.9)
        ax.set_yticks(range(len(d))); ax.set_yticklabels(d.method, fontsize=8)
        ax.invert_yaxis()
        for k, (e, b) in enumerate(zip(d.ece, d.brier)):
            ax.text(e + 0.012, k, f"ECE {e:.3f} · Brier {b:.3f}", va="center", fontsize=7,
                    color=MUTED)
        ax.set_xlim(0, max(d.ece) * 1.55)
        ax.set_xlabel("expected calibration error")
        ax.set_title(title, fontsize=9, loc="left")
    fig.suptitle("Figure 4 · Calibration method benchmark (grouped CV, held-out folds only)",
                 x=0.06, ha="left", fontsize=11, weight="bold", color=INK)
    save(fig, "F04_calibration_benchmark")


# ── F05 ──────────────────────────────────────────────────────────────────────
def f05_reliability(c):
    fig, axs = plt.subplots(1, 2, figsize=(9.0, 4.2),
                            gridspec_kw={"width_ratios": [1.15, 1.0]})
    ax = axs[0]
    r = c.rel.dropna(subset=["empirical_accuracy"])
    ax.plot([0, 1], [0, 1], color=LINE, ls="--", lw=1.0, label="perfect calibration")
    ax.plot(r.bin_center, r.empirical_accuracy, "o-", color=GREEN, lw=1.4, ms=5,
            label=f"{c.summ['split_a']['calibration']} (selected)")
    for _, row in r.iterrows():
        ax.annotate(f"n={int(row['count'])}", (row.bin_center, row.empirical_accuracy),
                    textcoords="offset points", xytext=(4, -9), fontsize=6, color=MUTED)
    ax.set_xlabel("reported confidence"); ax.set_ylabel("empirical accuracy")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title(f"a · reliability, Split A · ECE {c.summ['split_a']['ece']:.3f}",
                 fontsize=9, loc="left")
    ax = axs[1]
    w = c.rel.dropna(subset=["empirical_accuracy"])
    gap = w.empirical_accuracy - w.bin_center
    ax.bar(w.bin_center, gap, width=0.085, color=[GREEN if g >= 0 else RED for g in gap],
           alpha=0.85)
    ax.axhline(0, color=INK, lw=0.9)
    ax.set_xlabel("confidence bin"); ax.set_ylabel("accuracy − confidence")
    ax.set_title("b · signed calibration gap (positive = under-confident)", fontsize=9,
                 loc="left")
    fig.suptitle("Figure 5 · Reliability diagram", x=0.06, ha="left", fontsize=11,
                 weight="bold", color=INK)
    save(fig, "F05_reliability_diagram")


# ── F06 ──────────────────────────────────────────────────────────────────────
def f06_topk(c):
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 4.0))
    ax = axs[0]
    ax.plot(c.curve.k, c.curve.topk, "o-", color=BLUE, lw=1.6, ms=5)
    for k in (1, 3, 5):
        v = float(c.curve[c.curve.k == k].topk.iloc[0])
        ax.annotate(f"top-{k} = {v:.3f}", (k, v), textcoords="offset points", xytext=(6, -12),
                    fontsize=8, color=INK)
    ax.axhline(1 / 154, color=LINE, ls=":", lw=1.0)
    ax.text(9.6, 1 / 154 + 0.015, "chance (1/154)", ha="right", fontsize=7, color=MUTED)
    ax.set_xlabel("k"); ax.set_ylabel("molecule top-k retrieval")
    ax.set_ylim(0, 1); ax.set_title("a · Split A, the molecule is in the bank", fontsize=9,
                                    loc="left")
    ax = axs[1]
    ax.axis("off")
    ax.text(0.0, 0.90, "Split B · the molecule is absent from the bank", fontsize=9,
            weight="bold", color=INK)
    ax.text(0.0, 0.74, "Molecule top-k is **undefined**, not zero: the correct answer is\n"
            "not among the candidates. Reporting 0.000 would be a category error.",
            fontsize=8.4, color=INK, linespacing=1.6)
    b = c.summ["split_b"]
    for i, (lab, v) in enumerate([("class top-1", b["class_top1"]),
                                  ("class top-3", b["class_top3"]),
                                  ("macro F1", b["macro_f1"]),
                                  ("balanced accuracy", b["balanced_accuracy"])]):
        ax.barh(0.45 - i * 0.11, v, height=0.075, color=GREEN, alpha=0.85, left=0.32)
        ax.text(0.30, 0.485 - i * 0.11, lab, ha="right", fontsize=8.2, color=INK)
        ax.text(0.34 + v * 0.62, 0.485 - i * 0.11, f"{v:.3f}", fontsize=8, color=MUTED)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.suptitle("Figure 6 · Retrieval under the two evaluation splits", x=0.06, ha="left",
                 fontsize=11, weight="bold", color=INK)
    save(fig, "F06_topk_retrieval")


# ── F07 ──────────────────────────────────────────────────────────────────────
def f07_confidence(c):
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 4.0))
    ax = axs[0]
    ok = c.pa[c.pa.correct_top1]
    no = c.pa[~c.pa.correct_top1]
    bins = np.linspace(0, 1, 26)
    ax.hist(ok.confidence, bins=bins, color=GREEN, alpha=0.72, label=f"correct (n={len(ok)})")
    ax.hist(no.confidence, bins=bins, color=RED, alpha=0.62, label=f"wrong (n={len(no)})")
    ax.set_xlabel("calibrated confidence"); ax.set_ylabel("spectra")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("a · Split A, molecule identity", fontsize=9, loc="left")
    ax = axs[1]
    q = np.linspace(0.0, 0.95, 20)
    acc, cov = [], []
    for t in q:
        s = c.pa[c.pa.confidence >= t]
        if len(s) < 5:
            break
        acc.append(s.correct_top1.mean()); cov.append(len(s) / len(c.pa))
    ax.plot(cov, acc, "o-", color=BLUE, lw=1.5, ms=4)
    ax.set_xlabel("coverage (fraction of spectra answered)")
    ax.set_ylabel("accuracy among answered")
    ax.set_ylim(0, 1)
    ax.set_title("b · accuracy–coverage: abstention buys accuracy", fontsize=9, loc="left")
    fig.suptitle("Figure 7 · Confidence behaviour", x=0.06, ha="left", fontsize=11,
                 weight="bold", color=INK)
    save(fig, "F07_confidence")


# ── F08 ──────────────────────────────────────────────────────────────────────
def f08_openset(c):
    fig, axs = plt.subplots(1, 3, figsize=(11.0, 3.9),
                            gridspec_kw={"width_ratios": [1.05, 1.0, 1.05]})
    ax = axs[0]
    r = c.roc.sort_values("fpr")
    ax.plot(r.fpr, r.tpr, color=GREEN, lw=1.8,
            label=f"joint · AUROC {c.summ['openset']['joint_auroc']:.3f}")
    ax.plot([0, 1], [0, 1], color=LINE, ls="--", lw=1.0)
    op = c.summ["openset"]["operating_point"]
    ax.plot([0.05], [op["ood_reject"]], "o", color=RED, ms=7)
    ax.annotate(f"operating point\n5% in-domain rejected\n{op['ood_reject']:.0%} negatives caught",
                (0.05, op["ood_reject"]), textcoords="offset points", xytext=(12, -34),
                fontsize=7.4, color=RED)
    ax.set_xlabel("in-domain false rejection"); ax.set_ylabel("negatives rejected")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("a · joint rejection ROC", fontsize=9, loc="left")
    ax = axs[1]
    d = c.os_ch[c.os_ch.channel != "JOINT"].sort_values("auroc")
    cols = [GREEN if v >= 0.7 else (AMBER if v >= 0.5 else RED) for v in d.auroc]
    ax.barh(range(len(d)), d.auroc, color=cols, alpha=0.9)
    ax.axvline(0.5, color=INK, lw=0.9, ls="--")
    ax.set_yticks(range(len(d))); ax.set_yticklabels(d.channel, fontsize=7.4)
    ax.set_xlabel("AUROC (declared sign)"); ax.set_xlim(0, 1)
    ax.set_title("b · per-channel evidence", fontsize=9, loc="left")
    ax.text(0.02, -0.8, "below 0.5 = the channel is inverted", fontsize=6.8, color=RED)
    ax = axs[2]
    bins = np.linspace(min(c.j_in.min(), c.j_out.min()), max(c.j_in.max(), c.j_out.max()), 40)
    ax.hist(c.j_in, bins=bins, color=GREEN, alpha=0.75, label="in-domain Raman")
    ax.hist(c.j_out, bins=bins, color=RED, alpha=0.55, label="synthetic negatives")
    ax.axvline(op["threshold"], color=INK, lw=1.2, ls="--")
    ax.set_xlabel("joint rejection score"); ax.set_ylabel("spectra")
    ax.legend(frameon=False, fontsize=7.6)
    ax.set_title("c · score separation", fontsize=9, loc="left")
    fig.suptitle("Figure 8 · Open-set rejection (Raman only; negatives are synthetic)",
                 x=0.05, ha="left", fontsize=11, weight="bold", color=INK)
    save(fig, "F08_openset_rejection")


# ── F09 ──────────────────────────────────────────────────────────────────────
def _radar(ax, mag, conf, axes, title, color=PURPLE):
    n = len(axes)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    cl = np.concatenate([mag, mag[:1]])
    aa = np.concatenate([ang, ang[:1]])
    ax.plot(aa, cl, color=color, lw=1.5)
    ax.fill(aa, cl, color=color, alpha=0.18)
    for a, m, cf in zip(ang, mag, conf):
        ax.plot([a, a], [0, m], color=color, lw=0.8 + 2.6 * float(cf), alpha=0.35 + 0.65 * float(cf),
                solid_capstyle="round")
        ax.plot([a], [m], "o", ms=2.5 + 4.5 * float(cf), color=color,
                alpha=0.35 + 0.65 * float(cf))
    ax.set_xticks(ang)
    ax.set_xticklabels([a.replace("_", "\n") for a in axes], fontsize=6.2)
    ax.set_yticklabels([])
    ax.set_ylim(0, max(0.05, float(np.max(mag)) * 1.18))
    ax.set_title(title, fontsize=8.4, pad=13, color=INK)
    ax.grid(color=LINE, lw=0.4, alpha=0.6)


def f09_radars(c):
    picks = []
    for cl in ["peptide_protein", "mono_oligosaccharide", "purine", "fatty_acid",
               "chromophore_pigment", "sulfur_thiol_cofactor"]:
        idx = np.where(c.cls == cl)[0]
        if len(idx):
            picks.append((cl, int(idx[len(idx) // 2])))
    fig, axs = plt.subplots(2, 3, figsize=(10.6, 8.4), subplot_kw={"polar": True})
    for ax, (cl, i) in zip(axs.ravel(), picks):
        _radar(ax, c.E[i], c.Econf[i], c.axes, f"{c.y[i]}\n({cl})")
    fig.suptitle("Figure 9 · Biochemical Evidence Profiles\n"
                 "spoke thickness and marker size encode confidence — weak evidence "
                 "looks weak", x=0.04, ha="left", fontsize=11, weight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.subplots_adjust(hspace=0.52)
    save(fig, "F09_evidence_radars")


# ── F10 ──────────────────────────────────────────────────────────────────────
def f10_waterfall(c):
    rep = max(c.reports, key=lambda r: max(r["evidence_profile"]["magnitude"]))
    ch = max(rep["provenance"], key=lambda p: p["total_contribution"])
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    box(ax, 0.02, 0.80, 0.20, 0.11,
        f"axis\n{ch['axis']}\n{ch['total_contribution']:.3f}", "#f5f3ff", PURPLE, 8.2, "bold")
    tot = ch["total_contribution"] + 1e-12
    n = len(ch["csm_chain"])
    for k, link in enumerate(ch["csm_chain"][:4]):
        yk = 0.62 - k * 0.155
        box(ax, 0.27, yk, 0.22, 0.125,
            f"{link['csm_id']} · {link['contribution_share']:.0%} of the axis\n"
            f"activation {link['activation']:.2f} × loading {link['axis_loading']:.2f}",
            "#ecfdf5", GREEN, 7.6)
        arrow(ax, (0.12, 0.80), (0.27, yk + 0.062), LINE, "-|>", 0.8)
        bands = ", ".join(f"{b:.0f}" for b in link["dominant_bands"][:5])
        box(ax, 0.52, yk, 0.20, 0.125, f"LSMs: {', '.join(link['lsms'][:2])}\nbands: {bands}",
            "#eff6ff", BLUE, 7.2)
        arrow(ax, (0.49, yk + 0.062), (0.52, yk + 0.062), LINE, "-|>", 0.8)
        mols = ", ".join(link["molecules"][:3])
        box(ax, 0.75, yk, 0.23, 0.125,
            f"molecules ({len(link['molecules'])}):\n{mols}…", "#f8fafc", GREY, 7.2)
        arrow(ax, (0.72, yk + 0.062), (0.75, yk + 0.062), LINE, "-|>", 0.8)
    ax.text(0.02, 0.72, f"spectrum: {rep['truth_molecule']}\nclass: {rep['truth_class']}",
            fontsize=8, color=MUTED, va="top")
    ax.text(0.02, 0.055,
            f"Top {min(4, n)} of {n} contributing CSMs shown; the listed links account for "
            f"{ch['explained_share']:.0%} of the axis value. Every molecule name resolves to "
            "measured reference spectra in the frozen corpus.",
            fontsize=7.8, color=MUTED, style="italic")
    ax.set_title("Figure 10 · Provenance waterfall: axis → CSM → LSM → molecule → spectra",
                 loc="left", fontsize=11, weight="bold", color=INK)
    save(fig, "F10_provenance_waterfall")


# ── F11 ──────────────────────────────────────────────────────────────────────
def f11_robustness(c):
    kinds = list(dict.fromkeys(c.rob.perturbation))
    fig, axs = plt.subplots(2, 4, figsize=(12.0, 6.0))
    for ax, k in zip(axs.ravel(), kinds):
        d = c.rob[c.rob.perturbation == k]
        for r in REP_ORDER:
            s = d[d.representation == r].sort_values("level")
            ax.plot(range(len(s)), s.class_top1_grouped, "o-", color=REP_COL[r], lw=1.3, ms=3.4,
                    label=r)
        ax.set_xticks(range(len(d[d.representation == "raw"])))
        ax.set_xticklabels([f"{v:g}" for v in
                            d[d.representation == "raw"].sort_values("level").level],
                           fontsize=6.4)
        ax.set_title(k.replace("_", " "), fontsize=8.4, loc="left")
        ax.set_ylim(0, 1); ax.tick_params(labelsize=7)
        ax.set_xlabel("severity", fontsize=7.4)
    ax = axs.ravel()[-1]
    ax.axis("off")
    d = c.robs.set_index("representation")
    ax.text(0.0, 0.95, "class retention, molecule-grouped", fontsize=8.6, weight="bold")
    for i, r in enumerate(REP_ORDER):
        ax.text(0.0, 0.78 - i * 0.14,
                f"{r:9s} {d.loc[r,'clean_class_top1_grouped']:.3f} → "
                f"{d.loc[r,'mean_class_top1_grouped_perturbed']:.3f}   "
                f"(ret {d.loc[r,'class_retention_grouped']:.3f})",
                fontsize=7.8, color=REP_COL[r], family="DejaVu Sans Mono")
    ax.text(0.0, 0.15, "CSM beats raw on BOTH axes:\nmore accurate on unseen molecules\n"
            "and degrades more slowly.", fontsize=7.8, color=INK, style="italic")
    h = [Line2D([], [], color=REP_COL[r], marker="o", ms=4, label=r) for r in REP_ORDER]
    axs.ravel()[0].legend(handles=h, frameon=False, fontsize=7, loc="lower left")
    axs.ravel()[0].set_ylabel("class top-1 (molecule unseen)", fontsize=7.6)
    axs.ravel()[4].set_ylabel("class top-1 (molecule unseen)", fontsize=7.6)
    fig.suptitle("Figure 11 · Noise robustness across representations "
                 "(chemistry class, molecule-grouped)", x=0.05, ha="left", fontsize=11,
                 weight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, "F11_noise_robustness")


# ── F12 ──────────────────────────────────────────────────────────────────────
def f12_reports(c):
    picks = c.reports[:3]
    fig, axs = plt.subplots(3, 3, figsize=(11.0, 9.0),
                            gridspec_kw={"width_ratios": [1.25, 0.95, 1.0]})
    for row, rep in enumerate(picks):
        i = int(np.argmax([1 if (yy == rep["truth_molecule"]) else 0 for yy in c.y]))
        ax = axs[row, 0]
        ax.plot(c.grid, c.X[i], color=INK, lw=0.9)
        ax.set_title(f"{rep['truth_molecule']} · {rep['truth_class']}", fontsize=8.6, loc="left")
        ax.set_xlabel("cm$^{-1}$", fontsize=7.4); ax.tick_params(labelsize=7)
        ax = axs[row, 1]
        ax.axis("off")
        tm = rep["top_molecules"][:3]
        txt = ["ANALYTE RETRIEVAL"] + [f"  {k+1}. {m}  ({s:.3f})" for k, (m, s) in enumerate(tm)]
        txt += ["", f"confidence  {rep['confidence']:.3f}",
                f"margin      {rep['margin']:.3f}",
                f"entropy     {rep['entropy']:.3f}", "",
                "CHEMISTRY CLASS",
                f"  {rep['chemistry_class'][0]}  ({rep['chemistry_class'][1]:.3f})", "",
                "DIAGNOSTICS",
                f"  EV {rep['diagnostics']['explained_variance']:.3f} · "
                f"{int(rep['diagnostics']['n_active_csms'])} CSMs",
                f"  unassigned {rep['evidence_profile']['unassigned_mass']:.3f}"]
        for n in rep["notes"]:
            txt += ["", f"! {n[:52]}"]
        ax.text(0.0, 1.0, "\n".join(txt), fontsize=7.0, va="top", family="DejaVu Sans Mono",
                color=INK, linespacing=1.45)
        ax = fig.add_subplot(3, 3, row * 3 + 3, polar=True)
        axs[row, 2].axis("off")
        _radar(ax, np.array(rep["evidence_profile"]["magnitude"]),
               np.array(rep["evidence_profile"]["confidence"]), c.axes, "")
    fig.suptitle("Figure 12 · Representative inference reports", x=0.04, ha="left",
                 fontsize=11, weight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "F12_representative_reports")


# ── F13 ──────────────────────────────────────────────────────────────────────
def f13_active_csms(c):
    fig, axs = plt.subplots(1, 2, figsize=(11.0, 4.6),
                            gridspec_kw={"width_ratios": [1.55, 1.0]})
    ax = axs[0]
    order = np.argsort([list(sorted(set(c.cls))).index(x) for x in c.cls])
    im = ax.imshow(np.log1p(c.Acsm[order].T / (c.Acsm.max() + 1e-12) * 20),
                   aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_ylabel("frozen CSM"); ax.set_xlabel("spectra (grouped by chemistry class)")
    bounds, prev = [], None
    for k, i in enumerate(order):
        if c.cls[i] != prev:
            bounds.append((k, c.cls[i])); prev = c.cls[i]
    for b, _ in bounds[1:]:
        ax.axvline(b, color="white", lw=0.5, alpha=0.6)
    ax.set_xticks([b for b, _ in bounds])
    ax.set_xticklabels([l for _, l in bounds], rotation=60, ha="right", fontsize=5.8)
    fig.colorbar(im, ax=ax, shrink=0.8, label="log activation")
    ax.set_title("a · which motifs fire, and for what chemistry", fontsize=9, loc="left")
    ax = axs[1]
    freq = (c.Acsm > 1e-9).mean(axis=0)
    o = np.argsort(-freq)[:18]
    ax.barh(range(len(o)), freq[o][::-1], color=GREEN, alpha=0.85)
    ax.set_yticks(range(len(o)))
    ax.set_yticklabels([c.recs[j]["csm_id"] for j in o][::-1], fontsize=6.6)
    ax.set_xlabel("fraction of spectra activating this CSM")
    ax.set_title("b · the 18 most-used motifs", fontsize=9, loc="left")
    fig.suptitle("Figure 13 · Active CSMs across the corpus", x=0.05, ha="left", fontsize=11,
                 weight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, "F13_active_csms")


# ── F14 ──────────────────────────────────────────────────────────────────────
def f14_schematic(c):
    fig, axs = plt.subplots(1, 2, figsize=(11.4, 5.2),
                            gridspec_kw={"width_ratios": [1.0, 1.15]})
    ax = axs[0]
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    box(ax, 0.05, 0.80, 0.36, 0.11, "molecule\n(154 canonical Raman references)",
        "#f8fafc", GREY, 8.0, "bold")
    box(ax, 0.05, 0.56, 0.36, 0.11, "CSM activation\n49 consensus spectral motifs",
        "#ecfdf5", GREEN, 8.0, "bold")
    box(ax, 0.05, 0.32, 0.36, 0.11, "Biochemical Evidence Profile\n11 declared, grounded axes",
        "#f5f3ff", PURPLE, 8.0, "bold")
    box(ax, 0.05, 0.09, 0.36, 0.11, "radar + provenance\nevery spoke walks back to spectra",
        "#fffbeb", AMBER, 8.0)
    for y0, y1 in ((0.80, 0.67), (0.56, 0.43), (0.32, 0.20)):
        arrow(ax, (0.23, y0), (0.23, y1))
    ax.text(0.44, 0.735, "NNLS onto a\nfrozen dictionary", fontsize=7.4, color=MUTED, va="center")
    ax.text(0.44, 0.495, "fixed sparse map\n(no factorisation)", fontsize=7.4, color=MUTED,
            va="center")
    ax.text(0.44, 0.26, "additive, so every\nnumber decomposes", fontsize=7.4, color=MUTED,
            va="center")
    ax.set_title("a · the abstraction chain", fontsize=9.5, loc="left", weight="bold")
    ax = axs[1]
    keep = np.argsort(-c.M.sum(axis=1))[:24]
    im = ax.imshow(c.M[keep], aspect="auto", cmap="Purples", vmin=0, vmax=float(c.M.max()))
    ax.set_yticks(range(len(keep)))
    ax.set_yticklabels([c.recs[j]["csm_id"] for j in keep], fontsize=6.2)
    ax.set_xticks(range(len(c.axes)))
    ax.set_xticklabels([a.replace("_", " ") for a in c.axes], rotation=55, ha="right",
                       fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8, label="axis loading")
    ax.set_title("b · the frozen CSM → axis map (24 CSMs shown; mean 3.8 axes per CSM)",
                 fontsize=9, loc="left")
    fig.suptitle("Figure 14 · Molecule → CSM → Biochemical Evidence Profile", x=0.04,
                 ha="left", fontsize=11, weight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, "F14_molecule_csm_evidence")


# ── F15 ──────────────────────────────────────────────────────────────────────
def f15_summary(c):
    fig = plt.figure(figsize=(11.0, 7.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.95], hspace=0.45, wspace=0.32)
    ax = fig.add_subplot(gs[0, 0])
    d = c.robs.set_index("representation")
    xs = [d.loc[r, "clean_class_top1_grouped"] for r in REP_ORDER]
    ys = [d.loc[r, "class_retention_grouped"] for r in REP_ORDER]
    for r, x, yv in zip(REP_ORDER, xs, ys):
        ax.plot(x, yv, "o", ms=11, color=REP_COL[r])
        ax.annotate(r, (x, yv), textcoords="offset points", xytext=(9, -3), fontsize=8)
    ax.set_xlabel("class top-1, unseen molecules"); ax.set_ylabel("robustness retention")
    ax.set_title("a · accuracy vs robustness", fontsize=9, loc="left")
    ax = fig.add_subplot(gs[0, 1])
    v = c.axval.dropna(subset=["auroc"]).sort_values("auroc")
    cols = [GREEN if a >= 0.70 else (AMBER if a >= 0.60 else RED) for a in v.auroc]
    ax.barh(range(len(v)), v.auroc, color=cols, alpha=0.9)
    ax.axvline(0.70, color=INK, ls="--", lw=0.9)
    ax.set_yticks(range(len(v))); ax.set_yticklabels(v.axis, fontsize=6.6)
    ax.set_xlim(0.3, 1.0); ax.set_xlabel("axis AUROC vs declared chemistry")
    ax.set_title("b · are the axes real?", fontsize=9, loc="left")
    ax = fig.add_subplot(gs[0, 2])
    ax.axis("off")
    s = c.summ
    lines = [("Split A molecule top-1", f"{s['split_a']['molecule_top1']:.3f}"),
             ("Split A molecule top-5", f"{s['split_a']['molecule_top5']:.3f}"),
             ("Split A ECE", f"{s['split_a']['ece']:.3f}"),
             ("Split B class top-1", f"{s['split_b']['class_top1']:.3f}"),
             ("Split B macro F1", f"{s['split_b']['macro_f1']:.3f}"),
             ("open-set joint AUROC", f"{s['openset']['joint_auroc']:.3f}"),
             ("grounded axes", f"{s['evidence']['n_grounded']} of 11"),
             ("provenance chains broken", f"{s['provenance']['broken']}"),
             ("mean explained variance", f"{s['projection']['mean_ev']:.3f}")]
    ax.text(0.0, 1.0, "headline numbers", fontsize=9, weight="bold", va="top")
    for i, (k, v2) in enumerate(lines):
        ax.text(0.0, 0.88 - i * 0.093, k, fontsize=7.8, color=INK)
        ax.text(1.0, 0.88 - i * 0.093, v2, fontsize=7.8, color=GREEN, ha="right",
                family="DejaVu Sans Mono", weight="bold")
    ax = fig.add_subplot(gs[1, :])
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    g = c.gates
    for i, (_, row) in enumerate(g.iterrows()):
        x = 0.02 + (i % 5) * 0.196
        y = 0.68 - (i // 5) * 0.30
        ok = row.status == "PASS"
        box(ax, x, y, 0.185, 0.22, row.gate.replace(" (", "\n("),
            "#ecfdf5" if ok else "#fef2f2", GREEN if ok else RED, 6.4)
    ax.text(0.02, 0.03, f"{int((g.status == 'PASS').sum())} of {len(g)} gates pass · "
            f"engine fingerprint {c.state['engine_fingerprint']} · "
            "Raman only, no cross-modality experiment", fontsize=8, color=MUTED)
    fig.suptitle("Figure 15 · Phase 05 architecture summary", x=0.04, ha="left", fontsize=11,
                 weight="bold", color=INK)
    save(fig, "F15_architecture_summary")


def main():
    c = Ctx()
    print("[figures]")
    for fn in (f01_pipeline, f02_projection, f03_confusion, f04_calibration, f05_reliability,
               f06_topk, f07_confidence, f08_openset, f09_radars, f10_waterfall,
               f11_robustness, f12_reports, f13_active_csms, f14_schematic, f15_summary):
        fn(c)
    print(f"[figures] {len(list(F.glob('*.png')))} PNG written to {F}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
