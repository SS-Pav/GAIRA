#!/usr/bin/env python3
"""GAIRA V7 — Phase 09 figures. PNG at 300 dpi, deterministic."""
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
from gaira.v7.io import PhaseOutputs, frozen_root        # noqa: E402
from gaira.v7.canonical import GAIRAEngine               # noqa: E402

OUT = PhaseOutputs("09", extra=("interactive", "manifests", "reports_examples"))
T, A_, F = OUT.tables, OUT.artifacts, OUT.figures
FROZEN = frozen_root()
INK, MUTED, LINE = "#1a1a1a", "#6b7280", "#9ca3af"
BLUE, GREEN, AMBER, RED, GREY = "#2563eb", "#15803d", "#b45309", "#b91c1c", "#4b5563"
PURPLE, TEAL = "#7c3aed", "#0f766e"
DPI = 300
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
    fig.savefig(F / f"{name}.png", dpi=DPI); plt.close(fig); print(f"  {name}")


def box(ax, x, y, w, h, t, fc="#eef2ff", ec=BLUE, fs=8.0, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008", fc=fc, ec=ec, lw=1.1))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, color=INK,
            weight=weight, linespacing=1.35)


def arrow(ax, p0, p1, col=LINE, lw=1.2, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=11, color=col, lw=lw,
                                 linestyle=ls, shrinkA=2, shrinkB=2))


class C:
    def __init__(self):
        self.s = json.loads((A_ / "phase09_summary_v1.json").read_text())
        self.state = json.loads((OUT.root / "PHASE_STATE.json").read_text())
        z = np.load(A_ / "engine_activations_v1.npz", allow_pickle=True)
        self.L, self.A, self.E, self.P = z["L"], z["A"], z["E"], z["P"]
        self.y = np.array([str(v) for v in z["y"]])
        self.cls = np.array([str(v) for v in z["cls"]])
        self.axes = [str(v) for v in z["axes"]]
        self.ranks, self.margin = z["ranks"], z["margin"]
        self.src = np.array([str(v) for v in z["source"]])
        self.per = pd.read_csv(T / "engine_outputs_all_spectra_v1.csv")
        self.cm = pd.read_csv(T / "csm_confusion_matrix_v1.csv")
        self.pc = pd.read_csv(T / "csm_per_class_v1.csv")
        self.rd = pd.read_csv(T / "retrieval_rank_distribution_v1.csv")
        self.rc = pd.read_csv(T / "retrieval_risk_coverage_v1.csv")
        self.pa = pd.read_csv(T / "chemistry_per_axis_v1.csv")
        self.roc = pd.read_csv(T / "chemistry_roc_v1.csv")
        self.pr = pd.read_csv(T / "chemistry_pr_v1.csv")
        self.rel = pd.read_csv(T / "chemistry_reliability_v1.csv")
        self.rob = pd.read_csv(T / "noise_robustness_v1.csv")
        self.reps = pd.read_csv(T / "representative_analytes_v1.csv")
        self.gates = pd.read_csv(T / "phase09_gates_v1.csv")
        b = np.load(FROZEN / "phase01/artifacts/balanced_references_v1.npz", allow_pickle=True)
        self.X, self.grid = np.asarray(b["X"], float), np.asarray(b["grid"], float)
        self.CSM = np.load(FROZEN / "phase02/artifacts/csm_dictionary_v1.npz")["CSM"]
        self.H = np.load(FROZEN / "phase01/artifacts/lsm_dictionary_v1.npz")["H"]
        self.recs = json.loads(
            (FROZEN / "phase02/artifacts/csm_registry_v1.json").read_text())["csms"]
        self.canon = pd.read_csv(FROZEN / "phase00/tables/canonical_analytes_v1.csv")


# ── 1 architecture ───────────────────────────────────────────────────────────
def f01(c):
    fig, ax = plt.subplots(figsize=(12.4, 6.4)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.02, 0.955, "OFFLINE — learned once, then frozen", fontsize=9.6, weight="bold",
            color=GREEN)
    off = [(0.02, "375 pure Raman\nspectra · 154 molecules"), (0.20, "balanced references\n"
           "1 molecule = 1 unit"), (0.38, "16 class-local NMF fits\n→ 50 LSMs"),
           (0.58, "consensus graph\n→ 49 CSMs"), (0.78, "chemistry evidence map\n→ 16 axes")]
    for i, (x, t) in enumerate(off):
        box(ax, x, 0.80, 0.165, 0.10, t, "#ecfdf5", GREEN, 7.4)
        if i:
            arrow(ax, (off[i-1][0] + 0.165, 0.85), (x, 0.85), GREEN)
    ax.plot([0.0, 1.0], [0.75, 0.75], color=RULE if (RULE := "#d1d5db") else LINE, lw=1.2)
    ax.text(0.02, 0.715, "INFERENCE — the canonical path, fixed", fontsize=9.6, weight="bold",
            color=BLUE)
    inf = [(0.02, 0.15, "unknown Raman\nspectrum"), (0.19, 0.15, "canonical\npreprocessing"),
           (0.36, 0.13, "LSM\nprojection"), (0.51, 0.13, "CSM\nprojection (49-d)"),
           (0.66, 0.15, "molecular retrieval\n(CSM similarity)"),
           (0.83, 0.15, "16-axis Chemistry\nEvidence + radar")]
    for i, (x, w, t) in enumerate(inf):
        box(ax, x, 0.56, w, 0.11, t, "#eff6ff", BLUE, 7.4, "bold" if i in (3, 5) else "normal")
        if i:
            arrow(ax, (inf[i-1][0] + inf[i-1][1], 0.615), (x, 0.615), BLUE)
    box(ax, 0.30, 0.40, 0.40, 0.10,
        "interpretation report\nmolecules · chemistry radar · supporting CSMs & LSMs\n"
        "diagnostic bands · provenance · confidence", "#fffbeb", AMBER, 7.6, "bold")
    arrow(ax, (0.50, 0.56), (0.50, 0.50), AMBER, 1.4)
    box(ax, 0.02, 0.06, 0.46, 0.28,
        "NOT ON THE INFERENCE PATH\n\nBSV2 (Phase 07 — offline representation only)\n"
        "PCA · UMAP · clustering · latent geometry\ncontinuous coordinates · legacy themes · "
        "legacy BSV\n\nEach exclusion is a measured decision,\nnot a preference.",
        "#fef2f2", RED, 7.4)
    s = c.s
    box(ax, 0.52, 0.06, 0.46, 0.28,
        f"VALIDATED ACROSS ALL 375 SPECTRA\n\n"
        f"LSM reconstruction EV        {s['validation_1_lsm']['mean_explained_variance']:.3f}\n"
        f"CSM chemistry top-1          {s['validation_2_csm']['class_top1']:.3f}\n"
        f"molecule top-1 / top-5       {s['validation_3_retrieval']['top1']:.3f} / "
        f"{s['validation_3_retrieval']['top5']:.3f}\n"
        f"chemistry top-1 (held out)   {s['validation_4_chemistry']['fine_top1_heldout']:.3f}\n"
        f"radar reproducibility        {s['validation_4_chemistry']['radar_reproducibility']:.3f}\n"
        f"every retrieval score reconciles · deterministic",
        "#ecfdf5", GREEN, 7.2)
    ax.set_title("Figure 1 · The complete GAIRA V7 architecture — offline learning above, "
                 "frozen inference below", loc="left", fontsize=11.5, weight="bold", color=INK)
    save(fig, "F01_architecture")


# ── 2 corpus ─────────────────────────────────────────────────────────────────
def f02(c):
    fig, axs = plt.subplots(1, 3, figsize=(12.8, 4.6),
                            gridspec_kw={"width_ratios": [1.5, 1.0, 1.0]})
    d = pd.DataFrame({"cls": c.cls, "mol": c.y}).groupby("cls").agg(
        spectra=("mol", "size"), molecules=("mol", "nunique")).reset_index().sort_values("spectra")
    ax = axs[0]
    ax.barh(range(len(d)), d.spectra, color=BLUE, alpha=0.85, label="spectra")
    ax.barh(range(len(d)), d.molecules, color=GREEN, alpha=0.9, height=0.45, label="molecules")
    ax.set_yticks(range(len(d))); ax.set_yticklabels([SH(x) for x in d.cls], fontsize=7)
    for i, (sp, mo) in enumerate(zip(d.spectra, d.molecules)):
        ax.text(sp + 1.2, i, f"{sp}s/{mo}m", va="center", fontsize=6.4, color=MUTED)
    ax.set_xlim(0, d.spectra.max() * 1.25); ax.legend(frameon=False, fontsize=7.4)
    ax.set_xlabel("count")
    ax.set_title("a · the 16 chemistry families", fontsize=9, loc="left")
    ax = axs[1]
    srcs = sorted(set(c.src)); left = np.zeros(len(d))
    for s_ in srcs:
        v = np.array([int(((c.cls == cc) & (c.src == s_)).sum()) for cc in d.cls])
        ax.barh(range(len(d)), v, left=left, label=s_, alpha=0.9); left += v
    ax.set_yticks(range(len(d))); ax.set_yticklabels([])
    ax.legend(frameon=False, fontsize=6.4, loc="lower right"); ax.set_xlabel("spectra")
    ax.set_title("b · source coverage", fontsize=9, loc="left")
    ax = axs[2]
    reps = pd.Series(c.y).value_counts().value_counts().sort_index()
    ax.bar(reps.index.astype(str), reps.values, color=AMBER, alpha=0.9)
    for i, v in enumerate(reps.values):
        ax.text(i, v + 1.5, str(v), ha="center", fontsize=8, weight="bold")
    ax.set_xlabel("spectra per molecule"); ax.set_ylabel("molecules")
    ax.set_title(f"c · replicate structure ({int(reps.get(1,0))} singletons)", fontsize=9,
                 loc="left")
    fig.suptitle("Figure 2 · The grounded Raman corpus — 375 spectra, 154 canonical molecules, "
                 "16 chemistry families", x=0.03, ha="left", fontsize=11.5, weight="bold",
                 color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.91]); save(fig, "F02_corpus")


# ── 3 preprocessing ──────────────────────────────────────────────────────────
def f03(c):
    from gaira.v7.canonical.engine import _asls
    rng = np.random.default_rng(0)
    i = 40
    raw = c.X[i] * 100 + 20 + 0.02 * np.arange(len(c.grid)) + rng.normal(0, 0.3, len(c.grid))
    fig, axs = plt.subplots(1, 4, figsize=(13.0, 3.6))
    base = _asls(raw)
    steps = [(raw, "1 · raw + baseline + noise", GREY, base),
             (np.clip(raw - base, 0, None), "2 · asLS baseline removed", BLUE, None),
             (None, "3 · Savitzky–Golay (9, 3)", TEAL, None),
             (None, "4 · L2 normalised", GREEN, None)]
    from scipy.signal import savgol_filter
    s2 = np.clip(raw - base, 0, None)
    s3 = np.clip(savgol_filter(s2, 9, 3), 0, None)
    s4 = s3 / np.linalg.norm(s3)
    for ax, (v, t, col, b), vv in zip(axs, steps, [raw, s2, s3, s4]):
        ax.plot(c.grid, vv, color=col, lw=0.9)
        if b is not None:
            ax.plot(c.grid, b, color=RED, lw=1.0, ls="--", label="asLS baseline")
            ax.legend(frameon=False, fontsize=6.4)
        ax.set_title(t, fontsize=8, loc="left"); ax.set_xlabel("cm$^{-1}$", fontsize=7)
        ax.tick_params(labelsize=6.5)
    fig.suptitle("Figure 3 · Canonical preprocessing — crop to 450–1800 cm⁻¹, resample to 676 "
                 "bins at 2.0 cm⁻¹, asLS baseline, SG smoothing, L2 normalisation",
                 x=0.03, ha="left", fontsize=10.5, weight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.88]); save(fig, "F03_preprocessing")


# ── 4/5 representative LSMs and CSMs ─────────────────────────────────────────
def _motif_panel(c, M, act, ids, recs, title, name, n=6):
    use = np.argsort(-act.mean(axis=0))[:n]
    fig, axs = plt.subplots(2, 3, figsize=(12.4, 5.6))
    for ax, j in zip(axs.ravel(), use):
        ax.plot(c.grid, M[int(j)], color=PURPLE, lw=1.0)
        top = np.argsort(-act[:, int(j)])[:4]
        mols = ", ".join(sorted(set(c.y[top]))[:3])
        bands = ""
        if recs is not None:
            bands = ", ".join(f"{b:.0f}" for b in recs[int(j)].get("dominant_bands", [])[:5])
            for b in recs[int(j)].get("dominant_bands", [])[:5]:
                ax.axvline(b, color=AMBER, ls=":", lw=0.8, alpha=0.8)
        ax.set_title(f"{ids[int(j)]} · mean activation {act[:, int(j)].mean():.3f}\n"
                     f"top: {mols[:44]}\n{('bands: ' + bands) if bands else ''}",
                     fontsize=6.8, loc="left")
        ax.set_xlabel("cm$^{-1}$", fontsize=7); ax.tick_params(labelsize=6.4)
    fig.suptitle(title, x=0.03, ha="left", fontsize=11, weight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.90]); save(fig, name)


def f04(c):
    ids = [str(s) for s in np.load(FROZEN / "phase01/artifacts/lsm_dictionary_v1.npz",
                                   allow_pickle=True)["motif_ids"]]
    _motif_panel(c, c.H, c.L, ids, None,
                 "Figure 4 · Representative Local Spectral Motifs — the six most-activated of 50",
                 "F04_lsms")


def f05(c):
    ids = [r["csm_id"] for r in c.recs]
    _motif_panel(c, c.CSM, c.A, ids, c.recs,
                 "Figure 5 · Representative Consensus Spectral Motifs — the six most-activated "
                 "of 49 (amber = diagnostic bands)", "F05_csms")


# ── 6 CSM similarity space (UMAP, visualisation only) ────────────────────────
def f06(c):
    try:
        import umap
        mols = sorted(set(c.y.tolist()))
        M = np.vstack([c.A[c.y == m].mean(axis=0) for m in mols])
        cl = np.array([c.cls[c.y == m][0] for m in mols])
        Y = umap.UMAP(n_neighbors=15, min_dist=0.15, metric="cosine",
                      random_state=0).fit_transform(M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12))
    except Exception:
        from sklearn.decomposition import PCA
        mols = sorted(set(c.y.tolist()))
        M = np.vstack([c.A[c.y == m].mean(axis=0) for m in mols])
        cl = np.array([c.cls[c.y == m][0] for m in mols])
        Y = PCA(n_components=2, random_state=0).fit_transform(M)
    fig, ax = plt.subplots(figsize=(9.0, 6.6))
    cmap = plt.get_cmap("tab20")
    for i, f in enumerate(sorted(set(cl))):
        m = cl == f
        ax.scatter(Y[m, 0], Y[m, 1], s=42, color=cmap(i % 20), alpha=0.88, label=SH(f))
    ax.legend(frameon=False, fontsize=6.4, ncol=2, loc="upper left")
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.5, -0.045, "VISUALISATION ONLY — UMAP is not used during inference. Distances in "
            "this plot are not quantitative.", transform=ax.transAxes, ha="center", fontsize=8,
            color=RED, weight="bold")
    ax.set_title("Figure 6 · CSM similarity space, 154 molecules coloured by chemistry family",
                 loc="left", fontsize=11.5, weight="bold", color=INK)
    save(fig, "F06_csm_space")


# ── 7 example inference ──────────────────────────────────────────────────────
def _radar(ax, e, conf, axes, title, color=PURPLE):
    n = len(axes)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    v = e / (e.max() + 1e-12)
    ax.plot(np.concatenate([ang, ang[:1]]), np.concatenate([v, v[:1]]), color=color, lw=1.3)
    ax.fill(np.concatenate([ang, ang[:1]]), np.concatenate([v, v[:1]]), color=color, alpha=0.15)
    for a, m, cf in zip(ang, v, conf):
        ax.plot([a, a], [0, m], color=color, lw=0.5 + 3.0 * float(cf),
                alpha=0.3 + 0.7 * float(cf))
    ax.set_xticks(ang); ax.set_xticklabels([SH(x).replace(" ", "\n") for x in axes], fontsize=5.2)
    ax.set_yticklabels([]); ax.set_ylim(0, 1.15)
    ax.set_title(title, fontsize=7.6, pad=12)
    ax.grid(color=LINE, lw=0.35, alpha=0.6)


def f07(c):
    e = GAIRAEngine.load()
    i = int(np.where(c.cls == "purine")[0][0])
    r = e.infer(c.X[i], already_preprocessed=True)
    fig = plt.figure(figsize=(13.0, 7.0))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.05], hspace=0.42, wspace=0.32)
    ax = fig.add_subplot(gs[0, :2])
    ax.plot(c.grid, c.X[i], color=INK, lw=1.0)
    ax.set_title(f"1 · spectrum — {c.y[i]} ({SH(c.cls[i])})", fontsize=8.6, loc="left")
    ax.set_xlabel("cm$^{-1}$", fontsize=7.4)
    ax = fig.add_subplot(gs[0, 2])
    t = r.csm["top"][:6][::-1]
    ax.barh(range(len(t)), [x["weight"] for x in t], color=GREEN, alpha=0.9)
    ax.set_yticks(range(len(t))); ax.set_yticklabels([x["csm_id"] for x in t], fontsize=6.6)
    ax.set_title("2 · CSM activation", fontsize=8.6, loc="left")
    ax = fig.add_subplot(gs[0, 3])
    m = r.retrieval["top"][:6][::-1]
    ax.barh(range(len(m)), [x["similarity"] for x in m], color=BLUE, alpha=0.9)
    ax.set_yticks(range(len(m))); ax.set_yticklabels([x["molecule"][:20] for x in m],
                                                     fontsize=6.2)
    ax.set_xlim(0, 1.05)
    ax.set_title("3 · molecular retrieval", fontsize=8.6, loc="left")
    ax = fig.add_subplot(gs[1, 0], polar=True)
    _radar(ax, np.array(r.chemistry["evidence"]),
           np.array(r.chemistry["calibrated_probability"]) /
           (max(r.chemistry["calibrated_probability"]) + 1e-12), c.axes,
           "4 · Chemistry Evidence radar\n(relative evidence, NOT concentration)")
    ax = fig.add_subplot(gs[1, 1:3])
    o = np.argsort(-np.array(r.chemistry["evidence"]))
    ev = np.array(r.chemistry["evidence"])[o]
    ax.barh(range(len(o))[::-1], ev, color=PURPLE, alpha=0.9)
    ax.set_yticks(range(len(o))[::-1]); ax.set_yticklabels([SH(c.axes[int(j)]) for j in o],
                                                           fontsize=6.4)
    ax.set_xlabel("relative biochemical evidence (not concentration, not abundance)",
                  fontsize=7.2)
    ax.set_title("5 · ordered bar view of the same vector", fontsize=8.6, loc="left")
    ax = fig.add_subplot(gs[1, 3]); ax.axis("off")
    conf = r.confidence
    txt = (f"6 · INTERPRETATION\n\ntop molecule\n  {r.retrieval['top'][0]['molecule']}\n"
           f"  similarity {r.retrieval['top'][0]['similarity']:.3f}\n\n"
           f"chemistry\n  {SH(r.chemistry['predicted_class'])}\n\n"
           f"confidence     {conf['overall']:.3f}\n"
           f"coverage       {conf['evidence_coverage']:.3f}\n"
           f"margin         {conf['retrieval_margin']:.4f}\n\n"
           f"unknown warning  {conf['unknown_warning']}\n"
           f"outlier warning  {conf['outlier_warning']}")
    ax.text(0.0, 1.0, txt, fontsize=7.0, va="top", family="DejaVu Sans Mono", color=INK,
            linespacing=1.5)
    fig.suptitle("Figure 7 · One inference, end to end", x=0.03, ha="left", fontsize=11.5,
                 weight="bold", color=INK)
    save(fig, "F07_example_inference")


# ── 8 radar examples ─────────────────────────────────────────────────────────
def f08(c):
    fams = ["peptide_protein", "mono_oligosaccharide", "fatty_acid", "purine",
            "sterol_steroid", "sulfur_thiol_cofactor", "phosphate_metabolite",
            "chromophore_pigment"]
    fig, axs = plt.subplots(2, 4, figsize=(13.2, 7.4), subplot_kw={"polar": True})
    for ax, f in zip(axs.ravel(), fams):
        idx = np.where(c.cls == f)[0]
        if not len(idx):
            ax.axis("off"); continue
        i = int(idx[len(idx) // 2])
        _radar(ax, c.E[i], c.P[i] / (c.P[i].max() + 1e-12), c.axes,
               f"{c.y[i][:26]}\n({SH(f)})")
    fig.suptitle("Figure 8 · Chemistry Evidence radars across eight chemistry families\n"
                 "spoke thickness encodes calibrated confidence · radius is RELATIVE "
                 "BIOCHEMICAL EVIDENCE, not concentration and not abundance",
                 x=0.03, ha="left", fontsize=10.5, weight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.90]); fig.subplots_adjust(hspace=0.50)
    save(fig, "F08_radars")


# ── 9 retrieval examples ─────────────────────────────────────────────────────
def f09(c):
    e = GAIRAEngine.load()
    good = int(np.where(c.ranks == 1)[0][np.argmax(c.margin[c.ranks == 1])])
    near = int(np.where((c.ranks > 1) & (c.ranks <= 3))[0][0]) if ((c.ranks > 1) & (c.ranks <= 3)).any() else good
    bad = int(np.argmax(c.ranks))
    fig, axs = plt.subplots(1, 3, figsize=(13.0, 4.4))
    for ax, (i, lab) in zip(axs, ((good, "correct"), (near, "near miss"), (bad, "failure"))):
        r = e.infer(c.X[i], already_preprocessed=True)
        t = r.retrieval["top"][:6][::-1]
        cols = [GREEN if x["molecule"] == c.y[i] else GREY for x in t]
        ax.barh(range(len(t)), [x["similarity"] for x in t], color=cols, alpha=0.9)
        ax.set_yticks(range(len(t))); ax.set_yticklabels([x["molecule"][:24] for x in t],
                                                         fontsize=6.4)
        ax.set_xlim(0, 1.05)
        ax.set_title(f"{lab} · truth {c.y[i][:22]}\nrank {int(c.ranks[i])} · margin "
                     f"{c.margin[i]:.4f} · confidence {r.confidence['overall']:.3f}",
                     fontsize=7.6, loc="left")
    fig.suptitle("Figure 9 · Retrieval examples — green is the true molecule", x=0.03,
                 ha="left", fontsize=11.5, weight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.90]); save(fig, "F09_retrieval_examples")


# ── 10 confusion ─────────────────────────────────────────────────────────────
def f10(c):
    labs = list(c.cm.true_class)
    M = c.cm.drop(columns=["true_class"]).values.astype(float)
    N = M / (M.sum(axis=1, keepdims=True) + 1e-12)
    fig, ax = plt.subplots(figsize=(8.4, 7.4))
    im = ax.imshow(N, cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(range(len(labs))); ax.set_xticklabels([SH(x) for x in labs], rotation=55,
                                                        ha="right", fontsize=7)
    ax.set_yticks(range(len(labs)))
    ax.set_yticklabels([f"{SH(x)} (n={int(M[i].sum())})" for i, x in enumerate(labs)],
                       fontsize=7)
    for i in range(len(labs)):
        for j in range(len(labs)):
            if N[i, j] > 0.02:
                ax.text(j, i, f"{N[i,j]:.2f}", ha="center", va="center", fontsize=5.8,
                        color="white" if N[i, j] > 0.55 else INK)
    fig.colorbar(im, ax=ax, shrink=0.78, label="row-normalised rate")
    s = c.s["validation_2_csm"]
    ax.set_title(f"Figure 10 · Chemistry-class confusion, molecule-grouped\n"
                 f"top-1 {s['class_top1']:.3f} · macro-F1 {s['macro_f1']:.3f} · balanced "
                 f"{s['balanced_accuracy']:.3f}", loc="left", fontsize=10.5, weight="bold",
                 color=INK)
    save(fig, "F10_confusion")


# ── 11/12 ROC and PR ─────────────────────────────────────────────────────────
def f11(c):
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    cmap = plt.get_cmap("tab20")
    for i, a in enumerate(sorted(set(c.roc.axis))):
        d = c.roc[c.roc.axis == a]
        auc_ = float(c.pa[c.pa.axis == a].auc.iloc[0])
        ax.plot(d.fpr, d.tpr, lw=1.2, color=cmap(i % 20), label=f"{SH(a)} ({auc_:.3f})")
    ax.plot([0, 1], [0, 1], color=LINE, ls="--", lw=1.0)
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.legend(frameon=False, fontsize=5.8, ncol=2, loc="lower right")
    ax.set_title(f"Figure 11 · Per-axis ROC — macro AUC "
                 f"{c.s['validation_4_chemistry']['macro_auc']:.4f}\n"
                 "IN-SAMPLE: the chemistry map is fitted on all 375 spectra. Held-out top-1 is "
                 f"{c.s['validation_4_chemistry']['fine_top1_heldout']:.3f}.",
                 loc="left", fontsize=10, weight="bold", color=INK)
    save(fig, "F11_roc")


def f12(c):
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    cmap = plt.get_cmap("tab20")
    for i, a in enumerate(sorted(set(c.pr.axis))):
        d = c.pr[c.pr.axis == a]
        ap = float(c.pa[c.pa.axis == a].average_precision.iloc[0])
        ax.plot(d.recall, d.precision, lw=1.2, color=cmap(i % 20), label=f"{SH(a)} ({ap:.3f})")
    ax.set_xlabel("recall"); ax.set_ylabel("precision"); ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=5.8, ncol=2, loc="lower left")
    ax.set_title(f"Figure 12 · Per-axis precision–recall — macro AP "
                 f"{c.s['validation_4_chemistry']['macro_average_precision']:.4f}\n"
                 "IN-SAMPLE, as Figure 11.", loc="left", fontsize=10, weight="bold", color=INK)
    save(fig, "F12_pr")


# ── 13 calibration ───────────────────────────────────────────────────────────
def f13(c):
    fig, axs = plt.subplots(1, 3, figsize=(12.4, 4.0))
    ax = axs[0]
    r = c.rel.dropna(subset=["empirical_accuracy"])
    ax.plot([0, 1], [0, 1], color=LINE, ls="--", lw=1.0)
    ax.plot(r.bin_center, r.empirical_accuracy, "o-", color=PURPLE, lw=1.5, ms=5)
    for _, row in r.iterrows():
        ax.annotate(f"n={int(row['count'])}", (row.bin_center, row.empirical_accuracy),
                    textcoords="offset points", xytext=(4, -10), fontsize=6, color=MUTED)
    ax.set_xlabel("calibrated chemistry confidence"); ax.set_ylabel("empirical accuracy")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title(f"a · chemistry reliability · ECE "
                 f"{c.s['validation_4_chemistry']['ece']:.4f}", fontsize=9, loc="left")
    ax = axs[1]
    ax.plot(c.rc.coverage, c.rc.accuracy, "o-", color=BLUE, lw=1.5, ms=4)
    ax.axhline(0.80, color=RED, ls="--", lw=1.0)
    ax.text(0.05, 0.808, "accuracy 0.80", fontsize=6.8, color=RED)
    ax.set_xlabel("coverage"); ax.set_ylabel("molecule top-1 among answered")
    ax.set_title("b · risk–coverage: when to abstain", fontsize=9, loc="left")
    ax = axs[2]
    ok = c.ranks <= 1
    conf = c.per.confidence.values
    bins = np.linspace(0, 1, 26)
    ax.hist(conf[ok], bins=bins, color=GREEN, alpha=0.75, label=f"correct (n={ok.sum()})")
    ax.hist(conf[~ok], bins=bins, color=RED, alpha=0.6, label=f"wrong (n={(~ok).sum()})")
    ax.set_xlabel("engine confidence"); ax.set_ylabel("spectra")
    ax.legend(frameon=False, fontsize=7.4)
    ax.set_title("c · confidence separates right from wrong", fontsize=9, loc="left")
    fig.suptitle("Figure 13 · Calibration and selective prediction", x=0.03, ha="left",
                 fontsize=11.5, weight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.90]); save(fig, "F13_calibration")


# ── 14 noise ─────────────────────────────────────────────────────────────────
def f14(c):
    kinds = list(dict.fromkeys(c.rob.perturbation))
    fig, axs = plt.subplots(2, 4, figsize=(12.8, 5.6))
    for ax, k in zip(axs.ravel(), kinds):
        d = c.rob[c.rob.perturbation == k].sort_values("level")
        ax.plot(range(len(d)), d.retrieval_top1, "o-", color=BLUE, lw=1.2, ms=3,
                label="molecule top-1")
        ax.plot(range(len(d)), d.chemistry_top1, "s-", color=GREEN, lw=1.2, ms=3,
                label="chemistry top-1")
        ax.plot(range(len(d)), d.radar_cosine, "^--", color=PURPLE, lw=1.0, ms=3,
                label="radar cosine")
        ax.set_xticks(range(len(d))); ax.set_xticklabels([f"{v:g}" for v in d.level], fontsize=6)
        ax.set_ylim(0, 1.05); ax.set_title(k.replace("_", " "), fontsize=8, loc="left")
        ax.tick_params(labelsize=6.5)
    ax = axs.ravel()[-1]; ax.axis("off")
    n = c.s["noise_robustness"]
    ax.text(0.0, 0.9, "mean across all perturbations", fontsize=8.4, weight="bold")
    for i, (k, v) in enumerate((("molecule top-1", n["mean_retrieval_top1"]),
                                ("chemistry top-1", n["mean_chemistry_top1"]),
                                ("radar cosine", n["mean_radar_cosine"]))):
        ax.text(0.0, 0.72 - i * 0.13, f"{k:18s} {v:.4f}", fontsize=8,
                family="DejaVu Sans Mono")
    ax.text(0.0, 0.24, "The radar is the most robust output:\nit degrades far more slowly than "
            "molecule\nidentity, which is what a chemistry-level\nanswer should do.",
            fontsize=7.2, color=MUTED, style="italic")
    axs.ravel()[0].legend(frameon=False, fontsize=6)
    fig.suptitle("Figure 14 · Noise robustness of the complete engine — 7 perturbations × "
                 "5 levels", x=0.03, ha="left", fontsize=11.5, weight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.92]); save(fig, "F14_noise")


# ── 15 failure analysis ──────────────────────────────────────────────────────
def f15(c):
    fig, axs = plt.subplots(1, 3, figsize=(12.6, 4.4))
    ax = axs[0]
    d = c.pc.sort_values("f1")
    ax.barh(range(len(d)), d.f1, color=[RED if v < 0.6 else (AMBER if v < 0.8 else GREEN)
                                        for v in d.f1], alpha=0.9)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([f"{SH(x)} (n={int(n)})" for x, n in zip(d["class"], d.n)], fontsize=6.6)
    ax.set_xlim(0, 1.05); ax.set_xlabel("chemistry F1")
    ax.set_title("a · per-class chemistry F1", fontsize=9, loc="left")
    ax = axs[1]
    low = c.per.csm_ev < 0.5
    ax.scatter(c.per.csm_ev[~low], c.per.confidence[~low], s=10, color=GREY, alpha=0.5,
               label="EV ≥ 0.5")
    ax.scatter(c.per.csm_ev[low], c.per.confidence[low], s=26, color=RED, alpha=0.85,
               label="EV < 0.5")
    ax.axvline(0.5, color=RED, ls="--", lw=1.0)
    ax.set_xlabel("CSM explained variance"); ax.set_ylabel("engine confidence")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title(f"b · {int(low.sum())} low-EV spectra flagged", fontsize=9, loc="left")
    ax = axs[2]; ax.axis("off")
    w = c.s["warnings"]
    worst = c.per.assign(rank=c.ranks).sort_values("rank", ascending=False).head(10)
    ax.text(0.0, 0.97, "the ten worst-ranked spectra", fontsize=8.6, weight="bold")
    ax.text(0.0, 0.89, f"{'molecule':24s}{'rank':>5s}  {'EV':>5s}  warn", fontsize=6.2,
            family="DejaVu Sans Mono", color=MUTED)
    yy = 0.82
    for _, r in worst.iterrows():
        ax.text(0.0, yy, f"{r.molecule[:23]:24s}{int(r['rank']):5d}  {r.csm_ev:5.2f}  "
                f"{'U' if r.unknown_warning else ''}{'O' if r.outlier_warning else ''}",
                fontsize=6.2, family="DejaVu Sans Mono", color=INK)
        yy -= 0.065
    ax.text(0.0, 0.10, f"unknown warnings: {w['unknown']}   outlier warnings: {w['outlier']}\n"
            "The engine flags what it cannot explain rather\nthan answering confidently.",
            fontsize=7.2, color=INK, style="italic")
    fig.suptitle("Figure 15 · Failure analysis", x=0.03, ha="left", fontsize=11.5,
                 weight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.90]); save(fig, "F15_failures")


# ── 16 summary ───────────────────────────────────────────────────────────────
def f16(c):
    fig = plt.figure(figsize=(12.4, 7.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0], hspace=0.40, wspace=0.30)
    s = c.s
    ax = fig.add_subplot(gs[0, 0]); ax.axis("off")
    rows = [("LSM reconstruction EV", f"{s['validation_1_lsm']['mean_explained_variance']:.4f}"),
            ("LSM replicate consistency", f"{s['validation_1_lsm']['replicate_consistency']:.4f}"),
            ("CSM chemistry top-1", f"{s['validation_2_csm']['class_top1']:.4f}"),
            ("CSM macro-F1", f"{s['validation_2_csm']['macro_f1']:.4f}"),
            ("molecule top-1", f"{s['validation_3_retrieval']['top1']:.4f}"),
            ("molecule top-5", f"{s['validation_3_retrieval']['top5']:.4f}"),
            ("molecule MRR", f"{s['validation_3_retrieval']['mrr']:.4f}"),
            ("chemistry top-1 (held out)",
             f"{s['validation_4_chemistry']['fine_top1_heldout']:.4f}"),
            ("chemistry ECE", f"{s['validation_4_chemistry']['ece']:.4f}"),
            ("radar reproducibility",
             f"{s['validation_4_chemistry']['radar_reproducibility']:.4f}")]
    ax.text(0.0, 1.0, "validated across all 375 spectra", fontsize=9, weight="bold", va="top")
    for i, (k, v) in enumerate(rows):
        ax.text(0.0, 0.88 - i * 0.088, k, fontsize=7.4)
        ax.text(1.0, 0.88 - i * 0.088, v, fontsize=7.4, ha="right", weight="bold", color=GREEN,
                family="DejaVu Sans Mono")
    ax = fig.add_subplot(gs[0, 1])
    layers = ["raw\n676", "LSM\n50", "CSM\n49", "chemistry\n16"]
    vals = [0.592, 0.850, s["validation_2_csm"]["class_top1"],
            s["validation_4_chemistry"]["fine_top1_heldout"]]
    ax.plot(range(4), vals, "o-", color=GREEN, lw=2, ms=9)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.3f}", (i, v), textcoords="offset points", xytext=(0, 9), fontsize=8,
                    ha="center", weight="bold")
    ax.set_xticks(range(4)); ax.set_xticklabels(layers, fontsize=7.4)
    ax.set_ylim(0.5, 0.95); ax.set_ylabel("chemistry top-1, unseen molecule")
    ax.set_title("the abstraction stack", fontsize=9, loc="left")
    ax = fig.add_subplot(gs[0, 2])
    n = s["noise_robustness"]
    ax.bar(range(3), [n["mean_retrieval_top1"], n["mean_chemistry_top1"], n["mean_radar_cosine"]],
           color=[BLUE, GREEN, PURPLE], alpha=0.9, width=0.55)
    ax.set_xticks(range(3)); ax.set_xticklabels(["molecule\ntop-1", "chemistry\ntop-1",
                                                 "radar\ncosine"], fontsize=7.4)
    for i, v in enumerate([n["mean_retrieval_top1"], n["mean_chemistry_top1"],
                           n["mean_radar_cosine"]]):
        ax.text(i, v + 0.014, f"{v:.3f}", ha="center", fontsize=8, weight="bold")
    ax.set_ylim(0, 1.08); ax.set_title("mean under perturbation", fontsize=9, loc="left")
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
    ax.text(0.008, 0.03, f"{int((g.status=='PASS').sum())} of {len(g)} gates pass · atlas "
            f"{c.state['atlas_fingerprint'][:16]}… · deterministic · every score reconciles",
            fontsize=7.4, color=MUTED)
    fig.suptitle("Figure 16 · End-to-end engine summary", x=0.03, ha="left", fontsize=11.5,
                 weight="bold", color=INK)
    save(fig, "F16_summary")


def main():
    c = C(); print("[figures]")
    for fn in (f01, f02, f03, f04, f05, f06, f07, f08, f09, f10, f11, f12, f13, f14, f15, f16):
        fn(c)
    assert not list(F.glob("*.svg"))
    print(f"[figures] {len(list(F.glob('*.png')))} PNG at {DPI} dpi written to {F}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
