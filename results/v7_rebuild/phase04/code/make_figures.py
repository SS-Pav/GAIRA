#!/usr/bin/env python3
"""GAIRA V7 — Phase 04 figures (PNG, 200 dpi). Deterministic; seeds fixed."""
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
from gaira.v7.io import PhaseOutputs, frozen_root      # noqa: E402

OUT = PhaseOutputs("04")
T, A, V, F = OUT.tables, OUT.artifacts, OUT.validation, OUT.figures
FROZEN = frozen_root()
INK, MUTED, LINE = "#1a1a1a", "#6b7280", "#9ca3af"
BLUE, GREEN, AMBER, RED, GREY = "#2563eb", "#15803d", "#b45309", "#b91c1c", "#4b5563"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
                     "figure.facecolor": "white", "savefig.facecolor": "white",
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.18})
LEVELS = ["spectrum", "LSM", "CSM", "theme", "BSV", "geometry"]


def save(fig, name):
    F.mkdir(parents=True, exist_ok=True)
    fig.savefig(F / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"  {name}")


class Ctx:
    def __init__(self):
        z = np.load(A / "inference_v1.npz", allow_pickle=True)
        self.A_lsm, self.A_csm, self.T = z["A_lsm"], z["A_csm"], z["T"]
        self.BSV, self.COORD = z["BSV"], z["COORD"]
        self.conf, self.ood = z["confidence"], z["ood"]
        self.y = np.array([str(s) for s in z["y"]])
        self.cls = np.array([str(s) for s in z["cls"]])
        self.folds, self.X, self.grid = z["folds"], z["X"], z["grid"]
        self.state = json.loads((OUT.root / "PHASE_STATE.json").read_text())
        self.levels = pd.read_csv(T / "hierarchy_retrieval_v1.csv")
        self.flow = pd.read_csv(T / "information_flow_v1.csv")
        self.proj = pd.read_csv(T / "projection_benchmark_v1.csv")
        self.agg = pd.read_csv(T / "aggregation_benchmark_v1.csv")
        self.tm = pd.read_csv(T / "theme_mode_benchmark_v1.csv")
        self.bsvb = pd.read_csv(T / "bsv_variant_benchmark_v1.csv")
        self.geo = pd.read_csv(T / "geometry_extension_benchmark_v1.csv")
        self.leak = pd.read_csv(V / "leakage_control_v1.csv")
        self.perclass = pd.read_csv(V / "per_class_retrieval_v1.csv")
        self.noise = pd.read_csv(V / "bsv_noise_robustness_v1.csv")
        self.recov = pd.read_csv(V / "activation_recovery_v1.csv")
        self.diag = json.loads((A / "diagnostics_v1.json").read_text())
        self.probes = json.loads((A / "ood_probes_v1.json").read_text())
        self.cfg = json.loads((A / "engine_config_v1.json").read_text())
        treg = json.loads((FROZEN / "phase03/artifacts/theme_registry_v1.json").read_text())
        self.theme_names = [t["name"] for t in treg["themes"] if t["status"] == "accepted"]


def f01_pipeline(c):
    fig, ax = plt.subplots(figsize=(10.6, 3.3))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3.3); ax.set_axis_off()
    steps = [("new\nspectrum", GREY), ("QC", GREY), ("LSM\nactivations", BLUE),
             ("CSM\nactivations", BLUE), ("theme\nactivations", GREEN),
             ("BSV", GREEN), ("latent\ngeometry", AMBER), ("SpectrumState\n+ provenance", INK)]
    for k, (lab, col) in enumerate(steps):
        x = 0.1 + k * 1.235
        ax.add_patch(FancyBboxPatch((x, 1.55), 1.05, 1.0, boxstyle="round,pad=0.05",
                                    fc="white", ec=col, lw=1.3))
        ax.text(x + 0.525, 2.05, lab, ha="center", va="center", fontsize=7, color=INK)
        if k:
            ax.add_patch(FancyArrowPatch((x - 0.16, 2.05), (x - 0.02, 2.05),
                                         arrowstyle="-|>", mutation_scale=8, color=MUTED,
                                         lw=0.9))
    ax.text(0.1, 1.15, "projection only — no fitting, no RNG, batch-independent, "
                       "bit-identical on re-run", fontsize=8, color=RED)
    cfg = c.cfg["config"]
    ax.text(0.1, 0.72, "  ·  ".join(f"{k.replace('_method','').replace('_',' ')}: {v}"
                                    for k, v in cfg.items() if k != "knn"),
            fontsize=7.2, color=MUTED)
    ax.text(0.1, 0.3, "every output traces back: theme → CSM → LSM → canonical molecule → "
                      "source spectrum", fontsize=7.6, color=INK)
    ax.set_title("The canonical GAIRA inference pathway", fontsize=10.5, loc="left", color=INK)
    save(fig, "fig01_inference_pipeline")


def f02_projection(c):
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.2), gridspec_kw={"wspace": 0.34})
    p = c.proj
    sel = c.cfg["config"]["projection_method"]
    cols = [GREEN if m == sel else GREY for m in p.method]
    axes[0].barh(np.arange(len(p)), p.mean_ev, color=cols, height=0.6)
    axes[0].set_yticks(np.arange(len(p))); axes[0].set_yticklabels(p.method, fontsize=7)
    axes[0].set_xlabel("reconstruction EV")
    axes[0].set_title("Reconstruction", fontsize=9, loc="left", color=INK)
    axes[1].scatter(p.replicate_consistency, p.noise_stability, s=60, color=cols,
                    edgecolor=INK, linewidth=0.5)
    for r in p.itertuples():
        axes[1].annotate(r.method, (r.replicate_consistency, r.noise_stability), fontsize=6.2,
                         xytext=(4, 3), textcoords="offset points", color=MUTED)
    axes[1].set_xlabel("replicate consistency"); axes[1].set_ylabel("noise stability")
    axes[1].set_title("The two properties that decide it", fontsize=9, loc="left", color=INK)
    axes[2].barh(np.arange(len(p)), p.mean_active_components, color=cols, height=0.6)
    axes[2].set_yticks(np.arange(len(p))); axes[2].set_yticklabels([])
    axes[2].set_xlabel("active components per spectrum")
    axes[2].set_title("Sparsity", fontsize=9, loc="left", color=INK)
    for a in axes:
        for s in ("top", "right"): a.spines[s].set_visible(False)
    fig.suptitle(f"Projection estimators. Selected: {sel} — chosen on replicate consistency "
                 f"× noise stability among estimators with zero negative mass, not on "
                 f"reconstruction alone.",
                 fontsize=9.5, x=0.005, ha="left", y=1.06, color=INK)
    save(fig, "fig02_projection_benchmark")


def f03_aggregation_theme(c):
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.2), gridspec_kw={"wspace": 0.42})
    sel_a = c.cfg["config"]["aggregation_method"]
    ax = axes[0]
    ax.barh(np.arange(len(c.agg)), c.agg.mean_ev,
            color=[GREEN if m == sel_a else GREY for m in c.agg.aggregation], height=0.6)
    ax.set_yticks(np.arange(len(c.agg)))
    ax.set_yticklabels(c.agg.aggregation, fontsize=6.5)
    ax.set_xlabel("reconstruction EV")
    ax.set_title("B — LSM → CSM aggregation", fontsize=9, loc="left", color=INK)
    ax = axes[1]
    sel_t = c.cfg["config"]["theme_mode"]
    cols = [RED if not a else (GREEN if m == sel_t else GREY)
            for m, a in zip(c.tm.theme_mode, c.tm.admissible)]
    ax.barh(np.arange(len(c.tm)), c.tm.zero_evidence_leakage, color=cols, height=0.6)
    ax.set_yticks(np.arange(len(c.tm))); ax.set_yticklabels(c.tm.theme_mode, fontsize=6.5)
    ax.set_xlabel("zero-evidence leakage")
    ax.set_title("C — theme modes (red = inadmissible)", fontsize=9, loc="left", color=INK)
    ax = axes[2]
    sel_b = c.cfg["config"]["bsv_variant"]
    ax.barh(np.arange(len(c.bsvb)), c.bsvb.separation_ratio,
            color=[GREEN if v == sel_b else GREY for v in c.bsvb.variant], height=0.6)
    ax.set_yticks(np.arange(len(c.bsvb))); ax.set_yticklabels(c.bsvb.variant, fontsize=6.5)
    ax.set_xlabel("between/within molecule separation")
    ax.set_title("D — BSV definition", fontsize=9, loc="left", color=INK)
    for a in axes:
        for s in ("top", "right"): a.spines[s].set_visible(False)
    fig.suptitle("Aggregation, theme mode and BSV definition. The softmax mode scored best on "
                 "replicate consistency and was rejected:\nit activates themes for which the "
                 "spectrum has no CSM evidence at all.",
                 fontsize=9.5, x=0.005, ha="left", y=1.10, color=INK)
    save(fig, "fig03_aggregation_theme_bsv")


def f04_geometry_extension(c):
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.2), gridspec_kw={"wspace": 0.3})
    sel = c.cfg["config"]["geometry_extension"]
    cols = [GREEN if m == sel else GREY for m in c.geo.method]
    axes[0].barh(np.arange(len(c.geo)), c.geo.neighbour_preservation, color=cols, height=0.6)
    axes[0].set_yticks(np.arange(len(c.geo)))
    axes[0].set_yticklabels(c.geo.method, fontsize=7)
    axes[0].set_xlabel("leave-one-out neighbour preservation")
    axes[0].set_title("Which extension keeps neighbours", fontsize=9, loc="left", color=INK)
    axes[1].barh(np.arange(len(c.geo)), c.geo.relative_error, color=cols, height=0.6)
    axes[1].set_yticks(np.arange(len(c.geo))); axes[1].set_yticklabels([])
    axes[1].set_xlabel("relative coordinate error")
    axes[1].set_title("...and which places them accurately", fontsize=9, loc="left", color=INK)
    for a in axes:
        for s in ("top", "right"): a.spines[s].set_visible(False)
    fig.suptitle(f"Out-of-sample manifold extension. Selected: {sel}. Nyström is the "
                 f"principled choice for a diffusion map and the worst here — the frozen "
                 f"coordinates\nare too few and too spread for a kernel average to localise.",
                 fontsize=9.5, x=0.005, ha="left", y=1.08, color=INK)
    save(fig, "fig04_geometry_extension")


def f05_information_flow(c):
    fl = c.flow.set_index("level").loc[LEVELS].reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.4), gridspec_kw={"wspace": 0.34})
    x = np.arange(len(fl))
    ax = axes[0]
    ax.plot(x, fl.molecule_top1, "o-", color=BLUE, ms=6, label="molecule (split A)")
    ax.plot(x, fl.class_top1, "s-", color=GREEN, ms=6, label="chemistry class (split B)")
    ax.set_xticks(x); ax.set_xticklabels(fl.level, rotation=25, fontsize=7.5)
    ax.set_ylabel("held-out top-1"); ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("Retrieval through the hierarchy", fontsize=9, loc="left", color=INK)
    ax = axes[1]
    ax.plot(x, fl.replicate_consistency, "o-", color=AMBER, ms=6)
    ax.set_xticks(x); ax.set_xticklabels(fl.level, rotation=25, fontsize=7.5)
    ax.set_ylabel("replicate consistency")
    ax.set_title("Stability rises as identity falls", fontsize=9, loc="left", color=INK)
    ax = axes[2]
    ax.semilogy(x, fl.dim, "o-", color=GREY, ms=6, label="nominal dimension")
    ax.semilogy(x, fl.effective_rank, "s-", color=RED, ms=6, label="effective rank")
    ax.set_xticks(x); ax.set_xticklabels(fl.level, rotation=25, fontsize=7.5)
    ax.set_ylabel("dimension")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("Nominal vs effective dimensionality", fontsize=9, loc="left", color=INK)
    for a in axes:
        for s in ("top", "right"): a.spines[s].set_visible(False)
    fig.suptitle("Information flow. The LSM/CSM layers RAISE chemistry generalisation to "
                 "unseen molecules (0.608 → 0.855);\nthe theme layer trades both retrieval "
                 "axes for the highest replicate consistency in the stack.",
                 fontsize=9.5, x=0.005, ha="left", y=1.10, color=INK)
    save(fig, "fig05_information_flow")


def f06_retrieval(c):
    lv = c.levels
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.4), gridspec_kw={"wspace": 0.28})
    x = np.arange(len(lv))
    w = 0.26
    ax = axes[0]
    for i, (k, col) in enumerate((("A_molecule_top1", BLUE), ("A_molecule_top3", GREEN),
                                  ("A_molecule_top5", AMBER))):
        ax.bar(x + (i - 1) * w, lv[k], width=w, color=col, label=k.split("_")[-1])
    ax.set_xticks(x); ax.set_xticklabels(lv.level, rotation=22, fontsize=7)
    ax.set_ylabel("accuracy"); ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("Split A — molecule identity (molecule present in the reference set)",
                 fontsize=8.5, loc="left", color=INK)
    ax = axes[1]
    for i, (k, col) in enumerate((("B_class_top1", BLUE), ("B_class_top3", GREEN))):
        ax.bar(x + (i - 0.5) * w, lv[k], width=w, color=col, label=k.split("_")[-1])
    ax.set_xticks(x); ax.set_xticklabels(lv.level, rotation=22, fontsize=7)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("Split B — chemistry of an UNSEEN molecule\n"
                 "(molecule top-k is undefined here and is not shown)",
                 fontsize=8.5, loc="left", color=INK)
    for a in axes:
        for s in ("top", "right"): a.spines[s].set_visible(False)
    save(fig, "fig06_retrieval_by_level")


def f07_leakage(c):
    g = c.leak.groupby("dictionary")[["top1", "top3", "top5", "mrr"]].mean()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.3), gridspec_kw={"wspace": 0.3})
    ax = axes[0]
    x = np.arange(4)
    ax.bar(x - 0.2, g.loc["frozen_dictionary"], width=0.4, color=RED,
           label="frozen dictionary (has seen every molecule)")
    ax.bar(x + 0.2, g.loc["fold_honest_dictionary"], width=0.4, color=GREEN,
           label="refit without the held-out fold")
    ax.set_xticks(x); ax.set_xticklabels(["top1", "top3", "top5", "MRR"])
    ax.set_ylabel("retrieval"); ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=6.8)
    ax.set_title("Dictionary-level leakage, measured", fontsize=9, loc="left", color=INK)
    ax = axes[1]
    piv = c.leak.pivot(index="fold", columns="dictionary", values="top1")
    for col, colour in (("frozen_dictionary", RED), ("fold_honest_dictionary", GREEN)):
        ax.plot(piv.index, piv[col], "o-", color=colour, ms=5, label=col)
    ax.set_xlabel("fold"); ax.set_ylabel("top-1"); ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=6.8)
    ax.set_title("Per fold", fontsize=9, loc="left", color=INK)
    for a in axes:
        for s in ("top", "right"): a.spines[s].set_visible(False)
    gap = float(g.loc["frozen_dictionary", "top1"] - g.loc["fold_honest_dictionary", "top1"])
    fig.suptitle(f"The frozen dictionary was fitted on every molecule, so grouping the folds "
                 f"cannot remove dictionary-level leakage.\nRefitting per fold measures it: "
                 f"+{gap:.3f} top-1 inflation in every in-sample number this project has "
                 f"produced.",
                 fontsize=9.5, x=0.005, ha="left", y=1.10, color=INK)
    save(fig, "fig07_leakage_control")


def f08_ood(c):
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.3), gridspec_kw={"wspace": 0.3})
    real = c.probes["real_sers"]
    syn = c.probes["synthetic_band_shift"]
    ax = axes[0]
    ax.bar([0, 1], [syn["auroc"], real["auroc"]], color=[GREEN, RED], width=0.55)
    ax.axhline(0.5, color=INK, ls="--", lw=1.0)
    ax.text(1.35, 0.51, "chance", fontsize=7, color=INK)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["synthetic\nband shift", "REAL Ag-SERS"], fontsize=8)
    ax.set_ylabel("OOD AUROC"); ax.set_ylim(0, 1)
    ax.set_title("The probe that matters is the one that fails", fontsize=9, loc="left",
                 color=INK)
    ax = axes[1]
    ax.bar([0, 1, 2], [real["mean_in"], real["mean_out"], syn["mean_out"]],
           color=[BLUE, RED, GREY], width=0.55)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["in-domain\nRaman", "Ag-SERS", "band-shifted"], fontsize=8)
    ax.set_ylabel("residual OOD score")
    ax.set_title("SERS is BETTER explained than the references", fontsize=9, loc="left",
                 color=INK)
    for a in axes:
        for s in ("top", "right"): a.spines[s].set_visible(False)
    fig.suptitle("Out-of-domain detection. A non-negative dictionary of Raman motifs "
                 "reconstructs Ag-SERS of the same metabolites\ncomfortably, so the atlas "
                 "cannot tell modality. Reported as a failed gate, not compensated.",
                 fontsize=9.5, x=0.005, ha="left", y=1.10, color=INK)
    save(fig, "fig08_ood_detection")


def f09_bsv(c):
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.2), gridspec_kw={"wspace": 0.34})
    ax = axes[0]
    ax.plot(c.noise.sigma, c.noise.mean_cosine, "o-", color=BLUE, ms=6)
    ax.fill_between(c.noise.sigma, c.noise.min_cosine, c.noise.mean_cosine, color=BLUE,
                    alpha=0.2)
    ax.set_xlabel("noise σ (fraction of peak)"); ax.set_ylabel("BSV cosine to clean")
    ax.set_ylim(0.9, 1.005)
    ax.set_title("Noise robustness", fontsize=9, loc="left", color=INK)
    ax = axes[1]
    rep = c.diag["bsv_reproducibility"]
    ax.bar([0, 1], [rep["within_molecule_cosine"], rep["between_molecule_cosine"]],
           color=[GREEN, GREY], width=0.55)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["within\nmolecule", "between\nmolecules"],
                                              fontsize=8)
    ax.set_ylabel("mean cosine"); ax.set_ylim(0, 1.05)
    ax.set_title(f"Replicates cohere (separation ratio {rep['separation_ratio']:.1f})",
                 fontsize=9, loc="left", color=INK)
    ax = axes[2]
    er = c.state["bsv_effective_rank"]
    ax.bar([0, 1], [er["nominal_K"], er["participation_ratio"]], color=[GREY, RED], width=0.55)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["nominal K", "effective rank"], fontsize=8)
    ax.set_title(f"{er['participation_ratio']:.2f} of {er['nominal_K']} axes are independent",
                 fontsize=9, loc="left", color=INK)
    for a in axes:
        for s in ("top", "right"): a.spines[s].set_visible(False)
    fig.suptitle("The Biochemical State Vector — absolute, non-negative, four axes. "
                 "Effective rank is reported alongside K (risk R-12).",
                 fontsize=9.5, x=0.005, ha="left", y=1.08, color=INK)
    save(fig, "fig09_bsv_behaviour")


def f10_per_class(c):
    d = c.perclass.sort_values("top1")
    fig, ax = plt.subplots(figsize=(8.4, 0.28 * len(d) + 1.8))
    ax.barh(np.arange(len(d)), d.top1, color=BLUE, height=0.62)
    ax.barh(np.arange(len(d)), d.top5 - d.top1, left=d.top1, color=BLUE, alpha=0.35,
            height=0.62)
    for i, r in enumerate(d.itertuples()):
        ax.text(1.01, i, f"n={int(r.n)}", fontsize=6.2, va="center", color=MUTED)
    ax.set_yticks(np.arange(len(d)))
    ax.set_yticklabels(d.chemistry_class, fontsize=7)
    ax.set_xlabel("held-out retrieval (solid = top-1, pale = up to top-5)")
    ax.set_xlim(0, 1.12)
    ax.set_title("Per chemistry class, split B (unseen molecules) at the CSM level",
                 fontsize=9.5, loc="left", color=INK)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    save(fig, "fig10_per_class_retrieval")


def f11_geometry_map(c):
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.4), gridspec_kw={"wspace": 0.22})
    fams = sorted(set(c.cls))
    cm = plt.get_cmap("tab20")
    col = {f: cm(i % 20) for i, f in enumerate(fams)}
    ax = axes[0]
    for i in range(len(c.cls)):
        ax.scatter(c.COORD[i, 0], c.COORD[i, 1], s=22, color=col[c.cls[i]],
                   edgecolor="white", linewidth=0.3)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("375 spectra projected into the frozen manifold", fontsize=9, loc="left",
                 color=INK)
    ax = axes[1]
    sc = ax.scatter(c.COORD[:, 0], c.COORD[:, 1], s=22, c=c.conf, cmap="viridis",
                    vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(sc, ax=ax, shrink=0.8, label="confidence")
    ax.set_title("Engine confidence", fontsize=9, loc="left", color=INK)
    ax = axes[2]
    sc = ax.scatter(c.COORD[:, 0], c.COORD[:, 1], s=22, c=np.clip(c.ood, 0, 4),
                    cmap="magma_r")
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(sc, ax=ax, shrink=0.8, label="OOD score")
    ax.set_title("Residual OOD score", fontsize=9, loc="left", color=INK)
    fig.suptitle(f"Geometry projection. Neighbourhood purity "
                 f"{c.diag['geometry_neighbourhood_purity']:.3f} at "
                 f"{c.diag['geometry_lift_over_chance']:.1f}× chance.",
                 fontsize=9.5, x=0.005, ha="left", y=1.06, color=INK)
    save(fig, "fig11_geometry_projection")


def f12_calibration(c):
    cal = c.diag["calibration"]
    cur = pd.DataFrame(cal["curve"])
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.3), gridspec_kw={"wspace": 0.3})
    ax = axes[0]
    ax.plot([0, 1], [0, 1], ls="--", color=LINE)
    ax.plot(cur.mean_confidence, cur.accuracy, "o-", color=RED, ms=6)
    for r in cur.itertuples():
        ax.annotate(f"n={int(r.n)}", (r.mean_confidence, r.accuracy), fontsize=6,
                    xytext=(4, -8), textcoords="offset points", color=MUTED)
    ax.set_xlabel("engine confidence"); ax.set_ylabel("held-out top-1 accuracy")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title(f"Reliability — ECE {cal['ece']:.3f}", fontsize=9, loc="left", color=INK)
    ax = axes[1]
    r = c.recov.set_index("level")
    lv = [l for l in ("lsm", "csm", "theme", "bsv") if l in r.index]
    ax.bar(np.arange(len(lv)), [r.loc[l, "mean_cosine"] for l in lv], color=BLUE, width=0.5)
    ax.bar(np.arange(len(lv)), [r.loc[l, "top3_overlap"] for l in lv], color=GREEN, width=0.25)
    ax.set_xticks(np.arange(len(lv))); ax.set_xticklabels(lv)
    ax.set_ylabel("recovery"); ax.set_ylim(0, 1.05)
    ax.legend(handles=[Line2D([], [], color=BLUE, lw=6, label="cosine to the molecule's "
                                                              "reference profile"),
                       Line2D([], [], color=GREEN, lw=6, label="top-3 component overlap")],
              frameon=False, fontsize=6.5)
    ax.set_title("Activation recovery on held-out spectra", fontsize=9, loc="left", color=INK)
    for a in axes:
        for s in ("top", "right"): a.spines[s].set_visible(False)
    fig.suptitle("Confidence is poorly calibrated — it is monotone but overconfident, and "
                 "that is a reported failure.",
                 fontsize=9.5, x=0.005, ha="left", y=1.08, color=INK)
    save(fig, "fig12_calibration_and_recovery")


def f13_worked_example(c):
    ex = json.loads((A / "worked_example_v1.json").read_text())
    st, exp = ex["example_state"], ex["example_explanation"]
    fig = plt.figure(figsize=(10.6, 4.6))
    gs = fig.add_gridspec(1, 3, wspace=0.3, width_ratios=[1.4, 1, 1])
    ax = fig.add_subplot(gs[0])
    i = 7
    ax.plot(c.grid, c.X[i] / c.X[i].max(), lw=1.1, color=INK, label="spectrum")
    ax.set_xlabel("Raman shift (cm⁻¹)"); ax.set_yticks([])
    ax.legend(frameon=False, fontsize=7)
    ax.set_title(f"{st['spectrum_id']} · tier {st['confidence_tier']} · "
                 f"confidence {st['confidence']:.2f}", fontsize=8.5, loc="left", color=INK)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax = fig.add_subplot(gs[1])
    t = np.array(st["theme_activations"])
    ax.barh(np.arange(len(t)), t, color=GREEN, height=0.6)
    ax.set_yticks(np.arange(len(t)))
    ax.set_yticklabels([n[:24] for n in c.theme_names], fontsize=6.5)
    ax.set_xlabel("theme activation")
    ax.set_title("Themes", fontsize=8.5, loc="left", color=INK)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax = fig.add_subplot(gs[2]); ax.axis("off")
    lines = [f"top theme: {exp['theme_name'][:30]}", ""]
    for s in exp["supporting_csms"][:2]:
        lines.append(f"  {s['csm_id']}  contribution {s['contribution']:.3f}")
        lines.append(f"    LSMs: {', '.join(s['lsms'][:2])}")
        lines.append(f"    molecules: {', '.join(s['canonical_molecules'][:3])}")
        lines.append("")
    lines += [f"nearest molecules:"] + [f"  {m['canonical_id']} ({m['share']:.2f})"
                                        for m in st["nearest_molecules"][:3]]
    lines += ["", f"OOD {st['ood']['ood_score']:.2f} · EV "
                  f"{st['residual']['explained_variance']:.2f}"]
    ax.text(0, 1, "\n".join(lines), fontsize=7, va="top", family="DejaVu Sans", color=INK)
    ax.set_title("The explanation chain", fontsize=8.5, loc="left", color=INK)
    fig.suptitle("A worked example. Every activation resolves to CSMs, LSMs and named "
                 "molecules — nothing opaque.",
                 fontsize=9.5, x=0.005, ha="left", y=1.02, color=INK)
    save(fig, "fig13_worked_example")


def f14_architecture(c):
    fig, ax = plt.subplots(figsize=(10.4, 3.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3.6); ax.set_axis_off()
    steps = [("00\nbenchmark", GREEN), ("01\n50 LSMs", GREEN), ("02\n49 CSMs", GREEN),
             ("02.5\ngeometry", GREEN), ("03\n4 themes", GREEN),
             ("04\nengine + BSV", BLUE), ("05\nRaman\nvalidation", MUTED),
             ("06\nSERS\nobservation", MUTED)]
    for k, (lab, col) in enumerate(steps):
        x = 0.15 + k * 1.23
        ax.add_patch(FancyBboxPatch((x, 1.5), 1.05, 1.05, boxstyle="round,pad=0.05",
                                    fc="#eff6ff" if col == BLUE else
                                    ("#f0fdf4" if col == GREEN else "white"),
                                    ec=col, lw=1.7 if col == BLUE else 1.1))
        ax.text(x + 0.525, 2.02, lab, ha="center", va="center", fontsize=7, color=INK)
        if k:
            ax.add_patch(FancyArrowPatch((x - 0.15, 2.02), (x - 0.02, 2.02),
                                         arrowstyle="-|>", mutation_scale=8, color=MUTED,
                                         lw=0.9))
    st = c.state
    ax.text(0.15, 1.05,
            f"Engine frozen. BSV dimension {st['bsv_dimension']} "
            f"(effective rank {st['bsv_effective_rank']['participation_ratio']:.2f}).  "
            f"Held-out: molecule top-1 {st['retrieval']['L3_csm']['A_molecule_top1']:.3f}, "
            f"class top-1 {st['retrieval']['L3_csm']['B_class_top1']:.3f} at the CSM level.",
            fontsize=7.6, color=INK)
    ax.text(0.15, 0.55,
            f"Open: OOD cannot separate real Ag-SERS (AUROC {st['ood_auroc']:.2f}); "
            f"confidence poorly calibrated (ECE {st['calibration_ece']:.2f}); "
            f"dictionary leakage +{st['leakage_inflation_top1']:.3f} top-1.",
            fontsize=7.6, color=RED)
    ax.set_title("GAIRA V7 architecture after Phase 04", fontsize=10.5, loc="left", color=INK)
    save(fig, "fig14_architecture")


def main():
    c = Ctx()
    print("[phase04] figures")
    for fn in (f01_pipeline, f02_projection, f03_aggregation_theme, f04_geometry_extension,
               f05_information_flow, f06_retrieval, f07_leakage, f08_ood, f09_bsv,
               f10_per_class, f11_geometry_map, f12_calibration, f13_worked_example,
               f14_architecture):
        fn(c)


if __name__ == "__main__":
    main()
