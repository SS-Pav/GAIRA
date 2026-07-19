"""Stage B0 step 5 — figures + PDF report (READ-ONLY).

Palette (CVD-validated): accept/primary #2563EB, reject/secondary #D97706.
Single-hue sequential ramps; no dual axes, no rainbow, no radar, no UMAP.
"""
from __future__ import annotations
import sys, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

REPO = Path("/Users/surajpg/projects/GAIRA")
OUT = REPO / "results/v5_rebuild/preprocessing_autoresearch"
TAB, FIG, CFG, ART = OUT / "tables", OUT / "figures", OUT / "configs", OUT / "artifacts"
FIG.mkdir(parents=True, exist_ok=True)
PDF_PATH = REPO / "GAIRA_V5_PREPROCESSING_AUTORESEARCH_REPORT.pdf"

OK, BAD = "#2563EB", "#D97706"
INK, MUTED, GRIDC = "#1f2328", "#6B7280", "#E5E7EB"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "axes.grid": True,
    "grid.color": GRIDC, "grid.linewidth": 0.5, "xtick.color": MUTED,
    "ytick.color": MUTED, "text.color": INK, "axes.labelcolor": INK,
    "axes.titlecolor": INK, "legend.frameon": False, "figure.facecolor": "white",
})
PAGE = (8.5, 11.0)
BASE = "BASE_asls_sg_l2"


def _wrap(t, n=104):
    out, line = [], ""
    for w in t.split():
        if len(line) + len(w) + 1 > n:
            out.append(line); line = w
        else:
            line = (line + " " + w).strip()
    if line: out.append(line)
    return out


def text_page(pdf, title, blocks, subtitle=None):
    fig = plt.figure(figsize=PAGE); y = 0.95
    fig.text(0.07, y, title, fontsize=17, fontweight="bold", color=INK); y -= 0.033
    if subtitle:
        fig.text(0.07, y, subtitle, fontsize=9.5, color=MUTED); y -= 0.03
    for kind, txt in blocks:
        if y < 0.06:
            pdf.savefig(fig); plt.close(fig); fig = plt.figure(figsize=PAGE); y = 0.95
        if kind == "h":
            y -= 0.014
            fig.text(0.07, y, txt, fontsize=11, fontweight="bold", color=INK); y -= 0.025
        elif kind == "b":
            for ln in _wrap(txt):
                fig.text(0.07, y, ln, fontsize=8.6, color=INK); y -= 0.0165
            y -= 0.008
        elif kind == "m":
            fig.text(0.075, y, txt, fontsize=7.6, color=INK, family="DejaVu Sans Mono"); y -= 0.0145
    pdf.savefig(fig); plt.close(fig)


def save(fig, name):
    fig.savefig(FIG / name, dpi=150, bbox_inches="tight")


def main():
    j = pd.read_csv(TAB / "search_results_judged.csv")
    outer = pd.read_csv(TAB / "outer_test_results.csv", index_col=0)
    bg = pd.read_csv(TAB / "background_variance_vs_retention.csv")
    fam = pd.read_csv(TAB / "family_results.csv")
    ba = pd.read_csv(TAB / "per_analyte_before_after.csv")
    dec = json.loads((TAB / "final_decision.json").read_text())
    man = json.loads((ART / "preprocessing_manifest.json").read_text())
    acc = json.loads((CFG / "acceptance_thresholds.json").read_text())
    base_row = j[j.cid == BASE].iloc[0]

    # ─────────── FIGURE 1 — the central trade-off ───────────
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axs[0]
    m = j.rejected
    ax.scatter(j[~m].rep_sers_replicate_cos, j[~m].cm_mrr, s=22, c=OK, label="passes integrity rules")
    ax.scatter(j[m].rep_sers_replicate_cos, j[m].cm_mrr, s=22, c=BAD, marker="x", label="rejected")
    ax.scatter([base_row.rep_sers_replicate_cos], [base_row.cm_mrr], s=90, facecolors="none",
               edgecolors=INK, linewidths=1.6, label="reference baseline")
    ax.set_xlabel("Ag-SERS replicate cosine (preservation →)")
    ax.set_ylabel("cross-modal MRR (inner validation)")
    ax.set_title("Every retrieval gain costs Ag-SERS replicate structure")
    ax.legend(fontsize=7, loc="upper right")
    ax = axs[1]
    ax.scatter(j[~m].pk_effect_vs_mismatched, j[~m].cm_mrr, s=22, c=OK)
    ax.scatter(j[m].pk_effect_vs_mismatched, j[m].cm_mrr, s=22, c=BAD, marker="x")
    ax.axvline(base_row.pk_effect_vs_mismatched, ls="--", c=INK, lw=1)
    ax.text(base_row.pk_effect_vs_mismatched, ax.get_ylim()[1], " baseline peak specificity",
            fontsize=7, color=INK, va="top")
    ax.set_xlabel("matched − mismatched peak correspondence (specificity →)")
    ax.set_ylabel("cross-modal MRR (inner validation)")
    ax.set_title("Retrieval gains bring NO gain in band specificity")
    fig.tight_layout(); save(fig, "fig1_central_tradeoff.png"); plt.close(fig)

    # ─────────── FIGURE 2 — Pareto + rejection reasons ───────────
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    pf = pd.read_csv(TAB / "pareto_front.csv") if (TAB / "pareto_front.csv").exists() else pd.DataFrame()
    ax = axs[0]
    ax.scatter(j.n_stages, j.cm_mrr, s=18, c=np.where(j.rejected, BAD, OK), alpha=0.8)
    if len(pf):
        ax.scatter(pf.n_stages, pf.cm_mrr, s=60, facecolors="none", edgecolors=INK, linewidths=1.2,
                   label="Pareto front (integrity-passing)")
        ax.legend(fontsize=7)
    ax.set_xlabel("pipeline stages (complexity)"); ax.set_ylabel("cross-modal MRR")
    ax.set_title("Complexity vs retrieval")
    ax = axs[1]
    from collections import Counter
    cnt = Counter(x for s in j[j.rejected].reject_reasons.dropna() for x in str(s).split("|"))
    ks = [k for k, _ in cnt.most_common()][::-1]; vs = [cnt[k] for k in ks]
    ax.barh(ks, vs, color=BAD)
    for i, v in enumerate(vs): ax.text(v + 0.5, i, str(v), fontsize=7.5, va="center", color=INK)
    ax.set_xlabel("candidates rejected"); ax.set_title("Why candidates were rejected")
    fig.tight_layout(); save(fig, "fig2_pareto_and_rejections.png"); plt.close(fig)

    # ─────────── FIGURE 3 — background: variance removed vs analyte retention ───────────
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axs[0]
    ax.plot(bg.variance_explained, bg.sers_analyte_1nn_heldout, "o-", c=OK, ms=6)
    for _, r in bg.iterrows():
        ax.annotate(r.background, (r.variance_explained, r.sers_analyte_1nn_heldout),
                    fontsize=6, xytext=(3, 3), textcoords="offset points", color=MUTED)
    ax.axhline(bg.sers_analyte_1nn_heldout.iloc[0], ls="--", c=INK, lw=1)
    ax.set_xlabel("fraction of Ag-SERS variance removed")
    ax.set_ylabel("held-out Ag-SERS analyte 1-NN")
    ax.set_title("Control 4/5: background removal does NOT\ndestroy analyte information")
    ax = axs[1]
    d = j[j.arm == "D_background"]
    ax.scatter(d.bg_variance_explained, d.cm_mrr, s=40,
               c=np.where(d.rejected, BAD, OK))
    for _, r in d.iterrows():
        ax.annotate(str(r.cfg_background), (r.bg_variance_explained, r.cm_mrr), fontsize=6,
                    xytext=(3, 3), textcoords="offset points", color=MUTED)
    ax.axhline(base_row.cm_mrr, ls="--", c=INK, lw=1)
    ax.set_xlabel("fraction of Ag-SERS variance removed"); ax.set_ylabel("cross-modal MRR")
    ax.set_title("Explicit background correction:\nmodest MRR gain, all integrity-rejected")
    fig.tight_layout(); save(fig, "fig3_background_control.png"); plt.close(fig)

    # ─────────── FIGURE 4 — outer test ───────────
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2))
    idx = list(outer.index)
    ci = np.array([json.loads(str(x).replace("'", '"')) if isinstance(x, str) else x
                   for x in outer["mrr_ci"]])
    ax = axs[0]
    err = np.abs(ci.T - outer["mrr"].values)
    cols = [OK if i.startswith("BASE") else BAD for i in idx]
    ax.bar(range(len(idx)), outer["mrr"], yerr=err, capsize=3, color=cols)
    ax.axhline(outer.loc[BASE, "mrr"], ls="--", c=INK, lw=1)
    ax.set_xticks(range(len(idx))); ax.set_xticklabels(idx, rotation=40, ha="right", fontsize=6.5)
    ax.set_ylabel("held-out cross-modal MRR"); ax.set_title("Outer test (used once)")
    ax = axs[1]
    ax.bar(range(len(idx)), outer["sers_replicate_cos"], color=cols)
    ax.axhline(acc["sers_replicate_min_frac_of_L2"] * outer.loc[BASE, "sers_replicate_cos"],
               ls="--", c=INK, lw=1)
    ax.text(0, acc["sers_replicate_min_frac_of_L2"] * outer.loc[BASE, "sers_replicate_cos"] + 0.02,
            "integrity floor", fontsize=7, color=INK)
    ax.set_xticks(range(len(idx))); ax.set_xticklabels(idx, rotation=40, ha="right", fontsize=6.5)
    ax.set_ylabel("Ag-SERS replicate cosine"); ax.set_title("Replicate preservation")
    ax = axs[2]
    ax.bar(range(len(idx)), outer["peak_effect"], color=cols)
    ax.axhline(outer.loc[BASE, "peak_effect"], ls="--", c=INK, lw=1)
    ax.set_xticks(range(len(idx))); ax.set_xticklabels(idx, rotation=40, ha="right", fontsize=6.5)
    ax.set_ylabel("matched − mismatched peak correspondence")
    ax.set_title("Band specificity (never improves)")
    fig.tight_layout(); save(fig, "fig4_outer_test.png"); plt.close(fig)

    # ─────────── FIGURE 5 — smoothing / normalization studies ───────────
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    b = j[j.arm == "B_smoothing"]
    ax = axs[0]
    ax.scatter(b.si_peak_retention, b.cm_mrr, s=30, c=np.where(b.rejected, BAD, OK))
    ax.axvline(0.90, ls="--", c=INK, lw=1)
    ax.text(0.90, ax.get_ylim()[1], " retention floor", fontsize=7, color=INK, va="top")
    ax.set_xlabel("peak retention after smoothing"); ax.set_ylabel("cross-modal MRR")
    ax.set_title("Arm B — smoothing: peak loss vs retrieval")
    a = j[j.arm == "A_baseline_norm"]
    ax = axs[1]
    order = ["none", "l2", "area", "robust", "p95", "max", "snv"]
    present = [o for o in order if (a.cfg_norm_r == o).any()]
    vals = [a[a.cfg_norm_r == o].cm_mrr.values for o in present]
    bp = ax.boxplot(vals, labels=present, patch_artist=True)
    for patch, o in zip(bp["boxes"], present):
        patch.set_facecolor(BAD if o == "snv" else OK); patch.set_alpha(0.75)
    ax.axhline(base_row.cm_mrr, ls="--", c=INK, lw=1)
    ax.set_ylabel("cross-modal MRR"); ax.set_title("Arm A — normalization (SNV = declared control)")
    fig.tight_layout(); save(fig, "fig5_smoothing_normalization.png"); plt.close(fig)

    # ─────────── FIGURE 6 — per-analyte and family ───────────
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.6))
    ax = axs[0]
    if "matched_cos_B_0_savgol6" in ba:
        ax.scatter(ba.matched_cos_baseline, ba.matched_cos_B_0_savgol6, s=26, c=OK)
        lim = [0, max(1.0, ba.matched_cos_baseline.max())]
        ax.plot(lim, lim, ls="--", c=INK, lw=1)
        ax.set_xlabel("matched cosine — baseline"); ax.set_ylabel("matched cosine — best-MRR candidate")
        ax.set_title("Matched-pair similarity FALLS for nearly every analyte\n"
                     "(ranks improve only because mismatched falls faster)")
    ax = axs[1]
    f = fam.sort_values("rank_improvement")
    ax.barh(f.family, f.rank_improvement, color=np.where(f.rank_improvement >= 0, OK, BAD))
    ax.set_xlabel("mean rank improvement (baseline − candidate)")
    ax.set_title("Family-stratified rank change")
    fig.tight_layout(); save(fig, "fig6_per_analyte_family.png"); plt.close(fig)

    # ─────────── FIGURE 7 — fold stability ───────────
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for cid in outer.index:
        fm = outer.loc[cid, "fold_mrr"]
        fm = json.loads(str(fm).replace("'", '"')) if isinstance(fm, str) else fm
        ax.plot(range(len(fm)), fm, "o-", ms=5, lw=1.4,
                label=cid, color=OK if cid.startswith("BASE") else BAD, alpha=0.85)
    ax.set_xlabel("outer fold"); ax.set_ylabel("cross-modal MRR")
    ax.set_title("Fold-to-fold stability (outer test)"); ax.legend(fontsize=6.5, ncol=2)
    fig.tight_layout(); save(fig, "fig7_fold_stability.png"); plt.close(fig)

    # ─────────── PDF ───────────
    with PdfPages(PDF_PATH) as pdf:
        # cover
        fig = plt.figure(figsize=PAGE)
        fig.text(0.5, 0.79, "Preprocessing AutoResearch", ha="center", fontsize=25,
                 fontweight="bold", color=INK)
        fig.text(0.5, 0.745, "Raman ↔ Ag-SERS comparability — GAIRA V5 Stage B0",
                 ha="center", fontsize=13, color=MUTED)
        fig.text(0.5, 0.695, "Can leakage-safe preprocessing and explicit Ag-SERS background\n"
                             "modelling recover cross-modal biochemical correspondence?",
                 ha="center", fontsize=11, style="italic", color=INK)
        ax = fig.add_axes([0.13, 0.40, 0.74, 0.25]); ax.axis("off")
        rows = [["Candidate pipelines", f"{man['n_candidates_evaluated']}"],
                ["Rejected by integrity rules", f"{man['n_rejected_by_integrity_rules']}"],
                ["Corpus", "479 spectra · 214 Raman / 265 Ag-SERS · 51 matched analytes"],
                ["Design", "5 outer × 4 inner, analyte-grouped; outer test used once"],
                ["Reference baseline", "ASLS + Savitzky–Golay + L2"],
                ["Outcome", f"{dec['outcome']} — {dec['headline']}"],
                ["Frozen pipeline", "none"]]
        t = ax.table(cellText=rows, colWidths=[0.34, 0.66], loc="center", cellLoc="left")
        t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 1.65)
        for (r, c), cell in t.get_celld().items():
            cell.set_edgecolor(GRIDC)
            if c == 0: cell.set_text_props(fontweight="bold")
        fig.text(0.5, 0.31, "PRINCIPAL FINDING", ha="center", fontsize=11, fontweight="bold", color=INK)
        fig.text(0.5, 0.175,
                 "Cross-modal retrieval CAN be pushed well past the pre-declared success thresholds\n"
                 "(MRR +0.097, top-1 +0.108 on held-out analytes) — but only by pipelines that strip the\n"
                 "broad shared Ag-SERS component. Those same pipelines collapse Ag-SERS replicate\n"
                 "agreement (0.95 → 0.62) and do not improve matched-vs-mismatched band specificity,\n"
                 "which falls. The gain is contrast geometry, not recovered shared chemistry.",
                 ha="center", fontsize=9.5, color=INK)
        fig.text(0.5, 0.06, "Read-only study · branch gaira-v5-rebuild-plan · nothing pushed",
                 ha="center", fontsize=8, color=MUTED)
        pdf.savefig(fig); plt.close(fig)

        text_page(pdf, "Contents", [("m", x) for x in [
            "1.   Executive summary", "2.   Study design and leakage safety",
            "3.   The central trade-off (Figure 1)", "4.   Pareto front and rejection reasons (Figure 2)",
            "5.   Ag-SERS background controls (Figure 3)", "6.   Outer test (Figure 4)",
            "7.   Smoothing and normalization arms (Figure 5)",
            "8.   Per-analyte and family results (Figure 6)", "9.   Fold stability (Figure 7)",
            "10.  Decision, thresholds and next action"]])

        text_page(pdf, "Executive summary",
                  [("h", "Question"),
                   ("b", "Stage A/B found weak Raman↔Ag-SERS correspondence, and the spectral audit showed the "
                         "Ag-SERS arm is dominated by a shared colloid background with the analyte surviving as "
                         "a ~5% residual. This study asks whether physically reasonable, leakage-safe "
                         "preprocessing and explicit background modelling can recover cross-modal "
                         "correspondence without destroying real analyte information."),
                   ("h", "What was done"),
                   ("b", f"{man['n_candidates_evaluated']} complete candidate pipelines across seven controlled arms "
                         "(baseline correction, normalization, smoothing, replicate aggregation, Ag-SERS "
                         "background models, derivatives, modality-specific pipelines, and rational "
                         "combinations), evaluated under 5×4 nested analyte-grouped cross-validation. The outer "
                         "test analytes — both modalities — were held out entirely and consumed exactly once. "
                         "Acceptance thresholds and rejection rules were frozen before the outer test was run."),
                   ("h", "Result"),
                   ("b", "1. Cross-modal retrieval CAN be improved past the pre-declared bar: the best candidate "
                         "reached held-out MRR 0.464 vs baseline 0.366 (+0.097, threshold +0.08) and top-1 0.284 "
                         "vs 0.176 (+0.108, threshold +0.05). A retrieval-only study would have declared success."),
                   ("b", "2. Every such gain is produced by stripping the broad shared Ag-SERS component. Ag-SERS "
                         "replicate cosine collapses from 0.946 to 0.620 (and to 0.579 for the best explicit "
                         "background model), far below the 0.90× integrity floor."),
                   ("b", "3. Band-level specificity never improves. Matched-minus-mismatched peak correspondence "
                         "is highest for the RAW baseline (+0.035) and falls for every 'improved' pipeline "
                         "(+0.014, +0.020). Of 67 candidates that improved MRR, ZERO also improved peak "
                         "specificity — a result independent of any threshold choice."),
                   ("b", "4. Matched-pair cosine falls for nearly every analyte; ranks improve only because "
                         "mismatched similarity falls faster once the common component is removed. This is a "
                         "contrast-geometry effect, not newly revealed shared chemistry."),
                   ("b", "5. Explicit background correction does NOT destroy analyte information — held-out "
                         "Ag-SERS analyte 1-NN stays at 0.877–0.916 (baseline 0.896) even when 84% of Ag-SERS "
                         "variance is removed, and improves to 0.916 for scaled-mean subtraction. The analyte "
                         "residual is real and retained; it simply does not align with the Raman spectra."),
                   ("b", "6. Arm D's own winner was 'no background correction', and mean-subtraction made "
                         "held-out cross-modal retrieval worse (MRR 0.351 vs 0.366)."),
                   ("h", "Conclusion"),
                   ("b", f"Outcome {dec['outcome']} — {dec['headline']}. No preprocessing pipeline is frozen. The "
                         "limitation is acquisition contrast in the Ag-colloid measurement, not the preprocessing "
                         "pipeline: the analyte signal is present and reproducible but does not carry "
                         "analyte-specific correspondence with powder Raman at usable strength."),
                   ], f"{man['n_candidates_evaluated']} pipelines · outcome {dec['outcome']} · nothing frozen")

        for name, cap in [
            ("fig1_central_tradeoff.png", "Figure 1 — The central trade-off. Left: cross-modal MRR versus "
             "Ag-SERS replicate preservation; every high-MRR candidate sits at low replicate cosine. "
             "Right: MRR versus matched-minus-mismatched peak specificity; gains in retrieval are not "
             "accompanied by gains in band specificity."),
            ("fig2_pareto_and_rejections.png", "Figure 2 — Complexity versus retrieval, and the reasons "
             "candidates were rejected. Ag-SERS replicate destruction dominates (58 of 66 rejections)."),
            ("fig3_background_control.png", "Figure 3 — Ag-SERS background controls. Left (Control 4/5): "
             "removing up to 84% of Ag-SERS variance leaves held-out analyte identification intact — the "
             "analyte residual is genuinely retained. Right: explicit background correction yields only a "
             "modest MRR gain and every variant fails the integrity rules."),
            ("fig4_outer_test.png", "Figure 4 — The one-time outer test. Retrieval (left) rises exactly where "
             "replicate preservation (middle) collapses, while band specificity (right) never improves."),
            ("fig5_smoothing_normalization.png", "Figure 5 — Arm B (smoothing) and Arm A (normalization). "
             "Retrieval rises as peak retention falls. SNV, included only as a declared negative control, "
             "scores well on retrieval — precisely the failure mode this study was designed to catch."),
            ("fig6_per_analyte_family.png", "Figure 6 — Per-analyte and family results. Matched-pair cosine "
             "falls for nearly every analyte under the best-MRR candidate; rank 'improvements' come from "
             "mismatched similarity falling faster."),
            ("fig7_fold_stability.png", "Figure 7 — Fold-to-fold stability of held-out cross-modal MRR."),
        ]:
            fig = plt.figure(figsize=PAGE)
            img = plt.imread(FIG / name)
            ax = fig.add_axes([0.05, 0.42, 0.90, 0.46]); ax.imshow(img); ax.axis("off")
            fig.text(0.07, 0.94, name.split("_", 1)[1].replace(".png", "").replace("_", " ").title(),
                     fontsize=13, fontweight="bold", color=INK)
            y = 0.36
            for ln in _wrap(cap, 100):
                fig.text(0.07, y, ln, fontsize=8.8, color=INK); y -= 0.017
            pdf.savefig(fig); plt.close(fig)

        # decision page
        gates = dec.get("gates", {})
        blocks = [("h", f"Outcome {dec['outcome']} — {dec['headline']}")]
        for r in dec["reasons"]:
            blocks.append(("b", r))
        blocks += [("h", "Pre-declared acceptance thresholds (frozen before the outer test)")]
        for k, v in acc.items():
            blocks.append(("m", f"{k:42s} {v}"))
        blocks += [("h", "Next authorized action"),
                   ("b", "No preprocessing pipeline is frozen and no representation work is authorized by this "
                         "study. The evidence points to acquisition contrast rather than preprocessing: the "
                         "recommended next step is targeted Ag-SERS re-acquisition in which the analyte, not the "
                         "colloid, dominates the spectrum (higher effective surface coverage, blank-colloid "
                         "difference measurement, or Au-SERS references), followed by re-running this same "
                         "frozen study design on the enlarged corpus."),
                   ("h", "Scope"),
                   ("b", "Read-only study. Stage A/B results, historical V1–V3.1 pipelines, the demo, and all "
                         "governing documents were left unmodified. The outer test folds were consumed exactly "
                         "once, recorded in configs/study_manifest.json.")]
        text_page(pdf, "Decision", blocks)

        d = pdf.infodict()
        d["Title"] = "GAIRA V5 Stage B0 — Preprocessing AutoResearch"
        d["Subject"] = "Raman/Ag-SERS comparability: preprocessing and background-correction study"
    print(f"figures + PDF written: {PDF_PATH} ({PDF_PATH.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
