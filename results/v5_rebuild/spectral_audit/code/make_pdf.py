"""Build GAIRA_V5_MATCHED_ANALYTE_SPECTRAL_AUDIT.pdf (READ-ONLY audit deliverable).

Cover, TOC, executive summary, decisive-evidence figures, global statistics,
band-level + family analysis, one page per matched analyte, and conclusions.

Palette (CVD-validated, ΔE 32.3 protan): Raman #2563EB, Ag-SERS #D97706.
Sequential ramps single-hue; diverging blue–grey–amber for signed quantities.
No dual axes, no rainbow, no radar.
"""
from __future__ import annotations
import sys, json, pickle, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap

REPO = Path("/Users/surajpg/projects/GAIRA")
AUD = REPO / "results/v5_rebuild/spectral_audit"
TAB, FIG = AUD / "tables", AUD / "figures"
PDF_PATH = REPO / "GAIRA_V5_MATCHED_ANALYTE_SPECTRAL_AUDIT.pdf"

RAMAN, SERS = "#2563EB", "#D97706"
INK, MUTED, GRID = "#1f2328", "#6B7280", "#E5E7EB"
SEQ = LinearSegmentedColormap.from_list("seq", ["#F8FAFC", "#2563EB"])
DIV = LinearSegmentedColormap.from_list("div", ["#2563EB", "#F3F4F6", "#D97706"])

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8,
    "axes.titlesize": 9, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.labelcolor": INK, "axes.titlecolor": INK, "legend.frameon": False,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})
PAGE = (8.5, 11.0)


def _load():
    with open(AUD / "code" / "_audit_store.pkl", "rb") as f:
        blob = pickle.load(f)
    extra = {}
    for nm, fn in (("nulls", "null_control_summary.json"), ("sens", "preprocessing_sensitivity_summary.json"),
                   ("degen", "sers_degeneracy_test.json"), ("bg", "background_removal_test.json"),
                   ("glob", "global_statistics.json")):
        p = TAB / fn
        extra[nm] = json.loads(p.read_text()) if p.exists() else {}
    extra["nullsdf"] = pd.read_csv(TAB / "peak_correspondence_null_controls.csv")
    extra["bands"] = pd.read_csv(TAB / "band_level_comparison.csv")
    return blob, extra


def text_page(pdf, title, blocks, subtitle=None):
    fig = plt.figure(figsize=PAGE); fig.patch.set_facecolor("white")
    y = 0.95
    fig.text(0.07, y, title, fontsize=17, fontweight="bold", color=INK); y -= 0.032
    if subtitle:
        fig.text(0.07, y, subtitle, fontsize=9.5, color=MUTED); y -= 0.028
    fig.text(0.07, y, "", fontsize=1); y -= 0.012
    for kind, txt in blocks:
        if y < 0.06:
            pdf.savefig(fig); plt.close(fig)
            fig = plt.figure(figsize=PAGE); y = 0.95
        if kind == "h":
            y -= 0.012
            fig.text(0.07, y, txt, fontsize=11, fontweight="bold", color=INK); y -= 0.024
        elif kind == "b":
            wrapped = _wrap(txt, 108)
            for line in wrapped:
                fig.text(0.07, y, line, fontsize=8.6, color=INK); y -= 0.0165
            y -= 0.008
        elif kind == "m":
            fig.text(0.075, y, txt, fontsize=8.0, color=INK, family="DejaVu Sans Mono"); y -= 0.0155
    pdf.savefig(fig); plt.close(fig)


def _wrap(t, n):
    out, line = [], ""
    for w in t.split():
        if len(line) + len(w) + 1 > n:
            out.append(line); line = w
        else:
            line = (line + " " + w).strip()
    if line: out.append(line)
    return out


# ───────────────────────── cover / toc / summary ─────────────────────────
def cover(pdf, df, preproc):
    fig = plt.figure(figsize=PAGE)
    fig.text(0.5, 0.80, "Matched-Analyte Spectral Audit", ha="center", fontsize=25,
             fontweight="bold", color=INK)
    fig.text(0.5, 0.755, "Raman vs Ag-SERS — GAIRA V5 grounding corpus", ha="center",
             fontsize=13.5, color=MUTED)
    fig.text(0.5, 0.70, "Do Raman and Ag-SERS observe the same vibrational modes\n"
                        "for the same analyte, or fundamentally different spectra?",
             ha="center", fontsize=11, color=INK, style="italic")
    ax = fig.add_axes([0.12, 0.36, 0.76, 0.28]); ax.axis("off")
    rows = [["Matched analytes", f"{len(df)}"],
            ["Spectra compared", "435 (180 Raman / 255 Ag-SERS)"],
            ["Preprocessing (primary)", "A2 ASLS + Savitzky–Golay + SNV (exact Stage B)"],
            ["Common grid", "520–1750 cm⁻¹ @ 2 cm⁻¹ (616 bins)"],
            ["Peak-match tolerance", "±12 cm⁻¹"],
            ["Sensitivity pipeline", "A1 ASLS + Savitzky–Golay + L2"],
            ["Status", "READ-ONLY audit — no models fitted, no preprocessing changed"]]
    t = ax.table(cellText=rows, colWidths=[0.36, 0.64], loc="center", cellLoc="left")
    t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 1.7)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor(GRID)
        if c == 0: cell.set_text_props(fontweight="bold", color=INK)
    fig.text(0.5, 0.27, "PRINCIPAL FINDING", ha="center", fontsize=11,
             fontweight="bold", color=INK)
    fig.text(0.5, 0.155,
             "Ag-SERS spectra of 51 chemically distinct metabolites are ~95% identical to one another\n"
             "(median between-analyte cosine 0.945; 95% of each spectrum's variance is the corpus-mean\n"
             "background). Analyte identity survives as a small residual — sufficient for 73% within-SERS\n"
             "identification, but too faint to support cross-modal correspondence with Raman.\n"
             "Stage A/B's weak-similarity conclusion is spectroscopically corroborated.",
             ha="center", fontsize=9.6, color=INK)
    fig.text(0.5, 0.06, "Branch gaira-v5-rebuild-plan · read-only investigation", ha="center",
             fontsize=8, color=MUTED)
    pdf.savefig(fig); plt.close(fig)


def toc(pdf):
    items = [("1", "Executive summary"), ("2", "Decisive evidence — five diagnostics"),
             ("3", "Global statistics and distributions (Part 9)"),
             ("4", "Peak-correspondence null controls (Part 3)"),
             ("5", "Band-level comparison heatmaps (Part 7)"),
             ("6", "Chemical-family analysis (Part 10)"),
             ("7", "Per-analyte pages — 51 matched analytes (Parts 1, 2, 8, 11)"),
             ("8", "Scientific conclusions — twelve questions (Part 12)")]
    text_page(pdf, "Contents", [("m", f"{n}.   {t}") for n, t in items])


# ───────────────────────── decisive evidence figures ─────────────────────────
def decisive(pdf, blob, extra):
    store, grid = blob["store"], blob["grid"]
    df = blob["df"]

    # ---- Page: SERS degeneracy (the smoking gun) ----
    fig = plt.figure(figsize=PAGE)
    fig.text(0.07, 0.955, "Decisive evidence 1 — the Ag-SERS arm is background-dominated",
             fontsize=14, fontweight="bold", color=INK)
    fig.text(0.07, 0.932, "All 51 Ag-SERS mean spectra overlaid (L2) versus all 51 Raman mean spectra, same axes.",
             fontsize=9, color=MUTED)
    import sys as _s; _s.path.insert(0, str(REPO / "src"))
    from gaira.evidence import datasets as D
    dl = D.build("A1_asls_savgol_l2")
    A = dl.matched_analytes
    Rm = np.vstack([np.nan_to_num(dl.X[((dl.meta.analyte == a) & (dl.meta.modality == "raman")).values]).mean(0) for a in A])
    Sm = np.vstack([np.nan_to_num(dl.X[((dl.meta.analyte == a) & (dl.meta.modality == "sers")).values]).mean(0) for a in A])
    ax1 = fig.add_axes([0.09, 0.66, 0.39, 0.22]); ax2 = fig.add_axes([0.55, 0.66, 0.39, 0.22])
    for row in Rm: ax1.plot(dl.grid, row, color=RAMAN, lw=0.35, alpha=0.35)
    ax1.plot(dl.grid, Rm.mean(0), color=INK, lw=1.2)
    for row in Sm: ax2.plot(dl.grid, row, color=SERS, lw=0.35, alpha=0.35)
    ax2.plot(dl.grid, Sm.mean(0), color=INK, lw=1.2)
    for ax, t in ((ax1, "Raman — 51 analytes (distinct)"), (ax2, "Ag-SERS — 51 analytes (near-identical)")):
        ax.set_title(t); ax.set_xlabel("Raman shift (cm⁻¹)"); ax.set_yticks([])
    ax1.set_ylabel("intensity (L2, a.u.)")

    ax3 = fig.add_axes([0.09, 0.375, 0.39, 0.20])
    def between(X):
        Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
        S = Xn @ Xn.T; iu = np.triu_indices(len(X), 1); return S[iu]
    ax3.hist(between(Rm), bins=30, color=RAMAN, alpha=0.85, label="Raman")
    ax3.hist(between(Sm), bins=30, color=SERS, alpha=0.85, label="Ag-SERS")
    ax3.set_xlabel("cosine between DIFFERENT analytes"); ax3.set_ylabel("count"); ax3.legend()
    ax3.set_title("Between-analyte similarity within a modality")

    ax4 = fig.add_axes([0.55, 0.375, 0.39, 0.20])
    dg = extra["degen"].get("A1_asls_savgol_l2", {})
    labels = ["Raman", "Ag-SERS"]
    acc = [dg.get("raman", {}).get("loo_1nn_acc", 0), dg.get("sers", {}).get("loo_1nn_acc", 0)]
    ax4.bar(labels, acc, color=[RAMAN, SERS], width=0.5)
    ax4.axhline(1 / 51, color=MUTED, ls="--", lw=1)
    ax4.text(-0.42, 1 / 51 + 0.045, "chance (1/51)", color=MUTED, fontsize=7, ha="left")
    for i, v in enumerate(acc): ax4.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8.5, color=INK)
    ax4.set_ylim(0, 1.08); ax4.set_ylabel("leave-one-out 1-NN analyte ID")
    ax4.set_title("Analyte identity IS present in Ag-SERS")

    fig.text(0.07, 0.32, "Reading", fontsize=11, fontweight="bold", color=INK)
    for i, line in enumerate(_wrap(
        "Ag-SERS spectra of chemically unrelated metabolites are ~95% identical to each other "
        "(median between-analyte cosine 0.945 vs 0.349 for Raman), and 95% of each Ag-SERS spectrum's "
        "variance is captured by the single corpus-mean spectrum — the signature of a dominant shared "
        "colloid/surface background. Analyte-specific information nevertheless exists: leave-one-out "
        "1-NN identification reaches 0.73 within Ag-SERS against 0.02 chance. The analyte therefore "
        "contributes a small, real residual on a large common component that has no Raman counterpart.", 104)):
        fig.text(0.07, 0.295 - i * 0.017, line, fontsize=8.8, color=INK)
    pdf.savefig(fig); plt.close(fig); fig_save(fig, None)

    # ---- Page: reproducibility ceiling + nulls + background removal ----
    fig = plt.figure(figsize=PAGE)
    fig.text(0.07, 0.955, "Decisive evidence 2 — reproducibility, chance matching, background removal",
             fontsize=14, fontweight="bold", color=INK)

    ax = fig.add_axes([0.09, 0.70, 0.38, 0.19])
    pipes = ["SNV\n(Stage B)", "L2", "raw"]
    ram = [0.986, 0.990, 0.999]; ser = [0.491, 0.948, 0.999]
    x = np.arange(3); w = 0.34
    ax.bar(x - w / 2, ram, w, color=RAMAN, label="Raman")
    ax.bar(x + w / 2, ser, w, color=SERS, label="Ag-SERS")
    ax.set_xticks(x); ax.set_xticklabels(pipes); ax.set_ylim(0, 1.1)
    ax.set_ylabel("within-analyte replicate cosine"); ax.legend(loc="lower right")
    ax.set_title("Replicate reproducibility (the ceiling)")
    for xi, v in zip(x + w / 2, ser): ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=7.5, color=INK)

    ax = fig.add_axes([0.57, 0.70, 0.38, 0.19])
    n = extra["nulls"]
    cats = ["observed", "mismatched\nanalyte", "uniform\nrandom"]
    vals = [n["all_SERS_peaks"]["observed_recall_median"],
            n["all_SERS_peaks"]["mismatched_null_median"],
            n["all_SERS_peaks"]["random_null_median"]]
    ax.bar(cats, vals, color=[SERS, MUTED, GRID], width=0.55)
    for i, v in enumerate(vals): ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8.5, color=INK)
    ax.set_ylim(0, 1.0); ax.set_ylabel("fraction of Raman bands 'matched'")
    ax.set_title("Band matching is at chance level")

    ax = fig.add_axes([0.09, 0.42, 0.38, 0.19])
    bg = extra["bg"].get("A2_asls_savgol_snv", {})
    conds = ["as evaluated", "common\nremoved"]
    mt = [bg.get("as_evaluated", {}).get("matched_cos", 0), bg.get("common_removed_1", {}).get("matched_cos", 0)]
    mm = [bg.get("as_evaluated", {}).get("mismatched_cos", 0), bg.get("common_removed_1", {}).get("mismatched_cos", 0)]
    x = np.arange(2)
    ax.bar(x - 0.17, mt, 0.34, color=RAMAN, label="matched pair")
    ax.bar(x + 0.17, mm, 0.34, color=MUTED, label="mismatched pair")
    ax.set_xticks(x); ax.set_xticklabels(conds); ax.legend()
    ax.set_ylabel("cross-modal cosine"); ax.set_title("Removing the background makes\nthe residual specific — but small")
    ax.axhline(0, color=MUTED, lw=0.6)

    ax = fig.add_axes([0.57, 0.42, 0.38, 0.19])
    for pipe, c, lab in (("A1_asls_savgol_l2", RAMAN, "L2"), ("A2_asls_savgol_snv", SERS, "SNV")):
        b = extra["bg"].get(pipe, {})
        ks = ["as_evaluated", "common_removed_1", "common_removed_2", "common_removed_3", "common_removed_5"]
        ax.plot(range(len(ks)), [b.get(k, {}).get("top1", np.nan) for k in ks], "-o", color=c, label=lab, ms=4, lw=1.6)
    ax.axhline(1 / 51, color=MUTED, ls="--", lw=1)
    ax.set_xticks(range(5)); ax.set_xticklabels(["none", "1", "2", "3", "5"], fontsize=7)
    ax.set_xlabel("shared components removed"); ax.set_ylabel("cross-modal top-1")
    ax.legend(); ax.set_title("Retrieval stays weak after correction")

    fig.text(0.07, 0.36, "Reading", fontsize=11, fontweight="bold", color=INK)
    for i, line in enumerate(_wrap(
        "SNV — the pipeline Stage A/B selected — collapses Ag-SERS replicate agreement from 0.95 (L2) "
        "and 0.999 (raw) to 0.49, because SNV mean-centres each spectrum so cosine becomes a correlation "
        "dominated by the low-amplitude residual. This is a genuine methodological defect, but it does not "
        "create the negative result: matching a Raman spectrum to a DIFFERENT analyte's Ag-SERS spectrum "
        "recovers 71% of bands versus 80% observed (0/51 analytes specific), and this holds identically "
        "under L2. Projecting out the shared component makes the surviving cross-modal similarity "
        "specific (mismatched falls to ~0) yet leaves top-1 retrieval at only 0.12–0.18 against 0.02 chance.", 104)):
        fig.text(0.07, 0.335 - i * 0.017, line, fontsize=8.8, color=INK)
    pdf.savefig(fig); plt.close(fig)


def fig_save(fig, path):
    if path: fig.savefig(path, dpi=150, bbox_inches="tight")


# ───────────────────────── global stats ─────────────────────────
def global_stats(pdf, blob, extra):
    df = blob["df"]
    fig = plt.figure(figsize=PAGE)
    fig.text(0.07, 0.955, "Global statistics across 51 matched analytes (Part 9)",
             fontsize=14, fontweight="bold", color=INK)
    fig.text(0.07, 0.932, "Primary pipeline: SNV, exactly as Stage B evaluated.", fontsize=9, color=MUTED)
    panels = [("cosine", "cosine similarity"), ("pearson", "Pearson r"), ("spearman", "Spearman ρ"),
              ("spectral_angle_deg", "spectral angle (°)"), ("nrmse", "normalised RMSE"),
              ("dtw", "DTW distance"), ("derivative_corr", "derivative correlation"),
              ("peak_f1", "peak F1"), ("peak_jaccard", "peak Jaccard"),
              ("mean_abs_shift", "mean |Δν| (cm⁻¹)"), ("matched_pct_of_raman", "% Raman bands matched"),
              ("peak_correspondence_score", "peak correspondence score")]
    for i, (col, lab) in enumerate(panels):
        r, c = divmod(i, 3)
        ax = fig.add_axes([0.09 + c * 0.30, 0.74 - r * 0.175, 0.23, 0.115])
        v = df[col].dropna()
        ax.hist(v, bins=16, color=RAMAN if "peak" not in col else SERS, alpha=0.9)
        ax.axvline(v.median(), color=INK, lw=1.2)
        ax.set_title(f"{lab}\nmedian {v.median():.2f}", fontsize=7.6)
        ax.tick_params(labelsize=6.5); ax.set_yticks([])
    # rankings
    g = extra["glob"]["rankings"]
    y = 0.30
    fig.text(0.07, y, "Rankings", fontsize=11, fontweight="bold", color=INK); y -= 0.022
    for key, title in (("top10_best_match_cosine", "Best full-profile agreement"),
                       ("top10_worst_match_cosine", "Worst full-profile agreement"),
                       ("top10_largest_mean_abs_shift", "Largest mean |Δν|"),
                       ("top10_strongest_intensity_redistribution", "Strongest intensity redistribution")):
        items = g.get(key, [])[:8]
        k = [c for c in items[0] if c not in ("analyte", "family")][0] if items else ""
        s = ", ".join(f"{it['analyte']} ({it[k]:.2f})" for it in items)
        fig.text(0.07, y, title, fontsize=8.6, fontweight="bold", color=INK); y -= 0.016
        for line in _wrap(s, 104):
            fig.text(0.07, y, line, fontsize=8.2, color=INK); y -= 0.015
        y -= 0.006
    pdf.savefig(fig); plt.close(fig)


def band_and_family(pdf, blob, extra):
    bands = extra["bands"]; fam = blob["fam"]
    fig = plt.figure(figsize=PAGE)
    fig.text(0.07, 0.955, "Band-level comparison and chemical families (Parts 7, 10)",
             fontsize=14, fontweight="bold", color=INK)
    # heatmap: analyte x region cosine
    piv = bands.pivot_table(index="analyte", columns="region", values="cosine")
    order = ["600-800", "800-1000", "1000-1200", "1200-1400", "1400-1600", "1600-1800"]
    piv = piv[[c for c in order if c in piv.columns]]
    ax = fig.add_axes([0.20, 0.50, 0.34, 0.40])
    im = ax.imshow(piv.values, aspect="auto", cmap=DIV, vmin=-1, vmax=1)
    ax.set_yticks(range(len(piv))); ax.set_yticklabels(piv.index, fontsize=4.6)
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns, rotation=45, ha="right", fontsize=6)
    ax.set_title("Per-region cosine", fontsize=9); ax.grid(False)
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02).ax.tick_params(labelsize=6)
    # heatmap: mean abs shift
    piv2 = bands.pivot_table(index="analyte", columns="region", values="mean_abs_shift")
    piv2 = piv2[[c for c in order if c in piv2.columns]]
    ax = fig.add_axes([0.62, 0.50, 0.30, 0.40])
    im = ax.imshow(piv2.values, aspect="auto", cmap=SEQ, vmin=0, vmax=12)
    ax.set_yticks([]); ax.set_xticks(range(piv2.shape[1]))
    ax.set_xticklabels(piv2.columns, rotation=45, ha="right", fontsize=6)
    ax.set_title("Per-region mean |Δν| (cm⁻¹)", fontsize=9); ax.grid(False)
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02).ax.tick_params(labelsize=6)
    # family table
    ax = fig.add_axes([0.07, 0.10, 0.86, 0.33]); ax.axis("off")
    show = fam.copy()
    for c in ("cosine", "peak_f1", "mean_abs_shift", "pcs", "redistribution", "rank_corr", "matched_pct", "ceiling"):
        if c in show: show[c] = show[c].round(3)
    cols = ["family", "n", "cosine", "peak_f1", "mean_abs_shift", "pcs", "redistribution", "rank_corr", "matched_pct"]
    cols = [c for c in cols if c in show.columns]
    t = ax.table(cellText=show[cols].values, colLabels=cols, loc="upper center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(7); t.scale(1, 1.35)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor(GRID)
        if r == 0: cell.set_text_props(fontweight="bold")
    ax.set_title("Chemical-family summary (sorted by peak-correspondence score)", fontsize=9, pad=2)
    pdf.savefig(fig); plt.close(fig)


# ───────────────────────── per-analyte pages ─────────────────────────
def analyte_page(pdf, a, S, grid, save_png=True):
    R, Sp = np.nan_to_num(S["R"]), np.nan_to_num(S["S"])
    rm, sm, rsd, ssd = S["rm"], S["sm"], S["rsd"], S["ssd"]
    rec, rows, sim = S["rec"], S["rows"], S["sim"]
    lo = float(min(np.nanmin(R), np.nanmin(Sp))); hi = float(max(np.nanmax(R), np.nanmax(Sp)))
    pad = 0.06 * (hi - lo); ylim = (lo - pad, hi + pad)

    fig = plt.figure(figsize=PAGE)
    fig.text(0.07, 0.965, a, fontsize=16, fontweight="bold", color=INK)
    fig.text(0.07, 0.945, f"family: {rec['family']}"
                          + ("  (family assignment ambiguous)" if rec.get("family_ambiguous") else ""),
             fontsize=9, color=MUTED)
    meta = (f"Raman source: {rec['raman_sources']}   |   Ag-SERS source: {rec['sers_sources']}\n"
            f"excitation: 785 nm both arms   |   substrate: Raman = neat powder / library, "
            f"Ag-SERS = citrate Ag colloid\n"
            f"preprocessing: ASLS baseline + Savitzky–Golay + SNV (Stage B, unmodified)   |   "
            f"grid 520–1750 cm⁻¹ @ 2 cm⁻¹\n"
            f"replicates: Raman n={rec['n_raman']}, Ag-SERS n={rec['n_sers']}"
            + ("   [Raman replicates span two sources]" if rec.get("raman_multi_source") else ""))
    for i, line in enumerate(meta.split("\n")):
        fig.text(0.07, 0.925 - i * 0.0145, line, fontsize=7.6, color=INK)

    # row 1 — overlays, identical scaling
    ax1 = fig.add_axes([0.08, 0.688, 0.40, 0.158]); ax2 = fig.add_axes([0.555, 0.688, 0.40, 0.158])
    for row in R: ax1.plot(grid, row, color=RAMAN, lw=0.4, alpha=0.45)
    ax1.fill_between(grid, rm - rsd, rm + rsd, color=RAMAN, alpha=0.18, lw=0)
    ax1.plot(grid, rm, color=RAMAN, lw=1.5)
    for row in Sp: ax2.plot(grid, row, color=SERS, lw=0.4, alpha=0.45)
    ax2.fill_between(grid, sm - ssd, sm + ssd, color=SERS, alpha=0.18, lw=0)
    ax2.plot(grid, sm, color=SERS, lw=1.5)
    for ax, t in ((ax1, f"Raman — all {rec['n_raman']} replicates, mean ± SD"),
                  (ax2, f"Ag-SERS — all {rec['n_sers']} replicates, mean ± SD")):
        ax.set_ylim(*ylim); ax.set_xlim(grid[0], grid[-1]); ax.set_title(t, fontsize=8.4)
        ax.tick_params(labelsize=6.5)
    ax1.set_ylabel("SNV intensity"); ax2.set_yticklabels([])

    # row 2 — mean overlay + difference
    ax3 = fig.add_axes([0.08, 0.525, 0.875, 0.116])
    ax3.plot(grid, rm, color=RAMAN, lw=1.3, label="Raman mean")
    ax3.plot(grid, sm, color=SERS, lw=1.3, label="Ag-SERS mean")
    ax3.plot(grid, sm - rm, color=MUTED, lw=0.8, label="difference (SERS − Raman)")
    ax3.axhline(0, color=GRID, lw=0.6)
    ax3.set_xlim(grid[0], grid[-1]); ax3.legend(ncol=3, fontsize=7, loc="upper right")
    ax3.set_ylabel("SNV intensity"); ax3.set_title("Mean overlay and difference spectrum", fontsize=8.4)
    ax3.tick_params(labelsize=6.5)

    # row 3 — derivative + peak correspondence diagram
    ax4 = fig.add_axes([0.08, 0.375, 0.40, 0.100])
    ax4.plot(grid, np.gradient(rm), color=RAMAN, lw=0.8)
    ax4.plot(grid, np.gradient(sm), color=SERS, lw=0.8)
    ax4.set_xlim(grid[0], grid[-1]); ax4.set_title("First derivative", fontsize=8.4)
    ax4.set_xlabel("Raman shift (cm⁻¹)", fontsize=7); ax4.set_yticks([]); ax4.tick_params(labelsize=6.5)

    ax5 = fig.add_axes([0.555, 0.375, 0.40, 0.100])
    for p in S["rp"]: ax5.vlines(p["position"], 0, p["rel_intensity"], color=RAMAN, lw=1.0)
    for p in S["sp"]: ax5.vlines(p["position"], 0, -p["rel_intensity"], color=SERS, lw=1.0)
    for r in rows:
        if r["kind"] == "matched":
            ax5.plot([r["raman_peak"], r["sers_peak"]], [0.03, -0.03], color=MUTED, lw=0.5, alpha=0.8)
    ax5.axhline(0, color=INK, lw=0.6); ax5.set_xlim(grid[0], grid[-1])
    ax5.set_title("Peak map — Raman (up) vs Ag-SERS (down)", fontsize=8.4)
    ax5.set_xlabel("Raman shift (cm⁻¹)", fontsize=7); ax5.set_yticks([]); ax5.tick_params(labelsize=6.5)

    # metrics block
    m = (f"cosine {sim['cosine']:+.3f}   Pearson {sim['pearson']:+.3f}   Spearman {sim['spearman']:+.3f}   "
         f"spectral angle {sim['spectral_angle_deg']:.1f}°   NRMSE {sim['nrmse']:.3f}\n"
         f"DTW {sim['dtw']:.3f}   derivative r {sim['derivative_corr']:+.3f}   "
         f"peak F1 {rec['peak_f1']:.2f}   Jaccard {rec['peak_jaccard']:.2f}   "
         f"precision {rec['peak_precision']:.2f}   recall {rec['peak_recall']:.2f}\n"
         f"peaks: Raman {rec['n_raman_peaks']}, Ag-SERS {rec['n_sers_peaks']}, matched {rec['n_matched']}   "
         f"mean |Δν| {rec['mean_abs_shift']:.1f} cm⁻¹   correspondence score {rec['peak_correspondence_score']:.2f}\n"
         f"optimal rigid shift {rec['align_optimal_shift_cm']:+.0f} cm⁻¹ (cosine gain {rec['align_cosine_gain']:+.3f})   "
         f"replicate ceiling: Raman {rec['within_raman_cos']:.2f} / Ag-SERS {rec['within_sers_cos']:.2f}")
    for i, line in enumerate(m.split("\n")):
        fig.text(0.08, 0.305 - i * 0.0145, line, fontsize=7.2, color=INK, family="DejaVu Sans Mono")

    # correspondence table (strongest 10 matched + counts)
    mt = [r for r in rows if r["kind"] == "matched"]
    mt = sorted(mt, key=lambda r: -(r["raman_prom"] or 0))[:9]
    cells = [[f"{r['raman_peak']:.0f}", f"{r['sers_peak']:.0f}", f"{r['shift']:+.0f}",
              f"{r['intensity_ratio']:.2f}", r["confidence"], r["note"][:26]] for r in mt]
    if cells:
        ax6 = fig.add_axes([0.08, 0.085, 0.55, 0.135]); ax6.axis("off")
        t = ax6.table(cellText=cells,
                      colLabels=["Raman", "Ag-SERS", "Δν", "I ratio", "conf.", "note"],
                      loc="upper center", cellLoc="center")
        t.auto_set_font_size(False); t.set_fontsize(6.2); t.scale(1, 1.1)
        for (r, c), cell in t.get_celld().items():
            cell.set_edgecolor(GRID)
            if r == 0: cell.set_text_props(fontweight="bold")
        ax6.set_title(f"Peak correspondence — strongest matches "
                      f"({rec['n_raman_only']} Raman-only, {rec['n_sers_only']} SERS-only)",
                      fontsize=7.6, pad=2)

    # interpretation
    fig.text(0.655, 0.228, "Spectroscopic interpretation", fontsize=8.2, fontweight="bold", color=INK)
    for i, line in enumerate(_wrap(rec["interpretation"], 50)[:13]):
        fig.text(0.655, 0.212 - i * 0.0122, line, fontsize=6.4, color=INK)

    fig.text(0.07, 0.035,
             "Both panels share identical x- and y-limits; no independent autoscaling. "
             "Peak detection and all metrics use the spectra exactly as entered into Stage B.",
             fontsize=6.6, color=MUTED)
    pdf.savefig(fig)
    if save_png:
        fig.savefig(FIG / f"analyte_{a.replace(' ', '_').replace('/', '-')}.png", dpi=110)
    plt.close(fig)


# ───────────────────────── conclusions ─────────────────────────
def conclusions(pdf, blob, extra):
    n = extra["nulls"]; dg = extra["degen"]; bg = extra["bg"]; sens = extra["sens"]
    df = blob["df"]
    Q = [
     ("1. Does Stage A look spectroscopically believable?",
      "Yes. The weak cross-modal similarity is corroborated directly in the spectra and is not an artefact of "
      "representation. Matched Raman/Ag-SERS pairs separate from mismatched pairs by only +0.037 (SNV) and "
      "+0.027 (L2) in cosine, and band-level correspondence does not exceed a mismatched-analyte null in any "
      "analyte (0/51 at p<0.05). Stage A's conclusion stands."),
     ("2. Are Raman and Ag-SERS observing the same dominant vibrational modes?",
      "Not demonstrably, as measured here. The Ag-SERS spectra carry ~41–46 reproducible features versus ~12–13 "
      "well-defined Raman bands, and their positions show no analyte-specific alignment with the Raman bands. "
      "The dominant Ag-SERS features are shared across all 51 analytes, which is the behaviour of a common "
      "surface/colloid contribution rather than of the analyte's own normal modes."),
     ("3. Are peak positions generally preserved?",
      "No — not in an informative sense. 80% of Raman bands find an Ag-SERS peak within ±12 cm⁻¹, but a "
      "mismatched analyte achieves 71% and uniform-random peaks 60%, because the Ag-SERS peak list is dense "
      "(~24 cm⁻¹ spacing). The apparent preservation is a peak-density artefact; the excess over chance is only "
      "+0.08 and reaches p<0.05 in no analyte."),
     ("4. Are differences primarily intensity redistribution?",
      "Intensity redistribution is severe (median peak-rank ρ 0.12, redistribution index 0.24), but it is not the "
      "primary cause. The band inventories themselves differ in size and identity, so the discrepancy is not a "
      "re-weighting of one shared mode set. Redistribution is real but secondary to background dominance."),
     ("5. Are there systematic peak shifts?",
      "No. The optimal rigid shift is 0 cm⁻¹ at the median (|shift|>2 cm⁻¹ helps in a minority of analytes) and "
      "buys only +0.036 median cosine. The mean |Δν| of 5.8 cm⁻¹ is consistent with random matching inside a "
      "±12 cm⁻¹ window rather than a systematic chemical displacement."),
     ("6. Which biochemical families transfer best?",
      "Differences between families are small against the noise. Polysaccharide, amino-acid and protein entries "
      "rank highest by correspondence score (0.20–0.30) and purines/lipids/small-nitrogenous lowest (0.08–0.14); "
      "purines show the best preserved intensity ordering (rank ρ 0.47). No family reaches a level that would "
      "support cross-modal grounding."),
     ("7. Are there suspicious preprocessing artefacts?",
      "Yes — one significant defect. SNV collapses Ag-SERS replicate reproducibility from 0.999 (raw) and 0.948 "
      "(L2) to 0.491, because SNV mean-centres each spectrum so that cosine becomes Pearson correlation and the "
      "low-amplitude residual dominates. Stage A/B therefore evaluated the Ag-SERS arm at needlessly low "
      "effective SNR. Native resolutions (1.0–1.7 cm⁻¹) are all finer than the 2 cm⁻¹ grid, and Gobbato Raman "
      "and Ag-SERS share one instrument, so resampling and instrument mismatch are excluded. Critically, the "
      "negative result survives the correction: it is reproduced under L2."),
     ("8. Are there problematic analytes that should be excluded?",
      "Under SNV, 29/51 analytes have within-Ag-SERS replicate cosine below 0.50 (6 below 0.30: hydroxyproline, "
      "oleate, fructose-6-phosphate, methionine, serine, triolein), so their Ag-SERS means are unreliable in that "
      "representation. Under L2 the same replicates agree at 0.95, so these are pipeline casualties rather than "
      "bad measurements. The genuinely structural caveat is that 27/51 analytes draw Raman replicates from two "
      "different sources, mixing inter-instrument variation into within-Raman spread."),
     ("9. Should Stage A be rerun?",
      "Yes, but for methodological correctness rather than in expectation of a different verdict. Stage A should "
      "not have selected a normalisation that degrades one modality's reproducibility by half. A rerun should use "
      "L2 (or explicit background-component removal) and report similarity relative to the per-modality "
      "reproducibility ceiling. Expect the architectural conclusion to survive: after removing the shared "
      "component, cross-modal top-1 rises only to 0.12–0.18 against 0.02 chance."),
     ("10. What is the single biggest scientific insight?",
      "The Ag-SERS arm of this grounding corpus is background-dominated, not analyte-dominated. Ag-SERS spectra of "
      "51 chemically distinct metabolites are ~95% identical to one another and 95% of each spectrum's variance is "
      "the corpus-mean background; the analyte survives only as a ~5% residual — enough for 73% within-modality "
      "identification, but far too faint to anchor a correspondence with a powder Raman spectrum that contains no "
      "such background. The cross-modal failure is therefore a measurement-contrast problem in the Ag-colloid "
      "acquisition, not evidence that the two techniques observe unrelated chemistry, and not a shortcoming of "
      "the representation or the encoders. This also explains the Stage B encoder collapse: embeddings whose "
      "cross-analyte duplicate fraction reached 0.96–1.00 were faithfully representing inputs that are themselves "
      "near-identical."),
    ]
    blocks = []
    for h, b in Q:
        blocks.append(("h", h)); blocks.append(("b", b))
    text_page(pdf, "Scientific conclusions (Part 12)", blocks,
              "Answers follow strictly from the measurements in this audit.")


def main():
    blob, extra = _load()
    store, grid, df = blob["store"], blob["grid"], blob["df"]
    with PdfPages(PDF_PATH) as pdf:
        cover(pdf, df, blob["preproc"])
        toc(pdf)
        summary_blocks = [
            ("h", "What was asked"),
            ("b", "Stage A and Stage B concluded that Raman and Ag-SERS observations of the same analytes are "
                  "only weakly similar. This audit inspects the spectra themselves — using the frozen Stage B "
                  "corpus and the unmodified Stage B SNV pipeline — to decide whether the two techniques observe "
                  "fundamentally different spectra, or the same vibrational modes with different enhancement."),
            ("h", "What was found"),
            ("b", "1. The Stage A/B conclusion is spectroscopically corroborated. Matched pairs separate from "
                  "mismatched pairs by only +0.03 in cosine, and no analyte shows band correspondence above a "
                  "mismatched-analyte null."),
            ("b", "2. The mechanism is background dominance, not different chemistry. Ag-SERS spectra of 51 "
                  "chemically distinct metabolites are ~95% identical to each other (between-analyte cosine 0.945 "
                  "versus 0.349 for Raman); 95% of each Ag-SERS spectrum's variance is explained by the single "
                  "corpus-mean spectrum — the signature of a citrate/colloid surface background."),
            ("b", "3. The Ag-SERS data are not junk. Analyte identity is present as a small residual and supports "
                  "73% leave-one-out 1-NN identification within Ag-SERS against 2% chance; replicates agree at "
                  "0.999 raw. The analyte signal exists but contributes only ~5% of the spectrum."),
            ("b", "4. Apparent band correspondence is a peak-density artefact. Ag-SERS yields ~46 features spaced "
                  "~24 cm⁻¹ against ~12 Raman bands spaced ~75 cm⁻¹, so a ±12 cm⁻¹ window almost always finds a "
                  "hit: observed recall 0.80 versus 0.71 for a mismatched analyte and 0.60 for random peaks."),
            ("b", "5. A real preprocessing defect exists but does not drive the result. SNV collapses Ag-SERS "
                  "replicate reproducibility from 0.95 (L2) to 0.49; the negative result nonetheless reproduces "
                  "under L2 and after removing the shared component."),
            ("b", "6. Rigid alignment does not rescue agreement: median optimal shift 0 cm⁻¹, median cosine gain "
                  "+0.036. Intensity redistribution is severe (peak-rank ρ 0.12)."),
            ("h", "What it means"),
            ("b", "The cross-modal failure is a measurement-contrast problem in the Ag-colloid acquisition rather "
                  "than evidence that Raman and Ag-SERS observe unrelated chemistry, and it is not a failure of "
                  "representation learning. It also explains the Stage B encoder collapse: near-identical inputs "
                  "produce near-identical embeddings. Progress requires Ag-SERS measurements in which the analyte, "
                  "not the colloid, dominates the spectrum — higher effective surface coverage, blank-colloid "
                  "difference spectra, or explicit background modelling — before any further representation work."),
            ("h", "Scope and limits"),
            ("b", "Read-only audit: no models were fitted, no preprocessing was altered, no GAIRA code or governing "
                  "document was modified. All statistics are computed on the 51 matched analytes of the frozen "
                  "Stage B corpus. The L2 pipeline appears only as a clearly-labelled sensitivity analysis."),
        ]
        text_page(pdf, "Executive summary", summary_blocks,
                  "51 matched analytes · 435 spectra · 785 nm · read-only")
        decisive(pdf, blob, extra)
        global_stats(pdf, blob, extra)
        band_and_family(pdf, blob, extra)
        # per-analyte pages
        sep = plt.figure(figsize=PAGE)
        sep.text(0.5, 0.55, "Per-analyte pages", ha="center", fontsize=20, fontweight="bold", color=INK)
        sep.text(0.5, 0.50, "51 matched analytes — Parts 1, 2, 8 and 11", ha="center", fontsize=11, color=MUTED)
        pdf.savefig(sep); plt.close(sep)
        for i, a in enumerate(sorted(store)):
            analyte_page(pdf, a, store[a], grid)
            if (i + 1) % 10 == 0:
                print(f"  ... {i+1}/51 analyte pages", flush=True)
        conclusions(pdf, blob, extra)
        d = pdf.infodict()
        d["Title"] = "GAIRA V5 — Matched-Analyte Spectral Audit (Raman vs Ag-SERS)"
        d["Subject"] = "Read-only spectroscopic audit of the 51 matched analytes"
    print(f"written: {PDF_PATH}  ({PDF_PATH.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
