"""BSV Validation — figures + PDF (Part 15)."""
from __future__ import annotations
import sys, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA

REPO = Path("/Users/surajpg/projects/GAIRA"); sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).parent))
import bsv_val_lib as L

OUT = REPO / "results/v5_rebuild/bsv_validation"
TAB, FIG, ART = OUT / "tables", OUT / "figures", OUT / "artifacts"
FIG.mkdir(parents=True, exist_ok=True)
PDF = REPO / "GAIRA_BSV_VALIDATION.pdf"
P, S, INK, MUTED, GRIDC = "#2563EB", "#D97706", "#1f2328", "#6B7280", "#E5E7EB"
DIV = LinearSegmentedColormap.from_list("d", [P, "#F7F7F7", S])
SEQ = LinearSegmentedColormap.from_list("s", ["#F8FAFC", P])
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9,
                     "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "axes.grid": True,
                     "grid.color": GRIDC, "grid.linewidth": 0.5, "xtick.color": MUTED,
                     "ytick.color": MUTED, "text.color": INK, "axes.labelcolor": INK,
                     "axes.titlecolor": INK, "legend.frameon": False})
PAGE = (8.5, 11.0)
SH = lambda t: t.replace("nucleic_", "").replace("_metabolism", "").replace("_antioxidant", "")\
    .replace("_glycan", "").replace("_membrane", "").replace("_peptide", "").replace("_acyl", "")\
    .replace("_amino_acid", "AA").replace("aromatic", "arom")


def main():
    H = L.Harness(); bio = H.bio
    ada = pd.read_csv(TAB / "part2_bsv_ils_adenine.csv")
    erg = pd.read_csv(TAB / "part2_bsv_ergothioneine.csv")
    ref = pd.read_csv(TAB / "part2_bsv_pure_raman.csv")
    mono = pd.read_csv(TAB / "part3_monotonicity.csv")
    spec = pd.read_csv(TAB / "part4_specificity.csv")
    ct = pd.read_csv(TAB / "part4_crosstalk.csv")
    corr = pd.read_csv(TAB / "part11_theme_correlation.csv", index_col=0)
    conf = pd.read_csv(TAB / "part9_confidence_system.csv")
    comp = pd.read_csv(TAB / "part5_component_contributions.csv")
    stab = pd.read_csv(TAB / "part8_replicate_stability.csv")
    Cnn = pd.read_csv(TAB / "part7_analyte_cosine.csv", index_col=0)
    geom = json.loads((ART / "part12_state_space.json").read_text())

    # F1 — calibration curves + effect size + monotonicity
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
    ax = axs[0]
    for (sub, las), g in ada.groupby(["substrate", "laser_nm"]):
        m = g.groupby("conc_uM").theme_nucleic_purine.mean()
        ax.plot(m.index, m.values - m.values[0], "-o", ms=3, lw=1, label=f"{sub}@{las}")
    ax.set_xlabel("adenine conc (µM)"); ax.set_ylabel("Δ purine theme composition")
    ax.set_title("Part 3 — adenine calibration curves"); ax.legend(fontsize=6)
    ax = axs[1]
    me = erg.groupby("conc_uM").theme_sulfur_antioxidant.mean()
    ax.plot(me.index, me.values - me.values[0], "-o", color=S, ms=4)
    ax.set_xlabel("ergothioneine conc"); ax.set_ylabel("Δ sulfur theme")
    ax.set_title("Part 3 — ergothioneine → sulfur (ρ=0.97)")
    ax = axs[2]
    x = np.arange(len(mono))
    ax.bar(x, mono.spearman, color=[P if "cA" in e or "cAu" in e else S for e in mono.experiment])
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels([e.replace("adenine::", "").replace("::", "\n") for e in mono.experiment],
                                         rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("target-theme Spearman ρ")
    ax.set_title("Monotonicity (all p=0.002; blue=colloidal)")
    fig.tight_layout(); fig.savefig(FIG / "f1_calibration.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # F2 — cross-talk matrix + specificity
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    ax = axs[0]
    Mx = ct.set_index("experiment")[[f"rho_{t}" for t in bio]]
    im = ax.imshow(Mx.values, cmap=DIV, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(bio))); ax.set_xticklabels([SH(t) for t in bio], rotation=45, ha="right", fontsize=6.5)
    ax.set_yticks(range(len(ct))); ax.set_yticklabels([e.replace("adenine::", "") for e in ct.experiment], fontsize=6.5)
    # mark target cell
    for i, tgt in enumerate(ct.target):
        j = bio.index(tgt); ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False, edgecolor=INK, lw=1.6))
    ax.set_title("Part 4 — theme cross-talk (ρ vs dose; boxed = target)"); ax.grid(False)
    plt.colorbar(im, ax=ax, fraction=0.03)
    ax = axs[1]
    xx = np.arange(len(spec))
    ax.bar(xx - 0.2, spec.target_rho_abs, 0.4, color=P, label="target |ρ|")
    ax.bar(xx + 0.2, spec.mean_offtarget_rho_abs, 0.4, color=S, label="mean off-target |ρ|")
    ax.set_xticks(xx); ax.set_xticklabels([e.replace("adenine::", "").replace("ergothioneine", "ergo")
                                           for e in spec.experiment], rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("|Spearman ρ|"); ax.legend(fontsize=7)
    ax.set_title(f"Specificity: target > off-target, but leakage high\n(median margin "
                 f"{spec.specificity_margin.median():.2f}, leakage {spec.leakage_ratio.median():.2f})")
    fig.tight_layout(); fig.savefig(FIG / "f2_crosstalk.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # F3 — theme coupling (encodes biology) + effective dimensionality
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    ax = axs[0]
    im = ax.imshow(corr.values, cmap=DIV, vmin=-1, vmax=1)
    ax.set_xticks(range(len(bio))); ax.set_xticklabels([SH(t) for t in bio], rotation=45, ha="right", fontsize=6.5)
    ax.set_yticks(range(len(bio))); ax.set_yticklabels([SH(t) for t in bio], fontsize=6.5)
    ax.set_title("Part 11 — theme correlation (pure Raman)\ncoupling encodes shared biology"); ax.grid(False)
    plt.colorbar(im, ax=ax, fraction=0.04)
    ax = axs[1]
    evr = geom["explained_variance_ratio"]
    ax.bar(range(1, len(evr) + 1), evr, color=P)
    ax.plot(range(1, len(evr) + 1), np.cumsum(evr), "-o", color=S, ms=4, label="cumulative")
    ax.axhline(0.9, ls="--", c=MUTED, lw=1)
    ax.axvline(geom["n_components_90pct_variance"], ls=":", c=INK, lw=1)
    ax.set_xlabel("BSV principal component"); ax.set_ylabel("variance explained")
    ax.set_title(f"Part 12 — state space: effective dim {geom['effective_dimensionality_entropy']} of 11\n"
                 f"(90% variance by {geom['n_components_90pct_variance']} components)")
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "f3_orthogonality.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # F4 — inter-analyte geometry (cosine + tree) → purine cluster
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.2))
    ax = axs[0]
    im = ax.imshow(Cnn.values, cmap=SEQ, vmin=0.7, vmax=1)
    ax.set_xticks(range(len(Cnn))); ax.set_xticklabels(Cnn.columns, rotation=45, ha="right", fontsize=6)
    ax.set_yticks(range(len(Cnn))); ax.set_yticklabels(Cnn.index, fontsize=6)
    ax.set_title("Part 7 — inter-analyte BSV cosine"); ax.grid(False)
    plt.colorbar(im, ax=ax, fraction=0.04)
    ax = axs[1]
    D = 1 - Cnn.values; D = np.clip((D + D.T) / 2, 0, None); np.fill_diagonal(D, 0)
    Z = linkage(squareform(D, checks=False), method="average")
    dendrogram(Z, labels=list(Cnn.index), ax=ax, leaf_font_size=7, color_threshold=0.15)
    ax.set_title("Part 7 — analyte clustering in BSV space\n(purines cluster; glucose isomers merge)")
    fig.tight_layout(); fig.savefig(FIG / "f4_geometry.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # F5 — trajectories (PCA of BSV space) + scaling vs redistribution
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    Tref = ref[[f"theme_{t}" for t in bio]].values
    pca = PCA(2).fit(Tref - Tref.mean(0))
    ax = axs[0]
    for (sub, las), g in ada.groupby(["substrate", "laser_nm"]):
        m = g.groupby("conc_uM")[[f"theme_{t}" for t in bio]].mean().values
        pj = pca.transform(m - Tref.mean(0))
        ax.plot(pj[:, 0], pj[:, 1], "-o", ms=3, lw=1, label=f"ade {sub}@{las}", alpha=0.8)
    me = erg.groupby("conc_uM")[[f"theme_{t}" for t in bio]].mean().values
    pj = pca.transform(me - Tref.mean(0))
    ax.plot(pj[:, 0], pj[:, 1], "-s", ms=4, color=INK, lw=1.4, label="ergothioneine")
    ax.set_xlabel("BSV-PC1"); ax.set_ylabel("BSV-PC2")
    ax.set_title("Part 6 — dose trajectories in BSV space"); ax.legend(fontsize=6)
    ax = axs[1]
    ax.bar(range(len(comp)), comp.profile_corr_low_high,
           color=[S if m.startswith("redistribution") else P for m in comp["mode"]])
    ax.axhline(0.8, ls="--", c=INK, lw=1); ax.set_ylim(-0.3, 1)
    ax.set_xticks(range(len(comp))); ax.set_xticklabels([e.replace("adenine::", "").replace("ergothioneine", "ergo")
                                                         for e in comp.experiment], rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("component-profile corr (low vs high dose)")
    ax.set_title("Part 5 — scaling (blue, >0.8) vs redistribution (amber)")
    fig.tight_layout(); fig.savefig(FIG / "f5_trajectories.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # F6 — confidence system + replicate stability
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.0))
    ax = axs[0]
    ax.scatter(conf.median_ood, conf.median_confidence, s=60, c=P)
    for _, r in conf.iterrows():
        ax.annotate(r.group.replace("serum_spike_", "").replace("_adsorbers", ""),
                    (r.median_ood, r.median_confidence), fontsize=6, xytext=(3, 2), textcoords="offset points")
    ax.set_xlabel("median OOD"); ax.set_ylabel("median overall confidence")
    ax.set_title("Part 9 — confidence tracks domain OOD (ρ=−0.57)")
    ax = axs[1]
    strong = conf[conf.group == "serum_spike_STRONG_adsorbers"].median_confidence.iloc[0]
    weak = conf[conf.group == "serum_spike_WEAK_adsorbers"].median_confidence.iloc[0]
    ax.bar(["strong Ag\nadsorbers", "weak Ag\nadsorbers"], [strong, weak], color=[P, S])
    ax.set_ylabel("median confidence")
    ax.set_title("Confidence does NOT distinguish\nstrong vs weak adsorbers (a gap)")
    ax = axs[2]
    ax.scatter(stab.icc_purine, stab.within_dose_cv_median, s=60,
               c=[P if "cA" in e else S for e in stab.experiment])
    for _, r in stab.iterrows():
        ax.annotate(r.experiment.replace("adenine::", ""), (r.icc_purine, r.within_dose_cv_median),
                    fontsize=6, xytext=(3, 2), textcoords="offset points")
    ax.set_xlabel("ICC (replicate reliability)"); ax.set_ylabel("within-dose CV")
    ax.set_title("Part 8 — reproducibility is substrate-dependent")
    fig.tight_layout(); fig.savefig(FIG / "f6_confidence_stability.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # ── assemble PDF ──
    with PdfPages(PDF) as pdf:
        fig = plt.figure(figsize=PAGE)
        fig.text(0.5, 0.8, "Biochemical State Vector — Validation", ha="center", fontsize=21,
                 fontweight="bold", color=INK)
        fig.text(0.5, 0.76, "GAIRA V6 — characterizing the BSV as a scientific coordinate system",
                 ha="center", fontsize=12, color=MUTED)
        fig.text(0.5, 0.66,
                 "The V6 engine, atlas, ontology and weights are FROZEN and only MEASURED here.\n"
                 "The question is not whether the software runs, but whether the BSV is a stable,\n"
                 "meaningful biochemical coordinate system.",
                 ha="center", fontsize=10, color=INK, style="italic")
        rows = [["Monotonicity (dose→target theme)", "monotonic + saturating, all p=0.002; ρ 0.34–0.97"],
                ["Best calibration", "ergothioneine→sulfur ρ 0.97; colloidal adenine→purine ρ 0.91"],
                ["Theme specificity", "target > off-target, but leakage 0.65–0.82 (high cross-talk)"],
                ["Theme coupling", "mean |r| 0.24, max 0.65 — and it ENCODES biology"],
                ["Effective dimensionality", "~4 of 11 themes (PC1 = 50% variance)"],
                ["Inter-analyte geometry", "purines cluster; glucose isomers merge"],
                ["Confidence", "tracks domain OOD (ρ −0.57); NOT analyte recoverability"],
                ["Reproducibility", "substrate-dependent (ICC 0.14 paper → 0.83 colloid)"]]
        ax = fig.add_axes([0.08, 0.30, 0.84, 0.28]); ax.axis("off")
        t = ax.table(cellText=rows, colWidths=[0.36, 0.64], loc="center", cellLoc="left")
        t.auto_set_font_size(False); t.set_fontsize(8.2); t.scale(1, 1.5)
        for (r, c), cell in t.get_celld().items():
            cell.set_edgecolor(GRIDC)
            if c == 0: cell.set_text_props(fontweight="bold")
        fig.text(0.5, 0.075, "Read-only validation · engine frozen · nothing pushed", ha="center",
                 fontsize=8, color=MUTED)
        pdf.savefig(fig); plt.close(fig)
        for name, cap in [
            ("f1_calibration.png", "Calibration — dose→theme curves are monotonic and saturating."),
            ("f2_crosstalk.png", "Theme cross-talk — target moves most, but off-target leakage is high."),
            ("f3_orthogonality.png", "Theme coupling encodes biology; the space is effectively ~4-D."),
            ("f4_geometry.png", "Inter-analyte geometry — purines cluster in BSV space."),
            ("f5_trajectories.png", "Dose trajectories; adenine redistributes, ergothioneine scales."),
            ("f6_confidence_stability.png", "Confidence tracks domain OOD but not analyte recoverability; "
                                            "reproducibility is substrate-dependent."),
        ]:
            fig = plt.figure(figsize=PAGE)
            img = plt.imread(FIG / name)
            ax = fig.add_axes([0.05, 0.5, 0.9, 0.4]); ax.imshow(img); ax.axis("off")
            fig.text(0.07, 0.94, cap.split("—")[0].strip(), fontsize=13, fontweight="bold", color=INK)
            y = 0.45
            for ln in _wrap(cap, 96):
                fig.text(0.07, y, ln, fontsize=9, color=INK); y -= 0.018
            pdf.savefig(fig); plt.close(fig)
        d = pdf.infodict(); d["Title"] = "GAIRA BSV Validation"
    print(f"figures + PDF written: {PDF.stat().st_size/1e6:.1f} MB")


def _wrap(t, n):
    out, line = [], ""
    for w in str(t).split():
        if len(line) + len(w) + 1 > n:
            out.append(line); line = w
        else:
            line = (line + " " + w).strip()
    if line: out.append(line)
    return out


if __name__ == "__main__":
    main()
