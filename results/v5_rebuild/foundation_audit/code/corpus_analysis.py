"""Foundation audit — Part 1/2: build the canonical Raman-only corpus table,
chemical-class distribution, per-source / per-analyte structure, cross-source
duplicates, and coverage gaps. Deterministic; reads the same loader the atlas uses.

Writes tables + figures to results/v5_rebuild/foundation_audit/{tables,figures}/ and
prints a JSON summary consumed by the report writer.
"""
from __future__ import annotations
import sys, json
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/Users/surajpg/projects/GAIRA"); sys.path.insert(0, str(REPO / "src"))
from gaira.foundation import dataset as DS
from gaira.foundation.families_raman import family_of

AUD = REPO / "results/v5_rebuild/foundation_audit"
TAB, FIG = AUD / "tables", AUD / "figures"
for p in (TAB, FIG): p.mkdir(parents=True, exist_ok=True)

# report-level chemical classes (raw family_of() -> Nature-methods reporting bucket)
CLASS_MAP = {
    "protein": "Protein", "amino_acid": "Amino acid",
    "fatty_acid": "Lipid (fatty acid)", "triglyceride": "Lipid (triglyceride)",
    "phospholipid": "Lipid (phospholipid)", "sterol": "Sterol / steroid",
    "saccharide": "Saccharide", "polysaccharide": "Polysaccharide",
    "purine": "Purine", "pyrimidine": "Pyrimidine",
    "nucleic_acid": "Nucleic acid", "nucleoside": "Nucleoside", "nucleotide": "Nucleotide",
    "cofactor": "Cofactor / vitamin", "carotenoid": "Carotenoid",
    "organic_acid": "Organic acid", "lipid": "Lipid (other)",
    "unknown": "Other / unclassified",
}
INK = "#1b2430"; ACCENT = "#2a6f97"; GRID = "#d7dce3"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK, labelsize=8)
    ax.yaxis.label.set_color(INK); ax.xaxis.label.set_color(INK)
    ax.grid(axis="x", color=GRID, linewidth=0.6, zorder=0)


def main():
    c = DS.load_reference_corpus()
    m = c.meta.copy()
    m["family"] = m.analyte.map(family_of)
    m["report_class"] = m.family.map(lambda f: CLASS_MAP.get(f, f))

    # per-analyte table (one row per unique analyte; a spectrum count + sources)
    ana = (m.groupby("analyte")
             .agg(n_spectra=("spectrum_id", "size"),
                  family=("family", "first"),
                  report_class=("report_class", "first"),
                  sources=("source", lambda s: sorted(set(s))),
                  n_sources=("source", lambda s: len(set(s))),
                  excitations=("excitation_nm", lambda s: sorted(set(s))))
             .reset_index())
    ana.to_csv(TAB / "corpus_analytes.csv", index=False)

    # ── distributions ──
    class_counts_analytes = ana.report_class.value_counts()
    class_counts_spectra = m.report_class.value_counts()
    source_spectra = m.source.value_counts()
    source_analytes = m.groupby("source").analyte.nunique()
    exc_counts = m.excitation_nm.value_counts()
    spa = ana.n_spectra                                          # spectra per analyte

    # cross-source duplicate analytes (same canonical analyte from >1 dataset)
    dup = ana[ana.n_sources > 1][["analyte", "report_class", "sources", "n_spectra"]]
    dup.to_csv(TAB / "corpus_cross_source_duplicates.csv", index=False)

    summary = {
        "n_spectra": int(len(m)), "n_analytes": int(m.analyte.nunique()),
        "n_bins": int(c.X.shape[1]), "window_cm": list(DS.WINDOW), "grid_step_cm": 2.0,
        "sources_spectra": source_spectra.to_dict(),
        "sources_analytes": source_analytes.to_dict(),
        "excitations_spectra": {str(k): int(v) for k, v in exc_counts.items()},
        "class_counts_analytes": class_counts_analytes.to_dict(),
        "class_counts_spectra": class_counts_spectra.to_dict(),
        "raw_family_counts_analytes": ana.family.value_counts().to_dict(),
        "spectra_per_analyte": {
            "min": int(spa.min()), "median": float(spa.median()), "mean": float(spa.mean()),
            "max": int(spa.max()), "n_singletons": int((spa == 1).sum()),
            "n_with_replicates": int((spa > 1).sum())},
        "n_cross_source_duplicates": int(len(dup)),
        "n_analytes_multi_excitation": int((ana.excitations.map(len) > 1).sum()),
    }
    (TAB / "corpus_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    # ── figures ──
    # 1. class balance (by analytes)
    cc = class_counts_analytes.sort_values()
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.barh(cc.index, cc.values, color=ACCENT, zorder=3)
    for i, v in enumerate(cc.values):
        ax.text(v + 0.4, i, str(int(v)), va="center", fontsize=8, color=INK)
    ax.set_xlabel("unique analytes"); ax.set_title("Raman foundation corpus — chemical-class balance",
                                                   color=INK, fontsize=11)
    style(ax); fig.tight_layout(); fig.savefig(FIG / "class_balance_analytes.png", dpi=130); plt.close(fig)

    # 2. spectra per analyte (histogram)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.hist(spa, bins=range(1, int(spa.max()) + 2), color=ACCENT, zorder=3, align="left", rwidth=0.85)
    ax.set_xlabel("spectra per analyte"); ax.set_ylabel("number of analytes")
    ax.set_title(f"Spectra per analyte (median {spa.median():.0f}, max {spa.max()})",
                 color=INK, fontsize=11)
    style(ax); ax.grid(axis="y", color=GRID, linewidth=0.6)
    fig.tight_layout(); fig.savefig(FIG / "spectra_per_analyte.png", dpi=130); plt.close(fig)

    # 3. spectra + analytes per source
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    srcs = list(source_spectra.index); x = np.arange(len(srcs)); w = 0.4
    ax.bar(x - w / 2, source_spectra.values, w, label="spectra", color=ACCENT, zorder=3)
    ax.bar(x + w / 2, [source_analytes[s] for s in srcs], w, label="unique analytes",
           color="#b2182b", zorder=3)
    ax.set_xticks(x); ax.set_xticklabels([s.replace("_", "\n") for s in srcs], fontsize=8)
    ax.set_ylabel("count"); ax.legend(fontsize=8, frameon=False)
    ax.set_title("Per-source contribution", color=INK, fontsize=11)
    style(ax); ax.grid(axis="y", color=GRID, linewidth=0.6)
    fig.tight_layout(); fig.savefig(FIG / "per_source.png", dpi=130); plt.close(fig)

    # 4. excitation distribution
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    exc = exc_counts.sort_index()
    ax.bar([str(int(k)) for k in exc.index], exc.values, color=ACCENT, zorder=3)
    ax.set_xlabel("excitation wavelength (nm)"); ax.set_ylabel("spectra")
    ax.set_title("Excitation-wavelength distribution (tracked as nuisance)", color=INK, fontsize=11)
    style(ax); ax.grid(axis="y", color=GRID, linewidth=0.6)
    fig.tight_layout(); fig.savefig(FIG / "excitation_distribution.png", dpi=130); plt.close(fig)

    print(json.dumps(summary, indent=2, default=str))
    print("\ncross-source duplicate analytes:", len(dup))
    print(dup.to_string(index=False))


if __name__ == "__main__":
    main()
