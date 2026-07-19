"""GAIRA V5 Phase 1.5 — Canonical grounding corpus completion (785 nm only).

Assembles the 785 nm direct-grounding corpus (RamanBioLib 785 subset + adenine
Ag-SERS + Gobbato pure Raman + pure Ag-SERS), reconciles analyte identities,
rebuilds the grounding summary, and quantifies 785 Raman/Ag-SERS overlap.
NO PCA / clustering / NMF / embeddings / ontology / observation model / BSV / MSS.
Read-only. Outputs under results/v5_rebuild/phase1_5/.
"""
from __future__ import annotations
import sys, json, warnings, collections
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/Users/surajpg/projects/GAIRA"); sys.path.insert(0, str(REPO / "src"))
from gaira.data import loader, gobbato                     # noqa
from gaira.data.synonyms import canonical                   # noqa

PH = REPO / "results/v5_rebuild/phase1_5"
FIG = PH / "figures"; TAB = PH / "tables"
for d in (FIG, TAB): d.mkdir(parents=True, exist_ok=True)


def build_785_corpus():
    included, excluded = [], []
    # RamanBioLib: keep 785 only
    for s in loader.load_ramanbiolib():
        (included if s.record.excitation_nm == 785.0 else excluded).append(
            (s, "included:785" if s.record.excitation_nm == 785.0 else f"excluded:non-785({s.record.excitation_nm})"))
    # metabolite-63: 633 nm -> EXCLUDED entirely
    for s in loader.load_metabolite63():
        excluded.append((s, "excluded:633nm(not 785)"))
    # adenine Ag-SERS 785
    for s in loader.load_adenine():
        included.append((s, "included:785"))
    # Gobbato pure Raman + Ag-SERS (785)
    for s in gobbato.load_gobbato_785():
        included.append((s, "included:785"))
    # ORC-Ag peak-only -> excluded from representation (kept for MSS)
    for s in loader.load_orc_ag_peaks():
        excluded.append((s, "excluded:peak_only(kept for MSS)"))
    return included, excluded


def main():
    inc, exc = build_785_corpus()
    inc_specs = [s for s, _ in inc]
    # canonical analyte per included spectrum
    rows = []
    for s in inc_specs:
        can = canonical(s.record.canonical_analyte_name)
        rows.append(dict(spectrum_id=s.record.spectrum_id, canonical_analyte=can,
                         raw_name=s.record.canonical_analyte_name, modality=s.record.modality.value,
                         source=s.record.source_dataset, excitation=s.record.excitation_nm,
                         replicate=s.record.replicate))
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "grounding_spectrum_registry_785.csv", index=False)

    # canonical_analyte_registry_v5
    reg = (df.groupby("canonical_analyte")
           .agg(n_raman=("modality", lambda m: int((m == "raman").sum())),
                n_sers=("modality", lambda m: int((m == "sers").sum())),
                sources=("source", lambda s: sorted(set(s))),
                raw_names=("raw_name", lambda s: sorted(set(s)))).reset_index())
    reg["in_raman_785"] = reg.n_raman > 0
    reg["in_sers_785"] = reg.n_sers > 0
    reg["matched_785"] = reg.in_raman_785 & reg.in_sers_785
    reg.to_csv(TAB / "canonical_analyte_registry_v5.csv", index=False)

    raman_an = set(reg[reg.in_raman_785].canonical_analyte)
    sers_an = set(reg[reg.in_sers_785].canonical_analyte)
    matched = raman_an & sers_an
    n_raman_spec = int((df.modality == "raman").sum())
    n_sers_spec = int((df.modality == "sers").sum())

    summary = {
        "unique_analytes_785": int(reg.canonical_analyte.nunique()),
        "n_raman_spectra_785": n_raman_spec,
        "n_sers_spectra_785": n_sers_spec,
        "n_analytes_raman_785": len(raman_an),
        "n_analytes_sers_785": len(sers_an),
        "n_matched_analytes_785": len(matched),
        "pct_analytes_matched": round(100 * len(matched) / max(1, reg.canonical_analyte.nunique()), 1),
        "spectra_entering_representation": len(inc_specs),
        "spectra_excluded": len(exc),
        "exclusion_reasons": dict(collections.Counter(r for _, r in exc)),
        "included_sources": dict(collections.Counter(s.record.source_dataset for s in inc_specs)),
        "matched_analytes": sorted(matched),
        "raman_only": sorted(raman_an - sers_an),
        "sers_only": sorted(sers_an - raman_an),
    }
    (TAB / "phase1_5_grounding_summary.json").write_text(json.dumps(summary, indent=2))
    print("== Phase 1.5 grounding completion (785 nm) ==")
    for k in ("unique_analytes_785", "n_raman_spectra_785", "n_sers_spectra_785",
              "n_analytes_raman_785", "n_analytes_sers_785", "n_matched_analytes_785",
              "pct_analytes_matched", "spectra_entering_representation", "spectra_excluded"):
        print(f"  {k}: {summary[k]}")
    print("  exclusions:", summary["exclusion_reasons"])
    print("  matched analytes:", summary["matched_analytes"])

    # ── FIGURES ──
    # 1. overlap (Raman-only / matched / SERS-only)
    fig, ax = plt.subplots(figsize=(6, 4))
    vals = [len(raman_an - sers_an), len(matched), len(sers_an - raman_an)]
    ax.bar(["Raman-only", "matched\n(Raman∩Ag-SERS)", "Ag-SERS-only"], vals,
           color=["#2563eb", "#16a34a", "#dc2626"])
    for i, v in enumerate(vals): ax.text(i, v + 0.3, str(v), ha="center", fontweight="bold")
    ax.set_ylabel("# canonical analytes (785 nm)")
    ax.set_title(f"785 nm grounding overlap: {len(matched)} matched analytes "
                 f"(Phase 1 had 7)")
    fig.tight_layout(); fig.savefig(FIG / "matched_analyte_overlap_785.png", dpi=130); plt.close(fig)

    # 2. spectra per analyte (matched analytes, Raman vs SERS reps)
    md = reg[reg.matched_785].sort_values("n_sers", ascending=False)
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(md))
    ax.bar(x - 0.2, md.n_raman, 0.4, label="Raman reps", color="#2563eb")
    ax.bar(x + 0.2, md.n_sers, 0.4, label="Ag-SERS reps", color="#dc2626")
    ax.set_xticks(x); ax.set_xticklabels(md.canonical_analyte, rotation=90, fontsize=7)
    ax.set_ylabel("# spectra"); ax.legend(); ax.set_title("Matched analytes: replicate spectra per modality (785 nm)")
    fig.tight_layout(); fig.savefig(FIG / "spectra_per_matched_analyte.png", dpi=130); plt.close(fig)

    # 3. corpus composition
    fig, ax = plt.subplots(figsize=(7, 4))
    comp = df.groupby(["source", "modality"]).size().unstack(fill_value=0)
    comp.plot(kind="barh", stacked=True, ax=ax, color={"raman": "#2563eb", "sers": "#dc2626"})
    ax.set_xlabel("# spectra"); ax.set_title("785 nm grounding corpus composition (included)")
    fig.tight_layout(); fig.savefig(FIG / "corpus_composition_785.png", dpi=130); plt.close(fig)

    print("\nfigures + tables written to results/v5_rebuild/phase1_5/")
    return summary


if __name__ == "__main__":
    main()
