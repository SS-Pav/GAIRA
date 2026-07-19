"""Phases C2-C7 — manifold, axes, BSV, MSS, external validation, biological projection."""
from __future__ import annotations
import sys, json, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

REPO = Path("/Users/surajpg/projects/GAIRA"); sys.path.insert(0, str(REPO / "src"))
from gaira.foundation import (dataset as DS, latent_space as LS, axes as AX, bsv as BSV,
                              mss as MSS, validation as VAL, projection as PRJ,
                              serialization as SER)

OUT = REPO / "results/v5_rebuild/foundation"
TAB, ART, CFG = OUT / "tables", OUT / "artifacts", OUT / "configs"
for p in (TAB, ART, CFG): p.mkdir(parents=True, exist_ok=True)


def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def main():
    t0 = time.time()
    sel = json.loads((TAB / "c1_selection.json").read_text())
    NAME, K = sel["representation"], int(sel["k"])
    corpus = DS.load_reference_corpus()
    card = DS.dataset_card(corpus)
    log(f"corpus {card['n_spectra']} spectra / {card['n_analytes']} analytes | selected {NAME} k={K}")

    # ── C2 — manifold ──
    log("C2 building + freezing manifold …")
    man = LS.build_manifold(corpus, NAME, K, seed=0)
    st = man.stats
    log(f"  explained variance {st['explained_variance']:.3f} | "
        f"intrinsic dim (participation ratio) {st['intrinsic_dimensionality']['participation_ratio']:.1f} | "
        f"90% latent var at {st['intrinsic_dimensionality']['n_components_90pct_latent_var']} comps | "
        f"bootstrap stability {st['bootstrap_component_stability']['mean']:.3f}")
    (TAB / "c2_manifold_stats.json").write_text(json.dumps(st, indent=2, default=str))

    # ── C3 — emergent axes ──
    log("C3 discovering biochemical axes …")
    peaks_df = DS.load_peak_assignments()
    Zref = man.project(corpus.X)
    axes_list, comp_df, A, cl_stats = AX.build_axes(Zref, corpus.meta, man.rep.components_,
                                                    corpus.grid, peaks_df, max_axes=10)
    log(f"  {cl_stats['n_axes']} axes (silhouette {cl_stats['silhouette']:.3f})")
    for ax in axes_list:
        log(f"    axis {ax['axis']}: {ax['tentative_theme']:22s} conf={ax['theme_confidence']:12s} "
            f"ncomp={ax['n_components']} share={ax['variance_share']:.3f} "
            f"top={ax['top_analytes'][:4]}")
    (TAB / "c3_axes.json").write_text(json.dumps({"clustering": cl_stats, "axes": axes_list},
                                                 indent=2, default=str))
    comp_df.to_csv(TAB / "c3_components.csv", index=False)
    A.to_csv(TAB / "c3_analyte_activation_matrix.csv")

    # ── C4 — BSV ──
    log("C4 building BSV …")
    bsv_df, axis_df, Zn = BSV.build_bsv(man, corpus.X, corpus.meta, axes_list)
    bsv_df.to_csv(TAB / "c4_bsv_components.csv", index=False)
    if axis_df is not None:
        axis_df.to_csv(TAB / "c4_bsv_axes.csv", index=False)
    (TAB / "c4_bsv_summary.json").write_text(json.dumps(BSV.bsv_uncertainty_summary(bsv_df), indent=2))
    log(f"  BSV for {len(bsv_df)} analytes | median uncertainty "
        f"{bsv_df.mean_uncertainty.median():.4f}")

    # ── C5 — MSS ──
    log("C5 deriving molecular spectral signatures …")
    mss_df = MSS.build_mss(man, corpus.X, corpus.meta, axes_list, comp_df)
    mss_df.to_csv(TAB / "c5_mss.csv", index=False)
    log(f"  MSS for {len(mss_df)} analytes | median components used "
        f"{mss_df.n_components_used.median():.0f}/{K} (sparsity {mss_df.sparsity.median():.2f})")

    # ── C6 — external validation ──
    log("C6 external validation (no retraining) …")
    v1 = VAL.heldout_analyte_projection(corpus, NAME, K, n_splits=4, seed=0)
    v1.to_csv(TAB / "c6_heldout_analyte_projection.csv", index=False)
    exc_df, exc_null = VAL.excitation_transfer(man, corpus)
    src_df, src_null = VAL.source_transfer(man, corpus)
    exc_df.to_csv(TAB / "c6_excitation_transfer.csv", index=False)
    src_df.to_csv(TAB / "c6_source_transfer.csv", index=False)
    Xext, mext = DS.load_serum_raman()
    blanks = VAL.blank_control(man, Xext, mext, corpus) if Xext is not None else {}
    v_summary = {
        "excluded_out_of_domain": VAL.EXCLUDED_OUT_OF_DOMAIN,
        "heldout_analyte": {"neighbourhood_preservation": float(v1.neighbourhood_preservation.mean()),
                            "bsv_margin": float(v1.bsv_margin.mean()),
                            "within_analyte_cos": float(v1.within_analyte_cos.mean()),
                            "between_analyte_cos": float(v1.between_analyte_cos.mean()),
                            "recon_rel_error": float(v1.recon_rel_error.mean())},
        "excitation_transfer": {"n_analytes": int(len(exc_df)),
                                "mean_cross_excitation_cos": float(exc_df.cross_level_cos.mean()) if len(exc_df) else None,
                                "null_different_analyte_cos": exc_null},
        "source_transfer": {"n_analytes": int(len(src_df)),
                            "mean_cross_source_cos": float(src_df.cross_level_cos.mean()) if len(src_df) else None,
                            "null_different_analyte_cos": src_null},
        "blank_control": blanks,
    }
    (TAB / "c6_validation_summary.json").write_text(json.dumps(v_summary, indent=2, default=str))
    log(f"  held-out analytes: nbr {v_summary['heldout_analyte']['neighbourhood_preservation']:.3f}, "
        f"BSV margin {v_summary['heldout_analyte']['bsv_margin']:+.3f}")
    if len(exc_df):
        log(f"  excitation transfer: {exc_df.cross_level_cos.mean():.3f} vs null {exc_null:.3f} "
            f"(n={len(exc_df)})")
    if len(src_df):
        log(f"  source transfer:     {src_df.cross_level_cos.mean():.3f} vs null {src_null:.3f} "
            f"(n={len(src_df)})")

    # ── C7 — biological projection ──
    log("C7 projecting biological serum Raman …")
    c7 = {}
    if Xext is not None:
        proj, Zx = PRJ.project(man, Xext, mext, axes_list)
        proj["ood_score"] = PRJ.ood_score(man, Xext, corpus)
        axis_cols = [c for c in proj.columns if c.startswith("axis")]
        proj.to_csv(TAB / "c7_serum_projection.csv", index=False)
        gs = PRJ.group_summary(proj, axis_cols)
        gs.to_csv(TAB / "c7_serum_group_summary.csv", index=False)
        nn = PRJ.nearest_references(man, Xext, mext, corpus)
        nn.to_csv(TAB / "c7_serum_nearest_references.csv", index=False)
        c7 = {"n_spectra": int(len(proj)),
              "groups": {k: int(v) for k, v in mext.group.value_counts().items()},
              "axis_columns": axis_cols,
              "median_ood": float(proj.ood_score.median()),
              "ood_by_group": proj.groupby("group").ood_score.median().round(4).to_dict()}
        (TAB / "c7_projection_summary.json").write_text(json.dumps(c7, indent=2, default=str))
        log(f"  projected {len(proj)} serum spectra | median OOD {proj.ood_score.median():.3f}")
        log(f"  OOD by group: {c7['ood_by_group']}")

    # ── freeze ──
    fp = SER.freeze_manifold(man, ART, corpus_card=card,
                             extra={"selection": sel, "axes": axes_list,
                                    "validation": v_summary, "projection": c7})
    log(f"FROZEN manifold fingerprint {fp}")
    log(f"runtime {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
