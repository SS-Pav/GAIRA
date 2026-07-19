"""Matched-analyte Raman / Ag-SERS spectral audit — main analysis (READ-ONLY).

Computes, for all 51 matched analytes of the FROZEN Stage B corpus under the EXACT
Stage B SNV pipeline: peak tables, peak-correspondence matrices, multi-metric
similarity, rigid-shift alignment scan, intensity-redistribution mechanism,
band-level comparison, within-modality reproducibility controls, global
distributions, family analysis, and spectroscopic interpretations.

Writes tables/ + a pickle consumed by make_pdf.py. Modifies nothing.
"""
from __future__ import annotations
import sys, json, pickle, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

REPO = Path("/Users/surajpg/projects/GAIRA"); sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from gaira.evidence import datasets as D
from gaira.evidence.families import family_of, is_ambiguous
import audit_lib as AL
from interpret import interpret_analyte

AUD = REPO / "results/v5_rebuild/spectral_audit"
TAB, FIG = AUD / "tables", AUD / "figures"
for p in (TAB, FIG): p.mkdir(parents=True, exist_ok=True)
PREPROC = "A2_asls_savgol_snv"


def main():
    t0 = time.time()
    d = D.build(PREPROC)
    matched = d.matched_analytes
    print(f"corpus {d.X.shape} | matched analytes {len(matched)} | preproc {PREPROC}", flush=True)

    per, corr_rows, rpk_rows, spk_rows, band_rows = [], [], [], [], []
    store = {}
    for ai, a in enumerate(matched):
        mr = (d.meta.analyte == a) & (d.meta.modality == "raman")
        ms = (d.meta.analyte == a) & (d.meta.modality == "sers")
        R, S = d.X[mr.values], d.X[ms.values]
        rmeta, smeta = d.meta[mr], d.meta[ms]
        rm, sm = np.nanmean(R, axis=0), np.nanmean(S, axis=0)
        rsd, ssd = np.nanstd(R, axis=0), np.nanstd(S, axis=0)

        rp = AL.detect_peaks(rm, d.grid)
        sp = AL.detect_peaks(sm, d.grid)
        rows, st = AL.match_peaks(rp, sp)
        sim = AL.similarity_metrics(rm, sm, d.grid, rp, sp)
        ali = AL.alignment_scan(rm, sm, d.grid, rp, sp)
        red = AL.intensity_redistribution(rows)
        bands = AL.band_analysis(rm, sm, d.grid, rp, sp)
        wr = AL.within_modality_similarity(list(R))
        ws = AL.within_modality_similarity(list(S))

        fam = family_of(a)
        rec = {"analyte": a, "family": fam, "family_ambiguous": is_ambiguous(a),
               "n_raman": int(mr.sum()), "n_sers": int(ms.sum()),
               "raman_sources": "|".join(sorted(rmeta.source.unique())),
               "sers_sources": "|".join(sorted(smeta.source.unique())),
               "raman_multi_source": rmeta.source.nunique() > 1,
               "within_raman_cos": wr, "within_sers_cos": ws,
               **sim, **st, **{f"align_{k}": v for k, v in ali.items()},
               **{f"red_{k}": v for k, v in red.items() if k != "insufficient"}}
        # cross-modal similarity relative to the reproducibility ceiling
        ceiling = np.nanmean([wr, ws])
        rec["reproducibility_ceiling"] = float(ceiling)
        rec["cosine_vs_ceiling"] = float(sim["cosine"] / ceiling) if ceiling and ceiling > 0 else np.nan
        rec["interpretation"] = interpret_analyte(a, fam, rec, rows, bands, d.grid)
        per.append(rec)

        for r in rows:
            corr_rows.append({"analyte": a, "family": fam, **r})
        for p in rp:
            rpk_rows.append({"analyte": a, "modality": "raman", **p})
        for p in sp:
            spk_rows.append({"analyte": a, "modality": "sers", **p})
        for b in bands:
            band_rows.append({"analyte": a, "family": fam, **b})

        store[a] = {"R": R, "S": S, "rm": rm, "sm": sm, "rsd": rsd, "ssd": ssd,
                    "rp": rp, "sp": sp, "rows": rows, "sim": sim, "ali": ali,
                    "red": red, "bands": bands, "rec": rec,
                    "rmeta": rmeta.to_dict("records"), "smeta": smeta.to_dict("records")}
        print(f"  [{ai+1:2d}/51] {a:26s} cos={sim['cosine']:+.3f} "
              f"peakF1={st['peak_f1']:.2f} shift={st['mean_abs_shift']:.1f} "
              f"PCS={st['peak_correspondence_score']:.2f}", flush=True)

    df = pd.DataFrame(per)
    df.to_csv(TAB / "per_analyte_summary.csv", index=False)
    pd.DataFrame(corr_rows).to_csv(TAB / "peak_correspondence_matrix.csv", index=False)
    pd.DataFrame(rpk_rows).to_csv(TAB / "peak_table_raman.csv", index=False)
    pd.DataFrame(spk_rows).to_csv(TAB / "peak_table_sers.csv", index=False)
    pd.DataFrame(band_rows).to_csv(TAB / "band_level_comparison.csv", index=False)

    # ── global distributions + rankings (Part 9) ──
    def top(col, n=10, asc=False):
        s = df.dropna(subset=[col]).sort_values(col, ascending=asc).head(n)
        return [{"analyte": r.analyte, "family": r.family, col: float(getattr(r, col))}
                for r in s.itertuples()]
    metrics = ["cosine", "pearson", "spearman", "spectral_angle_deg", "nrmse", "dtw",
               "derivative_corr", "peak_f1", "peak_jaccard", "mean_abs_shift",
               "matched_pct_of_raman", "unmatched_pct_of_raman",
               "peak_correspondence_score", "red_intensity_redistribution_index",
               "red_peak_rank_corr", "red_norm_intensity_corr", "red_band_ratio_preservation",
               "within_raman_cos", "within_sers_cos", "cosine_vs_ceiling",
               "align_optimal_shift_cm", "align_cosine_gain"]
    dist = {m: {"mean": float(df[m].mean()), "median": float(df[m].median()),
                "std": float(df[m].std()), "min": float(df[m].min()), "max": float(df[m].max()),
                "q25": float(df[m].quantile(.25)), "q75": float(df[m].quantile(.75))}
            for m in metrics if m in df and df[m].notna().any()}
    rankings = {
        "top10_best_match_cosine": top("cosine"), "top10_worst_match_cosine": top("cosine", asc=True),
        "top10_best_peak_correspondence": top("peak_correspondence_score"),
        "top10_worst_peak_correspondence": top("peak_correspondence_score", asc=True),
        "top10_largest_mean_abs_shift": top("mean_abs_shift"),
        "top10_strongest_intensity_redistribution": top("red_intensity_redistribution_index"),
        "top10_visually_identical": top("peak_f1"),
        "top10_visually_different": top("peak_f1", asc=True),
    }
    (TAB / "global_statistics.json").write_text(json.dumps(
        {"n_matched_analytes": len(matched), "preprocessing": PREPROC,
         "match_tolerance_cm": AL.MATCH_TOL, "distributions": dist,
         "rankings": rankings}, indent=2, default=float))

    # ── family analysis (Part 10) ──
    fam = (df.groupby("family").agg(
        n=("analyte", "count"), cosine=("cosine", "mean"), pearson=("pearson", "mean"),
        peak_f1=("peak_f1", "mean"), mean_abs_shift=("mean_abs_shift", "mean"),
        pcs=("peak_correspondence_score", "mean"),
        redistribution=("red_intensity_redistribution_index", "mean"),
        rank_corr=("red_peak_rank_corr", "mean"),
        matched_pct=("matched_pct_of_raman", "mean"),
        ceiling=("reproducibility_ceiling", "mean")).reset_index()
        .sort_values("pcs", ascending=False))
    fam.to_csv(TAB / "family_analysis.csv", index=False)

    with open(AUD / "code" / "_audit_store.pkl", "wb") as f:
        pickle.dump({"store": store, "grid": d.grid, "df": df, "fam": fam,
                     "dist": dist, "rankings": rankings, "preproc": PREPROC}, f)

    print(f"\n=== GLOBAL SUMMARY (n={len(matched)}) ===")
    for k in ("cosine", "pearson", "peak_f1", "mean_abs_shift", "matched_pct_of_raman",
              "peak_correspondence_score", "red_peak_rank_corr",
              "red_intensity_redistribution_index", "within_raman_cos", "within_sers_cos",
              "align_optimal_shift_cm", "align_cosine_gain"):
        if k in dist:
            print(f"  {k:38s} median={dist[k]['median']:+.3f}  mean={dist[k]['mean']:+.3f}")
    print(f"\nruntime {time.time()-t0:.1f}s")
    return df


if __name__ == "__main__":
    main()
