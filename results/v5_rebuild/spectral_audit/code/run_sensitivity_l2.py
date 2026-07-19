"""Preprocessing sensitivity analysis (READ-ONLY, clearly labelled as SECONDARY).

The PRIMARY audit runs on the exact Stage B SNV pipeline. This script repeats the
peak-correspondence analysis on the L2 pipeline ONLY as a diagnostic, because the
audit found that SNV collapses Ag-SERS replicate reproducibility (0.95 -> 0.49),
which means SNV-derived Ag-SERS peak lists partly reflect noise. It answers the
audit questions "are there preprocessing artifacts?" and "should Stage A be rerun?".

Nothing here changes the Stage B corpus, pipeline, or any GAIRA code.
"""
from __future__ import annotations
import sys, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

REPO = Path("/Users/surajpg/projects/GAIRA"); sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from gaira.evidence import datasets as D
import audit_lib as AL

AUD = REPO / "results/v5_rebuild/spectral_audit"; TAB = AUD / "tables"
RNG = np.random.default_rng(0)


def analyse(pipeline):
    d = D.build(pipeline)
    peaks, out = {}, []
    for a in d.matched_analytes:
        R = np.nan_to_num(d.X[((d.meta.analyte == a) & (d.meta.modality == "raman")).values])
        S = np.nan_to_num(d.X[((d.meta.analyte == a) & (d.meta.modality == "sers")).values])
        rm, sm = R.mean(0), S.mean(0)
        rp, sp = AL.detect_peaks(rm, d.grid), AL.detect_peaks(sm, d.grid)
        peaks[a] = (rp, sp)
        rows, st = AL.match_peaks(rp, sp)
        out.append({"analyte": a, "n_raman_peaks": len(rp), "n_sers_peaks": len(sp),
                    "recall": st["peak_recall"], "precision": st["peak_precision"],
                    "f1": st["peak_f1"], "mean_abs_shift": st["mean_abs_shift"],
                    "pcs": st["peak_correspondence_score"]})
    df = pd.DataFrame(out)
    # mismatched-analyte null on recall
    analytes = list(peaks)
    exc, spec = [], 0
    for a in analytes:
        rp, sp = peaks[a]
        obs = AL.match_peaks(rp, sp)[1]["peak_recall"]
        mis = [AL.match_peaks(rp, peaks[b][1])[1]["peak_recall"] for b in analytes if b != a]
        exc.append(obs - float(np.mean(mis)))
        if (np.sum(np.array(mis) >= obs) + 1) / (len(mis) + 1) < 0.05:
            spec += 1
    df["excess_over_mismatched"] = exc
    return df, spec


def main():
    res = {}
    for pipe, label in (("A2_asls_savgol_snv", "SNV (Stage B primary)"),
                        ("A1_asls_savgol_l2", "L2 (sensitivity)")):
        df, spec = analyse(pipe)
        df.to_csv(TAB / f"peak_correspondence_sensitivity_{pipe}.csv", index=False)
        res[pipe] = {
            "label": label,
            "median_n_raman_peaks": float(df.n_raman_peaks.median()),
            "median_n_sers_peaks": float(df.n_sers_peaks.median()),
            "median_recall": float(df.recall.median()),
            "median_precision": float(df.precision.median()),
            "median_f1": float(df.f1.median()),
            "median_mean_abs_shift": float(df.mean_abs_shift.median()),
            "median_pcs": float(df.pcs.median()),
            "median_excess_over_mismatched_null": float(df.excess_over_mismatched.median()),
            "n_analytes_specific_p05": spec, "n_analytes": len(df)}
    (TAB / "preprocessing_sensitivity_summary.json").write_text(json.dumps(res, indent=2))

    print("=== PEAK CORRESPONDENCE: SNV (primary) vs L2 (sensitivity) ===")
    hdr = f"{'metric':38s} {'SNV':>12s} {'L2':>12s}"
    print(hdr); print("-" * len(hdr))
    a, b = res["A2_asls_savgol_snv"], res["A1_asls_savgol_l2"]
    for k in ("median_n_raman_peaks", "median_n_sers_peaks", "median_recall", "median_precision",
              "median_f1", "median_mean_abs_shift", "median_pcs",
              "median_excess_over_mismatched_null"):
        print(f"{k:38s} {a[k]:12.3f} {b[k]:12.3f}")
    print(f"{'analytes specific (p<0.05)':38s} {a['n_analytes_specific_p05']:12d} {b['n_analytes_specific_p05']:12d}")
    return res


if __name__ == "__main__":
    main()
