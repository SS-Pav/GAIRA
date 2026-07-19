"""Adversarial control for the peak-correspondence result (READ-ONLY).

The Ag-SERS mean spectra yield ~46 detected features spaced ~24 cm-1, versus ~12
Raman bands spaced ~75 cm-1. With a +/-12 cm-1 tolerance a Raman band will often
find SOME Ag-SERS feature by chance. This script quantifies that chance level so
the observed correspondence can be judged against it.

Nulls:
  (1) MISMATCHED-ANALYTE — match analyte A's Raman peaks to a DIFFERENT analyte's
      Ag-SERS peaks. Analyte-specific correspondence must beat this.
  (2) UNIFORM-RANDOM — random Ag-SERS peak positions with the same count.
Sensitivity: repeat with the Ag-SERS peak list truncated to the top-K most prominent
features (K = number of Raman bands), i.e. comparing like with like.
"""
from __future__ import annotations
import sys, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

REPO = Path("/Users/surajpg/projects/GAIRA"); sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).parent))
import audit_lib as AL

AUD = REPO / "results/v5_rebuild/spectral_audit"
TAB = AUD / "tables"
TOL = AL.MATCH_TOL
RNG = np.random.default_rng(0)


def recall(rp, sp, tol=TOL):
    """Fraction of Raman peaks with an Ag-SERS peak within tol (optimal assignment)."""
    if not rp:
        return np.nan
    _, st = AL.match_peaks(rp, sp, tol)
    return st["peak_recall"]


def topk(sp, k):
    return sorted(sp, key=lambda p: -p["prominence"])[:max(1, k)]


def main():
    import pickle
    with open(AUD / "code" / "_audit_store.pkl", "rb") as f:
        blob = pickle.load(f)
    store, grid = blob["store"], blob["grid"]
    analytes = list(store.keys())
    lo, hi = float(grid.min()), float(grid.max())

    rows = []
    for a in analytes:
        rp, sp = store[a]["rp"], store[a]["sp"]
        k = len(rp)
        obs_all = recall(rp, sp)
        obs_topk = recall(rp, topk(sp, k))

        # (1) mismatched-analyte null
        others = [b for b in analytes if b != a]
        mis_all, mis_topk = [], []
        for b in others:
            spb = store[b]["sp"]
            mis_all.append(recall(rp, spb))
            mis_topk.append(recall(rp, topk(spb, k)))
        # (2) uniform-random null (same count as the real SERS peak list)
        rnd_all, rnd_topk = [], []
        for _ in range(50):
            pos = np.sort(RNG.uniform(lo, hi, size=len(sp)))
            fake = [{"position": float(p), "prominence": 1.0, "rel_intensity": 0.5} for p in pos]
            rnd_all.append(recall(rp, fake))
            posk = np.sort(RNG.uniform(lo, hi, size=k))
            fakek = [{"position": float(p), "prominence": 1.0, "rel_intensity": 0.5} for p in posk]
            rnd_topk.append(recall(rp, fakek))

        rows.append({"analyte": a, "n_raman_peaks": k, "n_sers_peaks": len(sp),
                     "recall_observed_allSERS": obs_all,
                     "recall_mismatched_mean": float(np.nanmean(mis_all)),
                     "recall_mismatched_p95": float(np.nanpercentile(mis_all, 95)),
                     "recall_random_mean": float(np.nanmean(rnd_all)),
                     "excess_over_mismatched": obs_all - float(np.nanmean(mis_all)),
                     "recall_observed_topK": obs_topk,
                     "recall_mismatched_topK_mean": float(np.nanmean(mis_topk)),
                     "recall_random_topK_mean": float(np.nanmean(rnd_topk)),
                     "excess_topK": obs_topk - float(np.nanmean(mis_topk)),
                     # analyte-specificity p-value: how often a mismatched analyte does >= as well
                     "p_specificity": float((np.sum(np.array(mis_all) >= obs_all) + 1) / (len(mis_all) + 1)),
                     "p_specificity_topK": float((np.sum(np.array(mis_topk) >= obs_topk) + 1) / (len(mis_topk) + 1))})

    df = pd.DataFrame(rows)
    df.to_csv(TAB / "peak_correspondence_null_controls.csv", index=False)

    summ = {
        "tolerance_cm": TOL,
        "all_SERS_peaks": {
            "observed_recall_median": float(df.recall_observed_allSERS.median()),
            "mismatched_null_median": float(df.recall_mismatched_mean.median()),
            "random_null_median": float(df.recall_random_mean.median()),
            "median_excess_over_mismatched": float(df.excess_over_mismatched.median()),
            "n_analytes_specific_p05": int((df.p_specificity < 0.05).sum()),
        },
        "topK_matched_counts": {
            "observed_recall_median": float(df.recall_observed_topK.median()),
            "mismatched_null_median": float(df.recall_mismatched_topK_mean.median()),
            "random_null_median": float(df.recall_random_topK_mean.median()),
            "median_excess_over_mismatched": float(df.excess_topK.median()),
            "n_analytes_specific_p05": int((df.p_specificity_topK < 0.05).sum()),
        },
        "n_analytes": len(df),
    }
    (TAB / "null_control_summary.json").write_text(json.dumps(summ, indent=2))

    print("=== PEAK-CORRESPONDENCE NULL CONTROLS ===")
    for key in ("all_SERS_peaks", "topK_matched_counts"):
        s = summ[key]
        print(f"\n[{key}]")
        print(f"  observed recall (median)        {s['observed_recall_median']:.3f}")
        print(f"  MISMATCHED-analyte null (median){s['mismatched_null_median']:8.3f}")
        print(f"  uniform-random null (median)    {s['random_null_median']:.3f}")
        print(f"  excess over mismatched          {s['median_excess_over_mismatched']:+.3f}")
        print(f"  analytes with p<0.05 specificity{s['n_analytes_specific_p05']:4d}/{len(df)}")
    return df


if __name__ == "__main__":
    main()
