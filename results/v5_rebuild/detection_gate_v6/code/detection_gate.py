"""GAIRA V6 — Stage 0 Detection Gate.

Asks a MEASUREMENT question, not a GAIRA question: "does this Ag-SERS spectrum actually contain
reproducible analyte information above noise/background?" — separating measurement failure (analyte
invisible on silver) from representation failure (measured but chemistry not recovered).

Deterministic weighted Detection Confidence (0-1) from complementary spectroscopic metrics; NO ML.
Thresholds are VALIDATED interactively (validate_detection notebook) before being frozen here.
Uses raw pure Ag-SERS replicate spectra + the serum-blank Ag background. Frozen atlas untouched.

Outputs: results/v5_rebuild/detection_gate_v6/{tables,artifacts}.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr
from scipy.signal import find_peaks

REPO = Path("/Users/surajpg/projects/GAIRA")
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "results/v5_rebuild/spike_validation/code"))
from gaira.engine import GAIRAEngine
from gaira.foundation.families_raman import family_of
from gaira.data.synonyms import canonical
import spike_lib as SL

OUT = REPO / "results/v5_rebuild/detection_gate_v6"
(OUT / "tables").mkdir(parents=True, exist_ok=True); (OUT / "artifacts").mkdir(parents=True, exist_ok=True)
CANON = "09ed804a40836f4a05a91ba10900cded"

# ─────────────────────────────────────────────────────────────────────────────
# FROZEN, VALIDATED detection parameters (validated in validate_detection.ipynb;
# confirmed adenine/ergothioneine/urate/xanthine/hypoxanthine/guanine PASS and
# glucose/tyrosine/oleate/serine/alanine FAIL for physical adsorption reasons).
DET_WEIGHTS = {"rep_pearson": 0.45, "rep_spearman": 0.10, "peak_snr": 0.20,
               "var_concentration": 0.15, "repro_peaks": 0.10}
SNR_CAP = 100.0            # local peak SNR mapped log-scaled to [0,1] at this cap
VCONC_CAP = 0.25          # top-20-bin variance fraction mapped to [0,1] at this cap
PK_CAP = 12               # reproducible-peak count mapped to [0,1] at this cap
TIERS = [("GOOD", 0.65), ("MODERATE", 0.50), ("POOR", 0.40), ("UNDETECTABLE", 0.0)]
DETECTION_PASS = 0.50     # DC >= this (GOOD or MODERATE) passes Stage 0
# ─────────────────────────────────────────────────────────────────────────────


def tier(dc):
    for name, t in TIERS:
        if dc >= t:
            return name
    return "UNDETECTABLE"


def _metrics(Xi, blank_mean, blank_std):
    """All detection metrics for one analyte's replicate matrix (n x P)."""
    Xi = np.nan_to_num(Xi); n = len(Xi)
    pear, spear = [], []
    for i in range(n):
        for j in range(i + 1, n):
            pear.append(pearsonr(Xi[i], Xi[j])[0]); spear.append(spearmanr(Xi[i], Xi[j]).correlation)
    mean = Xi.mean(0); std = Xi.std(0) + 1e-9
    mn = mean - np.median(mean); mnorm = mn / (mn.max() + 1e-12)
    pk, _ = find_peaks(mnorm, prominence=0.05, distance=4)
    local_snr = mn[pk] / std[pk] if len(pk) else np.array([0.0])
    repro_peaks = int(np.sum(local_snr > 3.0))
    max_snr = float(local_snr.max()) if len(pk) else 0.0
    v = std ** 2; var_conc = float(np.sort(v)[-20:].sum() / (v.sum() + 1e-12))
    # blank-adjusted signal: peak excursion above blank mean vs blank noise
    blank_snr = float(np.max(mean - blank_mean) / (np.median(blank_std) + 1e-9))
    return {"rep_pearson": float(np.nanmean(pear)), "rep_spearman": float(np.nanmean(spear)),
            "repro_peaks": repro_peaks, "peak_snr": max_snr, "var_concentration": var_conc,
            "blank_snr": blank_snr, "n_replicates": n}


def detection_confidence(m):
    """Transparent deterministic weighted score in [0,1]."""
    parts = {
        "rep_pearson": np.clip(m["rep_pearson"], 0, 1),
        "rep_spearman": np.clip(m["rep_spearman"], 0, 1),
        "peak_snr": np.clip(np.log10(m["peak_snr"] + 1) / np.log10(SNR_CAP), 0, 1),
        "var_concentration": np.clip(m["var_concentration"] / VCONC_CAP, 0, 1),
        "repro_peaks": np.clip(m["repro_peaks"] / PK_CAP, 0, 1),
    }
    dc = float(sum(DET_WEIGHTS[k] * parts[k] for k in DET_WEIGHTS))
    return dc, {k: round(float(v), 4) for k, v in parts.items()}


def main():
    eng = GAIRAEngine()
    assert eng.atlas.meta["fingerprint"] == CANON, "FROZEN ATLAS CHANGED"
    grid = np.asarray(getattr(eng.atlas, "grid", np.arange(676)), float)
    X, rs = SL.load_pure_sers(); Xb, _ = SL.load_serum_baseline()
    blank_mean = np.nan_to_num(Xb).mean(0); blank_std = np.nan_to_num(Xb).std(0)
    rs_an = np.array([canonical(a) for a in rs.analyte.values])
    # restrict to the 51 matched analytes (same set as V4/V5)
    v5 = pd.read_csv(REPO / "results/v5_rebuild/abstraction_recovery_v5/tables/per_analyte_abstraction_recovery.csv")
    matched = sorted(set(v5.analyte) & set(rs_an))

    rows = []; spectra = {}
    for a in matched:
        Xi = X[rs_an == a]
        m = _metrics(Xi, blank_mean, blank_std)
        dc, parts = detection_confidence(m)
        rows.append({"analyte": a, "broad_family": family_of(a), **m,
                     "detection_confidence": round(dc, 4), "detection_tier": tier(dc),
                     "detection_pass": dc >= DETECTION_PASS,
                     **{f"score_{k}": v for k, v in parts.items()}})
        spectra[a] = np.nan_to_num(Xi).mean(0)
    df = pd.DataFrame(rows).sort_values("detection_confidence", ascending=False).reset_index(drop=True)
    df.to_csv(OUT / "tables/detection_metrics.csv", index=False)

    tier_counts = df.detection_tier.value_counts().to_dict()
    summary = {
        "atlas_fingerprint": CANON, "n_analytes": len(df),
        "weights": DET_WEIGHTS, "snr_cap": SNR_CAP, "vconc_cap": VCONC_CAP, "pk_cap": PK_CAP,
        "tier_thresholds": {n: t for n, t in TIERS}, "detection_pass_threshold": DETECTION_PASS,
        "tier_counts": tier_counts,
        "n_pass": int(df.detection_pass.sum()), "n_fail": int((~df.detection_pass).sum()),
        "pass_analytes": df[df.detection_pass].analyte.tolist(),
        "fail_analytes": df[~df.detection_pass].analyte.tolist(),
        "validation_anchors": {
            "expected_pass": {a: {"dc": float(df[df.analyte == a].detection_confidence.iloc[0]),
                                  "tier": df[df.analyte == a].detection_tier.iloc[0]}
                              for a in ["adenine", "ergothioneine", "urate", "xanthine"] if a in set(df.analyte)},
            "expected_fail": {a: {"dc": float(df[df.analyte == a].detection_confidence.iloc[0]),
                                  "tier": df[df.analyte == a].detection_tier.iloc[0]}
                              for a in ["glucose", "tyrosine", "oleate"] if a in set(df.analyte)}},
    }
    (OUT / "artifacts/detection_summary.json").write_text(json.dumps(summary, indent=2))
    # save mean spectra + blank + grid for figures
    np.savez(OUT / "artifacts/detection_spectra.npz", analytes=np.array(matched),
             grid=grid, blank_mean=blank_mean, blank_std=blank_std,
             spectra=np.array([spectra[a] for a in matched]))

    print(json.dumps({"fingerprint": CANON, "n": len(df), "tier_counts": tier_counts,
                      "n_pass": summary["n_pass"], "n_fail": summary["n_fail"],
                      "anchors": summary["validation_anchors"]}, indent=2))
    print("\ndetection ranking:")
    print(df[["analyte", "broad_family", "rep_pearson", "peak_snr", "var_concentration",
              "repro_peaks", "detection_confidence", "detection_tier"]].to_string(index=False))


if __name__ == "__main__":
    main()
