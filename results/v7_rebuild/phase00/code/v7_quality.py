"""GAIRA V7 Phase 00 — spectrum quality metadata and the frozen quality score `q`.

Strategy B (analyte-balanced weighted fitting) weights each replicate by a quality score:

    w_ai = q_ai / sum_j q_aj      so each canonical molecule contributes total weight 1

`q` must be FROZEN HERE, before Phase 01 runs. If it were tuned against Phase-01 outcomes
it would become a hidden hyperparameter chosen to produce a preferred answer (risk R-10).
It is computed from the PREPROCESSED spectrum alone: it never sees the analyte label, the
class, the fold assignment, or any downstream metric.

WHAT THIS CORPUS ACTUALLY IS, AND WHY THE FIRST DESIGN FAILED
-------------------------------------------------------------
A first version of `q` used classical acquisition-quality terms — cosmic-ray spike count,
detector saturation, first-difference SNR. Measured on this corpus, all three were
degenerate: `spike_free` scored 0.000 for all 375 spectra, `not_saturated` 1.000 for all
375, and the SNR term saturated at its ceiling for all 375. The reason is that these are
*curated reference-library* spectra — digitized, already baseline-corrected and smoothed by
their sources — so acquisition artefacts have been removed upstream. A first-difference
noise estimate on such a spectrum measures the sharpness of genuine Raman bands, not noise.

The score below therefore measures what genuinely varies here: how much resolvable band
structure a reference carries, and how cleanly it stands above its own baseline. This is
narrower than "acquisition quality" and is documented as such rather than dressed up.

COMPONENTS (all in [0, 1], all deterministic)
  snr_score     second-difference noise estimate vs signal, log-scaled (SNR 20 -> 0, 500 -> 1).
                The second difference is used because it is insensitive to the smooth band
                shape a first difference confuses with noise.
  grid_coverage fraction of the 676-bin canonical grid that is finite
  contrast      (p99 - p50) / (p99 - p1) — how far the peaks stand above the body of the
                spectrum; a flat or baseline-dominated spectrum scores low
  peak_density  count of prominent resolvable peaks / 30, clipped — band richness

  q = geometric mean of the four, so one catastrophic component cannot be averaged away.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

Q_VERSION = "v7_q_v2"
QC_FLOOR = 0.35             # below this a spectrum is flagged qc_pass = False
SNR_LO, SNR_HI = 20.0, 500.0
PEAK_TARGET = 30.0
PEAK_PROMINENCE_FRAC = 0.02


def spectrum_quality(y: np.ndarray) -> dict:
    y = np.asarray(y, float)
    n = y.size
    finite = np.isfinite(y)
    grid_coverage = float(finite.sum() / n) if n else 0.0
    x = np.nan_to_num(y)

    # second-difference robust noise estimate (sigma of a 2nd difference is sqrt(6)*sigma)
    d2 = np.diff(x, 2)
    noise = float(np.median(np.abs(d2)) * 1.4826 / np.sqrt(6.0))
    signal = float(np.percentile(x, 99) - np.median(x))
    snr_raw = signal / noise if noise > 0 else (SNR_HI if signal > 0 else 0.0)
    snr_score = float(np.clip(
        np.log10(max(snr_raw, 1.0) / SNR_LO) / np.log10(SNR_HI / SNR_LO), 0.0, 1.0))

    p99, p50, p1 = (float(np.percentile(x, 99)), float(np.median(x)),
                    float(np.percentile(x, 1)))
    contrast = float(np.clip((p99 - p50) / (p99 - p1 + 1e-12), 0.0, 1.0))

    rng = float(x.max() - x.min())
    npk = int(len(find_peaks(x, prominence=PEAK_PROMINENCE_FRAC * rng)[0])) if rng > 0 else 0
    peak_density = float(np.clip(npk / PEAK_TARGET, 0.0, 1.0))

    parts = np.array([snr_score, grid_coverage, contrast, peak_density], float)
    q = float(np.exp(np.mean(np.log(np.clip(parts, 1e-6, 1.0)))))
    return {
        "snr_estimate": round(float(snr_raw), 3),
        "snr_score": round(snr_score, 4),
        "grid_coverage": round(grid_coverage, 5),
        "n_nan_bins": int((~finite).sum()),
        "contrast": round(contrast, 4),
        "n_peaks": npk,
        "peak_density": round(peak_density, 4),
        "quality_score": round(q, 6),
        "qc_pass": bool(q >= QC_FLOOR),
    }


def quality_table(X: np.ndarray, meta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i in range(X.shape[0]):
        r = spectrum_quality(X[i])
        r["spectrum_id"] = meta.spectrum_id.iat[i]
        r["analyte"] = meta.analyte.iat[i]
        r["source"] = meta.source.iat[i]
        r["excitation_nm"] = meta.excitation_nm.iat[i]
        r["q_version"] = Q_VERSION
        rows.append(r)
    cols = ["spectrum_id", "analyte", "source", "excitation_nm", "snr_estimate", "snr_score",
            "grid_coverage", "n_nan_bins", "contrast", "n_peaks", "peak_density",
            "quality_score", "qc_pass", "q_version"]
    return pd.DataFrame(rows)[cols]


def analyte_weights(qual: pd.DataFrame, alias_to_cid: dict[str, str]) -> pd.DataFrame:
    """w_ai = q_ai / sum_j q_aj, summing to exactly 1.0 per CANONICAL molecule.

    `weight_uniform` is the mandatory B-uniform sensitivity arm (q identical for every
    replicate). If the two arms agree in Phase 01, `q` is doing nothing on this corpus —
    which is a result, not a failure.
    """
    df = qual.copy()
    df["canonical_id"] = df.analyte.map(alias_to_cid)
    tot = df.groupby("canonical_id").quality_score.transform("sum")
    df["weight_quality"] = df.quality_score / tot.replace(0, np.nan)
    n = df.groupby("canonical_id").spectrum_id.transform("count")
    df["weight_uniform"] = 1.0 / n
    df["weight_quality"] = df.weight_quality.fillna(df.weight_uniform)
    return df[["spectrum_id", "canonical_id", "quality_score",
               "weight_quality", "weight_uniform", "qc_pass"]]


def quality_summary(qual: pd.DataFrame, weights: pd.DataFrame) -> dict:
    q = qual.quality_score
    by_src = qual.groupby("source").quality_score.median().round(4).to_dict()
    g = qual.groupby("analyte").quality_score
    spread = (g.max() - g.min())[g.count() > 1]
    wsum = weights.groupby("canonical_id").weight_quality.sum()
    return {
        "q_version": Q_VERSION,
        "qc_floor": QC_FLOOR,
        "n_spectra": int(len(qual)),
        "q_min": round(float(q.min()), 4), "q_median": round(float(q.median()), 4),
        "q_max": round(float(q.max()), 4), "q_iqr": round(float(q.quantile(.75) - q.quantile(.25)), 4),
        "q_max_over_min": round(float(q.max() / max(q.min(), 1e-12)), 3),
        "n_below_qc_floor": int((~qual.qc_pass).sum()),
        "median_by_source": by_src,
        "n_spectra_with_nan_bins": int((qual.n_nan_bins > 0).sum()),
        "total_nan_bins": int(qual.n_nan_bins.sum()),
        "within_analyte_spread": {
            "n_replicated_analytes": int(len(spread)),
            "median": round(float(spread.median()), 4) if len(spread) else None,
            "max": round(float(spread.max()), 4) if len(spread) else None,
        },
        "weights_sum_to_one": bool(np.allclose(wsum.values, 1.0, atol=1e-9)),
        "caveat": (
            "This corpus is a curated reference library: acquisition artefacts were removed "
            "upstream, so q measures band structure and contrast rather than acquisition "
            "quality. Because within-analyte spread is small (median 0.03), Strategy B and "
            "the B-uniform arm are expected to be close in Phase 01; the arms are still run "
            "separately so that expectation is tested rather than assumed."
        ),
    }
