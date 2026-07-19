"""Matched-analyte Raman / Ag-SERS spectral audit — analysis library.

READ-ONLY spectroscopic audit. Uses the FROZEN Stage B corpus and the EXACT
Stage B SNV preprocessing (A2_asls_savgol_snv). No models, no refitting, no
preprocessing changes, no pre-alignment of stored spectra.

Note on SNV: SNV is an affine per-spectrum transform ((x-mean)/std). It therefore
PRESERVES peak positions and relative band ratios within a spectrum; it rescales
absolute intensity only. Peak-position analysis below is unaffected by SNV.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import find_peaks, peak_widths, medfilt
from scipy.stats import pearsonr, spearmanr
from scipy.optimize import linear_sum_assignment

# ── spectral regions (Part 7) ──
REGIONS = [(600, 800), (800, 1000), (1000, 1200), (1200, 1400), (1400, 1600), (1600, 1800)]

# peak-matching tolerance (cm-1). Grid step is 2 cm-1.
MATCH_TOL = 12.0
ALIGN_MAX = 10.0          # Part 5: +/-10 cm-1 rigid-shift search


# ─────────────────────── peak detection ───────────────────────
def detect_peaks(y, grid, prom_frac=0.10, min_sep_cm=8.0):
    """Peaks of a (SNV) spectrum. Returns list of dicts with position, prominence,
    width (FWHM, cm-1), relative intensity, local SNR."""
    y = np.nan_to_num(np.asarray(y, float))
    rng = float(y.max() - y.min())
    if rng <= 0:
        return []
    dx = float(np.median(np.diff(grid)))
    dist = max(1, int(round(min_sep_cm / dx)))
    prom = prom_frac * rng
    idx, props = find_peaks(y, prominence=prom, distance=dist)
    if len(idx) == 0:
        return []
    w = peak_widths(y, idx, rel_height=0.5)[0] * dx        # FWHM in cm-1
    # local noise: residual after a short median filter, robust MAD in a local window
    resid = y - medfilt(y, kernel_size=5)
    out = []
    ymax = float(y.max())
    for k, i in enumerate(idx):
        lo, hi = max(0, i - 30), min(len(y), i + 31)
        loc = resid[lo:hi]
        noise = 1.4826 * float(np.median(np.abs(loc - np.median(loc)))) + 1e-9
        out.append({
            "position": float(grid[i]), "index": int(i),
            "prominence": float(props["prominences"][k]),
            "width_cm": float(w[k]),
            "rel_intensity": float((y[i] - y.min()) / rng),
            "height": float(y[i]),
            "snr": float(props["prominences"][k] / noise),
        })
    return sorted(out, key=lambda p: -p["prominence"])


# ─────────────────────── peak correspondence (Part 3) ───────────────────────
def match_peaks(rp, sp, tol=MATCH_TOL):
    """Optimal (Hungarian) assignment of Raman->SERS peaks within `tol` cm-1.
    Returns (rows, stats). rows describe matched / Raman-only / SERS-only peaks."""
    rows = []
    if not rp and not sp:
        return rows, _match_stats(rows, 0, 0)
    matched_r, matched_s = set(), set()
    if rp and sp:
        C = np.abs(np.array([p["position"] for p in rp])[:, None]
                   - np.array([q["position"] for q in sp])[None, :])
        BIG = 1e6
        cost = np.where(C <= tol, C, BIG)
        ri, si = linear_sum_assignment(cost)
        for a, b in zip(ri, si):
            if cost[a, b] >= BIG:
                continue
            r, s = rp[a], sp[b]
            shift = s["position"] - r["position"]
            ratio = s["rel_intensity"] / (r["rel_intensity"] + 1e-9)
            conf = "High" if abs(shift) <= 5 else ("Medium" if abs(shift) <= 8 else "Low")
            note = ("Strong correspondence" if abs(shift) <= 5 and 0.7 <= ratio <= 1.4 else
                    ("Intensity enhanced" if ratio > 1.4 else
                     ("Intensity reduced" if ratio < 0.7 else "Shifted band")))
            rows.append({"raman_peak": r["position"], "sers_peak": s["position"],
                         "shift": float(shift), "intensity_ratio": float(ratio),
                         "confidence": conf, "note": note,
                         "raman_rel": r["rel_intensity"], "sers_rel": s["rel_intensity"],
                         "raman_prom": r["prominence"], "sers_prom": s["prominence"],
                         "kind": "matched"})
            matched_r.add(a); matched_s.add(b)
    for a, r in enumerate(rp):
        if a not in matched_r:
            rows.append({"raman_peak": r["position"], "sers_peak": None, "shift": None,
                         "intensity_ratio": None, "confidence": "None",
                         "note": "Absent in SERS", "raman_rel": r["rel_intensity"],
                         "sers_rel": None, "raman_prom": r["prominence"], "sers_prom": None,
                         "kind": "raman_only"})
    for b, s in enumerate(sp):
        if b not in matched_s:
            rows.append({"raman_peak": None, "sers_peak": s["position"], "shift": None,
                         "intensity_ratio": None, "confidence": "None",
                         "note": "SERS-only", "raman_rel": None, "sers_rel": s["rel_intensity"],
                         "raman_prom": None, "sers_prom": s["prominence"],
                         "kind": "sers_only"})
    rows.sort(key=lambda x: (x["raman_peak"] if x["raman_peak"] is not None else x["sers_peak"]))
    return rows, _match_stats(rows, len(rp), len(sp))


def _match_stats(rows, n_r, n_s):
    m = [r for r in rows if r["kind"] == "matched"]
    shifts = np.array([r["shift"] for r in m]) if m else np.array([])
    n_m = len(m)
    prec = n_m / n_s if n_s else 0.0      # of SERS peaks, fraction explained by Raman
    rec = n_m / n_r if n_r else 0.0       # of Raman peaks, fraction surviving in SERS
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    jac = n_m / (n_r + n_s - n_m) if (n_r + n_s - n_m) else 0.0
    mean_abs = float(np.mean(np.abs(shifts))) if n_m else float("nan")
    # Peak Correspondence Score: F1 penalised by mean |shift| relative to tolerance
    pcs = float(f1 * (1.0 - min(mean_abs, MATCH_TOL) / MATCH_TOL)) if n_m else 0.0
    return {"n_raman_peaks": n_r, "n_sers_peaks": n_s, "n_matched": n_m,
            "n_raman_only": n_r - n_m, "n_sers_only": n_s - n_m,
            "peak_precision": prec, "peak_recall": rec, "peak_f1": f1, "peak_jaccard": jac,
            "matched_pct_of_raman": 100 * rec, "unmatched_pct_of_raman": 100 * (1 - rec),
            "mean_shift": float(np.mean(shifts)) if n_m else float("nan"),
            "median_shift": float(np.median(shifts)) if n_m else float("nan"),
            "mean_abs_shift": mean_abs,
            "max_abs_shift": float(np.max(np.abs(shifts))) if n_m else float("nan"),
            "std_shift": float(np.std(shifts)) if n_m else float("nan"),
            "peak_correspondence_score": pcs}


# ─────────────────────── similarity metrics (Part 4) ───────────────────────
def _cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def dtw_distance(a, b, band=25):
    """Banded (Sakoe-Chiba) DTW on 1-D spectra; normalised by path length."""
    n = len(a)
    INF = np.inf
    prev = np.full(n + 1, INF); prev[0] = 0.0
    prev_len = np.zeros(n + 1)
    for i in range(1, n + 1):
        cur = np.full(n + 1, INF); cur_len = np.zeros(n + 1)
        lo, hi = max(1, i - band), min(n, i + band)
        for j in range(lo, hi + 1):
            c = abs(a[i - 1] - b[j - 1])
            opts = (prev[j], cur[j - 1], prev[j - 1])
            lens = (prev_len[j], cur_len[j - 1], prev_len[j - 1])
            k = int(np.argmin(opts))
            cur[j] = c + opts[k]; cur_len[j] = 1 + lens[k]
        prev, prev_len = cur, cur_len
    L = prev_len[n] if prev_len[n] > 0 else 1.0
    return float(prev[n] / L)


def similarity_metrics(r, s, grid, rp=None, sp=None, tol=MATCH_TOL):
    r = np.nan_to_num(r); s = np.nan_to_num(s)
    cos = _cos(r, s)
    pear = float(pearsonr(r, s)[0])
    spear = float(spearmanr(r, s)[0])
    sam = float(np.degrees(np.arccos(np.clip(cos, -1, 1))))
    rng = float(r.max() - r.min()) + 1e-12
    nrmse = float(np.sqrt(np.mean((r - s) ** 2)) / rng)
    dr, ds = np.gradient(r), np.gradient(s)
    dcorr = float(pearsonr(dr, ds)[0])
    # normalised cross-correlation: max and lag
    rc = (r - r.mean()) / (r.std() + 1e-12); sc = (s - s.mean()) / (s.std() + 1e-12)
    xc = np.correlate(rc, sc, mode="full") / len(r)
    lag = int(np.argmax(xc) - (len(r) - 1))
    out = {"cosine": cos, "pearson": pear, "spearman": spear, "spectral_angle_deg": sam,
           "nrmse": nrmse, "derivative_corr": dcorr,
           "xcorr_max": float(xc.max()), "xcorr_lag_cm": float(lag * (grid[1] - grid[0])),
           "dtw": dtw_distance(r, s)}
    if rp is not None and sp is not None:
        _, st = match_peaks(rp, sp, tol)
        out.update({"peak_overlap": st["peak_f1"], "peak_jaccard": st["peak_jaccard"],
                    "peak_precision": st["peak_precision"], "peak_recall": st["peak_recall"],
                    "peak_f1": st["peak_f1"]})
    return out


# ─────────────────────── alignment experiment (Part 5) ───────────────────────
def alignment_scan(r, s, grid, rp, sp, max_shift=ALIGN_MAX):
    """Rigid shift of the SERS spectrum within +/-max_shift cm-1 (analysis only —
    stored spectra are NOT modified). Returns best shift + improvement."""
    dx = float(grid[1] - grid[0])
    nb = int(round(max_shift / dx))
    base_cos = _cos(np.nan_to_num(r), np.nan_to_num(s))
    _, base_st = match_peaks(rp, sp)
    best = {"shift_cm": 0.0, "cosine": base_cos, "peak_f1": base_st["peak_f1"]}
    for k in range(-nb, nb + 1):
        ss = np.roll(np.nan_to_num(s), k)
        if k > 0: ss[:k] = ss[k]
        elif k < 0: ss[k:] = ss[k - 1]
        c = _cos(np.nan_to_num(r), ss)
        if c > best["cosine"]:
            sp_shift = [dict(p, position=p["position"] + k * dx) for p in sp]
            _, st = match_peaks(rp, sp_shift)
            best = {"shift_cm": k * dx, "cosine": c, "peak_f1": st["peak_f1"]}
    return {"baseline_cosine": base_cos, "baseline_peak_f1": base_st["peak_f1"],
            "optimal_shift_cm": best["shift_cm"], "aligned_cosine": best["cosine"],
            "aligned_peak_f1": best["peak_f1"],
            "cosine_gain": best["cosine"] - base_cos,
            "peak_f1_gain": best["peak_f1"] - base_st["peak_f1"]}


# ─────────────────────── intensity redistribution (Part 6) ───────────────────────
def intensity_redistribution(rows):
    """Position-vs-intensity mechanism analysis on MATCHED peak pairs."""
    m = [r for r in rows if r["kind"] == "matched"]
    if len(m) < 3:
        return {"n_pairs": len(m), "insufficient": True}
    ri = np.array([r["raman_rel"] for r in m]); si = np.array([r["sers_rel"] for r in m])
    rp_ = np.array([r["raman_prom"] for r in m]); sp_ = np.array([r["sers_prom"] for r in m])
    rank = float(spearmanr(rp_, sp_)[0])
    inten = float(pearsonr(ri / (ri.sum() + 1e-12), si / (si.sum() + 1e-12))[0])
    # band-ratio preservation: correlation of log pairwise ratios
    lr, ls = [], []
    for i in range(len(m)):
        for j in range(i + 1, len(m)):
            lr.append(np.log((rp_[i] + 1e-9) / (rp_[j] + 1e-9)))
            ls.append(np.log((sp_[i] + 1e-9) / (sp_[j] + 1e-9)))
    band_ratio = float(pearsonr(lr, ls)[0]) if len(lr) > 2 else float("nan")
    rn, sn = ri / (ri.sum() + 1e-12), si / (si.sum() + 1e-12)
    redistribution = float(0.5 * np.abs(rn - sn).sum())     # total variation distance
    return {"n_pairs": len(m), "peak_rank_corr": rank, "norm_intensity_corr": inten,
            "band_ratio_preservation": band_ratio,
            "intensity_redistribution_index": redistribution, "insufficient": False}


# ─────────────────────── band-level comparison (Part 7) ───────────────────────
def band_analysis(r, s, grid, rp, sp):
    out = []
    for lo, hi in REGIONS:
        m = (grid >= lo) & (grid < hi)
        if m.sum() < 5:
            continue
        rr, ss = np.nan_to_num(r[m]), np.nan_to_num(s[m])
        rpk = [p for p in rp if lo <= p["position"] < hi]
        spk = [p for p in sp if lo <= p["position"] < hi]
        _, st = match_peaks(rpk, spk)
        ir = intensity_redistribution(match_peaks(rpk, spk)[0])
        out.append({"region": f"{lo}-{hi}",
                    "cosine": _cos(rr, ss),
                    "pearson": float(pearsonr(rr, ss)[0]) if rr.std() > 0 and ss.std() > 0 else float("nan"),
                    "n_raman_peaks": len(rpk), "n_sers_peaks": len(spk),
                    "n_shared": st["n_matched"], "n_missing": st["n_raman_only"],
                    "mean_abs_shift": st["mean_abs_shift"],
                    "dominant_raman_peak": rpk[0]["position"] if rpk else None,
                    "dominant_sers_peak": spk[0]["position"] if spk else None,
                    "intensity_redistribution": ir.get("intensity_redistribution_index")})
    return out


# ─────────────────────── within-modality control ───────────────────────
def within_modality_similarity(specs):
    """Mean pairwise cosine among replicates of one modality — the practical
    CEILING for cross-modal similarity (measurement reproducibility)."""
    n = len(specs)
    if n < 2:
        return float("nan")
    v = [np.nan_to_num(x) for x in specs]
    vals = [_cos(v[i], v[j]) for i in range(n) for j in range(i + 1, n)]
    return float(np.mean(vals))
