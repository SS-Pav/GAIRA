"""GAIRA V5 — serum spike-in / perturbation projection validation library.

The Raman Reference Atlas v0.1 (NMF k=24) is FROZEN. This library only loads
controlled-perturbation datasets, preprocesses them into the atlas-native
representation, projects them, and measures whether the resulting motion is
coherent. It never refits, rotates or rescales the atlas.

IMPORTANT — preprocessing contract. A projection into a frozen NMF basis is only
meaningful if the spectra are expressed in the SAME representation the basis was
fitted on. The atlas-native pipeline (ASLS baseline + Savitzky-Golay + L2 on a
450-1800 cm-1, 2 cm-1 grid) is therefore mandatory as the FINAL step and is not a
free parameter. What Phase 2 legitimately evaluates is the PRE-steps that precede
it (cosmic-ray removal, replicate outlier rejection) and the diagnostic question of
how much those choices move the projection.
"""
from __future__ import annotations
import re, zipfile, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, linregress

from gaira.foundation import dataset as DS
from gaira.preprocessing import pipeline as pp
from gaira.data.synonyms import GOBBATO_ABBREV, canonical

RAW = Path("/Volumes/SSD_Rad/GAIRA_DATA/raw")
SERUM_ZIP = RAW / "serum_ag_colloids/dataset_spectral_data.zip"
ILS_CSV = RAW / "european_multi_instrument_adenine/ILSdata.csv"
ERG_CSV = RAW / "ergothioneine_serum/ERG_calibration.csv"
GRID, WINDOW, PREPROC = DS.GRID, DS.WINDOW, DS.PREPROC

# despiking is only identifiable when the axis oversamples the narrowest band
MAX_STEP_FOR_DESPIKE = 2.0   # cm-1 per sample


# ───────────────────────── preprocessing ─────────────────────────
def despike(y, wn=None, z=8.0, min_band_cm=12.0):
    """Cosmic-ray removal by SHARPNESS, not by amplitude alone.

    A cosmic ray is narrower than the instrument can resolve a real band. A naive
    "large residual" rule deletes genuine signal when the wavenumber sampling is
    coarse (at 3 cm-1/point a real 12 cm-1 band spans only ~4 points), so the test
    here requires a point to be BOTH a large positive outlier AND sharper than the
    narrowest physically plausible band: its immediate neighbours must not be
    elevated with it.
    """
    y = np.asarray(y, float).copy()
    n = len(y)
    if n < 7:
        return y, 0
    step = float(np.median(np.abs(np.diff(wn)))) if wn is not None and len(wn) > 1 else 1.0
    # Despiking is only identifiable when the axis OVERSAMPLES the narrowest real
    # band. At >2 cm-1/point a 12 cm-1 band spans <6 samples and a cosmic ray is
    # indistinguishable from a band apex, so we decline to despike rather than
    # risk deleting genuine signal. Recorded as skipped, not as "zero spikes".
    if step > MAX_STEP_FOR_DESPIKE:
        return y, -1
    band_pts = max(3, int(round(min_band_cm / max(step, 1e-6))))
    win = max(5, band_pts | 1)
    med = pd.Series(y).rolling(win, center=True, min_periods=1).median().values
    resid = y - med
    mad = 1.4826 * np.median(np.abs(resid - np.median(resid)))
    # Guard: on a noiseless or heavily pre-smoothed spectrum the residual MAD
    # collapses toward zero, which would make every band apex look like an
    # infinite-z outlier. Floor the noise estimate at a small fraction of the
    # signal range so despiking can never fire on clean data.
    mad = max(mad, 1e-4 * float(np.ptp(y)), 1e-12)
    zscore = resid / mad
    nb = np.maximum(np.r_[y[1], y[:-1]], np.r_[y[1:], y[-2]])      # higher neighbour
    # Sharpness, judged physically: a cosmic ray exceeds BOTH neighbours by a large
    # fraction of the local dynamic range, whereas a properly sampled band apex
    # differs from its neighbours only marginally. This does not rely on the noise
    # estimate, so it stays correct on noiseless or pre-smoothed spectra.
    loc_max = pd.Series(y).rolling(win, center=True, min_periods=1).max().values
    loc_min = pd.Series(y).rolling(win, center=True, min_periods=1).min().values
    loc_range = np.maximum(loc_max - loc_min, 1e-12)
    sharp = (y - nb) > 0.30 * loc_range
    bad = (zscore > z) & sharp
    if band_pts > 3:
        # with coarse sampling also require the spike to be a single isolated point
        prev_bad = np.r_[False, bad[:-1]]; next_bad = np.r_[bad[1:], False]
        bad &= ~(prev_bad | next_bad)
    y[bad] = med[bad]
    return y, int(bad.sum())


def to_atlas(wn, y, despike_first=True):
    """Atlas-native representation. Mandatory for a valid projection."""
    n_spikes = 0
    if despike_first:
        y, n_spikes = despike(y, wn=wn)
    v = pp.preprocess(np.asarray(wn, float), np.asarray(y, float), PREPROC, GRID, WINDOW)
    return v, n_spikes


def replicate_outliers(X, z=3.5):
    """Flag replicate spectra whose cosine to the group median is a robust outlier."""
    X = np.nan_to_num(np.atleast_2d(X))
    if len(X) < 3:
        return np.zeros(len(X), bool), np.ones(len(X))
    med = np.median(X, axis=0)
    c = np.array([float(np.dot(x, med) / (np.linalg.norm(x) * np.linalg.norm(med) + 1e-12))
                  for x in X])
    mad = 1.4826 * np.median(np.abs(c - np.median(c))) + 1e-12
    return ((np.median(c) - c) / mad > z), c


# ───────────────────────── loaders ─────────────────────────
def _bwtek(text):
    lines = text.splitlines()
    hdr = next((i for i, l in enumerate(lines)
                if "Raman Shift" in l and "Dark Subtracted" in l), None)
    if hdr is None:
        return None
    cols = [c.strip() for c in lines[hdr].split(";")]
    try:
        iw = cols.index("Raman Shift")
        iy = next(i for i, c in enumerate(cols) if c.startswith("Dark Subtracted"))
    except (ValueError, StopIteration):
        return None
    wn, y = [], []
    for l in lines[hdr + 1:]:
        p = l.split(";")
        if len(p) <= max(iw, iy):
            continue
        try:
            wn.append(float(p[iw].replace(",", "."))); y.append(float(p[iy].replace(",", ".")))
        except ValueError:
            continue
    if len(wn) < 100:
        return None
    o = np.argsort(wn)
    return np.array(wn)[o], np.array(y)[o]


def _zip_dir(prefix):
    if not SERUM_ZIP.exists():
        return []
    out = []
    with zipfile.ZipFile(SERUM_ZIP) as z:
        for n in z.namelist():
            if n.startswith(prefix) and n.endswith(".txt"):
                parsed = _bwtek(z.read(n).decode("cp1252", errors="replace"))
                if parsed:
                    out.append((n.split("/")[-1], *parsed))
    return out


SPIKE_RE = re.compile(r"SERS_spike_(.+?)_([0-9.]+)(uM|mM|nM|ngmL)_(\d+)\.txt")


def load_spiked_serum():
    """51 analytes spiked into pooled serum, 5 replicates each (Ag-SERS)."""
    rows, recs = [], []
    for fn, wn, y in _zip_dir("SERS spiked serum Merck/"):
        m = SPIKE_RE.match(fn)
        if m:
            abbr, conc, unit, rep = m.groups()
            analyte = canonical(GOBBATO_ABBREV.get(abbr, abbr.lower()))
            conc_uM = float(conc) * {"uM": 1, "mM": 1000, "nM": 1e-3}.get(unit, np.nan)
        else:
            m2 = re.match(r"SERS_spike_(.+?)_([0-9.]+)(ngmL)_(\d+)\.txt", fn)
            if not m2:
                continue
            abbr, conc, unit, rep = m2.groups()
            analyte = canonical(abbr.lower()); conc_uM = np.nan
        v, ns = to_atlas(wn, y)
        if not np.isfinite(v).any():
            continue
        rows.append(v)
        recs.append({"dataset": "spiked_serum", "analyte": analyte, "abbrev": abbr,
                     "conc": float(conc), "unit": unit, "conc_uM": conc_uM,
                     "replicate": rep, "file": fn, "n_cosmic": ns,
                     "substrate": "Ag colloid", "laser_nm": 785, "matrix": "serum"})
    return (np.vstack(rows), pd.DataFrame(recs)) if rows else (None, None)


def load_serum_baseline():
    """Unspiked pooled serum (control for the spike-in experiment)."""
    rows, recs = [], []
    for fn, wn, y in _zip_dir("SERS serum Merck/"):
        v, ns = to_atlas(wn, y)
        if np.isfinite(v).any():
            rows.append(v)
            recs.append({"dataset": "serum_baseline", "analyte": "unspiked_serum",
                         "conc_uM": 0.0, "replicate": fn.split("_")[1] if "_" in fn else "1",
                         "file": fn, "n_cosmic": ns, "substrate": "Ag colloid",
                         "laser_nm": 785, "matrix": "serum"})
    return (np.vstack(rows), pd.DataFrame(recs)) if rows else (None, None)


def load_pure_sers():
    """Pure-analyte Ag-SERS references (same analytes as the spikes)."""
    rows, recs = [], []
    for fn, wn, y in _zip_dir("SERS metabolites/"):
        m = re.match(r"SERS_met_(.+?)_(.+?)_(\d+)\.txt", fn)
        if not m:
            continue
        abbr, conc, rep = m.groups()
        v, ns = to_atlas(wn, y)
        if not np.isfinite(v).any():
            continue
        rows.append(v)
        recs.append({"dataset": "pure_sers", "analyte": canonical(GOBBATO_ABBREV.get(abbr, abbr.lower())),
                     "abbrev": abbr, "conc": conc, "replicate": rep, "file": fn,
                     "n_cosmic": ns, "substrate": "Ag colloid", "laser_nm": 785, "matrix": "buffer"})
    return (np.vstack(rows), pd.DataFrame(recs)) if rows else (None, None)


def load_uricase():
    """Uricase depletion: serum spiked with urate, +/- enzyme (a DEPLETION series)."""
    rows, recs = [], []
    for fn, wn, y in _zip_dir("dataset uricase/"):
        grp = ("spiked+uricase" if "Enzyme" in fn else
               "spiked" if "Serumspiked" in fn else "serum_reference")
        v, ns = to_atlas(wn, y)
        if np.isfinite(v).any():
            rows.append(v)
            recs.append({"dataset": "uricase", "analyte": "urate", "condition": grp,
                         "conc_uM": np.nan, "replicate": fn.split("_")[1], "file": fn,
                         "n_cosmic": ns, "substrate": "Ag colloid", "laser_nm": 785,
                         "matrix": "serum"})
    return (np.vstack(rows), pd.DataFrame(recs)) if rows else (None, None)


def load_isotopic():
    """Urate vs isotope-labelled urate (a specificity control)."""
    rows, recs = [], []
    for fn, wn, y in _zip_dir("isotopic/"):
        lab = "UAiso" if "iso" in fn else ("UA" if "UA" in fn else "other")
        v, ns = to_atlas(wn, y)
        if np.isfinite(v).any():
            rows.append(v)
            recs.append({"dataset": "isotopic", "analyte": "urate", "condition": lab,
                         "conc_uM": np.nan, "replicate": fn.split("_")[1], "file": fn,
                         "n_cosmic": ns, "substrate": "Ag colloid", "laser_nm": 785,
                         "matrix": "buffer"})
    return (np.vstack(rows), pd.DataFrame(recs)) if rows else (None, None)


def load_ils_adenine():
    """Inter-laboratory adenine dose-response: 15 labs, 4 substrates, 2 lasers."""
    if not ILS_CSV.exists():
        return None, None
    d = pd.read_csv(ILS_CSV, low_memory=False)
    wn = np.array([float(c) for c in d.columns[9:]])
    rows, recs = [], []
    for _, r in d.iterrows():
        y = r.values[9:].astype(float)
        v, ns = to_atlas(wn, y)
        if not np.isfinite(v).any():
            continue
        rows.append(v)
        recs.append({"dataset": "ils_adenine", "analyte": "adenine",
                     "labcode": r.labcode, "substrate": r.substrate,
                     "laser_nm": int(r.laser), "type": r.type,
                     "conc_uM": float(r.conc), "batch": str(r.batch),
                     "replicate": str(r.replica), "n_cosmic": ns,
                     "matrix": "buffer", "method": r.get("method", "")})
    return (np.vstack(rows), pd.DataFrame(recs)) if rows else (None, None)


def load_ergothioneine():
    """Ergothioneine Ag-SERS calibration series."""
    if not ERG_CSV.exists():
        return None, None
    d = pd.read_csv(ERG_CSV)
    wn = np.array([float(c) for c in d.columns[4:]])
    keep = wn > 0
    rows, recs = [], []
    for i, r in d.iterrows():
        y = r.values[4:].astype(float)
        v, ns = to_atlas(wn[keep], y[keep])
        if not np.isfinite(v).any():
            continue
        rows.append(v)
        recs.append({"dataset": "ergothioneine", "analyte": "ergothioneine",
                     "conc_uM": float(r.get("c", np.nan)), "laser_nm": int(r.get("laser", 785)),
                     "substrate": str(r.get("substrate", "cAg")), "power": r.get("power", np.nan),
                     "replicate": str(i), "n_cosmic": ns, "matrix": "buffer"})
    return (np.vstack(rows), pd.DataFrame(recs)) if rows else (None, None)


LOADERS = {"ils_adenine": load_ils_adenine, "spiked_serum": load_spiked_serum,
           "serum_baseline": load_serum_baseline, "pure_sers": load_pure_sers,
           "uricase": load_uricase, "isotopic": load_isotopic,
           "ergothioneine": load_ergothioneine}


# ───────────────────────── trajectory metrics ─────────────────────────
def _unit(X):
    X = np.nan_to_num(np.atleast_2d(X))
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def group_means(Z, labels):
    lab = np.asarray(labels)
    keys = sorted(pd.unique(lab), key=lambda v: (float(v) if _isnum(v) else str(v)))
    return keys, np.vstack([np.nan_to_num(Z[lab == k]).mean(axis=0) for k in keys])


def _isnum(v):
    try:
        float(v); return True
    except (TypeError, ValueError):
        return False


def trajectory_metrics(concs, M, ref=None):
    """Motion of a concentration series through the manifold.
    concs ascending; M = per-concentration mean coordinates."""
    concs = np.asarray(concs, float)
    ref = M[0] if ref is None else ref
    d = np.linalg.norm(M - ref, axis=1)                      # distance from control
    steps = np.diff(M, axis=0)
    step_norm = np.linalg.norm(steps, axis=1)
    # smoothness: cosine between consecutive displacement steps
    smooth = [float(np.dot(steps[i], steps[i + 1]) /
                    (np.linalg.norm(steps[i]) * np.linalg.norm(steps[i + 1]) + 1e-12))
              for i in range(len(steps) - 1)] if len(steps) > 1 else []
    rho, p = (spearmanr(concs, d) if len(concs) > 2 else (np.nan, np.nan))
    net = M[-1] - M[0]
    path = float(step_norm.sum())
    return {"n_levels": len(concs), "distance_from_control": d.tolist(),
            "total_path_length": path, "net_displacement": float(np.linalg.norm(net)),
            "straightness": float(np.linalg.norm(net) / (path + 1e-12)),
            "mean_step_cosine": float(np.mean(smooth)) if smooth else np.nan,
            "monotonicity_rho": float(rho), "monotonicity_p": float(p),
            "net_direction": net}


def monotonicity_null(concs, Z, labels, n_perm=1000, seed=0):
    """Permutation null: shuffle concentration labels and recompute |rho|."""
    rng = np.random.default_rng(seed)
    keys, M = group_means(Z, labels)
    obs = trajectory_metrics(np.array(keys, float), M)["monotonicity_rho"]
    null = []
    lab = np.asarray(labels)
    for _ in range(n_perm):
        perm = rng.permutation(lab)
        k2, M2 = group_means(Z, perm)
        r = trajectory_metrics(np.array(k2, float), M2)["monotonicity_rho"]
        if np.isfinite(r):
            null.append(abs(r))
    null = np.array(null)
    return {"observed_rho": obs, "null_mean_abs_rho": float(null.mean()) if null.size else np.nan,
            "p_value": float((np.sum(null >= abs(obs)) + 1) / (len(null) + 1)) if null.size else np.nan}


def dose_response_fits(concs, d):
    """Compare linear / log / saturating descriptions of distance vs concentration."""
    c = np.asarray(concs, float); y = np.asarray(d, float)
    ok = np.isfinite(c) & np.isfinite(y)
    c, y = c[ok], y[ok]
    out = {}
    if len(c) < 3 or np.ptp(y) < 1e-12:
        return {"insufficient": True}
    lr = linregress(c, y)
    out["linear_r2"] = float(lr.rvalue ** 2)
    pos = c > 0
    if pos.sum() >= 3:
        lg = linregress(np.log10(c[pos]), y[pos])
        out["log_r2"] = float(lg.rvalue ** 2)
    # saturating (Langmuir-like) y = a*c/(k+c)
    try:
        from scipy.optimize import curve_fit
        f = lambda x, a, k: a * x / (k + x)
        popt, _ = curve_fit(f, c, y, p0=[max(y), np.median(c[c > 0]) if (c > 0).any() else 1],
                            maxfev=8000)
        ss = 1 - np.sum((y - f(c, *popt)) ** 2) / (np.sum((y - y.mean()) ** 2) + 1e-12)
        out["saturating_r2"] = float(ss); out["saturating_K"] = float(popt[1])
    except Exception:
        pass
    best = max((k for k in out if k.endswith("_r2")), key=lambda k: out[k], default=None)
    out["best_model"] = (best or "none").replace("_r2", "")
    return out


def displacement(Z_treat, Z_control):
    """Mean displacement vector and its replicate-level dispersion."""
    a = np.nan_to_num(Z_treat).mean(axis=0)
    b = np.nan_to_num(Z_control).mean(axis=0)
    v = a - b
    per = np.nan_to_num(Z_treat) - b
    U = _unit(per)
    C = U @ U.T
    iu = np.triu_indices(len(U), 1)
    return {"vector": v, "norm": float(np.linalg.norm(v)),
            "replicate_direction_cos": float(C[iu].mean()) if len(U) > 1 else np.nan,
            "n": int(len(Z_treat))}


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def angle_deg(a, b):
    return float(np.degrees(np.arccos(np.clip(cos(a, b), -1, 1))))
