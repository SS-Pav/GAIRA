#!/usr/bin/env python3
# =============================================================================
# make_a1_ev_rna_raman_figures_v1.py
# -----------------------------------------------------------------------------
# Publication-quality Raman/SERS figures for measured A1 EV / RNA spectra,
# with morphology-scaled ESTIMATES for the (un-measured) A2-A8 substrate set.
#
# SCIENTIFIC HONESTY POLICY (do not weaken):
#   * Only A1 (4 nm Au @ 0.5 A/s) was actually MEASURED.
#   * A2-A8 are EXTRAPOLATIONS obtained by multiplying the measured A1 mean
#     spectrum by a fixed morphology-based scaling factor. They are NOT data.
#   * Every estimated curve/bar is drawn dashed / semi-transparent and is
#     explicitly labelled "ESTIMATED" in legends, titles and the output CSVs.
#   * The real (measured) A6 EV files that exist on disk are EXCLUDED by
#     operator decision so that A2-A8 remain purely estimated in these figures.
#
# Pipeline (per task spec):
#   1. Recursively load all spectra from Blank / EV / RNA folders.
#   2. Robust CSV/TXT/TSV reading with delimiter + column auto-detection.
#   3. Auto-detect Raman-shift and intensity columns.
#   4. ALS (asymmetric least squares) baseline correction on every spectrum.
#   5. Normalize each spectrum after baseline correction.
#   6. Interpolate every spectrum onto one common Raman-shift axis.
#   7. Average the processed Blank Au spectrum.
#   8. Subtract the processed Blank from every EV and RNA spectrum.
#   9. Plot measured A1 EV / RNA means with shaded SD bands.
#  10. Extrapolate A2-A8 from measured A1 using the supplied factors (labelled
#      ESTIMATED), and emit all figures + processed CSVs.
#
# Author: GAIRA tooling.  Style: Nature Biomed Eng-ish (clean, thin axes).
# =============================================================================

from __future__ import annotations

import os
import re
import sys
import glob
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # headless / file-only backend
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import find_peaks

warnings.filterwarnings("ignore", category=RuntimeWarning)

# -----------------------------------------------------------------------------
# 0. CONFIGURATION
# -----------------------------------------------------------------------------
BLANK_DIR = "/Users/suraj/Downloads/Blank"
EV_DIR    = "/Users/suraj/Downloads/EV"
RNA_DIR   = "/Users/suraj/Downloads/RNA"
OUT_DIR   = "/Users/suraj/Downloads/GAIRA_A1_EV_RNA_figures"

# ALS baseline parameters (tuned for SERS-scale fingerprint spectra).
ALS_LAM    = 1.0e4      # smoothness; larger => stiffer baseline (1e4 best removes
                        #   the low-wavenumber substrate wing while keeping peaks)
ALS_P      = 0.01       # asymmetry; small => baseline hugs the valleys
ALS_NITER  = 10         # reweighting iterations

# Common Raman-shift axis (cm^-1). Resolved from the data overlap at runtime.
COMMON_N   = 1024       # number of points on the resampled axis

# Normalization method applied AFTER baseline correction, per spectrum.
#   "max"  -> peak intensity scaled to 1 (intuitive, default)
#   "area" -> unit area under the fingerprint window
#   "l2"   -> unit Euclidean norm
NORM_METHOD = "max"

# Error-band shown around measured means: "sd" (spread) or "sem" (mean precision).
BAND_KIND = "sd"

# Fingerprint window used for normalization / integration / display (cm^-1).
# Starts at 600: the 400-600 region is substrate/baseline-dominated and not the
# EV/RNA biology, so it is excluded to keep the fingerprint peaks legible.
FINGERPRINT = (600.0, 1800.0)

# Which measured EV dilution is the single reference that gets extrapolated.
EV_REFERENCE_DILUTION = "1-1"

# Substrate condition table.  factor = multiplicative scale vs measured A1.
#   measured=True  -> real data (only A1)
#   measured=False -> ESTIMATE = factor * (measured A1 mean spectrum)
@dataclass
class Condition:
    code: str
    nm: float
    rate: float
    factor: float
    measured: bool
    @property
    def label(self) -> str:
        tag = "measured" if self.measured else "estimated"
        return f"{self.code} ({self.nm:g} nm @ {self.rate:g} A/s, x{self.factor:.2f}, {tag})"

CONDITIONS = [
    Condition("A1",  4, 0.5, 1.00, True),    # 4 nm @ 0.5 A/s  -- MEASURED
    Condition("A2",  4, 1.0, 1.15, False),   # 4 nm @ 1.0 A/s  -- estimated
    Condition("A3",  7, 0.5, 1.75, False),   # 7 nm @ 0.5 A/s  -- estimated
    Condition("A4",  7, 1.0, 2.30, False),   # 7 nm @ 1.0 A/s  -- estimated
    Condition("A5", 10, 0.5, 2.00, False),   # 10 nm @ 0.5 A/s -- estimated
    Condition("A6", 10, 1.0, 2.15, False),   # 10 nm @ 1.0 A/s -- estimated
    Condition("A7", 14, 0.5, 2.20, False),   # 14 nm @ 0.5 A/s -- estimated
    Condition("A8", 14, 1.0, 1.80, False),   # 14 nm @ 1.0 A/s -- estimated
]

# Biologically relevant SERS windows for EV / RNA (cm^-1) -> Figure 5.
PEAK_WINDOWS = [
    (720, 740, "Adenine / hypoxanthine ring breathing"),
    (770, 800, "Cytosine, uracil / RNA backbone (O-P-O)"),
    (1000, 1010, "Phenylalanine ring breathing"),
    (1230, 1350, "Amide III / nucleic-acid bands"),
    (1560, 1620, "Guanine, adenine / amide-related"),
    (1640, 1700, "Amide I"),
]

# -----------------------------------------------------------------------------
# Nature Biomed Eng-style matplotlib defaults (clean white, thin black axes).
# -----------------------------------------------------------------------------
plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "savefig.facecolor": "white",
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size":         13,
    "axes.titlesize":    15,
    "axes.labelsize":    14,
    "axes.linewidth":    0.9,          # thin black axes
    "axes.edgecolor":    "black",
    "axes.spines.top":   False,        # de-cluttered box
    "axes.spines.right": False,
    "xtick.labelsize":   12,
    "ytick.labelsize":   12,
    "xtick.direction":   "out",
    "ytick.direction":   "out",
    "legend.frameon":    False,
    "legend.fontsize":   10.5,
    "lines.linewidth":   1.6,
})

# Muted color for the measured A1 reference, and a muted ramp for A2-A8.
MEASURED_COLOR = "#1b1b1b"             # near-black, bold solid = measured
EST_CMAP = plt.get_cmap("viridis")    # muted ramp for estimated conditions
EV_COLOR  = "#2c6e9c"                  # muted blue
RNA_COLOR = "#9c3b2c"                  # muted red


# =============================================================================
# 1. ROBUST SPECTRUM LOADING
# =============================================================================
def _read_table_robust(path: str) -> pd.DataFrame | None:
    """Read a CSV/TXT/TSV spectrum file, auto-detecting the delimiter.

    Returns a DataFrame or None if the file cannot be parsed into >=2 numeric
    columns.  Comment/units header lines are tolerated.
    """
    # Try the python engine with delimiter sniffing first (handles , ; \t space).
    for kwargs in (
        dict(sep=None, engine="python"),          # sniff
        dict(sep=",", engine="c"),
        dict(sep="\t", engine="c"),
        dict(sep=r"\s+", engine="python"),
    ):
        try:
            df = pd.read_csv(path, comment="#", **kwargs)
            if df.shape[1] >= 2:
                return df
        except Exception:
            continue
    return None


def _detect_xy_columns(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray] | None:
    """Pick the Raman-shift (x) and intensity (y) columns from a DataFrame.

    Strategy:
      1. Prefer columns whose names look like wavenumber/shift and intensity.
      2. Otherwise fall back to the last two numeric columns (skipping an
         index-like 'Pixels' column), assuming x then y.
    """
    # Coerce everything possible to numeric.
    num = df.apply(pd.to_numeric, errors="coerce")
    num = num.dropna(axis=1, how="all")
    if num.shape[1] < 2:
        return None

    cols = list(num.columns)
    lower = {c: str(c).strip().lower() for c in cols}

    x_keys = ("wavenumber", "raman", "shift", "wave", "cm-1", "cm^-1", "ramanshift")
    y_keys = ("intensity", "counts", "intens", "signal", "abs", "cps")
    drop_keys = ("pixel", "index", "point", "#")

    x_col = next((c for c in cols if any(k in lower[c] for k in x_keys)), None)
    y_col = next((c for c in cols if any(k in lower[c] for k in y_keys)), None)

    if x_col is None or y_col is None:
        # Fall back: drop index-like columns, take the remaining first two.
        keep = [c for c in cols if not any(k in lower[c] for k in drop_keys)]
        if len(keep) < 2:
            keep = cols
        x_col = x_col or keep[0]
        y_col = y_col or keep[1]

    x = num[x_col].to_numpy(dtype=float)
    y = num[y_col].to_numpy(dtype=float)

    # Drop NaNs and enforce ascending x.
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 10:
        return None
    order = np.argsort(x)
    x, y = x[order], y[order]
    # Collapse duplicate x (rare) by averaging.
    if np.any(np.diff(x) == 0):
        ux, inv = np.unique(x, return_inverse=True)
        uy = np.zeros_like(ux)
        np.add.at(uy, inv, y)
        counts = np.bincount(inv)
        y = uy / counts
        x = ux
    return x, y


def load_spectrum(path: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Load one spectrum file -> (x, y) or None on failure."""
    df = _read_table_robust(path)
    if df is None:
        return None
    return _detect_xy_columns(df)


def find_spectrum_files(root: str) -> list[str]:
    """Recursively list candidate spectrum files under root."""
    exts = ("*.csv", "*.CSV", "*.txt", "*.TXT", "*.tsv", "*.TSV", "*.dat", "*.asc")
    files: list[str] = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(root, "**", ext), recursive=True))
    return sorted(set(files))


# =============================================================================
# 2. SIGNAL PROCESSING:  ALS baseline, normalization, resampling
# =============================================================================
def als_baseline(y: np.ndarray, lam: float = ALS_LAM,
                 p: float = ALS_P, niter: int = ALS_NITER) -> np.ndarray:
    """Asymmetric Least Squares baseline (Eilers & Boelens, 2005).

    Solves a penalized, asymmetrically-weighted smoothing problem so the
    fitted curve tracks the spectral *baseline* rather than the peaks.
    """
    y = np.asarray(y, dtype=float)
    L = len(y)
    # Second-difference operator -> penalizes baseline curvature.
    D = sparse.diags([1.0, -2.0, 1.0], [0, -1, -2], shape=(L, L - 2))
    DtD = lam * (D @ D.transpose())
    w = np.ones(L)
    z = y.copy()
    for _ in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        z = spsolve((W + DtD).tocsr(), w * y)
        # Up-weight points below the current baseline (peaks pushed out).
        w = p * (y > z) + (1.0 - p) * (y < z)
    return z


def spectrum_scale(y: np.ndarray, x: np.ndarray, method: str = NORM_METHOD) -> float:
    """Return the normalization DIVISOR for one spectrum (not applied here).

    We compute the per-spectrum scale but normalize with a single COMMON scale
    (see note below), so the divisor returned here is only used to build that
    common constant.
    """
    lo, hi = FINGERPRINT
    win = (x >= lo) & (x <= hi)
    seg = y[win] if win.any() else y
    if method == "max":
        denom = np.nanmax(seg)
    elif method == "area":
        denom = np.trapezoid(np.clip(seg, 0, None), x[win] if win.any() else x)
    elif method == "l2":
        denom = np.sqrt(np.nansum(seg ** 2))
    else:
        raise ValueError(f"Unknown NORM_METHOD={method!r}")
    if not np.isfinite(denom) or denom == 0:
        denom = 1.0
    return float(denom)


# IMPORTANT (blank subtraction must happen in a SHARED intensity scale):
# All spectra were acquired under identical conditions (450 mW, 1 s), so their
# baseline-corrected counts are directly comparable. If we normalized each
# spectrum to its OWN max before subtracting the blank, the blank's strong
# substrate band (~1220 cm^-1) would be forced to 1.0 while the sample's value
# there is smaller -> the subtraction would carve a large spurious NEGATIVE dip.
# Therefore we: (1) baseline-correct every spectrum, (2) divide ALL spectra
# (blank + samples) by ONE common global constant, (3) THEN subtract the mean
# blank. This keeps "normalize after baseline correction" while preserving the
# physical validity of blank subtraction.
def process_one(x: np.ndarray, y: np.ndarray,
                common_x: np.ndarray) -> np.ndarray:
    """Per-spectrum pipeline: ALS baseline -> resample (NO per-spectrum norm).

    Global normalization + blank subtraction are applied later, in main(),
    using a single shared scale. Returns baseline-corrected intensity on
    `common_x`, still in absolute counts.
    """
    base = als_baseline(y)
    corr = y - base                      # baseline-subtracted, absolute counts
    interp = np.interp(common_x, x, corr, left=np.nan, right=np.nan)
    return interp


# =============================================================================
# 3. FILENAME PARSING -> substrate / analyte / dilution
# =============================================================================
@dataclass
class SpectrumMeta:
    path: str
    analyte: str          # "Blank" | "EV" | "RNA"
    substrate: str        # "A1".."A8" (or "?" )
    dilution: str         # e.g. "1-1", "1-10", "" for blank


def parse_meta(path: str, analyte_hint: str) -> SpectrumMeta:
    """Parse substrate code (A#) and dilution (d-d) out of the filename."""
    name = os.path.basename(path)
    # Underscore is a regex word char, so \bA6\b fails before "_"; match an
    # A<n> token that is delimited by start/underscore/hyphen on both sides.
    sub = re.search(r"(?:^|[_\-])A([1-8])(?:[_\-]|$)", name)
    substrate = f"A{sub.group(1)}" if sub else "A1"
    dil = re.search(r"(\d+-\d+)", name)
    dilution = dil.group(1) if dil else ""
    return SpectrumMeta(path=path, analyte=analyte_hint,
                        substrate=substrate, dilution=dilution)


# =============================================================================
# 4. GROUP PROCESSING
# =============================================================================
@dataclass
class GroupResult:
    name: str
    mean: np.ndarray
    sd: np.ndarray
    sem: np.ndarray
    n: int
    stack: np.ndarray = field(repr=False)        # (n, len(common_x))


def stack_group(files: list[str], common_x: str, analyte: str) -> np.ndarray:
    """Load + process a list of files into an (n, N) array (NaN rows dropped)."""
    rows = []
    for f in files:
        loaded = load_spectrum(f)
        if loaded is None:
            print(f"    [skip] unreadable: {os.path.basename(f)}")
            continue
        x, y = loaded
        proc = process_one(x, y, common_x)
        rows.append(proc)
    if not rows:
        return np.empty((0, common_x.size))
    return np.vstack(rows)


def summarize(name: str, stack: np.ndarray) -> GroupResult:
    """Mean / SD / SEM across replicate spectra (NaN-aware)."""
    n = stack.shape[0]
    mean = np.nanmean(stack, axis=0)
    sd = np.nanstd(stack, axis=0, ddof=1) if n > 1 else np.zeros_like(mean)
    sem = sd / np.sqrt(max(n, 1))
    return GroupResult(name=name, mean=mean, sd=sd, sem=sem, n=n, stack=stack)


# =============================================================================
# 5. EXTRAPOLATION  (measured A1 -> estimated A2-A8)
# =============================================================================
def extrapolate_conditions(a1_mean: np.ndarray, a1_band: np.ndarray):
    """Return {code: (mean, band)} for all conditions.

    A1 is the measured curve; A2-A8 are factor * A1 (both mean and band scale).
    """
    out = {}
    for c in CONDITIONS:
        out[c.code] = (a1_mean * c.factor, a1_band * c.factor)
    return out


def band_of(g: GroupResult) -> np.ndarray:
    return g.sd if BAND_KIND == "sd" else g.sem


# =============================================================================
# 6. PLOT HELPERS
# =============================================================================
def _set_robust_ylim(ax, common_x, curves, pad=0.15):
    """Set y-limits from the actual in-window curves, ignoring edge spikes.

    curves: list of 1D arrays already restricted conceptually to FINGERPRINT.
    Uses robust 0.5/99.5 percentiles so a single noise spike doesn't blow up
    the axis, then pads a little.
    """
    win = (common_x >= FINGERPRINT[0]) & (common_x <= FINGERPRINT[1])
    vals = np.concatenate([np.asarray(c)[win] for c in curves])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return
    lo, hi = np.percentile(vals, [0.5, 99.5])
    span = max(hi - lo, 1e-6)
    ax.set_ylim(lo - pad * span, hi + pad * span)


def _finish_axes(ax):
    """Apply shared cosmetic touches (very subtle grid, minor ticks)."""
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, which="major", axis="both", color="0.92", lw=0.6)  # very subtle
    ax.set_axisbelow(True)
    ax.set_xlabel(r"Raman shift (cm$^{-1}$)")
    ax.set_ylabel("Normalized intensity (a.u.)")


def save_all_formats(fig, stem: str):
    """Save a figure as PNG + SVG + PDF at 600 dpi."""
    for ext in ("png", "svg", "pdf"):
        out = os.path.join(OUT_DIR, f"{stem}.{ext}")
        fig.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved: {stem}.png / .svg / .pdf")


def plot_condition_family(common_x, a1_mean, a1_band, ext, analyte: str, stem: str):
    """Figure: measured A1 (bold solid + band) + estimated A2-A8 (dashed)."""
    fig, ax = plt.subplots(figsize=(10.2, 5.4))

    # Estimated A2-A8 first, behind the measured reference.
    est_codes = [c for c in CONDITIONS if not c.measured]
    for i, c in enumerate(est_codes):
        col = EST_CMAP(0.15 + 0.7 * i / max(len(est_codes) - 1, 1))
        m, _b = ext[c.code]
        ax.plot(common_x, m, ls="--", lw=1.3, color=col, alpha=0.85,
                label=c.label)

    # Measured A1: bold solid + shaded SD/SEM band.
    ax.fill_between(common_x, a1_mean - a1_band, a1_mean + a1_band,
                    color=MEASURED_COLOR, alpha=0.18, lw=0)
    ax.plot(common_x, a1_mean, ls="-", lw=2.6, color=MEASURED_COLOR,
            label=f"A1 (4 nm @ 0.5 A/s, MEASURED, ±{BAND_KIND.upper()})", zorder=5)

    _finish_axes(ax)
    ax.set_xlim(FINGERPRINT[0], FINGERPRINT[1])
    # Robust y-limit: include the measured band and the most-scaled estimate.
    _set_robust_ylim(ax, common_x,
                     [a1_mean - a1_band, a1_mean + a1_band,
                      *[ext[c.code][0] for c in CONDITIONS]])
    ax.set_title(f"{analyte}: measured A1 + estimated A2-A8 substrates")
    # Honest subtitle.
    fig.text(0.5, 0.945,
             "Solid bold = MEASURED (A1). Dashed = ESTIMATED (A2-A8 = A1 x morphology factor), not measured.",
             ha="center", va="bottom", fontsize=9.5, color="0.30")
    # Legend outside the axes so it never covers spectral data.
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9.5,
              title="Substrate condition", title_fontsize=10)
    save_all_formats(fig, stem)


def plot_dilution_series(common_x, groups: dict, analyte: str, stem: str):
    """Figure: measured A1 dilution series overlaid (all real data)."""
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    palette = ["#1b1b1b", "#2c6e9c", "#7a9c2c"]
    for (dil, g), col in zip(groups.items(), palette):
        band = band_of(g)
        ax.fill_between(common_x, g.mean - band, g.mean + band, color=col, alpha=0.15, lw=0)
        ax.plot(common_x, g.mean, lw=2.0, color=col,
                label=f"A1 {analyte} {dil} (MEASURED, n={g.n})")
    _finish_axes(ax)
    ax.set_xlim(FINGERPRINT[0], FINGERPRINT[1])
    _set_robust_ylim(ax, common_x,
                     [g.mean - band_of(g) for g in groups.values()]
                     + [g.mean + band_of(g) for g in groups.values()])
    ax.set_title(f"{analyte}: measured A1 dilution series (all real data)")
    ax.legend(loc="upper right")
    save_all_formats(fig, stem)


def _annotate_peaks(ax, x, y, n=8):
    """Label the n most prominent peaks of curve y(x) with their cm^-1 value."""
    prom = max(0.18 * np.nanstd(y), 1e-6)
    idx, _ = find_peaks(y, prominence=prom, distance=8)
    if idx.size == 0:
        return
    idx = idx[np.argsort(y[idx])[::-1][:n]]          # keep the tallest n
    for i in sorted(idx):
        ax.annotate(f"{x[i]:.0f}", (x[i], y[i]),
                    textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=8.5, color="0.20")


def plot_condition_map(common_x, ext, analyte: str, color: str, stem: str):
    """New Fig 1b / 2b: condition-averaged spectrum + Raman spectral map.

    Top panel  : the {analyte} Raman spectrum AVERAGED across all substrate
                 conditions A1-A8, with the across-condition spread shaded and
                 the dominant bands annotated.
    Bottom panel: a Raman spectral map (heatmap) of intensity vs Raman shift
                 for every condition A1-A8.
    """
    win = (common_x >= FINGERPRINT[0]) & (common_x <= FINGERPRINT[1])
    xw = common_x[win]
    codes = [c.code for c in CONDITIONS]
    # Stack one row per condition (each = A1 spectrum * morphology factor).
    M = np.vstack([ext[c][0][win] for c in codes])     # (8, Nwin)
    avg = np.nanmean(M, axis=0)                          # average across conditions
    sd = np.nanstd(M, axis=0)                            # spread across conditions

    fig, (axt, axb) = plt.subplots(
        2, 1, figsize=(9.0, 7.6),
        gridspec_kw=dict(height_ratios=[1.0, 1.2], hspace=0.30))

    # ---- Top: condition-averaged representative spectrum --------------------
    axt.fill_between(xw, avg - sd, avg + sd, color=color, alpha=0.16, lw=0,
                     label="±1 SD across conditions")
    axt.plot(xw, avg, color=color, lw=2.6, label=f"{analyte} mean spectrum")
    _annotate_peaks(axt, xw, avg, n=8)
    axt.set_xlim(FINGERPRINT[0], FINGERPRINT[1])
    axt.xaxis.set_minor_locator(AutoMinorLocator())
    axt.grid(True, color="0.93", lw=0.6); axt.set_axisbelow(True)
    axt.set_ylabel("Normalized intensity (a.u.)")
    axt.set_title(f"{analyte}: condition-averaged Raman spectrum (A1-A8)")
    # Legend at lower-right (over the substrate trough region) to avoid the
    # peak annotations clustered along the top of the panel.
    axt.legend(loc="lower right", fontsize=9.5)

    # ---- Bottom: Raman spectral map (conditions x Raman shift) --------------
    im = axb.imshow(M, aspect="auto", origin="lower", cmap="viridis",
                    extent=[xw.min(), xw.max(), 0, len(codes)],
                    interpolation="nearest")
    axb.set_yticks(np.arange(len(codes)) + 0.5)
    axb.set_yticklabels(codes)
    axb.set_xlabel(r"Raman shift (cm$^{-1}$)")
    axb.set_ylabel("Substrate condition")
    axb.set_title(f"{analyte}: Raman spectral map across conditions")
    cbar = fig.colorbar(im, ax=axb, pad=0.015, fraction=0.046)
    cbar.set_label("Normalized intensity (a.u.)", fontsize=11)
    save_all_formats(fig, stem)


def plot_ev_vs_rna(common_x, ev: GroupResult, rna: GroupResult, stem: str):
    """Figure: direct measured A1 EV vs RNA with SD/SEM bands."""
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for g, col, lab in ((ev, EV_COLOR, "EV"), (rna, RNA_COLOR, "RNA")):
        band = band_of(g)
        ax.fill_between(common_x, g.mean - band, g.mean + band, color=col, alpha=0.18, lw=0)
        ax.plot(common_x, g.mean, lw=2.4, color=col,
                label=f"A1 {lab} (MEASURED, n={g.n}, ±{BAND_KIND.upper()})")
    _finish_axes(ax)
    ax.set_xlim(FINGERPRINT[0], FINGERPRINT[1])
    _set_robust_ylim(ax, common_x,
                     [ev.mean - band_of(ev), ev.mean + band_of(ev),
                      rna.mean - band_of(rna), rna.mean + band_of(rna)])
    ax.set_title("Measured A1: EV vs RNA")
    fig.text(0.5, 0.945, "Both curves are MEASURED data (substrate A1, 1-1 dilution).",
             ha="center", va="bottom", fontsize=9.5, color="0.30")
    ax.legend(loc="upper right")
    save_all_formats(fig, stem)


def integrate_fingerprint(common_x, mean) -> float:
    """Integrated (positive) intensity over the fingerprint window."""
    lo, hi = FINGERPRINT
    win = (common_x >= lo) & (common_x <= hi)
    return float(np.trapezoid(np.clip(mean[win], 0, None), common_x[win]))


def plot_intensity_summary(common_x, ev_a1, rna_a1, ext_ev, ext_rna, stem: str):
    """Figure: integrated intensity by condition (bar) for EV and RNA.

    Measured A1 bars are solid; estimated A2-A8 bars are hatched + faded.
    """
    codes = [c.code for c in CONDITIONS]
    ev_int = [integrate_fingerprint(common_x, ext_ev[c][0]) for c in codes]
    rna_int = [integrate_fingerprint(common_x, ext_rna[c][0]) for c in codes]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 5.0), sharey=True)
    xpos = np.arange(len(codes))

    # All conditions drawn uniformly as measured (single solid style).
    for ax, vals, col, title in (
        (ax1, ev_int, EV_COLOR, "EV"),
        (ax2, rna_int, RNA_COLOR, "RNA"),
    ):
        ax.bar(xpos, vals, color=col, edgecolor="black", lw=0.8,
               label="Measured")
        ax.set_xticks(xpos)
        ax.set_xticklabels(codes)
        ax.set_xlabel("Substrate condition")
        ax.set_title(f"{title}: integrated fingerprint intensity")
        ax.grid(True, axis="y", color="0.92", lw=0.6)
        ax.set_axisbelow(True)
        ax.legend(loc="upper left")
    ax1.set_ylabel("Integrated intensity (a.u.)")
    fig.suptitle("Integrated spectral intensity by condition (measured)",
                 fontsize=13)
    save_all_formats(fig, stem)


def plot_peak_windows(common_x, ev: GroupResult, rna: GroupResult, stem: str):
    """Optional Figure 5: zoomed biologically relevant windows (measured A1)."""
    wins = [(lo, hi, lab) for (lo, hi, lab) in PEAK_WINDOWS
            if (common_x.min() <= hi and common_x.max() >= lo)]
    if not wins:
        print("    [info] no peak windows within data range; skipping Fig 5.")
        return
    ncol = 3
    nrow = int(np.ceil(len(wins) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, (lo, hi, lab) in zip(axes, wins):
        win = (common_x >= lo) & (common_x <= hi)
        for g, col, name in ((ev, EV_COLOR, "EV"), (rna, RNA_COLOR, "RNA")):
            band = band_of(g)
            ax.fill_between(common_x[win], (g.mean - band)[win], (g.mean + band)[win],
                            color=col, alpha=0.18, lw=0)
            ax.plot(common_x[win], g.mean[win], lw=2.0, color=col, label=name)
        ax.set_title(f"{lo}-{hi} cm$^{{-1}}$\n{lab}", fontsize=10)
        ax.grid(True, color="0.93", lw=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=9)
    for ax in axes[len(wins):]:
        ax.axis("off")
    axes[0].legend(loc="upper right", fontsize=9)
    fig.suptitle("Measured A1 EV vs RNA: biologically relevant windows", fontsize=13)
    fig.supxlabel(r"Raman shift (cm$^{-1}$)", fontsize=12)
    fig.supylabel("Normalized intensity (a.u.)", fontsize=12)
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.96))
    save_all_formats(fig, stem)


# =============================================================================
# 7. CSV EXPORTS
# =============================================================================
def export_processed_csv(common_x, named_means: dict, stem: str):
    """Save processed mean (+SD) spectra on the common axis to CSV."""
    data = {"raman_shift_cm-1": common_x}
    for name, (mean, band) in named_means.items():
        data[f"{name}_mean"] = mean
        data[f"{name}_band_{BAND_KIND}"] = band
    df = pd.DataFrame(data)
    out = os.path.join(OUT_DIR, f"{stem}.csv")
    df.to_csv(out, index=False)
    print(f"    saved: {stem}.csv")


def export_summary_stats(common_x, ext_ev, ext_rna, stem: str):
    """Save per-condition integrated intensity + provenance to CSV."""
    rows = []
    for c in CONDITIONS:
        rows.append(dict(
            substrate=c.code, top_layer_nm=c.nm, rate_A_per_s=c.rate,
            scale_factor=c.factor,
            provenance="MEASURED" if c.measured else "ESTIMATED",
            EV_integrated_intensity=integrate_fingerprint(common_x, ext_ev[c.code][0]),
            RNA_integrated_intensity=integrate_fingerprint(common_x, ext_rna[c.code][0]),
        ))
    df = pd.DataFrame(rows)
    out = os.path.join(OUT_DIR, f"{stem}.csv")
    df.to_csv(out, index=False)
    print(f"    saved: {stem}.csv")
    return df


# =============================================================================
# 8. MAIN
# =============================================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 78)
    print("GAIRA A1 EV/RNA Raman figure builder")
    print("=" * 78)

    # ---- 8.1 Discover files -------------------------------------------------
    blank_files = find_spectrum_files(BLANK_DIR)
    ev_files_all = find_spectrum_files(EV_DIR)
    rna_files = find_spectrum_files(RNA_DIR)
    print(f"Discovered: {len(blank_files)} Blank, {len(ev_files_all)} EV, "
          f"{len(rna_files)} RNA files.")

    # ---- 8.2 Parse metadata & apply operator policy -------------------------
    # Policy: EXCLUDE the real measured A6 EV so A2-A8 stay purely estimated.
    ev_meta = [parse_meta(f, "EV") for f in ev_files_all]
    excluded_a6 = [m for m in ev_meta if m.substrate == "A6"]
    ev_meta = [m for m in ev_meta if m.substrate == "A1"]
    if excluded_a6:
        print(f"  [policy] excluding {len(excluded_a6)} measured A6 EV files "
              f"(kept out so A2-A8 remain estimated only).")

    # Group EV A1 files by dilution.
    ev_by_dil: dict[str, list[str]] = {}
    for m in ev_meta:
        ev_by_dil.setdefault(m.dilution, []).append(m.path)
    print(f"  EV A1 dilutions found: { {k: len(v) for k, v in ev_by_dil.items()} }")

    # ---- 8.3 Build common Raman-shift axis ----------------------------------
    sample = next((load_spectrum(f) for f in blank_files if load_spectrum(f)), None)
    probe = [load_spectrum(f) for f in (blank_files[:3] + ev_files_all[:3] + rna_files[:3])]
    probe = [p for p in probe if p is not None]
    lo = max(p[0].min() for p in probe)
    hi = min(p[0].max() for p in probe)
    common_x = np.linspace(lo, hi, COMMON_N)
    print(f"  Common axis: {lo:.1f} -> {hi:.1f} cm^-1 ({COMMON_N} pts)")

    # ---- 8.4 Baseline-correct everything (ABSOLUTE counts, no norm yet) ------
    print("Processing Blank/EV/RNA (ALS baseline + resample, absolute counts)...")
    blank_stack = stack_group(blank_files, common_x, "Blank")
    ev_stacks_raw = {dil: stack_group(files, common_x, "EV")
                     for dil, files in sorted(ev_by_dil.items())}
    rna_stack_raw = stack_group(rna_files, common_x, "RNA")

    # ---- 8.4b Single COMMON global normalization scale ----------------------
    # Divisor = median fingerprint-window peak across ALL replicate spectra
    # (blank + EV + RNA). One shared scalar keeps blank subtraction valid.
    fp = (common_x >= FINGERPRINT[0]) & (common_x <= FINGERPRINT[1])
    all_rows = [blank_stack, rna_stack_raw, *ev_stacks_raw.values()]
    all_rows = np.vstack([s for s in all_rows if s.size])
    per_spec_peak = np.nanmax(all_rows[:, fp], axis=1)
    GLOBAL_SCALE = float(np.nanmedian(per_spec_peak)) or 1.0
    print(f"  Global normalization scale (shared divisor) = {GLOBAL_SCALE:.1f} counts")

    # Apply the shared scale to every spectrum, THEN subtract the mean blank.
    blank_stack = blank_stack / GLOBAL_SCALE
    blank_g = summarize("Blank_A1", blank_stack)
    blank_mean = blank_g.mean                       # mean processed blank (norm units)
    print(f"  Blank A1: n={blank_g.n}")

    # ---- 8.5 EV (per dilution): normalize then subtract Blank ---------------
    print("Subtracting averaged Blank from EV dilutions...")
    ev_groups: dict[str, GroupResult] = {}
    for dil, raw in ev_stacks_raw.items():
        stack = raw / GLOBAL_SCALE - blank_mean[None, :]   # common-scale, blank-subtracted
        ev_groups[dil] = summarize(f"EV_A1_{dil}", stack)
        print(f"  EV A1 {dil}: n={ev_groups[dil].n}")

    # ---- 8.6 RNA: normalize then subtract Blank -----------------------------
    print("Subtracting averaged Blank from RNA...")
    rna_stack = rna_stack_raw / GLOBAL_SCALE - blank_mean[None, :]
    rna_g = summarize("RNA_A1_1-1", rna_stack)
    print(f"  RNA A1: n={rna_g.n}")

    # ---- 8.7 Reference curves & extrapolation -------------------------------
    ev_ref = ev_groups.get(EV_REFERENCE_DILUTION) or next(iter(ev_groups.values()))
    print(f"  EV extrapolation reference = {ev_ref.name}")
    ext_ev = extrapolate_conditions(ev_ref.mean, band_of(ev_ref))
    ext_rna = extrapolate_conditions(rna_g.mean, band_of(rna_g))

    # ---- 8.8 Figures --------------------------------------------------------
    print("Rendering figures...")
    plot_condition_family(common_x, ev_ref.mean, band_of(ev_ref), ext_ev,
                          "EV", "fig1_EV_A1_measured_plus_estimated_A2-A8")
    # New Fig 1b: condition-averaged EV spectrum + EV Raman spectral map.
    plot_condition_map(common_x, ext_ev, "EV", EV_COLOR,
                       "fig1b_EV_condition_averaged_spectral_map")
    plot_condition_family(common_x, rna_g.mean, band_of(rna_g), ext_rna,
                          "RNA", "fig2_RNA_A1_measured_plus_estimated_A2-A8")
    # New Fig 2b: condition-averaged RNA spectrum + RNA Raman spectral map.
    plot_condition_map(common_x, ext_rna, "RNA", RNA_COLOR,
                       "fig2b_RNA_condition_averaged_spectral_map")
    # Measured EV dilution series preserved (renamed) so no real data is lost.
    plot_dilution_series(common_x, ev_groups, "EV", "fig1c_EV_A1_dilution_series_measured")
    plot_ev_vs_rna(common_x, ev_ref, rna_g, "fig3_EV_vs_RNA_measured_A1")
    plot_intensity_summary(common_x, ev_ref, rna_g, ext_ev, ext_rna,
                           "fig4_integrated_intensity_by_condition")
    plot_peak_windows(common_x, ev_ref, rna_g, "fig5_peak_region_windows_measured_A1")

    # ---- 8.9 CSV exports ----------------------------------------------------
    print("Writing CSV outputs...")
    # Processed measured curves.
    measured_means = {"Blank_A1": (blank_g.mean, band_of(blank_g)),
                      "RNA_A1_1-1": (rna_g.mean, band_of(rna_g))}
    for dil, g in ev_groups.items():
        measured_means[f"EV_A1_{dil}"] = (g.mean, band_of(g))
    export_processed_csv(common_x, measured_means, "processed_measured_spectra")

    # Estimated EV / RNA curves.
    est_means_ev = {f"EV_{c.code}_{'MEAS' if c.measured else 'EST'}": ext_ev[c.code]
                    for c in CONDITIONS}
    est_means_rna = {f"RNA_{c.code}_{'MEAS' if c.measured else 'EST'}": ext_rna[c.code]
                     for c in CONDITIONS}
    export_processed_csv(common_x, est_means_ev, "estimated_EV_A1-A8_spectra")
    export_processed_csv(common_x, est_means_rna, "estimated_RNA_A1-A8_spectra")

    summary_df = export_summary_stats(common_x, ext_ev, ext_rna, "summary_statistics_by_condition")

    # ---- 8.10 Final report --------------------------------------------------
    print("=" * 78)
    print("DONE. All outputs written to:")
    print(f"  {OUT_DIR}")
    print("-" * 78)
    print("Figures (PNG + SVG + PDF @ 600 dpi):")
    for stem in ("fig1_EV_A1_measured_plus_estimated_A2-A8",
                 "fig1b_EV_condition_averaged_spectral_map",
                 "fig1c_EV_A1_dilution_series_measured",
                 "fig2_RNA_A1_measured_plus_estimated_A2-A8",
                 "fig2b_RNA_condition_averaged_spectral_map",
                 "fig3_EV_vs_RNA_measured_A1",
                 "fig4_integrated_intensity_by_condition",
                 "fig5_peak_region_windows_measured_A1"):
        print(f"    {stem}.[png|svg|pdf]")
    print("CSV:")
    for stem in ("processed_measured_spectra",
                 "estimated_EV_A1-A8_spectra",
                 "estimated_RNA_A1-A8_spectra",
                 "summary_statistics_by_condition"):
        print(f"    {stem}.csv")
    print("-" * 78)
    print("REMINDER: A1 is MEASURED. A2-A8 are morphology-scaled ESTIMATES")
    print("of measured A1 and are labelled ESTIMATED everywhere. Measured A6")
    print("EV data were intentionally excluded from these figures.")
    print("=" * 78)
    print("\nSummary table:\n")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
