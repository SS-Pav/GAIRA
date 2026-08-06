"""GAIRA V7 — Phase 04.5: physically-motivated Raman perturbations.

Twelve corruptions, each modelling something that actually happens to a Raman measurement
rather than an abstract noise model. Every one is deterministic given its seed, and every one
is applied to the *spectrum* — before preprocessing-equivalent normalisation — so the
downstream representations see the same spectrum a worse instrument would have produced.

The point of the study is not that performance degrades. It is *where* it degrades: if the
higher layers of the hierarchy hold up under corruptions that destroy raw retrieval, the
abstraction is buying something. If they degrade in lockstep, it is not.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12
PERTURBATIONS = ("gaussian_noise", "shot_noise", "baseline_drift", "fluorescence",
                 "cosmic_spikes", "intensity_scaling", "peak_scaling", "wavelength_shift",
                 "spectral_stretch", "peak_dropout", "band_broadening", "combined")
# Sweep levels, in the units each perturbation is natural in. Chosen to span "barely visible"
# to "an analyst would reject this spectrum".
LEVELS = {
    "gaussian_noise": (0.01, 0.02, 0.05, 0.10, 0.20),
    "shot_noise": (0.01, 0.02, 0.05, 0.10, 0.20),
    "baseline_drift": (0.05, 0.10, 0.20, 0.40, 0.80),
    "fluorescence": (0.05, 0.10, 0.20, 0.40, 0.80),
    "cosmic_spikes": (1, 2, 4, 8, 16),
    "intensity_scaling": (0.1, 0.25, 0.5, 1.0, 2.0),
    "peak_scaling": (0.05, 0.10, 0.20, 0.40, 0.80),
    "wavelength_shift": (1.0, 2.0, 4.0, 8.0, 16.0),
    "spectral_stretch": (0.001, 0.002, 0.005, 0.010, 0.020),
    "peak_dropout": (0.05, 0.10, 0.20, 0.40, 0.60),
    "band_broadening": (1.0, 2.0, 4.0, 8.0, 16.0),
    "combined": (0.25, 0.5, 1.0, 1.5, 2.0),
}


def _renorm(y):
    y = np.clip(y, 0.0, None)
    n = np.linalg.norm(y)
    return y / (n + EPS)


def gaussian_noise(y, grid, level, rng):
    """Detector read noise — additive, independent of signal."""
    return _renorm(y + rng.normal(0.0, level * float(np.abs(y).max()), y.shape))


def shot_noise(y, grid, level, rng):
    """Photon counting noise — variance proportional to signal, so peaks are noisiest."""
    scale = level * float(np.abs(y).max())
    return _renorm(y + rng.normal(0.0, 1.0, y.shape) * scale * np.sqrt(np.clip(y, 0, None)
                                                                       / (y.max() + EPS)))


def baseline_drift(y, grid, level, rng):
    """Slow instrumental drift — a low-order ramp under the whole spectrum."""
    g = (grid - grid.min()) / (grid.max() - grid.min() + EPS)
    a, b = rng.normal(0, 1, 2)
    return _renorm(y + level * float(np.abs(y).max()) * (a * g + b * g ** 2) / 2.0)


def fluorescence(y, grid, level, rng):
    """Broad polynomial fluorescence background — the dominant real-world corruption."""
    g = (grid - grid.min()) / (grid.max() - grid.min() + EPS)
    c = rng.normal(0, 1, 4)
    bg = c[0] + c[1] * g + c[2] * g ** 2 + c[3] * g ** 3
    bg = (bg - bg.min()) / (bg.max() - bg.min() + EPS)
    return _renorm(y + level * float(np.abs(y).max()) * bg)


def cosmic_spikes(y, grid, level, rng):
    """Cosmic-ray hits — a few very narrow, very tall spikes."""
    out = y.copy()
    for _ in range(int(level)):
        i = int(rng.integers(2, len(y) - 2))
        out[i] += float(y.max()) * float(rng.uniform(1.0, 5.0))
    return _renorm(out)


def intensity_scaling(y, grid, level, rng):
    """Global gain change — should be invisible to any L2-normalised representation."""
    return _renorm(y * float(np.exp(rng.normal(0, level))))


def peak_scaling(y, grid, level, rng):
    """Per-band intensity variation — polarisation, orientation, packing."""
    from scipy.ndimage import gaussian_filter1d
    f = gaussian_filter1d(rng.normal(0, 1, y.shape), 12.0)
    f = f / (np.abs(f).max() + EPS)
    return _renorm(y * (1.0 + level * f))


def wavelength_shift(y, grid, level, rng):
    """Calibration offset in cm-1 — the perturbation peak positions are supposed to resist."""
    step = float(np.median(np.diff(grid)))
    shift = float(rng.uniform(-level, level)) / step
    idx = np.arange(len(y)) - shift
    return _renorm(np.interp(idx, np.arange(len(y)), y, left=0.0, right=0.0))


def spectral_stretch(y, grid, level, rng):
    """Multiplicative axis error — a shift that grows across the window."""
    f = 1.0 + float(rng.uniform(-level, level))
    idx = np.clip((np.arange(len(y)) - len(y) / 2) * f + len(y) / 2, 0, len(y) - 1)
    return _renorm(np.interp(idx, np.arange(len(y)), y))


def peak_dropout(y, grid, level, rng):
    """Whole bands missing — masked regions, detector defects, or genuine absence."""
    from scipy.signal import find_peaks
    out = y.copy()
    idx, _ = find_peaks(y / (y.max() + EPS), prominence=0.02)
    if idx.size:
        drop = rng.choice(idx, size=max(1, int(level * idx.size)), replace=False)
        for i in drop:
            lo, hi = max(0, i - 5), min(len(y), i + 6)
            out[lo:hi] = np.linspace(out[lo], out[hi - 1], hi - lo)
    return _renorm(out)


def band_broadening(y, grid, level, rng):
    """Instrumental resolution loss — the corruption that most destroys band identity."""
    from scipy.ndimage import gaussian_filter1d
    step = float(np.median(np.diff(grid)))
    return _renorm(gaussian_filter1d(y, max(level / step, 0.1)))


def combined(y, grid, level, rng):
    """Everything at once, scaled — what a genuinely bad measurement looks like."""
    out = gaussian_noise(y, grid, 0.03 * level, rng)
    out = fluorescence(out, grid, 0.15 * level, rng)
    out = baseline_drift(out, grid, 0.10 * level, rng)
    out = wavelength_shift(out, grid, 2.0 * level, rng)
    out = peak_scaling(out, grid, 0.10 * level, rng)
    return out


DISPATCH = {
    "gaussian_noise": gaussian_noise, "shot_noise": shot_noise,
    "baseline_drift": baseline_drift, "fluorescence": fluorescence,
    "cosmic_spikes": cosmic_spikes, "intensity_scaling": intensity_scaling,
    "peak_scaling": peak_scaling, "wavelength_shift": wavelength_shift,
    "spectral_stretch": spectral_stretch, "peak_dropout": peak_dropout,
    "band_broadening": band_broadening, "combined": combined,
}


def apply(kind: str, X: np.ndarray, grid: np.ndarray, level: float, seed: int = 0) -> np.ndarray:
    """Perturb every row of `X`. Deterministic given `seed`."""
    rng = np.random.default_rng(seed)
    fn = DISPATCH[kind]
    return np.vstack([fn(np.asarray(x, float), grid, level, rng) for x in X])
