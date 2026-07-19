"""Bounded, documented spectral augmentations (§10) for encoder training.

Every augmentation reflects plausible measurement variation and is BOUNDED so it
cannot alter biochemical identity. No aggressive peak warping. A validity audit
(augmentation_audit) verifies major bands survive (not erased, not invented).
Deterministic given a seed.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class AugConfig:
    intensity_scale: float = 0.05     # ±5% multiplicative
    baseline_amp: float = 0.02        # low-order baseline drift amplitude (rel. to std)
    noise_amp: float = 0.01           # additive gaussian (rel. to std)
    smooth_prob: float = 0.3          # chance to apply mild extra smoothing
    max_shift_bins: int = 1           # ≤1 grid bin (~2 cm-1) wavenumber shift
    mask_frac: float = 0.05           # local contiguous dropout fraction
    mask_prob: float = 0.3


def _renorm(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def augment(v, grid, cfg: AugConfig, rng):
    """Apply a bounded random augmentation to one spectrum vector v (on grid)."""
    x = np.array(v, dtype=float)
    s = np.std(x) + 1e-9
    # small intensity scaling
    x = x * (1.0 + rng.uniform(-cfg.intensity_scale, cfg.intensity_scale))
    # low-order baseline drift (linear + quadratic), bounded
    t = np.linspace(-1, 1, len(x))
    x = x + s * cfg.baseline_amp * (rng.uniform(-1, 1) * t + rng.uniform(-1, 1) * 0.5 * (t ** 2 - 0.5))
    # additive noise
    x = x + rng.normal(0, s * cfg.noise_amp, size=len(x))
    # mild smoothing sometimes
    if rng.random() < cfg.smooth_prob:
        k = np.array([0.25, 0.5, 0.25])
        x = np.convolve(x, k, mode="same")
    # ≤1-bin wavenumber shift
    sh = int(rng.integers(-cfg.max_shift_bins, cfg.max_shift_bins + 1))
    if sh != 0:
        x = np.roll(x, sh)
        if sh > 0: x[:sh] = x[sh]
        else: x[sh:] = x[sh - 1]
    # local contiguous masking
    if rng.random() < cfg.mask_prob and cfg.mask_frac > 0:
        w = max(1, int(cfg.mask_frac * len(x)))
        st = int(rng.integers(0, len(x) - w))
        x[st:st + w] = x[st:st + w] * rng.uniform(0.0, 0.3)
    return _renorm(x)


def augmentation_audit(X, grid, cfg: AugConfig, seed=0, n_examples=6, band_tol=6.0):
    """Verify augmentations preserve major bands. For each example spectrum, compare
    the top peaks before/after augmentation. Returns audit stats + example pairs."""
    from ..representation.metrics import peaks
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n_examples, len(X)), replace=False)
    kept, invented, examples = [], [], []
    for i in idx:
        orig = _renorm(np.nan_to_num(X[i]))
        aug = augment(orig, grid, cfg, rng)
        po = peaks(orig, grid, prominence_frac=0.15)
        pa = peaks(aug, grid, prominence_frac=0.15)
        if len(po):
            k = np.mean([any(abs(p - q) <= band_tol for q in pa) for p in po])
            kept.append(k)
        # invented = strong aug peaks with no original support
        if len(pa):
            inv = np.mean([not any(abs(q - p) <= band_tol for p in po) for q in pa])
            invented.append(inv)
        examples.append({"index": int(i), "orig": orig.tolist(), "aug": aug.tolist(),
                         "orig_peaks": po.tolist(), "aug_peaks": pa.tolist()})
    return {"major_band_retention_mean": float(np.mean(kept)) if kept else None,
            "invented_peak_fraction_mean": float(np.mean(invented)) if invented else None,
            "band_tol_cm": band_tol, "config": cfg.__dict__, "examples": examples}
