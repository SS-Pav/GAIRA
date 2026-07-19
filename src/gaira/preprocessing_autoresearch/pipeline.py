"""Stage B0 — candidate pipeline execution, split into two stages.

STAGE 1 (fold-independent, cacheable): crop -> modality-specific baseline ->
         smoothing -> resample onto the FIXED common grid (520-1750 @ 2 cm-1).
         Purely per-spectrum, so it can never leak across folds.

STAGE 2 (fold-dependent): Ag-SERS background model (fitted on TRAIN Ag-SERS only)
         -> replicate aggregation -> derivative -> normalization -> peak transform.

Leakage rules enforced here:
  * background models see only training-fold Ag-SERS spectra;
  * no Raman spectrum ever influences an Ag-SERS spectrum;
  * no analyte labels are used to build spectral vectors;
  * aggregation only ever combines replicates WITHIN one analyte x modality group.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import hashlib, json
import numpy as np

from ..preprocessing.pipeline import common_grid, crop, resample  # reuse
from . import smoothing as SM
from . import normalization as NM
from . import background_models as BG
from .derivatives import apply_derivative

GRID = common_grid(520.0, 1750.0, 2.0)
WINDOW = (520.0, 1750.0)


# ───────────────────────── candidate definition ─────────────────────────
@dataclass
class Candidate:
    cid: str
    arm: str
    # modality-specific stage 1
    raman: dict = field(default_factory=lambda: dict(baseline="asls", baseline_params={},
                                                     smooth="none", smooth_params={}))
    sers: dict = field(default_factory=lambda: dict(baseline="asls", baseline_params={},
                                                    smooth="none", smooth_params={}))
    # stage 2
    background: tuple = ("none", {})          # Ag-SERS only
    aggregate: str = "mean"
    derivative: object = 0
    norm_raman: str = "l2"
    norm_sers: str = "l2"
    peak_transform: str = "full"

    def stage1_key(self, modality):
        cfg = self.raman if modality == "raman" else self.sers
        return hashlib.md5(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]

    def smooth_name(self, modality="raman"):
        cfg = self.raman if modality == "raman" else self.sers
        p = cfg.get("smooth_params", {})
        return cfg["smooth"] + ("" if not p else "(" + ",".join(f"{k}={v}" for k, v in sorted(p.items())) + ")")

    def baseline_name(self, modality="raman"):
        cfg = self.raman if modality == "raman" else self.sers
        p = cfg.get("baseline_params", {})
        return cfg["baseline"] + ("" if not p else "(" + ",".join(f"{k}={v}" for k, v in sorted(p.items())) + ")")

    def to_dict(self):
        return {"cid": self.cid, "arm": self.arm, "raman": self.raman, "sers": self.sers,
                "background": {"method": self.background[0], "params": self.background[1]},
                "aggregate": self.aggregate, "derivative": self.derivative,
                "norm_raman": self.norm_raman, "norm_sers": self.norm_sers,
                "peak_transform": self.peak_transform,
                "n_stages": self.n_stages(), "n_hyperparams": self.n_hyperparams()}

    def n_stages(self):
        n = 0
        for m in (self.raman, self.sers):
            n += int(m["baseline"] != "none") + int(m["smooth"] != "none")
        n += int(self.background[0] != "none") + int(self.aggregate not in ("none", "mean"))
        n += int(str(self.derivative) != "0") + int(self.norm_raman != "none") + \
             int(self.peak_transform != "full")
        return n

    def n_hyperparams(self):
        return (len(self.raman.get("baseline_params", {})) + len(self.raman.get("smooth_params", {}))
                + len(self.sers.get("baseline_params", {})) + len(self.sers.get("smooth_params", {}))
                + len(self.background[1]))

    def modality_specific(self):
        return (self.raman != self.sers) or (self.norm_raman != self.norm_sers)


# ───────────────────────── stage 1 ─────────────────────────
def stage1_spectrum(wn, y, cfg):
    """Crop -> baseline -> smooth -> resample. Per-spectrum; no cross-spectrum info."""
    wn2, y2 = crop(np.asarray(wn, float), np.asarray(y, float), *WINDOW)
    if len(wn2) < 20:
        return np.full(len(GRID), np.nan)
    z = SM.BASELINES[cfg["baseline"]](y2, **cfg.get("baseline_params", {}))
    y2 = y2 - z
    y2 = SM.SMOOTHERS[cfg["smooth"]](y2, **cfg.get("smooth_params", {}))
    return resample(wn2, y2, GRID)


class Stage1Cache:
    """Caches stage-1 output per (modality-config) so the search does not redo
    baseline/smoothing for every candidate. Fold-independent by construction."""
    def __init__(self, raw):
        self.raw = raw                      # list of (spectrum_id, wn, y, modality)
        self._cache = {}

    def get(self, cfg):
        key = hashlib.md5(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()
        if key not in self._cache:
            self._cache[key] = np.vstack([stage1_spectrum(wn, y, cfg) for _, wn, y, _ in self.raw])
        return self._cache[key]

    def build(self, cand):
        """Stage-1 matrix for all spectra using the modality-appropriate config."""
        R = self.get(cand.raman); S = self.get(cand.sers)
        mod = np.array([m for _, _, _, m in self.raw])
        out = np.where((mod == "raman")[:, None], R, S)
        return out


# ───────────────────────── stage 2 (fold-fitted) ─────────────────────────
def fit_stage2(cand, X1, meta, train_mask):
    """Fit the fold-dependent parts on TRAINING spectra only. Returns a fitted state."""
    is_sers = (meta.modality.values == "sers")
    tr_sers = train_mask & is_sers
    bgm = BG.make(cand.background[0], **cand.background[1])
    if tr_sers.sum() >= 2:
        bgm.fit(X1[tr_sers])
    else:
        bgm = BG.make("none")
    return {"background": bgm}


def apply_stage2(cand, X1, meta, state, aggregate=True):
    """Apply the fitted state. Returns (features, feat_meta).
    If aggregate, one row per (analyte, modality) group."""
    is_sers = (meta.modality.values == "sers")
    X = state["background"].transform(X1, is_sers_mask=is_sers)

    if aggregate:
        # An analyte-level representative is REQUIRED for cross-modal retrieval, so
        # aggregation always produces one row per (analyte, modality). "none" is not
        # a valid representative and falls back to the arithmetic mean.
        method = "mean" if cand.aggregate in (None, "none") else cand.aggregate
        rows, rmeta = [], []
        for (a, m), idx in meta.groupby(["analyte", "modality"], sort=True).groups.items():
            pos = [meta.index.get_loc(i) for i in idx]
            rows.append(NM.aggregate(X[pos], method))
            rmeta.append({"analyte": a, "modality": m, "n_rep": len(pos)})
        import pandas as pd
        X = np.vstack(rows); fmeta = pd.DataFrame(rmeta)
    else:
        fmeta = meta.reset_index(drop=True).copy()

    X = apply_derivative(X, cand.derivative)
    X = _peak_transform(X, cand.peak_transform)

    out = np.empty_like(X, dtype=float)
    mod = fmeta.modality.values
    for i in range(len(X)):
        nm = cand.norm_raman if mod[i] == "raman" else cand.norm_sers
        out[i] = NM.NORMALIZERS[nm](np.nan_to_num(X[i]))
    return out, fmeta


def _peak_transform(X, kind):
    if kind in ("full", "smoothed", "derivative"):
        return X                      # smoothing/derivative already handled upstream
    from scipy.ndimage import maximum_filter1d, gaussian_filter1d
    Xn = np.nan_to_num(X)
    if kind == "prominence":
        env = maximum_filter1d(Xn, size=25, mode="nearest")
        return np.clip(Xn - gaussian_filter1d(Xn, 8, mode="nearest"), 0, None) * (Xn >= env - 1e-12)
    if kind == "peak_likelihood":
        d2 = np.gradient(np.gradient(Xn, axis=-1), axis=-1)
        return np.clip(-d2, 0, None)
    if kind == "ridge":
        acc = np.zeros_like(Xn)
        for s in (2, 4, 8):
            acc += np.clip(-np.gradient(np.gradient(gaussian_filter1d(Xn, s, mode="nearest"),
                                                    axis=-1), axis=-1), 0, None)
        return acc
    return X
