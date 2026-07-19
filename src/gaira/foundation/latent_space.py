"""GAIRA V5 Foundation — Phase C2: the frozen biochemical manifold.

Fits the selected representation on the FULL Raman reference corpus (the reference
space is the deliverable; all external data is projected into it, never used to fit
it) and characterises explained variance, intrinsic dimensionality, neighbourhood
stability and bootstrap robustness before freezing.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from . import representation as RP
from .benchmark import _unit, neighbourhood_preservation


@dataclass
class Manifold:
    rep: object                 # fitted Representation
    grid: np.ndarray
    name: str
    k: int
    stats: dict

    def project(self, X):
        """Project spectra into the frozen manifold (non-negative activations)."""
        return self.rep.transform(np.nan_to_num(X))

    def coordinates(self, X, normalise=True):
        """BSV-style coordinates: activations, optionally L1-normalised to
        proportions so a coordinate reads as 'share of biochemical evidence'."""
        Z = np.clip(self.project(X), 0, None)
        if normalise:
            s = Z.sum(axis=1, keepdims=True)
            Z = np.divide(Z, s, out=np.zeros_like(Z), where=s > 1e-12)
        return Z


def explained_variance(X, rep):
    Xn = np.nan_to_num(X)
    R = rep.reconstruct(Xn)
    ss_res = float(np.sum((Xn - R) ** 2))
    ss_tot = float(np.sum((Xn - Xn.mean(0)) ** 2))
    return 1.0 - ss_res / (ss_tot + 1e-12)


def per_component_variance(X, rep):
    """Variance each component alone accounts for (non-orthogonal safe)."""
    Xn = np.nan_to_num(X)
    Z = rep.transform(Xn)
    ss_tot = float(np.sum((Xn - Xn.mean(0)) ** 2)) + 1e-12
    out = []
    for j in range(rep.components_.shape[0]):
        Rj = np.outer(Z[:, j], rep.components_[j])
        out.append(float(np.sum(Rj ** 2) / ss_tot))
    return np.array(out)


def intrinsic_dimensionality(Z, max_k=None):
    """Two estimates: participation ratio of the latent covariance spectrum, and
    the number of components needed for 90% of latent variance."""
    Zc = np.nan_to_num(Z) - np.nan_to_num(Z).mean(0)
    s = np.linalg.svd(Zc, compute_uv=False) ** 2
    p = s / (s.sum() + 1e-12)
    pr = float((s.sum() ** 2) / (np.sum(s ** 2) + 1e-12))
    n90 = int(np.searchsorted(np.cumsum(p), 0.90) + 1)
    ent = float(np.exp(-np.sum(p * np.log(p + 1e-12))))
    return {"participation_ratio": pr, "n_components_90pct_latent_var": n90,
            "effective_rank_entropy": ent}


def bootstrap_stability(X, groups, name, k, seed=0, n_boot=20):
    """Component reproducibility under analyte bootstrap (sign/order agnostic)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    ref = _unit(RP.FITTERS[name](X, k, seed).components_)
    sims = []
    for b in range(n_boot):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(groups == g)[0] for g in samp])
        if len(idx) < k + 2:
            continue
        try:
            r = RP.FITTERS[name](X[idx], k, seed + 500 + b)
        except Exception:
            continue
        sims.append(np.abs(ref @ _unit(r.components_).T).max(axis=1))
    if not sims:
        return {"mean": np.nan, "per_component": []}
    S = np.vstack(sims)
    return {"mean": float(S.mean()), "p10": float(np.percentile(S, 10)),
            "per_component": S.mean(axis=0).tolist()}


def neighbourhood_stability(X, Z, groups, seed=0, n_boot=20, k_nn=10):
    """How reproducible are latent analyte neighbourhoods under resampling?"""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    base = neighbourhood_preservation(X, Z, k=k_nn)
    vals = []
    for _ in range(n_boot):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(groups == g)[0] for g in samp])
        if len(idx) < k_nn + 3:
            continue
        vals.append(neighbourhood_preservation(X[idx], Z[idx], k=k_nn))
    return {"full": base, "bootstrap_mean": float(np.nanmean(vals)) if vals else np.nan,
            "bootstrap_std": float(np.nanstd(vals)) if vals else np.nan}


def build_manifold(corpus, name, k, seed=0):
    X, meta, grid = corpus.X, corpus.meta, corpus.grid
    rep = RP.FITTERS[name](X, k, seed)
    Z = rep.transform(np.nan_to_num(X))
    stats = {
        "representation": name, "k": int(k), "seed": seed,
        "n_spectra": int(len(X)), "n_analytes": int(meta.analyte.nunique()),
        "explained_variance": explained_variance(X, rep),
        "per_component_variance": per_component_variance(X, rep).tolist(),
        "intrinsic_dimensionality": intrinsic_dimensionality(Z),
        "bootstrap_component_stability": bootstrap_stability(X, meta.analyte.values, name, k, seed),
        "neighbourhood_stability": neighbourhood_stability(X, Z, meta.analyte.values, seed),
        "activation_sparsity_per_spectrum": float(np.mean(np.mean(
            np.clip(Z, 0, None) < 1e-8, axis=1))),
    }
    return Manifold(rep, grid, name, k, stats)
