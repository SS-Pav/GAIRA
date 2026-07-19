"""PCA + bootstrap loading stability with sign alignment (Phase 2 §9)."""
from __future__ import annotations
import numpy as np
from sklearn.decomposition import PCA


def fit_pca(X, n_components=10, seed=0):
    n_components = min(n_components, X.shape[0] - 1, X.shape[1])
    p = PCA(n_components=n_components, random_state=seed).fit(X)
    return p, p.transform(X)


def _sign_align(ref, comp):
    """Flip comp components to best match ref (resolves PCA sign ambiguity)."""
    out = comp.copy()
    for k in range(min(len(ref), len(comp))):
        if np.dot(ref[k], comp[k]) < 0:
            out[k] = -comp[k]
    return out


def bootstrap_stability(X, groups, n_components=6, n_boot=200, seed=0):
    """Bootstrap by GROUP (analyte), not spectrum (§13). Resample analytes with
    replacement, refit PCA, sign-align loadings to the full-data fit, and report
    per-component cosine stability of loading vectors."""
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    full, _ = fit_pca(X, n_components, seed)
    ref = full.components_
    nc = ref.shape[0]
    sims = [[] for _ in range(nc)]
    for _ in range(n_boot):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(groups == g)[0] for g in samp])
        if len(idx) < nc + 2:
            continue
        p, _ = fit_pca(X[idx], nc, seed)
        comp = _sign_align(ref, p.components_)
        for k in range(min(nc, comp.shape[0])):
            a, b = ref[k], comp[k]
            sims[k].append(abs(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))))
    stab = [float(np.mean(s)) if s else 0.0 for s in sims]
    stab_lo = [float(np.percentile(s, 5)) if s else 0.0 for s in sims]
    return {"explained_variance_ratio": full.explained_variance_ratio_.tolist(),
            "loading_stability_mean": stab, "loading_stability_p5": stab_lo,
            "components": ref, "n_boot_effective": len(sims[0])}
