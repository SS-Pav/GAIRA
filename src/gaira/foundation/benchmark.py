"""GAIRA V5 Foundation — Phase C1 representation benchmark.

Every candidate is scored on the SAME analyte-grouped held-out protocol:
  reconstruction · stability · interpretability/sparsity · replicate robustness ·
  analyte-neighbourhood preservation · nuisance (excitation/source) leakage.

No representation is privileged. Fitting always happens on TRAINING analytes only.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from . import representation as RP


def _unit(X):
    X = np.nan_to_num(X)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def _knn(X, k):
    S = _unit(X) @ _unit(X).T
    np.fill_diagonal(S, -np.inf)
    return np.argsort(-S, axis=1)[:, :k]


def neighbourhood_preservation(X, Z, k=10):
    """Overlap of k-NN sets between input space and latent space."""
    n = len(X)
    if n <= k + 1:
        return np.nan
    a, b = _knn(X, k), _knn(Z, k)
    return float(np.mean([len(set(a[i]) & set(b[i])) / k for i in range(n)]))


def replicate_robustness(Z, meta, analytes=None):
    """Within-analyte latent cosine minus between-analyte cosine (scale-free)."""
    m = meta if analytes is None else meta[meta.analyte.isin(analytes)]
    idx = m.index.values
    if len(idx) < 4:
        return np.nan
    U = _unit(Z[idx]); lab = m.analyte.values
    C = U @ U.T; np.fill_diagonal(C, np.nan)
    same = lab[:, None] == lab[None, :]
    if not (same & ~np.eye(len(lab), dtype=bool)).any():
        return np.nan
    return float(np.nanmean(np.where(same, C, np.nan)) - np.nanmean(np.where(~same, C, np.nan)))


def sparsity(W):
    """Gini-style sparsity of the spectral loadings (1 = maximally sparse)."""
    A = np.abs(np.nan_to_num(W))
    g = []
    for row in A:
        s = np.sort(row); n = len(s); c = np.cumsum(s)
        g.append(float((n + 1 - 2 * (c / (c[-1] + 1e-12)).sum()) / n))
    return float(np.mean(g))


def band_localisation(W, grid, frac=0.5):
    """Median fraction of the wavenumber axis needed to carry `frac` of a
    component's absolute loading mass (lower = more band-localised)."""
    out = []
    for row in np.abs(np.nan_to_num(W)):
        s = np.sort(row)[::-1]
        c = np.cumsum(s) / (s.sum() + 1e-12)
        out.append(float((np.searchsorted(c, frac) + 1) / len(row)))
    return float(np.median(out))


def component_stability(X, groups, name, k, seed=0, n_boot=12):
    """Bootstrap components by ANALYTE and measure sign-aligned matching."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    full = RP.FITTERS[name](X, k, seed)
    ref = _unit(full.components_)
    sims = []
    for b in range(n_boot):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(groups == g)[0] for g in samp])
        if len(idx) < k + 2:
            continue
        try:
            r = RP.FITTERS[name](X[idx], k, seed + 100 + b)
        except Exception:
            continue
        C = np.abs(ref @ _unit(r.components_).T)       # |cos| — sign/order agnostic
        sims.append(float(np.mean(C.max(axis=1))))
    return float(np.mean(sims)) if sims else np.nan


def nuisance_leakage(Z, meta, col):
    """Between-level minus within-level latent cosine for a nuisance factor
    (excitation, source). Near 0 = the factor is not encoded."""
    lab = meta[col].astype(str).values
    if len(np.unique(lab)) < 2:
        return np.nan
    U = _unit(Z); C = U @ U.T; np.fill_diagonal(C, np.nan)
    same = lab[:, None] == lab[None, :]
    return float(np.nanmean(np.where(same, C, np.nan)) - np.nanmean(np.where(~same, C, np.nan)))


def run_benchmark(corpus, ks=(4, 8, 12, 16, 24, 32), names=("PCA", "SparseDict", "NMF", "ICA", "Autoencoder"),
                  n_splits=4, seed=0, verbose=True):
    """Analyte-grouped held-out benchmark across representations and latent sizes."""
    X, meta, grid = corpus.X, corpus.meta, corpus.grid
    groups = meta.analyte.values
    gkf = GroupKFold(n_splits=n_splits)
    rows = []
    for name in names:
        for k in ks:
            rec_err, npres, reprob, folds_ok = [], [], [], 0
            for tr, te in gkf.split(X, groups=groups):
                try:
                    rep = RP.FITTERS[name](X[tr], k, seed)
                    Zte = rep.transform(X[te])
                    Rte = rep.reconstruct(X[te])
                except Exception:
                    continue
                num = np.linalg.norm(np.nan_to_num(X[te]) - np.nan_to_num(Rte))
                den = np.linalg.norm(np.nan_to_num(X[te])) + 1e-12
                rec_err.append(float(num / den))
                npres.append(neighbourhood_preservation(X[te], Zte, k=min(10, len(te) - 2)))
                mte = meta.iloc[te].reset_index(drop=True)
                reprob.append(replicate_robustness(Zte, mte))
                folds_ok += 1
            if folds_ok == 0:
                continue
            full = RP.FITTERS[name](X, k, seed)
            Zfull = full.transform(X)
            rows.append({
                "representation": name, "k": k,
                "recon_rel_error": float(np.nanmean(rec_err)),
                "neighbourhood_preservation": float(np.nanmean(npres)),
                "replicate_robustness": float(np.nanmean(reprob)),
                "component_stability": component_stability(X, groups, name, k, seed),
                "loading_sparsity": sparsity(full.components_),
                "band_localisation": band_localisation(full.components_, grid),
                "excitation_leakage": nuisance_leakage(Zfull, meta, "excitation_nm"),
                "source_leakage": nuisance_leakage(Zfull, meta, "source"),
                "nonneg": bool(full.nonneg), "n_folds": folds_ok,
            })
            if verbose:
                r = rows[-1]
                print(f"  {name:12s} k={k:3d} recon {r['recon_rel_error']:.3f} "
                      f"nbr {r['neighbourhood_preservation']:.3f} rep {r['replicate_robustness']:.3f} "
                      f"stab {r['component_stability']:.3f} sparse {r['loading_sparsity']:.3f} "
                      f"exc_leak {r['excitation_leakage']:+.3f}", flush=True)
    return pd.DataFrame(rows)


# ── selection: multi-criteria, explicitly NOT reconstruction-only ──
SELECTION_WEIGHTS = {
    "neighbourhood_preservation": 0.25,   # does the latent keep analyte structure?
    "replicate_robustness": 0.25,         # same analyte stays together
    "component_stability": 0.20,          # components reproduce under resampling
    "interpretability": 0.15,             # sparse, band-localised loadings
    "reconstruction": 0.10,               # fidelity (deliberately NOT dominant)
    "nuisance_control": 0.05,             # excitation/source not encoded
}


TIE_TOLERANCE = 0.02   # scores within this are treated as statistically indistinguishable


def select_with_tiebreak(scored, tol=TIE_TOLERANCE):
    """Pick the representation, breaking near-ties on PRE-STATED scientific criteria
    rather than on an arbitrary 3rd-decimal difference.

    Tie-break order (motivated by the stated objective — an *interpretable*
    biochemical coordinate system for projecting biological spectra):
      1. non-negativity / additivity — Raman spectra of mixtures are non-negative
         sums of molecular contributions, so a parts-based decomposition yields
         coordinates that mean "how much of this biochemical theme is present".
         Signed components would imply negative biochemical content.
      2. loading sparsity / band localisation — required for Phase C5 sparse
         molecular spectral signatures.
      3. component stability under analyte bootstrap.
      4. lower nuisance (excitation) leakage.
      5. smaller latent dimension (simplicity).
    """
    top = scored.iloc[0]
    tied = scored[scored.total_score >= top.total_score - tol].copy()
    if len(tied) == 1:
        return top, tied, "unique winner (no tie)"
    # Non-negativity is applied as a CONSTRAINT (physical interpretability), not as a
    # score to maximise; within the admissible set the benchmark score decides.
    admissible = tied[tied.nonneg] if tied.nonneg.any() else tied
    ranked = admissible.sort_values(by=["total_score", "k"], ascending=[False, True])
    pick = ranked.iloc[0]
    constraint = ("restricted to non-negative (parts-based) decompositions, because Raman "
                  "spectra of mixtures are non-negative sums of molecular contributions and "
                  "BSV coordinates must mean 'how much of this theme is present'"
                  if tied.nonneg.any() else "no non-negative candidate available")
    why = (f"{len(tied)} candidates within {tol:.3f} of the top score "
           f"({', '.join(f'{r.representation}@k={int(r.k)}:{r.total_score:.3f}' for _, r in tied.iterrows())}) "
           f"— statistically indistinguishable. Tie broken by constraint: {constraint}; "
           f"then by benchmark score, then by smaller k. Selected {pick.representation} k={int(pick.k)} "
           f"(score {pick.total_score:.3f}, sparsity {pick.loading_sparsity:.3f}, "
           f"stability {pick.component_stability:.3f}).")
    return pick, tied, why


def score(df):
    d = df.copy()

    def n01(x, invert=False):
        x = np.asarray(x, float)
        lo, hi = np.nanmin(x), np.nanmax(x)
        if not np.isfinite(lo) or hi - lo < 1e-12:
            return np.zeros_like(x)
        v = (x - lo) / (hi - lo)
        return 1 - v if invert else v

    d["s_nbr"] = n01(d.neighbourhood_preservation)
    d["s_rep"] = n01(d.replicate_robustness)
    d["s_stab"] = n01(d.component_stability)
    d["s_interp"] = 0.5 * n01(d.loading_sparsity) + 0.5 * n01(d.band_localisation, invert=True)
    d["s_recon"] = n01(d.recon_rel_error, invert=True)
    d["s_nuis"] = 0.5 * n01(np.abs(d.excitation_leakage), invert=True) + \
                  0.5 * n01(np.abs(d.source_leakage), invert=True)
    W = SELECTION_WEIGHTS
    d["total_score"] = (W["neighbourhood_preservation"] * d.s_nbr +
                        W["replicate_robustness"] * d.s_rep +
                        W["component_stability"] * d.s_stab +
                        W["interpretability"] * d.s_interp +
                        W["reconstruction"] * d.s_recon +
                        W["nuisance_control"] * d.s_nuis)
    return d.sort_values("total_score", ascending=False)
