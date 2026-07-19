"""GAIRA V5 Foundation — Phase C7: biological projection into the FROZEN manifold.

Projection only. No retraining, no supervised learning, no disease classification.
Produces biochemical coordinates, axis-level BSV, evidence and uncertainty for each
biological spectrum, plus group-level summaries and out-of-distribution scores.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .benchmark import _unit


def project(manifold, X_ext, meta_ext, axes=None):
    """Coordinates + uncertainty proxies for external biological spectra."""
    Z = manifold.coordinates(X_ext, normalise=True)
    df = pd.DataFrame(Z, columns=[f"c{j}" for j in range(Z.shape[1])])
    for c in meta_ext.columns:
        df[c] = meta_ext[c].values
    if axes:
        for ax in axes:
            members = [f"c{j}" for j in ax["components"]]
            df[f"axis{ax['axis']}_{ax['tentative_theme'][:18].replace(' ', '_')}"] = \
                df[members].sum(axis=1)
    return df, Z


def ood_score(manifold, X_ext, corpus, k=5):
    """Distance to the reference support — how far outside the pure-analyte space
    a biological spectrum sits. Reported as a caveat, not a probability."""
    Zref = _unit(np.clip(manifold.coordinates(corpus.X), 0, None))
    Zx = _unit(np.clip(manifold.coordinates(X_ext), 0, None))
    S = Zx @ Zref.T
    topk = np.sort(S, axis=1)[:, -k:]
    return 1.0 - topk.mean(axis=1)


def group_summary(proj_df, axis_cols, group_col="group"):
    """Group-level biochemical coordinates with dispersion (no classification)."""
    g = proj_df.groupby(group_col)[axis_cols]
    out = g.mean().round(4)
    out.columns = [f"mean_{c}" for c in out.columns]
    sd = g.std().round(4); sd.columns = [f"sd_{c}" for c in sd.columns]
    out = out.join(sd)
    out["n"] = proj_df.groupby(group_col).size()
    return out.reset_index()


def nearest_references(manifold, X_ext, meta_ext, corpus, top=5):
    """Which pure reference analytes each biological spectrum most resembles in the
    biochemical manifold. EVIDENCE ONLY — never a molecular identification."""
    Zref = _unit(np.clip(manifold.coordinates(corpus.X), 0, None))
    names = corpus.meta.analyte.values
    Zx = _unit(np.clip(manifold.coordinates(X_ext), 0, None))
    S = Zx @ Zref.T
    rows = []
    for i in range(len(Zx)):
        order = np.argsort(-S[i])
        seen, picks = set(), []
        for j in order:
            a = names[j]
            if a in seen:
                continue
            seen.add(a); picks.append((a, float(S[i, j])))
            if len(picks) == top:
                break
        rows.append({"spectrum_id": meta_ext.spectrum_id.iloc[i],
                     "group": meta_ext.group.iloc[i] if "group" in meta_ext else None,
                     "nearest_references": [p[0] for p in picks],
                     "similarities": [round(p[1], 4) for p in picks]})
    return pd.DataFrame(rows)
