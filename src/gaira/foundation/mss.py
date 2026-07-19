"""GAIRA V5 Foundation — Phase C5: Molecular Spectral Signatures.

For every reference analyte, a sparse interpretable signature: the few latent
components it actually uses, the spectral bands those components carry, and the
biochemical axis each contributes to. These are reusable reference objects.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.signal import find_peaks


def observed_bands(x, grid, n=8, prom_frac=0.12):
    x = np.nan_to_num(x)
    rng = x.max() - x.min()
    if rng <= 0:
        return []
    idx, props = find_peaks(x, prominence=prom_frac * rng, distance=4)
    if len(idx) == 0:
        return []
    order = np.argsort(-props["prominences"])[:n]
    return [float(grid[i]) for i in np.sort(idx[order])]


def build_mss(manifold, X, meta, axes, comp_df, energy=0.90, max_comp=6):
    """Sparse signature per analyte: minimal component set reaching `energy`."""
    Z = manifold.coordinates(X, normalise=True)
    grid = manifold.grid
    comp_bands = dict(zip(comp_df.component, comp_df.dominant_bands_cm))
    axis_of = dict(zip(comp_df.component, comp_df.axis))
    theme_of = {a["axis"]: a["tentative_theme"] for a in axes}

    df = pd.DataFrame(Z); df["analyte"] = meta.analyte.values
    meanZ = df.groupby("analyte").mean()
    meanX = pd.DataFrame(np.nan_to_num(X)); meanX["analyte"] = meta.analyte.values
    meanX = meanX.groupby("analyte").mean()

    rows = []
    for a in meanZ.index:
        w = meanZ.loc[a].values
        order = np.argsort(-w)
        cum = np.cumsum(w[order]) / (w.sum() + 1e-12)
        keep = order[: max(1, min(max_comp, int(np.searchsorted(cum, energy) + 1)))]
        bands = sorted({b for j in keep for b in comp_bands.get(int(j), [])})
        axis_contrib = {}
        for j in keep:
            ax = int(axis_of.get(int(j), -1))
            axis_contrib[f"axis{ax}:{theme_of.get(ax,'?')}"] = \
                axis_contrib.get(f"axis{ax}:{theme_of.get(ax,'?')}", 0.0) + float(w[j])
        rows.append({
            "analyte": a,
            "n_components_used": int(len(keep)),
            "components": [int(j) for j in keep],
            "component_weights": [round(float(w[j]), 4) for j in keep],
            "latent_energy_captured": float(cum[len(keep) - 1]),
            "signature_bands_cm": bands[:14],
            "observed_bands_cm": observed_bands(meanX.loc[a].values, grid),
            "axis_contributions": {k: round(v, 4) for k, v in
                                   sorted(axis_contrib.items(), key=lambda t: -t[1])},
            "sparsity": float(1.0 - len(keep) / Z.shape[1]),
        })
    return pd.DataFrame(rows)
