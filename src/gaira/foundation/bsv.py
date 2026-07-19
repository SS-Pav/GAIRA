"""GAIRA V5 Foundation — Phase C4: the Raman-derived Biochemical State Vector.

Each analyte receives coordinates in the frozen manifold, with uncertainty from
its own replicates/excitations and a record of which latent components contribute.
BSV coordinates are non-negative proportions ("share of biochemical evidence"),
which is meaningful only because the selected decomposition is parts-based.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def build_bsv(manifold, X, meta, axes=None):
    """Per-analyte BSV over latent components, plus axis-level aggregation."""
    Z = manifold.coordinates(X, normalise=True)          # per-spectrum proportions
    df = pd.DataFrame(Z, columns=[f"c{j}" for j in range(Z.shape[1])])
    df["analyte"] = meta.analyte.values
    g = df.groupby("analyte")
    mean = g.mean()
    sd = g.std().fillna(0.0)
    n = g.size().rename("n_spectra")

    comp_cols = [c for c in mean.columns]
    rows = []
    for a in mean.index:
        m = mean.loc[a].values; s = sd.loc[a].values
        top = np.argsort(-m)[:5]
        rows.append({
            "analyte": a, "n_spectra": int(n.loc[a]),
            **{f"bsv_{c}": float(m[i]) for i, c in enumerate(comp_cols)},
            **{f"sd_{c}": float(s[i]) for i, c in enumerate(comp_cols)},
            "top_components": [int(t) for t in top],
            "top_component_weights": [float(m[t]) for t in top],
            "mean_uncertainty": float(np.mean(s)),
            "max_uncertainty": float(np.max(s)),
            "concentration": float(np.sum(np.sort(m)[::-1][:3])),   # top-3 mass
        })
    bsv = pd.DataFrame(rows)

    axis_df = None
    if axes:
        cols = {}
        for ax in axes:
            members = [f"c{j}" for j in ax["components"]]
            cols[f"axis{ax['axis']}_{ax['tentative_theme'][:18].replace(' ', '_')}"] = \
                mean[members].sum(axis=1)
        axis_df = pd.DataFrame(cols)
        axis_df.insert(0, "analyte", mean.index)
        sd_cols = {}
        for ax in axes:
            members = [f"c{j}" for j in ax["components"]]
            sd_cols[f"sd_axis{ax['axis']}"] = np.sqrt((sd[members] ** 2).sum(axis=1))
        for k, v in sd_cols.items():
            axis_df[k] = v.values
    return bsv, axis_df, Z


def bsv_uncertainty_summary(bsv):
    return {"n_analytes": int(len(bsv)),
            "median_mean_uncertainty": float(bsv.mean_uncertainty.median()),
            "median_top3_mass": float(bsv.concentration.median()),
            "analytes_with_replicates": int((bsv.n_spectra > 1).sum())}
