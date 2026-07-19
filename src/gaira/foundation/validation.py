"""GAIRA V5 Foundation — Phase C6: external validation WITHOUT retraining.

The datasets named in the original plan (adenine concentration series, uricase,
metabolite-63) are Ag-SERS / Au-SERS in this repository and are therefore OUT OF
DOMAIN for a Raman-only foundation model; they are explicitly excluded and
recorded as such. The Raman-domain validation actually available is:

  V1 held-out analyte projection  — fit on training analytes only, project unseen
                                    analytes, test neighbourhood + BSV recovery;
  V2 excitation transfer          — the same analyte measured at DIFFERENT laser
                                    excitations must land in the same place;
  V3 source transfer              — the same analyte from different reference
                                    sources must agree;
  V4 tube-blank control           — non-biochemical blanks must not imitate analytes.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from . import representation as RP
from .benchmark import _unit, neighbourhood_preservation

EXCLUDED_OUT_OF_DOMAIN = {
    "adenine_sers_control": "Ag-SERS (bAgNPs) concentration series — not Raman",
    "european_multi_instrument_adenine": "cAg/sAg/cAu substrates — SERS/Au-SERS",
    "metabolite_sers63": "633 nm Ag-SERS",
    "serum_ag_colloids (uricase, spike-ins)": "Ag-SERS serum colloid",
}


def heldout_analyte_projection(corpus, name, k, n_splits=4, seed=0):
    """V1: fit on training analytes only; project unseen analytes."""
    X, meta = corpus.X, corpus.meta
    groups = meta.analyte.values
    gkf = GroupKFold(n_splits=n_splits)
    rows = []
    for f, (tr, te) in enumerate(gkf.split(X, groups=groups)):
        rep = RP.FITTERS[name](X[tr], k, seed)
        Zte = rep.transform(np.nan_to_num(X[te]))
        mte = meta.iloc[te].reset_index(drop=True)
        # BSV recovery: do replicates of an unseen analyte still co-locate?
        U = _unit(np.clip(Zte, 0, None)); lab = mte.analyte.values
        C = U @ U.T; np.fill_diagonal(C, np.nan)
        same = lab[:, None] == lab[None, :]
        within = float(np.nanmean(np.where(same, C, np.nan))) if same.any() else np.nan
        between = float(np.nanmean(np.where(~same, C, np.nan)))
        rows.append({"fold": f, "n_test_spectra": int(len(te)),
                     "n_test_analytes": int(mte.analyte.nunique()),
                     "neighbourhood_preservation": neighbourhood_preservation(
                         X[te], Zte, k=min(10, len(te) - 2)),
                     "within_analyte_cos": within, "between_analyte_cos": between,
                     "bsv_margin": (within - between) if np.isfinite(within) else np.nan,
                     "recon_rel_error": float(
                         np.linalg.norm(np.nan_to_num(X[te]) - rep.reconstruct(X[te])) /
                         (np.linalg.norm(np.nan_to_num(X[te])) + 1e-12))})
    return pd.DataFrame(rows)


def _agreement(manifold, X, meta, col):
    """Do coordinates of the same analyte agree across levels of `col`?"""
    Z = manifold.coordinates(X, normalise=True)
    rows = []
    for a, idx in meta.groupby("analyte").groups.items():
        sub = meta.loc[idx]
        if sub[col].nunique() < 2:
            continue
        U = _unit(Z[[meta.index.get_loc(i) for i in idx]])
        lv = sub[col].astype(str).values
        C = U @ U.T; np.fill_diagonal(C, np.nan)
        cross = lv[:, None] != lv[None, :]
        rows.append({"analyte": a, "n_levels": int(sub[col].nunique()),
                     "cross_level_cos": float(np.nanmean(np.where(cross, C, np.nan)))})
    if not rows:
        return pd.DataFrame(), np.nan
    df = pd.DataFrame(rows)
    # null: coordinates of DIFFERENT analytes
    Zu = _unit(Z); lab = meta.analyte.values
    C = Zu @ Zu.T; np.fill_diagonal(C, np.nan)
    diff = lab[:, None] != lab[None, :]
    null = float(np.nanmean(np.where(diff, C, np.nan)))
    return df, null


def excitation_transfer(manifold, corpus):
    """V2: same analyte, different laser excitation."""
    return _agreement(manifold, corpus.X, corpus.meta, "excitation_nm")


def source_transfer(manifold, corpus):
    """V3: same analyte, different reference source."""
    return _agreement(manifold, corpus.X, corpus.meta, "source")


def blank_control(manifold, X_ext, meta_ext, corpus, blank_group="Tube"):
    """V4: blanks must not look like reference analytes."""
    if X_ext is None or "group" not in meta_ext:
        return {}
    Zref = _unit(np.clip(manifold.coordinates(corpus.X), 0, None))
    out = {}
    for g in meta_ext.group.unique():
        m = (meta_ext.group == g).values
        Zg = _unit(np.clip(manifold.coordinates(X_ext[m]), 0, None))
        S = Zg @ Zref.T
        out[g] = {"n": int(m.sum()),
                  "max_similarity_to_any_reference": float(np.mean(S.max(axis=1))),
                  "mean_similarity_to_references": float(np.mean(S))}
    return out
