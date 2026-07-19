"""Stage B0 — scientific controls (Controls 4-8) and stratified diagnostics."""
from __future__ import annotations
import numpy as np

from . import pipeline as PL
from . import objectives as OB

# Control 6/7 — exemplar sets declared in advance (from the spectral audit)
CLEAN_SERS_EXEMPLARS = ["ergothioneine", "hypoxanthine", "guanine", "albumin", "cholesterol"]


def background_variance_and_retention(cand, cache, meta, train_analytes, eval_analytes):
    """Control 4 (variance removed) + Control 5 (analyte residual retention)."""
    X1 = cache.build(cand)
    train_mask = meta.analyte.isin(train_analytes).values | (~meta.matched.values)
    state = PL.fit_stage2(cand, X1, meta, train_mask)
    is_sers = meta.modality.values == "sers"
    ve = state["background"].variance_explained(X1[is_sers])
    F, fmeta = PL.apply_stage2(cand, X1, meta, state, aggregate=True)
    chem = OB.within_modality_chemistry(F, fmeta, eval_analytes)
    return {"bg_variance_explained": ve,
            "sers_1nn_after": chem.get("sers_1nn"),
            "raman_1nn_after": chem.get("raman_1nn")}


def stratified(df_per_analyte, meta, clean=None):
    """Controls 6-8: clean vs noisy Ag-SERS exemplars, and Raman-source sensitivity."""
    clean = clean or CLEAN_SERS_EXEMPLARS
    d = df_per_analyte
    out = {}
    if "analyte" in d:
        cm = d[d.analyte.isin(clean)]
        nz = d[~d.analyte.isin(clean)]
        for nm, sub in (("clean_exemplars", cm), ("other", nz)):
            if len(sub):
                out[nm] = {"n": int(len(sub)),
                           "mrr": float(np.nanmean(sub.get("mrr", np.nan))),
                           "peak_effect": float(np.nanmean(sub.get("peak_effect", np.nan)))}
        # Raman source sensitivity
        if "raman_multi_source" in d:
            for nm, sub in (("raman_multi_source", d[d.raman_multi_source]),
                            ("raman_single_source", d[~d.raman_multi_source])):
                if len(sub):
                    out[nm] = {"n": int(len(sub)),
                               "mrr": float(np.nanmean(sub.get("mrr", np.nan)))}
    return out
