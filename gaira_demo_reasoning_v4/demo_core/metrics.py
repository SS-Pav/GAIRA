"""Separated interpretation metrics (Part 5).

The frozen engine's per-theme `confidence` bundles several distinct notions. This
module does NOT redefine or modify that frozen metric — it exposes the underlying,
separately-named quantities for DISPLAY so the demo never shows one generic
"confidence" as though it covered every kind of uncertainty:

  - Atlas / spectral support   = how well the spectrum fits the frozen atlas (1 − OOD)
  - Theme specificity          = how concentrated a theme's evidence is (low leakage)
  - Matrix recoverability      = empirical serum-spike recoverability, ANALYTE-specific,
                                 or None where no matrix-spike evidence exists
  - Replicate reliability      = cross-replicate direction consistency, where available

For unknown biological spectra, matrix recoverability is unavailable BY DESIGN — a
universal value is never computed, and missing evidence is never scored as positive.
"""
from __future__ import annotations
import numpy as np


def atlas_support(bsv):
    """1 − OOD: fit to the frozen atlas / in-distribution status. [0,1], higher=better."""
    return float(np.clip(1.0 - bsv.ood_score, 0, 1))


def theme_specificity(builder, bsv, theme_id):
    """How specifically the component evidence supports ONE theme: 1 − normalized
    entropy of the theme's evidence mass W[:,t]·coord. High ⇒ concentrated (specific);
    low ⇒ spread across many components (collision-prone / leaky)."""
    ti = builder.onto.theme_index(theme_id)
    w = builder.W[:, ti]
    coord = np.asarray(bsv.component_coord, float)
    m = w * coord
    s = m.sum()
    if s < 1e-12:
        return 0.0
    p = m / s
    p = p[p > 0]
    ent = -np.sum(p * np.log(p)) / np.log(len(coord))
    return float(np.clip(1.0 - ent, 0, 1))


def overall_theme_specificity(builder, bsv):
    bio = builder.onto.biochemical_theme_ids
    # weight by each theme's composition so dominant themes matter more
    comp = np.array([bsv.composition[t] for t in bio])
    spec = np.array([theme_specificity(builder, bsv, t) for t in bio])
    w = comp / (comp.sum() + 1e-12)
    return float(np.sum(w * spec))


def replicate_reliability(replicate_direction_cos):
    """Cross-replicate direction consistency (from the serum stress-test), where
    available; None otherwise."""
    if replicate_direction_cos is None:
        return None
    return float(np.clip(replicate_direction_cos, 0, 1))


def composite_interpretation(atlas, specificity, recoverability):
    """Optional composite. Missing recoverability is NOT treated as positive — it is
    excluded from the average and flagged, never silently filled with a good score."""
    terms = {"atlas_support": atlas, "theme_specificity": specificity}
    if recoverability is not None:
        terms["matrix_recoverability"] = recoverability
    value = float(np.mean(list(terms.values())))
    return {"value": value, "terms": terms,
            "recoverability_known": recoverability is not None}
