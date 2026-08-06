"""GAIRA V7 Phase 01, Stage 1 — balanced reference construction.

The change V7 exists to make, in one line:

    one spectrum = one vote   →   one canonical molecule = one scientific reference unit

Global NMF minimises `‖X − WH‖²_F`, a sum over ROWS of X. A row is one spectrum, so an
analyte with three replicates exerts three times the pull of one with a single spectrum, and
a class with 30 molecules exerts thirty times the pull of one with two. That is limitation
L-01, and every arm below is a candidate answer to it.

ARMS (all eight are run; the selection rule is pre-registered in
`plan/VALIDATION_AND_DECISION_RULES.md` §1 and the comparison is published either way)

    A            all spectra, equal row weight            THE CONTROL — this is V5 behaviour
    B            analyte-balanced, quality-weighted        w_ai = q_ai / Σ_j q_aj
    B-uniform    analyte-balanced, uniform weight          isolates balancing from weighting
    C-mean       mean prototype per molecule
    C-median     per-bin median prototype
    C-trimmed    trimmed-mean prototype
    C-medoid     medoid prototype                          always a real measured spectrum
    C-quality    quality-weighted prototype

Replicates are used for quality, uncertainty and stability — never to increase a molecule's
representational influence (P-11). Rare classes are never bootstrapped by duplicating
spectra: a duplicate adds no information, it only moves the loss surface.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ARMS = ("A_all_spectra", "B_analyte_weighted", "B_uniform",
        "C_mean", "C_median", "C_trimmed", "C_medoid", "C_quality")
CONTROL_ARM = "A_all_spectra"
TRIM_FRACTION = 0.2


def _trimmed_mean(M: np.ndarray, frac: float = TRIM_FRACTION) -> np.ndarray:
    """Per-bin trimmed mean. With <3 rows the trim is a no-op, by construction."""
    n = M.shape[0]
    k = int(np.floor(n * frac))
    if n < 3 or k == 0:
        return M.mean(axis=0)
    S = np.sort(M, axis=0)
    return S[k:n - k].mean(axis=0)


def _medoid(M: np.ndarray) -> np.ndarray:
    """The member spectrum closest to all others — always a REAL measurement.

    Preferred when a molecule spans excitations: peak positions are excitation-invariant but
    relative intensities are not, so a per-bin mean or median can synthesise a band shape no
    instrument ever produced. A medoid cannot.
    """
    if M.shape[0] == 1:
        return M[0]
    N = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    d = 1.0 - N @ N.T
    return M[int(np.argmin(d.sum(axis=1)))]


def build_arm(arm: str, X: np.ndarray, meta: pd.DataFrame, quality: pd.DataFrame,
              ) -> tuple[np.ndarray, pd.DataFrame]:
    """Build one reference-construction arm.

    Returns (rows, row_meta). `row_meta` always carries `canonical_id`, `weight`,
    `n_source_spectra` and `provenance`, so every arm is a drop-in for the next stage.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")

    q = dict(zip(quality.spectrum_id, quality.quality_score))
    meta = meta.reset_index(drop=True)
    rows, recs = [], []

    if arm == CONTROL_ARM:
        for i in range(len(meta)):
            rows.append(X[i])
            recs.append({"canonical_id": meta.canonical_id.iat[i], "weight": 1.0,
                         "n_source_spectra": 1, "excitation_nm": meta.excitation_nm.iat[i],
                         "provenance": meta.spectrum_id.iat[i]})
        return np.vstack(rows), pd.DataFrame(recs)

    if arm in ("B_analyte_weighted", "B_uniform"):
        for cid, g in meta.groupby("canonical_id", sort=True):
            idx = g.index.to_numpy()
            if arm == "B_uniform":
                w = np.full(len(idx), 1.0 / len(idx))
            else:
                qq = np.array([q.get(s, 0.0) for s in g.spectrum_id], float)
                w = qq / qq.sum() if qq.sum() > 0 else np.full(len(idx), 1.0 / len(idx))
            for j, i in enumerate(idx):
                rows.append(X[i])
                recs.append({"canonical_id": cid, "weight": float(w[j]),
                             "n_source_spectra": 1,
                             "excitation_nm": meta.excitation_nm.iat[i],
                             "provenance": meta.spectrum_id.iat[i]})
        return np.vstack(rows), pd.DataFrame(recs)

    # C-family: exactly one prototype row per canonical molecule
    for cid, g in meta.groupby("canonical_id", sort=True):
        idx = g.index.to_numpy()
        M = X[idx]
        if arm == "C_mean":
            proto = M.mean(axis=0)
        elif arm == "C_median":
            proto = np.median(M, axis=0)
        elif arm == "C_trimmed":
            proto = _trimmed_mean(M)
        elif arm == "C_medoid":
            proto = _medoid(M)
        else:                                                   # C_quality
            qq = np.array([q.get(s, 0.0) for s in g.spectrum_id], float)
            proto = (qq @ M) / qq.sum() if qq.sum() > 0 else M.mean(axis=0)
        rows.append(proto)
        exc = g.excitation_nm.unique()
        recs.append({"canonical_id": cid, "weight": 1.0, "n_source_spectra": len(idx),
                     "excitation_nm": float(exc[0]) if len(exc) == 1 else np.nan,
                     "provenance": ";".join(sorted(g.spectrum_id))})
    return np.vstack(rows), pd.DataFrame(recs)


# ── evaluation ────────────────────────────────────────────────────────────────
def class_balance(row_meta: pd.DataFrame, class_of: dict) -> dict:
    """How evenly effective weight is spread across chemistry classes.

    `effective_class_gini` is the number that matters: 0 = every class carries equal weight
    in the objective, → 1 = one class dominates it.
    """
    df = row_meta.copy()
    df["fine_class"] = df.canonical_id.map(class_of)
    w = df.groupby("fine_class").weight.sum()
    w = w / w.sum()
    x = np.sort(w.values)
    n = len(x)
    gini = float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum())) if n and x.sum() else 0.0
    mw = df.groupby("canonical_id").weight.sum()
    return {
        "n_rows": int(len(df)),
        "n_molecules": int(df.canonical_id.nunique()),
        "effective_class_gini": round(gini, 4),
        "max_class_weight_share": round(float(w.max()), 4),
        "min_class_weight_share": round(float(w.min()), 4),
        "class_weight_ratio": round(float(w.max() / max(w.min(), 1e-12)), 2),
        "molecule_weight_ratio": round(float(mw.max() / max(mw.min(), 1e-12)), 3),
        "molecule_weight_equal": bool(np.allclose(mw.values, mw.values[0], atol=1e-9)),
    }


def band_fidelity(rows: np.ndarray, row_meta: pd.DataFrame, X: np.ndarray,
                  meta: pd.DataFrame, prominence: float = 0.05) -> float:
    """Do the constructed references preserve each molecule's diagnostic band pattern?

    Mean cosine between a molecule's reference row and its own measured spectra, restricted
    to the peaks of those measured spectra. A construction can look excellent on whole-vector
    correlation while flattening exactly the bands that carry the chemistry; this measures
    the bands.
    """
    from scipy.signal import find_peaks
    by_cid: dict[str, list[int]] = {}
    for i, c in enumerate(row_meta.canonical_id):
        by_cid.setdefault(c, []).append(i)
    sims = []
    for cid, g in meta.groupby("canonical_id", sort=True):
        if cid not in by_cid:
            continue
        ref = rows[by_cid[cid]].mean(axis=0)
        for i in g.index:
            x = np.nan_to_num(X[i])
            pk, _ = find_peaks(x, prominence=prominence * float(x.max()))
            if len(pk) < 3:
                continue
            a, b = x[pk], ref[pk]
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na > 0 and nb > 0:
                sims.append(float(a @ b / (na * nb)))
    return float(np.mean(sims)) if sims else float("nan")


def replicate_stability(rows: np.ndarray, row_meta: pd.DataFrame) -> float:
    """Mean within-molecule cosine among the rows a molecule contributes.

    1.0 for the C-family by construction (one row per molecule) — reported, not hidden, since
    a prototype arm cannot be credited for stability it obtained by discarding the variance.
    """
    sims = []
    for cid, g in row_meta.groupby("canonical_id", sort=True):
        idx = g.index.to_numpy()
        if len(idx) < 2:
            sims.append(1.0)
            continue
        M = rows[idx]
        N = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
        C = N @ N.T
        iu = np.triu_indices(len(idx), 1)
        sims.append(float(C[iu].mean()))
    return float(np.mean(sims)) if sims else float("nan")


def discarded_variance(X: np.ndarray, meta: pd.DataFrame) -> pd.DataFrame:
    """Within-molecule spread, retained separately when a prototype arm is selected.

    Collapsing replicates to a prototype throws away the only direct measurement-uncertainty
    estimate this corpus offers. The specification requires it be kept, not lost.
    """
    rows = []
    for cid, g in meta.groupby("canonical_id", sort=True):
        M = np.nan_to_num(X[g.index.to_numpy()])
        if M.shape[0] < 2:
            rows.append({"canonical_id": cid, "n_spectra": int(M.shape[0]),
                         "mean_pairwise_cosine": 1.0, "mean_bin_std": 0.0,
                         "max_bin_std": 0.0})
            continue
        N = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
        C = N @ N.T
        iu = np.triu_indices(M.shape[0], 1)
        sd = M.std(axis=0)
        rows.append({"canonical_id": cid, "n_spectra": int(M.shape[0]),
                     "mean_pairwise_cosine": round(float(C[iu].mean()), 5),
                     "mean_bin_std": round(float(sd.mean()), 6),
                     "max_bin_std": round(float(sd.max()), 6)})
    return pd.DataFrame(rows)
