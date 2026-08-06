"""GAIRA V7 — Phase 04, Parts B–D: LSM → CSM → theme → BSV, with uncertainty carried upward.

Three abstraction steps, each with candidate mappings that are benchmarked rather than
hard-coded. The binding constraint through all of them is that **uncertainty must survive the
abstraction**: a confident theme activation built on an unreliable CSM activation is a lie the
engine must not be able to tell.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12
AGGREGATIONS = ("direct_csm_projection", "membership_weighted", "provenance_mean",
                "stability_weighted", "nnls_on_csm_basis")
THEME_MODES = ("soft_membership", "sparse_topk", "probabilistic", "confidence_weighted")
BSV_VARIANTS = ("theme_only", "theme_plus_residual", "theme_plus_uncertainty", "full_state")


# ── Part B: LSM → CSM ────────────────────────────────────────────────────────
def lsm_to_csm(a_lsm: np.ndarray, lsm_ids: list[str], csm_members: dict[str, list[str]],
               csm_ids: list[str], method: str = "membership_weighted",
               stability: dict[str, float] | None = None,
               x: np.ndarray | None = None, CSM: np.ndarray | None = None) -> np.ndarray:
    """Aggregate LSM activations into CSM activations.

    48 of the 49 CSMs are a single LSM, so most of these mappings coincide on this atlas; they
    are still benchmarked, because the one merged CSM is where they differ and because a later
    atlas with more merges would make the choice matter everywhere.
    """
    idx = {m: i for i, m in enumerate(lsm_ids)}
    A = np.atleast_2d(a_lsm)
    out = np.zeros((A.shape[0], len(csm_ids)))
    if method == "direct_csm_projection":
        if x is None or CSM is None:
            raise ValueError("direct projection needs the spectra and the CSM basis")
        from .projection import project
        return project(x, CSM, "nnls")
    if method == "nnls_on_csm_basis":
        if x is None or CSM is None:
            raise ValueError("nnls_on_csm_basis needs the spectra and the CSM basis")
        from .projection import project
        return project(x, CSM, "nnls")
    for k, cid in enumerate(csm_ids):
        members = [idx[m] for m in csm_members[cid] if m in idx]
        if not members:
            continue
        if method == "membership_weighted" or method == "provenance_mean":
            out[:, k] = A[:, members].mean(axis=1)
        elif method == "stability_weighted":
            w = np.array([(stability or {}).get(csm_members[cid][j], 1.0)
                          for j in range(len(members))], float)
            w = w / (w.sum() + EPS)
            out[:, k] = A[:, members] @ w
        else:
            raise ValueError(f"unknown aggregation {method}")
    return out


def csm_uncertainty(a_lsm: np.ndarray, lsm_ids: list[str],
                    csm_members: dict[str, list[str]], csm_ids: list[str]) -> np.ndarray:
    """Disagreement among the LSMs of a CSM, as a fraction of their mean activation.

    A CSM built from one LSM has no internal disagreement and gets zero here — which is
    honest, not reassuring: its uncertainty comes from the layers above and below instead.
    """
    idx = {m: i for i, m in enumerate(lsm_ids)}
    A = np.atleast_2d(a_lsm)
    U = np.zeros((A.shape[0], len(csm_ids)))
    for k, cid in enumerate(csm_ids):
        members = [idx[m] for m in csm_members[cid] if m in idx]
        if len(members) < 2:
            continue
        sub = A[:, members]
        U[:, k] = sub.std(axis=1) / (sub.mean(axis=1) + EPS)
    return U


# ── Part C: CSM → theme ──────────────────────────────────────────────────────
def theme_activation(a_csm: np.ndarray, S: np.ndarray, mode: str = "soft_membership",
                     theme_confidence: np.ndarray | None = None,
                     accepted: np.ndarray | None = None, topk: int = 2,
                     temperature: float = 1.0) -> np.ndarray:
    """Theme activations from CSM activations through the frozen membership matrix.

    `t = Sᵀ c` is the architecture's definition (LEARNING_MODE Stage 5). The variants differ in
    what happens afterwards — whether the result is sparsified, turned into a distribution, or
    scaled by how much each theme is trusted.

    **No spectrum is forced into every theme.** Every mode here can return a theme activation
    of exactly zero, and `sparse_topk` will do so for most themes on most spectra.
    """
    A = np.atleast_2d(a_csm)
    T = A @ S                                        # (n, K)
    if mode == "soft_membership":
        pass
    elif mode == "sparse_topk":
        cut = np.sort(T, axis=1)[:, -topk][:, None]
        T = np.where(T >= cut, T, 0.0)
    elif mode == "probabilistic":
        Z = T / (temperature * (T.max(axis=1, keepdims=True) + EPS))
        E = np.exp(Z - Z.max(axis=1, keepdims=True))
        T = E / (E.sum(axis=1, keepdims=True) + EPS)
    elif mode == "confidence_weighted":
        if theme_confidence is None:
            raise ValueError("confidence_weighted needs per-theme confidence")
        T = T * theme_confidence[None, :]
    else:
        raise ValueError(f"unknown theme mode {mode}")
    return np.clip(T, 0.0, None)


def zero_evidence_leakage(a_csm: np.ndarray, S: np.ndarray, T: np.ndarray) -> float:
    """Activation a mode assigns to themes for which the CSM evidence is exactly zero.

    The test that catches theme collapse. A softmax over `Sᵀc` gives every theme a non-zero
    activation even when `Σ_m c_m S_mk = 0` — the spectrum activates no CSM belonging to that
    theme, so there is no evidence for it at all. That mode scored *best* on replicate
    consistency, because assigning every spectrum the same flat vector is perfectly
    reproducible and completely uninformative. Leakage must be zero: "do not force every
    spectrum into every theme" is a constraint on the mathematics, not a preference.
    """
    A = np.atleast_2d(a_csm)
    Tm = np.atleast_2d(T)
    evidence = A @ S                       # (n, K) raw, pre-mode
    zero = evidence <= EPS
    if not zero.any():
        return 0.0
    tot = Tm.sum(axis=1, keepdims=True) + EPS
    return float((Tm / tot)[zero].sum() / Tm.shape[0])


def rejected_theme_to_uncertainty(T_all: np.ndarray, accepted: np.ndarray) -> np.ndarray:
    """Mass a spectrum places on rejected themes, expressed as an uncertainty channel.

    A rejected theme is not a coordinate — Phase 03 rejected Theme-03 on stability, so an
    activation on it is not a biochemical statement. But it is not nothing either: a spectrum
    whose mass lands on a rejected theme is a spectrum the accepted themes do not describe, and
    that is exactly what an uncertainty channel is for.
    """
    T = np.atleast_2d(T_all)
    tot = T.sum(axis=1) + EPS
    return (T[:, ~accepted].sum(axis=1) / tot)


# ── Part D: the Biochemical State Vector ─────────────────────────────────────
def build_bsv(T_accepted: np.ndarray, residual: np.ndarray, rejected_mass: np.ndarray,
              bridge_mass: np.ndarray, variant: str = "theme_only") -> tuple[np.ndarray, list[str]]:
    """Candidate BSV definitions, all non-negative and absolute (contract C-09).

    The architecture defines the BSV as `t = Sᵀc` — theme coordinates and nothing else. The
    brief asks whether residual, uncertainty and bridge contribution belong *in* the vector or
    *beside* it. That is a real question with a real answer, and it is settled by benchmark
    rather than by taste, because the answer changes what a distance in BSV space means.

    A coordinate that is in the vector participates in every downstream distance, trajectory
    and comparison. Uncertainty in the vector means two spectra can be "far apart" because one
    was noisier — which is not a biochemical difference.
    """
    T = np.atleast_2d(T_accepted)
    r = np.atleast_1d(residual)[:, None]
    j = np.atleast_1d(rejected_mass)[:, None]
    b = np.atleast_1d(bridge_mass)[:, None]
    K = T.shape[1]
    names = [f"theme_{k}" for k in range(K)]
    if variant == "theme_only":
        V = T
    elif variant == "theme_plus_residual":
        V, names = np.hstack([T, r]), names + ["residual_unexplained"]
    elif variant == "theme_plus_uncertainty":
        V, names = np.hstack([T, j]), names + ["rejected_theme_mass"]
    elif variant == "full_state":
        V = np.hstack([T, r, j, b])
        names = names + ["residual_unexplained", "rejected_theme_mass", "bridge_mass"]
    else:
        raise ValueError(f"unknown BSV variant {variant}")
    return np.clip(V, 0.0, None), names


def bsv_reference_frame(BSV: np.ndarray) -> dict:
    """The frozen reference frame: per-axis location, spread, and the effective rank.

    Effective rank is reported alongside `K` because they are not the same number (risk R-12).
    At V5 the 24-component space had a participation ratio of 15.2 — a 38% gap that was only
    visible because someone measured it.
    """
    mu = BSV.mean(axis=0)
    sd = BSV.std(axis=0)
    C = np.cov(BSV.T) if BSV.shape[1] > 1 else np.array([[BSV.var()]])
    ev = np.clip(np.linalg.eigvalsh(np.atleast_2d(C)), 0, None)[::-1]
    p = ev / (ev.sum() + EPS)
    pr = float((ev.sum() ** 2) / ((ev ** 2).sum() + EPS))
    ent = float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum()))
    cum = np.cumsum(p)
    return {"reference_mean": mu.tolist(), "reference_spread": sd.tolist(),
            "effective_rank": {"participation_ratio": pr, "entropy_rank": ent,
                               "n_axes_90pct": int(np.searchsorted(cum, 0.90) + 1),
                               "nominal_K": int(BSV.shape[1])},
            "explained_variance_ratio": p.tolist()}


def bsv_elevation(BSV: np.ndarray, frame: dict) -> np.ndarray:
    """Signed z-score against the frozen frame. **Derived, never named `bsv`** (contract C-10)."""
    mu = np.asarray(frame["reference_mean"], float)
    sd = np.asarray(frame["reference_spread"], float)
    return (np.atleast_2d(BSV) - mu) / (sd + EPS)
