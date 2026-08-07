"""GAIRA V7 — Phase 06.5: Continuous Spectral Coordinates.

    u(x) ∈ ℝ^K,  Σ u = 1,  u_k = "how much x resembles emergent prototype k"

**These are not chemistry probabilities and must never be labelled as such.** A prototype is a
centroid of an unsupervised cluster; a coordinate is a similarity to it. The name is deliberate:
Phase 02.5 established that motif space is a continuum, and a continuum is better described by
coordinates than by a cluster id.

Every kernel is deterministic and every prototype is built inside training folds only.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12
KERNELS = ("softmax_cosine", "gaussian", "cosine_power", "inverse_distance", "wasserstein")
TEMPERATURES = (0.02, 0.05, 0.1, 0.2, 0.5, 1.0)


def unit(M):
    return np.asarray(M, float) / (np.linalg.norm(M, axis=1, keepdims=True) + EPS)


def prototypes(M: np.ndarray, lab: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Cluster centroids in CSM space. Unassigned members (-1) contribute to no prototype."""
    ids = sorted({int(v) for v in lab if v >= 0})
    P = np.vstack([np.asarray(M, float)[lab == c].mean(axis=0) for c in ids])
    return np.clip(P, 0.0, None), ids


def _wasserstein_sim(Q: np.ndarray, P: np.ndarray, grid_cost: np.ndarray) -> np.ndarray:
    """1-D Wasserstein between activation profiles treated as distributions over CSM index.

    The cost matrix is the frozen CSM geometry, so "moving mass" from one motif to another is
    cheap when the motifs are spectrally similar. Computed by the closed-form CDF difference on
    a 1-D ordering of the CSMs, which is exact for 1-D optimal transport and needs no solver.
    """
    Qn = np.clip(Q, 0, None) / (np.clip(Q, 0, None).sum(axis=1, keepdims=True) + EPS)
    Pn = np.clip(P, 0, None) / (np.clip(P, 0, None).sum(axis=1, keepdims=True) + EPS)
    order = grid_cost
    Qc = np.cumsum(Qn[:, order], axis=1)
    Pc = np.cumsum(Pn[:, order], axis=1)
    Dw = np.abs(Qc[:, None, :] - Pc[None, :, :]).sum(axis=2)
    return -Dw


def coordinates(Q: np.ndarray, P: np.ndarray, kernel: str = "softmax_cosine",
                temperature: float = 0.1, csm_order: np.ndarray | None = None) -> np.ndarray:
    """Query × prototype similarity, mapped to a simplex. Rows sum to 1."""
    Q = np.atleast_2d(np.asarray(Q, float))
    Nq, Np = unit(Q), unit(P)
    if kernel == "softmax_cosine":
        S = Nq @ Np.T / max(temperature, 1e-6)
    elif kernel == "gaussian":
        D = np.clip(1.0 - Nq @ Np.T, 0.0, 2.0)
        S = -(D ** 2) / (2 * max(temperature, 1e-6) ** 2)
    elif kernel == "cosine_power":
        C = np.clip(Nq @ Np.T, EPS, None)
        S = np.log(C) / max(temperature, 1e-6)
    elif kernel == "inverse_distance":
        D = np.clip(1.0 - Nq @ Np.T, EPS, 2.0)
        S = np.log(1.0 / D) / max(temperature, 1e-6)
    elif kernel == "wasserstein":
        if csm_order is None:
            csm_order = np.arange(Q.shape[1])
        S = _wasserstein_sim(Q, P, csm_order) / max(temperature, 1e-6)
    else:
        raise ValueError(f"unknown kernel {kernel}")
    S = S - S.max(axis=1, keepdims=True)
    U = np.exp(S)
    return U / (U.sum(axis=1, keepdims=True) + EPS)


# ── properties of a coordinate system ────────────────────────────────────────
def entropy(U: np.ndarray) -> np.ndarray:
    K = U.shape[1]
    return -(np.where(U > 0, U * np.log(U + EPS), 0.0)).sum(axis=1) / np.log(max(K, 2))


def reproducibility(U: np.ndarray, y: np.ndarray) -> float:
    """Mean within-molecule cosine of the coordinate vector across replicate spectra."""
    N = U / (np.linalg.norm(U, axis=1, keepdims=True) + EPS)
    vals = []
    for m in set(np.asarray(y).tolist()):
        idx = np.where(np.asarray(y) == m)[0]
        if len(idx) < 2:
            continue
        C = N[idx] @ N[idx].T
        iu = np.triu_indices(len(idx), 1)
        vals.append(float(C[iu].mean()))
    return float(np.mean(vals)) if vals else np.nan


def neighbour_preservation(A: np.ndarray, U: np.ndarray, k: int = 10) -> float:
    """Overlap of k-NN sets between the 49-d CSM space and the K-d coordinate space.

    The single most important property of a coordinate system that claims to summarise a
    manifold: if the neighbours change, the summary is a different manifold.
    """
    def nn(X):
        N = X / (np.linalg.norm(X, axis=1, keepdims=True) + EPS)
        D = 1.0 - N @ N.T
        np.fill_diagonal(D, np.inf)
        return np.argsort(D, axis=1)[:, :k]
    a, u = nn(np.asarray(A, float)), nn(np.asarray(U, float))
    return float(np.mean([len(set(a[i]) & set(u[i])) / k for i in range(len(a))]))


def bridge_score(U: np.ndarray) -> np.ndarray:
    """1 − (top1 − top2). High where a molecule sits between two prototypes."""
    S = np.sort(U, axis=1)
    return 1.0 - (S[:, -1] - S[:, -2])


def effective_rank(U: np.ndarray) -> float:
    X = np.asarray(U, float) - np.asarray(U, float).mean(axis=0)
    s = np.linalg.svd(X, compute_uv=False)
    p = s ** 2 / ((s ** 2).sum() + EPS)
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def sweep(A: np.ndarray, P: np.ndarray, y: np.ndarray, kernels=KERNELS,
          temperatures=TEMPERATURES, csm_order=None, log=None) -> "pd.DataFrame":
    """Kernel × temperature grid, scored on label-free properties only."""
    import pandas as pd
    rows = []
    for kern in kernels:
        for t in temperatures:
            try:
                U = coordinates(A, P, kern, t, csm_order)
            except Exception as exc:                              # pragma: no cover
                rows.append({"kernel": kern, "temperature": t, "usable": False,
                             "error": str(exc)[:60]})
                continue
            e = entropy(U)
            rows.append({"kernel": kern, "temperature": t, "usable": True,
                         "mean_entropy": float(e.mean()), "sd_entropy": float(e.std()),
                         "reproducibility": reproducibility(U, y),
                         "neighbour_preservation_k10": neighbour_preservation(A, U, 10),
                         "effective_rank": effective_rank(U),
                         "mean_bridge_score": float(bridge_score(U).mean()),
                         "fraction_degenerate": float(np.mean(e > 0.98)),
                         "fraction_one_hot": float(np.mean(e < 0.02))})
            if log:
                r = rows[-1]
                log(f"    {kern:18s} T={t:<5} entropy {r['mean_entropy']:.3f}  "
                    f"repro {r['reproducibility']:.3f}  nbr {r['neighbour_preservation_k10']:.3f} "
                    f" effrank {r['effective_rank']:.2f}")
    return pd.DataFrame(rows)


def select(tab: "pd.DataFrame", entropy_lo: float = 0.10, entropy_hi: float = 0.90) -> tuple:
    """Choose kernel and temperature on **label-free** criteria, declared before the sweep.

    Maximise neighbour preservation — the property that defines a faithful coordinate system —
    among settings that are neither degenerate (near-uniform) nor collapsed to a hard label.
    Chemistry is not consulted; it is the external validation target, not the objective.
    """
    ok = tab[(tab.usable) & (tab.mean_entropy > entropy_lo) & (tab.mean_entropy < entropy_hi)
             & (tab.fraction_degenerate < 0.05)]
    if not len(ok):
        return (str(tab.iloc[0]["kernel"]), float(tab.iloc[0]["temperature"]),
                "FALLBACK: no setting cleared the non-degeneracy window")
    best = ok.sort_values("neighbour_preservation_k10", ascending=False).iloc[0]
    return (str(best["kernel"]), float(best["temperature"]),
            "max neighbour preservation among non-degenerate settings")
