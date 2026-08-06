"""GAIRA V7 — Phase 04, Part E: projecting a new spectrum into the frozen latent manifold.

The Phase 02.5 geometry was built on 50 motifs, not on spectra, and it is frozen. A new
spectrum has to be placed *into* it without recomputing it — which is an out-of-sample
extension problem, and the method matters: an extension that re-derives the embedding would
make a spectrum's coordinates depend on which other spectra were in the batch, and the whole
point of a frozen atlas is that they do not.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12
EXTENSIONS = ("nystrom", "landmark_barycentric", "graph_interpolation", "knn_weighted")


def _sim(d: np.ndarray, sigma: float) -> np.ndarray:
    return np.exp(-(d ** 2) / (sigma ** 2 + EPS))


def nystrom(d_new: np.ndarray, D_ref: np.ndarray, coords_ref: np.ndarray,
            sigma: float | None = None) -> np.ndarray:
    """Nyström extension of the diffusion embedding.

    The principled choice for a diffusion map: the new point's coordinates are the kernel-
    weighted average of the reference coordinates, which is exactly the eigenfunction
    evaluated off-sample. It degrades gracefully — a point far from every reference lands near
    the weighted centre rather than somewhere arbitrary — and it is linear in the reference
    set, so a spectrum's coordinates never depend on its batch-mates.
    """
    off = D_ref[~np.eye(D_ref.shape[0], dtype=bool)]
    s = sigma if sigma is not None else float(np.median(off))
    W = _sim(np.atleast_2d(d_new), s)
    W = W / (W.sum(axis=1, keepdims=True) + EPS)
    return W @ coords_ref


def landmark_barycentric(d_new: np.ndarray, coords_ref: np.ndarray, k: int = 5) -> np.ndarray:
    """Barycentric coordinates over the k nearest landmarks, weights ∝ 1/distance."""
    Dn = np.atleast_2d(d_new)
    out = np.zeros((Dn.shape[0], coords_ref.shape[1]))
    for i, row in enumerate(Dn):
        nb = np.argsort(row)[:k]
        w = 1.0 / (row[nb] + EPS)
        out[i] = (w[:, None] * coords_ref[nb]).sum(0) / (w.sum() + EPS)
    return out


def graph_interpolation(d_new: np.ndarray, D_ref: np.ndarray, coords_ref: np.ndarray,
                        k: int = 5, n_iter: int = 20) -> np.ndarray:
    """Harmonic extension: attach the new point to its k nearest references and let its
    coordinate relax to the weighted average of its neighbours' on the frozen graph."""
    Dn = np.atleast_2d(d_new)
    out = landmark_barycentric(Dn, coords_ref, k)
    for i, row in enumerate(Dn):
        nb = np.argsort(row)[:k]
        w = 1.0 / (row[nb] + EPS)
        w /= w.sum() + EPS
        y = out[i]
        for _ in range(n_iter):
            y = 0.5 * y + 0.5 * (w[:, None] * coords_ref[nb]).sum(0)
        out[i] = y
    return out


def knn_weighted(d_new: np.ndarray, coords_ref: np.ndarray, k: int = 5) -> np.ndarray:
    Dn = np.atleast_2d(d_new)
    out = np.zeros((Dn.shape[0], coords_ref.shape[1]))
    for i, row in enumerate(Dn):
        nb = np.argsort(row)[:k]
        w = _sim(row[nb], float(np.median(row) + EPS))
        out[i] = (w[:, None] * coords_ref[nb]).sum(0) / (w.sum() + EPS)
    return out


def extend(method: str, d_new: np.ndarray, D_ref: np.ndarray, coords_ref: np.ndarray,
           k: int = 5) -> np.ndarray:
    if method == "nystrom":
        return nystrom(d_new, D_ref, coords_ref)
    if method == "landmark_barycentric":
        return landmark_barycentric(d_new, coords_ref, k)
    if method == "graph_interpolation":
        return graph_interpolation(d_new, D_ref, coords_ref, k)
    if method == "knn_weighted":
        return knn_weighted(d_new, coords_ref, k)
    raise ValueError(f"unknown extension {method}")


# ── geometry-derived diagnostics ─────────────────────────────────────────────
def local_density(d_new: np.ndarray, k: int = 5) -> np.ndarray:
    """Inverse mean distance to the k nearest references — how populated the neighbourhood is."""
    Dn = np.atleast_2d(d_new)
    return np.array([1.0 / (np.sort(r)[:k].mean() + EPS) for r in Dn])


def distance_to_support(d_new: np.ndarray) -> np.ndarray:
    return np.atleast_2d(d_new).min(axis=1)


def residual_ood(x: np.ndarray, D: np.ndarray, a: np.ndarray,
                 ref_residuals: np.ndarray | None = None) -> float:
    """Out-of-domain score from what the dictionary CANNOT explain.

    The geometric score below measures the distance from a spectrum's *reconstruction* to the
    dictionary — but a reconstruction is a non-negative combination of dictionary elements and
    therefore always lies inside their cone. It cannot see anything outside the dictionary's
    span, and on real Ag-SERS spectra it scored 0.409 AUROC: **below chance**, because the
    reconstruction of an out-of-domain spectrum lands as comfortably inside the atlas as an
    in-domain one does.

    The residual is what the reconstruction discards, and it is exactly where out-of-domain
    chemistry goes. Standardised against the reference set's own residuals, so 1.0 means "as
    unexplained as a typical reference spectrum".
    """
    rec = a @ D
    res = float(((x - rec) ** 2).sum() / ((x ** 2).sum() + EPS))
    if ref_residuals is None or len(ref_residuals) == 0:
        return res
    med = float(np.median(ref_residuals)) + EPS
    return res / med


def ood_score(d_new: np.ndarray, D_ref: np.ndarray, k: int = 5) -> np.ndarray:
    """Distance to the reference support, standardised by the reference set's own k-NN scale.

    An OOD score of 1.0 means the new spectrum sits as far from the atlas as a typical
    reference sits from its own neighbours; above about 2 it is outside the chemistry the atlas
    was built from, and downstream interpretation should not be trusted.
    """
    ref_knn = np.array([np.sort(r[r > 0])[:k].mean() for r in D_ref])
    scale = float(np.median(ref_knn)) + EPS
    return np.array([np.sort(r)[:k].mean() / scale for r in np.atleast_2d(d_new)])


def nearest_references(d_new: np.ndarray, ids: list[str], k: int = 5) -> list[list[dict]]:
    out = []
    for row in np.atleast_2d(d_new):
        nb = np.argsort(row)[:k]
        out.append([{"id": ids[j], "distance": float(row[j])} for j in nb])
    return out


def bridge_proximity(d_new: np.ndarray, ids: list[str], bridges: set[str],
                     k: int = 5) -> np.ndarray:
    """Fraction of a spectrum's nearest references that are bridge objects.

    High bridge proximity is a specific, actionable warning: the spectrum sits where the
    hierarchy itself is ambiguous, so its theme assignment is uncertain for a structural
    reason rather than a noise reason.
    """
    out = []
    for row in np.atleast_2d(d_new):
        nb = np.argsort(row)[:k]
        out.append(float(np.mean([ids[j] in bridges for j in nb])))
    return np.array(out)


def local_confidence(density: np.ndarray, ood: np.ndarray, bridge: np.ndarray) -> np.ndarray:
    """One number combining the three geometric warnings, in [0, 1]."""
    d = density / (np.median(density) + EPS)
    return np.clip(np.tanh(d) * np.exp(-np.clip(ood - 1.0, 0, None)) * (1.0 - 0.5 * bridge),
                   0.0, 1.0)


# ── benchmark criteria for the extension itself ──────────────────────────────
def extension_fidelity(method: str, D_ref: np.ndarray, coords_ref: np.ndarray,
                       k: int = 5) -> dict:
    """Leave-one-reference-out: re-place each reference object using only the others.

    The only honest way to benchmark an out-of-sample extension without new data — the
    reference's true coordinate is known, and a good extension must recover it from its
    neighbours alone.
    """
    n = D_ref.shape[0]
    err, cos = [], []
    for i in range(n):
        keep = [j for j in range(n) if j != i]
        y = extend(method, D_ref[i, keep][None, :], D_ref[np.ix_(keep, keep)],
                   coords_ref[keep], k)[0]
        t = coords_ref[i]
        err.append(float(np.linalg.norm(y - t)))
        cos.append(float(y @ t / (np.linalg.norm(y) * np.linalg.norm(t) + EPS)))
    scale = float(np.linalg.norm(coords_ref, axis=1).mean()) + EPS
    return {"method": method, "mean_error": float(np.mean(err)),
            "median_error": float(np.median(err)),
            "relative_error": float(np.mean(err) / scale),
            "mean_cosine": float(np.mean(cos))}


def neighbour_preservation(method: str, D_ref: np.ndarray, coords_ref: np.ndarray,
                           k: int = 5) -> float:
    """Fraction of each reference's true k nearest neighbours recovered from its extended
    position — the property that actually matters for retrieval."""
    n = D_ref.shape[0]
    hits = []
    for i in range(n):
        keep = [j for j in range(n) if j != i]
        y = extend(method, D_ref[i, keep][None, :], D_ref[np.ix_(keep, keep)],
                   coords_ref[keep], k)[0]
        d_emb = np.linalg.norm(coords_ref[keep] - y, axis=1)
        got = {keep[j] for j in np.argsort(d_emb)[:k]}
        true = set(np.argsort(D_ref[i])[1:k + 1])
        hits.append(len(got & true) / k)
    return float(np.mean(hits))
