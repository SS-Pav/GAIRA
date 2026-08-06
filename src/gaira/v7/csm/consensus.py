"""GAIRA V7 — Phase 02: the consensus operator and the CSM record.

A CSM is not an average of things that looked alike. It is a claim that several independent
class-local decompositions found the same spectral phenomenon, and it carries the evidence for
that claim — including the evidence against it.
"""
from __future__ import annotations

import numpy as np

from .csm import CSM

OPERATORS = ("stability_weighted_mean", "nonnegative_medoid", "leading_direction")
EPS = 1e-12


def consensus_spectrum(H: np.ndarray, weights: np.ndarray,
                       operator: str = "stability_weighted_mean") -> np.ndarray:
    """Pi: the non-negative consensus of a set of LSMs, renormalised to unit L2.

    Non-negativity is an invariant, not a preference — it is what keeps a CSM readable as a
    spectrum and what lets inference project onto it with NNLS.
    """
    if H.shape[0] == 1:
        return H[0] / (np.linalg.norm(H[0]) + EPS)
    if operator == "stability_weighted_mean":
        w = np.asarray(weights, float)
        w = w / (w.sum() + EPS)
        c = (w[:, None] * H).sum(axis=0)
    elif operator == "nonnegative_medoid":
        N = H / (np.linalg.norm(H, axis=1, keepdims=True) + EPS)
        c = H[int(np.argmax((N @ N.T).sum(axis=1)))]
    elif operator == "leading_direction":
        from sklearn.decomposition import NMF
        c = NMF(n_components=1, init="nndsvda", random_state=0,
                max_iter=2000).fit(H).components_[0]
    else:
        raise ValueError(f"unknown consensus operator: {operator}")
    c = np.clip(c, 0.0, None)
    return c / (np.linalg.norm(c) + EPS)


def cohesion(H: np.ndarray, c: np.ndarray) -> float:
    """Mean cosine of the contributing LSMs to their consensus."""
    if H.shape[0] == 0:
        return 0.0
    N = H / (np.linalg.norm(H, axis=1, keepdims=True) + EPS)
    return float(np.clip(N @ (c / (np.linalg.norm(c) + EPS)), 0, 1).mean())


def uncertainty(H: np.ndarray, c: np.ndarray) -> float:
    """Spread of the contributors about the consensus: 1 - min cosine.

    The minimum, not the mean: a CSM is only as trustworthy as its worst contributor, and a
    mean would let one distant motif hide behind three close ones.
    """
    if H.shape[0] < 2:
        return 0.0
    N = H / (np.linalg.norm(H, axis=1, keepdims=True) + EPS)
    return float(1.0 - np.clip(N @ (c / (np.linalg.norm(c) + EPS)), 0, 1).min())


def build_csm(index: int, members: list[int], H: np.ndarray, W: np.ndarray,
              lsm_meta: list[dict], grid: np.ndarray, A_mol: np.ndarray,
              mol_ids: list[str], mol_class: list[str], operator: str,
              coassign: np.ndarray | None = None,
              min_activation: float = 0.05) -> CSM:
    """Assemble one CSM with complete, resolvable provenance.

    Provenance is CSM -> LSM -> canonical molecule -> original spectrum. Every level is stored
    explicitly rather than recomputed later, because a provenance chain that has to be
    reconstructed is a provenance chain that can silently break.
    """
    from .csm import dominant_bands

    Hm = H[members]
    stab = np.array([lsm_meta[i]["stability"] for i in members], float)
    c = consensus_spectrum(Hm, stab, operator)

    classes = sorted({lsm_meta[i]["chemical_class"] for i in members})
    analytes: list[str] = []
    for i in members:
        analytes.extend(lsm_meta[i]["analytes"])
    analytes = sorted(set(a for a in analytes if a))

    # Projected support = molecules for which this motif is among the strongest responders
    # across the WHOLE pooled dictionary. Two other normalisations were tried and both make the
    # record useless: peak-normalised, and class-normalised (which is the right scale for the
    # provenance *feature* but here gives every motif in a one-motif class a share of 1.0 for
    # every molecule). Both listed all 154 molecules under every CSM, so a lipid motif appeared
    # to be supported by cellulose and DNA.
    shares = A_mol / (A_mol.sum(axis=1, keepdims=True) + EPS)
    proj_support = sorted({mol_ids[r] for i in members
                           for r in np.where(shares[:, i] >= min_activation)[0]})

    inner = [W[a, b] for x, a in enumerate(members) for b in members[x + 1:]]
    others = [j for j in range(H.shape[0]) if j not in members]
    outer = [W[a, b] for a in members for b in others]

    return CSM(
        csm_id=f"csm{index:02d}",
        index=index,
        contributing_lsms=[lsm_meta[i]["motif_id"] for i in members],
        contributing_lsm_weights=[float(s / (stab.sum() + EPS)) for s in stab],
        member_indices=list(members),
        supporting_classes=classes,
        supporting_analytes=analytes,
        projected_support=proj_support,
        n_lsms=len(members), n_classes=len(classes), n_analytes=len(analytes),
        spectrum=c,
        dominant_bands=dominant_bands(c, grid),
        cohesion=cohesion(Hm, c),
        uncertainty=uncertainty(Hm, c),
        mean_edge_weight=float(np.mean(inner)) if inner else 1.0,
        min_edge_weight=float(np.min(inner)) if inner else 1.0,
        max_external_weight=float(np.max(outer)) if outer else 0.0,
        min_coassignment=(1.0 if coassign is None or len(members) < 2 else
                          float(min(coassign[a, b] for x, a in enumerate(members)
                                    for b in members[x + 1:]))),
        lsm_types=sorted({lsm_meta[i]["lsm_type"] for i in members}),
        is_singleton=len(members) == 1,
        is_anchored=False,
        is_cross_class=len(classes) > 1,
        consensus_operator=operator,
    )
