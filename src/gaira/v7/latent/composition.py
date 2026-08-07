"""GAIRA V7 — Phase 06.5: what is inside each emergent cluster, and what it should be called.

Chemistry enters here for the first time, and only as *interpretation*. Nothing in this module
feeds back into `clustering.py`.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12
KINDS = ("chemically_coherent", "spectroscopically_coherent", "hierarchical_subfamily",
         "bridge", "acquisition_confounded", "mixed", "unresolved")


def _purity(v: np.ndarray) -> tuple[str, float]:
    u, n = np.unique(v, return_counts=True)
    return str(u[int(np.argmax(n))]), float(n.max() / n.sum())


def _entropy(v: np.ndarray) -> float:
    _, n = np.unique(v, return_counts=True)
    if len(n) < 2:
        return 0.0
    p = n / n.sum()
    return float(-(p * np.log(p)).sum() / np.log(len(n)))


def describe(lab: np.ndarray, M: np.ndarray, mols: list[str], cls: np.ndarray,
             broad: np.ndarray, src: np.ndarray, exc: np.ndarray, nrep: np.ndarray,
             csm_records: list[dict], grid: np.ndarray, CSM: np.ndarray) -> list[dict]:
    """One record per cluster: composition, dominant motifs and bands, neighbours, outliers."""
    from .clustering import cosine_distance, unit
    D = cosine_distance(M)
    N = unit(M)
    ids = sorted({int(v) for v in lab if v >= 0})
    cent = {c: N[lab == c].mean(axis=0) for c in ids}
    out = []
    for c in ids:
        sel = lab == c
        A = M[sel]
        mean_act = A.mean(axis=0)
        top_csm = np.argsort(-mean_act)[:6]
        top_csm = [int(j) for j in top_csm if mean_act[j] > 1e-6]
        lsms, bands = [], []
        for j in top_csm:
            r = csm_records[j]
            lsms += [l["lsm_id"] if isinstance(l, dict) else str(l)
                     for l in r.get("contributing_lsms", [])]
            bands += [float(b) for b in r.get("dominant_bands", [])]
        prof = mean_act @ CSM
        pk = np.argsort(-prof)[:8]
        others = [d for d in ids if d != c]
        dn = sorted(((float(1.0 - cent[c] @ cent[d] /
                            (np.linalg.norm(cent[c]) * np.linalg.norm(cent[d]) + EPS)), d)
                     for d in others))
        within = D[np.ix_(sel, sel)]
        iu = np.triu_indices(int(sel.sum()), 1)
        # A member whose mean distance to its own cluster exceeds the cluster's 90th percentile
        # is flagged; a member closer to another centroid than to its own is a bridge.
        md = D[np.ix_(sel, sel)].mean(axis=1)
        thr = float(np.quantile(md, 0.90)) if sel.sum() > 3 else np.inf
        members = [mols[i] for i in np.where(sel)[0]]
        outliers = [mols[i] for i, m in zip(np.where(sel)[0], md) if m > thr]
        bridges = []
        for i in np.where(sel)[0]:
            own = 1.0 - N[i] @ cent[c] / (np.linalg.norm(cent[c]) + EPS)
            best = min((1.0 - N[i] @ cent[d] / (np.linalg.norm(cent[d]) + EPS), d)
                       for d in others) if others else (np.inf, None)
            if best[0] < own:
                bridges.append({"molecule": mols[i], "nearer_cluster": int(best[1])})
        fine_top, fine_pur = _purity(cls[sel])
        broad_top, broad_pur = _purity(broad[sel])
        src_top, src_pur = _purity(src[sel])
        exc_top, exc_pur = _purity(exc[sel])
        out.append({
            "cluster": c, "n_molecules": int(sel.sum()),
            "n_spectra": int(nrep[sel].sum()),
            "members": members,
            "fine_classes": {str(k): int(v) for k, v in
                             zip(*np.unique(cls[sel], return_counts=True))},
            "dominant_fine_class": fine_top, "fine_purity": fine_pur,
            "fine_entropy": _entropy(cls[sel]),
            "dominant_broad_class": broad_top, "broad_purity": broad_pur,
            "sources": {str(k): int(v) for k, v in
                        zip(*np.unique(src[sel], return_counts=True))},
            "dominant_source": src_top, "source_purity": src_pur,
            "excitations": {str(k): int(v) for k, v in
                            zip(*np.unique(exc[sel], return_counts=True))},
            "dominant_excitation": exc_top, "excitation_purity": exc_pur,
            "mean_replicates": float(nrep[sel].mean()),
            "dominant_csms": [csm_records[j]["csm_id"] for j in top_csm],
            "dominant_csm_activations": [float(mean_act[j]) for j in top_csm],
            "dominant_lsms": sorted(set(lsms))[:8],
            "dominant_bands": sorted({round(b) for b in bands})[:10],
            "consensus_peak_positions": sorted(float(grid[p]) for p in pk),
            "within_cluster_mean_distance": float(within[iu].mean()) if len(iu[0]) else 0.0,
            "within_cluster_max_distance": float(within[iu].max()) if len(iu[0]) else 0.0,
            "nearest_clusters": [{"cluster": int(d), "centroid_distance": float(dist)}
                                 for dist, d in dn[:3]],
            "outlier_members": outliers,
            "bridge_members": bridges,
        })
    return out


def classify(rec: dict, n_total_mol: int, source_baseline: float,
             exc_baseline: float) -> tuple[str, str]:
    """Assign one of the seven kinds, with a written justification. Rule-based, not by eye.

    The thresholds are declared here, before any cluster was inspected, so the classification
    cannot drift toward a more flattering label once the composition is known.
    """
    fp, bp, sp, ep = (rec["fine_purity"], rec["broad_purity"], rec["source_purity"],
                      rec["excitation_purity"])
    nb = len(rec["bridge_members"])
    n = rec["n_molecules"]
    # Acquisition confounding first: if a cluster is purer in *source* or *excitation* than in
    # chemistry, and materially purer than the corpus baseline, the geometry is reading the
    # instrument rather than the molecule. That claim outranks any chemical reading.
    if (sp > fp and sp > source_baseline + 0.25) or (ep > fp and ep > exc_baseline + 0.25):
        return ("acquisition_confounded",
                f"source purity {sp:.2f} / excitation purity {ep:.2f} exceed chemistry purity "
                f"{fp:.2f} and the corpus baselines ({source_baseline:.2f} / "
                f"{exc_baseline:.2f}); the grouping tracks acquisition, not chemistry")
    if n <= 2:
        return ("unresolved",
                f"only {n} molecules — too small to characterise; reported, not interpreted")
    if fp >= 0.70:
        return ("chemically_coherent",
                f"{fp:.0%} of members share fine class '{rec['dominant_fine_class']}' "
                f"(entropy {rec['fine_entropy']:.2f})")
    if bp >= 0.75 and fp >= 0.40:
        return ("hierarchical_subfamily",
                f"broad class '{rec['dominant_broad_class']}' at {bp:.0%} while fine purity is "
                f"only {fp:.0%}: a superclass-level grouping that splits below it")
    if nb >= max(2, 0.25 * n):
        return ("bridge",
                f"{nb} of {n} members sit nearer another cluster's centroid than their own; "
                f"this is an interface region, not an island")
    if rec["within_cluster_mean_distance"] <= 0.35 and fp < 0.70:
        return ("spectroscopically_coherent",
                f"tight in CSM space (mean within-cluster distance "
                f"{rec['within_cluster_mean_distance']:.2f}) but chemically mixed "
                f"({fp:.0%} purity): shared vibrational structure across chemistries")
    if fp < 0.40 and bp < 0.60:
        return ("mixed",
                f"fine purity {fp:.0%} and broad purity {bp:.0%}: no dominant chemistry at "
                f"either level")
    return ("unresolved",
            f"fine {fp:.0%}, broad {bp:.0%}, within-cluster distance "
            f"{rec['within_cluster_mean_distance']:.2f}: no rule applies cleanly")


def baselines(cls, src, exc) -> dict:
    """Corpus-level purity a random cluster would achieve, so 'pure' means something."""
    return {"fine": float(np.max(np.unique(cls, return_counts=True)[1]) / len(cls)),
            "source": float(np.max(np.unique(src, return_counts=True)[1]) / len(src)),
            "excitation": float(np.max(np.unique(exc, return_counts=True)[1]) / len(exc))}
