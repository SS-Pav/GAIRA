"""Hierarchical clustering + agreement diagnostics (Phase 2 §11).

Reports cophenetic correlation, silhouette, and ARI of the cluster labels
against (a) analyte identity — do clusters recover chemistry? — and against
nuisance factors (modality, source) — do clusters instead recover acquisition?
"""
from __future__ import annotations
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, cophenet
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score, silhouette_score


def hierarchical(X, labels_by, metric="cosine", method="average", k=None):
    """X: matrix; labels_by: dict name->array of ground-truth-ish labels.
    Returns cophenetic corr, silhouette, and ARI vs each label set."""
    D = pdist(X, metric=metric)
    Z = linkage(D, method=method)
    coph, _ = cophenet(Z, D)
    n = X.shape[0]
    if k is None:
        # default: number of unique analytes if provided, else sqrt(n)
        k = len(np.unique(labels_by.get("analyte", np.arange(n)))) if "analyte" in labels_by else int(np.sqrt(n))
    k = max(2, min(k, n - 1))
    cl = fcluster(Z, t=k, criterion="maxclust")
    out = {"cophenetic_corr": float(coph), "k": int(k),
           "n_clusters_formed": int(len(np.unique(cl)))}
    try:
        out["silhouette_cosine"] = float(silhouette_score(X, cl, metric="cosine"))
    except Exception:
        out["silhouette_cosine"] = None
    out["ari_vs"] = {name: float(adjusted_rand_score(lab, cl)) for name, lab in labels_by.items()}
    return out, cl
