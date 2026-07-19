"""Representation reliability / uncertainty signals (§19).

NOT calibrated probabilities (do not label them as such). Reliability signals only:
distance-to-training-support, neighbor agreement, cross-modal agreement for matched
references, and an OOD score. Seed/model agreement is computed at benchmark level.
"""
from __future__ import annotations
import numpy as np
from ..representation.metrics import cosine_sim


def distance_to_support(F_eval, F_train):
    """Min cosine distance of each eval feature to any training feature."""
    sim = cosine_sim(F_eval, F_train)
    return 1.0 - sim.max(axis=1)


def neighbor_agreement(F_eval, F_train, labels_train, k=5):
    """Fraction of k nearest TRAINING neighbours sharing the modal label (label
    agreement as a confidence proxy)."""
    sim = cosine_sim(F_eval, F_train)
    knn = np.argsort(-sim, axis=1)[:, :k]
    labels_train = np.asarray(labels_train)
    agree = []
    for i in range(len(F_eval)):
        vals, cnt = np.unique(labels_train[knn[i]], return_counts=True)
        agree.append(cnt.max() / k)
    return np.array(agree)


def cross_modal_agreement(Fr, ar, Fs, as_):
    """For matched analytes, cosine similarity between the analyte's Raman and SERS
    feature centroids — a per-reference reliability signal."""
    out = {}
    for a in set(ar) & set(as_):
        r = Fr[np.asarray(ar) == a].mean(0)
        s = Fs[np.asarray(as_) == a].mean(0)
        out[a] = float(np.dot(r, s) / (np.linalg.norm(r) * np.linalg.norm(s) + 1e-12))
    return out


def ood_score(F_eval, F_train, k=5):
    """OOD = mean cosine distance to k nearest training points (higher = more OOD)."""
    sim = cosine_sim(F_eval, F_train)
    topk = np.sort(sim, axis=1)[:, -k:]
    return 1.0 - topk.mean(axis=1)
