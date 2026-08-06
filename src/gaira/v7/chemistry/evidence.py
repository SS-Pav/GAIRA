"""GAIRA V7 — Phase 06: candidate Chemistry Evidence models.

Every model maps a frozen 49-dimensional CSM activation vector to a non-negative 16-dimensional
evidence vector in the fixed `CLASS_ORDER`. Four families are benchmarked and the winner is
chosen inside nested molecule-grouped CV, never on the number finally reported.

    e_c(x)  =  "support that x occupies the reference region of chemistry class c"

**Not** a molar fraction, **not** a concentration, **not** a percent composition, and **not** a
probability unless a calibrator has been fitted and its reliability reported. The raw evidence
is retained; probabilities are a separate, calibrated view of it.
"""
from __future__ import annotations

import numpy as np

from .registry import CLASS_ORDER

EPS = 1e-12
NC = len(CLASS_ORDER)

AGGREGATIONS = ("max", "sum", "mean", "topk_mean", "logsumexp", "weighted_vote")
SIZE_CORRECTIONS = ("none", "divide_n", "divide_sqrt_n", "idf")
PROTOTYPES = ("mean", "median", "medoid", "multi2", "shrinkage")
PROBABILISTIC = ("logreg", "shrinkage_lda", "nearest_centroid", "prototype_likelihood",
                 "class_conditional_distance")
FAMILIES = ("A_similarity_evidence", "B_class_prototype", "C_probabilistic", "D_hierarchical")


def _unit(M):
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + EPS)


def cosine(Q, R):
    return np.clip(_unit(np.atleast_2d(Q)) @ _unit(np.atleast_2d(R)).T, 0.0, 1.0)


# ── Model A — similarity-weighted molecule evidence ──────────────────────────
def _size_weight(n: int, n_all: np.ndarray, correction: str) -> float:
    """Class-size correction, so a large class cannot win merely by holding more references."""
    if correction == "none":
        return 1.0
    if correction == "divide_n":
        return 1.0 / max(n, 1)
    if correction == "divide_sqrt_n":
        return 1.0 / np.sqrt(max(n, 1))
    if correction == "idf":
        # Inverse-frequency weight, normalised so the mean weight is 1 and the correction cannot
        # change the overall scale of the evidence vector — only its distribution across classes.
        w = np.log(n_all.sum() / max(n, 1)) + 1.0
        return float(w / (np.log(n_all.sum() / np.maximum(n_all, 1)) + 1.0).mean())
    raise ValueError(f"unknown size correction {correction}")


def fit_A(A_tr, y_tr, cls_tr, aggregation="max", size_correction="none", topk=3,
          temperature=0.05) -> dict:
    """Reference bank plus the aggregation rule. Nothing is fitted; the bank is a lookup."""
    mols = sorted(set(y_tr))
    R = np.vstack([A_tr[y_tr == m].mean(axis=0) for m in mols])
    mol_cls = np.array([cls_tr[y_tr == m][0] for m in mols])
    counts = np.array([int((mol_cls == c).sum()) for c in CLASS_ORDER])
    return {"family": "A_similarity_evidence", "R": R, "mols": mols, "mol_cls": mol_cls,
            "counts": counts, "aggregation": aggregation, "size_correction": size_correction,
            "topk": topk, "temperature": temperature}


def predict_A(model, A_q) -> np.ndarray:
    S = cosine(A_q, model["R"])                        # (n, n_mol) in [0, 1]
    agg, corr = model["aggregation"], model["size_correction"]
    E = np.zeros((S.shape[0], NC))
    for k, c in enumerate(CLASS_ORDER):
        m = model["mol_cls"] == c
        if not m.any():
            continue
        sub = S[:, m]
        if agg == "max":
            v = sub.max(axis=1)
        elif agg == "sum":
            v = sub.sum(axis=1)
        elif agg == "mean":
            v = sub.mean(axis=1)
        elif agg == "topk_mean":
            kk = min(model["topk"], sub.shape[1])
            v = np.sort(sub, axis=1)[:, -kk:].mean(axis=1)
        elif agg == "logsumexp":
            # Soft maximum: dominated by the best-matching molecule but not blind to the rest.
            t = model["temperature"]
            v = t * (np.log(np.exp(sub / t).sum(axis=1) + EPS))
        elif agg == "weighted_vote":
            w = np.exp((sub - sub.max(axis=1, keepdims=True)) / model["temperature"])
            v = (sub * w).sum(axis=1) / (w.sum(axis=1) + EPS)
        else:
            raise ValueError(f"unknown aggregation {agg}")
        E[:, k] = v * _size_weight(int(m.sum()), model["counts"], corr)
    return np.clip(E, 0.0, None)


# ── Model B — class-prototype similarity ─────────────────────────────────────
def fit_B(A_tr, y_tr, cls_tr, prototype="mean", shrinkage=0.3) -> dict:
    """One or more prototypes per class, built from training molecules only.

    Prototypes are built at the **molecule** level first (mean over a molecule's spectra) and
    only then aggregated to the class, so a molecule with three replicates does not get three
    votes in its own class prototype. That is principle P-11 applied one level up.
    """
    mols = sorted(set(y_tr))
    M = np.vstack([A_tr[y_tr == m].mean(axis=0) for m in mols])
    mol_cls = np.array([cls_tr[y_tr == m][0] for m in mols])
    grand = M.mean(axis=0)
    protos = {}
    for c in CLASS_ORDER:
        sub = M[mol_cls == c]
        if len(sub) == 0:
            protos[c] = np.zeros((1, A_tr.shape[1]))
            continue
        if prototype == "mean":
            P = sub.mean(axis=0, keepdims=True)
        elif prototype == "median":
            P = np.median(sub, axis=0, keepdims=True)
        elif prototype == "medoid":
            D = 1.0 - cosine(sub, sub)
            P = sub[int(np.argmin(D.sum(axis=1))), None]
        elif prototype == "multi2":
            P = _kmeans2(sub) if len(sub) >= 4 else sub.mean(axis=0, keepdims=True)
        elif prototype == "shrinkage":
            # Shrunk toward the corpus grand mean — the standard small-class remedy. Classes
            # with 3 molecules are the reason this variant exists.
            P = ((1 - shrinkage) * sub.mean(axis=0) + shrinkage * grand)[None, :]
        else:
            raise ValueError(f"unknown prototype {prototype}")
        protos[c] = np.clip(P, 0.0, None)
    return {"family": "B_class_prototype", "protos": protos, "prototype": prototype}


def _kmeans2(sub, iters: int = 50) -> np.ndarray:
    """Deterministic 2-means: seeded by the two most distant members, no RNG."""
    D = 1.0 - cosine(sub, sub)
    i, j = np.unravel_index(int(np.argmax(D)), D.shape)
    C = sub[[i, j]].copy()
    for _ in range(iters):
        lab = np.argmax(cosine(sub, C), axis=1)
        new = np.vstack([sub[lab == t].mean(axis=0) if (lab == t).any() else C[t]
                         for t in range(2)])
        if np.allclose(new, C):
            break
        C = new
    return C


def predict_B(model, A_q) -> np.ndarray:
    E = np.zeros((np.atleast_2d(A_q).shape[0], NC))
    for k, c in enumerate(CLASS_ORDER):
        E[:, k] = cosine(A_q, model["protos"][c]).max(axis=1)
    return np.clip(E, 0.0, None)


# ── Model C — transparent probabilistic models in CSM space ──────────────────
def fit_C(A_tr, y_tr, cls_tr, method="logreg", C=1.0, shrinkage=0.3) -> dict:
    """Only inspectable models: linear, distance-based, or a shrunk Gaussian.

    No neural network, no random forest, no boosting, no opaque embedding — the brief forbids
    them and so does P-04, because a chemistry score that cannot be decomposed cannot be
    provenanced.
    """
    mols = sorted(set(y_tr))
    M = np.vstack([A_tr[y_tr == m].mean(axis=0) for m in mols])
    mol_cls = np.array([cls_tr[y_tr == m][0] for m in mols])
    present = [c for c in CLASS_ORDER if (mol_cls == c).sum() > 0]
    if method == "logreg":
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(C=C, max_iter=5000, class_weight="balanced",
                                 random_state=0)
        clf.fit(_unit(M), mol_cls)
        return {"family": "C_probabilistic", "method": method, "clf": clf,
                "classes": list(clf.classes_)}
    if method == "shrinkage_lda":
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=shrinkage)
        clf.fit(_unit(M), mol_cls)
        return {"family": "C_probabilistic", "method": method, "clf": clf,
                "classes": list(clf.classes_)}
    if method == "nearest_centroid":
        cent = {c: _unit(M[mol_cls == c]).mean(axis=0) for c in present}
        return {"family": "C_probabilistic", "method": method, "cent": cent}
    if method == "prototype_likelihood":
        # Isotropic Gaussian per class with a pooled, shrunk variance — the smallest model that
        # yields a likelihood rather than a similarity.
        cent, var = {}, {}
        pooled = float(np.var(_unit(M) - np.vstack([_unit(M)[mol_cls == c].mean(axis=0)
                                                    for c in mol_cls])))
        for c in present:
            sub = _unit(M[mol_cls == c])
            cent[c] = sub.mean(axis=0)
            v = float(np.var(sub - cent[c])) if len(sub) > 1 else pooled
            var[c] = (1 - shrinkage) * v + shrinkage * pooled + EPS
        return {"family": "C_probabilistic", "method": method, "cent": cent, "var": var}
    if method == "class_conditional_distance":
        cent = {c: _unit(M[mol_cls == c]).mean(axis=0) for c in present}
        scale = {c: float(np.mean(1.0 - cosine(M[mol_cls == c], cent[c][None, :]))) + EPS
                 for c in present}
        return {"family": "C_probabilistic", "method": method, "cent": cent, "scale": scale}
    raise ValueError(f"unknown probabilistic method {method}")


def predict_C(model, A_q) -> np.ndarray:
    A_q = np.atleast_2d(A_q)
    E = np.zeros((A_q.shape[0], NC))
    m = model["method"]
    if m in ("logreg", "shrinkage_lda"):
        P = model["clf"].predict_proba(_unit(A_q))
        for j, c in enumerate(model["classes"]):
            E[:, CLASS_ORDER.index(c)] = P[:, j]
        return E
    Q = _unit(A_q)
    if m == "nearest_centroid":
        for c, v in model["cent"].items():
            E[:, CLASS_ORDER.index(c)] = np.clip(Q @ (v / (np.linalg.norm(v) + EPS)), 0, None)
        return E
    if m == "prototype_likelihood":
        L = np.full((A_q.shape[0], NC), -np.inf)
        for c, v in model["cent"].items():
            d = ((Q - v) ** 2).sum(axis=1)
            L[:, CLASS_ORDER.index(c)] = -0.5 * d / model["var"][c] \
                - 0.5 * Q.shape[1] * np.log(model["var"][c])
        L = np.where(np.isfinite(L), L, -1e9)
        L -= L.max(axis=1, keepdims=True)
        E = np.exp(L)
        return E / (E.sum(axis=1, keepdims=True) + EPS)
    if m == "class_conditional_distance":
        for c, v in model["cent"].items():
            d = 1.0 - np.clip(Q @ (v / (np.linalg.norm(v) + EPS)), -1, 1)
            E[:, CLASS_ORDER.index(c)] = np.exp(-d / model["scale"][c])
        return E
    raise ValueError(m)


# ── Model D — hierarchical broad → fine, with SOFT routing ───────────────────
def fit_D(A_tr, y_tr, cls_tr, broad_of: dict, base="A", lam=1.0, **kw) -> dict:
    """Broad superclass evidence multiplies fine evidence; it never excludes a fine class.

    The broad level comes from the frozen Phase 00 `broad_class` column — six superclasses,
    curated before any V7 model existed. It is not invented here and it is not tuned.

    Routing is soft by construction: `e_fine ← e_fine · (broad_evidence)^λ`. Because every broad
    evidence is strictly positive, a fine class stays reachable even when its superclass is not
    top-1. A hard filter would make a broad error unrecoverable, which at six-way accuracy would
    permanently lose a non-trivial share of queries.
    """
    fine = fit_A(A_tr, y_tr, cls_tr, **kw) if base == "A" else fit_B(A_tr, y_tr, cls_tr, **kw)
    broads = sorted({broad_of[m] for m in set(y_tr)})
    mols = sorted(set(y_tr))
    M = np.vstack([A_tr[y_tr == m].mean(axis=0) for m in mols])
    mb = np.array([broad_of[m] for m in mols])
    protos = {b: M[mb == b].mean(axis=0) for b in broads}
    fine_broad = {}
    for c in CLASS_ORDER:
        sel = [broad_of[m] for m in mols if cls_tr[y_tr == m][0] == c]
        fine_broad[c] = max(set(sel), key=sel.count) if sel else broads[0]
    return {"family": "D_hierarchical", "fine": fine, "base": base, "broads": broads,
            "protos": protos, "fine_broad": fine_broad, "lam": lam}


def predict_D(model, A_q) -> np.ndarray:
    E = predict_A(model["fine"], A_q) if model["base"] == "A" else predict_B(model["fine"], A_q)
    B = np.vstack([cosine(A_q, model["protos"][b][None, :]).ravel() for b in model["broads"]]).T
    B = np.clip(B, EPS, None)
    B = B / (B.max(axis=1, keepdims=True) + EPS)
    idx = {b: i for i, b in enumerate(model["broads"])}
    w = np.vstack([B[:, idx[model["fine_broad"][c]]] for c in CLASS_ORDER]).T
    return np.clip(E * (w ** model["lam"]), 0.0, None)


# ── dispatch ─────────────────────────────────────────────────────────────────
def fit(family: str, A_tr, y_tr, cls_tr, **kw):
    return {"A_similarity_evidence": fit_A, "B_class_prototype": fit_B,
            "C_probabilistic": fit_C, "D_hierarchical": fit_D}[family](A_tr, y_tr, cls_tr, **kw)


def predict(model, A_q) -> np.ndarray:
    return {"A_similarity_evidence": predict_A, "B_class_prototype": predict_B,
            "C_probabilistic": predict_C, "D_hierarchical": predict_D}[model["family"]](model, A_q)


# ── normalisation views ──────────────────────────────────────────────────────
NORMALISATIONS = ("raw", "l1", "calibrated")


def normalise(E: np.ndarray, how: str = "raw") -> np.ndarray:
    """Three views of the same evidence. They are not interchangeable.

    - `raw` — the model's own non-negative support. Comparable across spectra: a spectrum the
      atlas explains poorly has *low* evidence everywhere, and L1 normalisation destroys that.
    - `l1` — the composition-looking view. Useful for a radar only because a radar has no
      absolute scale, and dangerous because it invites a mixture reading.
    - `calibrated` — probabilities, produced by a fitted calibrator, not by this function.
    """
    E = np.atleast_2d(np.clip(E, 0.0, None))
    if how == "raw":
        return E
    if how == "l1":
        return E / (E.sum(axis=1, keepdims=True) + EPS)
    if how == "calibrated":
        raise ValueError("calibrated evidence comes from a fitted calibrator, not normalise()")
    raise ValueError(f"unknown normalisation {how}")
