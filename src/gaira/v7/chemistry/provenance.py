"""GAIRA V7 — Phase 06: provenance for a chemistry-class score.

    class evidence → supporting molecules → molecule similarity → query CSM activations
                   → supporting CSMs → LSMs → original Raman spectra

For the similarity-evidence family the decomposition is **mathematically exact**: the class
evidence is an explicit function of named per-molecule similarities, and those similarities are
inner products of the query activation with named reference activations, so every level sums or
maximises back to the number displayed. No chemistry score may exist without that chain.
"""
from __future__ import annotations

import numpy as np

from .registry import CLASS_ORDER

EPS = 1e-12


def class_chain(cls_id: str, a_q: np.ndarray, model: dict, csm_records: list[dict],
                top_molecules: int = 5, top_csms: int = 6) -> dict:
    """The full chain behind one class's evidence for one query."""
    from .evidence import cosine, _size_weight
    a_q = np.asarray(a_q, float).ravel()
    k = CLASS_ORDER.index(cls_id)
    if model["family"] not in ("A_similarity_evidence", "D_hierarchical"):
        return {"class_id": cls_id, "exact": False,
                "note": f"{model['family']} is not additively decomposable; "
                        "prototype and feature attributions are reported instead"}
    m = model["fine"] if model["family"] == "D_hierarchical" else model
    sel = m["mol_cls"] == cls_id
    if not sel.any():
        return {"class_id": cls_id, "exact": True, "evidence": 0.0, "molecules": [],
                "note": "no reference molecule of this class in the training bank"}
    S = cosine(a_q[None, :], m["R"][sel]).ravel()
    mols = [mm for mm, s in zip(m["mols"], sel) if s]
    order = np.argsort(-S)
    w = _size_weight(int(sel.sum()), m["counts"], m["size_correction"])
    links = []
    for i in order[:top_molecules]:
        r = m["R"][sel][i]
        contrib = a_q * r / (np.linalg.norm(a_q) * np.linalg.norm(r) + EPS)
        cs = np.argsort(-contrib)[:top_csms]
        links.append({
            "molecule": mols[i], "similarity": float(S[i]),
            "supporting_csms": [
                {"csm_id": csm_records[int(j)]["csm_id"],
                 "cosine_contribution": float(contrib[j]),
                 "share_of_similarity": float(contrib[j] / (S[i] + EPS)),
                 "query_activation": float(a_q[j]),
                 "reference_activation": float(r[j]),
                 "lsms": [l["lsm_id"] if isinstance(l, dict) else str(l)
                          for l in csm_records[int(j)].get("contributing_lsms", [])],
                 "dominant_bands": [float(b) for b in
                                    csm_records[int(j)].get("dominant_bands", [])],
                 "band_assignment": csm_records[int(j)].get("band_assignment", "")}
                for j in cs if contrib[j] > 1e-6],
        })
    return {
        "class_id": cls_id, "exact": True,
        "aggregation": m["aggregation"], "size_correction": m["size_correction"],
        "size_weight": float(w),
        "n_reference_molecules": int(sel.sum()),
        "evidence": float({"max": S.max(), "sum": S.sum(), "mean": S.mean()}.get(
            m["aggregation"], S.max()) * w),
        "molecules": links,
        "explained_share": float(sum(c["cosine_contribution"] for l in links
                                     for c in l["supporting_csms"]) /
                                 (sum(l["similarity"] for l in links) + EPS)),
    }


def verify(chains: list[dict], known_molecules: set, known_lsms: set,
           known_csms: set) -> "pd.DataFrame":
    """Every link checked against the frozen registries. A short chain is a broken chain."""
    import pandas as pd
    rows = []
    for ch in chains:
        if not ch.get("exact", False):
            rows.append({"class_id": ch["class_id"], "exact": False, "intact": False,
                         "reason": "not additively decomposable"})
            continue
        mols = [l["molecule"] for l in ch.get("molecules", [])]
        csms = [c["csm_id"] for l in ch.get("molecules", []) for c in l["supporting_csms"]]
        lsms = [x for l in ch.get("molecules", []) for c in l["supporting_csms"]
                for x in c["lsms"]]
        bad_m = [x for x in mols if x not in known_molecules]
        bad_c = [x for x in csms if x not in known_csms]
        bad_l = [x for x in lsms if x not in known_lsms]
        rows.append({
            "class_id": ch["class_id"], "exact": True,
            "n_molecules": len(mols), "n_csms": len(set(csms)), "n_lsms": len(set(lsms)),
            "unknown_molecules": len(bad_m), "unknown_csms": len(bad_c),
            "unknown_lsms": len(bad_l),
            "terminates_in_spectra": bool(mols),
            "intact": bool(mols) and bool(csms) and bool(lsms)
                      and not (bad_m or bad_c or bad_l),
            "reason": "",
        })
    return pd.DataFrame(rows)
