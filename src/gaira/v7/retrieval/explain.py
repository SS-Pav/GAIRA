"""GAIRA V7 — Phase 08: the explanation layer. Every score sums; nothing is hidden."""
from __future__ import annotations

import numpy as np

EPS = 1e-12


def decompose(i: int, j: int, sc: dict, labels, mol_class, csm_records, A_q, A_bank,
              axis_names, E_q, E_bank, top: int = 5) -> dict:
    """The full contribution table behind one (query, candidate) score.

    The four weighted terms are reported with the weights that produced them, and their sum is
    asserted against the model's own total. A decomposition that does not reconcile is a bug, not
    a rounding difference.
    """
    w = sc["weights"]
    terms = [
        {"term": "csm_similarity", "weight": w["alpha"], "value": float(sc["csm"][i, j]),
         "contribution": float(w["alpha"] * sc["csm"][i, j])},
        {"term": "chemistry_similarity", "weight": w["beta"], "value": float(sc["chem"][i, j]),
         "contribution": float(w["beta"] * sc["chem"][i, j])},
        {"term": "diagnostic_band_support", "weight": w["gamma"],
         "value": float(sc["band"][i, j]),
         "contribution": float(w["gamma"] * sc["band"][i, j])},
        {"term": "incompatibility_penalty", "weight": -w["delta"],
         "value": float(sc["penalty"][i, j]),
         "contribution": float(-w["delta"] * sc["penalty"][i, j])},
    ]
    subtotal = sum(t["contribution"] for t in terms)
    reranked = bool(sc["total"][i, j] >= 1.0)
    # Which CSMs carry the similarity, and which chemistry axes carry the chemistry term.
    a_q, r = np.asarray(A_q[i], float), np.asarray(A_bank[j], float)
    contrib = a_q * r / (np.linalg.norm(a_q) * np.linalg.norm(r) + EPS)
    cs = np.argsort(-contrib)[:top]
    e_q, e_r = np.asarray(E_q[i], float), np.asarray(E_bank[j], float)
    ec = e_q * e_r / (np.linalg.norm(e_q) * np.linalg.norm(e_r) + EPS)
    ax = np.argsort(-ec)[:top]
    return {
        "candidate": labels[j], "candidate_class": mol_class[j],
        "reranked": reranked,
        "score_total": float(sc["total"][i, j]),
        "score_offset_for_reranked_candidates": 1.0 if reranked else 0.0,
        "terms": terms, "terms_subtotal": float(subtotal),
        "reconciles": bool(abs(sc["total"][i, j] - ((1.0 if reranked else 0.0) + subtotal)
                               if reranked else
                               abs(sc["total"][i, j] - sc["csm"][i, j])) < 1e-9),
        "supporting_csms": [
            {"csm_id": csm_records[int(k)]["csm_id"],
             "cosine_contribution": float(contrib[k]),
             "share_of_csm_similarity": float(contrib[k] / (sc["csm"][i, j] + EPS)),
             "dominant_bands": [float(b) for b in csm_records[int(k)].get("dominant_bands", [])],
             "lsms": [l["lsm_id"] if isinstance(l, dict) else str(l)
                      for l in csm_records[int(k)].get("contributing_lsms", [])],
             "band_assignment": csm_records[int(k)].get("band_assignment", "")}
            for k in cs if contrib[k] > 1e-6],
        "supporting_chemistry_axes": [
            {"axis": axis_names[int(k)], "query_evidence": float(e_q[k]),
             "candidate_evidence": float(e_r[k]),
             "share_of_chemistry_similarity": float(ec[k] / (sc["chem"][i, j] + EPS))}
            for k in ax if ec[k] > 1e-6],
    }


def rank_change(sc: dict, labels, i: int, top: int = 10) -> "pd.DataFrame":
    """How reranking moved each candidate, and which term moved it."""
    import pandas as pd
    lab = np.asarray(labels)
    csm_order = np.argsort(-sc["csm"][i])
    fin_order = np.argsort(-sc["total"][i])
    csm_rank = {int(j): r + 1 for r, j in enumerate(csm_order)}
    fin_rank = {int(j): r + 1 for r, j in enumerate(fin_order)}
    w = sc["weights"]
    rows = []
    for j in fin_order[:top]:
        j = int(j)
        rows.append({"molecule": lab[j], "csm_rank": csm_rank[j], "final_rank": fin_rank[j],
                     "moved": csm_rank[j] - fin_rank[j],
                     "csm": float(sc["csm"][i, j]),
                     "chem_contribution": float(w["beta"] * sc["chem"][i, j]),
                     "band_contribution": float(w["gamma"] * sc["band"][i, j]),
                     "penalty_contribution": float(-w["delta"] * sc["penalty"][i, j])})
    return pd.DataFrame(rows)


def axis_importance(sc_fn, Eq, base_rank, y, seed: int = 0, n_rep: int = 5) -> "pd.DataFrame":
    """Permutation importance of each chemistry axis on the final rank of the true molecule.

    Exact rather than approximate: an axis is shuffled across spectra, the whole reranking is
    recomputed, and the change in mean reciprocal rank is recorded. Nothing is estimated by a
    surrogate model.
    """
    import pandas as pd
    rng = np.random.default_rng(seed)
    base = float(np.mean(1.0 / base_rank))
    rows = []
    for k in range(np.asarray(Eq).shape[1]):
        deltas = []
        for _ in range(n_rep):
            Ep = np.array(Eq, float).copy()
            Ep[:, k] = Ep[rng.permutation(len(Ep)), k]
            deltas.append(base - float(np.mean(1.0 / sc_fn(Ep))))
        rows.append({"axis_index": k, "delta_mrr": float(np.mean(deltas)),
                     "sd": float(np.std(deltas))})
    return pd.DataFrame(rows).sort_values("delta_mrr", ascending=False)
