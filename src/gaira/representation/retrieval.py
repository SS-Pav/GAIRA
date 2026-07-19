"""Cross-modal matched-analyte retrieval (Phase 2 §8) — the core asset test.

Given Raman centroids and Ag-SERS centroids for the 51 matched analytes, ask:
does an analyte's Raman spectrum retrieve its own Ag-SERS spectrum better than
chance? Reports top-k accuracy, mean reciprocal rank, reciprocal-nearest-neighbour
rate, and permutation-null comparisons with confidence intervals.

Determinism: permutations use a fixed np.random.default_rng(seed); no global RNG.
"""
from __future__ import annotations
import numpy as np
from .metrics import cosine_sim


def _align(raman_meta, sers_meta):
    """Return analytes present in both, plus index arrays into each centroid set."""
    r_by = {a: i for i, a in enumerate(raman_meta.analyte)}
    s_by = {a: i for i, a in enumerate(sers_meta.analyte)}
    common = sorted(set(r_by) & set(s_by))
    ri = np.array([r_by[a] for a in common])
    si = np.array([s_by[a] for a in common])
    return common, ri, si


def cross_modal_retrieval(R, raman_meta, S, sers_meta, ks=(1, 3, 5)):
    """R/S: centroid matrices (one row per analyte-source). Aggregates to one
    row per analyte per modality first (mean over sources) so retrieval is
    analyte↔analyte. Returns metrics dict."""
    # collapse to one centroid per analyte per modality (mean across sources)
    def collapse(M, meta):
        out, names = [], []
        for a, idx in meta.groupby("analyte").groups.items():
            rows = M[[meta.index.get_loc(i) for i in idx]]
            out.append(rows.mean(axis=0)); names.append(a)
        import pandas as pd
        return np.vstack(out), pd.DataFrame({"analyte": names})
    Rc, rm = collapse(R, raman_meta)
    Sc, sm = collapse(S, sers_meta)
    common, ri, si = _align(rm, sm)
    if len(common) < 3:
        return {"n_matched": len(common), "insufficient": True}
    Rq, Sdb = Rc[ri], Sc[si]                 # aligned so row j = same analyte
    sim = cosine_sim(Rq, Sdb)                 # query Raman → retrieve SERS
    n = len(common)
    # rank of the true match (diagonal) for each query row
    ranks = np.array([1 + int((sim[i] > sim[i, i]).sum()) for i in range(n)])
    topk = {f"top{k}": float(np.mean(ranks <= k)) for k in ks}
    mrr = float(np.mean(1.0 / ranks))
    # reciprocal nearest neighbour: R→S argmax and S→R argmax agree
    rn = sim.argmax(axis=1)
    sn = sim.argmax(axis=0)
    rnn = float(np.mean([sn[rn[i]] == i for i in range(n)]))
    diag = float(np.mean(np.diag(sim)))
    offdiag = float((sim.sum() - np.trace(sim)) / (n * n - n))
    return {"n_matched": n, "top_k": topk, "mrr": mrr, "reciprocal_nn_rate": rnn,
            "mean_matched_cos": diag, "mean_unmatched_cos": offdiag,
            "matched_minus_unmatched": diag - offdiag, "ranks": ranks.tolist(),
            "analytes": common, "_sim": sim}


def permutation_null(sim, n_perm=2000, seed=0):
    """Null: shuffle SERS labels, recompute top1 / MRR / matched-cos.
    Returns observed, null mean/CI, and p-values."""
    rng = np.random.default_rng(seed)
    n = sim.shape[0]
    obs_top1 = float(np.mean([(sim[i] >= sim[i].max()).argmax() == i for i in range(n)]))
    obs_mrr = float(np.mean([1.0 / (1 + int((sim[i] > sim[i, i]).sum())) for i in range(n)]))
    obs_diag = float(np.mean(np.diag(sim)))
    nt1, nmrr, ndiag = [], [], []
    for _ in range(n_perm):
        p = rng.permutation(n)
        sp = sim[:, p]
        nt1.append(np.mean([sp[i].argmax() == i for i in range(n)]))
        nmrr.append(np.mean([1.0 / (1 + int((sp[i] > sp[i, i]).sum())) for i in range(n)]))
        ndiag.append(np.mean(np.diag(sp)))
    def pack(obs, null):
        null = np.array(null)
        return {"observed": float(obs), "null_mean": float(null.mean()),
                "null_ci95": [float(np.percentile(null, 2.5)), float(np.percentile(null, 97.5))],
                "p_value": float((np.sum(null >= obs) + 1) / (len(null) + 1))}
    return {"top1": pack(obs_top1, nt1), "mrr": pack(obs_mrr, nmrr),
            "matched_cos": pack(obs_diag, ndiag), "n_perm": n_perm}
