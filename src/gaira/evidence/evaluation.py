"""Shared Stage B evaluation framework (§11-13). Every representation — direct,
interpretable, encoder, hybrid — is scored with the SAME functions on its output
features. Reuses Stage A retrieval / leakage / clustering where possible.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from ..representation import retrieval as _ret
from ..representation import leakage as _leak
from ..representation import clustering as _clu
from ..representation.metrics import cosine_sim


# ── feature centroids (one per analyte per modality; no cross-modality averaging) ──
def feature_centroids(F, meta):
    rows, rmeta = [], []
    for (a, mod), idx in meta.groupby(["analyte", "modality"]).groups.items():
        pos = [meta.index.get_loc(i) for i in idx]
        rows.append(F[pos].mean(axis=0)); rmeta.append({"analyte": a, "modality": mod})
    return np.vstack(rows), pd.DataFrame(rmeta)


# ── cross-modal retrieval among a given analyte set ──
def cross_modal_metrics(F, meta, n_perm=2000, seed=0):
    C, cm = feature_centroids(F, meta)
    R, rm = C[cm.modality == "raman"], cm[cm.modality == "raman"].reset_index(drop=True)
    S, sm = C[cm.modality == "sers"], cm[cm.modality == "sers"].reset_index(drop=True)
    res = _ret.cross_modal_retrieval(R, rm, S, sm)
    if res.get("insufficient"):
        return res
    sim = np.array(res.pop("_sim"))
    res["permutation_null"] = _ret.permutation_null(sim, n_perm=n_perm, seed=seed)
    return res


# ── pooled held-out cross-modal retrieval across folds (§12-13) ──
def _fold_sim(F, meta):
    """Aligned Raman→SERS similarity matrix for the matched analytes in one fold."""
    C, cm = feature_centroids(F, meta)
    rm = cm.modality.values == "raman"; sm = cm.modality.values == "sers"
    ra = list(cm[rm].analyte); sa = list(cm[sm].analyte)
    common = sorted(set(ra) & set(sa))
    if len(common) < 2:
        return None
    R = C[rm][[ra.index(a) for a in common]]
    S = C[sm][[sa.index(a) for a in common]]
    return cosine_sim(R, S)   # row i / col i == same analyte (diagonal = matched)


def pooled_heldout_retrieval(fold_feats, ks=(1, 3, 5, 10), n_perm=2000, seed=0):
    """fold_feats: list of (F_test, meta_test). Ranks are computed WITHIN each fold's
    held-out candidate set, then pooled. Permutation null shuffles within folds."""
    sims = [s for s in (_fold_sim(F, m) for F, m in fold_feats) if s is not None]
    if not sims:
        return {"insufficient": True, "n_folds": 0}
    ranks, fold_n = [], []
    for sim in sims:
        n = sim.shape[0]; fold_n.append(n)
        for i in range(n):
            ranks.append(1 + int((sim[i] > sim[i, i]).sum()))
    ranks = np.array(ranks)
    topk = {f"top{k}": float(np.mean(ranks <= k)) for k in ks}
    mrr = float(np.mean(1.0 / ranks))
    # reciprocal-NN per fold
    rnn = []
    for sim in sims:
        n = sim.shape[0]; rn = sim.argmax(1); sn = sim.argmax(0)
        rnn.append(np.mean([sn[rn[i]] == i for i in range(n)]))
    # matched vs unmatched cosine
    diag = np.mean([np.mean(np.diag(s)) for s in sims])
    offd = np.mean([(s.sum() - np.trace(s)) / (s.size - len(s)) for s in sims if s.size > len(s)])
    # permutation null on pooled top1 + mrr
    rng = np.random.default_rng(seed)
    obs_top1 = topk["top1"]; obs_mrr = mrr
    nt1, nmrr = [], []
    for _ in range(n_perm):
        rk = []
        for sim in sims:
            n = sim.shape[0]; p = rng.permutation(n); sp = sim[:, p]
            for i in range(n):
                rk.append(1 + int((sp[i] > sp[i, i]).sum()))
        rk = np.array(rk)
        nt1.append(np.mean(rk <= 1)); nmrr.append(np.mean(1.0 / rk))
    nt1 = np.array(nt1); nmrr = np.array(nmrr)
    chance_top1 = float(np.mean([1.0 / n for n in fold_n]))
    return {"n_folds": len(sims), "n_query_analytes": int(len(ranks)),
            "fold_sizes": fold_n, "chance_top1": chance_top1,
            "top_k": topk, "mrr": mrr, "reciprocal_nn_rate": float(np.mean(rnn)),
            "mean_matched_cos": float(diag), "mean_unmatched_cos": float(offd),
            "matched_minus_unmatched": float(diag - offd),
            "median_rank": float(np.median(ranks)),
            "perm_top1_p": float((np.sum(nt1 >= obs_top1) + 1) / (n_perm + 1)),
            "perm_mrr_p": float((np.sum(nmrr >= obs_mrr) + 1) / (n_perm + 1)),
            "perm_top1_null_mean": float(nt1.mean()), "perm_mrr_null_mean": float(nmrr.mean())}


def bootstrap_ci_mrr(fold_feats, n_boot=1000, seed=0):
    """Analyte-bootstrap 95% CI on pooled MRR (resample held-out analytes)."""
    sims = [s for s in (_fold_sim(F, m) for F, m in fold_feats) if s is not None]
    if not sims:
        return None
    per = []  # (fold_idx, i, rank)
    for fi, sim in enumerate(sims):
        for i in range(sim.shape[0]):
            per.append(1.0 / (1 + int((sim[i] > sim[i, i]).sum())))
    per = np.array(per)
    rng = np.random.default_rng(seed)
    boots = [per[rng.integers(0, len(per), len(per))].mean() for _ in range(n_boot)]
    return [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


# ── nuisance leakage on features ──
def leakage_metrics(F, meta, seed=0):
    out = {}
    if meta.modality.nunique() > 1:
        out["modality"] = _leak.grouped_leakage(F, meta.modality.values, meta.analyte.values, seed=seed)
    if meta.source.nunique() > 1:
        out["source"] = _leak.grouped_leakage(F, meta.source.values, meta.analyte.values, seed=seed)
    return out


# ── within-modality chemistry retention ──
def within_modality_chem(F, meta, seed=0):
    out = {}
    for mod in ("raman", "sers"):
        m = meta.modality.values == mod
        if m.sum() < 4 or meta[m].analyte.nunique() < 2:
            continue
        Fm, mm = F[m], meta[m].reset_index(drop=True)
        labs = {"analyte": pd.factorize(mm.analyte)[0]}
        if mm.source.nunique() > 1:
            labs["source"] = pd.factorize(mm.source)[0]
        cl, _ = _clu.hierarchical(Fm, labs, metric="cosine")
        # replicate retrieval: for each spectrum, is nearest neighbour same analyte?
        sim = cosine_sim(Fm, Fm); np.fill_diagonal(sim, -np.inf)
        nn = sim.argmax(axis=1)
        same = np.mean([mm.analyte.iloc[i] == mm.analyte.iloc[nn[i]] for i in range(len(mm))])
        out[mod] = {"ari_analyte": cl["ari_vs"].get("analyte"),
                    "ari_source": cl["ari_vs"].get("source"),
                    "silhouette": cl.get("silhouette_cosine"),
                    "nn_same_analyte_rate": float(same)}
    return out


# ── embedding-collapse & shortcut diagnostics (§11) ──
def collapse_diagnostics(F, analytes=None):
    """Embedding-collapse diagnostics. When `analytes` is given, distinguishes
    same-analyte near-duplicates (expected for replicates) from CROSS-analyte
    duplicates (the true collapse signal)."""
    Fc = F - F.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Fc, compute_uv=False)
    tot = s.sum()
    if tot < 1e-12:                       # degenerate: all embeddings identical
        eff_rank = 1.0
    else:
        sp = s / tot
        eff_rank = float(np.exp(-np.sum(sp * np.log(sp + 1e-12))))
    sim = cosine_sim(F, F); np.fill_diagonal(sim, -np.inf)
    dup = float(np.mean(sim.max(axis=1) > 0.999))
    out = {"n_features": int(F.shape[1]), "effective_rank": eff_rank,
           "singular_frac_top1": float(s[0]), "mean_dim_std": float(F.std(axis=0).mean()),
           "min_dim_std": float(F.std(axis=0).min()), "duplicate_fraction": dup}
    if analytes is not None:
        a = np.asarray(analytes)
        cross = (a[:, None] != a[None, :])
        simc = sim.copy(); simc[~cross] = -np.inf
        out["cross_analyte_duplicate_fraction"] = float(np.mean(simc.max(axis=1) > 0.999))
    off = ~np.eye(len(F), dtype=bool)
    d = 1.0 - cosine_sim(F, F)
    out["pairwise_dist_mean"] = float(d[off].mean()); out["pairwise_dist_std"] = float(d[off].std())
    return out


def signal_stat_correlation(F, X):
    """Correlate embedding norm / PC1 with simple signal stats (total intensity,
    baseline magnitude, smoothness) → detect trivial shortcuts."""
    tot = np.nan_to_num(X).sum(axis=1)
    base = np.nan_to_num(X).min(axis=1)
    smooth = -np.abs(np.diff(np.nan_to_num(X), axis=1)).mean(axis=1)
    Fc = F - F.mean(0)
    pc1 = np.linalg.svd(Fc, full_matrices=False)[0][:, 0]
    norm = np.linalg.norm(F, axis=1)
    def c(a, b):
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            return 0.0
        return float(abs(np.corrcoef(a, b)[0, 1]))
    return {"pc1_vs_total_intensity": c(pc1, tot), "pc1_vs_baseline": c(pc1, base),
            "pc1_vs_smoothness": c(pc1, smooth), "norm_vs_total_intensity": c(norm, tot)}


# ── chemical-family neighborhood (evaluation-only labels) ──
def family_neighborhood(F, meta, exclude_non_small=True, k=5):
    m = meta.copy()
    if exclude_non_small and "non_small_molecule" in m:
        keep = ~m.non_small_molecule.values
        F, m = F[keep], m[keep].reset_index(drop=True)
    known = (m.family.values != "unknown")
    if known.sum() < 6:
        return {"skipped": "too few known-family analytes"}
    Fk, mk = F[known], m[known].reset_index(drop=True)
    sim = cosine_sim(Fk, Fk); np.fill_diagonal(sim, -np.inf)
    fam = mk.family.values
    knn = np.argsort(-sim, axis=1)[:, :k]
    same = np.mean([np.mean(fam[knn[i]] == fam[i]) for i in range(len(mk))])
    # chance = weighted prob of drawing same family
    _, cnt = np.unique(fam, return_counts=True)
    chance = float(np.sum((cnt / cnt.sum()) ** 2))
    return {"family_knn_purity": float(same), "chance_purity": chance, "k": k,
            "n_known_family": int(known.sum()),
            "families": {f: int((fam == f).sum()) for f in np.unique(fam)}}
