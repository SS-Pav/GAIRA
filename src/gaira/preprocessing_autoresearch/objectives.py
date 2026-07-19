"""Stage B0 — the seven objective groups (multi-objective scorecard).

Nothing is collapsed into a hidden scalar. Each group is reported separately and
Pareto selection operates on the group summaries.
"""
from __future__ import annotations
import numpy as np

from ..representation.metrics import cosine_sim  # reuse
from . import peak_integrity as PI


def _unit(X):
    return np.nan_to_num(X) / (np.linalg.norm(np.nan_to_num(X), axis=1, keepdims=True) + 1e-12)


def _split_modalities(F, fmeta, analytes):
    """Aligned (R, S) matrices for the given analytes (one row per analyte)."""
    idx = {(a, m): i for i, (a, m) in enumerate(zip(fmeta.analyte, fmeta.modality))}
    keep = [a for a in analytes if (a, "raman") in idx and (a, "sers") in idx]
    if len(keep) < 3:
        return None, None, []
    R = np.vstack([F[idx[(a, "raman")]] for a in keep])
    S = np.vstack([F[idx[(a, "sers")]] for a in keep])
    return R, S, keep


# ── Objective 1 — cross-modal retrieval ──
def cross_modal(F, fmeta, analytes, n_perm=0, rng=None):
    R, S, keep = _split_modalities(F, fmeta, analytes)
    if R is None:
        return {"insufficient": True, "n": 0}
    Sim = _unit(R) @ _unit(S).T
    n = len(keep)

    def _dir(M):
        ranks = np.array([1 + int((M[i] > M[i, i]).sum()) for i in range(n)])
        return ranks

    r_rs = _dir(Sim); r_sr = _dir(Sim.T)
    ranks = np.concatenate([r_rs, r_sr])
    out = {"n": n, "chance_top1": 1.0 / n,
           "top1_r2s": float(np.mean(r_rs == 1)), "top1_s2r": float(np.mean(r_sr == 1)),
           "top1": float(np.mean(ranks == 1)), "top3": float(np.mean(ranks <= 3)),
           "top5": float(np.mean(ranks <= 5)), "mrr": float(np.mean(1.0 / ranks)),
           "median_rank": float(np.median(ranks)),
           "matched_cos": float(np.mean(np.diag(Sim))),
           "mismatched_cos": float((Sim.sum() - np.trace(Sim)) / (n * n - n)),
           "ranks": ranks.tolist()}
    out["matched_minus_mismatched"] = out["matched_cos"] - out["mismatched_cos"]
    if n_perm and rng is not None:                       # Control 3: label permutation
        null = []
        for _ in range(n_perm):
            p = rng.permutation(n)
            M = Sim[:, p]
            rr = np.array([1 + int((M[i] > M[i, i]).sum()) for i in range(n)])
            null.append(np.mean(1.0 / rr))
        null = np.array(null)
        out["perm_mrr_p"] = float((np.sum(null >= out["mrr"]) + 1) / (n_perm + 1))
        out["perm_mrr_null_mean"] = float(null.mean())
    return out


# ── Objective 2 — peak correspondence above null ──
def peak_correspondence(F, fmeta, analytes, grid, rng):
    R, S, keep = _split_modalities(F, fmeta, analytes)
    if R is None:
        return {"insufficient": True}
    if F.shape[1] != len(grid):
        # peak metrics require a wavenumber axis; concatenated representations
        # (e.g. intensity+derivative) have no single axis -> not applicable.
        return {"matched": np.nan, "mismatched": np.nan, "random": np.nan,
                "effect_vs_mismatched": np.nan, "effect_size": np.nan,
                "not_applicable": True}
    return PI.correspondence_with_nulls(R, S, keep, grid, rng)


# ── Objective 3 — replicate preservation (on UNAGGREGATED spectra) ──
def replicate_preservation(Xs, meta, analytes):
    """Absolute replicate cosine PLUS a scale-free replicate margin.

    Note: absolute replicate cosine is structurally reduced by removing any shared
    component (the common offset that inflates it is gone), so it penalises
    background correction by construction. `replicate_margin` (within-analyte minus
    between-analyte similarity in the SAME representation) is the scale-free
    counterpart and is reported alongside it.
    """
    out = {}
    for mod in ("raman", "sers"):
        vals, var, within_all = [], [], []
        m_all = ((meta.modality == mod) & meta.analyte.isin(analytes)).values
        for a in analytes:
            m = ((meta.analyte == a) & (meta.modality == mod)).values
            X = np.nan_to_num(Xs[m])
            if len(X) < 2:
                continue
            U = _unit(X); C = U @ U.T
            iu = np.triu_indices(len(X), 1)
            vals.append(float(C[iu].mean())); var.append(float(np.mean(np.var(X, axis=0))))
        out[f"{mod}_replicate_cos"] = float(np.median(vals)) if vals else np.nan
        out[f"{mod}_replicate_var"] = float(np.median(var)) if var else np.nan
        # between-analyte similarity in the same representation
        if m_all.sum() >= 4:
            U = _unit(np.nan_to_num(Xs[m_all])); lab = meta[m_all].analyte.values
            C = U @ U.T; np.fill_diagonal(C, np.nan)
            same = lab[:, None] == lab[None, :]
            btw = float(np.nanmean(np.where(~same, C, np.nan)))
            out[f"{mod}_between_analyte_cos"] = btw
            out[f"{mod}_replicate_margin"] = (out[f"{mod}_replicate_cos"] - btw) \
                if np.isfinite(out[f"{mod}_replicate_cos"]) else np.nan
        else:
            out[f"{mod}_between_analyte_cos"] = np.nan
            out[f"{mod}_replicate_margin"] = np.nan
    return out


# ── Objective 4 — within-modality chemistry ──
def within_modality_chemistry(F, fmeta, analytes):
    out = {}
    for mod in ("raman", "sers"):
        m = (fmeta.modality.values == mod) & fmeta.analyte.isin(analytes).values
        X, lab = np.nan_to_num(F[m]), fmeta[m].analyte.values
        if len(X) < 3 or len(np.unique(lab)) < 3:
            out[f"{mod}_1nn"] = np.nan; out[f"{mod}_margin"] = np.nan; continue
        U = _unit(X); C = U @ U.T; np.fill_diagonal(C, -np.inf)
        pred = lab[C.argmax(1)]
        out[f"{mod}_1nn"] = float(np.mean(pred == lab))
        same = lab[:, None] == lab[None, :]
        np.fill_diagonal(same, False)
        C2 = U @ U.T; np.fill_diagonal(C2, np.nan)
        w = np.nanmean(np.where(same, C2, np.nan)) if same.any() else np.nan
        b = np.nanmean(np.where(~same, C2, np.nan))
        out[f"{mod}_margin"] = float(w - b) if np.isfinite(w) else np.nan
    return out


# ── Objective 5 — nuisance suppression ──
def nuisance(F, fmeta, X_raw_ref=None):
    U = _unit(F)
    mod = (fmeta.modality.values == "sers").astype(int)
    # modality separability proxy: between-modality vs within-modality mean cosine
    C = U @ U.T; np.fill_diagonal(C, np.nan)
    same = mod[:, None] == mod[None, :]
    w = np.nanmean(np.where(same, C, np.nan)); b = np.nanmean(np.where(~same, C, np.nan))
    out = {"modality_separability": float(w - b)}
    if X_raw_ref is not None:
        tot = np.nan_to_num(X_raw_ref).sum(axis=1)
        base = np.nan_to_num(X_raw_ref).min(axis=1)
        pc1 = np.linalg.svd(U - U.mean(0), full_matrices=False)[0][:, 0]
        def c(a, b_):
            return 0.0 if (np.std(a) < 1e-12 or np.std(b_) < 1e-12) else float(abs(np.corrcoef(a, b_)[0, 1]))
        out["corr_total_intensity"] = c(pc1, tot)
        out["corr_baseline_magnitude"] = c(pc1, base)
    return out


# ── Objective 6 — spectral integrity ──
def spectral_integrity(F, fmeta, F_ref, grid):
    if F.shape[1] != len(grid):
        return {"peak_retention": np.nan, "peak_invention": np.nan,
                "peak_width_ratio": np.nan, "effective_rank": PI.effective_rank(F),
                "cross_analyte_duplicate_frac": np.nan,
                "negative_lobe_burden": np.nan, "edge_artefact_ratio": np.nan,
                "not_applicable": True}
    ret, inv, wid = [], [], []
    n = min(len(F), len(F_ref))          # guard: reference is analyte-level
    for i in range(n):
        r = PI.retention_invention(F_ref[i], F[i], grid)
        if np.isfinite(r["retention"]):
            ret.append(r["retention"]); inv.append(r["invention"])
            if np.isfinite(r["width_ratio"]):
                wid.append(r["width_ratio"])
    art = PI.artefact_burden(F, grid)
    U = _unit(F); C = U @ U.T; np.fill_diagonal(C, -np.inf)
    an = fmeta.analyte.values
    cross = an[:, None] != an[None, :]
    Cc = np.where(cross, C, -np.inf)
    return {"peak_retention": float(np.median(ret)) if ret else np.nan,
            "peak_invention": float(np.median(inv)) if inv else np.nan,
            "peak_width_ratio": float(np.median(wid)) if wid else np.nan,
            "effective_rank": PI.effective_rank(F),
            "cross_analyte_duplicate_frac": float(np.mean(Cc.max(axis=1) > 0.999)),
            **art}


# ── Objective 7 — complexity ──
def complexity(cand, runtime_s):
    return {"n_stages": cand.n_stages(), "n_hyperparams": cand.n_hyperparams(),
            "runtime_s": float(runtime_s), "modality_specific": bool(cand.modality_specific())}
