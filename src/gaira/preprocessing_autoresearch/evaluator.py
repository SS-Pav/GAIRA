"""Stage B0 — nested leakage-safe evaluation.

OUTER: held-out matched analytes (both modalities test-only) — used EXACTLY ONCE.
INNER: analyte-grouped CV over the remaining matched analytes, for all selection.

Everything fold-dependent (Ag-SERS background model) is fitted on training spectra
only. Non-matched analytes are always available for background fitting because they
can never be a test analyte.
"""
from __future__ import annotations
import time
import numpy as np
import pandas as pd

from . import pipeline as PL
from . import objectives as OB


# ───────────────────────── nested splits ─────────────────────────
def make_nested_splits(matched_analytes, n_outer=5, n_inner=4, seed=0):
    """Deterministic grouped nested folds over MATCHED analytes."""
    A = sorted(matched_analytes)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(A))
    outer = {A[order[i]]: i % n_outer for i in range(len(A))}
    folds = []
    for o in range(n_outer):
        test = sorted([a for a in A if outer[a] == o])
        devel = sorted([a for a in A if outer[a] != o])
        r2 = np.random.default_rng(seed + 100 + o)
        ordi = r2.permutation(len(devel))
        inner_of = {devel[ordi[i]]: i % n_inner for i in range(len(devel))}
        inner = [{"inner_fold": k,
                  "train": sorted([a for a in devel if inner_of[a] != k]),
                  "val": sorted([a for a in devel if inner_of[a] == k])}
                 for k in range(n_inner)]
        folds.append({"outer_fold": o, "test_analytes": test,
                      "devel_analytes": devel, "inner": inner})
    return {"seed": seed, "n_outer": n_outer, "n_inner": n_inner,
            "matched_analytes": A, "folds": folds}


def verify_nested_no_leakage(splits):
    """Assert outer test analytes never appear in any inner train/val list."""
    problems = []
    for f in splits["folds"]:
        te = set(f["test_analytes"])
        for inn in f["inner"]:
            if te & set(inn["train"]) or te & set(inn["val"]):
                problems.append(f"outer{f['outer_fold']}/inner{inn['inner_fold']}: test leak")
            if set(inn["train"]) & set(inn["val"]):
                problems.append(f"outer{f['outer_fold']}/inner{inn['inner_fold']}: train/val overlap")
    return {"ok": len(problems) == 0, "problems": problems}


# ───────────────────────── one candidate on one fold ─────────────────────────
def evaluate_fold(cand, cache, meta, grid, train_analytes, eval_analytes,
                  ref_features=None, rng=None, n_perm=0):
    """Fit stage-2 on TRAIN spectra, evaluate on EVAL analytes. Returns metrics."""
    X1 = cache.build(cand)
    # training spectra = all spectra whose analyte is a training analyte, PLUS
    # non-matched analytes (they are never evaluated, so they cannot leak).
    train_mask = meta.analyte.isin(train_analytes).values | (~meta.matched.values)
    state = PL.fit_stage2(cand, X1, meta, train_mask)

    F, fmeta = PL.apply_stage2(cand, X1, meta, state, aggregate=True)
    Xs, smeta = PL.apply_stage2(cand, X1, meta, state, aggregate=False)   # per-spectrum

    m = {}
    m.update({f"cm_{k}": v for k, v in OB.cross_modal(F, fmeta, eval_analytes,
                                                      n_perm=n_perm, rng=rng).items()
              if k != "ranks"})
    m.update({f"pk_{k}": v for k, v in OB.peak_correspondence(F, fmeta, eval_analytes,
                                                              grid, rng or np.random.default_rng(0)).items()})
    m.update({f"rep_{k}": v for k, v in OB.replicate_preservation(Xs, meta, eval_analytes).items()})
    # within-modality chemistry needs REPLICATE-level rows (aggregated features give a
    # single row per analyte, for which same-analyte 1-NN is undefined).
    m.update({f"chem_{k}": v for k, v in OB.within_modality_chemistry(Xs, smeta, eval_analytes).items()})
    m.update({f"nu_{k}": v for k, v in OB.nuisance(Xs, smeta, X_raw_ref=None).items()})
    if ref_features is not None:
        m.update({f"si_{k}": v for k, v in OB.spectral_integrity(F, fmeta, ref_features, grid).items()})
    m["bg_variance_explained"] = state["background"].variance_explained(
        X1[(meta.modality.values == "sers")])
    return m, (F, fmeta, state)


def evaluate_candidate_inner(cand, cache, meta, grid, fold, ref_features=None, seed=0):
    """Average a candidate's metrics over the inner folds of ONE outer fold."""
    t0 = time.time()
    rows = []
    for inn in fold["inner"]:
        rng = np.random.default_rng(seed + 7 * inn["inner_fold"])
        m, _ = evaluate_fold(cand, cache, meta, grid, inn["train"], inn["val"],
                             ref_features=ref_features, rng=rng, n_perm=0)
        m["inner_fold"] = inn["inner_fold"]
        rows.append(m)
    df = pd.DataFrame(rows)
    num = df.select_dtypes(include=[np.number])
    agg = num.mean(numeric_only=True).to_dict()
    agg.update({f"{k}_std": v for k, v in num.std(numeric_only=True).to_dict().items()
                if k.startswith(("cm_", "pk_"))})
    agg.update(OB.complexity(cand, time.time() - t0))
    agg["cid"] = cand.cid; agg["arm"] = cand.arm
    return agg, df
