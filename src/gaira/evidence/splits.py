"""GAIRA V5 Phase 2 Stage B — deterministic, leakage-safe split manifests (§5).

Predeclared BEFORE any model is trained. Four split families:
  A  held-out analytes           (no analyte in test occurs in train)
  B  held-out matched pairs       (whole matched analytes test-only, both modalities)
  C  replicate-group holdout      (no (analyte,modality,source) group crosses split)
  D  source sensitivity           (hold out a source; infeasible for single-source SERS)

Each fold carries train / val / test spectrum-id lists. Hyperparameters are chosen
on `val` (carved from train by the SAME grouping), never on `test`. Fully
deterministic given a seed; no global RNG.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def _grouped_folds(groups, k, seed):
    """Assign unique groups to k folds deterministically (seeded shuffle, round-robin)."""
    uniq = sorted(pd.unique(groups))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    fold_of = {uniq[order[i]]: i % k for i in range(len(uniq))}
    return fold_of


def _carve_val(train_groups_to_ids, groups_list, ids_list, frac, seed):
    """Carve a validation set from train by holding out whole groups (~frac of groups)."""
    uniq = sorted(set(groups_list))
    rng = np.random.default_rng(seed + 991)
    order = rng.permutation(len(uniq))
    n_val = max(1, int(round(frac * len(uniq))))
    val_groups = {uniq[order[i]] for i in range(n_val)}
    val_ids, tr_ids = [], []
    for g, i in zip(groups_list, ids_list):
        (val_ids if g in val_groups else tr_ids).append(i)
    return tr_ids, val_ids


def _make(meta, group_col, k, seed, restrict_mask=None, val_frac=0.2):
    """Generic grouped k-fold with nested val. Returns list of fold dicts."""
    m = meta if restrict_mask is None else meta[restrict_mask]
    fold_of = _grouped_folds(m[group_col].values, k, seed)
    folds = []
    for f in range(k):
        te_mask = m[group_col].map(lambda g: fold_of[g] == f)
        test_ids = m[te_mask].spectrum_id.tolist()
        tr_pool = m[~te_mask]
        tr_ids, val_ids = _carve_val(None, tr_pool[group_col].tolist(),
                                     tr_pool.spectrum_id.tolist(), val_frac, seed + f)
        folds.append({"fold": f, "train": tr_ids, "val": val_ids, "test": test_ids,
                      "n_train": len(tr_ids), "n_val": len(val_ids), "n_test": len(test_ids)})
    return folds


def make_all_splits(d, k=5, seed=0):
    """Build all split families for a StageBData. Returns a JSON-able dict."""
    meta = d.meta
    matched = set(d.matched_analytes)
    out = {"seed": seed, "k": k, "spectrum_id_universe": meta.spectrum_id.tolist(), "splits": {}}

    # A — held-out analytes (all analytes)
    out["splits"]["A_held_out_analytes"] = {
        "group_col": "analyte", "purpose": "chemical-family / broad generalization; NOT exact-analyte retrieval",
        "folds": _make(meta, "analyte", k, seed)}

    # B — held-out matched pairs (fold over matched analytes; test = matched analyte's BOTH modalities)
    mm = meta[meta.analyte.isin(matched)]
    fold_of = _grouped_folds(sorted(matched), k, seed + 7)
    bfolds = []
    for f in range(k):
        te_an = {a for a in matched if fold_of[a] == f}
        test_ids = mm[mm.analyte.isin(te_an)].spectrum_id.tolist()
        # train = ALL spectra whose analyte is not a held-out matched analyte
        tr_pool = meta[~meta.analyte.isin(te_an)]
        tr_ids, val_ids = _carve_val(None, tr_pool.analyte.tolist(), tr_pool.spectrum_id.tolist(), 0.2, seed + 100 + f)
        bfolds.append({"fold": f, "held_out_matched_analytes": sorted(te_an),
                       "train": tr_ids, "val": val_ids, "test": test_ids,
                       "n_train": len(tr_ids), "n_val": len(val_ids), "n_test": len(test_ids)})
    out["splits"]["B_held_out_matched_pairs"] = {
        "group_col": "analyte(matched)", "purpose": "cross-modal transfer to UNSEEN matched analytes", "folds": bfolds}

    # C — replicate-group holdout
    out["splits"]["C_replicate_group_holdout"] = {
        "group_col": "replicate_group", "purpose": "same-analyte generalization without technical leakage",
        "folds": _make(meta, "replicate_group", k, seed)}

    # D — source sensitivity (feasible only where a modality has >1 source)
    dsplit = {"purpose": "observation-source robustness", "feasible": {}, "folds": []}
    for mod in ("raman", "sers"):
        srcs = sorted(meta[meta.modality == mod].source.unique())
        dsplit["feasible"][mod] = {"n_sources": len(srcs), "sources": srcs,
                                   "leave_source_out_possible": len(srcs) > 1}
    # build leave-one-source-out folds for Raman (has 2 sources); SERS single-source → none
    raman = meta[meta.modality == "raman"]
    for src in sorted(raman.source.unique()):
        test_ids = raman[raman.source == src].spectrum_id.tolist()
        tr_ids = raman[raman.source != src].spectrum_id.tolist()
        if tr_ids and test_ids:
            dsplit["folds"].append({"held_out_source": src, "modality": "raman",
                                    "train": tr_ids, "test": test_ids,
                                    "n_train": len(tr_ids), "n_test": len(test_ids)})
    dsplit["sers_note"] = ("Ag-SERS is single-source (Gobbato) → leave-source-out is IMPOSSIBLE for SERS; "
                           "cross-source observation-domain invariance cannot be established in-corpus.")
    out["splits"]["D_source_sensitivity"] = dsplit
    return out


def verify_no_leakage(split_manifest, meta):
    """Assert grouped integrity for A/B/C. Returns dict of per-split check results."""
    id2 = meta.set_index("spectrum_id")
    checks = {}
    for name, cfg in split_manifest["splits"].items():
        if name.startswith("D_"):
            continue
        ok = True; detail = []
        for fold in cfg["folds"]:
            tr, va, te = set(fold["train"]), set(fold["val"]), set(fold["test"])
            if tr & te or tr & va or va & te:
                ok = False; detail.append(f"fold{fold.get('fold')}: id overlap")
            if name == "A_held_out_analytes":
                tr_an = set(id2.loc[list(tr | va), "analyte"]); te_an = set(id2.loc[list(te), "analyte"])
                if tr_an & te_an:
                    ok = False; detail.append(f"fold{fold.get('fold')}: analyte leak {tr_an & te_an}")
            if name == "B_held_out_matched_pairs":
                tr_an = set(id2.loc[list(tr | va), "analyte"]); te_an = set(id2.loc[list(te), "analyte"])
                if tr_an & te_an:
                    ok = False; detail.append(f"fold{fold.get('fold')}: matched-analyte leak")
            if name == "C_replicate_group_holdout":
                tr_g = set(id2.loc[list(tr | va), "replicate_group"]); te_g = set(id2.loc[list(te), "replicate_group"])
                if tr_g & te_g:
                    ok = False; detail.append(f"fold{fold.get('fold')}: replicate-group leak")
        checks[name] = {"ok": ok, "detail": detail}
    return checks
