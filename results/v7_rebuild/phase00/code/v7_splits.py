"""GAIRA V7 Phase 00 — frozen analyte-grouped cross-validation splits.

Grouping is by `canonical_id`, never by surface name. That is the whole point: 11 of the
13 declared duplicates are CROSS-SOURCE, so surface-name grouping would place the same
molecule in train and test under two spellings and inflate every downstream metric
invisibly (risk R-09).

Three leakage checks must all read False or Phase 00 does not pass:

  canonical_id_across_folds   a canonical ID in more than one test fold
  alias_collision             two surface forms of one molecule in different folds
  replicate_across_folds      two spectra of one molecule in different folds

Deterministic: a fixed seed, a sorted ID list, and a size-balanced greedy assignment, so
the same inputs give byte-identical folds on any machine.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SPLIT_VERSION = "v7_cv_v1"
SEED = 0
N_FOLDS = 5


def make_folds(canon: pd.DataFrame, part: pd.DataFrame,
               n_folds: int = N_FOLDS, seed: int = SEED) -> pd.DataFrame:
    """Assign every canonical ID to exactly one test fold.

    Stratified by fine class so each fold sees every chemistry it can, then balanced on
    spectrum count so folds carry comparable evidence. Deterministic given (seed, inputs).
    """
    df = canon[["canonical_id", "n_spectra"]].merge(
        part[["canonical_id", "fine_class", "broad_class"]], on="canonical_id")
    df = df.sort_values(["fine_class", "n_spectra", "canonical_id"],
                        ascending=[True, False, True]).reset_index(drop=True)

    rng = np.random.default_rng(seed)
    load = np.zeros(n_folds)                     # spectra already assigned per fold
    assign = {}
    for cls, g in df.groupby("fine_class", sort=True):
        # rotate the starting fold per class so small classes do not all land in fold 0
        offset = int(rng.integers(0, n_folds))
        for i, (_, r) in enumerate(g.iterrows()):
            order = np.argsort(load + 1e-9 * ((np.arange(n_folds) - offset - i) % n_folds))
            f = int(order[0])
            assign[r.canonical_id] = f
            load[f] += r.n_spectra
    df["fold"] = df.canonical_id.map(assign)
    df["split_version"] = SPLIT_VERSION
    return df.sort_values("canonical_id").reset_index(drop=True)


def leakage_checks(folds: pd.DataFrame, meta: pd.DataFrame,
                   alias_to_cid: dict[str, str]) -> dict:
    fold_of = dict(zip(folds.canonical_id, folds.fold))

    # 1. a canonical ID may appear in exactly one fold
    counts = folds.groupby("canonical_id").fold.nunique()
    canonical_across = sorted(counts[counts > 1].index.tolist())

    # 2. every surface form of a molecule must land in that molecule's fold
    m = meta.copy()
    m["canonical_id"] = m.analyte.map(alias_to_cid)
    m["fold"] = m.canonical_id.map(fold_of)
    per_form = m.groupby("analyte").fold.nunique()
    alias_collisions = sorted(per_form[per_form > 1].index.tolist())

    # 3. every spectrum of a molecule must land in that molecule's fold
    per_cid = m.groupby("canonical_id").fold.nunique()
    replicate_across = sorted(per_cid[per_cid > 1].index.tolist())

    unassigned = sorted(m[m.fold.isna()].canonical_id.unique().tolist())

    return {
        "schema": "cv_splits_v1",
        "split_version": SPLIT_VERSION,
        "seed": SEED,
        "n_folds": int(folds.fold.nunique()),
        "grouping": "canonical_id",
        "leakage_checks": {
            "canonical_id_across_folds": bool(canonical_across),
            "alias_collision": bool(alias_collisions),
            "replicate_across_folds": bool(replicate_across),
        },
        "offenders": {
            "canonical_id_across_folds": canonical_across,
            "alias_collision": alias_collisions,
            "replicate_across_folds": replicate_across,
            "unassigned_canonical_ids": unassigned,
        },
        "all_checks_false": not (canonical_across or alias_collisions
                                 or replicate_across or unassigned),
    }


def fold_summary(folds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for f, g in folds.groupby("fold"):
        rows.append({"fold": int(f),
                     "n_canonical_analytes": int(len(g)),
                     "n_spectra": int(g.n_spectra.sum()),
                     "n_fine_classes": int(g.fine_class.nunique()),
                     "n_broad_classes": int(g.broad_class.nunique())})
    return pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)


def split_manifest(folds: pd.DataFrame, checks: dict) -> dict:
    out = dict(checks)
    out["folds"] = [
        {"fold": int(f),
         "test": sorted(g.canonical_id.tolist()),
         "n_test": int(len(g)),
         "n_train": int(len(folds) - len(g))}
        for f, g in folds.groupby("fold")
    ]
    return out
