"""GAIRA V7 — Phase 06: the frozen 16-class chemistry ontology.

The ontology is **read, never constructed**. It was frozen in Phase 00 and it is also the label
space of the frozen Tier-1 success criteria (S-01, S-03 are defined on `v7_fine_16`). Renaming,
merging or splitting a class here would make every V7 result incomparable with its own bar, so
this module only describes what is already fixed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Canonical axis order for the 16-dimensional Chemistry Evidence Vector. Alphabetical, fixed
# once, and asserted by a unit test: a radar whose axes move between runs is not a coordinate
# system. Every artifact that stores a 16-vector stores it in this order.
CLASS_ORDER = (
    "acylglycerol", "carboxylic_acid_metabolite", "chromophore_pigment", "fatty_acid",
    "free_amino_acid", "mono_oligosaccharide", "nucleic_acid_polymer", "peptide_protein",
    "phosphate_metabolite", "phospholipid_sphingolipid", "polysaccharide", "purine",
    "pyrimidine", "small_nitrogenous", "sterol_steroid", "sulfur_thiol_cofactor",
)
# Chemically adjacent pairs, declared *before* any confusion matrix was inspected. Used only to
# ask whether errors are chemically reasonable — never to merge classes, never to score a
# near-miss as correct.
ADJACENT = (
    ("fatty_acid", "acylglycerol"),
    ("fatty_acid", "phospholipid_sphingolipid"),
    ("acylglycerol", "phospholipid_sphingolipid"),
    ("purine", "nucleic_acid_polymer"),
    ("pyrimidine", "nucleic_acid_polymer"),
    ("purine", "pyrimidine"),
    ("peptide_protein", "free_amino_acid"),
    ("mono_oligosaccharide", "polysaccharide"),
    ("carboxylic_acid_metabolite", "phosphate_metabolite"),
    ("free_amino_acid", "small_nitrogenous"),
    ("sulfur_thiol_cofactor", "free_amino_acid"),
    ("sterol_steroid", "acylglycerol"),
)


def load(frozen_root) -> pd.DataFrame:
    """The frozen per-molecule table: canonical id, fine class, broad class, provenance."""
    canon = pd.read_csv(frozen_root / "phase00/tables/canonical_analytes_v1.csv")
    part = pd.read_csv(frozen_root / "phase00/tables/chemical_partition_v1.csv")
    folds = pd.read_csv(frozen_root / "phase00/tables/cv_folds_v1.csv")
    df = (canon.merge(part[["canonical_id", "fine_class"]], on="canonical_id", how="left",
                      suffixes=("", "_p"))
               .merge(folds[["canonical_id", "fold"]], on="canonical_id", how="left"))
    if "fine_class_p" in df:
        df["fine_class"] = df["fine_class_p"].fillna(df["fine_class"])
        df = df.drop(columns=["fine_class_p"])
    return df


def build_registry(frozen_root, y: np.ndarray, cls: np.ndarray,
                   sources: np.ndarray | None = None,
                   excitations: np.ndarray | None = None) -> list[dict]:
    """One record per chemistry class, with everything the report must print explicitly."""
    df = load(frozen_root)
    by_mol = {r.canonical_id: r for r in df.itertuples()}
    adj = {}
    for a, b in ADJACENT:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    out = []
    for k, c in enumerate(CLASS_ORDER):
        sel = cls == c
        mols = sorted(set(y[sel]))
        rec = {
            "class_index": k,
            "class_id": c,
            "class_name": c.replace("_", " "),
            "broad_class": sorted({str(getattr(by_mol.get(m), "broad_class", "")) for m in mols}),
            "n_molecules": len(mols),
            "n_spectra": int(sel.sum()),
            "canonical_molecules": mols,
            "spectra_per_molecule": round(float(sel.sum() / max(len(mols), 1)), 3),
            "adjacent_classes": sorted(set(adj.get(c, []))),
        }
        if sources is not None:
            v, n = np.unique(sources[sel], return_counts=True)
            rec["source_distribution"] = dict(zip([str(x) for x in v], n.tolist()))
        if excitations is not None:
            v, n = np.unique(excitations[sel], return_counts=True)
            rec["excitation_distribution"] = dict(zip([str(x) for x in v], n.tolist()))
        reps = pd.Series(y[sel]).value_counts()
        rec["replicate_distribution"] = {str(int(k2)): int(v2)
                                         for k2, v2 in reps.value_counts().sort_index().items()}
        out.append(rec)
    tot_m = sum(r["n_molecules"] for r in out)
    tot_s = sum(r["n_spectra"] for r in out)
    for r in out:
        r["molecule_share"] = round(r["n_molecules"] / tot_m, 4)
        r["spectrum_share"] = round(r["n_spectra"] / tot_s, 4)
        r["imbalance_vs_uniform"] = round(r["n_spectra"] / (tot_s / len(CLASS_ORDER)), 3)
    return out


def check(cls: np.ndarray) -> None:
    """Abort conditions: the ontology in the data must be exactly the frozen one."""
    seen = set(cls.tolist())
    missing = set(CLASS_ORDER) - seen
    extra = seen - set(CLASS_ORDER)
    if missing or extra:
        raise ValueError(f"ontology mismatch — missing {sorted(missing)}, extra {sorted(extra)}")
    if len(CLASS_ORDER) != 16:
        raise ValueError(f"CLASS_ORDER has {len(CLASS_ORDER)} entries, expected 16")


def index_of(c: str) -> int:
    return CLASS_ORDER.index(c)


def one_hot(cls: np.ndarray) -> np.ndarray:
    Y = np.zeros((len(cls), len(CLASS_ORDER)))
    for i, c in enumerate(cls):
        Y[i, CLASS_ORDER.index(c)] = 1.0
    return Y
