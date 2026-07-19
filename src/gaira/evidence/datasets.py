"""GAIRA V5 Phase 2 Stage B — benchmark dataset object.

Wraps the FROZEN Phase-2 input manifest (reuses src/gaira/representation/datasets)
and exposes the per-spectrum fields Stage B needs: canonical analyte, modality,
source, acquisition domain, replicate group, condition, spectrum array, chemical
family (evaluation-only, curated), non-small-molecule flag, provenance, data role,
and matched-pair availability.

Does NOT invent metadata and does NOT infer chemistry from spectra. Read-only.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

from ..representation import datasets as _rep
from ..representation import centroids as _ct
from . import families

# analytes that are polymers/macromolecules, not single small molecules (from Stage A audit)
NON_SMALL_MOLECULE = {"dna", "rna", "albumin", "glycogen", "coenzyme a", "acetyl-coa"}

PREPROCS = _rep.PREPROCS
DEFAULT_PREPROC = "A2_asls_savgol_snv"   # Stage A best joint preprocessing


@dataclass
class StageBData:
    X: np.ndarray            # (n_spectra, n_bins) processed intensities on GRID
    grid: np.ndarray         # wavenumber axis
    meta: pd.DataFrame       # per-spectrum metadata (see build)
    preproc: str

    @property
    def matched_analytes(self):
        r = set(self.meta[self.meta.modality == "raman"].analyte)
        s = set(self.meta[self.meta.modality == "sers"].analyte)
        return sorted(r & s)

    def subset(self, mask):
        m = np.asarray(mask)
        return StageBData(self.X[m], self.grid, self.meta[m].reset_index(drop=True), self.preproc)

    def by_modality(self, modality):
        return self.subset((self.meta.modality == modality).values)

    def centroids(self, group_cols=("analyte", "modality", "source")):
        """Analyte×modality×source centroids (reuses Stage A; no cross-modality avg)."""
        return _ct.build_centroids(self.X, self.meta, group_cols=group_cols)


def build(preproc: str = DEFAULT_PREPROC) -> StageBData:
    rows, _ = _rep.build_phase2_input(preproc)
    X, meta = _rep.matrix(rows)
    meta = meta.copy()
    # replicate group = one technical-replicate set = (analyte, modality, source)
    meta["replicate_group"] = (meta.analyte + "|" + meta.modality + "|" + meta.source)
    # acquisition condition (grounding): source + modality (no perturbation conditions here)
    meta["acquisition_domain"] = meta.modality + "|" + meta.source
    meta["condition"] = meta.source
    meta["family"] = meta.analyte.map(families.family_of)
    meta["family_ambiguous"] = meta.analyte.map(families.is_ambiguous)
    meta["non_small_molecule"] = meta.analyte.isin(NON_SMALL_MOLECULE)
    meta["data_role"] = "grounding"
    # matched-pair availability
    r = set(meta[meta.modality == "raman"].analyte); s = set(meta[meta.modality == "sers"].analyte)
    matched = r & s
    meta["matched"] = meta.analyte.isin(matched)
    return StageBData(X, _rep.GRID, meta, preproc)


def dataset_card(d: StageBData) -> dict:
    """Corpus composition, missingness, imbalance, replicate + matched structure,
    generalization limits. Emitted to tables/stage_b_dataset_card.json."""
    m = d.meta
    matched = d.matched_analytes
    fam_counts = m.drop_duplicates("analyte").family.value_counts().to_dict()
    src_mod = m.groupby(["source", "modality"]).size().unstack(fill_value=0)
    reps = m.groupby("replicate_group").size()
    return {
        "preproc": d.preproc,
        "n_spectra": int(len(m)),
        "n_bins": int(d.X.shape[1]),
        "wavenumber_range": [float(d.grid.min()), float(d.grid.max())],
        "n_raman": int((m.modality == "raman").sum()),
        "n_sers": int((m.modality == "sers").sum()),
        "n_analytes": int(m.analyte.nunique()),
        "n_matched_analytes": len(matched),
        "matched_analytes": matched,
        "sources": {k: int(v) for k, v in m.source.value_counts().items()},
        "source_by_modality": {s: {mm: int(src_mod.loc[s, mm]) for mm in src_mod.columns}
                               for s in src_mod.index},
        "family_counts_by_analyte": fam_counts,
        "n_family_unknown": int(m.drop_duplicates("analyte").family.eq("unknown").sum()),
        "n_family_ambiguous_analytes": int(m.drop_duplicates("analyte").family_ambiguous.sum()),
        "non_small_molecule_analytes": sorted(set(m[m.non_small_molecule].analyte)),
        "replicate_group_size": {"min": int(reps.min()), "median": float(reps.median()),
                                 "max": int(reps.max()), "n_groups": int(reps.nunique() if False else reps.shape[0])},
        "missingness": {
            "outside_grid_fraction_mean": float(np.mean(~np.isfinite(d.X)) if not np.isfinite(d.X).all() else 0.0),
        },
        "imbalance_notes": [
            "Ag-SERS is single-source (Gobbato colloid) → no cross-source SERS split possible (Split D infeasible for SERS).",
            "Matched pairs are largely one instrument ecosystem (Gobbato) → cross-modal results are within-corpus, not observation-domain-invariant.",
            "Raman spans 2 sources (RamanBioLib 785 + Gobbato powders); SERS 1 source.",
        ],
        "generalization_limits": [
            "Held-out-analyte splits test chemical-family generalization, NOT exact-analyte retrieval (impossible for unseen identities).",
            "51 matched analytes is a feasibility scale, not a production-encoder scale (H7 high-risk).",
            "Technical replicates are not independent semantic samples; grouped splits enforced.",
        ],
    }
