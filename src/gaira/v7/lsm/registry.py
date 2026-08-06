"""GAIRA V7 — the Local Spectral Motif registry (canonical, class-indexed).

Indexed by **chemistry class**, as `DATA_CONTRACTS.md` C-05 specifies — not by atlas
component. Rejected LSMs are kept with their reasons so "what did we throw away" is
answerable by query.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from .lsm import LSM

REGISTRY_SCHEMA = "gaira_v7_lsm_registry_v1"


class LSMRegistry:
    def __init__(self, results: list[dict], discovery_version: str, config: dict,
                 reference_arm: str):
        self.results = results
        self.discovery_version = discovery_version
        self.config = dict(config)
        self.reference_arm = reference_arm

    @property
    def lsms(self) -> list[LSM]:
        return [m for r in self.results for m in r.get("lsms", [])]

    @property
    def retained(self) -> list[LSM]:
        return [m for m in self.lsms if m.retained]

    @property
    def rejected(self) -> list[LSM]:
        return [m for m in self.lsms if not m.retained]

    def by_class(self, cls: str, retained_only: bool = True) -> list[LSM]:
        ms = [m for m in self.lsms if m.chemical_class == cls]
        return [m for m in ms if m.retained] if retained_only else ms

    def by_id(self, motif_id: str) -> LSM | None:
        return next((m for m in self.lsms if m.motif_id == motif_id), None)

    def by_type(self, lsm_type: str) -> list[LSM]:
        return [m for m in self.retained if m.lsm_type == lsm_type]

    def motif_table(self) -> pd.DataFrame:
        df = pd.DataFrame([m.to_record() for m in self.lsms])
        cols = ["motif_id", "chemical_class", "index_in_class", "k_c", "n_class_analytes",
                "retained", "rejection_reason", "is_anchor", "lsm_type", "n_bands",
                "band_centers_cm", "n_analytes", "n_spectra", "dominant_broad_class",
                "purity", "stability", "matched_similarity", "activation_share",
                "activation_sparsity", "reconstruction_share", "redundancy_max",
                "anchor_justification", "analytes"]
        return df[[c for c in cols if c in df.columns]].sort_values(
            ["chemical_class", "index_in_class"]).reset_index(drop=True)

    def class_table(self) -> pd.DataFrame:
        rows = []
        for r in self.results:
            kept = [m for m in r.get("lsms", []) if m.retained]
            rows.append({
                "chemical_class": r["chemical_class"], "status": r["status"],
                "n_analytes": r["n_analytes"], "n_spectra": r["n_spectra"],
                "k_ceiling": r["k_ceiling"], "k_c": r.get("k_c", 0),
                "n_retained_lsms": len(kept),
                "n_rejected_lsms": len(r.get("lsms", [])) - len(kept),
                "explained_variance": r.get("explained_variance"),
                "mean_stability": (round(float(np.mean([m.stability for m in kept])), 4)
                                   if kept else None),
                "mean_purity": (round(float(np.mean([m.purity for m in kept])), 4)
                                if kept else None),
                "n_class_shared": sum(m.lsm_type == "class_shared" for m in kept),
                "n_subfamily": sum(m.lsm_type == "subfamily" for m in kept),
                "n_discriminating": sum(m.lsm_type == "molecule_discriminating" for m in kept),
                "dominant_source": r.get("dominant_source", ""),
                "dominant_source_fraction": r.get("dominant_source_fraction"),
                "source_confounded": r.get("source_confounded", False),
                "reason": r.get("reason", ""),
            })
        return pd.DataFrame(rows).sort_values("n_analytes", ascending=False).reset_index(drop=True)

    def rejection_table(self) -> pd.DataFrame:
        rows = [{"motif_id": m.motif_id, "chemical_class": m.chemical_class,
                 "n_analytes": m.n_analytes, "stability": round(m.stability, 4),
                 "redundancy_max": round(m.redundancy_max, 4),
                 "reason_category": m.rejection_reason.split(" ")[0],
                 "rejection_reason": m.rejection_reason} for m in self.rejected]
        if not rows:
            return pd.DataFrame(columns=["motif_id", "chemical_class", "n_analytes",
                                         "stability", "redundancy_max", "reason_category",
                                         "rejection_reason"])
        return pd.DataFrame(rows).sort_values(["chemical_class", "motif_id"]).reset_index(drop=True)

    def dictionary(self) -> tuple[np.ndarray, list[str]]:
        kept = self.retained
        if not kept:
            return np.zeros((0, 676)), []
        return np.vstack([m.spectrum for m in kept]), [m.motif_id for m in kept]

    def check_integrity(self) -> list[str]:
        bad = []
        ids = [m.motif_id for m in self.lsms]
        if len(ids) != len(set(ids)):
            bad.append(f"duplicate motif ids: {[i for i, c in Counter(ids).items() if c > 1]}")
        classes = {r["chemical_class"] for r in self.results}
        for m in self.lsms:
            if m.chemical_class not in classes:
                bad.append(f"{m.motif_id}: class {m.chemical_class} not in results")
            bad.extend(f"{m.motif_id}: {v}" for v in m.validate())
        for r in self.results:
            kept = [m for m in r.get("lsms", []) if m.retained]
            if r["status"] == "DECOMPOSED":
                if not kept:
                    bad.append(f"{r['chemical_class']}: DECOMPOSED with no retained LSM")
                k = r.get("k_c", 0)
                if k > r["k_ceiling"]:
                    bad.append(f"{r['chemical_class']}: k_c {k} exceeds ceiling {r['k_ceiling']}")
            if r["status"] == "ANCHOR_ROUTE" and any(not m.is_anchor for m in r.get("lsms", [])):
                bad.append(f"{r['chemical_class']}: anchor route produced a non-anchor LSM")
        return bad

    def summary(self) -> dict:
        ct = self.class_table()
        kept = self.retained
        return {
            "schema": REGISTRY_SCHEMA,
            "discovery_version": self.discovery_version,
            "reference_arm": self.reference_arm,
            "n_classes": int(len(self.results)),
            "n_classes_decomposed": int((ct.status == "DECOMPOSED").sum()),
            "n_classes_anchor_route": int((ct.status == "ANCHOR_ROUTE").sum()),
            "n_classes_no_stable_lsm": int((ct.status == "NO_STABLE_LSM").sum()),
            "n_lsms_total": int(len(self.lsms)),
            "n_lsms_retained": int(len(kept)),
            "n_lsms_rejected": int(len(self.rejected)),
            "n_anchors": int(sum(m.is_anchor for m in kept)),
            "k_c_min": int(ct[ct.k_c > 0].k_c.min()) if (ct.k_c > 0).any() else 0,
            "k_c_max": int(ct.k_c.max()) if len(ct) else 0,
            "k_c_distinct_values": sorted(int(x) for x in ct[ct.k_c > 0].k_c.unique()),
            "type_counts": dict(Counter(m.lsm_type for m in kept)),
            "mean_stability": (round(float(np.mean([m.stability for m in kept])), 4)
                               if kept else None),
            "mean_purity": (round(float(np.mean([m.purity for m in kept])), 4)
                            if kept else None),
            "config": self.config,
        }
