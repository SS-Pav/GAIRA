"""GAIRA V7 — the Local Spectral Motif registry.

The registry is the queryable index of the motif layer: every motif, its parent atlas
component, its spectral support, its participating molecules and classes, its scores and —
for rejected motifs — the deterministic reason it was rejected.

Rejected motifs are KEPT in the registry. "What did we throw away, and why" must be
answerable by query rather than by re-running the pipeline.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from .motif import LSM

REGISTRY_SCHEMA = "gaira_v7_lsm_registry_v1"


class LSMRegistry:
    """All motifs of all components, with the component-level discovery record."""

    def __init__(self, results: list[dict], atlas_fingerprint: str,
                 discovery_version: str, config: dict):
        self.results = results
        self.atlas_fingerprint = atlas_fingerprint
        self.discovery_version = discovery_version
        self.config = dict(config)

    # ── access ────────────────────────────────────────────────────────────────
    @property
    def motifs(self) -> list[LSM]:
        return [m for r in self.results for m in r.get("motifs", [])]

    @property
    def retained(self) -> list[LSM]:
        return [m for m in self.motifs if m.retained]

    @property
    def rejected(self) -> list[LSM]:
        return [m for m in self.motifs if not m.retained]

    def by_id(self, motif_id: str) -> LSM | None:
        return next((m for m in self.motifs if m.motif_id == motif_id), None)

    def by_component(self, k: int, retained_only: bool = True) -> list[LSM]:
        ms = [m for m in self.motifs if m.parent_component == k]
        return [m for m in ms if m.retained] if retained_only else ms

    def by_analyte(self, canonical_id: str) -> list[LSM]:
        return [m for m in self.retained if canonical_id in m.analytes]

    def by_class(self, fine_class: str) -> list[LSM]:
        return [m for m in self.retained if m.dominant_class == fine_class]

    # ── frames ────────────────────────────────────────────────────────────────
    def motif_table(self) -> pd.DataFrame:
        rows = [m.to_record() for m in self.motifs]
        cols = ["motif_id", "parent_component", "index_in_component", "retained",
                "rejection_reason", "n_bands", "band_centers_cm", "n_analytes", "n_spectra",
                "dominant_class", "purity", "stability", "coverage_analytes",
                "coverage_spectra", "band_fidelity", "redundancy_max", "fine_classes",
                "broad_classes", "sources", "analytes", "band_indices", "band_weights"]
        df = pd.DataFrame(rows)
        return df[[c for c in cols if c in df.columns]].sort_values(
            ["parent_component", "index_in_component"]).reset_index(drop=True)

    def component_table(self) -> pd.DataFrame:
        rows = []
        for r in self.results:
            ms = r.get("motifs", [])
            kept = [m for m in ms if m.retained]
            rows.append({
                "component": r["component"], "status": r["status"],
                "n_bands": r["n_bands"], "n_participants": r["n_participants"],
                "selected_n_motifs": r.get("selected_n_motifs", 1),
                "n_retained_motifs": len(kept), "n_rejected_motifs": len(ms) - len(kept),
                "silhouette": r.get("silhouette"), "size_gini": r.get("size_gini"),
                "max_motif_share": r.get("max_motif_share"),
                "mean_purity": (round(float(np.mean([m.purity for m in kept])), 4)
                                if kept else None),
                "mean_stability": (round(float(np.mean([m.stability for m in kept])), 4)
                                   if kept else None),
                "coverage_retained": round(float(sum(m.coverage_analytes for m in kept)), 4),
                "dominant_classes": ";".join(sorted({m.dominant_class for m in kept})),
                "reason": r.get("reason", ""),
            })
        return pd.DataFrame(rows).sort_values("component").reset_index(drop=True)

    def rejection_table(self) -> pd.DataFrame:
        rows = [{"motif_id": m.motif_id, "parent_component": m.parent_component,
                 "n_analytes": m.n_analytes, "n_bands": m.n_bands,
                 "stability": round(m.stability, 4),
                 "redundancy_max": round(m.redundancy_max, 4),
                 "reason_category": m.rejection_reason.split(" ")[0],
                 "rejection_reason": m.rejection_reason} for m in self.rejected]
        if not rows:
            return pd.DataFrame(columns=["motif_id", "parent_component", "n_analytes",
                                         "n_bands", "stability", "redundancy_max",
                                         "reason_category", "rejection_reason"])
        return (pd.DataFrame(rows).sort_values(["parent_component", "motif_id"])
                .reset_index(drop=True))

    def participation_matrix(self, analyte_ids: list[str]) -> pd.DataFrame:
        """Binary analyte x retained-motif participation."""
        kept = self.retained
        M = np.zeros((len(analyte_ids), len(kept)), dtype=int)
        pos = {a: i for i, a in enumerate(analyte_ids)}
        for j, m in enumerate(kept):
            for a in m.analytes:
                if a in pos:
                    M[pos[a], j] = 1
        return pd.DataFrame(M, index=analyte_ids, columns=[m.motif_id for m in kept])

    def spectra_matrix(self) -> tuple[np.ndarray, list[str]]:
        kept = self.retained
        if not kept:
            return np.zeros((0, 676)), []
        return np.vstack([m.spectrum for m in kept]), [m.motif_id for m in kept]

    # ── integrity ─────────────────────────────────────────────────────────────
    def check_integrity(self) -> list[str]:
        """Registry-level invariants. Empty list = intact."""
        bad = []
        ids = [m.motif_id for m in self.motifs]
        if len(ids) != len(set(ids)):
            dup = [i for i, c in Counter(ids).items() if c > 1]
            bad.append(f"duplicate motif ids: {dup}")
        comps = {r["component"] for r in self.results}
        for m in self.motifs:
            if m.parent_component not in comps:
                bad.append(f"{m.motif_id}: parent {m.parent_component} not in results")
            if not m.motif_id.startswith(f"c{m.parent_component:02d}."):
                bad.append(f"{m.motif_id}: id does not encode its parent component")
            bad.extend(f"{m.motif_id}: {v}" for v in m.validate())
        for r in self.results:
            kept = [m for m in r.get("motifs", []) if m.retained]
            if r["status"] == "DECOMPOSED" and len(kept) < 2:
                bad.append(f"component {r['component']}: DECOMPOSED with {len(kept)} motifs")
            if r["status"] == "IRREDUCIBLE" and len(kept) >= 2:
                bad.append(f"component {r['component']}: IRREDUCIBLE with {len(kept)} motifs")
        return bad

    def summary(self) -> dict:
        comp = self.component_table()
        kept = self.retained
        return {
            "schema": REGISTRY_SCHEMA,
            "discovery_version": self.discovery_version,
            "atlas_fingerprint": self.atlas_fingerprint,
            "n_components": int(len(self.results)),
            "n_components_decomposed": int((comp.status == "DECOMPOSED").sum()),
            "n_components_irreducible": int((comp.status == "IRREDUCIBLE").sum()),
            "n_components_not_analysable": int((comp.status == "NOT_ANALYSABLE").sum()),
            "n_motifs_total": int(len(self.motifs)),
            "n_motifs_retained": int(len(kept)),
            "n_motifs_rejected": int(len(self.rejected)),
            "motifs_per_component_mean": round(float(len(kept) / max(len(self.results), 1)), 3),
            "motifs_per_component_max": (int(comp.n_retained_motifs.max())
                                         if len(comp) else 0),
            "mean_purity": (round(float(np.mean([m.purity for m in kept])), 4)
                            if kept else None),
            "mean_stability": (round(float(np.mean([m.stability for m in kept])), 4)
                               if kept else None),
            "mean_coverage_analytes": (round(float(np.mean([m.coverage_analytes
                                                            for m in kept])), 4)
                                       if kept else None),
            "max_redundancy": round(float(max((m.redundancy_max for m in kept), default=0.0)), 4),
            "config": self.config,
        }
