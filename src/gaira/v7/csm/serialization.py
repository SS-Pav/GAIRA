"""GAIRA V7 — Phase 02: CSM serialisation (contract C-07).

Round-trip fidelity is a hard requirement: a CSM that does not deserialise to the object that
was validated is a different CSM, and the validation no longer applies to it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .csm import CSM
from .registry import CSMRegistry

SCHEMA = "v7_csm_serialization_v1"


def save(reg: CSMRegistry, out_dir: Path, method_selection: dict,
         graph_meta: dict, config: dict) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    D = reg.dictionary()
    np.savez_compressed(out_dir / "csm_dictionary_v1.npz", CSM=D,
                        csm_ids=np.array([c.csm_id for c in reg.csms]),
                        grid=graph_meta["grid"])
    payload = {
        "schema": SCHEMA,
        "registry_schema": "gaira_v7_csm_registry_v1",
        "atlas_build": reg.atlas_build,
        "M": len(reg.csms),
        "integration_method": reg.integration_method,
        "consensus_operator": reg.consensus_operator,
        "selected_threshold": reg.selected_threshold,
        "method_selection": method_selection,
        "fingerprint": reg.fingerprint(),
        "config": config,
        "summary": reg.summary(),
        "csms": [_csm_json(c) for c in reg.csms],
    }
    (out_dir / "csm_registry_v1.json").write_text(json.dumps(payload, indent=2))
    reg.table().to_csv(out_dir / "csm_registry_v1.csv", index=False)
    return payload


def _csm_json(c: CSM) -> dict:
    return {
        "csm_id": c.csm_id, "index": c.index,
        "contributing_lsms": [{"lsm_id": m, "weight": round(w, 4)}
                              for m, w in zip(c.contributing_lsms, c.contributing_lsm_weights)],
        "supporting_classes": c.supporting_classes,
        "supporting_analytes": c.supporting_analytes,
        "projected_support": c.projected_support,
        "n_lsms": c.n_lsms, "n_classes": c.n_classes, "n_analytes": c.n_analytes,
        "dominant_bands": [round(b, 1) for b in c.dominant_bands],
        "band_assignment": c.band_assignment,
        "interpretation": c.interpretation,
        "diagnostic_status": c.diagnostic_status,
        "cohesion": round(c.cohesion, 4), "uncertainty": round(c.uncertainty, 4),
        "bootstrap_confidence": _r(c.bootstrap_confidence),
        "ev_delta_vs_lsms": _r(c.ev_delta_vs_lsms),
        "loco_survival": _r(c.loco_survival),
        "source_robust": bool(c.source_robust),
        "mean_edge_weight": round(c.mean_edge_weight, 4),
        "min_edge_weight": round(c.min_edge_weight, 4),
        "max_external_weight": round(c.max_external_weight, 4),
        "min_coassignment": round(c.min_coassignment, 4),
        "lsm_types": c.lsm_types,
        "is_singleton": c.is_singleton, "is_anchored": c.is_anchored,
        "is_cross_class": c.is_cross_class,
        "anchor_justification": c.anchor_justification,
        "status": c.status, "rejection_reason": c.rejection_reason,
        "provenance": {
            "csm": c.csm_id,
            "lsms": c.contributing_lsms,
            "classes": c.supporting_classes,
            "canonical_molecules": c.supporting_analytes,
            "resolves_to_spectra_via": "phase00 canonical_molecule_registry -> spectrum_id",
        },
    }


def _r(v) -> float | None:
    import math
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), 4)


def load(out_dir: Path) -> tuple[np.ndarray, dict]:
    out_dir = Path(out_dir)
    z = np.load(out_dir / "csm_dictionary_v1.npz", allow_pickle=True)
    payload = json.loads((out_dir / "csm_registry_v1.json").read_text())
    return z["CSM"], payload


def save_graph(path: Path, nodes: list[dict], edges: list[dict], sweep: list[dict],
               selection: dict, feature_correlation: dict) -> None:
    """Contract C-06. The threshold sweep is part of the artefact, not part of the report:
    a graph shipped without its sweep cannot be audited for the arbitrariness of its cut."""
    Path(path).write_text(json.dumps({
        "schema": "lsm_graph_v1",
        "nodes": nodes, "edges": edges,
        "threshold_sweep": sweep,
        "selected_threshold": selection.get("selected_threshold"),
        "stable_region": selection.get("stable_region"),
        "selection_rationale": selection.get("rationale"),
        "feature_correlation": feature_correlation,
    }, indent=2))
