"""GAIRA V7 Phase 00 — benchmark lock: frozen-asset verification and the V5 control.

Three levels of verification, weakest to strongest:

  1. DECLARED   the fingerprint recorded in MANIFEST.json / manifold.json
  2. RECOMPUTED the fingerprint recomputed from the basis array itself, plus the SHA-256 of
                every frozen file re-checked against the manifest
  3. REBUILT    the basis REFITTED from the raw corpus through canonical preprocessing and
                NMF(k=24, seed=0), and compared element-by-element to the frozen basis

Level 3 is the real benchmark lock. It proves that the V7 corpus loader, the canonical
preprocessing chain and the frozen atlas are the same object — not merely that a hash
string was copied correctly. It requires the raw root; without it the lock degrades to
level 2 and that is recorded rather than glossed.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import v7_paths as P


def verify_declared() -> list[dict]:
    out = []
    man = json.loads((P.FOUNDATION / "MANIFEST.json").read_text())
    meta = json.loads((P.FOUNDATION / "manifold.json").read_text())
    for item, got in (
        ("MANIFEST.atlas_fingerprint", man.get("atlas_fingerprint")),
        ("MANIFEST.versions.atlas_fingerprint", man.get("versions", {}).get("atlas_fingerprint")),
        ("manifold.fingerprint", meta.get("fingerprint")),
    ):
        out.append({"check": item, "expected": P.CANONICAL_ATLAS_FINGERPRINT, "got": got,
                    "status": "PASS" if got == P.CANONICAL_ATLAS_FINGERPRINT else "FAIL"})
    out.append({"check": "manifold.k", "expected": 24, "got": meta.get("k"),
                "status": "PASS" if meta.get("k") == 24 else "FAIL"})
    pp = man.get("preprocessing", {})
    out.append({"check": "preprocessing.window_cm", "expected": list(P.WINDOW_CM),
                "got": pp.get("window_cm"),
                "status": "PASS" if pp.get("window_cm") == list(P.WINDOW_CM) else "FAIL"})
    out.append({"check": "preprocessing.grid_step_cm", "expected": P.GRID_STEP_CM,
                "got": pp.get("grid_step_cm"),
                "status": "PASS" if pp.get("grid_step_cm") == P.GRID_STEP_CM else "FAIL"})
    out.append({"check": "preprocessing.pipeline", "expected": dict(P.PREPROC),
                "got": pp.get("pipeline"),
                "status": "PASS" if pp.get("pipeline") == dict(P.PREPROC) else "FAIL"})
    return out


def verify_recomputed() -> tuple[list[dict], np.ndarray]:
    out = []
    z = np.load(P.FOUNDATION / "manifold_components.npz")
    H = np.asarray(z["components"], float)
    out.append({"check": "basis.shape", "expected": "(24, 676)", "got": str(H.shape),
                "status": "PASS" if H.shape == (24, 676) else "FAIL"})
    fp = P.sha256_array(H)
    out.append({"check": "basis.fingerprint_recomputed",
                "expected": P.CANONICAL_ATLAS_FINGERPRINT, "got": fp,
                "status": "PASS" if fp == P.CANONICAL_ATLAS_FINGERPRINT else "FAIL"})
    out.append({"check": "basis.nonnegative", "expected": True, "got": bool((H >= 0).all()),
                "status": "PASS" if (H >= 0).all() else "FAIL"})

    man = json.loads((P.FOUNDATION / "MANIFEST.json").read_text())
    for name, rec in sorted(man["files"].items()):
        p = P.FOUNDATION / name
        got = P.sha256_file(p) if p.is_file() else "MISSING"
        out.append({"check": f"file_sha256.{name}", "expected": rec["sha256"], "got": got,
                    "status": "PASS" if got == rec["sha256"] else "FAIL"})
    return out, H


def verify_rebuilt(X: np.ndarray, H_frozen: np.ndarray) -> list[dict]:
    """Refit NMF from the corpus and compare to the frozen basis element-wise."""
    P.add_src_to_path()
    from gaira.foundation.representation import fit_nmf              # noqa: E402

    model = fit_nmf(X, 24, seed=0)
    H = np.asarray(model.components_, float)
    fp = P.sha256_array(H)
    same_shape = H.shape == H_frozen.shape
    max_abs = float(np.max(np.abs(H - H_frozen))) if same_shape else float("nan")
    cos = (float(np.mean([np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
                          for a, b in zip(H, H_frozen)])) if same_shape else float("nan"))
    return [
        {"check": "rebuild.shape", "expected": str(H_frozen.shape), "got": str(H.shape),
         "status": "PASS" if same_shape else "FAIL"},
        {"check": "rebuild.fingerprint", "expected": P.CANONICAL_ATLAS_FINGERPRINT, "got": fp,
         "status": "PASS" if fp == P.CANONICAL_ATLAS_FINGERPRINT else "FAIL"},
        {"check": "rebuild.max_abs_difference", "expected": 0.0, "got": max_abs,
         "status": "PASS" if same_shape and max_abs == 0.0 else "FAIL"},
        {"check": "rebuild.mean_rowwise_cosine", "expected": 1.0, "got": round(cos, 12),
         "status": "PASS" if same_shape and cos > 1 - 1e-9 else "FAIL"},
    ]


def frozen_dependency_graph() -> pd.DataFrame:
    """Which frozen assets Phase 00 reads, and what depends on each. Read-only, always."""
    rows = [
        ("assets/foundation/manifold_components.npz", "NMF basis H (24x676)",
         "atlas fingerprint; V5 control representation", "READ"),
        ("assets/foundation/manifold.json", "frozen metadata + corpus card",
         "corpus card cross-check; preprocessing spec", "READ"),
        ("assets/foundation/MANIFEST.json", "fingerprint + per-file SHA-256",
         "frozen-asset integrity check", "READ"),
        ("assets/foundation/component_registry_v1.json", "per-component provenance",
         "baseline purity/stability statistics", "READ"),
        ("assets/foundation/component_theme_weights_v1.json", "component->theme weights",
         "not used in Phase 00", "READ"),
        ("assets/foundation/biochemical_ontology_v2.yaml", "13 V5 themes",
         "not used in Phase 00", "READ"),
        ("assets/foundation/mss_motifs_v1.yaml", "13 legacy MSS motifs",
         "not used in Phase 00", "READ"),
        ("assets/foundation/reference_normalization_v1.json", "reference frame",
         "not used in Phase 00", "READ"),
        ("assets/foundation/reference_support.npz", "OOD support",
         "not used in Phase 00", "READ"),
        ("results/v6_rebuild/semantic_validation/artifacts/sv_reps.npz",
         "per-analyte representations at 5 levels",
         "V5 control baseline under the V7 harness", "READ"),
        ("results/v6_rebuild/v63_ontology_revalidation/tables/v63_analyte_audit.csv",
         "V6.3 ontology + declared duplicates",
         "frozen partition; alias audit", "READ"),
        ("src/gaira/preprocessing/pipeline.py", "canonical preprocessing primitives",
         "corpus load; NMF rebuild", "READ"),
        ("src/gaira/foundation/representation.py", "NMF fitter (seed 0, nndsvda, 1500 it)",
         "NMF rebuild", "READ"),
        ("src/gaira/data/*", "RamanBioLib / Gobbato loaders + synonyms",
         "corpus load", "READ"),
    ]
    return pd.DataFrame(rows, columns=["frozen_asset", "what_it_is",
                                       "phase00_dependency", "access"])
