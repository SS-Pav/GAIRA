"""GAIRA V7 — LSM registry serialisation (canonical, class-indexed)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .lsm import LSM
from .registry import LSMRegistry

SERIALIZATION_VERSION = "v7_lsm_serialization_v1"


def registry_fingerprint(registry: LSMRegistry) -> str:
    kept = sorted(registry.retained, key=lambda m: m.motif_id)
    h = hashlib.sha256()
    for m in kept:
        h.update(m.motif_id.encode("utf-8"))
        h.update(np.ascontiguousarray(m.spectrum, dtype=np.float64).tobytes())
    h.update(json.dumps(registry.config, sort_keys=True, separators=(",", ":")).encode())
    return h.hexdigest()[:32]


def save_registry(registry: LSMRegistry, out_dir) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, df in (("lsm_registry_v1.csv", registry.motif_table()),
                     ("lsm_classes_v1.csv", registry.class_table()),
                     ("lsm_rejections_v1.csv", registry.rejection_table())):
        p = out_dir / name
        df.to_csv(p, index=False, lineterminator="\n")
        written[name] = hashlib.sha256(p.read_bytes()).hexdigest()

    H, ids = registry.dictionary()
    p = out_dir / "lsm_dictionary_v1.npz"
    np.savez_compressed(p, H=np.ascontiguousarray(H, dtype=np.float64),
                        motif_ids=np.array(ids, dtype=object),
                        classes=np.array([registry.by_id(i).chemical_class for i in ids],
                                         dtype=object))
    written["lsm_dictionary_v1.npz"] = hashlib.sha256(p.read_bytes()).hexdigest()

    manifest = {"schema": SERIALIZATION_VERSION,
                "registry_fingerprint": registry_fingerprint(registry),
                "discovery_version": registry.discovery_version,
                "reference_arm": registry.reference_arm,
                "config": registry.config, "summary": registry.summary(),
                "files": written}
    (out_dir / "lsm_manifest_v1.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str) + "\n")
    return manifest


def load_registry(out_dir):
    out_dir = Path(out_dir)
    df = pd.read_csv(out_dir / "lsm_registry_v1.csv")
    z = np.load(out_dir / "lsm_dictionary_v1.npz", allow_pickle=True)
    man = json.loads((out_dir / "lsm_manifest_v1.json").read_text())
    return df, np.asarray(z["H"], float), [str(x) for x in z["motif_ids"]], man


def lsms_from_table(df: pd.DataFrame, H: np.ndarray, ids: list[str]) -> list[LSM]:
    pos = {m: i for i, m in enumerate(ids)}
    out = []
    for _, r in df[df.retained].iterrows():
        centers = [float(x) for x in str(r.band_centers_cm).split(";") if x]
        out.append(LSM(
            motif_id=r.motif_id, chemical_class=r.chemical_class,
            index_in_class=int(r.index_in_class), spectrum=H[pos[r.motif_id]],
            dominant_bands=[{"center_cm": c, "prominence": 0.0, "weight": 1.0 / max(len(centers), 1)}
                            for c in centers],
            analytes=[a for a in str(r.analytes).split(";") if a],
            n_analytes=int(r.n_analytes), n_spectra=int(r.n_spectra),
            activation_share=float(r.activation_share),
            activation_sparsity=float(r.activation_sparsity),
            stability=float(r.stability), matched_similarity=float(r.matched_similarity),
            purity=float(r.purity), reconstruction_share=float(r.reconstruction_share),
            redundancy_max=float(r.redundancy_max), lsm_type=str(r.lsm_type),
            k_c=int(r.k_c), n_class_analytes=int(r.n_class_analytes),
            dominant_broad_class=str(r.dominant_broad_class), retained=bool(r.retained),
            rejection_reason="", is_anchor=bool(r.is_anchor),
            anchor_justification=("" if pd.isna(r.anchor_justification)
                                  else str(r.anchor_justification))))
    return out
