"""GAIRA V7 — motif registry serialisation.

Canonicalised so that a registry written on two machines hashes identically:
float64 C-contiguous arrays, JSON with sorted keys and no insignificant whitespace,
CSV with fixed column order and LF line endings.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .motif import LSM
from .registry import LSMRegistry

SERIALIZATION_VERSION = "v7_lsm_serialization_v1"


def _canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def registry_fingerprint(registry: LSMRegistry) -> str:
    """Content hash over the retained motif spectra and their identities.

    Covers the objects that determine downstream behaviour; ordering is fixed by motif id
    so the hash is stable.
    """
    kept = sorted(registry.retained, key=lambda m: m.motif_id)
    h = hashlib.sha256()
    for m in kept:
        h.update(m.motif_id.encode("utf-8"))
        h.update(np.ascontiguousarray(m.spectrum, dtype=np.float64).tobytes())
    h.update(_canonical_json(registry.config).encode("utf-8"))
    return h.hexdigest()[:32]


def save_registry(registry: LSMRegistry, out_dir: Path) -> dict:
    """Write the registry as tables + a dense spectra array + a manifest."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}

    for name, df in (("lsm_registry_v1.csv", registry.motif_table()),
                     ("lsm_components_v1.csv", registry.component_table()),
                     ("lsm_rejections_v1.csv", registry.rejection_table())):
        p = out_dir / name
        df.to_csv(p, index=False, lineterminator="\n")
        written[name] = hashlib.sha256(p.read_bytes()).hexdigest()

    S, ids = registry.spectra_matrix()
    p = out_dir / "lsm_spectra_v1.npz"
    np.savez_compressed(p, spectra=np.ascontiguousarray(S, dtype=np.float64),
                        motif_ids=np.array(ids, dtype=object))
    written["lsm_spectra_v1.npz"] = hashlib.sha256(p.read_bytes()).hexdigest()

    manifest = {
        "schema": SERIALIZATION_VERSION,
        "registry_fingerprint": registry_fingerprint(registry),
        "atlas_fingerprint": registry.atlas_fingerprint,
        "discovery_version": registry.discovery_version,
        "config": registry.config,
        "summary": registry.summary(),
        "files": written,
    }
    p = out_dir / "lsm_manifest_v1.json"
    p.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def load_registry(out_dir: Path) -> tuple[pd.DataFrame, np.ndarray, list[str], dict]:
    """Round-trip read. Returns (motif table, spectra, motif ids, manifest)."""
    out_dir = Path(out_dir)
    df = pd.read_csv(out_dir / "lsm_registry_v1.csv")
    z = np.load(out_dir / "lsm_spectra_v1.npz", allow_pickle=True)
    manifest = json.loads((out_dir / "lsm_manifest_v1.json").read_text())
    return df, np.asarray(z["spectra"], float), [str(x) for x in z["motif_ids"]], manifest


def motifs_from_table(df: pd.DataFrame, spectra: np.ndarray, ids: list[str]) -> list[LSM]:
    """Rebuild retained LSM objects from a serialised registry (round-trip check)."""
    pos = {mid: i for i, mid in enumerate(ids)}
    out = []
    for _, r in df[df.retained].iterrows():
        out.append(LSM(
            motif_id=r.motif_id, parent_component=int(r.parent_component),
            index_in_component=int(r.index_in_component),
            spectrum=spectra[pos[r.motif_id]],
            band_indices=[int(x) for x in str(r.band_indices).split(";") if x],
            band_centers_cm=[float(x) for x in str(r.band_centers_cm).split(";") if x],
            band_weights=[float(x) for x in str(r.band_weights).split(";") if x],
            analytes=[x for x in str(r.analytes).split(";") if x],
            n_analytes=int(r.n_analytes), n_spectra=int(r.n_spectra),
            fine_classes=_parse_counts(r.fine_classes),
            broad_classes=_parse_counts(r.broad_classes),
            sources=_parse_counts(r.sources),
            stability=float(r.stability), purity=float(r.purity),
            coverage_analytes=float(r.coverage_analytes),
            coverage_spectra=float(r.coverage_spectra),
            dominant_class=str(r.dominant_class), band_fidelity=float(r.band_fidelity),
            redundancy_max=float(r.redundancy_max), retained=bool(r.retained),
            rejection_reason="" if pd.isna(r.rejection_reason) else str(r.rejection_reason),
        ))
    return out


def _parse_counts(s) -> dict:
    if not isinstance(s, str) or not s:
        return {}
    out = {}
    for part in s.split(";"):
        if ":" in part:
            k, v = part.rsplit(":", 1)
            out[k] = int(v)
    return out
