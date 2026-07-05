"""Loader + typed accessor for substrate_families_v1.yaml."""
from __future__ import annotations

from pathlib import Path

import yaml

from gaira.substrate.schema import SubstrateFamily


def load_families(yaml_path: str | Path) -> dict[str, SubstrateFamily]:
    """Load the family vocabulary into an id → SubstrateFamily dict."""
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Substrate families YAML not found: {path}")
    doc = yaml.safe_load(path.read_text())
    if not doc or "families" not in doc:
        raise ValueError(f"{path}: malformed YAML; expected top-level 'families' list")

    out: dict[str, SubstrateFamily] = {}
    for f in doc["families"]:
        fid = f["id"]
        if fid in out:
            raise ValueError(f"Duplicate substrate family id: {fid}")
        out[fid] = SubstrateFamily(
            id=fid,
            display=f.get("display", fid),
            metal=str(f.get("metal", "unknown")),
            geometry_class=str(f.get("geometry_class", "unknown")),
            fabrication_class=str(f.get("fabrication_class", "unknown")),
            biofluid_use_common=bool(f.get("biofluid_use_common", False)),
            typical_excitation_nm=tuple(f.get("typical_excitation_nm") or ()),
            typical_capping_or_surface_chemistry=tuple(
                f.get("typical_capping_or_surface_chemistry") or ()
            ),
            known_strengths=tuple(f.get("known_strengths") or ()),
            known_weaknesses=tuple(f.get("known_weaknesses") or ()),
            notes=str(f.get("notes", "")).strip(),
        )
    return out
