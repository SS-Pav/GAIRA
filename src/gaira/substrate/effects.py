"""Loader + typed accessor for substrate_effect_types_v1.yaml."""
from __future__ import annotations

from pathlib import Path

import yaml

from gaira.substrate.schema import SubstrateEffectType


# Global bounds on confidence multipliers (policy v1).
MULTIPLIER_MIN = 0.40
MULTIPLIER_MAX = 1.15


def load_effect_types(yaml_path: str | Path) -> dict[str, SubstrateEffectType]:
    """Load the effect-type vocabulary into an id → SubstrateEffectType dict."""
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Effect types YAML not found: {path}")
    doc = yaml.safe_load(path.read_text())
    if not doc or "effect_types" not in doc:
        raise ValueError(f"{path}: malformed YAML; expected top-level 'effect_types' list")

    out: dict[str, SubstrateEffectType] = {}
    for e in doc["effect_types"]:
        eid = e["id"]
        if eid in out:
            raise ValueError(f"Duplicate effect type id: {eid}")
        m = float(e["confidence_multiplier"])
        if not (MULTIPLIER_MIN <= m <= MULTIPLIER_MAX):
            raise ValueError(
                f"Effect-type {eid}: confidence_multiplier {m} outside "
                f"[{MULTIPLIER_MIN}, {MULTIPLIER_MAX}]"
            )
        out[eid] = SubstrateEffectType(
            id=eid,
            display=e.get("display", eid),
            semantics=str(e.get("semantics", "")).strip(),
            confidence_multiplier=m,
            caution_flag=bool(e.get("caution_flag", True)),
            bias_direction=str(e.get("bias_direction", "none")),
            interpretation_note=str(e.get("interpretation_note", "")).strip(),
        )
    return out
