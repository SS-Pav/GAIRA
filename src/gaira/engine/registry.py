"""GAIRA engine — Component Registry v1 loader (Part 3).

Loads the structured registry of the 24 latent Raman components (generated from the
frozen atlas + Component Audit + Perturbation Response Audit). Read-only; every
field carries provenance.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from . import paths

REPO = paths.REPO
ARTIFACTS = paths.LEGACY_ENGINE
FROZEN = paths.LEGACY_FOUNDATION


class ComponentRegistry:
    def __init__(self, path=None):
        path = Path(path) if path else paths.frozen_file("component_registry_v1.json")
        self.raw = json.loads(path.read_text())
        self.fingerprint = self.raw["atlas_fingerprint"]
        self.k = self.raw["n_components"]
        self.components = {c["component_id"]["value"]: c for c in self.raw["components"]}
        self._basis = None

    def get(self, j):
        return self.components[j]

    def field(self, j, name):
        """Return (value, provenance) for a component field."""
        f = self.components[j][name]
        return f["value"], f["provenance"]

    def value(self, j, name):
        return self.components[j][name]["value"]

    def stability(self, j):
        return self.value(j, "bootstrap_stability")

    def basis(self):
        """Frozen basis spectra (lazily loaded, never modified)."""
        if self._basis is None:
            npz = np.load(paths.foundation_dir() / "manifold_components.npz")
            self._basis = (npz["components"], npz["grid"])
        return self._basis

    def summary_table(self):
        import pandas as pd
        rows = []
        for j, c in sorted(self.components.items()):
            rows.append({"component": j,
                         "interpretation": c["current_interpretation"]["value"][:60],
                         "confidence": c["interpretation_confidence"]["value"],
                         "stability": c["bootstrap_stability"]["value"],
                         "purity": c["purity"]["value"],
                         "audit_label": c["audit_label_v0_1"]["value"],
                         "n_dose_responsive": c["n_dose_experiments_responsive"]["value"]})
        return pd.DataFrame(rows)
