"""GAIRA engine — Biochemical Ontology v2 + Component→Theme mapping (Parts 4-5).

Loads the curated theme ontology (definitions, bands, literature, caveats) and the
evidence-derived many-to-many component→theme weight matrix. Provides the weight
matrix W (k_components x n_themes) used by the BSV layer.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[3]
DATA = Path(__file__).resolve().parent / "data"
ARTIFACTS = REPO / "results/v5_rebuild/engine_v1/artifacts"

# themes that are NOT biochemistry — surfaced for honesty, excluded from radar axes
NON_BIOCHEMICAL_THEMES = {"background_matrix", "unknown_mixed"}


class Ontology:
    def __init__(self, onto_path=None, weights_path=None):
        onto_path = Path(onto_path) if onto_path else DATA / "biochemical_ontology_v2.yaml"
        weights_path = Path(weights_path) if weights_path else ARTIFACTS / "component_theme_weights_v1.json"
        self.raw = yaml.safe_load(onto_path.read_text())
        self.version = self.raw["version"]
        self.themes = {t["id"]: t for t in self.raw["themes"]}
        self.theme_ids = [t["id"] for t in self.raw["themes"]]
        self.biochemical_theme_ids = [t for t in self.theme_ids if t not in NON_BIOCHEMICAL_THEMES]
        self._wraw = json.loads(weights_path.read_text())
        self.weights_version = self._wraw["version"]
        self.atlas_fingerprint = self._wraw["atlas_fingerprint"]
        self.k = max(int(j) for j in self._wraw["weights"]) + 1
        self._W, self._evidence = self._build_matrix()

    def _build_matrix(self):
        W = np.zeros((self.k, len(self.theme_ids)))
        ti = {t: i for i, t in enumerate(self.theme_ids)}
        evidence = {}
        for j_str, entries in self._wraw["weights"].items():
            j = int(j_str)
            for e in entries:
                W[j, ti[e["theme"]]] = e["weight"]
                evidence[(j, e["theme"])] = e
        return W, evidence

    @property
    def W(self):
        """component→theme weight matrix (rows sum to 1)."""
        return self._W

    def theme_index(self, theme_id):
        return self.theme_ids.index(theme_id)

    def theme(self, theme_id):
        return self.themes[theme_id]

    def weight_evidence(self, component, theme_id):
        """The three evidence lines behind one component→theme weight."""
        return self._evidence.get((component, theme_id))

    def contributors(self, theme_id, top=6):
        """Components contributing most to a theme, with their weights."""
        ti = self.theme_index(theme_id)
        order = np.argsort(-self._W[:, ti])
        return [(int(j), float(self._W[j, ti])) for j in order if self._W[j, ti] > 0.02][:top]

    def literature(self, theme_id):
        return self.themes[theme_id].get("literature")
