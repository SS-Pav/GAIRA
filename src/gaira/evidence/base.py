"""Common interface for Stage B representations (interpretable and encoder).

A Representation is fitted on TRAINING data only, then transforms any spectra to a
feature/embedding matrix. Every representation must expose enough to map features
back toward wavenumber space (for interpretability) and to serialize itself.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Representation:
    name: str
    branch: str                       # "direct" | "interpretable" | "encoder" | "hybrid"
    grid: np.ndarray                  # wavenumber axis of the input spectra
    modality_specific: bool = False   # True if fitted per modality
    params: dict = field(default_factory=dict)

    def transform(self, X, modality=None):
        raise NotImplementedError

    def feature_wavenumbers(self):
        """Return, per feature, the representative wavenumber(s) it maps to (or None)."""
        return None

    def to_dict(self):
        return {"name": self.name, "branch": self.branch, "modality_specific": self.modality_specific,
                "params": self.params, "n_features": int(getattr(self, "n_features", -1))}


def l2norm(X, eps=1e-12):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (n + eps)
