"""E4 — hybrid representation (§14): encoder embedding + sparse interpretable
evidence activations, concatenated in one L2-normalized feature vector.

Exploratory: only justified if it yields a reproducible held-out Pareto improvement
over both branches alone. Kept simple and transparent (concatenation with a weight),
so the interpretable half remains auditable.
"""
from __future__ import annotations
import numpy as np
from .base import Representation, l2norm


class HybridRepresentation(Representation):
    def __init__(self, encoder_rep, interp_rep, w_interp=1.0):
        grid = encoder_rep.grid
        super().__init__(name=f"E4_hybrid[{encoder_rep.name}+{interp_rep.name}]",
                         branch="hybrid", grid=grid,
                         modality_specific=encoder_rep.modality_specific,
                         params={"encoder": encoder_rep.name, "interpretable": interp_rep.name,
                                 "w_interp": w_interp})
        self.enc = encoder_rep
        self.interp = interp_rep
        self.w_interp = w_interp
        self.n_features = int(encoder_rep.n_features) + int(interp_rep.n_features)

    def transform(self, X, modality=None):
        ze = l2norm(self.enc.transform(X, modality))
        zi = l2norm(self.interp.transform(X, modality)) * self.w_interp
        return l2norm(np.hstack([ze, zi]))
