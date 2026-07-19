"""I1 — adaptive spectral-region representation (§7).

Contiguous spectral regions defined from TRAINING data only (equal-variance-mass
segmentation): region boundaries are placed so each region carries roughly equal
cumulative training-set variance, giving finer regions where the corpus varies
most. Feature = mean intensity within each contiguous region. Regions are
inspectable and map directly back to [wn_lo, wn_hi]. No test labels used.
"""
from __future__ import annotations
import numpy as np
from .base import Representation


class RegionRepresentation(Representation):
    def __init__(self, grid, edges):
        super().__init__(name="I1_adaptive_regions", branch="interpretable", grid=grid,
                         modality_specific=False, params={"n_regions": len(edges) - 1})
        self.edges = np.asarray(edges)         # bin-index edges, length n_regions+1
        self.n_features = len(edges) - 1

    def transform(self, X, modality=None):
        X = np.nan_to_num(np.atleast_2d(X))
        feats = np.empty((X.shape[0], self.n_features))
        for r in range(self.n_features):
            a, b = self.edges[r], self.edges[r + 1]
            feats[:, r] = X[:, a:b].mean(axis=1)
        n = np.linalg.norm(feats, axis=1, keepdims=True)
        return feats / (n + 1e-12)

    def feature_wavenumbers(self):
        return [(float(self.grid[self.edges[r]]), float(self.grid[self.edges[r + 1] - 1]))
                for r in range(self.n_features)]


def fit_regions(X_train, grid, n_regions=32):
    """Equal-variance-mass contiguous segmentation on training data."""
    v = np.var(np.nan_to_num(X_train), axis=0)
    v = v + 1e-9
    cum = np.cumsum(v); cum /= cum[-1]
    targets = np.linspace(0, 1, n_regions + 1)
    edges = np.searchsorted(cum, targets[1:-1])
    edges = np.unique(np.concatenate([[0], edges, [len(grid)]]))
    # guard against empty regions
    while len(edges) - 1 < n_regions and len(edges) - 1 < len(grid):
        gaps = np.diff(edges)
        j = int(np.argmax(gaps))
        edges = np.unique(np.concatenate([edges, [edges[j] + gaps[j] // 2]]))
    return RegionRepresentation(grid, edges)
