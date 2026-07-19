"""I2 — multiscale / wavelet-like representation (§7).

Deterministic multiscale decomposition via a Gaussian filter bank (à-trous style):
at each scale, the difference of Gaussians captures local bands / shoulders / broad
features; coefficients are subsampled per scale. No pywt dependency. Each coefficient
maps back to a (center wavenumber, scale). Captures modest band shifts because
coarser scales are shift-tolerant. Not fitted (fixed scales) — training-independent
by construction, which is a valid interpretable baseline.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter1d
from .base import Representation


class WaveletRepresentation(Representation):
    def __init__(self, grid, scales, stride):
        super().__init__(name="I2_multiscale", branch="interpretable", grid=grid,
                         modality_specific=False, params={"scales": list(scales), "stride": stride})
        self.scales = list(scales)
        self.stride = stride
        self._centers = None
        self.n_features = None

    def _decompose(self, x):
        coeffs, centers = [], []
        prev = gaussian_filter1d(x, self.scales[0], mode="nearest")
        for sc in self.scales[1:]:
            cur = gaussian_filter1d(x, sc, mode="nearest")
            band = prev - cur                       # difference-of-Gaussians detail
            sub = band[:: self.stride]
            coeffs.append(sub)
            if self._centers is None:
                centers.append(np.array([(self.grid[min(i, len(self.grid) - 1)], sc)
                                         for i in range(0, len(x), self.stride)], dtype=object))
            prev = cur
        coeffs.append(prev[:: self.stride])          # coarse approximation
        if self._centers is None:
            centers.append(np.array([(self.grid[min(i, len(self.grid) - 1)], self.scales[-1])
                                     for i in range(0, len(x), self.stride)], dtype=object))
            self._centers = np.concatenate(centers)
        return np.concatenate(coeffs)

    def transform(self, X, modality=None):
        X = np.nan_to_num(np.atleast_2d(X))
        out = np.vstack([self._decompose(x) for x in X])
        self.n_features = out.shape[1]
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / (n + 1e-12)

    def feature_wavenumbers(self):
        return None if self._centers is None else [(float(c[0]), float(c[1])) for c in self._centers]


def fit_wavelets(X_train, grid, scales=(1, 2, 4, 8, 16, 32), stride=4):
    rep = WaveletRepresentation(grid, scales, stride)
    rep.transform(X_train[:1])   # initialize centers/n_features deterministically
    return rep
