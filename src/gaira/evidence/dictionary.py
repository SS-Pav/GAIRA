"""I3 — sparse dictionary representation (§7).

Learns a SHARED spectral dictionary on joint training data (both modalities) so
sparse activation codes live in one comparable space — this is precisely the
shared-evidence hypothesis under test. Atoms are spectral shapes that map back to
wavenumber space. Fitted on training data only. (A modality-specific + matched-analyte
atom-alignment variant is noted as future work; the shared dictionary is the
comparable, tractable benchmark form.)
"""
from __future__ import annotations
import numpy as np
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.preprocessing import normalize
from .base import Representation


class DictionaryRepresentation(Representation):
    def __init__(self, grid, atoms, alpha):
        super().__init__(name="I3_sparse_dictionary", branch="interpretable", grid=grid,
                         modality_specific=False, params={"n_atoms": atoms.shape[0], "alpha": alpha})
        self.atoms = atoms            # (n_atoms, n_bins)
        self.alpha = alpha
        self.n_features = atoms.shape[0]

    def transform(self, X, modality=None):
        from sklearn.decomposition import sparse_encode
        X = np.nan_to_num(np.atleast_2d(X))
        codes = sparse_encode(X, self.atoms, algorithm="lasso_lars", alpha=self.alpha)
        n = np.linalg.norm(codes, axis=1, keepdims=True)
        return codes / (n + 1e-12)

    def feature_wavenumbers(self):
        # per atom: peak wavenumber of |atom|
        return [float(self.grid[int(np.argmax(np.abs(a)))]) for a in self.atoms]


def fit_dictionary(X_train, grid, n_atoms=24, alpha=1.0, seed=0):
    Xt = normalize(np.nan_to_num(X_train))
    dl = MiniBatchDictionaryLearning(n_components=n_atoms, alpha=alpha, max_iter=200,
                                     random_state=seed, transform_algorithm="lasso_lars",
                                     transform_alpha=alpha)
    dl.fit(Xt)
    return DictionaryRepresentation(grid, dl.components_, alpha)
