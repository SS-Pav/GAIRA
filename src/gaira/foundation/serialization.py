"""GAIRA V5 Foundation — freeze/restore the biochemical reference space."""
from __future__ import annotations
import json, hashlib
from pathlib import Path
import numpy as np


class FrozenAtlas:
    """A frozen reference atlas: fixed non-negative components, projection only.

    Projection uses NMF with the dictionary HELD FIXED (update_H=False), i.e. a
    non-negative least-squares fit of each spectrum onto the frozen components.
    Nothing here can refit or alter the atlas.
    """

    def __init__(self, components, grid, meta):
        self.components = np.asarray(components, float)
        self.grid = np.asarray(grid, float)
        self.meta = meta
        self.k = int(self.components.shape[0])
        self.name = meta.get("representation", "NMF")

    def project(self, X):
        from sklearn.decomposition import non_negative_factorization
        Xc = np.clip(np.nan_to_num(np.atleast_2d(X)), 0, None)
        W, _, _ = non_negative_factorization(
            Xc, H=self.components, n_components=self.k,
            update_H=False, init="custom", solver="cd", max_iter=500, random_state=0)
        return W

    def coordinates(self, X, normalise=True):
        Z = np.clip(self.project(X), 0, None)
        if normalise:
            s = Z.sum(axis=1, keepdims=True)
            Z = np.divide(Z, s, out=np.zeros_like(Z), where=s > 1e-12)
        return Z

    def reconstruct(self, X):
        return self.project(X) @ self.components


def load_frozen_manifold(in_dir) -> FrozenAtlas:
    """Load a frozen atlas from disk and verify its fingerprint."""
    p = Path(in_dir)
    npz = np.load(p / "manifold_components.npz")
    meta = json.loads((p / "manifold.json").read_text())
    comps = npz["components"]
    fp = hashlib.sha256(np.ascontiguousarray(comps).tobytes()).hexdigest()[:32]
    if meta.get("fingerprint") and fp != meta["fingerprint"]:
        raise ValueError(f"atlas fingerprint mismatch: {fp} != {meta['fingerprint']}")
    return FrozenAtlas(comps, npz["grid"], meta)


def freeze_manifold(manifold, out_dir, corpus_card=None, extra=None):
    p = Path(out_dir); p.mkdir(parents=True, exist_ok=True)
    np.savez(p / "manifold_components.npz",
             components=manifold.rep.components_,
             grid=manifold.grid,
             mean=(manifold.rep.mean_ if manifold.rep.mean_ is not None
                   else np.zeros(manifold.grid.shape)))
    fp = hashlib.sha256(np.ascontiguousarray(
        manifold.rep.components_).tobytes()).hexdigest()[:32]
    meta = {"representation": manifold.name, "k": int(manifold.k),
            "fingerprint": fp, "grid_min": float(manifold.grid.min()),
            "grid_max": float(manifold.grid.max()), "n_bins": int(len(manifold.grid)),
            "stats": manifold.stats, "corpus_card": corpus_card, **(extra or {})}
    (p / "manifold.json").write_text(json.dumps(meta, indent=2, default=str))
    return fp
