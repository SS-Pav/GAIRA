"""GAIRA V5 Foundation — freeze/restore the biochemical reference space."""
from __future__ import annotations
import json, hashlib
from pathlib import Path
import numpy as np


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
