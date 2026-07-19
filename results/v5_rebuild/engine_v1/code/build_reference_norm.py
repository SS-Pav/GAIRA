"""Build reference_normalization_v1 — the canonical reference frame.

The old radar normalised against COHORT means (circular, obsolete). The engine
normalises against the FROZEN Raman Reference Atlas: the distribution of the 375
pure-analyte reference spectra in the 24-component coordinate space. This gives a
fixed, cohort-independent 'centre' and 'spread' plus an out-of-distribution frame.
Robust statistics (median / MAD) so single reference outliers do not shift it.
"""
from __future__ import annotations
import sys, json, hashlib
from pathlib import Path
import numpy as np

REPO = Path("/Users/surajpg/projects/GAIRA"); sys.path.insert(0, str(REPO / "src"))
from gaira.foundation import serialization as SER, dataset as DS
OUT = REPO / "results/v5_rebuild/engine_v1/artifacts"
FROZEN = REPO / "results/v5_rebuild/foundation/artifacts"
K = 24


def main():
    atlas = SER.load_frozen_manifold(FROZEN)
    corpus = DS.load_reference_corpus()
    Z = atlas.coordinates(corpus.X, normalise=True)          # L1 share per component
    Zn = np.clip(np.nan_to_num(Z), 0, None)
    center = np.median(Zn, axis=0)
    spread = 1.4826 * np.median(np.abs(Zn - center), axis=0)
    spread = np.maximum(spread, 1e-3)                          # floor to avoid blow-up
    # reference support for OOD: unit vectors of the reference cloud
    U = Zn / (np.linalg.norm(Zn, axis=1, keepdims=True) + 1e-12)
    fp = atlas.meta["fingerprint"]
    out = {"schema": "reference_normalization_v1", "version": "1.0",
           "atlas_fingerprint": fp, "n_reference_spectra": int(len(Zn)),
           "n_reference_analytes": int(corpus.meta.analyte.nunique()),
           "statistic": "median / MAD (robust)",
           "component_center": [round(float(x), 6) for x in center],
           "component_spread": [round(float(x), 6) for x in spread],
           "note": "Query component coordinates are z-scored as (coord-center)/spread. "
                   "OOD uses cosine distance to the reference support (stored separately)."}
    (OUT / "reference_normalization_v1.json").write_text(json.dumps(out, indent=2))
    np.savez_compressed(OUT / "reference_support.npz",
                        support_unit=U.astype(np.float32),
                        analytes=corpus.meta.analyte.values)
    print(f"reference_normalization_v1.json written ({len(Zn)} spectra, fingerprint {fp})")
    print(f"  center range [{center.min():.4f}, {center.max():.4f}] | "
          f"spread range [{spread.min():.4f}, {spread.max():.4f}]")


if __name__ == "__main__":
    main()
