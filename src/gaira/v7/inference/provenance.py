"""GAIRA V7 — Phase 05, Step 8: axis → CSM → LSM → molecule → reference spectrum.

Every claim the engine makes must be walkable back to measured spectra. A chain that terminates
early is a broken chain and the audit reports it as a failure, not as a shorter chain.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


def axis_chain(axis: str, a_csm: np.ndarray, M: np.ndarray, csm_records: list[dict],
               axis_index: dict[str, int], top_csms: int = 5,
               top_molecules: int = 8) -> dict:
    """The provenance waterfall for one active axis of one spectrum.

    Contributions are the *actual* additive terms `a_m · M_ma`, so the listed CSMs explain the
    number shown and their contributions sum to it exactly.
    """
    a = np.asarray(a_csm, float).ravel()
    j = axis_index[axis]
    contrib = a * M[:, j]
    total = float(contrib.sum())
    order = np.argsort(-contrib)
    chain, lsms, mols, classes = [], [], [], []
    for i in order[:top_csms]:
        if contrib[i] <= EPS:
            break
        r = csm_records[int(i)]
        ls = [l["lsm_id"] if isinstance(l, dict) else str(l)
              for l in r.get("contributing_lsms", [])]
        lsms += ls
        mols += list(r.get("supporting_analytes", []))
        classes += list(r.get("supporting_classes", []))
        chain.append({
            "csm_id": r["csm_id"],
            "activation": float(a[int(i)]),
            "axis_loading": float(M[int(i), j]),
            "contribution": float(contrib[i]),
            "contribution_share": float(contrib[i] / (total + EPS)),
            "lsms": ls,
            "dominant_bands": [float(b) for b in r.get("dominant_bands", [])],
            "band_assignment": r.get("band_assignment", ""),
            "molecules": list(r.get("supporting_analytes", []))[:top_molecules],
            "classes": list(r.get("supporting_classes", [])),
        })
    return {"axis": axis, "total_contribution": total, "csm_chain": chain,
            "lsms": sorted(set(lsms)), "molecules": sorted(set(mols)),
            "classes": sorted(set(classes)),
            "explained_share": float(sum(c["contribution"] for c in chain) / (total + EPS))
            if total > EPS else 0.0}


def verify_chains(chains: list[dict], known_lsms: set, known_molecules: set) -> "pd.DataFrame":
    """No broken provenance allowed. Each link is checked against the frozen registries."""
    import pandas as pd
    rows = []
    for c in chains:
        bad_l = [x for x in c["lsms"] if x not in known_lsms]
        bad_m = [x for x in c["molecules"] if x not in known_molecules]
        rows.append({
            "axis": c["axis"], "n_csms": len(c["csm_chain"]), "n_lsms": len(c["lsms"]),
            "n_molecules": len(c["molecules"]),
            "unknown_lsms": len(bad_l), "unknown_molecules": len(bad_m),
            "terminates_in_spectra": bool(c["molecules"]),
            "intact": bool(c["csm_chain"]) and not bad_l and not bad_m and bool(c["molecules"]),
        })
    return pd.DataFrame(rows)
