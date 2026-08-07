"""GAIRA V7 — Phase 07: the biochemical explanation layer.

**Evidence first, description afterwards.** For every programme this module reports what the data
say — which chemistry axes it loads on, which molecules use it, which CSMs and Raman bands lie
underneath — and only then composes a description from those facts by template. No programme is
named by hand.

Frozen CSM and LSM artefacts are read **for explanation only**. They are not inputs to the
factorisation, which sees nothing but the 16-dimensional Chemistry Evidence matrix.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


def programme_evidence(P: np.ndarray, W: np.ndarray, axis_names, y, cls,
                       axis_map: np.ndarray | None = None,
                       csm_records: list[dict] | None = None,
                       top_axes: int = 5, top_molecules: int = 8) -> list[dict]:
    """One evidence record per programme. Facts only."""
    P = np.asarray(P, float)
    W = np.clip(np.asarray(W, float), 0, None)
    y, cls = np.asarray(y), np.asarray(cls)
    out = []
    for k in range(P.shape[0]):
        load = P[k]
        share = np.abs(load) / (np.abs(load).sum() + EPS)
        order = np.argsort(-np.abs(load))
        axes = [{"chemistry_axis": axis_names[int(j)], "loading": float(load[j]),
                 "share_of_programme": float(share[j])}
                for j in order[:top_axes] if abs(load[j]) > 1e-6]
        w = W[:, k]
        mol_use: dict[str, float] = {}
        for m in set(y.tolist()):
            mol_use[m] = float(w[y == m].mean())
        top_mol = sorted(mol_use.items(), key=lambda kv: -kv[1])[:top_molecules]
        thr = np.quantile(w, 0.80)
        top_spec = w > thr
        cvals, ccnt = np.unique(cls[top_spec], return_counts=True) if top_spec.any() \
            else (np.array([]), np.array([]))
        rec = {
            "programme": k,
            "top_chemistry_axes": axes,
            "cumulative_share_top3": float(share[order[:3]].sum()),
            "n_axes_above_5pct": int((share > 0.05).sum()),
            "representative_molecules": [{"molecule": m, "mean_activation": v}
                                         for m, v in top_mol],
            "representative_classes": [{"class": str(c), "n_spectra": int(n)}
                                       for c, n in sorted(zip(cvals, ccnt),
                                                          key=lambda t: -t[1])[:5]],
            "usage_share": float((np.argmax(W, axis=1) == k).mean()),
            "mean_activation": float(w.mean()),
        }
        if axis_map is not None and csm_records is not None:
            # Which CSMs and bands sit under this programme's chemistry, via the frozen
            # Phase 05 CSM -> axis map. Read for explanation; never an input to the fit.
            contrib = axis_map @ np.clip(load, 0, None)
            cs = np.argsort(-contrib)[:6]
            rec["supporting_csms"] = [
                {"csm_id": csm_records[int(j)]["csm_id"],
                 "contribution": float(contrib[j]),
                 "lsms": [l["lsm_id"] if isinstance(l, dict) else str(l)
                          for l in csm_records[int(j)].get("contributing_lsms", [])],
                 "dominant_bands": [float(b) for b in
                                    csm_records[int(j)].get("dominant_bands", [])],
                 "band_assignment": csm_records[int(j)].get("band_assignment", "")}
                for j in cs if contrib[j] > 1e-6]
            rec["dominant_bands"] = sorted({round(b) for c in rec["supporting_csms"]
                                            for b in c["dominant_bands"]})[:12]
        out.append(rec)
    return out


# Broad-superclass vocabulary, used only to *compose* a description from evidence already
# computed. It never influences the factorisation and it never renames a programme that the
# evidence does not support.
_BROAD_WORDS = {
    "lipid": "membrane and storage lipid",
    "protein_amino_acid": "protein and amino-acid",
    "carbohydrate": "carbohydrate",
    "nucleic": "nucleic",
    "energy_metabolism": "small-molecule energy metabolism",
    "redox_cofactor": "redox cofactor and pigment",
}


def describe(rec: dict, broad_of_axis: dict, min_share: float = 0.15) -> tuple[str, str]:
    """Compose a description from the evidence, by template. Never by hand.

    Returns (description, basis). If no chemistry axis reaches `min_share` the programme is
    described as *diffuse* rather than given a name it has not earned.
    """
    axes = [a for a in rec["top_chemistry_axes"] if a["share_of_programme"] >= min_share]
    if not axes:
        return ("diffuse — no chemistry axis reaches "
                f"{min_share:.0%} of the programme's loading",
                f"largest share {rec['top_chemistry_axes'][0]['share_of_programme']:.2f}"
                if rec["top_chemistry_axes"] else "no loading")
    broads = []
    for a in axes:
        b = broad_of_axis.get(a["chemistry_axis"])
        if b and b not in broads:
            broads.append(b)
    names = [a["chemistry_axis"].replace("_", " ") for a in axes]
    share = sum(a["share_of_programme"] for a in axes)
    if len(broads) == 1:
        desc = f"{_BROAD_WORDS.get(broads[0], broads[0])} programme"
    elif len(broads) == 2:
        desc = (f"{_BROAD_WORDS.get(broads[0], broads[0])} + "
                f"{_BROAD_WORDS.get(broads[1], broads[1])} programme")
    else:
        desc = "cross-superclass programme"
    basis = (f"{' + '.join(names)} carry {share:.0%} of the loading; "
             f"{len(broads)} broad superclass(es); usage share {rec['usage_share']:.2f}")
    return desc, basis


def compare_layers(Ev, W, cls, y, folds) -> "pd.DataFrame":
    """Chemistry Evidence vs BSV2 vs a PCA control, on the axes the brief asks about."""
    import pandas as pd
    from sklearn.decomposition import PCA
    from .factorization import effective_rank
    from .validation import (heldout_chemistry, mutual_information_with_chemistry,
                             replicate_consistency)
    K = np.asarray(W).shape[1]
    pca = PCA(n_components=K, random_state=0).fit(Ev)
    rows = []
    for name, Z, dim in (("chemistry_evidence_16", Ev, Ev.shape[1]),
                         ("BSV2_programmes", np.asarray(W), K),
                         ("PCA_control", pca.transform(Ev), K)):
        h = heldout_chemistry(Z, cls, folds, y)
        rows.append({"representation": name, "dim": dim,
                     "compression_ratio": Ev.shape[1] / dim,
                     "heldout_chemistry_top1": h["top1"],
                     "heldout_chemistry_top3": h["top3"],
                     "mutual_information_norm": mutual_information_with_chemistry(Z, cls),
                     "effective_rank": effective_rank(Z),
                     "replicate_consistency": replicate_consistency(np.abs(Z), y),
                     "non_negative": bool(np.min(Z) >= -1e-9)})
    return pd.DataFrame(rows)
