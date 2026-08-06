"""GAIRA V7 Phase 01 — end-to-end LSM discovery from balanced references.

    balanced references → split by class → independent class-local NMF → LSMs

Per class: sweep `k_c`, select by the pre-registered smallest-on-plateau rule, run repeated
fits with Hungarian alignment for stability, type each motif, score it, and reject with a
deterministic reason. Classes too small to fit route to the anchor mechanism.

The frozen V5 atlas is not an input to any step here (P-15).
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from . import classlocal as CLS
from .lsm import LSM, classify_type, dominant_bands

DISCOVERY_VERSION = "v7_lsm_classlocal_v1"

MIN_CLASS_ANALYTES = 2         # below this: no local fit, anchor route (Strategy F)
MIN_LSM_ANALYTES = 1           # an LSM must be activated by at least one molecule
ANCHOR_MIN_QUALITY = 0.35      # Phase-00 QC floor


def _class_folds(ids: list[str], fold_of: dict, n_min: int = 2) -> np.ndarray:
    """Analyte-grouped folds restricted to this class, using the FROZEN Phase-00 folds."""
    f = np.array([fold_of.get(a, 0) for a in ids])
    if len(np.unique(f)) < n_min:                    # too few distinct folds in this class
        f = np.arange(len(ids)) % max(n_min, 2)
    return f


def discover_class(cls: str, X: np.ndarray, ids: list[str], weights: np.ndarray,
                   n_spectra_of: dict, sources_of: dict, excitations_of: dict,
                   broad_of: dict, fold_of: dict, grid: np.ndarray,
                   alpha: float = CLS.SPARSE_ALPHA, quality_of: dict | None = None) -> dict:
    """Independent class-local decomposition of one chemistry class."""
    n = len(ids)
    base = {"chemical_class": cls, "n_analytes": n,
            "n_spectra": int(sum(n_spectra_of.get(a, 0) for a in ids)),
            "k_ceiling": n // 2, "alpha": alpha}

    # source / excitation composition — risk R-16, required per class by the specification
    src = Counter()
    for a in ids:
        src.update(sources_of.get(a, []))
    exc = Counter()
    for a in ids:
        exc.update(excitations_of.get(a, []))
    dom_frac = (max(src.values()) / sum(src.values())) if src else 0.0
    base |= {"sources": dict(src), "excitations": dict(exc),
             "dominant_source": (src.most_common(1)[0][0] if src else ""),
             "dominant_source_fraction": round(float(dom_frac), 4),
             "source_confounded": bool(dom_frac >= 0.9 and n >= 3)}

    if n < MIN_CLASS_ANALYTES or n // 2 < 1:
        return base | {"status": "ANCHOR_ROUTE", "lsms": [], "sweep": [],
                       "k_selection": None,
                       "reason": f"n_analytes {n} < {MIN_CLASS_ANALYTES} — local "
                                 f"decomposition is not meaningful; Strategy F applies"}

    folds = _class_folds(ids, fold_of)
    sweep = CLS.sweep_k(X, ids, broad_of, folds, weights=weights,
                        k_max=n // 2, alpha=alpha)
    if not sweep:
        return base | {"status": "ANCHOR_ROUTE", "lsms": [], "sweep": [],
                       "k_selection": None, "reason": "no admissible k"}

    sel = CLS.select_k(sweep)
    k = sel["k"]
    rep = CLS.repeated_fits(X, k, weights=weights, alpha=alpha)
    W, H = rep["W"], rep["H"]

    # per-molecule normalised activation
    P = W / (W.sum(axis=1, keepdims=True) + 1e-12)
    recon = W @ H
    ss_tot = float(np.sum(np.nan_to_num(X) ** 2)) or 1.0

    lsms: list[LSM] = []
    for j in range(k):
        act = P[:, j]
        members = [ids[i] for i in np.where(act > CLS.MIN_ACTIVATION)[0]]
        broads = Counter(broad_of.get(a, "") for a in members)
        dom, dom_n = (broads.most_common(1)[0] if broads else ("", 0))
        contrib = float(np.sum((np.outer(W[:, j], H[j])) ** 2)) / ss_tot
        lsms.append(LSM(
            motif_id=f"{cls}.m{j:02d}", chemical_class=cls, index_in_class=j,
            spectrum=np.asarray(H[j], float),
            dominant_bands=dominant_bands(H[j], grid),
            analytes=sorted(members), n_analytes=len(members),
            n_spectra=int(sum(n_spectra_of.get(a, 0) for a in members)),
            activation_share=float(W[:, j].sum() / (W.sum() + 1e-12)),
            activation_sparsity=float(1.0 - (act > CLS.MIN_ACTIVATION).mean()),
            stability=float(rep["recurrence"][j]),
            matched_similarity=float(rep["matched_similarity"][j]),
            purity=float(dom_n / len(members)) if members else 0.0,
            reconstruction_share=round(contrib, 6),
            redundancy_max=0.0,
            lsm_type=classify_type(act),
            k_c=k, n_class_analytes=n, dominant_broad_class=dom,
            provenance={"discovery_version": DISCOVERY_VERSION,
                        "k_selection_rule": sel["rule"], "alpha": alpha,
                        "n_repeats_effective": rep["n_repeats_effective"]},
        ))

    _score_redundancy(lsms)
    _reject(lsms)
    kept = [m for m in lsms if m.retained]
    return base | {"status": "DECOMPOSED" if kept else "NO_STABLE_LSM",
                   "lsms": lsms, "sweep": sweep, "k_selection": sel,
                   "k_c": k, "n_retained": len(kept),
                   "explained_variance": round(
                       float(1.0 - np.sum((np.nan_to_num(X) - recon) ** 2) / ss_tot), 4),
                   "analyte_ids": list(ids)}


def _score_redundancy(lsms: list[LSM]) -> None:
    for i, a in enumerate(lsms):
        a.redundancy_max = float(max((a.cosine(b) for j, b in enumerate(lsms) if j != i),
                                     default=0.0))


def _reject(lsms: list[LSM]) -> None:
    """Deterministic rejection, fixed order so reasons are reproducible."""
    for m in lsms:
        if float(m.spectrum.sum()) <= 0:
            m.retained, m.rejection_reason = False, "empty_component"
        elif m.n_analytes < MIN_LSM_ANALYTES:
            m.retained, m.rejection_reason = False, (
                f"no_activating_analyte ({m.n_analytes} < {MIN_LSM_ANALYTES})")
        elif m.stability < CLS.MIN_STABILITY:
            m.retained, m.rejection_reason = False, (
                f"low_stability ({m.stability:.3f} < {CLS.MIN_STABILITY})")

    alive = [m for m in lsms if m.retained]
    order = sorted(range(len(alive)),
                   key=lambda i: (-alive[i].n_analytes, alive[i].index_in_class))
    dropped: set[int] = set()
    for pos, ia in enumerate(order):
        if ia in dropped:
            continue
        for ib in order[pos + 1:]:
            if ib in dropped:
                continue
            c = alive[ia].cosine(alive[ib])
            if c >= CLS.REDUNDANCY_COSINE:
                alive[ib].retained = False
                alive[ib].rejection_reason = (
                    f"redundant (cosine {c:.3f} >= {CLS.REDUNDANCY_COSINE} "
                    f"with {alive[ia].motif_id})")
                dropped.add(ib)


def make_anchor(cls: str, analyte: str, spectrum: np.ndarray, grid: np.ndarray,
                broad_of: dict, n_spectra: int, quality: float,
                justification: str) -> LSM:
    """Strategy F: admit a single high-quality reference as an anchored atom.

    Permanently flagged, with exactly one supporting analyte. An anchor says "this chemistry
    exists and we have one clean reference for it" — never that it is well-characterised.
    """
    return LSM(
        motif_id=f"{cls}.anchor00", chemical_class=cls, index_in_class=0,
        spectrum=np.asarray(spectrum, float), dominant_bands=dominant_bands(spectrum, grid),
        analytes=[analyte], n_analytes=1, n_spectra=int(n_spectra),
        activation_share=1.0, activation_sparsity=0.0,
        stability=float("nan") if False else 1.0, matched_similarity=1.0,
        purity=1.0, reconstruction_share=float("nan") if False else 1.0,
        redundancy_max=0.0, lsm_type="molecule_discriminating",
        k_c=0, n_class_analytes=1, dominant_broad_class=broad_of.get(analyte, ""),
        retained=quality >= ANCHOR_MIN_QUALITY,
        rejection_reason=("" if quality >= ANCHOR_MIN_QUALITY
                          else f"anchor_below_quality_floor ({quality:.3f} < {ANCHOR_MIN_QUALITY})"),
        is_anchor=True, anchor_justification=justification,
        provenance={"discovery_version": DISCOVERY_VERSION, "route": "Strategy F anchor",
                    "quality": round(float(quality), 4)},
    )


def discover_all(blocks: dict[str, dict], grid: np.ndarray, n_spectra_of: dict,
                 sources_of: dict, excitations_of: dict, broad_of: dict, fold_of: dict,
                 quality_of: dict, alpha: float = CLS.SPARSE_ALPHA) -> list[dict]:
    """Run the class-local decomposition over every chemistry class."""
    out = []
    for cls in sorted(blocks):
        b = blocks[cls]
        res = discover_class(cls, b["X"], b["ids"], b["weights"], n_spectra_of,
                             sources_of, excitations_of, broad_of, fold_of, grid,
                             alpha=alpha, quality_of=quality_of)
        if res["status"] == "ANCHOR_ROUTE":
            anchors = []
            for i, a in enumerate(b["ids"]):
                anchors.append(make_anchor(
                    cls, a, b["X"][i], grid, broad_of, n_spectra_of.get(a, 0),
                    quality_of.get(a, 1.0),
                    f"Class '{cls}' holds {len(b['ids'])} canonical molecule(s) — below the "
                    f"{MIN_CLASS_ANALYTES}-molecule floor for a local decomposition. Admitted "
                    f"as an anchored reference so this chemistry has a route into the "
                    f"representation; support is one molecule and is flagged as such."))
            res["lsms"] = anchors
            res["n_retained"] = sum(m.retained for m in anchors)
        out.append(res)
    return out


def class_prior_bias(results: list[dict]) -> pd.DataFrame:
    """Risk R-01: is a class's decomposition driven by the partition rather than spectroscopy?

    Diagnostic: if a class's LSMs are all `class_shared` and its within-class activation is
    nearly uniform, the fit found no internal structure — the class boundary is doing the work.
    """
    rows = []
    for r in results:
        kept = [m for m in r.get("lsms", []) if m.retained]
        if not kept:
            continue
        types = Counter(m.lsm_type for m in kept)
        shared = types.get("class_shared", 0) / len(kept)
        rows.append({
            "chemical_class": r["chemical_class"], "n_analytes": r["n_analytes"],
            "k_c": r.get("k_c", 0), "n_retained": len(kept),
            "frac_class_shared": round(shared, 4),
            "frac_subfamily": round(types.get("subfamily", 0) / len(kept), 4),
            "frac_discriminating": round(types.get("molecule_discriminating", 0) / len(kept), 4),
            "mean_activation_sparsity": round(
                float(np.mean([m.activation_sparsity for m in kept])), 4),
            "prior_dominated": bool(shared >= 0.99 and len(kept) > 1),
            "note": ("all motifs class-shared — the partition, not spectroscopy, may be "
                     "driving this decomposition" if shared >= 0.99 and len(kept) > 1 else ""),
        })
    return pd.DataFrame(rows)
