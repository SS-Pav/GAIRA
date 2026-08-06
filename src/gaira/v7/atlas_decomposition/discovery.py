"""GAIRA V7 — deterministic Atlas Component Substructure discovery.

THE METHOD, IN ONE PARAGRAPH
----------------------------
An atlas component `h_k` is a single vector, so on its own it says nothing about whether it
is chemically pure. The information is in the *analytes that activate it*: if component k
carries one substructure, every analyte activating it should show all of k's bands in the
same relative proportions; if k is being asked to explain two chemistries, some analytes
will show one sub-pattern of its bands and others a different one. Phase 01 therefore
defines, for each analyte participating in component k, its **band profile** over k's own
diagnostic bands, and clusters those profiles. Each cluster is a Atlas Component Substructure: a
subset of k's bands that recurs together across a coherent group of analytes.

PIPELINE (all steps deterministic)

  1. bands        peaks of h_k, prominence >= BAND_PROMINENCE * max(h_k); window +-4 bins
  2. participants canonical analytes whose activation share of k >= SHARE_THRESHOLD
  3. profile      q_a[b] = sum of the analyte's spectrum over band b, normalised to unit sum
  4. cluster      hierarchical (pre-registered linkage), cut by silhouette
  5. score        stability (jackknife), purity, coverage, band fidelity, redundancy
  6. reject       deterministic reasons; a component with <2 survivors is IRREDUCIBLE

WHAT THIS DOES NOT DO
    * it does not fit anything new — motifs are masked restrictions of frozen components
    * it does not touch the atlas, the projection, or the fingerprint
    * it never uses a chemical class label to choose bands, profiles, linkage or cut
"""
from __future__ import annotations

from collections import Counter

import numpy as np
from scipy.signal import find_peaks

from . import clustering as CL
from .motif import ACS, Band, build_motif_spectrum

DISCOVERY_VERSION = "v7_acs_v1"

# ── pre-registered parameters (fixed before the final run) ───────────────────
BAND_PROMINENCE = 0.05        # of the component maximum
BAND_HALF_WIDTH = 4           # bins either side of the peak (+-8 cm-1)
SHARE_THRESHOLD = 0.03        # analyte's share of its own total activation
MIN_PARTICIPANTS = 10         # below this a component is not analysable
MIN_BANDS = 4                 # below this a component is not analysable
MIN_MOTIF_ANALYTES = 3        # "recurring" means at least three molecules
MIN_STABILITY = 0.50          # jackknife co-assignment
MIN_MOTIF_BANDS = 2           # a one-band motif is a peak, not a substructure
REDUNDANCY_COSINE = 0.98      # near-duplicate within the same component
PROFILE_MODE = "raw"          # {"raw", "attribution"} — see compare_profile_modes()


def component_bands(h: np.ndarray, grid: np.ndarray,
                    prominence: float = BAND_PROMINENCE,
                    half_width: int = BAND_HALF_WIDTH) -> list[Band]:
    """Diagnostic bands of one atlas component. Deterministic."""
    h = np.asarray(h, float)
    peaks, props = find_peaks(h, prominence=prominence * float(h.max()))
    bands = []
    for p, pr in zip(peaks, props["prominences"]):
        lo = int(max(0, p - half_width))
        hi = int(min(len(h) - 1, p + half_width))
        bands.append(Band(index=int(p), center_cm=float(grid[p]), lo_bin=lo, hi_bin=hi,
                          prominence=float(pr),
                          component_weight=float(h[lo:hi + 1].sum())))
    return bands


def band_profiles(Xa: np.ndarray, participants: np.ndarray, bands: list[Band],
                  *, mode: str = PROFILE_MODE, component: np.ndarray | None = None,
                  Wa: np.ndarray | None = None, k: int | None = None,
                  recon: np.ndarray | None = None) -> np.ndarray:
    """Per-analyte profile over the component's bands, normalised to unit sum.

    `raw`          observed mass of the analyte inside each band.
    `attribution`  observed mass weighted by the share of that band the component is
                   actually explaining for this analyte. Benchmarked and NOT selected —
                   dividing by the reconstruction amplifies noise where the reconstruction
                   is small, and it partly cancels the very per-analyte variation the
                   method exists to detect.
    """
    rows = []
    for i in participants:
        v = []
        for b in bands:
            sl = b.slice()
            if mode == "attribution":
                att = (Wa[i, k] * component[sl]) / (recon[i, sl] + 1e-12)
                v.append(float((Xa[i, sl] * att).sum()))
            else:
                v.append(float(Xa[i, sl].sum()))
        rows.append(v)
    Q = np.asarray(rows, float)
    return Q / (Q.sum(axis=1, keepdims=True) + 1e-12)


def _participants(share: np.ndarray, k: int,
                  threshold: float = SHARE_THRESHOLD) -> np.ndarray:
    return np.where(share[:, k] >= threshold)[0]


def discover_component(k: int, H: np.ndarray, grid: np.ndarray, Xa: np.ndarray,
                       Wa: np.ndarray, share: np.ndarray, analyte_ids: list[str],
                       fine_of: dict, broad_of: dict, sources_of: dict,
                       spectra_of: dict, *, linkage_method: str = CL.DEFAULT_LINKAGE,
                       profile_mode: str = PROFILE_MODE,
                       recon: np.ndarray | None = None) -> dict:
    """Discover the motifs of one atlas component."""
    h = np.asarray(H[k], float)
    bands = component_bands(h, grid)
    parts = _participants(share, k)

    base = {"component": int(k), "n_bands": len(bands), "n_participants": int(len(parts)),
            "linkage": linkage_method, "profile_mode": profile_mode}

    if len(parts) < MIN_PARTICIPANTS or len(bands) < MIN_BANDS:
        return base | {"status": "NOT_ANALYSABLE", "motifs": [], "sweep": [],
                       "reason": (f"participants {len(parts)} < {MIN_PARTICIPANTS}"
                                  if len(parts) < MIN_PARTICIPANTS
                                  else f"bands {len(bands)} < {MIN_BANDS}")}

    Q = band_profiles(Xa, parts, bands, mode=profile_mode, component=h,
                      Wa=Wa, k=k, recon=recon)
    sel = CL.select_cut(Q, method=linkage_method)
    labels = np.asarray(sel["labels"])
    uniq = sorted(set(labels.tolist()))

    stab = CL.jackknife_stability(Q, labels, method=linkage_method, n_motifs=len(uniq))
    stab_of = dict(zip(uniq, stab))

    total_spectra = sum(spectra_of.get(analyte_ids[i], 0) for i in parts)

    motifs: list[ACS] = []
    for m_i, u in enumerate(uniq):
        members = parts[labels == u]
        names = [analyte_ids[i] for i in members]
        centroid = Q[labels == u].mean(axis=0)

        # bands this motif carries: those it emphasises above a uniform share
        uniform = 1.0 / len(bands)
        carried = [b for b in range(len(bands)) if centroid[b] >= uniform]
        if not carried:
            carried = [int(np.argmax(centroid))]
        w = centroid[carried]

        spec = build_motif_spectrum(h, bands, carried, w)
        fine = Counter(fine_of.get(n, "") for n in names)
        broad = Counter(broad_of.get(n, "") for n in names)
        srcs: Counter = Counter()
        for n in names:
            srcs.update(sources_of.get(n, []))
        dom, dom_n = (fine.most_common(1)[0] if fine else ("", 0))

        # band fidelity: cosine between the motif and the parent inside the carried bands
        mask = np.zeros_like(h, dtype=bool)
        for b in carried:
            mask[bands[b].slice()] = True
        pv, mv = h[mask], spec[mask]
        fid = float(np.dot(pv, mv) / (np.linalg.norm(pv) * np.linalg.norm(mv) + 1e-12))

        n_spec = sum(spectra_of.get(n, 0) for n in names)
        motifs.append(ACS(
            motif_id=f"c{k:02d}.m{m_i:02d}",
            parent_component=int(k), index_in_component=m_i,
            spectrum=spec,
            band_indices=[bands[b].index for b in carried],
            band_centers_cm=[bands[b].center_cm for b in carried],
            band_weights=[float(x) for x in (w / (w.sum() + 1e-12))],
            analytes=sorted(names), n_analytes=len(names), n_spectra=int(n_spec),
            fine_classes=dict(fine), broad_classes=dict(broad), sources=dict(srcs),
            stability=float(stab_of.get(u, 1.0)),
            purity=float(dom_n / len(names)) if names else 0.0,
            coverage_analytes=float(len(names) / len(parts)),
            coverage_spectra=float(n_spec / total_spectra) if total_spectra else 0.0,
            dominant_class=dom, band_fidelity=fid, redundancy_max=0.0,
            provenance={"discovery_version": DISCOVERY_VERSION,
                        "linkage": linkage_method, "profile_mode": profile_mode,
                        "cut_silhouette": sel["silhouette"]},
        ))

    _score_redundancy(motifs)
    _reject(motifs)

    kept = [m for m in motifs if m.retained]
    status = "IRREDUCIBLE" if len(kept) < 2 else "DECOMPOSED"
    return base | {"status": status, "motifs": motifs, "sweep": sel["sweep"],
                   "selected_n_motifs": int(sel["n_motifs"]),
                   "silhouette": sel["silhouette"], "size_gini": sel["size_gini"],
                   "max_motif_share": sel["max_motif_share"],
                   "labels": labels.tolist(),
                   "participant_ids": [analyte_ids[i] for i in parts],
                   "bands": [b.to_dict() for b in bands],
                   "n_retained": len(kept)}


def _score_redundancy(motifs: list[ACS]) -> None:
    for i, a in enumerate(motifs):
        worst = 0.0
        for j, b in enumerate(motifs):
            if i == j:
                continue
            worst = max(worst, a.cosine(b))
        a.redundancy_max = float(worst)


def _reject(motifs: list[ACS]) -> None:
    """Deterministic rejection. Order matters and is fixed, so reasons are reproducible.

    Redundancy is resolved by keeping the motif with more participating analytes, then the
    lower index — never by a coin flip.
    """
    for m in motifs:
        if m.n_analytes < MIN_MOTIF_ANALYTES:
            m.retained, m.rejection_reason = False, (
                f"too_few_analytes ({m.n_analytes} < {MIN_MOTIF_ANALYTES})")
        elif m.n_bands < MIN_MOTIF_BANDS:
            m.retained, m.rejection_reason = False, (
                f"noise_single_band ({m.n_bands} < {MIN_MOTIF_BANDS})")
        elif m.stability < MIN_STABILITY:
            m.retained, m.rejection_reason = False, (
                f"low_stability ({m.stability:.3f} < {MIN_STABILITY})")

    alive = [m for m in motifs if m.retained]
    order = sorted(range(len(alive)), key=lambda i: (-alive[i].n_analytes,
                                                     alive[i].index_in_component))
    dropped: set[int] = set()
    for pos_a, ia in enumerate(order):
        if ia in dropped:
            continue
        for ib in order[pos_a + 1:]:
            if ib in dropped:
                continue
            if alive[ia].cosine(alive[ib]) >= REDUNDANCY_COSINE:
                alive[ib].retained = False
                alive[ib].rejection_reason = (
                    f"redundant (cosine {alive[ia].cosine(alive[ib]):.3f} >= "
                    f"{REDUNDANCY_COSINE} with {alive[ia].motif_id})")
                dropped.add(ib)


def discover_all(H: np.ndarray, grid: np.ndarray, Xa: np.ndarray, Wa: np.ndarray,
                 analyte_ids: list[str], fine_of: dict, broad_of: dict,
                 sources_of: dict, spectra_of: dict, *,
                 linkage_method: str = CL.DEFAULT_LINKAGE,
                 profile_mode: str = PROFILE_MODE) -> list[dict]:
    """Discover motifs across every atlas component."""
    share = Wa / (Wa.sum(axis=1, keepdims=True) + 1e-12)
    recon = Wa @ H
    return [discover_component(k, H, grid, Xa, Wa, share, analyte_ids, fine_of, broad_of,
                               sources_of, spectra_of, linkage_method=linkage_method,
                               profile_mode=profile_mode, recon=recon)
            for k in range(H.shape[0])]


def collect_profiles(H: np.ndarray, grid: np.ndarray, Xa: np.ndarray, Wa: np.ndarray,
                     *, profile_mode: str = PROFILE_MODE) -> dict[int, np.ndarray]:
    """Band-profile matrices for every analysable component (used by the comparisons)."""
    share = Wa / (Wa.sum(axis=1, keepdims=True) + 1e-12)
    recon = Wa @ H
    out = {}
    for k in range(H.shape[0]):
        h = np.asarray(H[k], float)
        bands = component_bands(h, grid)
        parts = _participants(share, k)
        if len(parts) < MIN_PARTICIPANTS or len(bands) < MIN_BANDS:
            continue
        out[k] = band_profiles(Xa, parts, bands, mode=profile_mode, component=h,
                               Wa=Wa, k=k, recon=recon)
    return out
