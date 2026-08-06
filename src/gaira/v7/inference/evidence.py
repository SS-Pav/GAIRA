"""GAIRA V7 — Phase 05, Step 6: the Biochemical Evidence Profile.

This replaces the Phase 03 theme layer as the interpretable output. It is **not** a
factorisation and nothing here is fitted. Eleven axes are defined *a priori* from Raman band
assignments; each frozen CSM is mapped onto them by where its own diagnostic bands fall; a
spectrum's profile is the sum of its CSM activations through that fixed, sparse, auditable map.

Why the change. Phase 03's themes were discovered by NMF over CSM co-activation and then named —
and Phase 04 measured what that cost: chemistry-class generalisation to unseen molecules fell
0.855 (CSM) → 0.405 (theme). A discovered layer that loses half the signal and still needs a
human to name it is worse on both counts than a declared layer that loses nothing, because the
evidence profile here is *additive on top of* CSM inference rather than *instead of* it.

The honest part is the residual. A CSM whose bands fall in no axis window contributes to
`unassigned`, never spread across the spokes. An axis reports zero when there is no evidence.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12
# A CSM *supports* an axis when it devotes at least this share of its diagnostic band strength
# to that axis. Below it the loading is a rounding-level contribution from a band that clipped a
# window edge, and counting it inflates every axis to near-universal support.
SUPPORT_FLOOR = 0.10

# ── the eleven axes ──────────────────────────────────────────────────────────
# Each window is (lo, hi, weight, what the band is). Windows are deliberately *wide* — this is
# region-based mapping, not exact-peak matching (project principle 5). Windows overlap between
# axes wherever the underlying chemistry overlaps; that ambiguity is reported, not hidden.
AXES: dict[str, list[tuple[float, float, float, str]]] = {
    "aliphatic_chain": [
        (1420, 1475, 1.0, "CH2/CH3 scissoring and bending"),
        (1280, 1320, 0.8, "CH2 twisting"),
        (1050, 1140, 0.6, "C–C skeletal stretch (trans/gauche)"),
        (890, 920, 0.4, "C–C skeletal terminal"),
    ],
    "unsaturation": [
        (1645, 1675, 1.0, "cis C=C stretch"),
        (1255, 1275, 0.7, "=C–H in-plane bend"),
        (965, 980, 0.5, "trans C=C out-of-plane =C–H"),
    ],
    "carbonyl_ester": [
        (1720, 1780, 1.0, "C=O stretch (ester/acid/ketone)"),
        (1150, 1200, 0.5, "C–O–C ester stretch"),
        (850, 880, 0.3, "O–C–O / glycerol C–C of acylglycerols"),
    ],
    "amide_protein": [
        (1630, 1700, 1.0, "amide I (C=O stretch, backbone)"),
        (1230, 1300, 0.8, "amide III (N–H bend / C–N stretch)"),
        (1540, 1580, 0.6, "amide II"),
        (930, 950, 0.3, "C–C backbone (α-helix)"),
    ],
    "carbohydrate_skeletal": [
        (1000, 1150, 1.0, "C–O / C–C ring and glycosidic stretch"),
        (1330, 1390, 0.6, "C–O–H deformation"),
        (830, 870, 0.5, "anomeric C–H deformation"),
        (510, 560, 0.4, "pyranose ring deformation"),
    ],
    "heterocyclic_ring": [
        (700, 800, 1.0, "ring breathing (5-/6-membered N heterocycle)"),
        (1330, 1400, 0.6, "ring stretch"),
        (1470, 1500, 0.5, "ring C=N / C=C stretch"),
    ],
    "purine": [
        (715, 740, 1.0, "purine ring breathing (adenine/guanine)"),
        (1330, 1350, 0.8, "purine ring stretch"),
        (1480, 1500, 0.7, "imidazole ring stretch"),
        (1570, 1590, 0.5, "purine C=N"),
    ],
    "sulfur_thiol": [
        (480, 560, 1.0, "S–S disulfide stretch"),
        (620, 700, 0.8, "C–S stretch"),
        (1080, 1110, 0.2, "C–S adjacent skeletal"),
    ],
    "phosphate_nucleic": [
        (960, 995, 1.0, "PO4 symmetric stretch"),
        (1075, 1100, 0.8, "PO2- symmetric stretch"),
        (805, 825, 0.6, "phosphodiester backbone O–P–O"),
        (1230, 1245, 0.4, "PO2- antisymmetric stretch"),
    ],
    "aromatic_residue": [
        (995, 1012, 1.0, "phenyl ring breathing"),
        (1580, 1615, 0.9, "aromatic C=C ring stretch"),
        (1195, 1215, 0.6, "C–C6H5 stretch"),
        (620, 645, 0.4, "ring deformation"),
    ],
    "chromophore_conjugated": [
        (1500, 1540, 1.0, "conjugated C=C (polyene / carotenoid ν1)"),
        (1145, 1175, 0.9, "conjugated C–C (polyene ν2)"),
        (1000, 1020, 0.3, "in-plane CH3 rock of polyene"),
    ],
}
AXIS_NAMES = tuple(AXES.keys())


def _prominence_profile(H: np.ndarray, grid: np.ndarray, window: float = 40.0) -> np.ndarray:
    """Local prominence: intensity above the local baseline, so a pedestal cannot count as a band.

    The same construction Phase 02 adopted after the pedestal-driven `band_overlap` feature was
    found to be 0.978-correlated with plain spectral cosine.
    """
    from scipy.ndimage import minimum_filter1d
    step = float(np.median(np.diff(grid)))
    w = max(int(window / step) | 1, 3)
    H = np.atleast_2d(np.clip(np.asarray(H, float), 0, None))
    return np.clip(H - minimum_filter1d(H, w, axis=1), 0.0, None)


def build_axis_map(H_csm: np.ndarray, grid: np.ndarray,
                   csm_records: list[dict] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """The frozen (49 x 11) CSM -> axis loading matrix, plus each CSM's unassigned mass.

    Two ingredients, and both must agree before an axis loads at all:

    1. **Support** — the CSM must have one of its *diagnostic* bands, as listed in the frozen
       Phase 02 registry, inside one of the axis's windows. Without this mask every axis loads
       on every CSM: local prominence is non-zero almost everywhere in a 676-bin window, so a
       purely intensity-based map assigned all 49 CSMs to 8 of the 11 axes and drove the
       specificity weight to 1.00 across the board — an axis that fires on everything.
    2. **Magnitude** — the prominence the CSM carries inside those windows, as a fraction of its
       total prominence. This is what makes a strong band count for more than a weak one.

    Axis windows overlap wherever the chemistry overlaps (amide I and cis C=C both live near
    1650; PO2- and C-C skeletal both near 1080), so the rows do **not** sum to one and the axes
    are not a partition. That ambiguity is the honest state of Raman band assignment and it is
    reported rather than resolved by fiat. The shortfall between the covered and total
    prominence is the `unassigned` mass, which is never redistributed onto the spokes.
    """
    P = _prominence_profile(H_csm, grid)
    if csm_records is None:
        # Fallback used only by unit tests on synthetic dictionaries: integrate prominence over
        # the whole window. On the real atlas this is too permissive — see the docstring.
        total = P.sum(axis=1) + EPS
        M = np.zeros((H_csm.shape[0], len(AXIS_NAMES)))
        for a, name in enumerate(AXIS_NAMES):
            for lo, hi, w, _ in AXES[name]:
                sel = (grid >= lo) & (grid <= hi)
                if sel.any():
                    M[:, a] += w * P[:, sel].sum(axis=1)
        covered = np.zeros(len(grid), bool)
        for name in AXIS_NAMES:
            for lo, hi, _, _ in AXES[name]:
                covered |= (grid >= lo) & (grid <= hi)
        return M / total[:, None], P[:, ~covered].sum(axis=1) / total

    M = np.zeros((H_csm.shape[0], len(AXIS_NAMES)))
    unassigned = np.zeros(H_csm.shape[0])
    for i, r in enumerate(csm_records):
        bands = [float(b) for b in r.get("dominant_bands", [])]
        if not bands:
            continue
        idx = [int(np.argmin(np.abs(grid - b))) for b in bands]
        strength = np.array([P[i, j] for j in idx], float)
        tot = strength.sum() + EPS
        hit_any = np.zeros(len(bands), bool)
        for a, name in enumerate(AXIS_NAMES):
            acc = 0.0
            for k, b in enumerate(bands):
                for lo, hi, w, _ in AXES[name]:
                    if lo <= b <= hi:
                        acc += w * strength[k]
                        hit_any[k] = True
                        break
            M[i, a] = acc / tot
        unassigned[i] = float(strength[~hit_any].sum() / tot)
    return M, unassigned


def window_overlap() -> "pd.DataFrame":
    """How much each pair of axes shares band territory — the ambiguity, stated up front."""
    import pandas as pd
    g = np.arange(450.0, 1800.1, 2.0)
    cov = {n: np.zeros(len(g), bool) for n in AXIS_NAMES}
    for n in AXIS_NAMES:
        for lo, hi, _, _ in AXES[n]:
            cov[n] |= (g >= lo) & (g <= hi)
    rows = []
    for i, a in enumerate(AXIS_NAMES):
        for b in AXIS_NAMES[i + 1:]:
            inter = (cov[a] & cov[b]).sum()
            if inter:
                rows.append({"axis_a": a, "axis_b": b, "shared_bins": int(inter),
                             "jaccard": float(inter / (cov[a] | cov[b]).sum())})
    return pd.DataFrame(rows).sort_values("jaccard", ascending=False).reset_index(drop=True)


def axis_specificity(M: np.ndarray) -> np.ndarray:
    """How selective each axis is across the dictionary — the ubiquity correction.

    Phase 03 named all four discovered themes "aliphatic chain + …" because CH2 bending is
    present in almost everything organic. An axis that loads on nearly every CSM carries little
    information when it fires; this is the inverse-document-frequency weight that says so. It
    scales *confidence*, never the evidence magnitude, so the profile stays additive.
    """
    frac = (M > SUPPORT_FLOOR).mean(axis=0)
    return np.log(1.0 / np.clip(frac, 1.0 / max(M.shape[0], 1), 1.0)) + 1.0


def profile(A: np.ndarray, M: np.ndarray, spec: np.ndarray,
            explained_variance: np.ndarray | None = None) -> dict:
    """A spectrum's Biochemical Evidence Profile: magnitude, confidence, coverage, support.

    - **magnitude** — `Σ_m a_m M_ma`, normalised across axes. Purely additive in the frozen
      activations, so any value can be traced back to the CSMs that produced it.
    - **coverage** — the share of the spectrum's *total* CSM activation that reaches this axis.
      Low coverage means the axis is inferred from a small corner of the evidence.
    - **support** — how many active CSMs contribute at all. One CSM is an assertion resting on
      one motif.
    - **confidence** — the product of coverage, specificity (normalised), a support term that
      saturates at three CSMs, and the spectrum's reconstruction quality. It is deliberately
      pessimistic: every factor is in [0, 1] and they multiply, so weak evidence looks weak.
    """
    A = np.atleast_2d(np.asarray(A, float))
    raw = A @ M                                            # (n, 11)
    tot = raw.sum(axis=1, keepdims=True) + EPS
    mag = raw / tot
    act_tot = A.sum(axis=1, keepdims=True) + EPS
    coverage = raw / act_tot
    support = ((A[:, :, None] > 1e-9) & (M[None, :, :] > SUPPORT_FLOOR)).sum(axis=1)
    s = spec / (spec.max() + EPS)
    ev = np.ones(A.shape[0]) if explained_variance is None else np.clip(explained_variance, 0, 1)
    conf = (np.clip(coverage / (coverage.max(axis=1, keepdims=True) + EPS), 0, 1)
            * s[None, :]
            * np.clip(support / 3.0, 0, 1)
            * ev[:, None])
    return {"axes": list(AXIS_NAMES), "magnitude": mag, "raw": raw, "coverage": coverage,
            "support": support, "confidence": conf}


def ground_axes(M: np.ndarray, csm_records: list[dict], grid: np.ndarray,
                H_csm: np.ndarray) -> list[dict]:
    """Step 6's grounding requirement: for each axis, what actually supports it."""
    P = _prominence_profile(H_csm, grid)
    out = []
    for a, name in enumerate(AXIS_NAMES):
        idx = np.argsort(-M[:, a])
        sup = [int(i) for i in idx if M[i, a] > SUPPORT_FLOOR]
        mols, classes, bands = [], [], []
        for i in sup:
            r = csm_records[i]
            mols += list(r.get("supporting_analytes", []))
            classes += list(r.get("supporting_classes", []))
            for b in r.get("dominant_bands", []):
                if any(lo <= b <= hi for lo, hi, _, _ in AXES[name]):
                    bands.append(float(b))
        out.append({
            "axis": name,
            "band_windows": [{"low": lo, "high": hi, "weight": w, "assignment": d}
                             for lo, hi, w, d in AXES[name]],
            "supporting_csms": [csm_records[i]["csm_id"] for i in sup],
            "n_supporting_csms": len(sup),
            "supporting_bands": sorted(set(bands)),
            "supporting_molecules": sorted(set(mols)),
            "n_supporting_molecules": len(set(mols)),
            "supporting_classes": sorted(set(classes)),
            "top_csm": csm_records[int(idx[0])]["csm_id"],
            "mean_loading": float(M[:, a].mean()),
            "max_loading": float(M[:, a].max()),
        })
    return out


def validate_axes(E: np.ndarray, y_class: np.ndarray, axis_to_classes: dict) -> "pd.DataFrame":
    """Falsifiability check: does each axis fire on the chemistry it claims?

    For every axis with a declared expected chemistry class, AUROC of that axis's magnitude
    separating the molecules of those classes from all others. Chemistry labels are used **only
    here, as evaluation** — nothing in the axis definition or the loading matrix saw a label.
    An axis at AUROC ≈ 0.5 is a decorative axis and must be reported as one.
    """
    import pandas as pd
    from .openset import auroc
    rows = []
    for a, name in enumerate(AXIS_NAMES):
        exp = axis_to_classes.get(name, [])
        if not exp:
            rows.append({"axis": name, "expected_classes": "", "n_positive": 0,
                         "auroc": np.nan, "verdict": "not testable"})
            continue
        pos = np.isin(y_class, exp)
        if pos.sum() == 0 or (~pos).sum() == 0:
            rows.append({"axis": name, "expected_classes": ";".join(exp),
                         "n_positive": int(pos.sum()), "auroc": np.nan,
                         "verdict": "not testable"})
            continue
        au = auroc(E[:, a], pos)
        rows.append({"axis": name, "expected_classes": ";".join(exp),
                     "n_positive": int(pos.sum()), "auroc": au,
                     "verdict": "grounded" if au >= 0.70 else
                                ("weak" if au >= 0.60 else "not discriminative")})
    return pd.DataFrame(rows)
