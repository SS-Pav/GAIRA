"""GAIRA V7 — Phase 02.5: seven complementary descriptions of a spectral motif.

No single description of a Raman motif is sufficient. A full profile is dominated by whichever
band happens to be tallest; a peak list throws away everything between the peaks; an activation
profile says what a motif responds to but nothing about what it looks like. The geometry of
motif space therefore has to be estimated from several views and cross-checked, which is what
this module builds.

**Chemistry-label firewall.** No function here takes a chemistry class as an input to a
representation. Class labels enter only after the geometry is fixed, to evaluate it. The
provenance view (F) deliberately carries *breadth* counts rather than class identity, for the
same reason Phase 02 discounted within-class provenance: a representation built on the class
partition would rediscover the class partition.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks, peak_prominences, peak_widths

D_GRID = 676
EPS = 1e-12

# Interpretable Raman windows. These are vibrational-mode windows, NOT chemistry classes —
# "1650-1690 amide I / C=C" describes a bond, and many classes contribute to it.
BAND_WINDOWS = [
    (450, 550, "S–S / skeletal deformation"),
    (550, 650, "C–S stretch"),
    (650, 730, "ring breathing (6-membered)"),
    (730, 800, "ring breathing (5-membered) / O–P–O"),
    (800, 880, "C–C skeletal / tyrosine doublet"),
    (880, 960, "C–C backbone / C–O–C ring"),
    (960, 1010, "PO4 / phenyl ring breathing"),
    (1010, 1070, "C–N / C–O stretch"),
    (1070, 1130, "C–C trans chain / PO2- / glycosidic C–O–C"),
    (1130, 1180, "C–C gauche / conjugated C–C"),
    (1180, 1250, "amide III / C–O–C asymmetric"),
    (1250, 1300, "=C–H in-plane bend / amide III"),
    (1300, 1360, "CH2 twist / CH deformation"),
    (1360, 1420, "COO- symmetric stretch"),
    (1420, 1480, "CH2 / CH3 scissoring"),
    (1480, 1540, "conjugated C=C"),
    (1540, 1600, "amide II / COO- asymmetric"),
    (1600, 1650, "aromatic ring C=C"),
    (1650, 1700, "amide I / cis C=C"),
    (1700, 1800, "C=O ester / carboxylic acid"),
]

VIEWS = ("spectral_profile", "peak_representation", "band_family", "activation",
         "reconstruction_contribution", "provenance", "edge_feature")


def _unit(M: np.ndarray) -> np.ndarray:
    return M / (np.linalg.norm(M, axis=-1, keepdims=True) + EPS)


# ── A. full spectral profile ─────────────────────────────────────────────────
def spectral_profile(H: np.ndarray) -> np.ndarray:
    """The 676-point preprocessed basis spectrum, L2-normalised.

    L2 rather than max: max-normalisation makes the whole vector hostage to a single bin, so a
    motif with one very sharp band and one with the same shape plus a little noise on that band
    land in different places for no chemical reason.
    """
    return _unit(np.asarray(H, float))


# ── B. peak representation ───────────────────────────────────────────────────
def peak_records(H: np.ndarray, grid: np.ndarray, prominence: float = 0.03) -> list[list[dict]]:
    """Position, prominence, width and relative intensity of every peak, per motif."""
    out = []
    for h in np.asarray(H, float):
        x = h / (h.max() + EPS)
        idx, _ = find_peaks(x, prominence=prominence)
        if idx.size == 0:
            out.append([])
            continue
        prom = peak_prominences(x, idx)[0]
        wid = peak_widths(x, idx, rel_height=0.5)[0] * float(np.diff(grid).mean())
        out.append([{"position": float(grid[i]), "prominence": float(p),
                     "width_cm": float(w), "relative_intensity": float(x[i])}
                    for i, p, w in zip(idx, prom, wid)])
    return out


def peak_vector(peaks: list[list[dict]], grid: np.ndarray, step: float = 8.0) -> np.ndarray:
    """Peaks rendered onto a shared coarse grid, weighted by prominence.

    Prominence, not height: a shoulder on a strong band is not an independent feature, and
    treating it as one inflates every position-based comparison.
    """
    centres = np.arange(grid.min(), grid.max() + step, step)
    V = np.zeros((len(peaks), centres.size))
    for i, ps in enumerate(peaks):
        for p in ps:
            V[i, int(np.argmin(np.abs(centres - p["position"])))] += p["prominence"]
    return V


def peak_summary(peaks: list[list[dict]]) -> np.ndarray:
    """Per-motif scalar summary: count, mean/median width, prominence spread, sharpness."""
    rows = []
    for ps in peaks:
        if not ps:
            rows.append([0, 0, 0, 0, 0])
            continue
        w = np.array([p["width_cm"] for p in ps])
        pr = np.array([p["prominence"] for p in ps])
        rows.append([len(ps), w.mean(), np.median(w), pr.std(), float((pr / (w + EPS)).mean())])
    return np.asarray(rows, float)


# ── C. band-family representation ────────────────────────────────────────────
def band_family(H: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Integrated intensity over interpretable vibrational windows, row-normalised.

    Row-normalised so the view describes *where* a motif puts its intensity rather than how
    much it has — amplitude is not chemistry.
    """
    H = np.asarray(H, float)
    V = np.zeros((H.shape[0], len(BAND_WINDOWS)))
    for k, (lo, hi, _) in enumerate(BAND_WINDOWS):
        m = (grid >= lo) & (grid < hi)
        V[:, k] = H[:, m].sum(axis=1)
    V = V / (V.sum(axis=1, keepdims=True) + EPS)
    return V, [f"{lo}-{hi} {lab}" for lo, hi, lab in BAND_WINDOWS]


# ── D. activation representation ─────────────────────────────────────────────
def activation_view(A_mol: np.ndarray) -> np.ndarray:
    """Motif activations across canonical molecules, molecule-balanced.

    `A_mol` is already one row per canonical molecule (never per spectrum), so a molecule with
    three replicates cannot outvote a molecule with one — limitation L-01 reappearing inside
    the geometry would be a quiet way to undo Phase 01.
    """
    A = np.asarray(A_mol, float).T                    # motifs x molecules
    return _unit(A)


# ── E. reconstruction contribution ───────────────────────────────────────────
def reconstruction_contribution(X: np.ndarray, canonical_id: np.ndarray, H: np.ndarray,
                                class_of: list[str]) -> tuple[np.ndarray, list[str]]:
    """Per-molecule reduction in residual error attributable to each motif.

    The marginal contribution within the motif's own class dictionary — what the motif uniquely
    supplies, given everything else that class already explains. This is a different question
    from "how strongly does it activate", and motifs can rank very differently on the two.

    `class_of` selects which dictionary the marginal is taken against. It is *not* a feature of
    the representation and never enters a distance: it only says which fit the motif came from.
    """
    from scipy.optimize import nnls
    X = np.asarray(X, float)
    mols = sorted(set(canonical_id))
    n = H.shape[0]
    V = np.zeros((n, len(mols)))
    cls = np.asarray(class_of)
    for i in range(n):
        members = np.where(cls == cls[i])[0]
        rest = [m for m in members if m != i]
        D_rest = H[rest] if rest else np.zeros((0, H.shape[1]))
        D_full = H[members]
        for k, mol in enumerate(mols):
            rows = X[canonical_id == mol]
            if rows.size == 0:
                continue
            r_rest = r_full = t = 0.0
            for x in rows:
                if D_rest.shape[0]:
                    c = nnls(D_rest.T, x)[0]
                    r_rest += float(((x - c @ D_rest) ** 2).sum())
                else:
                    r_rest += float((x ** 2).sum())
                c = nnls(D_full.T, x)[0]
                r_full += float(((x - c @ D_full) ** 2).sum())
                t += float((x ** 2).sum())
            V[i, k] = max(0.0, (r_rest - r_full) / (t + EPS))
    return V, mols


# ── F. provenance representation ─────────────────────────────────────────────
def provenance_view(lsm_meta: list[dict], sources_of: dict, excit_of: dict,
                    n_spectra_of: dict) -> tuple[np.ndarray, list[str]]:
    """Breadth of the evidence behind each motif — counts and diversity, never class identity.

    Class identity is deliberately excluded: a provenance view carrying the class label would
    re-encode the Phase 00 partition into the geometry, and every "discovered" community would
    be the partition looking back at us (risk R-01).
    """
    all_src = sorted({s for m in lsm_meta for a in m["analytes"] for s in sources_of.get(a, [])})
    all_exc = sorted({e for m in lsm_meta for a in m["analytes"] for e in excit_of.get(a, [])})
    cols = (["n_molecules", "n_spectra", "source_entropy", "excitation_entropy",
             "max_source_fraction"]
            + [f"src::{s}" for s in all_src] + [f"exc::{e}" for e in all_exc])
    V = np.zeros((len(lsm_meta), len(cols)))
    for i, m in enumerate(lsm_meta):
        an = [a for a in m["analytes"] if a]
        sc = {s: 0.0 for s in all_src}
        ec = {e: 0.0 for e in all_exc}
        for a in an:
            for s in sources_of.get(a, []):
                sc[s] += 1
            for e in excit_of.get(a, []):
                ec[e] += 1
        sv = np.array([sc[s] for s in all_src], float)
        ev = np.array([ec[e] for e in all_exc], float)
        sp = sv / (sv.sum() + EPS)
        ep = ev / (ev.sum() + EPS)
        V[i, 0] = len(an)
        V[i, 1] = sum(n_spectra_of.get(a, 0) for a in an)
        V[i, 2] = float(-(sp[sp > 0] * np.log(sp[sp > 0])).sum())
        V[i, 3] = float(-(ep[ep > 0] * np.log(ep[ep > 0])).sum())
        V[i, 4] = float(sp.max()) if sp.size else 0.0
        V[i, 5:5 + len(all_src)] = sp
        V[i, 5 + len(all_src):] = ep
    return V, cols


# ── G. Phase 02 edge-feature representation ──────────────────────────────────
def edge_feature_view(feat: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    """Each motif as its row of the seven Phase 02 edge-feature matrices, concatenated.

    A motif's *pattern of relationships* to all other motifs is itself a description of it —
    two motifs that relate to the rest of the dictionary in the same way are similar in a sense
    no pairwise spectral metric captures.
    """
    names = sorted(feat)
    blocks = []
    for f in names:
        M = np.asarray(feat[f], float).copy()
        np.fill_diagonal(M, 0.0)
        blocks.append(M)
    return np.hstack(blocks), [f"{f}::{i}" for f in names for i in range(blocks[0].shape[1])]


def build_panel(H, grid, A_mol, X, canonical_id, class_of, lsm_meta, sources_of,
                excit_of, n_spectra_of, feat) -> dict:
    """All seven views plus the manifest that says what each one means."""
    peaks = peak_records(H, grid)
    bf, bf_cols = band_family(H, grid)
    rc, rc_cols = reconstruction_contribution(X, canonical_id, H, class_of)
    pv, pv_cols = provenance_view(lsm_meta, sources_of, excit_of, n_spectra_of)
    ev, ev_cols = edge_feature_view(feat)
    views = {
        "spectral_profile": spectral_profile(H),
        "peak_representation": np.hstack([_unit(peak_vector(peaks, grid)), peak_summary(peaks)]),
        "band_family": bf,
        "activation": activation_view(A_mol),
        "reconstruction_contribution": _unit(rc),
        "provenance": pv,
        "edge_feature": _unit(ev),
    }
    manifest = {
        "spectral_profile": {"dim": views["spectral_profile"].shape[1],
                             "meaning": "L2-normalised 676-bin basis spectrum",
                             "captures": "overall band shape", "blind_to": "amplitude"},
        "peak_representation": {"dim": views["peak_representation"].shape[1],
                                "meaning": "prominence-weighted peaks on an 8 cm-1 grid + shape summary",
                                "captures": "discrete diagnostic features",
                                "blind_to": "the continuum between peaks"},
        "band_family": {"dim": views["band_family"].shape[1],
                        "meaning": "intensity share across 20 vibrational-mode windows",
                        "captures": "which bond systems carry the intensity",
                        "blind_to": "fine peak position", "columns": bf_cols},
        "activation": {"dim": views["activation"].shape[1],
                       "meaning": "molecule-balanced activation across 154 canonical molecules",
                       "captures": "what the motif responds to",
                       "blind_to": "what the motif looks like"},
        "reconstruction_contribution": {"dim": views["reconstruction_contribution"].shape[1],
                                        "meaning": "marginal residual reduction per molecule",
                                        "captures": "what the motif uniquely explains",
                                        "blind_to": "shared explanatory mass"},
        "provenance": {"dim": views["provenance"].shape[1],
                       "meaning": "evidence breadth: molecule/spectrum counts, source and excitation diversity",
                       "captures": "how well-supported a motif is",
                       "blind_to": "chemistry — class identity is deliberately excluded",
                       "columns": pv_cols},
        "edge_feature": {"dim": views["edge_feature"].shape[1],
                         "meaning": "the motif's row in all seven Phase 02 edge-feature matrices",
                         "captures": "relational position in the dictionary",
                         "blind_to": "absolute spectral identity"},
    }
    return {"views": views, "manifest": manifest, "peaks": peaks,
            "band_columns": bf_cols, "reconstruction_molecules": rc_cols}
