"""GAIRA V7 — Phase 03: the criteria that choose `K`, and the admissibility constraint.

`VALIDATION_AND_DECISION_RULES.md` §4 lists eight criteria including *mutual information with
chemistry*. The user brief forbids human ontology labels during discovery. These are reconciled
by splitting the list rather than by ignoring either:

* **Label-free criteria decide `K`** — information retained, held-out reconstruction, stability,
  compression, calibration, spectral coherence, and *band-based* chemical admissibility.
* **Mutual information with chemistry is computed post hoc**, after `K` is fixed and the themes
  are validated, and is reported as evidence rather than used as an objective.

Admissibility is a **hard constraint**, not a weighted term: a `K` whose themes cannot each be
named as coherent chemistry is rejected regardless of its score. It is judged from the themes'
own dominant bands, so it needs no labels.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import nnls

EPS = 1e-12

# Pre-registered weights over the label-free criteria. Committed before the sweep.
CRITERIA = {
    "information_retained": (0.24, +1),
    "heldout_reconstruction": (0.20, +1),
    "stability": (0.20, +1),
    "spectral_coherence": (0.14, +1),
    "compression": (0.10, +1),
    "calibration_ece": (0.06, -1),
    "membership_sparsity": (0.06, +1),
}
K_RANGE = range(2, 16)
PLATEAU_TOLERANCE = 0.02
ADMISSIBILITY_MIN_BANDS = 2
ADMISSIBILITY_MIN_CONCENTRATION = 0.60   # of band prominence in the top two mode families
SPARSITY_TARGET = 0.60                # of the mass in a CSM's top two themes
DEGENERATE_INFORMATION = 0.05         # below this a model has not fitted anything
DEGENERATE_MAX_THEME_SHARE = 0.60     # one theme may not own this much of the membership mass
DEGENERATE_MIN_EFFECTIVE_K = 2.0      # effective number of themes actually used
DISTINCTNESS_MAX_COSINE = 0.90        # two themes sharing a chemistry must not also
                                      # be near-duplicate spectra

# Vibrational windows — bond systems, not chemistry classes. Assignability against this table
# is a statement about physics, not about the curated ontology.
WINDOWS = [
    (450, 550, "S–S stretch"), (550, 650, "C–S stretch"),
    (650, 730, "six-membered ring breathing"), (730, 800, "five-membered ring / O–P–O"),
    (800, 880, "C–C skeletal"), (880, 960, "C–C backbone / C–O–C ring"),
    (960, 1010, "PO4 / phenyl ring breathing"), (1010, 1070, "C–N / C–O stretch"),
    (1070, 1130, "C–C trans chain / PO2⁻ / glycosidic C–O–C"),
    (1130, 1180, "C–C gauche / conjugated C–C"), (1180, 1250, "amide III / C–O–C asymmetric"),
    (1250, 1300, "=C–H in-plane bend"), (1300, 1360, "CH2 twist"),
    (1360, 1420, "COO⁻ symmetric stretch"), (1420, 1480, "CH2 / CH3 scissoring"),
    (1480, 1540, "conjugated C=C"), (1540, 1600, "amide II / COO⁻ asymmetric"),
    (1600, 1650, "aromatic ring C=C"), (1650, 1700, "amide I / cis C=C"),
    (1700, 1800, "C=O ester / carboxylic acid"),
]
# Mode families: assignments within one family are mutually consistent chemistry.
FAMILY = {
    "S–S stretch": "sulfur", "C–S stretch": "sulfur",
    "six-membered ring breathing": "ring", "five-membered ring / O–P–O": "ring",
    "conjugated C=C": "ring", "aromatic ring C=C": "ring",
    "C–C gauche / conjugated C–C": "ring",
    "C–C skeletal": "skeletal", "C–C backbone / C–O–C ring": "skeletal",
    "C–N / C–O stretch": "skeletal", "C–C trans chain / PO2⁻ / glycosidic C–O–C": "skeletal",
    "PO4 / phenyl ring breathing": "phosphate",
    "amide III / C–O–C asymmetric": "amide", "amide II / COO⁻ asymmetric": "amide",
    "amide I / cis C=C": "amide",
    "CH2 twist": "aliphatic", "CH2 / CH3 scissoring": "aliphatic",
    "=C–H in-plane bend": "unsaturation",
    "COO⁻ symmetric stretch": "carboxyl", "C=O ester / carboxylic acid": "carboxyl",
}


def assign_band(cm: float) -> str | None:
    for lo, hi, lab in WINDOWS:
        if lo <= cm < hi:
            return lab
    return None


def theme_bands(theme: np.ndarray, grid: np.ndarray, top: int = 5) -> list[float]:
    from scipy.signal import find_peaks, peak_prominences
    x = theme / (theme.max() + EPS)
    idx, _ = find_peaks(x, prominence=0.04)
    if idx.size == 0:
        return []
    prom = peak_prominences(x, idx)[0]
    order = np.argsort(prom)[::-1][:top]
    return sorted(float(grid[i]) for i in idx[order])


def admissibility(theme: np.ndarray, grid: np.ndarray,
                  specificity: dict[str, float] | None = None) -> dict:
    """Can a spectroscopist name this theme from its own bands?

    The test is **family concentration**: the share of the theme's dominant-band prominence
    carried by its two strongest vibrational mode families. A theme whose intensity is
    concentrated in one or two families describes a bond system; one whose prominence is spread
    evenly across five families is a mixture, and no honest single name exists for it.

    Two earlier formulations were tried and both were defective:

    - **assignable fraction ≥ 0.60.** The window table tiles 450–1800 cm-1 continuously, so
      every detected band falls in *some* window and this fraction was 1.000 for every theme of
      every model at every K. A criterion that cannot fail is not a criterion. It is still
      computed and reported, but it does not gate.
    - **at most three mode families among the top five bands.** Counting families with equal
      weight lets a fifth-ranked minor band veto a theme that its first and second bands
      clearly define. Under that rule no K was admissible under any of the five models — not
      because the themes were mixtures, but because five equally-weighted bands in a
      biological Raman spectrum almost always touch four families.

    Prominence-weighting asks the question a spectroscopist actually asks: is most of this
    spectrum's diagnostic intensity one chemistry?
    """
    from scipy.signal import find_peaks, peak_prominences
    bands = theme_bands(theme, grid)
    labs = [assign_band(b) for b in bands]
    named = [l for l in labs if l]
    frac = len(named) / max(len(bands), 1)

    x = theme / (theme.max() + EPS)
    idx, _ = find_peaks(x, prominence=0.04)
    prom_of = {}
    if idx.size:
        pr = peak_prominences(x, idx)[0]
        for i, pv in zip(idx, pr):
            prom_of[round(float(grid[i]), 3)] = float(pv)

    weight: dict[str, float] = {}
    for b, l in zip(bands, labs):
        if not l:
            continue
        weight[FAMILY.get(l, "other")] = weight.get(FAMILY.get(l, "other"), 0.0) + \
            prom_of.get(round(b, 3), 0.0)
    if specificity:
        weight = {k: v * specificity.get(k, 1.0) for k, v in weight.items()}
    total = sum(weight.values()) or EPS
    ranked = sorted(weight.items(), key=lambda kv: -kv[1])
    conc = sum(v for _, v in ranked[:2]) / total
    ok = len(bands) >= ADMISSIBILITY_MIN_BANDS and conc >= ADMISSIBILITY_MIN_CONCENTRATION
    return {"bands_cm1": bands, "assignments": named, "assigned_fraction": float(frac),
            "mode_families": [k for k, _ in ranked],
            "dominant_families": [k for k, _ in ranked[:2]],
            "family_concentration": float(conc),
            "admissible": bool(ok)}


# ── the label-free criteria ──────────────────────────────────────────────────
def information_retained(X: np.ndarray, S: np.ndarray, themes: np.ndarray) -> float:
    """Explained variance of the CSM spectra reconstructed from theme memberships alone."""
    R = S @ themes
    return float(max(0.0, 1.0 - ((X - R) ** 2).sum() / ((X ** 2).sum() + EPS)))


def heldout_reconstruction(X: np.ndarray, fit_fn, folds: np.ndarray) -> float:
    """Refit the themes without each fold and reconstruct the held-out CSMs by NNLS."""
    scores = []
    for f in sorted(set(folds)):
        te = folds == f
        if te.all() or (~te).sum() < 3:
            continue
        out = fit_fn(X[~te])
        Th = out["themes"]
        res = tot = 0.0
        for x in X[te]:
            c = nnls(Th.T, x)[0]
            res += float(((x - c @ Th) ** 2).sum())
            tot += float((x ** 2).sum())
        scores.append(max(0.0, 1.0 - res / (tot + EPS)))
    return float(np.mean(scores)) if scores else 0.0


def stability(fit_fn, X: np.ndarray, n_boot: int = 25, seed: int = 0) -> float:
    """Mean cosine between the full-data membership matrix and bootstrap-resampled ones,
    after matching themes across runs by the Hungarian algorithm."""
    from scipy.optimize import linear_sum_assignment
    base = fit_fn(X)["S"]
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    out = []
    for _ in range(n_boot):
        keep = np.sort(rng.choice(n, int(0.85 * n), replace=False))
        S = fit_fn(X[keep])["S"]
        B = base[keep]
        k = min(S.shape[1], B.shape[1])
        C = np.zeros((B.shape[1], S.shape[1]))
        for a in range(B.shape[1]):
            for b in range(S.shape[1]):
                C[a, b] = np.dot(B[:, a], S[:, b]) / (
                    np.linalg.norm(B[:, a]) * np.linalg.norm(S[:, b]) + EPS)
        r, c = linear_sum_assignment(-C)
        out.append(float(C[r, c].mean()))
    return float(np.mean(out)) if out else 0.0


def spectral_coherence(X: np.ndarray, S: np.ndarray) -> float:
    """Mean membership-weighted cosine of CSMs to their dominant theme's weighted centroid."""
    N = X / (np.linalg.norm(X, axis=1, keepdims=True) + EPS)
    K = S.shape[1]
    out = []
    for k in range(K):
        w = S[:, k]
        if w.sum() < EPS:
            continue
        c = (w[:, None] * X).sum(0)
        c /= (np.linalg.norm(c) + EPS)
        out.append(float((w * (N @ c)).sum() / (w.sum() + EPS)))
    return float(np.mean(out)) if out else 0.0


def calibration_ece(S: np.ndarray, X: np.ndarray, themes: np.ndarray,
                    n_bins: int = 8) -> float:
    """Expected calibration error of the top membership against per-CSM reconstruction quality.

    A membership of 0.9 should mean the theme actually explains that CSM well. This is what
    makes a confidence honest rather than decorative.
    """
    conf = S.max(axis=1)
    top = S.argmax(axis=1)
    acc = np.zeros(len(conf))
    for i, k in enumerate(top):
        t = themes[k]
        c = float(np.dot(X[i], t) / (np.dot(t, t) + EPS))
        acc[i] = max(0.0, 1.0 - ((X[i] - c * t) ** 2).sum() / ((X[i] ** 2).sum() + EPS))
    edges = np.linspace(conf.min(), conf.max() + 1e-9, n_bins + 1)
    ece = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        m = (conf >= a) & (conf < b)
        if m.sum() == 0:
            continue
        ece += m.mean() * abs(conf[m].mean() - acc[m].mean())
    return float(ece)


def membership_sparsity(S: np.ndarray) -> float:
    """Mean mass in each CSM's top two themes. Sparse membership is a contract requirement."""
    top2 = np.sort(S, axis=1)[:, -2:].sum(axis=1)
    return float(top2.mean())


def compression(M: int, K: int) -> float:
    return float(K / max(M, 1))       # normalised outside; smaller K = more compression


def membership_degenerate(S: np.ndarray) -> dict:
    """Is the membership matrix actually using its themes?

    Reconstruction quality does not detect this. A run where one theme is the top theme for
    every CSM scored 0.497 information retained and 0.964 stability — stability is trivially
    high when the answer never changes — and was selected. What it produced was a single
    community, kNN agreement 1.000 against a chance of 1.000, and an adjusted mutual
    information of exactly zero. The degeneracy is in `S`, so `S` is where it must be caught.
    """
    mass = S.sum(axis=0)
    share = mass / (mass.sum() + EPS)
    eff = float(np.exp(-(share[share > 0] * np.log(share[share > 0])).sum()))
    top = S.argmax(axis=1)
    used = len(set(top.tolist()))
    bad = (share.max() > DEGENERATE_MAX_THEME_SHARE or eff < DEGENERATE_MIN_EFFECTIVE_K
           or used < max(2, S.shape[1] // 2))
    return {"max_theme_share": float(share.max()), "effective_K": eff,
            "n_themes_ever_dominant": int(used), "degenerate": bool(bad)}


def family_specificity(all_adms: list[list[dict]]) -> dict[str, float]:
    """How discriminating each mode family is, across every theme in the sweep.

    CH2/CH3 scissoring near 1440 cm-1 is the strongest band in most biological Raman spectra,
    so under raw prominence it becomes a dominant family of almost every theme — and every
    theme is then named "aliphatic chain + something". That is precisely the trap the phase is
    warned about: **shared CH stretching is not lipid biology.** Weighting each family by the
    inverse of how often it dominates makes a theme's identity what distinguishes it rather
    than what it shares with everything.
    """
    from collections import Counter
    cnt = Counter()
    total = 0
    for adms in all_adms:
        for a in adms:
            for f in a.get("mode_families", []):
                cnt[f] += 1
            total += 1
    return {f: float(np.log((total + 1) / (c + 1)) + 1.0) for f, c in cnt.items()}


def theme_set_distinct(themes: np.ndarray, adms: list[dict]) -> dict:
    """Are the K themes distinguishable chemistry, or the same chemistry more than once?

    Two themes carrying the same pair of dominant mode families AND a spectral cosine above
    `DISTINCTNESS_MAX_COSINE` are one theme described twice. The requirement is label-free —
    it compares the theme spectra and their own band assignments — and it operationalises the
    standard the phase is held to: the smallest *chemically meaningful* set. Without it, K = 9
    was selected with two pairs of themes carrying identical names.
    """
    N = themes / (np.linalg.norm(themes, axis=1, keepdims=True) + EPS)
    C = N @ N.T
    dup = []
    for i in range(len(adms)):
        for j in range(i + 1, len(adms)):
            same = set(adms[i]["dominant_families"]) == set(adms[j]["dominant_families"])
            if same and C[i, j] >= DISTINCTNESS_MAX_COSINE:
                dup.append((i, j, float(C[i, j])))
            elif same:
                dup.append((i, j, float(C[i, j])))
    return {"distinct": len(dup) == 0, "duplicate_pairs": dup,
            "max_cosine": float(C[np.triu_indices(len(adms), 1)].max()) if len(adms) > 1 else 0.0}


def composite(rows: list[dict]) -> np.ndarray:
    """Min-max normalise each criterion across the sweep, apply direction, weight, sum."""
    out = np.zeros(len(rows))
    for crit, (w, direction) in CRITERIA.items():
        v = np.array([r[crit] for r in rows], float)
        span = v.max() - v.min()
        z = np.full_like(v, 0.5) if span < EPS else (v - v.min()) / span
        out += w * (z if direction > 0 else 1.0 - z)
    return out


def select_K(rows: list[dict], tolerance: float = PLATEAU_TOLERANCE) -> dict:
    """Smallest admissible `K` on the contiguous Pareto plateau.

    Admissibility is applied first and as a veto, per the pre-registered rule: an inadmissible
    `K` is rejected however well it scores. The plateau is the contiguous run containing the
    maximum, carried over from the Phase 01 `k_c` correction where a non-contiguous reading
    selected a degenerate answer.
    """
    rows = [r for r in rows if not r.get("degenerate", False)]
    if not rows:
        return {"K": None, "status": "FAIL",
                "rationale": "every K for this model is degenerate — the fit retains no "
                             "information about the CSM spectra"}
    order = np.argsort([r["K"] for r in rows])
    rows = [rows[i] for i in order]
    comp = composite(rows)
    adm = np.array([r["chemically_admissible"] and r.get("themes_distinct", True)
                    for r in rows], bool)
    if not adm.any():
        return {"K": None, "status": "FAIL",
                "rationale": "no K in the sweep is both chemically admissible and composed of "
                             "distinguishable themes"}
    masked = np.where(adm, comp, -np.inf)
    peak = int(np.argmax(masked))
    lo = peak
    while lo - 1 >= 0 and adm[lo - 1] and comp[lo - 1] >= comp[peak] - tolerance:
        lo -= 1
    hi = peak
    while hi + 1 < len(comp) and adm[hi + 1] and comp[hi + 1] >= comp[peak] - tolerance:
        hi += 1
    return {"K": int(rows[lo]["K"]), "peak_K": int(rows[peak]["K"]),
            "plateau": [int(rows[lo]["K"]), int(rows[hi]["K"])],
            "composite": float(comp[lo]), "status": "PASS",
            "n_admissible": int(adm.sum()), "n_swept": len(rows),
            "rationale": (f"{int(adm.sum())} of {len(rows)} K values are chemically "
                          f"admissible; the composite peaks at K={rows[peak]['K']} "
                          f"({comp[peak]:.4f}); the contiguous admissible plateau within "
                          f"{tolerance} spans K={rows[lo]['K']}–{rows[hi]['K']}; smallest "
                          f"selected")}
