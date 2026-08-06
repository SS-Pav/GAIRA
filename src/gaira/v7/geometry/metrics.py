"""GAIRA V7 — Phase 02.5: motif distance metrics and the benchmark that chooses between them.

Ten metrics, each answering a slightly different question about what makes two Raman motifs
"close". They are benchmarked on properties that matter spectroscopically — does the metric
care about amplitude it should ignore, does it survive a peak shifting by 4 cm-1, does it get
dragged around by broad generic structure — and then three are selected for three distinct
roles. Cosine is a candidate, not a default.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import jensenshannon, pdist, squareform
from scipy.stats import spearmanr, wasserstein_distance

EPS = 1e-12
METRICS = ("spectral_cosine", "pearson", "spearman", "euclidean_l2",
           "jensen_shannon", "wasserstein", "peak_set", "band_overlap",
           "activation_profile", "phase02_composite")


def _unit(M):
    return M / (np.linalg.norm(M, axis=-1, keepdims=True) + EPS)


def _prob(M):
    M = np.clip(np.asarray(M, float), 0, None)
    return M / (M.sum(axis=-1, keepdims=True) + EPS)


# ── the ten distances (all returned as DISTANCE, 0 = identical) ──────────────
def d_spectral_cosine(H, **kw):
    N = _unit(H)
    return np.clip(1.0 - N @ N.T, 0, None)


def d_pearson(H, **kw):
    return np.clip(1.0 - np.nan_to_num(np.corrcoef(np.asarray(H, float))), 0, None)


def _rank_corr(M: np.ndarray) -> np.ndarray:
    """Row-wise Spearman matrix. scipy returns a scalar for two rows, which silently produces
    a 1x1 matrix and an out-of-bounds read downstream — so the pairwise form is built here."""
    M = np.asarray(M, float)
    n = M.shape[0]
    if n == 2:
        r = float(np.nan_to_num(spearmanr(M[0], M[1]).statistic))
        return np.array([[1.0, r], [r, 1.0]])
    return np.nan_to_num(np.atleast_2d(spearmanr(M.T).statistic))


def d_spearman(H, **kw):
    return np.clip(1.0 - _rank_corr(H), 0, None)


def d_euclidean_l2(H, **kw):
    return squareform(pdist(_unit(H), metric="euclidean"))


def d_jensen_shannon(H, **kw):
    """Jensen-Shannon distance, computed directly rather than via scipy.

    `scipy.spatial.distance.jensenshannon` returns inf on these spectra — two Gaussians with
    80% mass overlap come back as maximally distant — because the underflowed tail bins produce
    0*log(0/0). Computed with an explicit floor the same pair gives a small, correct distance.
    """
    P = np.clip(_prob(H), 1e-15, None)
    P = P / P.sum(axis=1, keepdims=True)
    n = P.shape[0]
    Dm = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            m = 0.5 * (P[i] + P[j])
            kl = lambda a, b: float(np.sum(a * (np.log2(a) - np.log2(b))))
            Dm[i, j] = Dm[j, i] = float(np.sqrt(max(0.0, 0.5 * kl(P[i], m) + 0.5 * kl(P[j], m))))
    return np.clip(np.nan_to_num(Dm), 0.0, 1.0)


def d_wasserstein(H, grid=None, **kw):
    """Earth-mover distance along the Raman shift axis.

    The only metric here that knows 1440 is *near* 1450 and far from 700. Every bin-wise
    metric treats a 10 cm-1 shift and a 900 cm-1 shift as equally different, which is wrong
    for spectra: peak position drifts with excitation, substituent and hydrogen bonding.
    """
    P = _prob(H)
    g = np.asarray(grid, float)
    span = float(g.max() - g.min()) or 1.0
    n = P.shape[0]
    Dm = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            Dm[i, j] = Dm[j, i] = float(wasserstein_distance(g, g, P[i], P[j]))
    # normalised by the GRID SPAN, a fixed constant — not by the matrix maximum. Normalising
    # by the max makes the scale depend on which motifs are in the set, so any two-motif probe
    # returns 1.0 by construction and the metric appears to have zero shift tolerance.
    return Dm / span


def d_peak_set(H=None, *, peak_vec=None, **kw):
    N = _unit(peak_vec)
    return np.clip(1.0 - N @ N.T, 0, None)


def d_band_overlap(H=None, *, band_vec=None, **kw):
    N = _unit(band_vec)
    return np.clip(1.0 - N @ N.T, 0, None)


def d_activation_profile(H=None, *, act=None, **kw):
    A = np.asarray(act, float)
    return np.clip(1.0 - _rank_corr(A.T if A.shape[0] != A.shape[1] else A), 0, None)


def d_phase02_composite(H=None, *, W=None, **kw):
    if W is None:                       # probe calls have no Phase 02 graph to consult
        n = 2 if H is None else np.asarray(H).shape[0]
        return np.zeros((n, n))
    Wm = np.asarray(W, float)
    return np.clip(1.0 - Wm - np.eye(Wm.shape[0]), 0, None)


DISPATCH = {
    "spectral_cosine": d_spectral_cosine, "pearson": d_pearson, "spearman": d_spearman,
    "euclidean_l2": d_euclidean_l2, "jensen_shannon": d_jensen_shannon,
    "wasserstein": d_wasserstein, "peak_set": d_peak_set, "band_overlap": d_band_overlap,
    "activation_profile": d_activation_profile, "phase02_composite": d_phase02_composite,
}


def all_distances(H, grid, peak_vec, band_vec, act, W) -> dict[str, np.ndarray]:
    kw = dict(grid=grid, peak_vec=peak_vec, band_vec=band_vec, act=act, W=W)
    out = {}
    for m in METRICS:
        Dm = np.asarray(DISPATCH[m](H, **kw), float)
        Dm = (Dm + Dm.T) / 2
        np.fill_diagonal(Dm, 0.0)
        out[m] = Dm
    return out


# Metrics that a synthetic-spectrum probe can evaluate. `activation_profile` and
# `phase02_composite` are functions of the corpus and of the Phase 02 graph, not of a pair of
# spectra, so probing them on invented spectra would produce a number with no meaning. They are
# reported as not-probeable rather than given a fabricated score.
PROBEABLE = ("spectral_cosine", "pearson", "spearman", "euclidean_l2", "jensen_shannon",
             "wasserstein", "peak_set", "band_overlap")


# ── probe behaviours ─────────────────────────────────────────────────────────
def _gauss(grid, c, w=12.0):
    return np.exp(-((grid - c) ** 2) / (2 * w ** 2))


def probe_amplitude(fn, grid, **kw) -> float:
    """A metric should not separate a motif from a scaled copy of itself. Lower is better."""
    a = _gauss(grid, 1000) + 0.4 * _gauss(grid, 1440)
    H = np.vstack([a, 3.0 * a])
    return float(np.asarray(fn(H, grid=grid, peak_vec=H, band_vec=H, act=H,
                              W=np.ones((2, 2))), float)[0, 1])


def probe_peak_shift(fn, grid, shift=6.0, **kw) -> float:
    """A 6 cm-1 shift is within normal Raman variation; a good metric should barely notice."""
    a = _gauss(grid, 1000) + 0.4 * _gauss(grid, 1440)
    b = _gauss(grid, 1000 + shift) + 0.4 * _gauss(grid, 1440 + shift)
    H = np.vstack([a, b])
    return float(np.asarray(fn(H, grid=grid, peak_vec=H, band_vec=H, act=H,
                              W=np.ones((2, 2))), float)[0, 1])


def probe_width(fn, grid, **kw) -> float:
    """Sharp vs broad band at the same position — genuinely different, should be non-zero."""
    H = np.vstack([_gauss(grid, 1100, 6.0), _gauss(grid, 1100, 60.0)])
    return float(np.asarray(fn(H, grid=grid, peak_vec=H, band_vec=H, act=H,
                              W=np.ones((2, 2))), float)[0, 1])


def probe_generic_background(fn, grid, **kw) -> float:
    """Two motifs with disjoint peaks on a shared broad pedestal.

    The pedestal is the generic Raman background every biological spectrum carries. A metric
    that reports these as close is measuring the background, not the chemistry. Higher is
    better here — the metric SHOULD separate them.
    """
    ped = _gauss(grid, 1150, 380.0)
    H = np.vstack([ped + 0.6 * _gauss(grid, 700, 6.0), ped + 0.6 * _gauss(grid, 1600, 6.0)])
    return float(np.asarray(fn(H, grid=grid, peak_vec=H, band_vec=H, act=H,
                              W=np.ones((2, 2))), float)[0, 1])


def scale_free_probes(fn, grid, observed_D: np.ndarray) -> dict:
    """Probe results expressed as a fraction of the metric's own median observed distance.

    Raw probe values are not comparable across metrics: a background separation of 0.106 under
    Euclidean and 0.006 under cosine says nothing until both are read against the scale each
    metric actually works on. Dividing by the median off-diagonal distance on the real motif
    set makes every probe a scale-free ratio, which is the only form in which they can be
    ranked against one another.
    """
    med = float(np.median(observed_D[~np.eye(observed_D.shape[0], dtype=bool)])) or 1.0
    return {"median_observed_distance": med,
            "amplitude_leakage": probe_amplitude(fn, grid) / med,
            "peak_shift_cost": probe_peak_shift(fn, grid) / med,
            "width_discrimination": probe_width(fn, grid) / med,
            "background_separation": probe_generic_background(fn, grid) / med}


# ── data-driven benchmark criteria ───────────────────────────────────────────
def knn_label_coherence(Dm: np.ndarray, labels: list[str], k: int = 5) -> float:
    """Fraction of a motif's k nearest neighbours sharing its label.

    **Evaluation only.** Labels are revealed after the geometry exists; they never enter a
    representation or a distance. A metric that scores high here is one whose neighbourhoods
    line up with curated chemistry — informative, but not the target, since a metric that
    merely reproduced the class partition would tell us nothing new.
    """
    lab = np.asarray(labels)
    n = len(lab)
    hits = []
    for i in range(n):
        nb = np.argsort(Dm[i])[1:k + 1]
        hits.append(float((lab[nb] == lab[i]).mean()))
    return float(np.mean(hits))


def bootstrap_stability(build_fn, n: int, n_boot: int = 30, k: int = 5, seed: int = 0) -> float:
    """Mean Jaccard of each motif's k-NN set under resampling of the spectral axis."""
    rng = np.random.default_rng(seed)
    base = build_fn(None)
    base_nb = [set(np.argsort(base[i])[1:k + 1]) for i in range(n)]
    js = []
    for _ in range(n_boot):
        Dm = build_fn(rng)
        for i in range(n):
            nb = set(np.argsort(Dm[i])[1:k + 1])
            js.append(len(nb & base_nb[i]) / len(nb | base_nb[i]))
    return float(np.mean(js))


def null_separation(Dm: np.ndarray, null_D: list[np.ndarray]) -> float:
    """How far the observed nearest-neighbour distances sit below the null's.

    Reported as a standardised effect: (null mean - observed mean) / null sd of the
    first-neighbour distance. A metric with no null separation is describing generic Raman
    statistics, whatever its neighbourhoods look like.
    """
    def nn1(M):
        M = M.copy()
        np.fill_diagonal(M, np.inf)
        return M.min(axis=1)
    obs = nn1(Dm).mean()
    nulls = np.array([nn1(N).mean() for N in null_D])
    return float((nulls.mean() - obs) / (nulls.std() + EPS))
