"""GAIRA V7 — Phase 02: the seven edge features of the Consensus Spectral Graph.

One function per feature, each returning a full `n × n` matrix over the pooled LSM set. The
features are deliberately partly independent: a merge supported by only one of them is not a
merge, and the geometric-mean edge weight in `graph.py` enforces that continuously.

Everything here is a rule from `results/v7_rebuild/phase02/config/phase02_preregistration_v1.md`.
Constants are frozen there; nothing in this module may be tuned against an observed result.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import nnls
from scipy.stats import spearmanr

# ── frozen constants (pre-registration §10) ──────────────────────────────────
BAND_TOL_CM = 10.0
BAND_HALFWIDTH_CM = 8.0
MIN_ACTIVATION = 0.05
FEATURES = ("spectral_cosine", "band_overlap", "peak_agreement",
            "bootstrap_cooccurrence", "activation_cooccurrence",
            "provenance_overlap", "substitutability")
EPS = 1e-12


def _unit(M: np.ndarray) -> np.ndarray:
    return M / (np.linalg.norm(M, axis=-1, keepdims=True) + EPS)


# ── feature 1 — spectral cosine ──────────────────────────────────────────────
def spectral_cosine(H: np.ndarray) -> np.ndarray:
    """Overall shape agreement on the full grid.

    The trap feature, included because it is informative and excluded from dominance because
    it is not sufficient: biological Raman spectra share a great deal of broad structure and
    chemically distinct motifs routinely reach 0.7 on shared CH and skeletal modes.
    """
    N = _unit(H)
    return np.clip(N @ N.T, 0.0, 1.0)


# ── feature 2 — diagnostic-band overlap ──────────────────────────────────────
def _band_mask(bands: list[float], grid: np.ndarray,
               halfwidth: float = BAND_HALFWIDTH_CM) -> np.ndarray:
    m = np.zeros(grid.size, dtype=bool)
    for b in bands:
        m |= np.abs(grid - b) <= halfwidth
    return m


def band_prominence_profile(H: np.ndarray, bands: list[list[float]], grid: np.ndarray,
                            vocab_step: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    """Each motif as a sparse vector of peak *prominences* over a shared band vocabulary.

    The vocabulary is every band any motif declares, quantised to `vocab_step`. A motif's
    entry at a vocabulary position is the prominence of its peak there, or zero if it has no
    peak there — so the representation contains the diagnostic peaks and nothing else.
    """
    from scipy.signal import peak_prominences, find_peaks

    all_b = sorted({round(b / vocab_step) * vocab_step for bl in bands for b in bl})
    vocab = np.array(all_b, float)
    Pm = np.zeros((H.shape[0], vocab.size))
    for i, h in enumerate(H):
        x = h / (h.max() + EPS)
        idx, _ = find_peaks(x)
        if idx.size == 0:
            continue
        prom = peak_prominences(x, idx)[0]
        for k, b in enumerate(vocab):
            near = np.where(np.abs(grid[idx] - b) <= vocab_step)[0]
            if near.size:
                Pm[i, k] = prom[near].max()
    return Pm, vocab


def band_overlap(H: np.ndarray, bands: list[list[float]], grid: np.ndarray) -> np.ndarray:
    """Agreement of diagnostic band *prominences* — intensity-aware, pedestal-free.

    An earlier version masked the full spectra to the union of both motifs' ±8 cm-1 band
    windows and took the cosine inside the mask. It correlated with `spectral_cosine` at
    **0.978** over all 1225 pairs — the two were one line of evidence carrying 0.45 of the edge
    weight between them, which defeats the entire multi-evidence design. The cause is that a
    motif's energy is concentrated at its peaks, so a mask over those peaks reproduces the
    full-spectrum cosine almost exactly.

    Comparing peak prominences instead removes the shared broad pedestal, which is precisely
    the structure the architecture warns about: "two motifs can have high global cosine from
    shared broad structure while disagreeing on every diagnostic band". A pedestal contributes
    no prominence, so it cannot contribute agreement here.
    """
    Pm, _ = band_prominence_profile(H, bands, grid)
    N = _unit(Pm)
    return np.clip(N @ N.T, 0.0, 1.0)


# ── feature 3 — peak-position agreement ──────────────────────────────────────
def peak_agreement(bands: list[list[float]], tol: float = BAND_TOL_CM) -> np.ndarray:
    """Position-only, intensity-free.

    Peak position is excitation-invariant; relative intensity is not. A feature built purely
    on positions therefore survives the corpus's nine excitation domains in a way that any
    intensity-weighted measure does not.
    """
    n = len(bands)
    out = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            bi, bj = np.asarray(bands[i], float), np.asarray(bands[j], float)
            if bi.size == 0 or bj.size == 0:
                out[i, j] = out[j, i] = 0.0
                continue
            # greedy one-to-one matching, closest pairs first
            D = np.abs(bi[:, None] - bj[None, :])
            matched, used_i, used_j = 0, set(), set()
            for a, b in sorted(np.ndindex(D.shape), key=lambda p: D[p]):
                if D[a, b] > tol:
                    break
                if a in used_i or b in used_j:
                    continue
                used_i.add(a); used_j.add(b); matched += 1
            out[i, j] = out[j, i] = 2.0 * matched / (bi.size + bj.size)
    return out


# ── feature 4 — bootstrap co-occurrence ──────────────────────────────────────
def bootstrap_cooccurrence(resampled: list[dict[int, np.ndarray]], n: int) -> np.ndarray:
    """Is the pair relationship a property of the data, or of one particular fit?

    `resampled[r][i]` is motif `i` as recovered in resample `r`, absent if it was not
    recovered. The feature is

        (fraction of resamples recovering both) x (mean resampled pair cosine over those)

    which is continuous and introduces no threshold of its own. The literal contract reading —
    "how often do both appear" — is degenerate on this corpus, where every one of the 50 LSMs
    was retained above a 0.60 stability floor and nearly all pairs would co-occur in nearly
    every resample. Multiplying by the resampled agreement restores the discrimination.
    """
    R = len(resampled)
    out, cnt = np.zeros((n, n)), np.zeros((n, n))
    for rep in resampled:
        keys = sorted(rep)
        for a_idx, i in enumerate(keys):
            for j in keys[a_idx + 1:]:
                hi, hj = rep[i], rep[j]
                c = float(hi @ hj / (np.linalg.norm(hi) * np.linalg.norm(hj) + EPS))
                out[i, j] += max(0.0, c); out[j, i] += max(0.0, c)
                cnt[i, j] += 1; cnt[j, i] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_cos = np.where(cnt > 0, out / np.maximum(cnt, 1), 0.0)
    frac = cnt / max(R, 1)
    F = frac * mean_cos
    np.fill_diagonal(F, 1.0)
    return np.clip(F, 0.0, 1.0)


# ── activation matrix (shared input to features 5 and 6) ─────────────────────
def activation_matrix(X: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Independent non-negative scalar projection: `a_i(x) = max(0, <x,h_i>/||h_i||^2)`.

    NOT joint NNLS. Joint NNLS over 50 non-orthogonal motifs makes near-duplicates split each
    other's mass and appear anticorrelated — it would penalise exactly the pairs Phase 02
    exists to find. `joint_activation_matrix` computes the joint version for the sensitivity
    check, which is where it belongs.
    """
    denom = (H * H).sum(axis=1) + EPS
    return np.clip(X @ H.T / denom, 0.0, None)


def joint_activation_matrix(X: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Joint NNLS against the full pooled dictionary — reported as a sensitivity check only."""
    A = np.zeros((X.shape[0], H.shape[0]))
    for i, x in enumerate(X):
        A[i] = nnls(H.T, x)[0]
    return A


def to_molecule_level(A: np.ndarray, canonical_id: np.ndarray,
                      weight: np.ndarray | None = None) -> tuple[np.ndarray, list[str]]:
    """Collapse spectrum-level activations to one row per canonical molecule.

    Molecule level, not spectrum level: correlating over spectra would let a molecule with
    three replicates outvote a molecule with one, which is limitation L-01 reappearing inside
    Phase 02's own evidence.
    """
    ids = sorted(set(canonical_id))
    out = np.zeros((len(ids), A.shape[1]))
    for r, cid in enumerate(ids):
        m = canonical_id == cid
        w = None if weight is None else weight[m]
        out[r] = np.average(A[m], axis=0, weights=w)
    return out, ids


# ── feature 5 — activation co-occurrence ─────────────────────────────────────
def activation_cooccurrence(A_mol: np.ndarray) -> np.ndarray:
    """Do the two motifs respond to the same molecules?

    Spearman across molecules, so the answer depends on the *pattern* of preference and not on
    activation magnitude — two motifs of similar shape driven by different chemistry separate
    here even though their cosine is high.
    """
    n = A_mol.shape[1]
    out = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = A_mol[:, i], A_mol[:, j]
            if a.std() < EPS or b.std() < EPS:
                v = 0.0
            else:
                v = float(spearmanr(a, b).statistic)
                if not np.isfinite(v):
                    v = 0.0
            out[i, j] = out[j, i] = max(0.0, v)
    return out


# ── feature 6 — provenance overlap, within-class discounted ──────────────────
def provenance_overlap(A_mol: np.ndarray, classes: list[str], mol_class: list[str],
                       min_activation: float = MIN_ACTIVATION) -> np.ndarray:
    """Shared supporting evidence, with the class prior discounted (risk R-01).

    Support is the set of molecules whose normalised activation clears `min_activation` under
    projection against the whole reference set — not the class-local support, which is disjoint
    across classes by construction and would make this feature identically zero for every
    cross-class pair.

    The discount is against the Jaccard expected if both supports were drawn at random from the
    pair's own molecule pool: `max(0, (J - E[J]) / (1 - E[J]))`. Two motifs from one class draw
    from one pool and so start with a high null; without this the feature would re-encode the
    class partition and Phase 02 would rediscover the classes it began with.
    """
    n = A_mol.shape[1]
    S = activation_shares(A_mol, classes)
    supports = [set(np.where(S[:, i] >= min_activation)[0]) for i in range(n)]
    mol_class = np.asarray(mol_class)
    out = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            si, sj = supports[i], supports[j]
            if not si or not sj:
                out[i, j] = out[j, i] = 0.0
                continue
            inter, union = len(si & sj), len(si | sj)
            J = inter / union
            # null pool: molecules of the two motifs' classes
            pool = np.where((mol_class == classes[i]) | (mol_class == classes[j]))[0]
            P = max(len(pool), 1)
            a, b = len(si), len(sj)
            e_inter = a * b / P                        # expected |A ∩ B| for random subsets
            e_union = max(a + b - e_inter, EPS)
            e_j = min(e_inter / e_union, 1.0 - 1e-6)
            out[i, j] = out[j, i] = max(0.0, (J - e_j) / (1.0 - e_j))
    return out


# ── feature 7 — reconstruction substitutability ──────────────────────────────
def substitutability(H: np.ndarray, class_of: list[str], X: np.ndarray,
                     row_class: np.ndarray, support: list[set[int]]) -> np.ndarray:
    """Can motif j actually do motif i's job?

    The symmetric feature is the *minimum* of the two directions: a merge claim is only as
    strong as its weaker half. This is the sharpest falsification channel available — a pair
    can agree on shape, bands, positions and activations and still fail here, which is what
    "same phenomenon" ultimately has to mean.

    An earlier version swapped `h_j` for `h_i` inside `i`'s *full* class dictionary and compared
    explained variance. It had almost no power — median 0.971 over all 1225 pairs, and no pair
    below 0.108 — because a 10-motif class dictionary absorbs the loss of any single member: the
    other nine simply take up the slack. That measures dictionary redundancy, not
    substitutability.

    The marginal formulation asks the question that was intended. Let `D_-i` be the class
    dictionary without `h_i`. Then

        gain_i = EV(D_-i + h_i) - EV(D_-i)      what i uniquely contributes
        gain_j = EV(D_-i + h_j) - EV(D_-i)      what j would contribute in i's place
        s_i->j = gain_j / gain_i

    This isolates the contribution instead of hiding it in the ensemble, and it is not a
    restatement of shape similarity: it asks whether `j` supplies *the same missing piece* that
    `i` supplies, given everything else the class already explains.
    """
    n = H.shape[0]
    classes = sorted(set(class_of))
    idx_of_class = {c: [i for i in range(n) if class_of[i] == c] for c in classes}
    out = np.eye(n)

    def ev(rows: np.ndarray, D: np.ndarray) -> float:
        if rows.size == 0 or D.shape[0] == 0:
            return 0.0
        tot = res = 0.0
        for x in rows:
            c = nnls(D.T, x)[0]
            res += float(((x - c @ D) ** 2).sum())
            tot += float((x ** 2).sum())
        return max(0.0, 1.0 - res / (tot + EPS))

    cache: dict[int, tuple[np.ndarray, np.ndarray, float, float]] = {}
    for i in range(n):
        rows = X[sorted(support[i])] if support[i] else X[[]]
        members = idx_of_class[class_of[i]]
        rest = [m for m in members if m != i]
        D_rest = H[rest] if rest else np.zeros((0, H.shape[1]))
        base = ev(rows, D_rest)
        full = ev(rows, H[members])
        cache[i] = (rows, D_rest, base, full)

    for i in range(n):
        rows_i, D_rest, base_i, full_i = cache[i]
        gain_i = full_i - base_i
        for j in range(n):
            if i == j:
                continue
            if gain_i <= 1e-6:
                # i adds nothing its own class does not already explain; the marginal ratio is
                # undefined, so fall back to what a single motif can do on its own.
                a = ev(rows_i, H[[i]])
                b = ev(rows_i, H[[j]])
                out[i, j] = min(1.0, b / (a + EPS)) if a > 1e-6 else 0.0
                continue
            gain_j = ev(rows_i, np.vstack([D_rest, H[j]])) - base_i
            out[i, j] = float(np.clip(gain_j / gain_i, 0.0, 1.0))

    S = np.minimum(out, out.T)
    np.fill_diagonal(S, 1.0)
    return np.clip(S, 0.0, 1.0)


def activation_shares(A: np.ndarray, class_of: list[str]) -> np.ndarray:
    """Activation share, normalised **within each motif's own class**.

    `MIN_ACTIVATION = 0.05` was pre-registered as "a molecule activates a motif above this
    normalised share", carried over from Phase 01 where shares are taken within a class of at
    most ten motifs. Two wrong translations of that constant were tried before this one:

    - **peak-normalised** (`a / max_m a`): nearly every molecule clears 5% of a motif's peak,
      so supports came out near-total and `provenance_overlap` collapsed to a near-binary
      feature (median 1.000, 75th percentile 1.000);
    - **pooled row-normalised** (share across all 50 motifs): uniform share is 1/50 = 0.02, so
      a 0.05 floor keeps only a molecule's top two motifs — supports came out near-empty,
      `provenance_overlap` fell to a mean of 0.003 and `substitutability` to a median of 0.

    Normalising within class preserves what the constant means (a share among the motifs
    competing to describe one chemistry) while still being defined across classes, because
    every molecule is projected against every class's motifs.
    """
    cls = np.asarray(class_of)
    S = np.zeros_like(A)
    for c in np.unique(cls):
        m = cls == c
        S[:, m] = A[:, m] / (A[:, m].sum(axis=1, keepdims=True) + EPS)
    return S


def support_rows(A: np.ndarray, class_of: list[str],
                 min_activation: float = MIN_ACTIVATION) -> list[set[int]]:
    """Spectrum rows that activate each motif, used as the substitution test set."""
    S = activation_shares(A, class_of)
    return [set(np.where(S[:, i] >= min_activation)[0]) for i in range(A.shape[1])]
