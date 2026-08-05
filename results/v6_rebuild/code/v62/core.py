"""GAIRA V6.2 — semantic hierarchy core.

PHILOSOPHY (fixed before any number was computed)
-------------------------------------------------
GAIRA is a biochemical reasoning engine, not a molecular classifier. V6.2 optimises
MAXIMUM BIOCHEMICAL ABSTRACTION subject to MINIMUM INFORMATION LOSS. Priority order:

    1  interpretability
    2  information retention
    3  recoverability
    4  analyte accuracy   (secondary; a modest loss is acceptable)

WHAT IS FROZEN
--------------
The atlas (fingerprint 09ed804a…) and the V6 MSS layer. V6.2 reads M (24 x 18) and the
motif definitions as INPUTS and never rewrites them. Everything new lives above MSS.

THE V6.2 CHAIN
--------------
    coord (24)  --M-->  mss (17)  --S-->  soft theme (K)  -->  BSV  -->  domain

S is a SOFT membership: non-negative, rows sum to 1, sparse. A motif may belong to
several themes, because some biochemical motifs genuinely are shared (protein amide
III overlaps the saccharide C-O region). V6 forced a hard partition; V6.2 represents
the overlap instead of suppressing it.
"""
from __future__ import annotations
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

REPO = Path("/Users/surajpg/projects/GAIRA")
V6 = REPO / "results/v6_rebuild"
CANON = "09ed804a40836f4a05a91ba10900cded"

# documented, pre-stated soft-membership parameters (not tuned to any outcome)
SOFT_TEMPERATURE = 0.12      # softmax temperature on motif->theme-centroid cosine
SOFT_FLOOR = 0.02            # weights below this are dropped, then rows renormalised
EPS = 1e-12


# ── chemical superclasses: fixed on chemistry, reused verbatim from V6 ──
SUPERCLASS = {
    "nucleobase_purine": "nucleic", "nucleobase_pyrimidine": "nucleic",
    "phosphate_ester": "BRIDGING",
    "aromatic_sidechain": "protein", "polypeptide": "protein", "free_amino_acid": "protein",
    "fatty_acid": "lipid", "acylglycerol": "lipid", "sterol": "lipid",
    "monosaccharide": "carbohydrate", "polysaccharide": "carbohydrate",
    "organic_acid": "metabolite", "sulfur_metabolite": "metabolite",
    "tetrapyrrole": "cofactor", "redox_cofactor": "cofactor", "polyene": "cofactor",
}


def admissible(groups, class_of):
    for g in groups:
        sc = {SUPERCLASS.get(class_of[m], "?") for m in g}
        sc.discard("BRIDGING")
        if len(sc) > 1:
            return False
    return True


# ── loading ─────────────────────────────────────────────────────────────────
@dataclass
class V62Context:
    eng: object
    v6: object
    H: np.ndarray                 # (24, 676) frozen basis
    grid: np.ndarray
    M: np.ndarray                 # (24, 17) frozen V6 motif map, biochemical only
    motif_ids: list
    motifs: list
    analytes: list
    families: np.ndarray
    zA: np.ndarray                # (n_analytes, 24) mean coordinates
    A: np.ndarray                 # (n_analytes, 17) motif activations
    Zs: np.ndarray                # (n_spectra, 24) per-spectrum coordinates
    spec_analyte: np.ndarray
    corpusX: np.ndarray           # (n_analytes, 676) mean preprocessed spectra
    class_of: dict                # motif -> chemical_class

    @property
    def n_motifs(self):
        return len(self.motif_ids)


def load_context():
    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(V6 / "code"))
    from gaira.engine import GAIRAEngine
    from gaira.foundation import dataset as DS
    from gaira.foundation.families_raman import family_of
    from v6_semantic.mss_v6 import MSSLayerV6

    eng = GAIRAEngine()
    assert eng.atlas.meta["fingerprint"] == CANON, "FROZEN ATLAS CHANGED"
    H, grid = eng.atlas.components, eng.atlas.grid
    v6 = MSSLayerV6(V6 / "artifacts/mss_motifs_v6.yaml", eng.builder.reg, H, grid)
    bio = [i for i, m in enumerate(v6.motifs) if not m.non_biochemical]
    motif_ids = [v6.motifs[i].id for i in bio]
    motifs = [v6.motifs[i] for i in bio]

    corpus = DS.load_reference_corpus()
    Zs = eng.atlas.coordinates(corpus.X)
    ra = corpus.meta.analyte.values
    analytes = sorted(set(ra))
    zA = np.array([Zs[ra == a].mean(0) for a in analytes])
    A = np.array([v6.activate(z) for z in zA])[:, bio]
    corpusX = np.array([np.nan_to_num(corpus.X[ra == a]).mean(0) for a in analytes])
    return V62Context(eng=eng, v6=v6, H=H, grid=grid, M=v6.M[:, bio], motif_ids=motif_ids,
                      motifs=motifs, analytes=analytes,
                      families=np.array([family_of(a) for a in analytes]),
                      zA=zA, A=A, Zs=Zs, spec_analyte=ra, corpusX=corpusX,
                      class_of={m.id: m.chemical_class for m in motifs})


# ── soft theme membership ───────────────────────────────────────────────────
def motif_profiles(A):
    """L2-normalised activation profile of each motif across the corpus."""
    P = A.T.astype(float)                       # (n_motifs, n_analytes)
    return P / (np.linalg.norm(P, axis=1, keepdims=True) + EPS)


def soft_membership(A, groups, motif_ids, temperature=SOFT_TEMPERATURE, floor=SOFT_FLOOR):
    """S (n_motifs x K): non-negative, rows sum to 1, sparse.

    A motif's membership in theme t is a softmax over the cosine between the motif's
    corpus activation profile and the theme centroid (the mean profile of the theme's
    seed motifs). A motif that resembles two theme centroids is given to both — that
    is the point.
    """
    P = motif_profiles(A)
    idx = {m: i for i, m in enumerate(motif_ids)}
    C = np.zeros((len(groups), P.shape[1]))
    for t, g in enumerate(groups):
        C[t] = P[[idx[m] for m in g if m in idx]].mean(0)
    C = C / (np.linalg.norm(C, axis=1, keepdims=True) + EPS)
    sim = P @ C.T                                        # (n_motifs, K) cosine
    S = np.exp((sim - sim.max(axis=1, keepdims=True)) / temperature)
    S = S / S.sum(axis=1, keepdims=True)
    S[S < floor] = 0.0
    S = S / (S.sum(axis=1, keepdims=True) + EPS)
    return S, sim


def learn_membership(A, Y, n_iter=400, l1=0.0, seed=0):
    """Learn S by projected gradient: min ||Y - A S||_F  s.t. S >= 0, rows sum to 1.

    Y is the (n_analytes x K) target theme indicator. Deterministic (fixed init + step).
    """
    n_m, K = A.shape[1], Y.shape[1]
    S = np.full((n_m, K), 1.0 / K)
    L = np.linalg.norm(A, 2) ** 2 + EPS
    for _ in range(n_iter):
        G = A.T @ (A @ S - Y) + l1
        S = S - G / L
        S = np.clip(S, 0, None)
        S = S / (S.sum(axis=1, keepdims=True) + EPS)      # simplex projection (rows)
    S[S < SOFT_FLOOR] = 0.0
    S = S / (S.sum(axis=1, keepdims=True) + EPS)
    return S


# ── uncertainty ─────────────────────────────────────────────────────────────
def norm_entropy(p, axis=-1):
    p = np.clip(np.asarray(p, float), 0, None)
    s = p.sum(axis=axis, keepdims=True)
    p = np.divide(p, s, out=np.zeros_like(p), where=s > EPS)
    n = p.shape[axis]
    if n <= 1:
        return np.zeros(p.shape[:-1]) if p.ndim > 1 else 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -np.sum(np.where(p > 0, p * np.log(p), 0.0), axis=axis)
    return h / np.log(n)


def theme_posterior(mss, S):
    """Theme posterior + uncertainty for one or many motif-activation vectors."""
    X = np.atleast_2d(np.clip(np.asarray(mss, float), 0, None))
    T = X @ S
    tot = T.sum(axis=1, keepdims=True)
    Pp = np.divide(T, tot, out=np.zeros_like(T), where=tot > EPS)
    srt = np.sort(Pp, axis=1)[:, ::-1]
    margin = srt[:, 0] - (srt[:, 1] if Pp.shape[1] > 1 else 0.0)
    H = norm_entropy(Pp, axis=1)
    conf = srt[:, 0] * (1.0 - H)                      # peakedness x sharpness
    return dict(theme=T, posterior=Pp, entropy=H, margin=margin, confidence=conf,
                top=np.argmax(Pp, axis=1))


# ── linear (Bayesian) uncertainty propagation ───────────────────────────────
def propagate(cov_coord, M, S):
    """Var through two linear maps: coord -> mss -> theme. Returns the two covariances."""
    cov_mss = M.T @ cov_coord @ M
    cov_theme = S.T @ cov_mss @ S
    return cov_mss, cov_theme


def replicate_covariance(Zs, spec_analyte, analyte):
    """Empirical covariance of an analyte's replicate coordinates (the measurement noise)."""
    Z = Zs[spec_analyte == analyte]
    if len(Z) < 2:
        return np.zeros((Zs.shape[1], Zs.shape[1]))
    return np.cov(Z.T)


# ── information measures ────────────────────────────────────────────────────
def explained_variance(X, Xhat):
    ss_res = float(np.sum((X - Xhat) ** 2))
    ss_tot = float(np.sum((X - X.mean(0)) ** 2))
    return 1.0 - ss_res / (ss_tot + EPS)


def reconstruct_from_theme(A, S):
    """Best linear reconstruction of the motif activations from the theme vector."""
    T = A @ S
    W, *_ = np.linalg.lstsq(T, A, rcond=None)
    return T @ W


def mutual_information(labels, X, n_bins=8, seed=0):
    """I(theme representation ; chemical family), estimated by per-dimension binning.

    Discretises each theme dimension into quantile bins and sums the plug-in MI of the
    joint (bin, label) table, normalised by H(label). Deterministic.
    """
    lab = np.asarray(labels)
    uniq = {v: i for i, v in enumerate(sorted(set(lab.tolist())))}
    y = np.array([uniq[v] for v in lab])
    Hy = _entropy(np.bincount(y))
    if Hy <= 0:
        return 0.0
    mi = 0.0
    for d in range(X.shape[1]):
        col = X[:, d]
        qs = np.unique(np.quantile(col, np.linspace(0, 1, n_bins + 1)[1:-1]))
        b = np.digitize(col, qs)
        joint = np.zeros((b.max() + 1, len(uniq)))
        for bi, yi in zip(b, y):
            joint[bi, yi] += 1
        mi += _mi_table(joint)
    return float(mi / (X.shape[1] * Hy))          # mean normalised MI per dimension


def _entropy(counts):
    p = np.asarray(counts, float)
    p = p[p > 0] / p.sum()
    return float(-np.sum(p * np.log(p)))


def _mi_table(joint):
    j = np.asarray(joint, float)
    n = j.sum()
    if n <= 0:
        return 0.0
    p = j / n
    px = p.sum(1, keepdims=True); py = p.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(p > 0, p * np.log(p / (px * py + EPS) + EPS), 0.0)
    return float(t.sum())


def kl_divergence(P, Q):
    """Mean row-wise KL(P || Q) between two non-negative activation matrices."""
    P = np.clip(P, 0, None); Q = np.clip(Q, 0, None)
    P = P / (P.sum(1, keepdims=True) + EPS)
    Q = Q / (Q.sum(1, keepdims=True) + EPS)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(P > 0, P * np.log((P + EPS) / (Q + EPS)), 0.0)
    return float(t.sum(1).mean())


# ── persistence helpers ─────────────────────────────────────────────────────
def art(name):
    p = V6 / "artifacts" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def tab(name):
    p = V6 / "tables" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def dump_json(obj, name):
    art(name).write_text(json.dumps(obj, indent=2, default=str))


def dump_yaml(obj, name):
    art(name).write_text(yaml.safe_dump(obj, sort_keys=False, default_flow_style=False))
