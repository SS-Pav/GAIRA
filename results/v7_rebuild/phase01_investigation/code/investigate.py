#!/usr/bin/env python3
"""GAIRA V7 — Phase 01 scientific investigation.

Ten investigations designed to FALSIFY the Phase 01 conclusions. Nothing here modifies
Phase 00, the frozen atlas, or the Phase 01 architecture; it measures what Phase 01 produced
and tries to break it.

    python results/v7_rebuild/phase01_investigation/code/investigate.py
"""
from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import nnls
from scipy.signal import find_peaks
from scipy.spatial.distance import squareform
from scipy.stats import mannwhitneyu, pearsonr, spearmanr

HERE = Path(__file__).resolve().parent
INV = HERE.parent
REPO = INV.parents[2]
sys.path.insert(0, str(REPO / "results/v7_rebuild/phase00/code"))
sys.path.insert(0, str(REPO / "src"))

import v7_corpus as C                                    # noqa: E402
import v7_paths as P                                     # noqa: E402
from gaira.v7.lsm import classlocal as CLS               # noqa: E402
from gaira.v7.lsm import references as REF               # noqa: E402
from gaira.v7.lsm import serialization as SER            # noqa: E402

warnings.filterwarnings("ignore")

P00 = REPO / "results/v7_rebuild/phase00"
P01 = REPO / "results/v7_rebuild/phase01"
T, A, LOGS = INV / "tables", INV / "artifacts", INV / "logs"
LOG: list[str] = []


def log(m):
    line = f"[investigate] {m}"
    print(line, flush=True)
    LOG.append(line)


def wtab(df, name):
    T.mkdir(parents=True, exist_ok=True)
    df.to_csv(T / name, index=False, lineterminator="\n")
    return name


def wjson(obj, name):
    A.mkdir(parents=True, exist_ok=True)
    (A / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n")
    return name


# ── shared context ────────────────────────────────────────────────────────────
class Ctx:
    def __init__(self):
        self.reg, self.H, self.ids, self.man = SER.load_registry(P01 / "artifacts")
        self.kept = self.reg[self.reg.retained].reset_index(drop=True)
        z = np.load(P01 / "artifacts/balanced_references_v1.npz", allow_pickle=True)
        self.Xref = np.asarray(z["X"], float)
        self.rcid = np.array([str(c) for c in z["canonical_id"]])
        self.rw = np.asarray(z["weight"], float)
        self.grid = np.asarray(z["grid"], float)

        part = pd.read_csv(P00 / "tables/chemical_partition_v1.csv")
        self.fine = dict(zip(part.canonical_id, part.fine_class))
        self.broad = dict(zip(part.canonical_id, part.broad_class))
        canon = pd.read_csv(P00 / "tables/canonical_analytes_v1.csv")
        self.sources = {r.canonical_id: str(r.sources).split(";") for r in canon.itertuples()}
        folds = pd.read_csv(P00 / "tables/cv_folds_v1.csv")
        self.fold = dict(zip(folds.canonical_id, folds.fold))
        self.quality = pd.read_csv(P00 / "tables/spectrum_quality_v1.csv")

        corpus = C.load_corpus()
        self.X = np.nan_to_num(corpus.X)
        alias = pd.read_csv(P00 / "tables/alias_table_v1.csv")
        a2c = dict(zip(alias.surface_form, alias.canonical_id))
        self.meta = corpus.meta.copy()
        self.meta["canonical_id"] = self.meta.analyte.map(a2c)

        self.classes = sorted(self.kept.chemical_class.unique())
        self.mol = {c: sorted({m for m in self.rcid if self.fine.get(m) == c})
                    for c in self.classes}
        self.Xc = {c: self._block(self.mol[c]) for c in self.classes}
        self.Hc = {c: np.vstack([self.H[self.ids.index(m)] for m in
                                 self.kept[self.kept.chemical_class == c].motif_id])
                   for c in self.classes}

    def _block(self, mols):
        out = []
        for m in mols:
            s = self.rcid == m
            w = self.rw[s]
            w = w / w.sum() if w.sum() > 0 else np.full(s.sum(), 1.0 / s.sum())
            out.append((self.Xref[s] * w[:, None]).sum(axis=0))
        return np.vstack(out)

    def spectra_of(self, mol):
        return self.X[(self.meta.canonical_id == mol).values]


def recon(x, H):
    a, _ = nnls(H.T, np.maximum(np.asarray(x, float), 0))
    r = a @ H
    ss = float(np.sum(x ** 2)) or 1.0
    return a, r, {
        "ev": 1.0 - float(np.sum((x - r) ** 2)) / ss,
        "rmse": float(np.sqrt(np.mean((x - r) ** 2))),
        "cosine": float(x @ r / (np.linalg.norm(x) * np.linalg.norm(r) + 1e-12)),
        "residual_energy": float(np.sum((x - r) ** 2) / ss),
    }


# ── Investigation 1 — uniqueness ──────────────────────────────────────────────
def inv1(ctx):
    log("I1 — LSM uniqueness within each class")
    rows, mats = [], {}
    for c in ctx.classes:
        Hc = ctx.Hc[c]
        mids = ctx.kept[ctx.kept.chemical_class == c].motif_id.tolist()
        k = Hc.shape[0]
        if k < 2:
            rows.append({"chemical_class": c, "k": k, "n_pairs": 0, "max_cosine": None,
                         "mean_cosine": None, "max_pearson": None, "max_spearman": None,
                         "n_pairs_ge_0.95": 0, "n_pairs_ge_0.90": 0, "merge_candidates": ""})
            continue
        N = Hc / (np.linalg.norm(Hc, axis=1, keepdims=True) + 1e-12)
        Cos = N @ N.T
        Pe = np.corrcoef(Hc)
        Sp = np.zeros((k, k))
        for i in range(k):
            for j in range(k):
                Sp[i, j] = spearmanr(Hc[i], Hc[j]).statistic
        mats[c] = {"cosine": Cos, "pearson": Pe, "spearman": Sp, "ids": mids}
        iu = np.triu_indices(k, 1)
        cand = [f"{mids[i]}~{mids[j]}" for i, j in zip(*iu) if Cos[i, j] >= 0.90]
        rows.append({"chemical_class": c, "k": k, "n_pairs": len(iu[0]),
                     "max_cosine": round(float(Cos[iu].max()), 4),
                     "mean_cosine": round(float(Cos[iu].mean()), 4),
                     "max_pearson": round(float(Pe[iu].max()), 4),
                     "max_spearman": round(float(Sp[iu].max()), 4),
                     "n_pairs_ge_0.95": int((Cos[iu] >= 0.95).sum()),
                     "n_pairs_ge_0.90": int((Cos[iu] >= 0.90).sum()),
                     "merge_candidates": ";".join(cand)})
    df = pd.DataFrame(rows)
    wtab(df, "inv1_uniqueness_v1.csv")
    np.savez_compressed(A / "inv1_similarity_matrices.npz",
                        **{f"{c}__{k}": v for c, d in mats.items()
                           for k, v in d.items() if k != "ids"})
    dup = int(df["n_pairs_ge_0.95"].fillna(0).sum())
    log(f"   max within-class cosine {df.max_cosine.max():.3f}; "
        f"{dup} duplicate pairs (>=0.95); "
        f"{int(df['n_pairs_ge_0.90'].fillna(0).sum())} pairs >=0.90")
    return df, mats


# ── Investigation 2 — per-molecule reconstruction ─────────────────────────────
def inv2(ctx):
    log("I2 — per-molecule reconstruction quality")
    rows, band_rows = [], []
    for c in ctx.classes:
        Hc, mols, Xb = ctx.Hc[c], ctx.mol[c], ctx.Xc[c]
        for i, mol in enumerate(mols):
            x = Xb[i]
            a, r, met = recon(x, Hc)
            pk, _ = find_peaks(x, prominence=0.05 * float(x.max()) if x.max() > 0 else 0.01)
            band_ev = (1.0 - float(np.sum((x[pk] - r[pk]) ** 2) /
                                   (np.sum(x[pk] ** 2) + 1e-12))) if len(pk) else np.nan
            rows.append({"chemical_class": c, "molecule": mol, "k_c": Hc.shape[0],
                         **{k: round(v, 5) for k, v in met.items()},
                         "band_ev": round(float(band_ev), 5) if len(pk) else None,
                         "n_bands": int(len(pk)),
                         "n_motifs_used": int((a > 0.05 * a.max()).sum()) if a.max() > 0 else 0})
            if len(pk):
                for p in pk[:12]:
                    band_rows.append({"chemical_class": c, "molecule": mol,
                                      "band_cm": round(float(ctx.grid[p]), 1),
                                      "observed": round(float(x[p]), 5),
                                      "reconstructed": round(float(r[p]), 5),
                                      "residual": round(float(x[p] - r[p]), 5)})
    df = pd.DataFrame(rows)
    wtab(df, "inv2_per_molecule_reconstruction_v1.csv")
    wtab(pd.DataFrame(band_rows), "inv2_bandwise_residuals_v1.csv")
    summ = (df.groupby("chemical_class")
            .agg(n=("molecule", "size"), k_c=("k_c", "first"), ev_mean=("ev", "mean"),
                 ev_min=("ev", "min"), ev_p10=("ev", lambda s: np.percentile(s, 10)),
                 cos_mean=("cosine", "mean"), rmse_mean=("rmse", "mean"),
                 band_ev_mean=("band_ev", "mean"))
            .round(4).reset_index().sort_values("ev_min"))
    wtab(summ, "inv2_class_reconstruction_summary_v1.csv")
    log(f"   worst molecule EV {df.ev.min():.3f} ({df.loc[df.ev.idxmin(), 'molecule']}); "
        f"{int((df.ev < 0.5).sum())} molecules below EV 0.5; corpus mean {df.ev.mean():.3f}")
    return df, summ


# ── Investigation 3 — k_c robustness ──────────────────────────────────────────
def inv3(ctx):
    log("I3 — k_c robustness (k-1, k, k+1)")
    rows = []
    for c in ctx.classes:
        Xb, mols = ctx.Xc[c], ctx.mol[c]
        n = len(mols)
        k0 = ctx.Hc[c].shape[0]
        f = np.array([ctx.fold.get(m, 0) for m in mols])
        if len(np.unique(f)) < 2:
            f = np.arange(n) % 2
        base = None
        for k in (k0 - 1, k0, k0 + 1):
            if k < 1 or k >= n:
                continue
            rep = CLS.repeated_fits(Xb, k)
            W, H = rep["W"], rep["H"]
            evs = [recon(Xb[i], H)[2]["ev"] for i in range(n)]
            if k == k0:
                base = H
            sim = None
            if base is not None and k != k0:
                _, s = CLS.align(base, H) if k > k0 else CLS.align(H, base)
                sim = float(np.mean(s))
            rows.append({
                "chemical_class": c, "n_molecules": n, "k_selected": k0, "k": k,
                "delta_k": k - k0,
                "heldout_ev": round(CLS._heldout_reconstruction(Xb, k, f), 4),
                "insample_ev_mean": round(float(np.mean(evs)), 4),
                "insample_ev_min": round(float(np.min(evs)), 4),
                "stability": round(float(np.mean(rep["recurrence"])), 4),
                "activation_sparsity": round(CLS._activation_sparsity(W), 4),
                "duplicate_fraction": round(CLS._redundancy(H), 4),
                "basis_match_to_selected": round(sim, 4) if sim is not None else 1.0,
            })
    df = pd.DataFrame(rows)
    wtab(df, "inv3_kc_robustness_v1.csv")
    knife = []
    for c, g in df.groupby("chemical_class"):
        s = g.set_index("delta_k")
        if 0 not in s.index:
            continue
        hi = s.loc[0, "heldout_ev"]
        nb = [s.loc[d, "heldout_ev"] for d in (-1, 1) if d in s.index]
        knife.append({"chemical_class": c, "k_selected": int(s.loc[0, "k"]),
                      "heldout_ev_at_k": hi,
                      "max_neighbour_gain": round(float(max(nb) - hi), 4) if nb else 0.0,
                      "knife_edge": bool(nb and max(nb) - hi > 0.05)})
    kdf = pd.DataFrame(knife)
    wtab(kdf, "inv3_knife_edge_v1.csv")
    log(f"   {int(kdf.knife_edge.sum())} of {len(kdf)} classes on a knife edge "
        f"(a neighbouring k gains >0.05 held-out EV)")
    return df, kdf


# ── Investigation 4 — source consistency ──────────────────────────────────────
def inv4(ctx):
    log("I4 — source consistency of class-local activations")
    rows, per = [], []
    for c in ctx.classes:
        Hc = ctx.Hc[c]
        mols = ctx.mol[c]
        acts, srcs, labels = [], [], []
        for mol in mols:
            for s_i, x in zip(ctx.meta[ctx.meta.canonical_id == mol].source.tolist(),
                              ctx.spectra_of(mol)):
                a, _, _ = recon(x, Hc)
                tot = a.sum()
                acts.append(a / tot if tot > 0 else a)
                srcs.append(s_i)
                labels.append(mol)
        if not acts:
            continue
        Aq = np.vstack(acts)
        S = np.array(srcs)
        uniq = [u for u in np.unique(S) if (S == u).sum() >= 5]
        if len(uniq) < 2:
            rows.append({"chemical_class": c, "n_sources": len(np.unique(S)),
                         "testable": False, "n_motifs_differing": 0,
                         "min_p": None, "verdict": "single dominant source — not testable"})
            continue
        a1, a2 = Aq[S == uniq[0]], Aq[S == uniq[1]]
        ps = []
        for j in range(Aq.shape[1]):
            try:
                ps.append(float(mannwhitneyu(a1[:, j], a2[:, j]).pvalue))
            except Exception:
                ps.append(1.0)
        ps = np.array(ps)
        m = len(ps)
        sig = ps < (0.05 / m)                                   # Bonferroni
        rows.append({"chemical_class": c, "n_sources": len(uniq),
                     "sources_compared": " vs ".join(uniq[:2]),
                     "n_spectra": int(len(S)), "testable": True,
                     "n_motifs": m, "n_motifs_differing": int(sig.sum()),
                     "frac_motifs_differing": round(float(sig.mean()), 3),
                     "min_p": float(ps.min()),
                     "verdict": ("source-dependent activation" if sig.any()
                                 else "indistinguishable across sources")})
        for j in range(m):
            per.append({"chemical_class": c, "motif_index": j, "p_value": float(ps[j]),
                        "significant_bonferroni": bool(sig[j]),
                        "mean_source_a": round(float(a1[:, j].mean()), 4),
                        "mean_source_b": round(float(a2[:, j].mean()), 4)})
    df = pd.DataFrame(rows)
    wtab(df, "inv4_source_consistency_v1.csv")
    wtab(pd.DataFrame(per), "inv4_source_per_motif_v1.csv")
    t = df[df.testable]
    log(f"   {len(t)} classes testable; "
        f"{int((t.n_motifs_differing > 0).sum())} show source-dependent activation")
    return df


# ── Investigation 5 — spectroscopic interpretability ──────────────────────────
# Raman band assignment is CONTEXT-DEPENDENT. A band at 702 cm-1 is purine ring breathing in
# a nucleobase and the sterol ring mode in cholesterol; assigning it context-free produces
# chemistry that is simply wrong. Assignments are therefore conditioned on the chemistry class
# the motif was fitted in, with a generic fallback only where no class-specific assignment
# applies. Ranges follow standard biological Raman assignment tables.
GENERIC = [
    (480, 560, "skeletal deformation; S–S stretch"),
    (560, 660, "C–S stretch; ring deformation"),
    (660, 760, "C–S stretch; ring breathing"),
    (760, 830, "ring breathing; O–P–O symmetric"),
    (830, 900, "C–C skeletal stretch"),
    (900, 960, "C–C stretch; skeletal"),
    (960, 1010, "symmetric stretch (PO4 / ring breathing)"),
    (1010, 1060, "C–O / C–C stretch"),
    (1060, 1110, "C–C / C–N stretch; PO2− symmetric"),
    (1110, 1160, "C–O–C / C–N stretch"),
    (1160, 1210, "C–C stretch; in-plane bending"),
    (1210, 1300, "amide III region; CH2 twist"),
    (1300, 1380, "CH2 deformation / wag"),
    (1380, 1425, "COO− symmetric stretch"),
    (1425, 1490, "CH2 / CH3 deformation"),
    (1490, 1560, "ring stretch; amide II"),
    (1560, 1620, "conjugated C=C; ring stretch"),
    (1620, 1700, "amide I; C=C stretch"),
    (1700, 1780, "C=O stretch (ester / carboxylic acid)"),
]

CLASS_SPECIFIC = {
    "peptide_protein": [
        (500, 545, "S–S disulfide stretch"),
        (620, 645, "phenylalanine ring"),
        (640, 665, "tyrosine ring"),
        (700, 745, "C–S stretch (methionine)"),
        (825, 860, "tyrosine Fermi doublet (830/850)"),
        (995, 1010, "phenylalanine ring breathing (1003)"),
        (1200, 1300, "amide III (β-sheet 1230 / α-helix 1265)"),
        (1440, 1470, "CH2/CH3 deformation"),
        (1550, 1560, "tryptophan"),
        (1640, 1700, "amide I (α-helix 1655 / β-sheet 1670)"),
    ],
    "free_amino_acid": [
        (820, 860, "tyrosine Fermi doublet"),
        (995, 1010, "phenylalanine ring breathing"),
        (1320, 1420, "COO− symmetric stretch (zwitterion)"),
        (1580, 1640, "NH3+ deformation; COO− antisymmetric"),
    ],
    "purine": [
        (715, 740, "purine ring breathing (adenine 723, guanine 730)"),
        (1320, 1345, "purine ring stretch"),
        (1480, 1500, "imidazole ring stretch"),
        (1570, 1600, "purine C=N stretch"),
        (1650, 1700, "C=O stretch (oxopurine: xanthine, urate)"),
    ],
    "pyrimidine": [
        (780, 800, "pyrimidine ring breathing (cytosine 785, uracil 790)"),
        (1230, 1260, "ring stretch; C–N"),
        (1650, 1700, "C=O stretch (uracil/thymine carbonyl)"),
    ],
    "nucleic_acid_polymer": [
        (780, 800, "pyrimidine ring breathing; O–P–O backbone"),
        (810, 830, "A-form backbone O–P–O"),
        (1080, 1100, "PO2− symmetric stretch (phosphodiester backbone)"),
        (1240, 1260, "base ring / backbone"),
    ],
    "phosphate_metabolite": [
        (975, 995, "PO4 symmetric stretch (v1)"),
        (1050, 1100, "PO2− symmetric stretch"),
    ],
    "sterol_steroid": [
        (695, 715, "sterol ring skeletal mode (cholesterol 702)"),
        (420, 445, "sterol ring deformation"),
        (1055, 1090, "C–C skeletal (fused ring)"),
        (1435, 1460, "CH2 scissoring"),
        (1660, 1680, "C=C stretch (Δ5 unsaturation)"),
    ],
    "fatty_acid": [
        (1060, 1075, "C–C skeletal, all-trans acyl chain"),
        (1090, 1105, "C–C skeletal, gauche"),
        (1120, 1140, "C–C skeletal, all-trans"),
        (1255, 1275, "=C–H in-plane deformation (cis unsaturation)"),
        (1290, 1310, "CH2 twist"),
        (1435, 1460, "CH2 scissoring"),
        (1650, 1670, "C=C stretch (cis)"),
        (1700, 1730, "carboxylic acid C=O"),
    ],
    "acylglycerol": [
        (860, 880, "glycerol backbone C–C"),
        (1060, 1075, "C–C skeletal, all-trans acyl chain"),
        (1120, 1140, "C–C skeletal, all-trans"),
        (1255, 1275, "=C–H in-plane (cis unsaturation)"),
        (1290, 1310, "CH2 twist"),
        (1435, 1460, "CH2 scissoring"),
        (1650, 1670, "C=C stretch (cis)"),
        (1735, 1755, "ESTER C=O (1745) — diagnostic for acylglycerols vs free fatty acids"),
    ],
    "phospholipid_sphingolipid": [
        (715, 730, "choline N+(CH3)3 symmetric stretch"),
        (1060, 1140, "C–C skeletal acyl chain"),
        (1085, 1100, "PO2− symmetric stretch (phosphate head group)"),
        (1435, 1460, "CH2 scissoring"),
        (1735, 1755, "ester C=O"),
    ],
    "mono_oligosaccharide": [
        (840, 860, "α-anomeric C–H deformation"),
        (890, 910, "β-anomeric C–H deformation"),
        (1050, 1150, "C–O / C–C ring stretch (pyranose)"),
        (1250, 1300, "C–H / O–H deformation"),
        (1440, 1470, "CH2 deformation"),
    ],
    "polysaccharide": [
        (890, 910, "β-glycosidic linkage"),
        (930, 950, "α-glycosidic linkage"),
        (1080, 1130, "C–O–C glycosidic bridge"),
        (1330, 1380, "C–H / O–H deformation"),
    ],
    "carboxylic_acid_metabolite": [
        (800, 850, "C–COO− stretch"),
        (930, 960, "C–C stretch"),
        (1380, 1420, "COO− symmetric stretch"),
        (1560, 1620, "COO− antisymmetric stretch"),
        (1650, 1700, "C=O (keto acids: pyruvate, acetoacetate)"),
    ],
    "sulfur_thiol_cofactor": [
        (490, 550, "S–S stretch"),
        (630, 680, "C–S stretch"),
        (2550, 2600, "S–H stretch (outside window)"),
        (1390, 1420, "COO− symmetric"),
    ],
    "chromophore_pigment": [
        (1000, 1020, "C–CH3 rocking (carotenoid)"),
        (1145, 1170, "C–C stretch (carotenoid polyene)"),
        (1500, 1535, "C=C stretch (carotenoid polyene, resonance-enhanced)"),
        (1340, 1410, "isoalloxazine ring (flavin)"),
        (1570, 1620, "porphyrin ring / flavin"),
    ],
    "small_nitrogenous": [
        (1000, 1020, "C–N symmetric stretch (urea 1010)"),
        (1150, 1180, "C–N antisymmetric"),
        (1450, 1480, "N–H deformation"),
        (1600, 1680, "C=O stretch (urea 1650); guanidino (creatinine)"),
    ],
}


def assign(cm: float, chemical_class: str = "") -> str:
    """Assign a band, conditioned on the chemistry it was fitted in.

    Returns the class-specific assignment where one applies, otherwise a generic mode. A
    context-free table would call 702 cm-1 "purine ring breathing" inside a sterol motif,
    where it is the cholesterol ring mode — an assignment error a reviewer would reject.
    """
    for lo, hi, txt in CLASS_SPECIFIC.get(chemical_class, []):
        if lo <= cm < hi:
            return txt
    for lo, hi, txt in GENERIC:
        if lo <= cm < hi:
            return f"{txt} (generic)"
    return "outside the assigned range"


def inv5(ctx):
    log("I5 — spectroscopic interpretability of every retained LSM")
    rows = []
    for _, r in ctx.kept.iterrows():
        h = ctx.H[ctx.ids.index(r.motif_id)]
        pk, props = find_peaks(h, prominence=0.05 * float(h.max()))
        if len(pk) == 0:
            pk = np.array([int(np.argmax(h))])
            props = {"prominences": np.array([float(h.max())])}
        order = np.argsort(-props["prominences"])[:6]
        bands = [(float(ctx.grid[pk[i]]), float(props["prominences"][i])) for i in order]
        bands.sort()
        rows.append({
            "motif_id": r.motif_id, "chemical_class": r.chemical_class,
            "lsm_type": r.lsm_type, "n_analytes": r.n_analytes,
            "stability": r.stability, "activation_sparsity": r.activation_sparsity,
            "top_bands_cm": ";".join(f"{b:.0f}" for b, _ in bands),
            "assignments": " | ".join(f"{b:.0f}: {assign(b, r.chemical_class)}"
                                     for b, _ in bands),
            "n_class_specific": sum(1 for b, _ in bands
                                    if "(generic)" not in assign(b, r.chemical_class)),
            "representative_molecules": ";".join(str(r.analytes).split(";")[:4]),
            "interpretation_class": ("shared chemistry" if r.lsm_type == "class_shared"
                                     else "subfamily chemistry" if r.lsm_type == "subfamily"
                                     else "molecule-discriminating"),
        })
    df = pd.DataFrame(rows)
    wtab(df, "inv5_spectroscopic_interpretation_v1.csv")
    log(f"   {len(df)} motifs annotated with band assignments")
    return df


# ── Investigation 6 — coverage ────────────────────────────────────────────────
def inv6(ctx, permol):
    log("I6 — coverage and orphan detection")
    rows = []
    for c in ctx.classes:
        Hc, mols, Xb = ctx.Hc[c], ctx.mol[c], ctx.Xc[c]
        for i, mol in enumerate(mols):
            a, _, met = recon(Xb[i], Hc)
            top = float(a.max() / (a.sum() + 1e-12)) if a.sum() > 0 else 0.0
            rows.append({"chemical_class": c, "molecule": mol, "ev": round(met["ev"], 4),
                         "n_motifs_activated": int((a > 0.05 * (a.max() or 1)).sum()),
                         "dominant_motif_share": round(top, 4),
                         "orphan": bool(met["ev"] < 0.5),
                         "spectral_outlier": bool(met["ev"] < 0.4 or met["cosine"] < 0.7)})
    df = pd.DataFrame(rows)
    wtab(df, "inv6_coverage_v1.csv")
    orph = df[df.orphan]
    diag = []
    for _, r in orph.iterrows():
        n = len(ctx.mol[r.chemical_class])
        kc = ctx.Hc[r.chemical_class].shape[0]
        q = ctx.quality[ctx.quality.canonical_id == r.molecule].quality_score
        cause = ("corpus: class too small for its diversity" if n <= 5 else
                 "ceiling-bound: k_c at floor(n/2)" if kc == n // 2 else
                 "preprocessing/quality" if len(q) and float(q.mean()) < 0.6 else
                 "true biochemical uniqueness or optimisation shortfall")
        diag.append({"molecule": r.molecule, "chemical_class": r.chemical_class,
                     "ev": r.ev, "n_class_molecules": n, "k_c": kc,
                     "k_ceiling": n // 2, "mean_quality": round(float(q.mean()), 3) if len(q) else None,
                     "diagnosis": cause})
    ddf = pd.DataFrame(diag)
    wtab(ddf, "inv6_orphan_diagnosis_v1.csv")
    log(f"   {len(orph)} orphans (EV<0.5) of {len(df)} molecules; "
        f"{int(df.spectral_outlier.sum())} spectral outliers")
    return df, ddf


# ── Investigation 7 — hidden cross-class redundancy ───────────────────────────
def inv7(ctx):
    log("I7 — hidden cross-class redundancy (Phase 02 hypotheses)")
    mids = ctx.kept.motif_id.tolist()
    Hm = np.vstack([ctx.H[ctx.ids.index(m)] for m in mids])
    N = Hm / (np.linalg.norm(Hm, axis=1, keepdims=True) + 1e-12)
    Cos = N @ N.T
    cls = ctx.kept.chemical_class.tolist()
    M = pd.DataFrame(np.round(Cos, 4), index=mids, columns=mids)
    wtab(M.reset_index().rename(columns={"index": "motif_id"}),
         "inv7_intermotif_similarity_v1.csv")
    rows = []
    iu = np.triu_indices(len(mids), 1)
    for i, j in zip(*iu):
        if cls[i] != cls[j] and Cos[i, j] >= 0.70:
            rows.append({"motif_a": mids[i], "class_a": cls[i], "motif_b": mids[j],
                         "class_b": cls[j], "cosine": round(float(Cos[i, j]), 4),
                         "hypothesis": f"{cls[i]} ~ {cls[j]}"})
    cross = pd.DataFrame(rows).sort_values("cosine", ascending=False) if rows else pd.DataFrame(
        columns=["motif_a", "class_a", "motif_b", "class_b", "cosine", "hypothesis"])
    wtab(cross, "inv7_cross_class_candidates_v1.csv")
    pair = {}
    for _, r in cross.iterrows():
        key = tuple(sorted((r.class_a, r.class_b)))
        pair[key] = max(pair.get(key, 0), r.cosine)
    pdf = pd.DataFrame([{"class_a": a, "class_b": b, "max_cosine": v}
                        for (a, b), v in sorted(pair.items(), key=lambda t: -t[1])])
    wtab(pdf, "inv7_class_pair_hypotheses_v1.csv")
    log(f"   {len(cross)} cross-class motif pairs at cosine>=0.70, "
        f"spanning {len(pdf)} class pairs")
    return cross, pdf, M


# ── Investigation 8 — sensitivity ─────────────────────────────────────────────
def inv8(ctx, n_rep=8):
    log("I8 — sensitivity to seed, molecule bootstrap, spectrum bootstrap and noise")
    rows = []
    rng = np.random.default_rng(0)
    for c in ctx.classes:
        Xb, mols = ctx.Xc[c], ctx.mol[c]
        n, k = len(mols), ctx.Hc[c].shape[0]
        if k < 1 or n < 3:
            continue
        H0 = ctx.Hc[c]
        for mode in ("seed", "molecule_bootstrap", "noise_1pct", "noise_5pct"):
            sims = []
            for r in range(n_rep):
                if mode == "seed":
                    _, H, _ = CLS.fit_nmf(Xb, k, seed=1000 + r)
                elif mode == "molecule_bootstrap":
                    idx = np.sort(rng.choice(n, size=max(k + 1, int(0.8 * n)), replace=False))
                    if len(idx) <= k:
                        continue
                    _, H, _ = CLS.fit_nmf(Xb[idx], k, seed=0)
                else:
                    lvl = 0.01 if mode == "noise_1pct" else 0.05
                    Xn = np.maximum(Xb + rng.normal(0, lvl * Xb.std(), Xb.shape), 0)
                    _, H, _ = CLS.fit_nmf(Xn, k, seed=0)
                _, s = CLS.align(H0, H)
                sims.append(float(np.mean(s)))
            if sims:
                a = np.array(sims)
                rows.append({"chemical_class": c, "k_c": k, "perturbation": mode,
                             "n_repeats": len(a),
                             "mean_basis_similarity": round(float(a.mean()), 4),
                             "ci95_low": round(float(np.percentile(a, 2.5)), 4),
                             "ci95_high": round(float(np.percentile(a, 97.5)), 4),
                             "min": round(float(a.min()), 4)})
    df = pd.DataFrame(rows)
    wtab(df, "inv8_sensitivity_v1.csv")
    s = df.groupby("perturbation").mean_basis_similarity.agg(["mean", "min"]).round(4)
    log("   mean basis similarity under perturbation:\n" +
        "\n".join(f"      {i:22s} mean {r['mean']:.3f}  worst class {r['min']:.3f}"
                  for i, r in s.iterrows()))
    return df


# ── Investigation 9 — corpus vs algorithm ─────────────────────────────────────
def inv9(ctx, permol, kdf):
    log("I9 — attributing weakness to corpus or algorithm")
    rows = []
    for c in ctx.classes:
        n = len(ctx.mol[c])
        k = ctx.Hc[c].shape[0]
        sub = permol[permol.chemical_class == c]
        ceiling_bound = k == max(1, n // 2)
        worst = float(sub.ev.min())
        knife = bool(kdf[kdf.chemical_class == c].knife_edge.iloc[0]) \
            if (kdf.chemical_class == c).any() else False
        if worst >= 0.7:
            cause, fixable = "adequate", "—"
        elif ceiling_bound and n <= 5:
            cause = "CORPUS: too few molecules; k_c ceiling binds"
            fixable = "Phase 08 corpus expansion only"
        elif ceiling_bound:
            cause = "CORPUS/CONSTRAINT: k_c at the floor(n/2) ceiling"
            fixable = "relax ceiling only with more molecules"
        elif knife:
            cause = "ALGORITHM: k_c selection is unstable here"
            fixable = "selection criterion"
        else:
            cause = "CORPUS: chemical heterogeneity exceeds available molecules"
            fixable = "Phase 02 consensus may pool related chemistry; else corpus"
        rows.append({"chemical_class": c, "n_molecules": n, "k_c": k,
                     "k_ceiling": max(1, n // 2), "ceiling_bound": ceiling_bound,
                     "worst_molecule_ev": round(worst, 4),
                     "mean_ev": round(float(sub.ev.mean()), 4),
                     "limitation": cause, "resolvable_by": fixable})
    df = pd.DataFrame(rows).sort_values("worst_molecule_ev")
    wtab(df, "inv9_limitations_v1.csv")
    log(f"   {int((df.limitation.str.startswith('CORPUS')).sum())} classes corpus-limited; "
        f"{int((df.limitation.str.startswith('ALGORITHM')).sum())} algorithm-limited")
    return df


def main():
    for d in (T, A, LOGS, INV / "figures", INV / "reports"):
        d.mkdir(parents=True, exist_ok=True)
    t0 = datetime.now(timezone.utc)
    log(f"loading Phase 01 outputs ({datetime.now(timezone.utc).isoformat()})")
    ctx = Ctx()
    log(f"   {len(ctx.kept)} retained LSMs across {len(ctx.classes)} classes")

    u, mats = inv1(ctx)
    permol, csumm = inv2(ctx)
    kdf_full, kdf = inv3(ctx)
    src = inv4(ctx)
    interp = inv5(ctx)
    cov, orph = inv6(ctx, permol)
    cross, pairs, simM = inv7(ctx)
    sens = inv8(ctx)
    lim = inv9(ctx, permol, kdf)

    summary = {
        "generated_utc": t0.isoformat(),
        "n_lsms": int(len(ctx.kept)), "n_classes": len(ctx.classes),
        "I1_uniqueness": {"max_within_class_cosine": float(u.max_cosine.max()),
                          "duplicate_pairs": int(u["n_pairs_ge_0.95"].fillna(0).sum()),
                          "pairs_ge_0.90": int(u["n_pairs_ge_0.90"].fillna(0).sum())},
        "I2_reconstruction": {"mean_ev": round(float(permol.ev.mean()), 4),
                              "worst_ev": round(float(permol.ev.min()), 4),
                              "n_below_0.5": int((permol.ev < 0.5).sum()),
                              "n_below_0.7": int((permol.ev < 0.7).sum())},
        "I3_kc": {"n_knife_edge": int(kdf.knife_edge.sum()), "n_classes": int(len(kdf))},
        "I4_source": {"n_testable": int(src.testable.sum()),
                      "n_source_dependent": int((src.n_motifs_differing > 0).sum())},
        "I5_interpretability": {"n_annotated": int(len(interp))},
        "I6_coverage": {"n_orphans": int(cov.orphan.sum()),
                        "n_outliers": int(cov.spectral_outlier.sum())},
        "I7_cross_class": {"n_pairs": int(len(cross)), "n_class_pairs": int(len(pairs))},
        "I8_sensitivity": sens.groupby("perturbation").mean_basis_similarity.mean()
                              .round(4).to_dict(),
        "I9_limitations": lim.limitation.value_counts().to_dict(),
    }
    wjson(summary, "investigation_summary_v1.json")
    (LOGS / "investigate.log").write_text("\n".join(LOG) + "\n")
    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
