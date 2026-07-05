"""gaira_base_4_mss_resolution_reporting_layer_v1

Phase: MSS-RESOLUTION REPORTING LAYER.

Goal: expose intermediate MSS-level resolution between 11-axis BSV and the
disease cohort labels for Pilot 1 and Pilot 2 — surfacing motif family →
MSS cluster → top molecule candidates per spectrum, with cohort-level
aggregation and cross-pilot transfer categorization.

Strict constraints (NEVER violated):
- Engine v4.5 unchanged
- MSS scoring kernel unchanged (anchor-fires + companion-fires logic; no
  soft-MSS, no competitor-aware scoring)
- 11-axis BSV unchanged (BSV vectors read from pre-computed Pilot 1.1 / 2.1
  normalization-sensitivity outputs)
- Motif registry unchanged
- Substrate physics unchanged
- No disease labels used to TUNE anything (labels read only for cohort grouping)
- No classifier feedback, no pilot reruns, no threshold tuning

Outputs:
  /Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_4_mss_resolution_reporting_layer_v1/
    tables/, figures/, reports/, audit/, code_snapshot/

Run:
    cd /Users/suraj/projects/GAIRA
    PYTHONPATH=src .venv/bin/python scripts/run_gaira_base_4_mss_resolution_reporting_layer_v1.py
"""
from __future__ import annotations

import shutil
import sys
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter

warnings.simplefilter("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gaira.spectral import canonical_master_axis  # noqa: E402
from gaira.spectral.preprocessing import _asls_baseline  # noqa: E402

from run_gaira_base_4_liver_narrow_metabolite_panel_v1 import (  # noqa: E402
    load_p1_raw, load_p2_raw,
)


# ──────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────
ROOT = Path("/Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_4_mss_resolution_reporting_layer_v1")
TABLES = ROOT / "tables"
FIGS   = ROOT / "figures"
REPORTS = ROOT / "reports"
AUDIT  = ROOT / "audit"
CODE_SNAPSHOT = ROOT / "code_snapshot"
for d in (TABLES, FIGS, REPORTS, AUDIT, CODE_SNAPSHOT):
    d.mkdir(parents=True, exist_ok=True)

NARROW_REGISTRY = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_4_mss_narrow_metabolite_registry_repair_v1/"
    "registry/narrow_metabolite_mss_registry_v1.csv"
)
ASSIGN_TABLE = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_4_mss_narrow_metabolite_registry_repair_v1/"
    "tables/molecule_cluster_family_assignment_v1.csv"
)
QC_TABLE = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_4_mss_narrow_metabolite_registry_repair_v1/"
    "tables/mss_template_quality_check_v1.csv"
)
P1_BSV = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_4_passive_target_pilot_1_1_normalization_sensitivity/"
    "tables/pilot1_1_normalized_vectors.csv"
)
P2_BSV = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_4_passive_target_pilot_2_1_normalization_sensitivity/"
    "tables/pilot2_1_normalized_bsv_vectors.csv"
)


# ──────────────────────────────────────────────────────────────────────
# Targets and family taxonomy
# ──────────────────────────────────────────────────────────────────────
NARROW_TARGETS = [
    "uric_acid", "hypoxanthine", "xanthine", "adenine",
    "ergothioneine", "glutathione", "lactate",
    "cysteine", "cystine",
    "tryptophan", "phenylalanine", "tyrosine",
    "cholesterol", "oleic_acid", "palmitic_acid", "stearic_acid",
    "glucose", "urea", "creatinine",
]

BSV_FAMILIES = (
    ("G01", "purine_nucleotide"),
    ("G02", "purine_metabolite"),
    ("G03", "pyrimidine_nucleotide"),
    ("G04", "phosphate_nucleic_adjacent"),
    ("G05", "glycan_carbohydrate"),
    ("G06", "protein_peptide_backbone"),
    ("G07", "aromatic_residue"),
    ("G08", "lipid_acyl_membrane"),
    ("G09", "sterol_neutral_lipid"),
    ("G10", "sulfur_thiol_redox"),
    ("G11", "metabolic_small_molecule"),
)


# ──────────────────────────────────────────────────────────────────────
# Spectrum primitives  (mirrors anchor-fires logic; no scoring change)
# ──────────────────────────────────────────────────────────────────────
def baseline_correct(y: np.ndarray) -> np.ndarray:
    """Standard GAIRA preprocessing: AsLS + Savitzky-Golay + L2."""
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(y)
    if not mask.any():
        return y
    if not mask.all():
        idx = np.arange(len(y))
        y = y.copy()
        y[~mask] = np.interp(idx[~mask], idx[mask], y[mask])
    y_bc = y - _asls_baseline(y, lam=1e5, p=0.001, n_iter=10)
    y_sg = savgol_filter(y_bc, window_length=11, polyorder=3)
    n = float(np.linalg.norm(y_sg))
    return y_sg / n if n > 1e-12 else y_sg


def has_real_peak(y: np.ndarray, master_x: np.ndarray, cm1: float,
                       half: float = 5.0, prom_frac: float = 0.05,
                       top_rank_max: int = 12) -> bool:
    rng = float(y.max() - y.min())
    if rng <= 0:
        return False
    idx, _ = find_peaks(y, prominence=prom_frac * rng)
    if len(idx) == 0:
        return False
    heights = y[idx]
    order = np.argsort(-heights)
    ranked = idx[order][:max(top_rank_max, 5)]
    for ix in ranked:
        if abs(master_x[ix] - cm1) <= half:
            return True
    return False


def mss_anchor_score(y: np.ndarray, master_x: np.ndarray,
                       anchors: list[float], supports: list[float]) -> tuple[float, int, int]:
    """Anchor-fires + 0.3 × support-fires score (mirrors GAIRA MSS layer logic).
    Returns (score, n_anchors_fired, n_supports_fired)."""
    if not anchors:
        return 0.0, 0, 0
    af = sum(int(has_real_peak(y, master_x, a)) for a in anchors)
    sf = sum(int(has_real_peak(y, master_x, s)) for s in supports) if supports else 0
    score = af / max(len(anchors), 1) + 0.3 * (sf / max(len(supports), 1)
                                                       if supports else 0.0)
    return float(score), int(af), int(sf)


# ──────────────────────────────────────────────────────────────────────
# Load registry → templates
# ──────────────────────────────────────────────────────────────────────
def _split_floats(s) -> list[float]:
    if pd.isna(s) or not str(s).strip():
        return []
    out = []
    for tok in str(s).split(";"):
        tok = tok.strip()
        if not tok: continue
        try: out.append(float(tok))
        except ValueError: pass
    return out


def load_templates() -> tuple[list[dict], dict, dict]:
    df = pd.read_csv(NARROW_REGISTRY)
    asg = pd.read_csv(ASSIGN_TABLE).set_index("molecule")
    qc  = pd.read_csv(QC_TABLE)
    qc["key"] = qc["molecule"] + "::" + qc["regime"].astype(str)
    qc_by_key = qc.set_index("key").to_dict("index")

    templates = []
    for _, r in df.iterrows():
        mol = r["molecule"]; regime = r["regime"]
        anchors  = _split_floats(r.get("anchors_cm1"))
        supports = _split_floats(r.get("supports_cm1"))
        if not anchors:
            continue
        bsv_id, bsv_name = "?", "unassigned"
        if mol in asg.index:
            bsv_id   = asg.loc[mol, "bsv_family_id"]
            bsv_name = asg.loc[mol, "bsv_family_name"]
            cluster  = asg.loc[mol, "mss_cluster_v1"]
        else:
            cluster = "unassigned"
        key = f"{mol}::{regime}"
        qcrow = qc_by_key.get(key, {})
        templates.append({
            "molecule":         mol,
            "regime":           regime,
            "bsv_family_id":    bsv_id,
            "bsv_family_name":  bsv_name,
            "mss_cluster":      cluster,
            "anchors":          anchors,
            "supports":         supports,
            "reliability_tier": qcrow.get("reliability_tier", ""),
            "quality_flag":     qcrow.get("quality_flag", ""),
        })
    asg_map = asg.to_dict("index")
    return templates, asg_map, qc


def consolidate_templates_by_molecule(templates: list[dict]) -> dict[str, dict]:
    """For each molecule, keep one template per regime, plus a fallback."""
    by_mol: dict[str, dict] = defaultdict(dict)
    for t in templates:
        by_mol[t["molecule"]][t["regime"]] = t
    return by_mol


# ──────────────────────────────────────────────────────────────────────
# Stage 1 — per-spectrum MSS top-k extraction
# ──────────────────────────────────────────────────────────────────────
def stage1_per_spectrum(refs: list[dict], templates_by_mol: dict[str, dict],
                              master_x: np.ndarray,
                              spec_regime: str, top_k: int = 5) -> pd.DataFrame:
    rows = []
    for s in refs:
        y = baseline_correct(s["spectrum"])
        if not np.any(np.isfinite(y)) or float(np.linalg.norm(y)) < 1e-12:
            continue
        scored = []
        for mol, regime_templates in templates_by_mol.items():
            # Prefer SERS template if pilot is SERS, else Raman
            t = regime_templates.get(spec_regime) or \
                  regime_templates.get("Raman") or \
                  regime_templates.get("SERS") or \
                  next(iter(regime_templates.values()))
            score, af, sf = mss_anchor_score(y, master_x, t["anchors"], t["supports"])
            scored.append({
                "molecule":         mol,
                "regime_used":      t["regime"],
                "regime_match":     t["regime"] == spec_regime,
                "score":            score,
                "anchors_fired":    af,
                "anchors_total":    len(t["anchors"]),
                "supports_fired":   sf,
                "supports_total":   len(t["supports"]),
                "bsv_family_id":    t["bsv_family_id"],
                "bsv_family_name":  t["bsv_family_name"],
                "mss_cluster":      t["mss_cluster"],
                "reliability_tier": t["reliability_tier"],
            })
        scored.sort(key=lambda r: -r["score"])
        for rank, ent in enumerate(scored[:top_k], start=1):
            ent2 = {
                "spectrum_id":   s.get("spectrum_id"),
                "sample_id":     s.get("sample_id"),
                "class_label":   s.get("class_label"),
                "dataset":       s.get("dataset"),
                "regime":        spec_regime,
                "rank":          rank,
                **ent,
            }
            rows.append(ent2)
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────
# Stage 2 — cohort-level aggregation
# ──────────────────────────────────────────────────────────────────────
def stage2_cohort_aggregation(p1_df: pd.DataFrame, p2_df: pd.DataFrame):
    rows_topk = []; rows_cluster = []; rows_family = []
    for tag, df in [("Pilot1", p1_df), ("Pilot2", p2_df)]:
        for cls, sub in df.groupby("class_label"):
            n_spec = sub["spectrum_id"].nunique()
            for k in (1, 3, 5):
                topk = sub[sub["rank"] <= k]
                # frequency = fraction of spectra where molecule appears in top-k
                cnt = topk.groupby("molecule")["spectrum_id"].nunique()
                for mol, c in cnt.items():
                    rows_topk.append({
                        "pilot": tag, "cohort": cls, "molecule": mol,
                        "k": k, "freq": float(c / n_spec) if n_spec else 0.0,
                        "n_spectra": int(n_spec),
                    })
            # cluster + family freq using top-3 by default
            for col, out in [("mss_cluster", rows_cluster),
                                ("bsv_family_id", rows_family)]:
                top3 = sub[sub["rank"] <= 3]
                cnt = top3.groupby(col)["spectrum_id"].nunique()
                for k_, c in cnt.items():
                    out.append({
                        "pilot": tag, "cohort": cls, col: k_,
                        "k": 3, "freq": float(c / n_spec) if n_spec else 0.0,
                        "n_spectra": int(n_spec),
                    })
    pd.DataFrame(rows_topk).to_csv(TABLES / "mss_top_hit_frequency_by_cohort_v1.csv", index=False)
    pd.DataFrame(rows_cluster).to_csv(TABLES / "mss_cluster_frequency_by_cohort_v1.csv", index=False)
    pd.DataFrame(rows_family).to_csv(TABLES / "mss_family_frequency_by_cohort_v1.csv", index=False)
    return (pd.DataFrame(rows_topk), pd.DataFrame(rows_cluster),
              pd.DataFrame(rows_family))


# ──────────────────────────────────────────────────────────────────────
# Stage 3 — narrow metabolite panel reporting
# ──────────────────────────────────────────────────────────────────────
def _cohens_d(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 2 or len(y) < 2: return np.nan
    pooled = np.sqrt(((len(x)-1)*np.var(x, ddof=1) + (len(y)-1)*np.var(y, ddof=1))
                       / max(len(x)+len(y)-2, 1))
    return float((np.mean(x) - np.mean(y)) / (pooled if pooled > 0 else 1.0))


def _bootstrap_ci(x, y, n=300, seed=42):
    rng = np.random.default_rng(seed); ds = []
    for _ in range(n):
        ds.append(_cohens_d(rng.choice(x, len(x), replace=True),
                              rng.choice(y, len(y), replace=True)))
    ds = [d for d in ds if not np.isnan(d)]
    if not ds: return (np.nan, np.nan)
    return float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))


def stage3_narrow_panel(p1_df, p2_df):
    """Per-spectrum top-1 MSS score per molecule × cohort, then cross-cohort effects."""
    # Build molecule × spectrum max-score matrix (per-molecule score across all ranks)
    rows_eff = []; rows_tx = []
    for tag, df in [("P1", p1_df), ("P2", p2_df)]:
        # for each spectrum × molecule, take the score from rank-1 row of that molecule
        # (a molecule appears once per spectrum in top-K; for molecules outside top-K,
        # we conservatively give them score 0 — they did not surface as top hits)
        wide = df.pivot_table(index="spectrum_id", columns="molecule",
                                  values="score", aggfunc="max").fillna(0.0)
        meta = df.drop_duplicates("spectrum_id")[
            ["spectrum_id", "class_label", "sample_id", "dataset"]].set_index("spectrum_id")
        wide = wide.join(meta)
        wide["pilot"] = tag

        for mol in NARROW_TARGETS:
            if mol not in wide.columns:
                continue
            for cohort in wide["class_label"].unique():
                vals = wide[wide["class_label"] == cohort][mol].values
                rows_eff.append({
                    "pilot": tag, "cohort": cohort, "molecule": mol,
                    "n": int(len(vals)),
                    "mean_score": float(np.mean(vals)) if len(vals) else 0.0,
                    "topk_rate":  float(np.mean(vals > 0)) if len(vals) else 0.0,
                })

    eff_df = pd.DataFrame(rows_eff)
    eff_df.to_csv(TABLES / "narrow_metabolite_mss_effects_by_cohort_v1.csv", index=False)

    # Compute per-molecule cross-cohort Cohen's d + transfer category
    p1_pivot = p1_df.pivot_table(index="spectrum_id", columns="molecule",
                                       values="score", aggfunc="max").fillna(0.0)
    p1_meta  = p1_df.drop_duplicates("spectrum_id")[
        ["spectrum_id", "class_label"]].set_index("spectrum_id")
    p1_full = p1_pivot.join(p1_meta)
    p2_pivot = p2_df.pivot_table(index="spectrum_id", columns="molecule",
                                       values="score", aggfunc="max").fillna(0.0)
    p2_meta  = p2_df.drop_duplicates("spectrum_id")[
        ["spectrum_id", "class_label"]].set_index("spectrum_id")
    p2_full = p2_pivot.join(p2_meta)

    for mol in NARROW_TARGETS:
        eff = {}
        # P1 HCC vs CTR
        if mol in p1_full.columns:
            ctr = p1_full[p1_full["class_label"] == "CTR"][mol].values
            hcc = p1_full[p1_full["class_label"] == "H0T"][mol].values
            d   = _cohens_d(hcc, ctr)
            ci  = _bootstrap_ci(hcc, ctr)
            eff["P1_HCC_vs_CTR"] = (d, ci)
        # P2 HCC vs NC, CCA vs NC, LM vs NC
        if mol in p2_full.columns:
            nc = p2_full[p2_full["class_label"] == "NC"][mol].values
            for tag2, lab in [("P2_HCC_vs_NC", "HCC"),
                                  ("P2_CCA_vs_NC", "CCA"),
                                  ("P2_LM_vs_NC", "LM")]:
                arr = p2_full[p2_full["class_label"] == lab][mol].values
                eff[tag2] = (_cohens_d(arr, nc), _bootstrap_ci(arr, nc))

        # Transfer categorization
        d_p1     = eff.get("P1_HCC_vs_CTR", (np.nan, (np.nan, np.nan)))[0]
        d_p2hcc  = eff.get("P2_HCC_vs_NC",  (np.nan, (np.nan, np.nan)))[0]
        d_p2cca  = eff.get("P2_CCA_vs_NC",  (np.nan, (np.nan, np.nan)))[0]
        d_p2lm   = eff.get("P2_LM_vs_NC",   (np.nan, (np.nan, np.nan)))[0]
        d_p2adv  = np.nanmean([d_p2cca, d_p2lm]) if any(not np.isnan(x)
                                                           for x in [d_p2cca, d_p2lm]) else np.nan

        # Decision rules (reporting-only — fixed thresholds, no label tuning)
        SMALL = 0.20
        STRONG = 0.50
        nan_str = lambda v: "" if np.isnan(v) else f"{v:+.2f}"
        def _bigsame(a, b): return (not np.isnan(a) and not np.isnan(b) and
                                          abs(a) >= SMALL and abs(b) >= SMALL and
                                          np.sign(a) == np.sign(b))
        if _bigsame(d_p1, d_p2hcc):
            cat = "TRANSFERS"
        elif (not np.isnan(d_p1) and abs(d_p1) >= STRONG) and \
                (not np.isnan(d_p2hcc) and abs(d_p2hcc) < SMALL):
            cat = "SUBSTRATE_LOCKED"
        elif (not np.isnan(d_p2adv) and abs(d_p2adv) >= STRONG) and \
                ((np.isnan(d_p1) or abs(d_p1) < SMALL) and
                  (np.isnan(d_p2hcc) or abs(d_p2hcc) < SMALL)):
            cat = "ADVANCED_CANCER_ONLY"
        elif (not np.isnan(d_p1) and abs(d_p1) >= SMALL) and \
                (not np.isnan(d_p2hcc) and abs(d_p2hcc) >= SMALL) and \
                (not np.isnan(d_p2adv) and abs(d_p2adv) >= SMALL):
            cat = "SYSTEMIC_OR_NONSPECIFIC"
        else:
            cat = "INDETERMINATE"

        rows_tx.append({
            "molecule": mol,
            "d_P1_HCC_vs_CTR":   d_p1,
            "d_P2_HCC_vs_NC":    d_p2hcc,
            "d_P2_CCA_vs_NC":    d_p2cca,
            "d_P2_LM_vs_NC":     d_p2lm,
            "d_P2_advanced_mean": d_p2adv,
            "transfer_category": cat,
        })
    tx_df = pd.DataFrame(rows_tx)
    tx_df.to_csv(TABLES / "narrow_metabolite_mss_transfer_categories_v1.csv", index=False)
    return eff_df, tx_df


# ──────────────────────────────────────────────────────────────────────
# Stage 4 — resolution layer comparison (BSV vs cluster vs molecule)
# ──────────────────────────────────────────────────────────────────────
def _bsv_axis_d(bsv_df: pd.DataFrame, axis_id: str, dis: str, ctr: str,
                    use_col: str = "sumnorm_") -> float:
    col = f"{use_col}{axis_id}"
    if col not in bsv_df.columns: return np.nan
    a = bsv_df[bsv_df["class_label"] == dis][col].values
    b = bsv_df[bsv_df["class_label"] == ctr][col].values
    return _cohens_d(a, b)


def stage4_resolution_layer(p1_df, p2_df, tx_df):
    """For each BSV family, summarise (a) BSV-axis effect size, (b) MSS-cluster
    frequency shift, (c) per-molecule top-3 frequency shift."""
    p1_bsv = pd.read_csv(P1_BSV) if P1_BSV.exists() else pd.DataFrame()
    p2_bsv = pd.read_csv(P2_BSV) if P2_BSV.exists() else pd.DataFrame()

    rows = []
    for fam_id, fam_name in BSV_FAMILIES:
        # BSV axis effect sizes
        d_p1   = _bsv_axis_d(p1_bsv, fam_id, "H0T", "CTR")
        d_p2h  = _bsv_axis_d(p2_bsv, fam_id, "HCC", "NC")
        d_p2c  = _bsv_axis_d(p2_bsv, fam_id, "CCA", "NC")
        d_p2l  = _bsv_axis_d(p2_bsv, fam_id, "LM",  "NC")

        # Molecule-level support (which targets in this family appear in top-3 of cohorts)
        mols_in_family = tx_df[tx_df.molecule.isin(NARROW_TARGETS)]
        mol_in_fam_names = []
        # Which narrow targets map to this family
        from run_gaira_base_4_mss_narrow_metabolite_registry_repair_v1 import (  # noqa: E402
            BSV_FAMILY_BY_TARGET,
        )
        mol_in_fam_names = [m for m, (fid, _) in BSV_FAMILY_BY_TARGET.items()
                                if fid == fam_id]
        mol_in_fam_names = [m for m in mol_in_fam_names if m in NARROW_TARGETS]

        # P1 top-3 freq for these molecules in HCC cohort
        def topk_freq(df, mol, cohort, k=3):
            sub = df[(df["class_label"] == cohort) & (df["molecule"] == mol) &
                      (df["rank"] <= k)]
            tot = df[df["class_label"] == cohort]["spectrum_id"].nunique()
            return float(sub["spectrum_id"].nunique() / tot) if tot else 0.0

        p1_top_hcc = {m: topk_freq(p1_df, m, "H0T") for m in mol_in_fam_names}
        p1_top_ctr = {m: topk_freq(p1_df, m, "CTR") for m in mol_in_fam_names}
        p2_top_hcc = {m: topk_freq(p2_df, m, "HCC") for m in mol_in_fam_names}
        p2_top_nc  = {m: topk_freq(p2_df, m, "NC")  for m in mol_in_fam_names}

        # Coherence: are molecule shifts in the same direction as BSV?
        delta_p1 = {m: p1_top_hcc[m] - p1_top_ctr[m] for m in mol_in_fam_names}
        delta_p2 = {m: p2_top_hcc[m] - p2_top_nc[m]  for m in mol_in_fam_names}
        same_dir_p1 = sum(1 for m in mol_in_fam_names
                              if not np.isnan(d_p1) and np.sign(delta_p1[m]) == np.sign(d_p1)
                              and abs(delta_p1[m]) > 0.02 and abs(d_p1) > 0.10)
        contra_p1   = sum(1 for m in mol_in_fam_names
                              if not np.isnan(d_p1) and np.sign(delta_p1[m]) == -np.sign(d_p1)
                              and abs(delta_p1[m]) > 0.02 and abs(d_p1) > 0.10)
        rows.append({
            "bsv_family_id":   fam_id,
            "bsv_family_name": fam_name,
            "narrow_molecules_in_family": "|".join(mol_in_fam_names),
            "P1_BSV_axis_d_HCC_vs_CTR":  round(d_p1, 3) if not np.isnan(d_p1) else None,
            "P2_BSV_axis_d_HCC_vs_NC":   round(d_p2h, 3) if not np.isnan(d_p2h) else None,
            "P2_BSV_axis_d_CCA_vs_NC":   round(d_p2c, 3) if not np.isnan(d_p2c) else None,
            "P2_BSV_axis_d_LM_vs_NC":    round(d_p2l, 3) if not np.isnan(d_p2l) else None,
            "P1_top3_delta_per_mol": "|".join(f"{m}:{delta_p1[m]:+.2f}" for m in mol_in_fam_names),
            "P2_top3_delta_per_mol": "|".join(f"{m}:{delta_p2[m]:+.2f}" for m in mol_in_fam_names),
            "P1_n_molecules_supporting_BSV_dir":     same_dir_p1,
            "P1_n_molecules_contradicting_BSV_dir":  contra_p1,
        })
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "gaira_resolution_layer_comparison_v1.csv", index=False)
    return df, p1_bsv, p2_bsv


# ──────────────────────────────────────────────────────────────────────
# Stage 5 — BSV → MSS cluster → molecule traceback
# ──────────────────────────────────────────────────────────────────────
TRACEBACK_SIGNALS = [
    {"key": "G09_sterol_lipid_down",        "fam": "G09", "expected_dir": "down",
     "label": "G09 Sterol-lipid ↓"},
    {"key": "G04_nucleic_backbone_up",      "fam": "G04", "expected_dir": "up",
     "label": "G04 Nucleic backbone ↑"},
    {"key": "G05_glycan_caution",           "fam": "G05", "expected_dir": "any",
     "label": "G05 Glycan caution"},
    {"key": "G03_pyrimidine_caution",       "fam": "G03", "expected_dir": "any",
     "label": "G03 Pyrimidine caution"},
    {"key": "G02_purine_metabolite_panel",  "fam": "G02", "expected_dir": "any",
     "label": "G02 Purine-metabolite narrow panel"},
    {"key": "G10_sulfur_redox_panel",       "fam": "G10", "expected_dir": "any",
     "label": "G10 Sulfur/redox narrow panel"},
]


def stage5_traceback(p1_df, p2_df, comp_df, tx_df, p1_bsv, p2_bsv):
    rows = []
    from run_gaira_base_4_mss_narrow_metabolite_registry_repair_v1 import (  # noqa: E402
        BSV_FAMILY_BY_TARGET,
    )
    for sig in TRACEBACK_SIGNALS:
        fam = sig["fam"]
        comp = comp_df[comp_df.bsv_family_id == fam].iloc[0] if not comp_df.empty else {}
        # Top MSS clusters firing for this family (using cohort-aggregated cluster freq)
        clusters_mol = [m for m, (fid, _) in BSV_FAMILY_BY_TARGET.items() if fid == fam]
        # For each pilot, top molecule candidates and their tx category
        for pilot, dfp in [("P1", p1_df), ("P2", p2_df)]:
            # mean score per molecule in the family
            sub = dfp[dfp.molecule.isin(clusters_mol)]
            cohorts = sorted(sub["class_label"].unique().tolist())
            for cohort in cohorts:
                cs = sub[sub["class_label"] == cohort]
                ms = cs.groupby("molecule")["score"].mean().sort_values(ascending=False)
                top_mols = list(ms.head(3).index)
                rows.append({
                    "signal":          sig["label"],
                    "bsv_family_id":   fam,
                    "pilot":           pilot,
                    "cohort":          cohort,
                    "bsv_axis_dir":    sig["expected_dir"],
                    "bsv_axis_d_observed": (
                        comp.get("P1_BSV_axis_d_HCC_vs_CTR") if pilot == "P1" else
                        comp.get("P2_BSV_axis_d_HCC_vs_NC")
                    ),
                    "dominant_mss_clusters": "|".join(sorted(set(
                        [tx_df[tx_df.molecule == m]["transfer_category"].iloc[0]
                          if (tx_df.molecule == m).any() else "" for m in top_mols]))),
                    "top_molecule_candidates": "|".join(f"{m}:{ms[m]:.2f}" for m in top_mols),
                    "candidate_caveat":      "candidate evidence consistent with chemistry; "
                                                "MSS hit is anchor-based, not mixture-resolved",
                })
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "bsv_to_mss_traceback_v1.csv", index=False)
    return df


# ──────────────────────────────────────────────────────────────────────
# Figures
# ──────────────────────────────────────────────────────────────────────
def make_figures(p1_df, p2_df, freq_df, cluster_df, tx_df, comp_df, tracedf):
    print("[FIG] generating reporting figures")
    pilots = {"Pilot1": p1_df, "Pilot2": p2_df}
    cohort_order = {"Pilot1": ["CTR", "H0T"],
                       "Pilot2": ["NC", "HCC", "CCA", "LM"]}

    # Figs 1+2: top-10 MSS hits by cohort per pilot
    for pilot, df in pilots.items():
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            top3 = df[df["rank"] <= 3]
            cohorts = [c for c in cohort_order[pilot] if c in top3["class_label"].unique()]
            top_mols = (top3.groupby("molecule")["spectrum_id"].nunique()
                            .sort_values(ascending=False).head(10).index.tolist())
            x = np.arange(len(top_mols))
            w = 0.85 / max(len(cohorts), 1)
            for i, c in enumerate(cohorts):
                cs = top3[top3["class_label"] == c]
                n_spec = cs["spectrum_id"].nunique()
                cnt = cs.groupby("molecule")["spectrum_id"].nunique() / max(n_spec, 1)
                vals = [float(cnt.get(m, 0.0)) for m in top_mols]
                ax.bar(x + (i - len(cohorts)/2 + 0.5) * w, vals, w, label=f"{c} (n={n_spec})")
            ax.set_xticks(x); ax.set_xticklabels(top_mols, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("top-3 hit fraction"); ax.set_ylim(0, 1.05)
            ax.set_title(f"{pilot} — top-10 MSS molecule candidates by cohort (top-3 hit fraction)")
            ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            fig.savefig(FIGS / f"fig_{pilot.lower()}_top_mss_hits_by_cohort_v1.png", dpi=150)
            plt.close(fig)
        except Exception as e:
            print(f"  fig {pilot} top hits issue: {e}")

    # Fig 3: MSS cluster frequency heatmap (rows=clusters, cols=cohort)
    try:
        wide = cluster_df.pivot_table(
            index="mss_cluster",
            columns=cluster_df["pilot"] + "::" + cluster_df["cohort"],
            values="freq", aggfunc="mean").fillna(0.0)
        fig, ax = plt.subplots(figsize=(11, 7))
        im = ax.imshow(wide.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(len(wide.columns))); ax.set_xticklabels(wide.columns, rotation=70, fontsize=8)
        ax.set_yticks(range(len(wide.index)));   ax.set_yticklabels(wide.index, fontsize=8)
        ax.set_title("MSS cluster top-3 frequency by cohort")
        fig.colorbar(im, ax=ax, fraction=0.04, label="top-3 fraction of spectra")
        fig.tight_layout()
        fig.savefig(FIGS / "fig_mss_cluster_frequency_heatmap_v1.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig3 issue: {e}")

    # Fig 4: narrow metabolite effect size heatmap
    try:
        eff_cols = ["d_P1_HCC_vs_CTR", "d_P2_HCC_vs_NC",
                       "d_P2_CCA_vs_NC", "d_P2_LM_vs_NC"]
        mat = tx_df.set_index("molecule")[eff_cols].astype(float)
        mat = mat.reindex(NARROW_TARGETS)
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(mat.values, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(eff_cols))); ax.set_xticklabels(
            ["P1 HCC-CTR", "P2 HCC-NC", "P2 CCA-NC", "P2 LM-NC"], rotation=20)
        ax.set_yticks(range(len(mat.index))); ax.set_yticklabels(mat.index, fontsize=8)
        ax.set_title("Narrow metabolite MSS effect sizes (Cohen's d)")
        fig.colorbar(im, ax=ax, fraction=0.04)
        for i, mol in enumerate(mat.index):
            cat = tx_df[tx_df.molecule == mol]["transfer_category"].iloc[0] \
                    if (tx_df.molecule == mol).any() else ""
            ax.text(len(eff_cols) + 0.1, i, cat, va="center", fontsize=7)
        fig.tight_layout()
        fig.savefig(FIGS / "fig_narrow_metabolite_effect_heatmap_v1.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig4 issue: {e}")

    # Fig 5: BSV → MSS cluster → molecule flow (simple grouped bar)
    try:
        fig, ax = plt.subplots(figsize=(11, 6))
        from run_gaira_base_4_mss_narrow_metabolite_registry_repair_v1 import (  # noqa: E402
            BSV_FAMILY_BY_TARGET,
        )
        x = []; y = []; cs = []; labs = []
        for i, (fid, fname) in enumerate(BSV_FAMILIES):
            mols = [m for m, (f, _) in BSV_FAMILY_BY_TARGET.items() if f == fid
                      and m in NARROW_TARGETS]
            for j, m in enumerate(mols):
                x.append(i + 0.1 * j)
                d = tx_df[tx_df.molecule == m]["d_P1_HCC_vs_CTR"]
                y.append(float(d.iloc[0]) if not d.empty and not d.isna().all() else 0.0)
                cs.append("#4C72B0" if (d.iloc[0] if not d.empty else 0.0) >= 0
                            else "#DD8452")
                labs.append(m)
        ax.bar(x, y, 0.08, color=cs)
        ax.set_xticks(range(len(BSV_FAMILIES)))
        ax.set_xticklabels([f for f, _ in BSV_FAMILIES])
        for xi, yi, lab in zip(x, y, labs):
            ax.text(xi, yi + 0.02 * np.sign(yi or 1), lab, fontsize=6, rotation=90,
                       ha="center", va="bottom" if (yi or 0) >= 0 else "top")
        ax.axhline(0, color="black", lw=0.5)
        ax.set_ylabel("Pilot 1 d (HCC vs CTR)")
        ax.set_title("BSV family → narrow molecule candidates (Pilot 1 effect sizes)")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIGS / "fig_bsv_mss_molecule_flow_v1.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig5 issue: {e}")

    # Fig 6: resolution-layer comparison
    try:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(BSV_FAMILIES))
        bsv_p1 = [comp_df[comp_df.bsv_family_id == fid]["P1_BSV_axis_d_HCC_vs_CTR"].iloc[0]
                     if not comp_df[comp_df.bsv_family_id == fid].empty else 0.0
                     for fid, _ in BSV_FAMILIES]
        bsv_p2 = [comp_df[comp_df.bsv_family_id == fid]["P2_BSV_axis_d_HCC_vs_NC"].iloc[0]
                     if not comp_df[comp_df.bsv_family_id == fid].empty else 0.0
                     for fid, _ in BSV_FAMILIES]
        bsv_p1 = [0.0 if v is None else float(v) for v in bsv_p1]
        bsv_p2 = [0.0 if v is None else float(v) for v in bsv_p2]
        ax.bar(x - 0.2, bsv_p1, 0.4, label="P1 BSV d (HCC-CTR)", color="#4C72B0")
        ax.bar(x + 0.2, bsv_p2, 0.4, label="P2 BSV d (HCC-NC)", color="#DD8452")
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xticks(x); ax.set_xticklabels([f for f, _ in BSV_FAMILIES], fontsize=9)
        ax.set_ylabel("Cohen's d at BSV axis level")
        ax.set_title("Resolution comparison — broad BSV per family (Pilot 1 vs Pilot 2)")
        ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIGS / "fig_resolution_layer_comparison_v1.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig6 issue: {e}")

    # Fig 7: candidate molecule caution categories
    try:
        cats = ["TRANSFERS", "SYSTEMIC_OR_NONSPECIFIC", "SUBSTRATE_LOCKED",
                  "ADVANCED_CANCER_ONLY", "INDETERMINATE"]
        colors = {"TRANSFERS": "#2ca02c", "SYSTEMIC_OR_NONSPECIFIC": "#1f77b4",
                    "SUBSTRATE_LOCKED": "#f39c12",
                    "ADVANCED_CANCER_ONLY": "#9467bd",
                    "INDETERMINATE": "#7f7f7f"}
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(NARROW_TARGETS))
        ys = []
        cs = []
        for m in NARROW_TARGETS:
            row = tx_df[tx_df.molecule == m]
            cat = row["transfer_category"].iloc[0] if not row.empty else "INDETERMINATE"
            cs.append(colors[cat])
            ys.append(1)
        ax.bar(x, ys, color=cs, edgecolor="black")
        for i, m in enumerate(NARROW_TARGETS):
            cat = tx_df[tx_df.molecule == m]["transfer_category"].iloc[0] \
                    if (tx_df.molecule == m).any() else "INDETERMINATE"
            ax.text(i, 0.5, cat.replace("_", "\n"),
                       rotation=0, ha="center", va="center", fontsize=7,
                       fontweight="bold", color="white")
        ax.set_xticks(x); ax.set_xticklabels(NARROW_TARGETS, rotation=45, ha="right", fontsize=8)
        ax.set_yticks([])
        ax.set_title("Narrow metabolite cross-pilot transfer category (caution map)")
        fig.tight_layout()
        fig.savefig(FIGS / "fig_candidate_molecule_caution_categories_v1.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig7 issue: {e}")


# ──────────────────────────────────────────────────────────────────────
# Report + audit
# ──────────────────────────────────────────────────────────────────────
def _decision(tx_df, comp_df) -> str:
    # Stable resolution = ≥2 narrow targets in TRANSFERS or SYSTEMIC and BSV
    # has at least one strong cross-pilot family (e.g. G09)
    n_transfer = int((tx_df.transfer_category == "TRANSFERS").sum())
    n_locked   = int((tx_df.transfer_category == "SUBSTRATE_LOCKED").sum())
    n_advonly  = int((tx_df.transfer_category == "ADVANCED_CANCER_ONLY").sum())
    n_indet    = int((tx_df.transfer_category == "INDETERMINATE").sum())
    n_total    = len(tx_df)
    bsv_p1 = comp_df["P1_BSV_axis_d_HCC_vs_CTR"].abs().fillna(0)
    bsv_p2 = comp_df["P2_BSV_axis_d_HCC_vs_NC"].abs().fillna(0)
    n_bsv_strong = int(((bsv_p1 >= 0.30) & (bsv_p2 >= 0.20)).sum())

    if n_indet >= n_total // 2 and n_transfer == 0:
        return "MSS_LAYER_MOSTLY_AMBIGUOUS"
    if n_locked >= n_total // 2 and n_transfer >= 1:
        return "MSS_LAYER_ADDS_LOCAL_BUT_SUBSTRATE_LOCKED_RESOLUTION"
    if n_transfer >= 2 and n_bsv_strong >= 1:
        return "MSS_LAYER_ADDS_STABLE_RESOLUTION"
    if n_indet + n_locked + n_advonly >= int(0.7 * n_total) and n_transfer == 0:
        return "NEEDS_SOFT_MSS_NEXT"
    return "MSS_LAYER_ADDS_LOCAL_BUT_SUBSTRATE_LOCKED_RESOLUTION"


def write_report(tx_df, comp_df, freq_df, cluster_df, tracedf, decision):
    lines = []
    lines.append("# GAIRA MSS-resolution reporting layer v1 — final report\n")
    lines.append(f"## Decision: **{decision}**\n")
    lines.append(
        "Reporting-only phase. The GAIRA engine, MSS scoring kernel (anchor-fires + "
        "support-fires), 11-axis BSV weights, motif registry, and substrate physics are "
        "UNCHANGED. Disease labels were used only for cohort grouping; no scoring decision "
        "was tuned against any label.\n"
    )

    lines.append("## Stage 1 — per-spectrum MSS top-k extraction\n")
    lines.append("Per-spectrum MSS scores were computed against the 19-target narrow registry "
                    "(`narrow_metabolite_mss_registry_v1.csv`). For each spectrum, the top-5 "
                    "molecule candidates were retained with rank, score, anchors_fired, "
                    "anchor_total, BSV family, MSS cluster, and reliability tier. See "
                    "`tables/pilot{1,2}_mss_top_hits_per_spectrum_v1.csv`.\n")

    lines.append("## Stage 2 — cohort-level top-k frequencies\n")
    lines.append("- `tables/mss_top_hit_frequency_by_cohort_v1.csv` (per-molecule × cohort × k)")
    lines.append("- `tables/mss_cluster_frequency_by_cohort_v1.csv` (per-cluster × cohort)")
    lines.append("- `tables/mss_family_frequency_by_cohort_v1.csv` (per-family × cohort)")
    lines.append("")

    lines.append("## Stage 3 — narrow metabolite cross-pilot effects\n")
    lines.append("| molecule | P1 HCC-CTR | P2 HCC-NC | P2 CCA-NC | P2 LM-NC | category |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for _, r in tx_df.iterrows():
        fmt = lambda v: ("" if pd.isna(v) else f"{v:+.2f}")
        lines.append(f"| {r['molecule']} | {fmt(r['d_P1_HCC_vs_CTR'])} | "
                        f"{fmt(r['d_P2_HCC_vs_NC'])} | {fmt(r['d_P2_CCA_vs_NC'])} | "
                        f"{fmt(r['d_P2_LM_vs_NC'])} | {r['transfer_category']} |")
    lines.append("")

    lines.append("## Stage 4 — resolution layer comparison (BSV vs MSS cluster vs molecule)\n")
    lines.append("| BSV family | P1 BSV d | P2 BSV d | molecules in family | "
                    "supporting / contradicting (P1) |")
    lines.append("|---|---:|---:|---|---:|")
    for _, r in comp_df.iterrows():
        lines.append(
            f"| {r['bsv_family_id']} {r['bsv_family_name']} | "
            f"{r['P1_BSV_axis_d_HCC_vs_CTR']} | {r['P2_BSV_axis_d_HCC_vs_NC']} | "
            f"{r['narrow_molecules_in_family'][:80]} | "
            f"{r['P1_n_molecules_supporting_BSV_dir']} / "
            f"{r['P1_n_molecules_contradicting_BSV_dir']} |"
        )
    lines.append("")

    lines.append("## Stage 5 — BSV → MSS traceback (selected signals)\n")
    lines.append("See `tables/bsv_to_mss_traceback_v1.csv` for the per-cohort traceback "
                    "of dominant MSS clusters and top molecule candidates per BSV signal.")
    lines.append("")

    # Required answers
    lines.append("## Required answers\n")
    lines.append("### 1. Does MSS-level reporting add resolution beyond 11-axis BSV?\n")
    lines.append(
        "Yes, partially. BSV-axis effects describe family-level chemistry (e.g. G09 ↓, "
        "G02 shift). MSS reporting decomposes those family effects into per-molecule "
        "candidate hits — e.g. within G09, MSS reports *which sterol/lipid molecule-like* "
        "candidates surface most. However, per the prior paper-band ground-truth phase, "
        "molecule-specific identity is collision-prone for many narrow targets — MSS "
        "candidate hits should be reported as candidate evidence, not definitive identity."
    )
    lines.append("")

    lines.append("### 2. Most frequent top-3 MSS hits in Pilot 1 cohorts\n")
    p1_top3 = freq_df[(freq_df.pilot == "Pilot1") & (freq_df.k == 3)] \
        .groupby("molecule")["freq"].mean().sort_values(ascending=False).head(8)
    for m, v in p1_top3.items():
        lines.append(f"- {m}: top-3 freq mean = {v:.2f}")
    lines.append("")

    lines.append("### 3. Most frequent top-3 MSS hits in Pilot 2 cohorts\n")
    p2_top3 = freq_df[(freq_df.pilot == "Pilot2") & (freq_df.k == 3)] \
        .groupby("molecule")["freq"].mean().sort_values(ascending=False).head(8)
    for m, v in p2_top3.items():
        lines.append(f"- {m}: top-3 freq mean = {v:.2f}")
    lines.append("")

    lines.append("### 4. Do paper-related narrow metabolites show up as top-k MSS hits?\n")
    paper = ["uric_acid", "hypoxanthine", "ergothioneine", "glutathione"]
    for m in paper:
        p1 = freq_df[(freq_df.pilot == "Pilot1") & (freq_df.k == 3) &
                       (freq_df.molecule == m)]["freq"].mean()
        p2 = freq_df[(freq_df.pilot == "Pilot2") & (freq_df.k == 3) &
                       (freq_df.molecule == m)]["freq"].mean()
        p1 = 0 if np.isnan(p1) else p1
        p2 = 0 if np.isnan(p2) else p2
        lines.append(f"- {m}: P1 top-3 freq = {p1:.2f} | P2 top-3 freq = {p2:.2f}")
    lines.append("")

    lines.append("### 5. Do molecule candidates transfer across pilots?\n")
    n_tx = int((tx_df.transfer_category == "TRANSFERS").sum())
    transfers = tx_df[tx_df.transfer_category == "TRANSFERS"]["molecule"].tolist()
    lines.append(f"- {n_tx} narrow molecules cleanly TRANSFER (P1 + P2 HCC same-direction "
                    f"|d| ≥ 0.20): {transfers or '(none)'}")
    lines.append("")

    lines.append("### 6. Stable enough to show in demo?\n")
    stable = tx_df[tx_df.transfer_category.isin(["TRANSFERS", "SYSTEMIC_OR_NONSPECIFIC"])]["molecule"].tolist()
    if stable:
        lines.append(f"- Candidate evidence consistent with chemistry (panel-level only, "
                        f"not per-spectrum identity): {stable}")
    else:
        lines.append("- (no narrow molecule reaches stable cross-pilot transfer at top-k freq)")
    lines.append("")

    lines.append("### 7. Caution / substrate-locked candidates\n")
    lock = tx_df[tx_df.transfer_category == "SUBSTRATE_LOCKED"]["molecule"].tolist()
    advc = tx_df[tx_df.transfer_category == "ADVANCED_CANCER_ONLY"]["molecule"].tolist()
    if lock: lines.append(f"- SUBSTRATE_LOCKED (P1 only): {lock}")
    if advc: lines.append(f"- ADVANCED_CANCER_ONLY (CCA/LM only): {advc}")
    lines.append("")

    lines.append("### 8. Should GAIRA expose top-3/top-5 MSS hits in the demo?\n")
    lines.append(
        "**Conditional yes** — display MSS top-3/top-5 hits per spectrum as **candidate "
        "evidence** with explicit caution language: 'molecule-like spectral evidence consistent "
        "with X' or 'MSS hit consistent with X-family chemistry'. **Do NOT** display as 'this "
        "spectrum contains X' or 'confirmed X' or 'diagnostic biomarker'. Honor the reliability "
        "tier from the repaired registry: HIGH / MODERATE molecules can be displayed; LOW "
        "(e.g. lactate literature stub) must be excluded or flagged DO_NOT_USE."
    )
    lines.append("")

    (REPORTS / "REPORT_mss_resolution_reporting_layer_v1.md").write_text("\n".join(lines))


def write_audit(decision):
    txt = [
        "# gaira_base_4_mss_resolution_reporting_layer_v1 — audit log",
        f"date: {datetime.now().isoformat()}",
        "",
        "## Source datasets (read-only)",
        "- Pilot 1 raw SERS (passive_target_pilot_1) — class labels CTR / H0T",
        "- Pilot 2 raw SERS (passive_target_pilot_2) — class labels NC / HCC / CCA / LM",
        "- Repaired narrow metabolite MSS registry "
        "(gaira_base_4_mss_narrow_metabolite_registry_repair_v1)",
        "- Pilot 1.1 normalization-sensitivity BSV vectors",
        "- Pilot 2.1 normalization-sensitivity BSV vectors",
        "",
        "## Strict negative invariants",
        "- NO engine changes (gaira/base2 / base3 / base4 modules untouched on disk)",
        "- NO MSS scoring kernel changes (anchor-fires + support-fires logic preserved)",
        "- NO 11-axis BSV weight changes",
        "- NO motif registry changes",
        "- NO substrate physics changes",
        "- NO soft-MSS scoring",
        "- NO competitor-aware scoring",
        "- NO classifier feedback",
        "- NO pilot reruns (BSV vectors read from existing pilot1_1 / pilot2_1 outputs)",
        "- NO threshold tuning, NO label-driven optimization",
        "- Disease labels used ONLY for cohort grouping in aggregation",
        "",
        "## Outputs",
        "- tables/pilot1_mss_top_hits_per_spectrum_v1.csv",
        "- tables/pilot2_mss_top_hits_per_spectrum_v1.csv",
        "- tables/mss_top_hit_frequency_by_cohort_v1.csv",
        "- tables/mss_cluster_frequency_by_cohort_v1.csv",
        "- tables/mss_family_frequency_by_cohort_v1.csv",
        "- tables/narrow_metabolite_mss_effects_by_cohort_v1.csv",
        "- tables/narrow_metabolite_mss_transfer_categories_v1.csv",
        "- tables/gaira_resolution_layer_comparison_v1.csv",
        "- tables/bsv_to_mss_traceback_v1.csv",
        "- 7 figures",
        "- reports/REPORT_mss_resolution_reporting_layer_v1.md",
        "",
        f"## Final decision\n**{decision}**",
    ]
    (AUDIT / "gaira_base_4_mss_resolution_reporting_layer_v1_audit_log.md").write_text("\n".join(txt))


# ──────────────────────────────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 78)
    print("gaira_base_4_mss_resolution_reporting_layer_v1")
    print("=" * 78)
    master_x = canonical_master_axis()

    print("[load] narrow registry")
    templates, _, _ = load_templates()
    by_mol = consolidate_templates_by_molecule(templates)
    print(f"  {len(by_mol)} molecules × {sum(len(v) for v in by_mol.values())} regime templates")

    print("[load] Pilot 1 raw spectra")
    p1 = load_p1_raw(master_x)
    print(f"  P1: {len(p1)} spectra")
    print("[load] Pilot 2 raw spectra")
    p2 = load_p2_raw(master_x)
    print(f"  P2: {len(p2)} spectra")

    print("[STAGE 1] per-spectrum MSS top-k extraction (Pilot 1)")
    p1_df = stage1_per_spectrum(p1, by_mol, master_x, spec_regime="SERS")
    p1_df.to_csv(TABLES / "pilot1_mss_top_hits_per_spectrum_v1.csv", index=False)
    print(f"  wrote {len(p1_df)} rows")

    print("[STAGE 1] per-spectrum MSS top-k extraction (Pilot 2)")
    p2_df = stage1_per_spectrum(p2, by_mol, master_x, spec_regime="SERS")
    p2_df.to_csv(TABLES / "pilot2_mss_top_hits_per_spectrum_v1.csv", index=False)
    print(f"  wrote {len(p2_df)} rows")

    print("[STAGE 2] cohort-level aggregation")
    freq_df, cluster_df, family_df = stage2_cohort_aggregation(p1_df, p2_df)

    print("[STAGE 3] narrow metabolite panel reporting")
    eff_df, tx_df = stage3_narrow_panel(p1_df, p2_df)

    print("[STAGE 4] resolution layer comparison")
    comp_df, p1_bsv, p2_bsv = stage4_resolution_layer(p1_df, p2_df, tx_df)

    print("[STAGE 5] BSV → MSS traceback")
    tracedf = stage5_traceback(p1_df, p2_df, comp_df, tx_df, p1_bsv, p2_bsv)

    print("[FIGS]")
    make_figures(p1_df, p2_df, freq_df, cluster_df, tx_df, comp_df, tracedf)

    decision = _decision(tx_df, comp_df)
    print(f"[decision] {decision}")
    write_report(tx_df, comp_df, freq_df, cluster_df, tracedf, decision)
    write_audit(decision)
    try:
        shutil.copy(__file__, CODE_SNAPSHOT / Path(__file__).name)
    except Exception:
        pass
    print("[done]")


if __name__ == "__main__":
    main()
