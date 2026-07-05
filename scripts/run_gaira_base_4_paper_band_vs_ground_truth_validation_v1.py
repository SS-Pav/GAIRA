"""gaira_base_4_paper_band_vs_ground_truth_validation_v1

Phase: PAPER-BAND vs GROUND-TRUTH validation.

Goal: independently of any pilot cohort, evaluate whether the narrow-band
assignments used by the Pilot 1 HCC SERS paper (Gurian / Bonifacio) are
supported by GAIRA pure-molecule grounding spectra and whether those bands
are molecule-specific or collision-prone across a wider comparator panel.

Constraints (NEVER violated):
- Engine v4.5 unchanged
- MSS v4.3 / motif / taxonomy / substrate physics: read-only
- No threshold tuning, no classifier, no label-driven optimization
- Paper bands taken VERBATIM from the paper at fixed positions

Outputs:
  /Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_4_paper_band_vs_ground_truth_validation_v1/
    tables/      (CSV)
    figures/     (PNG)
    reports/     (markdown)
    audit/       (markdown)
    code_snapshot/

Run:
    cd /Users/suraj/projects/GAIRA
    PYTHONPATH=src .venv/bin/python scripts/run_gaira_base_4_paper_band_vs_ground_truth_validation_v1.py
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
from scipy.signal import find_peaks

warnings.simplefilter("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gaira.spectral import canonical_master_axis  # noqa: E402

# Reuse existing grounding loaders verbatim (no engine writes)
from run_gaira_validate_2_grounding import (  # noqa: E402
    load_ramanbiolib, load_gobbato_powder,
    load_amino_acid_xlsx, load_digitised_literature,
)
from run_gaira_base_3_full_grounding_audit_and_signature_build_v1 import (  # noqa: E402
    load_sers_metabolite_63,
)
from run_gaira_base_4_hybrid_bsv_controlled_calibration_v2 import (  # noqa: E402
    load_sers_fitting, load_isotopic, load_uricase, load_erg_calibration,
)


# ──────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────
ROOT = Path("/Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_4_paper_band_vs_ground_truth_validation_v1")
TABLES = ROOT / "tables"
FIGS   = ROOT / "figures"
REPORTS = ROOT / "reports"
AUDIT  = ROOT / "audit"
CODE_SNAPSHOT = ROOT / "code_snapshot"
for d in (TABLES, FIGS, REPORTS, AUDIT, CODE_SNAPSHOT):
    d.mkdir(parents=True, exist_ok=True)

MSS_REGISTRY = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_4_mss_decision_enrichment_v1/"
    "registry/grounding_molecular_signatures_v4_3.csv"
)


# ──────────────────────────────────────────────────────────────────────
# Paper band registry (verbatim from Pilot 1 paper)
# ──────────────────────────────────────────────────────────────────────
PAPER_BANDS = {
    "uric_acid":     [594.0, 638.0, 812.0, 888.0, 1132.0],
    "hypoxanthine":  [724.0],
    "ergothioneine": [480.0, 1220.0, 1442.0, 1582.0],
    "glutathione":   [664.0, 912.0],
}
PAPER_TARGETS = list(PAPER_BANDS.keys())

PAPER_DIRECTION = {
    "uric_acid":     "positive_panel",   # HCC > CTR
    "hypoxanthine":  "negative_panel",   # CTR > HCC
    "ergothioneine": "negative_panel",   # CTR > HCC
    "glutathione":   "tentative_negative_panel",
}


# ──────────────────────────────────────────────────────────────────────
# Canonical molecule normalization
# ──────────────────────────────────────────────────────────────────────
COMPARATORS = [
    "uric_acid", "hypoxanthine", "ergothioneine", "glutathione",
    "xanthine", "adenine", "guanine", "lactate",
    "cysteine", "cystine",
    "tryptophan", "phenylalanine", "tyrosine",
    "cholesterol", "oleic_acid", "palmitic_acid", "stearic_acid",
]

# Map raw component_key/cohort tokens (lower) → canonical molecule
NAME_MAP = {
    # uric acid family
    "uric acid": "uric_acid",
    "auric acid": "uric_acid",          # registry typo variant
    "ua": "uric_acid",
    "ua_free": "uric_acid",
    "ua_bound": "uric_acid",
    "uric_acid": "uric_acid",
    # hypoxanthine
    "hypoxanthine": "hypoxanthine",
    "hypox": "hypoxanthine",
    "hyp": "hypoxanthine",
    "hx": "hypoxanthine",
    # xanthine / purines
    "xanthine": "xanthine",
    "xan": "xanthine",
    "xanth": "xanthine",
    "adenine": "adenine",
    "ade": "adenine",
    "guanine": "guanine",
    "gua": "guanine",
    # ergothioneine
    "ergothioneine": "ergothioneine",
    "ergo": "ergothioneine",
    "erg": "ergothioneine",
    # glutathione
    "glutathione": "glutathione",
    "gsh": "glutathione",
    "g-glu": "glutathione",
    "γ-glu-cys-gly": "glutathione",
    # cysteine
    "cysteine": "cysteine",
    "l-cysteine": "cysteine",
    "cys": "cysteine",
    "cystine": "cystine",
    # aromatic AAs
    "tryptophan": "tryptophan",
    "l-tryptophan": "tryptophan",
    "trp": "tryptophan",
    "phenylalanine": "phenylalanine",
    "l-phenylalanine": "phenylalanine",
    "phe": "phenylalanine",
    "tyrosine": "tyrosine",
    "l-tyrosine": "tyrosine",
    "tyr": "tyrosine",
    # lipids
    "cholesterol": "cholesterol",
    "chol": "cholesterol",
    "oleic acid": "oleic_acid",
    "oleic_acid": "oleic_acid",
    "palmitic acid": "palmitic_acid",
    "palmitic_acid": "palmitic_acid",
    "stearic acid": "stearic_acid",
    "stearic_acid": "stearic_acid",
    # lactate
    "lactic acid": "lactate",
    "l-lactic acid": "lactate",
    "lactate": "lactate",
    "sodium lactate": "lactate",
    # serum-only / background uricase cohorts (for completeness)
    "serumsigma": "serum_background",
    "serumsigma+enzyme": "serum_background_uricase_treated",
    "serumspiked": "serum_spiked",
    "serumspiked+enzyme": "serum_spiked_uricase_treated",
}

def canonicalize(name: str | None) -> str | None:
    if name is None:
        return None
    s = str(name).strip().lower()
    s = s.replace("(", " ").replace(")", " ").replace(",", " ")
    s = " ".join(s.split())
    # try exact, then prefix-stripped
    if s in NAME_MAP:
        return NAME_MAP[s]
    # strip leading "l-" / "d-" / "d-(+)-" / "(+/-)-" etc
    for pre in ["l-", "d-", "d-(+)-", "d-(-)-", "(+)-", "(-)-", "(+/-)-",
                  "n-acetyl-", "n-acetyl ", "alpha-", "beta-", "γ-", "g-"]:
        if s.startswith(pre):
            stripped = s[len(pre):]
            if stripped in NAME_MAP:
                return NAME_MAP[stripped]
    # known suffix-tolerant matches
    for k in NAME_MAP:
        if s == k or s.startswith(k + " ") or s.endswith(" " + k):
            return NAME_MAP[k]
    # fatty-acid family — palmitoleic/linoleic/stearic etc.
    if "palmitic" in s: return "palmitic_acid"
    if "stearic"  in s: return "stearic_acid"
    if "oleic"    in s and "linoleic" not in s and "palmitoleic" not in s: return "oleic_acid"
    if "cholesterol" in s: return "cholesterol"
    return None


# ──────────────────────────────────────────────────────────────────────
# Spectrum primitives
# ──────────────────────────────────────────────────────────────────────
def _idx_at(master_x: np.ndarray, cm1: float) -> int:
    return int(np.argmin(np.abs(master_x - cm1)))

def _local_window(y: np.ndarray, master_x: np.ndarray, cm1: float, half: float):
    lo = _idx_at(master_x, cm1 - half); hi = _idx_at(master_x, cm1 + half) + 1
    lo = max(lo, 0); hi = min(hi, len(y))
    return y[lo:hi], lo, hi

def _local_max(y: np.ndarray, master_x: np.ndarray, cm1: float, half: float):
    win, lo, _ = _local_window(y, master_x, cm1, half)
    if len(win) == 0:
        return None, None
    j = int(np.argmax(win))
    return float(win[j]), float(master_x[lo + j])

def _local_prominence(y: np.ndarray, master_x: np.ndarray, cm1: float, half: float):
    win, lo, _ = _local_window(y, master_x, cm1, half)
    if len(win) < 3:
        return 0.0
    bg_left  = float(np.percentile(y[max(lo - 25, 0):lo], 30)) if lo > 5 else 0.0
    bg_right_lo = lo + len(win)
    bg_right = float(np.percentile(y[bg_right_lo:min(bg_right_lo + 25, len(y))], 30)) if bg_right_lo + 5 < len(y) else 0.0
    bg = (bg_left + bg_right) / 2.0
    return max(float(win.max()) - bg, 0.0)

def _spectrum_peaks(y: np.ndarray, master_x: np.ndarray, prom_frac: float = 0.05):
    rng = float(y.max() - y.min())
    if rng <= 0:
        return np.array([], dtype=int)
    idx, _ = find_peaks(y, prominence=prom_frac * rng)
    return idx

def _peak_rank(y: np.ndarray, master_x: np.ndarray, cm1: float, half: float = 5.0):
    """Rank of the local-max within ±half cm⁻¹ among all spectrum peaks
    (1 = tallest peak in spectrum). If no peak found, returns None."""
    idx = _spectrum_peaks(y, master_x)
    if len(idx) == 0:
        return None
    heights = y[idx]
    order = np.argsort(-heights)  # tallest first
    ranked = idx[order]
    in_win = []
    for k, ix in enumerate(ranked, start=1):
        if abs(master_x[ix] - cm1) <= half:
            in_win.append((k, ix))
    if not in_win:
        return None
    return in_win[0][0]


def _has_real_peak(y: np.ndarray, master_x: np.ndarray, cm1: float,
                       half: float, top_rank_max: int = 12,
                       prom_frac: float = 0.05) -> bool:
    """Strict 'is there a real peak here' test.

    Requires: a `find_peaks` peak (prominence ≥ prom_frac × spectrum range)
    within ±half cm⁻¹ AND that peak ranks within top_rank_max of all
    spectrum peaks. This excludes baseline wobble and minor shoulders."""
    idx = _spectrum_peaks(y, master_x, prom_frac=prom_frac)
    if len(idx) == 0:
        return False
    heights = y[idx]
    order = np.argsort(-heights)
    ranked = idx[order][:max(top_rank_max, 5)]
    for ix in ranked:
        if abs(master_x[ix] - cm1) <= half:
            return True
    return False


# ──────────────────────────────────────────────────────────────────────
# Stage 1 — Ground-truth inventory
# ──────────────────────────────────────────────────────────────────────
def stage1_inventory(master_x):
    print("[STAGE 1] Building ground-truth inventory")
    inventory = []
    refs_by_mol: dict[str, list[dict]] = defaultdict(list)

    sources = []

    # Pure Raman / SERS grounding loaders
    for tag, regime, substrate, fn in [
        ("ramanbiolib",            "Raman", "n/a",                   load_ramanbiolib),
        ("gobbato_powder_raman",   "Raman", "n/a (powder)",          load_gobbato_powder),
        ("amino_acid_raman",       "Raman", "n/a",                   load_amino_acid_xlsx),
        ("digitised_literature",   "Raman", "n/a (digitised)",       load_digitised_literature),
        ("sers_metabolite_63",     "SERS",  "Au-on-Si plasmonic",    load_sers_metabolite_63),
        ("serum_ag_colloids_fitting",   "SERS", "Ag colloid",        load_sers_fitting),
        ("serum_ag_colloids_isotopic",  "SERS", "Ag colloid",        load_isotopic),
        ("serum_ag_colloids_uricase",   "SERS", "Ag colloid (cAg-like)", load_uricase),
        ("serum_ag_colloids_erg_cal",   "SERS", "Ag colloid",        load_erg_calibration),
    ]:
        try:
            refs = fn(master_x)
        except Exception as e:
            print(f"  loader {tag} failed: {e}")
            refs = []
        # Normalize each ref into a uniform schema
        for r in refs:
            raw_name = r.get("component_key")
            if raw_name is None:
                raw_name = r.get("cohort") or r.get("conc_label")
            mol = canonicalize(raw_name)
            entry = {
                "spectrum_id":   r.get("spectrum_id", ""),
                "dataset":       r.get("dataset", tag),
                "regime":        r.get("regime", regime),
                "substrate":     r.get("substrate_type") or r.get("substrate_family") or substrate,
                "raw_label":     raw_name,
                "molecule_canonical": mol,
                "spectrum":      r["spectrum"],
            }
            if mol is not None:
                refs_by_mol[mol].append(entry)
            sources.append(entry)
        print(f"  {tag}: {len(refs)} refs loaded ({sum(1 for r in refs if canonicalize(r.get('component_key') or r.get('cohort') or r.get('conc_label')) is not None)} mapped)")

    # Build inventory CSV (one row per molecule × dataset × regime)
    grouped = defaultdict(list)
    for s in sources:
        if s["molecule_canonical"] is None:
            continue
        key = (s["molecule_canonical"], s["dataset"], s["regime"], s["substrate"])
        grouped[key].append(s)

    for (mol, ds, regime, substrate), spectra in sorted(grouped.items()):
        inventory.append({
            "molecule_canonical": mol,
            "dataset":            ds,
            "regime":             regime,
            "substrate":          substrate,
            "preprocessing":      "canonical_preprocess (crop 400-1800, AsLS, SG, L2)",
            "n_spectra":          len(spectra),
            "wn_min":             float(master_x[0]),
            "wn_max":             float(master_x[-1]),
            "wn_step":            float(master_x[1] - master_x[0]),
            "raw_labels_seen":    "|".join(sorted({str(s["raw_label"]) for s in spectra})[:8]),
        })

    inv_df = pd.DataFrame(inventory).sort_values(["molecule_canonical", "regime", "dataset"])
    inv_df.to_csv(TABLES / "ground_truth_inventory_v1.csv", index=False)
    print(f"  inventory: {len(inv_df)} rows; molecules covered: {sorted(refs_by_mol.keys())}")
    return refs_by_mol, inv_df


# ──────────────────────────────────────────────────────────────────────
# Stage 2 — Band presence test
# ──────────────────────────────────────────────────────────────────────
def stage2_band_presence(refs_by_mol, master_x):
    print("[STAGE 2] Per-molecule paper-band presence test")
    rows = []
    for target, bands in PAPER_BANDS.items():
        for band in bands:
            for mol in COMPARATORS:
                refs = refs_by_mol.get(mol, [])
                if not refs:
                    rows.append({
                        "paper_target":  target,
                        "paper_band":    band,
                        "molecule":      mol,
                        "n_spectra":     0,
                        "regime_mix":    "",
                        "intensity_mean": np.nan,
                        "prom_mean":     np.nan,
                        "rank_median":   np.nan,
                        "presence_3":    np.nan,
                        "presence_5":    np.nan,
                        "presence_10":   np.nan,
                        "anchor_or_support": "ABSENT_NO_REF",
                    })
                    continue
                ints, proms, ranks = [], [], []
                pres3, pres5, pres10 = [], [], []
                regimes = []
                for s in refs:
                    y = s["spectrum"]; regimes.append(s["regime"])
                    pk_int, _ = _local_max(y, master_x, band, 10.0)
                    if pk_int is None:
                        continue
                    ints.append(pk_int)
                    proms.append(_local_prominence(y, master_x, band, 10.0))
                    rk = _peak_rank(y, master_x, band, half=5.0)
                    if rk is not None:
                        ranks.append(rk)
                    # presence: real find_peaks peak within tolerance, ranking in top-12
                    pres3.append(int(_has_real_peak(y, master_x, band, 3.0)))
                    pres5.append(int(_has_real_peak(y, master_x, band, 5.0)))
                    pres10.append(int(_has_real_peak(y, master_x, band, 10.0)))
                # classify
                p5 = float(np.mean(pres5)) if pres5 else 0.0
                rank_med = float(np.median(ranks)) if ranks else np.nan
                if p5 >= 0.5 and (not np.isnan(rank_med) and rank_med <= 5):
                    cls = "ANCHOR_LIKE"
                elif p5 >= 0.5:
                    cls = "PRESENT_BUT_WEAK_RANK"
                elif p5 >= 0.2:
                    cls = "OCCASIONAL_SUPPORT"
                else:
                    cls = "ABSENT_OR_SUBLIMINAL"
                rows.append({
                    "paper_target":  target,
                    "paper_band":    band,
                    "molecule":      mol,
                    "n_spectra":     len(refs),
                    "regime_mix":    "|".join(sorted(set(regimes))),
                    "intensity_mean": float(np.mean(ints)) if ints else np.nan,
                    "prom_mean":     float(np.mean(proms)) if proms else np.nan,
                    "rank_median":   rank_med,
                    "presence_3":    float(np.mean(pres3)) if pres3 else 0.0,
                    "presence_5":    p5,
                    "presence_10":   float(np.mean(pres10)) if pres10 else 0.0,
                    "anchor_or_support": cls,
                })
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "paper_band_presence_by_molecule_v1.csv", index=False)
    print(f"  emitted {len(df)} (target_band × molecule) rows")
    return df


# ──────────────────────────────────────────────────────────────────────
# Stage 3 — Specificity / collision test
# ──────────────────────────────────────────────────────────────────────
def stage3_specificity(presence_df: pd.DataFrame):
    print("[STAGE 3] Specificity / collision matrix")
    rows = []
    coll_rows = []
    for target, bands in PAPER_BANDS.items():
        for band in bands:
            sub = presence_df[(presence_df.paper_target == target) &
                                 (presence_df.paper_band == band)].copy()
            # restrict to molecules with spectra
            sub = sub[sub.n_spectra > 0]
            present = sub[sub.presence_5 >= 0.5]
            n_present = len(present)
            n_target_present = int(((present.molecule == target) & (present.presence_5 >= 0.5)).sum())
            specificity = (n_target_present / n_present) if n_present > 0 else 0.0
            target_anchor = bool(((sub.molecule == target) &
                                     (sub.anchor_or_support == "ANCHOR_LIKE")).any())
            colliders = sorted(present[present.molecule != target]["molecule"].tolist())
            if specificity >= 0.75 and target_anchor:
                flag = "HIGH_SPECIFICITY"
            elif specificity >= 0.5:
                flag = "MODERATE_SPECIFICITY"
            elif n_present == 0:
                flag = "WEAK_OR_ABSENT_EVERYWHERE"
            else:
                flag = "LOW_SPECIFICITY_COLLISION_PRONE"
            rows.append({
                "paper_target":      target,
                "paper_band":        band,
                "n_molecules_present": n_present,
                "target_present":      int(n_target_present > 0),
                "target_is_anchor":    int(target_anchor),
                "specificity":         round(specificity, 3),
                "colliders":           "|".join(colliders),
                "specificity_flag":    flag,
            })
            for mol in COMPARATORS:
                row = sub[sub.molecule == mol]
                p5 = float(row.presence_5.iloc[0]) if not row.empty else 0.0
                cls = row.anchor_or_support.iloc[0] if not row.empty else "ABSENT_NO_REF"
                coll_rows.append({
                    "paper_target": target, "paper_band": band,
                    "molecule": mol, "presence_5": p5,
                    "anchor_or_support": cls,
                })
    spec_df = pd.DataFrame(rows)
    coll_df = pd.DataFrame(coll_rows)
    spec_df.to_csv(TABLES / "paper_band_specificity_v1.csv", index=False)
    # Wide collision matrix: molecules x (target::band)
    coll_df["band_id"] = coll_df.paper_target + "::" + coll_df.paper_band.astype(str)
    wide = coll_df.pivot_table(index="molecule", columns="band_id",
                                  values="presence_5", aggfunc="mean").fillna(0.0)
    wide.to_csv(TABLES / "paper_band_collision_matrix_v1.csv")
    print(f"  emitted {len(spec_df)} band-level specificity rows")
    return spec_df, coll_df, wide


# ──────────────────────────────────────────────────────────────────────
# Stage 4 — Panel-level validation
# ──────────────────────────────────────────────────────────────────────
def stage4_panel(presence_df, refs_by_mol, master_x):
    print("[STAGE 4] Panel-level validation")
    rows, missing_rows = [], []
    for target, bands in PAPER_BANDS.items():
        sub = presence_df[(presence_df.paper_target == target) &
                             (presence_df.molecule == target)]
        n_present = int((sub.presence_5 >= 0.5).sum())
        coverage = n_present / len(bands) if bands else 0.0
        # Panel specificity = mean band-level specificity across the panel
        spec_vals = []
        for band in bands:
            band_sub = presence_df[(presence_df.paper_target == target) &
                                       (presence_df.paper_band == band) &
                                       (presence_df.n_spectra > 0)]
            present_set = band_sub[band_sub.presence_5 >= 0.5]
            n_p = len(present_set)
            n_t = int(((present_set.molecule == target)).sum())
            spec_vals.append((n_t / n_p) if n_p > 0 else 0.0)
        panel_specificity = float(np.mean(spec_vals)) if spec_vals else 0.0

        # Discover strong ground-truth peaks for this molecule and check
        # how many are NOT in the paper panel (allow ±10 cm⁻¹ tolerance)
        refs = refs_by_mol.get(target, [])
        strong_peaks = []
        if refs:
            ymean = np.mean([s["spectrum"] for s in refs], axis=0)
            idx = _spectrum_peaks(ymean, master_x, prom_frac=0.10)
            heights = ymean[idx]
            order = np.argsort(-heights)
            top = idx[order][:10]  # top-10 strong peaks in mean GT spectrum
            for ix in top:
                cm1 = float(master_x[ix])
                in_panel = any(abs(cm1 - b) <= 10 for b in bands)
                strong_peaks.append((cm1, float(ymean[ix]), in_panel))

        n_strong_in_panel  = sum(1 for c, _, ok in strong_peaks if ok)
        n_strong_off_panel = sum(1 for c, _, ok in strong_peaks if not ok)

        # Missing strong peaks (rows for separate CSV)
        for cm1, ht, ok in strong_peaks:
            if not ok:
                missing_rows.append({
                    "molecule": target,
                    "ground_truth_peak_cm1": round(cm1, 1),
                    "ground_truth_height":   round(ht, 4),
                    "in_paper_panel":        False,
                })

        # Weak-or-absent paper bands per target
        weak = sub[sub.presence_5 < 0.5]
        rows.append({
            "molecule": target,
            "paper_panel_n":            len(bands),
            "paper_panel_present":      n_present,
            "paper_panel_coverage":     round(coverage, 3),
            "panel_specificity_mean":   round(panel_specificity, 3),
            "n_strong_gt_peaks":        len(strong_peaks),
            "n_strong_in_panel":        n_strong_in_panel,
            "n_strong_off_panel":       n_strong_off_panel,
            "weak_or_absent_paper_bands": "|".join(str(int(b)) for b in weak.paper_band.tolist()),
        })

    panel_df = pd.DataFrame(rows)
    panel_df.to_csv(TABLES / "paper_panel_validation_v1.csv", index=False)
    miss_df = pd.DataFrame(missing_rows)
    miss_df.to_csv(TABLES / "ground_truth_peaks_missing_from_paper_panel_v1.csv", index=False)
    print(f"  emitted panel validation for {len(panel_df)} targets, "
            f"{len(miss_df)} off-panel strong peaks")
    return panel_df, miss_df


# ──────────────────────────────────────────────────────────────────────
# Stage 5 — Paper panel vs MSS template comparison
# ──────────────────────────────────────────────────────────────────────
def _split_floats(s):
    if pd.isna(s) or not str(s).strip():
        return []
    out = []
    for tok in str(s).split(";"):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(float(tok))
        except ValueError:
            pass
    return out


def stage5_mss_compare():
    print("[STAGE 5] Paper panel vs MSS template comparison")
    if not MSS_REGISTRY.exists():
        print(f"  MSS registry not found at {MSS_REGISTRY} — emitting empty stage5")
        df = pd.DataFrame()
        df.to_csv(TABLES / "paper_panel_vs_mss_template_v1.csv", index=False)
        return df

    mss = pd.read_csv(MSS_REGISTRY)
    mss["analyte_lower"] = mss["analyte_name"].str.lower().str.strip()

    rows = []
    for target, bands in PAPER_BANDS.items():
        # find MSS analyte rows whose canonical name matches target
        candidates = []
        for _, r in mss.iterrows():
            if canonicalize(r["analyte_lower"]) == target:
                candidates.append(r)
        if not candidates:
            rows.append({
                "molecule": target,
                "mss_template_present": False,
                "mss_signatures_matched": 0,
                "mss_anchor_bands": "",
                "mss_companion_bands": "",
                "mss_anti_evidence_bands": "",
                "paper_bands": "|".join(str(int(b)) for b in bands),
                "paper_overlap_anchor": 0,
                "paper_overlap_companion": 0,
                "paper_not_used_by_mss": "|".join(str(int(b)) for b in bands),
                "mss_required_absent_from_paper": "",
                "classification": "MSS_TEMPLATE_MISSING_OR_INCOMPLETE",
            })
            continue
        # union of all matching MSS signatures (target may be split into variants)
        anchors, supports, anti = [], [], []
        for r in candidates:
            anchors  += _split_floats(r.get("mandatory_anchors_cm1"))
            supports += _split_floats(r.get("optional_support_cm1"))
            anti     += _split_floats(r.get("anti_evidence_cm1"))
        anchors  = sorted(set(round(x, 1) for x in anchors))
        supports = sorted(set(round(x, 1) for x in supports))
        anti     = sorted(set(round(x, 1) for x in anti))

        def overlap(p, ref, tol=10.0):
            return [b for b in p if any(abs(b - x) <= tol for x in ref)]
        in_anchor    = overlap(bands, anchors)
        in_companion = overlap([b for b in bands if b not in in_anchor], supports)
        not_used     = [b for b in bands if b not in in_anchor and b not in in_companion]
        # MSS anchors absent from paper panel (paper does not invoke them)
        absent_anchors = [a for a in anchors if not any(abs(a - b) <= 10.0 for b in bands)]

        if len(in_anchor) >= max(1, len(bands) // 2):
            cls = "PAPER_PANEL_MATCHES_FULL_MSS" if len(not_used) == 0 \
                    else "PAPER_PANEL_PARTIAL_MSS"
        elif len(in_anchor) + len(in_companion) >= 1:
            cls = "PAPER_PANEL_PARTIAL_MSS"
        else:
            cls = "PAPER_PANEL_WEAK_OR_COLLISION_PRONE"

        rows.append({
            "molecule": target,
            "mss_template_present": True,
            "mss_signatures_matched": len(candidates),
            "mss_anchor_bands":   "|".join(str(int(x)) for x in anchors),
            "mss_companion_bands": "|".join(str(int(x)) for x in supports),
            "mss_anti_evidence_bands": "|".join(str(int(x)) for x in anti),
            "paper_bands": "|".join(str(int(b)) for b in bands),
            "paper_overlap_anchor":    len(in_anchor),
            "paper_overlap_companion": len(in_companion),
            "paper_not_used_by_mss":   "|".join(str(int(b)) for b in not_used),
            "mss_required_absent_from_paper": "|".join(str(int(x)) for x in absent_anchors),
            "classification": cls,
        })
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "paper_panel_vs_mss_template_v1.csv", index=False)
    print(f"  MSS comparison: {len(df)} target rows")
    return df


# ──────────────────────────────────────────────────────────────────────
# Stage 6 — Figures
# ──────────────────────────────────────────────────────────────────────
def stage6_figures(refs_by_mol, presence_df, spec_df, panel_df, mss_df, master_x):
    print("[STAGE 6] Figures")

    # Fig 1: ground-truth spectra of each target with paper bands marked
    try:
        fig, axes = plt.subplots(len(PAPER_TARGETS), 1, figsize=(10, 11), sharex=True)
        for ax, target in zip(axes, PAPER_TARGETS):
            refs = refs_by_mol.get(target, [])
            ax.set_title(f"{target}  (n_ref={len(refs)})", fontsize=10)
            if refs:
                # Plot at most 3 spectra (1 per regime if possible)
                shown = []
                for s in refs:
                    if len(shown) >= 3:
                        break
                    if all(s2["regime"] != s["regime"] for s2 in shown) or len(shown) == 0:
                        shown.append(s)
                # Fall back: if still <3, just take first few
                if len(shown) < 2:
                    shown = refs[:3]
                for s in shown:
                    y = s["spectrum"]
                    ax.plot(master_x, y / max(y.max(), 1e-9),
                              label=f"{s['dataset']} ({s['regime']})", alpha=0.75, lw=0.9)
                ax.legend(fontsize=7, loc="upper right")
            for b in PAPER_BANDS[target]:
                ax.axvline(b, color="red", ls="--", alpha=0.5, lw=0.8)
                ax.text(b, 1.02, f"{int(b)}", color="red",
                          ha="center", va="bottom", fontsize=7, rotation=90)
            ax.set_ylim(-0.05, 1.15)
            ax.set_ylabel("intensity (norm)", fontsize=8)
        axes[-1].set_xlabel("wavenumber cm⁻¹")
        fig.suptitle("Ground-truth spectra with paper bands (red dashed)", y=0.995)
        fig.tight_layout()
        fig.savefig(FIGS / "fig_ground_truth_spectra_with_paper_bands_v1.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig1 issue: {e}")

    # Fig 2: band overlap heatmap (rows=molecules, cols=paper bands)
    try:
        wide = presence_df.pivot_table(
            index="molecule", values="prom_mean",
            columns=presence_df.paper_target + "::" + presence_df.paper_band.astype(int).astype(str),
            aggfunc="mean").fillna(0.0)
        # reorder rows to put PAPER_TARGETS first
        ordered_rows = PAPER_TARGETS + [m for m in COMPARATORS if m not in PAPER_TARGETS]
        wide = wide.reindex(ordered_rows).fillna(0.0)
        fig, ax = plt.subplots(figsize=(11, 6))
        im = ax.imshow(wide.values, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(wide.columns))); ax.set_xticklabels(wide.columns, rotation=70, fontsize=7)
        ax.set_yticks(range(len(wide.index)));   ax.set_yticklabels(wide.index, fontsize=8)
        ax.set_title("Paper-band local prominence — molecules × bands")
        fig.colorbar(im, ax=ax, fraction=0.04, label="local prominence (mean)")
        fig.tight_layout()
        fig.savefig(FIGS / "fig_paper_band_presence_heatmap_v1.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig2 issue: {e}")

    # Fig 3: specificity heatmap (rows=paper bands, cols=molecules) showing presence_5
    try:
        cdf = presence_df.copy()
        cdf["band_id"] = cdf.paper_target + "::" + cdf.paper_band.astype(int).astype(str)
        wide = cdf.pivot_table(index="band_id", columns="molecule",
                                  values="presence_5", aggfunc="mean").fillna(0.0)
        ordered_cols = PAPER_TARGETS + [m for m in COMPARATORS if m not in PAPER_TARGETS]
        wide = wide.reindex(columns=[c for c in ordered_cols if c in wide.columns])
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(wide.values, aspect="auto", cmap="magma", vmin=0, vmax=1)
        ax.set_xticks(range(len(wide.columns))); ax.set_xticklabels(wide.columns, rotation=70, fontsize=8)
        ax.set_yticks(range(len(wide.index)));   ax.set_yticklabels(wide.index, fontsize=7)
        ax.set_title("Paper bands × molecules — presence_5 (collision pattern)")
        fig.colorbar(im, ax=ax, fraction=0.04, label="presence (≥60th-pctile fire rate)")
        fig.tight_layout()
        fig.savefig(FIGS / "fig_paper_band_specificity_heatmap_v1.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig3 issue: {e}")

    # Fig 4: panel coverage + specificity bar
    try:
        fig, ax1 = plt.subplots(figsize=(8, 5))
        x = np.arange(len(panel_df)); w = 0.4
        ax1.bar(x - w/2, panel_df.paper_panel_coverage, w,
                  label="panel coverage (frac present)", color="#4C72B0")
        ax1.bar(x + w/2, panel_df.panel_specificity_mean, w,
                  label="panel specificity (mean per band)", color="#DD8452")
        ax1.set_xticks(x); ax1.set_xticklabels(panel_df.molecule, rotation=15)
        ax1.set_ylim(0, 1.05); ax1.set_ylabel("fraction")
        ax1.set_title("Paper panel — coverage vs specificity (per target)")
        ax1.legend(loc="upper right", fontsize=8); ax1.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIGS / "fig_panel_coverage_specificity_v1.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig4 issue: {e}")

    # Fig 5: paper panel vs MSS schematic
    try:
        if mss_df is not None and not mss_df.empty:
            fig, axes = plt.subplots(len(mss_df), 1, figsize=(11, 1.6 * len(mss_df)),
                                          sharex=True)
            if len(mss_df) == 1: axes = [axes]
            for ax, (_, row) in zip(axes, mss_df.iterrows()):
                ax.set_title(f"{row['molecule']}  —  {row['classification']}", fontsize=9)
                # paper bands (red)
                for b in str(row["paper_bands"]).split("|"):
                    if b: ax.axvline(float(b), color="red", lw=2, alpha=0.7)
                # MSS anchors (blue)
                for b in str(row["mss_anchor_bands"]).split("|"):
                    if b: ax.axvline(float(b), color="blue", lw=2, alpha=0.5)
                # MSS companions (cyan thin)
                for b in str(row["mss_companion_bands"]).split("|"):
                    if b: ax.axvline(float(b), color="cyan", lw=1, alpha=0.5)
                # MSS anti (gray dotted)
                for b in str(row["mss_anti_evidence_bands"]).split("|"):
                    if b: ax.axvline(float(b), color="gray", ls=":", lw=1, alpha=0.5)
                ax.set_xlim(400, 1800); ax.set_yticks([])
            axes[-1].set_xlabel("wavenumber cm⁻¹")
            fig.suptitle("Paper bands (red) vs MSS anchors (blue) / companions (cyan) / anti (gray)",
                            y=1.001, fontsize=10)
            fig.tight_layout()
            fig.savefig(FIGS / "fig_paper_panel_vs_mss_template_v1.png", dpi=150)
            plt.close(fig)
    except Exception as e:
        print(f"  fig5 issue: {e}")


# ──────────────────────────────────────────────────────────────────────
# Stage 7 — Report + final decision
# ──────────────────────────────────────────────────────────────────────
def stage7_report(inv_df, presence_df, spec_df, panel_df, miss_df, mss_df) -> str:
    print("[STAGE 7] Building report + final decision")

    # Decision logic
    n_targets = len(PAPER_TARGETS)
    n_high_spec_bands  = int((spec_df.specificity_flag == "HIGH_SPECIFICITY").sum())
    n_mod_spec_bands   = int((spec_df.specificity_flag == "MODERATE_SPECIFICITY").sum())
    n_low_spec_bands   = int((spec_df.specificity_flag == "LOW_SPECIFICITY_COLLISION_PRONE").sum())
    n_weak_bands       = int((spec_df.specificity_flag == "WEAK_OR_ABSENT_EVERYWHERE").sum())
    n_total_bands      = len(spec_df)

    # Per-target quick summary
    well_supported = []
    partial = []
    weak = []
    for _, r in panel_df.iterrows():
        target = r["molecule"]
        cov = r["paper_panel_coverage"]
        psp = r["panel_specificity_mean"]
        if cov >= 0.5 and psp >= 0.5:
            well_supported.append(target)
        elif cov >= 0.5 or psp >= 0.4:
            partial.append(target)
        else:
            weak.append(target)

    # Insufficient ground truth check
    targets_with_refs = set(inv_df["molecule_canonical"].unique()) & set(PAPER_TARGETS)
    if len(targets_with_refs) < 3:
        decision = "INSUFFICIENT_GROUND_TRUTH"
    elif n_high_spec_bands >= max(2, n_total_bands // 3) and len(well_supported) >= 3:
        decision = "PAPER_BANDS_STRONGLY_SUPPORTED_AND_SPECIFIC"
    elif (n_high_spec_bands + n_mod_spec_bands) >= n_total_bands // 2 \
            and (len(well_supported) + len(partial)) >= 3:
        decision = "PAPER_BANDS_SUPPORTED_BUT_PARTIAL"
    elif n_low_spec_bands >= n_total_bands // 2:
        decision = "PAPER_BANDS_COLLISION_PRONE"
    else:
        decision = "PAPER_BANDS_WEAKLY_SUPPORTED"

    lines = []
    lines.append("# GAIRA paper-band vs ground-truth validation v1 — final report\n")
    lines.append(f"## Decision: **{decision}**\n")
    lines.append("Pure-molecule ground-truth corpus assembled from RamanBioLib + Gobbato powder + "
                    "amino-acid xlsx + digitised literature + NIHMS1547448 SERS metabolites + "
                    "serum_ag_colloids pure-component cohorts (Hypox/UAfree/UAbound, isotopic, uricase, ERG cal). "
                    "Engine, MSS, motif, taxonomy, substrate physics — UNCHANGED.\n")

    # Inventory summary
    inv_summary = (inv_df.groupby("molecule_canonical")
                            .agg(n=("n_spectra", "sum"),
                                  datasets=("dataset", lambda s: "|".join(sorted(set(s))[:5])))
                            .reset_index())
    lines.append("## Stage 1 — Ground-truth inventory (per molecule)\n")
    lines.append("| molecule | n_spectra | datasets |")
    lines.append("|---|---:|---|")
    for _, r in inv_summary.iterrows():
        if r["molecule_canonical"] in COMPARATORS or r["molecule_canonical"] in PAPER_TARGETS:
            lines.append(f"| {r['molecule_canonical']} | {int(r['n'])} | {r['datasets']} |")
    lines.append("")

    # Stage 2 / 3 — band-level table per target
    lines.append("## Stage 2 + 3 — Band-level presence and specificity\n")
    for target in PAPER_TARGETS:
        lines.append(f"### {target}\n")
        lines.append("| paper band | target presence_5 | target rank_med | specificity | flag | top colliders |")
        lines.append("|---:|---:|---:|---:|---|---|")
        for band in PAPER_BANDS[target]:
            tp = presence_df[(presence_df.paper_target == target) &
                                  (presence_df.paper_band == band) &
                                  (presence_df.molecule == target)]
            sp = spec_df[(spec_df.paper_target == target) &
                              (spec_df.paper_band == band)]
            tp_p5  = tp.presence_5.iloc[0] if not tp.empty else 0.0
            tp_rk  = tp.rank_median.iloc[0] if not tp.empty else np.nan
            spec   = sp.specificity.iloc[0] if not sp.empty else 0.0
            flag   = sp.specificity_flag.iloc[0] if not sp.empty else ""
            colls  = sp.colliders.iloc[0] if not sp.empty else ""
            lines.append(f"| {int(band)} | {tp_p5:.2f} | {tp_rk if not np.isnan(tp_rk) else 'NA'} | "
                            f"{spec:.2f} | {flag} | {colls[:80]} |")
        lines.append("")

    # Stage 4 — panel summary
    lines.append("## Stage 4 — Panel-level validation\n")
    lines.append("| molecule | panel n | present | coverage | mean specificity | strong GT peaks (in/off) | weak/absent paper bands |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for _, r in panel_df.iterrows():
        lines.append(f"| {r['molecule']} | {r['paper_panel_n']} | {r['paper_panel_present']} | "
                        f"{r['paper_panel_coverage']:.2f} | {r['panel_specificity_mean']:.2f} | "
                        f"{r['n_strong_in_panel']}/{r['n_strong_off_panel']} | "
                        f"{r['weak_or_absent_paper_bands']} |")
    lines.append("")

    if not miss_df.empty:
        lines.append("### Strong ground-truth peaks NOT included in paper panel\n")
        lines.append("| molecule | GT peak cm⁻¹ | GT height (norm) |")
        lines.append("|---|---:|---:|")
        for _, r in miss_df.sort_values(["molecule", "ground_truth_height"], ascending=[True, False]).iterrows():
            lines.append(f"| {r['molecule']} | {r['ground_truth_peak_cm1']:.0f} | {r['ground_truth_height']:.3f} |")
        lines.append("")

    # Stage 5 — MSS comparison
    if mss_df is not None and not mss_df.empty:
        lines.append("## Stage 5 — Paper panel vs MSS template\n")
        lines.append("| molecule | MSS sigs | paper-bands | overlap anchor / companion | MSS-required absent from paper | classification |")
        lines.append("|---|---:|---|---:|---|---|")
        for _, r in mss_df.iterrows():
            lines.append(f"| {r['molecule']} | {r['mss_signatures_matched']} | "
                            f"{r['paper_bands']} | {r['paper_overlap_anchor']} / {r['paper_overlap_companion']} | "
                            f"{r['mss_required_absent_from_paper']} | {r['classification']} |")
        lines.append("")

    # Required answers
    lines.append("## Required answers\n")
    lines.append("### 1. Do paper bands match ground-truth spectra for UA / HX / ERG / GSH?\n")
    for target in PAPER_TARGETS:
        sub = presence_df[(presence_df.paper_target == target) &
                              (presence_df.molecule == target)]
        n_present = int((sub.presence_5 >= 0.5).sum())
        lines.append(f"- **{target}**: {n_present} / {len(PAPER_BANDS[target])} paper bands present in target ground truth")
    lines.append("")

    lines.append("### 2. Which assignments are strongly supported?\n")
    strong_bands = spec_df[(spec_df.specificity_flag == "HIGH_SPECIFICITY")].copy()
    if not strong_bands.empty:
        for _, r in strong_bands.iterrows():
            lines.append(f"- {r['paper_target']} @ {int(r['paper_band'])} — specificity {r['specificity']:.2f}")
    else:
        lines.append("- (no band reached HIGH_SPECIFICITY threshold)")
    lines.append("")

    lines.append("### 3. Which assignments are weak or collision-prone?\n")
    weak_bands = spec_df[spec_df.specificity_flag.isin(
        ["LOW_SPECIFICITY_COLLISION_PRONE", "WEAK_OR_ABSENT_EVERYWHERE"])]
    for _, r in weak_bands.iterrows():
        lines.append(f"- {r['paper_target']} @ {int(r['paper_band'])} — {r['specificity_flag']}; "
                        f"colliders: {r['colliders'][:120]}")
    if weak_bands.empty:
        lines.append("- (none — all bands at least MODERATE_SPECIFICITY)")
    lines.append("")

    lines.append("### 4. Are the paper bands sufficient for molecule-specific evidence?\n")
    lines.append(f"- well-supported targets: {well_supported or '(none)'}")
    lines.append(f"- partial-support targets: {partial or '(none)'}")
    lines.append(f"- weak-support targets: {weak or '(none)'}")
    lines.append("")

    lines.append("### 5. Which important ground-truth peaks are missing from the paper panels?\n")
    if not miss_df.empty:
        for tgt in PAPER_TARGETS:
            miss_t = miss_df[miss_df.molecule == tgt].sort_values("ground_truth_height", ascending=False)
            if not miss_t.empty:
                top = ", ".join(f"{int(r['ground_truth_peak_cm1'])} ({r['ground_truth_height']:.2f})"
                                  for _, r in miss_t.head(5).iterrows())
                lines.append(f"- **{tgt}**: {top}")
    else:
        lines.append("- (no off-panel strong GT peaks above prominence threshold)")
    lines.append("")

    lines.append("### 6. Why might paper-feature scoring succeed while full MSS scoring fails?\n")
    lines.append(
        "Paper-feature scoring tests **height + local prominence at fixed paper-band positions** — "
        "a narrow, additive readout calibrated to the bands the paper claims. MSS scoring requires "
        "**anchor co-firing + competitor / anti-evidence + amplitude-aware gating**, which is a stricter "
        "decision rule. When paper bands are partially specific (e.g. the band fires for the target but "
        "also for related comparators), the additive paper score still tracks the target on average, but "
        "the MSS decision can be vetoed by competitor/anti-evidence patterns. The two tests therefore "
        "answer different questions: paper-feature scoring asks 'does the literature-claimed band move '"
        "in the expected direction?', while MSS asks 'is this band's firing consistent with a clean "
        "molecule-specific assignment?'\n"
    )

    lines.append("### 7. Implications for GAIRA's narrow metabolite layer\n")
    if decision == "PAPER_BANDS_STRONGLY_SUPPORTED_AND_SPECIFIC":
        lines.append(
            "- Paper bands hold up against pure-molecule ground truth. The narrow paper-band panel can "
            "be promoted to a literature-claim verification layer alongside MSS for paper replication studies.\n"
            "- MSS templates that miss paper-claimed anchors should add them as companions where appropriate."
        )
    elif decision == "PAPER_BANDS_SUPPORTED_BUT_PARTIAL":
        lines.append(
            "- Paper bands are partially supported: targets with high coverage and acceptable specificity "
            "can be used as literature-claim verification; targets with low coverage or low specificity "
            "should NOT be reported as molecule-specific without MSS-style co-fire constraints.\n"
            "- Highest-leverage GAIRA next steps: (a) ingest the off-panel strong peaks listed in Stage 4 "
            "as MSS companions, (b) keep the paper-band panel as PAPER_LITERATURE tier, separate from MSS_DECISION tier."
        )
    elif decision == "PAPER_BANDS_COLLISION_PRONE":
        lines.append(
            "- Paper bands fire across multiple comparators — using them alone for molecule identity overclaims. "
            "Paper-band scoring stays valid as a panel-mean directional readout, but per-band identity claims "
            "should be marked AMBIGUOUS unless co-fire / anti-evidence constraints are added."
        )
    elif decision == "PAPER_BANDS_WEAKLY_SUPPORTED":
        lines.append(
            "- Paper bands do not consistently fire in pure-molecule ground truth. The Pilot 1 paper-claim "
            "replication earlier likely succeeded because of correlated mixture chemistry (panel-mean tracking "
            "covarying biology), not because of molecule-specific evidence."
        )
    else:
        lines.append(
            "- Insufficient pure-molecule ground truth for one or more target molecules (e.g. lactate has no "
            "MSS template; some targets have only 1-2 reference spectra). Expand grounding corpus before "
            "drawing strong conclusions."
        )

    out = "\n".join(lines)
    (REPORTS / "REPORT_paper_band_vs_ground_truth_validation_v1.md").write_text(out)
    print(f"  decision: {decision}")
    return decision


# ──────────────────────────────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 78)
    print("gaira_base_4_paper_band_vs_ground_truth_validation_v1")
    print("=" * 78)
    master_x = canonical_master_axis()

    refs_by_mol, inv_df = stage1_inventory(master_x)
    presence_df         = stage2_band_presence(refs_by_mol, master_x)
    spec_df, coll_df, _ = stage3_specificity(presence_df)
    panel_df, miss_df   = stage4_panel(presence_df, refs_by_mol, master_x)
    mss_df              = stage5_mss_compare()
    stage6_figures(refs_by_mol, presence_df, spec_df, panel_df, mss_df, master_x)
    decision = stage7_report(inv_df, presence_df, spec_df, panel_df, miss_df, mss_df)

    # Code snapshot
    try:
        shutil.copy(__file__, CODE_SNAPSHOT / Path(__file__).name)
    except Exception:
        pass

    # Audit log
    audit = [
        "# gaira_base_4_paper_band_vs_ground_truth_validation_v1 — audit log",
        f"date: {datetime.now().isoformat()}",
        "",
        "## Inputs (pure-molecule ground truth)",
        "- ramanbiolib (Raman pure-component)",
        "- gobbato_powder_raman (Raman pure-component, multiple replicates)",
        "- amino_acid_raman (Raman pure-component)",
        "- digitised_literature (Raman digitised)",
        "- sers_metabolite_63 (SERS pure-component, NIHMS1547448)",
        "- serum_ag_colloids fitting / isotopic / uricase / ERG calibration cohorts (SERS pure-component or controlled spike)",
        "",
        "## Paper bands (verbatim)",
        f"- uric_acid: {PAPER_BANDS['uric_acid']}",
        f"- hypoxanthine: {PAPER_BANDS['hypoxanthine']}",
        f"- ergothioneine: {PAPER_BANDS['ergothioneine']}",
        f"- glutathione: {PAPER_BANDS['glutathione']}",
        "",
        "## Comparators",
        ", ".join(COMPARATORS),
        "",
        "## Outputs",
        "- tables/ground_truth_inventory_v1.csv",
        "- tables/paper_band_presence_by_molecule_v1.csv",
        "- tables/paper_band_specificity_v1.csv",
        "- tables/paper_band_collision_matrix_v1.csv",
        "- tables/paper_panel_validation_v1.csv",
        "- tables/ground_truth_peaks_missing_from_paper_panel_v1.csv",
        "- tables/paper_panel_vs_mss_template_v1.csv",
        "- figures/fig_ground_truth_spectra_with_paper_bands_v1.png",
        "- figures/fig_paper_band_presence_heatmap_v1.png",
        "- figures/fig_paper_band_specificity_heatmap_v1.png",
        "- figures/fig_panel_coverage_specificity_v1.png",
        "- figures/fig_paper_panel_vs_mss_template_v1.png",
        "- reports/REPORT_paper_band_vs_ground_truth_validation_v1.md",
        "",
        "## Invariants preserved",
        "- Engine v4.5: unchanged",
        "- MSS v4.3 / motif / taxonomy / substrate physics: read-only",
        "- No threshold tuning, no classifier, no label-driven optimization",
        "- Paper bands taken VERBATIM at fixed positions",
        "",
        f"## Final decision\n**{decision}**",
    ]
    (AUDIT / "gaira_base_4_paper_band_vs_ground_truth_validation_v1_audit_log.md").write_text("\n".join(audit))
    print("[done]")


if __name__ == "__main__":
    main()
