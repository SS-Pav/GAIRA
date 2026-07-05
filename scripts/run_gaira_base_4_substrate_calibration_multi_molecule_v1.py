"""gaira_base_4_substrate_calibration_multi_molecule_v1

Phase: extend the substrate-aware POST-HOC calibration wrapper from adenine
to additional narrow targets (UA, HX, ERG, GSH) using calibration / pure
datasets ONLY.

STRICT INVARIANTS (NEVER violated):
- Engine v4.5: unchanged
- MSS scoring kernel: unchanged (anchor-fires + 0.3 × support-fires)
- Motif registry: unchanged
- MSS templates: unchanged (use repaired narrow registry v1)
- 11-axis BSV: unchanged
- Preprocessing: unchanged

NO soft-MSS, NO global threshold changes, NO retraining, NO classifier-first,
NO feedback into GAIRA, NO disease labels (these calibration datasets carry
known PURE-MOLECULE labels only).

Run:
    cd /Users/suraj/projects/GAIRA
    PYTHONPATH=src .venv/bin/python scripts/run_gaira_base_4_substrate_calibration_multi_molecule_v1.py
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

warnings.simplefilter("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gaira.spectral import canonical_master_axis  # noqa: E402

from run_gaira_base_4_mss_resolution_reporting_layer_v1 import (  # noqa: E402
    baseline_correct, mss_anchor_score, has_real_peak, load_templates,
)
from run_gaira_base_4_paper_band_vs_ground_truth_validation_v1 import (  # noqa: E402
    canonicalize,
)
import run_gaira_base_4_paper_band_vs_ground_truth_validation_v1 as _pbv  # noqa: E402

# Extend canonicalization for the optional small-molecule targets
_pbv.NAME_MAP.update({
    "glucose": "glucose", "d-(+)-glucose": "glucose", "gluc": "glucose",
    "urea": "urea", "creatinine": "creatinine", "creat": "creatinine",
})

from run_gaira_validate_2_grounding import (  # noqa: E402
    load_ramanbiolib, load_gobbato_powder,
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
ROOT = Path("/Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_4_substrate_calibration_multi_molecule_v1")
TABLES  = ROOT / "tables"
FIGS    = ROOT / "figures"
REPORTS = ROOT / "reports"
AUDIT   = ROOT / "audit"
CODE_SNAPSHOT = ROOT / "code_snapshot"
for d in (TABLES, FIGS, REPORTS, AUDIT, CODE_SNAPSHOT):
    d.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────
# Targets + interference set
# ──────────────────────────────────────────────────────────────────────
TARGETS = ["uric_acid", "hypoxanthine", "ergothioneine", "glutathione"]
COMPETITORS = {
    "uric_acid":     ["hypoxanthine", "xanthine", "adenine", "guanine", "ergothioneine"],
    "hypoxanthine":  ["uric_acid",   "xanthine", "adenine", "guanine"],
    "ergothioneine": ["cysteine", "cystine", "glutathione",
                          "tryptophan", "tyrosine", "phenylalanine"],
    "glutathione":   ["cysteine", "cystine", "ergothioneine"],
}
ALL_SCORE_TARGETS = sorted({m for ts in [[t] + COMPETITORS[t] for t in TARGETS] for m in ts}
                                | set(TARGETS))

# Per-molecule core/discriminative bands (from paper-band-vs-ground-truth findings + repaired registry)
# Pick anchors that are most discriminative per target. Use the SERS-mode anchors
# from the repaired registry where available, plus the prior phase's HIGH_SPECIFICITY bands.
CORE_BANDS = {
    "uric_acid":     {"primary_window":  (635, 645),    # 640 anchor
                          "secondary_window": (1130, 1145)},  # 1138 anchor (companion)
    "hypoxanthine":  {"primary_window":  (720, 735),    # 720-735 ring
                          "secondary_window": (1090, 1110)},  # 1099 anchor
    "ergothioneine": {"primary_window":  (1215, 1230),  # 1220 — HIGH_SPECIFICITY
                          "secondary_window": (485, 500)},     # 491 SERS anchor
    "glutathione":   {"primary_window":  (905, 920),    # 912 — HIGH_SPECIFICITY
                          "secondary_window": (655, 665)},     # 657 SERS anchor
}


# ──────────────────────────────────────────────────────────────────────
# Helpers — same scoring kernel + ring features as adenine wrapper
# ──────────────────────────────────────────────────────────────────────
def _ring_features(y, master_x, lo, hi):
    mask = (master_x >= lo) & (master_x <= hi)
    if not mask.any() or not np.isfinite(y).any():
        return (np.nan, 0.0, 0.0)
    win = y[mask]
    j = int(np.nanargmax(win))
    pk_pos = float(master_x[mask][j])
    area = float(np.trapezoid(np.clip(win, 0, None), master_x[mask]))
    idx_lo = int(np.where(mask)[0][0])
    idx_hi = int(np.where(mask)[0][-1])
    bg_left  = float(np.percentile(y[max(idx_lo-25, 0):idx_lo], 30)) if idx_lo > 5 else 0.0
    bg_right = float(np.percentile(y[idx_hi+1:min(idx_hi+1+25, len(y))], 30)) if idx_hi+5 < len(y) else 0.0
    prom = max(float(np.nanmax(win)) - (bg_left + bg_right) / 2.0, 0.0)
    return (pk_pos, area, prom)


def _spearman(x, y):
    x = pd.Series(x); y = pd.Series(y)
    valid = x.notna() & y.notna()
    if valid.sum() < 3: return np.nan
    rx = x[valid].rank(); ry = y[valid].rank()
    if rx.std() == 0 or ry.std() == 0: return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def _cohens_d(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 2 or len(y) < 2: return np.nan
    pooled = np.sqrt(((len(x)-1)*np.var(x, ddof=1) + (len(y)-1)*np.var(y, ddof=1))
                       / max(len(x)+len(y)-2, 1))
    return float((np.mean(x) - np.mean(y)) / (pooled if pooled > 0 else 1.0))


# ──────────────────────────────────────────────────────────────────────
# Dataset loading
# ──────────────────────────────────────────────────────────────────────
def gather_calibration_refs(master_x):
    refs = []
    for tag, regime, substrate, ground_truth_field, fn in [
        ("sers_metab_63",   "SERS", "Au-on-Si", "component_key", load_sers_metabolite_63),
        ("sers_fitting",    "SERS", "Ag colloid", "cohort",       load_sers_fitting),
        ("uricase",         "SERS", "Ag colloid (cAg-like)",
                                                 "cohort",       load_uricase),
        ("isotopic",        "SERS", "Ag colloid", "cohort",       load_isotopic),
        ("erg_calibration", "SERS", "Ag colloid", "cohort",       load_erg_calibration),
        ("gobbato_powder",  "Raman", "n/a (powder)", "component_key", load_gobbato_powder),
        ("ramanbiolib",     "Raman", "n/a", "component_key",          load_ramanbiolib),
    ]:
        try:
            r = fn(master_x)
        except Exception as e:
            print(f"  loader {tag} failed: {e}")
            r = []
        for ent in r:
            ent_meta = {
                "spectrum_id": ent.get("spectrum_id", ""),
                "dataset": ent.get("dataset", tag),
                "regime":  ent.get("regime", regime),
                "substrate": ent.get("substrate_type") or
                                 ent.get("substrate_family") or substrate,
                "raw_label": ent.get(ground_truth_field) or
                                ent.get("component_key") or ent.get("cohort"),
                "molecule_truth": canonicalize(
                    ent.get(ground_truth_field) or ent.get("component_key") or ent.get("cohort")),
                "spectrum": ent["spectrum"],
                # Extras from cohort-rich loaders
                "conc_label": ent.get("conc_label"),
                "conc": ent.get("concentration") or
                          ent.get("conc") or ent.get("conc_label"),
                "rep_id": ent.get("rep_id"),
                "calibration_type": ent.get("calibration_type"),
                "control_cohort":   ent.get("control_cohort"),
            }
            refs.append(ent_meta)
        print(f"  {tag}: {len(r)} refs loaded")
    return refs


def score_per_spectrum(refs, master_x, templates_by_mol):
    print("[score] per-spectrum × per-molecule scoring")
    n = len(refs)
    score_mat = {m: np.zeros(n) for m in ALL_SCORE_TARGETS}
    for i, ref in enumerate(refs):
        if i % 100 == 0: print(f"  {i}/{n}")
        y_pp = baseline_correct(ref["spectrum"]) if not np.isfinite(ref["spectrum"]).all() \
                  else ref["spectrum"]
        if not np.isfinite(y_pp).any() or float(np.linalg.norm(y_pp)) < 1e-12:
            continue
        ref["spectrum_pp"] = y_pp
        for mol in ALL_SCORE_TARGETS:
            if mol not in templates_by_mol: continue
            tps = templates_by_mol[mol]
            t = tps.get(ref["regime"]) or tps.get("SERS") or tps.get("Raman") or \
                  next(iter(tps.values()))
            sc, _, _ = mss_anchor_score(y_pp, master_x, t["anchors"], t["supports"])
            score_mat[mol][i] = sc
    return score_mat


# ──────────────────────────────────────────────────────────────────────
# STEP 1 — Baseline MSS performance
# ──────────────────────────────────────────────────────────────────────
def step1_baseline(refs, score_mat):
    print("[STEP 1] baseline MSS performance")
    rows = []
    for tgt in TARGETS:
        for ds, refs_ds in _group_by_dataset(refs).items():
            idx = np.array([i for i, _ in refs_ds])
            tgt_idx = ALL_SCORE_TARGETS.index(tgt)
            scores_block = np.stack([score_mat[m][idx] for m in ALL_SCORE_TARGETS])
            ranks = (np.argsort(-scores_block, axis=0) == tgt_idx).argmax(axis=0) + 1
            mol_truth = np.array([r["molecule_truth"] for _, r in refs_ds])
            tgt_mask = mol_truth == tgt
            n_tgt = int(tgt_mask.sum())
            row = {
                "molecule":          tgt,
                "dataset":           ds,
                "n_spectra_total":   len(idx),
                "n_spectra_target":  n_tgt,
                "baseline_mss_mean": float(score_mat[tgt][idx].mean()),
                "baseline_mss_sd":   float(score_mat[tgt][idx].std()),
            }
            if n_tgt >= 2:
                row.update({
                    "target_top1_rate": float((ranks[tgt_mask] == 1).mean()),
                    "target_top3_rate": float((ranks[tgt_mask] <= 3).mean()),
                    "target_top5_rate": float((ranks[tgt_mask] <= 5).mean()),
                })
            else:
                row.update({"target_top1_rate": np.nan,
                              "target_top3_rate": np.nan,
                              "target_top5_rate": np.nan})
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "baseline_per_molecule_dataset_v1.csv", index=False)
    return df


def _group_by_dataset(refs):
    """Return {dataset: [(index_in_refs, ref), ...]}."""
    by_ds = defaultdict(list)
    for i, r in enumerate(refs):
        by_ds[r["dataset"]].append((i, r))
    return dict(by_ds)


# ──────────────────────────────────────────────────────────────────────
# STEP 2 — Per-molecule observation profiles
# ──────────────────────────────────────────────────────────────────────
def step2_molecule_profiles(refs, master_x, templates_by_mol, score_mat):
    print("[STEP 2] per-molecule observation profiles")
    profiles = {}
    for tgt in TARGETS:
        # Restrict to spectra whose ground-truth molecule IS this target
        idx = [i for i, r in enumerate(refs) if r["molecule_truth"] == tgt]
        if len(idx) < 3:
            print(f"  {tgt}: only {len(idx)} pure spectra — minimal profile")
        if not idx:
            profiles[tgt] = None; continue

        # Per-regime split
        regimes = sorted({refs[i]["regime"] for i in idx})
        regime_data = {}
        for regime in regimes:
            r_idx = [i for i in idx if refs[i]["regime"] == regime]
            if not r_idx: continue
            cb = CORE_BANDS[tgt]["primary_window"]
            sb = CORE_BANDS[tgt]["secondary_window"]
            pos_p, area_p, prom_p, pos_s, area_s, prom_s = ([] for _ in range(6))
            for i in r_idx:
                y = refs[i].get("spectrum_pp", refs[i]["spectrum"])
                if not np.isfinite(y).any(): continue
                p1, a1, pr1 = _ring_features(y, master_x, cb[0], cb[1])
                p2, a2, pr2 = _ring_features(y, master_x, sb[0], sb[1])
                pos_p.append(p1); area_p.append(a1); prom_p.append(pr1)
                pos_s.append(p2); area_s.append(a2); prom_s.append(pr2)
            if not pos_p: continue
            pos_p = np.array(pos_p); area_p = np.array(area_p); prom_p = np.array(prom_p)
            valid = np.isfinite(pos_p) & (prom_p > 0)
            if valid.sum() < 2: continue
            pos_v = pos_p[valid]; prom_v = prom_p[valid]
            pos_mean, pos_sd = float(pos_v.mean()), float(pos_v.std() if len(pos_v) > 1 else 1.0)

            # Companion fire rates: anchor + support of this molecule's MSS template
            tps = templates_by_mol.get(tgt, {})
            t = tps.get(regime) or tps.get("SERS") or tps.get("Raman") or {}
            anchors = t.get("anchors", []); supports = t.get("supports", [])
            anchor_rates = {}
            support_rates = {}
            for a in anchors:
                hits = sum(1 for i in r_idx
                              if has_real_peak(refs[i].get("spectrum_pp", refs[i]["spectrum"]),
                                                  master_x, a, 5.0))
                anchor_rates[a] = hits / max(len(r_idx), 1)
            for s in supports:
                hits = sum(1 for i in r_idx
                              if has_real_peak(refs[i].get("spectrum_pp", refs[i]["spectrum"]),
                                                  master_x, s, 5.0))
                support_rates[s] = hits / max(len(r_idx), 1)
            consistent = {b: r for b, r in {**anchor_rates, **support_rates}.items() if r >= 0.50}
            absent     = {b: r for b, r in {**anchor_rates, **support_rates}.items() if r < 0.10}

            # Interference: mean MSS score for competitors on this molecule's pure spectra
            comp_scores = {m: float(score_mat[m][r_idx].mean())
                            for m in COMPETITORS.get(tgt, []) if m in score_mat}
            interference_load = float(np.mean(list(comp_scores.values()))) if comp_scores else 0.0

            regime_data[regime] = {
                "n":               int(len(r_idx)),
                "pos_mean":        pos_mean, "pos_sd": pos_sd,
                "pos_window_lo":   max(pos_mean - 2*pos_sd, CORE_BANDS[tgt]["primary_window"][0] - 5),
                "pos_window_hi":   min(pos_mean + 2*pos_sd, CORE_BANDS[tgt]["primary_window"][1] + 5),
                "prom_q25":        float(np.percentile(prom_v, 25)),
                "prom_q50":        float(np.percentile(prom_v, 50)),
                "prom_q75":        float(np.percentile(prom_v, 75)),
                "consistent_companions":    consistent,
                "absent_companions":        absent,
                "anchor_fire_rates":        anchor_rates,
                "support_fire_rates":       support_rates,
                "interference_per_competitor": comp_scores,
                "interference_load_mean":   interference_load,
            }
        profiles[tgt] = {
            "core_band_primary":   CORE_BANDS[tgt]["primary_window"],
            "core_band_secondary": CORE_BANDS[tgt]["secondary_window"],
            "regimes":             regime_data,
        }

    rows = []
    for tgt, p in profiles.items():
        if p is None:
            rows.append({"molecule": tgt, "regime": "—",
                            "n_pure_refs": 0, "status": "INSUFFICIENT_GT"})
            continue
        for regime, d in p["regimes"].items():
            rows.append({
                "molecule":          tgt,
                "regime":            regime,
                "n_pure_refs":       d["n"],
                "core_band_primary": f"{p['core_band_primary'][0]}-{p['core_band_primary'][1]}",
                "core_band_secondary": f"{p['core_band_secondary'][0]}-{p['core_band_secondary'][1]}",
                "primary_pos_mean":  d["pos_mean"],
                "primary_pos_sd":    d["pos_sd"],
                "pos_window_lo":     d["pos_window_lo"],
                "pos_window_hi":     d["pos_window_hi"],
                "prom_q50":          d["prom_q50"],
                "consistent_companions": "|".join(f"{int(b)}:{r:.2f}"
                                                       for b, r in d["consistent_companions"].items()),
                "absent_companions":     "|".join(f"{int(b)}:{r:.2f}"
                                                       for b, r in d["absent_companions"].items()),
                "interference_load": d["interference_load_mean"],
                "interference_per_competitor": ";".join(
                    f"{m}:{s:.2f}" for m, s in d["interference_per_competitor"].items()),
                "status":            "OK",
            })
    pd.DataFrame(rows).to_csv(TABLES / "molecule_profiles_v1.csv", index=False)
    return profiles


# ──────────────────────────────────────────────────────────────────────
# STEP 3 — Per-molecule post-hoc calibration wrappers
# ──────────────────────────────────────────────────────────────────────
# Per-molecule weights — kept conservative and almost identical to adenine v1.
# Per spec: weights MAY be molecule-specific but stay in the same family.
WRAPPER_WEIGHTS = {
    "uric_acid":     {"w_mss": 0.50, "w_core": 0.30, "w_prom": 0.10, "w_comp": 0.20, "w_int": 0.15},
    "hypoxanthine":  {"w_mss": 0.50, "w_core": 0.30, "w_prom": 0.10, "w_comp": 0.20, "w_int": 0.15},
    "ergothioneine": {"w_mss": 0.50, "w_core": 0.40, "w_prom": 0.10, "w_comp": 0.15, "w_int": 0.10},
    "glutathione":   {"w_mss": 0.50, "w_core": 0.40, "w_prom": 0.10, "w_comp": 0.15, "w_int": 0.10},
}


def step3_calibrate(refs, score_mat, profiles, master_x):
    print("[STEP 3] computing per-molecule calibrated wrapper scores")
    n = len(refs)
    cal = {tgt: np.zeros(n) for tgt in TARGETS}
    contrib = {tgt: {k: np.zeros(n) for k in
                          ("base_mss", "core_in_window", "prom_z", "companion_agree",
                           "interference_pen")} for tgt in TARGETS}

    for i, ref in enumerate(refs):
        y = ref.get("spectrum_pp", ref["spectrum"])
        if not np.isfinite(y).any(): continue
        regime = ref["regime"]
        for tgt in TARGETS:
            prof = profiles.get(tgt)
            w = WRAPPER_WEIGHTS[tgt]
            if prof is None: continue
            d = prof["regimes"].get(regime) or next(iter(prof["regimes"].values()), None)
            if d is None: continue

            base = float(score_mat[tgt][i])

            # Core-band-in-window
            cb = prof["core_band_primary"]
            pos, area, prom = _ring_features(y, master_x, cb[0], cb[1])
            in_window = float(d["pos_window_lo"] <= pos <= d["pos_window_hi"]) \
                            if np.isfinite(pos) else 0.0

            # Prominence z
            iqr = max(d["prom_q75"] - d["prom_q25"], 1e-9)
            prom_z = 0.0
            if np.isfinite(prom) and prom > 0:
                prom_z = float(np.clip((prom - d["prom_q50"]) / iqr, -1.5, 1.5))

            # Companion agreement
            n_consistent = max(len(d["consistent_companions"]), 1)
            companion_agree = float(np.clip(base / max(n_consistent / 5.0, 0.1), 0.0, 1.5))

            # Interference penalty: excess interferer signal vs molecule's profile baseline
            local_ifl_terms = [score_mat[m][i] for m in COMPETITORS.get(tgt, [])
                                  if m in score_mat]
            local_ifl = float(np.mean(local_ifl_terms)) if local_ifl_terms else 0.0
            interference_pen = float(max(local_ifl - max(d["interference_load_mean"], 1e-6), 0.0))

            calibrated = (
                w["w_mss"]  * base
              + w["w_core"] * in_window
              + w["w_prom"] * (prom_z + 0.5)
              + w["w_comp"] * companion_agree
              - w["w_int"]  * interference_pen
            )
            calibrated = float(np.clip(calibrated, 0.0, 1.5))
            cal[tgt][i] = calibrated
            contrib[tgt]["base_mss"][i]         = w["w_mss"]  * base
            contrib[tgt]["core_in_window"][i]   = w["w_core"] * in_window
            contrib[tgt]["prom_z"][i]           = w["w_prom"] * (prom_z + 0.5)
            contrib[tgt]["companion_agree"][i]  = w["w_comp"] * companion_agree
            contrib[tgt]["interference_pen"][i] = -w["w_int"] * interference_pen

    # Save per-spectrum calibrated scores + meta
    rows = []
    for i, ref in enumerate(refs):
        rows.append({
            "spectrum_id":    ref["spectrum_id"],
            "dataset":        ref["dataset"],
            "regime":         ref["regime"],
            "substrate":      ref["substrate"],
            "molecule_truth": ref["molecule_truth"],
            "raw_label":      ref["raw_label"],
            **{f"baseline_{tgt}_mss":   float(score_mat[tgt][i]) for tgt in TARGETS},
            **{f"calibrated_{tgt}":     float(cal[tgt][i])       for tgt in TARGETS},
        })
    pd.DataFrame(rows).to_csv(TABLES / "calibrated_per_spectrum_v1.csv", index=False)
    return cal, contrib


# ──────────────────────────────────────────────────────────────────────
# STEP 4 — Evaluation per molecule
# ──────────────────────────────────────────────────────────────────────
def step4_evaluate(refs, score_mat, cal, contrib):
    print("[STEP 4] evaluation baseline vs calibrated per molecule")
    rows_id = []
    rows_tx = []
    rows_spec = []

    for tgt in TARGETS:
        tgt_idx_in_score = ALL_SCORE_TARGETS.index(tgt)
        # Identity per dataset
        for ds, refs_ds in _group_by_dataset(refs).items():
            local_idx = np.array([i for i, _ in refs_ds])
            mol_truth = np.array([r["molecule_truth"] for _, r in refs_ds])
            tgt_mask = mol_truth == tgt
            n_tgt = int(tgt_mask.sum())
            if n_tgt < 2: continue
            scores_block = np.stack([score_mat[m][local_idx] for m in ALL_SCORE_TARGETS])
            base_ranks = (np.argsort(-scores_block, axis=0) == tgt_idx_in_score) \
                              .argmax(axis=0) + 1
            cal_block = scores_block.copy()
            cal_block[tgt_idx_in_score] = cal[tgt][local_idx]
            cal_ranks = (np.argsort(-cal_block, axis=0) == tgt_idx_in_score) \
                            .argmax(axis=0) + 1

            rows_id.append({
                "molecule": tgt, "dataset": ds, "n_target_spectra": n_tgt,
                "baseline_top1": float((base_ranks[tgt_mask] == 1).mean()),
                "calibrated_top1": float((cal_ranks[tgt_mask] == 1).mean()),
                "baseline_top3": float((base_ranks[tgt_mask] <= 3).mean()),
                "calibrated_top3": float((cal_ranks[tgt_mask] <= 3).mean()),
                "baseline_top5": float((base_ranks[tgt_mask] <= 5).mean()),
                "calibrated_top5": float((cal_ranks[tgt_mask] <= 5).mean()),
                "delta_top1":   float((cal_ranks[tgt_mask] == 1).mean() -
                                          (base_ranks[tgt_mask] == 1).mean()),
                "delta_top3":   float((cal_ranks[tgt_mask] <= 3).mean() -
                                          (base_ranks[tgt_mask] <= 3).mean()),
                "delta_top5":   float((cal_ranks[tgt_mask] <= 5).mean() -
                                          (base_ranks[tgt_mask] <= 5).mean()),
            })

            # Specificity check: does the wrapper inflate target's calibrated score
            # for spectra whose truth IS NOT this target? (should not push other-truth
            # spectra into target top-1)
            other_mask = (mol_truth != tgt) & (mol_truth.astype(object) != None)
            other_mask = other_mask & np.array([m is not None for m in mol_truth])
            n_other = int(other_mask.sum())
            if n_other >= 2:
                rows_spec.append({
                    "molecule": tgt, "dataset": ds, "n_other_spectra": n_other,
                    "baseline_other_top1_rate":   float((base_ranks[other_mask] == 1).mean()),
                    "calibrated_other_top1_rate": float((cal_ranks[other_mask] == 1).mean()),
                    "false_positive_increase":    float((cal_ranks[other_mask] == 1).mean() -
                                                            (base_ranks[other_mask] == 1).mean()),
                })

    # Transformation tests (UA-specific)
    # Uricase: SerumSigma vs SerumSigma+Enzyme — UA should DROP with enzyme
    sigma_no = [i for i, r in enumerate(refs) if r["raw_label"] == "SerumSigma"]
    sigma_e  = [i for i, r in enumerate(refs) if r["raw_label"] == "SerumSigma+Enzyme"]
    if len(sigma_no) >= 2 and len(sigma_e) >= 2:
        d_base = _cohens_d(score_mat["uric_acid"][sigma_e],
                              score_mat["uric_acid"][sigma_no])
        d_cal  = _cohens_d(cal["uric_acid"][sigma_e], cal["uric_acid"][sigma_no])
        rows_tx.append({
            "molecule": "uric_acid",
            "test":     "uricase_SerumSigma_vs_+Enzyme",
            "expected_direction": "negative_d (UA drops with enzyme)",
            "baseline_d":   d_base,
            "calibrated_d": d_cal,
            "directionally_correct_baseline":   bool(not np.isnan(d_base) and d_base < -0.3),
            "directionally_correct_calibrated": bool(not np.isnan(d_cal) and d_cal < -0.3),
        })
    # Isotopic: UA vs UAiso
    ua_only = [i for i, r in enumerate(refs) if r["raw_label"] == "UA"]
    uaiso   = [i for i, r in enumerate(refs) if r["raw_label"] == "UAiso"]
    if len(ua_only) >= 2 and len(uaiso) >= 2:
        d_base = _cohens_d(score_mat["uric_acid"][uaiso], score_mat["uric_acid"][ua_only])
        d_cal  = _cohens_d(cal["uric_acid"][uaiso], cal["uric_acid"][ua_only])
        rows_tx.append({
            "molecule": "uric_acid",
            "test":     "isotopic_UA_vs_UAiso",
            "expected_direction": "near-zero or modest shift; UA template should still fire",
            "baseline_d": d_base,
            "calibrated_d": d_cal,
            "directionally_correct_baseline":   bool(not np.isnan(d_base) and abs(d_base) < 1.0),
            "directionally_correct_calibrated": bool(not np.isnan(d_cal) and abs(d_cal) < 1.0),
        })
    # SERS_fitting: HX vs UA cohorts
    hyp = [i for i, r in enumerate(refs) if r["raw_label"] == "Hypoxanthine"]
    uaf = [i for i, r in enumerate(refs) if r["raw_label"] in ("UA_free", "UA_bound")]
    if len(hyp) >= 2 and len(uaf) >= 2:
        d_base = _cohens_d(score_mat["hypoxanthine"][hyp],
                              score_mat["hypoxanthine"][uaf])
        d_cal  = _cohens_d(cal["hypoxanthine"][hyp], cal["hypoxanthine"][uaf])
        rows_tx.append({
            "molecule": "hypoxanthine",
            "test":     "SERS_fitting_Hypox_vs_UA_cohorts",
            "expected_direction": "positive_d (HX score elevated in Hypox cohort)",
            "baseline_d":   d_base,
            "calibrated_d": d_cal,
            "directionally_correct_baseline":   bool(not np.isnan(d_base) and d_base > 0.5),
            "directionally_correct_calibrated": bool(not np.isnan(d_cal) and d_cal > 0.5),
        })
    # ERG calibration: ρ(log conc, ERG score)
    erg_idx = [i for i, r in enumerate(refs) if r["dataset"] == "erg_calibration"]
    if len(erg_idx) >= 5:
        # Parse conc from cohort label
        conc_vals = []
        for i in erg_idx:
            lab = refs[i]["raw_label"] or refs[i].get("conc_label", "") or ""
            try:
                v = float(str(lab).replace("ng_mL", "").replace("ngmL", "").strip())
            except ValueError:
                v = np.nan
            conc_vals.append(v)
        conc = np.array(conc_vals, dtype=float)
        # Use ERG_calibration loader's conc field if present (most reliable)
        for k, i in enumerate(erg_idx):
            c_field = refs[i].get("conc")
            if c_field is not None:
                try: conc[k] = float(c_field)
                except (TypeError, ValueError): pass
        log_c = np.log10(np.where(conc > 0, conc, 1e-3))
        rho_base = _spearman(log_c, score_mat["ergothioneine"][erg_idx])
        rho_cal  = _spearman(log_c, cal["ergothioneine"][erg_idx])
        rows_tx.append({
            "molecule": "ergothioneine",
            "test":     "ERG_calibration_dose_response",
            "expected_direction": "positive_rho (ERG score scales with log conc)",
            "baseline_d":   rho_base,
            "calibrated_d": rho_cal,
            "directionally_correct_baseline":   bool(not np.isnan(rho_base) and rho_base > 0.5),
            "directionally_correct_calibrated": bool(not np.isnan(rho_cal) and rho_cal > 0.5),
        })

    pd.DataFrame(rows_id).to_csv(TABLES / "evaluation_identity_per_dataset_v1.csv", index=False)
    pd.DataFrame(rows_tx).to_csv(TABLES / "evaluation_transformation_v1.csv", index=False)
    pd.DataFrame(rows_spec).to_csv(TABLES / "evaluation_specificity_v1.csv", index=False)
    return pd.DataFrame(rows_id), pd.DataFrame(rows_tx), pd.DataFrame(rows_spec)


# ──────────────────────────────────────────────────────────────────────
# STEP 5 — Failure mode analysis
# ──────────────────────────────────────────────────────────────────────
def step5_failure_modes(id_df, spec_df, profiles, tx_df):
    print("[STEP 5] failure mode analysis")
    rows = []
    for tgt in TARGETS:
        sub_id = id_df[id_df.molecule == tgt]
        sub_sp = spec_df[spec_df.molecule == tgt]
        sub_tx = tx_df[tx_df.molecule == tgt]
        if sub_id.empty:
            rows.append({"molecule": tgt, "outcome": "INSUFFICIENT_DATA",
                            "delta_top3_mean": np.nan,
                            "max_specificity_loss": np.nan,
                            "core_band_present": "n/a", "diagnosis": "no per-dataset evaluation rows"})
            continue
        delta_top3_mean = float(sub_id["delta_top3"].mean())
        max_fp_increase = float(sub_sp["false_positive_increase"].max()) if not sub_sp.empty else 0.0
        # Identify whether core band fires consistently in pure spectra
        core_present = "n/a"
        prof = profiles.get(tgt)
        if prof and prof["regimes"]:
            present_regimes = [r for r, d in prof["regimes"].items()
                                  if d["pos_sd"] < 6 and d["prom_q50"] > 0]
            core_present = "|".join(present_regimes) if present_regimes else "weak"

        # Outcome
        if max_fp_increase > 0.10 and delta_top3_mean > 0:
            outcome = "PARTIAL_SPECIFICITY_DROP"
            diagnosis = "wrapper boosts target but inflates other-cohort top-1 — likely shared core band"
        elif delta_top3_mean >= 0.05 and max_fp_increase <= 0.10:
            outcome = "SUCCESS"
            diagnosis = "identity recovered without specificity loss"
        elif delta_top3_mean > 0:
            outcome = "MODEST_GAIN"
            diagnosis = "small identity gain; core band fires intermittently"
        elif delta_top3_mean == 0 or np.isnan(delta_top3_mean):
            outcome = "NEUTRAL"
            diagnosis = "wrapper neither helps nor hurts"
        else:
            outcome = "FAIL"
            diagnosis = "calibrated identity worse than baseline"

        # Transformation directionality
        tx_ok_base = sub_tx["directionally_correct_baseline"].astype(bool).tolist()
        tx_ok_cal  = sub_tx["directionally_correct_calibrated"].astype(bool).tolist()
        rows.append({
            "molecule":             tgt,
            "delta_top3_mean":      delta_top3_mean,
            "max_specificity_loss": max_fp_increase,
            "n_datasets_evaluated": int(len(sub_id)),
            "core_band_present_in": core_present,
            "transformation_tests_baseline_ok":   sum(tx_ok_base),
            "transformation_tests_calibrated_ok": sum(tx_ok_cal),
            "outcome":              outcome,
            "diagnosis":            diagnosis,
        })
    pd.DataFrame(rows).to_csv(TABLES / "failure_mode_analysis_v1.csv", index=False)
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────
# STEP 6 — Generalization assessment
# ──────────────────────────────────────────────────────────────────────
def step6_generalization(failure_df):
    print("[STEP 6] generalization assessment")
    families = {
        "purine": ["uric_acid", "hypoxanthine"],
        "sulfur_redox": ["ergothioneine", "glutathione"],
    }
    rows = []
    for fam, mols in families.items():
        sub = failure_df[failure_df.molecule.isin(mols)]
        if sub.empty: continue
        rows.append({
            "family":           fam,
            "molecules":        "|".join(mols),
            "mean_delta_top3":  float(sub.delta_top3_mean.mean()),
            "outcomes":         "|".join(sub.outcome.tolist()),
            "n_success":        int((sub.outcome == "SUCCESS").sum()),
            "n_partial_or_fail": int(sub.outcome.isin(["PARTIAL_SPECIFICITY_DROP",
                                                                  "FAIL", "MODEST_GAIN", "NEUTRAL"]).sum()),
        })
    pd.DataFrame(rows).to_csv(TABLES / "generalization_assessment_v1.csv", index=False)
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────
# Figures
# ──────────────────────────────────────────────────────────────────────
def make_figures(id_df, spec_df, tx_df, failure_df, cal, score_mat, refs):
    print("[FIG] generating figures")
    # Fig 1: per-molecule mean Δtop1 / Δtop3
    try:
        agg = id_df.groupby("molecule").agg(
            mean_delta_top1=("delta_top1", "mean"),
            mean_delta_top3=("delta_top3", "mean"),
        ).reindex(TARGETS)
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(TARGETS)); w = 0.4
        ax.bar(x - w/2, agg.mean_delta_top1.fillna(0), w, label="Δtop-1", color="#4C72B0")
        ax.bar(x + w/2, agg.mean_delta_top3.fillna(0), w, label="Δtop-3", color="#DD8452")
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xticks(x); ax.set_xticklabels(TARGETS, rotation=15)
        ax.set_ylabel("calibrated − baseline hit rate"); ax.legend()
        ax.set_title("Mean Δ identity hit rate per molecule (across datasets)")
        for i, (a, b) in enumerate(zip(agg.mean_delta_top1.fillna(0),
                                              agg.mean_delta_top3.fillna(0))):
            ax.text(i - w/2, a + 0.01 if a >= 0 else a - 0.02, f"{a:+.2f}",
                       ha="center", fontsize=8)
            ax.text(i + w/2, b + 0.01 if b >= 0 else b - 0.02, f"{b:+.2f}",
                       ha="center", fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGS / "fig_per_molecule_delta_topk_v1.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig delta topk issue: {e}")

    # Fig 2: per-molecule baseline vs calibrated top-3 by dataset
    try:
        fig, axes = plt.subplots(1, len(TARGETS), figsize=(4 * len(TARGETS), 5), sharey=True)
        for ax, tgt in zip(axes, TARGETS):
            sub = id_df[id_df.molecule == tgt]
            if sub.empty:
                ax.set_title(tgt + " — no data"); continue
            x = np.arange(len(sub)); w = 0.4
            ax.bar(x - w/2, sub.baseline_top3.values, w, label="baseline", color="#888")
            ax.bar(x + w/2, sub.calibrated_top3.values, w, label="calibrated", color="#4C72B0")
            ax.set_xticks(x); ax.set_xticklabels(sub.dataset.values, rotation=30, fontsize=7)
            ax.set_title(tgt); ax.set_ylim(0, 1.05)
            if ax is axes[0]: ax.set_ylabel("top-3 hit rate"); ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGS / "fig_per_molecule_top3_baseline_vs_calibrated_v1.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig per-mol top3 issue: {e}")

    # Fig 3: specificity check — false positive increase per molecule × dataset
    try:
        fig, ax = plt.subplots(figsize=(9, 5))
        for tgt, color in zip(TARGETS, ["#4C72B0", "#DD8452", "#2ca02c", "#9467bd"]):
            sub = spec_df[spec_df.molecule == tgt].sort_values("dataset")
            if sub.empty: continue
            ax.plot(sub.dataset.values, sub.false_positive_increase.values,
                       marker="o", label=tgt, color=color)
        ax.axhline(0, color="black", lw=0.5)
        ax.axhline(0.10, color="red", ls="--", lw=0.8, label="+10pp threshold")
        ax.set_ylabel("false positive top-1 rate (calibrated − baseline)")
        ax.set_title("Specificity check — wrapper inflation of other-truth spectra")
        ax.legend(fontsize=8)
        plt.xticks(rotation=20, fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGS / "fig_specificity_per_molecule_dataset_v1.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig specificity issue: {e}")

    # Fig 4: transformation tests
    try:
        if not tx_df.empty:
            fig, ax = plt.subplots(figsize=(9, 5))
            x = np.arange(len(tx_df)); w = 0.4
            ax.bar(x - w/2, tx_df.baseline_d.fillna(0), w, label="baseline", color="#888")
            ax.bar(x + w/2, tx_df.calibrated_d.fillna(0), w, label="calibrated", color="#4C72B0")
            ax.axhline(0, color="black", lw=0.5)
            ax.set_xticks(x); ax.set_xticklabels(tx_df.test.values, rotation=20, fontsize=7)
            ax.set_ylabel("Cohen's d (or ρ for ERG calibration)")
            ax.set_title("Transformation / dose-response tests — baseline vs calibrated")
            ax.legend()
            for i, (b, c) in enumerate(zip(tx_df.baseline_d.fillna(0), tx_df.calibrated_d.fillna(0))):
                ax.text(i - w/2, b, f"{b:+.2f}", ha="center", fontsize=7,
                           va="bottom" if b >= 0 else "top")
                ax.text(i + w/2, c, f"{c:+.2f}", ha="center", fontsize=7,
                           va="bottom" if c >= 0 else "top")
            fig.tight_layout()
            fig.savefig(FIGS / "fig_transformation_tests_v1.png", dpi=150)
            plt.close(fig)
    except Exception as e:
        print(f"  fig transformation issue: {e}")

    # Fig 5: outcome summary per molecule
    try:
        outcome_color = {"SUCCESS": "#2ca02c", "MODEST_GAIN": "#f39c12",
                            "PARTIAL_SPECIFICITY_DROP": "#9467bd",
                            "NEUTRAL": "#888", "FAIL": "#c0392b",
                            "INSUFFICIENT_DATA": "#aaaaaa"}
        fig, ax = plt.subplots(figsize=(9, 4))
        for i, (_, r) in enumerate(failure_df.iterrows()):
            ax.bar(i, 1, color=outcome_color.get(r["outcome"], "#888"))
            ax.text(i, 0.5, r["outcome"], rotation=0, ha="center", va="center",
                       color="white", fontweight="bold", fontsize=9)
        ax.set_xticks(range(len(failure_df)))
        ax.set_xticklabels(failure_df.molecule.values, rotation=15)
        ax.set_yticks([])
        ax.set_title("Outcome per molecule — substrate calibration wrapper")
        fig.tight_layout()
        fig.savefig(FIGS / "fig_outcome_per_molecule_v1.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig outcome issue: {e}")


# ──────────────────────────────────────────────────────────────────────
# Reports
# ──────────────────────────────────────────────────────────────────────
def write_profile_summary(profiles):
    lines = ["# Molecule profile summary — substrate calibration multi-molecule v1\n",
                f"date: {datetime.now().isoformat()}", ""]
    for tgt in TARGETS:
        p = profiles.get(tgt)
        if p is None:
            lines.append(f"## {tgt}  — INSUFFICIENT_GT\n"); continue
        lines.append(f"## {tgt}\n")
        lines.append(f"- core band primary window: {p['core_band_primary'][0]}-{p['core_band_primary'][1]} cm⁻¹")
        lines.append(f"- core band secondary window: {p['core_band_secondary'][0]}-{p['core_band_secondary'][1]} cm⁻¹")
        for regime, d in p["regimes"].items():
            lines.append(f"### {regime} regime (n={d['n']} pure refs)")
            lines.append(f"- primary peak position: {d['pos_mean']:.1f} ± {d['pos_sd']:.2f} cm⁻¹  "
                            f"(method-typical window {d['pos_window_lo']:.1f}-{d['pos_window_hi']:.1f})")
            lines.append(f"- prominence quartiles: q25={d['prom_q25']:.3f} / q50={d['prom_q50']:.3f} / q75={d['prom_q75']:.3f}")
            cc = list(d["consistent_companions"].items())
            ac = list(d["absent_companions"].items())
            lines.append(f"- consistent companion bands (≥50%): "
                            f"{', '.join(f'{int(b)} ({r:.0%})' for b, r in cc) or '(none)'}")
            lines.append(f"- absent companion bands (<10%): "
                            f"{', '.join(f'{int(b)} ({r:.0%})' for b, r in ac) or '(none)'}")
            ic = list(d["interference_per_competitor"].items())
            lines.append(f"- interferer mean MSS on this molecule's pure spectra: "
                            f"{', '.join(f'{m}={s:.2f}' for m, s in ic)}")
            lines.append("")
    (REPORTS / "MOLECULE_PROFILE_SUMMARY.md").write_text("\n".join(lines))


def write_effectiveness_analysis(failure_df, gen_df, decision):
    lines = ["# Wrapper effectiveness analysis\n",
                f"date: {datetime.now().isoformat()}", "",
                f"## Decision: **{decision}**\n",
                "## Per-molecule outcome\n",
                "| molecule | n datasets | mean Δtop3 | max specificity loss | tx baseline ok | tx calibrated ok | outcome | diagnosis |",
                "|---|---:|---:|---:|---:|---:|---|---|"]
    for _, r in failure_df.iterrows():
        lines.append(f"| {r['molecule']} | {int(r['n_datasets_evaluated'])} | "
                        f"{r['delta_top3_mean']:+.2f} | {r['max_specificity_loss']:+.2f} | "
                        f"{int(r['transformation_tests_baseline_ok'])} | "
                        f"{int(r['transformation_tests_calibrated_ok'])} | "
                        f"{r['outcome']} | {r['diagnosis']} |")
    lines.append("")
    lines.append("## Generalization by chemistry family\n")
    lines.append("| family | molecules | mean Δtop3 | n_success / n_total |")
    lines.append("|---|---|---:|---:|")
    for _, r in gen_df.iterrows():
        n_total = r["n_success"] + r["n_partial_or_fail"]
        lines.append(f"| {r['family']} | {r['molecules']} | {r['mean_delta_top3']:+.2f} | "
                        f"{r['n_success']}/{n_total} |")
    (REPORTS / "WRAPPER_EFFECTIVENESS_ANALYSIS.md").write_text("\n".join(lines))


def write_final_report(decision, id_df, tx_df, spec_df, failure_df, gen_df):
    lines = [
        "# REPORT — substrate calibration multi-molecule v1\n",
        f"## Decision: **{decision}**\n",
        "## Setup",
        "- Extends the substrate-aware POST-HOC calibration wrapper from adenine to UA, HX, ERG, GSH using calibration / pure datasets ONLY (no disease cohorts).",
        "- Calibration sources: serum_ag_colloids (SERS_metab_63, SERS fitting Hypox/UAfree/UAbound, uricase, isotopic, ERG_calibration), Gobbato Raman powder, RamanBioLib.",
        "- Wrapper formula per molecule (post-hoc only):",
        "  `calibrated = w1·MSS + w2·core_in_window + w3·prom_z + w4·companion_agree − w5·interference_pen`",
        "  Default weights kept conservative; weights MAY be molecule-specific (see WRAPPER_WEIGHTS in driver).",
        "- Engine v4.5 / MSS scoring kernel / motif registry / MSS templates (repaired narrow registry v1) / 11-axis BSV / preprocessing — UNCHANGED.",
        "",
        "## Per-molecule outcomes\n",
    ]
    for _, r in failure_df.iterrows():
        lines.append(f"- **{r['molecule']}** — {r['outcome']} ({r['diagnosis']}). "
                        f"n datasets = {int(r['n_datasets_evaluated'])}, "
                        f"mean Δtop3 = {r['delta_top3_mean']:+.2f}, "
                        f"max specificity loss = {r['max_specificity_loss']:+.2f}.")
    lines.append("")

    # Identity summary
    if not id_df.empty:
        lines.append("## Identity per-molecule × dataset (top-3)\n")
        lines.append("| molecule | dataset | n | baseline top-3 | calibrated top-3 | Δtop3 |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for _, r in id_df.sort_values(["molecule", "dataset"]).iterrows():
            lines.append(f"| {r['molecule']} | {r['dataset']} | {int(r['n_target_spectra'])} | "
                            f"{r['baseline_top3']:.2f} | {r['calibrated_top3']:.2f} | "
                            f"{r['delta_top3']:+.2f} |")
        lines.append("")

    if not tx_df.empty:
        lines.append("## Transformation / dose-response\n")
        lines.append("| molecule | test | expected | baseline | calibrated | base ok | cal ok |")
        lines.append("|---|---|---|---:|---:|---|---|")
        for _, r in tx_df.iterrows():
            lines.append(f"| {r['molecule']} | {r['test']} | {r['expected_direction']} | "
                            f"{r['baseline_d']:+.2f} | {r['calibrated_d']:+.2f} | "
                            f"{'✓' if r['directionally_correct_baseline'] else '✗'} | "
                            f"{'✓' if r['directionally_correct_calibrated'] else '✗'} |")
        lines.append("")

    if not spec_df.empty:
        lines.append("## Specificity (false positive top-1 rate on other-cohort spectra)\n")
        lines.append("| molecule | dataset | n other | baseline FP | calibrated FP | Δ |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for _, r in spec_df.sort_values(["molecule", "dataset"]).iterrows():
            lines.append(f"| {r['molecule']} | {r['dataset']} | {int(r['n_other_spectra'])} | "
                            f"{r['baseline_other_top1_rate']:.2f} | "
                            f"{r['calibrated_other_top1_rate']:.2f} | "
                            f"{r['false_positive_increase']:+.2f} |")
        lines.append("")

    lines.append("## Generalization assessment\n")
    lines.append("| family | molecules | mean Δtop3 | n_success / n_total |")
    lines.append("|---|---|---:|---:|")
    for _, r in gen_df.iterrows():
        n_total = r["n_success"] + r["n_partial_or_fail"]
        lines.append(f"| {r['family']} | {r['molecules']} | {r['mean_delta_top3']:+.2f} | "
                        f"{r['n_success']}/{n_total} |")
    lines.append("")

    lines.append("## Honest reading")
    lines.append("This phase tests whether the substrate-aware post-hoc wrapper pattern proven on adenine generalizes "
                    "to UA, HX, ERG, GSH using calibration / pure-component datasets only. The wrapper is identical in "
                    "structure (5 components: MSS + core_in_window + prom_z + companion_agree − interference_pen) with "
                    "molecule-specific core-band windows derived empirically from the molecule's own pure-component "
                    "spectra. Engine, MSS, motif, BSV and preprocessing are all unchanged. Disease labels are NOT used; "
                    "evaluation uses pure-component cohort labels (which molecule each spectrum is) as ground truth — "
                    "those are provenance metadata, not disease outcomes.")
    (REPORTS / "REPORT_substrate_calibration_multi_molecule_v1.md").write_text("\n".join(lines))


def write_audit(decision):
    txt = [
        "# gaira_base_4_substrate_calibration_multi_molecule_v1 — audit log",
        f"date: {datetime.now().isoformat()}",
        "",
        "## Strict negative invariants",
        "- NO engine changes",
        "- NO MSS scoring kernel changes",
        "- NO motif registry changes",
        "- NO MSS template changes (uses repaired narrow registry v1 read-only)",
        "- NO 11-axis BSV weight changes",
        "- NO preprocessing changes",
        "- NO soft-MSS scoring",
        "- NO global threshold changes",
        "- NO retraining of MSS",
        "- NO classifier-first framing",
        "- NO feedback into GAIRA",
        "- NO disease labels (cohort labels used here are pure-component molecule provenance, not disease outcomes)",
        "",
        "## Wrapper contract",
        "- Post-hoc per-molecule calibrated_score = w1·MSS + w2·core_in_window + w3·(prom_z+0.5) + w4·companion_agree − w5·interference_pen",
        "- Per-molecule weights allowed (UA/HX = adenine-default; ERG/GSH have slightly higher core weight)",
        "- Reranking substitutes that target's column in the score block; OTHER molecules retain raw MSS",
        "",
        "## Outputs",
        "- tables/baseline_per_molecule_dataset_v1.csv",
        "- tables/molecule_profiles_v1.csv",
        "- tables/calibrated_per_spectrum_v1.csv",
        "- tables/evaluation_identity_per_dataset_v1.csv",
        "- tables/evaluation_transformation_v1.csv",
        "- tables/evaluation_specificity_v1.csv",
        "- tables/failure_mode_analysis_v1.csv",
        "- tables/generalization_assessment_v1.csv",
        "- 5 figures (delta topk, per-molecule top3 by dataset, specificity, transformation, outcome)",
        "- reports/REPORT_substrate_calibration_multi_molecule_v1.md",
        "- reports/MOLECULE_PROFILE_SUMMARY.md",
        "- reports/WRAPPER_EFFECTIVENESS_ANALYSIS.md",
        "",
        f"## Final decision\n**{decision}**",
    ]
    (AUDIT / "gaira_base_4_substrate_calibration_multi_molecule_v1_audit_log.md").write_text("\n".join(txt))


# ──────────────────────────────────────────────────────────────────────
# Decision logic
# ──────────────────────────────────────────────────────────────────────
def _decision(failure_df) -> str:
    n_success = int((failure_df.outcome == "SUCCESS").sum())
    n_partial = int(failure_df.outcome.isin(["PARTIAL_SPECIFICITY_DROP", "MODEST_GAIN"]).sum())
    n_fail    = int(failure_df.outcome.isin(["FAIL", "INSUFFICIENT_DATA"]).sum())
    if n_success >= 2:
        return "WRAPPER_GENERALIZES_BEYOND_ADENINE"
    if n_success == 1 and n_partial >= 2:
        return "WRAPPER_PARTIALLY_GENERALIZES_NEEDS_MOLECULE_SPECIFIC_TUNING"
    if n_success == 0 and n_partial >= 2:
        return "WRAPPER_LIMITED_GENERALIZATION_REQUIRES_PER_MOLECULE_REWORK"
    if n_fail >= 2:
        return "WRAPPER_FAILS_TO_GENERALIZE"
    return "WRAPPER_PARTIALLY_GENERALIZES_NEEDS_MOLECULE_SPECIFIC_TUNING"


# ──────────────────────────────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 78)
    print("gaira_base_4_substrate_calibration_multi_molecule_v1")
    print("=" * 78)
    master_x = canonical_master_axis()

    print("[load] templates")
    templates, _, _ = load_templates()
    by_mol = defaultdict(dict)
    for t in templates:
        by_mol[t["molecule"]][t["regime"]] = t

    print("[load] gathering calibration refs")
    refs = gather_calibration_refs(master_x)

    print("[score] per-spectrum × per-target/competitor MSS scores")
    score_mat = score_per_spectrum(refs, master_x, by_mol)

    baseline_df = step1_baseline(refs, score_mat)
    profiles    = step2_molecule_profiles(refs, master_x, by_mol, score_mat)
    cal, contrib = step3_calibrate(refs, score_mat, profiles, master_x)
    id_df, tx_df, spec_df = step4_evaluate(refs, score_mat, cal, contrib)
    failure_df = step5_failure_modes(id_df, spec_df, profiles, tx_df)
    gen_df     = step6_generalization(failure_df)

    make_figures(id_df, spec_df, tx_df, failure_df, cal, score_mat, refs)
    write_profile_summary(profiles)
    decision = _decision(failure_df)
    write_effectiveness_analysis(failure_df, gen_df, decision)
    write_final_report(decision, id_df, tx_df, spec_df, failure_df, gen_df)
    write_audit(decision)
    try:
        shutil.copy(__file__, CODE_SNAPSHOT / Path(__file__).name)
    except Exception:
        pass
    print(f"[done] decision: {decision}")


if __name__ == "__main__":
    main()
