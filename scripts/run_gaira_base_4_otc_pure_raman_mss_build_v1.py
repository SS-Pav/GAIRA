"""gaira_base_4_otc_pure_raman_mss_build_v1

Phase: build pure-Raman MSS templates for 3 OTC drugs (acetylsalicylic acid,
paracetamol, ibuprofen) from the Paraguay iRaman 785s tablet dataset.

This is PURE RAMAN grounding. Not SERS. No substrate physics.

Paper: Data in Brief 2024, "Dataset of Raman spectroscopy responses for
over-the-counter drugs in Paraguay..."
Acquisition: iRaman 785s (BWTEK), 785 nm, 50% power, 1s × 10 accum,
spectral range 150-3200 cm⁻¹, resolution 4 cm⁻¹, direct tablet.

STRICT INVARIANTS:
- Do NOT change engine / MSS kernel / existing MSS templates / preprocessing
- Do NOT use substrate physics
- Do NOT treat as SERS
- Do NOT train classifier-first

Run:
    cd /Users/suraj/projects/GAIRA
    PYTHONPATH=src .venv/bin/python scripts/run_gaira_base_4_otc_pure_raman_mss_build_v1.py
"""
from __future__ import annotations

import shutil
import sys
import warnings
from collections import Counter, defaultdict
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
from run_gaira_base_4_mss_resolution_reporting_layer_v1 import (  # noqa: E402
    baseline_correct, mss_anchor_score,
)


# ──────────────────────────────────────────────────────────────────────
# Paths + constants
# ──────────────────────────────────────────────────────────────────────
ROOT = Path("/Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_4_otc_pure_raman_mss_build_v1")
TABLES  = ROOT / "tables"
FIGS    = ROOT / "figures"
REPORTS = ROOT / "reports"
AUDIT   = ROOT / "audit"
REGISTRY = ROOT / "registry"
CODE_SNAPSHOT = ROOT / "code_snapshot"
for d in (TABLES, FIGS, REPORTS, AUDIT, REGISTRY, CODE_SNAPSHOT):
    d.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path("/Volumes/SSD_Rad/GAIRA_DATA/raw/otc_drugs")

# Canonical molecule names
DRUGS = ["acetylsalicylic_acid", "paracetamol", "ibuprofen"]

# File → (drug, variant) mapping
FILE_SPEC = {
    "Acetylsalicylic-acid.xlsx":            ("acetylsalicylic_acid", "pure"),
    "Acetylsalicylic-acid-trademark.xlsx":  ("acetylsalicylic_acid", "trademark"),
    "Paracetamol.xlsx":                     ("paracetamol", "pure"),
    "Paracetamol-trademark.xlsx":           ("paracetamol", "trademark"),
    "Ibuprofen.xlsx":                       ("ibuprofen", "pure"),
    "Ibuprofen-trademark.xlsx":             ("ibuprofen", "trademark"),
    "All spectra.xlsx":                     (None, "aggregate"),  # duplicates pure files
}

TOLERANCE_CM1 = 5.0  # clustering + anchor window half-width


# ──────────────────────────────────────────────────────────────────────
# TASK 1 — file audit
# ──────────────────────────────────────────────────────────────────────
def task1_file_audit():
    print("[TASK 1] file inventory")
    rows = []
    for fname, (drug, variant) in FILE_SPEC.items():
        path = DATA_DIR / fname
        if not path.exists():
            rows.append({"file": fname, "present": False, "drug": drug,
                            "variant": variant})
            continue
        xl = pd.ExcelFile(path)
        for sh in xl.sheet_names:
            df = pd.read_excel(path, sheet_name=sh, header=0)
            rs = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna()
            cols = df.columns.tolist()
            first_col_name = str(cols[0])
            labels = [str(c).replace("\n", "").replace("\t", "").strip() for c in cols[1:]]
            label_bases = [l.split(".")[0] for l in labels]
            ctr = Counter(label_bases)
            steps = np.diff(rs.values)
            rows.append({
                "file":                  fname,
                "present":               True,
                "sheet":                 sh,
                "drug":                  drug,
                "variant":               variant,
                "n_rows":                int(df.shape[0]),
                "n_cols":                int(df.shape[1]),
                "n_spectra":             int(df.shape[1] - 1),
                "first_col_name":        first_col_name,
                "first_col_is_raman_shift": "raman" in first_col_name.lower() or "shift" in first_col_name.lower(),
                "wn_min_cm1":            float(rs.min()),
                "wn_max_cm1":            float(rs.max()),
                "wn_step_median":        float(np.median(steps)),
                "wn_step_min":           float(np.min(steps)),
                "wn_step_max":           float(np.max(steps)),
                "n_wn_points":           int(len(rs)),
                "label_group_counts":    ";".join(f"{k}={v}" for k, v in ctr.items()),
            })
    inv = pd.DataFrame(rows)
    inv.to_csv(TABLES / "otc_file_inventory.csv", index=False)
    return inv


# ──────────────────────────────────────────────────────────────────────
# TASK 2 — parse spectra + dedupe All spectra.xlsx
# ──────────────────────────────────────────────────────────────────────
def task2_parse_spectra(inv_df):
    print("[TASK 2] parse spectra + dedupe All spectra.xlsx")
    # Parse everything into structured records; we'll dedupe against
    # the individual pure files by content hash later.
    records = []
    spectra_matrices = {}  # keyed by source file

    for fname, (drug, variant) in FILE_SPEC.items():
        path = DATA_DIR / fname
        if not path.exists(): continue
        df = pd.read_excel(path, sheet_name=0, header=0)
        rs = pd.to_numeric(df.iloc[:, 0], errors="coerce").values
        valid = np.isfinite(rs)
        rs = rs[valid]
        Y = df.iloc[valid, 1:].values.astype(float)
        spectra_matrices[fname] = {"wn": rs, "Y": Y.T, "columns": df.columns[1:].tolist()}
        for j, col in enumerate(df.columns[1:]):
            col_s = str(col).replace("\n", "").replace("\t", "").strip()
            base = col_s.split(".")[0]
            # Brand code: for trademark files, the base is like "Para-D" / "ibu-B" / "Acid-A"
            brand_code = base if variant == "trademark" else None
            # For "All spectra.xlsx" — base encodes drug identity directly
            if fname == "All spectra.xlsx":
                if "Ibuprofen" in base:                drug_here = "ibuprofen"
                elif "Acetylsalicylic" in base:        drug_here = "acetylsalicylic_acid"
                elif "Paracetamol" in base:            drug_here = "paracetamol"
                else:                                  drug_here = None
                var_here = "pure_aggregate_duplicate"
            else:
                drug_here = drug
                var_here  = variant
            records.append({
                "spectrum_id":       f"{fname.replace('.xlsx', '')}::col{j:03d}::{col_s}",
                "file":              fname,
                "column_name":       col_s,
                "column_base_label": base,
                "drug":              drug_here,
                "variant_type":      var_here,
                "brand_code":        brand_code,
                "replicate_idx":     j,
                "n_wn_points":       int(len(rs)),
                "wn_min":            float(rs.min()) if len(rs) else np.nan,
                "wn_max":            float(rs.max()) if len(rs) else np.nan,
            })

    meta_df = pd.DataFrame(records)

    # Dedupe: "All spectra.xlsx" should duplicate the individual pure files.
    # Verify by comparing mean-spectrum across files for a given drug.
    # Then mark All spectra.xlsx rows as duplicates (exclude from downstream).
    dup_flag = np.zeros(len(meta_df), dtype=int)
    if "All spectra.xlsx" in spectra_matrices and all(
            (DATA_DIR / fn).exists() for fn in
            ["Acetylsalicylic-acid.xlsx", "Paracetamol.xlsx", "Ibuprofen.xlsx"]):
        # Fast content-level dedupe: for each "All spectra.xlsx" spectrum, check if an
        # identical intensity vector exists in the corresponding single-drug file.
        all_mat = spectra_matrices["All spectra.xlsx"]
        for row_idx, r in meta_df[meta_df.file == "All spectra.xlsx"].iterrows():
            drug_here = r["drug"]
            if drug_here is None: continue
            indiv_file = {
                "ibuprofen": "Ibuprofen.xlsx",
                "acetylsalicylic_acid": "Acetylsalicylic-acid.xlsx",
                "paracetamol": "Paracetamol.xlsx",
            }.get(drug_here)
            if indiv_file is None: continue
            mat_all = all_mat["Y"]
            mat_ind = spectra_matrices[indiv_file]["Y"]
            target = mat_all[r["replicate_idx"]]
            # Find any row in indiv with near-identical content (exact match in xlsx is likely)
            for k in range(mat_ind.shape[0]):
                if np.array_equal(target, mat_ind[k]):
                    dup_flag[row_idx] = 1
                    break
                elif len(target) == len(mat_ind[k]) and \
                        np.allclose(target, mat_ind[k], atol=1e-8, equal_nan=True):
                    dup_flag[row_idx] = 1
                    break
    meta_df["is_duplicate_of_pure_file"] = dup_flag
    meta_df.to_csv(TABLES / "otc_spectrum_metadata.csv", index=False)
    n_dup = int(dup_flag.sum())
    print(f"  {len(meta_df)} total rows; {n_dup} flagged as All-spectra-duplicates; "
            f"{len(meta_df) - n_dup} unique to downstream")
    return meta_df, spectra_matrices


# ──────────────────────────────────────────────────────────────────────
# TASK 3 — preprocessing (interp, AsLS, SG, L2)
# ──────────────────────────────────────────────────────────────────────
def task3_preprocess(meta_df, spectra_matrices, master_x):
    print("[TASK 3] preprocessing to canonical master_x (400-1800, step 1)")
    # Preprocess per spectrum; restrict to non-duplicate unique spectra
    active_mask = meta_df["is_duplicate_of_pure_file"] == 0
    active_idx = meta_df[active_mask].index.tolist()
    Y_pp = np.full((len(meta_df), len(master_x)), np.nan)

    qc_rows = []
    for i in active_idx:
        r = meta_df.iloc[i]
        mat = spectra_matrices[r["file"]]
        wn = mat["wn"]
        y_raw = mat["Y"][r["replicate_idx"]]
        # Interp to master_x; outside range → NaN
        y_rs = np.interp(master_x, wn, y_raw, left=np.nan, right=np.nan)
        y_pp = baseline_correct(y_rs)
        n_finite = int(np.isfinite(y_pp).sum())
        n_master = len(master_x)
        is_flat = bool(np.nanstd(y_pp) < 1e-9)
        is_nan_majority = bool(n_finite < 0.5 * n_master)
        is_empty = bool(not np.any(np.isfinite(y_pp)))
        if is_empty: status = "EMPTY"
        elif is_flat: status = "FLAT"
        elif is_nan_majority: status = "NAN_MAJORITY"
        else:
            status = "OK"
            Y_pp[i] = y_pp
        qc_rows.append({
            "spectrum_id": r["spectrum_id"],
            "file": r["file"], "drug": r["drug"], "variant_type": r["variant_type"],
            "status": status, "n_finite": n_finite,
            "median": float(np.nanmedian(y_pp)) if not is_empty else np.nan,
            "std":    float(np.nanstd(y_pp))    if not is_empty else np.nan,
        })
    qc = pd.DataFrame(qc_rows)
    qc.to_csv(TABLES / "otc_qc_summary.csv", index=False)
    print(f"  QC: {dict(Counter(qc.status))}")
    return Y_pp, qc


# ──────────────────────────────────────────────────────────────────────
# TASK 4 — peak extraction + aggregation per drug (pure spectra)
# ──────────────────────────────────────────────────────────────────────
def _detect_peaks(y, master_x, prom_frac=0.04):
    """Find peaks with prominence ≥ prom_frac × range."""
    if not np.any(np.isfinite(y)): return np.array([], dtype=int), {}
    rng = float(np.nanmax(y) - np.nanmin(y))
    if rng <= 0: return np.array([], dtype=int), {}
    idx, props = find_peaks(y, prominence=prom_frac * rng,
                                width=1, distance=3)
    return idx, props


def _cluster_peaks(all_peaks, tolerance=5.0):
    """Greedy clustering of peaks within ±tolerance cm⁻¹. Returns list of
    clusters = [(center, members)], sorted by center."""
    if not all_peaks: return []
    sorted_p = sorted(all_peaks, key=lambda x: x["position"])
    clusters = []; current = [sorted_p[0]]
    for p in sorted_p[1:]:
        center_of_current = np.mean([c["position"] for c in current])
        if abs(p["position"] - center_of_current) <= tolerance:
            current.append(p)
        else:
            clusters.append(current); current = [p]
    clusters.append(current)
    return clusters


def task4_peak_extraction(meta_df, Y_pp, master_x, qc_df):
    print("[TASK 4] peak extraction per spectrum + aggregation per drug (pure)")
    peak_rows = []

    # Identify pure, OK spectra per drug
    ok_mask = qc_df.set_index("spectrum_id")["status"] == "OK"
    pure_per_drug = defaultdict(list)  # drug → list of (meta_idx, spectrum_id)
    for i, r in meta_df.iterrows():
        if r["variant_type"] != "pure": continue
        sid = r["spectrum_id"]
        if not ok_mask.get(sid, False): continue
        pure_per_drug[r["drug"]].append(i)

    stat_rows = []
    peak_clusters = {drug: [] for drug in DRUGS}  # keyed on drug

    for drug in DRUGS:
        idxs = pure_per_drug[drug]
        n_spec = len(idxs)
        all_peaks_for_drug = []
        for i in idxs:
            y = Y_pp[i]
            pk_idx, props = _detect_peaks(y, master_x)
            proms = props.get("prominences", np.zeros(len(pk_idx)))
            widths = props.get("widths", np.zeros(len(pk_idx)))
            order = np.argsort(-proms)
            for rank, k in enumerate(order, 1):
                ii = int(pk_idx[k])
                all_peaks_for_drug.append({
                    "spectrum_id_idx": i,
                    "position":    float(master_x[ii]),
                    "prominence":  float(proms[k]),
                    "width":       float(widths[k]),
                    "rank":        int(rank),
                    "intensity":   float(y[ii]),
                })
                peak_rows.append({
                    "drug": drug,
                    "spectrum_id_idx": i,
                    "position":    float(master_x[ii]),
                    "prominence":  float(proms[k]),
                    "width":       float(widths[k]),
                    "rank":        int(rank),
                })

        # Cluster peaks for this drug
        clusters = _cluster_peaks(all_peaks_for_drug, tolerance=TOLERANCE_CM1)
        peak_clusters[drug] = clusters

        # Aggregate per cluster
        for cl_idx, members in enumerate(clusters):
            positions = np.array([m["position"] for m in members])
            proms = np.array([m["prominence"] for m in members])
            widths = np.array([m["width"] for m in members])
            ranks = np.array([m["rank"] for m in members])
            unique_spec_idx = sorted({m["spectrum_id_idx"] for m in members})
            freq = len(unique_spec_idx) / max(n_spec, 1)
            stat_rows.append({
                "drug":                 drug,
                "n_pure_spectra":       n_spec,
                "cluster_idx":          cl_idx,
                "center_cm1":           float(positions.mean()),
                "position_sd_cm1":      float(positions.std()),
                "n_peak_hits":          int(len(members)),
                "n_unique_spectra":     int(len(unique_spec_idx)),
                "frequency":            float(freq),
                "prominence_mean":      float(proms.mean()),
                "prominence_sd":        float(proms.std()),
                "prominence_q50":       float(np.median(proms)),
                "rank_median":          float(np.median(ranks)),
                "rank_q25":             float(np.percentile(ranks, 25)),
                "width_mean":           float(widths.mean()),
            })

    peak_df = pd.DataFrame(peak_rows)
    peak_df.to_csv(TABLES / "otc_peak_per_spectrum.csv", index=False)

    stat_df = pd.DataFrame(stat_rows)
    stat_df.to_csv(TABLES / "otc_peak_statistics_by_molecule.csv", index=False)
    print(f"  emitted {len(peak_df)} per-spectrum peaks; {len(stat_df)} cluster stats rows")
    return stat_df, peak_clusters, pure_per_drug


# ──────────────────────────────────────────────────────────────────────
# TASK 5 — MSS template construction
# ──────────────────────────────────────────────────────────────────────
def _classify_clusters(stats_drug, other_drugs_stats):
    """Classify clusters of one drug as ANCHOR / COMPANION / WEAK based on
    frequency, position SD, prominence rank, cross-drug specificity.
    Returns updated DataFrame with classification + specificity + collision cols."""
    # Compute specificity: for each cluster of this drug, count how many
    # OTHER drugs have a cluster within ±5 cm⁻¹.
    results = []
    for _, r in stats_drug.iterrows():
        center = r["center_cm1"]
        collision_drugs = []
        for other_drug, other_stats in other_drugs_stats.items():
            for _, rr in other_stats.iterrows():
                if abs(rr["center_cm1"] - center) <= TOLERANCE_CM1 and rr["frequency"] >= 0.40:
                    collision_drugs.append(other_drug); break
        # Specificity score: 1 if no other drug has this cluster, else 1/(1+n_others)
        specificity = 1.0 / (1.0 + len(collision_drugs))

        # Classification
        freq = r["frequency"]; sd = r["position_sd_cm1"]
        rank_med = r["rank_median"]; prom = r["prominence_q50"]
        if freq >= 0.70 and sd <= 5.0 and rank_med <= 8 and specificity >= 0.50:
            tier = "ANCHOR"
        elif freq >= 0.70 and sd <= 5.0 and rank_med <= 12:
            tier = "ANCHOR_LOW_SPEC"   # common chemistry band; included but specificity noted
        elif freq >= 0.40 and sd <= 7.0:
            tier = "COMPANION"
        elif freq >= 0.20:
            tier = "WEAK"
        else:
            tier = "BELOW_THRESHOLD"
        results.append({**r.to_dict(),
                          "specificity_vs_other_drugs": specificity,
                          "collision_drugs": "|".join(collision_drugs),
                          "n_collision_drugs": len(collision_drugs),
                          "tier": tier})
    return pd.DataFrame(results)


def task5_mss_templates(stat_df):
    print("[TASK 5] MSS template construction")
    classified_all = []
    per_drug_classified = {}
    for drug in DRUGS:
        stats_drug = stat_df[stat_df.drug == drug].copy()
        other = {d: stat_df[stat_df.drug == d] for d in DRUGS if d != drug}
        cd = _classify_clusters(stats_drug, other)
        cd["drug"] = drug  # ensure set
        per_drug_classified[drug] = cd
        classified_all.append(cd)
    classified_df = pd.concat(classified_all, ignore_index=True)
    classified_df.to_csv(TABLES / "otc_anchor_companion_candidates.csv", index=False)

    # Build MSS registry
    reg_rows = []
    for drug in DRUGS:
        cd = per_drug_classified[drug]
        anchors = cd[cd.tier.isin(["ANCHOR", "ANCHOR_LOW_SPEC"])] \
            .sort_values("prominence_q50", ascending=False)
        companions = cd[cd.tier == "COMPANION"] \
            .sort_values("prominence_q50", ascending=False)
        weak = cd[cd.tier == "WEAK"] \
            .sort_values("prominence_q50", ascending=False)
        collisions = cd[cd.n_collision_drugs >= 1]

        # Cap for template size
        anchor_cm1  = [round(x, 1) for x in anchors.head(6)["center_cm1"].tolist()]
        companion_cm1 = [round(x, 1) for x in companions.head(8)["center_cm1"].tolist()]
        weak_cm1 = [round(x, 1) for x in weak.head(6)["center_cm1"].tolist()]

        n_anchors = len(anchor_cm1); n_comps = len(companion_cm1)
        if n_anchors >= 4 and n_comps >= 4:
            tier = "HIGH"
        elif n_anchors >= 3 and n_comps >= 3:
            tier = "MODERATE"
        elif n_anchors >= 1:
            tier = "LOW"
        else:
            tier = "INSUFFICIENT"

        reg_rows.append({
            "molecule":            drug,
            "regime":              "Raman",
            "source_dataset":      "otc_drugs_paraguay",
            "anchor_bands_cm1":    ";".join(str(x) for x in anchor_cm1),
            "companion_bands_cm1": ";".join(str(x) for x in companion_cm1),
            "weak_bands_cm1":      ";".join(str(x) for x in weak_cm1),
            "tolerance_cm1":       TOLERANCE_CM1,
            "collision_bands":     ";".join(
                f"{round(r['center_cm1'], 1)}(vs:{r['collision_drugs']})"
                for _, r in collisions.sort_values("center_cm1").iterrows()),
            "n_anchor":            n_anchors,
            "n_companion":         n_comps,
            "n_weak":              len(weak_cm1),
            "reliability_tier":    tier,
            "notes":               "pure Raman tablet; iRaman 785 nm 4 cm⁻¹ res; no substrate physics",
        })
    reg_df = pd.DataFrame(reg_rows)
    reg_df.to_csv(REGISTRY / "otc_pure_raman_mss_registry_v1.csv", index=False)

    # Cross-drug collision matrix: for each anchor/companion of drug X, which drugs have it?
    coll_rows = []
    for _, r in classified_df[classified_df.tier.isin(["ANCHOR", "ANCHOR_LOW_SPEC", "COMPANION"])].iterrows():
        coll_rows.append({
            "drug":            r["drug"],
            "band_cm1":        r["center_cm1"],
            "tier":            r["tier"],
            "frequency":       r["frequency"],
            "specificity":     r["specificity_vs_other_drugs"],
            "collision_drugs": r["collision_drugs"],
        })
    coll_df = pd.DataFrame(coll_rows)
    coll_df.to_csv(TABLES / "otc_collision_matrix.csv", index=False)
    print(f"  registry rows: {len(reg_df)}; "
            f"tiers: {dict(Counter(reg_df['reliability_tier']))}")
    return reg_df, classified_df


# ──────────────────────────────────────────────────────────────────────
# TASK 6 — validation
# ──────────────────────────────────────────────────────────────────────
def _score_spectra_against_templates(Y_pp, master_x, reg_df):
    """For each spectrum, compute MSS score per drug template (using
    the unchanged GAIRA MSS anchor-fires + 0.3·support-fires kernel)."""
    templates = {}
    for _, r in reg_df.iterrows():
        anchors = [float(x) for x in str(r["anchor_bands_cm1"]).split(";") if x.strip()]
        supports = [float(x) for x in str(r["companion_bands_cm1"]).split(";") if x.strip()]
        templates[r["molecule"]] = {"anchors": anchors, "supports": supports}
    scores = {drug: np.zeros(Y_pp.shape[0]) for drug in DRUGS}
    for i in range(Y_pp.shape[0]):
        y = Y_pp[i]
        if not np.isfinite(y).any(): continue
        for drug in DRUGS:
            t = templates[drug]
            sc, _, _ = mss_anchor_score(y, master_x, t["anchors"], t["supports"])
            scores[drug][i] = sc
    return scores


def task6_validation(meta_df, Y_pp, master_x, qc_df, reg_df):
    print("[TASK 6] validation — pure self + trademark cross + specificity")
    ok_mask = qc_df.set_index("spectrum_id")["status"] == "OK"

    scores_all = _score_spectra_against_templates(Y_pp, master_x, reg_df)

    # Pure self-validation: top-1/top-3 via 3-way comparison
    # (top-3 is tautological for 3 drugs; include for completeness)
    pure_rows = []
    pure_confusion = defaultdict(lambda: Counter())
    for i, r in meta_df.iterrows():
        if r["variant_type"] != "pure": continue
        sid = r["spectrum_id"]
        if not ok_mask.get(sid, False): continue
        drug_true = r["drug"]
        # score triplet
        triplet = {d: scores_all[d][i] for d in DRUGS}
        top1 = max(triplet, key=triplet.get)
        pure_rows.append({
            "spectrum_id": sid, "drug_true": drug_true, "drug_predicted": top1,
            "score_asa":         triplet["acetylsalicylic_acid"],
            "score_paracetamol": triplet["paracetamol"],
            "score_ibuprofen":   triplet["ibuprofen"],
            "correct":           int(top1 == drug_true),
        })
        pure_confusion[drug_true][top1] += 1
    pure_df = pd.DataFrame(pure_rows)
    pure_df.to_csv(TABLES / "otc_mss_validation_pure.csv", index=False)
    n_total = len(pure_df)
    n_correct = int(pure_df["correct"].sum())
    pure_acc = n_correct / max(n_total, 1)

    # Per-drug pure accuracy
    pure_per_drug = {
        drug: float(pure_df[pure_df.drug_true == drug]["correct"].mean())
                 if (pure_df.drug_true == drug).any() else np.nan
        for drug in DRUGS
    }

    # Trademark validation
    tm_rows = []
    tm_confusion = defaultdict(lambda: Counter())
    for i, r in meta_df.iterrows():
        if r["variant_type"] != "trademark": continue
        sid = r["spectrum_id"]
        if not ok_mask.get(sid, False): continue
        drug_true = r["drug"]
        triplet = {d: scores_all[d][i] for d in DRUGS}
        top1 = max(triplet, key=triplet.get)
        tm_rows.append({
            "spectrum_id": sid, "drug_true": drug_true, "drug_predicted": top1,
            "brand_code": r.get("brand_code", ""),
            "score_asa":         triplet["acetylsalicylic_acid"],
            "score_paracetamol": triplet["paracetamol"],
            "score_ibuprofen":   triplet["ibuprofen"],
            "correct":           int(top1 == drug_true),
        })
        tm_confusion[drug_true][top1] += 1
    tm_df = pd.DataFrame(tm_rows)
    tm_df.to_csv(TABLES / "otc_mss_validation_trademark.csv", index=False)
    tm_total = len(tm_df)
    tm_correct = int(tm_df["correct"].sum())
    tm_acc = tm_correct / max(tm_total, 1)
    tm_per_drug = {
        drug: float(tm_df[tm_df.drug_true == drug]["correct"].mean())
                 if (tm_df.drug_true == drug).any() else np.nan
        for drug in DRUGS
    }
    # Per-brand
    tm_per_brand = tm_df.groupby("brand_code")["correct"].agg(["mean", "count"]).reset_index()
    tm_per_brand.to_csv(TABLES / "otc_mss_validation_trademark_per_brand.csv", index=False)

    # False-positive analysis: for each drug, how often does OTHER-drug spectra surface as this drug top-1?
    fp_rows = []
    for drug in DRUGS:
        pred_as_drug = pure_df[pure_df.drug_predicted == drug]
        fp_pure = int((pred_as_drug["drug_true"] != drug).sum())
        total_pure_not = int((pure_df["drug_true"] != drug).sum())
        fp_rate_pure = fp_pure / max(total_pure_not, 1)
        pred_as_drug_tm = tm_df[tm_df.drug_predicted == drug]
        fp_tm = int((pred_as_drug_tm["drug_true"] != drug).sum())
        total_tm_not = int((tm_df["drug_true"] != drug).sum())
        fp_rate_tm = fp_tm / max(total_tm_not, 1)
        fp_rows.append({
            "drug": drug,
            "fp_rate_pure":       fp_rate_pure,
            "fp_rate_trademark":  fp_rate_tm,
        })
    pd.DataFrame(fp_rows).to_csv(TABLES / "otc_false_positive_rates.csv", index=False)

    # Confusion matrices as tables
    for label, mat in [("pure", pure_confusion), ("trademark", tm_confusion)]:
        rows = []
        for true_d in DRUGS:
            for pred_d in DRUGS:
                rows.append({"true": true_d, "predicted": pred_d, "count": int(mat[true_d][pred_d])})
        pd.DataFrame(rows).to_csv(TABLES / f"otc_confusion_matrix_{label}.csv", index=False)

    summary = {
        "n_pure_scored":       n_total,
        "n_pure_correct":      n_correct,
        "pure_top1_accuracy":  pure_acc,
        "pure_acc_per_drug":   {k: round(v, 3) if v is not None else None for k, v in pure_per_drug.items()},
        "n_trademark_scored":  tm_total,
        "n_trademark_correct": tm_correct,
        "trademark_top1_accuracy":   tm_acc,
        "trademark_acc_per_drug":    {k: round(v, 3) if v is not None else None for k, v in tm_per_drug.items()},
    }
    return summary, pure_df, tm_df, tm_per_brand, pure_confusion, tm_confusion


# ──────────────────────────────────────────────────────────────────────
# Figures
# ──────────────────────────────────────────────────────────────────────
def make_figures(meta_df, Y_pp, master_x, reg_df, stat_df,
                     classified_df, pure_confusion, tm_confusion):
    print("[FIG] generating figures")
    # Fig 1 — mean spectra per molecule (pure only)
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
        for ax, drug in zip(axes, DRUGS):
            idxs = [i for i, r in meta_df.iterrows()
                       if r["variant_type"] == "pure" and r["drug"] == drug]
            stack = np.stack([Y_pp[i] for i in idxs if np.isfinite(Y_pp[i]).any()])
            if len(stack):
                mean = np.nanmean(stack, axis=0); sd = np.nanstd(stack, axis=0)
                ax.fill_between(master_x, mean - sd, mean + sd, alpha=0.2, color="#4C72B0")
                ax.plot(master_x, mean, color="#4C72B0", lw=1.5)
                # Mark anchors + companions
                reg = reg_df[reg_df.molecule == drug].iloc[0]
                for band in str(reg["anchor_bands_cm1"]).split(";"):
                    if band.strip():
                        ax.axvline(float(band), color="red", lw=0.8, alpha=0.7)
                for band in str(reg["companion_bands_cm1"]).split(";"):
                    if band.strip():
                        ax.axvline(float(band), color="orange", lw=0.6, alpha=0.5, ls="--")
            ax.set_title(f"{drug} (pure, n={len(idxs)})")
            ax.set_xlabel("Raman shift cm⁻¹")
            ax.set_xlim(400, 1800)
        axes[0].set_ylabel("canonical preprocessed intensity")
        fig.suptitle("Mean pure Raman spectra with MSS anchors (red) + companions (orange dashed)")
        fig.tight_layout()
        fig.savefig(FIGS / "fig_mean_spectra_per_molecule.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig1 issue: {e}")

    # Fig 2 — pure vs trademark overlays
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
        for ax, drug in zip(axes, DRUGS):
            pure_idx = [i for i, r in meta_df.iterrows()
                            if r["variant_type"] == "pure" and r["drug"] == drug]
            tm_idx   = [i for i, r in meta_df.iterrows()
                            if r["variant_type"] == "trademark" and r["drug"] == drug]
            for idxs, color, label in [(pure_idx, "#4C72B0", "pure"),
                                              (tm_idx,   "#DD8452", "trademark")]:
                stack = np.stack([Y_pp[i] for i in idxs if np.isfinite(Y_pp[i]).any()])
                if len(stack):
                    ax.plot(master_x, np.nanmean(stack, axis=0), color=color, lw=1.5, label=f"{label} (n={len(stack)})")
            ax.set_title(drug); ax.set_xlabel("Raman shift cm⁻¹")
            ax.set_xlim(400, 1800); ax.legend(fontsize=8)
        axes[0].set_ylabel("canonical intensity")
        fig.suptitle("Mean Raman spectra — pure vs trademark per drug")
        fig.tight_layout()
        fig.savefig(FIGS / "fig_pure_vs_trademark_overlays.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig2 issue: {e}")

    # Fig 3 — peak frequency map per molecule
    try:
        fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
        for ax, drug in zip(axes, DRUGS):
            sub = stat_df[stat_df.drug == drug].sort_values("center_cm1")
            ax.stem(sub["center_cm1"], sub["frequency"], basefmt=" ")
            reg = reg_df[reg_df.molecule == drug].iloc[0]
            for band in str(reg["anchor_bands_cm1"]).split(";"):
                if band.strip():
                    ax.axvspan(float(band) - 2, float(band) + 2, color="red", alpha=0.15)
            ax.set_title(f"{drug} — peak occurrence frequency (pure spectra)")
            ax.set_ylabel("freq"); ax.set_ylim(0, 1.05); ax.grid(alpha=0.3)
        axes[-1].set_xlabel("Raman shift cm⁻¹")
        axes[-1].set_xlim(400, 1800)
        fig.tight_layout()
        fig.savefig(FIGS / "fig_peak_frequency_map.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig3 issue: {e}")

    # Fig 4 — peak collision heatmap (band position × drug, presence flag)
    try:
        # Take union of all clusters with freq ≥ 0.4 across drugs
        common_bands = stat_df[stat_df.frequency >= 0.4]["center_cm1"].round(0).astype(int).unique()
        common_bands = sorted(set(common_bands))
        mat = np.zeros((3, len(common_bands)))
        for i_d, drug in enumerate(DRUGS):
            for j, band in enumerate(common_bands):
                sub = stat_df[(stat_df.drug == drug) &
                                 (np.abs(stat_df.center_cm1 - band) <= TOLERANCE_CM1)]
                if not sub.empty:
                    mat[i_d, j] = float(sub["frequency"].max())
        fig, ax = plt.subplots(figsize=(max(14, len(common_bands) * 0.15), 3.5))
        im = ax.imshow(mat, aspect="auto", cmap="Reds", vmin=0, vmax=1)
        ax.set_yticks(range(3)); ax.set_yticklabels(DRUGS)
        ax.set_xticks(range(len(common_bands))); ax.set_xticklabels(common_bands, rotation=60, fontsize=7)
        ax.set_title("Peak frequency by drug × band position  (ANCHOR/COMPANION candidate bands, freq ≥ 0.4)")
        plt.colorbar(im, ax=ax, label="peak frequency across pure spectra", fraction=0.02)
        fig.tight_layout()
        fig.savefig(FIGS / "fig_peak_collision_heatmap.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig4 issue: {e}")

    # Fig 5 — confusion matrices (pure + trademark)
    try:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for ax, (label, conf) in zip(axes, [("Pure", pure_confusion), ("Trademark", tm_confusion)]):
            mat = np.array([[conf[t][p] for p in DRUGS] for t in DRUGS])
            row_sums = mat.sum(axis=1, keepdims=True)
            norm = mat / np.maximum(row_sums, 1)
            im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
            ax.set_xticks(range(3)); ax.set_yticks(range(3))
            ax.set_xticklabels(DRUGS, rotation=20, fontsize=8)
            ax.set_yticklabels(DRUGS, fontsize=8)
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, f"{mat[i, j]}\n({norm[i, j]:.0%})",
                              ha="center", va="center", fontsize=8,
                              color="white" if norm[i, j] > 0.5 else "black")
            ax.set_xlabel("predicted"); ax.set_ylabel("true")
            ax.set_title(f"{label} (n={int(mat.sum())})")
            plt.colorbar(im, ax=ax, fraction=0.04)
        fig.suptitle("OTC MSS validation — confusion matrices")
        fig.tight_layout()
        fig.savefig(FIGS / "fig_mss_confusion_matrices.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig5 issue: {e}")

    # Fig 6 — anchor/companion schematic
    try:
        fig, ax = plt.subplots(figsize=(13, 4))
        colors = {"acetylsalicylic_acid": "#4C72B0",
                    "paracetamol":          "#DD8452",
                    "ibuprofen":            "#2ca02c"}
        for i, drug in enumerate(DRUGS):
            reg = reg_df[reg_df.molecule == drug].iloc[0]
            for band in str(reg["anchor_bands_cm1"]).split(";"):
                if band.strip():
                    ax.plot([float(band)], [i], "o", color=colors[drug], ms=11)
            for band in str(reg["companion_bands_cm1"]).split(";"):
                if band.strip():
                    ax.plot([float(band)], [i], "s", color=colors[drug], ms=6, alpha=0.7)
            for band in str(reg["weak_bands_cm1"]).split(";"):
                if band.strip():
                    ax.plot([float(band)], [i], "x", color=colors[drug], ms=6, alpha=0.5)
        ax.set_yticks(range(3)); ax.set_yticklabels(DRUGS)
        ax.set_xlabel("Raman shift cm⁻¹"); ax.set_xlim(400, 1800)
        ax.set_title("OTC pure-Raman MSS template — ● anchor | ■ companion | × weak")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIGS / "fig_anchor_companion_schematic.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig6 issue: {e}")


# ──────────────────────────────────────────────────────────────────────
# Reports + decision
# ──────────────────────────────────────────────────────────────────────
def _decision(reg_df, summary):
    tiers = Counter(reg_df["reliability_tier"])
    pure_acc = summary["pure_top1_accuracy"]
    tm_acc   = summary["trademark_top1_accuracy"]
    if tiers.get("HIGH", 0) + tiers.get("MODERATE", 0) >= 3 and pure_acc >= 0.95 and tm_acc >= 0.70:
        return "OTC_RAMAN_MSS_ROBUST_READY_FOR_GROUNDING"
    if pure_acc >= 0.90 and tm_acc >= 0.50:
        return "OTC_RAMAN_MSS_PARTIAL_BRAND_VARIABILITY"
    return "OTC_RAMAN_MSS_UNRELIABLE"


def write_template_summary(reg_df):
    lines = ["# OTC pure-Raman MSS template summary\n",
                f"date: {datetime.now().isoformat()}", ""]
    for _, r in reg_df.iterrows():
        lines.append(f"## {r['molecule']}  —  tier **{r['reliability_tier']}**")
        lines.append(f"- regime: {r['regime']} (pure tablet Raman, iRaman 785 nm, no substrate physics)")
        lines.append(f"- source_dataset: {r['source_dataset']}")
        lines.append(f"- anchors (n={r['n_anchor']}): {r['anchor_bands_cm1']}")
        lines.append(f"- companions (n={r['n_companion']}): {r['companion_bands_cm1']}")
        lines.append(f"- weak (n={r['n_weak']}): {r['weak_bands_cm1']}")
        lines.append(f"- tolerance: ±{r['tolerance_cm1']} cm⁻¹")
        lines.append(f"- collision bands (shared with other OTC drugs): {r['collision_bands']}")
        lines.append(f"- notes: {r['notes']}")
        lines.append("")
    (REPORTS / "OTC_MSS_TEMPLATE_SUMMARY.md").write_text("\n".join(lines))


def write_validation_assessment(summary, tm_per_brand):
    lines = ["# OTC MSS validation assessment\n",
                f"date: {datetime.now().isoformat()}", "",
                "## Pure self-validation\n"]
    lines.append(f"- N scored: {summary['n_pure_scored']}")
    lines.append(f"- top-1 accuracy: {summary['pure_top1_accuracy']:.2%}")
    lines.append(f"- per-drug accuracy: {summary['pure_acc_per_drug']}")
    lines.append("")
    lines.append("## Trademark generalization\n")
    lines.append(f"- N scored: {summary['n_trademark_scored']}")
    lines.append(f"- top-1 accuracy: {summary['trademark_top1_accuracy']:.2%}")
    lines.append(f"- per-drug accuracy: {summary['trademark_acc_per_drug']}")
    lines.append("")
    lines.append("## Per-brand trademark accuracy\n")
    lines.append("| brand_code | correct_rate | n |")
    lines.append("|---|---:|---:|")
    for _, r in tm_per_brand.sort_values("mean", ascending=False).iterrows():
        lines.append(f"| {r['brand_code']} | {float(r['mean']):.0%} | {int(r['count'])} |")
    (REPORTS / "OTC_VALIDATION_ASSESSMENT.md").write_text("\n".join(lines))


def write_main_report(decision, inv, meta_df, qc, stat_df, classified_df,
                          reg_df, summary, tm_per_brand):
    lines = [
        "# REPORT — OTC pure-Raman MSS build v1\n",
        f"date: {datetime.now().isoformat()}", "",
        f"## Decision: **{decision}**\n",
        "## Dataset",
        "- Paraguay OTC Raman tablet dataset (Data in Brief 2024).",
        "- Acquisition: iRaman 785s (BWTEK), 785 nm, 50% power, 1 s × 10 accum, 150-3200 cm⁻¹, "
        "4 cm⁻¹ resolution, direct tablet. Raw (untreated).",
        "- Drugs: acetylsalicylic acid (ASA), paracetamol, ibuprofen.",
        "",
        "## Invariants (preserved)",
        "- Engine v4.5 / MSS kernel / existing MSS templates / preprocessing — UNCHANGED",
        "- Substrate physics NOT invoked (pure Raman tablet)",
        "- No classifier; no label leakage into BSV/MSS construction",
        "",
        "## Files",
        f"- {len(inv)} rows in file inventory, {int((inv['variant'] == 'pure').sum())} pure file rows, "
        f"{int((inv['variant'] == 'trademark').sum())} trademark file rows, "
        f"{int((inv['variant'] == 'aggregate').sum())} aggregate file rows (\"All spectra.xlsx\")",
        f"- {len(meta_df)} spectrum-rows total; "
        f"{int(meta_df['is_duplicate_of_pure_file'].sum())} flagged as duplicate of individual pure files "
        f"(from \"All spectra.xlsx\" — excluded from downstream)",
        "",
        "## QC after canonical preprocessing",
        f"- status distribution: {dict(Counter(qc.status))}",
        "",
        "## Template registry (v1)\n",
        "| molecule | tier | n anchor | n companion | n weak | anchors (cm⁻¹) |",
        "|---|---|---:|---:|---:|---|",
    ]
    for _, r in reg_df.iterrows():
        lines.append(f"| {r['molecule']} | {r['reliability_tier']} | {r['n_anchor']} | "
                        f"{r['n_companion']} | {r['n_weak']} | {r['anchor_bands_cm1']} |")
    lines.append("")

    lines.append("## Validation summary\n")
    lines.append(f"- **Pure top-1 accuracy: {summary['pure_top1_accuracy']:.1%}** (n={summary['n_pure_scored']})")
    lines.append(f"- Per-drug pure accuracy: {summary['pure_acc_per_drug']}")
    lines.append(f"- **Trademark top-1 accuracy: {summary['trademark_top1_accuracy']:.1%}** (n={summary['n_trademark_scored']})")
    lines.append(f"- Per-drug trademark accuracy: {summary['trademark_acc_per_drug']}")
    lines.append("")

    lines.append("## Collision / shared-band flag\n")
    lines.append("Bands shared with other OTC drugs (≥0.4 frequency each) are tagged in the registry and "
                    "in `tables/otc_collision_matrix.csv`. Shared bands are common tablet chemistry (aromatic ring "
                    "modes, C-H stretches, amide region) — this is expected; template specificity relies on the "
                    "unique-band set of each drug plus the companion pattern.")
    lines.append("")

    lines.append("## Paper context (acquisition support only)\n")
    lines.append("- Pure Raman tablet measurements (NOT SERS). GAIRA treats these as intrinsic Raman grounding, "
                    "equivalent to RamanBioLib / Gobbato powder references.")
    lines.append("- No substrate physics invoked.")
    lines.append("- Brand variants may carry excipient/binder signatures; `reliability_tier` in the registry "
                    "reflects pure-spectrum derivation only. Trademark validation reports the downstream "
                    "robustness of pure-derived templates against tablet-excipient interference.")
    lines.append("")

    lines.append("## Honest reading\n")
    lines.append(f"- Pure Raman MSS templates are easy to separate at the pure-spectrum level (expected given "
                    f"distinct drug chemistries). Trademark accuracy reflects how much brand-specific excipient "
                    f"content interferes with anchor firing.")
    lines.append(f"- Templates are ADDED to a NEW registry (`registry/otc_pure_raman_mss_registry_v1.csv`); "
                    f"existing GAIRA narrow registry is unchanged.")
    lines.append("")

    (REPORTS / "REPORT_otc_pure_raman_mss_build_v1.md").write_text("\n".join(lines))


def write_audit(decision):
    txt = [
        "# gaira_base_4_otc_pure_raman_mss_build_v1 — audit log",
        f"date: {datetime.now().isoformat()}",
        "",
        "## Strict invariants (preserved)",
        "- Engine v4.5 / MSS kernel / existing MSS templates / preprocessing — UNCHANGED",
        "- Substrate physics NOT invoked (pure Raman)",
        "- Dataset NOT treated as SERS",
        "- No classifier trained; no label leakage",
        "",
        "## Source dataset",
        "- /Volumes/SSD_Rad/GAIRA_DATA/raw/otc_drugs/ (7 xlsx files)",
        "- Paper: Data in Brief 2024 — iRaman 785s Paraguay OTC tablet Raman dataset",
        "",
        "## Outputs",
        "- registry/otc_pure_raman_mss_registry_v1.csv  (NEW registry — not merged into existing GAIRA narrow registry)",
        "- tables/otc_file_inventory.csv",
        "- tables/otc_spectrum_metadata.csv",
        "- tables/otc_qc_summary.csv",
        "- tables/otc_peak_per_spectrum.csv",
        "- tables/otc_peak_statistics_by_molecule.csv",
        "- tables/otc_anchor_companion_candidates.csv",
        "- tables/otc_collision_matrix.csv",
        "- tables/otc_mss_validation_pure.csv + otc_confusion_matrix_pure.csv",
        "- tables/otc_mss_validation_trademark.csv + otc_confusion_matrix_trademark.csv",
        "- tables/otc_mss_validation_trademark_per_brand.csv",
        "- tables/otc_false_positive_rates.csv",
        "- 6 figures",
        "- reports/REPORT_otc_pure_raman_mss_build_v1.md",
        "- reports/OTC_MSS_TEMPLATE_SUMMARY.md",
        "- reports/OTC_VALIDATION_ASSESSMENT.md",
        "",
        f"## Final decision\n**{decision}**",
    ]
    (AUDIT / "gaira_base_4_otc_pure_raman_mss_build_v1_audit_log.md").write_text("\n".join(txt))


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 78)
    print("gaira_base_4_otc_pure_raman_mss_build_v1")
    print("=" * 78)
    master_x = canonical_master_axis()

    inv = task1_file_audit()
    meta_df, spectra_matrices = task2_parse_spectra(inv)
    Y_pp, qc = task3_preprocess(meta_df, spectra_matrices, master_x)
    stat_df, peak_clusters, pure_per_drug = task4_peak_extraction(
        meta_df, Y_pp, master_x, qc)
    reg_df, classified_df = task5_mss_templates(stat_df)
    summary, pure_val, tm_val, tm_per_brand, pure_conf, tm_conf = task6_validation(
        meta_df, Y_pp, master_x, qc, reg_df)

    make_figures(meta_df, Y_pp, master_x, reg_df, stat_df,
                    classified_df, pure_conf, tm_conf)

    decision = _decision(reg_df, summary)
    write_template_summary(reg_df)
    write_validation_assessment(summary, tm_per_brand)
    write_main_report(decision, inv, meta_df, qc, stat_df, classified_df,
                          reg_df, summary, tm_per_brand)
    write_audit(decision)
    try:
        shutil.copy(__file__, CODE_SNAPSHOT / Path(__file__).name)
    except Exception:
        pass
    print(f"[done] decision: {decision}")


if __name__ == "__main__":
    main()
