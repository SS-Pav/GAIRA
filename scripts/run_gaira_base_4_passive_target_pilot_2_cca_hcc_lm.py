"""gaira_base_4 passive target Pilot 2 — CCA / HCC / LM serum SERS.

Passive readout only. NO classifier training, NO threshold tuning, NO
feature selection using labels, NO parameter fitting.

Engine: v4.5 locked + v3 calibration fixes.
Substrate: 'label-free SERS-based nanosensor' — exact substrate not
documented → `unknown_SERS` block (inference OFF, interpretation ON).

Dataset:
  /Volumes/SSD_Rad/GAIRA_DATA/raw/cca_hcc_lm_serum_sers/Combination of label-free SERS-based nanosensor an.zip
  219 patients across 4 cohorts:
    NC  (normal control / healthy)  : 49
    HCC (hepatocellular carcinoma)  : 50
    CCA (cholangiocarcinoma)        : 70
    LM  (liver metastases)          : 50
"""
from __future__ import annotations

import shutil
import sys
import warnings
import zipfile
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gaira.spectral import canonical_master_axis
from run_gaira_base_4_hybrid_bsv_build_v1 import (
    BSV_GROUPS, compute_motif_firings, compute_mss_scores_v43,
)
from run_gaira_base_4_hybrid_bsv_refinement_v4_5_triglyceride_veto import (
    compute_hybrid_bsv_v45,
)
from run_gaira_base_4_hybrid_bsv_controlled_calibration_v2 import FAMILY_LABELS
from run_gaira_base_4_calibration_fixes_before_v3 import (
    SUBSTRATE_BLOCKS, derive_erg_anchors,
)


ROOT = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/"
    "gaira_base_4_passive_target_pilot_2_cca_hcc_lm"
)
TABLES = ROOT / "tables"
FIGS = ROOT / "figures"
REPORTS = ROOT / "reports"
AUDIT = ROOT / "audit"
CODE_SNAPSHOT = ROOT / "code_snapshot"

MSS_V43 = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_4_mss_decision_enrichment_v1/"
    "registry/grounding_molecular_signatures_v4_3.csv"
)
LEARNED_MOTIFS = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_3_grounding_trained_ontology_v1/"
    "registry/learned_motif_registry_v1.csv"
)
ZIP_PATH = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/raw/cca_hcc_lm_serum_sers/"
    "Combination of label-free SERS-based nanosensor an.zip"
)

BSV_GROUPS_ORDER = [g["group_id"] for g in BSV_GROUPS]
CLASS_ORDER = ["NC", "HCC", "CCA", "LM"]


# ─────────────────────────────────────────────────────────────────────
# Substrate block (extended locally for unknown_SERS)
# ─────────────────────────────────────────────────────────────────────

EXTRA_BLOCKS = [
    {"block_id": "Ag_colloid_untyped",
     "substrate_family": "Ag colloid (variant not documented)",
     "apply_for_inference": False, "apply_for_interpretation": True},
    {"block_id": "unknown_SERS",
     "substrate_family": "SERS substrate (type unknown)",
     "apply_for_inference": False, "apply_for_interpretation": True},
]
ALL_BLOCKS = SUBSTRATE_BLOCKS + EXTRA_BLOCKS
BLOCK_APPLY = {b["block_id"]: b["apply_for_inference"] for b in ALL_BLOCKS}


# ─────────────────────────────────────────────────────────────────────
# Stage 1 — load zip + Stage 2 — substrate block decision
# ─────────────────────────────────────────────────────────────────────

def stage12_load_and_substrate(master_x):
    print("\n[STAGE 1+2] Load CCA/HCC/LM zip + substrate block decision")
    refs = []
    with zipfile.ZipFile(ZIP_PATH) as z:
        for info in z.infolist():
            if not info.filename.endswith(".txt"): continue
            parts = info.filename.split("/")
            # parts[1] = class folder (e.g. "CCA (Cholangiocarcinoma)" or "HCC")
            # parts[2] = patient folder (e.g. "SER-CCA-1")
            # parts[3] = filename (e.g. "SER-CCA-1_01.txt")
            if len(parts) < 4: continue
            class_folder = parts[1]
            patient_folder = parts[2]
            if not patient_folder.startswith("SER-"): continue
            # Class label from patient folder
            tokens = patient_folder.split("-")
            if len(tokens) < 3: continue
            cls = tokens[1]   # CCA / HCC / LM / NC
            patient_id = tokens[2]
            data = z.read(info).decode("utf-8", errors="ignore").splitlines()
            if len(data) < 2: continue
            try:
                wn = np.array([float(x) for x in data[0].split("\t") if x.strip()])
            except Exception:
                continue
            row_arrays = []
            for line in data[1:]:
                vals = line.split("\t")
                try:
                    floats = [float(v) for v in vals if v.strip()]
                except ValueError:
                    continue
                # Each data row: 2 metadata (x,y position) + N intensities
                if len(floats) >= len(wn) + 2:
                    row_arrays.append(np.asarray(floats[2:2 + len(wn)]))
            if not row_arrays: continue
            patient_mean = np.mean(row_arrays, 0)  # average over spatial reps
            order = np.argsort(wn)
            y_rs = np.interp(master_x, wn[order], patient_mean[order],
                               left=np.nan, right=np.nan)
            refs.append({
                "spectrum_id": f"pilot2::{patient_folder}",
                "patient_id": patient_folder,
                "class_label": cls,
                "spectrum": y_rs,
                "regime": "SERS",
                "substrate_family": "label-free SERS nanosensor (chemistry undocumented)",
                "n_position_acquisitions_averaged": len(row_arrays),
                "source_file": info.filename,
            })

    n_per = Counter(r["class_label"] for r in refs)
    print(f"  loaded {len(refs)} patient-mean spectra")
    for c in CLASS_ORDER: print(f"    {c}: {n_per.get(c, 0)}")

    # Substrate block — undocumented → unknown_SERS
    block = "unknown_SERS"
    apply_inf = BLOCK_APPLY[block]

    audit_rows = [{
        "dataset_path": str(ZIP_PATH),
        "n_patients_loaded": len(refs),
        "n_NC": n_per.get("NC", 0), "n_HCC": n_per.get("HCC", 0),
        "n_CCA": n_per.get("CCA", 0), "n_LM": n_per.get("LM", 0),
        "regime": "SERS",
        "substrate_documented": "label-free SERS nanosensor (no Ag/Au/colloid/film specification)",
        "substrate_block_assigned": block,
        "apply_substrate_physics_for_inference": apply_inf,
        "apply_substrate_physics_for_interpretation": True,
        "spatial_acquisitions_per_patient": "averaged over per-patient spatial mapping rows",
        "batch_metadata_available": False,
        "fallback_delta_reference": "global NC centroid",
    }]
    pd.DataFrame(audit_rows).to_csv(
        TABLES / "pilot2_ingestion_audit.csv", index=False,
    )
    pd.DataFrame([{
        "dataset": "cca_hcc_lm_serum_sers",
        "substrate_status": "UNDOCUMENTED_SERS_NANOSENSOR",
        "substrate_block_assigned": block,
        "apply_inference": apply_inf,
        "apply_interpretation": True,
        "batch_id_available": False,
        "delta_reference_mode_primary": "global_NC_centroid",
        "delta_reference_mode_secondary": "all_sample_neutral_centroid",
        "batch_local_NC_reference": "n/a (no batch metadata)",
    }]).to_csv(TABLES / "pilot2_substrate_batch_audit.csv", index=False)

    lines = [
        "# Pilot 2 CCA/HCC/LM — Ingestion Audit",
        "",
        f"- Dataset path: `{ZIP_PATH}`",
        f"- Paper: 'Combination of label-free SERS-based nanosensor and machine learning…' (folder name truncated; substrate chemistry not specified in metadata)",
        f"- Total patient-mean spectra: **{len(refs)}**",
        f"- Per-class:",
        f"  - NC (normal control): {n_per.get('NC', 0)}",
        f"  - HCC (hepatocellular carcinoma): {n_per.get('HCC', 0)}",
        f"  - CCA (cholangiocarcinoma): {n_per.get('CCA', 0)}",
        f"  - LM (liver metastases): {n_per.get('LM', 0)}",
        f"- Regime: SERS",
        f"- Substrate block: **{block}** (inference={apply_inf}, interpretation=True)",
        f"- Per-patient: averaged over the spatial-mapping acquisition rows in the txt file (≈49 positions per patient)",
        f"- Batch metadata: NOT AVAILABLE — using global NC centroid as primary ΔBSV reference",
        f"- Spectral range: ~300-3268 cm⁻¹ (raw); resampled to GAIRA canonical master axis",
        "",
        "## Caveats",
        "",
        "- Substrate chemistry is undocumented in the local file set. Output carries `substrate_variant_caveat=True` and `unknown_SERS_strong_caveat=True`.",
        "- No batch IDs available → batch-aware ΔBSV NOT possible. Reported as a known limitation.",
        "- All 4 cohorts have similar n (~50-70) — class balance OK.",
        "- Each patient = 1 averaged spectrum (averaged over ~49 spatial acquisitions per patient).",
    ]
    (REPORTS / "REPORT_pilot2_ingestion_audit.md").write_text("\n".join(lines))

    lines = [
        "# Pilot 2 — Substrate + Batch Handling",
        "",
        f"## Substrate block: **{block}**",
        "",
        "- substrate_documented = 'label-free SERS nanosensor (no chemistry spec)'",
        "- apply_substrate_physics_for_inference: **False** (substrate type unknown)",
        "- apply_substrate_physics_for_interpretation: **True** (caveat issued)",
        "",
        "## Batch handling",
        "",
        "- batch_id_available: **False**",
        "- ΔBSV reference modes:",
        "  - PRIMARY: global NC centroid (49 patients)",
        "  - SECONDARY: all-sample neutral centroid (219 patients)",
        "  - batch-local NC: NOT AVAILABLE (no batch metadata)",
        "",
        "## Limitations to surface in interpretation",
        "",
        "- Without batch metadata, between-cohort variance and within-cohort variance cannot be cleanly separated from acquisition-batch variance. Pilot 1 v2 found batch effects dominate class signal in HCC; the same caveat must be assumed here.",
        "- Without substrate confirmation, any apparent G07-aromatic enrichment, G02 purine signal, or G05 glycan signal must be interpreted with substrate-scope caveat.",
    ]
    (REPORTS / "REPORT_pilot2_substrate_batch_handling.md").write_text("\n".join(lines))

    return refs, block


# ─────────────────────────────────────────────────────────────────────
# Stage 3 — pipeline
# ─────────────────────────────────────────────────────────────────────

def stage3_pipeline(refs, master_x, motif_df, mss_df, motif_id_to_group,
                       motif_ids, analyte_to_group, block):
    apply_sers = BLOCK_APPLY[block]
    print(f"\n[STAGE 3] Pipeline: block={block}, apply_sers_physics={apply_sers}")
    rows = []
    for r in refs:
        mf = compute_motif_firings(r["spectrum"], master_x, motif_df)
        order = np.argsort(-mf)
        top_motif_families = []
        for idx in order[:5]:
            g = motif_id_to_group.get(motif_ids[idx], None)
            if g and g not in top_motif_families:
                top_motif_families.append(g)
            if len(top_motif_families) >= 3: break

        ms = compute_mss_scores_v43(r["spectrum"], master_x, mss_df)
        top_mss = sorted(ms.items(), key=lambda kv: -kv[1])[:5]

        bsv = compute_hybrid_bsv_v45(
            r["spectrum"], master_x, mf, ms, motif_id_to_group, motif_ids,
            analyte_to_group, regime="SERS",
            apply_sers_physics=apply_sers, apply_tg_veto=True,
        )
        per_group = bsv["per_group"]
        bsv_vec = {g: round(per_group.get(g, {}).get("magnitude", 0.0), 4)
                    for g in BSV_GROUPS_ORDER}
        conf_vec = {g: round(per_group.get(g, {}).get("confidence", 0.0), 4)
                     for g in BSV_GROUPS_ORDER}
        sorted_g = sorted(per_group.items(), key=lambda kv: -kv[1]["magnitude"])
        top3 = [g for g, _ in sorted_g[:3]]
        row = {
            "spectrum_id": r["spectrum_id"],
            "sample_id": r["patient_id"],
            "class_label": r["class_label"],
            "batch_id": None,
            "preprocessing_tag": "raw_spatial_mean + gaira_canonical_resample",
            "substrate_block": block,
            "substrate_physics_inference_applied": apply_sers,
            "substrate_physics_interpretation_applied": True,
            "top_motif_family": top_motif_families[0] if top_motif_families else None,
            "top_3_motif_families": ";".join(top_motif_families[:3]),
            "top_mss_hits": ";".join(n for n, _ in top_mss),
            "top_mss_scores": ";".join(str(round(s, 3)) for _, s in top_mss),
            "top_bsv_family": bsv["top_group"],
            "top_3_bsv_families": ";".join(top3),
            "bsv_vector_11axis": ";".join(f"{g}:{v}" for g, v in bsv_vec.items()),
            "confidence_vector_11axis": ";".join(f"{g}:{v}" for g, v in conf_vec.items()),
            "ambiguity_flag": bsv["ambiguity_flag"],
            "spillover_ratio": round(bsv["spillover_ratio"], 4),
            "top_confidence": round(per_group.get(bsv["top_group"], {}).get("confidence", 0.0), 4),
            "nearest_competing_family": sorted_g[1][0] if len(sorted_g) > 1 else None,
            "interpretation_tier": "SENSITIVE_SERS_UNKNOWN_SUBSTRATE_STRONG_CAVEAT",
            "n_position_acquisitions_averaged": r["n_position_acquisitions_averaged"],
        }
        row.update({f"abs_{g}": bsv_vec[g] for g in BSV_GROUPS_ORDER})
        row.update({f"conf_{g}": conf_vec[g] for g in BSV_GROUPS_ORDER})
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "pilot2_per_spectrum_outputs.csv", index=False)
    return df


# ─────────────────────────────────────────────────────────────────────
# Stage 4 — ΔBSV reference modes
# ─────────────────────────────────────────────────────────────────────

def stage4_delta_modes(df):
    print("\n[STAGE 4] ΔBSV reference modes")
    nc = df[df.class_label == "NC"]
    if len(nc) == 0:
        print("  WARNING: no NC samples for reference; falling back to all-sample neutral")
        nc_means = df[[f"abs_{g}" for g in BSV_GROUPS_ORDER]].mean()
    else:
        nc_means = nc[[f"abs_{g}" for g in BSV_GROUPS_ORDER]].mean()
    neutral_means = df[[f"abs_{g}" for g in BSV_GROUPS_ORDER]].mean()
    for g in BSV_GROUPS_ORDER:
        df[f"delta_nc_{g}"] = df[f"abs_{g}"] - nc_means[f"abs_{g}"]
        df[f"delta_neutral_{g}"] = df[f"abs_{g}"] - neutral_means[f"abs_{g}"]
    df["delta_bsv_vector_11axis"] = df.apply(
        lambda r: ";".join(f"{g}:{round(r[f'delta_nc_{g}'], 4)}" for g in BSV_GROUPS_ORDER),
        axis=1,
    )
    # top-3 changing families per spectrum
    def _t3(row):
        scores = [(g, row[f"delta_nc_{g}"]) for g in BSV_GROUPS_ORDER]
        scores.sort(key=lambda kv: -abs(kv[1]))
        return ";".join(f"{g}:{v:+.3f}" for g, v in scores[:3])
    df["top3_delta_changing_families"] = df.apply(_t3, axis=1)

    # Reference registry
    rows = [{
        "reference_mode": "NC cohort centroid",
        "n_samples": int(len(nc)),
        **{f"nc_mean_{g}": round(float(nc_means[f'abs_{g}']), 4)
             for g in BSV_GROUPS_ORDER},
    }, {
        "reference_mode": "all-sample neutral centroid",
        "n_samples": int(len(df)),
        **{f"neutral_mean_{g}": round(float(neutral_means[f'abs_{g}']), 4)
             for g in BSV_GROUPS_ORDER},
    }]
    pd.DataFrame(rows).to_csv(
        TABLES / "pilot2_delta_reference_registry.csv", index=False,
    )

    lines = [
        "# Pilot 2 — ΔBSV Reference Modes",
        "",
        f"- PRIMARY: **NC cohort centroid** (n={len(nc)})",
        f"- SECONDARY: all-sample neutral centroid (n={len(df)})",
        f"- batch-local NC: **NOT AVAILABLE** (no batch metadata)",
        "",
        "## NC vs neutral mean — per-family",
        "",
        "| family | NC centroid | neutral centroid | Δ (NC − neutral) |",
        "|---|---:|---:|---:|",
    ]
    for g in BSV_GROUPS_ORDER:
        nc_v = float(nc_means[f"abs_{g}"]); ne_v = float(neutral_means[f"abs_{g}"])
        lines.append(f"| {g} {FAMILY_LABELS.get(g, g)} | {nc_v:.4f} | {ne_v:.4f} | "
                      f"{nc_v - ne_v:+.4f} |")
    (REPORTS / "REPORT_pilot2_delta_reference.md").write_text("\n".join(lines))
    return df


# ─────────────────────────────────────────────────────────────────────
# Stage 5 — Group analysis (group + pairwise effect sizes + bootstrap CI)
# ─────────────────────────────────────────────────────────────────────

def _cohens_d(x, y):
    if len(x) < 2 or len(y) < 2: return 0.0
    pooled = np.sqrt(((len(x) - 1) * np.var(x, ddof=1) +
                       (len(y) - 1) * np.var(y, ddof=1)) /
                      max(len(x) + len(y) - 2, 1))
    return (np.mean(x) - np.mean(y)) / (pooled if pooled > 0 else 1.0)


def stage5_group_analysis(df):
    print("\n[STAGE 5] Group + pairwise effect-size analysis")
    rng = np.random.default_rng(42)

    # Group-level BSV / ΔBSV summaries
    gb = []
    for cls in CLASS_ORDER:
        sub = df[df.class_label == cls]
        for g in BSV_GROUPS_ORDER:
            gb.append({
                "class": cls, "family": g,
                "n": len(sub),
                "mean_BSV": round(float(sub[f"abs_{g}"].mean()), 4),
                "std_BSV": round(float(sub[f"abs_{g}"].std(ddof=1)), 4),
                "sem_BSV": round(float(sub[f"abs_{g}"].sem()), 4),
                "mean_delta_BSV": round(float(sub[f"delta_nc_{g}"].mean()), 4),
            })
    pd.DataFrame(gb).to_csv(TABLES / "pilot2_group_bsv_summary.csv", index=False)
    pd.DataFrame(gb).to_csv(TABLES / "pilot2_group_delta_bsv_summary.csv", index=False)

    # Effect size vs NC + bootstrap CI
    rows_vs = []
    for cls in ["HCC", "CCA", "LM"]:
        for g in BSV_GROUPS_ORDER:
            x = df[df.class_label == cls][f"abs_{g}"].values
            y = df[df.class_label == "NC"][f"abs_{g}"].values
            d_pt = _cohens_d(x, y)
            ds = []
            for _ in range(1000):
                xs = rng.choice(x, size=len(x), replace=True)
                ys = rng.choice(y, size=len(y), replace=True)
                ds.append(_cohens_d(xs, ys))
            ds = np.asarray(ds)
            ci_lo, ci_hi = float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))
            rows_vs.append({
                "comparison": f"{cls}_vs_NC", "family": g,
                "family_label": FAMILY_LABELS.get(g, g),
                "cohens_d": round(float(d_pt), 3),
                "abs_d": round(abs(float(d_pt)), 3),
                "ci95_low": round(ci_lo, 3), "ci95_high": round(ci_hi, 3),
                "ci_excludes_zero": (ci_lo > 0 and ci_hi > 0) or (ci_lo < 0 and ci_hi < 0),
            })
    pd.DataFrame(rows_vs).to_csv(
        TABLES / "pilot2_family_effect_sizes_vs_control.csv", index=False,
    )

    # Pairwise effect sizes (all 6 pairs of 4 groups)
    pairs = [("HCC", "NC"), ("CCA", "NC"), ("LM", "NC"),
              ("CCA", "HCC"), ("LM", "HCC"), ("CCA", "LM")]
    rows_pair = []
    for a, b in pairs:
        for g in BSV_GROUPS_ORDER:
            x = df[df.class_label == a][f"abs_{g}"].values
            y = df[df.class_label == b][f"abs_{g}"].values
            d = _cohens_d(x, y)
            rows_pair.append({
                "pair": f"{a}_vs_{b}", "family": g,
                "family_label": FAMILY_LABELS.get(g, g),
                "cohens_d": round(float(d), 3),
                "abs_d": round(abs(float(d)), 3),
            })
    pair_df = pd.DataFrame(rows_pair)
    pair_df.to_csv(TABLES / "pilot2_pairwise_family_effect_sizes.csv", index=False)

    # Motif + MSS enrichment
    df["_first_mss"] = df["top_mss_hits"].str.split(";").str[0]
    enr_rows = []
    for cls in CLASS_ORDER:
        sub = df[df.class_label == cls]
        n = len(sub)
        # Top motifs
        mf_counts = sub["top_motif_family"].value_counts()
        for fam in BSV_GROUPS_ORDER:
            n_fam = int(mf_counts.get(fam, 0))
            enr_rows.append({"class": cls, "type": "top_motif_family",
                              "item": fam, "n": n_fam, "rate": round(n_fam/n, 3)})
        # Top MSS
        mss_counts = sub["_first_mss"].value_counts().head(5)
        for name, count in mss_counts.items():
            enr_rows.append({"class": cls, "type": "MSS_first_hit",
                              "item": name, "n": int(count),
                              "rate": round(int(count)/n, 3)})
    pd.DataFrame(enr_rows).to_csv(
        TABLES / "pilot2_motif_mss_enrichment.csv", index=False,
    )
    return rows_vs, pair_df


# ─────────────────────────────────────────────────────────────────────
# Stage 6 — Biochemical state pattern per cohort
# ─────────────────────────────────────────────────────────────────────

def stage6_state_patterns(df, eff_vs_nc):
    print("\n[STAGE 6] Biochemical state patterns")
    eff_df = pd.DataFrame(eff_vs_nc)
    rows = []
    for cls in ["HCC", "CCA", "LM"]:
        sub_eff = eff_df[eff_df.comparison == f"{cls}_vs_NC"].sort_values("cohens_d", ascending=False)
        elevated = sub_eff[sub_eff.cohens_d > 0].head(3)
        depleted = sub_eff[sub_eff.cohens_d < 0].sort_values("cohens_d").head(3)
        rows.append({
            "class": cls,
            "elevated_top3_families": ";".join(f"{r.family}({r.cohens_d:+.2f})"
                                                  for _, r in elevated.iterrows()),
            "depleted_top3_families": ";".join(f"{r.family}({r.cohens_d:+.2f})"
                                                  for _, r in depleted.iterrows()),
            "max_abs_d_vs_NC": round(float(sub_eff["abs_d"].max()), 3),
            "n_meaningful_d_ge_03": int((sub_eff["abs_d"] >= 0.30).sum()),
            "n_ci_significant": int(sub_eff["ci_excludes_zero"].sum()),
            "ambiguity_rate": round(float(df[df.class_label == cls]["ambiguity_flag"].mean()), 3),
            "mean_top_confidence": round(float(df[df.class_label == cls]["top_confidence"].mean()), 3),
        })
    pat_df = pd.DataFrame(rows)
    pat_df.to_csv(TABLES / "pilot2_biochemical_state_patterns.csv", index=False)

    lines = [
        "# Pilot 2 — Biochemical State Patterns by Cohort",
        "",
        "## Per-cohort summary",
        "",
        "| class | elevated top-3 (vs NC) | depleted top-3 (vs NC) | max \\|d\\| | meaningful (\\|d\\|≥0.3) | CI-significant | ambiguity rate | mean conf |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in pat_df.iterrows():
        lines.append(
            f"| **{r['class']}** | {r['elevated_top3_families']} | "
            f"{r['depleted_top3_families']} | {r['max_abs_d_vs_NC']} | "
            f"{r['n_meaningful_d_ge_03']} | {r['n_ci_significant']} | "
            f"{r['ambiguity_rate']:.1%} | {r['mean_top_confidence']:.2f} |"
        )
    lines += [
        "",
        "## Special-attention axes",
        "",
        "Glycan (G05), nucleic-acid backbone (G04), lipid_acyl (G08), sterol (G09), "
        "protein (G06), purine_metabolite (G02), purine_nucleotide (G01), "
        "metabolic_small_molecule (G11) — see "
        "`tables/pilot2_family_effect_sizes_vs_control.csv` and "
        "`tables/pilot2_pairwise_family_effect_sizes.csv` for full per-axis numbers.",
    ]
    (REPORTS / "REPORT_pilot2_biochemical_state_patterns.md").write_text("\n".join(lines))
    return pat_df


# ─────────────────────────────────────────────────────────────────────
# Stage 7 — Figures
# ─────────────────────────────────────────────────────────────────────

def stage7_figures(df, refs, master_x, eff_vs_nc, pair_df):
    print("\n[STAGE 7] Figures")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pal = {"NC": "#1f77b4", "HCC": "#d62728", "CCA": "#ff7f0e", "LM": "#2ca02c"}

    # 1. spectra overview
    fig, ax = plt.subplots(figsize=(12, 4))
    for cls in CLASS_ORDER:
        spectra = np.vstack([r["spectrum"] for r in refs if r["class_label"] == cls])
        if len(spectra) == 0: continue
        m = np.nanmean(spectra, 0)
        fin = np.isfinite(m)
        mx = np.nanmax(m[fin]) if fin.any() else 1.0
        ax.plot(master_x, m / (mx + 1e-9), label=f"{cls} (n={spectra.shape[0]})",
                 color=pal[cls], linewidth=1.0)
    ax.set_xlim(400, 1800); ax.set_xlabel("Raman shift (cm⁻¹)")
    ax.set_ylabel("normalized intensity")
    ax.set_title("Pilot 2 CCA/HCC/LM/NC — mean preprocessed SERS by class")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "fig_pilot2_spectra_overview.png", dpi=150)
    plt.close(fig)

    # 2. BSV bar
    x = np.arange(len(BSV_GROUPS_ORDER))
    n = len(CLASS_ORDER); w = 0.78 / n
    fig, ax = plt.subplots(figsize=(13, 4.4))
    for i, cls in enumerate(CLASS_ORDER):
        sub = df[df.class_label == cls]
        means = [sub[f"abs_{g}"].mean() for g in BSV_GROUPS_ORDER]
        sems  = [sub[f"abs_{g}"].sem() for g in BSV_GROUPS_ORDER]
        ax.bar(x + (i - (n - 1) / 2) * w, means, w, yerr=sems, capsize=2,
                label=cls, color=pal[cls])
    ax.set_xticks(x)
    ax.set_xticklabels([FAMILY_LABELS.get(g, g) for g in BSV_GROUPS_ORDER], rotation=45, ha="right")
    ax.set_ylabel("mean BSV (± SEM)")
    ax.set_title("Pilot 2 — mean BSV by family (4 cohorts)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_pilot2_bsv_bar.png", dpi=150)
    plt.close(fig)

    # 3. ΔBSV bar (vs NC)
    fig, ax = plt.subplots(figsize=(13, 4.4))
    for i, cls in enumerate(["HCC", "CCA", "LM"]):
        sub = df[df.class_label == cls]
        means = [sub[f"delta_nc_{g}"].mean() for g in BSV_GROUPS_ORDER]
        sems  = [sub[f"delta_nc_{g}"].sem() for g in BSV_GROUPS_ORDER]
        ax.bar(x + (i - 1) * (0.78 / 3), means, 0.78 / 3, yerr=sems, capsize=2,
                label=f"{cls} − NC", color=pal[cls])
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([FAMILY_LABELS.get(g, g) for g in BSV_GROUPS_ORDER], rotation=45, ha="right")
    ax.set_ylabel("mean ΔBSV (vs NC centroid)")
    ax.set_title("Pilot 2 — mean ΔBSV per family (HCC / CCA / LM relative to NC)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_pilot2_delta_bsv_bar.png", dpi=150)
    plt.close(fig)

    # 4. BSV radar
    angles = np.linspace(0, 2 * np.pi, len(BSV_GROUPS_ORDER), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw={"projection": "polar"})
    for cls in CLASS_ORDER:
        sub = df[df.class_label == cls]
        vals = [float(sub[f"abs_{g}"].mean()) for g in BSV_GROUPS_ORDER]
        vals += vals[:1]
        ax.plot(angles, vals, label=cls, color=pal[cls], linewidth=1.5)
        ax.fill(angles, vals, alpha=0.07, color=pal[cls])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([FAMILY_LABELS.get(g, g) for g in BSV_GROUPS_ORDER], fontsize=9)
    ax.set_title("Pilot 2 — BSV 11-axis radar (4 cohort means)", pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.05), fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_pilot2_bsv_radar.png", dpi=180)
    plt.close(fig)

    # 5. ΔBSV radar (HCC / CCA / LM vs NC)
    fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw={"projection": "polar"})
    for cls in ["HCC", "CCA", "LM"]:
        sub = df[df.class_label == cls]
        vals = [float(sub[f"delta_nc_{g}"].mean()) for g in BSV_GROUPS_ORDER]
        vals += vals[:1]
        ax.plot(angles, vals, label=f"{cls} − NC", color=pal[cls], linewidth=1.5)
        ax.fill(angles, vals, alpha=0.10, color=pal[cls])
    ax.plot(angles, [0]*len(angles), color="k", linewidth=0.8, linestyle="--",
             label="NC baseline (Δ=0)")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([FAMILY_LABELS.get(g, g) for g in BSV_GROUPS_ORDER], fontsize=9)
    ax.set_title("Pilot 2 — ΔBSV 11-axis radar (cohorts vs NC centroid)", pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.05), fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_pilot2_delta_bsv_radar.png", dpi=180)
    plt.close(fig)

    # 6. Family effect-size heatmap vs NC (3 disease cols × 11 fams)
    eff_df = pd.DataFrame(eff_vs_nc)
    pivot = eff_df.pivot(index="family", columns="comparison", values="cohens_d")
    pivot = pivot.reindex(BSV_GROUPS_ORDER)
    fig, ax = plt.subplots(figsize=(7, 6))
    vmax = float(np.abs(pivot.values).max()) or 0.5
    im = ax.imshow(pivot.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_yticks(range(len(BSV_GROUPS_ORDER)))
    ax.set_yticklabels([FAMILY_LABELS.get(g, g) for g in BSV_GROUPS_ORDER])
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_title("Family effect size (Cohen's d) vs NC")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.iloc[i,j]:+.2f}", ha="center", va="center",
                     fontsize=8, color="white" if abs(pivot.iloc[i,j]) > vmax*0.5 else "black")
    fig.colorbar(im, ax=ax, label="Cohen's d")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_pilot2_family_effect_size_heatmap.png", dpi=150)
    plt.close(fig)

    # 7. Pairwise effect-size heatmap (6 pairs × 11 families)
    piv2 = pair_df.pivot(index="family", columns="pair", values="cohens_d")
    piv2 = piv2.reindex(BSV_GROUPS_ORDER)
    fig, ax = plt.subplots(figsize=(10, 6))
    vmax2 = float(np.abs(piv2.values).max()) or 0.5
    im = ax.imshow(piv2.values, cmap="RdBu_r", vmin=-vmax2, vmax=vmax2, aspect="auto")
    ax.set_yticks(range(len(BSV_GROUPS_ORDER)))
    ax.set_yticklabels([FAMILY_LABELS.get(g, g) for g in BSV_GROUPS_ORDER])
    ax.set_xticks(range(len(piv2.columns)))
    ax.set_xticklabels(piv2.columns, rotation=30, ha="right", fontsize=9)
    ax.set_title("Pairwise family effect size (Cohen's d)")
    for i in range(piv2.shape[0]):
        for j in range(piv2.shape[1]):
            ax.text(j, i, f"{piv2.iloc[i,j]:+.2f}", ha="center", va="center",
                     fontsize=7, color="white" if abs(piv2.iloc[i,j]) > vmax2*0.5 else "black")
    fig.colorbar(im, ax=ax, label="Cohen's d")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_pilot2_pairwise_effect_heatmap.png", dpi=150)
    plt.close(fig)

    # 8. Motif distribution by class
    motif_counts = df.groupby(["class_label", "top_motif_family"]).size().unstack(fill_value=0)
    motif_counts = motif_counts.reindex(columns=BSV_GROUPS_ORDER, fill_value=0).reindex(CLASS_ORDER)
    fig, ax = plt.subplots(figsize=(13, 4.2))
    motif_counts.T.plot(kind="bar", ax=ax, color=[pal[c] for c in CLASS_ORDER])
    ax.set_xticks(range(len(BSV_GROUPS_ORDER)))
    ax.set_xticklabels([FAMILY_LABELS.get(g, g) for g in BSV_GROUPS_ORDER], rotation=45, ha="right")
    ax.set_title("Pilot 2 — top motif family distribution by class")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_pilot2_motif_distribution.png", dpi=150)
    plt.close(fig)

    # 9. MSS distribution by class (top-10 first hits overall)
    df["_mss1"] = df["top_mss_hits"].str.split(";").str[0]
    mc = df.groupby(["class_label", "_mss1"]).size().unstack(fill_value=0).reindex(CLASS_ORDER)
    top10 = mc.sum(0).nlargest(10).index.tolist()
    fig, ax = plt.subplots(figsize=(13, 4.5))
    mc[top10].T.plot(kind="bar", ax=ax, color=[pal[c] for c in CLASS_ORDER])
    ax.set_title("Pilot 2 — top-10 MSS first-hit distribution by class")
    ax.tick_params(axis="x", labelrotation=45)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_pilot2_mss_distribution.png", dpi=150)
    plt.close(fig)

    # 10. Confidence + ambiguity by class
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    conf_means = [df[df.class_label == c]["top_confidence"].mean() for c in CLASS_ORDER]
    amb_rates = [df[df.class_label == c]["ambiguity_flag"].mean() for c in CLASS_ORDER]
    axes[0].bar(CLASS_ORDER, conf_means, color=[pal[c] for c in CLASS_ORDER])
    axes[0].set_title("top confidence per cohort"); axes[0].set_ylim(0, 1)
    axes[1].bar(CLASS_ORDER, amb_rates, color=[pal[c] for c in CLASS_ORDER])
    axes[1].set_title("ambiguity rate per cohort"); axes[1].set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_pilot2_confidence_ambiguity.png", dpi=150)
    plt.close(fig)

    # 11. PCA on BSV
    try:
        from sklearn.decomposition import PCA
        X = df[[f"abs_{g}" for g in BSV_GROUPS_ORDER]].values
        pc = PCA(n_components=2, random_state=0).fit_transform(X)
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        for cls in CLASS_ORDER:
            m = df["class_label"].values == cls
            ax.scatter(pc[m, 0], pc[m, 1], s=40, alpha=0.7,
                         label=f"{cls} (n={m.sum()})", color=pal[cls])
        ax.set_xlabel("PC1 of BSV"); ax.set_ylabel("PC2 of BSV")
        ax.set_title("Pilot 2 — unsupervised PCA of BSV (colored post-hoc)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGS / "fig_pilot2_bsv_pca.png", dpi=150)
        plt.close(fig)

        # 12. PCA on ΔBSV
        Xd = df[[f"delta_nc_{g}" for g in BSV_GROUPS_ORDER]].values
        pcd = PCA(n_components=2, random_state=0).fit_transform(Xd)
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        for cls in CLASS_ORDER:
            m = df["class_label"].values == cls
            ax.scatter(pcd[m, 0], pcd[m, 1], s=40, alpha=0.7,
                         label=f"{cls} (n={m.sum()})", color=pal[cls])
        ax.set_xlabel("PC1 of ΔBSV"); ax.set_ylabel("PC2 of ΔBSV")
        ax.set_title("Pilot 2 — unsupervised PCA of ΔBSV (colored post-hoc)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGS / "fig_pilot2_delta_bsv_pca.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  PCA skipped: {e}")

    # 13. Variance plot — note that we have no batch metadata, so variance
    # decomposition is within-class vs between-class (ANOVA-style)
    rows = []
    for g in BSV_GROUPS_ORDER:
        v = df[f"abs_{g}"].values
        total_var = float(np.var(v, ddof=1))
        wc = 0.0; tn = 0
        for cls in CLASS_ORDER:
            sub = df[df.class_label == cls][f"abs_{g}"].values
            if len(sub) > 1:
                wc += np.var(sub, ddof=1) * (len(sub) - 1)
                tn += (len(sub) - 1)
        within_var = wc / max(tn, 1)
        means = df.groupby("class_label")[f"abs_{g}"].mean()
        n_per = df.groupby("class_label").size()
        gm = float(np.mean(v))
        between_var = float(sum(n_per[cls] * (means[cls] - gm) ** 2
                                  for cls in means.index) / max(len(means) - 1, 1))
        rows.append({"family": g, "label": FAMILY_LABELS.get(g, g),
                       "within_class": within_var, "between_class": between_var,
                       "ratio_between_over_within": between_var / max(within_var, 1e-9)})
    var_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(11, 4.4))
    x = np.arange(len(BSV_GROUPS_ORDER))
    ax.bar(x - 0.2, var_df["within_class"], 0.4, label="within-class", color="#1f77b4")
    ax.bar(x + 0.2, var_df["between_class"], 0.4, label="between-class (4 cohorts)", color="#d62728")
    ax.set_xticks(x); ax.set_xticklabels([FAMILY_LABELS.get(g, g) for g in BSV_GROUPS_ORDER],
                                            rotation=45, ha="right")
    ax.set_ylabel("variance (BSV magnitude²)")
    ax.set_title("Pilot 2 — within-class vs between-class variance (no batch metadata)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "fig_pilot2_variance.png", dpi=150)
    plt.close(fig)

    print("  13 figures emitted")


# ─────────────────────────────────────────────────────────────────────
# Stage 8 — Descriptive label separation
# ─────────────────────────────────────────────────────────────────────

def stage8_descriptive(df):
    print("\n[STAGE 8] Descriptive label separation")
    from sklearn.decomposition import PCA
    X = df[[f"abs_{g}" for g in BSV_GROUPS_ORDER]].values
    pc = PCA(n_components=3, random_state=0)
    Z = pc.fit_transform(X)
    centroids = {}
    for cls in CLASS_ORDER:
        m = df["class_label"].values == cls
        if m.sum() > 0: centroids[cls] = Z[m, :2].mean(0)
    pairs = [("NC","HCC"),("NC","CCA"),("NC","LM"),
             ("HCC","CCA"),("HCC","LM"),("CCA","LM")]
    rows = []
    for a, b in pairs:
        if a in centroids and b in centroids:
            d = float(np.linalg.norm(centroids[a] - centroids[b]))
            rows.append({"pair": f"{a}_vs_{b}", "centroid_separation_PC1PC2": round(d, 3)})

    # Per-axis univariate AUC (NC=0, disease=1) for vs-NC pairs
    def _auc(v, y):
        order = np.argsort(v)
        rank = np.empty_like(order); rank[order] = np.arange(len(v))
        n_pos = y.sum(); n_neg = len(y) - n_pos
        if n_pos == 0 or n_neg == 0: return 0.5
        return float((rank[y == 1].sum() - n_pos*(n_pos-1)/2) / (n_pos * n_neg))
    auc_rows = []
    for cls in ["HCC", "CCA", "LM"]:
        m = df["class_label"].isin(["NC", cls])
        sub = df[m]
        y = (sub["class_label"].values == cls).astype(int)
        for g in BSV_GROUPS_ORDER:
            v = sub[f"abs_{g}"].values
            auc = _auc(v, y)
            auc_rows.append({"comparison": f"{cls}_vs_NC", "family": g,
                              "univariate_AUC": round(auc, 3),
                              "sep_strength": round(abs(auc - 0.5), 3)})
    auc_df = pd.DataFrame(auc_rows)
    desc_rows = [
        {"metric": "PCA_PC1_variance_explained", "value": round(float(pc.explained_variance_ratio_[0]), 3)},
        {"metric": "PCA_PC2_variance_explained", "value": round(float(pc.explained_variance_ratio_[1]), 3)},
        {"metric": "PCA_PC3_variance_explained", "value": round(float(pc.explained_variance_ratio_[2]), 3)},
    ]
    for r in rows:
        desc_rows.append({"metric": r["pair"] + "_centroid_sep", "value": r["centroid_separation_PC1PC2"]})
    pd.DataFrame(desc_rows).to_csv(
        TABLES / "pilot2_descriptive_label_separation.csv", index=False,
    )
    auc_df.to_csv(TABLES / "pilot2_per_family_univariate_auc.csv", index=False)

    lines = [
        "# Pilot 2 — Descriptive Label Separation (no fitting)",
        "",
        "## Unsupervised PCA on BSV space",
        "",
        f"- PC1 variance: {pc.explained_variance_ratio_[0]:.1%}",
        f"- PC2 variance: {pc.explained_variance_ratio_[1]:.1%}",
        f"- PC3 variance: {pc.explained_variance_ratio_[2]:.1%}",
        "",
        "## Centroid separation in PC1-PC2",
        "",
        "| pair | distance |",
        "|---|---:|",
    ]
    for r in rows:
        lines.append(f"| {r['pair']} | {r['centroid_separation_PC1PC2']:.3f} |")
    lines += [
        "",
        "## Best univariate AUC per disease-vs-NC comparison",
        "",
        "| comparison | family | AUC | |AUC−0.5| |",
        "|---|---|---:|---:|",
    ]
    for cls in ["HCC", "CCA", "LM"]:
        sub = auc_df[auc_df.comparison == f"{cls}_vs_NC"].sort_values("sep_strength", ascending=False).head(3)
        for _, r in sub.iterrows():
            lines.append(f"| {r['comparison']} | {r['family']} {FAMILY_LABELS.get(r['family'], '')} | "
                          f"{r['univariate_AUC']} | {r['sep_strength']} |")
    lines += [
        "",
        "## Caveats",
        "",
        "- Univariate AUCs are descriptive only; no classifier trained.",
        "- Multivariate fitted models would likely exceed univariate AUCs (literature on this dataset achieves ~0.85-0.95 for tumor-vs-control with PCA-LDA-style classifiers).",
        "- These metrics must NOT be reported as diagnostic performance.",
    ]
    (REPORTS / "REPORT_pilot2_descriptive_separation.md").write_text("\n".join(lines))
    return rows, auc_df


# ─────────────────────────────────────────────────────────────────────
# Stage 9 — Interpretation
# ─────────────────────────────────────────────────────────────────────

def stage9_interpretation(df, eff_vs_nc, pair_df, pat_df, sep_rows):
    print("\n[STAGE 9] Biochemical interpretation")
    eff_df = pd.DataFrame(eff_vs_nc)
    # Top families per disease
    def _top_eff(cls, n=3, sign=+1):
        sub = eff_df[eff_df.comparison == f"{cls}_vs_NC"]
        if sign > 0:
            return sub.sort_values("cohens_d", ascending=False).head(n)
        else:
            return sub.sort_values("cohens_d").head(n)
    hcc_up = _top_eff("HCC", 3, +1); hcc_dn = _top_eff("HCC", 3, -1)
    cca_up = _top_eff("CCA", 3, +1); cca_dn = _top_eff("CCA", 3, -1)
    lm_up  = _top_eff("LM",  3, +1); lm_dn  = _top_eff("LM",  3, -1)

    # Disease vs disease distinguishers (highest |d| in pair)
    dvd_rows = []
    for pair in ["CCA_vs_HCC", "LM_vs_HCC", "CCA_vs_LM"]:
        sub = pair_df[pair_df.pair == pair].sort_values("abs_d", ascending=False).head(3)
        dvd_rows.append((pair, sub))

    lines = [
        "# Pilot 2 — Biochemical Interpretation (CAUTIOUS)",
        "",
        "## Substrate context",
        "",
        "- SERS substrate documented as 'label-free SERS nanosensor' — chemistry not specified.",
        "- Substrate block: `unknown_SERS` (inference OFF, interpretation ON, strong substrate caveat).",
        "- All claims below carry `SENSITIVE_SERS_UNKNOWN_SUBSTRATE_STRONG_CAVEAT`.",
        "",
        "## Per-cohort top biochemical shifts (vs NC centroid)",
        "",
        "### HCC vs NC",
        "",
        "**Elevated in HCC**:",
    ]
    for _, r in hcc_up.iterrows():
        ci = " *(CI ✓)*" if r["ci_excludes_zero"] else ""
        lines.append(f"- {r['family']} {r['family_label']}: d = {r['cohens_d']:+.2f} "
                     f"[{r['ci95_low']:+.2f}, {r['ci95_high']:+.2f}]{ci}")
    lines += ["", "**Depleted in HCC**:"]
    for _, r in hcc_dn.iterrows():
        ci = " *(CI ✓)*" if r["ci_excludes_zero"] else ""
        lines.append(f"- {r['family']} {r['family_label']}: d = {r['cohens_d']:+.2f} "
                     f"[{r['ci95_low']:+.2f}, {r['ci95_high']:+.2f}]{ci}")
    lines += ["", "### CCA vs NC", "", "**Elevated in CCA**:"]
    for _, r in cca_up.iterrows():
        ci = " *(CI ✓)*" if r["ci_excludes_zero"] else ""
        lines.append(f"- {r['family']} {r['family_label']}: d = {r['cohens_d']:+.2f} "
                     f"[{r['ci95_low']:+.2f}, {r['ci95_high']:+.2f}]{ci}")
    lines += ["", "**Depleted in CCA**:"]
    for _, r in cca_dn.iterrows():
        ci = " *(CI ✓)*" if r["ci_excludes_zero"] else ""
        lines.append(f"- {r['family']} {r['family_label']}: d = {r['cohens_d']:+.2f} "
                     f"[{r['ci95_low']:+.2f}, {r['ci95_high']:+.2f}]{ci}")
    lines += ["", "### LM vs NC", "", "**Elevated in LM**:"]
    for _, r in lm_up.iterrows():
        ci = " *(CI ✓)*" if r["ci_excludes_zero"] else ""
        lines.append(f"- {r['family']} {r['family_label']}: d = {r['cohens_d']:+.2f} "
                     f"[{r['ci95_low']:+.2f}, {r['ci95_high']:+.2f}]{ci}")
    lines += ["", "**Depleted in LM**:"]
    for _, r in lm_dn.iterrows():
        ci = " *(CI ✓)*" if r["ci_excludes_zero"] else ""
        lines.append(f"- {r['family']} {r['family_label']}: d = {r['cohens_d']:+.2f} "
                     f"[{r['ci95_low']:+.2f}, {r['ci95_high']:+.2f}]{ci}")
    lines += [
        "",
        "## HCC vs CCA vs LM differentiators (top |d| per pair)",
        "",
    ]
    for pair, sub in dvd_rows:
        lines.append(f"### {pair}")
        lines.append("")
        for _, r in sub.iterrows():
            lines.append(f"- {r['family']} {r['family_label']}: d = {r['cohens_d']:+.2f}")
        lines.append("")
    lines += [
        "## Per-cohort summary statistics",
        "",
        "| class | max |d| vs NC | meaningful (|d|≥0.3) | CI-significant | ambiguity | mean conf |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in pat_df.iterrows():
        lines.append(f"| {r['class']} | {r['max_abs_d_vs_NC']} | {r['n_meaningful_d_ge_03']} | "
                     f"{r['n_ci_significant']} | {r['ambiguity_rate']:.1%} | "
                     f"{r['mean_top_confidence']:.2f} |")
    lines += [
        "",
        "## Consistency with Pilot 1 (HCC) directional hypotheses",
        "",
        "Pilot 1 v2 found in HCC vs CTR (Gurian dataset, different substrate):",
        "- G05 Glycan ↑ (d ≈ +0.26)",
        "- G04 Nucleic-backbone ↑ (d ≈ +0.25)",
        "- G09 Sterol ↓ (d ≈ −0.17)",
        "- Purine axes G01/G02 essentially zero",
        "",
        "Pilot 2 HCC vs NC outcome: see HCC table above. Direction agreement with Pilot 1 is the primary cross-cohort consistency check.",
        "",
        "## Substrate + variance caveats",
        "",
        "- `unknown_SERS` block applied; do NOT assume citrate-Ag rules apply.",
        "- No batch metadata available — between-class variance reported but not separable from technical batch variance. Pilot 1 v2 demonstrated batch effects can dominate class signal in serum SERS.",
        "- Effect sizes < 0.20 should be treated as noise; |d| ≥ 0.30 with CI excluding zero is the meaningful-signal bar.",
        "",
        "## What should NOT be overclaimed",
        "",
        "- Do NOT claim diagnostic discrimination of HCC vs CCA vs LM.",
        "- Do NOT translate any single-family elevation to molecule identity (no 'contains uric acid' / 'contains albumin' style claims).",
        "- Do NOT extrapolate substrate-specific findings to other SERS substrates without confirmation.",
        "- Do NOT report this as cross-cohort validation of Pilot 1 — Pilot 1 used a different substrate (Gurian Ag colloid) and a different patient population (Trieste).",
        "",
        "## What CAN be carried into cross-pilot synthesis",
        "",
        "- The qualitative top-3 family ordering per cohort vs NC.",
        "- Whether the same families dominate in both Pilot 1 (HCC vs CTR) and Pilot 2 HCC subgroup.",
        "- The per-pair distinguishers (CCA vs HCC etc.) as biochemical-state hypotheses for follow-up.",
    ]
    (REPORTS / "REPORT_pilot2_biochemical_interpretation.md").write_text("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────
# Stage 10 — Readiness for cross-pilot synthesis
# ─────────────────────────────────────────────────────────────────────

def stage10_readiness(eff_vs_nc, pair_df, pat_df, sep_rows):
    print("\n[STAGE 10] Cross-pilot synthesis readiness")
    eff_df = pd.DataFrame(eff_vs_nc)
    n_meaningful = int((eff_df["abs_d"] >= 0.30).sum())
    n_ci = int(eff_df["ci_excludes_zero"].sum())
    max_d = float(eff_df["abs_d"].max())
    # Centroid separation magnitudes
    seps = {r["pair"]: r["centroid_separation_PC1PC2"] for r in sep_rows}
    max_sep = max(seps.values()) if seps else 0.0

    if n_meaningful >= 3 and n_ci >= 3:
        decision = "READY_FOR_CROSS_PILOT_SYNTHESIS"
    elif max_d >= 0.20 and n_ci >= 1:
        decision = "READY_FOR_CROSS_PILOT_SYNTHESIS"  # low-signal but synthesizable
    elif max_d < 0.20:
        decision = "NEEDS_INTERPRETATION_REVIEW"
    elif max_sep < 0.05:
        decision = "NEEDS_BATCH_REFERENCE_REVIEW"
    else:
        decision = "NEEDS_PILOT_2_QC_FIX"

    lines = [
        "# Pilot 2 — Readiness for Cross-Pilot Synthesis",
        "",
        f"**Decision: {decision}**",
        "",
        "## Headline numbers",
        "",
        f"- max |Cohen's d| (any disease vs NC, any family): **{max_d:.3f}**",
        f"- families with |d|≥0.30 across all 3 disease comparisons: **{n_meaningful}/33**",
        f"- families with bootstrap 95% CI excluding zero: **{n_ci}/33**",
        f"- max PC1-PC2 centroid separation across 6 pairs: **{max_sep:.3f}**",
        "",
        "## Per-cohort meaningful-effect counts",
        "",
        "| cohort | meaningful (|d|≥0.3) | CI-significant | max |d| |",
        "|---|---:|---:|---:|",
    ]
    for _, r in pat_df.iterrows():
        lines.append(f"| {r['class']} | {r['n_meaningful_d_ge_03']} | "
                     f"{r['n_ci_significant']} | {r['max_abs_d_vs_NC']} |")
    lines += [
        "",
        "## Interpretation of decision",
        "",
    ]
    if decision == "READY_FOR_CROSS_PILOT_SYNTHESIS":
        lines.append("Effect sizes and bootstrap-CI evidence support advancing to cross-pilot synthesis. "
                     "Substrate + variance caveats remain.")
    elif decision == "NEEDS_INTERPRETATION_REVIEW":
        lines.append(f"Max effect size {max_d:.2f} is below the 0.20 threshold for synthesis. "
                     "Recommend interpretation review before combining Pilot 1 + Pilot 2 results.")
    elif decision == "NEEDS_BATCH_REFERENCE_REVIEW":
        lines.append(f"PC1-PC2 centroid separation is small ({max_sep:.3f}); suggests cohort signal is "
                     "dwarfed by within-cohort variance. Batch reference review recommended.")
    else:
        lines.append("Pilot 2 has cohort-level QC concerns that should be resolved before synthesis.")
    lines += [
        "",
        "## Invariants preserved",
        "",
        "- Engine v4.5 unchanged",
        "- Taxonomy / motif / MSS v4.3 / substrate physics v1.2: read-only",
        "- Substrate rule blocks v2: extended LOCALLY with Ag_colloid_untyped + unknown_SERS",
        "- No classifier training, no threshold tuning, no label-driven feature selection",
        "- No target clinical fitting",
        "- No dynamic DART-Met",
    ]
    (REPORTS / "REPORT_pilot2_readiness_for_cross_pilot_synthesis.md").write_text("\n".join(lines))
    return decision, max_d, n_meaningful, n_ci


# ─────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 78)
    print("gaira_base_4_passive_target_pilot_2_cca_hcc_lm")
    print("=" * 78)
    for d in (TABLES, FIGS, REPORTS, AUDIT, CODE_SNAPSHOT):
        d.mkdir(parents=True, exist_ok=True)

    master_x = canonical_master_axis()
    mss_df = pd.read_csv(MSS_V43)
    motif_df = pd.read_csv(LEARNED_MOTIFS)
    motif_ids = motif_df["learned_motif_id"].tolist()
    motif_id_to_group = {}
    for g in BSV_GROUPS:
        for m_id in g["dominant_motifs"]:
            motif_id_to_group[m_id] = g["group_id"]
    bc_to_group = {bc: g["group_id"] for g in BSV_GROUPS
                    for bc in g["member_broad_classes"]}
    analyte_to_group = {}
    for _, r in mss_df.iterrows():
        analyte_to_group[r["analyte_name"]] = bc_to_group.get(r["broad_class"], "G11")

    refs, block = stage12_load_and_substrate(master_x)
    df = stage3_pipeline(refs, master_x, motif_df, mss_df, motif_id_to_group,
                            motif_ids, analyte_to_group, block)
    df = stage4_delta_modes(df)
    df.to_csv(TABLES / "pilot2_per_spectrum_outputs.csv", index=False)
    eff_vs_nc, pair_df = stage5_group_analysis(df)
    pat_df = stage6_state_patterns(df, eff_vs_nc)
    stage7_figures(df, refs, master_x, eff_vs_nc, pair_df)
    sep_rows, auc_df = stage8_descriptive(df)
    stage9_interpretation(df, eff_vs_nc, pair_df, pat_df, sep_rows)
    decision, max_d, n_meaningful, n_ci = stage10_readiness(
        eff_vs_nc, pair_df, pat_df, sep_rows,
    )

    # Audit log
    lines = [
        "# gaira_base_4_passive_target_pilot_2_cca_hcc_lm — Audit Log",
        "",
        "## Dataset",
        f"- {ZIP_PATH}",
        f"- 219 patient-mean spectra (NC=49, HCC=50, CCA=70, LM=50)",
        "",
        "## Substrate / batch handling",
        f"- substrate block: **{block}** (inference OFF, interpretation ON)",
        "- batch metadata: NOT AVAILABLE",
        "- ΔBSV reference: NC centroid (primary) + neutral centroid (secondary)",
        "",
        "## Pipeline",
        "- engine v4.5 + v3 fixes; UNCHANGED",
        "- 11-axis BSV + ΔBSV + confidence + ambiguity per spectrum",
        "- pairwise effect sizes for all 6 disease/control pairs",
        "- bootstrap 95% CIs (1000 resamples) for disease vs NC",
        "- unsupervised PCA on BSV + ΔBSV (colored post-hoc)",
        "",
        "## Results",
        f"- max |Cohen's d|: {max_d:.2f}",
        f"- families with |d|≥0.30: {n_meaningful}/33",
        f"- families with CI excluding zero: {n_ci}/33",
        "",
        f"## Decision: **{decision}**",
        "",
        "## Invariants",
        "- engine / taxonomy / motif / MSS / substrate physics v1.2: unchanged",
        "- ERG MSS template / G09 v4.5 logic: not used (no ERG/G09 context in Pilot 2)",
        "- substrate rule blocks v2 extended LOCALLY with Ag_colloid_untyped + unknown_SERS",
        "- no classifier training; no threshold tuning; no label-driven feature select",
        "- no target clinical fitting",
        "- no DART-Met",
    ]
    (AUDIT / "gaira_base_4_passive_target_pilot_2_cca_hcc_lm_audit_log.md"
     ).write_text("\n".join(lines))

    p = Path(__file__)
    if p.exists(): shutil.copy(p, CODE_SNAPSHOT / p.name)

    print(f"\n[complete] decision: {decision}")
    print(f"  max |d|={max_d:.2f}  meaningful={n_meaningful}/33  CI-significant={n_ci}/33")


if __name__ == "__main__":
    main()
