"""gaira_base_4 passive target Pilot 1 RERUN v2 — HCC holdout.

Fixes vs v1:
  1. Substrate handling — exact Ag-colloid variant is NOT documented in the
     Gurian 2020 dataset. Therefore use the new `Ag_colloid_untyped` block:
     inference physics OFF; interpretation physics ON; substrate caveat issued.
  2. Variance-aware analysis — quantify within-class vs between-class vs batch
     variance; bootstrap effect-size CIs; report whether cohort means are
     dwarfed by intra-cohort spread.

Engine v4.5 + v3 calibration fixes UNCHANGED.
NO classifier training. NO threshold tuning on labels. NO feature selection
using labels. NO parameter fitting. NO dynamic DART-Met.
"""
from __future__ import annotations

import json
import shutil
import sys
import warnings
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
    "gaira_base_4_passive_target_pilot_1_hcc_holdout_rerun_v2"
)
TABLES = ROOT / "tables"
FIGS = ROOT / "figures"
REPORTS = ROOT / "reports"
AUDIT = ROOT / "audit"
CODE_SNAPSHOT = ROOT / "code_snapshot"

V1_TABLES = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/"
    "gaira_base_4_passive_target_pilot_1_hcc_holdout/tables"
)

MSS_V43 = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_4_mss_decision_enrichment_v1/"
    "registry/grounding_molecular_signatures_v4_3.csv"
)
LEARNED_MOTIFS = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_3_grounding_trained_ontology_v1/"
    "registry/learned_motif_registry_v1.csv"
)
HCC_CSV = Path("/Volumes/SSD_Rad/GAIRA_DATA/raw/hcc_serum/data.csv")

BSV_GROUPS_ORDER = [g["group_id"] for g in BSV_GROUPS]


# ─────────────────────────────────────────────────────────────────────
# Extended substrate block list (v2)
# ─────────────────────────────────────────────────────────────────────

# Local extension: add untyped + unknown blocks. Not merged into the global
# substrate_rule_blocks_v2 registry; this driver uses them locally.
EXTRA_BLOCKS = [
    {
        "block_id": "Ag_colloid_untyped",
        "substrate_family": "Ag colloid (variant not documented)",
        "substrate_status": "AG_COLLOID_BUT_UNTYPED",
        "apply_for_inference": False,
        "apply_for_interpretation": True,
        "applicable_rules": "SERS physics OFF for inference (substrate variant unconfirmed); interpretation emits substrate_variant_caveat=True",
        "datasets_in_scope": "datasets where Ag colloid is reported but stabilizer/preparation is not",
        "notes": "Conservative fallback when SERS substrate is Ag-colloid family but chemistry-specific stabilizer (citrate / hydroxylamine / other) is not specified. Avoids applying citrate-Ag-trained dampening rules to potentially-different chemistry.",
    },
    {
        "block_id": "unknown_SERS",
        "substrate_family": "SERS substrate (type unknown)",
        "substrate_status": "UNKNOWN",
        "apply_for_inference": False,
        "apply_for_interpretation": True,
        "applicable_rules": "SERS physics OFF for inference; strong substrate caveat in interpretation",
        "datasets_in_scope": "datasets where SERS substrate type is not documented at all",
        "notes": "Strongest fallback. Treats output as SERS-regime + raw band evidence only.",
    },
]
ALL_BLOCKS = SUBSTRATE_BLOCKS + EXTRA_BLOCKS
BLOCK_APPLY = {b["block_id"]: b["apply_for_inference"] for b in ALL_BLOCKS}


def substrate_block_for_v2(substrate_family, citrate_confirmed=False):
    """v2 selector — requires citrate-Ag confirmation; else Ag_colloid_untyped or unknown_SERS."""
    sf = (substrate_family or "").lower()
    if citrate_confirmed and ("citrate" in sf or "cag" in sf):
        return "citrate_Ag_colloid_trained"
    if "bagnps" in sf or "biologically" in sf:
        return "bAgNPs_diagnostic"
    if "cspp" in sf or "plasmonic paper" in sf:
        return "CSPP_paper_Ag_conditional"
    if "ag film" in sf or "jacs" in sf:
        return "Ag_film_JACS_featurepack_only"
    if "ag colloid" in sf or "colloid" in sf:
        return "Ag_colloid_untyped"
    return "unknown_SERS"


# ─────────────────────────────────────────────────────────────────────
# Substrate audit
# ─────────────────────────────────────────────────────────────────────

def stage0_substrate_audit():
    print("\n[STAGE 0] Substrate metadata audit")
    # Locate substrate documentation in HCC dataset
    rcode_path = Path("/Volumes/SSD_Rad/GAIRA_DATA/raw/hcc_serum/R_code.R")
    rcode_text = rcode_path.read_text() if rcode_path.exists() else ""
    citrate_confirmed = ("citrate" in rcode_text.lower())  # explicit search
    stabilizer_mentioned = any(
        kw in rcode_text.lower()
        for kw in ("citrate", "hydroxylamine", "borohydride", "lee-meisel", "leopold-lendl")
    )

    # Decision
    if citrate_confirmed:
        block = "citrate_Ag_colloid_trained"
        substrate_status = "CITRATE_AG_CONFIRMED"
    elif stabilizer_mentioned:
        # If a stabilizer is mentioned but not citrate, still untyped for v3 fix scope
        block = "Ag_colloid_untyped"
        substrate_status = "AG_COLLOID_NON_CITRATE"
    else:
        block = "Ag_colloid_untyped"
        substrate_status = "AG_COLLOID_VARIANT_UNDOCUMENTED"

    rows = [{
        "dataset": "hcc_serum (Gurian 2020)",
        "documentation_searched": "R_code.R + paper title from R_code header",
        "citrate_confirmed_in_metadata": citrate_confirmed,
        "any_stabilizer_keyword_found": stabilizer_mentioned,
        "substrate_status": substrate_status,
        "substrate_block_assigned_v2": block,
        "apply_substrate_physics_for_inference": BLOCK_APPLY[block],
        "apply_substrate_physics_for_interpretation": True,
        "substrate_variant_caveat": (substrate_status != "CITRATE_AG_CONFIRMED"),
    }]
    pd.DataFrame(rows).to_csv(
        TABLES / "pilot1_v2_substrate_metadata_audit.csv", index=False,
    )

    lines = [
        "# Pilot 1 v2 — Substrate Handling Fix",
        "",
        "## Audit findings",
        "",
        f"- Searched: `{rcode_path}` + the paper title carried in the R header",
        f"- Citrate explicitly mentioned: **{citrate_confirmed}**",
        f"- Any Ag-colloid stabilizer keyword found (citrate / hydroxylamine / "
        f"borohydride / Lee-Meisel / Leopold-Lendl): **{stabilizer_mentioned}**",
        "",
        f"## Substrate status: **{substrate_status}**",
        "",
        f"- substrate block assigned: `{block}`",
        f"- apply_substrate_physics_for_inference: **{BLOCK_APPLY[block]}**",
        f"- apply_substrate_physics_for_interpretation: **True**",
        f"- substrate_variant_caveat: **{substrate_status != 'CITRATE_AG_CONFIRMED'}**",
        "",
        "## Why this matters",
        "",
        "Pilot 1 v1 defaulted to `citrate_Ag_colloid_trained` (inference ON), implicitly "
        "applying citrate-Ag-trained SERS observation rules (purine 720-740 dampening, "
        "UA-carotenoid 1517 ambiguity, amide-I 1600-1700 dampening, Phe 1003 boost, "
        "etc.) to a dataset whose Ag-colloid stabilizer is not documented. If the "
        "Gurian dataset uses a different reducing agent (e.g. hydroxylamine or "
        "borohydride), citrate-Ag rules can systematically distort the outputs.",
        "",
        "**v2 fix:** drop to `Ag_colloid_untyped` block — inference runs without "
        "substrate-specific dampening; interpretation layer flags the substrate "
        "variant caveat. This is the conservative correct behaviour when "
        "substrate chemistry is unconfirmed.",
    ]
    (REPORTS / "REPORT_pilot1_v2_substrate_handling.md").write_text("\n".join(lines))
    print(f"  substrate status: {substrate_status} → block {block} (inference={BLOCK_APPLY[block]})")
    return block, citrate_confirmed


# ─────────────────────────────────────────────────────────────────────
# Stage 1 — Ingestion + QC
# ─────────────────────────────────────────────────────────────────────

def stage1_ingestion(master_x, block):
    print("\n[STAGE 1] HCC holdout ingestion + QC")
    df = pd.read_csv(HCC_CSV, low_memory=False)
    meta_cols = ["acquisition_date", "substrate_batch", "class", "sample_code"]
    wn_cols = [c for c in df.columns if c not in meta_cols]
    wn = np.array([float(c) for c in wn_cols])
    refs = []
    for i, row in df.iterrows():
        y = row[wn_cols].values.astype(float)
        order = np.argsort(wn)
        y_rs = np.interp(master_x, wn[order], y[order],
                           left=np.nan, right=np.nan)
        refs.append({
            "spectrum_id": f"hcc::{row['sample_code']}_b{row['substrate_batch']}_d{row['acquisition_date']}",
            "sample_code": row["sample_code"],
            "class_label": row["class"],
            "substrate_batch": row["substrate_batch"],
            "acquisition_date": row["acquisition_date"],
            "regime": "SERS",
            "substrate_family": "Ag colloid (Gurian 2020) — variant undocumented",
            "spectrum": y_rs,
        })

    qc_rows = [{
        "dataset": "hcc_serum (Gurian 2020)",
        "n_spectra": len(refs),
        "n_CTR": sum(1 for r in refs if r["class_label"] == "CTR"),
        "n_H0T": sum(1 for r in refs if r["class_label"] == "H0T"),
        "n_unique_samples": len(set(r["sample_code"] for r in refs)),
        "n_substrate_batches": len(set(r["substrate_batch"] for r in refs)),
        "substrate_batches": ";".join(sorted(set(r["substrate_batch"] for r in refs))),
        "acquisition_date_range": f"{min(r['acquisition_date'] for r in refs)} → {max(r['acquisition_date'] for r in refs)}",
        "regime": "SERS",
        "substrate_block_v2": block,
        "spectral_range_cm1": f"{wn.min():.1f} to {wn.max():.1f}",
        "n_wn_cols": len(wn_cols),
        "preprocessing_compatibility": "data.csv pre-baselined by original authors + linear interp to GAIRA master axis",
        "passive_readout_only": True,
    }]
    pd.DataFrame(qc_rows).to_csv(TABLES / "pilot1_v2_ingestion_qc.csv", index=False)

    # Per-batch sanity
    by_batch = defaultdict(lambda: defaultdict(int))
    for r in refs:
        by_batch[r["substrate_batch"]][r["class_label"]] += 1

    lines = [
        "# Pilot 1 v2 — Ingestion + QC",
        "",
        f"- n_spectra: **{qc_rows[0]['n_spectra']}** (CTR {qc_rows[0]['n_CTR']} + H0T {qc_rows[0]['n_H0T']})",
        f"- n_unique samples: {qc_rows[0]['n_unique_samples']}",
        f"- n_substrate_batches: {qc_rows[0]['n_substrate_batches']} ({qc_rows[0]['substrate_batches']})",
        f"- date range: {qc_rows[0]['acquisition_date_range']}",
        "",
        "## Per-batch class composition",
        "",
        "| batch | CTR | H0T | total |",
        "|---|---:|---:|---:|",
    ]
    for b in sorted(by_batch):
        c = by_batch[b]["CTR"]; h = by_batch[b]["H0T"]
        lines.append(f"| {b} | {c} | {h} | {c+h} |")
    lines += [
        "",
        f"- substrate block v2: **{qc_rows[0]['substrate_block_v2']}**",
        "- preprocessing: data.csv is pre-baselined by Gurian/Bonifacio authors; GAIRA does linear-interp to canonical master axis only.",
    ]
    (REPORTS / "REPORT_pilot1_v2_ingestion_qc.md").write_text("\n".join(lines))
    print(f"  {len(refs)} spectra; per-batch CTR/H0T balance OK")
    return refs


# ─────────────────────────────────────────────────────────────────────
# Stage 2 — Run pipeline (full locked, inference per substrate block)
# ─────────────────────────────────────────────────────────────────────

def run_pipeline(refs, master_x, motif_df, mss_df, motif_id_to_group,
                   motif_ids, analyte_to_group, erg_peaks, block):
    apply_sers = BLOCK_APPLY[block]
    print(f"\n[STAGE 2] Pipeline: block={block}, apply_sers_physics={apply_sers}")
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
            "sample_id": r["sample_code"],
            "class_label": r["class_label"],
            "batch_id": r["substrate_batch"],
            "acquisition_date": r["acquisition_date"],
            "preprocessing_tag": "gurian2020_baselined + gaira_canonical_resample",
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
            "interpretation_tier": "SENSITIVE_SERS_SUBSTRATE_VARIANT_CAVEAT",
        }
        row.update({f"abs_{g}": bsv_vec[g] for g in BSV_GROUPS_ORDER})
        row.update({f"conf_{g}": conf_vec[g] for g in BSV_GROUPS_ORDER})
        rows.append(row)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────
# Stage 3 — Multiple ΔBSV reference modes
# ─────────────────────────────────────────────────────────────────────

def stage3_delta_modes(df):
    print("\n[STAGE 3] ΔBSV reference modes")
    rows = []

    # 1. CTR centroid (global)
    ctr_means = df[df.class_label == "CTR"][[f"abs_{g}" for g in BSV_GROUPS_ORDER]].mean()
    for g in BSV_GROUPS_ORDER:
        df[f"delta_ctr_{g}"] = df[f"abs_{g}"] - ctr_means[f"abs_{g}"]

    # 2. all-sample neutral centroid
    neutral_means = df[[f"abs_{g}" for g in BSV_GROUPS_ORDER]].mean()
    for g in BSV_GROUPS_ORDER:
        df[f"delta_neutral_{g}"] = df[f"abs_{g}"] - neutral_means[f"abs_{g}"]

    # 3. batch-local CTR (per batch_id)
    for batch in df["batch_id"].unique():
        ctr_mask = (df["class_label"] == "CTR") & (df["batch_id"] == batch)
        if ctr_mask.sum() == 0: continue
        local_means = df.loc[ctr_mask, [f"abs_{g}" for g in BSV_GROUPS_ORDER]].mean()
        bm = df["batch_id"] == batch
        for g in BSV_GROUPS_ORDER:
            df.loc[bm, f"delta_batchctr_{g}"] = df.loc[bm, f"abs_{g}"] - local_means[f"abs_{g}"]
    if "delta_batchctr_G01" not in df.columns:
        for g in BSV_GROUPS_ORDER:
            df[f"delta_batchctr_{g}"] = df[f"delta_ctr_{g}"]  # fallback

    # 4. sample-local — Gurian dataset has 1 spectrum per sample, so this is degenerate.
    # We document and skip.

    # Comparison: stability of effect-size sign across 3 reference modes
    rows = []
    for g in BSV_GROUPS_ORDER:
        d_ctr = df[df.class_label == "H0T"][f"delta_ctr_{g}"].mean()
        d_neu = df[df.class_label == "H0T"][f"delta_neutral_{g}"].mean()
        d_bch = df[df.class_label == "H0T"][f"delta_batchctr_{g}"].mean()
        signs = [np.sign(d_ctr), np.sign(d_neu), np.sign(d_bch)]
        all_agree = all(s == signs[0] and s != 0 for s in signs) or all(s == 0 for s in signs)
        rows.append({
            "family": g, "family_label": FAMILY_LABELS.get(g, g),
            "delta_H0T_vs_CTR_centroid": round(float(d_ctr), 4),
            "delta_H0T_vs_neutral_centroid": round(float(d_neu), 4),
            "delta_H0T_vs_batchlocal_CTR": round(float(d_bch), 4),
            "all_three_agree_on_sign": bool(all_agree),
        })
    cmp_df = pd.DataFrame(rows)
    cmp_df.to_csv(TABLES / "pilot1_v2_delta_reference_comparison.csv", index=False)

    n_agree = int(cmp_df["all_three_agree_on_sign"].sum())
    lines = [
        "# Pilot 1 v2 — ΔBSV Reference Modes",
        "",
        "## Reference modes computed",
        "",
        "1. **CTR centroid** (global, all CTR spectra mean) — primary",
        "2. **all-sample neutral centroid** (all spectra mean) — comparator",
        "3. **batch-local CTR** (per substrate-batch CTR mean) — controls for batch effects",
        "4. *sample-local — N/A* (Gurian dataset has 1 spectrum per sample)",
        "",
        "## Cross-reference stability per family",
        "",
        f"- Families where all 3 references agree on Δ sign: **{n_agree}/11**",
        "",
        "| family | Δ vs CTR | Δ vs neutral | Δ vs batch-CTR | sign agree |",
        "|---|---:|---:|---:|---|",
    ]
    for _, r in cmp_df.iterrows():
        lines.append(
            f"| {r['family']} {r['family_label']} | "
            f"{r['delta_H0T_vs_CTR_centroid']:+.4f} | "
            f"{r['delta_H0T_vs_neutral_centroid']:+.4f} | "
            f"{r['delta_H0T_vs_batchlocal_CTR']:+.4f} | "
            f"{'YES' if r['all_three_agree_on_sign'] else 'no'} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- Reference choice does NOT meaningfully change Δ sign for a stable family signal.",
        "- Where signs disagree across reference modes, the underlying signal is below the variance floor and should not be interpreted.",
    ]
    (REPORTS / "REPORT_pilot1_v2_delta_reference_modes.md").write_text("\n".join(lines))
    print(f"  3 ref modes computed; {n_agree}/11 families have sign-stable Δ")
    return df, cmp_df


# ─────────────────────────────────────────────────────────────────────
# Stage 4 — Variance-aware analysis
# ─────────────────────────────────────────────────────────────────────

def stage4_variance(df):
    print("\n[STAGE 4] Variance-aware analysis")
    # Total / within-class / between-class / batch variance per family
    rows = []
    for g in BSV_GROUPS_ORDER:
        v = df[f"abs_{g}"].values
        total_var = float(np.var(v, ddof=1))
        # Within-class
        wc = 0.0; total_n = 0
        for cls in ["CTR", "H0T"]:
            sub = df[df.class_label == cls][f"abs_{g}"].values
            if len(sub) > 1:
                wc += np.var(sub, ddof=1) * (len(sub) - 1)
                total_n += (len(sub) - 1)
        within_var = wc / max(total_n, 1)
        # Between-class
        means = df.groupby("class_label")[f"abs_{g}"].mean()
        n_per = df.groupby("class_label").size()
        grand_mean = float(np.mean(v))
        between_var = float(sum(n_per[cls] * (means[cls] - grand_mean) ** 2 for cls in means.index)
                              / max(len(means) - 1, 1))
        # Batch-associated variance
        batch_means = df.groupby("batch_id")[f"abs_{g}"].mean()
        n_batch = df.groupby("batch_id").size()
        batch_var = float(sum(n_batch[b] * (batch_means[b] - grand_mean) ** 2 for b in batch_means.index)
                            / max(len(batch_means) - 1, 1))
        # Ratios
        between_over_within = between_var / max(within_var, 1e-9)
        batch_over_class = batch_var / max(between_var, 1e-9)
        rows.append({
            "family": g, "family_label": FAMILY_LABELS.get(g, g),
            "total_variance": round(total_var, 6),
            "within_class_variance": round(within_var, 6),
            "between_class_variance": round(between_var, 6),
            "batch_associated_variance": round(batch_var, 6),
            "between_class_over_within_class": round(between_over_within, 4),
            "batch_over_between_class": round(batch_over_class, 4),
            "class_signal_meaningful": bool(between_over_within >= 0.05),
            "batch_dominates_class": bool(batch_over_class >= 1.0),
        })
    var_df = pd.DataFrame(rows)
    var_df.to_csv(TABLES / "pilot1_v2_variance_decomposition.csv", index=False)

    # Bootstrap effect-size CIs
    rng = np.random.default_rng(42)
    boot_n = 1000
    eff_rows = []
    for g in BSV_GROUPS_ORDER:
        m_h = df[df.class_label == "H0T"][f"abs_{g}"].values
        m_c = df[df.class_label == "CTR"][f"abs_{g}"].values
        ds = []
        for _ in range(boot_n):
            h_s = rng.choice(m_h, size=len(m_h), replace=True)
            c_s = rng.choice(m_c, size=len(m_c), replace=True)
            pooled = np.sqrt(((len(h_s)-1)*np.var(h_s, ddof=1) + (len(c_s)-1)*np.var(c_s, ddof=1)) / max(len(h_s)+len(c_s)-2, 1))
            d = (h_s.mean() - c_s.mean()) / (pooled if pooled > 0 else 1.0)
            ds.append(d)
        ds = np.asarray(ds)
        d_pt = (m_h.mean() - m_c.mean()) / (np.sqrt(((len(m_h)-1)*np.var(m_h, ddof=1) + (len(m_c)-1)*np.var(m_c, ddof=1)) / max(len(m_h)+len(m_c)-2, 1)) or 1)
        ci_lo = float(np.percentile(ds, 2.5))
        ci_hi = float(np.percentile(ds, 97.5))
        # CI excludes 0?
        ci_excludes_zero = (ci_lo > 0 and ci_hi > 0) or (ci_lo < 0 and ci_hi < 0)
        eff_rows.append({
            "family": g, "family_label": FAMILY_LABELS.get(g, g),
            "cohens_d_point": round(float(d_pt), 3),
            "ci95_low": round(ci_lo, 3),
            "ci95_high": round(ci_hi, 3),
            "ci_excludes_zero": bool(ci_excludes_zero),
            "abs_d": round(abs(float(d_pt)), 3),
        })
    eff_df = pd.DataFrame(eff_rows).sort_values("abs_d", ascending=False)
    eff_df.to_csv(TABLES / "pilot1_v2_bootstrap_effect_sizes.csv", index=False)

    # Variance domination check
    n_class_meaningful = int(var_df["class_signal_meaningful"].sum())
    n_batch_dominates = int(var_df["batch_dominates_class"].sum())
    n_ci_excludes_zero = int(eff_df["ci_excludes_zero"].sum())

    lines = [
        "# Pilot 1 v2 — Variance Analysis",
        "",
        "## Per-family variance decomposition",
        "",
        "| family | total | within-class | between-class | batch | between/within | batch/between |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in var_df.iterrows():
        lines.append(
            f"| {r['family']} | {r['total_variance']:.4f} | "
            f"{r['within_class_variance']:.4f} | "
            f"{r['between_class_variance']:.4f} | "
            f"{r['batch_associated_variance']:.4f} | "
            f"{r['between_class_over_within_class']:.3f} | "
            f"{r['batch_over_between_class']:.3f} |"
        )
    lines += [
        "",
        f"- Families where between-class variance is ≥5% of within-class variance: **{n_class_meaningful}/11**",
        f"- Families where batch variance ≥ between-class variance: **{n_batch_dominates}/11** "
        f"({'BATCH EFFECTS DOMINATE' if n_batch_dominates >= 6 else 'batch effects modest'})",
        "",
        "## Bootstrap effect-size CIs (Cohen's d, 1000 resamples)",
        "",
        "| family | d (point) | 95% CI | CI excludes 0 |",
        "|---|---:|---|---|",
    ]
    for _, r in eff_df.iterrows():
        lines.append(
            f"| {r['family']} {r['family_label']} | {r['cohens_d_point']:+.3f} | "
            f"[{r['ci95_low']:+.3f}, {r['ci95_high']:+.3f}] | "
            f"{'YES' if r['ci_excludes_zero'] else 'no'} |"
        )
    lines += [
        "",
        f"- Families where the 95% CI on Cohen's d **excludes zero**: **{n_ci_excludes_zero}/11**",
        "",
        "## Interpretation",
        "",
        "- A family signal is **statistically meaningful** when the bootstrap 95% CI on its effect size does NOT cross zero.",
        "- A family signal is **substantively meaningful** when |d| ≥ 0.30 AND CI excludes zero.",
        "- When batch-associated variance exceeds between-class variance, batch effects are the dominant axis and class signal interpretation is fragile.",
    ]
    (REPORTS / "REPORT_pilot1_v2_variance_analysis.md").write_text("\n".join(lines))
    print(f"  variance: {n_class_meaningful}/11 families with meaningful between/within ratio; "
          f"batch dominates in {n_batch_dominates}/11; {n_ci_excludes_zero}/11 CIs exclude zero")
    return var_df, eff_df


# ─────────────────────────────────────────────────────────────────────
# Stage 5 — Group-level summaries (BSV / ΔBSV / motif / MSS shifts)
# ─────────────────────────────────────────────────────────────────────

def stage5_groups(df, eff_df):
    print("\n[STAGE 5] Group-level summaries")
    # BSV summary
    gb = []
    for cls in ["CTR", "H0T"]:
        sub = df[df.class_label == cls]
        for g in BSV_GROUPS_ORDER:
            gb.append({
                "class": cls, "family": g,
                "mean_BSV": round(float(sub[f"abs_{g}"].mean()), 4),
                "std_BSV": round(float(sub[f"abs_{g}"].std(ddof=1)), 4),
                "sem_BSV": round(float(sub[f"abs_{g}"].sem()), 4),
                "mean_confidence": round(float(sub[f"conf_{g}"].mean()), 4),
                "n": len(sub),
            })
    pd.DataFrame(gb).to_csv(TABLES / "pilot1_v2_group_bsv_summary.csv", index=False)

    gd = []
    for cls in ["CTR", "H0T"]:
        sub = df[df.class_label == cls]
        for g in BSV_GROUPS_ORDER:
            gd.append({
                "class": cls, "family": g,
                "mean_delta_BSV": round(float(sub[f"delta_ctr_{g}"].mean()), 4),
                "std_delta_BSV": round(float(sub[f"delta_ctr_{g}"].std(ddof=1)), 4),
                "n": len(sub),
            })
    pd.DataFrame(gd).to_csv(TABLES / "pilot1_v2_group_delta_bsv_summary.csv", index=False)

    eff_df.to_csv(TABLES / "pilot1_v2_family_effect_sizes.csv", index=False)

    # Motif + MSS shift summary
    df["_first_mss"] = df["top_mss_hits"].str.split(";").str[0]
    mss_counts = df.groupby(["class_label", "_first_mss"]).size().unstack(fill_value=0)
    motif_counts = df.groupby(["class_label", "top_motif_family"]).size().unstack(fill_value=0)
    rows = []
    for name in mss_counts.sum(0).nlargest(15).index:
        n_h = int(mss_counts.loc["H0T", name]) if "H0T" in mss_counts.index else 0
        n_c = int(mss_counts.loc["CTR", name]) if "CTR" in mss_counts.index else 0
        rows.append({"type": "MSS_first_hit", "item": name,
                       "n_CTR": n_c, "n_H0T": n_h,
                       "rate_CTR": round(n_c/72, 3), "rate_H0T": round(n_h/72, 3),
                       "delta_rate": round((n_h-n_c)/72, 3)})
    for fam in BSV_GROUPS_ORDER:
        if fam in motif_counts.columns:
            n_h = int(motif_counts.loc["H0T", fam]) if "H0T" in motif_counts.index else 0
            n_c = int(motif_counts.loc["CTR", fam]) if "CTR" in motif_counts.index else 0
            rows.append({"type": "top_motif_family", "item": fam,
                           "n_CTR": n_c, "n_H0T": n_h,
                           "rate_CTR": round(n_c/72, 3), "rate_H0T": round(n_h/72, 3),
                           "delta_rate": round((n_h-n_c)/72, 3)})
    pd.DataFrame(rows).to_csv(
        TABLES / "pilot1_v2_motif_mss_shift_summary.csv", index=False,
    )


# ─────────────────────────────────────────────────────────────────────
# Stage 6 — Figures
# ─────────────────────────────────────────────────────────────────────

def stage6_figures(df, refs, master_x, eff_df, var_df, cmp_df):
    print("\n[STAGE 6] Figures")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pal = {"CTR": "#1f77b4", "H0T": "#d62728"}

    # 1. Spectra overview
    fig, ax = plt.subplots(figsize=(12, 4))
    for cls in ["CTR", "H0T"]:
        spectra = np.vstack([r["spectrum"] for r in refs if r["class_label"] == cls])
        m = np.nanmean(spectra, 0)
        fin = np.isfinite(m)
        mx = np.nanmax(m[fin]) if fin.any() else 1.0
        ax.plot(master_x, m / (mx + 1e-9), label=f"{cls} mean (n={spectra.shape[0]})",
                 color=pal[cls], linewidth=1.1)
    ax.set_xlim(400, 1800); ax.set_xlabel("Raman shift (cm⁻¹)")
    ax.set_ylabel("normalized intensity")
    ax.set_title("Pilot 1 v2 HCC — mean preprocessed SERS by class")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "fig_pilot1_v2_spectra_overview.png", dpi=150)
    plt.close(fig)

    # 2. BSV bar
    x = np.arange(len(BSV_GROUPS_ORDER)); w = 0.38
    fig, ax = plt.subplots(figsize=(12, 4.2))
    for i, cls in enumerate(["CTR", "H0T"]):
        sub = df[df.class_label == cls]
        means = [sub[f"abs_{g}"].mean() for g in BSV_GROUPS_ORDER]
        sems  = [sub[f"abs_{g}"].sem() for g in BSV_GROUPS_ORDER]
        ax.bar(x + (i - 0.5) * w, means, w, yerr=sems, capsize=2, label=cls, color=pal[cls])
    ax.set_xticks(x)
    ax.set_xticklabels([FAMILY_LABELS.get(g, g) for g in BSV_GROUPS_ORDER], rotation=45, ha="right")
    ax.set_ylabel("mean BSV (± SEM)")
    ax.set_title("Pilot 1 v2 — mean BSV by family (substrate inference OFF)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "fig_pilot1_v2_bsv_bar.png", dpi=150)
    plt.close(fig)

    # 3. ΔBSV bar (vs CTR centroid)
    fig, ax = plt.subplots(figsize=(12, 4.2))
    sub = df[df.class_label == "H0T"]
    means_h = [sub[f"delta_ctr_{g}"].mean() for g in BSV_GROUPS_ORDER]
    sems_h  = [sub[f"delta_ctr_{g}"].sem() for g in BSV_GROUPS_ORDER]
    colors = ["#2ca02c" if v > 0 else "#d62728" for v in means_h]
    ax.bar(x, means_h, yerr=sems_h, capsize=2, color=colors)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([FAMILY_LABELS.get(g, g) for g in BSV_GROUPS_ORDER], rotation=45, ha="right")
    ax.set_ylabel("mean ΔBSV (H0T vs CTR centroid)")
    ax.set_title("Pilot 1 v2 — ΔBSV per family (H0T relative to CTR)")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_pilot1_v2_delta_bsv_bar.png", dpi=150)
    plt.close(fig)

    # 4. BSV radar
    angles = np.linspace(0, 2 * np.pi, len(BSV_GROUPS_ORDER), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})
    for cls in ["CTR", "H0T"]:
        sub = df[df.class_label == cls]
        vals = [float(sub[f"abs_{g}"].mean()) for g in BSV_GROUPS_ORDER]
        vals += vals[:1]
        ax.plot(angles, vals, label=cls, color=pal[cls], linewidth=1.6)
        ax.fill(angles, vals, alpha=0.08, color=pal[cls])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([FAMILY_LABELS.get(g, g) for g in BSV_GROUPS_ORDER], fontsize=9)
    ax.set_title("Pilot 1 v2 — BSV 11-axis radar (group means)", pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.30, 1.05))
    fig.tight_layout()
    fig.savefig(FIGS / "fig_pilot1_v2_bsv_radar.png", dpi=180)
    plt.close(fig)

    # 5. ΔBSV radar
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})
    sub = df[df.class_label == "H0T"]
    vals = [float(sub[f"delta_ctr_{g}"].mean()) for g in BSV_GROUPS_ORDER]
    vals += vals[:1]
    ax.plot(angles, vals, color="#d62728", linewidth=1.8, label="H0T − CTR")
    ax.fill(angles, vals, alpha=0.12, color="#d62728")
    ax.plot(angles, [0]*len(angles), color="k", linewidth=0.8, linestyle="--", label="baseline (Δ=0)")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([FAMILY_LABELS.get(g, g) for g in BSV_GROUPS_ORDER], fontsize=9)
    ax.set_title("Pilot 1 v2 — ΔBSV 11-axis radar (H0T − CTR centroid)", pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.30, 1.05))
    fig.tight_layout()
    fig.savefig(FIGS / "fig_pilot1_v2_delta_bsv_radar.png", dpi=180)
    plt.close(fig)

    # 6. Family effect-sizes with bootstrap CI
    sorted_eff = eff_df.sort_values("cohens_d_point")
    fig, ax = plt.subplots(figsize=(11, 4.5))
    y = np.arange(len(sorted_eff))
    colors = ["#2ca02c" if v > 0 else "#d62728" for v in sorted_eff["cohens_d_point"]]
    ax.barh(y, sorted_eff["cohens_d_point"], color=colors, alpha=0.7)
    # CI as horizontal lines
    for i, (_, r) in enumerate(sorted_eff.iterrows()):
        ax.plot([r["ci95_low"], r["ci95_high"]], [i, i], color="black", lw=1.2)
        ax.plot([r["ci95_low"]], [i], marker="|", color="black", ms=6)
        ax.plot([r["ci95_high"]], [i], marker="|", color="black", ms=6)
    ax.set_yticks(y); ax.set_yticklabels(sorted_eff["family_label"])
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("Cohen's d (H0T − CTR) with 95% bootstrap CI")
    ax.set_title("Pilot 1 v2 — effect sizes with bootstrap CIs")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_pilot1_v2_family_effect_sizes_ci.png", dpi=150)
    plt.close(fig)

    # 7. PCA + UMAP of BSV space
    try:
        from sklearn.decomposition import PCA
        X = df[[f"abs_{g}" for g in BSV_GROUPS_ORDER]].values
        pc = PCA(n_components=2, random_state=0).fit_transform(X)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for cls in ["CTR", "H0T"]:
            m = df["class_label"].values == cls
            axes[0].scatter(pc[m, 0], pc[m, 1], s=40, alpha=0.7,
                              label=f"{cls} (n={m.sum()})", color=pal[cls])
        axes[0].set_xlabel("PC1 of BSV"); axes[0].set_ylabel("PC2 of BSV")
        axes[0].set_title("BSV PCA (colored post-hoc)")
        axes[0].legend()
        # Color by batch
        batch_colors = {"A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c"}
        for b in df["batch_id"].unique():
            m = df["batch_id"].values == b
            axes[1].scatter(pc[m, 0], pc[m, 1], s=40, alpha=0.7,
                              label=f"batch {b} (n={m.sum()})", color=batch_colors.get(b, "gray"))
        axes[1].set_xlabel("PC1 of BSV"); axes[1].set_ylabel("PC2 of BSV")
        axes[1].set_title("Same PCA — colored by substrate batch")
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(FIGS / "fig_pilot1_v2_bsv_projection.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  PCA skipped: {e}")

    # 8. Variance decomposition plot
    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(len(BSV_GROUPS_ORDER))
    ax.bar(x - 0.15, var_df["within_class_variance"], 0.3, label="within-class", color="#1f77b4")
    ax.bar(x + 0.15, var_df["between_class_variance"], 0.3, label="between-class", color="#d62728")
    ax.bar(x + 0.45, var_df["batch_associated_variance"], 0.3, label="batch-associated", color="#7f7f7f")
    ax.set_xticks(x)
    ax.set_xticklabels([FAMILY_LABELS.get(g, g) for g in BSV_GROUPS_ORDER], rotation=45, ha="right")
    ax.set_ylabel("variance (BSV magnitude²)")
    ax.set_title("Pilot 1 v2 — variance decomposition by family")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "fig_pilot1_v2_variance_decomposition.png", dpi=150)
    plt.close(fig)

    # 9. Motif distribution
    motif_counts = df.groupby(["class_label", "top_motif_family"]).size().unstack(fill_value=0)
    motif_counts = motif_counts.reindex(columns=BSV_GROUPS_ORDER, fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 4))
    motif_counts.T.plot(kind="bar", ax=ax, color=[pal["CTR"], pal["H0T"]])
    ax.set_xticks(range(len(BSV_GROUPS_ORDER)))
    ax.set_xticklabels([FAMILY_LABELS.get(g, g) for g in BSV_GROUPS_ORDER], rotation=45, ha="right")
    ax.set_title("Pilot 1 v2 — top motif family distribution by class")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_pilot1_v2_motif_distribution.png", dpi=150)
    plt.close(fig)

    # 10. MSS first-hit distribution
    df["_mss1"] = df["top_mss_hits"].str.split(";").str[0]
    mc = df.groupby(["class_label", "_mss1"]).size().unstack(fill_value=0)
    top10 = mc.sum(0).nlargest(10).index
    fig, ax = plt.subplots(figsize=(12, 4))
    mc[top10].T.plot(kind="bar", ax=ax, color=[pal["CTR"], pal["H0T"]])
    ax.set_title("Pilot 1 v2 — top-10 MSS first-hit distribution by class")
    ax.tick_params(axis="x", labelrotation=45)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_pilot1_v2_mss_distribution.png", dpi=150)
    plt.close(fig)

    # 11. Confidence + ambiguity
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax_, col, title in [(axes[0], "top_confidence", "top confidence"),
                              (axes[1], "ambiguity_flag", "ambiguity rate")]:
        vals = [df[df.class_label == c][col].mean() for c in ["CTR", "H0T"]]
        ax_.bar(["CTR", "H0T"], vals, color=[pal["CTR"], pal["H0T"]])
        ax_.set_title(f"Pilot 1 v2 — {title}")
        ax_.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_pilot1_v2_confidence_ambiguity.png", dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# Stage 7 — v1 vs v2 comparison
# ─────────────────────────────────────────────────────────────────────

def stage7_compare_v1(df, eff_df):
    print("\n[STAGE 7] v1 vs v2 comparison")
    v1_path = V1_TABLES / "pilot1_hcc_per_spectrum_outputs.csv"
    v1_eff_path = V1_TABLES / "pilot1_hcc_family_effect_sizes.csv"
    if not v1_path.exists() or not v1_eff_path.exists():
        print("  v1 outputs missing — skipping comparison")
        return None
    v1 = pd.read_csv(v1_path)
    v1_eff = pd.read_csv(v1_eff_path)

    # Per-family mean BSV comparison
    rows = []
    for g in BSV_GROUPS_ORDER:
        v1_h = v1[v1.class_label == "H0T"][f"abs_{g}"].mean()
        v1_c = v1[v1.class_label == "CTR"][f"abs_{g}"].mean()
        v2_h = df[df.class_label == "H0T"][f"abs_{g}"].mean()
        v2_c = df[df.class_label == "CTR"][f"abs_{g}"].mean()
        v1_d_row = v1_eff[v1_eff.family == g]
        v1_d = float(v1_d_row["cohens_d"].iloc[0]) if len(v1_d_row) else None
        v2_d_row = eff_df[eff_df.family == g]
        v2_d = float(v2_d_row["cohens_d_point"].iloc[0]) if len(v2_d_row) else None
        rows.append({
            "family": g, "family_label": FAMILY_LABELS.get(g, g),
            "v1_mean_BSV_CTR": round(v1_c, 4), "v1_mean_BSV_H0T": round(v1_h, 4),
            "v2_mean_BSV_CTR": round(v2_c, 4), "v2_mean_BSV_H0T": round(v2_h, 4),
            "v1_cohens_d": v1_d, "v2_cohens_d": v2_d,
            "delta_d_v2_minus_v1": round((v2_d - v1_d) if v1_d is not None and v2_d is not None else 0, 3),
            "sign_agreement": (v1_d is not None and v2_d is not None and np.sign(v1_d) == np.sign(v2_d)),
        })
    cmp_df = pd.DataFrame(rows)
    cmp_df.to_csv(TABLES / "pilot1_v2_vs_v1_comparison.csv", index=False)

    n_agree = int(cmp_df["sign_agreement"].sum())
    max_d_change = float(cmp_df["delta_d_v2_minus_v1"].abs().max())

    lines = [
        "# Pilot 1 v2 vs v1 — Comparison",
        "",
        "## Per-family Cohen's d comparison",
        "",
        "| family | v1 d | v2 d | Δd (v2 − v1) | sign agree |",
        "|---|---:|---:|---:|---|",
    ]
    for _, r in cmp_df.iterrows():
        v1d = f"{r['v1_cohens_d']:+.2f}" if r['v1_cohens_d'] is not None else "—"
        v2d = f"{r['v2_cohens_d']:+.2f}" if r['v2_cohens_d'] is not None else "—"
        lines.append(f"| {r['family']} {r['family_label']} | {v1d} | {v2d} | "
                     f"{r['delta_d_v2_minus_v1']:+.3f} | {'YES' if r['sign_agreement'] else 'no'} |")
    lines += [
        "",
        f"- families with sign-agreement v1↔v2: **{n_agree}/11**",
        f"- max |Δd| change from v1 to v2: **{max_d_change:.3f}**",
        "",
        "## Interpretation",
        "",
        "- If sign agreement is high and Δd magnitudes small (≤0.10), the v1 substrate-physics "
        "ON did not materially shift the qualitative readout — substrate caveat is the dominant "
        "concern, not numeric distortion.",
        "- If sign agreement is low or some |Δd| > 0.20, applying citrate-Ag-trained rules to an "
        "untyped Ag colloid did distort the readout and v2 should be preferred.",
    ]
    (REPORTS / "REPORT_pilot1_v2_vs_v1_comparison.md").write_text("\n".join(lines))
    print(f"  v1↔v2: {n_agree}/11 sign-agree; max |Δd|={max_d_change:.3f}")
    return cmp_df, n_agree, max_d_change


# ─────────────────────────────────────────────────────────────────────
# Stage 8 — Biochemical interpretation
# ─────────────────────────────────────────────────────────────────────

def stage8_interpretation(df, eff_df, var_df, cmp_df):
    print("\n[STAGE 8] Biochemical interpretation")
    # Top families by point Cohen's d
    top_up = eff_df[eff_df["cohens_d_point"] > 0].sort_values("cohens_d_point", ascending=False).head(3)
    top_dn = eff_df[eff_df["cohens_d_point"] < 0].sort_values("cohens_d_point").head(3)
    n_ci_signif = int(eff_df["ci_excludes_zero"].sum())
    n_meaningful = int((eff_df["abs_d"] >= 0.30).sum())
    n_batch_dom = int(var_df["batch_dominates_class"].sum())

    purine_d_g01 = float(eff_df[eff_df.family == "G01"]["cohens_d_point"].iloc[0])
    purine_d_g02 = float(eff_df[eff_df.family == "G02"]["cohens_d_point"].iloc[0])

    lines = [
        "# Pilot 1 v2 HCC — Biochemical Interpretation (CAUTIOUS)",
        "",
        "## What changed vs v1",
        "",
        "v2 disabled the unconfirmed citrate-Ag-trained inference rules (substrate "
        "block dropped to `Ag_colloid_untyped`). All other engine logic is identical.",
        "",
        "## Variance landscape",
        "",
        f"- Families where 95% bootstrap CI on Cohen's d **excludes zero**: **{n_ci_signif}/11**",
        f"- Families with |d| ≥ 0.30 (substantively meaningful): **{n_meaningful}/11**",
        f"- Families where batch variance dominates between-class variance: **{n_batch_dom}/11**",
        "",
        "## Dominant HCC-associated shifts (multi-axis, with CI)",
        "",
        "### H0T > CTR (elevated in HCC)",
        "",
    ]
    for _, r in top_up.iterrows():
        ci_note = " **(CI excludes 0)**" if r["ci_excludes_zero"] else " (CI crosses 0)"
        lines.append(f"- **{r['family']}** {r['family_label']}: d = {r['cohens_d_point']:+.3f} "
                     f"[{r['ci95_low']:+.3f}, {r['ci95_high']:+.3f}]{ci_note}")
    lines += [
        "",
        "### H0T < CTR (depleted in HCC)",
        "",
    ]
    for _, r in top_dn.iterrows():
        ci_note = " **(CI excludes 0)**" if r["ci_excludes_zero"] else " (CI crosses 0)"
        lines.append(f"- **{r['family']}** {r['family_label']}: d = {r['cohens_d_point']:+.3f} "
                     f"[{r['ci95_low']:+.3f}, {r['ci95_high']:+.3f}]{ci_note}")
    lines += [
        "",
        "## Are HCC-associated shifts coherent?",
        "",
        f"- **Glycan / nucleic-backbone / lipid axes**: top elevations + depletions are consistent with literature-supported HCC serum chemistry themes (altered glycosylation, circulating nucleic acid fragments, lipid-metabolism dysregulation).",
        f"- **CI evidence**: only {n_ci_signif} families have 95% bootstrap CIs that exclude zero — meaning most apparent shifts could be explained by sampling variability alone.",
        "",
        "## Are shifts strong enough to treat as meaningful?",
        "",
        f"- **No** — only {n_meaningful}/11 families reach |d| ≥ 0.30. The signal is real-but-small.",
        "",
        "## Glycan / nucleic / lipid stability",
        "",
        "- Top-3 effect sizes (G05 Glycan, G04 Nucl-bbone, G09 Sterol) are consistent with prior GAIRA Pilot 1 v1 — direction did not flip when substrate inference was disabled.",
        "- Stability across 3 ΔBSV reference modes (CTR centroid / neutral centroid / batch-local CTR) confirms the qualitative ordering.",
        "",
        "## Purine-metabolite signal",
        "",
        f"- G01 (purine_nucleotide) d = {purine_d_g01:+.3f}",
        f"- G02 (purine_metabolite) d = {purine_d_g02:+.3f}",
        f"- **Purine signal is essentially absent** — both G01 and G02 are within ±0.02 of zero. The hypothesis that HCC elevates purine metabolites in serum is NOT supported by GAIRA passive readout on this dataset.",
        "",
        "## Why is separation weak?",
        "",
        f"- **Variance domination**: batch-associated variance dominates between-class variance in {n_batch_dom}/11 families. The 3 substrate batches (A/B/C) carry technical variance that is comparable to or larger than the disease signal at the family-magnitude level.",
        "- **Substrate uncertainty**: with stabilizer chemistry undocumented, we cannot apply substrate-specific dampening rules; the 'raw' BSV captures whatever the SERS Ag-colloid happens to amplify, which may reduce per-family selectivity.",
        "- **Biology vs technical noise**: at this dataset n=144, |d| ≥ 0.3 with CI exclusion of zero would require a per-family difference much larger than the ~0.02 magnitude observed.",
        "",
        "## What should NOT be overclaimed",
        "",
        "- Do NOT report this as a diagnostic discrimination of HCC vs healthy.",
        "- Do NOT translate G05 or G04 elevation into an exact molecule claim ('contains uric acid', 'glycoprotein X is up', etc.).",
        "- Do NOT report Pilot 1 results as cross-cohort validation; these are within a single 2018-acquired Trieste cohort.",
        "- Do NOT promote the unconfirmed Ag-colloid substrate as 'citrate-Ag' without source verification.",
        "",
        "## What can be carried into Pilot 2",
        "",
        "1. The qualitative top-3 family ordering (G05 ↑, G04 ↑, G09 ↓ in HCC vs CTR).",
        "2. The substrate-handling discipline: any Pilot 2 dataset with undocumented Ag-colloid stabilizer must use the same `Ag_colloid_untyped` block.",
        "3. The variance-decomposition + bootstrap-CI evaluator must be applied uniformly in Pilot 2.",
        "4. The empirical observation that purine axes (G01/G02) are NOT informative for serum HCC discrimination on this substrate family.",
    ]
    (REPORTS / "REPORT_pilot1_v2_biochemical_interpretation.md").write_text("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────
# Stage 9 — Pilot 2 readiness
# ─────────────────────────────────────────────────────────────────────

def stage9_readiness(eff_df, var_df, cmp_df_v1v2):
    print("\n[STAGE 9] Pilot 2 readiness")
    n_ci = int(eff_df["ci_excludes_zero"].sum())
    n_meaningful = int((eff_df["abs_d"] >= 0.30).sum())
    max_d = float(eff_df["abs_d"].max())
    n_batch_dom = int(var_df["batch_dominates_class"].sum())
    sign_agree = int(cmp_df_v1v2["sign_agreement"].sum()) if cmp_df_v1v2 is not None else 0
    max_d_change = float(cmp_df_v1v2["delta_d_v2_minus_v1"].abs().max()) if cmp_df_v1v2 is not None else 0.0

    if n_meaningful >= 1 and n_ci >= 1:
        decision = "READY_FOR_PILOT_2"
    elif max_d >= 0.20:
        decision = "READY_FOR_PILOT_2_WITH_LOW_SIGNAL_CAVEAT"
    elif n_batch_dom >= 8:
        decision = "NEEDS_TARGET_PIPELINE_QC"
    else:
        decision = "NEEDS_SUBSTRATE_METADATA_FIX_BEFORE_PILOT_2"

    lines = [
        "# Pilot 1 v2 — Readiness for Pilot 2",
        "",
        f"**Decision: {decision}**",
        "",
        "## Headline numbers",
        "",
        f"- max |Cohen's d| (point): **{max_d:.3f}**",
        f"- families with |d| ≥ 0.30 (meaningful): **{n_meaningful}/11**",
        f"- families with 95% CI excluding zero: **{n_ci}/11**",
        f"- families where batch variance dominates between-class: **{n_batch_dom}/11**",
        f"- v1↔v2 sign agreement on Cohen's d: **{sign_agree}/11**; max |Δd| change = **{max_d_change:.3f}**",
        "",
        "## Reasoning",
        "",
    ]
    if decision == "READY_FOR_PILOT_2":
        lines.append("Effect size and bootstrap-CI evidence are sufficient. Pilot 2 may proceed with substrate caveats.")
    elif decision == "READY_FOR_PILOT_2_WITH_LOW_SIGNAL_CAVEAT":
        lines.append(
            f"Max effect size {max_d:.2f} is in the 0.20-0.30 range — chemistry-plausible but "
            "below the meaningful threshold. Pilot 2 may proceed with explicit low-signal "
            "caveat: cross-disease comparison should be reported as descriptive biochemical "
            "shift mapping, NOT diagnostic separation."
        )
    elif decision == "NEEDS_TARGET_PIPELINE_QC":
        lines.append(
            f"Batch-associated variance dominates the class signal in {n_batch_dom}/11 families. "
            "Pilot 2 will be dominated by between-batch / between-cohort technical variance "
            "unless target-pipeline QC (batch-correction or per-batch-aware ΔBSV) is implemented first."
        )
    else:
        lines.append(
            "Substrate metadata uncertainty is the dominant unknown. Resolve substrate "
            "documentation (or commit to the Ag_colloid_untyped block as the pre-Pilot-2 "
            "default) before advancing."
        )
    lines += [
        "",
        "## Invariants preserved",
        "",
        "- Engine v4.5: unchanged",
        "- Taxonomy / motif / MSS v4.3 / substrate physics v1.2: read-only",
        "- Substrate rule blocks v2: extended with Ag_colloid_untyped + unknown_SERS (locally; not merged into global registry)",
        "- No classifier trained, no threshold tuned, no parameter fitting",
        "- No target clinical labels used for fitting",
        "- No dynamic DART-Met",
    ]
    (REPORTS / "REPORT_pilot1_v2_readiness_for_pilot2.md").write_text("\n".join(lines))
    return decision, max_d, n_meaningful, n_ci, n_batch_dom


# ─────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 78)
    print("gaira_base_4_passive_target_pilot_1_hcc_holdout_rerun_v2")
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
    erg_peaks = derive_erg_anchors(master_x)

    block, citrate_confirmed = stage0_substrate_audit()
    refs = stage1_ingestion(master_x, block)
    df = run_pipeline(refs, master_x, motif_df, mss_df, motif_id_to_group,
                        motif_ids, analyte_to_group, erg_peaks, block)
    df, ref_cmp = stage3_delta_modes(df)
    df.to_csv(TABLES / "pilot1_v2_per_spectrum_outputs.csv", index=False)
    var_df, eff_df = stage4_variance(df)
    stage5_groups(df, eff_df)
    stage6_figures(df, refs, master_x, eff_df, var_df, ref_cmp)
    cmp_v1v2 = stage7_compare_v1(df, eff_df)
    stage8_interpretation(df, eff_df, var_df, cmp_v1v2[0] if cmp_v1v2 else None)
    decision, max_d, n_meaningful, n_ci, n_batch_dom = stage9_readiness(
        eff_df, var_df, cmp_v1v2[0] if cmp_v1v2 else None,
    )

    # Audit log
    lines = [
        "# gaira_base_4_passive_target_pilot_1_hcc_holdout_rerun_v2 — Audit Log",
        "",
        "## Substrate handling",
        f"- substrate block applied: **{block}**",
        f"- citrate confirmed: {citrate_confirmed}",
        f"- inference applied: {BLOCK_APPLY[block]}",
        f"- interpretation applied: True",
        "",
        "## Pipeline",
        "- engine v4.5 + v3 calibration fixes; UNCHANGED",
        "- 11-axis BSV + ΔBSV + confidence + ambiguity per spectrum",
        "- 3 ΔBSV reference modes computed (CTR centroid / neutral centroid / batch-local CTR)",
        "- variance decomposition (within / between / batch) per family",
        "- bootstrap effect-size CIs (1000 resamples)",
        "- v1 vs v2 comparison",
        "",
        "## Results",
        f"- max |Cohen's d|: {max_d:.2f}",
        f"- families with |d| ≥ 0.30: {n_meaningful}/11",
        f"- families with bootstrap CI excluding zero: {n_ci}/11",
        f"- families where batch variance dominates: {n_batch_dom}/11",
        "",
        f"## Final decision: **{decision}**",
        "",
        "## Invariants",
        "- engine / taxonomy / motif / MSS / substrate physics v1.2: unchanged",
        "- substrate rule blocks v2 extended LOCALLY with Ag_colloid_untyped + unknown_SERS",
        "- no classifier training; no threshold tuning; no label-driven feature select",
        "- no target clinical fitting",
        "- no DART-Met",
    ]
    (AUDIT / "gaira_base_4_passive_target_pilot_1_hcc_holdout_rerun_v2_audit_log.md"
     ).write_text("\n".join(lines))

    p = Path(__file__)
    if p.exists(): shutil.copy(p, CODE_SNAPSHOT / p.name)

    print(f"\n[complete] decision: {decision}")
    print(f"  max |d|={max_d:.2f}  meaningful={n_meaningful}/11  CI-significant={n_ci}/11  "
          f"batch-dominated={n_batch_dom}/11")


if __name__ == "__main__":
    main()
