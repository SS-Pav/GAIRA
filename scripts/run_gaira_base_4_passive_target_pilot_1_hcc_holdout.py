"""gaira_base_4 passive target Pilot 1 — HCC holdout cohort.

Passive readout only. NO parameter fitting, NO target-label-driven feature
selection, NO threshold tuning. Labels used only AFTER inference for
group-level descriptive comparison.

Engine: v4.5 locked + v3 controlled-calibration fixes active:
  - 11-axis BSV + ΔBSV emitted per spectrum
  - substrate-block gated SERS physics (citrate-Ag trained block applies)
  - SENSITIVE-tier output policy
  - ERG MSS auxiliary template available (not used here — no ERG spike)

Dataset:
  /Volumes/SSD_Rad/GAIRA_DATA/raw/hcc_serum/data.csv
  Gurian / Bonifacio 2020. 144 spectra. 72 CTR + 72 H0T (HCC).
  SERS on Ag colloid (PCA-LDA paper baseline). 3 substrate batches A/B/C.
"""
from __future__ import annotations

import shutil
import sys
import warnings
from collections import Counter
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
    substrate_block_for, SUBSTRATE_BLOCKS, derive_erg_anchors,
)


ROOT = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/"
    "gaira_base_4_passive_target_pilot_1_hcc_holdout"
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
HCC_CSV = Path("/Volumes/SSD_Rad/GAIRA_DATA/raw/hcc_serum/data.csv")

BSV_GROUPS_ORDER = [g["group_id"] for g in BSV_GROUPS]


# ─────────────────────────────────────────────────────────────────────
# Stage 1 — Load + ingestion audit
# ─────────────────────────────────────────────────────────────────────

def stage1_load(master_x):
    print("\n[STAGE 1] HCC holdout ingestion audit")
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
            "class_label": row["class"],  # "CTR" or "H0T"
            "substrate_batch": row["substrate_batch"],
            "acquisition_date": row["acquisition_date"],
            "regime": "SERS",
            "substrate_family": "Ag colloid (HCC serum SERS, Gurian 2020)",
            "spectrum": y_rs,
        })

    audit_rows = [{
        "dataset_path": str(HCC_CSV),
        "n_spectra": len(refs),
        "n_CTR": sum(1 for r in refs if r["class_label"] == "CTR"),
        "n_H0T": sum(1 for r in refs if r["class_label"] == "H0T"),
        "substrate_batches": ";".join(sorted(set(r["substrate_batch"] for r in refs))),
        "regime": "SERS",
        "substrate_family": "Ag colloid (serum SERS; Gurian/Bonifacio 2020)",
        "substrate_block_assigned": substrate_block_for("Ag colloid (HCC serum SERS, Gurian 2020)"),
        "spectral_range_cm1": f"{wn.min():.1f} to {wn.max():.1f}",
        "n_wavenumber_columns": len(wn_cols),
        "paper_reference": "Gurian et al. 2020, PCA-LDA of SERS serum for HCC (Bonifacio lab)",
        "preprocessing_compatibility": "resampled to GAIRA canonical master axis via linear interp",
        "missing_metadata_notes": "exact Ag-colloid substrate variant not specified; substrate block assigned to citrate_Ag_colloid_trained based on paper context but marked with substrate_variant_caveat",
        "passive_readout_only": True,
        "labels_used_for_fitting": False,
    }]
    pd.DataFrame(audit_rows).to_csv(
        TABLES / "pilot1_hcc_ingestion_audit.csv", index=False,
    )
    r = audit_rows[0]
    print(f"  {r['n_spectra']} spectra ({r['n_CTR']} CTR + {r['n_H0T']} H0T) across batches {r['substrate_batches']}")

    lines = [
        "# Pilot 1 HCC Holdout — Ingestion Audit",
        "",
        "## Dataset",
        "",
        f"- Path: `{HCC_CSV}`",
        f"- Paper: Gurian et al. 2020 (Bonifacio lab, University of Trieste)",
        f"  — *PCA-LDA of SERS spectra for HCC serum classification*",
        f"- n_spectra: **{r['n_spectra']}** (72 CTR + 72 H0T)",
        f"- Substrate batches: **{r['substrate_batches']}** (A=51, B=47, C=46)",
        f"- Regime: SERS",
        f"- Substrate family: Ag colloid (serum SERS)",
        f"- Substrate block assigned: `{r['substrate_block_assigned']}` "
        "(trained block; caveat that exact Ag-colloid variant is not documented in "
        "the paper's publicly available metadata — could be citrate-reduced Ag or a "
        "related Ag-colloid prep)",
        f"- Spectral range: {r['spectral_range_cm1']} cm⁻¹",
        f"- n wavenumber columns: {r['n_wavenumber_columns']}",
        "",
        "## Preprocessing",
        "",
        "- data.csv is pre-preprocessed by the original authors (Gurian et al.) — baseline "
        "correction + normalisation per their R pipeline.",
        "- GAIRA step: linear interpolation onto the canonical GAIRA master axis for "
        "pipeline compatibility. No additional preprocessing applied.",
        "",
        "## Passive-readout rules enforced",
        "",
        "- Labels (CTR / H0T) used ONLY for group-level descriptive comparison AFTER inference.",
        "- No classifier training.",
        "- No threshold tuning on target labels.",
        "- No feature selection using target labels.",
        "- No parameter fitting.",
        "",
        "## Caveats",
        "",
        "- Substrate variant not explicitly documented — controlled calibration on citrate-Ag colloid provides the trained baseline, but cross-batch reproducibility (A/B/C) should be inspected.",
        "- All predictions carry SENSITIVE-tier output policy + substrate caveat.",
    ]
    (REPORTS / "REPORT_pilot1_hcc_ingestion_audit.md").write_text("\n".join(lines))
    return refs


# ─────────────────────────────────────────────────────────────────────
# Stage 2 — Run pipeline + Stage 3 — ΔBSV reference
# ─────────────────────────────────────────────────────────────────────

def run_pipeline(refs, master_x, motif_df, mss_df, motif_id_to_group,
                   motif_ids, analyte_to_group, erg_peaks):
    # Substrate block
    block = substrate_block_for(refs[0]["substrate_family"])
    block_apply = {b["block_id"]: b["apply_for_inference"] for b in SUBSTRATE_BLOCKS}
    apply_sers = block_apply.get(block, False)
    print(f"\n[STAGE 2] Running pipeline: substrate_block={block}, "
          f"apply_sers_physics={apply_sers}")

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

        # ERG aux score (not expected to matter for HCC but computed for completeness)
        fin = np.isfinite(r["spectrum"])
        sp_max = float(np.max(r["spectrum"][fin])) if fin.any() else 1.0
        erg_vals = []
        for cm, _ in erg_peaks[:6]:
            idx = int(np.argmin(np.abs(master_x - cm)))
            w = r["spectrum"][max(0, idx - 4):idx + 5]
            erg_vals.append(float(np.nanmax(w)) / max(sp_max, 1e-9))
        erg_score = float(np.mean(erg_vals)) if erg_vals else 0.0

        row = {
            "spectrum_id": r["spectrum_id"],
            "sample_code": r["sample_code"],
            "class_label": r["class_label"],
            "substrate_batch": r["substrate_batch"],
            "acquisition_date": r["acquisition_date"],
            "regime": r["regime"],
            "substrate_family": r["substrate_family"],
            "substrate_block": block,
            "apply_sers_physics": apply_sers,
            "preprocessing_tag": "gurian2020_baselined + gaira_canonical_resample",
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
            "interpretation_tier": "SENSITIVE_SERS_SUBSTRATE_CAVEAT",
            "erg_aux_score": round(erg_score, 4),
        }
        row.update({f"abs_{g}": bsv_vec[g] for g in BSV_GROUPS_ORDER})
        row.update({f"conf_{g}": conf_vec[g] for g in BSV_GROUPS_ORDER})
        rows.append(row)
    return pd.DataFrame(rows)


def stage3_delta_reference(df):
    print("\n[STAGE 3] Δ reference: CTR centroid")
    ctr_mask = df["class_label"] == "CTR"
    n_ctr = int(ctr_mask.sum())
    ctr_means = df.loc[ctr_mask, [f"abs_{g}" for g in BSV_GROUPS_ORDER]].mean()
    for g in BSV_GROUPS_ORDER:
        df[f"delta_{g}"] = df[f"abs_{g}"] - ctr_means[f"abs_{g}"]
    # Top-3 ΔBSV-changing families per spectrum
    def _t3(row):
        scores = [(g, row[f"delta_{g}"]) for g in BSV_GROUPS_ORDER]
        scores.sort(key=lambda kv: -abs(kv[1]))
        return ";".join(f"{g}:{v:+.3f}" for g, v in scores[:3])
    df["top3_delta_changing_families"] = df.apply(_t3, axis=1)
    df["delta_bsv_vector_11axis"] = df.apply(
        lambda r: ";".join(f"{g}:{round(r[f'delta_{g}'], 4)}" for g in BSV_GROUPS_ORDER),
        axis=1,
    )

    # Emit reference registry
    ref_rows = [{
        "reference_mode": "CTR cohort centroid (per-family mean BSV magnitude)",
        "n_control_spectra": n_ctr,
        **{f"ctr_mean_{g}": round(float(ctr_means[f'abs_{g}']), 4)
             for g in BSV_GROUPS_ORDER},
    }]
    pd.DataFrame(ref_rows).to_csv(
        TABLES / "pilot1_hcc_delta_reference_registry.csv", index=False,
    )

    lines = [
        "# Pilot 1 HCC Holdout — ΔBSV Reference",
        "",
        f"## Reference mode: **CTR cohort centroid**",
        "",
        f"- CTR n = {n_ctr} spectra",
        f"- Per-family mean BSV magnitudes used as the subtraction baseline:",
        "",
        "| family | CTR mean BSV |",
        "|---|---:|",
    ]
    for g in BSV_GROUPS_ORDER:
        lines.append(f"| {g} {FAMILY_LABELS.get(g, g)} | {ctr_means[f'abs_{g}']:.4f} |")
    lines += [
        "",
        "## Fallback policy (not triggered here)",
        "",
        "- If CTR cohort absent or n<10: use the trained citrate-Ag SERS "
        "neutral-centroid (mean across ramanbiolib + gobbato + sers_metabolite_63 "
        "standard grounding refs). CTR cohort is the preferred reference when present.",
    ]
    (REPORTS / "REPORT_pilot1_hcc_delta_reference.md").write_text("\n".join(lines))
    return df


# ─────────────────────────────────────────────────────────────────────
# Stage 4 — Group-level analysis
# ─────────────────────────────────────────────────────────────────────

def stage4_group_analysis(df):
    print("\n[STAGE 4] Group-level state-vector analysis")
    # Mean BSV + ΔBSV per group
    grp = df.groupby("class_label")
    bsv_means = grp[[f"abs_{g}" for g in BSV_GROUPS_ORDER]].mean().round(4)
    bsv_stds  = grp[[f"abs_{g}" for g in BSV_GROUPS_ORDER]].std(ddof=1).round(4)
    delta_means = grp[[f"delta_{g}" for g in BSV_GROUPS_ORDER]].mean().round(4)
    conf_means = grp[[f"conf_{g}" for g in BSV_GROUPS_ORDER]].mean().round(4)
    amb_rates  = grp["ambiguity_flag"].mean().round(3)

    # Effect size per family: Cohen's d = (μ_H0T − μ_CTR) / pooled_std
    rows = []
    for g in BSV_GROUPS_ORDER:
        m_h = df[df.class_label == "H0T"][f"abs_{g}"]
        m_c = df[df.class_label == "CTR"][f"abs_{g}"]
        if m_h.std() == 0 and m_c.std() == 0: d = 0.0
        else:
            pooled = np.sqrt(((len(m_h) - 1) * m_h.var(ddof=1) +
                                (len(m_c) - 1) * m_c.var(ddof=1)) /
                               max(len(m_h) + len(m_c) - 2, 1))
            d = (m_h.mean() - m_c.mean()) / (pooled if pooled > 0 else 1.0)
        rows.append({
            "family": g, "family_label": FAMILY_LABELS.get(g, g),
            "mean_BSV_CTR": round(float(m_c.mean()), 4),
            "mean_BSV_H0T": round(float(m_h.mean()), 4),
            "mean_delta_BSV_H0T": round(float(delta_means.loc["H0T", f"delta_{g}"]), 4),
            "cohens_d": round(float(d), 3),
            "abs_cohens_d": round(abs(float(d)), 3),
            "direction": "H0T>CTR" if d > 0 else ("H0T<CTR" if d < 0 else "equal"),
        })
    eff_df = pd.DataFrame(rows).sort_values("abs_cohens_d", ascending=False)
    eff_df.to_csv(TABLES / "pilot1_hcc_family_effect_sizes.csv", index=False)

    # Group BSV summary
    gbrows = []
    for cls in ["CTR", "H0T"]:
        sub = df[df.class_label == cls]
        for g in BSV_GROUPS_ORDER:
            gbrows.append({
                "class": cls, "family": g,
                "mean_BSV": round(float(sub[f"abs_{g}"].mean()), 4),
                "std_BSV": round(float(sub[f"abs_{g}"].std(ddof=1)), 4),
                "mean_confidence": round(float(sub[f"conf_{g}"].mean()), 4),
                "n": len(sub),
            })
    pd.DataFrame(gbrows).to_csv(TABLES / "pilot1_hcc_group_bsv_summary.csv", index=False)

    gdrows = []
    for cls in ["CTR", "H0T"]:
        sub = df[df.class_label == cls]
        for g in BSV_GROUPS_ORDER:
            gdrows.append({
                "class": cls, "family": g,
                "mean_delta_BSV": round(float(sub[f"delta_{g}"].mean()), 4),
                "std_delta_BSV": round(float(sub[f"delta_{g}"].std(ddof=1)), 4),
                "n": len(sub),
            })
    pd.DataFrame(gdrows).to_csv(TABLES / "pilot1_hcc_group_delta_bsv_summary.csv", index=False)

    # Motif + MSS enrichment
    df["_first_mss"] = df["top_mss_hits"].str.split(";").str[0]
    df["_first_motif_fam"] = df["top_motif_family"]
    mss_counts = df.groupby(["class_label", "_first_mss"]).size().unstack(fill_value=0)
    motif_counts = df.groupby(["class_label", "_first_motif_fam"]).size().unstack(fill_value=0)

    enr_rows = []
    # MSS enrichment: Δ frequency H0T vs CTR on top 15 combined
    top15_mss = mss_counts.sum(0).nlargest(15).index.tolist()
    for name in top15_mss:
        n_h = int(mss_counts.loc["H0T", name]) if "H0T" in mss_counts.index else 0
        n_c = int(mss_counts.loc["CTR", name]) if "CTR" in mss_counts.index else 0
        enr_rows.append({
            "type": "MSS_first_hit", "item": name,
            "n_CTR": n_c, "n_H0T": n_h,
            "rate_CTR": round(n_c / 72, 3),
            "rate_H0T": round(n_h / 72, 3),
            "delta_rate": round((n_h - n_c) / 72, 3),
        })
    for fam in BSV_GROUPS_ORDER:
        if fam in motif_counts.columns:
            n_h = int(motif_counts.loc["H0T", fam]) if "H0T" in motif_counts.index else 0
            n_c = int(motif_counts.loc["CTR", fam]) if "CTR" in motif_counts.index else 0
            enr_rows.append({
                "type": "top_motif_family", "item": fam,
                "n_CTR": n_c, "n_H0T": n_h,
                "rate_CTR": round(n_c / 72, 3),
                "rate_H0T": round(n_h / 72, 3),
                "delta_rate": round((n_h - n_c) / 72, 3),
            })
    pd.DataFrame(enr_rows).to_csv(
        TABLES / "pilot1_hcc_motif_mss_enrichment.csv", index=False,
    )

    print(f"  top-3 effect sizes: "
          f"{', '.join(f'{r.family}(d={r.cohens_d:+.2f})' for r in eff_df.head(3).itertuples())}")
    return eff_df, bsv_means, delta_means, amb_rates


# ─────────────────────────────────────────────────────────────────────
# Stage 5 — Figures
# ─────────────────────────────────────────────────────────────────────

def _palette(): return {"CTR": "#1f77b4", "H0T": "#d62728"}


def stage5_figures(df, refs, master_x, eff_df):
    print("\n[STAGE 5] Figures")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pal = _palette()

    # 1. spectra overview — mean spectrum per class
    fig, ax = plt.subplots(figsize=(12, 4))
    for cls in ["CTR", "H0T"]:
        spectra = np.vstack([r["spectrum"] for r in refs if r["class_label"] == cls])
        mean_spec = np.nanmean(spectra, 0)
        fin = np.isfinite(mean_spec)
        mx = np.nanmax(mean_spec[fin]) if fin.any() else 1.0
        ax.plot(master_x, mean_spec / (mx + 1e-9),
                 label=f"{cls} mean (n={spectra.shape[0]})",
                 color=pal[cls], linewidth=1.1)
    ax.set_xlim(400, 1800); ax.set_xlabel("Raman shift (cm⁻¹)")
    ax.set_ylabel("normalized intensity")
    ax.set_title("HCC holdout — mean preprocessed SERS spectra by class")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_pilot1_hcc_spectra_overview.png", dpi=150)
    plt.close(fig)

    # 2. Mean BSV bar by class
    fig, ax = plt.subplots(figsize=(12, 4.2))
    x = np.arange(len(BSV_GROUPS_ORDER))
    w = 0.38
    for i, cls in enumerate(["CTR", "H0T"]):
        sub = df[df.class_label == cls]
        means = [sub[f"abs_{g}"].mean() for g in BSV_GROUPS_ORDER]
        stds = [sub[f"abs_{g}"].std(ddof=1) for g in BSV_GROUPS_ORDER]
        ax.bar(x + (i - 0.5) * w, means, w, yerr=stds, capsize=2,
                label=cls, color=pal[cls])
    ax.set_xticks(x)
    ax.set_xticklabels([FAMILY_LABELS.get(g, g) for g in BSV_GROUPS_ORDER],
                        rotation=45, ha="right")
    ax.set_ylabel("mean BSV magnitude")
    ax.set_title("Pilot 1 HCC — mean BSV by family (CTR vs H0T)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "fig_pilot1_hcc_bsv_bar.png", dpi=150)
    plt.close(fig)

    # 3. Mean ΔBSV bar
    fig, ax = plt.subplots(figsize=(12, 4.2))
    means_h = df[df.class_label == "H0T"][[f"delta_{g}" for g in BSV_GROUPS_ORDER]].mean()
    stds_h  = df[df.class_label == "H0T"][[f"delta_{g}" for g in BSV_GROUPS_ORDER]].std(ddof=1)
    colors = ["#2ca02c" if v > 0 else "#d62728" for v in means_h.values]
    ax.bar(x, means_h.values, yerr=stds_h.values, capsize=2, color=colors)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([FAMILY_LABELS.get(g, g) for g in BSV_GROUPS_ORDER],
                        rotation=45, ha="right")
    ax.set_ylabel("mean ΔBSV (H0T vs CTR centroid)")
    ax.set_title("Pilot 1 HCC — mean ΔBSV per family (H0T relative to CTR)")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_pilot1_hcc_delta_bsv_bar.png", dpi=150)
    plt.close(fig)

    # 4. BSV radar overlay
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
    ax.set_title("Pilot 1 HCC — BSV 11-axis radar (group means)", pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.30, 1.05), fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_pilot1_hcc_bsv_radar.png", dpi=180)
    plt.close(fig)

    # 5. ΔBSV radar (H0T only, since CTR Δ is 0 by definition)
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})
    vals = [float(df[df.class_label == "H0T"][f"delta_{g}"].mean())
             for g in BSV_GROUPS_ORDER]
    vals += vals[:1]
    ax.plot(angles, vals, color="#d62728", linewidth=1.8, label="H0T − CTR")
    ax.fill(angles, vals, alpha=0.12, color="#d62728")
    # Zero baseline
    ax.plot(angles, [0] * len(angles), color="k", linewidth=0.8, linestyle="--",
             label="CTR baseline (Δ=0)")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([FAMILY_LABELS.get(g, g) for g in BSV_GROUPS_ORDER], fontsize=9)
    ax.set_title("Pilot 1 HCC — ΔBSV 11-axis radar (H0T vs CTR centroid)", pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.30, 1.05), fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_pilot1_hcc_delta_bsv_radar.png", dpi=180)
    plt.close(fig)

    # 6. Family effect-size plot (Cohen's d)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    sorted_eff = eff_df.sort_values("cohens_d")
    colors = ["#2ca02c" if v > 0 else "#d62728" for v in sorted_eff["cohens_d"].values]
    ax.barh(sorted_eff["family_label"], sorted_eff["cohens_d"], color=colors)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("Cohen's d (H0T vs CTR); positive = H0T higher")
    ax.set_title("Pilot 1 HCC — per-family effect size on absolute BSV")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_pilot1_hcc_family_effect_sizes.png", dpi=150)
    plt.close(fig)

    # 7. Top MSS hits by class
    df["_mss1"] = df["top_mss_hits"].str.split(";").str[0]
    mc = df.groupby(["class_label", "_mss1"]).size().unstack(fill_value=0)
    top = mc.sum(0).nlargest(10).index.tolist()
    fig, ax = plt.subplots(figsize=(12, 4.2))
    mc[top].T.plot(kind="bar", ax=ax, color=[pal["CTR"], pal["H0T"]])
    ax.set_title("Pilot 1 HCC — top-10 MSS first-hits by class")
    ax.set_ylabel("count")
    ax.tick_params(axis="x", labelrotation=45)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_pilot1_hcc_mss_hits.png", dpi=150)
    plt.close(fig)

    # 8. Motif enrichment
    mfc = df.groupby(["class_label", "top_motif_family"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 4.2))
    mfc = mfc.reindex(columns=BSV_GROUPS_ORDER, fill_value=0)
    mfc.T.plot(kind="bar", ax=ax, color=[pal["CTR"], pal["H0T"]])
    ax.set_xticks(range(len(BSV_GROUPS_ORDER)))
    ax.set_xticklabels([FAMILY_LABELS.get(g, g) for g in BSV_GROUPS_ORDER],
                        rotation=45, ha="right")
    ax.set_title("Pilot 1 HCC — top motif family distribution by class")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_pilot1_hcc_motif_enrichment.png", dpi=150)
    plt.close(fig)

    # 9. Confidence + ambiguity by group
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax_, col, title in [(axes[0], "top_confidence", "top confidence"),
                              (axes[1], "ambiguity_flag", "ambiguity rate")]:
        vals = [df[df.class_label == c][col].mean() for c in ["CTR", "H0T"]]
        ax_.bar(["CTR", "H0T"], vals, color=[pal["CTR"], pal["H0T"]])
        ax_.set_title(f"Pilot 1 HCC — {title}")
        ax_.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_pilot1_hcc_confidence_ambiguity.png", dpi=150)
    plt.close(fig)

    # 10. Unsupervised PCA of BSV space, colored by label post-hoc (no fitting with labels)
    try:
        from sklearn.decomposition import PCA
        X = df[[f"abs_{g}" for g in BSV_GROUPS_ORDER]].values
        pc = PCA(n_components=2, random_state=0).fit_transform(X)
        fig, ax = plt.subplots(figsize=(7, 5))
        for cls in ["CTR", "H0T"]:
            m = df["class_label"].values == cls
            ax.scatter(pc[m, 0], pc[m, 1], s=40, alpha=0.7,
                         label=f"{cls} (n={m.sum()})", color=pal[cls])
        ax.set_xlabel("PC1 of BSV"); ax.set_ylabel("PC2 of BSV")
        ax.set_title("Pilot 1 HCC — unsupervised PCA of BSV space (colored post-hoc)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGS / "fig_pilot1_hcc_bsv_projection.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  PCA projection skipped: {e}")

    print("  10 figures emitted")


# ─────────────────────────────────────────────────────────────────────
# Stage 6 — Biochemical interpretation (cautious)
# ─────────────────────────────────────────────────────────────────────

FAMILY_LONG = {
    "G01": "purine_nucleotide",
    "G02": "purine_metabolite",
    "G03": "pyrimidine",
    "G04": "nucleic_acid_phosphate_backbone",
    "G05": "glycan",
    "G06": "protein_polypeptide",
    "G07": "aromatic_residue",
    "G08": "lipid_acyl_membrane",
    "G09": "sterol_neutral_lipid",
    "G10": "free_amino_acid / sulfur_thiol_redox (shared axis)",
    "G11": "metabolic_small_molecule",
}


def stage6_interpretation(df, eff_df):
    print("\n[STAGE 6] Biochemical interpretation")
    top3_up = eff_df[eff_df.cohens_d > 0].head(3)
    top3_dn = eff_df[eff_df.cohens_d < 0].tail(3)

    amb_ctr = float(df[df.class_label == "CTR"]["ambiguity_flag"].mean())
    amb_h   = float(df[df.class_label == "H0T"]["ambiguity_flag"].mean())
    conf_ctr = float(df[df.class_label == "CTR"]["top_confidence"].mean())
    conf_h  = float(df[df.class_label == "H0T"]["top_confidence"].mean())

    # Top MSS hits per class — top-3 first-hits
    top_mss_ctr = df[df.class_label == "CTR"]["_mss1"].value_counts().head(3).index.tolist() if "_mss1" in df.columns else []
    top_mss_h   = df[df.class_label == "H0T"]["_mss1"].value_counts().head(3).index.tolist() if "_mss1" in df.columns else []

    lines = [
        "# Pilot 1 HCC — Biochemical Interpretation (CAUTIOUS)",
        "",
        "## Readout mode",
        "",
        "Passive readout only. No classifier training, no threshold tuning, no "
        "target-label-driven feature selection. Labels used only AFTER inference "
        "for group-level descriptive summary.",
        "",
        "## Substrate context",
        "",
        "- SERS on Ag colloid (Gurian/Bonifacio 2020 serum SERS protocol).",
        "- Substrate block assigned: `citrate_Ag_colloid_trained`.",
        "- **Caveat**: exact Ag-colloid substrate variant (citrate-reduced vs "
        "alternative reducing agents) is not documented in the paper's public "
        "metadata. Outputs carry `SENSITIVE_SERS_SUBSTRATE_CAVEAT`.",
        "",
        "## Dominant HCC-associated BSV shifts (Cohen's d, |d|≥0.3 highlighted)",
        "",
        "### H0T > CTR (families elevated in HCC)",
        "",
    ]
    for _, r in top3_up.iterrows():
        mark = " **(meaningful)**" if abs(r.cohens_d) >= 0.3 else ""
        lines.append(f"- **{r.family}** ({FAMILY_LONG.get(r.family, r.family)}): "
                     f"d = {r.cohens_d:+.2f}; ΔBSV mean = {r.mean_delta_BSV_H0T:+.3f}{mark}")
    lines += [
        "",
        "### H0T < CTR (families depleted in HCC)",
        "",
    ]
    for _, r in top3_dn.iterrows():
        mark = " **(meaningful)**" if abs(r.cohens_d) >= 0.3 else ""
        lines.append(f"- **{r.family}** ({FAMILY_LONG.get(r.family, r.family)}): "
                     f"d = {r.cohens_d:+.2f}; ΔBSV mean = {r.mean_delta_BSV_H0T:+.3f}{mark}")
    lines += [
        "",
        "## Strongest biochemical themes (cautious language)",
        "",
    ]
    # Derive narrative from top shifts (no molecule claims)
    for _, r in eff_df.head(5).iterrows():
        fam = r.family
        direction = "elevated" if r.cohens_d > 0 else "depleted"
        if fam == "G01":
            theme = "consistent with a purine-nucleotide-associated shift (SERS purine ring chemistry)"
        elif fam == "G02":
            theme = "consistent with a purine-metabolite-associated shift (uric-acid/hypoxanthine-adjacent chemistry)"
        elif fam == "G03":
            theme = "consistent with pyrimidine-family chemistry"
        elif fam == "G04":
            theme = "consistent with nucleic-acid backbone/phosphate chemistry"
        elif fam == "G05":
            theme = "consistent with a glycan/carbohydrate-associated shift"
        elif fam == "G06":
            theme = "consistent with protein-backbone / serum-proteome shift"
        elif fam == "G07":
            theme = "consistent with aromatic-residue shift (Phe/Tyr/Trp-like chemistry)"
        elif fam == "G08":
            theme = "consistent with acyl/lipid chemistry shift"
        elif fam == "G09":
            theme = "consistent with sterol / neutral-lipid chemistry"
        elif fam == "G10":
            theme = "consistent with free-amino-acid / sulfur-thiol-redox chemistry"
        elif fam == "G11":
            theme = "consistent with small-molecule metabolite shift"
        else:
            theme = "family-specific shift"
        lines.append(f"- `{fam}` {direction} in HCC ({theme}); d = {r.cohens_d:+.2f}")

    lines += [
        "",
        "## Motif + MSS evidence",
        "",
        f"- Top MSS first-hits in CTR: {', '.join(top_mss_ctr) if top_mss_ctr else '—'}",
        f"- Top MSS first-hits in H0T: {', '.join(top_mss_h) if top_mss_h else '—'}",
        "- See `tables/pilot1_hcc_motif_mss_enrichment.csv` for full enrichment counts.",
        "",
        "## Confidence and ambiguity",
        "",
        f"- CTR: mean top-confidence = {conf_ctr:.2f}; ambiguity rate = {amb_ctr:.1%}",
        f"- H0T: mean top-confidence = {conf_h:.2f}; ambiguity rate = {amb_h:.1%}",
        "- Ambiguity above 50% in either group signals serum-matrix competition; output tier = SENSITIVE for both classes.",
        "",
        "## What should NOT be overclaimed",
        "",
        "- No exact molecule identity claims (e.g. do NOT say 'contains uric acid' — "
        "say 'consistent with purine-metabolite-associated shift').",
        "- No diagnostic accuracy claim — this is biochemical state interpretation, not a classifier.",
        "- No claim that GAIRA distinguishes HCC subtypes (Pilot 1 is binary CTR vs HCC only).",
        "- No claim that findings generalize beyond this substrate / cohort until cross-cohort (Pilot 2+) validation.",
        "- Effect sizes at |d| < 0.2 are not meaningful and should be treated as noise.",
        "",
        "## Agreement with prior GAIRA pilot expectations",
        "",
        "- Serum SERS is expected to show dominant G06 (serum proteome) background with "
        "secondary G02 purine-metabolite signal. If HCC-associated purine/nucleotide "
        "perturbation is real, G01/G02 effect sizes should be non-trivial.",
        "- Lipid-family axes (G08/G09) may show secondary shifts consistent with "
        "lipid-metabolism dysregulation in HCC (literature-supported but substrate-dependent).",
    ]
    (REPORTS / "REPORT_pilot1_hcc_biochemical_interpretation.md").write_text("\n".join(lines))
    print("  emitted cautious biochemical interpretation")


# ─────────────────────────────────────────────────────────────────────
# Stage 7 — Descriptive label-separation (no fitting)
# ─────────────────────────────────────────────────────────────────────

def stage7_descriptive(df):
    print("\n[STAGE 7] Descriptive label-separation (unsupervised; no fitting)")
    from sklearn.decomposition import PCA

    # Simple unsupervised projection
    X = df[[f"abs_{g}" for g in BSV_GROUPS_ORDER]].values
    pc = PCA(n_components=3, random_state=0)
    Z = pc.fit_transform(X)
    # Compute centroid separation in PC1-PC2 space
    labels = df["class_label"].values
    c_ctr = Z[labels == "CTR", :2].mean(0) if (labels == "CTR").any() else np.zeros(2)
    c_h = Z[labels == "H0T", :2].mean(0) if (labels == "H0T").any() else np.zeros(2)
    centroid_sep = float(np.linalg.norm(c_h - c_ctr))
    pc_var = pc.explained_variance_ratio_

    # Per-family signed mean Δ (just a rolling summary)
    family_signed_means = []
    for g in BSV_GROUPS_ORDER:
        family_signed_means.append({
            "family": g,
            "delta_mean_H0T": float(df[df.class_label == "H0T"][f"delta_{g}"].mean()),
            "delta_mean_CTR": float(df[df.class_label == "CTR"][f"delta_{g}"].mean()),
        })

    # Simple 1-D axis separation: for each family, what's the auc-like separation?
    from numpy import argsort
    sep_rows = []
    for g in BSV_GROUPS_ORDER:
        v = df[f"abs_{g}"].values
        y = (labels == "H0T").astype(int)
        # AUC via Mann-Whitney style
        order = argsort(v)
        rank = np.empty_like(order)
        rank[order] = np.arange(len(v))
        # u = sum of ranks of positives minus expected
        n_pos = y.sum(); n_neg = len(y) - n_pos
        r_pos = rank[y == 1].sum()
        auc = (r_pos - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg) if n_pos > 0 and n_neg > 0 else 0.5
        # Center AUC at 0.5 → |AUC − 0.5| is separation strength
        sep_rows.append({
            "family": g, "family_label": FAMILY_LABELS.get(g, g),
            "univariate_AUC": round(float(auc), 3),
            "sep_strength": round(abs(auc - 0.5), 3),
        })
    sep_df = pd.DataFrame(sep_rows).sort_values("sep_strength", ascending=False)

    # Write tables
    desc_rows = [
        {"metric": "PCA_PC1_variance_explained", "value": round(float(pc_var[0]), 3)},
        {"metric": "PCA_PC2_variance_explained", "value": round(float(pc_var[1]), 3)},
        {"metric": "PCA_PC3_variance_explained", "value": round(float(pc_var[2]), 3)},
        {"metric": "centroid_separation_PC1PC2_euclid", "value": round(centroid_sep, 3)},
        {"metric": "best_univariate_family", "value": sep_df.iloc[0]["family"]},
        {"metric": "best_univariate_AUC", "value": sep_df.iloc[0]["univariate_AUC"]},
        {"metric": "best_univariate_sep_strength", "value": sep_df.iloc[0]["sep_strength"]},
    ]
    pd.DataFrame(desc_rows).to_csv(
        TABLES / "pilot1_hcc_descriptive_label_separation.csv", index=False,
    )
    sep_df.to_csv(TABLES / "pilot1_hcc_per_family_univariate_sep.csv", index=False)

    lines = [
        "# Pilot 1 HCC — Descriptive Label-Separation (NO FITTING)",
        "",
        "Unsupervised descriptive analysis only. No classifier trained. No threshold tuned. "
        "Labels used post-hoc for colouring / separation estimation.",
        "",
        "## PCA of BSV space (unsupervised, colour-by-label post-hoc)",
        "",
        f"- PC1 variance: {pc_var[0]:.1%}",
        f"- PC2 variance: {pc_var[1]:.1%}",
        f"- PC3 variance: {pc_var[2]:.1%}",
        f"- CTR/H0T centroid separation in PC1-PC2: **{centroid_sep:.3f}**",
        "",
        "## Per-family univariate separation (AUC-like; not a classifier)",
        "",
        "Each family's BSV magnitude treated as a univariate score against the label; "
        "AUC is the Mann-Whitney rank statistic (no threshold chosen, no training).",
        "",
        "| family | AUC | |AUC−0.5| |",
        "|---|---:|---:|",
    ]
    for _, r in sep_df.iterrows():
        lines.append(f"| {r['family']} {r['family_label']} | {r['univariate_AUC']} | "
                     f"{r['sep_strength']} |")
    lines += [
        "",
        "## Interpretation",
        "",
        "- |AUC−0.5| ≥ 0.15 suggests the single family carries non-trivial CTR/H0T separation.",
        "- No multivariate fitting performed — a true classifier would almost certainly exceed any univariate AUC listed here.",
        "- These metrics are purely descriptive and must NOT be reported as diagnostic performance.",
    ]
    (REPORTS / "REPORT_pilot1_hcc_descriptive_separation.md").write_text("\n".join(lines))
    return sep_df, centroid_sep


# ─────────────────────────────────────────────────────────────────────
# Stage 8 — Readiness for Pilot 2
# ─────────────────────────────────────────────────────────────────────

def stage8_readiness(eff_df, sep_df, centroid_sep, df):
    print("\n[STAGE 8] Readiness decision for Pilot 2")
    # Criteria
    max_eff = float(eff_df["abs_cohens_d"].max())
    n_meaningful = int((eff_df["abs_cohens_d"] >= 0.3).sum())
    n_trivial = int((eff_df["abs_cohens_d"] < 0.1).sum())
    best_univariate = float(sep_df.iloc[0]["sep_strength"])
    amb_h = float(df[df.class_label == "H0T"]["ambiguity_flag"].mean())

    # Decision rule
    if max_eff >= 0.3 and best_univariate >= 0.1 and centroid_sep > 0:
        decision = "READY_FOR_PILOT_2_CCA_HCC_LM"
    elif max_eff >= 0.2:
        decision = "NEEDS_INTERPRETATION_REVIEW"
    elif amb_h > 0.9:
        decision = "NEEDS_SUBSTRATE_METADATA_FIX"
    else:
        decision = "NEEDS_PILOT_1_QC_FIX"

    lines = [
        "# Pilot 1 HCC — Readiness for Pilot 2",
        "",
        f"**Decision: {decision}**",
        "",
        "## Criteria snapshot",
        "",
        f"- Maximum |Cohen's d| across families: **{max_eff:.2f}**",
        f"- Families with |d| ≥ 0.3 (meaningful effect): {n_meaningful}",
        f"- Families with |d| < 0.1 (trivial): {n_trivial}",
        f"- Best univariate |AUC−0.5|: **{best_univariate:.3f}** "
        f"(family `{sep_df.iloc[0]['family']}`)",
        f"- Centroid separation PC1-PC2: **{centroid_sep:.3f}**",
        f"- H0T ambiguity rate: {amb_h:.1%}",
        "",
        "## Interpretation of decision",
        "",
    ]
    if decision == "READY_FOR_PILOT_2_CCA_HCC_LM":
        lines.append(
            "HCC holdout shows non-trivial biochemical-state differences from CTR. "
            "Effect sizes and descriptive separation are sufficient for advancing to "
            "Pilot 2 (CCA/HCC/LM cross-disease comparison) with full substrate + "
            "SENSITIVE-tier caveats."
        )
    elif decision == "NEEDS_INTERPRETATION_REVIEW":
        lines.append(
            "Maximum effect size is between 0.2 and 0.3 — a chemistry-plausible "
            "but modest signal. Recommend human interpretation review before "
            "advancing to Pilot 2."
        )
    elif decision == "NEEDS_SUBSTRATE_METADATA_FIX":
        lines.append(
            "Ambiguity rate is very high — substrate/serum-matrix is dominating. "
            "Re-check substrate metadata and consider requesting Ag-colloid "
            "preparation details from the source paper before Pilot 2."
        )
    else:
        lines.append(
            "Effect sizes and separation are insufficient to interpret meaningfully. "
            "Re-audit pipeline on this dataset before advancing."
        )
    lines += [
        "",
        "## Passive-readout invariants preserved",
        "",
        "- Engine v4.5 unchanged",
        "- Taxonomy / motif / MSS v4.3 / substrate physics v1.2: read-only",
        "- ERG auxiliary template loaded but unused (no ERG context in HCC cohort)",
        "- Substrate block: `citrate_Ag_colloid_trained` (with substrate-variant caveat)",
        "- Labels used ONLY for group-level post-inference description",
        "- No classifier trained, no threshold tuned, no parameter fitting",
        "- SENSITIVE-tier output policy applied to all H0T and CTR predictions",
    ]
    (REPORTS / "REPORT_pilot1_hcc_readiness_for_pilot2.md").write_text("\n".join(lines))
    return decision, max_eff, n_meaningful


# ─────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 78)
    print("gaira_base_4_passive_target_pilot_1_hcc_holdout")
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
        analyte_to_group[r["analyte_name"]] = bc_to_group.get(
            r["broad_class"], "G11",
        )

    erg_peaks = derive_erg_anchors(master_x)

    refs = stage1_load(master_x)
    df = run_pipeline(refs, master_x, motif_df, mss_df, motif_id_to_group,
                        motif_ids, analyte_to_group, erg_peaks)
    df = stage3_delta_reference(df)
    df.to_csv(TABLES / "pilot1_hcc_per_spectrum_outputs.csv", index=False)
    eff_df, bsv_means, delta_means, amb_rates = stage4_group_analysis(df)
    stage5_figures(df, refs, master_x, eff_df)
    stage6_interpretation(df, eff_df)
    sep_df, centroid_sep = stage7_descriptive(df)
    decision, max_eff, n_meaningful = stage8_readiness(eff_df, sep_df, centroid_sep, df)

    # Audit log
    lines = [
        "# gaira_base_4_passive_target_pilot_1_hcc_holdout — Audit Log",
        "",
        "## Dataset",
        f"- {HCC_CSV}",
        f"- 144 spectra (72 CTR + 72 H0T), 3 substrate batches (A/B/C)",
        f"- Paper: Gurian et al. 2020 (Bonifacio lab)",
        "",
        "## Substrate",
        f"- SERS on Ag colloid",
        f"- substrate block applied: citrate_Ag_colloid_trained (with substrate-variant caveat)",
        "",
        "## Pipeline",
        "- engine: v4.5 locked + v3 controlled-calibration fixes",
        "- full 11-axis BSV + ΔBSV + confidence vector per spectrum",
        "- substrate-gated SERS physics (ON)",
        "- SENSITIVE-tier output policy",
        "- ERG MSS template loaded but unused",
        "",
        "## Results (passive)",
        f"- Max |Cohen's d|: {max_eff:.2f}",
        f"- Families with |d| ≥ 0.3 (meaningful): {n_meaningful}",
        f"- Centroid separation PC1-PC2: {centroid_sep:.3f}",
        "- See per-spectrum outputs, effect-size, motif/MSS enrichment tables",
        "",
        f"## Decision: **{decision}**",
        "",
        "## Invariants",
        "- no classifier trained",
        "- no parameter fitting",
        "- no threshold tuning on target labels",
        "- no feature selection using labels",
        "- labels used for post-inference group-level description only",
        "- engine / taxonomy / motif / MSS v4.3 / substrate physics v1.2: unchanged",
        "- no dynamic DART-Met",
    ]
    (AUDIT / "gaira_base_4_passive_target_pilot_1_hcc_holdout_audit_log.md"
     ).write_text("\n".join(lines))

    p = Path(__file__)
    if p.exists():
        shutil.copy(p, CODE_SNAPSHOT / p.name)

    print(f"\n[complete] decision: {decision}")
    print(f"  max |d|: {max_eff:.2f}; meaningful families (|d|≥0.3): {n_meaningful}")


if __name__ == "__main__":
    main()
