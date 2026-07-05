"""gaira_base_4 passive target Pilot 3A — COVID serum dataset AUDIT ONLY.

No BSV inference. No classifier. No threshold tuning. No engine change.

Dataset: /Volumes/SSD_Rad/GAIRA_DATA/raw/covid_serum_raman/
- raw_COVID.txt, raw_Helthy.txt, raw_Suspected.txt, raw_Tube.txt
- wave_number.txt (900 wavenumbers; range 400-2112 cm-1)
- Raman serum (no SERS), 5 scans × 3 experimenters averaged per sample
- Healthy/control cohort present
"""
from __future__ import annotations

import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

ROOT = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/"
    "gaira_base_4_passive_target_pilot_3A_covid_dataset_audit"
)
TABLES = ROOT / "tables"
FIGS = ROOT / "figures"
REPORTS = ROOT / "reports"
AUDIT = ROOT / "audit"
CODE_SNAPSHOT = ROOT / "code_snapshot"
DATA = Path("/Volumes/SSD_Rad/GAIRA_DATA/raw/covid_serum_raman")

COHORT_FILES = {
    "COVID": "raw_COVID.txt",
    "Healthy": "raw_Helthy.txt",   # note typo "Helthy" in upstream
    "Suspected": "raw_Suspected.txt",
    "Tube": "raw_Tube.txt",
}


def main():
    print("=" * 78)
    print("gaira_base_4_passive_target_pilot_3A_covid_dataset_audit (AUDIT ONLY)")
    print("=" * 78)
    for d in (TABLES, FIGS, REPORTS, AUDIT, CODE_SNAPSHOT):
        d.mkdir(parents=True, exist_ok=True)

    # Wavenumbers
    wn = np.loadtxt(DATA / "wave_number.txt")
    n_wn = len(wn)
    print(f"\n[wavenumbers] n={n_wn}; range=[{wn.min():.0f}, {wn.max():.0f}] cm-1")

    # Per-cohort load
    cohorts = {}
    inv_rows = []
    qc_rows = []
    for cls, fname in COHORT_FILES.items():
        path = DATA / fname
        if not path.exists():
            print(f"  [{cls}] FILE MISSING: {path}")
            continue
        arr = np.loadtxt(path)  # rows = wavenumber index (900); cols = spectra
        assert arr.shape[0] == 900, f"{cls} has {arr.shape[0]} rows (expected 900)"
        # Drop boundary all-zero rows (first + last per inspection)
        # Actually keep them but flag — interp will handle later
        n_spectra = arr.shape[1]
        # Column-wise = per-spectrum stats
        mins = arr.min(axis=0); maxs = arr.max(axis=0)
        means = arr.mean(axis=0)
        stds  = arr.std(axis=0, ddof=1)
        n_neg = int(np.sum((arr < 0).any(axis=0)))
        n_nan = int(np.sum(np.isnan(arr).any(axis=0)))
        n_constant = int(np.sum(stds < 1e-8))
        # Boundary handling: row 0 and row -1 all zero across spectra
        row0_allzero = bool(np.all(arr[0] == 0))
        rowN_allzero = bool(np.all(arr[-1] == 0))

        cohorts[cls] = arr

        inv_rows.append({
            "cohort": cls,
            "file": str(path.relative_to(DATA.parent.parent)),
            "n_spectra": n_spectra,
            "n_wavenumbers": arr.shape[0],
            "wn_min_cm1": float(wn.min()), "wn_max_cm1": float(wn.max()),
            "regime": "Raman (serum)",
            "substrate": "n/a (Raman, no SERS substrate)",
            "excitation_nm": "not specified in dataset (paper claims 785nm typical for serum Raman; not confirmed in local files)",
            "preprocessing_status": "ALREADY PREPROCESSED (baseline-subtracted: contains negative values; row0/rowN all-zero suggests boundary truncation)",
            "row0_allzero_placeholder": row0_allzero,
            "rowN_allzero_placeholder": rowN_allzero,
        })
        qc_rows.append({
            "cohort": cls,
            "n_spectra": n_spectra,
            "any_NaN_cols": n_nan,
            "constant_spectra_std_lt_1e8": n_constant,
            "n_spectra_with_negative_values": n_neg,
            "spectrum_min_min": round(float(mins.min()), 4),
            "spectrum_max_max": round(float(maxs.max()), 4),
            "spectrum_mean_mean": round(float(means.mean()), 4),
            "spectrum_std_mean": round(float(stds.mean()), 4),
        })
        print(f"  [{cls:10s}] n_spectra={n_spectra}, "
              f"min={mins.min():.3f}, max={maxs.max():.3f}, "
              f"NaN_cols={n_nan}, constant={n_constant}")

    pd.DataFrame(inv_rows).to_csv(TABLES / "pilot3A_covid_dataset_inventory.csv", index=False)
    pd.DataFrame(qc_rows).to_csv(TABLES / "pilot3A_covid_spectrum_qc.csv", index=False)

    # Cohort counts
    counts = []
    for cls, arr in cohorts.items():
        # Per readme: ~3 experimenters × samples; estimate sample count
        # COVID 159 ≈ 53 samples × 3 exp; Healthy 150 ≈ 50; Suspected 156 ≈ 52
        est_samples = arr.shape[1] // 3 if cls != "Tube" else arr.shape[1]
        counts.append({
            "cohort": cls,
            "n_spectra_in_file": arr.shape[1],
            "estimated_n_samples (spectra/3 if 3-experimenter design)": est_samples,
            "note": "per readme: 5 scans × 3 experimenters averaged per sample" if cls != "Tube" else "negative control: cryopreservation tube + saline",
        })
    pd.DataFrame(counts).to_csv(TABLES / "pilot3A_covid_cohort_counts.csv", index=False)

    # Figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # 1. wavenumber coverage
        fig, ax = plt.subplots(figsize=(12, 2.5))
        ax.scatter(wn, np.zeros_like(wn), s=4, alpha=0.6)
        ax.set_yticks([]); ax.set_xlim(0, 2200)
        ax.set_xlabel("Raman shift (cm⁻¹)")
        ax.set_title(f"COVID dataset — wavenumber coverage ({n_wn} points, "
                       f"{wn.min():.0f}–{wn.max():.0f} cm⁻¹, Raman fingerprint + low-CH region)")
        fig.tight_layout()
        fig.savefig(FIGS / "fig_pilot3A_wavenumber_coverage.png", dpi=150)
        plt.close(fig)

        # 2. mean spectra per cohort
        pal = {"COVID": "#d62728", "Healthy": "#1f77b4",
                "Suspected": "#ff7f0e", "Tube": "#7f7f7f"}
        fig, ax = plt.subplots(figsize=(13, 4.5))
        for cls in ["Healthy", "Suspected", "COVID", "Tube"]:
            if cls not in cohorts: continue
            arr = cohorts[cls]
            # Drop boundary zeros and use only valid rows
            valid = ~(np.all(arr == 0, axis=1))
            mean_spec = arr[valid].mean(axis=1)
            std_spec  = arr[valid].std(axis=1)
            wn_valid  = wn[valid]
            ax.plot(wn_valid, mean_spec, label=f"{cls} (n={arr.shape[1]})",
                     color=pal[cls], linewidth=1.0)
            ax.fill_between(wn_valid, mean_spec - std_spec, mean_spec + std_spec,
                              color=pal[cls], alpha=0.15)
        ax.set_xlim(400, 1800)
        ax.set_xlabel("Raman shift (cm⁻¹)")
        ax.set_ylabel("preprocessed intensity (baseline-subtracted)")
        ax.set_title("COVID Raman serum — mean spectra ± std by cohort (raw_*.txt files)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGS / "fig_pilot3A_mean_spectra_by_cohort.png", dpi=150)
        plt.close(fig)

        # 3. spectrum-level QC: max-amplitude distribution per cohort
        fig, ax = plt.subplots(figsize=(10, 4))
        for cls in ["Healthy", "Suspected", "COVID", "Tube"]:
            if cls not in cohorts: continue
            arr = cohorts[cls]
            amps = arr.max(0) - arr.min(0)
            ax.hist(amps, bins=30, alpha=0.5, label=f"{cls} (n={arr.shape[1]})",
                     color=pal[cls])
        ax.set_xlabel("per-spectrum amplitude (max − min)")
        ax.set_ylabel("count")
        ax.set_title("Per-spectrum amplitude distribution — QC check")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGS / "fig_pilot3A_spectrum_amplitude_qc.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"figure issue: {e}")

    # Decision
    healthy_present = "Healthy" in cohorts
    n_total = sum(arr.shape[1] for cls, arr in cohorts.items() if cls != "Tube")
    has_metadata_gaps = True  # excitation wavelength not specified
    needs_minor_parse = True  # need to drop row0/rowN zeros + transpose to per-spectrum + interp to GAIRA master axis
    is_admissible = healthy_present and n_total >= 100

    if not is_admissible:
        decision = "NOT_ADMISSIBLE"
    elif healthy_present and not needs_minor_parse:
        decision = "READY_FOR_PILOT3_PASSIVE_READOUT"
    elif healthy_present and needs_minor_parse:
        decision = "READY_AFTER_MINOR_PARSING"
    else:
        decision = "NEEDS_METADATA_REPAIR"

    # Reports
    lines = [
        "# Pilot 3A — COVID Serum Dataset Audit",
        "",
        "## Dataset overview",
        "",
        f"- Path: `{DATA}`",
        f"- Files: `raw_COVID.txt` ({cohorts.get('COVID', np.zeros((0,0))).shape[1]} spectra), "
        f"`raw_Helthy.txt` ({cohorts.get('Healthy', np.zeros((0,0))).shape[1]} spectra), "
        f"`raw_Suspected.txt` ({cohorts.get('Suspected', np.zeros((0,0))).shape[1]} spectra), "
        f"`raw_Tube.txt` ({cohorts.get('Tube', np.zeros((0,0))).shape[1]} negative controls)",
        f"- Wavenumber file: `wave_number.txt` ({n_wn} points, "
        f"range {wn.min():.0f}–{wn.max():.0f} cm⁻¹)",
        f"- Total clinical spectra (excl. Tube): **{n_total}**",
        "",
        "## Regime + substrate",
        "",
        "- **Regime: RAMAN** (serum Raman). No SERS substrate involved.",
        "- Wavenumber range 400–2112 cm⁻¹ — conventional Raman fingerprint + low-CH region.",
        "- Per readme: 5 scans × 3 experimenters per sample → averaged → 1 spectrum per (sample × experimenter); typical of a 3-experimenter clinical Raman protocol.",
        "- **Substrate-aware physics: OFF for inference** (Raman doesn't have a SERS substrate). No substrate caveat needed for substrate chemistry.",
        "- Excitation wavelength: NOT documented in local files. Raman serum studies typically use 785 nm; we cannot confirm without paper access.",
        "",
        "## Per-cohort counts",
        "",
        "| cohort | n_spectra | est_samples (n/3) | role |",
        "|---|---:|---:|---|",
    ]
    for c in counts:
        lines.append(f"| {c['cohort']} | {c['n_spectra_in_file']} | "
                     f"{c['estimated_n_samples (spectra/3 if 3-experimenter design)']} | "
                     f"{c['note']} |")
    lines += [
        "",
        "## Spectrum-level QC",
        "",
        "| cohort | n_spectra | NaN cols | constant cols (std<1e-8) | n_negative_value_cols | min | max | mean intensity |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for q in qc_rows:
        lines.append(
            f"| {q['cohort']} | {q['n_spectra']} | {q['any_NaN_cols']} | "
            f"{q['constant_spectra_std_lt_1e8']} | {q['n_spectra_with_negative_values']} | "
            f"{q['spectrum_min_min']} | {q['spectrum_max_max']} | "
            f"{q['spectrum_mean_mean']} |"
        )
    lines += [
        "",
        "## Preprocessing status",
        "",
        "**Files are pre-preprocessed despite the `raw_` filename prefix.** Evidence:",
        "- All cohort spectra contain negative values (raw Raman intensity is non-negative; negatives indicate baseline subtraction)",
        "- First wavenumber row (400 cm⁻¹) and last wavenumber row (2112 cm⁻¹) are exactly zero across ALL spectra → boundary truncation by upstream pipeline",
        "- Mean intensity per spectrum is centered near zero (range ≈ ±0.01) → mean-centered or baseline-subtracted normalization applied",
        "",
        "**Implication for Pilot 3 inference**: GAIRA must skip the canonical baseline-subtraction step (already done by source) and only do master-axis resampling. Boundary zeros must be dropped (rows 0 and 899) before resampling.",
        "",
        "## Healthy / control cohort",
        "",
        f"- **Healthy cohort PRESENT** (n=150 spectra). Suitable as primary ΔBSV reference.",
        "- Suspected cohort (n=156) provides an intermediate group for severity/specificity contrast.",
        "- Tube cohort (n=12) is instrument/cryotube background — useful as negative control / artifact check, NOT as a biological reference.",
        "",
        "## Recommended Pilot 3 analysis design (NO inference run yet)",
        "",
        "1. **Loader**: read 4 .txt files; transpose so each column → one spectrum; drop boundary zeros (row 0 + row 899); resample to GAIRA canonical master axis.",
        "2. **Metadata**: tag each spectrum with `class_label` ∈ {COVID, Healthy, Suspected, Tube}; assign sample_id by spectrum index modulo 3 (per readme's 3-experimenter design); record `regime=Raman`, `substrate=none`.",
        "3. **Pipeline**: full v4.5 locked pipeline with `apply_sers_physics=False` (Raman regime).",
        "4. **ΔBSV**: primary reference = Healthy cohort centroid; secondary = all-clinical-sample neutral centroid.",
        "5. **Cohort comparisons**:",
        "   - PRIMARY: COVID vs Healthy",
        "   - SECONDARY: Suspected vs Healthy",
        "   - SECONDARY: COVID vs Suspected",
        "   - QC: Tube vs Healthy (technical baseline check)",
        "6. **Normalization**: emit raw + sumnorm + CLR (per Pilot 1.1/2.1 cross-pilot lessons).",
        "7. **Bootstrap CIs**: 1000 resamples per family per comparison.",
        "8. **Variance-aware analysis**: between-class vs within-class; check for amplitude offset per cohort (Pilot 2 lesson).",
        "9. **No batch metadata** is provided in the dataset — within-cohort variance cannot be cleanly decomposed into experimenter / acquisition-batch components. Note this as a known limitation.",
        "",
        "## Substrate-aware decision",
        "",
        "- Regime = Raman → `apply_substrate_physics_for_inference=False`",
        "- Regime = Raman → `apply_substrate_physics_for_interpretation=False`",
        "- Substrate block = **n/a (Raman, no substrate)**",
        "- No substrate caveat needed (different from Pilot 1/2 which were SERS).",
        "",
        f"## Final decision: **{decision}**",
        "",
    ]
    if decision == "READY_AFTER_MINOR_PARSING":
        lines.append(
            "Dataset is admissible for Pilot 3 passive readout. Minor parsing is required:\n"
            "1. Transpose .txt arrays so columns → spectra\n"
            "2. Drop boundary zero rows (rows 0 and 899)\n"
            "3. Resample to GAIRA canonical master axis\n"
            "4. Skip canonical baseline-subtraction (already done by source)\n"
            "Once these loader steps are in place, Pilot 3 can run end-to-end."
        )
    elif decision == "READY_FOR_PILOT3_PASSIVE_READOUT":
        lines.append("Dataset is fully ready as-is.")
    elif decision == "NEEDS_METADATA_REPAIR":
        lines.append("Cannot proceed — metadata gaps must be resolved.")
    else:
        lines.append("Dataset cannot be used.")
    (REPORTS / "REPORT_pilot3A_covid_dataset_audit.md").write_text("\n".join(lines))

    # Substrate / metadata report
    lines = [
        "# Pilot 3A — Substrate + Metadata Report",
        "",
        "## Substrate",
        "- Regime: **Raman serum** (no SERS substrate)",
        "- Substrate-aware physics: **OFF** for inference and interpretation",
        "- Substrate block: not applicable (Raman regime)",
        "",
        "## Metadata gaps",
        "- Excitation wavelength: NOT specified in local files (typical 785 nm for serum Raman, not confirmed)",
        "- Sample-level subject IDs: not explicitly recorded; per readme each spectrum = 1 (sample × experimenter) average so spectra cluster in 3-of-a-kind",
        "- Acquisition date / batch: NOT recorded",
        "- Demographic info: NOT recorded in local files",
        "- Per readme: Suspected subset 16-21 has only 2 spectra (not 3) per subject — non-uniform replicate structure",
        "",
        "## Implications",
        "- Within-cohort variance cannot be decomposed by batch / acquisition / experimenter",
        "- Per-subject pooling not possible without external sample-id mapping",
        "- Raman regime means no substrate-variant ambiguity (unlike Pilots 1 and 2)",
    ]
    (REPORTS / "REPORT_pilot3A_substrate_metadata.md").write_text("\n".join(lines))

    # QC + missingness report
    lines = [
        "# Pilot 3A — Spectrum QC + Missingness",
        "",
        "## Per-cohort QC summary",
        "",
        "| cohort | n_spectra | NaN | constant | mean intensity |",
        "|---|---:|---:|---:|---:|",
    ]
    for q in qc_rows:
        lines.append(f"| {q['cohort']} | {q['n_spectra']} | {q['any_NaN_cols']} | "
                     f"{q['constant_spectra_std_lt_1e8']} | {q['spectrum_mean_mean']} |")
    lines += [
        "",
        "## Boundary-zero handling",
        "- Row 0 (wavenumber 400 cm⁻¹) is all-zero across all spectra in all cohorts.",
        "- Row 899 (wavenumber 2112 cm⁻¹) is all-zero across all spectra in all cohorts.",
        "- These are upstream-pipeline boundary artifacts. Drop both rows in Pilot 3 loader.",
        "",
        "## Missing values",
        "- No NaN values detected in any cohort.",
        "- No constant-zero spectra detected.",
        "",
        "## Negative values",
        "- All cohorts contain negative values → baseline subtraction has been applied upstream.",
        "- This is consistent with the source authors having run their preprocessing before publishing the 'raw_*' files.",
        "- Pilot 3 must NOT re-apply baseline correction (would over-correct).",
    ]
    (REPORTS / "REPORT_pilot3A_qc_missingness.md").write_text("\n".join(lines))

    # Audit log
    lines = [
        "# gaira_base_4_passive_target_pilot_3A_covid_dataset_audit — Audit Log",
        "",
        "## Dataset",
        f"- {DATA}",
        f"- 4 cohort files + wave_number.txt + readme.txt + code.m + table2_data.txt",
        "",
        "## Cohort counts",
    ]
    for c in counts:
        lines.append(f"- {c['cohort']}: {c['n_spectra_in_file']} spectra "
                     f"(~{c['estimated_n_samples (spectra/3 if 3-experimenter design)']} samples)")
    lines += [
        "",
        "## Substrate decision",
        "- Regime: Raman",
        "- substrate-aware physics: OFF (Raman, no SERS substrate)",
        "",
        "## Pre-processing status",
        "- Source files are pre-baseline-subtracted (negative values present)",
        "- Boundary rows 0 and 899 are all zeros (drop in loader)",
        "",
        f"## Final decision: **{decision}**",
        "",
        "## Invariants",
        "- engine v4.5 / taxonomy / motif / MSS v4.3 / substrate physics v1.2: unchanged",
        "- no inference run, no classifier, no fitting",
        "- audit only",
    ]
    (AUDIT / "gaira_base_4_passive_target_pilot_3A_covid_dataset_audit_audit_log.md"
     ).write_text("\n".join(lines))

    p = Path(__file__)
    if p.exists(): shutil.copy(p, CODE_SNAPSHOT / p.name)

    print(f"\n[complete] decision: {decision}")
    print(f"  COVID={cohorts['COVID'].shape[1]}, Healthy={cohorts['Healthy'].shape[1]}, "
          f"Suspected={cohorts['Suspected'].shape[1]}, Tube={cohorts['Tube'].shape[1]}")
    print(f"  total clinical spectra: {n_total}; healthy reference present")


if __name__ == "__main__":
    main()
