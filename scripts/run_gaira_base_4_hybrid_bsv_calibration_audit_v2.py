"""gaira_base_4 hybrid BSV calibration audit v2 — PRE-CALIBRATION AUDIT.

Analyte-centric audit of calibration datasets already present in GAIRA.

Hard constraints (user-explicit):
  - does NOT run calibration tests
  - does NOT rebuild motif / MSS / taxonomy
  - does NOT use target clinical cohorts for fitting
  - no dynamic DART-Met modeling

Outputs only: discovery CSVs, admissibility decisions, substrate-physics
applicability per dataset, next-phase checklist, gap analysis, readiness.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/"
    "gaira_base_4_hybrid_bsv_calibration_audit_v2"
)
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
AUDIT = ROOT / "audit"
CODE_SNAPSHOT = ROOT / "code_snapshot"

DATA_RAW = Path("/Volumes/SSD_Rad/GAIRA_DATA/raw")
PROCESSED = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed")


# ═════════════════════════════════════════════════════════════════════
# STAGE 1 — Global calibration dataset discovery
# ═════════════════════════════════════════════════════════════════════

# Comprehensive enumeration from storage sweep. Each row is one distinct
# calibration-relevant block (a dataset or a sub-block within a dataset ZIP).
CANDIDATES = [
    # ── ERG_calibration.csv: 55 spectra, 11 conc × 5 reps, 785 nm, 30 mW, cAg
    {
        "dataset_name": "ergothioneine_serum / ERG_calibration.csv",
        "source_path": str(DATA_RAW / "ergothioneine_serum/ERG_calibration.csv"),
        "analyte_or_process": "ergothioneine (titration in serum matrix)",
        "regime": "SERS",
        "substrate_type": "cAg (citrate-Ag colloid)",
        "sample_matrix": "serum (titrated with ERGO)",
        "n_spectra": 55,
        "control_variable": "ERGO concentration 0.0-2.0 µM (11 points, 5 reps each)",
        "is_calibration_relevant": True,
        "calibration_type_tag": "DOSE_RESPONSE + REPLICATE",
    },
    # ── CSPP Figure-7 (already used by prior calibration eval)
    {
        "dataset_name": "cspp_serum / Figure-7_all-spectra-and-metadata.csv",
        "source_path": str(DATA_RAW / "cspp_serum/Figure-7_all-spectra-and-metadata.csv"),
        "analyte_or_process": "hypoxanthine + ergothioneine spiked into serum (Bkg / Erg / Hyp cohorts)",
        "regime": "SERS",
        "substrate_type": "plasmonic paper Ag (CSPP)",
        "sample_matrix": "serum",
        "n_spectra": 150,
        "control_variable": "Bkg (50) vs Erg-spiked (50) vs Hyp-spiked (50)",
        "is_calibration_relevant": True,
        "calibration_type_tag": "MIXTURE_OVERLAP + DOSE_RESPONSE",
    },
    # ── CSPP Figure-2/4/5/6 (other figure contexts — uncertain admissibility)
    {
        "dataset_name": "cspp_serum / Figure-2_all-spectra-and-metadata.csv",
        "source_path": str(DATA_RAW / "cspp_serum/Figure-2_all-spectra-and-metadata.csv"),
        "analyte_or_process": "CSPP figure-2 context (cohort labels not obvious from header)",
        "regime": "SERS",
        "substrate_type": "plasmonic paper Ag (CSPP)",
        "sample_matrix": "serum (variants)",
        "n_spectra": 70,
        "control_variable": "unclear from header — requires manual inspection of metadata",
        "is_calibration_relevant": True,
        "calibration_type_tag": "OTHER_CONTROLLED_SHIFT (needs parsing)",
    },
    {
        "dataset_name": "cspp_serum / Figure-4_all-spectra-and-metadata.csv",
        "source_path": str(DATA_RAW / "cspp_serum/Figure-4_all-spectra-and-metadata.csv"),
        "analyte_or_process": "CSPP figure-4 context (needs manual parse)",
        "regime": "SERS",
        "substrate_type": "plasmonic paper Ag (CSPP)",
        "sample_matrix": "serum (variants)",
        "n_spectra": 125,
        "control_variable": "encoded in X1 column — unparsed",
        "is_calibration_relevant": True,
        "calibration_type_tag": "OTHER_CONTROLLED_SHIFT (needs parsing)",
    },
    {
        "dataset_name": "cspp_serum / Figure-5_all-spectra-and-metadata.csv",
        "source_path": str(DATA_RAW / "cspp_serum/Figure-5_all-spectra-and-metadata.csv"),
        "analyte_or_process": "CSPP figure-5 context (needs manual parse)",
        "regime": "SERS",
        "substrate_type": "plasmonic paper Ag (CSPP)",
        "sample_matrix": "serum (variants)",
        "n_spectra": 120,
        "control_variable": "encoded in X1 column — unparsed",
        "is_calibration_relevant": True,
        "calibration_type_tag": "OTHER_CONTROLLED_SHIFT (needs parsing)",
    },
    {
        "dataset_name": "cspp_serum / Figure-6_all-spectra-and-metadata.csv",
        "source_path": str(DATA_RAW / "cspp_serum/Figure-6_all-spectra-and-metadata.csv"),
        "analyte_or_process": "CSPP figure-6 context (needs manual parse)",
        "regime": "SERS",
        "substrate_type": "plasmonic paper Ag (CSPP)",
        "sample_matrix": "serum (variants)",
        "n_spectra": 63,
        "control_variable": "unclear from header — requires manual inspection",
        "is_calibration_relevant": True,
        "calibration_type_tag": "OTHER_CONTROLLED_SHIFT (needs parsing)",
    },
    # ── serum_ag_colloids ZIP — multiple sub-blocks
    {
        "dataset_name": "serum_ag_colloids ZIP :: dataset uricase/",
        "source_path": str(DATA_RAW / "serum_ag_colloids/dataset_spectral_data.zip::dataset uricase/"),
        "analyte_or_process": "uricase enzymatic depletion (UA → allantoin): 4 cohorts × 5 reps (Serumspiked, Serumspiked+Enzyme, SerumSigma, SerumSigma+Enzyme)",
        "regime": "SERS",
        "substrate_type": "Ag colloid (citrate-Ag, serum matrix)",
        "sample_matrix": "serum (commercial Sigma + spiked variants) with/without uricase enzyme",
        "n_spectra": 20,
        "control_variable": "enzymatic depletion (+Enzyme vs −Enzyme) × spiked-vs-baseline (2×2 = 4 cohorts)",
        "is_calibration_relevant": True,
        "calibration_type_tag": "TRANSFORMATION_ENZYMATIC + MIXTURE_OVERLAP",
    },
    {
        "dataset_name": "serum_ag_colloids ZIP :: Raman metabolites/",
        "source_path": str(DATA_RAW / "serum_ag_colloids/dataset_spectral_data.zip::Raman metabolites/"),
        "analyte_or_process": "51 pure metabolites (powder Raman): AcCoA/Ade/Ala/Alb/Arg/Asc/Chol/Citric/CoA/Creat/Cys/Ergo/Gluc/Gua/His/Hypox/Methio/Oleic/PEP/Phe/Pro/Ribo/Stearic/Thy/Triolein/Trp/Tyr/UA/Urea/Xanth/…",
        "regime": "Raman",
        "substrate_type": "pure powder (no substrate)",
        "sample_matrix": "pure analyte (powder)",
        "n_spectra": 418,
        "control_variable": "identity only (multi-replicate per analyte)",
        "is_calibration_relevant": True,
        "calibration_type_tag": "IDENTITY_PURE + REPLICATE",
    },
    {
        "dataset_name": "serum_ag_colloids ZIP :: SERS metabolites/ (100 µM)",
        "source_path": str(DATA_RAW / "serum_ag_colloids/dataset_spectral_data.zip::SERS metabolites/"),
        "analyte_or_process": "53 pure metabolites SERS at 100 µM (matches Raman metabolites set + adds DNA + RNA)",
        "regime": "SERS",
        "substrate_type": "Ag colloid",
        "sample_matrix": "pure analyte (100 µM solution)",
        "n_spectra": 81,
        "control_variable": "identity + SERS regime perturbation",
        "is_calibration_relevant": True,
        "calibration_type_tag": "IDENTITY_PURE + SUBSTRATE_PERTURBATION",
    },
    {
        "dataset_name": "serum_ag_colloids ZIP :: SERS metabolites for fitting/",
        "source_path": str(DATA_RAW / "serum_ag_colloids/dataset_spectral_data.zip::SERS metabolites for fitting/"),
        "analyte_or_process": "Hypoxanthine (10 reps) + UAfree (10 reps) + UAbound (10 reps)",
        "regime": "SERS",
        "substrate_type": "Ag colloid",
        "sample_matrix": "pure analyte + UA:HSA complex variants",
        "n_spectra": 30,
        "control_variable": "UAfree vs UAbound (matrix effect on UA)",
        "is_calibration_relevant": True,
        "calibration_type_tag": "MIXTURE_OVERLAP + REPLICATE",
    },
    {
        "dataset_name": "serum_ag_colloids ZIP :: isotopic/",
        "source_path": str(DATA_RAW / "serum_ag_colloids/dataset_spectral_data.zip::isotopic/"),
        "analyte_or_process": "UA vs UAiso (isotopic substitution) × ±HSA × ±filterLower/Upper",
        "regime": "SERS",
        "substrate_type": "Ag colloid",
        "sample_matrix": "UA / UAiso / UA+HSA / UAiso+HSA / UA+HSAfilterLower / UA+HSAfilterUpper / UAiso+HSAfilterLower / UAiso+HSAfilterUpper",
        "n_spectra": 73,
        "control_variable": "isotope label (UA vs UAiso) + matrix (±HSA) + filter fractionation",
        "is_calibration_relevant": True,
        "calibration_type_tag": "OTHER_CONTROLLED_SHIFT (isotopic labeling + matrix effects)",
    },
    {
        "dataset_name": "serum_ag_colloids ZIP :: SERS serum Merck/",
        "source_path": str(DATA_RAW / "serum_ag_colloids/dataset_spectral_data.zip::SERS serum Merck/"),
        "analyte_or_process": "whole serum SERS (commercial Merck donor pool)",
        "regime": "SERS",
        "substrate_type": "Ag colloid",
        "sample_matrix": "clinical donor serum (commercial pool)",
        "n_spectra": 285,
        "control_variable": "clinical pool — no controlled perturbation",
        "is_calibration_relevant": False,  # clinical target-like matrix
        "calibration_type_tag": "CLINICAL_POOL (excluded)",
    },
    {
        "dataset_name": "serum_ag_colloids ZIP :: donors serum SERS/",
        "source_path": str(DATA_RAW / "serum_ag_colloids/dataset_spectral_data.zip::donors serum SERS/"),
        "analyte_or_process": "multi-donor serum SERS",
        "regime": "SERS",
        "substrate_type": "Ag colloid",
        "sample_matrix": "clinical donor serum",
        "n_spectra": 100,  # approximate
        "control_variable": "donor identity — no controlled perturbation",
        "is_calibration_relevant": False,
        "calibration_type_tag": "CLINICAL_POOL (excluded)",
    },
    {
        "dataset_name": "serum_ag_colloids ZIP :: digitized literature spectra/",
        "source_path": str(DATA_RAW / "serum_ag_colloids/dataset_spectral_data.zip::digitized literature spectra/"),
        "analyte_or_process": "Gelder 2007, Kim 1987, Stewart 1999 (digitized)",
        "regime": "Raman",
        "substrate_type": "literature digitized",
        "sample_matrix": "literature references",
        "n_spectra": 3,
        "control_variable": "none — literature reference spectra",
        "is_calibration_relevant": True,
        "calibration_type_tag": "IDENTITY_PURE (literature, already in GAIRA load_digitised_literature)",
    },
    # ── adenine_sers_control (already in prior suite)
    {
        "dataset_name": "adenine_sers_control / bAgNPs LOD ladder",
        "source_path": str(DATA_RAW / "adenine_sers_control/"),
        "analyte_or_process": "adenine dose-response on bAgNPs (10pg → 10µM, 7 points)",
        "regime": "SERS",
        "substrate_type": "bAgNPs (biologically synthesized Ag nanoparticles)",
        "sample_matrix": "pure adenine solution",
        "n_spectra": 7,
        "control_variable": "concentration (7 log-spaced levels)",
        "is_calibration_relevant": True,
        "calibration_type_tag": "DOSE_RESPONSE + SUBSTRATE_PERTURBATION",
    },
    {
        "dataset_name": "adenine_sers_control / 1ng replicates",
        "source_path": str(DATA_RAW / "adenine_sers_control/"),
        "analyte_or_process": "adenine 1ng × 5 replicate SERS on bAgNPs",
        "regime": "SERS",
        "substrate_type": "bAgNPs",
        "sample_matrix": "pure adenine 1ng",
        "n_spectra": 5,
        "control_variable": "replicate only (fixed concentration)",
        "is_calibration_relevant": True,
        "calibration_type_tag": "REPLICATE_REPRODUCIBILITY",
    },
    # ── Standard grounding refs (already used as identity bench)
    {
        "dataset_name": "ramanbiolib",
        "source_path": str(DATA_RAW / "ramanbiolib"),
        "analyte_or_process": "pure Raman analyte references across all 11 families",
        "regime": "Raman",
        "substrate_type": "pure (no substrate)",
        "sample_matrix": "pure reference",
        "n_spectra": 202,
        "control_variable": "identity",
        "is_calibration_relevant": True,
        "calibration_type_tag": "IDENTITY_PURE",
    },
    {
        "dataset_name": "gobbato powder Raman (currently loaded 3-rep subset)",
        "source_path": str(DATA_RAW / "serum_ag_colloids/dataset_spectral_data.zip::Raman metabolites/"),
        "analyte_or_process": "51 analytes × 3 reps (Raman powder)",
        "regime": "Raman",
        "substrate_type": "pure powder",
        "sample_matrix": "pure reference",
        "n_spectra": 153,
        "control_variable": "identity + replicate",
        "is_calibration_relevant": True,
        "calibration_type_tag": "IDENTITY_PURE + REPLICATE",
    },
    {
        "dataset_name": "amino_acid_raman_grounding / aa.xlsx",
        "source_path": str(DATA_RAW / "amino_acid_raman_grounding/aa.xlsx"),
        "analyte_or_process": "20 amino acids (Raman)",
        "regime": "Raman",
        "substrate_type": "pure",
        "sample_matrix": "pure amino acid",
        "n_spectra": 20,
        "control_variable": "identity",
        "is_calibration_relevant": True,
        "calibration_type_tag": "IDENTITY_PURE",
    },
    {
        "dataset_name": "digitised_literature (in loader)",
        "source_path": "(GAIRA digitised literature loader)",
        "analyte_or_process": "2 digitised literature references",
        "regime": "Raman",
        "substrate_type": "literature",
        "sample_matrix": "literature",
        "n_spectra": 2,
        "control_variable": "identity",
        "is_calibration_relevant": True,
        "calibration_type_tag": "IDENTITY_PURE",
    },
    {
        "dataset_name": "sers_metabolite_63 (NIHMS1547448, citrate-Ag)",
        "source_path": str(DATA_RAW / "sers_metabolite_63/NIHMS1547448-supplement-2.xlsx"),
        "analyte_or_process": "63 pure SERS metabolite references",
        "regime": "SERS",
        "substrate_type": "citrate-Ag colloid",
        "sample_matrix": "pure analyte",
        "n_spectra": 63,
        "control_variable": "identity + regime",
        "is_calibration_relevant": True,
        "calibration_type_tag": "IDENTITY_PURE + CROSS_REGIME_COHERENCE",
    },
    # ── Explicitly EXCLUDED clinical cohorts (documented, not run)
    {
        "dataset_name": "nature_serum_sers (and other clinical cohorts)",
        "source_path": str(DATA_RAW / "nature_serum_sers/"),
        "analyte_or_process": "disease-cohort serum SERS",
        "regime": "SERS",
        "substrate_type": "varies",
        "sample_matrix": "clinical",
        "n_spectra": 0,
        "control_variable": "disease vs control",
        "is_calibration_relevant": False,
        "calibration_type_tag": "CLINICAL_TARGET_COHORT (excluded from calibration fitting)",
    },
    # ── Prior calibration outputs (metadata-only reference, not a new dataset)
    {
        "dataset_name": "gaira_calibration_eval_v1/v2/v3 (prior GAIRA calibration outputs)",
        "source_path": str(PROCESSED / "gaira_calibration_eval_v3/"),
        "analyte_or_process": "5 contrasts already run: cspp_fig7_hypox/ergo spikes, uricase_sigma_depletion, uricase_spiked_hypoxanthine_serum, ergothioneine_titration_top_vs_zero",
        "regime": "SERS",
        "substrate_type": "multiple (CSPP paper + Ag colloid)",
        "sample_matrix": "serum",
        "n_spectra": 0,
        "control_variable": "see individual contrasts",
        "is_calibration_relevant": False,  # reference only — not a new dataset
        "calibration_type_tag": "REFERENCE_PRIOR_OUTPUTS (not a new data source)",
    },
]


def stage1_discovery():
    print("\n[STAGE 1] Calibration dataset discovery")
    df = pd.DataFrame(CANDIDATES)
    df.to_csv(TABLES / "calibration_dataset_discovery_v2.csv", index=False)

    admissible = [r for r in CANDIDATES if r["is_calibration_relevant"]]
    excluded   = [r for r in CANDIDATES if not r["is_calibration_relevant"]]

    lines = [
        "# Calibration Dataset Discovery v2",
        "",
        "## Comprehensive storage sweep completed",
        "",
        f"- Total candidates identified: **{len(CANDIDATES)}**",
        f"- Marked calibration-relevant: **{len(admissible)}**",
        f"- Excluded (clinical cohort / reference-only): **{len(excluded)}**",
        "",
        "## Key new discoveries vs prior calibration suite",
        "",
        "The prior `gaira_base_4_hybrid_bsv_calibration_suite_v1` phase (2026-04-23) "
        "only enumerated 9 datasets; this sweep finds materially more, including:",
        "",
        "### Inside `serum_ag_colloids/dataset_spectral_data.zip`",
        "- **Raman metabolites/** — 418 pure powder Raman spectra across 51 analytes "
        "(includes Ergo, HX, Xanth, UA, Gua, Ade, Hypox and many more). The current "
        "loader `load_gobbato_powder` only reads 153 of these.",
        "- **SERS metabolites/** — 81 pure-analyte SERS spectra across 53 analytes "
        "at 100 µM on Ag colloid (includes DNA + RNA, adds analytes beyond the 63-only "
        "NIHMS1547448 set). **Largest pure-SERS expansion opportunity in storage.**",
        "- **SERS metabolites for fitting/** — Hypoxanthine (10 reps) + UAbound (10 reps) "
        "+ UAfree (10 reps) = 30 spectra. Clean **UAfree vs UAbound matrix-effect** "
        "calibration — directly relevant to G01 purine_nucleotide.",
        "- **isotopic/** — 73 UA / UAiso ± HSA ± filter fractionation. "
        "**Isotopic substitution + protein-matrix effect** on UA — unique controlled "
        "shift for G01/G02 calibration.",
        "- **dataset uricase/** — the 20-spectrum uricase enzymatic depletion "
        "(4 cohorts × 5) already used in prior `gaira_calibration_eval_v1`.",
        "",
        "### `cspp_serum/`",
        "- Figures 2/4/5/6/7 — the prior calibration eval only used Figure-7 (150 "
        "spectra). Figures 2/4/5/6 contain additional controlled contrasts (375 more "
        "spectra total) that require manual metadata parsing to identify cohort structure.",
        "",
        "### `ergothioneine_serum/ERG_calibration.csv`",
        "- 55 spectra = 11 ERGO concentrations (0.0-2.0 µM in 0.2 µM steps) × 5 reps "
        "on cAg / 785 nm / 30 mW. **Full SERS titration ladder on the matched-substrate "
        "(cAg matches the NIHMS1547448 training substrate).**",
        "",
        "### Prior processed outputs",
        "- `gaira_calibration_eval_v1`/`v2`/`v3` — 5 contrasts with per-contrast "
        "tables + figures already produced. These are **reference outputs**, not a "
        "data source in themselves.",
        "",
        "## Categorization",
        "",
        "| dataset block | regime | substrate | analyte scope | type tag |",
        "|---|---|---|---|---|",
    ]
    for r in admissible:
        lines.append(
            f"| {r['dataset_name']} | {r['regime']} | {r['substrate_type']} | "
            f"{r['analyte_or_process']} | {r['calibration_type_tag']} |"
        )
    lines += [
        "",
        "## Excluded from calibration (documented)",
        "",
    ]
    for r in excluded:
        lines.append(
            f"- **{r['dataset_name']}** — {r['calibration_type_tag']} "
            f"({r['analyte_or_process']})"
        )
    (REPORTS / "REPORT_calibration_dataset_discovery_v2.md").write_text("\n".join(lines))
    print(f"  emitted calibration_dataset_discovery_v2.csv + REPORT "
          f"({len(admissible)} admissible / {len(excluded)} excluded)")
    return CANDIDATES


# ═════════════════════════════════════════════════════════════════════
# STAGE 2 — Admissibility audit
# ═════════════════════════════════════════════════════════════════════

# Per-dataset admissibility judgment. Label vocabulary:
#   - ADMISSIBLE_READY            — immediately usable by next phase
#   - ADMISSIBLE_AFTER_MINOR_PARSING — small parsing step needed
#   - ADMISSIBLE_BUT_LIMITED      — usable but n or scope is limited
#   - NOT_ADMISSIBLE              — cohort/target-like or not a data source

ADMISSIBILITY = [
    # key = dataset_name
    ("ergothioneine_serum / ERG_calibration.csv",
     "ADMISSIBLE_READY",
     "Wide-format CSV with laser/power/substrate/c metadata cols + wavenumber columns; 55 rows; already clean. Requires a loader that reshapes wide-to-long.",
     "can support: dose-response for ERGO on cAg, replicate consistency (5 reps × 11 concs), low-effect-size floor characterization, ΔBSV reference-mode testing (control = 0.0 µM)",
     "cannot support: substrate generalization (single substrate), multi-analyte family-selectivity"),
    ("cspp_serum / Figure-7_all-spectra-and-metadata.csv",
     "ADMISSIBLE_READY",
     "Already used by prior calibration_eval_v1. 150 spectra with clear cohort structure (Bkg/Erg/Hyp).",
     "can support: spike-in dose-response, within-family overlap (ERGO vs HX both in purine-related chemistry), ambiguity behaviour on spiked analytes",
     "cannot support: multi-substrate generalization (CSPP paper Ag only)"),
    ("cspp_serum / Figure-2_all-spectra-and-metadata.csv",
     "ADMISSIBLE_AFTER_MINOR_PARSING",
     "70 spectra, cohort labels not obvious from column headers. Requires manual metadata inspection to identify cohort structure.",
     "potential: additional CSPP-paper-Ag contrasts for purine/aromatic/glycan calibration",
     "cannot support without parsing: anything"),
    ("cspp_serum / Figure-4_all-spectra-and-metadata.csv",
     "ADMISSIBLE_AFTER_MINOR_PARSING",
     "125 spectra, cohort labels encoded in X1 column (unparsed). Requires metadata interpretation.",
     "potential: additional controlled perturbation contrasts on CSPP substrate",
     "cannot support without parsing: anything"),
    ("cspp_serum / Figure-5_all-spectra-and-metadata.csv",
     "ADMISSIBLE_AFTER_MINOR_PARSING",
     "120 spectra; same issue as Figure-4.",
     "potential: additional CSPP contrasts",
     "cannot support without parsing: anything"),
    ("cspp_serum / Figure-6_all-spectra-and-metadata.csv",
     "ADMISSIBLE_AFTER_MINOR_PARSING",
     "63 spectra; cohort labels unclear from header.",
     "potential: additional CSPP contrasts",
     "cannot support without parsing: anything"),
    ("serum_ag_colloids ZIP :: dataset uricase/",
     "ADMISSIBLE_READY",
     "Already used by prior calibration_eval_v1. 20 spectra, 4 cohorts × 5 reps. Clean +Enzyme / −Enzyme contrast.",
     "can support: enzymatic depletion (UA → allantoin) on purine_nucleotide axis; TRANSFORMATION_ENZYMATIC is the single calibration category flagged as gap in prior suite",
     "cannot support: rate kinetics (no time-course)"),
    ("serum_ag_colloids ZIP :: Raman metabolites/",
     "ADMISSIBLE_READY",
     "418 pure Raman spectra across 51 analytes. Currently loaded only as 153 (3-rep subset) by load_gobbato_powder. Additional reps exist and are usable.",
     "can support: expanded Raman identity + replicate consistency; more reps → tighter CV bounds",
     "cannot support: SERS or substrate-perturbation questions (Raman only)"),
    ("serum_ag_colloids ZIP :: SERS metabolites/ (100 µM)",
     "ADMISSIBLE_READY",
     "81 pure-analyte SERS spectra at 100 µM on Ag colloid. Clean naming convention `SERS_met_<analyte>_100uM_<rep>.txt`.",
     "can support: SERS identity across 53 analytes on Ag colloid (expands the 63-only NIHMS1547448 set, particularly adds DNA + RNA chemistry); cross-regime coherence vs Raman metabolites (same analytes in Raman metabolites/)",
     "cannot support: concentration response (all at 100 µM)"),
    ("serum_ag_colloids ZIP :: SERS metabolites for fitting/",
     "ADMISSIBLE_READY",
     "30 spectra: Hypox (10 reps) + UAfree (10 reps) + UAbound (10 reps). Clean per-rep structure.",
     "can support: UAfree vs UAbound matrix-effect on purine_nucleotide (G01); separate Hypoxanthine 10-rep replicate consistency",
     "cannot support: full protein-matrix generality (only HSA-bound variant)"),
    ("serum_ag_colloids ZIP :: isotopic/",
     "ADMISSIBLE_BUT_LIMITED",
     "73 UA / UAiso ± HSA ± filter fractionation. Rich controlled-shift design but requires careful interpretation (isotope labeling may not map cleanly to current motif registry).",
     "can support: UA 1330/1380 isotopic shift characterization; protein-matrix (HSA) effect on UA band pattern; size-fractionation effect",
     "cannot support: broader isotopic calibration (only UA is labeled)"),
    ("serum_ag_colloids ZIP :: SERS serum Merck/",
     "NOT_ADMISSIBLE",
     "Clinical serum pool (Merck commercial). No controlled perturbation. Falls under 'clinical target cohort' exclusion.",
     "none",
     "none"),
    ("serum_ag_colloids ZIP :: donors serum SERS/",
     "NOT_ADMISSIBLE",
     "Multi-donor serum. Clinical identity, no calibration perturbation.",
     "none",
     "none"),
    ("serum_ag_colloids ZIP :: digitized literature spectra/",
     "ADMISSIBLE_BUT_LIMITED",
     "3 literature digitizations (Gelder 2007, Kim 1987, Stewart 1999). Already partially present in GAIRA `load_digitised_literature`.",
     "can support: cross-source identity check on 3 analytes",
     "cannot support: replicate or perturbation analysis (n=3 total)"),
    ("adenine_sers_control / bAgNPs LOD ladder",
     "ADMISSIBLE_BUT_LIMITED",
     "Already used by prior suite. 7 concentration points. **Confirmed substrate mismatch against v4.5 engine** (bAgNPs ≠ citrate-Ag training substrate).",
     "can support: substrate-specific ERROR characterization (what happens when SERS rules don't transfer); NOT a clean G01 dose-response on the trained substrate",
     "cannot support: G01 dose-response on the trained substrate without substrate-aware recalibration"),
    ("adenine_sers_control / 1ng replicates",
     "ADMISSIBLE_READY",
     "5 reps at fixed 1 ng. Excellent measurement reproducibility bench for SERS at bAgNPs.",
     "can support: SERS measurement-noise CV at bAgNPs (already showed 1.5% CV in v1 suite)",
     "cannot support: chemistry validation (substrate mismatch applies)"),
    ("ramanbiolib",
     "ADMISSIBLE_READY",
     "Primary Raman identity bench. 202 spectra.",
     "can support: family identity + selectivity on Raman",
     "cannot support: SERS or substrate-perturbation questions"),
    ("gobbato powder Raman (currently loaded 3-rep subset)",
     "ADMISSIBLE_READY",
     "153 spectra already loaded; 51 analytes × 3 reps.",
     "can support: Raman replicate consistency + identity",
     "cannot support: SERS"),
    ("amino_acid_raman_grounding / aa.xlsx",
     "ADMISSIBLE_READY",
     "20 amino acids × 1 = 20 spectra.",
     "can support: G10 free_amino_acid identity",
     "cannot support: replicate analysis (n=1 per analyte)"),
    ("digitised_literature (in loader)",
     "ADMISSIBLE_BUT_LIMITED",
     "2 digitised literature spectra in current loader.",
     "can support: trivial sanity check",
     "cannot support: any statistical calibration"),
    ("sers_metabolite_63 (NIHMS1547448, citrate-Ag)",
     "ADMISSIBLE_READY",
     "63 pure-analyte SERS on citrate-Ag. Primary SERS identity bench on the trained substrate.",
     "can support: SERS identity + cross-regime coherence vs Raman of the same analyte",
     "cannot support: dose-response or replicate (n=1 per analyte)"),
    ("nature_serum_sers (and other clinical cohorts)",
     "NOT_ADMISSIBLE",
     "Target clinical cohort. Excluded from calibration fitting per user doctrine.",
     "none",
     "none"),
    ("gaira_calibration_eval_v1/v2/v3 (prior GAIRA calibration outputs)",
     "NOT_ADMISSIBLE",
     "Prior output directory, not a new data source. Referenced for historical context.",
     "none (reference only)",
     "none"),
]


def stage2_admissibility():
    print("\n[STAGE 2] Admissibility audit")
    rows = []
    for name, label, reason, supports, not_supports in ADMISSIBILITY:
        rows.append({
            "dataset_name": name,
            "admissibility_label": label,
            "reason": reason,
            "supports": supports,
            "does_not_support": not_supports,
        })
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "calibration_admissibility_audit_v2.csv", index=False)

    counts = df["admissibility_label"].value_counts().to_dict()
    lines = [
        "# Calibration Admissibility Audit v2",
        "",
        "## Summary counts",
        "",
    ]
    for k, v in counts.items():
        lines.append(f"- **{k}**: {v}")
    lines += [
        "",
        "## Per-dataset decisions",
        "",
        "| dataset | label | reason |",
        "|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['dataset_name']} | {r['admissibility_label']} | "
                      f"{r['reason']} |")
    lines += [
        "",
        "## What the ADMISSIBLE_READY set supports together",
        "",
        "- **Identity bench** (Raman + SERS) across ≥ 50 analytes",
        "- **Replicate consistency** on Raman (gobbato 3-rep; expanded via full 418-spectra zip) and SERS (adenine 1ng × 5, hypox × 10, UAfree × 10, UAbound × 10)",
        "- **Dose-response / titration** on ERGO (11 points × 5 reps, cAg) — the cleanest titration we have on the matched training substrate",
        "- **Transformation / enzymatic depletion** on UA via uricase (4 cohorts × 5) — covers the category flagged as 'gap' in the prior calibration suite",
        "- **Spike-in overlap** on CSPP (Hyp-spike + Erg-spike into serum)",
        "- **Matrix effect** via UAfree vs UAbound (HSA complex)",
        "- **Isotopic shift** on UA / UAiso ± HSA ± filter",
        "- **Cross-regime coherence** on ≥ 50 analytes (Raman metabolites + SERS metabolites at 100 µM share a large intersection)",
        "- **Substrate-mismatch characterization** on adenine bAgNPs (v1 suite already revealed this as a calibration failure mode)",
        "",
        "## ADMISSIBLE_AFTER_MINOR_PARSING items",
        "",
        "CSPP Figures 2/4/5/6 — 378 additional spectra that would become usable with a small metadata-parse step (inspection of the X1 column + per-figure text in the CSPP paper).",
    ]
    (REPORTS / "REPORT_calibration_admissibility_audit_v2.md"
     ).write_text("\n".join(lines))
    print(f"  emitted admissibility audit ({counts})")
    return rows


# ═════════════════════════════════════════════════════════════════════
# STAGE 3 — Expected-signal registry per admissible dataset
# ═════════════════════════════════════════════════════════════════════

EXPECTED_SIGNAL = [
    # (dataset, calibration_type, expected_family, expected_mss_hits,
    #  expected_bsv_direction, expected_delta_bsv_direction, expected_ambiguity,
    #  monotonic, anchor_strength, notes)
    ("ergothioneine_serum / ERG_calibration.csv",
     "DOSE_RESPONSE",
     "G01 purine_nucleotide (ERGO imidazole 720-740 zone fires G01; may also touch G10 via the thiolate carboxyl)",
     "ergothioneine MSS (if registered) OR imidazole-family nearest MSS",
     "G01 up monotonically with [ERGO]; weak effect size at µM concentrations",
     "ΔG01 (relative to 0.0 µM control) cleaner than absolute G01",
     "low at top concentration; higher near zero",
     "yes (weakly)",
     "MEDIUM_WEAK (µM effect size, prior v1 run returned 'inconsistent')",
     "Serum matrix + µM signal is a known weak-recovery case; expected to be a FLOOR test"),
    ("cspp_serum / Figure-7 hypoxanthine spike",
     "DOSE_RESPONSE + MIXTURE_OVERLAP",
     "G01/G02 purine family (HX is purine metabolite; 725 cm⁻¹ ring breathing in 700-740 window)",
     "hypoxanthine MSS",
     "Bkg → Hyp: G01/G02 up",
     "ΔG01 cleanly up (prior v1 PASS)",
     "low in pure HX-spike; higher if G07 aromatic competes",
     "yes",
     "STRONG (prior PASS in v1 calibration eval; effect = +1.70)",
     "Confirmed strong calibration anchor by prior GAIRA work"),
    ("cspp_serum / Figure-7 ergothioneine spike",
     "DOSE_RESPONSE",
     "G01 via ERGO imidazole (weaker effect) or G10 if thiolate dominates",
     "ergothioneine MSS",
     "Bkg → Erg: G01 weakly up",
     "ΔG01 small",
     "moderate (prior top window was 1020-1080 nucleic_acid_backbone — unexpected axis)",
     "yes (weak)",
     "WEAK (prior v1 result = weak)",
     "Known weak-recovery case — useful as a FLOOR / SENSITIVE-tier benchmark"),
    ("serum_ag_colloids ZIP :: dataset uricase/",
     "TRANSFORMATION_ENZYMATIC",
     "G01/G02 (uricase converts UA → allantoin; G01 purine signal should DROP)",
     "uric_acid MSS; secondary allantoin signal (if registered)",
     "+Enzyme: G01 DOWN vs −Enzyme",
     "ΔG01 cleanly negative (depletion direction)",
     "moderate-high (UA also fires G07 via 635 aromatic and G05 via 890 glycan windows)",
     "not monotonic (binary ±Enzyme)",
     "STRONG_BUT_INCONSISTENT_IN_V1 (prior v1 result = 'inconsistent' on 5v5 — may be substrate artifact)",
     "Single cleanest TRANSFORMATION_ENZYMATIC anchor GAIRA has; prior inconsistency deserves re-audit"),
    ("serum_ag_colloids ZIP :: Raman metabolites/",
     "IDENTITY_PURE + REPLICATE",
     "per-analyte family (broad coverage across 11 families)",
     "per-analyte MSS",
     "correct top-family per spectrum",
     "(ΔBSV unnecessary for identity)",
     "low",
     "n/a",
     "STRONG_IDENTITY_ANCHOR",
     "418 spectra is a larger replicate base than current 153; CV bounds can tighten"),
    ("serum_ag_colloids ZIP :: SERS metabolites/ (100 µM)",
     "IDENTITY_PURE + CROSS_REGIME_COHERENCE",
     "per-analyte family on Ag colloid",
     "per-analyte MSS",
     "correct top-family; cross-regime agreement vs Raman metabolites",
     "(identity → ΔBSV optional)",
     "moderate (SERS purine overfire expected)",
     "n/a",
     "STRONG_IDENTITY_ANCHOR (expanded SERS coverage)",
     "Likely the highest-leverage single new dataset in this audit — expands SERS diversity materially"),
    ("serum_ag_colloids ZIP :: SERS metabolites for fitting/",
     "MIXTURE_OVERLAP + REPLICATE",
     "G01 purine for UAfree/UAbound; G02 for Hypox",
     "uric_acid + hypoxanthine MSS",
     "UAfree vs UAbound: UA band pattern shifts with HSA binding; Hypox 10-rep: tight CV",
     "ΔBSV(bound - free) testable for matrix effect",
     "moderate (UA↔HX purine overfire zone)",
     "n/a (binary contrast + replicate)",
     "STRONG",
     "Directly relevant to G01 purine_nucleotide matrix effects"),
    ("serum_ag_colloids ZIP :: isotopic/",
     "OTHER_CONTROLLED_SHIFT",
     "G01 purine via UA isotopic shift (UAiso shifts 1380 → ~1370 region)",
     "uric_acid MSS",
     "absolute BSV should not collapse but band-position shift may alter top-group ordering",
     "ΔBSV(UA vs UAiso) measures isotope-sensitivity of GAIRA motifs",
     "moderate",
     "n/a",
     "NICHE_BUT_UNIQUE",
     "Isotopic substitution is a rare controlled shift — great for validating band-specificity of motifs"),
    ("adenine_sers_control / bAgNPs LOD ladder",
     "DOSE_RESPONSE + SUBSTRATE_PERTURBATION",
     "G01 purine_nucleotide (adenine) on bAgNPs",
     "adenine MSS",
     "expected G01 up but prior v1 suite empirically observed G01 top-1 = 0% (substrate mismatch)",
     "ΔG01 likely flat or negative due to substrate-calibration gap",
     "high (substrate mismatch amplifies ambiguity)",
     "expected yes — empirically NO (ρ = -0.79 in v1 suite)",
     "SUBSTRATE_ERROR_ANCHOR",
     "Use as a DIAGNOSTIC for substrate-generalization failure, not as a positive dose-response test"),
    ("adenine_sers_control / 1ng replicates",
     "REPLICATE_REPRODUCIBILITY",
     "whatever family the bAgNPs substrate routes to (empirically G07 / G11 / G10 — NOT G01)",
     "n/a (substrate mismatch)",
     "stable across 5 reps (CV ~1.5%)",
     "n/a",
     "stable",
     "n/a",
     "MEASUREMENT_STABILITY_ANCHOR",
     "Confirms SERS measurement noise is NOT the issue on bAgNPs; substrate rules are"),
    ("ramanbiolib",
     "IDENTITY_PURE",
     "per-analyte (all 11 families)",
     "per-analyte MSS",
     "correct top-family",
     "n/a",
     "low",
     "n/a",
     "STRONG",
     "Primary Raman identity bench"),
    ("gobbato powder Raman (currently loaded 3-rep subset)",
     "IDENTITY_PURE + REPLICATE",
     "per-analyte",
     "per-analyte MSS",
     "correct top-family; 3-rep stability",
     "n/a",
     "low",
     "n/a",
     "STRONG",
     "Already showing 98% agreement, 1% CV in v1 suite"),
    ("amino_acid_raman_grounding / aa.xlsx",
     "IDENTITY_PURE",
     "G10 free_amino_acid",
     "per-AA MSS",
     "G10 top-1",
     "n/a",
     "moderate (free-AA within-family overlap)",
     "n/a",
     "STRONG",
     "G10 family bench"),
    ("sers_metabolite_63 (NIHMS1547448, citrate-Ag)",
     "IDENTITY_PURE + CROSS_REGIME_COHERENCE",
     "per-analyte",
     "per-analyte MSS",
     "correct top-family on citrate-Ag (trained substrate)",
     "ΔBSV vs Raman equivalent = coherence metric",
     "moderate (purine SERS overfire)",
     "n/a",
     "STRONG_ON_TRAINED_SUBSTRATE",
     "Primary SERS identity bench"),
]


def stage3_expected_signal():
    print("\n[STAGE 3] Expected-signal registry")
    rows = []
    for (ds, ctype, fam, mss, bsv_dir, dbsv_dir, amb, mono, anchor, notes) in EXPECTED_SIGNAL:
        rows.append({
            "dataset": ds,
            "calibration_type": ctype,
            "expected_dominant_family": fam,
            "expected_top_mss_hits": mss,
            "expected_bsv_direction": bsv_dir,
            "expected_delta_bsv_direction": dbsv_dir,
            "expected_ambiguity": amb,
            "monotonic": mono,
            "anchor_strength": anchor,
            "notes": notes,
        })
    pd.DataFrame(rows).to_csv(
        TABLES / "calibration_expected_signal_registry_v2.csv", index=False,
    )

    lines = [
        "# Calibration Expected-Signal Registry v2",
        "",
        "## Per-dataset expected behaviour",
        "",
        "| dataset | type | expected family | anchor strength |",
        "|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['dataset']} | {r['calibration_type']} | "
            f"{r['expected_dominant_family']} | {r['anchor_strength']} |"
        )
    lines += [
        "",
        "## Strong vs weak anchors",
        "",
        "### Strong (high information per dataset)",
        "- cspp_fig7_hypoxanthine_spike (prior PASS)",
        "- uricase_sigma_depletion (UNIQUE enzymatic anchor)",
        "- SERS metabolites/ (100 µM pure-analyte SERS across 53 analytes)",
        "- SERS metabolites for fitting/ (UAfree vs UAbound)",
        "- Raman metabolites/ (418 pure-analyte Raman)",
        "- sers_metabolite_63 (citrate-Ag SERS identity)",
        "- ramanbiolib (Raman identity)",
        "- gobbato 3-rep (replicate)",
        "",
        "### Weak / floor",
        "- ERG titration (µM effect size)",
        "- CSPP Erg spike (prior weak)",
        "- digitised literature (n=2-3)",
        "",
        "### Substrate-error anchors (diagnostic, not positive)",
        "- adenine bAgNPs LOD (v1 already showed substrate mismatch)",
        "",
        "### Niche",
        "- isotopic UA / UAiso ± HSA ± filter",
        "",
        "## Expected-shift chemistry notes (analyte-centric)",
        "",
        "- **UA (uric acid)**: G01 anchor; characteristic SERS at ~635 / 890 / 1130 cm⁻¹; can leak to G07 aromatic (635 zone) and G05 glycan (890 zone).",
        "- **Hypoxanthine (HX)**: G01/G02 anchor; 725 cm⁻¹ ring breathing in the 700-740 purine window; 640 cm⁻¹ secondary.",
        "- **Adenine**: G01 purine_nucleotide; 720-740 SERS zone; empirically substrate-specific.",
        "- **Ergothioneine (ERGO)**: G01 via imidazole 720-740; thiolate carboxyl can touch G10.",
        "- **ERG-bound to serum protein**: 1020-1080 region (nucleic_acid_backbone confusion) — known weak.",
    ]
    (REPORTS / "REPORT_calibration_expected_signal_registry_v2.md"
     ).write_text("\n".join(lines))
    print(f"  emitted expected-signal registry ({len(rows)} datasets)")
    return rows


# ═════════════════════════════════════════════════════════════════════
# STAGE 4 — Substrate-aware physics applicability per dataset
# ═════════════════════════════════════════════════════════════════════

SUBSTRATE_PHYSICS = [
    # (dataset, apply_for_inference, apply_for_interpretation, substrate_family,
    #  rationale, rules_already_exist, new_rule_needed)
    ("ergothioneine_serum / ERG_calibration.csv",
     "YES",
     "YES",
     "cAg (citrate-Ag colloid)",
     "SERS on the MATCHED training substrate (citrate-Ag is the NIHMS1547448 regime). The current SERS observation-model rules v1 (dampen purine 720-740, UA-carotenoid 1517, amide-I 1600-1700, HX 640, boost Phe 1003, MSS-boost glycan) were calibrated to citrate-Ag and should apply here. Serum-matrix background should be acknowledged but not corrected for in inference.",
     "YES (current v1 SERS observation rules cover cAg)",
     "NO new rules required"),
    ("cspp_serum / Figure-7 hypoxanthine spike",
     "CONDITIONAL",
     "YES",
     "plasmonic paper Ag (CSPP) — different substrate from citrate-Ag",
     "CSPP paper Ag has different enhancement physics from citrate-Ag. Apply SERS observation rules ONLY if inference treats CSPP as a known in-scope substrate. Safer alternative: inference without substrate-aware dampening (generic SERS baseline), then flag substrate caveat in interpretation.",
     "PARTIAL (substrate_family caveat exists; no CSPP-specific rules)",
     "would benefit from CSPP-specific physics rules in a future pass; currently acceptable to run without scoring-level dampening and emit substrate caveat"),
    ("cspp_serum / Figure-7 ergothioneine spike",
     "CONDITIONAL",
     "YES",
     "plasmonic paper Ag (CSPP)",
     "Same as Fig-7 HX: CSPP-specific rules pending; run with substrate caveat.",
     "PARTIAL",
     "same as above"),
    ("cspp_serum / Figure-2/4/5/6",
     "CONDITIONAL",
     "YES",
     "plasmonic paper Ag",
     "Same substrate as Fig-7; same conditional applicability",
     "PARTIAL",
     "same as above"),
    ("serum_ag_colloids ZIP :: dataset uricase/",
     "YES",
     "YES",
     "Ag colloid (citrate-Ag-like; serum-matrix)",
     "Matched training substrate family. Current SERS rules apply. Serum-matrix background should be acknowledged in interpretation.",
     "YES",
     "NO"),
    ("serum_ag_colloids ZIP :: Raman metabolites/",
     "NO",
     "NO",
     "pure powder (no substrate)",
     "Pure powder Raman does not involve plasmonic substrate; no SERS physics applies. Run with SERS rules OFF (regime = Raman).",
     "n/a",
     "NO"),
    ("serum_ag_colloids ZIP :: SERS metabolites/ (100 µM)",
     "YES",
     "YES",
     "Ag colloid (citrate-Ag-like)",
     "Same substrate family as trained; pure-analyte SERS. SERS rules apply fully.",
     "YES",
     "NO"),
    ("serum_ag_colloids ZIP :: SERS metabolites for fitting/",
     "YES",
     "YES",
     "Ag colloid",
     "Same as above. Matrix-bound variants (UAbound) require interpretation caveat about HSA binding.",
     "YES",
     "NO (matrix-effect notes pending empirical ingest)"),
    ("serum_ag_colloids ZIP :: isotopic/",
     "YES",
     "YES",
     "Ag colloid",
     "Same training substrate family. Isotopic shift interpretation requires a separate annotation layer (not in current substrate-physics rules).",
     "PARTIAL (substrate rules apply; isotope-shift interpretation does not have dedicated rules)",
     "isotope-shift interpretation rule block would be NEW and narrow-scope"),
    ("adenine_sers_control / bAgNPs LOD ladder",
     "CONDITIONAL",
     "YES (mandatory)",
     "bAgNPs (biologically synthesised Ag NPs) — NOT the training substrate",
     "v1 suite empirically showed current SERS rules DON'T transfer from citrate-Ag to bAgNPs (G01 top-1 = 0%, ρ = -0.79). Inference on bAgNPs should either (a) RUN WITH NO SERS PHYSICS (treat as enhanced-Raman baseline) or (b) require dedicated bAgNPs substrate-conditioned rules which GAIRA does NOT currently have. Interpretation MUST flag substrate scope.",
     "NO (bAgNPs rules do not exist)",
     "YES — dedicated bAgNPs substrate rule block is REQUIRED before valid inference; absent that, the dataset functions as a substrate-mismatch diagnostic"),
    ("adenine_sers_control / 1ng replicates",
     "CONDITIONAL",
     "YES (mandatory)",
     "bAgNPs",
     "Same as the LOD ladder — substrate outside scope.",
     "NO",
     "same — needs dedicated bAgNPs rule block for chemistry-valid inference"),
    ("ramanbiolib",
     "NO",
     "NO",
     "pure (no substrate)",
     "Raman pure references; regime = Raman; SERS rules off.",
     "n/a",
     "NO"),
    ("gobbato powder Raman (currently loaded 3-rep subset)",
     "NO",
     "NO",
     "pure powder",
     "Raman pure references; SERS rules off.",
     "n/a",
     "NO"),
    ("amino_acid_raman_grounding / aa.xlsx",
     "NO",
     "NO",
     "pure",
     "Raman; SERS rules off.",
     "n/a",
     "NO"),
    ("sers_metabolite_63 (NIHMS1547448, citrate-Ag)",
     "YES",
     "YES",
     "citrate-Ag colloid (TRAINED SUBSTRATE)",
     "Primary SERS identity bench on the trained substrate. All existing SERS rules apply.",
     "YES",
     "NO"),
]


def stage4_substrate_physics():
    print("\n[STAGE 4] Substrate-aware physics applicability")
    rows = []
    for (ds, inf, interp, sub, rat, exists, need) in SUBSTRATE_PHYSICS:
        rows.append({
            "dataset": ds,
            "apply_substrate_physics_for_inference": inf,
            "apply_substrate_physics_for_interpretation": interp,
            "substrate_family": sub,
            "rationale": rat,
            "required_rules_already_exist": exists,
            "new_rule_support_needed": need,
        })
    pd.DataFrame(rows).to_csv(
        TABLES / "calibration_substrate_physics_audit_v1.csv", index=False,
    )

    lines = [
        "# Calibration Substrate-Aware Physics Audit v1",
        "",
        "## Per-dataset applicability",
        "",
        "| dataset | inference | interpretation | substrate | rules exist | new needed |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['dataset']} | {r['apply_substrate_physics_for_inference']} | "
            f"{r['apply_substrate_physics_for_interpretation']} | "
            f"{r['substrate_family']} | "
            f"{r['required_rules_already_exist']} | "
            f"{r['new_rule_support_needed']} |"
        )
    lines += [
        "",
        "## Substrate-family partition",
        "",
        "- **Pure Raman (no substrate)** — ramanbiolib, gobbato powder, aa.xlsx: SERS physics OFF. Run unmodified Raman path. (4 datasets)",
        "- **Trained SERS substrate (citrate-Ag colloid)** — ergothioneine_serum ERG_calibration, sers_metabolite_63, serum_ag_colloids uricase/SERS-met/fitting/isotopic: SERS physics ON for both inference and interpretation. **No new rules needed.** (6 datasets)",
        "- **Related SERS substrate (plasmonic paper Ag)** — CSPP Figures 2/4/5/6/7: physics CONDITIONAL. Safer path is inference without substrate-specific dampening + interpretation caveat emitting `substrate_family=CSPP_paper_Ag` and `substrate_scope_caveat=True`. (5 datasets)",
        "- **Out-of-scope SERS substrate (bAgNPs)** — adenine_sers_control LOD + replicates: v1 calibration suite already demonstrated empirically that current SERS rules do NOT transfer. Inference is INVALID without new bAgNPs rules. These datasets function as **substrate-mismatch diagnostics only** until new rules exist. (2 datasets)",
        "",
        "## Safe to evaluate directly (no new substrate rules required)",
        "",
        "- ramanbiolib",
        "- gobbato powder Raman (3-rep subset)",
        "- aa.xlsx",
        "- digitised_literature (2 refs in current loader)",
        "- sers_metabolite_63",
        "- ergothioneine_serum / ERG_calibration.csv",
        "- serum_ag_colloids :: dataset uricase/",
        "- serum_ag_colloids :: Raman metabolites/",
        "- serum_ag_colloids :: SERS metabolites/",
        "- serum_ag_colloids :: SERS metabolites for fitting/",
        "- serum_ag_colloids :: digitised literature spectra/",
        "",
        "## Require substrate-aware inference/interpretation to be scientifically valid",
        "",
        "- CSPP Figures 2/4/5/6/7 — run with substrate caveat; future work: dedicated CSPP rule block",
        "- serum_ag_colloids :: isotopic/ — isotope-shift interpretation annotations (narrow, new)",
        "- adenine_sers_control (both) — currently UNUSABLE for positive calibration; use only as substrate-mismatch diagnostic until bAgNPs rule block is implemented",
    ]
    (REPORTS / "REPORT_calibration_substrate_physics_audit_v1.md"
     ).write_text("\n".join(lines))
    print(f"  emitted substrate physics audit ({len(rows)} datasets)")
    return rows


# ═════════════════════════════════════════════════════════════════════
# STAGE 5 — Pre-calibration implementation checklist
# ═════════════════════════════════════════════════════════════════════

CHECKLIST = [
    ("ergothioneine_serum / ERG_calibration.csv", "DOSE_RESPONSE + REPLICATE", "ergothioneine",
     "YES", "0.0 µM cohort (explicit control)", "YES (inference + interpretation)",
     "G01 purine_nucleotide (weak)", "ergothioneine MSS",
     "wide-to-long reshape loader (metadata cols → long-form)",
     "Highest-leverage ERGO anchor on matched substrate"),
    ("cspp_serum / Figure-7 hypoxanthine spike", "MIXTURE_OVERLAP + DOSE_RESPONSE", "hypoxanthine",
     "YES", "Bkg cohort (explicit control)", "YES interpretation; CONDITIONAL inference",
     "G01/G02 purine family", "hypoxanthine MSS",
     "(none — parsed by prior calibration_eval_v1)",
     "Re-use prior parser; extend with substrate caveat output"),
    ("cspp_serum / Figure-7 ergothioneine spike", "DOSE_RESPONSE", "ergothioneine",
     "YES", "Bkg cohort", "YES interpretation; CONDITIONAL inference",
     "G01 (weak)", "ergothioneine MSS",
     "(none)",
     "Known weak anchor — useful for SENSITIVE-tier calibration"),
    ("cspp_serum / Figures 2/4/5/6", "OTHER_CONTROLLED_SHIFT", "multiple",
     "NOT_YET (needs parsing)", "unknown", "YES interpretation",
     "unknown", "unknown",
     "metadata parser to identify cohort structure per figure",
     "378 additional spectra unlocked after parsing"),
    ("serum_ag_colloids :: dataset uricase/", "TRANSFORMATION_ENZYMATIC + MIXTURE_OVERLAP", "uric acid / hypoxanthine",
     "YES", "SerumSigma cohort (no-enzyme, no-spike)", "YES (both)",
     "G01/G02", "uric_acid MSS",
     "(none — parsed by prior calibration_eval_v1)",
     "Single strongest ENZYMATIC anchor GAIRA has"),
    ("serum_ag_colloids :: Raman metabolites/", "IDENTITY_PURE + REPLICATE", "51 metabolites",
     "YES", "no control needed (identity)", "NO",
     "per-analyte", "per-analyte MSS",
     "expanded loader reading zip (currently only 3-rep subset loaded)",
     "Expands replicate base from 153 → 418 spectra"),
    ("serum_ag_colloids :: SERS metabolites/", "IDENTITY_PURE + CROSS_REGIME_COHERENCE", "53 metabolites",
     "YES", "no control needed for identity; cross-regime compares to Raman metabolites", "YES (both)",
     "per-analyte", "per-analyte MSS",
     "loader reading zip sub-folder",
     "Largest SERS pure-analyte expansion available"),
    ("serum_ag_colloids :: SERS metabolites for fitting/", "MIXTURE_OVERLAP + REPLICATE", "UAfree / UAbound / hypoxanthine",
     "YES", "UAfree cohort as control (for UAbound)", "YES (both)",
     "G01 purine", "uric_acid MSS",
     "loader for 30-spectrum block",
     "Clean matrix-effect anchor (UAfree vs UAbound)"),
    ("serum_ag_colloids :: isotopic/", "OTHER_CONTROLLED_SHIFT", "UA / UAiso ± HSA",
     "YES_BUT_LIMITED", "UA cohort as control (for UAiso)", "YES (both); isotope-shift annotations new",
     "G01", "uric_acid MSS",
     "loader for 73-spectrum block + isotope annotation layer",
     "Niche unique anchor for band-specificity"),
    ("adenine_sers_control / bAgNPs LOD", "SUBSTRATE_MISMATCH_DIAGNOSTIC", "adenine",
     "DIAGNOSTIC_ONLY", "10pg or 100pg lowest concentration (or a no-spike baseline if available)", "YES interpretation; NO inference",
     "G01 (expected) / G07/G10/G11 (observed)", "adenine MSS",
     "existing loader",
     "Keep as substrate-mismatch diagnostic; do not treat as positive G01 dose-response"),
    ("adenine_sers_control / 1ng replicates", "REPLICATE_REPRODUCIBILITY", "adenine at 1ng",
     "YES", "(none)", "YES interpretation; NO inference",
     "n/a (substrate mismatch)", "adenine MSS",
     "existing loader",
     "Substrate-mismatch diagnostic — measures SERS measurement CV only"),
    ("ramanbiolib", "IDENTITY_PURE", "all 11 families",
     "YES", "(identity)", "NO", "per-analyte", "per-analyte MSS",
     "(none — existing loader)",
     "Primary Raman bench"),
    ("gobbato powder Raman 3-rep", "IDENTITY_PURE + REPLICATE", "51 metabolites",
     "YES", "(identity)", "NO", "per-analyte", "per-analyte MSS",
     "(existing loader)",
     "Expand later if the full 418-spectra loader is built"),
    ("aa.xlsx", "IDENTITY_PURE", "20 amino acids", "YES", "(identity)", "NO",
     "G10 free_amino_acid", "per-AA MSS", "(existing)", "G10 bench"),
    ("digitised_literature", "IDENTITY_PURE", "2 refs", "YES", "(identity)", "NO",
     "per-analyte", "per-analyte MSS", "(existing)", "Trivial n=2"),
    ("sers_metabolite_63", "IDENTITY_PURE + CROSS_REGIME_COHERENCE", "63 metabolites",
     "YES", "(identity)", "YES (both)", "per-analyte", "per-analyte MSS",
     "(existing)", "Primary SERS bench"),
]


def stage5_checklist():
    print("\n[STAGE 5] Pre-calibration implementation checklist")
    rows = []
    for (ds, ctype, analyte, admissible, dbsv_ref, phys, fam, mss, missing, comments) in CHECKLIST:
        rows.append({
            "dataset_name": ds,
            "calibration_type": ctype,
            "analyte_or_process": analyte,
            "admissible_for_next_phase": admissible,
            "delta_bsv_reference_mode": dbsv_ref,
            "substrate_physics_required": phys,
            "expected_main_family": fam,
            "expected_main_mss_hits": mss,
            "missing_prerequisites": missing,
            "comments": comments,
        })
    pd.DataFrame(rows).to_csv(
        TABLES / "calibration_next_phase_checklist_v1.csv", index=False,
    )
    lines = [
        "# Pre-Calibration Implementation Checklist v1",
        "",
        "## Ready-to-run subset (no new prerequisites)",
        "",
    ]
    for r in rows:
        if r["admissible_for_next_phase"] == "YES" and not r["missing_prerequisites"].strip().startswith("(none"):
            continue
        if r["admissible_for_next_phase"] == "YES":
            lines.append(f"- **{r['dataset_name']}** ({r['calibration_type']}) — "
                         f"expected family `{r['expected_main_family']}`, "
                         f"ΔBSV ref = `{r['delta_bsv_reference_mode']}`")
    lines += [
        "",
        "## Requires minor parsing / loader work before running",
        "",
    ]
    for r in rows:
        if r["admissible_for_next_phase"] == "YES" and not r["missing_prerequisites"].strip().startswith("(none"):
            lines.append(f"- **{r['dataset_name']}** — missing: {r['missing_prerequisites']}")
        elif r["admissible_for_next_phase"].startswith("NOT_YET"):
            lines.append(f"- **{r['dataset_name']}** — missing: {r['missing_prerequisites']}")
    lines += [
        "",
        "## Use only as diagnostic (not as positive calibration)",
        "",
    ]
    for r in rows:
        if "DIAGNOSTIC" in r["admissible_for_next_phase"]:
            lines.append(f"- **{r['dataset_name']}** — {r['comments']}")
    (REPORTS / "REPORT_calibration_next_phase_checklist_v1.md"
     ).write_text("\n".join(lines))
    print(f"  emitted checklist ({len(rows)} items)")
    return rows


# ═════════════════════════════════════════════════════════════════════
# STAGE 6 — Gap analysis
# ═════════════════════════════════════════════════════════════════════

def stage6_gap_analysis():
    print("\n[STAGE 6] Gap analysis")
    lines = [
        "# Calibration Gap Analysis v1",
        "",
        "## Categories now covered by admissible data",
        "",
        "| category | covered by | strength |",
        "|---|---|---|",
        "| IDENTITY_PURE | ramanbiolib + gobbato + aa.xlsx + lit + sers_metabolite_63 + serum_ag_colloids::Raman metabolites + serum_ag_colloids::SERS metabolites | STRONG (6 datasets, 900+ spectra, Raman + SERS, 11 families) |",
        "| DOSE_RESPONSE | ERG_calibration (11 conc × 5 rep, cAg) + adenine bAgNPs LOD (substrate-mismatch) + CSPP fig7 HX/Erg spike (binary) | MEDIUM — one strong matched-substrate titration, one binary spike, one diagnostic substrate-mismatch |",
        "| REPLICATE_REPRODUCIBILITY | gobbato 3-rep × 51 + adenine 1ng × 5 + fitting Hypox/UAfree/UAbound × 10 each | STRONG |",
        "| TRANSFORMATION_ENZYMATIC | serum_ag_colloids uricase dataset (4 cohorts × 5) | PRESENT (was flagged as gap in v1 suite — now RESOLVED) |",
        "| MIXTURE_OVERLAP | CSPP fig7 spikes + serum_ag_colloids UAfree/UAbound + synthetic 50/50 proxies (v1 suite) | MEDIUM (no real experimental multi-analyte mixture series) |",
        "| SUBSTRATE_PERTURBATION | adenine bAgNPs vs citrate-Ag (different substrate, same analyte); Raman metabolites vs SERS metabolites (same analyte, different regime) | MEDIUM |",
        "| CROSS_REGIME_COHERENCE | Raman metabolites + SERS metabolites at 100 µM (same analyte set, Raman + SERS on Ag colloid) — **best cross-regime anchor in GAIRA storage** | STRONG (newly discovered in this audit) |",
        "| OTHER_CONTROLLED_SHIFT | isotopic UA/UAiso ± HSA ± filter | NICHE |",
        "",
        "## Still missing — highest priority",
        "",
        "1. **Enzymatic kinetics (time-course)** — uricase dataset is binary ±Enzyme; no time-course. Kinetic depletion modelling is NOT possible with current data. **Blocking only for rate-based validation** — not blocking for depletion-direction validation.",
        "2. **Non-purine DOSE_RESPONSE series** — all titrations in GAIRA storage are purine-family (adenine, HX, ERGO). No clean dose-response on G05 glycan, G08 lipid_acyl, G09 sterol, G10 free_amino_acid, etc. **Non-blocking** for initial calibration but limits per-family titration calibration.",
        "3. **bAgNPs substrate calibration** — the only SERS substrate outside the trained scope that we HAVE data on. Requires dedicated rule block OR empirical recalibration using the bAgNPs adenine replicate set. **Non-blocking** for calibration phase (can proceed with adenine_sers_control as diagnostic-only).",
        "4. **Real experimental mixtures** — no controlled multi-analyte mixture at known ratios. Synthetic 50/50 proxies are the current stand-in. **Non-blocking** for initial readiness.",
        "5. **Family-specific calibration anchors for G02 / G06 / G10** — SENSITIVE-tier families have no dedicated calibration dataset. G02 is adjacent to G01 purine anchors but only indirectly tested; G06 (protein_polypeptide) and G10 (free_amino_acid) lack direct titration or transformation data. **Non-blocking** but limits SENSITIVE-tier calibration.",
        "",
        "## Previously flagged gap that is now RESOLVED",
        "",
        "The `gaira_base_4_hybrid_bsv_calibration_suite_v1` (2026-04-23) explicitly "
        "recorded **TRANSFORMATION / ENZYMATIC** as an uncovered calibration "
        "category. This audit finds the **serum_ag_colloids :: dataset uricase/** "
        "block (already used in prior `gaira_calibration_eval_v1`) which provides "
        "a clean enzymatic depletion anchor. The prior suite missed this because "
        "it did not search `/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_calibration_eval_*/` "
        "nor extract the full zip contents of `serum_ag_colloids/dataset_spectral_data.zip`.",
        "",
        "## Blocking vs non-blocking",
        "",
        "- **Blocking for the next analyte-centric calibration phase**: none. The "
        "ADMISSIBLE_READY set is sufficient to run a scientifically valid analyte-"
        "centric calibration.",
        "- **Blocking for production SERS deployment on non-citrate-Ag substrates**: "
        "yes (bAgNPs rule block absent). Passive readout on bAgNPs is NOT "
        "scientifically supported by the current engine.",
    ]
    (REPORTS / "REPORT_calibration_gap_analysis_v1.md"
     ).write_text("\n".join(lines))
    print(f"  emitted gap analysis")


# ═════════════════════════════════════════════════════════════════════
# STAGE 7 — Final audit readiness decision
# ═════════════════════════════════════════════════════════════════════

def stage7_readiness(admissibility_rows, substrate_rows, checklist_rows):
    print("\n[STAGE 7] Final audit readiness decision")
    n_ready = sum(1 for r in admissibility_rows if r["admissibility_label"] == "ADMISSIBLE_READY")
    n_parsing = sum(1 for r in admissibility_rows if r["admissibility_label"] == "ADMISSIBLE_AFTER_MINOR_PARSING")
    n_limited = sum(1 for r in admissibility_rows if r["admissibility_label"] == "ADMISSIBLE_BUT_LIMITED")
    n_new_rules_needed = sum(1 for r in substrate_rows if r["new_rule_support_needed"].upper().startswith("YES"))

    if n_ready >= 6 and n_new_rules_needed == 0:
        decision = "READY_TO_RUN_ANALYTE_CENTRIC_CALIBRATION"
    elif n_ready >= 5 and n_new_rules_needed >= 1:
        decision = "READY_BUT_NEEDS_SUBSTRATE_RULES_ON_SPECIFIC_DATASETS"
    elif n_parsing >= 1:
        decision = "READY_BUT_SOME_DATASETS_REQUIRE_MINOR_PARSING"
    else:
        decision = "NOT_READY_FOR_FULL_CALIBRATION"

    lines = [
        "# Calibration Audit Readiness v2",
        "",
        f"**Decision: {decision}**",
        "",
        "## Counts by admissibility",
        "",
        f"- ADMISSIBLE_READY: **{n_ready}**",
        f"- ADMISSIBLE_AFTER_MINOR_PARSING: **{n_parsing}**",
        f"- ADMISSIBLE_BUT_LIMITED: **{n_limited}**",
        f"- datasets requiring new substrate rules before VALID inference: **{n_new_rules_needed}**",
        "",
        "## Answers to required audit questions",
        "",
        "### 1. What calibration datasets are actually present?",
        "",
        "- **Pure Raman identity**: ramanbiolib (202), gobbato 3-rep (153 currently; 418 available), aa.xlsx (20), digitised_literature (2)",
        "- **Pure SERS identity**: sers_metabolite_63 NIHMS1547448 (63, citrate-Ag); serum_ag_colloids::SERS metabolites (81, Ag colloid 100 µM)",
        "- **Raman pure metabolites (expanded)**: serum_ag_colloids::Raman metabolites (418 across 51 analytes)",
        "- **Dose-response titration (matched substrate)**: ergothioneine_serum ERG_calibration (55 = 11 conc × 5 rep, cAg 785 nm)",
        "- **Dose-response (binary spike)**: CSPP Figure-7 Hyp (50) + Erg (50) + Bkg (50) = 150",
        "- **Enzymatic depletion**: serum_ag_colloids::dataset uricase (20 = 4 cohorts × 5)",
        "- **Matrix effect**: serum_ag_colloids::SERS metabolites for fitting (UAfree 10 + UAbound 10 + Hypox 10 = 30)",
        "- **Isotopic shift**: serum_ag_colloids::isotopic (73: UA/UAiso ± HSA ± filter)",
        "- **Substrate mismatch (diagnostic)**: adenine_sers_control bAgNPs LOD (7) + 1ng × 5 reps (5)",
        "- **Additional CSPP context**: Figures 2 (70), 4 (125), 5 (120), 6 (63) = 378 requiring parsing",
        "",
        "### 2. Which analytes/processes can we calibrate now?",
        "",
        "- **UA (uric acid)**: identity (Raman + SERS) + enzymatic depletion (uricase) + matrix effect (UAfree vs UAbound) + isotopic (UA vs UAiso) — STRONGEST anchor set",
        "- **Hypoxanthine**: identity (Raman + SERS) + spike-in (CSPP fig7) + replicate (fitting 10-rep) — STRONG",
        "- **Ergothioneine**: identity (Raman + SERS) + titration (ERG_calibration) + spike-in (CSPP fig7) — MEDIUM (µM effects weak)",
        "- **Adenine**: identity (Raman + SERS); dose-response only via SUBSTRATE-MISMATCH diagnostic",
        "- **Phe/Tyr/Trp + other amino acids**: identity (ramanbiolib + aa.xlsx + SERS metabolites) — STRONG identity, no titration",
        "- **Glucose / Lactate / Cholesterol**: identity only (Raman metabolites + SERS metabolites) — no titration",
        "- **DNA / RNA (new in SERS metabolites)**: identity check only — these 2 are NEW to the SERS bench",
        "",
        "### 3. Where must substrate-aware physics be applied?",
        "",
        "- **MANDATORY for inference + interpretation** (trained citrate-Ag family): ERG_calibration, sers_metabolite_63, serum_ag_colloids::uricase/SERS-met/fitting/isotopic",
        "- **CONDITIONAL for inference; MANDATORY for interpretation** (CSPP paper Ag, different substrate family): CSPP Figures 2/4/5/6/7 — currently without CSPP-specific rules, run with substrate caveat",
        "- **OFF** (pure Raman): ramanbiolib, gobbato powder, aa.xlsx, digitised_literature, Raman metabolites",
        "- **REQUIRED BEFORE VALID INFERENCE** (out-of-scope bAgNPs): adenine_sers_control — currently DIAGNOSTIC ONLY until bAgNPs rule block is implemented",
        "",
        "### 4. What is ready for the next calibration phase?",
        "",
        "- All ADMISSIBLE_READY datasets + the ADMISSIBLE_BUT_LIMITED subset that functions as diagnostic.",
        "- 3 datasets require small loader work (wide-to-long reshape for ERG_calibration; zip-aware loaders for serum_ag_colloids sub-folders).",
        "- 5 datasets (CSPP Figures 2/4/5/6) need metadata parsing before use.",
        "",
        "### 5. What remains missing but not blocking?",
        "",
        "- Enzymatic time-course / kinetics — binary depletion only",
        "- Non-purine dose-response titrations",
        "- Real multi-analyte experimental mixtures",
        "- G02 / G06 / G10 dedicated calibration anchors",
        "- bAgNPs dedicated substrate-aware rule block",
        "- Additional non-citrate-Ag substrate scope (solid-Ag chip / Ag-film / Au / core-shell)",
        "",
        "## Next-phase implementation notes",
        "",
        "1. **Build loaders** for:",
        "   - `ERG_calibration.csv` wide-to-long reshape",
        "   - `serum_ag_colloids/dataset_spectral_data.zip::Raman metabolites/` (expanded beyond current 3-rep)",
        "   - `serum_ag_colloids/dataset_spectral_data.zip::SERS metabolites/`",
        "   - `serum_ag_colloids/dataset_spectral_data.zip::SERS metabolites for fitting/`",
        "   - `serum_ag_colloids/dataset_spectral_data.zip::dataset uricase/`",
        "   - `serum_ag_colloids/dataset_spectral_data.zip::isotopic/`",
        "2. **Run engine** with correct substrate-physics switch per dataset "
        "(Stage 4 substrate physics audit table).",
        "3. **Compute** BSV, ΔBSV (with dataset-appropriate reference), top motif family + top-3, top MSS hits, confidence, ambiguity — for every dataset.",
        "4. **Compare observed shifts to expected** chemistry per Stage 3 expected-signal registry.",
        "5. **DO NOT** use bAgNPs adenine as positive anchor.",
        "6. **DO NOT** fit on clinical cohorts (serum Merck / donors serum / nature_serum_sers / etc.).",
    ]
    (REPORTS / "REPORT_calibration_audit_readiness_v2.md"
     ).write_text("\n".join(lines))
    print(f"  [decision] {decision}")
    return decision


def write_audit(discovery, admiss, expected, substrate, checklist, decision):
    lines = [
        "# gaira_base_4_hybrid_bsv_calibration_audit_v2 — Audit Log",
        "",
        "## Purpose",
        "",
        "Pre-calibration audit only. No calibration tests run. No engine / taxonomy / "
        "motif / MSS change.",
        "",
        "## Datasets discovered",
        "",
        f"- Total candidates enumerated: {len(discovery)}",
        f"- Admissible for calibration: {sum(1 for r in admiss if r['admissibility_label'].startswith('ADMISSIBLE'))}",
        f"- Not admissible (clinical cohort / reference-only): {sum(1 for r in admiss if r['admissibility_label'] == 'NOT_ADMISSIBLE')}",
        "",
        "## Admissibility counts",
        "",
    ]
    from collections import Counter
    counts = Counter(r["admissibility_label"] for r in admiss)
    for k, v in counts.items():
        lines.append(f"- {k}: {v}")
    lines += [
        "",
        "## Substrate-aware physics applicability decisions (summary)",
        "",
        f"- Datasets requiring SERS physics ON (both inference + interpretation): "
        f"{sum(1 for r in substrate if r['apply_substrate_physics_for_inference'] == 'YES' and r['apply_substrate_physics_for_interpretation'] == 'YES')}",
        f"- Datasets with SERS physics OFF (pure Raman): "
        f"{sum(1 for r in substrate if r['apply_substrate_physics_for_inference'] == 'NO')}",
        f"- Datasets CONDITIONAL (CSPP paper Ag / bAgNPs): "
        f"{sum(1 for r in substrate if r['apply_substrate_physics_for_inference'].upper().startswith('CONDITIONAL') or r['apply_substrate_physics_for_inference'] == 'DIAGNOSTIC_ONLY')}",
        f"- Datasets requiring NEW rule support before valid inference: "
        f"{sum(1 for r in substrate if r['new_rule_support_needed'].upper().startswith('YES'))}",
        "",
        "## Next-phase readiness checklist entries",
        "",
        f"- Total checklist entries: {len(checklist)}",
        f"- YES (ready to run): {sum(1 for r in checklist if r['admissible_for_next_phase'] == 'YES')}",
        f"- YES_BUT_LIMITED: {sum(1 for r in checklist if r['admissible_for_next_phase'] == 'YES_BUT_LIMITED')}",
        f"- NOT_YET (needs parsing): {sum(1 for r in checklist if r['admissible_for_next_phase'].startswith('NOT_YET'))}",
        f"- DIAGNOSTIC_ONLY: {sum(1 for r in checklist if 'DIAGNOSTIC' in r['admissible_for_next_phase'])}",
        "",
        "## Final decision",
        "",
        f"**{decision}**",
        "",
        "## Invariants",
        "",
        "- `src/gaira/base3/mss_engine.py`: unchanged",
        "- All prior phase drivers: unchanged",
        "- v4.5 engine: unchanged",
        "- Taxonomy / motif / MSS / substrate physics: read-only",
        "- No calibration tests executed in this audit phase",
        "- No target clinical cohorts used",
    ]
    (AUDIT / "gaira_base_4_hybrid_bsv_calibration_audit_v2_audit_log.md"
     ).write_text("\n".join(lines))


# ═════════════════════════════════════════════════════════════════════
# Driver
# ═════════════════════════════════════════════════════════════════════

def main():
    print("=" * 78)
    print("gaira_base_4_hybrid_bsv_calibration_audit_v2 (AUDIT ONLY)")
    print("=" * 78)
    for d in (TABLES, REPORTS, AUDIT, CODE_SNAPSHOT):
        d.mkdir(parents=True, exist_ok=True)

    discovery = stage1_discovery()
    admiss = stage2_admissibility()
    expected = stage3_expected_signal()
    substrate = stage4_substrate_physics()
    checklist = stage5_checklist()
    stage6_gap_analysis()
    decision = stage7_readiness(admiss, substrate, checklist)
    write_audit(discovery, admiss, expected, substrate, checklist, decision)

    p = Path(__file__)
    if p.exists():
        shutil.copy(p, CODE_SNAPSHOT / p.name)

    print(f"\n[complete] decision: {decision}")


if __name__ == "__main__":
    main()
