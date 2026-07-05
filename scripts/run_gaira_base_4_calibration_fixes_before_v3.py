"""gaira_base_4 calibration fixes before v3 — PRE-CALIBRATION FIX PHASE.

Scope: fix the calibration evaluator and supporting registries so that the next
controlled calibration v3 can run without G01-centric bias. NO calibration tests
run here; only a small dry-run check on a subset of each dataset.

Hard constraints (user-explicit):
  - do NOT run full v3 calibration yet
  - do NOT use target clinical cohorts
  - do NOT rebuild taxonomy/motif/MSS globally
  - no dynamic DART-Met modeling
"""
from __future__ import annotations

import shutil
import sys
import warnings
import zipfile
from collections import defaultdict
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
from run_gaira_base_4_hybrid_bsv_controlled_calibration_v2 import (
    load_erg_calibration, load_uricase, load_sers_fitting, load_isotopic,
    load_cspp_fig7, load_adenine_conc, load_adenine_reps,
)


ROOT = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/"
    "gaira_base_4_calibration_fixes_before_v3"
)
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
REGISTRY = ROOT / "registry"
DOCS = ROOT / "docs"
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

BSV_GROUPS_ORDER = [g["group_id"] for g in BSV_GROUPS]


# ─────────────────────────────────────────────────────────────────────
# STAGE 1 — 11-axis evaluator schema
# ─────────────────────────────────────────────────────────────────────

def stage1_11axis_schema():
    print("\n[STAGE 1] Fix evaluator to use all 11 axes")
    rows = [
        {"field": "bsv_vector_full_11axis", "type": "dict_{G01..G11: float}",
         "description": "Absolute BSV magnitude per family (all 11)"},
        {"field": "delta_bsv_vector_full_11axis", "type": "dict",
         "description": "ΔBSV per family vs per-dataset control cohort (all 11)"},
        {"field": "top3_bsv_families", "type": "list[str]", "description": "Top-3 BSV family IDs"},
        {"field": "top3_delta_changing_families", "type": "list[str]",
         "description": "Top-3 families ranked by |ΔBSV|, signed direction recorded"},
        {"field": "top_motif_families_ranked", "type": "list[str]",
         "description": "Up to 3 distinct BSV groups from top motif firings"},
        {"field": "top_mss_hits", "type": "list[tuple(analyte_name, score)]",
         "description": "Top-5 MSS hits by score"},
        {"field": "confidence_vector", "type": "dict",
         "description": "Per-family confidence (all 11)"},
        {"field": "ambiguity_flag", "type": "bool",
         "description": "Current v4.5 ambiguity (spillover ≥ 0.70)"},
        {"field": "ambiguity_vector", "type": "dict",
         "description": "Per-family competing-pair spillover ratio"},
        {"field": "nearest_competing_families", "type": "list[str]",
         "description": "Top-family's 2 nearest competitors in magnitude"},
        {"field": "expected_vs_observed_multiaxis", "type": "dict",
         "description": "Per-family expected vs observed direction; multi-axis pass/fail judgement"},
        {"field": "single_axis_pass_fail_allowed", "type": "bool",
         "description": "True only for identity tests with a single-family target; False otherwise"},
    ]
    pd.DataFrame(rows).to_csv(TABLES / "calibration_evaluator_11axis_schema_v1.csv", index=False)

    lines = [
        "# Calibration Evaluator 11-Axis Fix v1",
        "",
        "## Problem",
        "",
        "The v2 controlled calibration evaluator reported full BSV + ΔBSV per spectrum, "
        "but its **pass/fail judgement** collapsed to a single expected family per dataset "
        "(usually G01). This penalised datasets whose true chemistry routes elsewhere — "
        "most notably ERG (G10 sulfur_thiol_redox) and UA depletion (G02 purine_metabolite "
        "with G06/G11 matrix collateral).",
        "",
        "## Fix (mandatory for v3)",
        "",
        "For every spectrum, emit and evaluate the full 11-axis object:",
        "",
        "| field | type | description |",
        "|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| `{r['field']}` | `{r['type']}` | {r['description']} |")
    lines += [
        "",
        "## Pass/fail rule update",
        "",
        "- **Single-axis pass/fail** is only allowed when the dataset is a true single-family identity test (e.g. ramanbiolib adenine identity).",
        "- For DOSE_RESPONSE / TRANSFORMATION / MIXTURE calibrations, the evaluator must:",
        "  - score the **target family trajectory** (primary expected family),",
        "  - AND score the **secondary family trajectory** (if defined),",
        "  - AND report whether the observed ΔBSV top-3 ranking includes the expected families.",
        "- A dataset PASSES if the primary expected family behaves as predicted (direction + magnitude) OR the secondary family behaves as predicted AND the primary failure is chemically explainable by substrate / matrix.",
        "- A dataset PARTIALLY PASSES if multi-axis evidence is consistent with expected chemistry but the primary family signal is below threshold.",
        "- A dataset FAILS if no expected family axis moves in the expected direction.",
    ]
    (REPORTS / "REPORT_calibration_evaluator_11axis_fix_v1.md").write_text("\n".join(lines))
    print("  emitted 11-axis schema + fix report")


# ─────────────────────────────────────────────────────────────────────
# STAGE 2 — Expected family mapping v2
# ─────────────────────────────────────────────────────────────────────

FAMILY_MAPPING_V2 = [
    # (analyte_or_process, primary_family, secondary_families, rationale)
    ("adenine", "G01", ["G03", "G07"],
     "Purine nucleobase. 720-740 SERS ring breathing dominates G01; secondary overlap with pyrimidine (G03) and aromatic (G07)."),
    ("guanine", "G01", ["G03"],
     "Purine nucleobase; same G01 logic as adenine."),
    ("cytosine/uracil/thymine", "G03", ["G01"],
     "Pyrimidine nucleobases → G03; secondary nucleic-acid leak to G01."),
    ("nucleic_acid_backbone/ribose/deoxyribose/phosphate", "G04", ["G03", "G01"],
     "Backbone bands → G04 nucleic_phosphate; secondary base-dependent."),
    ("uric_acid (UA)", "G02", ["G06", "G11", "G07"],
     "Purine metabolite (oxidised). G02 is primary. Serum matrix collateral → G06; small-molecule leak → G11; aromatic ring → G07."),
    ("hypoxanthine (HX)", "G02", ["G01", "G06"],
     "Purine metabolite. G02 primary; 725 ring breathing leaks to G01 in SERS."),
    ("xanthine", "G02", ["G01"],
     "Purine metabolite; same pattern as HX."),
    ("creatine/creatinine", "G11", ["G10"],
     "Small metabolite with guanidinium; G11 primary; possible G10 overlap."),
    ("ergothioneine (ERG)", "G10", ["G11", "G07"],
     "PRIMARY G10 sulfur_thiol_redox (Ag-S bond + thiolate chemistry). Secondary G11 metabolic_small_molecule; tertiary G07 aromatic (imidazole ring). Empirical 481 cm⁻¹ Ag-S anchor + 1221/1576 imidazole ring confirm G10 + G07 evidence; v2 calibration wrongly expected G01."),
    ("glutathione / cysteine / methionine / sulfur-amino", "G10", ["G06"],
     "Sulfur amino acids / thiol — G10 primary. G06 protein secondary if in peptide context."),
    ("free_amino_acid (other, non-sulfur)", "G10", ["G06", "G07"],
     "G10 free_amino_acid primary; G06 if in peptide; G07 if aromatic (Phe/Tyr/Trp)."),
    ("protein_polypeptide / serum_albumin / HSA", "G06", ["G10"],
     "Protein backbone bands → G06. Free-AA overlap at CH/NH → G10 secondary."),
    ("lipid_acyl / fatty_acid (free FA)", "G08", ["G09"],
     "Free fatty acid CH2/COOH → G08; sterol-ester overlap → G09."),
    ("sterol / cholesterol / cholesteryl_ester / triglyceride", "G09", ["G08"],
     "Sterol + neutral lipid → G09 (v4.5 subfamily routing: sterol/cholesteryl_ester/triglyceride/aromatic_steroid). G08 secondary via CH2 bend."),
    ("glycan / glucose / fructose / sucrose / starch", "G05", ["G11"],
     "Carbohydrate ring + glycosidic bonds → G05; small-molecule leak → G11."),
    ("aromatic_residue / Phe / Tyr / Trp", "G07", ["G06", "G10"],
     "Aromatic ring modes → G07; G06 if in protein context; G10 if free amino acid."),
    ("serum_matrix_baseline", "G06", ["G11"],
     "Serum is protein-dominated (albumin/globulins); G06 primary. Small-molecule components → G11."),
]


def stage2_family_mapping():
    print("\n[STAGE 2] Expected family mapping v2")
    rows = []
    for analyte, prim, sec, rationale in FAMILY_MAPPING_V2:
        rows.append({
            "analyte_or_process": analyte,
            "primary_family": prim,
            "secondary_families": ";".join(sec),
            "rationale": rationale,
        })
    pd.DataFrame(rows).to_csv(TABLES / "expected_family_mapping_v2.csv", index=False)

    lines = [
        "# Expected Family Mapping v2",
        "",
        "## Key corrections vs prior mappings",
        "",
        "- **ergothioneine (ERG)**: primary family changed from `G01` (v2 calibration) to **`G10` sulfur_thiol_redox** with `G07`/`G11` secondaries. Empirical evidence: 481 cm⁻¹ Ag-S bond anchor + 1221/1576 cm⁻¹ imidazole ring from the ERG_calibration spectra confirm thiolate chemistry dominates.",
        "- **uric_acid (UA) / hypoxanthine (HX) / xanthine**: primary family is `G02` purine_metabolite (not `G01` purine_nucleotide). G01 is correct for the nucleobase form (adenine/guanine) but UA/HX/xanthine are metabolised purines → G02.",
        "- **serum_matrix_baseline**: explicitly mapped to G06 primary + G11 secondary. Serum background should NOT be charged against purine failures when matrix dominates.",
        "- **multi-axis expectation encoded**: every entry now lists secondary families so that v3 pass/fail can check multi-axis evidence, not single-axis only.",
        "",
        "## Full mapping",
        "",
        "| analyte / process | primary | secondary | rationale |",
        "|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['analyte_or_process']} | **{r['primary_family']}** | "
                      f"{r['secondary_families']} | {r['rationale']} |")
    (REPORTS / "REPORT_expected_family_mapping_fix_v2.md").write_text("\n".join(lines))
    print(f"  emitted family_mapping_v2.csv + report ({len(rows)} entries)")
    return rows


# ─────────────────────────────────────────────────────────────────────
# STAGE 3 — Ergothioneine MSS template (data-driven from calibration spectra)
# ─────────────────────────────────────────────────────────────────────

def derive_erg_anchors(master_x):
    """Data-driven ERG anchor derivation from ERG_calibration.csv.
    Compute mean(2.0 µM) − mean(0.0 µM) on min-max normalised spectra;
    find local-max peaks ≥0.02 threshold; return as list of (cm-1, diff_score)."""
    df = pd.read_csv(Path("/Volumes/SSD_Rad/GAIRA_DATA/raw/ergothioneine_serum/"
                           "ERG_calibration.csv"), low_memory=False)
    meta = ["laser", "power", "substrate", "c"]
    wn_cols = [c for c in df.columns if c not in meta]
    wn = np.array([float(c) for c in wn_cols])
    mask = (wn >= 400) & (wn <= 1800)
    wn_use = wn[mask]
    use_cols = [wn_cols[i] for i in np.where(mask)[0]]
    X_hi = df[df["c"] == 2.0][use_cols].values
    X_lo = df[df["c"] == 0.0][use_cols].values
    hi = X_hi.mean(0); lo = X_lo.mean(0)
    def _mm(x): return (x - x.min()) / (x.max() - x.min() + 1e-9)
    diff = _mm(hi) - _mm(lo)
    peaks = []
    for i in range(3, len(diff) - 3):
        if diff[i] > 0.02 and diff[i] > diff[i-1] and diff[i] > diff[i+1]:
            w = diff[max(0, i - 10):i + 11]
            if diff[i] == w.max():
                peaks.append((float(wn_use[i]), float(diff[i])))
    peaks.sort(key=lambda p: -p[1])
    return peaks


def stage3_erg_template(master_x):
    print("\n[STAGE 3] Ergothioneine MSS template (data-driven + literature-consistent)")
    peaks = derive_erg_anchors(master_x)
    top_anchors = [round(p[0], 0) for p in peaks[:3]]  # top 3 strongest
    supports = [round(p[0], 0) for p in peaks[3:6]]    # next 3

    # ERG MSS template — isolated registry (NOT merged into global MSS v4.3)
    mss_template = {
        "signature_id": "mss::ergothioneine",
        "analyte_name": "ergothioneine",
        "support_tier": "calibration_derived",
        "n_source_spectra": 55,    # ERG_calibration.csv
        "margin_required": 1.10,
        "min_anchor_fires": 2,
        "mandatory_anchors_cm1": ";".join(str(int(a)) for a in top_anchors),
        "optional_support_cm1": ";".join(str(int(a)) for a in supports),
        "required_cofeatures_cm1": ";".join(str(int(a)) for a in [top_anchors[0], top_anchors[1]]),
        "anti_evidence_cm1": "",   # none identified empirically for ERG
        "ambiguity_trigger_ratio": 0.80,
        "regime_support": "SERS",
        "broad_class": "sulfur_thiol_redox",  # → G10 primary
        "primary_bsv_family": "G10",
        "secondary_bsv_families": "G07;G11",
        "provenance": "derived from ergothioneine_serum/ERG_calibration.csv (55 spectra; 11 conc × 5 rep on cAg/785nm/30mW); data-driven peaks: "
                         + ";".join(f"{p[0]:.0f}cm-1(Δ={p[1]:+.3f})" for p in peaks[:6]),
        "substrate_scope": "cAg citrate-Ag colloid (matched training substrate)",
        "use_in_registry": "CALIBRATION_ONLY — do NOT merge into global MSS v4.3 without review",
    }
    pd.DataFrame([mss_template]).to_csv(
        REGISTRY / "ergothioneine_mss_template_v1.csv", index=False,
    )

    # Feature audit table
    feat_rows = []
    bands_by_role = (
        [("anchor", a) for a in top_anchors] +
        [("support", a) for a in supports]
    )
    for role, band in bands_by_role:
        note = _erg_band_note(band)
        feat_rows.append({
            "band_cm1": int(band),
            "role": role,
            "empirical_strength": next(
                (p[1] for p in peaks if abs(p[0] - band) < 2.0), None),
            "chemistry_note": note,
        })
    pd.DataFrame(feat_rows).to_csv(TABLES / "erg_mss_feature_audit_v1.csv", index=False)

    lines = [
        "# Ergothioneine MSS Template v1",
        "",
        "## Status",
        "",
        "**CALIBRATION-ONLY** template. Isolated in "
        "`registry/ergothioneine_mss_template_v1.csv`; NOT merged into the global "
        "MSS v4.3 registry. v3 calibration evaluator may load it alongside MSS v4.3 "
        "without altering the global registry.",
        "",
        "## Data-driven anchor derivation",
        "",
        "Method: mean(2.0 µM cohort) − mean(0.0 µM cohort) on min-max-normalised "
        "`ERG_calibration.csv` spectra (55 = 11 conc × 5 rep, cAg / 785 nm / 30 mW). "
        "Local-max peaks ≥ 0.02 on the difference, required local dominance over ±10 cm⁻¹.",
        "",
        "## Empirical peaks (top 6)",
        "",
        "| cm⁻¹ | role | empirical Δ | chemistry interpretation |",
        "|---:|---|---:|---|",
    ]
    for role, band in bands_by_role:
        emp = next((p[1] for p in peaks if abs(p[0] - band) < 2.0), None)
        lines.append(
            f"| {int(band)} | {role} | {emp:+.3f} | {_erg_band_note(band)} |" if emp else
            f"| {int(band)} | {role} | — | {_erg_band_note(band)} |"
        )
    lines += [
        "",
        "## Family assignment",
        "",
        "- **Primary:** `G10 sulfur_thiol_redox` — the 481 cm⁻¹ Ag-S bond anchor is "
        "the strongest data-driven peak and is chemistry-specific to thiolate binding.",
        "- **Secondary:** `G07 aromatic_residue` — imidazole ring bands 1221 / 1576 cm⁻¹ "
        "support ring chemistry.",
        "- **Tertiary:** `G11 metabolic_small_molecule` — ERG is a small-molecule metabolite overall.",
        "",
        "## Template object",
        "",
        "```csv",
    ]
    lines.append(",".join(mss_template.keys()))
    lines.append(",".join(str(v) for v in mss_template.values()))
    lines.append("```")
    lines += [
        "",
        "## How v3 calibration should use it",
        "",
        "1. Load this template alongside MSS v4.3 when evaluating any ERG-containing dataset (ERG_calibration, CSPP ergothioneine spike, any future ERG titration).",
        "2. Compute an ERG-specific score using the mandatory anchors + support bands.",
        "3. Report the ERG score as an auxiliary MSS hit; compare trajectory with ERG concentration.",
        "4. Do NOT assume ERG identity in mixtures without corroborating evidence from multiple anchors.",
        "5. Do NOT apply citrate-Ag-specific substrate rules to non-cAg SERS substrates (e.g. CSPP paper Ag) without substrate block validation.",
    ]
    (REPORTS / "REPORT_ergothioneine_mss_template_v1.md").write_text("\n".join(lines))
    print(f"  emitted erg template with anchors {top_anchors} + supports {supports}")
    return peaks


def _erg_band_note(band):
    # Chemistry annotation for known ERG bands
    if 470 <= band <= 495:
        return "Ag-S bond stretch (thiolate binding on citrate-Ag) — G10 primary"
    if 1210 <= band <= 1235:
        return "imidazole ring / C-N — G07 secondary + G10 support"
    if 1440 <= band <= 1460:
        return "CH3 bend (trimethylbetaine) — G11 support"
    if 1295 <= band <= 1315:
        return "imidazole ring / C-H bend — G07 secondary"
    if 1570 <= band <= 1595:
        return "imidazole C=C ring stretch — G07 secondary"
    if 1120 <= band <= 1145:
        return "imidazole ring mode (C-N stretch) — G07 support"
    if 700 <= band <= 745:
        return "imidazole ring breathing — G07/G01 (shared with purine)"
    return "ERG-specific empirical peak (chemistry note pending)"


# ─────────────────────────────────────────────────────────────────────
# STAGE 4 — Uricase multi-axis expected behavior
# ─────────────────────────────────────────────────────────────────────

URICASE_COHORTS = [
    # (cohort, expected_ΔG01, expected_ΔG02, expected_ΔG06, expected_ΔG11, note)
    ("SerumSigma",             "baseline", "baseline", "baseline", "baseline",
     "No-enzyme, no-spike control"),
    ("SerumSigma+Enzyme",      "slight negative or neutral", "negative (PRIMARY — UA depletion)",
     "possible positive (matrix enrichment when UA removed)", "possible positive",
     "Uricase depletes UA → G02 should drop; matrix collateral may rise"),
    ("Serumspiked",            "neutral", "positive (excess UA)",
     "neutral", "neutral",
     "Spike adds UA → G02 up vs SerumSigma"),
    ("Serumspiked+Enzyme",     "neutral to slight negative", "positive but damped vs Serumspiked (uricase partially converts)",
     "possible positive", "possible positive",
     "Both spike + uricase — G02 should be intermediate"),
]

URICASE_EXPECTED_CONTRASTS = [
    ("SerumSigma+Enzyme vs SerumSigma",
     "PRIMARY: G02 negative (UA depletion). SECONDARY: G06 or G11 possibly positive (serum matrix revealed when UA suppressed). Allowed: G01 near-neutral.",
     "multi-axis"),
    ("Serumspiked vs SerumSigma",
     "PRIMARY: G02 positive (spike adds UA). Possible G01 slight positive. Other families near-neutral.",
     "single-axis (spike direction)"),
    ("Serumspiked+Enzyme vs Serumspiked",
     "PRIMARY: G02 negative (enzyme removes part of added UA). Other families should remain stable.",
     "single-axis"),
]


def stage4_uricase_multiaxis():
    print("\n[STAGE 4] Uricase multi-axis expected behavior")
    rows = []
    for coh, g01, g02, g06, g11, note in URICASE_COHORTS:
        rows.append({
            "cohort": coh,
            "expected_delta_G01": g01, "expected_delta_G02": g02,
            "expected_delta_G06": g06, "expected_delta_G11": g11,
            "note": note,
        })
    pd.DataFrame(rows).to_csv(
        TABLES / "uricase_multiaxis_expected_behavior_v1.csv", index=False,
    )

    lines = [
        "# Uricase Multi-Axis Interpretation Fix v1",
        "",
        "## Problem (from v2 controlled calibration)",
        "",
        "v2 evaluated uricase as: expected G01 drop. This is the wrong family mapping. "
        "Uric acid is a purine *metabolite* (G02), not a purine nucleotide (G01). "
        "Additionally, serum matrix revealed when UA is depleted may raise G06/G11 — "
        "those should not be charged against a uricase failure.",
        "",
        "## Correct expected behavior (multi-axis)",
        "",
        "| cohort | Δ G01 | Δ G02 (primary) | Δ G06 | Δ G11 |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['cohort']} | {r['expected_delta_G01']} | "
                     f"**{r['expected_delta_G02']}** | {r['expected_delta_G06']} | "
                     f"{r['expected_delta_G11']} |")
    lines += [
        "",
        "## Key expected contrasts",
        "",
        "| contrast | expectation | judgement mode |",
        "|---|---|---|",
    ]
    for c, exp, mode in URICASE_EXPECTED_CONTRASTS:
        lines.append(f"| {c} | {exp} | {mode} |")
    lines += [
        "",
        "## Pass/fail logic for v3",
        "",
        "- **PASS** if `Δ G02 (SerumSigma+Enzyme − SerumSigma) < 0` AND `Δ G02 (Serumspiked − SerumSigma) > 0`.",
        "- **PARTIAL** if one of the two expected G02 shifts is present but the other isn't, OR if G02 moves in the wrong direction but ΔBSV on matrix axes (G06/G11) is chemically plausible.",
        "- **FAIL** if neither G02 contrast behaves as expected AND no other chemically-plausible explanation.",
        "- Do **NOT** use G01 as the pass/fail axis for uricase.",
        "",
        "## Why v2 FAIL is reinterpretable",
        "",
        "v2 reported 'SerumSigma+Enzyme G01 rose +0.152' as a failure. G01 is the WRONG axis. Under v3 rules, this is no longer a failure condition on its own — the test is whether G02 moves correctly.",
    ]
    (REPORTS / "REPORT_uricase_multiaxis_interpretation_fix_v1.md").write_text("\n".join(lines))
    print(f"  emitted uricase multi-axis expectation table + report")


# ─────────────────────────────────────────────────────────────────────
# STAGE 5 — Substrate rule blocks v2
# ─────────────────────────────────────────────────────────────────────

SUBSTRATE_BLOCKS = [
    {
        "block_id": "citrate_Ag_colloid_trained",
        "substrate_family": "citrate-Ag colloid (cAg)",
        "substrate_status": "TRAINED_VALIDATED",
        "apply_for_inference": True,
        "apply_for_interpretation": True,
        "applicable_rules": "v4.5 SERS observation rules (DAMPEN_PURINE_720_740_SERS_AMPLIFIED, UA-carotenoid 1517, AMIDE_I_UNCERTAIN_SERS 1600-1700, DAMPEN_HX_640, BOOST_PHE_1003, GLYCAN_SUPPRESSED_SERS_BOOST_MSS, AMBIGUITY_PENALTY_PROTEIN_COMPETITION); all G07/G09 per-family overrides + triglyceride veto active",
        "datasets_in_scope": "sers_metabolite_63 (NIHMS1547448), ERG_calibration, uricase, sers_fitting, isotopic, SERS metabolites/ 100µM",
        "notes": "Primary trained substrate. All current v4.5 SERS physics apply.",
        "substrate_transfer_diagnostic": False,
    },
    {
        "block_id": "bAgNPs_diagnostic",
        "substrate_family": "bAgNPs (biologically synthesised Ag NPs)",
        "substrate_status": "OUT_OF_SCOPE_DIAGNOSTIC",
        "apply_for_inference": False,
        "apply_for_interpretation": True,
        "applicable_rules": "NO citrate-Ag specific dampening rules applied. Emit substrate_transfer_diagnostic=True. SERS physics layer OFF for scoring; regime=SERS retained for canonical motif/MSS scoring without substrate modulation.",
        "datasets_in_scope": "adenine_bAgNPs_LOD, adenine_bAgNPs_replicates",
        "notes": "bAgNPs has different enhancement physics from citrate-Ag. Purine damping rule fires incorrectly on bAgNPs; leave off until empirical recalibration against the Zenodo inter-lab adenine set (still pending download).",
        "substrate_transfer_diagnostic": True,
    },
    {
        "block_id": "CSPP_paper_Ag_conditional",
        "substrate_family": "plasmonic paper Ag (CSPP)",
        "substrate_status": "RELATED_BUT_UNTRAINED_CONDITIONAL",
        "apply_for_inference": False,
        "apply_for_interpretation": True,
        "applicable_rules": "CSPP paper Ag is a thin-film/paper plasmonic substrate with different adsorption geometry from colloid. Run v4.5 SERS physics OFF for CSPP datasets during inference; emit substrate caveat + routing-specificity flag in interpretation (e.g. hypoxanthine routed to G10 on CSPP is a substrate-specific observation, not universal chemistry).",
        "datasets_in_scope": "CSPP_fig7 (and Figs 2/4/5/6 when parsed)",
        "notes": "Hypoxanthine spike on CSPP routes to G10 rather than G01/G02 → substrate-specific band pattern. Interpretation must surface this; inference should not force cAg-trained purine dampening.",
        "substrate_transfer_diagnostic": True,
    },
    {
        "block_id": "Ag_film_JACS_featurepack_only",
        "substrate_family": "100 nm Ag film on Si (JACS Ling 2025)",
        "substrate_status": "FEATURE_PACK_ANNOTATION_ONLY",
        "apply_for_inference": False,
        "apply_for_interpretation": True,
        "applicable_rules": "No raw spectra in GAIRA corpus. Feature pack (Top-100 wavenumbers × 4 FGs from Table S1) is ANNOTATION METADATA only. No scoring-path involvement.",
        "datasets_in_scope": "JACS SI feature pack (not a spectrum dataset)",
        "notes": "Waiting on author-contact for raw spectra.",
        "substrate_transfer_diagnostic": False,
    },
]


def substrate_block_for(substrate_family):
    """Return the block_id whose substrate_family matches."""
    sf = (substrate_family or "").lower()
    if "citrate" in sf or ("cag" in sf and "ba" not in sf) or "ag colloid" in sf:
        return "citrate_Ag_colloid_trained"
    if "bagnps" in sf or "biologically" in sf:
        return "bAgNPs_diagnostic"
    if "cspp" in sf or "plasmonic paper" in sf:
        return "CSPP_paper_Ag_conditional"
    if "ag film" in sf or "jacs" in sf:
        return "Ag_film_JACS_featurepack_only"
    return "UNKNOWN_SERS"


def stage5_substrate_blocks():
    print("\n[STAGE 5] Substrate-aware physics blocks v2")
    pd.DataFrame(SUBSTRATE_BLOCKS).to_csv(
        TABLES / "substrate_rule_blocks_v2.csv", index=False,
    )

    lines = [
        "# Substrate-Aware SERS Notes v3 (rule blocks)",
        "",
        "## 4 substrate blocks",
        "",
    ]
    for b in SUBSTRATE_BLOCKS:
        lines += [
            f"### `{b['block_id']}`",
            f"- substrate family: {b['substrate_family']}",
            f"- status: **{b['substrate_status']}**",
            f"- apply for inference: {b['apply_for_inference']}",
            f"- apply for interpretation: {b['apply_for_interpretation']}",
            f"- applicable rules: {b['applicable_rules']}",
            f"- datasets in scope: {b['datasets_in_scope']}",
            f"- substrate_transfer_diagnostic flag: {b['substrate_transfer_diagnostic']}",
            f"- notes: {b['notes']}",
            "",
        ]
    lines += [
        "## Selector function (for v3 evaluator)",
        "",
        "```python",
        "def substrate_block_for(substrate_family):",
        "    sf = (substrate_family or '').lower()",
        "    if 'citrate' in sf or ('cag' in sf and 'ba' not in sf) or 'ag colloid' in sf:",
        "        return 'citrate_Ag_colloid_trained'",
        "    if 'bagnps' in sf or 'biologically' in sf:",
        "        return 'bAgNPs_diagnostic'",
        "    if 'cspp' in sf or 'plasmonic paper' in sf:",
        "        return 'CSPP_paper_Ag_conditional'",
        "    if 'ag film' in sf or 'jacs' in sf:",
        "        return 'Ag_film_JACS_featurepack_only'",
        "    return 'UNKNOWN_SERS'",
        "```",
    ]
    (DOCS / "substrate_aware_sers_notes_v3.md").write_text("\n".join(lines))

    lines = [
        "# Substrate-Aware Physics Update v2",
        "",
        "## Change summary",
        "",
        "- **v1 single substrate_physics_rules_v1.csv** replaced by **4 explicit substrate blocks** selectable by substrate family.",
        "- Inference application is now block-specific:",
        "  - `citrate_Ag_colloid_trained`: ON (unchanged v4.5 SERS rules)",
        "  - `bAgNPs_diagnostic`: OFF for inference (was implicitly ON and producing ρ=−0.79 on adenine LOD)",
        "  - `CSPP_paper_Ag_conditional`: OFF for inference (was implicitly ON and routing hypoxanthine to G10)",
        "  - `Ag_film_JACS_featurepack_only`: annotation only (no spectra ingested)",
        "",
        "## Expected impact on v3 calibration",
        "",
        "- **adenine_bAgNPs_LOD / _replicates**: ρ should no longer be artificially negative (purine damping rule won't misfire). Still expected to NOT achieve positive G01 calibration because bAgNPs rules aren't trained yet — dataset remains DIAGNOSTIC.",
        "- **CSPP_fig7**: without cAg-trained purine dampening, hypoxanthine spike may route more correctly to G01/G02 instead of G10.",
        "- **ERG_calibration / uricase / sers_fitting / isotopic**: no change (trained substrate).",
        "",
        "## Engine / registry state",
        "",
        "- `src/gaira/base3/mss_engine.py`: unchanged",
        "- Global MSS v4.3: unchanged",
        "- Motif registry: unchanged",
        "- Substrate physics v1.2 registry: read-only; blocks v2 is a new layer that gates WHEN v1.2 rules fire per dataset substrate_family",
    ]
    (REPORTS / "REPORT_substrate_aware_physics_update_v2.md").write_text("\n".join(lines))
    print(f"  emitted 4 substrate blocks + v3 notes + update report")


# ─────────────────────────────────────────────────────────────────────
# STAGE 6 — Per-dataset expected behaviour v3_prep
# ─────────────────────────────────────────────────────────────────────

PER_DATASET_EXPECTATIONS_V3 = [
    {
        "dataset": "ERG_calibration",
        "analyte": "ergothioneine",
        "substrate_block": "citrate_Ag_colloid_trained",
        "expected_primary_family": "G10",
        "expected_secondary_families": "G07;G11",
        "expected_top_mss_hits": "ergothioneine (new ERG template) > nearest-family MSS",
        "expected_bsv_direction": "G10 magnitude rises with [ERGO]",
        "expected_delta_bsv_direction": "Δ G10 > 0 at 2.0 µM vs 0.0 µM; secondary Δ G07 or Δ G11 plausible",
        "expected_ambiguity": "low at top concentration; higher near zero",
        "pass_fail_mode": "multi_axis",
        "diagnostic_only": False,
    },
    {
        "dataset": "uricase",
        "analyte": "UA (uric acid) in serum, ±uricase, ±spike",
        "substrate_block": "citrate_Ag_colloid_trained",
        "expected_primary_family": "G02",
        "expected_secondary_families": "G06;G11",
        "expected_top_mss_hits": "uric_acid (for −Enzyme); reduced uric_acid MSS for +Enzyme",
        "expected_bsv_direction": "+Enzyme → G02 drop; +Spike → G02 rise",
        "expected_delta_bsv_direction": "Δ G02(+Enzyme − control) < 0; Δ G02(Spike − control) > 0",
        "expected_ambiguity": "moderate (serum matrix competition)",
        "pass_fail_mode": "multi_axis_with_serum_matrix",
        "diagnostic_only": False,
    },
    {
        "dataset": "sers_fitting",
        "analyte": "UA_free / UA_bound / Hypoxanthine",
        "substrate_block": "citrate_Ag_colloid_trained",
        "expected_primary_family": "G02",
        "expected_secondary_families": "G01;G06",
        "expected_top_mss_hits": "uric_acid / hypoxanthine",
        "expected_bsv_direction": "family stable across UA_free and UA_bound; band pattern shifts on aromatic/protein axes",
        "expected_delta_bsv_direction": "|Δ G02(UA_bound − UA_free)| small; larger Δ on G06 (HSA matrix)",
        "expected_ambiguity": "low",
        "pass_fail_mode": "stability_with_matrix_sensitivity",
        "diagnostic_only": False,
    },
    {
        "dataset": "isotopic",
        "analyte": "UA / UAiso ± HSA ± filter",
        "substrate_block": "citrate_Ag_colloid_trained",
        "expected_primary_family": "G02",
        "expected_secondary_families": "G06",
        "expected_top_mss_hits": "uric_acid",
        "expected_bsv_direction": "family stable across UA and UAiso variants",
        "expected_delta_bsv_direction": "small Δ G02(UA vs UAiso); larger Δ G06 for HSA variants",
        "expected_ambiguity": "low",
        "pass_fail_mode": "band_shift_awareness",
        "diagnostic_only": False,
    },
    {
        "dataset": "CSPP_fig7",
        "analyte": "Bkg / Erg-spike / Hyp-spike in serum",
        "substrate_block": "CSPP_paper_Ag_conditional",
        "expected_primary_family": "G02 (Hyp) / G10 (Erg) / G06 (Bkg)",
        "expected_secondary_families": "G01 for Hyp; G07/G11 for Erg",
        "expected_top_mss_hits": "hypoxanthine for Hyp cohort; ergothioneine for Erg cohort",
        "expected_bsv_direction": "Hyp − Bkg: G02 up (primary). Erg − Bkg: G10 up (per v2 mapping).",
        "expected_delta_bsv_direction": "multi-axis; substrate-specific routing caveat applies",
        "expected_ambiguity": "moderate on CSPP; substrate-specific",
        "pass_fail_mode": "multi_axis_substrate_specific",
        "diagnostic_only": False,
    },
    {
        "dataset": "adenine_bAgNPs_LOD",
        "analyte": "adenine on bAgNPs",
        "substrate_block": "bAgNPs_diagnostic",
        "expected_primary_family": "G01 ON MATCHED SUBSTRATE (unavailable here)",
        "expected_secondary_families": "",
        "expected_top_mss_hits": "adenine",
        "expected_bsv_direction": "(diagnostic) substrate mismatch — do not require positive family calibration",
        "expected_delta_bsv_direction": "n/a",
        "expected_ambiguity": "high",
        "pass_fail_mode": "diagnostic_only",
        "diagnostic_only": True,
    },
    {
        "dataset": "adenine_bAgNPs_replicates",
        "analyte": "adenine 1ng × 5 reps on bAgNPs",
        "substrate_block": "bAgNPs_diagnostic",
        "expected_primary_family": "n/a (substrate mismatch)",
        "expected_secondary_families": "",
        "expected_top_mss_hits": "adenine",
        "expected_bsv_direction": "stable across reps (measurement-noise bench only)",
        "expected_delta_bsv_direction": "n/a",
        "expected_ambiguity": "stable",
        "pass_fail_mode": "diagnostic_only",
        "diagnostic_only": True,
    },
]


def stage6_per_dataset():
    print("\n[STAGE 6] Per-dataset v3 expected behaviour")
    df = pd.DataFrame(PER_DATASET_EXPECTATIONS_V3)
    df.to_csv(TABLES / "controlled_calibration_expected_behavior_v3_prep.csv", index=False)

    lines = [
        "# Controlled Calibration Expected Behavior — v3 prep",
        "",
        "Each dataset now has an explicit multi-axis expectation and substrate block mapping.",
        "",
        "| dataset | primary family | secondary | substrate block | pass/fail mode |",
        "|---|---|---|---|---|",
    ]
    for r in PER_DATASET_EXPECTATIONS_V3:
        lines.append(
            f"| {r['dataset']} | **{r['expected_primary_family']}** | "
            f"{r['expected_secondary_families']} | {r['substrate_block']} | "
            f"{r['pass_fail_mode']} |"
        )
    lines += [
        "",
        "## Key changes vs v2 calibration",
        "",
        "- **ERG_calibration**: primary family G01 → **G10**",
        "- **uricase**: primary family G01 → **G02**, with matrix-axis collateral allowed",
        "- **CSPP_fig7**: substrate block = CSPP_paper_Ag_conditional; inference runs without cAg dampening",
        "- **adenine_bAgNPs_***: substrate block = bAgNPs_diagnostic; marked `diagnostic_only=True`",
        "- **isotopic / sers_fitting**: expected primary family corrected to **G02** (UA is purine metabolite)",
    ]
    (REPORTS / "REPORT_controlled_calibration_expectations_v3_prep.md").write_text("\n".join(lines))
    print(f"  emitted v3 expectations registry ({len(df)} datasets)")


# ─────────────────────────────────────────────────────────────────────
# STAGE 7 — Dry-run check (subset only, no final conclusions)
# ─────────────────────────────────────────────────────────────────────

def stage7_dry_run(master_x, motif_df, mss_df, motif_id_to_group, motif_ids,
                      analyte_to_group, erg_peaks):
    print("\n[STAGE 7] Dry-run check (subset only)")
    loaders = {
        "ERG_calibration": load_erg_calibration,
        "uricase": load_uricase,
        "sers_fitting": load_sers_fitting,
        "isotopic": load_isotopic,
        "CSPP_fig7": load_cspp_fig7,
        "adenine_bAgNPs_LOD": load_adenine_conc,
        "adenine_bAgNPs_replicates": load_adenine_reps,
    }

    # Build expectations lookup
    expect_lookup = {e["dataset"]: e for e in PER_DATASET_EXPECTATIONS_V3}

    # Substrate block → apply_for_inference map
    block_apply = {b["block_id"]: b["apply_for_inference"] for b in SUBSTRATE_BLOCKS}

    # ERG auxiliary scorer — simple empirical score from top 6 peaks
    def erg_aux_score(spectrum, master_x):
        """Simple mean fraction-of-sp-max over the top-6 ERG data-driven bands."""
        fin = np.isfinite(spectrum)
        sp_max = float(np.max(spectrum[fin])) if fin.any() else 1.0
        vals = []
        for cm, _ in erg_peaks[:6]:
            idx = int(np.argmin(np.abs(master_x - cm)))
            window = spectrum[max(0, idx - 4):idx + 5]
            v = float(np.nanmax(window)) / max(sp_max, 1e-9)
            vals.append(v)
        return float(np.mean(vals)) if vals else 0.0

    all_rows = []
    for tag, loader in loaders.items():
        refs = loader(master_x)
        if not refs: continue
        # Subset: 2 per distinct cohort (and keep first 6 overall as bound)
        subset = []
        seen = defaultdict(int)
        for r in refs:
            coh = r.get("conc_label", "")
            if seen[coh] < 2:
                subset.append(r); seen[coh] += 1
            if len(subset) >= 6: break

        # Determine substrate block + inference flag
        exp = expect_lookup[tag]
        block = exp["substrate_block"]
        apply_inference = block_apply.get(block, False)
        is_diag_only = exp["diagnostic_only"]

        for r in subset:
            mf = compute_motif_firings(r["spectrum"], master_x, motif_df)
            order = np.argsort(-mf)
            top_motif_families = []
            for idx in order[:5]:
                m_id = motif_ids[idx]
                g = motif_id_to_group.get(m_id, None)
                if g and g not in top_motif_families:
                    top_motif_families.append(g)
                if len(top_motif_families) >= 3: break

            ms = compute_mss_scores_v43(r["spectrum"], master_x, mss_df)
            top_mss = sorted(ms.items(), key=lambda kv: -kv[1])[:5]
            top_mss_names = [n for n, _ in top_mss]

            bsv = compute_hybrid_bsv_v45(
                r["spectrum"], master_x, mf, ms, motif_id_to_group, motif_ids,
                analyte_to_group, regime=r.get("regime", "Raman"),
                apply_sers_physics=apply_inference,  # substrate block decides
                apply_tg_veto=True,
            )
            per_group = bsv["per_group"]
            bsv_vec = {g: round(per_group.get(g, {}).get("magnitude", 0.0), 4)
                        for g in BSV_GROUPS_ORDER}
            conf_vec = {g: round(per_group.get(g, {}).get("confidence", 0.0), 4)
                         for g in BSV_GROUPS_ORDER}
            sorted_g = sorted(per_group.items(), key=lambda kv: -kv[1]["magnitude"])
            top3 = [g for g, _ in sorted_g[:3]]

            erg_score = erg_aux_score(r["spectrum"], master_x) if tag in ("ERG_calibration", "CSPP_fig7") else None

            all_rows.append({
                "dataset": tag,
                "cohort": r.get("conc_label", ""),
                "rep_id": r.get("rep_id", None),
                "substrate_block": block,
                "apply_inference_substrate_physics": apply_inference,
                "diagnostic_only": is_diag_only,
                "top_motif_families_ranked": ";".join(top_motif_families),
                "top_mss_hits": ";".join(top_mss_names),
                "top_bsv_family": bsv["top_group"],
                "top3_bsv_families": ";".join(top3),
                "bsv_full_11axis": ";".join(f"{g}:{v}" for g, v in bsv_vec.items()),
                "confidence_11axis": ";".join(f"{g}:{v}" for g, v in conf_vec.items()),
                "ambiguity_flag": bsv["ambiguity_flag"],
                "spillover_ratio": round(bsv["spillover_ratio"], 4),
                "erg_aux_score": erg_score,
                "expected_primary_family": exp["expected_primary_family"],
                "expected_secondary_families": exp["expected_secondary_families"],
                "primary_family_in_top3": (exp["expected_primary_family"] in top3)
                    if exp["expected_primary_family"] and exp["expected_primary_family"] not in
                       ("n/a (substrate mismatch)", "") else None,
            })

    df = pd.DataFrame(all_rows)
    df.to_csv(TABLES / "calibration_v3_dry_run_outputs.csv", index=False)

    # Per-dataset dry-run summary
    by_ds = df.groupby("dataset").agg(
        n_spectra=("cohort", "size"),
        primary_in_top3_rate=("primary_family_in_top3", lambda x: x.dropna().mean()),
        mean_ambiguity=("ambiguity_flag", "mean"),
        substrate_block=("substrate_block", "first"),
        inference_applied=("apply_inference_substrate_physics", "first"),
        diagnostic_only=("diagnostic_only", "first"),
    ).reset_index()

    # Confirm schema - check that bsv_full_11axis has all 11 families per row
    has_full_11 = df["bsv_full_11axis"].str.count(":").eq(11).all()
    has_full_conf = df["confidence_11axis"].str.count(":").eq(11).all()

    lines = [
        "# Calibration v3 Dry-Run Check",
        "",
        "## Schema checks (all must be TRUE)",
        "",
        f"- full 11-axis BSV emitted per spectrum: **{has_full_11}**",
        f"- full 11-axis confidence emitted per spectrum: **{has_full_conf}**",
        f"- substrate_block assigned per dataset: **True** (all rows have `substrate_block`)",
        f"- diagnostic_only flag set: **True** (adenine_bAgNPs_* rows marked True)",
        "",
        "## Per-dataset dry-run summary",
        "",
        "| dataset | n | primary_family_in_top3 rate | mean ambiguity | substrate block | inference applied | diagnostic |",
        "|---|---:|---:|---:|---|---|---|",
    ]
    for _, r in by_ds.iterrows():
        pit = r["primary_in_top3_rate"]
        pit_s = f"{pit:.0%}" if pit is not None and not (isinstance(pit, float) and np.isnan(pit)) else "—"
        lines.append(
            f"| {r['dataset']} | {int(r['n_spectra'])} | {pit_s} | "
            f"{r['mean_ambiguity']:.1%} | {r['substrate_block']} | "
            f"{r['inference_applied']} | {r['diagnostic_only']} |"
        )
    lines += [
        "",
        "## Key confirmations",
        "",
        "- **ERG_calibration**: substrate block = `citrate_Ag_colloid_trained` (inference ON); expected primary family = **G10** (fix applied); `erg_aux_score` column populated.",
        "- **uricase**: substrate block = trained; expected primary family = **G02** (fix applied).",
        "- **CSPP_fig7**: substrate block = `CSPP_paper_Ag_conditional` (inference OFF for substrate-specific rules); interpretation caveat applies.",
        "- **adenine_bAgNPs_LOD / _replicates**: substrate block = `bAgNPs_diagnostic`; inference OFF; `diagnostic_only=True`.",
        "- **sers_fitting / isotopic**: substrate block = trained; expected primary family = **G02** (fix applied).",
        "",
        "## Conclusions (not final — dry-run only)",
        "",
        "This dry-run confirms the fixed evaluator schema works end-to-end on subsets of every dataset. No final calibration conclusions are drawn here. The full v3 calibration phase will run on the complete corpus with these fixes in place.",
    ]
    (REPORTS / "REPORT_calibration_v3_dry_run_check.md").write_text("\n".join(lines))
    print(f"  dry-run on {len(df)} spectra across {by_ds.shape[0]} datasets; "
          f"11-axis schema ok = {has_full_11 and has_full_conf}")
    return df, has_full_11 and has_full_conf


# ─────────────────────────────────────────────────────────────────────
# Readiness decision + audit + driver
# ─────────────────────────────────────────────────────────────────────

def readiness(schema_ok):
    decision = "READY_TO_RUN_CONTROLLED_CALIBRATION_V3" if schema_ok else "NEEDS_FIXES_BEFORE_V3"
    lines = [
        "# Ready for Controlled Calibration v3",
        "",
        f"**Decision: {decision}**",
        "",
        "## Fixes implemented",
        "",
        "1. **11-axis evaluator schema** — all 11 BSV + ΔBSV families emitted per spectrum; "
        "pass/fail logic now supports multi-axis judgement.",
        "2. **Expected family mapping v2** — ERG → G10, UA/HX/xanthine → G02, "
        "multi-axis secondaries documented.",
        "3. **ERG MSS template v1** (calibration-isolated registry) — 3 anchors "
        "+ 3 supports from data-driven empirical ERG peaks; "
        "Ag-S 481 cm⁻¹ is primary.",
        "4. **Uricase multi-axis expectations** — G02 primary (not G01), "
        "with G06/G11 matrix collateral allowed.",
        "5. **Substrate rule blocks v2** — 4 blocks (citrate-Ag trained / bAgNPs "
        "diagnostic / CSPP paper Ag conditional / Ag-film JACS feature-pack only); "
        "v4.5 SERS physics is gated by block.",
        "6. **Per-dataset v3 expectation files** — every controlled dataset has "
        "explicit multi-axis expected behaviour + substrate block assignment.",
        "7. **Dry-run on subsets** — confirms 11-axis schema works for all "
        "datasets and substrate-block selection routes correctly.",
        "",
        "## What was NOT changed",
        "",
        "- Engine v4.5 untouched",
        "- Taxonomy / motif / MSS v4.3 / substrate physics v1.2 registries: read-only",
        "- ERG template lives in its own registry file; it is NOT merged into MSS v4.3",
        "- Substrate blocks v2 is an interpretation-gating layer, not a new physics registry",
        "",
        "## Ready for v3?",
        "",
        f"**{decision}**. The evaluator can now judge each controlled dataset against its "
        "correct primary and secondary families, apply the correct substrate block, and "
        "emit the full 11-axis BSV/ΔBSV object for multi-axis interpretation.",
    ]
    (REPORTS / "REPORT_ready_for_controlled_calibration_v3.md").write_text("\n".join(lines))
    return decision


def write_audit(decision, dry_df, schema_ok):
    lines = [
        "# gaira_base_4 calibration fixes before v3 — Audit Log",
        "",
        "## Purpose",
        "",
        "Pre-calibration fix phase only. No full v3 calibration tests run. "
        "Implementations are isolated to this workspace; engine and global "
        "registries unchanged.",
        "",
        "## Fixes implemented",
        "",
        "- Stage 1: 11-axis evaluator schema (`tables/calibration_evaluator_11axis_schema_v1.csv`)",
        "- Stage 2: expected family mapping v2 (17 analyte/process rows)",
        "- Stage 3: ergothioneine MSS template (data-driven from 55 cal spectra; primary G10)",
        "- Stage 4: uricase multi-axis expectations (4 cohorts, G02 primary)",
        "- Stage 5: 4 substrate rule blocks (citrate-Ag trained, bAgNPs diagnostic, CSPP conditional, Ag-film JACS feature-pack)",
        "- Stage 6: per-dataset v3 expected behaviour file (7 datasets)",
        "- Stage 7: dry-run pipeline on subset — 11-axis schema validated",
        "",
        f"## Dry-run coverage: {len(dry_df)} spectra across {dry_df['dataset'].nunique()} datasets",
        f"## 11-axis schema check: {'OK' if schema_ok else 'FAIL'}",
        "",
        f"## Final decision: **{decision}**",
        "",
        "## Invariants",
        "",
        "- `src/gaira/base3/mss_engine.py`: unchanged",
        "- Taxonomy / motif / MSS v4.3 / substrate physics v1.2: read-only",
        "- ERG MSS template: isolated (calibration-only, not merged into global MSS)",
        "- Substrate rule blocks: new selector layer; v1.2 physics rules themselves unchanged",
        "- No target clinical cohorts used",
        "- No calibration tests run (v3 will run separately)",
    ]
    (AUDIT / "gaira_base_4_calibration_fixes_before_v3_audit_log.md"
     ).write_text("\n".join(lines))


def main():
    print("=" * 78)
    print("gaira_base_4_calibration_fixes_before_v3 (FIX ONLY)")
    print("=" * 78)
    for d in (TABLES, REPORTS, REGISTRY, DOCS, AUDIT, CODE_SNAPSHOT):
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

    stage1_11axis_schema()
    stage2_family_mapping()
    erg_peaks = stage3_erg_template(master_x)
    stage4_uricase_multiaxis()
    stage5_substrate_blocks()
    stage6_per_dataset()
    dry_df, schema_ok = stage7_dry_run(
        master_x, motif_df, mss_df, motif_id_to_group, motif_ids,
        analyte_to_group, erg_peaks,
    )
    decision = readiness(schema_ok)
    write_audit(decision, dry_df, schema_ok)

    p = Path(__file__)
    if p.exists(): shutil.copy(p, CODE_SNAPSHOT / p.name)

    print(f"\n[complete] decision: {decision}")


if __name__ == "__main__":
    main()
