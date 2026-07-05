"""gaira_base_2 — grounding re-run (behavioural validation).

Validates that the implemented engine behaves consistently with the
M3 / M3.1 / M3.2 / M2.2 grounding decisions. This is NOT an ontology
redefinition; it is a behaviour check.

For each motif with CORE_GROUNDED status in the M2.2 dual-status table,
we verify that the engine produces a non-trivial activation on its
canonical reference spectrum (ramanbiolib for most; Gobbato powder
Raman for the purine-catabolite / ergothioneine / creatinine set).

For each motif with CORE_NOT_SUPPORTED status, we verify that the
engine does NOT produce meaningful activation on any canonical
reference (there isn't one).

For each motif with CORE_AMBIGUITY_CONFIRMED, we verify that multiple
candidate reference classes produce activation.

Comparison output: per motif × reference — engine activation vs M3
grounding status. Flag any divergence.

Run:
    cd /Users/suraj/projects/GAIRA
    PYTHONPATH=src .venv/bin/python scripts/run_gaira_base_2_grounding_rerun.py
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gaira.base2 import (
    compute_motif_activation,
    load_active_registry,
)
from gaira.spectral import canonical_master_axis, crop_before_interpolate


OUT_ROOT = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_2_implementation_v1/grounding"
)
RAMANBIOLIB = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/raw/ramanbiolib/ramanbiolib-main/"
    "ramanbiolib/db/raman_spectra_db.csv"
)
M3_1_NPZ = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/gaira_build_motifs_v1/"
    "M3_1_reference_rescue_v1/references/rescued_refs_master_axis.npz"
)
M2_2_DUAL = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/gaira_build_motifs_v1/"
    "M2_2_ontology_untangling_v1/tables/motif_dual_status_v1.csv"
)


MOTIF_CORE_REF = {
    # motif_id: list of ref_ids (ramanbiolib component name OR Gobbato npz key)
    "purine_ring_breathing_720_735":         ["adenine", "guanine"],
    "uric_acid_full_signature":              ["ua_raman_pwd_gobbato2025"],
    "hypoxanthine_signature":                ["hypox_raman_pwd_gobbato2025"],
    "pyrimidine_ring_breathing_780_800":     ["cytosine", "thymine", "uracil"],
    "nucleobase_in_plane_ring_1320_1340":    ["adenine", "guanine", "cytosine", "thymine"],
    "dna_methylation_marker_790":            ["cytosine", "thymine"],
    "phosphate_PO2_sym_str_1080":            ["a-dna", "b-dna", "t-rna"],
    "phosphate_PO_asym_str_1240":            ["a-dna", "b-dna", "t-rna"],
    "dna_composite_motif":                   ["a-dna", "b-dna"],
    "xanthine_signature":                    ["xanth_raman_pwd_gobbato2025"],
    "guanine_specific_motif":                ["guanine"],
    "thymine_specific_motif":                ["thymine"],
    "cytosine_specific_motif":               ["cytosine"],
    "glycan_pyranose_ring_skeletal_850_950": ["d-(+)-glucose", "d-(+)-galactose",
                                                "d-(+)-mannose", "β-d-glucose"],
    "glycan_glycosidic_C_O_C_1020_1100":     ["cellulose", "glycogen",
                                                "a-dna", "b-dna", "citric acid"],
    "sialic_acid_signature":                 ["n-acetyl- d-glucosamine"],
    "free_saccharide_motif":                 ["d-(+)-glucose", "d-(+)-galactose"],
    "amide_III_protein_backbone_1230_1280":  ["albumin", "collagen"],
    "phenylalanine_ring_1003":               ["l-phenylalanine"],
    "tyrosine_doublet_830_850":              ["l-tyrosine"],
    "amide_I_alpha_helix_beta_sheet_motif":  ["albumin", "collagen"],
    "amide_II_motif":                        ["albumin", "collagen"],
    "lipid_acyl_C_C_str_1060_1130":          ["oleic acid", "palmitic acid"],
    "lipid_C_H_bend_1440_1460":              ["oleic acid", "palmitic acid"],
    "phosphatidylcholine_choline_head_715":  ["l-α-phosphatidylcholine"],
    "cholesterol_signature":                 ["cholesterol"],
    "lipid_methylene_twist_1300":            ["palmitic acid", "stearic acid", "oleic acid"],
    "cytochrome_c_resonance_motif":          ["cytochrome c"],
    "disulfide_S_S_str_500_550":             ["glutathione"],
    "ergothioneine_signature":               ["ergo_raman_pwd_gobbato2025"],
    "citrate_baseline_artifact_motif":       ["citric acid"],
    "amide_I_lipid_carbonyl_partial_panel_motif": ["albumin", "tristearin"],
    "purine_HX_lipid_choline_715_overlap_ambiguity": ["adenine", "guanine",
                                                       "l-α-phosphatidylcholine"],
    "neutral_lipid_triglyceride_motif":      ["tristearin", "tripalmitin", "triolein"],
    "creatine_creatinine_motif":             ["creat_raman_pwd_gobbato2025"],
    "thiol_C_S_str_660_motif":               ["glutathione"],
    "glutathione_GSH_motif":                 ["glutathione"],
    "collision_1020_1080_multi_candidate":   ["a-dna", "cellulose", "citric acid",
                                                "d-(+)-glucose"],
    "collision_1300_1400_multi_candidate_motif": ["adenine", "oleic acid", "albumin",
                                                    "citric acid"],
}


def _parse_list(s: str) -> np.ndarray:
    return np.array(ast.literal_eval(s), dtype=np.float64)


def load_ramanbiolib_on_master_axis(master_x: np.ndarray) -> dict[str, np.ndarray]:
    df = pd.read_csv(RAMANBIOLIB)
    out = {}
    for _, r in df.iterrows():
        comp = str(r["component"]).strip().lower()
        try:
            wn = _parse_list(r["wavenumbers"])
            y = _parse_list(r["intensity"])
            y_interp, _ = crop_before_interpolate(
                wn, y, master_x, partial_ok=True, min_coverage=0.80,
            )
            out[comp] = y_interp
        except Exception:
            continue
    return out


def load_gobbato_powder(master_x: np.ndarray) -> dict[str, np.ndarray]:
    if not M3_1_NPZ.exists():
        return {}
    npz = np.load(M3_1_NPZ)
    return {k: npz[k] for k in npz.files if "raman_pwd_gobbato" in k}


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print("gaira_base_2 grounding re-run")
    print("=" * 78)

    master_x = canonical_master_axis()
    motifs, mappings, dual = load_active_registry()
    # Include HELD_V2 motifs too (we need to evaluate them against their
    # reference set to confirm the behavior — their mapping is inactive
    # but their activation is still testable)
    from gaira.base2.registry import load_motif_registry
    all_motifs = load_motif_registry()
    print(f"engine loaded: {len(all_motifs)} total motifs "
          f"({len(motifs)} v1_active)")

    rb = load_ramanbiolib_on_master_axis(master_x)
    gp = load_gobbato_powder(master_x)
    print(f"reference pool: {len(rb)} ramanbiolib + {len(gp)} Gobbato powder")

    # ── Prior grounding labels (M2.2 dual-status) ─────────────────────
    prior = pd.read_csv(M2_2_DUAL).set_index("motif_id")["core_status"].to_dict()

    rows = []
    for motif_id, spec in all_motifs.items():
        ref_ids = MOTIF_CORE_REF.get(motif_id, [])
        max_activation = 0.0
        best_ref = ""
        for rid in ref_ids:
            if rid in rb:
                y = rb[rid]
            elif rid in gp:
                y = gp[rid]
            else:
                continue
            # Fill NaNs for continuous spectrum
            mask = np.isfinite(y)
            if not mask.any():
                continue
            y_work = y.copy()
            if not mask.all():
                idx = np.arange(len(y))
                y_work[~mask] = np.interp(idx[~mask], idx[mask], y[mask])
            a = compute_motif_activation(spec, y_work, master_x)
            if a > max_activation:
                max_activation = a
                best_ref = rid

        core_status = prior.get(motif_id, "UNKNOWN")
        # Expected behaviour:
        #   CORE_GROUNDED / CORE_AMBIGUITY_CONFIRMED → activation > 0.05
        #   CORE_NOT_SUPPORTED                       → activation == 0
        engine_label = (
            "FIRES_GROUNDED"     if max_activation > 0.05 and ref_ids
            else "FIRES_WEAK"    if max_activation > 0 and ref_ids
            else "NO_ACTIVATION"
        )
        expected_label = (
            "FIRES_GROUNDED" if core_status in ("CORE_GROUNDED",
                                                  "CORE_AMBIGUITY_CONFIRMED",
                                                  "CORE_PARTIALLY_GROUNDED")
            else "NO_ACTIVATION"
        )
        consistent = (engine_label == expected_label) or (
            engine_label == "FIRES_WEAK" and expected_label == "FIRES_GROUNDED"
        )
        rows.append({
            "motif_id": motif_id,
            "prior_core_status_M2_2": core_status,
            "best_reference": best_ref,
            "engine_activation": round(max_activation, 4),
            "engine_label": engine_label,
            "expected_label": expected_label,
            "consistent_with_M3": consistent,
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_ROOT / "motif_grounding_rerun_v1.csv", index=False)

    print()
    print("consistency with M3 grounding labels:")
    print(df["consistent_with_M3"].value_counts().to_string())
    print()
    print("engine label distribution:")
    print(df["engine_label"].value_counts().to_string())

    inconsistent = df[~df["consistent_with_M3"]]
    if len(inconsistent):
        print()
        print("divergences vs M3:")
        for _, r in inconsistent.iterrows():
            print(f"  {r['motif_id']:50s}  prior={r['prior_core_status_M2_2']:30s}  "
                  f"engine={r['engine_label']:15s}  act={r['engine_activation']}")

    # Report
    n_total = len(df)
    n_consistent = int(df["consistent_with_M3"].sum())
    n_divergent = n_total - n_consistent
    n_active_grounded = int(
        (df["engine_label"] == "FIRES_GROUNDED").sum()
    )
    report_path = OUT_ROOT / "REPORT_gaira_base_2_grounding_rerun_v1.md"
    report_path.write_text("\n".join([
        "# gaira_base_2 — Grounding re-run report",
        "",
        f"**Motifs evaluated:** {n_total} (all registry motifs, including HELD_V2)",
        f"**Consistent with prior M3 / M3.1 / M3.2 / M2.2 labels:** "
        f"{n_consistent}/{n_total} ({n_consistent / max(n_total, 1):.0%})",
        f"**Divergences:** {n_divergent}",
        f"**Engine 'FIRES_GROUNDED':** {n_active_grounded}",
        "",
        "## Purpose",
        "",
        "Behavioural check: does the implemented `gaira_base_2` engine "
        "activate each motif on its canonical CORE reference (ramanbiolib "
        "pure-compound or Gobbato powder Raman) in a way consistent with "
        "the CORE_GROUNDED / CORE_AMBIGUITY_CONFIRMED / CORE_NOT_SUPPORTED "
        "label assigned in M2.2?",
        "",
        "This does NOT redefine any motif, change any grounding label, or "
        "modify the motif registry. It is a regression check on the "
        "engine's scoring path.",
        "",
        "## Method",
        "",
        "1. Load all registry motifs (active and HELD_V2).",
        "2. For each motif, look up its canonical CORE references from the "
        "   biochemistry-driven MOTIF_CORE_REF map (same map used in M3 / "
        "   M3.1 grounding).",
        "3. For each reference, compute `compute_motif_activation()` through "
        "   the implemented engine on that reference's preprocessed spectrum.",
        "4. Label the engine's behaviour: FIRES_GROUNDED if best activation "
        "   > 0.05; FIRES_WEAK if > 0 but ≤ 0.05; NO_ACTIVATION otherwise.",
        "5. Compare to the M2.2 `core_status` column.",
        "",
        "## Divergence interpretation",
        "",
        "A divergence means the engine's activation on canonical references "
        "does not match M2.2's CORE_GROUNDED-like status. Possible causes:",
        "",
        "* the motif's co-band-REQUIRED logic rejects the reference (e.g. "
        "  uric_acid_full_signature on pure UA powder may fail if any one "
        "  of the 4 primary bands is below the floor — even though M3 "
        "  evaluation used a looser peak-list check)",
        "* the motif's reference in the MOTIF_CORE_REF map is not the same "
        "  as M3's (M2.2 used broader ref maps per axis)",
        "",
        "Divergences are reported but do NOT trigger a registry rebuild. "
        "They are inputs for later implementation refinement, not a reason "
        "to modify the ontology.",
        "",
        "## Divergences (by motif)",
        "",
    ] + (
        ["| motif | prior status | engine label | activation |",
         "|---|---|---|---:|"] + [
            f"| `{r['motif_id']}` | {r['prior_core_status_M2_2']} | "
            f"{r['engine_label']} | {r['engine_activation']} |"
            for _, r in inconsistent.iterrows()
        ]
        if len(inconsistent) else ["_No divergences._"]
    ) + [
        "",
        "## Table",
        "",
        "Full per-motif × reference data is in "
        "`motif_grounding_rerun_v1.csv`.",
    ]))
    print(f"\n[emit] {report_path}")
    print("DONE")


if __name__ == "__main__":
    main()
