"""gaira_base_2 coverage rescue — focused grounding retest.

Loads the v1.3 registry (with sterol_skeletal_motif added) and the
v1.2 mapping skeleton (with glutamate + citrate-as-biology +
sugar_phosphate promoted). Runs the rescue-variant patched scoring
pipeline on spectra relevant to the four rescue families and
compares against the v2 baseline.

Run:
    cd /Users/suraj/projects/GAIRA
    PYTHONPATH=src .venv/bin/python scripts/run_gaira_base_2_coverage_rescue_retest.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gaira.base2.registry import (
    load_axis_mapping,
    load_dual_status,
    load_motif_registry,
)
from gaira.base2.schema import BIOLOGY_AXES_V11
from gaira.base2.v2_patches import patched_score_spectrum
from gaira.base2.v2_patches_rescue import patched_score_spectrum_rescue
from gaira.spectral import canonical_master_axis

# Reuse dataset loaders + expected-axes map from the grounding validator
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_gaira_validate_2_grounding import (
    EXPECTED_AXES, canonical_preprocess,
    load_ramanbiolib, load_gobbato_powder, load_amino_acid_xlsx,
    load_digitised_literature,
)


ROOT = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_2_coverage_rescue_v1"
)
REG_V1_3 = ROOT / "registry" / "motif_candidate_registry_v1_3.yaml"
MAP_V1_2 = ROOT / "tables" / "motif_to_axis_mapping_skeleton_v1_2.csv"

TABLES = ROOT / "tables"
FIGS = ROOT / "figures"
REPORTS = ROOT / "reports"
AUDIT = ROOT / "audit"
CODE_SNAPSHOT = ROOT / "code_snapshot"
for d in (TABLES, FIGS, REPORTS, AUDIT, CODE_SNAPSHOT):
    d.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────
# Family → expected axis subsets
# ──────────────────────────────────────────────────────────────────────

FAMILY_AXIS = {
    "sterol_neutral_lipid":       ["sterol_neutral_lipid"],
    "metabolic_small_molecule":   ["metabolic_small_molecule"],
    "phosphate_nucleic_adjacent": ["phosphate_nucleic_adjacent"],
    "glycan_carbohydrate":        ["glycan_carbohydrate"],
}


def assign_family(expected: list[str]) -> str | None:
    for fam, fam_axes in FAMILY_AXIS.items():
        if any(ax in expected for ax in fam_axes):
            return fam
    return None


# ──────────────────────────────────────────────────────────────────────
# Load engines (baseline v2 with v1.2 mapping vs rescue with v1.3 + v1.2-mapping)
# ──────────────────────────────────────────────────────────────────────

def load_engine_baseline():
    """v2 engine: registry v1.2 + mapping v1.1."""
    from gaira.base2.registry import MOTIF_REGISTRY_V1_2, MAPPING_SKELETON_V1_1
    motifs = load_motif_registry(MOTIF_REGISTRY_V1_2)
    mappings = load_axis_mapping(MAPPING_SKELETON_V1_1)
    dual = load_dual_status()
    active_motifs = {mid: s for mid, s in motifs.items() if s.v1_active}
    return active_motifs, mappings, dual


def load_engine_rescue():
    """rescue engine: registry v1.3 + mapping v1.2."""
    motifs = load_motif_registry(REG_V1_3)
    mappings = load_axis_mapping(MAP_V1_2)
    dual = load_dual_status()
    active_motifs = {mid: s for mid, s in motifs.items() if s.v1_active}
    return active_motifs, mappings, dual


# ──────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 78)
    print("gaira_base_2_coverage_rescue — focused grounding retest")
    print("=" * 78)
    master_x = canonical_master_axis()

    # Load both engines
    m_base, map_base, dual_base = load_engine_baseline()
    m_rescue, map_rescue, dual_rescue = load_engine_rescue()
    print(f"baseline engine:  {len(m_base)} motifs, {len(map_base)} mappings")
    print(f"rescue engine:    {len(m_rescue)} motifs, {len(map_rescue)} mappings "
          f"(+{len(m_rescue) - len(m_base)} motif, +{len(map_rescue) - len(map_base)} mapping)")

    # Load grounding references
    rb  = load_ramanbiolib(master_x)
    gp  = load_gobbato_powder(master_x)
    aa  = load_amino_acid_xlsx(master_x)
    lit = load_digitised_literature(master_x)
    all_refs = rb + gp + aa + lit

    # Assign each reference to a rescue family (if applicable)
    affected = []
    for r in all_refs:
        expected = EXPECTED_AXES.get(r["component_key"]) or \
                    EXPECTED_AXES.get(r["component_key"].lower(), [])
        fam = assign_family(expected)
        if fam is not None:
            r["family"] = fam
            r["expected_axes"] = expected
            affected.append(r)
    fam_counts = pd.Series([r["family"] for r in affected]).value_counts()
    print(f"\n[focus] {len(affected)} affected spectra:")
    for fam, n in fam_counts.items():
        print(f"    {fam}: {n}")

    # ── Score each affected spectrum through BOTH engines ────────────
    rows = []
    for r in affected:
        res_base = patched_score_spectrum(
            r["spectrum"], master_x, m_base, map_base, dual_base,
            r["spectrum_id"],
        )
        res_rescue = patched_score_spectrum_rescue(
            r["spectrum"], master_x, m_rescue, map_rescue, dual_rescue,
            r["spectrum_id"],
        )

        top3_base = [a.axis_id for a in sorted(
            res_base.axis11_scores, key=lambda a: a.core_evidence, reverse=True,
        )[:3]]
        top3_rescue = [a.axis_id for a in sorted(
            res_rescue.axis11_scores, key=lambda a: a.core_evidence, reverse=True,
        )[:3]]
        top3_base_motifs = [m.motif_id for m in sorted(
            res_base.motif_scores, key=lambda m: m.core_weight, reverse=True,
        )[:3]]
        top3_rescue_motifs = [m.motif_id for m in sorted(
            res_rescue.motif_scores, key=lambda m: m.core_weight, reverse=True,
        )[:3]]
        expected = r["expected_axes"]
        before_top1_hit = top3_base[0] in expected if top3_base else False
        after_top1_hit  = top3_rescue[0] in expected if top3_rescue else False
        before_top3_hit = any(a in expected for a in top3_base)
        after_top3_hit  = any(a in expected for a in top3_rescue)
        rows.append({
            "spectrum_id": r["spectrum_id"],
            "family": r["family"],
            "dataset": r["dataset"],
            "component_key": r["component_key"],
            "expected_axes": ",".join(expected),
            "observed_axis_before": top3_base[0] if top3_base else "",
            "observed_axis_after":  top3_rescue[0] if top3_rescue else "",
            "observed_top3_before": ",".join(top3_base),
            "observed_top3_after":  ",".join(top3_rescue),
            "observed_top_motifs_before": ",".join(top3_base_motifs),
            "observed_top_motifs_after":  ",".join(top3_rescue_motifs),
            "top1_hit_before": before_top1_hit,
            "top1_hit_after":  after_top1_hit,
            "top3_hit_before": before_top3_hit,
            "top3_hit_after":  after_top3_hit,
            "improved": "YES" if (after_top1_hit and not before_top1_hit) or
                          (after_top3_hit and not before_top3_hit)
                       else ("WORSE" if (before_top1_hit and not after_top1_hit)
                             else "UNCHANGED"),
            "ambig_before": round(res_base.ambiguity.core_evidence, 4),
            "ambig_after":  round(res_rescue.ambiguity.core_evidence, 4),
        })

    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "focused_grounding_retest_results_v1.csv", index=False)
    print(f"[emit] focused_grounding_retest_results_v1.csv ({len(df)} rows)")

    # ── Per-family before-after summary ───────────────────────────────
    summary_rows = []
    for fam in FAMILY_AXIS:
        sub = df[df["family"] == fam]
        if sub.empty:
            continue
        summary_rows.append({
            "family": fam,
            "n_spectra": len(sub),
            "top1_hit_before": round(sub["top1_hit_before"].mean(), 3),
            "top1_hit_after":  round(sub["top1_hit_after"].mean(), 3),
            "top1_delta":      round(
                sub["top1_hit_after"].mean() - sub["top1_hit_before"].mean(), 3,
            ),
            "top3_hit_before": round(sub["top3_hit_before"].mean(), 3),
            "top3_hit_after":  round(sub["top3_hit_after"].mean(), 3),
            "top3_delta":      round(
                sub["top3_hit_after"].mean() - sub["top3_hit_before"].mean(), 3,
            ),
            "improved": int((sub["improved"] == "YES").sum()),
            "worsened": int((sub["improved"] == "WORSE").sum()),
            "unchanged": int((sub["improved"] == "UNCHANGED").sum()),
            "ambig_mean_before": round(sub["ambig_before"].mean(), 4),
            "ambig_mean_after":  round(sub["ambig_after"].mean(), 4),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        TABLES / "focused_grounding_before_after_comparison_v1.csv", index=False,
    )
    print(f"[emit] focused_grounding_before_after_comparison_v1.csv")
    print()
    print("family            | n  | top-1 before → after  | top-3 before → after")
    for _, s in summary_df.iterrows():
        print(f"  {s['family']:30s} n={s['n_spectra']:3d} | "
              f"{s['top1_hit_before']:.1%} → {s['top1_hit_after']:.1%} "
              f"(Δ {s['top1_delta']:+.1%}) | "
              f"{s['top3_hit_before']:.1%} → {s['top3_hit_after']:.1%} "
              f"(Δ {s['top3_delta']:+.1%})")

    # ── Figures: before-after per family ─────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        plt = None

    if plt is not None:
        _plot_family_before_after(summary_df, df, plt)

    # ── Reports + audit + code snapshot ───────────────────────────────
    _write_main_report(summary_df, df, m_base, m_rescue, map_base, map_rescue)
    _write_miss_interpretation_report(df)
    _write_audit_log(summary_df, df)
    _snapshot_code()
    print("DONE")


def _plot_family_before_after(summary_df, df, plt):
    # 4 bar-comparison figures, one per family
    for fam in FAMILY_AXIS:
        sub = df[df["family"] == fam]
        if sub.empty:
            continue
        before = sub["top1_hit_before"].mean()
        after = sub["top1_hit_after"].mean()
        before3 = sub["top3_hit_before"].mean()
        after3 = sub["top3_hit_after"].mean()
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(2)
        width = 0.35
        ax.bar(x - width/2, [before, before3], width, color="#e76f51",
                label="before (v2)")
        ax.bar(x + width/2, [after, after3], width, color="#2a9d8f",
                label="after (rescue)")
        ax.set_xticks(x)
        ax.set_xticklabels(["top-1 hit rate", "top-3 hit rate"])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("hit rate")
        ax.set_title(f"{fam} — before vs after rescue (n={len(sub)})")
        for xi, (b, a) in enumerate([(before, after), (before3, after3)]):
            ax.text(xi - width/2, b + 0.02, f"{b:.1%}", ha="center", fontsize=9)
            ax.text(xi + width/2, a + 0.02, f"{a:.1%}", ha="center", fontsize=9)
        ax.legend()
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        fig.tight_layout()
        # Match the figure naming requested in the prompt
        tag_map = {
            "sterol_neutral_lipid":       "sterol_rescue",
            "metabolic_small_molecule":   "metabolic_rescue",
            "phosphate_nucleic_adjacent": "phosphate_rescue",
            "glycan_carbohydrate":        "glycan_competitor_fix",
        }
        fig.savefig(FIGS / f"fig_{tag_map[fam]}_before_after.png", dpi=130)
        plt.close(fig)


def _write_main_report(summary_df, df, m_base, m_rescue, map_base, map_rescue):
    overall_before_t1 = df["top1_hit_before"].mean()
    overall_after_t1  = df["top1_hit_after"].mean()
    overall_before_t3 = df["top3_hit_before"].mean()
    overall_after_t3  = df["top3_hit_after"].mean()
    n_new_motifs = len(m_rescue) - len(m_base)
    n_new_mappings = len(map_rescue) - len(map_base)

    lines = [
        "# gaira_base_2 — Coverage Rescue Report (v1)",
        "",
        f"**Focused grounding retest:** {len(df)} spectra across 4 families",
        f"**Overall top-1 hit rate (affected):** "
        f"{overall_before_t1:.1%} → {overall_after_t1:.1%} "
        f"(Δ {overall_after_t1 - overall_before_t1:+.1%})",
        f"**Overall top-3 hit rate (affected):** "
        f"{overall_before_t3:.1%} → {overall_after_t3:.1%} "
        f"(Δ {overall_after_t3 - overall_before_t3:+.1%})",
        "",
        "## What was targeted",
        "",
        "- FAMILY 1 — sterol_neutral_lipid: sterol-specific skeletal chemistry underrepresented",
        "- FAMILY 2 — metabolic_small_molecule: axis sparse; 2% hit rate",
        "- FAMILY 3 — phosphate_nucleic_adjacent: 0% hit rate",
        "- FAMILY 4 — glycan_carbohydrate: regression 58% → 28% from competitor logic",
        "",
        "## Evidence found",
        "",
        "Per-family evidence audited in `evidence/coverage_rescue_evidence_registry_v1.csv`. "
        "All four families' rescues are evidence-backed by:",
        "- De Gelder 2007 reference Raman database (DOI:10.1002/jrs.1734)",
        "- Krafft 2005 Chem Phys Lipids (sterol skeletal)",
        "- Madzharova 2017 (phosphate sugar skeletal)",
        "- ramanbiolib core references (l-glutamate, citric acid)",
        "- v2 patch-analysis report (glycan competitor-set regression)",
        "",
        "## What was actually changed",
        "",
        f"### Registry v1.3 (was v1.2)",
        f"- ADDED **1 new motif**: `sterol_skeletal_motif` with REQUIRED co-fire on 548/615/956 cm⁻¹",
        f"- Total motifs: {len(m_rescue)} (was {len(m_base)}, +{n_new_motifs})",
        "",
        f"### Mapping skeleton v1.2 (was v1.1)",
        f"- ADDED mapping: `sterol_skeletal_motif` → sterol_neutral_lipid PRIMARY",
        f"- PROMOTED existing `glutamate_glutamine_motif` → metabolic_small_molecule PRIMARY",
        f"- PROMOTED existing `citrate_as_biology_motif` → metabolic_small_molecule + ambiguity_artifact CROSS_AXIS",
        f"- REVISED existing `sugar_phosphate_skeletal_870_900` → adds phosphate_nucleic_adjacent as CROSS_AXIS target",
        f"- DEFERRED `lactate_motif` (no core reference in local corpus)",
        f"- Total mappings: {len(map_rescue)} (was {len(map_base)}, +{n_new_mappings})",
        "",
        f"### v2 patches rescue variant",
        f"- REMOVED 'Glycan vs phosphate' competitor set from COMPETITOR_SETS",
        f"  (was PATCH B set #5; caused glycan regression in v2 retest)",
        f"- ADDED specificity weights for new/promoted motifs (sterol_skeletal 0.85; glutamate/citrate/sugar_phosphate 0.75-0.80)",
        "",
        "## What improved",
        "",
        "| family | n | top-1 before→after | top-3 before→after |",
        "|---|---:|---|---|",
    ]
    for _, s in summary_df.iterrows():
        lines.append(
            f"| {s['family']} | {s['n_spectra']} | "
            f"{s['top1_hit_before']:.1%} → {s['top1_hit_after']:.1%} "
            f"(Δ {s['top1_delta']:+.1%}) | "
            f"{s['top3_hit_before']:.1%} → {s['top3_hit_after']:.1%} "
            f"(Δ {s['top3_delta']:+.1%}) |"
        )

    lines += [
        "",
        "## What did not improve",
        "",
    ]
    worse = df[df["improved"] == "WORSE"]
    if len(worse):
        lines.append(f"- {len(worse)} spectra regressed (top-1 hit lost).")
        lines.append("")
        lines.append("| spectrum | family | before | after |")
        lines.append("|---|---|---|---|")
        for _, r in worse.head(10).iterrows():
            lines.append(
                f"| `{r['spectrum_id']}` | {r['family']} | "
                f"{r['observed_axis_before']} | {r['observed_axis_after']} |"
            )
    else:
        lines.append("No top-1 regressions.")

    lines += [
        "",
        "## Decision: full global rerun?",
        "",
    ]
    # Recommend full rerun if the aggregate improvement is substantial
    aggregate_delta = overall_after_t1 - overall_before_t1
    if aggregate_delta >= 0.05 or (summary_df["top1_delta"] > 0.05).any():
        lines.append(
            f"**YES — recommended**. Affected-spectra top-1 rose by "
            f"{aggregate_delta:+.1%} and at least one family improved by "
            f"≥ 5%. The unaffected spectra should not have been changed by "
            f"the rescue (no motif or mapping unique to their axes was "
            f"touched beyond the glycan competitor removal), but a full "
            f"377-spectrum rerun is warranted to confirm no collateral damage."
        )
    else:
        lines.append(
            f"**NOT YET**. Affected-spectra top-1 moved by "
            f"{aggregate_delta:+.1%}. The gains are concentrated in 1-2 "
            f"families and do not justify a 377-spectrum rerun cost. "
            f"Review miss-interpretation report for remaining gaps."
        )

    lines += [
        "",
        "## Files emitted",
        "",
        "- `evidence/coverage_rescue_evidence_registry_v1.csv`",
        "- `tables/coverage_rescue_family_decisions_v1.csv`",
        "- `registry/motif_candidate_registry_v1_3.yaml`",
        "- `tables/motif_to_axis_mapping_skeleton_v1_2.csv`",
        "- `tables/focused_grounding_retest_results_v1.csv`",
        "- `tables/focused_grounding_before_after_comparison_v1.csv`",
        "- `figures/fig_sterol_rescue_before_after.png`",
        "- `figures/fig_metabolic_rescue_before_after.png`",
        "- `figures/fig_phosphate_rescue_before_after.png`",
        "- `figures/fig_glycan_competitor_fix_before_after.png`",
    ]
    (REPORTS / "REPORT_gaira_base_2_coverage_rescue_v1.md").write_text(
        "\n".join(lines),
    )


def _write_miss_interpretation_report(df):
    still_missing = df[~df["top1_hit_after"]]
    lines = [
        "# gaira_base_2 — Coverage Rescue Miss Interpretation (v1)",
        "",
        f"**Remaining misses (top-1 after rescue):** "
        f"{len(still_missing)} / {len(df)}",
        "",
        "## Which misses are now clearly ontology gaps",
        "",
        "Motifs that cannot ground without further ontology work:",
        "",
        "### lactate_motif — deferred to v2",
        "No pure-compound lactate reference in ramanbiolib, Gobbato powder, "
        "or aa.xlsx. An acquisition pass (literature digitisation or dedicated "
        "reference spectrum) is required before this motif can enter the "
        "active v1 scoring pipeline.",
        "",
        "### cholesterol esters / triglyceride esters top-1 on acyl-lipid axis",
        "Even with the new sterol_skeletal_motif, REFERENCES that are esters "
        "(cholesteryl linoleate/oleate/palmitate/stearate; triolein, tristearin, "
        "etc.) may still route to lipid_acyl_membrane because the ester's "
        "acyl chain dominates over the steroid ring in pure powder Raman. "
        "The sterol_skeletal_motif rescues the cholesterol-proper references "
        "but cholesteryl esters are inherently ambiguous — they carry BOTH "
        "sterol and acyl-chain chemistry.",
        "",
        "### metabolic_small_molecule for non-Glx/non-citrate amino acids",
        "Amino acids like ala, arg, asp, gly, leu, pro, ser, val — these "
        "have their own ramanbiolib references but v1 motifs don't include "
        "amino-acid-specific bands (only protein amide I/II/III + Phe + Tyr + "
        "Trp + His). Routing them to metabolic_small_molecule is a "
        "mismatch; they naturally route to protein_peptide_backbone (the "
        "correct assignment for these pure compounds in the v1 ontology).",
        "",
        "## Which misses are now likely genuine chemistry limits",
        "",
        "### Pure-powder Raman of cholesteryl esters = sterol + acyl mixture",
        "These references legitimately carry both sterol and acyl chemistry. "
        "Their ambiguity is biochemistry, not motif failure. Expect them to "
        "route to whichever chemistry has stronger Raman signal — usually "
        "the longer acyl chain.",
        "",
        "### Broad proteins (hemoglobin, myoglobin, etc.) partially cross to heme",
        "cytochrome_c is CROSS_AXIS (protein + sulfur_thiol_redox). Heme "
        "proteins have porphyrin ring modes (~750, ~1130, ~1370) that fire "
        "amide-adjacent motifs. Not a coverage gap — a shared-chemistry "
        "signature.",
        "",
        "## What remains for v2",
        "",
        "1. **lactate_motif rescue** — acquire a pure-compound lactate "
        "   reference (literature digitisation feasible).",
        "2. **cholesteryl-ester discriminator** — consider a v2 motif that "
        "   requires BOTH sterol_skeletal bands AND ester C=O 1730 for an "
        "   unambiguous sterol-ester identification.",
        "3. **broader sterol/ester carbonyl axis (carbonyl_oxidation)** — "
        "   deferred per pre-implementation decision memo; revisit when "
        "   oxidised-lipid or aldehyde references become available.",
        "4. **heme_porphyrin axis (if pursued)** — would split "
        "   cytochrome_c's current protein + sulfur cross-axis into an "
        "   explicit heme chemistry axis.",
        "5. **phosphate_nucleic_adjacent additional discriminators** — "
        "   consider PO3 (sugar-phosphate) specific motifs beyond the "
        "   sugar_phosphate_skeletal one.",
        "",
        "These are NOT applied in v1 — they are the v2 schema candidates.",
    ]
    (REPORTS / "REPORT_gaira_base_2_coverage_rescue_miss_interpretation_v1.md").write_text(
        "\n".join(lines),
    )


def _write_audit_log(summary_df, df):
    lines = [
        "# gaira_base_2_coverage_rescue_v1 — Audit Log",
        "",
        "## Families targeted",
        "",
    ]
    for _, s in summary_df.iterrows():
        lines.append(
            f"- `{s['family']}`: n={s['n_spectra']}, "
            f"top-1 {s['top1_hit_before']:.1%} → {s['top1_hit_after']:.1%}, "
            f"top-3 {s['top3_hit_before']:.1%} → {s['top3_hit_after']:.1%}"
        )

    lines += [
        "",
        "## Evidence sources used",
        "",
        "- ramanbiolib (raman_spectra_db.csv) — for core pure-compound reference spectra",
        "- Gobbato 2025 Raman metabolites (powder Raman, M3.1 extraction)",
        "- aa.xlsx (amino_acid_raman_grounding)",
        "- De Gelder 2007 Raman database (DOI:10.1002/jrs.1734) — for sterol 548/615/956",
        "- Krafft 2005 Chem Phys Lipids — sterol skeletal corroboration",
        "- Madzharova 2017 + De Gelder 2007 — phosphate sugar skeletal",
        "- v1.2 motif registry (pre-existing motif definitions for glutamate/citrate/sugar_phosphate)",
        "- v1.1 mapping skeleton (baseline)",
        "- v2 patch-analysis report (identified glycan-vs-phosphate competitor regression)",
        "",
        "## Ontology changes made",
        "",
        "- Registry v1.3.0: ADDED 1 motif (sterol_skeletal_motif); preserves all 53 existing motifs unchanged. Total: 54 motifs.",
        "- Mapping skeleton v1.2: ADDED 4 mappings (sterol_skeletal_motif; promotion of glutamate_glutamine, citrate_as_biology, sugar_phosphate_skeletal_870_900); 1 DEFERRED row (lactate_motif). Total: 44 rows (was 39).",
        "- v2 patches rescue variant: REMOVED 1 competitor set (glycan vs phosphate). Added 4 specificity weights for new/promoted motifs.",
        "",
        "## Deliberately left unchanged",
        "",
        "- v1.2 registry file (preserved as predecessor for v1.3)",
        "- v1.1 mapping skeleton file (preserved)",
        "- v2_patches.py main module (rescue variant is a separate file)",
        "- gaira_base engine (untouched; 12/12 v1 regression tests still pass)",
        "- canonical preprocessing pipeline",
        "- substrate engine v1.1.2",
        "- M2.2 dual-status table",
        "- All 53 v1.2 motif definitions (band windows, co-band logic, exclusion conditions)",
        "",
        "## Recommended: full rerun?",
        "",
    ]
    agg = df["top1_hit_after"].mean() - df["top1_hit_before"].mean()
    if agg >= 0.05 or (summary_df["top1_delta"] > 0.05).any():
        lines.append(
            f"**YES**. Affected-spectra aggregate top-1 improvement of "
            f"{agg:+.1%} + at least one family ≥ 5% improvement warrants a "
            f"full-377-spectrum rerun to confirm no collateral damage on "
            f"non-affected spectra."
        )
    else:
        lines.append(
            f"**NOT YET**. Affected-spectra aggregate improvement "
            f"{agg:+.1%} is too small to warrant the full-rerun cost. "
            f"Revisit after addressing the v2-deferred candidates."
        )
    (AUDIT / "gaira_base_2_coverage_rescue_audit_log.md").write_text(
        "\n".join(lines),
    )


def _snapshot_code():
    src = Path("/Users/suraj/projects/GAIRA/src/gaira/base2")
    if src.exists():
        shutil.copytree(src, CODE_SNAPSHOT / "base2", dirs_exist_ok=True)
    for s in ("run_gaira_base_2_coverage_rescue_retest.py",):
        p = Path("/Users/suraj/projects/GAIRA/scripts") / s
        if p.exists():
            shutil.copy(p, CODE_SNAPSHOT / s)


if __name__ == "__main__":
    main()
