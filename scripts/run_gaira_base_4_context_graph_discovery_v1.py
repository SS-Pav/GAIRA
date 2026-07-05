"""gaira_base_4_context_graph_discovery_v1.

Backend discovery pass — crawl every completed GAIRA phase, build a unified
evidence-event table, derive multi-view context graphs, run unsupervised
clustering on dataset/condition embeddings, and write a ranked-findings
report.

Strict invariants
-----------------
- GAIRA core unchanged; this script is read-only over build artifacts.
- No GAIRA scoring rerun.
- Heterogeneous table schemas tolerated — every parser is catch-and-skip.
- Output lives ENTIRELY under
  /Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_4_context_graph_discovery_v1/
"""
from __future__ import annotations

import json
import re
import shutil
import sys
import warnings
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

warnings.simplefilter("ignore")

# ─────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────
BUILD_ROOT = Path("/Volumes/SSD_Rad/GAIRA_BUILD")
ROOT = BUILD_ROOT / "gaira_base_4_context_graph_discovery_v1"
T = ROOT / "tables"
F = ROOT / "figures"
R = ROOT / "reports"
G = ROOT / "graph"
C = ROOT / "code_snapshot"
for d in (T, F, R, G, C): d.mkdir(parents=True, exist_ok=True)

EXCLUDE_DIR = ROOT.name  # never crawl our own outputs

BIOLOGY_AXES_V11 = [f"G{i:02d}" for i in range(1, 12)]
AXIS_NAMES = {
    "G01": "purine_nucleotide", "G02": "purine_metabolite",
    "G03": "pyrimidine_nucleotide", "G04": "nucleic_acid_phosphate",
    "G05": "glycan_carbohydrate", "G06": "protein_peptide_backbone",
    "G07": "aromatic_residue", "G08": "lipid_acyl_membrane",
    "G09": "sterol_neutral_lipid", "G10": "sulfur_thiol_redox",
    "G11": "metabolic_small_molecule",
}

# ─────────────────────────────────────────────────────────────────────────
# Phase classification
# ─────────────────────────────────────────────────────────────────────────

PHASE_RULES = [
    # (substring, sample_type, pilot_group, condition_family)
    ("diabetes_ev",                "EV",          "EV",          "diabetes_metabolic"),
    ("shine_ev",                   "EV",          "EV",          "toxicity_drug_response"),
    ("shine_otc",                  "EV",          "EV",          "toxicity_drug_response"),
    ("shine_regression",           "EV",          "EV",          "toxicity_drug_response"),
    ("small_ev",                   "EV",          "EV",          "dual_probe_invariance"),
    ("otc_pure_raman_mss",         "pure_Raman",  "grounding",   "pure_calibration"),
    ("otc_bsv_mss",                "pure_Raman",  "grounding",   "pure_calibration"),
    ("otc_drug_detection",         "EV",          "EV",          "toxicity_drug_response"),
    ("otc_mss_detector",           "pure_Raman",  "grounding",   "pure_calibration"),
    ("substrate_calibration",      "SERS",        "calibration", "substrate_calibration"),
    ("european_adenine",           "SERS",        "calibration", "substrate_calibration"),
    ("hybrid_bsv_calibration",     "mixed",       "calibration", "calibration_suite"),
    ("hybrid_bsv_build",           "mixed",       "grounding",   "hybrid_build"),
    ("hybrid_bsv_calibration_audit","mixed",      "calibration", "calibration_suite"),
    ("calibration_fixes_before_v3","mixed",       "calibration", "calibration_suite"),
    ("liver_narrow_metabolite",    "serum",       "serum",       "liver_cancer"),
    ("paper_band_vs_ground_truth", "pure_Raman",  "grounding",   "paper_band_validation"),
    ("passive_target_pilot_1",     "serum",       "serum",       "liver_cancer"),
    ("passive_target_pilot_2",     "serum",       "serum",       "liver_cancer"),
    ("passive_target_pilot_3a",    "serum",       "serum",       "infection_covid"),
    ("passive_target_pilot_3A",    "serum",       "serum",       "infection_covid"),
    ("passive_target_pilot_3b",    "serum",       "serum",       "infection_covid"),
    ("pilot3c",                    "serum",       "serum",       "cross_disease"),
    ("hcc",                        "serum",       "serum",       "liver_cancer"),
    ("cross_pilot_synthesis",      "mixed",       "synthesis",   "cross_pilot"),
    ("cross_pilot_generalization", "serum",       "serum",       "cross_pilot"),
    ("validate_2_grounding",       "pure_Raman",  "grounding",   "grounding_validation"),
    ("mss_core_build",             "pure_Raman",  "grounding",   "mss_build"),
    ("mss_repair_loop",            "pure_Raman",  "grounding",   "mss_build"),
    ("mss_decision_enrichment",    "pure_Raman",  "grounding",   "mss_build"),
    ("mss_readiness",              "pure_Raman",  "grounding",   "mss_build"),
    ("representation_cluster",     "pure_Raman",  "grounding",   "representation"),
    ("packet_ontology",            "pure_Raman",  "grounding",   "ontology"),
    ("structural_anti_evidence",   "pure_Raman",  "grounding",   "ontology"),
    ("competitor_anti_evidence",   "pure_Raman",  "grounding",   "ontology"),
    ("core_signature_validation",  "pure_Raman",  "grounding",   "ontology"),
    ("full_grounding_audit",       "pure_Raman",  "grounding",   "ontology"),
    ("grounding_full_corpus",      "pure_Raman",  "grounding",   "ontology"),
    ("grounding_trained_ontology", "pure_Raman",  "grounding",   "ontology"),
    ("grounding_repair",           "pure_Raman",  "grounding",   "ontology"),
    ("targeted_anchor_acquisition","pure_Raman",  "grounding",   "ontology"),
    ("discriminative_motif",       "pure_Raman",  "grounding",   "ontology"),
    ("v1_closure_pass",            "pure_Raman",  "grounding",   "ontology"),
    ("revert_v4",                  "pure_Raman",  "grounding",   "ontology"),
    ("evidence_gating_repair",     "pure_Raman",  "grounding",   "ontology"),
    ("final_ranking_repair",       "pure_Raman",  "grounding",   "ontology"),
    ("coverage_rescue",            "pure_Raman",  "grounding",   "ontology"),
    ("backend_validation",         "pure_Raman",  "grounding",   "ontology"),
    ("base_2",                     "pure_Raman",  "grounding",   "ontology"),
    ("base_3",                     "pure_Raman",  "grounding",   "ontology"),
    ("preimplementation_pressure", "pure_Raman",  "grounding",   "ontology"),
    ("substrate_physics",          "n/a",         "physics",     "substrate_physics"),
    ("sers_chemical_space",        "SERS",        "grounding",   "sers_corpus"),
]


def classify_phase(phase: str) -> tuple[str, str, str]:
    p = phase.lower()
    for sub, samp, pg, cf in PHASE_RULES:
        if sub.lower() in p:
            return samp, pg, cf
    return "unknown", "unknown", "unknown"


# ─────────────────────────────────────────────────────────────────────────
# Stage 1 — artifact inventory
# ─────────────────────────────────────────────────────────────────────────

USEFUL_PATTERNS = (
    "bsv", "mss", "delta", "cohort", "classifier", "regression",
    "feature_importance", "axis_comparison", "cross_pilot", "candidate",
    "effect", "transfer", "calibration", "scorecard", "binary",
    "cluster_assignments", "monotonicity", "dose_response",
    "stability", "selectivity", "axis_rank", "rank_comparison",
)


def stage1_inventory() -> pd.DataFrame:
    print("[stage 1] artifact inventory walk")
    rows = []
    for p in BUILD_ROOT.rglob("*.csv"):
        if "._" in p.name or "__pycache__" in str(p) or EXCLUDE_DIR in p.parts:
            continue
        try:
            phase = next((part for part in p.parts
                          if part.startswith(("gaira_base_", "gaira_validate_",
                                                "gaira_representation_",
                                                "gaira_sers_", "substrate_physics"))),
                         p.parent.name)
            samp, pg, cf = classify_phase(phase)
            usable = any(pat in p.name.lower() for pat in USEFUL_PATTERNS)
            rows.append({
                "file_path": str(p),
                "inferred_phase": phase,
                "dataset": phase,
                "sample_type": samp,
                "pilot_group": pg,
                "condition_family": cf,
                "artifact_type": "csv",
                "name": p.name,
                "size_bytes": p.stat().st_size,
                "usable_for_graph": usable,
            })
        except Exception:
            continue
    for p in BUILD_ROOT.rglob("*.md"):
        if "._" in p.name or EXCLUDE_DIR in p.parts:
            continue
        phase = next((part for part in p.parts
                      if part.startswith(("gaira_base_", "gaira_validate_",
                                            "gaira_representation_",
                                            "gaira_sers_", "substrate_physics"))),
                     p.parent.name)
        samp, pg, cf = classify_phase(phase)
        rows.append({
            "file_path": str(p),
            "inferred_phase": phase,
            "dataset": phase,
            "sample_type": samp,
            "pilot_group": pg,
            "condition_family": cf,
            "artifact_type": "md",
            "name": p.name,
            "size_bytes": p.stat().st_size,
            "usable_for_graph": "report" in p.name.lower() or "audit" in p.name.lower(),
        })
    df = pd.DataFrame(rows)
    df.to_csv(T / "context_graph_artifact_inventory.csv", index=False)
    print(f"  inventoried {len(df)} files "
          f"({(df.artifact_type=='csv').sum()} csv / {(df.artifact_type=='md').sum()} md)")
    return df


# ─────────────────────────────────────────────────────────────────────────
# Stage 2 — parse evidence events into long form
# ─────────────────────────────────────────────────────────────────────────

def _read_safe(p: str) -> pd.DataFrame | None:
    try:
        return pd.read_csv(p)
    except Exception:
        return None


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    # case-insensitive fallback
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def _direction(value: float, pos="up", neg="down", thresh: float = 0.10) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "ambiguous"
    if value > thresh:
        return pos
    if value < -thresh:
        return neg
    return "stable"


def _emit(events: list, **kw) -> None:
    e = {k: kw.get(k, "") for k in [
        "event_id", "dataset", "pilot_group", "sample_type", "condition_family",
        "condition_A", "condition_B", "comparison_type",
        "bsv_axis", "bsv_axis_name", "direction",
        "effect_size", "metric_type", "metric_value",
        "mss_candidate", "motif_cluster", "band_cm",
        "confidence_tier", "caveat", "source_file"]}
    if not e["event_id"]:
        e["event_id"] = f"E{len(events):07d}"
    events.append(e)


def parse_binary_effects(p: Path, events: list, meta: dict) -> int:
    df = _read_safe(p)
    if df is None or df.empty:
        return 0
    # axis can be encoded as `axis`, `bsv_axis`, or — in serum/cross-pilot
    # phases — `family` (with G01..G11 string values).
    axis_col = _pick_col(df, ["axis", "bsv_axis", "family"])
    d_col = _pick_col(df, ["cohens_d", "cohens_d_C40_vs_C0_clr",
                           "cohens_d_owd_minus_nwd", "effect_size", "delta",
                           "abs_d", "mean_d", "d"])
    if axis_col is None or d_col is None:
        return 0
    n = 0
    comp_col = _pick_col(df, ["comparison", "comparison_label", "pair"])
    rep_col = _pick_col(df, ["representation", "feature_set"])
    sub_col = _pick_col(df, ["substrate", "substrate_block"])
    pilot_col = _pick_col(df, ["pilot", "set_id", "cohort"])
    for _, r in df.iterrows():
        ax = str(r[axis_col]).strip()
        if ax not in BIOLOGY_AXES_V11:
            continue
        try:
            d = float(r[d_col])
        except Exception:
            continue
        ci_excl = r.get("ci_excludes_zero", None)
        tier = "MODERATE" if abs(d) >= 0.20 else "WEAK"
        if abs(d) >= 0.50:
            tier = "STRONG"
        condA = (str(r[comp_col]) if comp_col else meta.get("condA", ""))
        condB = meta.get("condB", "")
        # Encode substrate / pilot / representation hints into condition_A
        extras = []
        if rep_col and pd.notna(r.get(rep_col)): extras.append(f"rep={r[rep_col]}")
        if sub_col and pd.notna(r.get(sub_col)): extras.append(f"substrate={r[sub_col]}")
        if pilot_col and pd.notna(r.get(pilot_col)): extras.append(f"pilot={r[pilot_col]}")
        if extras:
            condA = f"{condA}|{'|'.join(extras)}"
        _emit(events,
              dataset=meta["phase"], pilot_group=meta["pg"],
              sample_type=meta["samp"], condition_family=meta["cf"],
              condition_A=condA, condition_B=condB,
              comparison_type=meta.get("comp", "binary_effect"),
              bsv_axis=ax, bsv_axis_name=AXIS_NAMES.get(ax, ""),
              direction=_direction(d, "up", "down", 0.20),
              effect_size=d,
              metric_type="cohens_d", metric_value=d,
              confidence_tier=("STRONG" if ci_excl is True else tier),
              source_file=str(p))
        n += 1
    return n


def parse_dose_response(p: Path, events: list, meta: dict) -> int:
    df = _read_safe(p)
    if df is None or df.empty:
        return 0
    axis_col = _pick_col(df, ["axis", "bsv_axis"])
    rho_col = _pick_col(df, ["spearman_rho_dose", "spearman_rho", "rho"])
    pearson = _pick_col(df, ["pearson_r_dose", "pearson_r"])
    endpoint = _pick_col(df, ["endpoint_C40_minus_C0_clr", "endpoint_delta",
                              "endpoint_clr"])
    if axis_col is None:
        return 0
    n = 0
    for _, r in df.iterrows():
        ax = str(r[axis_col]).strip()
        if ax not in BIOLOGY_AXES_V11:
            continue
        rho = None
        if rho_col is not None:
            try: rho = float(r[rho_col])
            except: rho = None
        ep = None
        if endpoint is not None:
            try: ep = float(r[endpoint])
            except: ep = None
        condA = str(r.get("set_id", "")) + ("_" + str(r.get("day", ""))
                                              if "day" in df.columns else "")
        if rho is not None:
            _emit(events,
                  dataset=meta["phase"], pilot_group=meta["pg"],
                  sample_type=meta["samp"], condition_family=meta["cf"],
                  condition_A=condA, condition_B="dose_axis",
                  comparison_type="dose_response",
                  bsv_axis=ax, bsv_axis_name=AXIS_NAMES.get(ax, ""),
                  direction=_direction(rho, "up", "down", 0.30),
                  effect_size=rho,
                  metric_type="spearman_rho_dose", metric_value=rho,
                  confidence_tier=("STRONG" if abs(rho) >= 0.80
                                    else "MODERATE" if abs(rho) >= 0.50
                                    else "WEAK"),
                  source_file=str(p))
            n += 1
        if ep is not None:
            _emit(events,
                  dataset=meta["phase"], pilot_group=meta["pg"],
                  sample_type=meta["samp"], condition_family=meta["cf"],
                  condition_A=condA, condition_B="endpoint",
                  comparison_type="endpoint_delta",
                  bsv_axis=ax, bsv_axis_name=AXIS_NAMES.get(ax, ""),
                  direction=_direction(ep, "up", "down", 0.20),
                  effect_size=ep,
                  metric_type="endpoint_clr_delta", metric_value=ep,
                  confidence_tier=("STRONG" if abs(ep) >= 1.0
                                    else "MODERATE" if abs(ep) >= 0.5
                                    else "WEAK"),
                  source_file=str(p))
            n += 1
    return n


def parse_axis_rank_comparison(p: Path, events: list, meta: dict) -> int:
    df = _read_safe(p)
    if df is None or df.empty:
        return 0
    axis_col = _pick_col(df, ["axis"])
    if axis_col is None:
        return 0
    rank_col = _pick_col(df, ["rank_combined", "combined_score"])
    n = 0
    for _, r in df.iterrows():
        ax = str(r[axis_col]).strip()
        if ax not in BIOLOGY_AXES_V11:
            continue
        try:
            score = float(r["combined_score"])
        except Exception:
            score = None
        try:
            rank = int(r["rank_combined"]) if "rank_combined" in df.columns else None
        except Exception:
            rank = None
        probe = str(r.get("probe", "panel"))
        # transfer evidence: rank 1-3 = strong axis for that probe
        if rank is not None and rank <= 3:
            _emit(events,
                  dataset=meta["phase"], pilot_group=meta["pg"],
                  sample_type=meta["samp"], condition_family=meta["cf"],
                  condition_A=probe, condition_B="top_axis",
                  comparison_type="probe_axis_rank",
                  bsv_axis=ax, bsv_axis_name=AXIS_NAMES.get(ax, ""),
                  direction="stable" if score is None else "up",
                  effect_size=score,
                  metric_type="rank_combined", metric_value=float(rank),
                  confidence_tier="MODERATE",
                  source_file=str(p))
            n += 1
    return n


def parse_classifier_importance(p: Path, events: list, meta: dict) -> int:
    df = _read_safe(p)
    if df is None or df.empty:
        return 0
    feat_col = _pick_col(df, ["feature"])
    coef_col = _pick_col(df, ["abs_coef", "coef", "importance"])
    if feat_col is None or coef_col is None:
        return 0
    df = df.copy()
    df[coef_col] = pd.to_numeric(df[coef_col], errors="coerce")
    df = df.sort_values(coef_col, ascending=False).head(40)
    n = 0
    for _, r in df.iterrows():
        feat = str(r[feat_col])
        try:
            v = float(r[coef_col])
        except Exception:
            continue
        # Try to extract axis and molecule tokens from feature name
        ax_match = re.search(r"\b(G\d{2})\b", feat)
        ax = ax_match.group(1) if ax_match else ""
        mol_match = re.search(r"(palmitic_acid|ergothioneine|tyrosine|lactate|"
                              r"uric_acid|hypoxanthine|xanthine|adenine|cholesterol|"
                              r"glutathione|cysteine|cystine|creatinine|phenylalanine|"
                              r"tryptophan|stearic_acid|oleic_acid|urea|glucose)", feat,
                              flags=re.I)
        mol = mol_match.group(1).lower() if mol_match else ""
        _emit(events,
              dataset=meta["phase"], pilot_group=meta["pg"],
              sample_type=meta["samp"], condition_family=meta["cf"],
              condition_A=str(r.get("model", "logreg")), condition_B="top_feature",
              comparison_type="classifier_importance",
              bsv_axis=ax, bsv_axis_name=AXIS_NAMES.get(ax, ""),
              direction="up", effect_size=v,
              metric_type="abs_coef", metric_value=v,
              mss_candidate=mol,
              confidence_tier="MODERATE",
              source_file=str(p))
        n += 1
    return n


def parse_mss_top_hits(p: Path, events: list, meta: dict) -> int:
    df = _read_safe(p)
    if df is None or df.empty:
        return 0
    mol_col = _pick_col(df, ["molecule", "candidate", "mss_candidate"])
    freq_col = _pick_col(df, ["top1_freq", "frequency", "freq"])
    if mol_col is None or freq_col is None:
        return 0
    n = 0
    for _, r in df.iterrows():
        try:
            v = float(r[freq_col])
        except Exception:
            continue
        if v < 0.05:
            continue
        cond = str(r.get("label", r.get("condition", "")))
        _emit(events,
              dataset=meta["phase"], pilot_group=meta["pg"],
              sample_type=meta["samp"], condition_family=meta["cf"],
              condition_A=cond, condition_B="top_mss_hit",
              comparison_type="mss_topK_frequency",
              direction="up", effect_size=v,
              metric_type="top1_freq", metric_value=v,
              mss_candidate=str(r[mol_col]),
              confidence_tier=("STRONG" if v >= 0.30
                                else "MODERATE" if v >= 0.15
                                else "WEAK"),
              source_file=str(p))
        n += 1
    return n


def parse_cross_pilot_synthesis(p: Path, events: list, meta: dict) -> int:
    """Parse axis-level cross-pilot summary tables (incl. wide consensus tables
    where columns themselves encode pilot/comparison + family-row gives axis)."""
    df = _read_safe(p)
    if df is None or df.empty:
        return 0
    axis_col = _pick_col(df, ["axis", "bsv_axis", "family"])
    if axis_col is None:
        return 0
    n = 0
    eff_col = _pick_col(df, ["mean_effect", "effect", "mean_d", "transfer_score",
                              "cohens_d", "abs_d", "max_abs_d_disease_vs_control",
                              "liver_mean_d"])
    # Wide-form parser: any column name matching `*_d_sn` or `*_vs_*` with float
    # values is treated as one comparison-specific effect.
    wide_cols = [c for c in df.columns
                 if (c.endswith("_d_sn") or "_vs_" in c)
                 and pd.api.types.is_numeric_dtype(df[c])]
    for _, r in df.iterrows():
        ax = str(r[axis_col]).strip()
        if ax not in BIOLOGY_AXES_V11:
            continue
        # Wide-form: emit one event per matching comparison column
        for c in wide_cols:
            try:
                v = float(r[c])
            except Exception:
                continue
            if pd.isna(v):
                continue
            _emit(events,
                  dataset=meta["phase"], pilot_group=meta["pg"],
                  sample_type=meta["samp"], condition_family=meta["cf"],
                  condition_A=c, condition_B="",
                  comparison_type="cross_pilot_axis_wide",
                  bsv_axis=ax, bsv_axis_name=AXIS_NAMES.get(ax, ""),
                  direction=_direction(v, "up", "down", 0.20),
                  effect_size=v, metric_type="cohens_d_wide",
                  metric_value=v,
                  confidence_tier=("STRONG" if abs(v) >= 0.50
                                    else "MODERATE" if abs(v) >= 0.20
                                    else "WEAK"),
                  source_file=str(p))
            n += 1
        # Long-form fallback: emit a single event with eff_col if present
        if eff_col is not None and eff_col not in wide_cols:
            try:
                v = float(r[eff_col])
                _emit(events,
                      dataset=meta["phase"], pilot_group=meta["pg"],
                      sample_type=meta["samp"], condition_family=meta["cf"],
                      comparison_type="cross_pilot_axis",
                      bsv_axis=ax, bsv_axis_name=AXIS_NAMES.get(ax, ""),
                      direction=_direction(v, "up", "down", 0.20),
                      effect_size=v, metric_type=eff_col, metric_value=v,
                      confidence_tier="MODERATE",
                      source_file=str(p))
                n += 1
            except Exception:
                pass
    return n


def parse_calibration_family_selectivity(p: Path, events: list, meta: dict) -> int:
    """Parse calibration_family_selectivity_v1.csv style tables."""
    df = _read_safe(p)
    if df is None or df.empty:
        return 0
    fam_col = _pick_col(df, ["family", "axis", "bsv_axis"])
    top1_col = _pick_col(df, ["top1", "top1_rate", "top1_freq"])
    if fam_col is None or top1_col is None:
        return 0
    n = 0
    for _, r in df.iterrows():
        ax = str(r[fam_col]).strip()
        if ax not in BIOLOGY_AXES_V11:
            continue
        try:
            v = float(r[top1_col])
        except Exception:
            continue
        if v < 0.05:
            continue
        _emit(events,
              dataset=meta["phase"], pilot_group=meta["pg"],
              sample_type=meta["samp"], condition_family=meta["cf"],
              condition_A="calibration_self", condition_B="",
              comparison_type="calibration_family_selectivity",
              bsv_axis=ax, bsv_axis_name=AXIS_NAMES.get(ax, ""),
              direction="up", effect_size=v,
              metric_type="top1_rate", metric_value=v,
              confidence_tier=("STRONG" if v >= 0.70
                                else "MODERATE" if v >= 0.40
                                else "WEAK"),
              source_file=str(p))
        n += 1
    return n


def parse_cohort_state_map(p: Path, events: list, meta: dict) -> int:
    """Parse cohort/condition state-map tables that encode elevated/depleted
    families as semicolon strings."""
    df = _read_safe(p)
    if df is None or df.empty:
        return 0
    elev_col = _pick_col(df, ["elevated_top3 (sumnorm)",
                                "elevated_top3_families",
                                "elevated_families_d_ge_03"])
    depl_col = _pick_col(df, ["depleted_top3 (sumnorm)",
                                "depleted_top3_families",
                                "depleted_families_d_le_neg03"])
    if elev_col is None and depl_col is None:
        return 0
    cohort_col = _pick_col(df, ["cohort", "comparison", "comparison_label",
                                  "class"])
    sub_col = _pick_col(df, ["substrate"])
    n = 0
    for _, r in df.iterrows():
        cond = str(r[cohort_col]) if cohort_col else ""
        substrate = str(r[sub_col]) if sub_col and pd.notna(r.get(sub_col)) else ""
        for col, direction in [(elev_col, "up"), (depl_col, "down")]:
            if col is None: continue
            txt = r.get(col)
            if pd.isna(txt) or not str(txt).strip(): continue
            for tok in re.split(r"[;,]", str(txt)):
                tok = tok.strip()
                m = re.search(r"\b(G\d{2})\b", tok)
                if not m: continue
                ax = m.group(1)
                if ax not in BIOLOGY_AXES_V11: continue
                _emit(events,
                      dataset=meta["phase"], pilot_group=meta["pg"],
                      sample_type=meta["samp"], condition_family=meta["cf"],
                      condition_A=cond + (f"|substrate={substrate}" if substrate else ""),
                      condition_B="state_map",
                      comparison_type="cohort_state_map",
                      bsv_axis=ax, bsv_axis_name=AXIS_NAMES.get(ax, ""),
                      direction=direction,
                      effect_size=("" if direction == "stable" else (1 if direction == "up" else -1)),
                      metric_type="state_map_token", metric_value="",
                      confidence_tier="MODERATE",
                      source_file=str(p))
                n += 1
    return n


def stage2_evidence_events(inv: pd.DataFrame) -> pd.DataFrame:
    print("[stage 2] evidence-event extraction")
    events: list[dict] = []
    parsed = Counter()

    for _, row in inv[inv.usable_for_graph & (inv.artifact_type == "csv")].iterrows():
        p = Path(row["file_path"])
        meta = {"phase": row["dataset"], "pg": row["pilot_group"],
                "samp": row["sample_type"], "cf": row["condition_family"]}
        nm = p.name.lower()
        n = 0
        # Hand-tuned per-pattern dispatch (most specific first)
        if "binary" in nm and "effect" in nm:
            meta.update(comp="binary_effect", condA="OWD", condB="NWD")
            n = parse_binary_effects(p, events, meta)
        elif ("family_effect_sizes_vs_control" in nm or
                "pairwise_family_effect_sizes" in nm):
            meta.update(comp="serum_family_effect", condA="vs_control", condB="")
            n = parse_binary_effects(p, events, meta)
        elif "covid_liver_cross_disease_effects" in nm:
            meta.update(comp="cross_disease_effect", condA="cohort", condB="")
            n = parse_binary_effects(p, events, meta)
        elif "covid_liver_shared_vs_specific_axes" in nm:
            n = parse_cross_pilot_synthesis(p, events, meta)
        elif "cross_pilot_family_consensus" in nm:
            n = parse_cross_pilot_synthesis(p, events, meta)
        elif "cross_pilot_harmonized_effect_sizes" in nm:
            meta.update(comp="cross_pilot_harmonized", condA="comparison",
                          condB="")
            n = parse_binary_effects(p, events, meta)
        elif "cross_pilot_robust_vs_caution_signals" in nm:
            n = parse_cross_pilot_synthesis(p, events, meta)
        elif "effect_sizes" in nm or "effect_size" in nm:
            meta.update(comp="binary_effect", condA="comparison", condB="")
            n = parse_binary_effects(p, events, meta)
        elif "dose_response" in nm or "monotonicity" in nm:
            n = parse_dose_response(p, events, meta)
        elif "axis_rank_comparison" in nm or "rank_comparison" in nm:
            n = parse_axis_rank_comparison(p, events, meta)
        elif "classifier_feature_importance" in nm or "feature_importance" in nm:
            n = parse_classifier_importance(p, events, meta)
        elif "mss_top_hits" in nm or "top_hits_by_condition" in nm:
            n = parse_mss_top_hits(p, events, meta)
        elif "calibration_family_selectivity" in nm:
            n = parse_calibration_family_selectivity(p, events, meta)
        elif ("cross_pilot_cohort_state_map" in nm or
                "biochemical_state_patterns" in nm or
                "covid_liver_state_map" in nm):
            n = parse_cohort_state_map(p, events, meta)
        elif "trajectory_scores" in nm:
            # trajectory_scores has Healthy_mean / COVID_mean / delta_C_minus_H
            # use generic binary_effects with delta column
            meta.update(comp="trajectory_delta", condA="vs_Healthy", condB="")
            n = parse_binary_effects(p, events, meta)
        elif "per_family_univariate_auc" in nm:
            df0 = _read_safe(p)
            if df0 is not None and not df0.empty:
                fam_col = _pick_col(df0, ["family"])
                auc_col = _pick_col(df0, ["univariate_AUC"])
                comp_col = _pick_col(df0, ["comparison"])
                if fam_col and auc_col:
                    for _, rr in df0.iterrows():
                        ax = str(rr[fam_col]).strip()
                        if ax not in BIOLOGY_AXES_V11: continue
                        try: v = float(rr[auc_col])
                        except: continue
                        # AUC > 0.5 → axis discriminates
                        d = (v - 0.5) * 2  # crude effect proxy
                        _emit(events, dataset=meta["phase"], pilot_group=meta["pg"],
                              sample_type=meta["samp"], condition_family=meta["cf"],
                              condition_A=str(rr.get(comp_col, "")), condition_B="AUC",
                              comparison_type="univariate_auc",
                              bsv_axis=ax, bsv_axis_name=AXIS_NAMES.get(ax, ""),
                              direction="up" if d > 0.05 else "stable",
                              effect_size=d, metric_type="univariate_AUC",
                              metric_value=v,
                              confidence_tier=("STRONG" if v >= 0.75
                                                else "MODERATE" if v >= 0.60
                                                else "WEAK"),
                              source_file=str(p))
                        n += 1
        elif "scorecard" in nm:
            n = parse_binary_effects(p, events, meta)
        if n:
            parsed[nm] += n

    df = pd.DataFrame(events)
    df.to_csv(T / "gaira_evidence_events_long.csv", index=False)
    print(f"  emitted {len(df)} evidence events from "
          f"{len(parsed)} unique table-name patterns")
    return df


# ─────────────────────────────────────────────────────────────────────────
# Stage 3 — graph nodes + edges
# ─────────────────────────────────────────────────────────────────────────

def stage3_graph(events: pd.DataFrame, inv: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("[stage 3] graph nodes + edges")
    nodes = []
    edges = []

    if events.empty:
        return pd.DataFrame(), pd.DataFrame()

    datasets = events["dataset"].dropna().unique()
    sample_types = events["sample_type"].dropna().unique()
    cond_families = events["condition_family"].dropna().unique()
    axes = sorted(set(BIOLOGY_AXES_V11) & set(events["bsv_axis"].dropna().unique()))
    mss_candidates = sorted(set(c for c in events["mss_candidate"].dropna().unique() if c))

    for d in datasets:
        meta = events[events.dataset == d].iloc[0].to_dict()
        nodes.append({
            "node_id": f"DATASET::{d}",
            "node_type": "DATASET", "label": d,
            "sample_type": meta["sample_type"],
            "family": meta["condition_family"],
            "size_metric": int((events.dataset == d).sum()),
            "confidence": "",
            "metadata_json": json.dumps({"pilot_group": meta["pilot_group"]})})
    for s in sample_types:
        nodes.append({"node_id": f"SAMPLE_TYPE::{s}", "node_type": "SAMPLE_TYPE",
                      "label": s, "sample_type": s, "family": "",
                      "size_metric": int((events.sample_type == s).sum()),
                      "confidence": "", "metadata_json": "{}"})
    for cf in cond_families:
        nodes.append({"node_id": f"CONDITION::{cf}", "node_type": "CONDITION",
                      "label": cf, "sample_type": "", "family": cf,
                      "size_metric": int((events.condition_family == cf).sum()),
                      "confidence": "", "metadata_json": "{}"})
    for ax in axes:
        nodes.append({"node_id": f"BSV_AXIS::{ax}", "node_type": "BSV_AXIS",
                      "label": f"{ax} · {AXIS_NAMES.get(ax,'')}",
                      "sample_type": "", "family": "",
                      "size_metric": int((events.bsv_axis == ax).sum()),
                      "confidence": "", "metadata_json": "{}"})
    for m in mss_candidates:
        nodes.append({"node_id": f"MSS::{m}", "node_type": "MSS_CANDIDATE",
                      "label": m, "sample_type": "", "family": "",
                      "size_metric": int((events.mss_candidate == m).sum()),
                      "confidence": "", "metadata_json": "{}"})

    # Edges: dataset → axis (effect) and dataset → mss + axis ↔ axis (collision proxy)
    for (d, ax), sub in events.groupby(["dataset", "bsv_axis"]):
        if not ax or ax not in BIOLOGY_AXES_V11:
            continue
        dirs = sub["direction"].dropna().value_counts()
        dom_dir = dirs.idxmax() if len(dirs) else "ambiguous"
        try:
            effs = pd.to_numeric(sub["effect_size"], errors="coerce").dropna()
            mean_eff = float(effs.mean()) if len(effs) else 0.0
            max_abs_eff = float(effs.abs().max()) if len(effs) else 0.0
        except Exception:
            mean_eff, max_abs_eff = 0.0, 0.0
        edge_type = ("enriched_in" if dom_dir == "up"
                     else "depleted_in" if dom_dir == "down"
                     else "stable_in")
        edges.append({
            "source": f"DATASET::{d}", "target": f"BSV_AXIS::{ax}",
            "edge_type": edge_type, "weight": round(max_abs_eff, 3),
            "direction": dom_dir, "confidence": "",
            "evidence_count": len(sub),
            "source_files": ";".join(sorted(set(sub["source_file"].dropna()))[:3])})
    for (d, m), sub in events.groupby(["dataset", "mss_candidate"]):
        if not m:
            continue
        try:
            effs = pd.to_numeric(sub["effect_size"], errors="coerce").dropna()
            w = float(effs.abs().max()) if len(effs) else 0.0
        except Exception:
            w = 0.0
        edges.append({
            "source": f"DATASET::{d}", "target": f"MSS::{m}",
            "edge_type": "candidate_only" if "biofluid" in d else "supports",
            "weight": round(w, 3), "direction": "",
            "confidence": "", "evidence_count": len(sub),
            "source_files": ";".join(sorted(set(sub["source_file"].dropna()))[:3])})
    # Sample-type → axis recurrence
    for (s, ax), sub in events.groupby(["sample_type", "bsv_axis"]):
        if not ax or ax not in BIOLOGY_AXES_V11 or not s:
            continue
        n_datasets = sub["dataset"].nunique()
        edges.append({
            "source": f"SAMPLE_TYPE::{s}", "target": f"BSV_AXIS::{ax}",
            "edge_type": "axis_used_by_sample_type",
            "weight": int(n_datasets), "direction": "",
            "confidence": "", "evidence_count": len(sub),
            "source_files": ""})

    nodes_df = pd.DataFrame(nodes)
    edges_df = pd.DataFrame(edges)
    nodes_df.to_csv(T / "context_graph_nodes.csv", index=False)
    edges_df.to_csv(T / "context_graph_edges.csv", index=False)
    print(f"  nodes={len(nodes_df)} edges={len(edges_df)}")
    return nodes_df, edges_df


# ─────────────────────────────────────────────────────────────────────────
# Stage 4 — recurrence + transfer + caveat tables
# ─────────────────────────────────────────────────────────────────────────

def stage4_recurrence(events: pd.DataFrame) -> dict[str, pd.DataFrame]:
    print("[stage 4] recurrence + transfer + caveats")
    out = {}
    if events.empty:
        return out

    # Sample-type axis recurrence (datasets per (sample_type, axis))
    rec = (events.dropna(subset=["bsv_axis"])
           .groupby(["sample_type", "bsv_axis"])["dataset"].nunique()
           .reset_index(name="n_datasets"))
    rec.to_csv(T / "sample_type_axis_recurrence.csv", index=False)
    out["sample_type_axis"] = rec

    # Condition-family axis recurrence + dominant direction
    rows = []
    for (cf, ax), sub in events.groupby(["condition_family", "bsv_axis"]):
        if ax not in BIOLOGY_AXES_V11:
            continue
        dirs = sub["direction"].value_counts().to_dict()
        dom_dir = max(dirs, key=dirs.get) if dirs else "ambiguous"
        try:
            effs = pd.to_numeric(sub["effect_size"], errors="coerce").dropna()
            mean_eff = float(effs.mean()) if len(effs) else 0.0
        except Exception:
            mean_eff = 0.0
        rows.append({"condition_family": cf, "bsv_axis": ax,
                     "n_events": len(sub), "n_datasets": sub["dataset"].nunique(),
                     "dom_direction": dom_dir,
                     "mean_effect": round(mean_eff, 3)})
    cf_axis = pd.DataFrame(rows)
    cf_axis.to_csv(T / "condition_axis_motif_recurrence.csv", index=False)
    out["condition_family_axis"] = cf_axis

    # Per-axis neighborhood summary
    rows = []
    for ax in BIOLOGY_AXES_V11:
        sub = events[events.bsv_axis == ax]
        if sub.empty:
            continue
        dom_dir = (sub["direction"].value_counts().idxmax()
                   if len(sub["direction"].dropna()) else "ambiguous")
        rows.append({
            "bsv_axis": ax, "axis_name": AXIS_NAMES.get(ax, ""),
            "n_events": len(sub),
            "n_datasets": sub["dataset"].nunique(),
            "n_sample_types": sub["sample_type"].nunique(),
            "n_condition_families": sub["condition_family"].nunique(),
            "dom_direction": dom_dir,
            "datasets": ";".join(sorted(sub["dataset"].dropna().unique())[:8]),
            "top_mss_candidates": ";".join(
                sub["mss_candidate"].dropna().value_counts().head(5).index.tolist()),
        })
    axis_nb = pd.DataFrame(rows)
    axis_nb.to_csv(T / "axis_neighborhood_summary.csv", index=False)
    out["axis_neighborhood"] = axis_nb

    # MSS transfer classification
    mss_rows = []
    for m, sub in events.groupby("mss_candidate"):
        if not m:
            continue
        n_datasets = sub["dataset"].nunique()
        n_sample_types = sub["sample_type"].nunique()
        sample_dist = sub["sample_type"].value_counts(normalize=True).to_dict()
        dirs = sub["direction"].value_counts(normalize=True).to_dict()
        dom_dir = max(dirs, key=dirs.get) if dirs else "ambiguous"
        dir_consistency = max(dirs.values()) if dirs else 0.0
        ev_n = len(sub)
        # Classify
        if n_datasets >= 2 and dir_consistency >= 0.65:
            cls = "TRANSFERABLE"
        elif n_sample_types == 1 and ev_n >= 3:
            cls = "SAMPLE_TYPE_SPECIFIC"
        elif n_datasets >= 2 and dir_consistency < 0.50:
            cls = "SUBSTRATE_LOCKED"
        elif "pure_Raman" in sample_dist and len(sample_dist) == 1:
            cls = "PURE_ONLY"
        else:
            cls = "CANDIDATE_ONLY"
        mss_rows.append({
            "mss_candidate": m, "n_datasets": n_datasets,
            "n_sample_types": n_sample_types, "n_events": ev_n,
            "dom_direction": dom_dir,
            "direction_consistency": round(dir_consistency, 2),
            "sample_type_dist": json.dumps(
                {k: round(v, 2) for k, v in sample_dist.items()}),
            "classification": cls,
            "datasets": ";".join(sorted(sub["dataset"].dropna().unique())[:6]),
        })
    mss_df = pd.DataFrame(mss_rows).sort_values(
        ["n_datasets", "n_events"], ascending=False)
    mss_df.to_csv(T / "mss_transfer_classification.csv", index=False)
    out["mss_transfer"] = mss_df

    return out


# ─────────────────────────────────────────────────────────────────────────
# Stage 5 — caveat extraction (from MD reports/audits)
# ─────────────────────────────────────────────────────────────────────────

CAVEAT_PATTERNS = [
    ("substrate_sensitive", r"substrate.{0,30}(sensitive|locked|caveat|specific)"),
    ("collision_prone", r"(collision[- ]prone|paper.?band.{0,40}collision|"
                         r"collision[s]? in)"),
    ("candidate_only_biofluid", r"(candidate[- ]level|candidate only|"
                                  r"not.{0,10}molecule.{0,10}identif)"),
    ("pure_only_mss", r"(pure[- ]?only|pure[- ]?cohort|pure context)"),
    ("qc_failure", r"(QC[_ ]fail|quality control fail|excluded|"
                    r"qc[ _]caution|qc[ _]warning)"),
    ("regression_failed_trajectory_strong",
       r"(regression.{0,30}fail|R²[_ ]?neg|cohort.?level.{0,30}strong)"),
    ("patient_level_only",
       r"(patient[- ]level|aggregation INSIDE|aggregation cleanup)"),
    ("low_purity_cluster",
       r"(low purity|cluster purity (?:0\.[0-2]|<\s*0\.3))"),
]


def stage5_caveats(inv: pd.DataFrame) -> pd.DataFrame:
    print("[stage 5] caveat extraction")
    rows = []
    md_files = inv[(inv.artifact_type == "md") & inv.usable_for_graph]
    for _, row in md_files.iterrows():
        try:
            text = Path(row["file_path"]).read_text(errors="ignore")
        except Exception:
            continue
        for cv_id, pat in CAVEAT_PATTERNS:
            hits = len(re.findall(pat, text, flags=re.I))
            if hits == 0:
                continue
            rows.append({
                "dataset": row["dataset"], "sample_type": row["sample_type"],
                "condition_family": row["condition_family"],
                "caveat_id": cv_id, "n_mentions": hits,
                "source_file": row["file_path"],
            })
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=["dataset", "sample_type", "condition_family",
                                     "caveat_id", "n_mentions", "source_file"])
    df.to_csv(T / "caveat_recurrence.csv", index=False)
    print(f"  caveats found in {df['dataset'].nunique() if not df.empty else 0} datasets")
    return df


# ─────────────────────────────────────────────────────────────────────────
# Stage 6 — embeddings + clustering
# ─────────────────────────────────────────────────────────────────────────

def stage6_embeddings(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("[stage 6] embeddings + clustering")
    if events.empty:
        return pd.DataFrame(), pd.DataFrame()

    # BSV effect vector per dataset (11-dim) — mean signed effect by axis
    rows = []
    for d, sub in events.groupby("dataset"):
        vec = {}
        for ax in BIOLOGY_AXES_V11:
            ax_sub = sub[sub.bsv_axis == ax]
            try:
                effs = pd.to_numeric(ax_sub["effect_size"], errors="coerce").dropna()
                vec[ax] = float(effs.mean()) if len(effs) else 0.0
            except Exception:
                vec[ax] = 0.0
        meta = sub.iloc[0]
        vec["dataset"] = d
        vec["sample_type"] = meta["sample_type"]
        vec["pilot_group"] = meta["pilot_group"]
        vec["condition_family"] = meta["condition_family"]
        rows.append(vec)
    bsv_df = pd.DataFrame(rows)

    if len(bsv_df) < 3:
        bsv_df.to_csv(T / "context_dataset_bsv_features.csv", index=False)
        return bsv_df, pd.DataFrame()

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import AgglomerativeClustering

    X = bsv_df[BIOLOGY_AXES_V11].fillna(0.0).values
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=42).fit_transform(Xs)
    bsv_df["pca_1"] = pca[:, 0]
    bsv_df["pca_2"] = pca[:, 1]

    # Try UMAP, fall back to t-SNE
    try:
        import umap  # noqa: F401
        reducer = umap.UMAP(n_components=2, random_state=42,
                            n_neighbors=min(15, max(2, len(bsv_df) - 1)),
                            min_dist=0.10)
        emb = reducer.fit_transform(Xs)
        bsv_df["umap_1"], bsv_df["umap_2"] = emb[:, 0], emb[:, 1]
    except Exception:
        try:
            from sklearn.manifold import TSNE
            tsne = TSNE(n_components=2, perplexity=min(8, max(3, len(bsv_df) // 3)),
                        random_state=42, init="pca").fit_transform(Xs)
            bsv_df["umap_1"], bsv_df["umap_2"] = tsne[:, 0], tsne[:, 1]
        except Exception:
            bsv_df["umap_1"], bsv_df["umap_2"] = pca[:, 0], pca[:, 1]

    # Hierarchical clustering
    n_clusters = max(2, min(8, len(bsv_df) // 3))
    try:
        agg = AgglomerativeClustering(n_clusters=n_clusters).fit(Xs)
        bsv_df["cluster_id"] = agg.labels_
    except Exception:
        bsv_df["cluster_id"] = 0

    bsv_df.to_csv(T / "context_dataset_bsv_features.csv", index=False)
    bsv_df[["dataset", "sample_type", "pilot_group", "condition_family",
             "cluster_id", "pca_1", "pca_2", "umap_1", "umap_2"]].to_csv(
        T / "context_cluster_assignments.csv", index=False)
    return bsv_df, pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────
# Stage 7 — quantitative metrics
# ─────────────────────────────────────────────────────────────────────────

def stage7_metrics(events: pd.DataFrame, mss_transfer: pd.DataFrame) -> dict:
    print("[stage 7] quantitative metrics")
    out = {}

    if events.empty:
        return out

    # Axis transfer scores
    rows = []
    for ax in BIOLOGY_AXES_V11:
        sub = events[events.bsv_axis == ax]
        if sub.empty:
            continue
        n_datasets = sub["dataset"].nunique()
        n_sample_types = sub["sample_type"].nunique()
        dirs = sub["direction"].value_counts(normalize=True)
        dom = dirs.idxmax() if len(dirs) else "ambiguous"
        consistency = float(dirs.max()) if len(dirs) else 0.0
        try:
            effs = pd.to_numeric(sub["effect_size"], errors="coerce").dropna()
            mean_abs = float(effs.abs().mean()) if len(effs) else 0.0
        except Exception:
            mean_abs = 0.0
        score = n_datasets * consistency * (mean_abs + 0.1)
        rows.append({
            "bsv_axis": ax, "axis_name": AXIS_NAMES.get(ax, ""),
            "n_datasets": n_datasets, "n_sample_types": n_sample_types,
            "dom_direction": dom,
            "direction_consistency": round(consistency, 2),
            "mean_abs_effect": round(mean_abs, 3),
            "axis_transfer_score": round(score, 3),
        })
    axis_score = pd.DataFrame(rows)
    if "axis_transfer_score" in axis_score.columns:
        axis_score = axis_score.sort_values("axis_transfer_score",
                                            ascending=False)
    axis_score.to_csv(T / "axis_transfer_scores.csv", index=False)
    out["axis_transfer"] = axis_score

    # MSS scores already in mss_transfer; copy with explicit name
    if mss_transfer is not None and not mss_transfer.empty:
        mss_transfer.copy().rename(columns={
            "n_datasets": "n_datasets",
            "direction_consistency": "direction_consistency",
        }).to_csv(T / "mss_transfer_scores.csv", index=False)

    # Context-dependence score per axis (variance of effect explained by sample_type)
    rows = []
    for ax in BIOLOGY_AXES_V11:
        sub = events[events.bsv_axis == ax]
        if sub.empty:
            continue
        try:
            sub2 = sub.copy()
            sub2["e"] = pd.to_numeric(sub2["effect_size"], errors="coerce")
            sub2 = sub2.dropna(subset=["e"])
            if len(sub2) < 3:
                continue
            grand = sub2["e"].mean()
            ss_total = float(((sub2["e"] - grand) ** 2).sum())
            ss_between = 0.0
            for st, gp in sub2.groupby("sample_type"):
                gm = gp["e"].mean()
                ss_between += len(gp) * (gm - grand) ** 2
            r2 = ss_between / ss_total if ss_total > 0 else 0.0
            rows.append({
                "bsv_axis": ax, "axis_name": AXIS_NAMES.get(ax, ""),
                "n_events": len(sub2),
                "variance_explained_by_sample_type": round(r2, 3),
            })
        except Exception:
            continue
    ctx_dep = pd.DataFrame(rows)
    ctx_dep.to_csv(T / "context_dependence_scores.csv", index=False)
    out["context_dependence"] = ctx_dep

    # Emergent-behaviour metrics per (axis, condition_family)
    rows = []
    for (ax, cf), sub in events.groupby(["bsv_axis", "condition_family"]):
        if ax not in BIOLOGY_AXES_V11:
            continue
        n_d = sub["dataset"].nunique()
        if n_d < 2:
            continue
        try:
            effs = pd.to_numeric(sub["effect_size"], errors="coerce").dropna()
            mean_abs = float(effs.abs().mean()) if len(effs) else 0.0
        except Exception:
            mean_abs = 0.0
        dirs = sub["direction"].value_counts(normalize=True)
        cons = float(dirs.max()) if len(dirs) else 0.0
        score = n_d * cons * (mean_abs + 0.1)
        rows.append({
            "bsv_axis": ax, "condition_family": cf,
            "n_datasets": n_d, "mean_abs_effect": round(mean_abs, 3),
            "direction_consistency": round(cons, 2),
            "emergence_score": round(score, 3),
        })
    em = pd.DataFrame(rows)
    if "emergence_score" in em.columns:
        em = em.sort_values("emergence_score", ascending=False)
    em.to_csv(T / "emergent_behavior_metrics.csv", index=False)
    out["emergent"] = em

    return out


# ─────────────────────────────────────────────────────────────────────────
# Stage 8 — top emergent findings
# ─────────────────────────────────────────────────────────────────────────

def stage8_findings(events: pd.DataFrame, axis_score: pd.DataFrame,
                    mss_transfer: pd.DataFrame, em: pd.DataFrame,
                    caveats: pd.DataFrame) -> pd.DataFrame:
    print("[stage 8] ranked emergent findings")
    findings = []

    # Top 5 axes by transfer score
    for i, r in axis_score.head(5).iterrows():
        ax = r["bsv_axis"]
        sub = events[events.bsv_axis == ax]
        ds = ";".join(sorted(sub["dataset"].dropna().unique())[:5])
        mss = ";".join(sub["mss_candidate"].dropna().value_counts().head(5).index)
        findings.append({
            "finding_id": f"F{len(findings)+1:02d}",
            "finding_title": f"{ax} ({r['axis_name']}) recurs across "
                              f"{int(r['n_datasets'])} datasets · "
                              f"dominant {r['dom_direction']}",
            "evidence_type": "axis_transfer",
            "datasets_involved": ds,
            "axes_involved": ax,
            "mss_candidates": mss,
            "metric_summary": (f"transfer_score={r['axis_transfer_score']} · "
                               f"mean|d|={r['mean_abs_effect']} · "
                               f"consistency={r['direction_consistency']}"),
            "confidence": ("STRONG" if r["axis_transfer_score"] >= 1.0
                           else "MODERATE" if r["axis_transfer_score"] >= 0.4
                           else "WEAK"),
            "caveats": "candidate-level in biofluids; substrate effects possible",
            "recommended_demo_figure": "context_graph_global / sample_type_axis_recurrence",
        })

    # Top transferable MSS candidates
    if mss_transfer is not None and not mss_transfer.empty:
        top_mss = mss_transfer[mss_transfer.classification == "TRANSFERABLE"].head(5)
        for _, r in top_mss.iterrows():
            findings.append({
                "finding_id": f"F{len(findings)+1:02d}",
                "finding_title": (f"MSS candidate `{r['mss_candidate']}` transfers "
                                   f"across {int(r['n_datasets'])} datasets · "
                                   f"dom {r['dom_direction']}"),
                "evidence_type": "mss_transfer",
                "datasets_involved": r["datasets"],
                "axes_involved": "",
                "mss_candidates": r["mss_candidate"],
                "metric_summary": (f"n_events={int(r['n_events'])} · "
                                    f"sample_type_spread={int(r['n_sample_types'])} · "
                                    f"consistency={r['direction_consistency']}"),
                "confidence": "MODERATE",
                "caveats": "candidate-level evidence in biofluids",
                "recommended_demo_figure": "mss_transfer_graph",
            })

    # Top emergent (axis × condition_family)
    for _, r in em.head(8).iterrows():
        if r["emergence_score"] < 0.4:
            break
        findings.append({
            "finding_id": f"F{len(findings)+1:02d}",
            "finding_title": (f"{r['bsv_axis']} reappears in `{r['condition_family']}` "
                               f"across {int(r['n_datasets'])} datasets"),
            "evidence_type": "axis_x_condition_emergence",
            "datasets_involved": "",
            "axes_involved": r["bsv_axis"],
            "mss_candidates": "",
            "metric_summary": (f"emergence_score={r['emergence_score']} · "
                               f"mean|d|={r['mean_abs_effect']} · "
                               f"consistency={r['direction_consistency']}"),
            "confidence": ("STRONG" if r["emergence_score"] >= 1.0
                           else "MODERATE" if r["emergence_score"] >= 0.5
                           else "WEAK"),
            "caveats": "verify against substrate / cohort effects",
            "recommended_demo_figure": "condition_axis_motif_recurrence",
        })

    # Caveat findings
    if caveats is not None and not caveats.empty:
        cav_summary = (caveats.groupby("caveat_id")["dataset"]
                        .nunique().sort_values(ascending=False).head(5))
        for cv, n in cav_summary.items():
            findings.append({
                "finding_id": f"F{len(findings)+1:02d}",
                "finding_title": f"Caveat `{cv}` flagged across {int(n)} datasets",
                "evidence_type": "caveat",
                "datasets_involved": ";".join(
                    caveats[caveats.caveat_id == cv]["dataset"].unique()[:5]),
                "axes_involved": "", "mss_candidates": "",
                "metric_summary": f"n_datasets_with_caveat={int(n)}",
                "confidence": "MODERATE",
                "caveats": "this IS the caveat",
                "recommended_demo_figure": "context_graph_caveats",
            })

    df = pd.DataFrame(findings)
    df.to_csv(T / "top_emergent_findings.csv", index=False)
    print(f"  produced {len(df)} ranked findings")
    return df


# ─────────────────────────────────────────────────────────────────────────
# Stage 9 — figures
# ─────────────────────────────────────────────────────────────────────────

def _heatmap_png(values: np.ndarray, x_labels, y_labels, title: str,
                 path: Path, cmap: str = "viridis") -> None:
    fig, ax = plt.subplots(figsize=(max(8, len(x_labels) * 0.45),
                                       max(5, len(y_labels) * 0.30)))
    im = ax.imshow(values, aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(x_labels))); ax.set_xticklabels(x_labels, rotation=45,
                                                              ha="right", fontsize=8)
    ax.set_yticks(range(len(y_labels))); ax.set_yticklabels(y_labels, fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.025)
    ax.set_title(title, fontsize=11)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def stage9_figures(events: pd.DataFrame,
                   nodes: pd.DataFrame, edges: pd.DataFrame,
                   sample_axis: pd.DataFrame, cf_axis: pd.DataFrame,
                   axis_score: pd.DataFrame,
                   mss_transfer: pd.DataFrame, ctx_dataset: pd.DataFrame,
                   caveats: pd.DataFrame) -> None:
    print("[stage 9] figures")

    # ── (1) Sample-type × axis recurrence heatmap ────────────────────────
    if not sample_axis.empty:
        pivot = (sample_axis.pivot(index="sample_type", columns="bsv_axis",
                                     values="n_datasets")
                 .reindex(columns=BIOLOGY_AXES_V11).fillna(0))
        _heatmap_png(pivot.values,
                     [f"{c} · {AXIS_NAMES.get(c,'')}" for c in pivot.columns],
                     pivot.index.tolist(),
                     "Datasets per (sample_type × BSV axis)",
                     F / "sample_type_axis_recurrence_heatmap.png")

    # ── (2) Condition-family × axis (mean effect) ────────────────────────
    if not cf_axis.empty:
        pivot = (cf_axis.pivot(index="condition_family", columns="bsv_axis",
                                 values="mean_effect")
                 .reindex(columns=BIOLOGY_AXES_V11).fillna(0))
        _heatmap_png(pivot.values,
                     [f"{c} · {AXIS_NAMES.get(c,'')}" for c in pivot.columns],
                     pivot.index.tolist(),
                     "Mean effect per (condition family × BSV axis)",
                     F / "condition_family_axis_mean_effect.png", cmap="RdBu_r")

    # ── (3) Axis transfer score bar ──────────────────────────────────────
    if not axis_score.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(axis_score["bsv_axis"],
               axis_score["axis_transfer_score"], color="#4C72B0")
        ax.set_xlabel("BSV axis"); ax.set_ylabel("transfer score")
        ax.set_title("Axis transfer score (datasets × consistency × mean|effect|)")
        ax.set_xticklabels(
            [f"{ax_id}\n{AXIS_NAMES.get(ax_id,'')[:14]}"
             for ax_id in axis_score["bsv_axis"]], fontsize=8)
        for i, v in enumerate(axis_score["axis_transfer_score"]):
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
        plt.tight_layout()
        fig.savefig(F / "axis_transfer_score_bar.png", dpi=150)
        plt.close(fig)

    # ── (4) Context UMAP/PCA scatter ─────────────────────────────────────
    if not ctx_dataset.empty and "umap_1" in ctx_dataset.columns:
        fig, axarr = plt.subplots(1, 2, figsize=(14, 6))
        for ax, xcol, ycol, title in [
            (axarr[0], "umap_1", "umap_2", "Context UMAP · datasets"),
            (axarr[1], "pca_1", "pca_2", "Context PCA · datasets")]:
            for st, sub in ctx_dataset.groupby("sample_type"):
                ax.scatter(sub[xcol], sub[ycol], label=st, s=70, alpha=0.85)
                for _, r in sub.iterrows():
                    ax.annotate(r["dataset"][:24], (r[xcol], r[ycol]),
                                 fontsize=6, alpha=0.7)
            ax.set_title(title); ax.set_xlabel(xcol); ax.set_ylabel(ycol)
            ax.legend(fontsize=8, loc="best")
        plt.tight_layout()
        fig.savefig(F / "context_embedding_dataset_umap_pca.png", dpi=150)
        plt.close(fig)

    # ── (5) MSS transfer classification stacked bar ──────────────────────
    if mss_transfer is not None and not mss_transfer.empty:
        cls_counts = mss_transfer["classification"].value_counts()
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(cls_counts.index, cls_counts.values,
               color=["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728", "#9467bd"][:len(cls_counts)])
        ax.set_ylabel("# MSS candidates")
        ax.set_title("MSS transfer classification across pilots")
        for i, v in enumerate(cls_counts.values):
            ax.text(i, v + 0.5, str(int(v)), ha="center", fontsize=10)
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        fig.savefig(F / "mss_transfer_classification_bar.png", dpi=150)
        plt.close(fig)

    # ── (6) Caveat heatmap ───────────────────────────────────────────────
    if caveats is not None and not caveats.empty:
        pivot = (caveats.pivot_table(index="dataset", columns="caveat_id",
                                       values="n_mentions", aggfunc="sum")
                 .fillna(0))
        if pivot.size > 0:
            _heatmap_png(pivot.values, pivot.columns.tolist(),
                         pivot.index.tolist(),
                         "Caveat mentions per dataset",
                         F / "caveat_recurrence_heatmap.png", cmap="OrRd")

    # ── (7) Plotly interactive global graph (subset for readability) ────
    if not nodes.empty and not edges.empty:
        # Pick top edges by weight to keep readable
        edges_view = edges.copy()
        edges_view["w"] = pd.to_numeric(edges_view["weight"],
                                          errors="coerce").fillna(0)
        ds_axis_edges = edges_view[edges_view.target.str.startswith("BSV_AXIS::")]
        ds_axis_edges = ds_axis_edges.sort_values("w", ascending=False).head(120)

        # Position layout: BSV axes on a circle, datasets clustered around them
        ax_pos = {}
        n_ax = len(BIOLOGY_AXES_V11)
        for i, ax_id in enumerate(BIOLOGY_AXES_V11):
            angle = 2 * np.pi * i / n_ax - np.pi / 2
            ax_pos[ax_id] = (1.5 * np.cos(angle), 1.5 * np.sin(angle))

        ds_pos = {}
        rng = np.random.default_rng(0)
        for d in nodes[nodes.node_type == "DATASET"]["label"]:
            # find dominant axis edge for this dataset
            d_edges = ds_axis_edges[ds_axis_edges.source == f"DATASET::{d}"]
            if not d_edges.empty:
                top_ax = d_edges.iloc[0].target.replace("BSV_AXIS::", "")
                ax_xy = ax_pos.get(top_ax, (0, 0))
                jitter = rng.normal(0, 0.20, size=2)
                ds_pos[d] = (ax_xy[0] * 0.55 + jitter[0],
                             ax_xy[1] * 0.55 + jitter[1])
            else:
                ds_pos[d] = (rng.normal(0, 0.3), rng.normal(0, 0.3))

        fig = go.Figure()
        # Edges
        for _, e in ds_axis_edges.iterrows():
            d = e.source.replace("DATASET::", "")
            ax = e.target.replace("BSV_AXIS::", "")
            x0, y0 = ds_pos.get(d, (0, 0))
            x1, y1 = ax_pos.get(ax, (0, 0))
            color = ("rgba(125, 200, 125, 0.35)" if e.direction == "up"
                     else "rgba(200, 125, 125, 0.35)" if e.direction == "down"
                     else "rgba(160, 160, 160, 0.20)")
            width = max(0.5, min(3.5, e.w * 1.5))
            fig.add_trace(go.Scatter(
                x=[x0, x1], y=[y0, y1], mode="lines",
                line=dict(color=color, width=width),
                hoverinfo="skip", showlegend=False))

        # Axis nodes
        for ax_id, (x, y) in ax_pos.items():
            fig.add_trace(go.Scatter(
                x=[x], y=[y], mode="markers+text",
                marker=dict(size=42, color="#ffa657",
                            line=dict(color="#0d1117", width=2)),
                text=[f"<b>{ax_id}</b>"],
                textposition="middle center",
                textfont=dict(color="#0d1117", size=11),
                hovertext=[f"<b>{ax_id} · {AXIS_NAMES.get(ax_id,'')}</b>"],
                hoverinfo="text", showlegend=False))

        # Dataset nodes
        sample_palette = {"EV": "#79c0ff", "serum": "#bc8cff",
                          "pure_Raman": "#7ee787", "SERS": "#56d4dd",
                          "mixed": "#d2a8ff", "unknown": "#6e7681"}
        for d, (x, y) in ds_pos.items():
            row = nodes[nodes.label == d].iloc[0] if (nodes.label == d).any() else None
            st = row["sample_type"] if row is not None else "unknown"
            color = sample_palette.get(st, "#9ecbff")
            fig.add_trace(go.Scatter(
                x=[x], y=[y], mode="markers",
                marker=dict(size=10, color=color, opacity=0.85,
                            line=dict(color="#0d1117", width=0.5)),
                hovertext=[f"<b>{d}</b><br>sample_type: {st}"],
                hoverinfo="text", showlegend=False))

        # Sample-type legend (synthetic)
        for st, color in sample_palette.items():
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(size=10, color=color),
                name=st, showlegend=True))

        fig.update_layout(
            template="plotly_dark", height=700,
            title=dict(text="Global GAIRA evidence graph · datasets ↔ BSV axes "
                              "(top 120 edges by effect)",
                       font=dict(size=13, color="#c9d1d9")),
            margin=dict(l=10, r=10, t=44, b=10),
            plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
            xaxis=dict(visible=False, range=[-2.2, 2.2]),
            yaxis=dict(visible=False, range=[-2.2, 2.2],
                       scaleanchor="x", scaleratio=1),
            legend=dict(font=dict(size=10, color="#c9d1d9"),
                        bgcolor="rgba(13,17,23,0.6)"))
        try:
            fig.write_html(str(F / "context_graph_global.html"),
                           include_plotlyjs="cdn")
            fig.write_image(str(F / "context_graph_global.png"), width=1200, height=800)
        except Exception as e:
            print(f"  plotly export issue: {e}")

    # ── (8) Plotly interactive context UMAP ─────────────────────────────
    if not ctx_dataset.empty and "umap_1" in ctx_dataset.columns:
        sample_palette = {"EV": "#79c0ff", "serum": "#bc8cff",
                          "pure_Raman": "#7ee787", "SERS": "#56d4dd",
                          "mixed": "#d2a8ff", "unknown": "#6e7681"}
        fig = go.Figure()
        for st, sub in ctx_dataset.groupby("sample_type"):
            fig.add_trace(go.Scatter(
                x=sub["umap_1"], y=sub["umap_2"], mode="markers+text",
                name=st,
                marker=dict(size=12, color=sample_palette.get(st, "#9ecbff"),
                            line=dict(color="#0d1117", width=0.5)),
                text=sub["dataset"].str[:22],
                textposition="top center",
                textfont=dict(size=8, color="#c9d1d9"),
                hovertext=[f"<b>{r['dataset']}</b><br>"
                           f"sample: {r['sample_type']} · "
                           f"family: {r['condition_family']}"
                           for _, r in sub.iterrows()],
                hoverinfo="text"))
        fig.update_layout(
            template="plotly_dark", height=620,
            title=dict(text="Context UMAP · datasets in BSV-effect space",
                       font=dict(size=12, color="#c9d1d9")),
            margin=dict(l=10, r=10, t=44, b=10),
            plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
            xaxis=dict(title="UMAP-1", gridcolor="#21262d"),
            yaxis=dict(title="UMAP-2", gridcolor="#21262d"),
            legend=dict(font=dict(size=10, color="#c9d1d9"),
                        bgcolor="rgba(13,17,23,0.6)"))
        try:
            fig.write_html(str(F / "context_embedding_dataset.html"),
                           include_plotlyjs="cdn")
        except Exception as e:
            print(f"  plotly umap export issue: {e}")


# ─────────────────────────────────────────────────────────────────────────
# Stage 10 — report + decision
# ─────────────────────────────────────────────────────────────────────────

def stage10_report(events: pd.DataFrame, axis_score: pd.DataFrame,
                   mss_transfer: pd.DataFrame, em: pd.DataFrame,
                   caveats: pd.DataFrame, ctx_dataset: pd.DataFrame,
                   findings: pd.DataFrame, sample_axis: pd.DataFrame) -> str:
    print("[stage 10] report + decision")

    # Decision logic
    n_phases = events["dataset"].nunique() if not events.empty else 0
    n_events = len(events)
    n_axes_with_evidence = events["bsv_axis"].nunique() if not events.empty else 0
    n_strong_axis = (axis_score["axis_transfer_score"] >= 1.0).sum() if not axis_score.empty else 0
    n_moderate_axis = ((axis_score["axis_transfer_score"] >= 0.4) &
                       (axis_score["axis_transfer_score"] < 1.0)).sum() if not axis_score.empty else 0
    n_transferable_mss = (mss_transfer["classification"] == "TRANSFERABLE").sum() \
        if mss_transfer is not None and not mss_transfer.empty else 0
    n_sample_types = events["sample_type"].nunique() if not events.empty else 0
    sample_clusters = ctx_dataset["sample_type"].nunique() if not ctx_dataset.empty else 0

    if n_events < 50:
        decision = "CONTEXT_GRAPH_INSUFFICIENT_ARTIFACTS"
    elif n_strong_axis >= 3 and n_transferable_mss >= 3 and n_sample_types >= 3:
        decision = "CONTEXT_GRAPH_EMERGENT_STRUCTURE_STRONG"
    elif n_strong_axis + n_moderate_axis >= 3 and n_transferable_mss >= 1:
        decision = "CONTEXT_GRAPH_EMERGENT_STRUCTURE_MODERATE"
    elif sample_clusters >= 3 and n_transferable_mss == 0:
        decision = "CONTEXT_GRAPH_SAMPLE_TYPE_STRUCTURE_ONLY"
    elif n_transferable_mss == 0 and n_strong_axis >= 2:
        decision = "CONTEXT_GRAPH_MSS_TOO_NOISY_BSV_ONLY"
    else:
        decision = "CONTEXT_GRAPH_EMERGENT_STRUCTURE_MODERATE"

    lines = [
        f"# REPORT — gaira_base_4_context_graph_discovery_v1\n",
        f"date: {datetime.now().isoformat()}",
        "",
        f"## Decision: **{decision}**",
        "",
        "## Headline numbers",
        f"- evidence events: **{n_events}** across **{n_phases}** datasets",
        f"- BSV axes with evidence: **{n_axes_with_evidence}/11**",
        f"- axes with STRONG transfer (score ≥ 1.0): **{n_strong_axis}**",
        f"- axes with MODERATE transfer (0.4–1.0): **{n_moderate_axis}**",
        f"- MSS candidates classified TRANSFERABLE: **{n_transferable_mss}**",
        f"- sample types covered: {sorted(events['sample_type'].dropna().unique().tolist()) if not events.empty else []}",
        "",
        "## Required answers",
    ]

    # Q1
    lines += ["", "### 1. What emergent biochemical structures appear across GAIRA?"]
    if not axis_score.empty:
        top3 = axis_score.head(3)
        for _, r in top3.iterrows():
            lines.append(f"- **{r['bsv_axis']} ({r['axis_name']})** — "
                         f"{int(r['n_datasets'])} datasets · dominant "
                         f"{r['dom_direction']} · transfer_score "
                         f"{r['axis_transfer_score']:.2f}")
    else:
        lines.append("- No emergent axis structure detected.")

    # Q2
    lines += ["", "### 2. Which BSV axes are most transferable?"]
    if not axis_score.empty:
        for _, r in axis_score.head(5).iterrows():
            lines.append(f"- {r['bsv_axis']} · transfer_score "
                         f"{r['axis_transfer_score']:.2f} · "
                         f"datasets={int(r['n_datasets'])} · "
                         f"sample_types={int(r['n_sample_types'])} · "
                         f"consistency={r['direction_consistency']}")

    # Q3
    lines += ["", "### 3. MSS candidates: recurrence + caveats"]
    if mss_transfer is not None and not mss_transfer.empty:
        for cls, sub in mss_transfer.groupby("classification"):
            lines.append(f"- **{cls}** ({len(sub)} candidates): "
                         + ", ".join(sub["mss_candidate"].head(8).tolist()))

    # Q4
    lines += ["", "### 4. Do EV and serum datasets form distinct context spaces?"]
    if not ctx_dataset.empty:
        per_st = ctx_dataset.groupby("sample_type").size().to_dict()
        lines.append(f"- datasets per sample_type: {per_st}")
        if "umap_1" in ctx_dataset.columns:
            ev_pts = ctx_dataset[ctx_dataset.sample_type == "EV"]
            ser_pts = ctx_dataset[ctx_dataset.sample_type == "serum"]
            if len(ev_pts) > 0 and len(ser_pts) > 0:
                from scipy.spatial.distance import cdist
                ev_xy = ev_pts[["umap_1", "umap_2"]].values
                ser_xy = ser_pts[["umap_1", "umap_2"]].values
                dist = float(np.median(cdist(ev_xy, ser_xy)))
                lines.append(f"- median UMAP distance EV↔serum: {dist:.2f} "
                             f"(higher = more distinct context spaces)")

    # Q5
    lines += ["", "### 5. Do disease/condition families cluster by biochemical response?"]
    if not em.empty:
        for _, r in em.head(8).iterrows():
            lines.append(f"- {r['bsv_axis']} in `{r['condition_family']}` "
                         f"· {int(r['n_datasets'])} datasets · "
                         f"emergence_score {r['emergence_score']:.2f}")

    # Q6
    lines += ["", "### 6. Findings robust enough for the demo"]
    if not findings.empty:
        for _, r in findings[findings.confidence == "STRONG"].head(8).iterrows():
            lines.append(f"- **{r['finding_id']}** · {r['finding_title']}")
        if (findings.confidence == "STRONG").sum() == 0:
            for _, r in findings.head(5).iterrows():
                lines.append(f"- {r['finding_id']} · {r['finding_title']} "
                             f"({r['confidence']})")

    # Q7
    lines += ["", "### 7. Hypothesis-generating only"]
    if not findings.empty:
        for _, r in findings[findings.confidence == "WEAK"].head(5).iterrows():
            lines.append(f"- {r['finding_id']} · {r['finding_title']}")

    # Q8
    lines += [
        "", "### 8. Backend tests recommended next",
        "- promote TRANSFERABLE MSS candidates to per-pilot validation panels",
        "- run substrate-aware re-scoring on SUBSTRATE_LOCKED MSS candidates",
        "- compute paired EV vs serum statistics on shared axes (G08, G09, G11)",
        "- build a per-condition-family ΔBSV reference panel",
        "- audit caveat-flagged datasets for QC + substrate documentation",
    ]

    lines += [
        "",
        "## Caveat summary",
    ]
    if caveats is not None and not caveats.empty:
        cav_sum = (caveats.groupby("caveat_id")["dataset"].nunique()
                    .sort_values(ascending=False))
        for k, v in cav_sum.items():
            lines.append(f"- `{k}` — flagged across {int(v)} datasets")
    else:
        lines.append("- (no caveats found in MD reports)")

    lines += [
        "",
        "## Strict invariants preserved",
        "- GAIRA core unchanged.",
        "- No GAIRA scoring rerun.",
        "- Read-only crawl over /Volumes/SSD_Rad/GAIRA_BUILD.",
        "- All tables / figures / report under "
        f"`{ROOT}/`.",
        "",
        f"## Final decision: **{decision}**",
    ]

    (R / "REPORT_context_graph_discovery_v1.md").write_text("\n".join(lines))
    return decision


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("="*78)
    print("gaira_base_4_context_graph_discovery_v1")
    print("="*78)

    inv = stage1_inventory()
    events = stage2_evidence_events(inv)
    nodes, edges = stage3_graph(events, inv)
    rec = stage4_recurrence(events)
    caveats = stage5_caveats(inv)
    ctx_dataset, _ = stage6_embeddings(events)
    metrics = stage7_metrics(events, rec.get("mss_transfer", pd.DataFrame()))
    findings = stage8_findings(events,
                                metrics.get("axis_transfer", pd.DataFrame()),
                                rec.get("mss_transfer", pd.DataFrame()),
                                metrics.get("emergent", pd.DataFrame()),
                                caveats)
    stage9_figures(events, nodes, edges,
                    rec.get("sample_type_axis", pd.DataFrame()),
                    rec.get("condition_family_axis", pd.DataFrame()),
                    metrics.get("axis_transfer", pd.DataFrame()),
                    rec.get("mss_transfer", pd.DataFrame()),
                    ctx_dataset, caveats)
    decision = stage10_report(events,
                                metrics.get("axis_transfer", pd.DataFrame()),
                                rec.get("mss_transfer", pd.DataFrame()),
                                metrics.get("emergent", pd.DataFrame()),
                                caveats, ctx_dataset, findings,
                                rec.get("sample_type_axis", pd.DataFrame()))
    # snapshot script
    try:
        shutil.copy(__file__, C / Path(__file__).name)
    except Exception:
        pass
    print(f"\n[done] decision: {decision}")


if __name__ == "__main__":
    main()
