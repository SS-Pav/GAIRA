"""Structured Evidence v2 data loaders for the Literature Ops Console.

Reads from the v2 CSV files under structured_evidence_v2/ and returns
DataFrames compatible with the existing Streamlit render functions.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import pandas as pd

V2_ROOT = Path("/Volumes/SSD_Rad/GAIRA_DATA/structured_evidence_v2")
REPORTS = V2_ROOT / "reports"
REGISTRY = V2_ROOT / "registry"
PROCESSED = V2_ROOT / "processed"

BATCH1_EVIDENCE = PROCESSED / "extraction_runs" / "phaseG_batch1_20260409_evidence_v3_corrected.csv"
BATCH2A_EVIDENCE = PROCESSED / "extraction_runs" / "phaseG_batch2A_20260409_evidence.csv"
INGESTED_REGISTRY = REGISTRY / "ingested_source_registry.csv"
CANDIDATE_PARTITION = REGISTRY / "candidate_partition_registry.csv"
BLOCKED_REGISTRY = REGISTRY / "blocked_high_quality_registry.csv"
ACTIVE_WORKING_SET = REPORTS / "phaseF_active_working_set.csv"
RECLASSIFIED_LAKE = REPORTS / "phaseF_reclassified_candidate_lake.csv"
MEANING_HIERARCHY = REPORTS / "meaning_hierarchy.csv"
CONDITION_CONFIDENCE = REPORTS / "condition_link_confidence_v2.csv"
REGION_AMBIGUITY = REPORTS / "spectral_region_ambiguity_table.csv"
NEIGHBORHOODS = PROCESSED / "updated_neighborhoods.csv"
ASSIGNMENT_PATTERNS = PROCESSED / "updated_assignment_patterns.csv"
MODALITY_AUDIT = REPORTS / "modality_audit.csv"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=str).fillna("")
    except Exception:
        return pd.DataFrame()


def _classify_modality(title: str) -> str:
    t = title.lower()
    if re.search(r"\bsers\b|surface.enhanced", t):
        return "SERS"
    if re.search(r"\braman\b", t):
        return "Raman"
    return "unclear"


# ---------- OVERVIEW ---------- #

def load_overview_metrics() -> dict:
    """Return overview metrics from v2 CSV state."""
    # Candidates
    rcl = _read_csv(RECLASSIFIED_LAKE)
    if rcl.empty:
        rcl = _read_csv(CANDIDATE_PARTITION)

    lane_counts = rcl["new_lane"].value_counts().to_dict() if "new_lane" in rcl.columns else {}

    # Ingested evidence
    ev = load_all_evidence()
    reg = _read_csv(INGESTED_REGISTRY)

    # Meanings & patterns
    meanings = ev["normalized_meaning"].nunique() if not ev.empty else 0
    patterns = _read_csv(ASSIGNMENT_PATTERNS)
    motifs = len(patterns)

    # Condition links
    cond_motif_path = list((PROCESSED / "extraction_runs").glob("*_condition_motif_links.csv"))
    cond_region_path = list((PROCESSED / "extraction_runs").glob("*_condition_region_links.csv"))
    cond_links = 0
    for p in cond_motif_path:
        cond_links += len(_read_csv(p))
    for p in cond_region_path:
        cond_links += len(_read_csv(p))

    # Neighborhoods
    neighborhoods = _read_csv(NEIGHBORHOODS)

    # Modality gap
    modality_missing = 0
    if not ev.empty:
        if "modality" not in ev.columns:
            modality_missing = len(ev)
        else:
            modality_missing = int((ev["modality"] == "").sum())

    # Blocked
    blocked = _read_csv(BLOCKED_REGISTRY)

    return {
        "total_candidates": len(rcl),
        "oa_ready": int(lane_counts.get("ready_now", 0)),
        "oa_fetch": int(lane_counts.get("oa_fetchable", 0)),
        "blocked": int(lane_counts.get("blocked_high_value", 0)),
        "low_value": int(lane_counts.get("low_value_archive", 0)),
        "needs_resolution": int(lane_counts.get("needs_resolution", 0)),
        "ingested": len(reg),
        "total_structured_rows": len(ev),
        "total_meanings": meanings,
        "motifs_count": motifs,
        "condition_links": cond_links,
        "spectral_regions": len(neighborhoods),
        "modality_missing_rows": modality_missing,
        "blocked_high_quality": len(blocked),
        "last_run": _latest_run_info(reg),
    }


def _latest_run_info(reg: pd.DataFrame) -> dict:
    if reg.empty or "ingestion_run_id" not in reg.columns:
        return {"source_kind": "unknown", "papers_processed": 0, "contributing_papers": 0, "rows_added": 0, "motifs_affected": 0}
    latest_id = reg["ingestion_run_id"].dropna().iloc[-1] if not reg["ingestion_run_id"].dropna().empty else ""
    latest = reg[reg["ingestion_run_id"] == latest_id] if latest_id else pd.DataFrame()
    return {
        "source_kind": latest_id,
        "papers_processed": len(latest),
        "contributing_papers": len(latest),
        "rows_added": int(pd.to_numeric(latest["structured_rows"], errors="coerce").sum()) if not latest.empty else 0,
        "motifs_affected": int(pd.to_numeric(latest["motifs_affected"], errors="coerce").sum()) if not latest.empty else 0,
    }


# ---------- EVIDENCE ---------- #

def load_all_evidence() -> pd.DataFrame:
    """Load all ingested evidence from v2 extraction runs."""
    dfs = []
    for p in [BATCH1_EVIDENCE, BATCH2A_EVIDENCE]:
        df = _read_csv(p)
        if not df.empty:
            dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    ev = pd.concat(dfs, ignore_index=True)
    return ev


# ---------- CANDIDATE QUEUE ---------- #

def load_candidate_queue_df() -> pd.DataFrame:
    """Load candidate queue with current lane assignments."""
    aws = _read_csv(ACTIVE_WORKING_SET)
    rcl = _read_csv(RECLASSIFIED_LAKE)
    if aws.empty:
        return pd.DataFrame()

    if not rcl.empty and "new_lane" in rcl.columns:
        lane_map = rcl.set_index("paper_id")["new_lane"].to_dict()
        aws["current_lane"] = aws["paper_id"].map(lane_map).fillna("unknown")
    else:
        aws["current_lane"] = aws.get("current_lane", "unknown")

    # Map lanes to display states
    lane_to_state = {
        "ready_now": "OA READY",
        "oa_fetchable": "OA FETCH",
        "blocked_high_value": "BLOCKED",
        "low_value_archive": "LOW VALUE",
        "needs_resolution": "NEEDS RESOLUTION",
    }

    # Check which are ingested
    reg = _read_csv(INGESTED_REGISTRY)
    ingested_pids = set(reg["paper_id"].tolist()) if not reg.empty and "paper_id" in reg.columns else set()

    aws["operational_state"] = aws.apply(
        lambda r: "INGESTED" if r["paper_id"] in ingested_pids else lane_to_state.get(r["current_lane"], "UNKNOWN"),
        axis=1,
    )
    aws["yield_signal"] = ""
    aws["modality"] = aws["title"].apply(_classify_modality) if "title" in aws.columns else ""

    return aws


# ---------- BLOCKED ---------- #

def load_blocked_df() -> pd.DataFrame:
    """Load the high-quality blocked registry."""
    df = _read_csv(BLOCKED_REGISTRY)
    if df.empty:
        return pd.DataFrame()
    # Ensure columns match what the render function expects
    col_map = {
        "publisher_blocker": "blocker_type",
        "expected_evidence_contribution": "expected_yield",
        "canonical_url": "canonical_article_url",
        "rescue_rationale": "blocker_detail",
        "upload_target_path": "local_upload_target_path",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    if "suggested_manual_action" not in df.columns:
        df["suggested_manual_action"] = "Manual institutional download"
    if "preferred_asset_type_to_rescue_first" not in df.columns:
        df["preferred_asset_type_to_rescue_first"] = "manuscript_PDF"
    if "supplementary_files_json" not in df.columns:
        df["supplementary_files_json"] = "[]"
    return df


# ---------- EVIDENCE CONTRIBUTION / LIVE REGISTRY ---------- #

def load_live_evidence_registry_df() -> pd.DataFrame:
    """One row per ingested source with impact metrics."""
    reg = _read_csv(INGESTED_REGISTRY)
    if reg.empty:
        return pd.DataFrame()

    ev = load_all_evidence()

    rows = []
    for _, r in reg.iterrows():
        pid = r.get("paper_id", "")
        sid = r.get("source_id", "")

        # Get evidence rows for this paper
        paper_ev = ev[ev["paper_id"] == pid] if not ev.empty and "paper_id" in ev.columns else pd.DataFrame()

        # Source breakdown
        src_counts = paper_ev["extraction_source"].value_counts().to_dict() if not paper_ev.empty and "extraction_source" in paper_ev.columns else {}
        src_mix_parts = [f"{s}:{c}" for s, c in sorted(src_counts.items())]
        src_mix = " | ".join(src_mix_parts) if src_mix_parts else "unknown"

        # Modality from title
        title = r.get("title", "")
        modality = _classify_modality(title)

        # Regions covered
        regions = sorted(paper_ev["spectral_region"].unique().tolist()) if not paper_ev.empty and "spectral_region" in paper_ev.columns else []

        # Condition links
        conditions = set()
        if not paper_ev.empty and "condition" in paper_ev.columns:
            for c in paper_ev["condition"].dropna():
                for part in c.split(";"):
                    conditions.add(part.strip())

        rows.append({
            "source_id": sid,
            "paper_id": pid,
            "title": title,
            "doi": r.get("doi", ""),
            "structured_evidence_rows": int(r.get("structured_rows", 0) or 0),
            "strengthened_support_rows": int(r.get("strengthened_rows", 0) or 0),
            "meanings_touched": int(r.get("meanings_touched", 0) or 0),
            "new_meanings_introduced": int(r.get("new_meanings_introduced", 0) or 0),
            "motifs_affected": int(r.get("motifs_affected", 0) or 0),
            "condition_motif_links": int(r.get("condition_links_added", 0) or 0),
            "condition_neighborhood_links": 0,
            "extraction_modes": src_mix,
            "modality": modality,
            "biosample": paper_ev["biosample"].iloc[0] if not paper_ev.empty and "biosample" in paper_ev.columns else "",
            "condition": "; ".join(sorted(conditions)),
            "regions_covered": len(regions),
            "region_list": "; ".join(regions),
            "impact_score": float(r.get("structured_rows", 0) or 0) * 0.5 + float(r.get("meanings_touched", 0) or 0) * 2.0,
            "ingestion_run_id": r.get("ingestion_run_id", ""),
            "assigned_value_class": r.get("assigned_value_class", ""),
        })

    return pd.DataFrame(rows)


# ---------- ONTOLOGY / COVERAGE ---------- #

def load_ontology_summary() -> dict:
    """Load L1/L2 meaning coverage and region summary."""
    ev = load_all_evidence()
    if ev.empty:
        return {"l1_counts": {}, "l2_counts": {}, "region_counts": {}, "condition_counts": {}}

    # L1
    l1_counts = {}
    if "l1_category" in ev.columns:
        for val in ev["l1_category"].dropna():
            for part in val.split("; "):
                part = part.strip()
                if part:
                    l1_counts[part] = l1_counts.get(part, 0) + 1

    # L2
    l2_counts = {}
    if "l2_category" in ev.columns:
        for val in ev["l2_category"].dropna():
            for part in val.split("; "):
                part = part.strip()
                if part:
                    l2_counts[part] = l2_counts.get(part, 0) + 1

    # Region
    region_counts = ev["spectral_region"].value_counts().to_dict() if "spectral_region" in ev.columns else {}

    # Condition
    condition_counts = {}
    if "condition" in ev.columns:
        for val in ev["condition"].dropna():
            for part in val.split(";"):
                part = part.strip()
                if part:
                    condition_counts[part] = condition_counts.get(part, 0) + 1

    return {
        "l1_counts": l1_counts,
        "l2_counts": l2_counts,
        "region_counts": region_counts,
        "condition_counts": condition_counts,
    }
