from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import pandas as pd

from gaira.config import get_database_path


ROOT = Path(__file__).resolve().parents[1]
PHASE1_DIR = ROOT / "reports" / "gaira_phase1_registry_audit_v2"
RAW_PILOT_DIR = ROOT / "reports" / "gaira_bsv_raw_pilot_v1"
OUTPUT_DIR = ROOT / "reports" / "gaira_grounding_coverage_expansion_audit_v1"

COMPOUND_SPECS = [
    {
        "compound_name": "hypoxanthine",
        "compound_family": "purine_target",
        "grounding_patterns": ["hypox", "hypoxanthine"],
        "grounding_exact_labels": ["Hypox"],
        "biosample_patterns": ["hypox"],
        "biosample_exact_labels": ["Hypox"],
        "blocking_priority": 1,
    },
    {
        "compound_name": "xanthine",
        "compound_family": "purine_neighbor",
        "grounding_patterns": ["xanth", "xanthine"],
        "grounding_exact_labels": ["Xanth"],
        "biosample_patterns": ["xanth"],
        "biosample_exact_labels": ["Xanth"],
        "blocking_priority": 2,
    },
    {
        "compound_name": "uric_acid",
        "compound_family": "purine_neighbor",
        "grounding_patterns": ["uric", "urate"],
        "grounding_exact_labels": ["UA", "UAfree", "UAbound", "UAiso", "UA+HSA", "UAiso+HSA", "UA+HSAfilterLower", "UA+HSAfilterUpper", "UAiso+HSAfilterLower", "UAiso+HSAfilterUpper", "Ura"],
        "biosample_patterns": [],
        "biosample_exact_labels": ["UA"],
        "blocking_priority": 2,
    },
    {
        "compound_name": "inosine",
        "compound_family": "purine_neighbor",
        "grounding_patterns": ["inosine"],
        "biosample_patterns": ["inosine"],
        "blocking_priority": 3,
    },
    {
        "compound_name": "adenine",
        "compound_family": "purine_reference_present",
        "grounding_patterns": ["adenine", "ade"],
        "grounding_exact_labels": ["Ade"],
        "biosample_patterns": ["adenine", "ade"],
        "biosample_exact_labels": ["Ade"],
        "blocking_priority": 4,
    },
    {
        "compound_name": "guanine",
        "compound_family": "purine_neighbor",
        "grounding_patterns": ["guanine", "gua"],
        "grounding_exact_labels": ["Gua"],
        "biosample_patterns": ["guanine", "gua"],
        "biosample_exact_labels": ["Gua"],
        "blocking_priority": 3,
    },
    {
        "compound_name": "ergothioneine",
        "compound_family": "sulfur_target",
        "grounding_patterns": ["ergo", "ergoth"],
        "grounding_exact_labels": ["Ergo"],
        "biosample_patterns": ["ergo", "erg_"],
        "biosample_exact_labels": ["Ergo"],
        "blocking_priority": 1,
    },
    {
        "compound_name": "glutathione",
        "compound_family": "sulfur_neighbor",
        "grounding_patterns": ["glutathione"],
        "biosample_patterns": ["glutathione"],
        "blocking_priority": 4,
    },
    {
        "compound_name": "cysteamine",
        "compound_family": "sulfur_neighbor",
        "grounding_patterns": ["cysteamine"],
        "biosample_patterns": ["cysteamine"],
        "blocking_priority": 4,
    },
    {
        "compound_name": "homocysteine",
        "compound_family": "sulfur_neighbor",
        "grounding_patterns": ["homocysteine"],
        "biosample_patterns": ["homocysteine"],
        "blocking_priority": 4,
    },
    {
        "compound_name": "homocystine",
        "compound_family": "sulfur_neighbor",
        "grounding_patterns": ["homocystine"],
        "biosample_patterns": ["homocystine"],
        "blocking_priority": 4,
    },
    {
        "compound_name": "cystathionine",
        "compound_family": "sulfur_neighbor",
        "grounding_patterns": ["cystathionine"],
        "grounding_exact_labels": [],
        "biosample_patterns": ["cystathionine"],
        "biosample_exact_labels": [],
        "blocking_priority": 4,
    },
    {
        "compound_name": "cysteine",
        "compound_family": "sulfur_neighbor",
        "grounding_patterns": ["cysteine", " cys"],
        "grounding_exact_labels": ["Cys"],
        "biosample_patterns": ["cysteine", " cys"],
        "biosample_exact_labels": ["Cys"],
        "blocking_priority": 4,
    },
    {
        "compound_name": "methionine_related",
        "compound_family": "sulfur_neighbor",
        "grounding_patterns": ["methio", "methionine"],
        "grounding_exact_labels": ["Methio"],
        "biosample_patterns": ["methio", "methionine"],
        "biosample_exact_labels": ["Methio"],
        "blocking_priority": 4,
    },
]

SERUM_LOCAL_GROUNDING_DATASETS = {"serum_ag_colloids_grounding"}
SERUM_LOCAL_BIOSAMPLE_CANDIDATES = {"serum_ag_colloids", "ergothioneine_serum"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit missing grounding coverage for the Phase 2 raw/direct BSV pilot.")
    parser.add_argument("--dataset-registry-path", type=Path, default=PHASE1_DIR / "phase1_dataset_registry_v2.csv")
    parser.add_argument("--raw-pilot-summary-path", type=Path, default=RAW_PILOT_DIR / "run_summary.csv")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def lower_text(*parts: object) -> str:
    return " ".join(str(part or "") for part in parts).lower()


def match_any(text: str, patterns: list[str]) -> bool:
    text = text.lower()
    return any(pattern.lower() in text for pattern in patterns)


def load_context(args: argparse.Namespace) -> tuple[pd.DataFrame, set[str], set[str]]:
    registry = pd.read_csv(args.dataset_registry_path)
    pilot_summary = pd.read_csv(args.raw_pilot_summary_path)
    active_sources = {
        part.strip()
        for part in str(pilot_summary.iloc[0]["available_grounding_sources"]).split(";")
        if part.strip()
    }
    universal_rows = registry[
        registry["proposed_phase1_role"].astype(str) == "grounding_reference_universal_pure"
    ].copy()
    universal_sources = set(universal_rows["dataset_id"].astype(str))
    active_universal_sources = active_sources & universal_sources
    return registry, active_sources, active_universal_sources


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    db_path = get_database_path()
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        grounding_metadata = con.execute(
            """
            select dataset_id, class_label, compound_label, grounding_role
            from grounding_metadata
            """
        ).fetchdf()
        grounding_processed = con.execute(
            """
            select dataset_id, class_label, count(*) as processed_n
            from grounding_processed_spectra
            group by 1,2
            """
        ).fetchdf()
        biosample_metadata = con.execute(
            """
            select dataset_id, subclass_label, class_label
            from biosample_metadata
            where dataset_id in ('serum_ag_colloids', 'ergothioneine_serum')
            """
        ).fetchdf()
        biosample_processed = con.execute(
            """
            select p.dataset_id, m.subclass_label, m.class_label, count(*) as processed_n
            from biosample_processed_spectra p
            join biosample_metadata m
              on p.biosample_id = m.biosample_id
             and p.dataset_id = m.dataset_id
            where p.dataset_id in ('serum_ag_colloids', 'ergothioneine_serum')
            group by 1,2,3
            """
        ).fetchdf()
    finally:
        con.close()
    return grounding_metadata, grounding_processed, biosample_metadata, biosample_processed


def build_compound_coverage(
    active_universal_sources: set[str],
    grounding_metadata: pd.DataFrame,
    grounding_processed: pd.DataFrame,
    biosample_metadata: pd.DataFrame,
    biosample_processed: pd.DataFrame,
) -> pd.DataFrame:
    gp_lookup = grounding_processed.set_index(["dataset_id", "class_label"])["processed_n"].to_dict()
    bp_lookup = biosample_processed.set_index(["dataset_id", "subclass_label", "class_label"])["processed_n"].to_dict()
    rows: list[dict[str, object]] = []

    for spec in COMPOUND_SPECS:
        gmask = grounding_metadata.apply(
            lambda row: (
                match_any(lower_text(row["class_label"], row["compound_label"]), spec["grounding_patterns"])
                or str(row["class_label"]) in spec.get("grounding_exact_labels", [])
                or str(row["compound_label"]) in spec.get("grounding_exact_labels", [])
            ),
            axis=1,
        )
        gsub = grounding_metadata[gmask].copy()
        if not gsub.empty:
            grouped = (
                gsub.groupby(["dataset_id", "class_label", "compound_label", "grounding_role"], dropna=False)
                .size()
                .reset_index(name="metadata_n")
            )
            for _, row in grouped.iterrows():
                dataset_id = str(row["dataset_id"])
                class_label = str(row["class_label"])
                source_status = (
                    "active_universal_grounding"
                    if dataset_id in active_universal_sources
                    else ("local_serum_specific_grounding" if dataset_id in SERUM_LOCAL_GROUNDING_DATASETS else "local_other_grounding")
                )
                rows.append(
                    {
                        "compound_name": spec["compound_name"],
                        "compound_family": spec["compound_family"],
                        "record_type": "grounding",
                        "dataset_id": dataset_id,
                        "subset_id": "all",
                        "class_label": class_label,
                        "compound_label": str(row["compound_label"]),
                        "source_status": source_status,
                        "metadata_n": int(row["metadata_n"]),
                        "processed_spectra_n": int(gp_lookup.get((dataset_id, class_label), 0)),
                        "in_grounding_metadata": True,
                        "in_grounding_processed_spectra": (dataset_id, class_label) in gp_lookup,
                        "recommended_role": (
                            "use_now_validation_targeted"
                            if source_status == "local_serum_specific_grounding" and spec["blocking_priority"] <= 2
                            else ("already_active_universal" if source_status == "active_universal_grounding" else "supporting_neighbor_only")
                        ),
                        "notes": "",
                    }
                )

        bmask = biosample_metadata.apply(
            lambda row: (
                match_any(lower_text(row["class_label"], row["subclass_label"]), spec["biosample_patterns"])
                or str(row["class_label"]) in spec.get("biosample_exact_labels", [])
            ),
            axis=1,
        )
        bsub = biosample_metadata[bmask].copy()
        if not bsub.empty:
            grouped = (
                bsub.groupby(["dataset_id", "subclass_label", "class_label"], dropna=False)
                .size()
                .reset_index(name="metadata_n")
            )
            for _, row in grouped.iterrows():
                dataset_id = str(row["dataset_id"])
                subset_id = str(row["subclass_label"])
                source_status = "local_biosample_candidate" if dataset_id in SERUM_LOCAL_BIOSAMPLE_CANDIDATES else "local_biosample_other"
                rows.append(
                    {
                        "compound_name": spec["compound_name"],
                        "compound_family": spec["compound_family"],
                        "record_type": "biosample_candidate",
                        "dataset_id": dataset_id,
                        "subset_id": subset_id,
                        "class_label": str(row["class_label"]),
                        "compound_label": "",
                        "source_status": source_status,
                        "metadata_n": int(row["metadata_n"]),
                        "processed_spectra_n": int(bp_lookup.get((dataset_id, subset_id, str(row["class_label"])), 0)),
                        "in_grounding_metadata": False,
                        "in_grounding_processed_spectra": False,
                        "recommended_role": "convert_only_with_caution",
                        "notes": "",
                    }
                )

        if gsub.empty and bsub.empty:
            rows.append(
                {
                    "compound_name": spec["compound_name"],
                    "compound_family": spec["compound_family"],
                    "record_type": "missing",
                    "dataset_id": "",
                    "subset_id": "",
                    "class_label": "",
                    "compound_label": "",
                    "source_status": "not_confirmed_locally",
                    "metadata_n": 0,
                    "processed_spectra_n": 0,
                    "in_grounding_metadata": False,
                    "in_grounding_processed_spectra": False,
                    "recommended_role": "needs_external_acquisition_or_new_local_source",
                    "notes": "",
                }
            )

    audit = pd.DataFrame(rows)
    note_map = {
        "hypoxanthine": "Current validation blocker. Explicit local serum-specific grounding exists as `Hypox`, but it is not in the active universal pool.",
        "ergothioneine": "Current validation blocker. Explicit local serum-specific grounding exists as `Ergo`, and a separate 55-spectrum serum calibration archive exists, but neither is in the active universal pool.",
        "uric_acid": "Local serum-specific purine neighborhood exists as `UA`, `UAfree`, `UAbound`, `UAiso`, and HSA variants.",
        "xanthine": "Local serum-specific purine neighborhood exists as `Xanth`, but not in the active universal pool.",
        "adenine": "Already present in the active universal pool via `adenine_sers_control`.",
        "inosine": "Not found in local grounding or biosample archives inspected here.",
        "guanine": "Present only as shorthand `Gua` in serum-specific grounding; not in the active universal pool.",
    }
    audit["notes"] = audit["compound_name"].map(note_map).fillna("")
    audit = audit.drop_duplicates(
        subset=["compound_name", "record_type", "dataset_id", "subset_id", "class_label", "compound_label", "source_status"]
    )
    return audit.sort_values(["compound_name", "record_type", "dataset_id", "subset_id", "class_label"]).reset_index(drop=True)


def build_priority_list(compound_audit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for spec in sorted(COMPOUND_SPECS, key=lambda item: (item["blocking_priority"], item["compound_name"])):
        sub = compound_audit[compound_audit["compound_name"] == spec["compound_name"]].copy()
        active = sub[sub["source_status"] == "active_universal_grounding"]
        serum_grounding = sub[sub["source_status"] == "local_serum_specific_grounding"]
        biosample = sub[sub["source_status"] == "local_biosample_candidate"]

        if not active.empty:
            status = "already_covered_in_active_universal_grounding"
            action = "no_expansion_needed_for_current_blocker"
        elif not serum_grounding.empty:
            status = "locally_available_in_serum_specific_grounding_only"
            action = "add_as_validation_targeted_serum_grounding"
        elif not biosample.empty:
            status = "locally_available_in_biosample_only"
            action = "consider_manual_conversion_only_if_serum_grounding_is_insufficient"
        else:
            status = "not_confirmed_locally"
            action = "external_acquisition_needed_if_still_required"

        rows.append(
            {
                "priority_rank": spec["blocking_priority"],
                "compound_name": spec["compound_name"],
                "compound_family": spec["compound_family"],
                "current_local_status": status,
                "recommended_next_action": action,
                "why_it_matters": (
                    "Direct blocker for cspp_metabolite_spike_validation rerun"
                    if spec["blocking_priority"] == 1
                    else ("Useful neighboring purine/sulfur control" if spec["blocking_priority"] == 2 else "Lower-priority support compound")
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["priority_rank", "compound_name"]).reset_index(drop=True)


def build_local_candidates(compound_audit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    priority_map = {spec["compound_name"]: spec["blocking_priority"] for spec in COMPOUND_SPECS}
    for _, row in compound_audit.iterrows():
        if priority_map.get(str(row["compound_name"]), 99) > 2:
            continue
        if row["record_type"] == "grounding" and row["source_status"] == "local_serum_specific_grounding":
            rows.append(
                {
                    "compound_name": row["compound_name"],
                    "candidate_source_dataset_id": row["dataset_id"],
                    "candidate_source_subset_id": row["subset_id"],
                    "candidate_class_label": row["class_label"],
                    "candidate_source_type": "grounding",
                    "eligibility_status": "immediate_local_addition",
                    "caution_level": "serum_specific_only_not_universal",
                    "recommended_next_step": "add to validation-targeted serum grounding pool",
                    "rationale": row["notes"],
                }
            )
        elif row["record_type"] == "biosample_candidate" and row["source_status"] == "local_biosample_candidate":
            rows.append(
                {
                    "compound_name": row["compound_name"],
                    "candidate_source_dataset_id": row["dataset_id"],
                    "candidate_source_subset_id": row["subset_id"],
                    "candidate_class_label": row["class_label"],
                    "candidate_source_type": "biosample_candidate",
                    "eligibility_status": "manual_review_only",
                    "caution_level": "not_universal_and_not_yet_grounding",
                    "recommended_next_step": "use only if serum-specific grounding rows remain insufficient",
                    "rationale": row["notes"],
                }
            )
    candidates = pd.DataFrame(rows).drop_duplicates()
    return candidates.sort_values(["compound_name", "candidate_source_dataset_id", "candidate_class_label"]).reset_index(drop=True)


def write_summary(
    output_path: Path,
    compound_audit: pd.DataFrame,
    priority_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
) -> None:
    def status_line(name: str) -> str:
        sub = compound_audit[compound_audit["compound_name"] == name]
        active = sorted(sub[sub["source_status"] == "active_universal_grounding"]["dataset_id"].unique().tolist())
        serum = sorted(sub[sub["source_status"] == "local_serum_specific_grounding"]["dataset_id"].unique().tolist())
        bios = sorted(sub[sub["source_status"] == "local_biosample_candidate"]["dataset_id"].unique().tolist())
        return f"- `{name}`: active_universal={active or ['none']}; serum_specific={serum or ['none']}; biosample_only={bios or ['none']}"

    lines = [
        "# GAIRA Phase 2e Grounding Coverage Expansion Audit",
        "",
        "This audit stays focused on the current `cspp_metabolite_spike_validation` bottleneck and uses the live canonical grounding/biosample tables.",
        "",
        "## What Is Blocking Clean Validation",
        status_line("hypoxanthine"),
        status_line("ergothioneine"),
        status_line("xanthine"),
        status_line("uric_acid"),
        "",
        "## Current Interpretation",
        "- The active universal pool still lacks explicit hypoxanthine and ergothioneine references.",
        "- Local serum-specific grounding already contains explicit `Hypox`, `Xanth`, `UA`, `UAfree`, `UAbound`, `UAiso`, `UA+HSA`, `UAiso+HSA`, and `Ergo` rows.",
        "- `ergothioneine_serum` is already ingested locally as a 55-spectrum biosample calibration archive, but it is not a grounding dataset and should not be promoted silently to a universal reference source.",
        "",
        "## Smallest Useful Expansion Before Rerun",
        "- Add `serum_ag_colloids_grounding` classes `Hypox` and `Ergo` into a validation-targeted serum grounding pool for this single validation family.",
        "- Add purine-neighborhood serum-specific controls `Xanth`, `UA`, `UAfree`, `UAbound`, `UAiso`, and `UA+HSA` as neighboring references, still marked serum-specific rather than universal.",
        "- Keep `ergothioneine_serum` as a secondary fallback candidate only if the serum-specific grounding rows remain insufficient after the first targeted rerun.",
        "",
        "## Still Missing Entirely",
        "- `inosine` is not confirmed locally in grounding or biosample tables inspected here.",
        "- No new acquisition is needed yet for the immediate rerun if the goal is only to fix the current serum validation panel, because the local serum-specific grounding pool already covers the critical missing labels.",
        "",
        "## Priority Table",
    ]
    for _, row in priority_df.iterrows():
        lines.append(
            f"- `{row['compound_name']}`: {row['current_local_status']} -> {row['recommended_next_action']}"
        )
    if not candidate_df.empty:
        lines.extend(["", "## Immediate Local Candidates"])
        for _, row in candidate_df.head(20).iterrows():
            lines.append(
                f"- `{row['compound_name']}` from `{row['candidate_source_dataset_id']}::{row['candidate_source_subset_id']}` / `{row['candidate_class_label']}` -> {row['recommended_next_step']}"
            )
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    registry, active_sources, active_universal_sources = load_context(args)
    grounding_metadata, grounding_processed, biosample_metadata, biosample_processed = load_tables()

    compound_audit = build_compound_coverage(
        active_universal_sources,
        grounding_metadata,
        grounding_processed,
        biosample_metadata,
        biosample_processed,
    )
    priority_df = build_priority_list(compound_audit)
    candidate_df = build_local_candidates(compound_audit)

    compound_audit.to_csv(args.output_dir / "compound_coverage_audit.csv", index=False)
    priority_df.to_csv(args.output_dir / "missing_reference_priority_list.csv", index=False)
    candidate_df.to_csv(args.output_dir / "local_grounding_expansion_candidates.csv", index=False)
    write_summary(args.output_dir / "grounding_expansion_summary.md", compound_audit, priority_df, candidate_df)

    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
