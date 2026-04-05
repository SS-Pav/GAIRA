from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tmp/global_v2_dataset_audit"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def is_saliva_shard(url: str) -> bool:
    return any(
        token in url
        for token in [
            "20428395",
            "20427957",
            "20427954",
            "20427951",
            "20427948",
            "20427945",
            "20427939",
            "20427936",
            "20427933",
            "20427930",
            "20427927",
            "20427924",
            "20427921",
            "20427918",
            "20427909",
            "20427906",
            "20427903",
            "20282238",
            "20406102",
        ]
    )


FINAL_ACTIONS = {
    "https://zenodo.org/records/4941488": ("ingest_now", 4.4, "Strong pathogen raw-spectrum matrix in CSV; useful for robustness and non-serum domain transfer."),
    "https://zenodo.org/records/5947010": ("ingest_now", 4.2, "Real raw TXT cohort with disease/control labels and replicate-rich acquisition; adds non-blood biofluid diversity."),
    "https://figshare.com/articles/dataset/Raman_spectroscopic_techniques_to_detect_ovarian_cancer_biomarkers_in_blood_plasma/6744206": ("ingest_now", 4.1, "Large raw plasma Raman archive with hundreds of TXT spectra; clear disease benchmark value."),
    "https://figshare.com/articles/dataset/SERS_and_Raman_spectra_of_WT_and_mutant_cytochromes_c/4903091": ("ingest_now", 3.9, "Clean molecule-level grounding asset with native text/CSV spectra for cytochrome variants."),
    "https://figshare.com/articles/dataset/Surface-enhanced_SERS_and_tip-enhanced_TERS_Raman_scattering_in_label-free_characterization_of_erythrocyte_membranes_and_extracellular_vesicles_in_nano-scale_and_at_the_single-molecule_level_/24105993": ("ingest_later", 3.6, "Useful EV/RBC raw TXT archive, but specialized and small; better as manual reconstruction target than first ingest wave."),
    "https://zenodo.org/records/19369604": ("ingest_later", 3.5, "Potentially valuable urine stroke cohort, but 1.4 GB RAR package needs manual extraction and versioned cleanup."),
    "https://zenodo.org/records/19109120": ("reject", 1.8, "Superseded by newer Zenodo version 19369604; do not ingest both."),
    "https://zenodo.org/records/12740805": ("ingest_later", 3.4, "Real pathogen raw spectra, but narrower task framing and partial overlap with existing pathogen/reference assets."),
    "https://zenodo.org/records/10851312": ("ingest_later", 3.4, "Interesting cell-state trajectory archive, but large RAR and probe-specific biology argue for second wave."),
    "https://zenodo.org/records/5021659": ("ingest_later", 3.3, "Useful urine/pathogen diversity, but proprietary OPJ/OPJU packaging raises ingest friction."),
    "https://figshare.com/articles/dataset/SERS_spectra_of_43_patients_with_ACS_xlsx/24747531?file=43481136": ("ingest_later", 3.0, "Small but potentially usable disease-benchmark workbook; verify axis/metadata structure manually before ingest."),
    "https://figshare.com/articles/dataset/Suplementary_material_DIB_ACS_40_samples_xlsx/24564787?file=43183257": ("ingest_later", 2.9, "Likely related ACS workbook; hold for manual inspection rather than immediate ingest."),
    "https://zenodo.org/records/5806132": ("method_only", 2.5, "Good quantitative SERS methodology, but drug-focused substrate benchmarking is not a Global v2 priority."),
    "https://zenodo.org/records/14755439": ("method_only", 2.4, "Support zip appears figure/panel oriented rather than a clean cohort release."),
    "https://zenodo.org/records/5806264": ("ingest_later", 3.0, "Bacterial metabolism xlsx may support metabolite/pathogen axes, but it is not a straightforward benchmark dataset."),
    "https://zenodo.org/records/8130216": ("ingest_later", 3.0, "Interesting purine secretome archive, but cell-secretome framing and large zip make this a deferred grounding-style ingest."),
    "https://zenodo.org/records/3994312": ("method_only", 2.2, "Substrate paper archive with figure zips; not a usable biological training dataset."),
    "https://figshare.com/articles/dataset/Additional_file_2_of_Combined_miRNA_and_SERS_urine_liquid_biopsy_for_the_point-of-care_diagnosis_and_molecular_stratification_of_bladder_cancer/19498603?file=34649167": ("literature_only", 1.9, "Uploaded workbook contains validated miRNA target sheets, not reusable raw spectra."),
    "https://figshare.com/articles/dataset/Raw_Raman_data_/26059145?file=47123702": ("ingest_later", 3.2, "Interesting single-vesicle EV heterogeneity archive, but RAR packaging and unclear cohort design put it in manual-reconstruction tier."),
    "https://figshare.com/articles/dataset/DFT-Based_Theoretical_Study_on_Label-Free_SERS_Detection_of_Type_B_Fumonisins_New_Insights_into_Molecular_Substrate_Interactions_and_Quantification_Strategies/28565671?file=52894126": ("method_only", 2.3, "Primarily theoretical and substrate-interaction focused; not a priority biological encoder asset."),
    "https://figshare.com/articles/dataset/Source_Data_file_xlsxDataset_ArticleNatureComm_Dallarietal_2024/26411992?file=48039661": ("method_only", 2.2, "Figure-wise source workbook for a materials study; useful reference, weak ingest target."),
    "https://zenodo.org/records/18670010": ("method_only", 2.0, "Tear dopamine substrate paper; not aligned with current Global v2 gaps."),
    "https://zenodo.org/records/18284194": ("method_only", 2.0, "Drug quantification on nanocone arrays is not a priority shared-encoder dataset."),
    "https://zenodo.org/records/17023716": ("method_only", 2.0, "Metasurface sensing archive is mostly materials characterization."),
    "https://zenodo.org/records/7523579": ("method_only", 2.0, "Useful correlation study, but chemistry/method support rather than training data."),
    "https://zenodo.org/records/17035751": ("reject", 1.2, "Already ingested in GAIRA as adenine_sers_control."),
    "https://zenodo.org/records/14294417": ("reject", 1.0, "Already covered as support-only grounding in GAIRA."),
}


def main() -> None:
    inventory = read_csv(OUT / "tables/global_v2_candidate_dataset_inventory.csv")
    scoring = read_csv(OUT / "tables/global_v2_candidate_dataset_scoring.csv")
    dedup = read_csv(OUT / "tables/global_v2_deduplication_check.csv")
    priority = read_csv(OUT / "tables/global_v2_ingest_priority_list.csv")

    inv_by = {r["source_url"]: r for r in inventory}
    score_by = {r["source_url"]: r for r in scoring}
    dedup_by = {r["source_url"]: r for r in dedup}
    pri_by = {r["source_url"]: r for r in priority}

    for url, (action, score, rationale) in FINAL_ACTIONS.items():
        if url in score_by:
            score_by[url]["recommended_action"] = action
            score_by[url]["overall_value_score"] = f"{score:.2f}"
            score_by[url]["rationale"] = rationale
            pri_by[url]["recommended_action"] = action
            pri_by[url]["overall_value_score"] = f"{score:.2f}"
            pri_by[url]["rationale"] = rationale

    for url, row in inv_by.items():
        if is_saliva_shard(url):
            row["domain_category"] = "saliva"
            row["label_type"] = "disease class; patient ID" if "Patient" in row["title"] else "disease class"
            row["reusable_for_ml"] = "partial"
            row["provenance_clean"] = "partial" if "ERCC/20406102" in url else "yes"
            score_by[url]["recommended_action"] = "reject"
            score_by[url]["overall_value_score"] = "2.10"
            score_by[url]["rationale"] = "Single-patient or metadata-only saliva shard; only reconsider as part of a reconstructed cohort."
            pri_by[url]["recommended_action"] = "reject"
            pri_by[url]["overall_value_score"] = "2.10"
            pri_by[url]["rationale"] = "Single-patient or metadata-only saliva shard; only reconsider as part of a reconstructed cohort."
            dedup_by[url]["duplicate_status"] = "fragmented_cohort_component"
            dedup_by[url]["overlap_existing_dataset"] = "ucla_saliva_ev_shards"
            dedup_by[url]["duplicate_notes"] = "Fragmented patient-level shard; not worth ingesting standalone."

    conservative_domains = {
        "https://zenodo.org/records/4941488": "pathogen",
        "https://zenodo.org/records/3994312": "substrate / materials",
        "https://zenodo.org/records/8130216": "cells",
        "https://figshare.com/articles/dataset/DFT-Based_Theoretical_Study_on_Label-Free_SERS_Detection_of_Type_B_Fumonisins_New_Insights_into_Molecular_Substrate_Interactions_and_Quantification_Strategies/28565671?file=52894126": "molecule_reference",
        "https://figshare.com/articles/dataset/Raw_Raman_data_/26059145?file=47123702": "EV",
        "https://figshare.com/articles/dataset/Additional_file_2_of_Combined_miRNA_and_SERS_urine_liquid_biopsy_for_the_point-of-care_diagnosis_and_molecular_stratification_of_bladder_cancer/19498603?file=34649167": "urine",
    }
    for url, domain in conservative_domains.items():
        if url in inv_by:
            inv_by[url]["domain_category"] = domain
            pri_by[url]["domain_category"] = domain

    sorted_priority = sorted(
        priority,
        key=lambda r: (
            {"ingest_now": 0, "ingest_later": 1, "method_only": 2, "literature_only": 3, "reject": 4}[r["recommended_action"]],
            -float(r["overall_value_score"]),
            r["title"],
        ),
    )
    for idx, row in enumerate(sorted_priority, start=1):
        row["priority_rank"] = str(idx)

    write_csv(OUT / "tables/global_v2_candidate_dataset_inventory.csv", inventory)
    write_csv(OUT / "tables/global_v2_candidate_dataset_scoring.csv", scoring)
    write_csv(OUT / "tables/global_v2_deduplication_check.csv", dedup)
    write_csv(OUT / "tables/global_v2_ingest_priority_list.csv", sorted_priority)

    top_now = [r for r in sorted_priority if r["recommended_action"] == "ingest_now"][:10]
    top_later = [r for r in sorted_priority if r["recommended_action"] == "ingest_later"][:10]
    redundant = [r for r in dedup if r["already_ingested"] == "yes" or r["duplicate_status"] in {"scope_overlap", "already_ingested_exact", "fragmented_cohort_component"}]
    method_only = [r for r in sorted_priority if r["recommended_action"] == "method_only"][:12]
    tempting = [r for r in sorted_priority if r["recommended_action"] in {"ingest_later", "reject"}][:12]

    lines = []
    lines.append("# GAIRA Global v2 Dataset Audit")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    counts = {}
    for row in sorted_priority:
        counts[row["recommended_action"]] = counts.get(row["recommended_action"], 0) + 1
    lines.append(f"- Audited candidates: {len(sorted_priority)}")
    for key in ["ingest_now", "ingest_later", "method_only", "literature_only", "reject"]:
        lines.append(f"- {key}: {counts.get(key, 0)}")
    lines.append("")
    lines.append("## Best Additions Right Now")
    lines.append("")
    for row in top_now:
        lines.append(f"- {row['title']} | {row['source_url']} | score={row['overall_value_score']} | {row['rationale']}")
    lines.append("")
    lines.append("## Redundant With Current GAIRA Assets")
    lines.append("")
    for row in redundant[:12]:
        lines.append(f"- {row['title']} | {row['source_url']} | {row['duplicate_status']} | {row['overlap_existing_dataset']}")
    lines.append("")
    lines.append("## Tempting But Low-Value")
    lines.append("")
    for row in tempting:
        lines.append(f"- {row['title']} | {row['source_url']} | action={row['recommended_action']} | {row['rationale']}")
    lines.append("")
    lines.append("## Methodology References Only")
    lines.append("")
    for row in method_only:
        lines.append(f"- {row['title']} | {row['source_url']}")
    lines.append("")
    lines.append("## Tiered Plan")
    lines.append("")
    lines.append("### Tier A: ingest immediately")
    for row in top_now:
        lines.append(f"- {row['title']} | {row['source_url']}")
    lines.append("")
    lines.append("### Tier B: deeper manual reconstruction")
    for row in top_later:
        lines.append(f"- {row['title']} | {row['source_url']}")
    lines.append("")
    lines.append("### Tier C: skip for now")
    for row in [r for r in sorted_priority if r["recommended_action"] in {"method_only", "literature_only", "reject"}][:30]:
        lines.append(f"- {row['title']} | {row['source_url']} | {row['recommended_action']}")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- The UCLA saliva sEV Figshare uploads are real raw TXT shards, but they are fragmented patient-level deposits rather than a clean cohort package. Treat them as reconstruction work, not direct ingest targets.")
    lines.append("- Zenodo 19369604 supersedes Zenodo 19109120 for the ischemic stroke urine release.")
    lines.append("- Figshare 19498603 exposes validated miRNA-target sheets rather than reusable spectral matrices; it is not a direct ingest target.")
    lines.append("- Substrate/materials papers with figure-wise source files were pushed out of Tier A even when numeric files exist, because they add little to the Global v2 encoder objective.")
    lines.append("")
    lines.append("## Candidate Priority Table")
    lines.append("")
    lines.append("| Rank | Action | Score | Domain | Candidate |")
    lines.append("|---:|---|---:|---|---|")
    for row in sorted_priority:
        lines.append(f"| {row['priority_rank']} | {row['recommended_action']} | {row['overall_value_score']} | {row['domain_category']} | {row['title']} |")

    (OUT / "report/global_v2_dataset_audit.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
