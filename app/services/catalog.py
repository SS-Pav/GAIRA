from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from gaira.config import get_database_path, get_storage_paths, require_data_root_exists


GROUNDING_SPECTRA_BUCKETS = {
    "Reference library": ["ramanbiolib"],
    "Controlled grounding": ["adenine_sers_control", "serum_ag_colloids_grounding"],
    "Metabolite fingerprints": ["metabolite_sers63_support"],
}

GROUNDING_SUPPORT_DATASETS = [
    "serum_ag_colloids_literature_grounding",
    "sers24_metabolite_support",
    "sers_fingerprint_workingpaper_support",
]

EV_BUCKETS = {
    "EV general": ["small2023_ev"],
    "EV disease/stress": ["shine_ev_sers", "diabetes_plasma_ev_sers"],
}

SERUM_BUCKETS = {
    "Serum general": [
        "serum_ag_colloids",
        "serum_protocol_comparison",
        "cspp_serum",
        "ergothioneine_serum",
        "covid_serum_raman",
    ],
    "Serum liver/hepatobiliary": ["cca_hcc_lm_serum_sers"],
}

DATASET_LABELS = {
    "ramanbiolib": "RamanBioLib",
    "adenine_sers_control": "Adenine controls",
    "metabolite_sers63_support": "Metabolite SERS fingerprints",
    "serum_ag_colloids_grounding": "Serum Ag colloids grounding",
    "serum_ag_colloids_literature_grounding": "Serum Ag colloids literature support",
    "sers24_metabolite_support": "24-metabolite support",
    "sers_fingerprint_workingpaper_support": "Fingerprint working-paper support",
    "small2023_ev": "small2023 EV",
    "shine_ev_sers": "SHINE / SPECTRA EV",
    "diabetes_plasma_ev_sers": "Diabetes plasma EV",
    "serum_ag_colloids": "Serum Ag colloids",
    "serum_protocol_comparison": "Serum protocol comparison",
    "cspp_serum": "CSPP serum",
    "ergothioneine_serum": "Ergothioneine serum",
    "covid_serum_raman": "COVID serum Raman",
    "cca_hcc_lm_serum_sers": "CCA / HCC / LM serum SERS",
    "hcc_serum": "HCC serum holdout",
}


def _connect(db_path: Path | None = None) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(db_path or get_database_path()), read_only=True)


def get_demo_paths() -> dict[str, Path]:
    storage = require_data_root_exists()
    return {
        "db_path": get_database_path(),
        "processed_data": storage["processed_data"],
        "hcc_eval_db": storage["processed_data"] / "hcc_holdout_evaluation" / "eval_db" / "gaira_hcc_holdout_eval.duckdb",
        "hcc_calibration_dir": storage["processed_data"] / "hcc_holdout_calibration",
    }


def _parse_json_array(value: str) -> list[float]:
    return [float(item) for item in json.loads(value)]


def _display_dataset(dataset_id: str) -> str:
    return DATASET_LABELS.get(dataset_id, dataset_id)


def _classify_grounding_type(class_label: str) -> str:
    value = class_label.lower()
    if any(token in value for token in ["aden", "guan", "xanth", "hypox", "uric", "caffeine", "nicotinamide"]):
        return "Purine-like / analytes"
    if any(token in value for token in ["acid", "lact", "carnit", "lipid", "chol", "ole", "linole"]):
        return "Acids / lipid-like"
    if any(token in value for token in ["amine", "tyramine", "quinoline", "imidazole"]):
        return "Amines / aromatics"
    return "Other metabolites"


@lru_cache(maxsize=1)
def get_global_kpis() -> dict[str, Any]:
    with _connect() as con:
        dataset_count = 16
        spectra_count = int(
            con.execute(
                """
                SELECT
                    (SELECT COUNT(*) FROM reference_spectra)
                    + (SELECT COUNT(*) FROM grounding_spectra)
                    + (SELECT COUNT(*) FROM biosample_spectra)
                """
            ).fetchone()[0]
        )
        support_docs = int(
            con.execute("SELECT COUNT(*) FROM grounding_support_documents").fetchone()[0]
            + con.execute("SELECT COUNT(*) FROM domain_context_documents").fetchone()[0]
            + con.execute("SELECT COUNT(DISTINCT source_id) FROM knowledge_chunks WHERE dataset_id = 'raman_knowledge_core'").fetchone()[0]
        )
    return {
        "total_live_datasets": dataset_count,
        "total_live_spectra": spectra_count,
        "total_support_context_docs": support_docs,
        "live_layers": 3,
        "holdout_status": "hcc_serum holdout-only",
    }


@lru_cache(maxsize=1)
def get_layer_tree() -> dict[str, Any]:
    with _connect() as con:
        registry_df = con.execute(
            """
            SELECT dataset_id, dataset_family, sample_type, notes
            FROM read_csv_auto(?)
            """,
            [str(Path(__file__).resolve().parents[2] / "data/registry/datasets.csv")],
        ).fetchdf()
        reference_meta = con.execute(
            """
            SELECT dataset_id, biochemical_class, COUNT(*) AS n_spectra
            FROM reference_metadata
            WHERE dataset_id = 'ramanbiolib'
            GROUP BY dataset_id, biochemical_class
            ORDER BY biochemical_class
            """
        ).fetchdf()
        grounding_classes = con.execute(
            """
            SELECT dataset_id, class_label, experiment_family, n_spectra
            FROM grounding_class_summary
            WHERE dataset_id IN ('adenine_sers_control', 'metabolite_sers63_support', 'serum_ag_colloids_grounding')
            ORDER BY dataset_id, class_label
            """
        ).fetchdf()
        support_docs = con.execute(
            """
            SELECT dataset_id, support_type, title, COUNT(*) AS n_docs
            FROM grounding_support_documents
            GROUP BY dataset_id, support_type, title
            ORDER BY dataset_id, title
            """
        ).fetchdf()
        biosample_classes = con.execute(
            """
            SELECT dataset_id, class_label, subclass_label, n_spectra
            FROM biosample_class_summary
            WHERE dataset_id IN (
                'small2023_ev', 'shine_ev_sers', 'diabetes_plasma_ev_sers',
                'serum_ag_colloids', 'serum_protocol_comparison', 'cspp_serum',
                'ergothioneine_serum', 'covid_serum_raman', 'cca_hcc_lm_serum_sers'
            )
            ORDER BY dataset_id, class_label
            """
        ).fetchdf()

    tree: dict[str, Any] = {"GAIRA_GROUNDING": {}, "GAIRA_EV": {}, "GAIRA_SERUM": {}}

    for bucket, dataset_ids in GROUNDING_SPECTRA_BUCKETS.items():
        datasets = []
        for dataset_id in dataset_ids:
            if dataset_id == "ramanbiolib":
                dataset_rows = reference_meta[reference_meta["dataset_id"] == dataset_id].copy()
                compounds = [
                    {
                        "class_label": row["biochemical_class"],
                        "family_label": "reference class",
                        "n_spectra": int(row["n_spectra"]),
                    }
                    for row in dataset_rows.to_dict("records")
                ]
                n_spectra = int(dataset_rows["n_spectra"].sum()) if not dataset_rows.empty else 0
                dataset_kind = "reference_spectra"
            else:
                dataset_rows = grounding_classes[grounding_classes["dataset_id"] == dataset_id].copy()
                compounds = [
                    {
                        "class_label": row["class_label"],
                        "type_bucket": _classify_grounding_type(str(row["class_label"])),
                        "experiment_family": row["experiment_family"],
                        "n_spectra": int(row["n_spectra"]),
                    }
                    for row in dataset_rows.to_dict("records")
                ]
                n_spectra = int(dataset_rows["n_spectra"].sum()) if not dataset_rows.empty else 0
                dataset_kind = "grounding_spectra"
            datasets.append(
                {
                    "dataset_id": dataset_id,
                    "display_name": _display_dataset(dataset_id),
                    "dataset_kind": dataset_kind,
                    "n_spectra": n_spectra,
                    "items": compounds,
                }
            )
        tree["GAIRA_GROUNDING"][bucket] = datasets

    support_only_datasets = []
    for dataset_id in GROUNDING_SUPPORT_DATASETS:
        dataset_rows = support_docs[support_docs["dataset_id"] == dataset_id].copy()
        support_only_datasets.append(
            {
                "dataset_id": dataset_id,
                "display_name": _display_dataset(dataset_id),
                "dataset_kind": "support_only",
                "n_spectra": 0,
                "n_docs": int(dataset_rows["n_docs"].sum()) if not dataset_rows.empty else 0,
                "items": [
                    {
                        "class_label": row["title"],
                        "family_label": row["support_type"],
                        "n_spectra": int(row["n_docs"]),
                    }
                    for row in dataset_rows.to_dict("records")[:12]
                ],
            }
        )
    tree["GAIRA_GROUNDING"]["Support-only grounding literature"] = support_only_datasets

    for bucket, dataset_ids in EV_BUCKETS.items():
        bucket_rows = biosample_classes[biosample_classes["dataset_id"].isin(dataset_ids)].copy()
        datasets = []
        for dataset_id in dataset_ids:
            dataset_rows = bucket_rows[bucket_rows["dataset_id"] == dataset_id].copy()
            datasets.append(
                {
                    "dataset_id": dataset_id,
                    "display_name": _display_dataset(dataset_id),
                    "n_spectra": int(dataset_rows["n_spectra"].sum()) if not dataset_rows.empty else 0,
                    "items": [
                        {
                            "class_label": row["class_label"],
                            "subclass_label": row["subclass_label"],
                            "n_spectra": int(row["n_spectra"]),
                        }
                        for row in dataset_rows.to_dict("records")
                    ],
                }
            )
        tree["GAIRA_EV"][bucket] = datasets

    for bucket, dataset_ids in SERUM_BUCKETS.items():
        bucket_rows = biosample_classes[biosample_classes["dataset_id"].isin(dataset_ids)].copy()
        datasets = []
        for dataset_id in dataset_ids:
            dataset_rows = bucket_rows[bucket_rows["dataset_id"] == dataset_id].copy()
            datasets.append(
                {
                    "dataset_id": dataset_id,
                    "display_name": _display_dataset(dataset_id),
                    "n_spectra": int(dataset_rows["n_spectra"].sum()) if not dataset_rows.empty else 0,
                    "items": [
                        {
                            "class_label": row["class_label"],
                            "subclass_label": row["subclass_label"],
                            "n_spectra": int(row["n_spectra"]),
                        }
                        for row in dataset_rows.to_dict("records")
                    ],
                }
            )
        tree["GAIRA_SERUM"][bucket] = datasets

    return tree


@lru_cache(maxsize=1)
def get_dataset_selector_options() -> dict[str, Any]:
    return {
        "grounding": {
            "Controlled grounding": [dataset for dataset in get_layer_tree()["GAIRA_GROUNDING"]["Controlled grounding"]],
            "Metabolite fingerprints": [dataset for dataset in get_layer_tree()["GAIRA_GROUNDING"]["Metabolite fingerprints"]],
        },
        "ev": get_layer_tree()["GAIRA_EV"],
        "serum": get_layer_tree()["GAIRA_SERUM"],
    }


@lru_cache(maxsize=1)
def get_rag_inventory() -> dict[str, dict[str, Any]]:
    with _connect() as con:
        support_docs = con.execute(
            """
            SELECT document_id, dataset_id, source_dataset_id, support_type, title, notes
            FROM grounding_support_documents
            ORDER BY dataset_id, title
            """
        ).fetchdf()
        support_chunks = con.execute(
            """
            SELECT document_id, dataset_id, section, chunk_text
            FROM grounding_support_chunks
            ORDER BY dataset_id, chunk_order
            """
        ).fetchdf()
        context_docs = con.execute(
            """
            SELECT document_id, intended_domain, context_type, title, notes, source_dataset_id
            FROM domain_context_documents
            ORDER BY intended_domain, title
            """
        ).fetchdf()
        context_chunks = con.execute(
            """
            SELECT document_id, intended_domain, section, chunk_text
            FROM domain_context_chunks
            ORDER BY intended_domain, chunk_order
            """
        ).fetchdf()
        knowledge_docs = con.execute(
            """
            SELECT source_id, section, chunk_text, chunk_order
            FROM knowledge_chunks
            WHERE dataset_id = 'raman_knowledge_core'
            ORDER BY source_id, chunk_order
            """
        ).fetchdf()

    def build_support(layer_name: str, dataset_filter: list[str] | None = None, domain_filter: str | None = None) -> dict[str, Any]:
        if dataset_filter is not None:
            docs_df = support_docs[support_docs["dataset_id"].isin(dataset_filter)].copy()
            chunks_df = support_chunks[support_chunks["dataset_id"].isin(dataset_filter)].copy()
        else:
            docs_df = pd.DataFrame(columns=support_docs.columns)
            chunks_df = pd.DataFrame(columns=support_chunks.columns)
        if domain_filter is not None:
            ctx_docs_df = context_docs[context_docs["intended_domain"] == domain_filter].copy()
            ctx_chunks_df = context_chunks[context_chunks["intended_domain"] == domain_filter].copy()
        else:
            ctx_docs_df = pd.DataFrame(columns=context_docs.columns)
            ctx_chunks_df = pd.DataFrame(columns=context_chunks.columns)

        rows = []
        for row in docs_df.to_dict("records"):
            snippet = chunks_df[chunks_df["document_id"] == row["document_id"]]["chunk_text"].astype(str).head(1)
            rows.append(
                {
                    "document_id": row["document_id"],
                    "title": row["title"] or row["document_id"],
                    "kind": row["support_type"] or "literature support",
                    "source_dataset_id": row["source_dataset_id"] or row["dataset_id"],
                    "snippet": snippet.iloc[0][:320] if not snippet.empty else str(row["notes"] or "")[:320],
                }
            )
        for row in ctx_docs_df.to_dict("records"):
            snippet = ctx_chunks_df[ctx_chunks_df["document_id"] == row["document_id"]]["chunk_text"].astype(str).head(1)
            rows.append(
                {
                    "document_id": row["document_id"],
                    "title": row["title"] or row["document_id"],
                    "kind": row["context_type"] or "context note",
                    "source_dataset_id": row["source_dataset_id"] or layer_name.lower(),
                    "snippet": snippet.iloc[0][:320] if not snippet.empty else str(row["notes"] or "")[:320],
                }
            )
        return {
            "doc_count": len(rows),
            "chunk_count": int(len(chunks_df) + len(ctx_chunks_df)),
            "documents": rows,
        }

    grounding_rows = build_support(
        "Grounding",
        dataset_filter=[
            "adenine_sers_control",
            "metabolite_sers63_support",
            "serum_ag_colloids_literature_grounding",
            "sers24_metabolite_support",
            "sers_fingerprint_workingpaper_support",
        ],
        domain_filter=None,
    )
    knowledge_groups = []
    for source_id, chunk_df in knowledge_docs.groupby("source_id", sort=False):
        knowledge_groups.append(
            {
                "document_id": str(source_id),
                "title": str(source_id).replace("_", " ").title(),
                "kind": "knowledge support",
                "source_dataset_id": "raman_knowledge_core",
                "snippet": str(chunk_df.iloc[0]["chunk_text"])[:320],
            }
        )
    grounding_rows["doc_count"] += len(knowledge_groups)
    grounding_rows["chunk_count"] += len(knowledge_docs)
    grounding_rows["documents"].extend(knowledge_groups)

    return {
        "Grounding": grounding_rows,
        "EV": build_support("EV", dataset_filter=["diabetes_ev_context_support", "shine_spectra_context_support"], domain_filter="ev"),
        "Serum": build_support(
            "Serum",
            dataset_filter=["liver_serum_literature_support", "serum_ag_colloids_literature_grounding"],
            domain_filter="serum",
        ),
    }


def _fetch_class_summary_row(domain: str, dataset_id: str, class_label: str, family_label: str | None = None) -> dict[str, Any]:
    with _connect() as con:
        if domain == "grounding":
            row = con.execute(
                """
                SELECT class_label, experiment_family, processing_version, mean_wavenumbers_json, mean_intensity_json, n_spectra
                FROM grounding_class_summary
                WHERE dataset_id = ? AND class_label = ? AND (? IS NULL OR experiment_family = ?)
                ORDER BY experiment_family
                LIMIT 1
                """,
                [dataset_id, class_label, family_label, family_label],
            ).fetchone()
            if row is None:
                raise ValueError(f"Missing grounding class summary for {dataset_id} / {class_label}")
            return {
                "dataset_id": dataset_id,
                "class_label": str(row[0]),
                "family_label": str(row[1]),
                "processing_version": str(row[2]),
                "x": _parse_json_array(row[3]),
                "y": _parse_json_array(row[4]),
                "n_spectra": int(row[5]),
            }
        row = con.execute(
            """
            SELECT class_label, subclass_label, processing_version, mean_wavenumbers_json, mean_intensity_json, n_spectra
            FROM biosample_class_summary
            WHERE dataset_id = ? AND class_label = ? AND (? IS NULL OR subclass_label = ?)
            ORDER BY subclass_label
            LIMIT 1
            """,
            [dataset_id, class_label, family_label, family_label],
        ).fetchone()
        if row is None:
            raise ValueError(f"Missing biosample class summary for {dataset_id} / {class_label}")
        return {
            "dataset_id": dataset_id,
            "class_label": str(row[0]),
            "family_label": str(row[1]),
            "processing_version": str(row[2]),
            "x": _parse_json_array(row[3]),
            "y": _parse_json_array(row[4]),
            "n_spectra": int(row[5]),
        }


def get_class_mean_spectrum(domain: str, dataset_id: str, class_label: str, family_label: str | None = None) -> dict[str, Any]:
    return _fetch_class_summary_row(domain, dataset_id, class_label, family_label)


def get_dataset_metadata(dataset_id: str) -> dict[str, Any]:
    tree = get_layer_tree()
    for layer in tree.values():
        for datasets in layer.values():
            for dataset in datasets:
                if dataset["dataset_id"] == dataset_id:
                    return dataset
    return {"dataset_id": dataset_id, "display_name": _display_dataset(dataset_id), "items": [], "n_spectra": 0}


def get_processed_pair_spectrum(db_path: Path, biosample_id: str) -> dict[str, Any]:
    with _connect(db_path) as con:
        row = con.execute(
            """
            SELECT p.wavenumbers_json, p.intensity_json, m.class_label, m.subclass_label, m.sample_id, m.replicate_id
            FROM biosample_processed_spectra p
            JOIN biosample_metadata m USING (biosample_id, dataset_id)
            WHERE p.biosample_id = ?
            ORDER BY p.processing_version
            LIMIT 1
            """,
            [biosample_id],
        ).fetchone()
    if row is None:
        raise ValueError(f"Missing processed biosample spectrum for {biosample_id}")
    return {
        "x": _parse_json_array(row[0]),
        "y": _parse_json_array(row[1]),
        "class_label": str(row[2]),
        "subclass_label": str(row[3]),
        "sample_id": str(row[4]),
        "replicate_id": str(row[5]),
    }
