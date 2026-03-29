from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    import sys

    sys.path.insert(0, str(project_root / "src"))
    from gaira.embedding.runtime import REMOTE_OUTPUT_ROOT, add_common_io_args

    parser = argparse.ArgumentParser(description="Evaluate frozen linear probes on spectral embeddings.")
    add_common_io_args(parser, default_run_name="embedding_v2", default_root=REMOTE_OUTPUT_ROOT)
    parser.add_argument("--min-class-count", type=int, default=3, help="Minimum samples per class for a task.")
    parser.add_argument("--report-dir", default=None, help="Directory to write probe outputs. Defaults to output-dir.")
    return parser.parse_args()


def family_label(metadata_df: pd.DataFrame) -> pd.Series:
    mapping = {
        "adenine_sers_control": "grounding_analyte",
        "metabolite_sers63_support": "grounding_analyte",
        "amino_acid_raman_grounding": "grounding_analyte",
        "small2023_ev": "ev_general",
        "shine_ev_sers": "ev_disease_or_stress",
        "diabetes_plasma_ev_sers": "ev_disease_or_stress",
        "covid_serum_raman": "serum_general",
        "serum_protocol_comparison": "serum_general",
        "serum_ag_colloids": "serum_general",
        "cspp_serum": "serum_general",
        "ergothioneine_serum": "serum_general",
        "cca_hcc_lm_serum_sers": "serum_liver_hepatobiliary",
    }
    return metadata_df["dataset_id"].map(mapping).fillna(metadata_df["sample_type"].astype(str))


def run_probe(
    embeddings: np.ndarray,
    labels: pd.Series,
    task_name: str,
    *,
    min_class_count: int,
) -> dict[str, object] | None:
    y = labels.fillna("").astype(str)
    valid_mask = y != ""
    y = y[valid_mask]
    if y.nunique() < 2:
        return None

    value_counts = y.value_counts()
    keep_labels = value_counts[value_counts >= min_class_count].index
    keep_mask = valid_mask.copy()
    keep_mask[valid_mask] = y.isin(keep_labels).to_numpy()
    filtered_y = labels[keep_mask].astype(str)
    if filtered_y.nunique() < 2:
        return None

    X = embeddings[keep_mask.to_numpy()]
    y_array = filtered_y.to_numpy()
    min_count = filtered_y.value_counts().min()
    if min_count < 2:
        return None
    n_splits = min(5, int(min_count))
    if n_splits < 2:
        return None

    clf = LogisticRegression(max_iter=2000)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=7)
    preds = cross_val_predict(clf, X, y_array, cv=cv)
    return {
        "task_name": task_name,
        "n_samples": int(len(y_array)),
        "n_classes": int(filtered_y.nunique()),
        "accuracy": float(accuracy_score(y_array, preds)),
        "macro_f1": float(f1_score(y_array, preds, average="macro")),
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    import sys

    sys.path.insert(0, str(project_root / "src"))
    from gaira.embedding.runtime import resolve_output_dir

    args = parse_args()
    output_dir = resolve_output_dir(args)
    report_dir = Path(args.report_dir).expanduser().resolve() if args.report_dir else output_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    embeddings = np.load(output_dir / "embeddings.npy")
    metadata_df = pd.read_csv(output_dir / "metadata.csv")
    if "family_label" not in metadata_df.columns:
        metadata_df["family_label"] = family_label(metadata_df)

    tasks: list[tuple[str, pd.Series]] = [
        ("sample_type", metadata_df["sample_type"]),
        ("family_label", metadata_df["family_label"]),
    ]

    cca_mask = metadata_df["dataset_id"].astype(str) == "cca_hcc_lm_serum_sers"
    tasks.append(("cca_hcc_lm_serum_sers_class", metadata_df.loc[cca_mask, "label_optional"]))

    ev_mask = metadata_df["dataset_id"].astype(str).isin(["small2023_ev", "shine_ev_sers", "diabetes_plasma_ev_sers"])
    ev_df = metadata_df.loc[ev_mask].copy()
    ev_df["ev_dataset_task"] = ev_df["dataset_id"].astype(str)
    tasks.append(("ev_dataset_family", ev_df["ev_dataset_task"]))

    aa_mask = metadata_df["dataset_id"].astype(str).isin(
        ["adenine_sers_control", "metabolite_sers63_support", "amino_acid_raman_grounding"]
    )
    analyte_df = metadata_df.loc[aa_mask].copy()
    tasks.append(("grounding_label_family", analyte_df["label_optional"]))

    rows: list[dict[str, object]] = []
    base_index = metadata_df.index
    for task_name, label_series in tasks:
        aligned = pd.Series("", index=base_index, dtype=object)
        aligned.loc[label_series.index] = label_series.astype(str)
        result = run_probe(embeddings, aligned, task_name, min_class_count=args.min_class_count)
        if result is not None:
            rows.append(result)

    probe_df = pd.DataFrame(rows)
    probe_df.to_csv(report_dir / "probe_metrics.csv", index=False)
    report = textwrap.dedent(
        f"""
        Frozen probe report

        Tasks evaluated:

        {probe_df.to_string(index=False) if not probe_df.empty else '_empty_'}

        Notes:
        - Probes are trained only on frozen embeddings.
        - Tasks are limited to coherent label spaces; no fake cross-dataset class task was evaluated.
        - Higher accuracy and macro-F1 indicate more usable downstream signal in the learned geometry.
        """
    )
    (report_dir / "probe_report.md").write_text(report, encoding="utf-8")
    print(f"Saved probe metrics: {report_dir / 'probe_metrics.csv'}")


if __name__ == "__main__":
    main()
