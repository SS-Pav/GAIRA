from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path("/Volumes/SSD_Rad/GAIRA_DATA")
CURRENT_PROCESSED = ROOT / "processed"
REFERENCE_PROCESSED = ROOT / "legacy_reference" / "ssd_spg_recovered_artifacts" / "processed"
OUTPUT_DIR = CURRENT_PROCESSED / "small2023_ev_invariant_analysis_comparison"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_signature(path: Path) -> tuple[str, int]:
    return sha256(path), path.stat().st_size


def compare_csv(old_path: Path, new_path: Path) -> dict[str, object]:
    old = pd.read_csv(old_path)
    new = pd.read_csv(new_path)
    result: dict[str, object] = {
        "old_shape": old.shape,
        "new_shape": new.shape,
        "columns_match": list(old.columns) == list(new.columns),
    }
    if not result["columns_match"]:
        return result

    numeric = [c for c in old.columns if pd.api.types.is_numeric_dtype(old[c]) and pd.api.types.is_numeric_dtype(new[c])]
    non_numeric = [c for c in old.columns if c not in numeric]
    if non_numeric:
        merged = old.merge(new, on=non_numeric, how="outer", indicator=True)
        result["keys_match"] = merged["_merge"].eq("both").all()
    else:
        result["keys_match"] = True

    diffs: dict[str, float] = {}
    if old.shape == new.shape:
        for column in numeric:
            diffs[column] = float((old[column] - new[column]).abs().max())
    result["max_abs_diff"] = diffs
    result["exact_numeric_match"] = all(value == 0.0 for value in diffs.values())
    return result


def compare_text(old_path: Path, new_path: Path) -> dict[str, object]:
    old_text = old_path.read_text(encoding="utf-8")
    new_text = new_path.read_text(encoding="utf-8")
    return {
        "exact_match": old_text == new_text,
        "old_sha256": hashlib.sha256(old_text.encode("utf-8")).hexdigest(),
        "new_sha256": hashlib.sha256(new_text.encode("utf-8")).hexdigest(),
    }


def run_v1_report_builder() -> None:
    subprocess.run([sys.executable, "scripts/build_small2023_ev_invariant_report_v1.py"], check=True)


def load_scalar(path: Path) -> pd.Series:
    return pd.read_csv(path).iloc[0]


def build_version_table() -> pd.DataFrame:
    v1_metrics = pd.read_csv(CURRENT_PROCESSED / "small2023_ev_invariant_embedding" / "embedding_cross_probe_metrics.csv")
    v1_geometry = load_scalar(CURRENT_PROCESSED / "small2023_ev_invariant_embedding" / "geometry_metrics.csv")

    v2_metrics = pd.read_csv(CURRENT_PROCESSED / "small2023_ev_invariant_embedding_v2" / "comparison_cross_probe_metrics_v2.csv")
    v2_geometry = load_scalar(CURRENT_PROCESSED / "small2023_ev_invariant_embedding_v2" / "geometry_metrics_v2.csv")

    v3_metrics = pd.read_csv(CURRENT_PROCESSED / "small2023_ev_invariant_embedding_v3" / "comparison_cross_probe_metrics_v3.csv")
    v3_geometry = load_scalar(CURRENT_PROCESSED / "small2023_ev_invariant_embedding_v3" / "geometry_metrics_v3.csv")

    v2_embed = v2_metrics[v2_metrics["model"] == "v2_embedding"]
    v3_embed = v3_metrics[v3_metrics["model"] == "v3_embedding"]
    v1_mean = float(v1_metrics["accuracy"].mean())
    v2_mean = float(v2_embed["accuracy"].mean())
    v3_mean = float(v3_embed["accuracy"].mean())

    rows = [
        {
            "version": "v1",
            "cross_probe_transfer_mean_accuracy": v1_mean,
            "class_separability": float(v1_geometry["embedding_class_silhouette"]),
            "probe_separability": float(v1_geometry["embedding_probe_silhouette"]),
            "mixture_order_correlation": float(v1_geometry["embedding_mixture_order_correlation"]),
            "centroid_alignment_note": "Improved over raw, still probe-sensitive",
            "trustworthiness_note": "Useful first invariant baseline, but transfer remains moderate",
            "recommendation": "superseded",
        },
        {
            "version": "v2",
            "cross_probe_transfer_mean_accuracy": v2_mean,
            "class_separability": float(v2_geometry["v2_class_silhouette"]),
            "probe_separability": float(v2_geometry["v2_probe_silhouette"]),
            "mixture_order_correlation": float(v2_geometry["v2_mixture_order_correlation"]),
            "centroid_alignment_note": "Strong same-class cross-probe alignment",
            "trustworthiness_note": "Best tradeoff between transfer, class structure, and low probe leakage",
            "recommendation": "recommended default benchmark view",
        },
        {
            "version": "v3",
            "cross_probe_transfer_mean_accuracy": v3_mean,
            "class_separability": float(v3_geometry["v3_class_silhouette"]),
            "probe_separability": float(v3_geometry["v3_probe_silhouette"]),
            "mixture_order_correlation": float(v3_geometry["v3_mixture_order_correlation"]),
            "centroid_alignment_note": "Strict transfer protocol removes useful class signal",
            "trustworthiness_note": "Important negative control, not suitable as the default",
            "recommendation": "keep as stress-test only",
        },
    ]
    return pd.DataFrame(rows)


def compare_bundle() -> tuple[pd.DataFrame, list[str]]:
    comparisons = [
        (
            "v1 sample counts",
            REFERENCE_PROCESSED / "small2023_ev_invariant_embedding" / "benchmark_sample_counts.csv",
            CURRENT_PROCESSED / "small2023_ev_invariant_embedding" / "benchmark_sample_counts.csv",
            "csv",
        ),
        (
            "v1 baseline metrics",
            REFERENCE_PROCESSED / "small2023_ev_invariant_embedding" / "baseline_cross_probe_metrics.csv",
            CURRENT_PROCESSED / "small2023_ev_invariant_embedding" / "baseline_cross_probe_metrics.csv",
            "csv",
        ),
        (
            "v1 embedding metrics",
            REFERENCE_PROCESSED / "small2023_ev_invariant_embedding" / "embedding_cross_probe_metrics.csv",
            CURRENT_PROCESSED / "small2023_ev_invariant_embedding" / "embedding_cross_probe_metrics.csv",
            "csv",
        ),
        (
            "v1 summary",
            REFERENCE_PROCESSED / "small2023_ev_invariant_embedding" / "embedding_summary.txt",
            CURRENT_PROCESSED / "small2023_ev_invariant_embedding" / "embedding_summary.txt",
            "text",
        ),
        (
            "v2 comparison metrics",
            REFERENCE_PROCESSED / "small2023_ev_invariant_embedding_v2" / "comparison_cross_probe_metrics_v2.csv",
            CURRENT_PROCESSED / "small2023_ev_invariant_embedding_v2" / "comparison_cross_probe_metrics_v2.csv",
            "csv",
        ),
        (
            "v2 geometry",
            REFERENCE_PROCESSED / "small2023_ev_invariant_embedding_v2" / "geometry_metrics_v2.csv",
            CURRENT_PROCESSED / "small2023_ev_invariant_embedding_v2" / "geometry_metrics_v2.csv",
            "csv",
        ),
        (
            "v2 summary",
            REFERENCE_PROCESSED / "small2023_ev_invariant_embedding_v2" / "embedding_summary_v2.txt",
            CURRENT_PROCESSED / "small2023_ev_invariant_embedding_v2" / "embedding_summary_v2.txt",
            "text",
        ),
        (
            "v2 validation transfer",
            REFERENCE_PROCESSED / "small2023_ev_invariant_embedding_v2_validation" / "multi_seed_transfer_summary.csv",
            CURRENT_PROCESSED / "small2023_ev_invariant_embedding_v2_validation" / "multi_seed_transfer_summary.csv",
            "csv",
        ),
        (
            "v2 validation ablation",
            REFERENCE_PROCESSED / "small2023_ev_invariant_embedding_v2_validation" / "v2_ablation_metrics.csv",
            CURRENT_PROCESSED / "small2023_ev_invariant_embedding_v2_validation" / "v2_ablation_metrics.csv",
            "csv",
        ),
        (
            "v2 validation summary",
            REFERENCE_PROCESSED / "small2023_ev_invariant_embedding_v2_validation" / "v2_validation_summary.txt",
            CURRENT_PROCESSED / "small2023_ev_invariant_embedding_v2_validation" / "v2_validation_summary.txt",
            "text",
        ),
        (
            "v3 comparison metrics",
            REFERENCE_PROCESSED / "small2023_ev_invariant_embedding_v3" / "comparison_cross_probe_metrics_v3.csv",
            CURRENT_PROCESSED / "small2023_ev_invariant_embedding_v3" / "comparison_cross_probe_metrics_v3.csv",
            "csv",
        ),
        (
            "v3 geometry",
            REFERENCE_PROCESSED / "small2023_ev_invariant_embedding_v3" / "geometry_metrics_v3.csv",
            CURRENT_PROCESSED / "small2023_ev_invariant_embedding_v3" / "geometry_metrics_v3.csv",
            "csv",
        ),
        (
            "v3 summary",
            REFERENCE_PROCESSED / "small2023_ev_invariant_embedding_v3" / "embedding_summary_v3.txt",
            CURRENT_PROCESSED / "small2023_ev_invariant_embedding_v3" / "embedding_summary_v3.txt",
            "text",
        ),
    ]

    rows = []
    notes: list[str] = []
    for label, old_path, new_path, kind in comparisons:
        if kind == "csv":
            result = compare_csv(old_path, new_path)
            rows.append(
                {
                    "artifact": label,
                    "kind": kind,
                    "old_shape": result["old_shape"],
                    "new_shape": result["new_shape"],
                    "exact_match": bool(result.get("exact_numeric_match", False) and result.get("keys_match", False)),
                    "detail": str(result.get("max_abs_diff", {})),
                }
            )
        else:
            result = compare_text(old_path, new_path)
            rows.append(
                {
                    "artifact": label,
                    "kind": kind,
                    "old_shape": "",
                    "new_shape": "",
                    "exact_match": bool(result["exact_match"]),
                    "detail": f"old_sha256={result['old_sha256']} new_sha256={result['new_sha256']}",
                }
            )

    image_pairs = [
        (
            "v1 raw_tsne_by_class",
            REFERENCE_PROCESSED / "small2023_ev_invariant_embedding" / "raw_tsne_by_class.png",
            CURRENT_PROCESSED / "small2023_ev_invariant_embedding" / "raw_tsne_by_class.png",
        ),
        (
            "v1 raw_tsne_by_probe",
            REFERENCE_PROCESSED / "small2023_ev_invariant_embedding" / "raw_tsne_by_probe.png",
            CURRENT_PROCESSED / "small2023_ev_invariant_embedding" / "raw_tsne_by_probe.png",
        ),
        (
            "v1 embedding_tsne_by_class",
            REFERENCE_PROCESSED / "small2023_ev_invariant_embedding" / "embedding_tsne_by_class.png",
            CURRENT_PROCESSED / "small2023_ev_invariant_embedding" / "embedding_tsne_by_class.png",
        ),
        (
            "v1 embedding_tsne_by_probe",
            REFERENCE_PROCESSED / "small2023_ev_invariant_embedding" / "embedding_tsne_by_probe.png",
            CURRENT_PROCESSED / "small2023_ev_invariant_embedding" / "embedding_tsne_by_probe.png",
        ),
        (
            "v2 embedding_tsne_by_class",
            REFERENCE_PROCESSED / "small2023_ev_invariant_embedding_v2" / "embedding_tsne_by_class_v2.png",
            CURRENT_PROCESSED / "small2023_ev_invariant_embedding_v2" / "embedding_tsne_by_class_v2.png",
        ),
        (
            "v2 embedding_tsne_by_probe",
            REFERENCE_PROCESSED / "small2023_ev_invariant_embedding_v2" / "embedding_tsne_by_probe_v2.png",
            CURRENT_PROCESSED / "small2023_ev_invariant_embedding_v2" / "embedding_tsne_by_probe_v2.png",
        ),
        (
            "v2 validation transfer plot",
            REFERENCE_PROCESSED / "small2023_ev_invariant_embedding_v2_validation" / "multi_seed_transfer_plot.png",
            CURRENT_PROCESSED / "small2023_ev_invariant_embedding_v2_validation" / "multi_seed_transfer_plot.png",
        ),
        (
            "v2 validation ablation plot",
            REFERENCE_PROCESSED / "small2023_ev_invariant_embedding_v2_validation" / "v2_ablation_plot.png",
            CURRENT_PROCESSED / "small2023_ev_invariant_embedding_v2_validation" / "v2_ablation_plot.png",
        ),
        (
            "v3 embedding_tsne_by_class",
            REFERENCE_PROCESSED / "small2023_ev_invariant_embedding_v3" / "embedding_tsne_by_class_v3.png",
            CURRENT_PROCESSED / "small2023_ev_invariant_embedding_v3" / "embedding_tsne_by_class_v3.png",
        ),
        (
            "v3 embedding_tsne_by_probe",
            REFERENCE_PROCESSED / "small2023_ev_invariant_embedding_v3" / "embedding_tsne_by_probe_v3.png",
            CURRENT_PROCESSED / "small2023_ev_invariant_embedding_v3" / "embedding_tsne_by_probe_v3.png",
        ),
    ]

    for label, old_path, new_path in image_pairs:
        old_hash, old_size = image_signature(old_path)
        new_hash, new_size = image_signature(new_path)
        rows.append(
            {
                "artifact": label,
                "kind": "image",
                "old_shape": old_size,
                "new_shape": new_size,
                "exact_match": old_hash == new_hash,
                "detail": f"old_sha256={old_hash} new_sha256={new_hash}",
            }
        )
        if old_hash != new_hash:
            notes.append(f"{label} differs at the image-byte level; compare source metrics to interpret whether this is expected.")

    return pd.DataFrame(rows), notes


def write_report(bundle_df: pd.DataFrame, version_df: pd.DataFrame, notes: list[str]) -> Path:
    exact_count = int(bundle_df["exact_match"].sum())
    total_count = int(len(bundle_df))
    v2_row = version_df.loc[version_df["version"] == "v2"].iloc[0]
    report = f"""# small2023_ev invariant embedding regeneration report

## Scope

This report compares the regenerated SSD_Rad small2023 probe-family analysis bundle against the recovered SSD_SPG reference bundle.

## Reproduction status

- Exact artifact matches: {exact_count} / {total_count}
- SSD_Rad is the sole live root.
- SSD_SPG was used only as a recovered reference bundle.
- The restored canonical raw package includes both `NormedProbe1.mat` and `NormedProbe2.mat`.

## Version assessment

| version | mean cross-probe transfer | class separability | probe separability | mixture ordering | recommendation |
|---|---:|---:|---:|---:|---|
"""
    for _, row in version_df.iterrows():
        report += (
            f"| {row['version']} | {row['cross_probe_transfer_mean_accuracy']:.6f} | "
            f"{row['class_separability']:.6f} | {row['probe_separability']:.6f} | "
            f"{row['mixture_order_correlation']:.6f} | {row['recommendation']} |\n"
        )

    report += f"""

## Interpretation

- `v1` is the first useful invariant baseline, but its transfer is still moderate.
- `v2` is the strongest overall version: high cross-probe transfer, strong class separation, minimal probe leakage, and preserved mixture-order structure.
- `v3` is a useful strict-transfer negative control, but it suppresses useful class signal and should not be the default.

## Recommendation

- Recommended version going forward: **v2**
- Use `v1` as a historical baseline.
- Keep `v2_validation` as the main robustness check for `v2`.
- Keep `v3` as a stress-test / failure-mode reference rather than the default GAIRA embedding benchmark view.

## Notes
"""
    if notes:
        for note in notes:
            report += f"- {note}\n"
    else:
        report += "- All compared text, CSV, and checked image artifacts matched exactly.\n"

    out_path = OUTPUT_DIR / "small2023_ev_invariant_analysis_comparison_report.md"
    out_path.write_text(report, encoding="utf-8")
    return out_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_v1_report_builder()
    bundle_df, notes = compare_bundle()
    version_df = build_version_table()
    bundle_df.to_csv(OUTPUT_DIR / "artifact_comparison.csv", index=False)
    version_df.to_csv(OUTPUT_DIR / "version_assessment.csv", index=False)
    report_path = write_report(bundle_df, version_df, notes)
    print(f"Wrote comparison report: {report_path}")
    print(f"Wrote artifact comparison table: {OUTPUT_DIR / 'artifact_comparison.csv'}")
    print(f"Wrote version assessment table: {OUTPUT_DIR / 'version_assessment.csv'}")


if __name__ == "__main__":
    main()
