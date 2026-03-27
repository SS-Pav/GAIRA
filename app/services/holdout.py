from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from gaira.inference import HCC_SERUM_PROCESSING_VERSION, load_serum_class_mean_query

from app.services.catalog import get_demo_paths, get_processed_pair_spectrum


@lru_cache(maxsize=1)
def _load_calibration_tables() -> dict[str, pd.DataFrame]:
    paths = get_demo_paths()
    base = paths["hcc_calibration_dir"]
    return {
        "sample_summary": pd.read_csv(base / "raw_outputs" / "hcc_holdout_calibrated_sample_summary.csv"),
        "shifted_themes": pd.read_csv(base / "tables" / "hcc_holdout_shifted_themes_before_after.csv"),
        "shared_background": pd.read_csv(base / "tables" / "hcc_holdout_shared_background_summary.csv"),
        "differential_evidence": pd.read_csv(base / "tables" / "hcc_holdout_differential_evidence_summary.csv"),
        "metrics": pd.read_csv(base / "tables" / "hcc_holdout_calibration_before_after_metrics.csv"),
    }


def get_holdout_banner_metrics() -> dict[str, Any]:
    metrics_df = _load_calibration_tables()["metrics"]
    metrics = {}
    for row in metrics_df.to_dict("records"):
        if "metric_value" in row:
            metrics[row["metric_name"]] = row["metric_value"]
        else:
            metrics[row["metric_name"]] = row.get("after", row.get("before", 0.0))
    return metrics


def _select_representative_row(class_label: str) -> pd.Series:
    sample_df = _load_calibration_tables()["sample_summary"]
    subset = sample_df[sample_df["class_label"] == class_label].copy()
    if subset.empty:
        raise ValueError(f"Missing holdout class {class_label}")
    subset = subset.sort_values(["mean_comparison_positive_confidence", "mean_differential_signal"], ascending=False)
    return subset.iloc[0]


def get_holdout_pair(mode: str = "representative pair") -> dict[str, Any]:
    paths = get_demo_paths()
    hcc_eval_db: Path = paths["hcc_eval_db"]
    if mode == "class-mean pair":
        ctr = load_serum_class_mean_query(hcc_eval_db, "hcc_serum", "CTR", "released_txt_archive", HCC_SERUM_PROCESSING_VERSION)
        h0t = load_serum_class_mean_query(hcc_eval_db, "hcc_serum", "H0T", "released_txt_archive", HCC_SERUM_PROCESSING_VERSION)
        left = {
            "label": "CTR class mean",
            "class_label": "CTR",
            "x": ctr.spectrum_query.x.tolist(),
            "y": ctr.spectrum_query.y.tolist(),
            "summary_row": None,
        }
        right = {
            "label": "H0T class mean",
            "class_label": "H0T",
            "x": h0t.spectrum_query.x.tolist(),
            "y": h0t.spectrum_query.y.tolist(),
            "summary_row": None,
        }
    else:
        ctr_row = _select_representative_row("CTR")
        h0t_row = _select_representative_row("H0T")
        ctr_spec = get_processed_pair_spectrum(hcc_eval_db, str(ctr_row["query_id"]))
        h0t_spec = get_processed_pair_spectrum(hcc_eval_db, str(h0t_row["query_id"]))
        left = {
            "label": "CTR representative",
            "class_label": "CTR",
            "x": ctr_spec["x"],
            "y": ctr_spec["y"],
            "summary_row": ctr_row.to_dict(),
        }
        right = {
            "label": "H0T representative",
            "class_label": "H0T",
            "x": h0t_spec["x"],
            "y": h0t_spec["y"],
            "summary_row": h0t_row.to_dict(),
        }

    shifted = _load_calibration_tables()["shifted_themes"]
    after_shifted = shifted[shifted["stage"] == "after"].copy()
    after_shifted["h0t_minus_ctr"] = -after_shifted["difference_a_minus_b"]
    enriched_h0t = after_shifted.sort_values("h0t_minus_ctr", ascending=False).head(3)
    enriched_ctr = after_shifted.sort_values("h0t_minus_ctr", ascending=True).head(3)
    shared = _load_calibration_tables()["shared_background"].sort_values("shared_background_score", ascending=False).head(5)
    evidence = _load_calibration_tables()["differential_evidence"].head(12)
    top_shared = str(shared.iloc[0]["theme_name"]).replace("_associated", "").replace("_", " ") if not shared.empty else "shared serum background"
    top_h0t = str(enriched_h0t.iloc[0]["theme_name"]).replace("_associated", "").replace("_", " ") if not enriched_h0t.empty else "shifted signal"
    top_ctr = str(enriched_ctr.iloc[0]["theme_name"]).replace("_associated", "").replace("_", " ") if not enriched_ctr.empty else "counter-shifted signal"
    summary_text = (
        f"GAIRAM v1 separates broad shared serum background from modest differential structure. "
        f"Shared background is dominated by {top_shared}, while the strongest H0T-shifted signal is {top_h0t} "
        f"and the strongest CTR-relative shift is {top_ctr}. This remains calibrated differential interpretation rather than diagnosis."
    )

    return {
        "left": left,
        "right": right,
        "shared_background": shared,
        "h0t_shifted": enriched_h0t,
        "ctr_shifted": enriched_ctr,
        "differential_evidence": evidence,
        "summary_text": summary_text,
    }
