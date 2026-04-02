from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from gaira.demo.ev_analysis_utils import THEME_ORDER as LEGACY_THEME_ORDER, compute_theme_profiles


LEGACY_PURINE_THEME = "purine_metabolite_associated"
MASTER_THEME_ORDER = [
    "protein_peptide_associated",
    "oxidative_redox_associated",
    "purine_associated",
    "general_metabolite_associated",
    "lipid_membrane_associated",
    "carbohydrate_associated",
    "nucleic_acid_associated",
    "serum_matrix_associated",
]
MASTER_THEME_COLORS = {
    "protein_peptide_associated": "#3e6ea1",
    "oxidative_redox_associated": "#7a4b9d",
    "purine_associated": "#cf7a2d",
    "general_metabolite_associated": "#d9a441",
    "lipid_membrane_associated": "#2c8c69",
    "carbohydrate_associated": "#c24d67",
    "nucleic_acid_associated": "#8d5b2a",
    "serum_matrix_associated": "#8f9499",
    "unresolved": "#b6bcc4",
}
PURINE_PATTERNS = [
    r"\badenine\b",
    r"\bguanine\b",
    r"\bxanthine\b",
    r"\bhypox",
    r"\buric[_\s-]?acid\b",
    r"\burate\b",
    r"\binosine\b",
    r"\bpurine\b",
    r"\bua(?:bound|iso)?\b",
]


def split_legacy_theme_label(
    theme: str,
    *,
    label_optional: str = "",
    semantic_group: str = "",
    proposed_harmonized_anchor: str = "",
    notes: str = "",
    sample_key: str = "",
    dataset_id: str = "",
) -> str:
    if str(theme) != LEGACY_PURINE_THEME:
        return str(theme)
    text = " ".join(
        [
            str(label_optional),
            str(semantic_group),
            str(proposed_harmonized_anchor),
            str(notes),
            str(sample_key),
            str(dataset_id),
        ]
    ).lower()
    for pattern in PURINE_PATTERNS:
        if re.search(pattern, text):
            return "purine_associated"
    return "general_metabolite_associated"


def split_grounding_theme_table(theme_table: pd.DataFrame) -> pd.DataFrame:
    split = theme_table.copy()
    split["grounding_theme_split"] = split.apply(
        lambda row: split_legacy_theme_label(
            str(row.get("grounding_theme", "")),
            label_optional=str(row.get("label_optional", "")),
            semantic_group=str(row.get("semantic_group", "")),
            proposed_harmonized_anchor=str(row.get("proposed_harmonized_anchor", "")),
            notes=str(row.get("notes", "")),
            sample_key=str(row.get("sample_key", "")),
            dataset_id=str(row.get("dataset_id", "")),
        ),
        axis=1,
    )
    return split


def _empty_master_profile(n_rows: int) -> pd.DataFrame:
    return pd.DataFrame({theme: np.zeros(n_rows, dtype=float) for theme in MASTER_THEME_ORDER})


def split_theme_profile_frame(profile_df: pd.DataFrame) -> pd.DataFrame:
    split = _empty_master_profile(len(profile_df))
    for theme in profile_df.columns:
        if theme not in LEGACY_THEME_ORDER:
            continue
        if theme == LEGACY_PURINE_THEME:
            continue
        if theme in split.columns:
            split[theme] = profile_df[theme].to_numpy(dtype=float)
    if LEGACY_PURINE_THEME in profile_df.columns:
        # Default split for already-aggregated legacy profiles: keep the old weight but place it in the
        # general metabolite channel unless a separate per-grounding split was available upstream.
        split["general_metabolite_associated"] = profile_df[LEGACY_PURINE_THEME].to_numpy(dtype=float)
    return split


def compute_split_theme_profiles(
    query_embeddings: np.ndarray,
    grounding_embeddings: np.ndarray,
    grounding_theme_table: pd.DataFrame,
    *,
    top_k: int,
) -> pd.DataFrame:
    split_grounding = split_grounding_theme_table(grounding_theme_table)
    split_profiles = compute_theme_profiles(
        query_embeddings,
        grounding_embeddings,
        split_grounding["grounding_theme_split"].astype(str).to_numpy(),
        top_k=top_k,
    )
    for theme in MASTER_THEME_ORDER:
        if theme not in split_profiles.columns:
            split_profiles[theme] = 0.0
    return split_profiles[MASTER_THEME_ORDER].copy()


def split_existing_composition_frame(frame: pd.DataFrame) -> pd.DataFrame:
    split = frame.copy()
    if LEGACY_PURINE_THEME in split.columns:
        split["purine_associated"] = 0.0
        split["general_metabolite_associated"] = split[LEGACY_PURINE_THEME].to_numpy(dtype=float)
        split = split.drop(columns=[LEGACY_PURINE_THEME])
    for theme in MASTER_THEME_ORDER:
        if theme not in split.columns:
            split[theme] = 0.0
    ordered_cols = [col for col in split.columns if col not in LEGACY_THEME_ORDER]
    front = [col for col in split.columns if col not in MASTER_THEME_ORDER]
    return split[front + MASTER_THEME_ORDER]


def summarize_theme_split(split_grounding: pd.DataFrame) -> pd.DataFrame:
    purine_subset = split_grounding[split_grounding["grounding_theme"] == LEGACY_PURINE_THEME].copy()
    if purine_subset.empty:
        return pd.DataFrame(columns=["split_theme", "record_count"])
    counts = (
        purine_subset.groupby("grounding_theme_split", as_index=False)
        .size()
        .rename(columns={"size": "record_count", "grounding_theme_split": "split_theme"})
        .sort_values("record_count", ascending=False)
        .reset_index(drop=True)
    )
    return counts


def write_theme_split_note(output_path: Path, split_grounding: pd.DataFrame) -> None:
    counts = summarize_theme_split(split_grounding)
    lines = [
        "# Theme Split Note",
        "",
        "The legacy `purine_metabolite_associated` bucket has been split into two auditable master channels for the v8 master workflow:",
        "- `purine_associated`: grounding records whose controlled labels or anchors explicitly reference purine-like compounds such as adenine, xanthine, hypoxanthine, inosine, uric acid, urate, or guanine.",
        "- `general_metabolite_associated`: the remaining legacy metabolite-like records that were previously merged into the same coarse bucket.",
        "",
        "This is still a broad, compatibility-preserving split. It is more honest than leaving purines silently merged, but it is not yet a final mechanistic ontology.",
        "",
        "Split counts from the current grounding table:",
    ]
    if counts.empty:
        lines.append("- no legacy purine/metabolite records were found")
    else:
        lines.extend([f"- {row.split_theme}: {int(row.record_count):,}" for row in counts.itertuples(index=False)])
    (output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
