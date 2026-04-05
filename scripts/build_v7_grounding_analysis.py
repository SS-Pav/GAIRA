#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_RUN_DIR = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/embedding_v7_anchor_gpu_run1")
DEFAULT_EVAL_DIR = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/embedding_eval_v2/embedding_v7_anchor_gpu_run1_eval_v2")
DEFAULT_CLUSTER_DIR = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/embedding_cluster_analysis_v7")
DEFAULT_ANCHOR_TABLE = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/embedding_anchor_audit/embedding_anchor_table_v1.csv")
DEFAULT_OUTPUT_DIR = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/embedding_grounding_analysis_v7")

THEME_ORDER = [
    "nucleic_acid_associated",
    "purine_metabolite_associated",
    "protein_peptide_associated",
    "lipid_membrane_associated",
    "carbohydrate_associated",
    "oxidative_redox_associated",
    "serum_matrix_associated",
    "grounding_reference_unresolved",
]

KEYWORD_THEME_MAP = {
    "nucleic_acid_associated": [
        "dna",
        "rna",
        "adenine",
        "ade",
        "guan",
        "gua",
        "urac",
        "ura",
        "nucle",
        "pterin",
        "cyclic_amp",
    ],
    "purine_metabolite_associated": [
        "adenine",
        "ade",
        "hypox",
        "xanth",
        "uric",
        "ua",
        "urea",
        "ure",
        "creat",
        "caffeine",
        "dopamine",
        "histamine",
        "kynurenine",
        "trypt",
        "nicot",
        "purine",
        "methy",
        "pipec",
        "octopamine",
        "mandelic",
    ],
    "protein_peptide_associated": [
        "alb",
        "ser",
        "pro",
        "phe",
        "met",
        "leu",
        "glu",
        "gly",
        "arg",
        "asp",
        "his",
        "val",
        "ala",
        "thy",
        "trp",
        "tyr",
        "ile",
        "cys",
        "lys",
        "homocys",
        "glutath",
        "cytochrome",
        "peptide",
        "protein",
        "cyst",
        "spermid",
        "agmatine",
    ],
    "lipid_membrane_associated": [
        "oleic",
        "stearic",
        "triolein",
        "chol",
        "lipid",
        "membrane",
        "accoa",
        "coa",
        "pep",
    ],
    "carbohydrate_associated": [
        "gluc",
        "glucose",
        "fruct",
        "galact",
        "glycogen",
        "glycerol",
        "mann",
        "ribo",
        "lact",
        "carbo",
        "dfruct6p",
        "phinositol",
        "malic acid",
        "havuc",
    ],
    "oxidative_redox_associated": [
        "ergo",
        "ergoth",
        "asc",
        "glutath",
        "homocys",
        "homocyst",
        "cyste",
        "seleno",
        "biliverdin",
        "redox",
        "oxid",
        "lipoamide",
        "dihydrofolate",
        "tetrahydrofolate",
        "riboflavin",
        "thiamine",
        "vitaminb12",
    ],
    "serum_matrix_associated": [
        "hsa",
        "filter",
        "bound",
        "free",
        "serum",
        "matrix",
        "colloid",
        "background",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build EV cluster grounding analysis for GAIRAM v7.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--cluster-dir", type=Path, default=DEFAULT_CLUSTER_DIR)
    parser.add_argument("--anchor-table", type=Path, default=DEFAULT_ANCHOR_TABLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-k-grounding", type=int, default=12)
    parser.add_argument("--top-k-member-grounding", type=int, default=6)
    parser.add_argument("--representative-members", type=int, default=3)
    return parser.parse_args()


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_inputs(args: argparse.Namespace) -> dict[str, pd.DataFrame | np.ndarray]:
    embeddings = np.load(args.run_dir / "embeddings.npy")
    metadata = pd.read_csv(args.run_dir / "metadata.csv")
    projection = pd.read_csv(args.eval_dir / "embedding_projection_v2.csv")
    cluster_assignments = pd.read_csv(args.cluster_dir / "cluster_assignments.csv")
    cluster_summary = pd.read_csv(args.cluster_dir / "cluster_summary.csv")
    cluster_interpretation = pd.read_csv(args.cluster_dir / "cluster_interpretation_table.csv")
    anchor_table = pd.read_csv(args.anchor_table)

    anchor_cols = [
        "sample_key",
        "proposed_harmonized_anchor",
        "anchor_type",
        "anchor_confidence",
        "cross_dataset_usable",
        "notes",
        "provenance",
    ]
    metadata = metadata.merge(anchor_table[anchor_cols], on="sample_key", how="left")
    metadata["proposed_harmonized_anchor"] = metadata["proposed_harmonized_anchor"].fillna("")
    metadata["anchor_type"] = metadata["anchor_type"].fillna("")
    metadata["anchor_confidence"] = metadata["anchor_confidence"].fillna("")
    metadata["cross_dataset_usable"] = metadata["cross_dataset_usable"].fillna(False)
    metadata["notes"] = metadata["notes"].fillna("")
    metadata["provenance"] = metadata["provenance"].fillna("")
    return {
        "embeddings": embeddings,
        "metadata": metadata,
        "projection": projection,
        "cluster_assignments": cluster_assignments,
        "cluster_summary": cluster_summary,
        "cluster_interpretation": cluster_interpretation,
        "anchor_table": anchor_table,
    }


def normalize_vectors(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return matrix / norms


def infer_theme_from_label(label: str, dataset_id: str, anchor: str) -> str:
    label_lower = (label or "").strip().lower()
    dataset_lower = (dataset_id or "").strip().lower()
    anchor_lower = (anchor or "").strip().lower()

    if anchor in THEME_ORDER:
        return anchor
    if dataset_lower == "adenine_sers_control":
        return "purine_metabolite_associated"
    if dataset_lower == "serum_ag_colloids_grounding" and anchor_lower == "grounding_other_controlled_reference":
        return "serum_matrix_associated"
    if dataset_lower == "amino_acid_raman_grounding" and anchor_lower == "grounding_other_controlled_reference":
        return "protein_peptide_associated"

    combined = f"{label_lower} {dataset_lower} {anchor_lower}"
    for theme, keywords in KEYWORD_THEME_MAP.items():
        if any(keyword in combined for keyword in keywords):
            return theme
    return "grounding_reference_unresolved"


def build_grounding_theme_table(metadata: pd.DataFrame) -> pd.DataFrame:
    grounding = metadata[metadata["sample_type"] == "grounding"].copy()
    grounding["grounding_theme"] = grounding.apply(
        lambda row: infer_theme_from_label(
            str(row.get("label_optional", "")),
            str(row.get("dataset_id", "")),
            str(row.get("proposed_harmonized_anchor", "")),
        ),
        axis=1,
    )
    grounding["theme_source"] = np.where(
        grounding["proposed_harmonized_anchor"].isin(THEME_ORDER),
        "anchor_table",
        "dataset_label_heuristic",
    )
    return grounding


def write_grounding_theme_report(output_dir: Path, grounding_themes: pd.DataFrame) -> None:
    summary = (
        grounding_themes.groupby(["grounding_theme", "dataset_id"])
        .size()
        .reset_index(name="record_count")
        .sort_values(["grounding_theme", "record_count"], ascending=[True, False])
    )
    report = textwrap.dedent(
        f"""
        # Grounding Theme Report

        Grounding records were mapped to broad biochemical themes only. Narrow molecules are preserved as retrieved examples,
        but cluster meaning is assigned at the theme level.

        Theme coverage by dataset:

        {summary.to_string(index=False)}
        """
    ).strip() + "\n"
    (output_dir / "grounding_theme_report.md").write_text(report, encoding="utf-8")


def retrieve_top_grounding_hits(
    query_vector: np.ndarray,
    grounding_embeddings: np.ndarray,
    grounding_table: pd.DataFrame,
    *,
    top_k: int,
) -> pd.DataFrame:
    similarities = grounding_embeddings @ query_vector
    top_idx = np.argsort(-similarities)[:top_k]
    hits = grounding_table.iloc[top_idx].copy()
    hits["similarity"] = similarities[top_idx]
    hits["rank"] = np.arange(1, len(hits) + 1)
    return hits


def theme_strength_label(top_share: float, entropy_norm: float, top_similarity: float) -> str:
    if top_share >= 0.55 and entropy_norm <= 0.55 and top_similarity >= 0.20:
        return "strong"
    if top_share >= 0.40 and entropy_norm <= 0.75 and top_similarity >= 0.10:
        return "moderate"
    if top_share >= 0.28:
        return "weak"
    return "mixed_ambiguous"


def build_interpretation_summary(row: pd.Series) -> tuple[str, str, str]:
    anchor = str(row["dominant_harmonized_anchor"])
    top_theme = str(row["top_biochemical_theme"])
    secondary = str(row["secondary_biochemical_theme"])
    strength = str(row["theme_support_strength"])
    cross_dataset = bool(row["cross_dataset_mixed"])
    dataset_purity = float(row["dataset_purity"])

    if top_theme == "":
        summary = "Grounding support is too diffuse to assign a stable broad biochemical theme."
        uncertainty = "Latent neighbors do not concentrate on one grounding theme."
        caveat = "Interpretation should remain exploratory."
        return summary, uncertainty, caveat

    if secondary and secondary != "none":
        summary = f"Cluster is consistent with {top_theme.replace('_', '/')} contributions, with secondary {secondary.replace('_', '/')} support."
    else:
        summary = f"Cluster is consistent with {top_theme.replace('_', '/')} contributions."

    if cross_dataset:
        summary = "Cross-dataset mixed EV cluster; " + summary[0].lower() + summary[1:]
    if anchor:
        summary += f" Dominant harmonized anchor is {anchor.replace('_', '/')}."

    if strength == "mixed_ambiguous":
        uncertainty = "Grounding retrieval is mixed across themes and does not concentrate strongly."
    elif strength == "weak":
        uncertainty = "Grounding retrieval leans toward one theme, but support remains weak."
    else:
        uncertainty = "Grounding support concentrates on a limited set of broad themes."

    if dataset_purity >= 0.90:
        caveat = "Cluster remains largely dataset-pure, so biochemical interpretation may still be entangled with cohort or protocol effects."
    else:
        caveat = "Interpretation is broad-theme only and should not be read as molecule-level assignment."
    return summary, uncertainty, caveat


def build_cluster_cards(interp_df: pd.DataFrame) -> str:
    cards: list[tuple[str, pd.Series]] = []
    cross_dataset = interp_df[interp_df["cross_dataset_mixed"] == True]
    if not cross_dataset.empty:
        cards.append(("Largest cross-dataset mixed EV cluster", cross_dataset.sort_values("cluster_size", ascending=False).iloc[0]))

    disease = interp_df[interp_df["dominant_harmonized_anchor"] == "ev_disease_or_stress"]
    if not disease.empty:
        cards.append(("Strongest disease/stress EV cluster", disease.sort_values(["theme_support_strength_rank", "cluster_size"]).iloc[0]))

    control = interp_df[interp_df["dominant_harmonized_anchor"] == "ev_control_or_baseline"]
    if not control.empty:
        cards.append(("Strongest control/baseline EV cluster", control.sort_values(["theme_support_strength_rank", "cluster_size"]).iloc[0]))

    ambiguous = interp_df.sort_values(["theme_entropy", "dataset_purity"], ascending=[False, False])
    if not ambiguous.empty:
        cards.append(("Most ambiguous EV cluster", ambiguous.iloc[0]))

    lines = ["# EV Cluster Cards", ""]
    seen: set[str] = set()
    for title, row in cards:
        cluster_id = str(row["cluster_id"])
        if cluster_id in seen:
            continue
        seen.add(cluster_id)
        lines.extend(
            [
                f"## {title}",
                f"- cluster_id: `{cluster_id}`",
                f"- size: {int(row['cluster_size'])}",
                f"- datasets: {row['datasets_represented']}",
                f"- dominant anchor: {row['dominant_harmonized_anchor'] or 'none'}",
                f"- top themes: {row['top_biochemical_theme']}, {row['secondary_biochemical_theme']}, {row['third_biochemical_theme']}",
                f"- representative grounding hits: {row['nearest_grounding_examples']}",
                f"- interpretation: {row['interpretation_summary']}",
                f"- caveat: {row['caveat_notes']}",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def plot_scatter(df: pd.DataFrame, color_col: str, output_path: Path, title: str, top_n: int | None = None) -> None:
    if df.empty:
        return
    plot_df = df.copy()
    plot_df[color_col] = plot_df[color_col].fillna("").astype(str)
    if top_n is not None:
        top_labels = plot_df[color_col].value_counts().head(top_n).index.tolist()
        plot_df[color_col] = plot_df[color_col].where(plot_df[color_col].isin(top_labels), other="other")
    labels = plot_df[color_col].astype("category")
    codes = labels.cat.codes.to_numpy()
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(plot_df["dim1"], plot_df["dim2"], c=codes, cmap="tab20", s=7, alpha=0.8, linewidths=0)
    ax.set_title(title)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    legend_labels = labels.cat.categories.tolist()[:15]
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=scatter.cmap(scatter.norm(idx)), label=label, markersize=5)
        for idx, label in enumerate(legend_labels)
    ]
    if handles:
        ax.legend(handles=handles, loc="upper right", fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_highlight_cross_dataset(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    mask = df["cross_dataset_mixed"].astype("boolean").to_numpy(dtype=bool, na_value=False)
    ax.scatter(df.loc[~mask, "dim1"], df.loc[~mask, "dim2"], s=6, alpha=0.20, color="#bdbdbd", linewidths=0, label="other EV clusters")
    ax.scatter(df.loc[mask, "dim1"], df.loc[mask, "dim2"], s=8, alpha=0.85, color="#c0392b", linewidths=0, label="cross-dataset mixed EV clusters")
    ax.set_title("v7 EV sampled UMAP: cross-dataset mixed clusters highlighted")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ensure_output_dir(args.output_dir)

    loaded = load_inputs(args)
    embeddings = loaded["embeddings"]
    metadata = loaded["metadata"]
    projection = loaded["projection"]
    cluster_assignments = loaded["cluster_assignments"]
    cluster_summary = loaded["cluster_summary"]
    cluster_interpretation = loaded["cluster_interpretation"]

    metadata = metadata.merge(
        cluster_assignments[["sample_key", "global_cluster_id", "within_type_cluster_id"]],
        on="sample_key",
        how="left",
    )
    normalized_embeddings = normalize_vectors(embeddings)

    grounding_table = build_grounding_theme_table(metadata)
    grounding_table.to_csv(args.output_dir / "grounding_theme_table.csv", index=False)
    write_grounding_theme_report(args.output_dir, grounding_table)

    grounding_norm = normalized_embeddings[grounding_table.index.to_numpy()]

    ev_records = metadata[metadata["sample_type"] == "ev"].copy()
    ev_cluster_rows = cluster_summary[
        (cluster_summary["cluster_scope"] == "within_type_cluster_id")
        & (cluster_summary["dominant_sample_type"] == "ev")
    ].copy()
    ev_cluster_interp = cluster_interpretation[
        (cluster_interpretation["cluster_scope"] == "within_type_cluster_id")
        & (cluster_interpretation["dominant_sample_type"] == "ev")
    ].copy()
    ev_cluster_rows = ev_cluster_rows.merge(
        ev_cluster_interp[["cluster_id", "interpretation_label", "notes"]],
        on="cluster_id",
        how="left",
        suffixes=("", "_interp"),
    )

    hit_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    theme_rows: list[dict[str, object]] = []
    interpretation_rows: list[dict[str, object]] = []

    strength_rank = {"strong": 0, "moderate": 1, "weak": 2, "mixed_ambiguous": 3}

    for _, cluster_row in ev_cluster_rows.sort_values("cluster_size", ascending=False).iterrows():
        cluster_id = str(cluster_row["cluster_id"])
        cluster_members = ev_records[ev_records["within_type_cluster_id"] == cluster_id].copy()
        if cluster_members.empty:
            continue

        member_indices = cluster_members.index.to_numpy()
        cluster_vectors = normalized_embeddings[member_indices]
        centroid = cluster_vectors.mean(axis=0)
        centroid /= max(np.linalg.norm(centroid), 1e-8)

        centroid_hits = retrieve_top_grounding_hits(
            centroid,
            grounding_norm,
            grounding_table,
            top_k=args.top_k_grounding,
        )
        centroid_hits["cluster_id"] = cluster_id
        centroid_hits["query_source"] = "cluster_centroid"

        member_similarity = cluster_vectors @ centroid
        rep_order = np.argsort(-member_similarity)[: args.representative_members]
        rep_hits_frames = []
        for rep_rank, idx in enumerate(rep_order, start=1):
            sample_key = str(cluster_members.iloc[idx]["sample_key"])
            query_vector = cluster_vectors[idx]
            rep_hits = retrieve_top_grounding_hits(
                query_vector,
                grounding_norm,
                grounding_table,
                top_k=args.top_k_member_grounding,
            )
            rep_hits["cluster_id"] = cluster_id
            rep_hits["query_source"] = f"representative_member_{rep_rank}"
            rep_hits["query_sample_key"] = sample_key
            rep_hits_frames.append(rep_hits)

        combined_hits = pd.concat([centroid_hits, *rep_hits_frames], ignore_index=True)
        combined_hits["query_sample_key"] = combined_hits["query_sample_key"].fillna("")
        combined_hits["retrieval_weight"] = combined_hits["similarity"].clip(lower=0.0) + 1e-6
        combined_hits["retrieval_source_weight"] = combined_hits["query_source"].map(
            lambda value: 1.0 if value == "cluster_centroid" else 0.65
        )
        combined_hits["weighted_score"] = combined_hits["retrieval_weight"] * combined_hits["retrieval_source_weight"]
        hit_rows.extend(combined_hits.to_dict(orient="records"))

        dedup_hits = (
            combined_hits.sort_values(["sample_key", "weighted_score"], ascending=[True, False])
            .drop_duplicates(subset=["sample_key"])
            .copy()
        )
        theme_scores = (
            dedup_hits.groupby("grounding_theme")
            .agg(
                weighted_score=("weighted_score", "sum"),
                hit_count=("sample_key", "count"),
                mean_similarity=("similarity", "mean"),
                max_similarity=("similarity", "max"),
                datasets=("dataset_id", lambda s: "; ".join(sorted(set(map(str, s))))),
            )
            .reset_index()
            .sort_values(["weighted_score", "mean_similarity"], ascending=[False, False])
        )
        total_score = float(theme_scores["weighted_score"].sum()) if not theme_scores.empty else 0.0
        if total_score > 0:
            theme_scores["theme_share"] = theme_scores["weighted_score"] / total_score
        else:
            theme_scores["theme_share"] = 0.0
        probs = theme_scores["theme_share"].to_numpy(dtype=float)
        nonzero = probs[probs > 0]
        theme_entropy = float(-(nonzero * np.log2(nonzero)).sum()) if len(nonzero) else 0.0
        max_entropy = math.log2(len(theme_scores)) if len(theme_scores) > 1 else 1.0
        theme_entropy_norm = theme_entropy / max_entropy if max_entropy > 0 else 0.0

        top_theme = theme_scores.iloc[0]["grounding_theme"] if len(theme_scores) >= 1 else ""
        secondary_theme = theme_scores.iloc[1]["grounding_theme"] if len(theme_scores) >= 2 else "none"
        third_theme = theme_scores.iloc[2]["grounding_theme"] if len(theme_scores) >= 3 else "none"
        top_theme_share = float(theme_scores.iloc[0]["theme_share"]) if len(theme_scores) >= 1 else 0.0
        top_theme_similarity = float(theme_scores.iloc[0]["mean_similarity"]) if len(theme_scores) >= 1 else 0.0
        support_strength = theme_strength_label(top_theme_share, theme_entropy_norm, top_theme_similarity)

        for _, theme_row in theme_scores.iterrows():
            theme_rows.append(
                {
                    "cluster_id": cluster_id,
                    "grounding_theme": theme_row["grounding_theme"],
                    "weighted_score": theme_row["weighted_score"],
                    "theme_share": theme_row["theme_share"],
                    "hit_count": int(theme_row["hit_count"]),
                    "mean_similarity": float(theme_row["mean_similarity"]),
                    "max_similarity": float(theme_row["max_similarity"]),
                    "support_strength": support_strength,
                    "theme_entropy": theme_entropy_norm,
                    "datasets": theme_row["datasets"],
                }
            )

        top_examples = (
            dedup_hits.sort_values(["weighted_score", "similarity"], ascending=[False, False])
            .head(5)[["dataset_id", "label_optional", "grounding_theme", "similarity"]]
        )
        nearest_grounding_examples = "; ".join(
            f"{row.dataset_id}/{row.label_optional}/{row.grounding_theme}:{row.similarity:.3f}"
            for row in top_examples.itertuples(index=False)
        )

        datasets_represented = "; ".join(cluster_members["dataset_id"].astype(str).value_counts().index.tolist())
        summary_rows.append(
            {
                "cluster_id": cluster_id,
                "cluster_size": int(cluster_row["cluster_size"]),
                "cross_dataset_mixed": bool(cluster_row["cross_dataset_mixed"]),
                "dataset_purity": float(cluster_row["dominant_dataset_share"]),
                "dataset_count": int(cluster_row["dataset_count"]),
                "dominant_harmonized_anchor": str(cluster_row["dominant_anchor"]),
                "top_biochemical_theme": top_theme,
                "secondary_biochemical_theme": secondary_theme,
                "third_biochemical_theme": third_theme,
                "theme_support_strength": support_strength,
                "theme_entropy": theme_entropy_norm,
                "top_theme_share": top_theme_share,
                "top_theme_mean_similarity": top_theme_similarity,
                "datasets_represented": datasets_represented,
                "nearest_grounding_examples": nearest_grounding_examples,
                "cluster_notes": str(cluster_row.get("notes_interp", "") or cluster_row.get("notes", "")),
            }
        )

    hit_df = pd.DataFrame(hit_rows)
    hit_df.to_csv(args.output_dir / "ev_cluster_grounding_hits.csv", index=False)

    summary_df = pd.DataFrame(summary_rows).sort_values(["cluster_size"], ascending=False).reset_index(drop=True)
    summary_df["theme_support_strength_rank"] = summary_df["theme_support_strength"].map(strength_rank).fillna(9).astype(int)
    summary_df.to_csv(args.output_dir / "ev_cluster_grounding_summary.csv", index=False)

    theme_df = pd.DataFrame(theme_rows).sort_values(["cluster_id", "weighted_score"], ascending=[True, False]).reset_index(drop=True)
    theme_df.to_csv(args.output_dir / "ev_cluster_theme_scores.csv", index=False)

    interpretation_records = []
    for _, row in summary_df.iterrows():
        summary_text, uncertainty_text, caveat_text = build_interpretation_summary(row)
        interpretation_records.append(
            {
                "cluster_id": row["cluster_id"],
                "cluster_size": int(row["cluster_size"]),
                "cross_dataset_mixed": bool(row["cross_dataset_mixed"]),
                "dataset_purity": float(row["dataset_purity"]),
                "dominant_harmonized_anchor": row["dominant_harmonized_anchor"],
                "top_biochemical_theme": row["top_biochemical_theme"],
                "secondary_biochemical_theme": row["secondary_biochemical_theme"],
                "theme_support_strength": row["theme_support_strength"],
                "nearest_grounding_examples": row["nearest_grounding_examples"],
                "datasets_represented": row["datasets_represented"],
                "interpretation_summary": summary_text,
                "uncertainty_notes": uncertainty_text,
                "caveat_notes": caveat_text,
                "theme_entropy": float(row["theme_entropy"]),
                "top_theme_share": float(row["top_theme_share"]),
                "theme_support_strength_rank": int(row["theme_support_strength_rank"]),
                "third_biochemical_theme": row["third_biochemical_theme"],
            }
        )
    interpretation_df = pd.DataFrame(interpretation_records).sort_values(
        ["theme_support_strength_rank", "cluster_size"],
        ascending=[True, False],
    )
    interpretation_df.to_csv(args.output_dir / "ev_cluster_interpretation_table.csv", index=False)

    theme_report = textwrap.dedent(
        f"""
        # EV Cluster Theme Report

        Theme scores were built by retrieving nearest grounding records to each EV cluster centroid and to a few representative
        cluster members in latent space. Scores aggregate weighted cosine similarity and report broad biochemical themes only.

        Support strength counts:

        {interpretation_df['theme_support_strength'].value_counts().to_string()}
        """
    ).strip() + "\n"
    (args.output_dir / "ev_cluster_theme_report.md").write_text(theme_report, encoding="utf-8")

    interp_report = textwrap.dedent(
        f"""
        # EV Cluster Interpretation Report

        Cluster interpretations are cautious and evidence-backed. They describe consistency with broad grounding themes, not
        direct molecule-level claims.

        {interpretation_df[['cluster_id','cluster_size','dominant_harmonized_anchor','top_biochemical_theme','secondary_biochemical_theme','theme_support_strength','interpretation_summary','uncertainty_notes','caveat_notes']].to_string(index=False)}
        """
    ).strip() + "\n"
    (args.output_dir / "ev_cluster_interpretation_report.md").write_text(interp_report, encoding="utf-8")

    cards_md = build_cluster_cards(interpretation_df)
    (args.output_dir / "ev_cluster_cards.md").write_text(cards_md, encoding="utf-8")

    strong_count = int((interpretation_df["theme_support_strength"] == "strong").sum())
    moderate_count = int((interpretation_df["theme_support_strength"] == "moderate").sum())
    weak_count = int((interpretation_df["theme_support_strength"] == "weak").sum())
    mixed_count = int((interpretation_df["theme_support_strength"] == "mixed_ambiguous").sum())
    decision_lines = [
        "# v7 Grounding Decision Memo",
        "",
        "1. Are EV latent clusters now interpretable in biochemical-theme terms?",
        f"Yes, at a broad-theme level. Strong={strong_count}, moderate={moderate_count}, weak={weak_count}, mixed={mixed_count}.",
        "",
        "2. Are the retrieved grounding themes coherent enough to support a first inference layer?",
        "Yes for EV-first broad-theme interpretation, provided outputs remain explicitly cautious and evidence-backed.",
        "",
        "3. Which EV clusters are strongest candidates for demo use?",
        interpretation_df.head(8)[['cluster_id','cluster_size','dominant_harmonized_anchor','top_biochemical_theme','theme_support_strength']].to_string(index=False),
        "",
        "4. Is the current system ready for an EV-first graph / inference prototype?",
        "Yes. The current latent clusters plus broad grounding themes are sufficient for a first EV-only prototype focused on cluster-level biochemical interpretation rather than sample-level diagnosis.",
        "",
    ]
    (args.output_dir / "v7_grounding_decision_memo.md").write_text("\n".join(decision_lines), encoding="utf-8")

    projection_ev = projection.merge(
        cluster_assignments[["sample_key", "within_type_cluster_id"]],
        on="sample_key",
        how="left",
    )
    projection_ev = projection_ev.merge(
        interpretation_df[["cluster_id", "top_biochemical_theme", "dominant_harmonized_anchor", "theme_support_strength", "cross_dataset_mixed"]],
        left_on="within_type_cluster_id",
        right_on="cluster_id",
        how="left",
    )
    projection_ev = projection_ev[projection_ev["sample_type"] == "ev"].copy()

    plot_scatter(projection_ev, "within_type_cluster_id", args.output_dir / "umap_ev_cluster_id.png", "v7 EV sampled UMAP by cluster", top_n=20)
    plot_scatter(projection_ev, "top_biochemical_theme", args.output_dir / "umap_ev_biochemical_theme.png", "v7 EV sampled UMAP by dominant biochemical theme", top_n=8)
    plot_scatter(projection_ev, "dominant_harmonized_anchor", args.output_dir / "umap_ev_harmonized_anchor.png", "v7 EV sampled UMAP by dominant harmonized anchor", top_n=8)
    plot_highlight_cross_dataset(projection_ev, args.output_dir / "umap_ev_cross_dataset_mixed.png")

    print(f"Saved grounding analysis to {args.output_dir}")
    print(f"EV clusters analyzed: {interpretation_df['cluster_id'].nunique()}")
    print("Theme support counts: " + ", ".join(
        f"{label}={count}" for label, count in interpretation_df["theme_support_strength"].value_counts().to_dict().items()
    ))


if __name__ == "__main__":
    main()
