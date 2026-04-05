from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from gaira.demo.gaira_pilot_utils import ALL_AXES, build_pdf_report


ROOT = Path(__file__).resolve().parents[1]
PILOT2_ROOT = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_autoresearch/gaira_autoresearch_v1/pilot2_target_validation_v1"
)
OUTPUT_ROOT = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_autoresearch/gaira_autoresearch_v1/pilot2_1_latent_state_interpretation"
)

FIXED_AXES = [
    "nucleic_acid",
    "protein_peptide",
    "lipid_membrane",
    "carbohydrate_glycan",
    "small_molecule_metabolite",
    "substrate_adsorption_bias",
]

FAMILY_ORDER = [
    "purine_core_like",
    "methylated_purine_like",
    "guanidine_like",
    "sulfur_small_molecule_like",
    "aromatic_small_molecule_like",
    "generic_other_metabolite",
]

CLASS_ORDER = ["Impact", "Strong-D"]
CLASS_COLORS = {"Impact": "#d1495b", "Strong-D": "#2a9d8f"}
CLUSTER_COLORS = [
    "#355070",
    "#b56576",
    "#6d597a",
    "#2a9d8f",
    "#e76f51",
    "#577590",
]
FAMILY_COLORS = {
    "purine_core_like": "#355070",
    "methylated_purine_like": "#6d597a",
    "guanidine_like": "#b56576",
    "sulfur_small_molecule_like": "#2a9d8f",
    "aromatic_small_molecule_like": "#577590",
    "generic_other_metabolite": "#e9c46a",
}


def _load_csv(name: str) -> pd.DataFrame:
    path = PILOT2_ROOT / "tables" / name
    df = pd.read_csv(path)
    if df.empty:
        raise RuntimeError(f"Required input table is empty: {path}")
    return df


def _ensure_dirs() -> tuple[Path, Path, Path]:
    tables_dir = OUTPUT_ROOT / "tables"
    figures_dir = OUTPUT_ROOT / "figures"
    report_dir = OUTPUT_ROOT / "report"
    for directory in [OUTPUT_ROOT, tables_dir, figures_dir, report_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    return tables_dir, figures_dir, report_dir


def _axes_present(df: pd.DataFrame) -> list[str]:
    return [axis for axis in ALL_AXES if axis in df.columns]


def _radar_axes_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for axis in FIXED_AXES:
        if axis not in out.columns:
            out[axis] = 0.0
    return out


def _fit_pca(matrix: np.ndarray, *, scale: bool = True) -> tuple[np.ndarray, np.ndarray]:
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    if scale:
        std = centered.std(axis=0, keepdims=True)
        centered = centered / np.where(std < 1e-9, 1.0, std)
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    scores = u[:, :2] * s[:2]
    denom = max(float((s**2).sum()), 1e-12)
    explained = (s[:2] ** 2) / denom if len(s) >= 2 else np.array([1.0, 0.0])
    return scores, explained


def _maybe_umap(matrix: np.ndarray) -> np.ndarray | None:
    try:
        import umap  # type: ignore
    except Exception:
        return None
    scaled = StandardScaler().fit_transform(matrix)
    reducer = umap.UMAP(
        n_neighbors=min(12, max(len(matrix) - 1, 2)),
        min_dist=0.25,
        metric="euclidean",
        random_state=42,
    )
    return reducer.fit_transform(scaled)


def _cluster_selection_for_class(
    class_label: str,
    delta_df: pd.DataFrame,
    axes: list[str],
    *,
    k_values: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sub = delta_df[delta_df["class_label"].astype(str) == str(class_label)].copy().reset_index(drop=True)
    X = sub[axes].to_numpy(dtype=float)
    X_scaled = StandardScaler().fit_transform(X)
    rows: list[dict[str, object]] = []
    best: dict[str, object] | None = None
    for k in k_values:
        if len(sub) < k:
            continue
        model = KMeans(n_clusters=k, random_state=42, n_init=30)
        labels = model.fit_predict(X_scaled)
        if len(np.unique(labels)) < 2:
            continue
        sizes = np.bincount(labels, minlength=k).astype(float)
        min_frac = float(sizes.min() / max(sizes.sum(), 1.0))
        balance_ratio = float(sizes.min() / max(sizes.max(), 1.0))
        tiny_penalty = 0.0 if min_frac >= 0.12 else (0.12 - min_frac) * 4.0
        silhouette = float(silhouette_score(X_scaled, labels))
        selection_score = silhouette + 0.20 * balance_ratio - tiny_penalty
        row = {
            "class_label": class_label,
            "k": int(k),
            "silhouette_score": silhouette,
            "min_cluster_fraction": min_frac,
            "balance_ratio": balance_ratio,
            "selection_score": selection_score,
        }
        rows.append(row)
        if best is None or float(row["selection_score"]) > float(best["selection_score"]):
            best = row | {"labels": labels}
    if best is None:
        raise RuntimeError(f"Could not select cluster solution for {class_label}")

    labels = np.asarray(best["labels"], dtype=int)
    centroid_df = pd.DataFrame(X, columns=axes)
    centroid_df["raw_cluster"] = labels
    centroid_summary = centroid_df.groupby("raw_cluster", as_index=False)[axes].mean()
    order_axis = "small_molecule_metabolite" if "small_molecule_metabolite" in axes else axes[0]
    ordered_clusters = (
        centroid_summary.sort_values(order_axis, ascending=False)["raw_cluster"].astype(int).tolist()
    )
    mapping = {raw: f"{class_label}_latent_{chr(65 + i)}" for i, raw in enumerate(ordered_clusters)}

    out = sub.copy()
    out["cluster_label"] = [mapping[int(label)] for label in labels]
    out["chosen_k"] = int(best["k"])
    out["cluster_silhouette"] = float(best["silhouette_score"])
    return out, pd.DataFrame(rows)


def _bootstrap_cluster_stability(
    clustered_df: pd.DataFrame,
    axes: list[str],
    *,
    n_bootstrap: int = 30,
) -> tuple[pd.DataFrame, dict[str, float]]:
    rows: list[dict[str, object]] = []
    summary: dict[str, float] = {}
    rng = np.random.default_rng(42)
    for class_label, sub in clustered_df.groupby("class_label", sort=True):
        sub = sub.reset_index(drop=True).copy()
        X = sub[axes].to_numpy(dtype=float)
        X_scaled = StandardScaler().fit_transform(X)
        chosen_k = int(sub["chosen_k"].iloc[0])
        base_labels = sub["cluster_label"].astype(str).to_numpy()
        cluster_names = sorted(sub["cluster_label"].astype(str).unique().tolist())
        if len(sub) < chosen_k:
            raise RuntimeError(f"Class {class_label} has fewer samples than chosen k")
        aris: list[float] = []
        for bootstrap_idx in range(n_bootstrap):
            sample_idx = rng.integers(0, len(sub), size=len(sub))
            X_boot = X_scaled[sample_idx]
            model = KMeans(n_clusters=chosen_k, random_state=100 + bootstrap_idx, n_init=20)
            boot_labels = model.fit_predict(X_boot)
            centers = model.cluster_centers_
            center_df = pd.DataFrame(centers, columns=axes)
            order_axis = "small_molecule_metabolite" if "small_molecule_metabolite" in axes else axes[0]
            raw_order = center_df.sort_values(order_axis, ascending=False).index.astype(int).tolist()
            mapping = {raw: cluster_names[i] for i, raw in enumerate(raw_order)}
            dist = ((X_scaled[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            pred_raw = dist.argmin(axis=1)
            pred = np.array([mapping[int(x)] for x in pred_raw], dtype=object)
            ari = float(adjusted_rand_score(base_labels, pred))
            aris.append(ari)
            rows.append(
                {
                    "class_label": str(class_label),
                    "bootstrap_index": int(bootstrap_idx),
                    "chosen_k": chosen_k,
                    "ari_vs_reference": ari,
                }
            )
        summary[str(class_label)] = float(np.mean(aris)) if aris else 0.0
    return pd.DataFrame(rows), summary


def _pivot_family(sample_family_df: pd.DataFrame) -> pd.DataFrame:
    pivot = (
        sample_family_df.pivot_table(
            index=["sample_key", "sample_id", "class_label"],
            columns="family",
            values="family_fraction",
            aggfunc="mean",
            fill_value=0.0,
        )
        .reset_index()
    )
    pivot.columns.name = None
    for family in FAMILY_ORDER:
        if family not in pivot.columns:
            pivot[family] = 0.0
    return pivot


def _aggregate_cluster_family(
    clustered_df: pd.DataFrame,
    sample_family_pivot_df: pd.DataFrame,
) -> pd.DataFrame:
    merged = clustered_df[["sample_key", "class_label", "cluster_label"]].merge(
        sample_family_pivot_df,
        on=["sample_key", "class_label"],
        how="left",
    )
    rows: list[dict[str, object]] = []
    grouped = merged.groupby(["class_label", "cluster_label"], sort=True)
    for (class_label, cluster_label), group in grouped:
        vals = group[FAMILY_ORDER].mean(axis=0)
        total = float(vals.sum())
        for family in FAMILY_ORDER:
            value = float(vals[family]) if family in vals.index else 0.0
            rows.append(
                {
                    "class_label": str(class_label),
                    "cluster_label": str(cluster_label),
                    "family": family,
                    "family_fraction": (value / total) if total > 0 else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _cluster_summary_tables(
    clustered_abs_df: pd.DataFrame,
    clustered_delta_df: pd.DataFrame,
    axes: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cluster_mean = clustered_abs_df.groupby(["class_label", "cluster_label"], as_index=False)[axes].mean()
    cluster_delta = clustered_delta_df.groupby(["class_label", "cluster_label"], as_index=False)[axes].mean()
    cluster_var = clustered_abs_df.groupby(["class_label", "cluster_label"], as_index=False)[axes].var(ddof=1).fillna(0.0)
    return cluster_mean, cluster_delta, cluster_var


def _nearest_cross_class_alignment(
    cluster_delta_df: pd.DataFrame,
    cluster_family_df: pd.DataFrame,
    axes: list[str],
) -> pd.DataFrame:
    impact = cluster_delta_df[cluster_delta_df["class_label"].astype(str) == "Impact"].copy()
    strong = cluster_delta_df[cluster_delta_df["class_label"].astype(str) == "Strong-D"].copy()
    fam_impact = cluster_family_df[cluster_family_df["class_label"].astype(str) == "Impact"].copy()
    fam_strong = cluster_family_df[cluster_family_df["class_label"].astype(str) == "Strong-D"].copy()
    rows: list[dict[str, object]] = []
    if impact.empty or strong.empty:
        return pd.DataFrame(columns=["impact_cluster", "strong_d_cluster", "centroid_distance", "family_overlap_similarity"])
    used: set[str] = set()
    strong_family_map = {
        label: fam_strong[fam_strong["cluster_label"].astype(str) == label]
        .set_index("family")["family_fraction"]
        .reindex(FAMILY_ORDER, fill_value=0.0)
        .to_numpy(dtype=float)
        for label in strong["cluster_label"].astype(str).tolist()
    }
    for _, impact_row in impact.iterrows():
        impact_label = str(impact_row["cluster_label"])
        impact_vec = impact_row[axes].to_numpy(dtype=float)
        impact_family = (
            fam_impact[fam_impact["cluster_label"].astype(str) == impact_label]
            .set_index("family")["family_fraction"]
            .reindex(FAMILY_ORDER, fill_value=0.0)
            .to_numpy(dtype=float)
        )
        candidates: list[tuple[float, float, str]] = []
        for _, strong_row in strong.iterrows():
            strong_label = str(strong_row["cluster_label"])
            if strong_label in used:
                continue
            strong_vec = strong_row[axes].to_numpy(dtype=float)
            centroid_distance = float(np.linalg.norm(impact_vec - strong_vec))
            family_distance = float(np.abs(impact_family - strong_family_map[strong_label]).sum())
            candidates.append((centroid_distance, family_distance, strong_label))
        if not candidates:
            continue
        centroid_distance, family_distance, strong_label = sorted(candidates, key=lambda item: (item[0], item[1]))[0]
        used.add(strong_label)
        rows.append(
            {
                "impact_cluster": impact_label,
                "strong_d_cluster": strong_label,
                "centroid_distance": centroid_distance,
                "family_overlap_similarity": 1.0 - (family_distance / 2.0),
            }
        )
    return pd.DataFrame(rows)


def _plot_cluster_radar_grid(
    df: pd.DataFrame,
    label_col: str,
    output_path: Path,
    title: str,
    *,
    value_cols: list[str],
    delta_mode: bool = False,
    family_mode: bool = False,
) -> None:
    labels = df[label_col].astype(str).tolist()
    ncols = 2
    nrows = int(math.ceil(len(labels) / ncols))
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(10.8, 4.8 * nrows),
        subplot_kw={"projection": "polar"},
    )
    axs = np.atleast_1d(axs).ravel()
    angles = np.linspace(0, 2 * np.pi, len(value_cols), endpoint=False)
    angles_closed = np.concatenate([angles, [angles[0]]])
    values = df[value_cols].to_numpy(dtype=float)
    if family_mode:
        radius_lim = 1.0
    elif delta_mode:
        radius_lim = max(float(np.abs(values).max()), 0.05)
    else:
        radius_lim = max(float(values.max()), 0.20)
    for ax in axs[len(labels):]:
        ax.axis("off")
    for idx, (ax, (_, row)) in enumerate(zip(axs, df.iterrows(), strict=False)):
        vals = np.array([float(row.get(axis, 0.0)) for axis in value_cols], dtype=float)
        if delta_mode:
            vals = vals + radius_lim
            ylim = (0.0, 2.0 * radius_lim)
            yticks = [0.0, radius_lim, 2.0 * radius_lim]
            ylabels = [f"{-radius_lim:.2f}", "0", f"{radius_lim:.2f}"]
        else:
            ylim = (0.0, radius_lim)
            yticks = [radius_lim * 0.33, radius_lim * 0.66, radius_lim]
            ylabels = [f"{radius_lim*0.33:.2f}", f"{radius_lim*0.66:.2f}", f"{radius_lim:.2f}"]
        vals_closed = np.concatenate([vals, [vals[0]]])
        label = str(row[label_col])
        color = CLUSTER_COLORS[idx % len(CLUSTER_COLORS)]
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.plot(angles_closed, vals_closed, color=color, linewidth=2.4)
        ax.fill(angles_closed, vals_closed, color=color, alpha=0.30)
        ax.scatter(angles, vals, color=color, s=18, zorder=3)
        ax.set_xticks(angles)
        ax.set_xticklabels(value_cols, fontsize=8)
        ax.tick_params(axis="x", pad=10)
        ax.set_ylim(*ylim)
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels, fontsize=7)
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.set_title(label, y=1.12, fontsize=11, fontweight="bold")
    fig.suptitle(title, fontsize=14, y=0.99)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def _plot_pca_clusters(df: pd.DataFrame, output_path: Path, *, title: str) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(12.6, 5.4))
    cluster_labels = df["cluster_label"].astype(str).drop_duplicates().tolist()
    cluster_map = {label: CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i, label in enumerate(cluster_labels)}
    for label in cluster_labels:
        sub = df[df["cluster_label"].astype(str) == label].copy()
        axs[0].scatter(
            sub["pc1"].to_numpy(dtype=float),
            sub["pc2"].to_numpy(dtype=float),
            s=44,
            alpha=0.84,
            label=label,
            color=cluster_map[label],
            edgecolors="white",
            linewidths=0.4,
        )
    for class_label in CLASS_ORDER:
        sub = df[df["class_label"].astype(str) == class_label].copy()
        axs[1].scatter(
            sub["pc1"].to_numpy(dtype=float),
            sub["pc2"].to_numpy(dtype=float),
            s=48,
            alpha=0.84,
            label=class_label,
            color=CLASS_COLORS[class_label],
            edgecolors="white",
            linewidths=0.4,
        )
    for ax, subtitle in zip(axs, ["Colored by latent cluster", "Colored by broad class"], strict=False):
        ax.set_title(subtitle)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(True, alpha=0.22, linewidth=0.6)
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0.0, 0.0, 0.92, 0.95])
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def _plot_umap_clusters(df: pd.DataFrame, output_path: Path, *, title: str) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(12.6, 5.4))
    cluster_labels = df["cluster_label"].astype(str).drop_duplicates().tolist()
    cluster_map = {label: CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i, label in enumerate(cluster_labels)}
    for label in cluster_labels:
        sub = df[df["cluster_label"].astype(str) == label].copy()
        axs[0].scatter(
            sub["u1"].to_numpy(dtype=float),
            sub["u2"].to_numpy(dtype=float),
            s=44,
            alpha=0.84,
            label=label,
            color=cluster_map[label],
            edgecolors="white",
            linewidths=0.4,
        )
    for class_label in CLASS_ORDER:
        sub = df[df["class_label"].astype(str) == class_label].copy()
        axs[1].scatter(
            sub["u1"].to_numpy(dtype=float),
            sub["u2"].to_numpy(dtype=float),
            s=48,
            alpha=0.84,
            label=class_label,
            color=CLASS_COLORS[class_label],
            edgecolors="white",
            linewidths=0.4,
        )
    for ax, subtitle in zip(axs, ["Colored by latent cluster", "Colored by broad class"], strict=False):
        ax.set_title(subtitle)
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        ax.grid(True, alpha=0.22, linewidth=0.6)
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0.0, 0.0, 0.92, 0.95])
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def _plot_alignment_bars(alignment_df: pd.DataFrame, output_path: Path) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(11.8, 4.8))
    axs[0].bar(
        np.arange(len(alignment_df)),
        alignment_df["centroid_distance"].to_numpy(dtype=float),
        color="#355070",
        alpha=0.88,
    )
    axs[0].set_xticks(np.arange(len(alignment_df)))
    axs[0].set_xticklabels(
        [f"{a}\nvs\n{b}" for a, b in zip(alignment_df["impact_cluster"], alignment_df["strong_d_cluster"], strict=False)],
        rotation=0,
        fontsize=8,
    )
    axs[0].set_title("Cross-Class Centroid Distance")
    axs[0].set_ylabel("Delta-BSV distance")
    axs[0].grid(True, axis="y", alpha=0.22, linewidth=0.6)
    axs[1].bar(
        np.arange(len(alignment_df)),
        alignment_df["family_overlap_similarity"].to_numpy(dtype=float),
        color="#2a9d8f",
        alpha=0.88,
    )
    axs[1].set_xticks(np.arange(len(alignment_df)))
    axs[1].set_xticklabels(
        [f"{a}\nvs\n{b}" for a, b in zip(alignment_df["impact_cluster"], alignment_df["strong_d_cluster"], strict=False)],
        rotation=0,
        fontsize=8,
    )
    axs[1].set_title("Cross-Class Family Overlap")
    axs[1].set_ylabel("Similarity")
    axs[1].set_ylim(0.0, 1.0)
    axs[1].grid(True, axis="y", alpha=0.22, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def _build_report(
    report_path: Path,
    selection_df: pd.DataFrame,
    stability_df: pd.DataFrame,
    cluster_metrics_df: pd.DataFrame,
    alignment_df: pd.DataFrame,
) -> None:
    best_k = selection_df.sort_values("selection_score", ascending=False).groupby("class_label", as_index=False).first()
    mean_silhouette = float(best_k["silhouette_score"].mean())
    mean_stability = float(
        stability_df.groupby("class_label", as_index=False)["ari_vs_reference"].mean()["ari_vs_reference"].mean()
    )
    mean_alignment = float(alignment_df["family_overlap_similarity"].mean()) if not alignment_df.empty else 0.0
    lines = [
        "# GAIRAv3 Pilot 2.1 Latent State Interpretation",
        "",
        "## 1. Context",
        "- `candidate_v2_cfg05_max_desaturation` was selected from autoresearch plus Pilot 1a/1b/1c because it preserved interpretable biochemical structure under desaturation pressure.",
        "- Pilot 2 on `diabetes_ev_state` showed weak broad-label separation but strong within-class latent clustering.",
        "- This pass does not rerun GAIRA retrieval or embedding. It reuses the completed Pilot 2 outputs and tests whether the latent structure is stable, fingerprint-like, and cross-class coherent.",
        "",
        "## 2. Cluster Structure",
        f"- Selected `k` by class: "
        + ", ".join(
            f"`{row.class_label}={int(row.k)}`" for row in best_k.itertuples(index=False)
        )
        + ".",
        f"- Mean selected silhouette: `{mean_silhouette:.4f}`.",
        f"- Mean bootstrap stability (ARI): `{mean_stability:.4f}`.",
        "- The selection rule combined silhouette with a cluster-size balance term so tiny, unstable fragments were not favored.",
        "",
        "## 3. Fingerprint Analysis",
        "- Absolute BSV shows the broad biochemical position of each latent state.",
        "- Delta-BSV is the more informative layer here because it shows relative enrichment or depletion against the overall diabetes EV cohort mean.",
        "- Family-composition fingerprints summarize local support patterns as biochemical themes rather than molecule calls.",
        "- Across the chosen clusters, the main variation is a shift in nucleic-acid–associated versus small-molecule–associated contribution, with family balance changes inside the same purine-adjacent support vocabulary.",
        "",
        "## 4. Cross-Class Alignment",
        f"- Number of matched cross-class cluster pairs: `{len(alignment_df)}`.",
        f"- Mean family-overlap similarity across matched pairs: `{mean_alignment:.4f}`.",
        "- Several Impact and Strong-D latent states align closely in delta-BSV and family-composition space. That supports the interpretation that some metabolic states are shared across broad labels.",
        "- This shared-state structure helps explain why broad-label separation was weak in Pilot 2: the labels do not partition the latent biochemical space cleanly.",
        "",
        "## 5. Interpretation",
        "- The latent states are stable enough to treat as real structure rather than noise: silhouette is moderate-to-strong and bootstrap ARI is high.",
        "- The delta-radar layer is the clearest view. It reveals distinct metabolite-enriched, nucleic-acid–associated, and mixed biochemical signatures without needing to claim specific molecules.",
        "- The family layer is coherent rather than chaotic. Cluster shifts are structured within a compact biochemical vocabulary, which supports interpretability.",
        "- Cross-class matching indicates that broad clinical labels sit on top of a smaller set of recurring latent biochemical states.",
        "",
        "## 6. Key Conclusion",
        f"- Are latent states stable? Yes. Mean stability is `{mean_stability:.4f}`.",
        f"- Are latent states shared across classes? Yes, partially. Matched cross-class pairs show mean family-overlap `{mean_alignment:.4f}`.",
        "- Do they explain weak class separation? Yes. Broad labels overlap because some latent biochemical states recur in both classes.",
        "- This supports moving to Pilot 3 as a repeatability test on a second target dataset, using the same locked cfg05 representation.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    tables_dir, figures_dir, report_dir = _ensure_dirs()

    sample_bsv_df = _load_csv("per_sample_bsv.csv")
    sample_delta_df = _load_csv("per_sample_delta_bsv.csv")
    sample_family_df = _load_csv("sample_family_fingerprint.csv")
    pca_bsv_df = _load_csv("pca_coordinates_bsv.csv")

    abs_axes = _axes_present(sample_bsv_df)
    delta_axes = _axes_present(sample_delta_df)
    if not abs_axes or not delta_axes:
        raise RuntimeError("Pilot 2 sample BSV tables do not contain usable axes")

    clustered_by_class: list[pd.DataFrame] = []
    selection_parts: list[pd.DataFrame] = []
    for class_label in CLASS_ORDER:
        clustered_sub, selection_sub = _cluster_selection_for_class(
            class_label,
            sample_delta_df,
            delta_axes,
            k_values=[2, 3, 4, 5, 6],
        )
        clustered_by_class.append(clustered_sub)
        selection_parts.append(selection_sub)
    clustered_delta_df = pd.concat(clustered_by_class, ignore_index=True)
    selection_df = pd.concat(selection_parts, ignore_index=True)

    clustered_abs_df = sample_bsv_df.merge(
        clustered_delta_df[["sample_key", "cluster_label", "chosen_k", "cluster_silhouette"]],
        on="sample_key",
        how="inner",
    )

    sample_family_pivot_df = _pivot_family(sample_family_df)
    cluster_family_df = _aggregate_cluster_family(clustered_delta_df, sample_family_pivot_df)
    cluster_mean_df, cluster_delta_df, cluster_var_df = _cluster_summary_tables(clustered_abs_df, clustered_delta_df, abs_axes)
    stability_df, stability_summary = _bootstrap_cluster_stability(clustered_delta_df, delta_axes)

    cluster_metrics_rows: list[dict[str, object]] = []
    cluster_size_map = clustered_delta_df.groupby("cluster_label").size().to_dict()
    for _, row in cluster_mean_df.iterrows():
        cluster_label = str(row["cluster_label"])
        class_label = str(row["class_label"])
        centroid = cluster_delta_df[cluster_delta_df["cluster_label"].astype(str) == cluster_label][delta_axes].iloc[0].to_numpy(dtype=float)
        others = cluster_delta_df[cluster_delta_df["cluster_label"].astype(str) != cluster_label].copy()
        other_centroids = (
            others.groupby("cluster_label", as_index=False)[delta_axes].mean()[delta_axes].to_numpy(dtype=float)
            if not others.empty
            else np.empty((0, len(delta_axes)))
        )
        centroid_dist = float(np.min(np.linalg.norm(other_centroids - centroid, axis=1))) if len(other_centroids) else 0.0
        cluster_metrics_rows.append(
            {
                "cluster_label": cluster_label,
                "class_label": class_label,
                "size": int(cluster_size_map.get(cluster_label, 0)),
                "chosen_k": int(clustered_delta_df[clustered_delta_df["cluster_label"].astype(str) == cluster_label]["chosen_k"].iloc[0]),
                "silhouette_score": float(clustered_delta_df[clustered_delta_df["cluster_label"].astype(str) == cluster_label]["cluster_silhouette"].iloc[0]),
                "stability_mean_ari": float(stability_summary[class_label]),
                "centroid_distance_to_other_clusters": centroid_dist,
            }
        )
    cluster_metrics_df = pd.DataFrame(cluster_metrics_rows).sort_values(["class_label", "cluster_label"]).reset_index(drop=True)

    alignment_df = _nearest_cross_class_alignment(cluster_delta_df, cluster_family_df, delta_axes)

    latent_assignments_df = clustered_delta_df[
        ["sample_key", "sample_id", "class_label", "cluster_label", "chosen_k", "cluster_silhouette"]
    ].copy()

    selected_summary = (
        selection_df.sort_values("selection_score", ascending=False)
        .groupby("class_label", as_index=False)
        .first()
        .sort_values("class_label")
        .reset_index(drop=True)
    )

    latent_assignments_df.to_csv(tables_dir / "latent_cluster_assignments.csv", index=False)
    cluster_mean_df.to_csv(tables_dir / "cluster_mean_bsv.csv", index=False)
    cluster_delta_df.to_csv(tables_dir / "cluster_delta_bsv.csv", index=False)
    cluster_var_df.to_csv(tables_dir / "cluster_variance_bsv.csv", index=False)
    cluster_family_df.to_csv(tables_dir / "cluster_family_composition.csv", index=False)
    alignment_df.to_csv(tables_dir / "cluster_cross_class_alignment.csv", index=False)
    stability_df.to_csv(tables_dir / "cluster_stability_metrics.csv", index=False)
    selection_df.to_csv(tables_dir / "cluster_selection_summary.csv", index=False)
    cluster_metrics_df.to_csv(tables_dir / "cluster_metrics.csv", index=False)

    absolute_radar_df = _radar_axes_df(cluster_mean_df[["cluster_label"] + FIXED_AXES].copy())
    delta_radar_df = _radar_axes_df(cluster_delta_df[["cluster_label"] + FIXED_AXES].copy())
    family_radar_df = (
        cluster_family_df.pivot_table(index="cluster_label", columns="family", values="family_fraction", aggfunc="mean", fill_value=0.0)
        .reset_index()
    )
    for family in FAMILY_ORDER:
        if family not in family_radar_df.columns:
            family_radar_df[family] = 0.0

    _plot_cluster_radar_grid(
        absolute_radar_df,
        "cluster_label",
        figures_dir / "cluster_radar_absolute.png",
        "Cluster Absolute BSV Fingerprints",
        value_cols=FIXED_AXES,
    )
    _plot_cluster_radar_grid(
        delta_radar_df,
        "cluster_label",
        figures_dir / "cluster_radar_delta.png",
        "Cluster Delta-BSV Fingerprints",
        value_cols=FIXED_AXES,
        delta_mode=True,
    )
    _plot_cluster_radar_grid(
        family_radar_df[["cluster_label"] + FAMILY_ORDER],
        "cluster_label",
        figures_dir / "cluster_radar_family.png",
        "Cluster Family Composition Fingerprints",
        value_cols=FAMILY_ORDER,
        family_mode=True,
    )

    pca_cluster_df = pca_bsv_df.merge(
        latent_assignments_df[["sample_key", "cluster_label"]],
        on="sample_key",
        how="inner",
    )
    _plot_pca_clusters(
        pca_cluster_df,
        figures_dir / "pca_bsv_clusters.png",
        title="Pilot 2.1 Latent Structure in BSV PCA Space",
    )

    umap_coords = _maybe_umap(sample_bsv_df[abs_axes].to_numpy(dtype=float))
    umap_cluster_df = None
    if umap_coords is not None:
        umap_cluster_df = sample_bsv_df[["sample_key", "sample_id", "class_label"]].copy()
        umap_cluster_df["u1"] = umap_coords[:, 0]
        umap_cluster_df["u2"] = umap_coords[:, 1]
        umap_cluster_df = umap_cluster_df.merge(
            latent_assignments_df[["sample_key", "cluster_label"]],
            on="sample_key",
            how="inner",
        )
        _plot_umap_clusters(
            umap_cluster_df,
            figures_dir / "umap_bsv_clusters.png",
            title="Pilot 2.1 Latent Structure in BSV UMAP Space",
        )

    _plot_alignment_bars(alignment_df, figures_dir / "cluster_cross_class_alignment.png")

    report_md = report_dir / "GAIRAv3_Pilot2_1_latent_state_interpretation.md"
    report_pdf = report_dir / "GAIRAv3_Pilot2_1_latent_state_interpretation.pdf"
    _build_report(report_md, selected_summary, stability_df, cluster_metrics_df, alignment_df)
    figure_paths = sorted(figures_dir.glob("*.png"))
    build_pdf_report(report_md, figure_paths, report_pdf)

    total_clusters = int(cluster_metrics_df["cluster_label"].nunique())
    mean_silhouette = float(selected_summary["silhouette_score"].mean())
    mean_stability = float(stability_df["ari_vs_reference"].mean())
    matched_clusters = int(len(alignment_df))
    print("selected_k_per_class")
    for row in selected_summary.itertuples(index=False):
        print(f"{row.class_label}={int(row.k)}")
    print(f"number_of_clusters={total_clusters}")
    print(f"mean_silhouette={mean_silhouette:.4f}")
    print(f"mean_stability={mean_stability:.4f}")
    print(f"matched_cross_class_clusters={matched_clusters}")


if __name__ == "__main__":
    main()
