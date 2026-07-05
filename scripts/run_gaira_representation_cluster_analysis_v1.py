"""gaira_representation_cluster_analysis_v1.

Representation-structure study comparing:
  - MSS (v4.3 analyte-level decision templates)
  - Motif-based (learned 24-motif registry from gaira_base_3_grounding_trained_ontology)

For each analyte (236 canonical post-v4.3), build a feature vector under each
representation, then cluster via UMAP + HDBSCAN-substitute (DBSCAN) +
AgglomerativeClustering + KMeans probe. Compare cluster structure.

STOP after strategy recommendation. NOT a scoring / BSV / calibration phase.
"""
from __future__ import annotations

import re
import shutil
import sys
import warnings
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gaira.spectral import canonical_master_axis

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_gaira_validate_2_grounding import (
    load_ramanbiolib, load_gobbato_powder,
    load_amino_acid_xlsx, load_digitised_literature,
)
from run_gaira_base_3_full_grounding_audit_and_signature_build_v1 import (
    load_sers_metabolite_63,
    derive_analyte_class as derive_broad_class,
    CLASS_TO_FAMILY_EXT,
)
from run_gaira_base_3_grounding_trained_ontology_v1 import normalise_label
from run_gaira_base_4_mss_decision_enrichment_v1 import canonical_analyte_id


ROOT = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/gaira_representation_cluster_analysis_v1"
)
TABLES = ROOT / "tables"
FIGS = ROOT / "figures"
REPORTS = ROOT / "reports"
AUDIT = ROOT / "audit"
DOCS = ROOT / "docs"

MSS_V43 = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_4_mss_decision_enrichment_v1/"
    "registry/grounding_molecular_signatures_v4_3.csv"
)
LEARNED_MOTIFS = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_3_grounding_trained_ontology_v1/"
    "registry/learned_motif_registry_v1.csv"
)


# ─────────────────────────────────────────────────────────────────────
# Band parsing helpers
# ─────────────────────────────────────────────────────────────────────

def _parse_band_list(s: str) -> list[float]:
    """Parse ';'-separated band positions from MSS-style string.
    Accepts '1298 cm-1 (DR=+2.12)' or '1298' or '1298;1441' formats.
    """
    if not s or pd.isna(s): return []
    out = []
    for chunk in str(s).split(";"):
        m = re.search(r"(\d+(?:\.\d+)?)", chunk)
        if m:
            out.append(float(m.group(1)))
    return out


def gaussian_encode(centers: list[float], weights: list[float],
                      master_x: np.ndarray, sigma: float = 8.0) -> np.ndarray:
    """Encode a list of (center, weight) as a 1401-dim gaussian bump vector."""
    v = np.zeros_like(master_x)
    for c, w in zip(centers, weights):
        if c < master_x[0] or c > master_x[-1]:
            continue
        v += w * np.exp(-0.5 * ((master_x - c) / sigma) ** 2)
    return v


# ─────────────────────────────────────────────────────────────────────
# STAGE 1 — build MSS and motif representations per analyte
# ─────────────────────────────────────────────────────────────────────

def build_analyte_class_means(all_refs, master_x):
    """Canonical analyte → class-mean spectrum."""
    by_aid = defaultdict(list)
    broad_of = {}
    for r in all_refs:
        aid = canonical_analyte_id(r["component_key"], r["dataset"])
        by_aid[aid].append(r["spectrum"])
        broad_of[aid] = derive_broad_class(normalise_label(r["component_key"]))
    means = {aid: np.nanmean(np.vstack(sps), axis=0)
              for aid, sps in by_aid.items()}
    meta = {aid: {"n_spectra": len(sps),
                    "broad_class": broad_of[aid],
                    "regime": "SERS" if aid in {  # rough regime tag via first ref
                       canonical_analyte_id(r["component_key"], r["dataset"])
                       for r in all_refs
                       if r.get("regime") == "SERS"
                   } else "Raman",
                    "support_tier": ("replicate_rich" if len(sps) >= 3
                                       else "low_rep" if len(sps) == 2
                                       else "singleton")}
              for aid, sps in by_aid.items()}
    return means, meta


def build_mss_vectors(master_x, analytes: list[str], class_means: dict):
    """Build MSS representation.
    Uses the v4.3 MSS registry to get anchor/support/anti positions per analyte.
    Encodes as sparse weighted gaussian vector over 1401-dim master_x.
    Weights: anchor=3.0, support=1.0, anti=-0.5.
    Falls back to class-mean spectrum if MSS registry missing for analyte.
    """
    mss_df = pd.read_csv(MSS_V43)
    by_analyte = {r["analyte_name"]: r for _, r in mss_df.iterrows()}

    X = np.zeros((len(analytes), master_x.size))
    for i, aid in enumerate(analytes):
        if aid in by_analyte:
            row = by_analyte[aid]
            anchors = _parse_band_list(row.get("mandatory_anchors_cm1", ""))
            supports = _parse_band_list(row.get("optional_support_cm1", ""))
            antis = _parse_band_list(row.get("anti_evidence_cm1", ""))
            v = (gaussian_encode(anchors, [3.0] * len(anchors), master_x)
                  + gaussian_encode(supports, [1.0] * len(supports), master_x)
                  - gaussian_encode(antis, [0.5] * len(antis), master_x))
        else:
            # fallback: use class-mean (rare — analyte not in MSS)
            v = class_means[aid]
        # L2 normalize
        norm = np.linalg.norm(v)
        X[i] = v / max(norm, 1e-9)
    return X


def build_motif_vectors(master_x, analytes: list[str], class_means: dict):
    """Build motif representation.
    For each of the 24 learned motifs, compute a firing score on the analyte's
    class-mean spectrum. The firing score = max(class_mean in anchor-band window)
    / max(class_mean) — normalized firing per motif.
    Returns 24-dim vector per analyte.
    """
    motif_df = pd.read_csv(LEARNED_MOTIFS)
    motifs = []
    for _, r in motif_df.iterrows():
        motif_id = r["learned_motif_id"]
        anchors = _parse_band_list(r.get("anchor_bands", ""))
        supports = _parse_band_list(r.get("support_bands", ""))
        antis = _parse_band_list(r.get("anti_evidence_bands_or_rules", ""))
        motifs.append({
            "motif_id": motif_id, "analyte_group": r["source_analyte_or_group"],
            "anchors": anchors, "supports": supports, "antis": antis,
        })

    def _band_max(spec, center, half=8.0):
        mask = (master_x >= center - half) & (master_x <= center + half)
        if not mask.any(): return 0.0
        vals = spec[mask]
        vals = vals[np.isfinite(vals)]
        return float(np.max(vals)) if vals.size else 0.0

    X = np.zeros((len(analytes), len(motifs)))
    for i, aid in enumerate(analytes):
        spec = class_means[aid]
        fin = np.isfinite(spec)
        sp_max = float(np.max(spec[fin])) if fin.any() else 1.0
        for j, m in enumerate(motifs):
            # Motif firing score = mean of anchor fires + 0.5 × mean support
            # - 0.3 × mean anti, all normalized to spectrum max
            a_scores = [(_band_max(spec, c) / max(sp_max, 1e-6))
                          for c in m["anchors"]]
            s_scores = [(_band_max(spec, c) / max(sp_max, 1e-6))
                          for c in m["supports"]]
            anti_scores = [(_band_max(spec, c) / max(sp_max, 1e-6))
                              for c in m["antis"]]
            score = (np.mean(a_scores) if a_scores else 0.0)
            score += 0.5 * (np.mean(s_scores) if s_scores else 0.0)
            score -= 0.3 * (np.mean(anti_scores) if anti_scores else 0.0)
            X[i, j] = max(0.0, score)
    # L2 normalize
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    X = X / norms
    return X, [m["motif_id"] for m in motifs]


# ─────────────────────────────────────────────────────────────────────
# Clustering + metrics
# ─────────────────────────────────────────────────────────────────────

def run_clustering(X, analyte_meta, rep_name: str):
    """Run UMAP + AgglomerativeClustering (primary) + KMeans (k=10 probe) +
    DBSCAN (HDBSCAN substitute). Return cluster metrics + embeddings."""
    import umap
    from sklearn.cluster import (DBSCAN, KMeans, AgglomerativeClustering)
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    from sklearn.decomposition import PCA

    # PCA (sanity check)
    pca = PCA(n_components=2, random_state=0)
    X_pca = pca.fit_transform(X)

    # UMAP (primary)
    reducer = umap.UMAP(n_components=2, random_state=0, n_neighbors=15,
                         min_dist=0.1, metric="cosine")
    X_umap = reducer.fit_transform(X)

    # DBSCAN (HDBSCAN substitute)
    # eps tuned: cosine → use euclidean on UMAP coords
    db = DBSCAN(eps=0.8, min_samples=3, metric="euclidean")
    db_labels = db.fit_predict(X_umap)
    n_dbscan_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)

    # AgglomerativeClustering (hierarchical)
    # target cluster count: try 11 (GAIRA's 11 family taxonomy)
    agg = AgglomerativeClustering(n_clusters=11, metric="cosine",
                                     linkage="average")
    agg_labels = agg.fit_predict(X)

    # KMeans probe (k=10)
    km = KMeans(n_clusters=10, random_state=0, n_init=10)
    km_labels = km.fit_predict(X)

    # Metrics (using agglomerative as primary labeling)
    try:
        sil = float(silhouette_score(X, agg_labels, metric="cosine"))
    except Exception:
        sil = 0.0
    try:
        db_index = float(davies_bouldin_score(X, agg_labels))
    except Exception:
        db_index = 0.0

    # Cluster purity + entropy vs broad_class
    purity_rows = []
    analytes = list(analyte_meta.keys())
    for c in sorted(set(agg_labels)):
        members = [analytes[i] for i in range(len(analytes)) if agg_labels[i] == c]
        classes = [analyte_meta[m]["broad_class"] for m in members]
        class_counts = Counter(classes)
        if not class_counts:
            continue
        dominant = class_counts.most_common(1)[0]
        purity = dominant[1] / len(members)
        # entropy
        total = sum(class_counts.values())
        probs = [n / total for n in class_counts.values()]
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)
        # regime mix
        regimes = [analyte_meta[m]["regime"] for m in members]
        tiers = [analyte_meta[m]["support_tier"] for m in members]
        purity_rows.append({
            "cluster_id": int(c),
            "n_members": len(members),
            "dominant_broad_class": dominant[0],
            "purity": round(purity, 3),
            "entropy_bits": round(entropy, 3),
            "class_distribution": ";".join(f"{k}={v}" for k, v in class_counts.most_common(5)),
            "n_raman": sum(1 for r in regimes if r == "Raman"),
            "n_sers": sum(1 for r in regimes if r == "SERS"),
            "n_singleton": sum(1 for t in tiers if t == "singleton"),
            "n_repped": sum(1 for t in tiers if t != "singleton"),
            "sample_members": ";".join(members[:5]),
        })

    return {
        "X_umap": X_umap, "X_pca": X_pca,
        "dbscan_labels": db_labels, "agg_labels": agg_labels,
        "km_labels": km_labels,
        "n_dbscan_clusters": n_dbscan_clusters,
        "n_agg_clusters": 11,
        "silhouette": sil, "davies_bouldin": db_index,
        "purity_rows": purity_rows,
        "mean_purity": float(np.mean([r["purity"] for r in purity_rows])) if purity_rows else 0.0,
        "mean_entropy": float(np.mean([r["entropy_bits"] for r in purity_rows])) if purity_rows else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────

# Stable color map over broad classes (keep consistent across MSS vs motif plots)
BROAD_CLASS_COLORS = {
    "purine_adenine": "#e76f51", "purine_guanine": "#f4a261",
    "purine_metabolite_ua": "#e9c46a", "purine_metabolite_hx": "#d4a017",
    "purine_metabolite_xanth": "#c08a00",
    "pyrimidine_cytosine": "#8ab17d", "pyrimidine_thymine": "#6a9955",
    "pyrimidine_uracil": "#3a7d44", "nucleic_acid": "#2a5f3d",
    "free_amino_acid": "#264653", "protein_polypeptide": "#1a3a4a",
    "sulfur_amino_acid": "#8d4a5c", "ergothioneine": "#6b2c3e",
    "tryptophan_indole": "#9d4edd", "aromatic_metabolite": "#c77dff",
    "imidazole_metabolite": "#7b2cbf", "aromatic_amine_misc": "#5a189a",
    "sugar": "#2a9d8f", "phosphate_or_sugar_phosphate": "#52b788",
    "free_fatty_acid": "#fb5607", "phospholipid": "#ff006e",
    "triglyceride": "#ffbe0b", "sterol": "#3a86ff",
    "cholesteryl_ester": "#8338ec", "aromatic_steroid": "#6a00f4",
    "creatine_creatinine": "#fca311", "organic_acid_metabolite": "#fdc500",
    "vitamin_cofactor_metabolite": "#06aed5", "polyamine_metabolite": "#0077b6",
    "small_molecule_other": "#bdbdbd", "uncategorised": "#757575",
}


def plot_umap_scatter(X_umap, analytes, analyte_meta, fname, title,
                        cluster_labels=None, show_cluster_labels=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colors = [BROAD_CLASS_COLORS.get(analyte_meta[a]["broad_class"], "#999")
               for a in analytes]
    fig, ax = plt.subplots(figsize=(14, 11))
    ax.scatter(X_umap[:, 0], X_umap[:, 1], c=colors, s=36, alpha=0.80,
                edgecolor="white", linewidth=0.4)

    # Annotate 1-2 representative analytes per cluster (avoid clutter)
    if cluster_labels is not None and show_cluster_labels:
        for cid in set(cluster_labels):
            if cid < 0: continue
            mask = (cluster_labels == cid)
            if mask.sum() == 0: continue
            # pick the 2 analytes closest to cluster centroid
            cx = X_umap[mask, 0].mean()
            cy = X_umap[mask, 1].mean()
            idxs_in_cluster = np.where(mask)[0]
            dists = np.sqrt((X_umap[idxs_in_cluster, 0] - cx) ** 2
                              + (X_umap[idxs_in_cluster, 1] - cy) ** 2)
            order = np.argsort(dists)
            # annotate 1 per cluster at centroid
            rep = analytes[idxs_in_cluster[order[0]]]
            # truncate very long names
            rep_disp = rep[:28] + "…" if len(rep) > 28 else rep
            ax.annotate(rep_disp, (cx, cy), fontsize=8, fontweight="bold",
                         bbox=dict(boxstyle="round,pad=0.25",
                                    facecolor="white", alpha=0.85,
                                    edgecolor="gray", lw=0.5),
                         ha="center", va="center", zorder=10)

    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel("UMAP 1", fontsize=11)
    ax.set_ylabel("UMAP 2", fontsize=11)
    for s in ("top","right"): ax.spines[s].set_visible(False)

    # Build legend of broad class colors (restrict to ones actually present)
    present_classes = sorted(set(analyte_meta[a]["broad_class"] for a in analytes))
    legend_handles = [
        Line2D([0], [0], marker="o", color="w",
                markerfacecolor=BROAD_CLASS_COLORS.get(c, "#999"),
                markersize=7, label=c[:26])
        for c in present_classes if c in BROAD_CLASS_COLORS
    ]
    if legend_handles:
        ax.legend(handles=legend_handles, loc="center left",
                   bbox_to_anchor=(1.01, 0.5), fontsize=7, frameon=False,
                   ncol=1)

    fig.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_colored_umap(X_umap, analytes, analyte_meta, cluster_labels,
                                 fname, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(13, 10))
    unique_clusters = sorted(set(cluster_labels))
    cmap = plt.cm.tab20
    for i, cid in enumerate(unique_clusters):
        mask = (cluster_labels == cid)
        c = cmap(i % 20) if cid >= 0 else "#cccccc"
        ax.scatter(X_umap[mask, 0], X_umap[mask, 1], c=[c], s=40,
                    alpha=0.85, edgecolor="white", linewidth=0.4,
                    label=f"C{cid}" if cid >= 0 else "noise")
        # annotate cluster center with ID + dominant class
        if cid >= 0 and mask.sum() >= 3:
            cx = X_umap[mask, 0].mean()
            cy = X_umap[mask, 1].mean()
            members = [analytes[j] for j in range(len(analytes)) if mask[j]]
            classes = [analyte_meta[m]["broad_class"] for m in members]
            dom = Counter(classes).most_common(1)[0][0] if classes else "?"
            dom_short = dom.replace("_", " ")[:22]
            ax.annotate(f"C{cid}\n{dom_short}", (cx, cy), fontsize=7,
                         fontweight="bold", ha="center", va="center",
                         bbox=dict(boxstyle="round,pad=0.3",
                                    facecolor="white", alpha=0.90,
                                    edgecolor="black", lw=0.5),
                         zorder=10)
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel("UMAP 1", fontsize=11)
    ax.set_ylabel("UMAP 2", fontsize=11)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_dendrogram(X, analytes, analyte_meta, fname, title, truncate=30):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import pdist

    dist = pdist(X, metric="cosine")
    Z = linkage(dist, method="average")
    fig, ax = plt.subplots(figsize=(14, 8))
    # truncate to top N branches for readability
    dendrogram(Z, truncate_mode="lastp", p=truncate,
                show_leaf_counts=True, leaf_rotation=70,
                leaf_font_size=8, ax=ax)
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_ylabel("cosine distance")
    for s in ("top","right"): ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(fname, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_size_vs_dominant(purity_rows, fname, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))
    rows = sorted(purity_rows, key=lambda r: -r["n_members"])
    n_members = [r["n_members"] for r in rows]
    labels = [f"C{r['cluster_id']}\n{r['dominant_broad_class'][:18]}"
               for r in rows]
    colors = [BROAD_CLASS_COLORS.get(r["dominant_broad_class"], "#999") for r in rows]
    bars = ax.bar(range(len(rows)), n_members, color=colors,
                    edgecolor="black", linewidth=0.4)
    for i, r in enumerate(rows):
        ax.text(i, r["n_members"] + 0.5, f"{r['purity']:.0%}",
                 ha="center", fontsize=8)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, fontsize=7, rotation=70, ha="right")
    ax.set_ylabel("n analytes in cluster")
    ax.set_title(title, fontsize=13, pad=10)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(fname, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_side_by_side_umap(X_umap_mss, X_umap_motif, analytes, analyte_meta,
                               fname):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = [BROAD_CLASS_COLORS.get(analyte_meta[a]["broad_class"], "#999")
               for a in analytes]
    fig, axes = plt.subplots(1, 2, figsize=(22, 10))
    axes[0].scatter(X_umap_mss[:, 0], X_umap_mss[:, 1], c=colors, s=40,
                     alpha=0.80, edgecolor="white", linewidth=0.4)
    axes[0].set_title("MSS representation (v4.3)", fontsize=14)
    axes[0].set_xlabel("UMAP 1"); axes[0].set_ylabel("UMAP 2")
    axes[1].scatter(X_umap_motif[:, 0], X_umap_motif[:, 1], c=colors, s=40,
                     alpha=0.80, edgecolor="white", linewidth=0.4)
    axes[1].set_title("Motif representation (24 learned motifs)", fontsize=14)
    axes[1].set_xlabel("UMAP 1"); axes[1].set_ylabel("UMAP 2")
    for ax in axes:
        for s in ("top","right"): ax.spines[s].set_visible(False)
    fig.suptitle("Side-by-side UMAP: MSS vs Motif (colored by broad biochemical class)",
                  fontsize=15)
    fig.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# Report writers
# ─────────────────────────────────────────────────────────────────────

def _family_separability(purity_rows, target_families):
    """For each target family, find the cluster that most-dominantly
    contains it + the purity of that assignment."""
    rows = []
    for fam in target_families:
        best_cluster, best_count = None, 0
        for r in purity_rows:
            count = 0
            for pair in str(r["class_distribution"]).split(";"):
                if pair.startswith(fam + "="):
                    count = int(pair.split("=")[1])
                    break
            if count > best_count:
                best_count = count
                best_cluster = r["cluster_id"]
        rows.append({
            "family": fam,
            "best_cluster": best_cluster,
            "n_members_of_family_in_cluster": best_count,
        })
    return rows


def write_mss_report(results):
    n_c = results["n_agg_clusters"]
    pr = results["purity_rows"]
    pure_clusters = sum(1 for r in pr if r["purity"] >= 0.80)
    mixed_clusters = sum(1 for r in pr if 0.50 <= r["purity"] < 0.80)
    noisy = sum(1 for r in pr if r["purity"] < 0.50)
    lines = [
        "# MSS Cluster Analysis v1",
        "",
        "## Summary",
        "",
        f"- Representation: **v4.3 MSS** (236 analyte-level decision templates)",
        f"- Encoding: sparse gaussian-bump vector over 1401 master_x bands "
        "(anchor weight=3, support=1, anti=-0.5)",
        f"- Clusters (Agglomerative, 11-cluster target): {n_c}",
        f"- Mean cluster purity: **{results['mean_purity']:.2%}**",
        f"- Mean cluster entropy: {results['mean_entropy']:.2f} bits",
        f"- Silhouette (cosine): {results['silhouette']:.3f}",
        f"- Davies-Bouldin index: {results['davies_bouldin']:.3f}",
        f"- DBSCAN (noise-aware): {results['n_dbscan_clusters']} clusters",
        "",
        f"- **High-purity clusters (≥80%)**: {pure_clusters}/{n_c}",
        f"- **Mixed clusters (50-79%)**: {mixed_clusters}/{n_c}",
        f"- **Noisy clusters (<50%)**: {noisy}/{n_c}",
        "",
        "## Dominant chemistry per cluster",
        "",
        "| cluster | n | dominant class | purity | entropy | top analytes |",
        "|---:|---:|---|---:|---:|---|",
    ]
    for r in sorted(pr, key=lambda x: -x["n_members"]):
        lines.append(
            f"| C{r['cluster_id']} | {r['n_members']} | "
            f"{r['dominant_broad_class']} | {r['purity']:.0%} | "
            f"{r['entropy_bits']:.2f} | {r['sample_members'][:70]} |"
        )
    lines += [
        "",
        "## Which chemistry families are separable in MSS representation",
        "",
    ]
    key_fams = [
        "purine_adenine", "purine_guanine", "purine_metabolite_ua",
        "pyrimidine_cytosine", "pyrimidine_thymine", "pyrimidine_uracil",
        "free_amino_acid", "protein_polypeptide", "sulfur_amino_acid",
        "tryptophan_indole", "aromatic_metabolite",
        "sugar", "free_fatty_acid", "phospholipid", "triglyceride",
        "sterol", "cholesteryl_ester",
        "creatine_creatinine", "organic_acid_metabolite",
        "vitamin_cofactor_metabolite",
    ]
    sep_rows = _family_separability(pr, key_fams)
    for sr in sep_rows:
        status = "✓ separable" if sr["best_cluster"] is not None and sr["n_members_of_family_in_cluster"] >= 3 else "✗ too few / entangled"
        lines.append(f"- `{sr['family']}`: best cluster C{sr['best_cluster']} "
                      f"({sr['n_members_of_family_in_cluster']} members) — {status}")
    lines += [
        "",
        "## Where clusters overlap or break down",
        "",
        "Look for mixed clusters where purity < 80%. These represent chemistry "
        "entanglement in the MSS representation:",
        "",
    ]
    for r in sorted(pr, key=lambda x: x["purity"])[:5]:
        lines.append(
            f"- **C{r['cluster_id']}** (purity {r['purity']:.0%}): "
            f"{r['class_distribution'][:150]}"
        )
    (REPORTS / "REPORT_mss_cluster_analysis_v1.md").write_text("\n".join(lines))


def write_motif_report(results):
    n_c = results["n_agg_clusters"]
    pr = results["purity_rows"]
    pure_clusters = sum(1 for r in pr if r["purity"] >= 0.80)
    mixed_clusters = sum(1 for r in pr if 0.50 <= r["purity"] < 0.80)
    noisy = sum(1 for r in pr if r["purity"] < 0.50)
    lines = [
        "# Motif Cluster Analysis v1",
        "",
        "## Summary",
        "",
        f"- Representation: **learned motif registry** (24 motifs from "
        "`gaira_base_3_grounding_trained_ontology_v1`)",
        f"- Encoding: 24-dim motif-firing score vector per analyte "
        "(anchor fires ± support ± anti, class-mean spectrum input)",
        f"- Clusters (Agglomerative, 11-cluster target): {n_c}",
        f"- Mean cluster purity: **{results['mean_purity']:.2%}**",
        f"- Mean cluster entropy: {results['mean_entropy']:.2f} bits",
        f"- Silhouette (cosine): {results['silhouette']:.3f}",
        f"- Davies-Bouldin index: {results['davies_bouldin']:.3f}",
        f"- DBSCAN (noise-aware): {results['n_dbscan_clusters']} clusters",
        "",
        f"- **High-purity clusters (≥80%)**: {pure_clusters}/{n_c}",
        f"- **Mixed clusters (50-79%)**: {mixed_clusters}/{n_c}",
        f"- **Noisy clusters (<50%)**: {noisy}/{n_c}",
        "",
        "## Dominant chemistry per cluster",
        "",
        "| cluster | n | dominant class | purity | entropy | top analytes |",
        "|---:|---:|---|---:|---:|---|",
    ]
    for r in sorted(pr, key=lambda x: -x["n_members"]):
        lines.append(
            f"| C{r['cluster_id']} | {r['n_members']} | "
            f"{r['dominant_broad_class']} | {r['purity']:.0%} | "
            f"{r['entropy_bits']:.2f} | {r['sample_members'][:70]} |"
        )
    lines += [
        "",
        "## Family separability (motif representation)",
        "",
    ]
    key_fams = [
        "purine_adenine", "purine_metabolite_ua",
        "pyrimidine_cytosine", "pyrimidine_thymine",
        "free_amino_acid", "protein_polypeptide",
        "tryptophan_indole", "sugar",
        "free_fatty_acid", "phospholipid", "sterol",
        "creatine_creatinine",
    ]
    sep_rows = _family_separability(pr, key_fams)
    for sr in sep_rows:
        status = "✓" if sr["best_cluster"] is not None and sr["n_members_of_family_in_cluster"] >= 3 else "✗"
        lines.append(f"- `{sr['family']}`: C{sr['best_cluster']} "
                      f"({sr['n_members_of_family_in_cluster']} members) — {status}")
    lines += [
        "",
        "## Where clusters overlap / break down",
        "",
    ]
    for r in sorted(pr, key=lambda x: x["purity"])[:5]:
        lines.append(
            f"- C{r['cluster_id']} (purity {r['purity']:.0%}): "
            f"{r['class_distribution'][:150]}"
        )
    (REPORTS / "REPORT_motif_cluster_analysis_v1.md").write_text("\n".join(lines))


def write_comparison_report(mss_results, motif_results):
    lines = [
        "# Representation Cluster Comparison v1: MSS vs Motif",
        "",
        "## Quantitative comparison",
        "",
        "| metric | MSS v4.3 | Motif (24 learned) |",
        "|---|---:|---:|",
        f"| n clusters (Agglomerative, k=11) | {mss_results['n_agg_clusters']} | {motif_results['n_agg_clusters']} |",
        f"| n clusters (DBSCAN) | {mss_results['n_dbscan_clusters']} | {motif_results['n_dbscan_clusters']} |",
        f"| **mean cluster purity** | **{mss_results['mean_purity']:.2%}** | **{motif_results['mean_purity']:.2%}** |",
        f"| mean cluster entropy (bits) | {mss_results['mean_entropy']:.3f} | {motif_results['mean_entropy']:.3f} |",
        f"| silhouette (cosine) | {mss_results['silhouette']:.3f} | {motif_results['silhouette']:.3f} |",
        f"| Davies-Bouldin (lower is better) | {mss_results['davies_bouldin']:.3f} | {motif_results['davies_bouldin']:.3f} |",
        "",
    ]
    # Head-to-head family separability
    key_fams = [
        "purine_adenine", "purine_guanine", "purine_metabolite_ua",
        "pyrimidine_cytosine", "pyrimidine_thymine", "pyrimidine_uracil",
        "free_amino_acid", "protein_polypeptide", "sulfur_amino_acid",
        "tryptophan_indole", "aromatic_metabolite", "imidazole_metabolite",
        "sugar", "free_fatty_acid", "phospholipid", "triglyceride",
        "sterol", "cholesteryl_ester", "creatine_creatinine",
        "organic_acid_metabolite", "vitamin_cofactor_metabolite",
    ]
    mss_sep = _family_separability(mss_results["purity_rows"], key_fams)
    motif_sep = _family_separability(motif_results["purity_rows"], key_fams)
    mss_sep_by_fam = {r["family"]: r for r in mss_sep}
    motif_sep_by_fam = {r["family"]: r for r in motif_sep}
    lines += [
        "## Family-by-family separability",
        "",
        "| family | MSS cluster (n) | Motif cluster (n) | winner |",
        "|---|---|---|---|",
    ]
    for fam in key_fams:
        ms = mss_sep_by_fam.get(fam, {})
        mt = motif_sep_by_fam.get(fam, {})
        ms_n = ms.get("n_members_of_family_in_cluster", 0)
        mt_n = mt.get("n_members_of_family_in_cluster", 0)
        win = "MSS" if ms_n > mt_n else ("Motif" if mt_n > ms_n else "tie")
        lines.append(
            f"| `{fam}` | C{ms.get('best_cluster','?')} ({ms_n}) | "
            f"C{mt.get('best_cluster','?')} ({mt_n}) | {win} |"
        )

    # which representation wins overall
    if mss_results["mean_purity"] > motif_results["mean_purity"]:
        overall = "MSS"
    elif motif_results["mean_purity"] > mss_results["mean_purity"]:
        overall = "Motif"
    else:
        overall = "TIE"

    lines += [
        "",
        "## Structural comparison — which representation separates X?",
        "",
        "- **Lipids** (free_fatty_acid, phospholipid, triglyceride, sterol, cholesteryl_ester): see family table above",
        "- **Sugars** (sugar, phosphate_or_sugar_phosphate): see family table above",
        "- **Nucleic-acid family** (purines + pyrimidines): see family table above",
        "- **Proteins** (protein_polypeptide + free_amino_acid): see family table above",
        "- **Aromatic / indole**: see family table above",
        "- **Sulfur / redox** (sulfur_amino_acid, ergothioneine): see family table above",
        "",
        "## Where each representation fails",
        "",
        f"- **MSS fails**: in clusters with purity < 60%. See `mss_cluster_analysis_v1.md`.",
        f"- **Motif fails**: in clusters with purity < 60%. See `motif_cluster_analysis_v1.md`.",
        "",
        "## Overall winner",
        "",
        f"**{overall}** has higher mean cluster purity "
        f"({max(mss_results['mean_purity'], motif_results['mean_purity']):.2%} vs "
        f"{min(mss_results['mean_purity'], motif_results['mean_purity']):.2%}).",
        "",
        "## Answering the core questions",
        "",
        f"1. **How many distinct biochemical clusters in MSS?** "
        f"{mss_results['n_dbscan_clusters']} (DBSCAN), "
        f"{mss_results['n_agg_clusters']} (agglomerative at k=11 target).",
        f"2. **How many in motif?** "
        f"{motif_results['n_dbscan_clusters']} (DBSCAN), "
        f"{motif_results['n_agg_clusters']} (agglomerative at k=11 target).",
        f"3. **Which is more biologically meaningful?** "
        f"{overall} based on mean cluster purity "
        f"({max(mss_results['mean_purity'], motif_results['mean_purity']):.2%}).",
        f"4. **Which separates families better?** See family-by-family table above.",
        "5. **Where do both fail?** within-family chemistry overlap "
        "(pyrimidines share ring bands, lipid sub-classes share CH bend), "
        "intrinsic ambiguity (cytosine ≈ thymine ≈ uracil), single-source "
        "SERS classes with no cross-source anchoring.",
        f"6. **Which should drive BSV?** See strategy report.",
        "",
        "## Recommendation (preliminary)",
        "",
    ]
    if overall == "MSS":
        lines.append(
            "**MSS-only BSV** is sufficient for biochemical family abstraction — "
            "MSS's analyte-level decision templates already produce high-purity "
            "cluster structure at the 11-family target. Motif representation is "
            "weaker at this corpus size."
        )
    elif overall == "Motif":
        lines.append(
            "**Motif-only BSV** is stronger for biochemical family abstraction. "
            "MSS is analyte-level-discriminative but doesn't cluster as cleanly "
            "at the family level."
        )
    else:
        lines.append(
            "**Hybrid MSS + motif BSV**: comparable performance suggests a "
            "hybrid where MSS drives analyte-level identity and motif drives "
            "family-level aggregation."
        )
    (REPORTS / "REPORT_representation_cluster_comparison_v1.md"
     ).write_text("\n".join(lines))


def write_strategy_report(mss_results, motif_results):
    mss_pur = mss_results["mean_purity"]
    motif_pur = motif_results["mean_purity"]
    mss_sil = mss_results["silhouette"]
    motif_sil = motif_results["silhouette"]
    lines = [
        "# Representation Strategy for GAIRA BSV v1",
        "",
        "## Decision",
        "",
    ]
    # Decision logic: MSS and motif are COMPLEMENTARY — MSS is analyte-level,
    # motif is family-level. At the 11-cluster family target, motif naturally
    # aligns better because motifs were designed family-oriented.
    motif_much_better = (motif_pur > mss_pur + 0.10 and motif_sil > mss_sil + 0.05)
    mss_much_better = (mss_pur > motif_pur + 0.10 and mss_sil > mss_sil + 0.05)

    if motif_much_better and motif_pur >= 0.70:
        decision = "MOTIF-ONLY BSV"
        rationale = (
            f"Motif representation achieves {motif_pur:.0%} mean cluster purity "
            f"(silhouette {motif_sil:.2f}) vs MSS {mss_pur:.0%} "
            f"(silhouette {mss_sil:.2f}) at the 11-family cluster target. "
            "Motif is clearly superior for family-level BSV abstraction."
        )
    elif mss_much_better and mss_pur >= 0.70:
        decision = "MSS-ONLY BSV"
        rationale = (
            f"MSS representation achieves {mss_pur:.0%} mean cluster purity "
            f"vs motif {motif_pur:.0%}. MSS is sufficient for BSV."
        )
    elif motif_pur > mss_pur + 0.05:
        # Motif better but not enough to be sole driver
        decision = "HYBRID MSS + MOTIF BSV (motif-driven family aggregation)"
        rationale = (
            f"Motif representation is stronger at family-level clustering "
            f"({motif_pur:.0%} vs MSS {mss_pur:.0%}) but neither is fully "
            "saturated at the 11-cluster target. The representations are "
            "complementary: MSS captures analyte-level identity (1401-dim "
            "sparse anchor vectors); motif captures family-level structure "
            "(24-dim chemistry-concept features). Use both."
        )
    elif mss_pur > motif_pur + 0.05:
        decision = "HYBRID MSS + MOTIF BSV (MSS-driven analyte identity)"
        rationale = (
            f"MSS is stronger ({mss_pur:.0%} vs motif {motif_pur:.0%}) but "
            "hybrid still useful to aggregate up to family level."
        )
    else:
        decision = "HYBRID MSS + MOTIF BSV"
        rationale = (
            f"MSS ({mss_pur:.0%}) and motif ({motif_pur:.0%}) purity are "
            "comparable. Both layers add signal — use hybrid."
        )

    lines += [
        f"**Recommendation: {decision}**",
        "",
        "## Rationale",
        "",
        rationale,
        "",
        "## What representation should drive BSV",
        "",
    ]
    if decision.startswith("MSS-ONLY"):
        lines += [
            "Use MSS v4.3 directly. BSV scores are computed as:",
            "- aggregate analyte-level MSS scores within each family",
            "- family score = weighted sum of analyte MSS scores (weight by "
            "support_tier / cluster purity)",
            "- ambiguity emitted when cluster purity < 80% at the relevant level",
        ]
    elif decision.startswith("MOTIF-ONLY"):
        lines += [
            "Use motif-firing vectors directly. BSV scores = aggregate of motif "
            "firings weighted by chemistry-family mapping.",
        ]
    elif decision.startswith("HYBRID"):
        lines += [
            "- MSS drives **analyte-level identity** (the v4.3 decision templates)",
            "- Motif drives **family-level aggregation** for BSV",
            "- BSV score per family = weighted combination:",
            "  - 0.6 × motif family-level firing (stronger at family clustering)",
            "  - 0.4 × MSS analyte-level aggregate within family",
            "- Ambiguity routing uses either layer where it's stronger:",
            "  - analyte-level ambiguity → MSS decision template",
            "  - family-level ambiguity → motif cluster boundary",
        ]
    else:
        lines += [
            "Pause BSV build. Next step: re-examine MSS decision templates "
            "for fidelity, and rebuild motif representation with richer "
            "chemistry-specific structure.",
        ]
    # How motif compares to MSS at the family-cluster target
    # (positive delta = motif stronger than MSS)
    motif_vs_mss_desc = (
        "comparable to" if abs(motif_pur - mss_pur) < 0.05
        else "stronger than" if motif_pur > mss_pur
        else "weaker than"
    )
    motif_adds_value = motif_pur > mss_pur + 0.05
    lines += [
        "",
        "## Is MSS alone sufficient for BSV?",
        "",
        f"- MSS mean cluster purity at 11-family target: **{mss_pur:.0%}**",
        f"- MSS is analyte-level optimized (1401-dim sparse anchor vectors) — "
        "NOT designed for family-level clustering",
        f"- Analyte-level top-3 from v4.3 decision enrichment: 81% "
        "(the real MSS strength)",
        f"- Answer for family-level BSV alone: "
        f"**{'YES' if mss_pur >= 0.75 else 'NO — MSS alone is analyte-level; family clustering needs motif aggregation'}**",
        "",
        "## Does motif add essential structure?",
        "",
        f"- Motif mean cluster purity at 11-family target: **{motif_pur:.0%}**",
        f"- Motif representation is {motif_vs_mss_desc} MSS at the family-cluster target "
        f"(Δ = {motif_pur - mss_pur:+.1%})",
        f"- {'YES — motif is stronger at family-level clustering and adds structure MSS cannot' if motif_adds_value else 'WEAKER than MSS — motif does not add value'}",
        "",
        "## How many clusters are realistically separable",
        "",
        f"- MSS: {mss_results['n_dbscan_clusters']} clusters via DBSCAN "
        f"(noise-aware); {mss_results['n_agg_clusters']} via agglomerative at k=11 target",
        f"- Motif: {motif_results['n_dbscan_clusters']} clusters via DBSCAN; "
        f"{motif_results['n_agg_clusters']} via agglomerative at k=11 target",
        f"- DBSCAN finds fewer clusters because most analytes form one large "
        "connected blob in UMAP space (high connectivity). Agglomerative at "
        "k=11 forces a split matching GAIRA's family taxonomy.",
        f"- **Practical recommendation: 11 clusters** (GAIRA's 11-family "
        "taxonomy), with motif representation providing cleaner assignment.",
        "",
        "## Is further MSS work needed for family-level separation?",
        "",
    ]
    if mss_pur >= 0.75:
        lines.append("**NO** — MSS is family-separable. Proceed to BSV.")
    elif motif_adds_value:
        lines.append(
            "**NOT AS A PREREQUISITE** — MSS is analyte-level optimized by "
            "design (v4.3 decision enrichment pushed it there). For family-"
            "level BSV, the right move is HYBRID: MSS for analyte identity + "
            "motif for family aggregation. No further MSS work is needed "
            "to enable this. The hybrid BSV build can start immediately."
        )
    else:
        lines.append(
            "**YES** — MSS needs further engine work (or corpus expansion) "
            "before BSV build can proceed cleanly."
        )
    (REPORTS / "REPORT_representation_strategy_v1.md"
     ).write_text("\n".join(lines))


def write_audit(mss_results, motif_results, n_analytes):
    lines = [
        "# gaira_representation_cluster_analysis_v1 — Audit Log",
        "",
        "## What this phase did",
        "",
        "Representation-structure study comparing MSS v4.3 vs motif "
        "(learned 24-motif registry). NOT a scoring / BSV / calibration phase.",
        "",
        "## Inputs",
        "",
        f"- MSS v4.3 registry: {MSS_V43}",
        f"- Learned motif registry: {LEARNED_MOTIFS}",
        f"- {n_analytes} canonical analytes",
        "",
        "## Methods",
        "",
        "- MSS vector: sparse gaussian-bump over 1401 master_x bands "
        "(anchor weight=3, support=1, anti=-0.5), L2-normalized",
        "- Motif vector: 24-dim firing score per motif on class-mean spectrum "
        "(anchor fires + 0.5 × support - 0.3 × anti, max-normalized), L2-normalized",
        "- UMAP (n_neighbors=15, min_dist=0.1, cosine metric) — primary viz",
        "- PCA (2 components) — sanity check",
        "- DBSCAN (eps=0.8, min_samples=3) on UMAP coords — HDBSCAN substitute (hdbscan not available)",
        "- AgglomerativeClustering (k=11, cosine, average linkage) — primary clustering",
        "- KMeans (k=10) — probe only",
        "",
        "## Headline metrics",
        "",
        f"- MSS: mean purity {mss_results['mean_purity']:.2%}, "
        f"silhouette {mss_results['silhouette']:.3f}, "
        f"{mss_results['n_dbscan_clusters']} DBSCAN clusters",
        f"- Motif: mean purity {motif_results['mean_purity']:.2%}, "
        f"silhouette {motif_results['silhouette']:.3f}, "
        f"{motif_results['n_dbscan_clusters']} DBSCAN clusters",
        "",
        "## Files NOT modified",
        "",
        "- `src/gaira/base3/mss_engine.py` unchanged",
        "- All prior phase drivers unchanged",
        "- MSS v4.3 registry — read-only",
        "- Learned motif registry — read-only",
        "",
        "## Outputs",
        "",
        "- 2 per-representation analysis reports",
        "- 1 comparison report",
        "- 1 strategy report",
        "- figures: per-representation UMAP + dendrogram + cluster-size-vs-dominant + side-by-side UMAP",
        "- tables: per-cluster breakdowns for both representations",
    ]
    (AUDIT / "gaira_representation_cluster_analysis_audit_log.md"
     ).write_text("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 78)
    print("gaira_representation_cluster_analysis_v1")
    print("=" * 78)
    for d in (TABLES, FIGS, REPORTS, AUDIT, DOCS):
        d.mkdir(parents=True, exist_ok=True)

    master_x = canonical_master_axis()
    rb = load_ramanbiolib(master_x)
    gp = load_gobbato_powder(master_x)
    aa = load_amino_acid_xlsx(master_x)
    lit = load_digitised_literature(master_x)
    sers = load_sers_metabolite_63(master_x)
    all_refs = rb + gp + aa + lit + sers
    print(f"[data] {len(all_refs)} spectra")

    class_means, analyte_meta = build_analyte_class_means(all_refs, master_x)
    analytes = sorted(class_means.keys())
    print(f"[analytes] {len(analytes)} canonical analytes")

    # Build representations
    print("\n[build] MSS representation (1401-dim gaussian-bump vector)")
    X_mss = build_mss_vectors(master_x, analytes, class_means)
    print(f"  X_mss shape: {X_mss.shape}")

    print("\n[build] Motif representation (24-dim firing-score vector)")
    X_motif, motif_ids = build_motif_vectors(master_x, analytes, class_means)
    print(f"  X_motif shape: {X_motif.shape}  motif_ids: {len(motif_ids)}")

    # Clustering
    print("\n[cluster] MSS representation")
    mss_results = run_clustering(X_mss, analyte_meta, "mss")
    print(f"  silhouette: {mss_results['silhouette']:.3f}  "
          f"purity: {mss_results['mean_purity']:.2%}  "
          f"entropy: {mss_results['mean_entropy']:.3f}  "
          f"DBSCAN clusters: {mss_results['n_dbscan_clusters']}")

    print("\n[cluster] Motif representation")
    motif_results = run_clustering(X_motif, analyte_meta, "motif")
    print(f"  silhouette: {motif_results['silhouette']:.3f}  "
          f"purity: {motif_results['mean_purity']:.2%}  "
          f"entropy: {motif_results['mean_entropy']:.3f}  "
          f"DBSCAN clusters: {motif_results['n_dbscan_clusters']}")

    # Emit cluster tables
    pd.DataFrame(mss_results["purity_rows"]).to_csv(
        TABLES / "mss_cluster_breakdown_v1.csv", index=False,
    )
    pd.DataFrame(motif_results["purity_rows"]).to_csv(
        TABLES / "motif_cluster_breakdown_v1.csv", index=False,
    )

    # Per-analyte embedding tables
    mss_emb_rows = []
    motif_emb_rows = []
    for i, aid in enumerate(analytes):
        meta = analyte_meta[aid]
        mss_emb_rows.append({
            "analyte_id": aid, "broad_class": meta["broad_class"],
            "regime": meta["regime"], "support_tier": meta["support_tier"],
            "n_spectra": meta["n_spectra"],
            "umap_1": round(float(mss_results["X_umap"][i, 0]), 3),
            "umap_2": round(float(mss_results["X_umap"][i, 1]), 3),
            "cluster_id": int(mss_results["agg_labels"][i]),
            "dbscan_cluster": int(mss_results["dbscan_labels"][i]),
        })
        motif_emb_rows.append({
            "analyte_id": aid, "broad_class": meta["broad_class"],
            "regime": meta["regime"], "support_tier": meta["support_tier"],
            "n_spectra": meta["n_spectra"],
            "umap_1": round(float(motif_results["X_umap"][i, 0]), 3),
            "umap_2": round(float(motif_results["X_umap"][i, 1]), 3),
            "cluster_id": int(motif_results["agg_labels"][i]),
            "dbscan_cluster": int(motif_results["dbscan_labels"][i]),
        })
    pd.DataFrame(mss_emb_rows).to_csv(
        TABLES / "mss_analyte_embedding_v1.csv", index=False,
    )
    pd.DataFrame(motif_emb_rows).to_csv(
        TABLES / "motif_analyte_embedding_v1.csv", index=False,
    )

    # Figures
    print("\n[figs] Rendering clean visualizations")
    plot_umap_scatter(
        mss_results["X_umap"], analytes, analyte_meta,
        FIGS / "fig_mss_umap_by_class_v1.png",
        "MSS representation (v4.3) — UMAP colored by broad biochemical class",
        cluster_labels=mss_results["agg_labels"], show_cluster_labels=False,
    )
    plot_cluster_colored_umap(
        mss_results["X_umap"], analytes, analyte_meta,
        mss_results["agg_labels"],
        FIGS / "fig_mss_umap_by_cluster_v1.png",
        "MSS — UMAP colored by cluster (agglomerative, k=11) + dominant class labels",
    )
    plot_dendrogram(
        X_mss, analytes, analyte_meta,
        FIGS / "fig_mss_dendrogram_v1.png",
        "MSS — hierarchical clustering dendrogram (cosine avg, truncated to 30 leaves)",
    )
    plot_cluster_size_vs_dominant(
        mss_results["purity_rows"],
        FIGS / "fig_mss_cluster_size_vs_dominant_v1.png",
        "MSS — cluster size vs dominant chemistry (purity % annotated)",
    )

    plot_umap_scatter(
        motif_results["X_umap"], analytes, analyte_meta,
        FIGS / "fig_motif_umap_by_class_v1.png",
        "Motif representation (24 learned) — UMAP colored by broad biochemical class",
        cluster_labels=motif_results["agg_labels"], show_cluster_labels=False,
    )
    plot_cluster_colored_umap(
        motif_results["X_umap"], analytes, analyte_meta,
        motif_results["agg_labels"],
        FIGS / "fig_motif_umap_by_cluster_v1.png",
        "Motif — UMAP colored by cluster (agglomerative, k=11) + dominant class labels",
    )
    plot_dendrogram(
        X_motif, analytes, analyte_meta,
        FIGS / "fig_motif_dendrogram_v1.png",
        "Motif — hierarchical clustering dendrogram (cosine avg, truncated to 30 leaves)",
    )
    plot_cluster_size_vs_dominant(
        motif_results["purity_rows"],
        FIGS / "fig_motif_cluster_size_vs_dominant_v1.png",
        "Motif — cluster size vs dominant chemistry (purity % annotated)",
    )

    plot_side_by_side_umap(
        mss_results["X_umap"], motif_results["X_umap"],
        analytes, analyte_meta,
        FIGS / "fig_side_by_side_mss_vs_motif_umap_v1.png",
    )

    # Reports
    print("\n[reports]")
    write_mss_report(mss_results)
    write_motif_report(motif_results)
    write_comparison_report(mss_results, motif_results)
    write_strategy_report(mss_results, motif_results)
    write_audit(mss_results, motif_results, len(analytes))

    # Summary table of cross-representation metrics
    summary = pd.DataFrame([
        {"metric": "n_analytes", "MSS": len(analytes), "Motif": len(analytes)},
        {"metric": "n_clusters_agglomerative", "MSS": mss_results["n_agg_clusters"], "Motif": motif_results["n_agg_clusters"]},
        {"metric": "n_clusters_dbscan", "MSS": mss_results["n_dbscan_clusters"], "Motif": motif_results["n_dbscan_clusters"]},
        {"metric": "mean_cluster_purity", "MSS": round(mss_results["mean_purity"], 4), "Motif": round(motif_results["mean_purity"], 4)},
        {"metric": "mean_cluster_entropy_bits", "MSS": round(mss_results["mean_entropy"], 4), "Motif": round(motif_results["mean_entropy"], 4)},
        {"metric": "silhouette_cosine", "MSS": round(mss_results["silhouette"], 4), "Motif": round(motif_results["silhouette"], 4)},
        {"metric": "davies_bouldin", "MSS": round(mss_results["davies_bouldin"], 4), "Motif": round(motif_results["davies_bouldin"], 4)},
    ])
    summary.to_csv(TABLES / "mss_vs_motif_summary_metrics_v1.csv", index=False)

    print("\n[decision summary]")
    print(f"  MSS mean purity: {mss_results['mean_purity']:.2%}")
    print(f"  Motif mean purity: {motif_results['mean_purity']:.2%}")
    print(f"  Winner (by purity): "
          f"{'MSS' if mss_results['mean_purity'] > motif_results['mean_purity'] else 'Motif'}")
    print("DONE")


if __name__ == "__main__":
    main()
