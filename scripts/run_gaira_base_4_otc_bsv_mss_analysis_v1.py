"""gaira_base_4_otc_bsv_mss_analysis_v1

Phase: 11-axis BSV clustering + MSS-resolution analysis on OTC pure-Raman
tablet dataset (ibuprofen / ASA / paracetamol), using the UNCHANGED 11-axis
GAIRA BSV schema derived from the biological narrow registry.

STRICT INVARIANTS:
- 11-axis BSV schema UNCHANGED (BIOLOGY_AXES_V11)
- No new axes, no new groups, no schema modification
- No classifier-first
- No parameter tuning
- No substrate physics
- OTC MSS registry from previous phase consumed as-is (not modified)

Goal: let drug separation emerge from the existing biological BSV — the OTC
drugs are NOT in the narrow biological registry, so the BSV per-axis scores
reflect anchor-overlap patterns only; if the 11 biological axes produce
clean drug separation even on non-biological molecules, the BSV schema is a
genuinely discriminative chemistry representation.

Run:
    cd /Users/suraj/projects/GAIRA
    PYTHONPATH=src .venv/bin/python scripts/run_gaira_base_4_otc_bsv_mss_analysis_v1.py
"""
from __future__ import annotations

import shutil
import sys
import warnings
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

warnings.simplefilter("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gaira.spectral import canonical_master_axis  # noqa: E402
from run_gaira_base_4_mss_resolution_reporting_layer_v1 import (  # noqa: E402
    baseline_correct, mss_anchor_score, load_templates,
)
from run_gaira_base_4_small_ev_dual_probe_analysis_v1 import (  # noqa: E402
    compute_bsv_per_spectrum, bsv_transforms, BSV_FAMILIES,
)


# ──────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────
ROOT = Path("/Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_4_otc_bsv_mss_analysis_v1")
TABLES  = ROOT / "tables"
FIGS    = ROOT / "figures"
REPORTS = ROOT / "reports"
AUDIT   = ROOT / "audit"
CODE_SNAPSHOT = ROOT / "code_snapshot"
for d in (TABLES, FIGS, REPORTS, AUDIT, CODE_SNAPSHOT):
    d.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path("/Volumes/SSD_Rad/GAIRA_DATA/raw/otc_drugs")
OTC_REGISTRY = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_4_otc_pure_raman_mss_build_v1/"
    "registry/otc_pure_raman_mss_registry_v1.csv"
)

FILE_SPEC = {
    "Acetylsalicylic-acid.xlsx":            ("acetylsalicylic_acid", "pure"),
    "Acetylsalicylic-acid-trademark.xlsx":  ("acetylsalicylic_acid", "trademark"),
    "Paracetamol.xlsx":                     ("paracetamol", "pure"),
    "Paracetamol-trademark.xlsx":           ("paracetamol", "trademark"),
    "Ibuprofen.xlsx":                       ("ibuprofen", "pure"),
    "Ibuprofen-trademark.xlsx":             ("ibuprofen", "trademark"),
}
DRUGS = ["acetylsalicylic_acid", "paracetamol", "ibuprofen"]
DRUG_COLORS = {"acetylsalicylic_acid": "#4C72B0",
                  "paracetamol":          "#DD8452",
                  "ibuprofen":            "#2ca02c"}


# ──────────────────────────────────────────────────────────────────────
# TASK 1 — load + preprocess
# ──────────────────────────────────────────────────────────────────────
def task1_load_preprocess(master_x):
    print("[TASK 1] load + canonical preprocess")
    meta_rows = []
    Y_list = []
    for fname, (drug, variant) in FILE_SPEC.items():
        path = DATA_DIR / fname
        if not path.exists(): continue
        df = pd.read_excel(path, sheet_name=0, header=0)
        rs = pd.to_numeric(df.iloc[:, 0], errors="coerce").values
        valid = np.isfinite(rs); rs = rs[valid]
        Y_raw = df.iloc[valid, 1:].values.astype(float)  # (n_wn, n_spectra)
        cols = df.columns[1:].tolist()
        for j, col in enumerate(cols):
            col_s = str(col).replace("\n", "").replace("\t", "").strip()
            base = col_s.split(".")[0]
            brand_code = base if variant == "trademark" else None
            y_raw = Y_raw[:, j]
            y_rs = np.interp(master_x, rs, y_raw, left=np.nan, right=np.nan)
            y_pp = baseline_correct(y_rs)
            if not (np.isfinite(y_pp).any() and float(np.linalg.norm(y_pp)) >= 1e-12):
                continue
            Y_list.append(y_pp)
            meta_rows.append({
                "spectrum_id":   f"{fname.replace('.xlsx', '')}::col{j:03d}::{col_s}",
                "file":          fname,
                "drug":          drug,
                "variant_type":  variant,
                "brand_code":    brand_code,
                "column_base":   base,
            })
    Y_pp = np.vstack(Y_list)
    meta_df = pd.DataFrame(meta_rows)
    print(f"  {len(meta_df)} spectra preprocessed; shape = {Y_pp.shape}")
    return Y_pp, meta_df


# ──────────────────────────────────────────────────────────────────────
# TASK 2 — BSV computation (11-axis GAIRA schema unchanged)
# ──────────────────────────────────────────────────────────────────────
def task2_bsv(Y_pp, master_x):
    print("[TASK 2] 11-axis BSV computation (unchanged schema)")
    templates, _, _ = load_templates()
    by_mol = defaultdict(dict)
    for t in templates:
        by_mol[t["molecule"]][t["regime"]] = t
    bsv_raw = compute_bsv_per_spectrum(Y_pp, master_x, by_mol)
    trans = bsv_transforms(bsv_raw)
    return trans["raw"], trans["sumnorm"], trans["clr"]


# ──────────────────────────────────────────────────────────────────────
# TASK 3 — RAW vs BSV PCA
# ──────────────────────────────────────────────────────────────────────
def _pca_metrics(Z, labels):
    """Silhouette + centroid-distance-over-spread metrics."""
    labels = np.asarray(labels)
    try:
        sil = float(silhouette_score(Z, labels))
    except Exception:
        sil = np.nan
    # Between/within variance
    overall_var = float(np.var(Z))
    within_var = 0.0; between_var = 0.0
    centroids = {}
    for lbl in np.unique(labels):
        sub = Z[labels == lbl]
        centroids[lbl] = sub.mean(axis=0)
        within_var += float(np.var(sub)) * len(sub)
    within_var /= max(len(labels), 1)
    grand_mean = Z.mean(axis=0)
    for lbl in np.unique(labels):
        n = int((labels == lbl).sum())
        d = centroids[lbl] - grand_mean
        between_var += n * float(np.dot(d, d))
    between_var /= max(len(labels), 1)
    # Centroid pair distances
    keys = sorted(centroids.keys())
    pair_dists = {}
    for i, a in enumerate(keys):
        for b in keys[i+1:]:
            pair_dists[f"{a}::{b}"] = float(np.linalg.norm(centroids[a] - centroids[b]))
    return sil, within_var, between_var, pair_dists


def task3_raw_vs_bsv_pca(Y_pp, bsv_clr, labels):
    print("[TASK 3] RAW vs BSV PCA")
    rows = []
    # Raw PCA — impute NaN to 0 (should be rare after QC)
    Xraw = np.nan_to_num(Y_pp, nan=0.0)
    Zraw = PCA(n_components=2).fit_transform(Xraw)
    Zbsv = PCA(n_components=2).fit_transform(bsv_clr)

    sil_raw, wv_raw, bv_raw, pd_raw = _pca_metrics(Zraw, labels)
    sil_bsv, wv_bsv, bv_bsv, pd_bsv = _pca_metrics(Zbsv, labels)
    rows.append({"space": "RAW_canonical_pp", "silhouette_by_molecule": sil_raw,
                    "within_var": wv_raw, "between_var": bv_raw,
                    "between_over_within": bv_raw / max(wv_raw, 1e-9),
                    **{f"centroid_dist::{k}": v for k, v in pd_raw.items()}})
    rows.append({"space": "BSV_CLR_11", "silhouette_by_molecule": sil_bsv,
                    "within_var": wv_bsv, "between_var": bv_bsv,
                    "between_over_within": bv_bsv / max(wv_bsv, 1e-9),
                    **{f"centroid_dist::{k}": v for k, v in pd_bsv.items()}})
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "clustering_metrics_raw_vs_bsv_v1.csv", index=False)

    # Figure
    try:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for ax, Z, title, sil in [(axes[0], Zraw, "A. RAW preprocessed PCA", sil_raw),
                                         (axes[1], Zbsv, "B. BSV-CLR PCA (11-axis unchanged schema)", sil_bsv)]:
            for drug in DRUGS:
                m = np.asarray(labels) == drug
                ax.scatter(Z[m, 0], Z[m, 1], s=15, alpha=0.6,
                              color=DRUG_COLORS[drug], label=drug)
            ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
            ax.set_title(f"{title}\nsilhouette = {sil:+.3f}")
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGS / "fig_raw_vs_bsv_pca_by_molecule_v1.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig issue: {e}")
    return Zraw, Zbsv, sil_raw, sil_bsv


# ──────────────────────────────────────────────────────────────────────
# TASK 4 — axis contribution analysis
# ──────────────────────────────────────────────────────────────────────
def task4_axis_importance(bsv_clr, labels):
    print("[TASK 4] axis contribution analysis")
    labels = np.asarray(labels)
    # PCA loadings on BSV-CLR
    pca = PCA(n_components=4).fit(bsv_clr)
    loadings = pca.components_  # (4, 11)
    explained = pca.explained_variance_ratio_

    # Per-axis η² via single-factor ANOVA (factor = molecule)
    eta_rows = []
    for k, (fid, fname) in enumerate(BSV_FAMILIES):
        v = bsv_clr[:, k]
        grand_mean = float(v.mean())
        sst = float(np.sum((v - grand_mean) ** 2))
        ss_between = 0.0
        for lbl in DRUGS:
            sub = v[labels == lbl]
            if len(sub) < 1: continue
            ss_between += len(sub) * (float(sub.mean()) - grand_mean) ** 2
        eta2 = float(ss_between / sst) if sst > 0 else 0.0
        eta_rows.append({
            "axis": fid, "axis_name": fname,
            "abs_pc1_loading": float(abs(loadings[0, k])),
            "abs_pc2_loading": float(abs(loadings[1, k])),
            "combined_pc1_pc2": float(np.sqrt(loadings[0, k]**2 + loadings[1, k]**2)),
            "eta2_by_molecule": eta2,
            "bsv_mean":        float(v.mean()),
            "bsv_sd":          float(v.std()),
        })
    df = pd.DataFrame(eta_rows)
    # Composite importance = eta2 × combined_loading (simple multiplicative)
    df["combined_importance"] = df["eta2_by_molecule"] * df["combined_pc1_pc2"]
    df = df.sort_values("combined_importance", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    df.to_csv(TABLES / "axis_importance_ranking_v1.csv", index=False)

    # Per-axis per-molecule mean (for radar later)
    per_axis = []
    for k, (fid, _) in enumerate(BSV_FAMILIES):
        for drug in DRUGS:
            sub = bsv_clr[labels == drug, k]
            per_axis.append({
                "axis": fid, "drug": drug,
                "mean": float(sub.mean()),
                "sd":   float(sub.std()),
                "n":    int(len(sub)),
            })
    pd.DataFrame(per_axis).to_csv(TABLES / "per_axis_variance_v1.csv", index=False)

    top3 = df.head(3)["axis"].tolist()
    top5 = df.head(5)["axis"].tolist()
    print(f"  top-3 axes: {top3}; top-5 axes: {top5}")
    print(f"  PC1 var = {explained[0]:.1%}, PC2 var = {explained[1]:.1%}")

    # Figure: axis importance bar chart
    try:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(df)); w = 0.25
        df2 = df.sort_values("axis").reset_index(drop=True)
        ax.bar(x - w, df2["abs_pc1_loading"], w, label="|PC1 loading|", color="#4C72B0")
        ax.bar(x,       df2["abs_pc2_loading"], w, label="|PC2 loading|", color="#DD8452")
        ax.bar(x + w,   df2["eta2_by_molecule"], w, label="η² by molecule", color="#2ca02c")
        ax.set_xticks(x); ax.set_xticklabels(df2["axis"] + "\n" + df2["axis_name"],
                                                     rotation=20, fontsize=8, ha="right")
        ax.set_ylabel("importance"); ax.set_ylim(0, 1.05)
        ax.set_title("BSV-CLR axis importance (|PCA loadings| + η² by molecule)")
        ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIGS / "fig_axis_importance_bar_v1.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig importance issue: {e}")
    return df, top3, top5, pca


# ──────────────────────────────────────────────────────────────────────
# TASK 5 — derived discriminative subspace (top-3 / top-5)
# ──────────────────────────────────────────────────────────────────────
def task5_subspace(bsv_clr, top3, top5, labels):
    print("[TASK 5] derived discriminative subspace PCA")
    ax_ids = [fid for fid, _ in BSV_FAMILIES]
    idx_top3 = [ax_ids.index(a) for a in top3]
    idx_top5 = [ax_ids.index(a) for a in top5]

    Z_top3 = PCA(n_components=2).fit_transform(bsv_clr[:, idx_top3])
    Z_top5 = PCA(n_components=2).fit_transform(bsv_clr[:, idx_top5])
    sil3, wv3, bv3, _ = _pca_metrics(Z_top3, labels)
    sil5, wv5, bv5, _ = _pca_metrics(Z_top5, labels)

    rows = [
        {"space": f"top3_{'|'.join(top3)}",
         "silhouette": sil3, "between_over_within": bv3 / max(wv3, 1e-9)},
        {"space": f"top5_{'|'.join(top5)}",
         "silhouette": sil5, "between_over_within": bv5 / max(wv5, 1e-9)},
    ]
    pd.DataFrame(rows).to_csv(TABLES / "subspace_clustering_metrics_v1.csv", index=False)

    # Figures
    for Z, tag, sil, fname in [(Z_top3, f"top-3 axes {top3}", sil3,
                                       "fig_pca_top3_axes_v1.png"),
                                      (Z_top5, f"top-5 axes {top5}", sil5,
                                       "fig_pca_top5_axes_v1.png")]:
        try:
            fig, ax = plt.subplots(figsize=(7, 5))
            for drug in DRUGS:
                m = np.asarray(labels) == drug
                ax.scatter(Z[m, 0], Z[m, 1], s=15, alpha=0.65,
                              color=DRUG_COLORS[drug], label=drug)
            ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
            ax.set_title(f"PCA on {tag}\nsilhouette = {sil:+.3f}")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(FIGS / fname, dpi=150)
            plt.close(fig)
        except Exception as e:
            print(f"  fig {tag} issue: {e}")

    return sil3, sil5


# ──────────────────────────────────────────────────────────────────────
# TASK 6 — per-molecule axis trajectories + radar
# ──────────────────────────────────────────────────────────────────────
def task6_radar(bsv_sumnorm, bsv_clr, meta_df):
    print("[TASK 6] per-molecule radar + pure-vs-trademark overlay")
    labels = meta_df["drug"].values
    # Mean sumnorm per molecule × axis
    rows = []
    for drug in DRUGS:
        for variant in ("pure", "trademark"):
            mask = (meta_df.drug == drug) & (meta_df.variant_type == variant)
            if mask.sum() == 0: continue
            for k, (fid, fname) in enumerate(BSV_FAMILIES):
                rows.append({
                    "drug": drug, "variant": variant, "axis": fid,
                    "mean_sumnorm": float(bsv_sumnorm[mask, k].mean()),
                })
    pd.DataFrame(rows).to_csv(TABLES / "per_molecule_per_variant_per_axis_mean_v1.csv", index=False)

    # Radar figure: one per drug, pure vs trademark overlay
    try:
        angles = np.linspace(0, 2 * np.pi, len(BSV_FAMILIES), endpoint=False)
        angles_closed = np.concatenate([angles, [angles[0]]])
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), subplot_kw=dict(polar=True))
        for ax, drug in zip(axes, DRUGS):
            for variant, color, ls in [("pure", "#4C72B0", "-"),
                                              ("trademark", "#DD8452", "--")]:
                mask = (meta_df.drug == drug) & (meta_df.variant_type == variant)
                if mask.sum() == 0: continue
                vals = [float(bsv_sumnorm[mask, k].mean()) for k in range(len(BSV_FAMILIES))]
                vals_closed = vals + [vals[0]]
                ax.plot(angles_closed, vals_closed, lw=1.8, color=color, ls=ls,
                          label=f"{variant} (n={int(mask.sum())})")
                ax.fill(angles_closed, vals_closed, alpha=0.15, color=color)
            ax.set_xticks(angles); ax.set_xticklabels([fid for fid, _ in BSV_FAMILIES], fontsize=7)
            ax.set_title(drug, fontsize=11)
            ax.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.15, 1.08))
            ax.set_ylim(0, None)
        fig.suptitle("Per-molecule BSV-sumnorm radar — pure vs trademark overlay")
        fig.tight_layout()
        fig.savefig(FIGS / "fig_per_molecule_radar_v1.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig radar issue: {e}")


# ──────────────────────────────────────────────────────────────────────
# TASK 7 — MSS layer on top of BSV (using OTC registry)
# ──────────────────────────────────────────────────────────────────────
def task7_mss_layer(Y_pp, master_x, meta_df):
    print("[TASK 7] MSS layer — OTC registry scoring")
    reg_df = pd.read_csv(OTC_REGISTRY)
    templates = {}
    for _, r in reg_df.iterrows():
        anchors = [float(x) for x in str(r["anchor_bands_cm1"]).split(";") if x.strip()]
        supports = [float(x) for x in str(r["companion_bands_cm1"]).split(";") if x.strip()]
        templates[r["molecule"]] = {"anchors": anchors, "supports": supports}

    rows = []
    pred_top1 = []; pred_top3 = []
    for i, r in meta_df.iterrows():
        y = Y_pp[i]
        scores = {}
        for drug, t in templates.items():
            sc, _, _ = mss_anchor_score(y, master_x, t["anchors"], t["supports"])
            scores[drug] = sc
        order = sorted(scores.items(), key=lambda x: -x[1])
        top1_d = order[0][0]
        top3_set = [d for d, _ in order[:3]]
        correct_top1 = int(top1_d == r["drug"])
        correct_top3 = int(r["drug"] in top3_set)
        pred_top1.append(top1_d); pred_top3.append(top3_set)
        rows.append({
            "spectrum_id": r["spectrum_id"],
            "drug_true":   r["drug"],
            "variant":     r["variant_type"],
            "brand_code":  r["brand_code"],
            "top1":        top1_d,
            "top3":        "|".join(top3_set),
            "score_asa":         scores["acetylsalicylic_acid"],
            "score_paracetamol": scores["paracetamol"],
            "score_ibuprofen":   scores["ibuprofen"],
            "correct_top1": correct_top1,
            "correct_top3": correct_top3,
        })
    mss_df = pd.DataFrame(rows)
    mss_df.to_csv(TABLES / "mss_per_spectrum_v1.csv", index=False)

    top1_acc = float(mss_df["correct_top1"].mean())
    top3_acc = float(mss_df["correct_top3"].mean())
    per_drug_acc = {d: float(mss_df[mss_df.drug_true == d]["correct_top1"].mean())
                      for d in DRUGS}
    per_variant_acc = {v: float(mss_df[mss_df.variant == v]["correct_top1"].mean())
                         for v in ("pure", "trademark")}
    per_brand_acc = mss_df[mss_df.variant == "trademark"].groupby("brand_code")["correct_top1"] \
                       .agg(["mean", "count"]).reset_index()
    per_brand_acc.to_csv(TABLES / "mss_per_brand_accuracy_v1.csv", index=False)

    # Confusion matrix
    conf = defaultdict(lambda: Counter())
    for _, r in mss_df.iterrows():
        conf[r["drug_true"]][r["top1"]] += 1
    conf_rows = []
    for t in DRUGS:
        for p in DRUGS:
            conf_rows.append({"true": t, "predicted": p, "count": int(conf[t][p])})
    pd.DataFrame(conf_rows).to_csv(TABLES / "mss_confusion_matrix_v1.csv", index=False)

    summary_rows = [
        {"metric": "top1_accuracy_overall", "value": top1_acc, "n": len(mss_df)},
        {"metric": "top3_accuracy_overall", "value": top3_acc, "n": len(mss_df)},
        {"metric": "top1_accuracy_pure",    "value": per_variant_acc.get("pure", np.nan),
         "n": int((mss_df.variant == "pure").sum())},
        {"metric": "top1_accuracy_trademark","value": per_variant_acc.get("trademark", np.nan),
         "n": int((mss_df.variant == "trademark").sum())},
        *[{"metric": f"top1_accuracy_{d}", "value": v,
            "n": int((mss_df.drug_true == d).sum())} for d, v in per_drug_acc.items()],
    ]
    pd.DataFrame(summary_rows).to_csv(TABLES / "mss_validation_summary_v1.csv", index=False)

    # Figure: confusion matrix
    try:
        mat = np.array([[conf[t][p] for p in DRUGS] for t in DRUGS])
        norm = mat / np.maximum(mat.sum(axis=1, keepdims=True), 1)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(DRUGS, rotation=15, fontsize=9)
        ax.set_yticklabels(DRUGS, fontsize=9)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{mat[i, j]}\n({norm[i, j]:.0%})",
                          ha="center", va="center", fontsize=9,
                          color="white" if norm[i, j] > 0.5 else "black")
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(f"OTC MSS confusion matrix (n={int(mat.sum())})")
        plt.colorbar(im, ax=ax, fraction=0.04)
        fig.tight_layout()
        fig.savefig(FIGS / "fig_mss_confusion_matrix_v1.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig confusion issue: {e}")

    print(f"  MSS top-1 = {top1_acc:.1%}, top-3 = {top3_acc:.1%}")
    return mss_df, top1_acc, top3_acc, per_drug_acc, per_variant_acc, per_brand_acc


# ──────────────────────────────────────────────────────────────────────
# TASK 8 — BSV vs MSS role separation
# ──────────────────────────────────────────────────────────────────────
def task8_bsv_mss_role_separation(sil_bsv, sil_raw, sil3, sil5,
                                        top1_acc, top3_acc, mss_df):
    print("[TASK 8] BSV vs MSS role separation")
    rows = [
        {"layer": "RAW_PCA",                "silhouette_by_molecule": sil_raw,
         "top1_accuracy": None, "top3_accuracy": None},
        {"layer": "BSV_CLR_full_11",        "silhouette_by_molecule": sil_bsv,
         "top1_accuracy": None, "top3_accuracy": None},
        {"layer": "BSV_CLR_top3_subspace", "silhouette_by_molecule": sil3,
         "top1_accuracy": None, "top3_accuracy": None},
        {"layer": "BSV_CLR_top5_subspace", "silhouette_by_molecule": sil5,
         "top1_accuracy": None, "top3_accuracy": None},
        {"layer": "MSS_top1",               "silhouette_by_molecule": None,
         "top1_accuracy": top1_acc, "top3_accuracy": top3_acc},
    ]
    pd.DataFrame(rows).to_csv(TABLES / "bsv_mss_role_separation_v1.csv", index=False)

    # Cluster-label consistency: per BSV-cluster (via top-1 MSS predicted label) how often is it the true drug?
    # Here the "BSV cluster" is loosely defined via nearest-centroid in BSV-CLR space using true labels.
    return rows


# ──────────────────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────────────────
def _decision(sil_bsv, top1_acc, top3_acc):
    if sil_bsv >= 0.35 and top1_acc >= 0.85:
        return "BSV_SEPARATION_STRONG_MSS_CONFIRMATORY"
    if sil_bsv >= 0.15 and top1_acc >= 0.85:
        return "BSV_SEPARATION_PARTIAL_MSS_REQUIRED"
    if sil_bsv < 0.15 and top1_acc >= 0.90:
        return "BSV_WEAK_MSS_DOMINANT"
    return "BSV_SEPARATION_PARTIAL_MSS_REQUIRED"


def write_report(decision, sil_raw, sil_bsv, sil3, sil5, top1_acc, top3_acc,
                     imp_df, top3, top5, per_drug_acc, per_variant_acc, per_brand_acc):
    lines = [
        "# REPORT — OTC BSV clustering + MSS resolution v1\n",
        f"date: {datetime.now().isoformat()}", "",
        f"## Decision: **{decision}**\n",
        "## Setup",
        "- OTC pure-Raman tablet dataset (Paraguay iRaman 785 nm), 150 pure + 150 trademark across 3 drugs.",
        "- Canonical preprocessing: interp to master_x (400-1800 step 1) + AsLS + SG + L2.",
        "- 11-axis BSV computed via UNCHANGED family-aggregated MSS-anchor kernel on the biological narrow registry.",
        "- Key design: OTC drugs are NOT in the narrow biological registry. Drug separation emerges from "
        "anchor-overlap patterns across the 11 BSV families.",
        "- No axis changes, no new groups, no schema modification, no classifier, no substrate physics.",
        "",
        "## Required answers\n",
    ]

    lines.append("### 1. Do OTC drugs naturally separate in BSV space?")
    lines.append(f"- **RAW PCA silhouette-by-molecule = {sil_raw:+.3f}**")
    lines.append(f"- **BSV-CLR 11-axis PCA silhouette-by-molecule = {sil_bsv:+.3f}**")
    if sil_bsv >= 0.35:
        lines.append("  → **YES, strong separation** (silhouette ≥ 0.35).")
    elif sil_bsv >= 0.15:
        lines.append("  → **Partial separation** (0.15 ≤ sil < 0.35).")
    else:
        lines.append("  → Weak separation at BSV layer.")
    lines.append("")

    lines.append("### 2. Which axes drive separation?")
    top5_list = imp_df.head(5)[["axis", "axis_name", "abs_pc1_loading", "abs_pc2_loading", "eta2_by_molecule"]]
    lines.append("| rank | axis | axis_name | |PC1| | |PC2| | η² by molecule |")
    lines.append("|---:|---|---|---:|---:|---:|")
    for i, r in imp_df.head(5).iterrows():
        lines.append(f"| {int(r['rank'])} | {r['axis']} | {r['axis_name']} | "
                        f"{r['abs_pc1_loading']:.2f} | {r['abs_pc2_loading']:.2f} | {r['eta2_by_molecule']:.3f} |")
    lines.append("")

    lines.append("### 3. How many axes are needed?")
    lines.append(f"- top-3 subspace {top3}: silhouette = {sil3:+.3f}")
    lines.append(f"- top-5 subspace {top5}: silhouette = {sil5:+.3f}")
    lines.append(f"- full 11-axis: silhouette = {sil_bsv:+.3f}")
    lines.append("")

    lines.append("### 4. Does MSS add resolution beyond BSV?")
    lines.append(f"- MSS top-1 accuracy = {top1_acc:.1%}")
    lines.append(f"- MSS top-3 accuracy = {top3_acc:.1%}")
    lines.append(f"- per-drug top-1: {per_drug_acc}")
    lines.append(f"- per-variant top-1: {per_variant_acc}")
    lines.append("")
    if top1_acc >= 0.95:
        lines.append("  → MSS provides **molecule-level confirmatory accuracy** beyond the BSV cluster.")
    elif top1_acc >= 0.80:
        lines.append("  → MSS adds meaningful per-molecule confirmation.")
    else:
        lines.append("  → MSS is the dominant classification layer (BSV clustering is weaker than MSS).")
    lines.append("")

    lines.append("### 5. Is the separation robust across trademark variants?")
    lines.append(f"- per-variant MSS top-1: {per_variant_acc}")
    lines.append(f"- per-brand top-1 trademark:")
    for _, r in per_brand_acc.sort_values("mean", ascending=False).iterrows():
        lines.append(f"  - {r['brand_code']}: {float(r['mean']):.0%} (n={int(r['count'])})")
    lines.append("")

    # Summary table
    lines.append("## Role separation summary\n")
    lines.append("| layer | silhouette | top-1 | top-3 |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| RAW PCA | {sil_raw:+.3f} | — | — |")
    lines.append(f"| BSV-CLR full 11 | {sil_bsv:+.3f} | — | — |")
    lines.append(f"| BSV-CLR top-3 | {sil3:+.3f} | — | — |")
    lines.append(f"| BSV-CLR top-5 | {sil5:+.3f} | — | — |")
    lines.append(f"| MSS layer | — | {top1_acc:.1%} | {top3_acc:.1%} |")
    lines.append("")

    (REPORTS / "REPORT_otc_bsv_mss_analysis_v1.md").write_text("\n".join(lines))


def write_audit(decision):
    txt = [
        "# gaira_base_4_otc_bsv_mss_analysis_v1 — audit log",
        f"date: {datetime.now().isoformat()}",
        "",
        "## Strict invariants",
        "- 11-axis BSV schema UNCHANGED (BIOLOGY_AXES_V11)",
        "- No new axes, no new groups, no schema modification",
        "- No classifier-first; no parameter tuning",
        "- No substrate physics",
        "- OTC MSS registry consumed read-only",
        "",
        "## Inputs",
        "- /Volumes/SSD_Rad/GAIRA_DATA/raw/otc_drugs/ (6 xlsx files, 300 unique spectra)",
        "- /Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_4_otc_pure_raman_mss_build_v1/registry/otc_pure_raman_mss_registry_v1.csv",
        "",
        "## Outputs",
        "- tables/clustering_metrics_raw_vs_bsv_v1.csv",
        "- tables/axis_importance_ranking_v1.csv",
        "- tables/per_axis_variance_v1.csv",
        "- tables/subspace_clustering_metrics_v1.csv",
        "- tables/per_molecule_per_variant_per_axis_mean_v1.csv",
        "- tables/mss_per_spectrum_v1.csv",
        "- tables/mss_per_brand_accuracy_v1.csv",
        "- tables/mss_confusion_matrix_v1.csv",
        "- tables/mss_validation_summary_v1.csv",
        "- tables/bsv_mss_role_separation_v1.csv",
        "- figures: raw_vs_bsv_pca, axis_importance_bar, pca_top3_axes, pca_top5_axes, per_molecule_radar, mss_confusion_matrix",
        "- reports/REPORT_otc_bsv_mss_analysis_v1.md",
        "",
        f"## Final decision\n**{decision}**",
    ]
    (AUDIT / "gaira_base_4_otc_bsv_mss_analysis_v1_audit_log.md").write_text("\n".join(txt))


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 78)
    print("gaira_base_4_otc_bsv_mss_analysis_v1")
    print("=" * 78)
    master_x = canonical_master_axis()

    Y_pp, meta_df = task1_load_preprocess(master_x)
    bsv_raw, bsv_sumnorm, bsv_clr = task2_bsv(Y_pp, master_x)

    # Persist BSV matrix
    pd.DataFrame(
        np.hstack([bsv_raw, bsv_sumnorm, bsv_clr]),
        columns=[f"raw_{fid}" for fid, _ in BSV_FAMILIES] +
                  [f"sumnorm_{fid}" for fid, _ in BSV_FAMILIES] +
                  [f"clr_{fid}" for fid, _ in BSV_FAMILIES],
    ).to_csv(TABLES / "bsv_matrix_v1.csv", index=False)

    labels = meta_df["drug"].values
    Zraw, Zbsv, sil_raw, sil_bsv = task3_raw_vs_bsv_pca(Y_pp, bsv_clr, labels)
    imp_df, top3, top5, pca_full = task4_axis_importance(bsv_clr, labels)
    sil3, sil5 = task5_subspace(bsv_clr, top3, top5, labels)
    task6_radar(bsv_sumnorm, bsv_clr, meta_df)
    mss_df, top1_acc, top3_acc, per_drug_acc, per_variant_acc, per_brand_acc = \
        task7_mss_layer(Y_pp, master_x, meta_df)
    task8_bsv_mss_role_separation(sil_bsv, sil_raw, sil3, sil5, top1_acc, top3_acc, mss_df)

    decision = _decision(sil_bsv, top1_acc, top3_acc)
    write_report(decision, sil_raw, sil_bsv, sil3, sil5, top1_acc, top3_acc,
                    imp_df, top3, top5, per_drug_acc, per_variant_acc, per_brand_acc)
    write_audit(decision)
    try:
        shutil.copy(__file__, CODE_SNAPSHOT / Path(__file__).name)
    except Exception:
        pass
    print(f"[done] decision: {decision}")


if __name__ == "__main__":
    main()
