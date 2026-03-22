import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold


SEED = 42
DATASET_ID = "hcc_serum"
POS_LABEL = "H0T"
NEG_LABEL = "CTR"


def ensure_output_dir() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))
    from gaira.config import ensure_storage_dirs, resolve_storage_path

    storage_config = ensure_storage_dirs()
    processed_root = resolve_storage_path(storage_config.get("processed_data"))
    if processed_root is None:
        raise ValueError("The storage config is missing processed_data.")
    output_dir = processed_root / "hcc_serum_paper_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_raw_dataset(csv_path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    metadata_cols = ["acquisition_date", "substrate_batch", "class", "sample_code"]
    x = np.asarray([float(col) for col in df.columns if col not in metadata_cols], dtype=float)
    y = df[[col for col in df.columns if col not in metadata_cols]].to_numpy(dtype=float)
    return df, x, y


def interp_rows(x_src: np.ndarray, y_rows: np.ndarray, x_dst: np.ndarray) -> np.ndarray:
    return np.asarray([np.interp(x_dst, x_src, row) for row in y_rows], dtype=float)


def approximate_modpoly_baseline(x: np.ndarray, y: np.ndarray, degree: int = 4, max_iter: int = 50) -> np.ndarray:
    working = y.copy()
    baseline = np.polyval(np.polyfit(x, working, degree), x)
    for _ in range(max_iter):
        clipped = np.minimum(working, baseline)
        updated = np.polyval(np.polyfit(x, clipped, degree), x)
        if np.max(np.abs(updated - baseline)) < 1e-8:
            baseline = updated
            break
        baseline = updated
        working = clipped
    return baseline


def vector_normalize(y_rows: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(y_rows, axis=1, keepdims=True)
    norms = np.where(norms <= 0, 1.0, norms)
    return y_rows / norms


def preprocess_like_r_code(x_raw: np.ndarray, y_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    interp_grid = np.arange(400.0, 1800.0 + 2.0, 2.0)
    y_interp = interp_rows(x_raw, y_raw, interp_grid)
    baselines = np.asarray(
        [approximate_modpoly_baseline(interp_grid, row, degree=4) for row in y_interp],
        dtype=float,
    )
    y_corrected = y_interp - baselines
    crop_mask = (interp_grid >= 430.0) & (interp_grid <= 1730.0)
    x_crop = interp_grid[crop_mask]
    y_crop = y_corrected[:, crop_mask]
    y_norm = vector_normalize(y_crop)
    return x_crop, y_norm


def one_standard_error_choice(mean_err: np.ndarray, std_err: np.ndarray, params: np.ndarray) -> int:
    best_idx = int(np.argmin(mean_err))
    threshold = mean_err[best_idx] + std_err[best_idx]
    allowed = params[mean_err <= threshold]
    return int(allowed[0])


def run_repeated_double_cv(X: np.ndarray, y: np.ndarray, params: np.ndarray, repetitions: int, out_segments: int, inn_segments: int) -> tuple[pd.DataFrame, list[np.ndarray], list[int]]:
    repetition_rows = []
    confusion_mats = []
    opt_params = []
    y_binary = (y == POS_LABEL).astype(int)

    for rep in range(repetitions):
        outer = StratifiedKFold(n_splits=out_segments, shuffle=True, random_state=SEED + rep)
        rep_pred = np.empty_like(y, dtype=object)
        rep_prob = np.empty(len(y), dtype=float)
        rep_opt = []

        for fold_idx, (opt_idx, test_idx) in enumerate(outer.split(X, y), start=1):
            X_opt, y_opt = X[opt_idx], y[opt_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            inner = StratifiedKFold(
                n_splits=inn_segments,
                shuffle=True,
                random_state=SEED * 100 + rep * 10 + fold_idx,
            )
            cv_errors = np.zeros((inn_segments, len(params)), dtype=float)

            for p_idx, n_pc in enumerate(params):
                for inner_fold, (train_idx, val_idx) in enumerate(inner.split(X_opt, y_opt)):
                    X_train, y_train = X_opt[train_idx], y_opt[train_idx]
                    X_val, y_val = X_opt[val_idx], y_opt[val_idx]

                    pca = PCA(n_components=max(params), svd_solver="full")
                    X_train_scores = pca.fit_transform(X_train)
                    X_val_scores = pca.transform(X_val)

                    lda = LinearDiscriminantAnalysis()
                    lda.fit(X_train_scores[:, :n_pc], y_train)
                    pred = lda.predict(X_val_scores[:, :n_pc])
                    cv_errors[inner_fold, p_idx] = 1.0 - accuracy_score(y_val, pred)

            mean_err = cv_errors.mean(axis=0)
            std_err = cv_errors.std(axis=0, ddof=0)
            chosen_pc = one_standard_error_choice(mean_err, std_err, params)
            rep_opt.append(chosen_pc)
            opt_params.append(chosen_pc)

            pca = PCA(n_components=max(20, max(params)), svd_solver="full")
            X_opt_scores = pca.fit_transform(X_opt)
            X_test_scores = pca.transform(X_test)
            lda = LinearDiscriminantAnalysis()
            lda.fit(X_opt_scores[:, :chosen_pc], y_opt)
            pred = lda.predict(X_test_scores[:, :chosen_pc])
            prob = lda.predict_proba(X_test_scores[:, :chosen_pc])[:, list(lda.classes_).index(POS_LABEL)]

            rep_pred[test_idx] = pred
            rep_prob[test_idx] = prob

        cm = confusion_matrix(y, rep_pred, labels=[NEG_LABEL, POS_LABEL])
        confusion_mats.append(cm)
        repetition_rows.append(
            {
                "repetition": rep + 1,
                "accuracy": float(accuracy_score(y, rep_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y, rep_pred)),
                "roc_auc": float(roc_auc_score(y_binary, rep_prob)),
                "mean_opt_pc": float(np.mean(rep_opt)),
                "median_opt_pc": float(np.median(rep_opt)),
            }
        )

    return pd.DataFrame(repetition_rows), confusion_mats, opt_params


def build_pca_variance_df(X: np.ndarray) -> pd.DataFrame:
    pca = PCA(n_components=min(10, X.shape[1]), svd_solver="full")
    pca.fit(X)
    return pd.DataFrame(
        {
            "pc_index": np.arange(1, len(pca.explained_variance_ratio_) + 1),
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
        }
    )


def save_confusion_matrix_plot(cm: np.ndarray, output_path: Path) -> None:
    plt.figure(figsize=(4.5, 4), dpi=300)
    plt.imshow(cm, cmap="Blues")
    plt.xticks([0, 1], [NEG_LABEL, POS_LABEL])
    plt.yticks([0, 1], [NEG_LABEL, POS_LABEL])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Mean confusion matrix across RDCV repetitions")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f"{cm[i, j]:.1f}", ha="center", va="center", color="black")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_projection_plot(X: np.ndarray, y: np.ndarray, output_path: Path) -> None:
    pca = PCA(n_components=2, svd_solver="full")
    coords = pca.fit_transform(X)
    plt.figure(figsize=(5.5, 4.5), dpi=300)
    for label, color in [(NEG_LABEL, "#3366aa"), (POS_LABEL, "#cc5533")]:
        mask = y == label
        plt.scatter(coords[mask, 0], coords[mask, 1], s=20, alpha=0.8, label=label, color=color, edgecolors="none")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Paper-like PCA projection after preprocessing")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def load_gaira_v1_summary(metrics_path: Path) -> pd.DataFrame:
    df = pd.read_csv(metrics_path)
    return (
        df.groupby(["split_name", "model_name"])[["accuracy", "balanced_accuracy", "roc_auc"]]
        .mean()
        .reset_index()
    )


def write_summary(
    output_path: Path,
    released_metrics: pd.DataFrame,
    pc14_metrics: pd.DataFrame,
    comparison_df: pd.DataFrame,
    preprocessing_note: str,
    cv_note: str,
) -> None:
    lines = [
        "hcc_serum paper comparison summary",
        "",
        "Released-code-like PCA-LDA reproduction:",
        released_metrics.to_string(index=False),
        "",
        "PC1-4 constrained PCA-LDA variant:",
        pc14_metrics.to_string(index=False),
        "",
        "Comparison versus GAIRA serum benchmark v1:",
        comparison_df.to_string(index=False),
        "",
        "Preprocessing used:",
        preprocessing_note,
        "",
        "Validation used:",
        cv_note,
        "",
        "Interpretation:",
        "- The released R code uses repeated double cross-validation with PCA model selection inside each outer split.",
        "- GAIRA v1 used simpler processed spectra and direct classical CV models, not nested RDCV.",
        "- Any gap between the paper-like reproduction and GAIRA v1 is most likely driven by preprocessing and validation protocol differences rather than search or embedding logic.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_variant(name: str, repetition_df: pd.DataFrame, opt_params: list[int]) -> dict:
    return {
        "variant": name,
        "mean_accuracy": float(repetition_df["accuracy"].mean()),
        "std_accuracy": float(repetition_df["accuracy"].std(ddof=0)),
        "mean_balanced_accuracy": float(repetition_df["balanced_accuracy"].mean()),
        "mean_roc_auc": float(repetition_df["roc_auc"].mean()),
        "mean_opt_pc": float(np.mean(opt_params)),
        "median_opt_pc": float(np.median(opt_params)),
        "min_opt_pc": int(np.min(opt_params)),
        "max_opt_pc": int(np.max(opt_params)),
        "repetitions": int(len(repetition_df)),
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_dir = ensure_output_dir()
    raw_csv_path = Path("/Volumes/SSD_Rad/GAIRA_DATA/raw/hcc_serum/data.csv")
    gaira_v1_metrics_path = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/hcc_serum_benchmark_v1/hcc_serum_benchmark_metrics.csv")

    metadata_df, x_raw, y_raw = load_raw_dataset(raw_csv_path)
    x_proc, X = preprocess_like_r_code(x_raw, y_raw)
    y = metadata_df["class"].astype(str).to_numpy()

    released_rep_df, released_confusions, released_opt = run_repeated_double_cv(
        X=X,
        y=y,
        params=np.arange(1, 8),
        repetitions=100,
        out_segments=3,
        inn_segments=7,
    )
    pc14_rep_df, _, pc14_opt = run_repeated_double_cv(
        X=X,
        y=y,
        params=np.arange(1, 5),
        repetitions=100,
        out_segments=3,
        inn_segments=7,
    )

    released_summary = summarize_variant("released_code_like_param1_7", released_rep_df, released_opt)
    pc14_summary = summarize_variant("pc1_4_variant", pc14_rep_df, pc14_opt)
    metrics_df = pd.DataFrame([released_summary, pc14_summary])
    repetition_export_df = pd.concat(
        [
            released_rep_df.assign(variant="released_code_like_param1_7"),
            pc14_rep_df.assign(variant="pc1_4_variant"),
        ],
        ignore_index=True,
    )
    pca_variance_df = build_pca_variance_df(X)
    gaira_v1_df = load_gaira_v1_summary(gaira_v1_metrics_path)

    comparison_df = pd.DataFrame(
        [
            {
                "benchmark": "paper_reproduction_released_code_like",
                "accuracy": released_summary["mean_accuracy"],
                "balanced_accuracy": released_summary["mean_balanced_accuracy"],
                "roc_auc": released_summary["mean_roc_auc"],
                "notes": "Raw-data PCA-LDA with repeated double CV, released-code-like parameter range 1:7.",
            },
            {
                "benchmark": "paper_reproduction_pc1_4_variant",
                "accuracy": pc14_summary["mean_accuracy"],
                "balanced_accuracy": pc14_summary["mean_balanced_accuracy"],
                "roc_auc": pc14_summary["mean_roc_auc"],
                "notes": "Same as above but constrained to 1:4 PCs.",
            },
            {
                "benchmark": "gaira_v1_stratified_cv_lda",
                "accuracy": float(gaira_v1_df.query("split_name == 'stratified_cv' and model_name == 'lda'")["accuracy"].iloc[0]),
                "balanced_accuracy": float(gaira_v1_df.query("split_name == 'stratified_cv' and model_name == 'lda'")["balanced_accuracy"].iloc[0]),
                "roc_auc": float(gaira_v1_df.query("split_name == 'stratified_cv' and model_name == 'lda'")["roc_auc"].iloc[0]),
                "notes": "Existing GAIRA serum benchmark on processed spectra with simple stratified CV.",
            },
            {
                "benchmark": "gaira_v1_leave_one_batch_out_logreg",
                "accuracy": float(gaira_v1_df.query("split_name == 'leave_one_batch_out' and model_name == 'logreg'")["accuracy"].iloc[0]),
                "balanced_accuracy": float(gaira_v1_df.query("split_name == 'leave_one_batch_out' and model_name == 'logreg'")["balanced_accuracy"].iloc[0]),
                "roc_auc": float(gaira_v1_df.query("split_name == 'leave_one_batch_out' and model_name == 'logreg'")["roc_auc"].iloc[0]),
                "notes": "Existing GAIRA serum benchmark on processed spectra with leave-one-batch-out evaluation.",
            },
        ]
    )

    mean_cm = np.mean(np.stack(released_confusions, axis=0), axis=0)
    save_confusion_matrix_plot(mean_cm, output_dir / "hcc_serum_pcalda_confusion_matrix.png")
    save_projection_plot(X, y, output_dir / "hcc_serum_pcalda_projection.png")

    preprocessing_note = (
        "Loaded raw data.csv from the release, then applied the released-code sequence as closely as feasible: "
        "loess-like interpolation target replaced by deterministic linear interpolation to 400–1800 cm^-1 at 2 cm^-1 spacing; "
        "baseline correction approximated with an iterative degree-4 modified polynomial fit; "
        "final crop to 430–1730 cm^-1; vector normalization. "
        "The exact R baseline::modpolyfit implementation was not reproduced bit-for-bit."
    )
    cv_note = (
        "Repeated double cross-validation matched the released R structure: 100 repetitions, 3 outer stratified folds, "
        "7 inner stratified folds, PCA centered but not scaled, one-standard-error model selection, and LDA on the chosen PCs. "
        "A second comparison variant constrained PCA selection to PCs 1–4 only."
    )

    metrics_df.to_csv(output_dir / "hcc_serum_paper_comparison_metrics.csv", index=False)
    repetition_export_df.to_csv(output_dir / "hcc_serum_pcalda_reproduction_metrics.csv", index=False)
    comparison_df.to_csv(output_dir / "hcc_serum_pcalda_vs_gaira_v1_comparison.csv", index=False)
    pca_variance_df.to_csv(output_dir / "hcc_serum_pcalda_pca_variance.csv", index=False)
    write_summary(
        output_dir / "hcc_serum_paper_comparison_summary.txt",
        pd.DataFrame([released_summary]),
        pd.DataFrame([pc14_summary]),
        comparison_df,
        preprocessing_note,
        cv_note,
    )

    print(f"Outputs written to: {output_dir}")
    print("\nReleased-code-like PCA-LDA summary:")
    print(pd.DataFrame([released_summary]).to_string(index=False))
    print("\nPC1-4 constrained summary:")
    print(pd.DataFrame([pc14_summary]).to_string(index=False))
    print("\nComparison versus GAIRA v1:")
    print(comparison_df.to_string(index=False))


if __name__ == "__main__":
    main()
