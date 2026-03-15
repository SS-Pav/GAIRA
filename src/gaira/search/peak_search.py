from pathlib import Path
import json

import duckdb
import numpy as np
import pandas as pd
from scipy.signal import find_peaks


class PeakSearchEngine:
    """Simple Raman peak search with full-spectrum reranking."""

    def __init__(
        self,
        db_path: Path,
        peak_tolerance_cm: float = 5.0,
        min_peak_prominence: float = 0.03,
        min_peak_height: float = 0.03,
    ) -> None:
        self.db_path = Path(db_path)
        self.peak_tolerance_cm = peak_tolerance_cm
        self.min_peak_prominence = min_peak_prominence
        self.min_peak_height = min_peak_height
        self._reference_peaks_df: pd.DataFrame | None = None

    def load_query_spectrum(self, file_path: Path) -> pd.DataFrame:
        """Load a query CSV and rename columns to wavenumber/intensity."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Query spectrum file not found: {file_path}")

        query_df = pd.read_csv(file_path)
        if len(query_df.columns) != 2:
            raise ValueError("The query CSV must contain exactly two columns.")

        lowered_columns = [str(column).strip().lower() for column in query_df.columns]
        if lowered_columns in (
            ["x", "y"],
            ["wavenumber", "intensity"],
            ["wavenumbers", "intensity"],
        ):
            query_df.columns = ["wavenumber", "intensity"]
        else:
            # Try again for a simple two-column file with no header row.
            query_df = pd.read_csv(file_path, header=None, names=["wavenumber", "intensity"])

        query_df["wavenumber"] = pd.to_numeric(query_df["wavenumber"], errors="coerce")
        query_df["intensity"] = pd.to_numeric(query_df["intensity"], errors="coerce")
        query_df = query_df.dropna(subset=["wavenumber", "intensity"]).sort_values("wavenumber")
        query_df = query_df.reset_index(drop=True)

        if query_df.empty:
            raise ValueError("The query spectrum contains no valid numeric rows.")

        return query_df

    def crop_query_spectrum(
        self,
        df: pd.DataFrame,
        x_min: float = 450.0,
        x_max: float = 1800.0,
    ) -> pd.DataFrame:
        """Keep the spectral window that overlaps the RamanBioLib references."""
        cropped_df = df[(df["wavenumber"] >= x_min) & (df["wavenumber"] <= x_max)].copy()
        cropped_df = cropped_df.reset_index(drop=True)

        if cropped_df.empty:
            raise ValueError("No spectrum points remain after cropping to 450-1800 cm-1.")

        return cropped_df

    def normalize_query_spectrum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale intensity to 0-1 so spectra are easier to compare."""
        normalized_df = df.copy()
        min_intensity = float(normalized_df["intensity"].min())
        max_intensity = float(normalized_df["intensity"].max())

        if max_intensity == min_intensity:
            raise ValueError("The query spectrum is flat and cannot be normalized.")

        normalized_df["intensity"] = (
            normalized_df["intensity"] - min_intensity
        ) / (max_intensity - min_intensity)
        return normalized_df

    def detect_query_peaks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect local peaks in the normalized query spectrum."""
        peak_indices, peak_properties = find_peaks(
            df["intensity"].to_numpy(),
            prominence=self.min_peak_prominence,
            height=self.min_peak_height,
        )

        peaks_df = pd.DataFrame(
            {
                "peak_rank": range(1, len(peak_indices) + 1),
                "peak_cm": df.iloc[peak_indices]["wavenumber"].to_numpy(),
                "peak_intensity": peak_properties["peak_heights"],
            }
        )

        if peaks_df.empty:
            raise ValueError(
                "No peaks were detected. Try lowering the prominence or height threshold."
            )

        return peaks_df

    def _get_reference_peaks(self) -> pd.DataFrame:
        """Load reference peaks once and add an inverse-frequency weight."""
        if self._reference_peaks_df is not None:
            return self._reference_peaks_df

        if not self.db_path.exists():
            raise FileNotFoundError(f"DuckDB database not found: {self.db_path}")

        try:
            with duckdb.connect(str(self.db_path), read_only=True) as connection:
                reference_df = connection.execute(
                    """
                    SELECT
                        rp.ref_id,
                        rp.component,
                        rm.biochemical_class,
                        rp.peak_cm,
                        rp.rel_intensity
                    FROM reference_peaks AS rp
                    LEFT JOIN reference_metadata AS rm
                        ON rp.ref_id = rm.ref_id
                    """
                ).fetchdf()
        except duckdb.Error as exc:
            raise RuntimeError(
                "Could not load DuckDB reference peaks. Make sure RamanBioLib ingestion has completed."
            ) from exc

        if reference_df.empty:
            raise RuntimeError("The reference_peaks table is empty. Ingest RamanBioLib first.")

        reference_df["peak_bin"] = reference_df["peak_cm"].round().astype(int)
        total_ref_count = reference_df["ref_id"].nunique()
        bin_counts_df = (
            reference_df.groupby("peak_bin")["ref_id"]
            .nunique()
            .reset_index(name="refs_in_bin")
        )
        reference_df = reference_df.merge(bin_counts_df, on="peak_bin", how="left")

        # Common peaks get slightly smaller weights.
        reference_df["idf_weight"] = np.log(
            (1 + total_ref_count) / (1 + reference_df["refs_in_bin"])
        ) + 1.0
        self._reference_peaks_df = reference_df
        return self._reference_peaks_df

    def search_reference_peaks(self, peaks_df: pd.DataFrame) -> pd.DataFrame:
        """Find reference peaks within the tolerance window of each query peak."""
        if peaks_df.empty:
            return pd.DataFrame()

        reference_peaks_df = self._get_reference_peaks()
        matches: list[pd.DataFrame] = []

        for row in peaks_df.itertuples(index=False):
            match_df = reference_peaks_df[
                (reference_peaks_df["peak_cm"] >= float(row.peak_cm) - self.peak_tolerance_cm)
                & (reference_peaks_df["peak_cm"] <= float(row.peak_cm) + self.peak_tolerance_cm)
            ].copy()

            if match_df.empty:
                continue

            match_df["query_peak_cm"] = float(row.peak_cm)
            match_df["query_peak_intensity"] = float(row.peak_intensity)
            match_df["reference_peak_cm"] = match_df["peak_cm"]
            match_df["peak_delta_cm"] = (match_df["reference_peak_cm"] - float(row.peak_cm)).abs()
            matches.append(
                match_df[
                    [
                        "query_peak_cm",
                        "query_peak_intensity",
                        "ref_id",
                        "component",
                        "biochemical_class",
                        "reference_peak_cm",
                        "rel_intensity",
                        "peak_delta_cm",
                        "idf_weight",
                    ]
                ]
            )

        if not matches:
            return pd.DataFrame()

        return pd.concat(matches, ignore_index=True)

    def _scale_series_0_1(self, values: pd.Series) -> pd.Series:
        """Scale a numeric series to 0-1 before combining scores."""
        values = values.astype(float)
        min_value = values.min()
        max_value = values.max()

        if pd.isna(min_value) or pd.isna(max_value) or max_value == min_value:
            return pd.Series(np.ones(len(values)), index=values.index, dtype=float)

        return (values - min_value) / (max_value - min_value)

    def _get_strong_reference_peaks(self, reference_df: pd.DataFrame) -> pd.DataFrame:
        """Keep stronger peaks so missing important peaks can be penalized."""
        if reference_df.empty:
            return reference_df

        threshold = max(0.2, float(reference_df["rel_intensity"].quantile(0.75)))
        strong_df = reference_df[reference_df["rel_intensity"] >= threshold].copy()

        if strong_df.empty:
            strong_df = reference_df.nlargest(min(5, len(reference_df)), "rel_intensity").copy()

        return strong_df

    def score_candidates(
        self,
        matches_df: pd.DataFrame,
        query_peaks_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Aggregate peak matches into ref-level and component-level scores."""
        empty_ref_df = pd.DataFrame(
            columns=[
                "ref_id",
                "component",
                "biochemical_class",
                "matched_peak_count",
                "query_peak_count",
                "matched_fraction",
                "mean_abs_delta_cm",
                "intensity_weighted_score",
                "strong_peak_coverage",
                "missing_peak_penalty",
                "overall_score",
            ]
        )
        empty_component_df = pd.DataFrame(
            columns=[
                "component",
                "biochemical_class",
                "supporting_ref_count",
                "best_ref_id",
                "matched_peak_count",
                "query_peak_count",
                "matched_fraction",
                "mean_abs_delta_cm",
                "intensity_weighted_score",
                "strong_peak_coverage",
                "missing_peak_penalty",
                "overall_score",
            ]
        )

        if matches_df.empty:
            return empty_ref_df, empty_component_df

        reference_peaks_df = self._get_reference_peaks()

        best_query_matches_df = matches_df.sort_values("peak_delta_cm").drop_duplicates(
            subset=["ref_id", "query_peak_cm"],
            keep="first",
        )
        best_reference_matches_df = matches_df.sort_values("peak_delta_cm").drop_duplicates(
            subset=["ref_id", "reference_peak_cm"],
            keep="first",
        )

        query_peak_count = len(query_peaks_df)
        candidate_rows: list[dict] = []

        for (ref_id, component, biochemical_class), group_df in best_query_matches_df.groupby(
            ["ref_id", "component", "biochemical_class"],
            dropna=False,
        ):
            reference_df = reference_peaks_df[reference_peaks_df["ref_id"] == ref_id].copy()
            strong_reference_df = self._get_strong_reference_peaks(reference_df)
            strong_reference_peaks_df = strong_reference_df[["peak_cm"]].rename(
                columns={"peak_cm": "reference_peak_cm"}
            )
            strong_matches_df = best_reference_matches_df[
                best_reference_matches_df["ref_id"] == ref_id
            ].copy()
            strong_matches_df = strong_matches_df.merge(
                strong_reference_peaks_df,
                on="reference_peak_cm",
                how="inner",
            )

            matched_peak_count = group_df["query_peak_cm"].nunique()
            matched_fraction = matched_peak_count / query_peak_count if query_peak_count else 0.0
            mean_abs_delta_cm = float(group_df["peak_delta_cm"].mean())

            raw_match_strength = (
                group_df["query_peak_intensity"]
                * group_df["rel_intensity"]
                * group_df["idf_weight"]
                / (1 + group_df["peak_delta_cm"])
            ).sum()

            total_strong_weight = (
                strong_reference_df["rel_intensity"] * strong_reference_df["idf_weight"]
            ).sum()
            matched_strong_weight = (
                strong_matches_df["rel_intensity"] * strong_matches_df["idf_weight"]
            ).sum()

            if total_strong_weight > 0:
                strong_peak_coverage = float(matched_strong_weight / total_strong_weight)
            else:
                strong_peak_coverage = 0.0

            missing_peak_penalty = max(0.0, 1.0 - strong_peak_coverage)

            candidate_rows.append(
                {
                    "ref_id": ref_id,
                    "component": component,
                    "biochemical_class": biochemical_class,
                    "matched_peak_count": matched_peak_count,
                    "query_peak_count": query_peak_count,
                    "matched_fraction": matched_fraction,
                    "mean_abs_delta_cm": mean_abs_delta_cm,
                    "raw_match_strength": float(raw_match_strength),
                    "strong_peak_coverage": strong_peak_coverage,
                    "missing_peak_penalty": missing_peak_penalty,
                }
            )

        candidate_df_ref = pd.DataFrame(candidate_rows)
        if candidate_df_ref.empty:
            return empty_ref_df, empty_component_df

        candidate_df_ref["intensity_weighted_score"] = self._scale_series_0_1(
            candidate_df_ref["raw_match_strength"]
        )
        candidate_df_ref["overall_score"] = (
            0.35 * candidate_df_ref["matched_fraction"]
            + 0.30 * candidate_df_ref["intensity_weighted_score"]
            + 0.20 * (1 / (1 + candidate_df_ref["mean_abs_delta_cm"]))
            + 0.15 * candidate_df_ref["strong_peak_coverage"]
            - 0.20 * candidate_df_ref["missing_peak_penalty"]
        )
        candidate_df_ref["overall_score"] = candidate_df_ref["overall_score"].clip(lower=0.0)
        candidate_df_ref = candidate_df_ref.sort_values(
            ["overall_score", "matched_peak_count", "intensity_weighted_score"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

        component_rows: list[dict] = []
        for (component, biochemical_class), group_df in candidate_df_ref.groupby(
            ["component", "biochemical_class"],
            dropna=False,
        ):
            best_row = group_df.sort_values("overall_score", ascending=False).iloc[0]
            component_score = 0.7 * float(group_df["overall_score"].max()) + 0.3 * float(
                group_df["overall_score"].mean()
            )
            component_rows.append(
                {
                    "component": component,
                    "biochemical_class": biochemical_class,
                    "supporting_ref_count": int(group_df["ref_id"].nunique()),
                    "best_ref_id": best_row["ref_id"],
                    "matched_peak_count": int(group_df["matched_peak_count"].max()),
                    "query_peak_count": int(best_row["query_peak_count"]),
                    "matched_fraction": float(group_df["matched_fraction"].max()),
                    "mean_abs_delta_cm": float(group_df["mean_abs_delta_cm"].min()),
                    "intensity_weighted_score": float(group_df["intensity_weighted_score"].max()),
                    "strong_peak_coverage": float(group_df["strong_peak_coverage"].max()),
                    "missing_peak_penalty": float(group_df["missing_peak_penalty"].min()),
                    "overall_score": component_score,
                }
            )

        candidate_df_component = pd.DataFrame(component_rows).sort_values(
            ["overall_score", "matched_peak_count", "intensity_weighted_score"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

        candidate_df_ref = candidate_df_ref[
            [
                "ref_id",
                "component",
                "biochemical_class",
                "matched_peak_count",
                "query_peak_count",
                "matched_fraction",
                "mean_abs_delta_cm",
                "intensity_weighted_score",
                "strong_peak_coverage",
                "missing_peak_penalty",
                "overall_score",
            ]
        ]
        return candidate_df_ref, candidate_df_component

    def load_reference_spectrum(self, ref_id: str) -> pd.DataFrame:
        """Load one reference spectrum from DuckDB and unpack the JSON arrays."""
        if not self.db_path.exists():
            raise FileNotFoundError(f"DuckDB database not found: {self.db_path}")

        try:
            with duckdb.connect(str(self.db_path), read_only=True) as connection:
                row = connection.execute(
                    """
                    SELECT wavenumbers_json, intensity_json
                    FROM reference_spectra
                    WHERE ref_id = ?
                    """,
                    [ref_id],
                ).fetchone()
        except duckdb.Error as exc:
            raise RuntimeError(
                "Could not load reference spectra from DuckDB. Make sure RamanBioLib ingestion has completed."
            ) from exc

        if row is None:
            raise ValueError(f"No reference spectrum found for ref_id: {ref_id}")

        wavenumbers = json.loads(row[0])
        intensity = json.loads(row[1])

        if len(wavenumbers) != len(intensity):
            raise ValueError(f"Reference spectrum {ref_id} has mismatched x/y array lengths.")

        spectrum_df = pd.DataFrame(
            {"wavenumber": pd.to_numeric(wavenumbers), "intensity": pd.to_numeric(intensity)}
        )
        return spectrum_df.sort_values("wavenumber").reset_index(drop=True)

    def interpolate_reference_to_query_grid(
        self,
        reference_df: pd.DataFrame,
        query_grid: pd.Series,
    ) -> pd.DataFrame:
        """Interpolate the reference spectrum onto the query wavenumber grid."""
        interpolated_intensity = np.interp(
            query_grid.to_numpy(),
            reference_df["wavenumber"].to_numpy(),
            reference_df["intensity"].to_numpy(),
        )
        return pd.DataFrame(
            {
                "wavenumber": query_grid.to_numpy(),
                "intensity": interpolated_intensity,
            }
        )

    def _normalize_intensity_vector(self, values: np.ndarray) -> np.ndarray | None:
        """Min-max normalize a raw intensity vector."""
        min_value = float(values.min())
        max_value = float(values.max())

        if max_value == min_value:
            return None

        return (values - min_value) / (max_value - min_value)

    def _cosine_similarity(self, left: np.ndarray, right: np.ndarray) -> float:
        """Compute cosine similarity between two spectra."""
        left_norm = float(np.linalg.norm(left))
        right_norm = float(np.linalg.norm(right))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0

        return float(np.dot(left, right) / (left_norm * right_norm))

    def _pearson_correlation(self, left: np.ndarray, right: np.ndarray) -> float:
        """Compute Pearson correlation between two spectra."""
        if np.std(left) == 0.0 or np.std(right) == 0.0:
            return 0.0

        correlation = float(np.corrcoef(left, right)[0, 1])
        if np.isnan(correlation):
            return 0.0

        return correlation

    def rerank_with_spectral_similarity(
        self,
        query_spectrum_df: pd.DataFrame,
        candidate_df_ref: pd.DataFrame,
        top_k: int = 50,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Rerank top peak candidates using full-spectrum similarity."""
        empty_df = pd.DataFrame(
            columns=[
                "ref_id",
                "component",
                "biochemical_class",
                "peak_score",
                "cosine_similarity",
                "pearson_correlation",
                "spectral_similarity_score",
                "final_score",
            ]
        )

        if candidate_df_ref.empty:
            return empty_df, empty_df

        query_values = query_spectrum_df["intensity"].to_numpy(dtype=float)
        normalized_query = self._normalize_intensity_vector(query_values)
        if normalized_query is None:
            raise ValueError("The normalized query spectrum became flat before reranking.")

        reranked_rows: list[dict] = []
        top_candidates_df = candidate_df_ref.head(top_k).copy()

        for row in top_candidates_df.itertuples(index=False):
            try:
                reference_df = self.load_reference_spectrum(row.ref_id)
                interpolated_df = self.interpolate_reference_to_query_grid(
                    reference_df,
                    query_spectrum_df["wavenumber"],
                )
            except (ValueError, RuntimeError, FileNotFoundError) as exc:
                print(f"Skipping spectral reranking for {row.ref_id}: {exc}")
                continue

            normalized_reference = self._normalize_intensity_vector(
                interpolated_df["intensity"].to_numpy(dtype=float)
            )
            if normalized_reference is None:
                print(f"Skipping spectral reranking for {row.ref_id}: reference spectrum is flat.")
                continue

            cosine_similarity = self._cosine_similarity(normalized_query, normalized_reference)
            pearson_correlation = self._pearson_correlation(
                normalized_query,
                normalized_reference,
            )
            spectral_similarity_score = 0.5 * cosine_similarity + 0.5 * pearson_correlation
            final_score = (
                0.4 * float(row.overall_score)
                + 0.3 * cosine_similarity
                + 0.3 * pearson_correlation
            )

            reranked_rows.append(
                {
                    "ref_id": row.ref_id,
                    "component": row.component,
                    "biochemical_class": row.biochemical_class,
                    "peak_score": float(row.overall_score),
                    "cosine_similarity": cosine_similarity,
                    "pearson_correlation": pearson_correlation,
                    "spectral_similarity_score": spectral_similarity_score,
                    "final_score": final_score,
                }
            )

        if not reranked_rows:
            return empty_df, empty_df

        candidate_df_reranked = pd.DataFrame(reranked_rows).sort_values(
            ["final_score", "spectral_similarity_score", "peak_score"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        candidate_df_similarity = candidate_df_reranked.sort_values(
            ["spectral_similarity_score", "cosine_similarity", "pearson_correlation"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        return candidate_df_similarity, candidate_df_reranked

    def aggregate_reranked_components(self, candidate_df_reranked: pd.DataFrame) -> pd.DataFrame:
        """Combine reranked ref-level rows into component-level scores."""
        if candidate_df_reranked.empty:
            return pd.DataFrame(
                columns=[
                    "component",
                    "biochemical_class",
                    "supporting_ref_count",
                    "best_ref_id",
                    "best_final_score",
                    "mean_final_score",
                    "mean_spectral_similarity_score",
                    "mean_peak_score",
                    "overall_component_score",
                ]
            )

        component_rows: list[dict] = []
        for (component, biochemical_class), group_df in candidate_df_reranked.groupby(
            ["component", "biochemical_class"],
            dropna=False,
        ):
            best_row = group_df.sort_values("final_score", ascending=False).iloc[0]
            best_final_score = float(group_df["final_score"].max())
            mean_final_score = float(group_df["final_score"].mean())
            overall_component_score = 0.7 * best_final_score + 0.3 * mean_final_score

            component_rows.append(
                {
                    "component": component,
                    "biochemical_class": biochemical_class,
                    "supporting_ref_count": int(group_df["ref_id"].nunique()),
                    "best_ref_id": best_row["ref_id"],
                    "best_final_score": best_final_score,
                    "mean_final_score": mean_final_score,
                    "mean_spectral_similarity_score": float(
                        group_df["spectral_similarity_score"].mean()
                    ),
                    "mean_peak_score": float(group_df["peak_score"].mean()),
                    "overall_component_score": overall_component_score,
                }
            )

        return pd.DataFrame(component_rows).sort_values(
            ["overall_component_score", "best_final_score", "mean_final_score"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

    def summarize_classes(self, candidate_df_component: pd.DataFrame) -> pd.DataFrame:
        """Summarize the reranked component table by biochemical class."""
        if candidate_df_component.empty:
            return pd.DataFrame(
                columns=["biochemical_class", "candidate_count", "mean_score", "max_score"]
            )

        return (
            candidate_df_component.groupby("biochemical_class", dropna=False)[
                "overall_component_score"
            ]
            .agg(candidate_count="count", mean_score="mean", max_score="max")
            .reset_index()
            .sort_values(["max_score", "mean_score"], ascending=[False, False])
            .reset_index(drop=True)
        )

    def interpret_mixture(
        self,
        candidate_df_component: pd.DataFrame,
        class_df: pd.DataFrame,
    ) -> str:
        """Produce a short human-readable summary of the search result."""
        if candidate_df_component.empty:
            return "No strong RamanBioLib matches were found for this query."

        top_components = candidate_df_component.head(3)["component"].tolist()
        text_parts = [f"Top component-level matches are {', '.join(top_components)}."]

        if len(candidate_df_component) > 1:
            top_score = float(candidate_df_component.iloc[0]["overall_component_score"])
            second_score = float(candidate_df_component.iloc[1]["overall_component_score"])
            if top_score >= second_score + 0.15:
                text_parts.append("One component stands out more strongly than the rest.")
            else:
                text_parts.append("Several components remain plausible matches.")

        if len(class_df) > 1:
            top_class_gap = abs(float(class_df.iloc[0]["max_score"]) - float(class_df.iloc[1]["max_score"]))
            if top_class_gap <= 0.08:
                text_parts.append(
                    "Several biochemical classes have similar support, so the sample may be mixed."
                )

        joined_classes = " ".join(
            str(class_name).lower() for class_name in class_df.head(3)["biochemical_class"].tolist()
        )
        if "lipid" in joined_classes or "fattyacid" in joined_classes:
            text_parts.append("The top matches suggest a lipid-enriched sample.")
        if "amino" in joined_classes or "protein" in joined_classes or "peptide" in joined_classes:
            text_parts.append("The top matches suggest a proteinaceous or amino-acid-rich sample.")
        if "carbo" in joined_classes or "saccharide" in joined_classes or "sugar" in joined_classes:
            text_parts.append("The top matches suggest a carbohydrate-rich sample.")

        return " ".join(text_parts)

    def run(self, file_path: Path) -> dict:
        """Run the baseline GAIRA search workflow."""
        query_spectrum_df = self.load_query_spectrum(file_path)
        query_spectrum_df = self.crop_query_spectrum(query_spectrum_df)
        query_spectrum_df = self.normalize_query_spectrum(query_spectrum_df)
        query_peaks_df = self.detect_query_peaks(query_spectrum_df)
        peak_matches_df = self.search_reference_peaks(query_peaks_df)
        candidate_df_ref, _candidate_df_component_peak = self.score_candidates(
            peak_matches_df,
            query_peaks_df,
        )
        candidate_df_similarity, candidate_df_reranked = self.rerank_with_spectral_similarity(
            query_spectrum_df,
            candidate_df_ref,
        )
        candidate_df_component_reranked = self.aggregate_reranked_components(
            candidate_df_reranked
        )
        class_df_reranked = self.summarize_classes(candidate_df_component_reranked)
        interpretation_text = self.interpret_mixture(
            candidate_df_component_reranked,
            class_df_reranked,
        )

        return {
            "query_spectrum_df": query_spectrum_df,
            "query_peaks_df": query_peaks_df,
            "peak_matches_df": peak_matches_df,
            "candidate_df_ref": candidate_df_ref,
            "candidate_df_similarity": candidate_df_similarity,
            "candidate_df_reranked": candidate_df_reranked,
            "candidate_df_component_reranked": candidate_df_component_reranked,
            "class_df_reranked": class_df_reranked,
            "interpretation_text": interpretation_text,
        }
