from __future__ import annotations

import re
from pathlib import Path

import duckdb
import pandas as pd


SERUM_CONTEXT_LAYER = "GAIRA_SERUM_CONTEXT"


def extract_chunk_bands(chunk_text: str) -> list[tuple[float, float]]:
    ranges: list[tuple[float, float]] = []
    for match in re.finditer(r"(?<![\d.])(\d{3,4})(?:\s*-\s*(\d{2,4}))?", chunk_text):
        start_value = float(match.group(1))
        end_text = match.group(2)
        if end_text is None:
            ranges.append((start_value, start_value))
            continue
        end_value = float(end_text)
        if end_value < 100:
            prefix = match.group(1)[: len(match.group(1)) - len(end_text)]
            end_value = float(f"{prefix}{end_text}")
        ranges.append((min(start_value, end_value), max(start_value, end_value)))
    return ranges


class SerumContextRetriever:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.documents_df, self.chunks_df = self._load_tables()

    def _load_tables(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        with duckdb.connect(str(self.db_path), read_only=True) as connection:
            documents_df = connection.execute(
                """
                SELECT *
                FROM domain_context_documents
                WHERE context_layer = ?
                  AND intended_domain = 'serum'
                ORDER BY document_id
                """,
                [SERUM_CONTEXT_LAYER],
            ).fetchdf()

            chunks_df = connection.execute(
                """
                SELECT *
                FROM domain_context_chunks
                WHERE context_layer = ?
                  AND intended_domain = 'serum'
                ORDER BY document_id, chunk_order
                """,
                [SERUM_CONTEXT_LAYER],
            ).fetchdf()
        return documents_df, chunks_df

    def search_by_bands(self, bands_cm: list[float], top_n: int = 8) -> pd.DataFrame:
        rows: list[dict] = []
        document_type_map = self.documents_df.set_index("document_id")["context_type"].to_dict()
        for row in self.chunks_df.to_dict(orient="records"):
            chunk_text = str(row["chunk_text"])
            extracted_ranges = extract_chunk_bands(chunk_text)
            score = 0.0
            matched_tokens: list[str] = []
            for band_cm in bands_cm:
                rounded_band = int(round(float(band_cm)))
                for start_value, end_value in extracted_ranges:
                    if start_value <= band_cm <= end_value:
                        score += 1.0
                        if start_value == end_value:
                            matched_tokens.append(str(int(round(start_value))))
                        else:
                            matched_tokens.append(
                                f"{int(round(start_value))}-{int(round(end_value))}"
                            )
                        break
            if score <= 0:
                continue
            rows.append(
                {
                    "query_type": "band_context_search",
                    "query_bands_cm": ", ".join(str(int(round(value))) for value in bands_cm),
                    "document_id": row["document_id"],
                    "context_type": document_type_map[row["document_id"]],
                    "section": row["section"],
                    "score": score,
                    "matched_tokens": ", ".join(sorted(set(matched_tokens))),
                    "chunk_text": chunk_text,
                }
            )

        result_df = pd.DataFrame(rows)
        if result_df.empty:
            return result_df
        return result_df.sort_values(["score", "document_id"], ascending=[False, True]).head(top_n)

    def search_by_grounding_labels(self, labels: list[str], top_n: int = 8) -> pd.DataFrame:
        rows: list[dict] = []
        document_type_map = self.documents_df.set_index("document_id")["context_type"].to_dict()
        tokens = [label.lower() for label in labels if str(label).strip()]
        for row in self.chunks_df.to_dict(orient="records"):
            chunk_text = str(row["chunk_text"]).lower()
            matches = [token for token in tokens if token in chunk_text]
            if not matches:
                continue
            rows.append(
                {
                    "query_type": "label_context_search",
                    "query_labels": ", ".join(labels),
                    "document_id": row["document_id"],
                    "context_type": document_type_map[row["document_id"]],
                    "section": row["section"],
                    "score": float(len(matches)),
                    "matched_tokens": ", ".join(sorted(set(matches))),
                    "chunk_text": str(row["chunk_text"]),
                }
            )
        result_df = pd.DataFrame(rows)
        if result_df.empty:
            return result_df
        return result_df.sort_values(["score", "document_id"], ascending=[False, True]).head(top_n)

    def search_by_text(self, query_text: str, top_n: int = 8) -> pd.DataFrame:
        query_tokens = {
            token.lower()
            for token in re.findall(r"[A-Za-z0-9\+\-]+", query_text)
            if len(token) >= 3
        }
        rows: list[dict] = []
        document_type_map = self.documents_df.set_index("document_id")["context_type"].to_dict()
        for row in self.chunks_df.to_dict(orient="records"):
            chunk_text = str(row["chunk_text"])
            chunk_tokens = {
                token.lower()
                for token in re.findall(r"[A-Za-z0-9\+\-]+", chunk_text)
                if len(token) >= 3
            }
            overlap = sorted(query_tokens & chunk_tokens)
            if not overlap:
                continue
            rows.append(
                {
                    "query_type": "text_context_search",
                    "query_text": query_text,
                    "document_id": row["document_id"],
                    "context_type": document_type_map[row["document_id"]],
                    "section": row["section"],
                    "score": float(len(overlap)),
                    "matched_tokens": ", ".join(overlap[:12]),
                    "chunk_text": chunk_text,
                }
            )
        result_df = pd.DataFrame(rows)
        if result_df.empty:
            return result_df
        return result_df.sort_values(["score", "document_id"], ascending=[False, True]).head(top_n)
