from __future__ import annotations

import re
from pathlib import Path

import duckdb
import pandas as pd


EV_CONTEXT_LAYER = "GAIRA_EV_CONTEXT"


class EVContextRetriever:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.documents_df, self.chunks_df = self._load_tables()
        self.document_type_map = (
            self.documents_df.set_index("document_id")["context_type"].to_dict()
            if not self.documents_df.empty
            else {}
        )

    def _load_tables(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        with duckdb.connect(str(self.db_path), read_only=True) as connection:
            documents_df = connection.execute(
                """
                SELECT *
                FROM domain_context_documents
                WHERE context_layer = ?
                  AND intended_domain = 'ev'
                ORDER BY document_id
                """,
                [EV_CONTEXT_LAYER],
            ).fetchdf()

            chunks_df = connection.execute(
                """
                SELECT *
                FROM domain_context_chunks
                WHERE context_layer = ?
                  AND intended_domain = 'ev'
                ORDER BY document_id, chunk_order
                """,
                [EV_CONTEXT_LAYER],
            ).fetchdf()
        return documents_df, chunks_df

    def search_by_text(self, query_text: str, top_n: int = 8) -> pd.DataFrame:
        query_tokens = {
            token.lower()
            for token in re.findall(r"[A-Za-z0-9_\+\-]+", query_text)
            if len(token) >= 3
        }
        rows: list[dict] = []
        for row in self.chunks_df.to_dict(orient="records"):
            chunk_text = str(row["chunk_text"])
            chunk_tokens = {
                token.lower()
                for token in re.findall(r"[A-Za-z0-9_\+\-]+", chunk_text)
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
                    "context_type": self.document_type_map.get(row["document_id"], "unknown"),
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

    def search_by_labels(self, labels: list[str], top_n: int = 8) -> pd.DataFrame:
        tokens = [str(label).lower() for label in labels if str(label).strip()]
        rows: list[dict] = []
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
                    "context_type": self.document_type_map.get(row["document_id"], "unknown"),
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
