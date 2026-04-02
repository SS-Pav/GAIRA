from __future__ import annotations

from pathlib import Path

import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from gaira.demo.v8_report_utils import add_text_page, image_grid_page, image_page, table_page


def maybe_table_page(
    pdf: PdfPages,
    title: str,
    dataframe: pd.DataFrame,
    *,
    subtitle: str | None = None,
    footer: str | None = None,
    font_size: float = 8.2,
) -> None:
    if dataframe.empty:
        return
    table_page(pdf, title, dataframe, subtitle=subtitle, footer=footer, font_size=font_size)


def add_figure_manifest_rows(rows: list[dict[str, str]], section: str, image_items: list[tuple[Path, str]]) -> None:
    for path, caption in image_items:
        rows.append({"section": section, "file": str(path), "caption": caption})


def maybe_image_grid(
    pdf: PdfPages,
    title: str,
    image_items: list[tuple[Path, str]],
    *,
    subtitle: str | None = None,
) -> None:
    existing = [(path, caption) for path, caption in image_items if path.exists()]
    if not existing:
        return
    image_grid_page(pdf, title, existing, subtitle=subtitle)


def maybe_image_page(
    pdf: PdfPages,
    title: str,
    image_path: Path,
    *,
    caption: str | None = None,
    subtitle: str | None = None,
) -> None:
    if not image_path.exists():
        return
    image_page(pdf, title, image_path, caption=caption, subtitle=subtitle)


__all__ = [
    "add_text_page",
    "maybe_table_page",
    "maybe_image_grid",
    "maybe_image_page",
    "add_figure_manifest_rows",
]
