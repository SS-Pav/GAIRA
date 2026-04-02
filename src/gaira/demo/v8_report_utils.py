from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Iterable

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


def wrap_paragraphs(paragraphs: Iterable[str], width: int = 98) -> str:
    chunks = []
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        chunks.append(textwrap.fill(paragraph, width=width))
    return "\n\n".join(chunks)


def add_text_page(
    pdf: PdfPages,
    title: str,
    paragraphs: list[str],
    *,
    subtitle: str | None = None,
    footer: str | None = None,
) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    fig.text(0.07, 0.955, title, fontsize=20, fontweight="bold", va="top")
    if subtitle:
        fig.text(0.07, 0.925, textwrap.fill(subtitle, 100), fontsize=10.5, va="top", color="#444444")
        top = 0.89
    else:
        top = 0.915
    fig.text(0.07, top, wrap_paragraphs(paragraphs), fontsize=10.5, va="top")
    if footer:
        fig.text(0.07, 0.035, footer, fontsize=8.2, color="#666666")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def table_page(
    pdf: PdfPages,
    title: str,
    dataframe: pd.DataFrame,
    *,
    subtitle: str | None = None,
    footer: str | None = None,
    font_size: float = 8.2,
    scale_y: float = 1.32,
) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_axes([0.05, 0.08, 0.90, 0.82])
    ax.axis("off")
    fig.text(0.05, 0.96, title, fontsize=18, fontweight="bold", va="top")
    if subtitle:
        fig.text(0.05, 0.93, textwrap.fill(subtitle, 110), fontsize=9.4, va="top")
    table = ax.table(
        cellText=dataframe.values,
        colLabels=dataframe.columns,
        cellLoc="center",
        loc="upper left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1, scale_y)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#dde7f5")
        elif row % 2 == 0:
            cell.set_facecolor("#f7f9fc")
    if footer:
        fig.text(0.05, 0.03, footer, fontsize=8.0, color="#666666")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def image_page(
    pdf: PdfPages,
    title: str,
    image_path: Path,
    *,
    caption: str | None = None,
    subtitle: str | None = None,
) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_axes([0.06, 0.16, 0.88, 0.70])
    ax.axis("off")
    fig.text(0.06, 0.96, title, fontsize=18, fontweight="bold", va="top")
    if subtitle:
        fig.text(0.06, 0.93, textwrap.fill(subtitle, 110), fontsize=9.5, va="top")
    image = mpimg.imread(image_path)
    ax.imshow(image)
    if caption:
        fig.text(0.06, 0.08, textwrap.fill(caption, 108), fontsize=9.3, color="#444444")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def image_grid_page(
    pdf: PdfPages,
    title: str,
    image_items: list[tuple[Path, str]],
    *,
    subtitle: str | None = None,
) -> None:
    rows = len(image_items)
    fig, axes = plt.subplots(rows, 1, figsize=(8.27, 11.69))
    if rows == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=18, fontweight="bold", x=0.06, y=0.98, ha="left")
    if subtitle:
        fig.text(0.06, 0.945, textwrap.fill(subtitle, 110), fontsize=9.5, va="top")
    top = 0.90 if subtitle else 0.94
    fig.subplots_adjust(left=0.06, right=0.96, top=top, bottom=0.06, hspace=0.22)
    for ax, (image_path, caption) in zip(axes, image_items):
        ax.axis("off")
        ax.imshow(mpimg.imread(image_path))
        ax.set_title(caption, fontsize=10.5, loc="left")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
