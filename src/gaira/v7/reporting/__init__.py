"""GAIRA V7 deterministic report generation. One implementation, five callers."""
from .generator import SCOPE_NOTES, SECTIONS, render, render_html, render_json, render_pdf

__all__ = ["render", "render_json", "render_html", "render_pdf", "SECTIONS", "SCOPE_NOTES"]
