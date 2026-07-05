"""Tab 1 — Overview / Evidence Stack.

Explains what GAIRA is and lays out the evidence layers that produce
biochemical interpretations from Raman/SERS spectra.
"""
from __future__ import annotations

from pathlib import Path
import streamlit as st

from components import ui_blocks as ui


def _hero(app_cfg: dict) -> None:
    st.markdown(
        f"# {app_cfg['title']}\n"
        f"**{app_cfg['subtitle']}**"
    )
    st.markdown(
        '<div style="color:#8b949e; font-size:0.95rem; margin-top:-6px;">'
        'GAIRA does not replace Raman classifiers — it explains what they learn, '
        'what transfers across substrates / cohorts / instruments, and what is '
        'chemically defensible.</div>',
        unsafe_allow_html=True,
    )


def _pipeline(steps: list[str]) -> None:
    ui.section_header("GAIRA pipeline",
                      "From a raw spectrum to an interpretable, evidence-tiered report.")
    ui.pipeline_flow(steps)


def _evidence_cards(layers: list[dict]) -> None:
    ui.section_header("Evidence-stack layers",
                      "Each layer contributes evidence; none is treated as ground truth alone.")

    cols = st.columns(2)
    for i, layer in enumerate(layers):
        col = cols[i % 2]
        with col:
            sources_html = "<ul style='margin: 4px 0 4px 18px; padding: 0;'>"
            for s in layer.get("sources", []):
                sources_html += f"<li style='font-size:0.88rem;'>{s}</li>"
            sources_html += "</ul>"
            note = layer.get("note", "")
            note_html = (f'<div style="color:#8b949e; font-size:0.82rem; '
                         f'margin-top:6px;"><em>{note}</em></div>') if note else ""
            ui.card(
                title=layer["title"],
                subtitle=layer.get("role", ""),
                body_md=sources_html + note_html,
            )


def _coverage_summary(manifest: dict, manifest_path: Path) -> None:
    ui.section_header("Dataset & artifact coverage",
                      "Detected from the latest artifact manifest.")
    summary = manifest.get("summary", {})
    metrics = {
        "phases present": str(summary.get("phases_present", 0)),
        "phases missing": str(summary.get("phases_missing", 0)),
        "tables (.csv)": str(summary.get("csv_total", 0)),
        "figures (.png)": str(summary.get("png_total", 0)),
        "reports (.md)": str(summary.get("md_total", 0)),
        "total artifacts": str(summary.get("artifacts_total", 0)),
    }
    ui.metric_row(metrics, cols_per_row=6)

    st.caption(
        f"Manifest: `{manifest_path}` — generated "
        f"`{manifest.get('generated', '?')}` from "
        f"`{manifest.get('build_root', '?')}`"
    )

    # Manual baseline coverage (does not depend on manifest)
    ui.section_header("Grounding corpus baseline (canonical)",
                      "Known coverage independent of build artifacts.")
    baseline = {
        "unique analytes": "267",
        "grounding spectra": "440",
        "core datasets": "5",
        "analyte-level MSS": "257",
        "BSV axes": "11",
    }
    ui.metric_row(baseline, cols_per_row=5)


def _principles() -> None:
    ui.section_header("Core principles",
                      "Grounding rules baked into every GAIRA output.")
    cols = st.columns(3)
    principles = [
        ("Evidence-first interpretation",
         "Every claim is tied to anchor + companion + competitor evidence — "
         "not to one peak."),
        ("Biochemical themes > exact molecules",
         "GAIRA prefers family-level / motif-level claims unless the context "
         "is pure and grounded."),
        ("Domain & context aware",
         "Serum, EV, plasma, tissue, OTC drug, and pure powder each get "
         "their own observation rules."),
        ("Reproducibility-aware",
         "GroupKFold, in-fold ΔBSV references, leakage audits — methodology "
         "is part of the deliverable."),
        ("Candidate-level MSS in biofluids",
         "In complex matrices, MSS hits are evidence — never identity claims."),
        ("No overclaiming",
         "Confidence tiers (HIGH / CANDIDATE / NOT_DETECTED / NOT_RUN) are "
         "explicit on every output."),
    ]
    for i, (title, body) in enumerate(principles):
        with cols[i % 3]:
            ui.card(title, f"<div style='font-size:0.88rem;'>{body}</div>")


def _roadmap(roadmap: list[dict]) -> None:
    ui.section_header("Demo roadmap (coming next)",
                      "Tabs scheduled after Tab 1 + Tab 2.")
    cols = st.columns(min(3, max(1, len(roadmap))))
    for i, item in enumerate(roadmap):
        with cols[i % len(cols)]:
            ui.card(
                title=f"🔜 {item['title']}",
                subtitle=item.get("blurb", ""),
                body_md="<div style='font-size:0.82rem; color:#6e7681;'>not yet implemented</div>",
                disabled=True,
            )


# ─── public entry ───────────────────────────────────────────────────────────

def render(app_cfg: dict, evidence_layers_cfg: dict, manifest: dict,
           manifest_path: Path) -> None:
    _hero(app_cfg)
    ui.divider()

    _pipeline(evidence_layers_cfg.get("pipeline_steps", []))
    ui.divider()

    _evidence_cards(evidence_layers_cfg.get("layers", []))
    ui.divider()

    _coverage_summary(manifest, manifest_path)
    ui.divider()

    _principles()
    ui.divider()

    _roadmap(app_cfg.get("roadmap", []))
