"""Build the Spectral Anchor Evidence Layer (SAEL v1) artifacts.

Backend-only. Does NOT touch the Streamlit demo or the direct spectral BSV
engine. Does NOT use calibration datasets.

Run:
    cd /Users/suraj/projects/GAIRA
    PYTHONPATH=src python scripts/build_sael_v1.py

Writes:
    outputs/gaira_spectral_anchor_evidence_raw.csv
    config/spectral_anchor_windows_v1.csv
    outputs/gaira_expected_delta_anchor_v1.json
    outputs/gaira_expected_comparators_anchor_v1.json
    docs/spectral_anchor_windows_v1.md
    docs/gaira_expected_delta_anchor_v1.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from gaira.sael.anchor_builder import build_sael_anchor_windows
from gaira.sael.bsv_derivation import derive_expected_comparators
from gaira.sael.delta_builder import (
    build_sael_expected_deltas, to_serializable,
)
from gaira.sael.extractor import extract_anchor_evidence


REPO = Path(__file__).resolve().parent.parent
OUTPUTS = REPO / "outputs"
CONFIG = REPO / "config"
DOCS = REPO / "docs"
for d in (OUTPUTS, CONFIG, DOCS):
    d.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────
# Markdown writers
# ─────────────────────────────────────────────────────────────────────

def _df_md(df: pd.DataFrame) -> str:
    if df.empty:
        return "_empty_"
    cols = [str(c) for c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for _, r in df.iterrows():
        cells = []
        for v in r.tolist():
            if isinstance(v, float):
                cells.append(f"{v:g}")
            else:
                s = str(v)
                if len(s) > 80:
                    s = s[:77] + "..."
                cells.append(s)
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *rows])


def _write_anchor_windows_md(windows_df: pd.DataFrame) -> Path:
    by_axis = windows_df.groupby("primary_axis", dropna=False)
    blocks = []
    for axis, sub in by_axis:
        anchors = sub[sub["classification"] == "anchor"]
        secondary = sub[sub["classification"] == "secondary"]
        ambiguous = sub[sub["classification"] == "ambiguous"]
        blocks.append(f"""### `{axis or '_unassigned'}`

- **anchors**: {len(anchors)}
- **secondary**: {len(secondary)}
- **ambiguous**: {len(ambiguous)}

**Anchors**

{_df_md(anchors[['window_id', 'start_cm1', 'end_cm1', 'source_count', 'condition_count', 'direction_distribution', 'ambiguity_score', 'priority_tags']]) if len(anchors) else '_none_'}

**Secondary** (top 5)

{_df_md(secondary[['window_id', 'start_cm1', 'end_cm1', 'source_count', 'direction_distribution', 'ambiguity_score']].head(5)) if len(secondary) else '_none_'}
""")

    n_anchor = int((windows_df["classification"] == "anchor").sum())
    n_second = int((windows_df["classification"] == "secondary").sum())
    n_ambig = int((windows_df["classification"] == "ambiguous").sum())

    body = f"""# SAEL v1 — Anchor Window Registry

Built from the raw SAEL anchor evidence extraction (see
`outputs/gaira_spectral_anchor_evidence_raw.csv`).

## Classification rules

- **anchor** — ≥ 2 distinct sources, ambiguity ≤ 0.4, AND either a direction
  claim or a multi-source assignment supporting the window
- **secondary** — ≥ 1 source and matches a canonical axis-hint range, OR
  multi-source assignment without direction
- **ambiguous** — ambiguity > 0.4, conflicting up/down from multiple sources,
  or insufficient support

Ambiguity score = fraction of peaks inside the window that are attributed to a
different BSV axis (cross-axis overlap).

## Totals

- **anchor**: **{n_anchor}**
- **secondary**: **{n_second}**
- **ambiguous**: **{n_ambig}**

## How this differs from the previous expected-BSV v2 registry

SAEL v1 windows carry **per-window contrast metadata** — direction_distribution,
matrix_distribution, substrate_distribution, condition_count — that the prior
expected-BSV v2 registry did not. In practice, only 3 of the current
extraction rows carry a direction verb, so most direction_distribution cells
are empty. This is honest: the underlying corpus has essentially no
explicit contrast-direction sentences; SAEL v1 surfaces the gap rather than
imputing direction.

## Per-axis detail

{chr(10).join(blocks)}
"""
    p = DOCS / "spectral_anchor_windows_v1.md"
    p.write_text(body)
    return p


def _write_expected_delta_md(deltas: list) -> Path:
    lines = [
        "# SAEL v1 — Anchor-Based Expected-Delta Objects",
        "",
        "Expected biochemical shifts derived **from anchor windows**, not from",
        "broad text averages. Each contrast spec resolves to a SAEL expected-delta",
        "object with per-axis direction, per-axis confidence, and anchor windows",
        "used as support.",
        "",
        "## Modes",
        "",
        "- `analyte_based` — perturbation is a known molecule (spike / depletion).",
        "  SAEL uses assignment-level anchors to LOCATE the analyte's peaks; the",
        "  direction comes from the spec (spike → up, depletion → down). This is",
        "  literature-grounded location + explicit spec direction, never a text-",
        "  inferred contrast prose claim.",
        "- `condition_based` — disease vs reference (e.g. HCC vs healthy serum).",
        "  Requires SAEL contrast-type rows that carry both a matching condition",
        "  and a direction verb. If the corpus has none, SAEL reports",
        "  `status = unavailable` rather than inventing direction.",
        "",
        "## Registered contrasts",
        "",
        "| contrast_id | mode | condition_a | matrix | status | overall_confidence | # per_axis entries | provenance_count |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for d in deltas:
        mode = "analyte_based" if d.contrast_id.endswith("_literature") else "condition_based"
        lines.append(
            f"| {d.contrast_id} | {mode} | {d.condition_a} | {d.matrix} | "
            f"{d.status} | {d.overall_confidence} | {len(d.per_axis)} | "
            f"{len(d.provenance)} |"
        )

    lines.append("")
    lines.append("## Per-contrast detail")
    for d in deltas:
        lines.append("")
        lines.append(f"### `{d.contrast_id}`")
        lines.append("")
        lines.append(f"- **Condition A**: `{d.condition_a}` · **Condition B**: `{d.condition_b}`")
        lines.append(f"- **Matrix**: {d.matrix} · **Substrate**: {d.substrate_context or '—'}")
        lines.append(f"- **Status**: `{d.status}` · **Overall confidence**: `{d.overall_confidence}`")
        lines.append(f"- **Rationale**: {d.rationale}")
        lines.append(f"- **Ambiguity summary**: {d.ambiguity_summary}")
        lines.append(f"- **Anchor windows used**: " + (", ".join(d.anchor_windows_used) if d.anchor_windows_used else "_none_"))
        if d.per_axis:
            lines.append("")
            lines.append("| axis | direction | confidence | supporting windows | ambiguity notes |")
            lines.append("| --- | --- | --- | --- | --- |")
            for a in d.per_axis:
                supp = ", ".join(a.supporting_windows) or "—"
                amb = "; ".join(a.ambiguity_notes) or "—"
                if len(amb) > 100:
                    amb = amb[:97] + "..."
                if len(supp) > 100:
                    supp = supp[:97] + "..."
                lines.append(f"| {a.axis} | {a.direction} | {a.confidence} | {supp} | {amb} |")
        else:
            lines.append("")
            lines.append("_No per-axis entries (no anchor windows matched or condition evidence absent)._")

    lines.append("")
    lines.append("## How SAEL deltas differ from expected-BSV v2 deltas")
    lines.append("")
    lines.append("- **Source of direction**: SAEL direction comes from the contrast spec")
    lines.append("  (spike / depletion) combined with the direction verbs detected in the")
    lines.append("  SAEL evidence rows. Expected-BSV v2 pulled direction from the")
    lines.append("  `condition_differential_profile.csv` landscape aggregate, which was")
    lines.append("  itself a coarse average over a larger evidence pool.")
    lines.append("- **Anchor support is explicit**: each per-axis direction lists the")
    lines.append("  window_ids that supported it. Expected-BSV v2 attached windows at the")
    lines.append("  contrast level, not per axis.")
    lines.append("- **Ambiguity is not averaged away**: SAEL reports status =")
    lines.append("  'unavailable' when no direction-bearing rows exist. Expected-BSV v2")
    lines.append("  would still emit a delta based on landscape averages.")
    lines.append("- **Context conditioning**: SAEL filters anchor windows by declared")
    lines.append("  matrix and substrate. Expected-BSV v2 did not.")

    p = DOCS / "gaira_expected_delta_anchor_v1.md"
    p.write_text("\n".join(lines) + "\n")
    return p


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("[A] extracting local spectral anchor evidence...")
    evidence = extract_anchor_evidence()
    evidence_csv = OUTPUTS / "gaira_spectral_anchor_evidence_raw.csv"
    evidence.to_csv(evidence_csv, index=False)
    n_total = len(evidence)
    n_contrast = int((evidence["kind"] == "contrast").sum())
    n_assignment = int((evidence["kind"] == "assignment").sum())
    print(f"    {evidence_csv.relative_to(REPO)}  "
          f"(rows={n_total}, contrast={n_contrast}, assignment={n_assignment})")

    print("[B] clustering into anchor windows...")
    windows = build_sael_anchor_windows(evidence)
    windows_csv = CONFIG / "spectral_anchor_windows_v1.csv"
    windows.to_csv(windows_csv, index=False)
    windows_md = _write_anchor_windows_md(windows)
    n_anc = int((windows["classification"] == "anchor").sum())
    n_sec = int((windows["classification"] == "secondary").sum())
    n_amb = int((windows["classification"] == "ambiguous").sum())
    print(f"    {windows_csv.relative_to(REPO)}  "
          f"(anchor={n_anc}, secondary={n_sec}, ambiguous={n_amb})")
    print(f"    {windows_md.relative_to(REPO)}")

    print("[C] building anchor-based expected-delta objects...")
    deltas = build_sael_expected_deltas(evidence, windows)
    deltas_json = OUTPUTS / "gaira_expected_delta_anchor_v1.json"
    deltas_json.write_text(
        json.dumps([to_serializable(d) for d in deltas], indent=2)
    )
    deltas_md = _write_expected_delta_md(deltas)
    n_direct = sum(1 for d in deltas if d.status == "direct")
    n_approx = sum(1 for d in deltas if d.status == "approximate")
    n_weak = sum(1 for d in deltas if d.status == "weak")
    n_unav = sum(1 for d in deltas if d.status == "unavailable")
    print(f"    {deltas_json.relative_to(REPO)}  "
          f"(direct={n_direct}, approximate={n_approx}, weak={n_weak}, unavailable={n_unav})")
    print(f"    {deltas_md.relative_to(REPO)}")

    print("[D] deriving expected comparators from anchor deltas...")
    comparators = derive_expected_comparators(deltas)
    comp_json = OUTPUTS / "gaira_expected_comparators_anchor_v1.json"
    comp_json.write_text(json.dumps([c.to_dict() for c in comparators], indent=2))
    print(f"    {comp_json.relative_to(REPO)}")

    # ── Console summary ─────────────────────────────────────────────
    print()
    print("=== SAEL v1 summary ===")
    print(f"Raw anchor evidence rows: {n_total} "
          f"({n_contrast} contrast, {n_assignment} assignment)")
    print(f"Anchor windows: anchor={n_anc}, secondary={n_sec}, ambiguous={n_amb}")
    print(f"Expected deltas: direct={n_direct}, approximate={n_approx}, "
          f"weak={n_weak}, unavailable={n_unav}")
    print()
    print("Per-contrast status:")
    for d in deltas:
        axes_nonflat = [a.axis for a in d.per_axis if a.direction in ("up", "down", "mixed")]
        axes_str = ", ".join(axes_nonflat) or "—"
        print(f"  {d.contrast_id:45s}  {d.status:12s}  axes: {axes_str}")


if __name__ == "__main__":
    main()
