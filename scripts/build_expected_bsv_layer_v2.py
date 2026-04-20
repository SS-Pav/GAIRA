"""Build the literature-grounded expected-BSV layer v2 artifacts.

Runs:
  Part A — per-axis evidence audit
  Part B — anchor-window registry (peak clustering with ambiguity flags)
  Part C — expected-delta objects (contrast-specific, JSON)
  Part D — ambiguity-aware expected comparator v2 (batch-built)

Writes:
  reports/gaira_expected_bsv_axis_audit.csv
  config/expected_bsv_anchor_windows.csv
  outputs/gaira_expected_delta_objects.json
  outputs/gaira_expected_comparators_v2.json
  docs/gaira_expected_bsv_axis_audit.md
  docs/expected_bsv_anchor_windows.md
  docs/gaira_expected_delta_objects.md

NOTHING in this script touches the spectral BSV engine or the v4 demo.

Run:
    cd /Users/suraj/projects/GAIRA
    PYTHONPATH=src python scripts/build_expected_bsv_layer_v2.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from gaira.expected.axis_audit import build_axis_audit
from gaira.expected.anchor_windows import build_anchor_window_registry
from gaira.expected.comparator_v2 import (
    build_all_expected_comparators_v2, _load_peaks,
)
from gaira.expected.delta_objects import (
    build_expected_delta_objects, to_serializable,
)


REPO = Path(__file__).resolve().parent.parent
REPORTS = REPO / "reports"
CONFIG = REPO / "config"
OUTPUTS = REPO / "outputs"
DOCS = REPO / "docs"

for d in (REPORTS, CONFIG, OUTPUTS, DOCS):
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
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *rows])


def _write_axis_audit_md(df: pd.DataFrame) -> Path:
    real = df[df["axis"] != "_ambiguous_unmapped"].copy()
    ambig = df[df["axis"] == "_ambiguous_unmapped"]

    strong = real[real["support_strength"] == "strong"]["axis"].tolist()
    moderate = real[real["support_strength"] == "moderate"]["axis"].tolist()
    sparse = real[real["support_strength"] == "sparse"]["axis"].tolist()

    body = f"""# GAIRA Expected-BSV Axis Audit (v2)

Per-axis summary of literature evidence that feeds the expected comparator.
Numbers come from `peak_assignments` (local, peak-level), `knowledge_chunks`
(diffuse prose), and `condition_differential_profile.csv` (landscape v4
contrast-explicit directionality). Calibration status is READ-ONLY — this
audit does not adjust anything based on calibration results.

## Support strength at a glance

- **Strong:** `{'`, `'.join(strong) or '—'}`
- **Moderate:** `{'`, `'.join(moderate) or '—'}`
- **Sparse:** `{'`, `'.join(sparse) or '—'}`

## Full audit

{_df_md(real[['axis', 'n_peak_rows', 'n_sources', 'n_molecules', 'share_high_conf',
               'share_medium_conf', 'anchor_hint_hit_rate',
               'n_conditions_explicit', 'n_conditions_up', 'n_conditions_down',
               'locality_score', 'support_strength', 'calibration_status']])}

## Ambiguous / unmapped pool

Peaks that could not be confidently attributed to a single BSV axis. Kept for
visibility so they are counted in the ambiguity tally rather than silently
dropped or force-assigned.

{_df_md(ambig[['n_peak_rows', 'n_sources', 'n_molecules', 'distinct_molecules_top5', 'peak_cm_min', 'peak_cm_max', 'peak_cm_median', 'notes']])}

## Field meanings

- `n_peak_rows` — rows in `peak_assignments` mapped to this axis via `axis_mapping.py`.
- `n_sources` / `n_molecules` — distinct `source_id` / `assigned_molecule` values.
- `share_*_conf` — distribution of `confidence_text` (high/medium/low) among the peaks.
- `anchor_hint_hit_rate` — fraction of peaks whose `peak_cm` lies inside a canonical
  anchor range for this axis (`AXIS_ANCHOR_HINTS`).
- `n_conditions_explicit` — conditions in `condition_differential_profile.csv` with
  `direction ∈ {{up, down}}` on this axis.
- `locality_score` — `n_peak_rows / (n_peak_rows + n_prose_chunks)`. Higher = more local, peak-level support.
- `support_strength` — `strong` if ≥40 peak rows AND ≥75% medium-or-better confidence;
  `moderate` if ≥15 peak rows; else `sparse`.
- `calibration_status` — result from `gaira_calibration_eval_v1` (downstream); not used by the audit.
"""
    p = DOCS / "gaira_expected_bsv_axis_audit.md"
    p.write_text(body)
    return p


def _write_anchor_windows_md(df: pd.DataFrame) -> Path:
    axes = sorted(df["axis"].unique())
    blocks = []

    for axis in axes:
        sub = df[df["axis"] == axis]
        anchors = sub[sub["classification"] == "anchor"]
        secondaries = sub[sub["classification"] == "secondary"]
        ambiguous = sub[sub["classification"] == "ambiguous"]
        blocks.append(f"""### `{axis}`

**Anchors** ({len(anchors)}) — ≥3 sources, ≥2 molecules, ambiguity ≤ 0.4.

{_df_md(anchors[['start_cm', 'end_cm', 'n_peak_rows', 'n_sources', 'n_molecules', 'top_molecules', 'anchor_hint_match', 'ambiguity_score']]) if len(anchors) else '_none_'}

**Secondary** ({len(secondaries)}) — ≥2 sources OR anchor-hint match with thinner evidence.

{_df_md(secondaries[['start_cm', 'end_cm', 'n_peak_rows', 'n_sources', 'n_molecules', 'top_molecules', 'anchor_hint_match', 'ambiguity_score']].head(8)) if len(secondaries) else '_none_'}

**Ambiguous** ({len(ambiguous)}) — ambiguity score > 0.4 or single-source single-molecule.

{_df_md(ambiguous[['start_cm', 'end_cm', 'n_peak_rows', 'n_sources', 'top_molecules', 'ambiguity_score']].head(4)) if len(ambiguous) else '_none_'}
""")

    n_anchor = int((df["classification"] == "anchor").sum())
    n_second = int((df["classification"] == "secondary").sum())
    n_ambig = int((df["classification"] == "ambiguous").sum())

    body = f"""# Expected-BSV Anchor Windows (v2)

Per-axis anchor / secondary / ambiguous windows clustered from
`peak_assignments.peak_cm`. Windows are honest — most raw peak clusters end
up ambiguous because the underlying literature evidence is sparse or
cross-axis.

## Totals

- **anchor** windows: **{n_anchor}**
- **secondary** windows: **{n_second}**
- **ambiguous** windows: **{n_ambig}**

## Clustering rules

- Peaks within an axis are sorted; a new cluster begins whenever two
  consecutive `peak_cm` values differ by more than **25 cm⁻¹**.
- Each cluster is padded by **±5 cm⁻¹** to avoid claiming false precision.
- Classification:
  - `anchor` — ≥ 3 distinct sources, ≥ 2 distinct molecules, ambiguity ≤ 0.4
  - `secondary` — ≥ 2 sources, OR anchor-hint match with thinner support
  - `ambiguous` — everything else, or ambiguity > 0.4

Ambiguity score = fraction of peaks inside the window range that are attributed
to a DIFFERENT BSV axis (i.e. cross-axis cm overlap).

## Per-axis detail

{chr(10).join(blocks)}
"""
    p = DOCS / "expected_bsv_anchor_windows.md"
    p.write_text(body)
    return p


def _write_delta_objects_md(deltas: list) -> Path:
    lines = ["# Expected-Delta Objects (v2)",
             "",
             "Contrast-specific, literature-grounded expected shifts. "
             "Each object carries per-axis direction + confidence + anchor "
             "windows + ambiguity notes, rather than a single averaged profile.",
             "",
             "## Schema",
             "",
             "```",
             "ExpectedDelta {",
             "  contrast_id: str",
             "  condition_a: str                     // perturbed / disease side",
             "  condition_b: str                     // reference",
             "  matrix: str                          // serum, EV, biofluid, ...",
             "  substrate_context: str               // Ag colloid, Au, plasmonic paper, ...",
             "  status: 'direct' | 'approximate' | 'unavailable'",
             "  overall_confidence: 'high' | 'moderate' | 'low' | 'none'",
             "  expected_axes: [",
             "    {",
             "      axis: str",
             "      direction: 'up' | 'down' | 'flat' | 'mixed' | 'unknown'",
             "      confidence: 'high' | 'moderate' | 'low'",
             "      anchor_windows: [[start_cm, end_cm], ...]",
             "      ambiguity_notes: [str, ...]",
             "      source_ids: [str, ...]",
             "      rationale: str",
             "    }, ...",
             "  ]",
             "  ambiguity_summary: str",
             "  rationale: str",
             "  provenance: [str, ...]",
             "}",
             "```",
             "",
             "## Status semantics",
             "",
             "- `direct` — contrast-specific landscape-v4 row exists; numeric delta "
             "vector is attached.",
             "- `approximate` — contrast inferred from analyte fingerprint "
             "(e.g. hypoxanthine peak assignments) rather than a direct contrast "
             "statement in the source literature.",
             "- `unavailable` — no literature support; downstream should not treat "
             "this as a real comparator.",
             "",
             "## Registered contrasts",
             "",
             "| contrast_id | condition_a | condition_b | matrix | status | overall_confidence | #up | #down | #mixed |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for d in deltas:
        ups = sum(1 for a in d.expected_axes if a.direction == "up")
        downs = sum(1 for a in d.expected_axes if a.direction == "down")
        mixed = sum(1 for a in d.expected_axes if a.direction == "mixed")
        lines.append(f"| {d.contrast_id} | {d.condition_a} | {d.condition_b} | "
                     f"{d.matrix} | {d.status} | {d.overall_confidence} | "
                     f"{ups} | {downs} | {mixed} |")

    lines.append("")
    lines.append("## Per-contrast detail")
    for d in deltas:
        lines.append("")
        lines.append(f"### `{d.contrast_id}`")
        lines.append("")
        lines.append(f"- **Label**: {d.condition_a} vs {d.condition_b}  ·  matrix: {d.matrix}  ·  substrate: {d.substrate_context}")
        lines.append(f"- **Status**: `{d.status}`  ·  **Overall confidence**: `{d.overall_confidence}`")
        lines.append(f"- **Rationale**: {d.rationale}")
        lines.append(f"- **Ambiguity**: {d.ambiguity_summary}")
        lines.append("")
        lines.append("| axis | direction | confidence | anchor windows | ambiguity | rationale |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for a in d.expected_axes:
            anc = "; ".join(f"{s:.0f}–{e:.0f}" for s, e in a.anchor_windows) or "—"
            amb = "; ".join(a.ambiguity_notes) if a.ambiguity_notes else "—"
            # truncate rationale and ambiguity for markdown tidiness
            if len(amb) > 120:
                amb = amb[:117] + "..."
            rat = a.rationale or "—"
            if len(rat) > 120:
                rat = rat[:117] + "..."
            lines.append(f"| {a.axis} | {a.direction} | {a.confidence} | {anc} | {amb} | {rat} |")

    p = DOCS / "gaira_expected_delta_objects.md"
    p.write_text("\n".join(lines) + "\n")
    return p


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("[A] building axis audit...")
    audit_df = build_axis_audit()
    audit_csv = REPORTS / "gaira_expected_bsv_axis_audit.csv"
    audit_df.to_csv(audit_csv, index=False)
    audit_md = _write_axis_audit_md(audit_df)
    print(f"    {audit_csv.relative_to(REPO)}")
    print(f"    {audit_md.relative_to(REPO)}")

    print("[B] building anchor-window registry...")
    anchor_df = build_anchor_window_registry()
    anchor_csv = CONFIG / "expected_bsv_anchor_windows.csv"
    anchor_df.to_csv(anchor_csv, index=False)
    anchor_md = _write_anchor_windows_md(anchor_df)
    print(f"    {anchor_csv.relative_to(REPO)}")
    print(f"    {anchor_md.relative_to(REPO)}")

    print("[C] building expected-delta objects...")
    peaks_df = _load_peaks()
    deltas = build_expected_delta_objects(anchor_df, peaks_df)
    delta_json = OUTPUTS / "gaira_expected_delta_objects.json"
    delta_json.write_text(
        json.dumps([to_serializable(d) for d in deltas], indent=2)
    )
    delta_md = _write_delta_objects_md(deltas)
    print(f"    {delta_json.relative_to(REPO)}")
    print(f"    {delta_md.relative_to(REPO)}")

    print("[D] building expected-comparator v2 objects...")
    comparators = build_all_expected_comparators_v2(anchor_df, peaks_df)
    comp_json = OUTPUTS / "gaira_expected_comparators_v2.json"
    comp_json.write_text(
        json.dumps([c.to_dict() for c in comparators], indent=2)
    )
    print(f"    {comp_json.relative_to(REPO)}")

    # ── Console summary ────────────────────────────────────────────
    print()
    real = audit_df[audit_df["axis"] != "_ambiguous_unmapped"]
    print("Axes by support strength:")
    for level in ("strong", "moderate", "sparse"):
        axes = real[real["support_strength"] == level]["axis"].tolist()
        print(f"  {level:10s}  {axes}")

    n_anchor = int((anchor_df["classification"] == "anchor").sum())
    n_second = int((anchor_df["classification"] == "secondary").sum())
    n_ambig = int((anchor_df["classification"] == "ambiguous").sum())
    print(f"Anchor windows: anchor={n_anchor}, secondary={n_second}, ambiguous={n_ambig}")

    print(f"Expected-delta objects: {len(deltas)} ({sum(1 for d in deltas if d.status=='direct')} direct, "
          f"{sum(1 for d in deltas if d.status=='approximate')} approximate)")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
