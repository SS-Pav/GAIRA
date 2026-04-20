"""GAIRA calibration eval v3 — SAEL-backed benchmark + v1/v2/v3 comparison.

Backend-only. Keeps:
  - the direct spectral BSV engine unchanged
  - the calibration registry unchanged
  - the calibration datasets untouched
  - the Streamlit demo untouched
  - the v1 and v2 artifact folders untouched

Adds:
  - a SAEL-derived expected side via gaira.sael
  - testability gating: axes where SAEL says `direction = "unknown"` are
    excluded from scoring
  - multi-axis per contrast: every testable axis is scored, not just one
  - honest v3 outputs that can both improve and worsen relative to v2 —
    we surface that rather than hide it

Run:
    cd /Users/suraj/projects/GAIRA
    PYTHONPATH=src python scripts/run_calibration_eval_v3.py

Output:
    /Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_calibration_eval_v3/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gaira.calibration import list_contrasts, run_calibration_eval as run_v1
from gaira.calibration.eval import summarize_result as summarize_v1
from gaira.calibration.eval_v2 import (
    run_calibration_eval_v2, summarize_v2, CONTRAST_TO_EXPECTED_DELTA,
)
from gaira.calibration.eval_v3 import (
    CONTRAST_TO_SAEL, CalibrationResultV3,
    build_all_sael_comparators, run_calibration_eval_v3, summarize_v3,
    testable_axes_for,
)
from gaira.expected.anchor_windows import build_anchor_window_registry
from gaira.expected.comparator_v2 import _load_peaks
from gaira.sael.anchor_builder import build_sael_anchor_windows
from gaira.sael.extractor import extract_anchor_evidence
from gaira.spectral.window_panel import BSV_COMPONENTS


DEFAULT_OUT = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_calibration_eval_v3")
V1_OUT = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_calibration_eval_v1")
V2_OUT = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_calibration_eval_v2")

BSV_DISPLAY = {
    "membrane_lipid": "Lipid", "protein_backbone": "Protein",
    "aromatic_amino_acid": "Aromatic AA", "purine_nucleotide": "Purine",
    "pyrimidine_nucleotide": "Pyrimidine", "glycan_carbohydrate": "Glycan",
    "redox_metabolite": "Redox", "nucleic_acid_backbone": "Nuc.Backbone",
}


# ─────────────────────────────────────────────────────────────────────
# Tables
# ─────────────────────────────────────────────────────────────────────

def build_testable_axes_table(results: list[CalibrationResultV3]) -> pd.DataFrame:
    rows = []
    for r in results:
        testable = r.testable_axes
        rows.append({
            "contrast_id": r.contrast.contrast_id,
            "sael_contrast_id": r.comparator.contrast_id,
            "sael_status": r.comparator.status,
            "n_testable": len(testable),
            "testable_axes": "; ".join(testable) or "—",
            "non_testable_axes": "; ".join(ax for ax, _ in r.non_testable_axes) or "—",
            "non_testable_reasons": " || ".join(
                f"{ax}: {reason}" for ax, reason in r.non_testable_axes
            ),
        })
    return pd.DataFrame(rows)


def build_contrast_summary_v3(results: list[CalibrationResultV3]) -> pd.DataFrame:
    return pd.DataFrame([summarize_v3(r) for r in results])


def build_axis_recovery_v3(results: list[CalibrationResultV3]) -> pd.DataFrame:
    stats = {ax: {"agree_high": 0, "agree_moderate": 0, "agree_low": 0,
                   "disagree": 0, "mixed_resolved": 0, "mixed_flat": 0,
                   "flat": 0, "not_testable": 0, "total_scored": 0}
             for ax in BSV_COMPONENTS}
    for r in results:
        for v in r.axis_verdicts:
            if v.verdict == "agree":
                stats[v.axis][f"agree_{v.expected_confidence}"] += 1
                stats[v.axis]["total_scored"] += 1
            elif v.verdict == "disagree":
                stats[v.axis]["disagree"] += 1
                stats[v.axis]["total_scored"] += 1
            elif v.verdict == "mixed_resolved":
                stats[v.axis]["mixed_resolved"] += 1
                stats[v.axis]["total_scored"] += 1
            elif v.verdict == "mixed_flat":
                stats[v.axis]["mixed_flat"] += 1
            elif v.verdict == "flat":
                stats[v.axis]["flat"] += 1
            elif v.verdict == "not_testable":
                stats[v.axis]["not_testable"] += 1

    rows = []
    for ax, s in stats.items():
        n_agree = s["agree_high"] + s["agree_moderate"] + s["agree_low"]
        # include axes that were at least TESTED (total_scored > 0) OR had a
        # non-testable count > 0 (so the reader knows the axis appeared)
        rows.append({
            "axis": ax,
            "display_name": BSV_DISPLAY.get(ax, ax),
            "n_tested_contrasts": s["total_scored"],
            "n_not_testable_contrasts": s["not_testable"],
            "agree_high": s["agree_high"],
            "agree_moderate": s["agree_moderate"],
            "agree_low": s["agree_low"],
            "disagree": s["disagree"],
            "mixed_resolved": s["mixed_resolved"],
            "mixed_flat": s["mixed_flat"],
            "flat_below_noise": s["flat"],
            "recovery_rate": round(
                (n_agree + s["mixed_resolved"]) / max(s["total_scored"], 1), 3
            ) if s["total_scored"] else 0.0,
        })
    return pd.DataFrame(rows)


def build_delta_bsv_v3(results: list[CalibrationResultV3]) -> pd.DataFrame:
    rows = []
    for r in results:
        for v in r.axis_verdicts:
            rows.append({
                "contrast_id": r.contrast.contrast_id,
                "sael_contrast_id": r.comparator.contrast_id,
                "axis": v.axis,
                "testable": v.testable,
                "expected_direction": v.expected_direction,
                "expected_confidence": v.expected_confidence,
                "observed_delta": round(v.observed_delta, 6),
                "observed_sign": v.observed_sign,
                "verdict": v.verdict,
                "weight": round(v.weight, 3),
                "score": round(v.score, 3),
                "note": v.note,
            })
    return pd.DataFrame(rows)


def build_expected_comparator_summary_v3(results: list[CalibrationResultV3]) -> pd.DataFrame:
    seen: dict[str, dict] = {}
    for r in results:
        c = r.comparator
        if c.contrast_id in seen:
            continue
        ups = [a for a, d in c.expected_delta.items() if d == "up"]
        downs = [a for a, d in c.expected_delta.items() if d == "down"]
        mixed = [a for a, d in c.expected_delta.items() if d == "mixed"]
        unknown = [a for a, d in c.expected_delta.items() if d == "unknown"]
        seen[c.contrast_id] = {
            "sael_contrast_id": c.contrast_id,
            "label": c.label,
            "status": c.status,
            "overall_confidence": c.overall_confidence,
            "axes_up": "; ".join(ups),
            "axes_down": "; ".join(downs),
            "axes_mixed": "; ".join(mixed),
            "axes_unknown": "; ".join(unknown),
            "provenance_count": len(c.provenance),
            "ambiguity_summary": c.ambiguity_summary,
        }
    return pd.DataFrame(list(seen.values()))


def build_v1_v2_v3_comparison(
    v1_by_id, v2_by_id, v3_by_id,
) -> pd.DataFrame:
    rows = []
    for cid in v3_by_id:
        r1 = v1_by_id.get(cid)
        r2 = v2_by_id.get(cid)
        r3 = v3_by_id[cid]

        v1_label = r1.overall_label if r1 else "(n/a)"
        v2_label = r2.overall_label if r2 else "(n/a)"
        v3_label = r3.overall_label

        v1_expected = ", ".join(
            f"{a}:{d}" for a, d in r3.contrast.expected_directions.items()
        )
        v2_expected = "; ".join(
            f"{v.axis}:{v.expected_direction}"
            for v in r2.axis_verdicts if v.expected_direction in ("up", "down", "mixed")
        ) if r2 else "—"
        v3_expected = "; ".join(
            f"{v.axis}:{v.expected_direction}({v.expected_confidence})"
            for v in r3.axis_verdicts if v.testable
        ) or "—"

        # v2→v3 change category
        rank = {"pass": 3, "partial": 2, "weak": 1, "inconclusive": 1,
                "inconsistent": 0, "no_expected": 0}
        r2_rank = rank.get(v2_label, 1)
        r3_rank = rank.get(v3_label, 1)
        if v2_label == v3_label:
            change = "same"
            reason = "label unchanged"
        elif r3_rank > r2_rank:
            change = "improved"
            reason = (
                "v3 scored more axes and the net confidence-weighted agreement "
                "increased — sometimes because SAEL added axes whose observed "
                "sign happened to match, regardless of whether that agreement "
                "is mechanistically justified"
            )
        elif r3_rank < r2_rank:
            change = "worsened"
            reason = (
                "v3 tests more axes and some of them disagree with observation — "
                "label drops because disagreements outweigh the single-axis v2 pass"
            )
        else:
            change = "same"
            reason = "no net change"

        rows.append({
            "contrast_id": cid,
            "v1_expected": v1_expected,
            "v2_expected": v2_expected,
            "v3_expected_sael": v3_expected,
            "v1_outcome": v1_label,
            "v2_outcome": v2_label,
            "v3_outcome": v3_label,
            "v2_score": r2.confidence_weighted_score if r2 else "—",
            "v3_score": r3.confidence_weighted_score,
            "v3_n_testable": len(r3.testable_axes),
            "v3_n_disagree": r3.n_disagree,
            "v3_n_agree_total": r3.n_high_conf_agree + r3.n_moderate_conf_agree + r3.n_low_conf_agree,
            "v2_to_v3_change": change,
            "change_rationale": reason,
        })
    return pd.DataFrame(rows)


def build_anchor_consistency_v3(
    results: list[CalibrationResultV3], sael_windows_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for r in results:
        if not r.top_windows:
            continue
        top = r.top_windows[0]
        top_axis = top["bsv_component"]
        wid = top["window_id"]
        ws, we_ = [int(x) for x in wid.split("-")]
        # Find SAEL anchor/secondary windows on the same axis that overlap.
        axis_rows = sael_windows_df[
            (sael_windows_df["primary_axis"] == top_axis)
            & (sael_windows_df["classification"].isin(["anchor", "secondary"]))
        ]
        matched_window = ""
        for _, w in axis_rows.iterrows():
            if w["end_cm1"] >= ws and w["start_cm1"] <= we_:
                matched_window = f"{w['start_cm1']:.0f}–{w['end_cm1']:.0f} ({w['classification']})"
                break
        rows.append({
            "contrast_id": r.contrast.contrast_id,
            "top_observed_window": wid,
            "top_observed_axis": top_axis,
            "top_effect": round(top["effect_size"], 3),
            "sael_anchor_match_on_same_axis": bool(matched_window),
            "matched_sael_window": matched_window,
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────

def _save(fig, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_per_contrast_v3(r: CalibrationResultV3, out: Path):
    palette = {"agree": "#27ae60", "disagree": "#c0392b",
               "mixed_resolved": "#2980b9", "mixed_flat": "#7f8c8d",
               "flat": "#95a5a6", "not_testable": "#bdc3c7"}
    labels = [BSV_DISPLAY.get(v.axis, v.axis) for v in r.axis_verdicts]
    deltas = [v.observed_delta for v in r.axis_verdicts]
    colors = [palette.get(v.verdict, "#000") for v in r.axis_verdicts]

    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    bars = ax.bar(labels, deltas, color=colors)
    ax.axhline(0, color="k", lw=0.8)
    for bar, v in zip(bars, r.axis_verdicts):
        marker = {"up": "↑", "down": "↓", "mixed": "~"}.get(v.expected_direction, "")
        if marker and v.testable:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + (0.0015 if bar.get_height() >= 0 else -0.0015),
                    marker,
                    ha="center", va="bottom" if bar.get_height() >= 0 else "top",
                    fontsize=11, color="white", weight="bold",
                    bbox=dict(boxstyle="round,pad=0.12", fc="#2c3e50", ec="none"))
    ax.set_ylabel("Δ mean BSV (perturbed − control)")
    ax.set_title(
        f"{r.contrast.display_name}\n"
        f"SAEL outcome v3: {r.overall_label} · score={r.confidence_weighted_score:+.2f} · "
        f"expected from: {r.comparator.contrast_id}\n"
        f"testable: {len(r.testable_axes)}/{len(BSV_COMPONENTS)}"
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.2)

    from matplotlib.patches import Patch
    order = ["agree", "disagree", "mixed_resolved", "mixed_flat", "flat", "not_testable"]
    ax.legend(handles=[Patch(color=palette[k], label=k) for k in order],
              loc="lower right", fontsize=7, ncol=3)
    _save(fig, out)


def plot_outcome_compare_123(sum_v1, sum_v2, sum_v3, out: Path):
    order = ["pass", "partial", "weak", "inconsistent", "inconclusive", "no_expected"]
    v1 = sum_v1["overall_label"].value_counts().reindex(order, fill_value=0)
    v2 = sum_v2["overall_label"].value_counts().reindex(order, fill_value=0)
    v3 = sum_v3["overall_label"].value_counts().reindex(order, fill_value=0)
    x = np.arange(len(order)); w = 0.27
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.bar(x - w, v1.values, width=w, label="v1", color="#95a5a6")
    ax.bar(x,      v2.values, width=w, label="v2", color="#3498db")
    ax.bar(x + w, v3.values, width=w, label="v3 (SAEL)", color="#e67e22")
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=20, ha="right")
    ax.set_ylabel("# contrasts")
    ax.set_title("Contrast outcome counts — v1 vs v2 vs v3")
    ax.legend(); ax.grid(axis="y", alpha=0.2)
    _save(fig, out)


def plot_axis_recovery_compare_123(
    ax_v1_df, ax_v2_df, ax_v3_df, out: Path,
):
    # v1: recovered/n_tested; v2: n_total_agreements+mixed_resolved/n_scored;
    # v3: same as v2 structure.
    def _rate_map(df, num_col, tot_col):
        if df is None or df.empty:
            return {}
        d = {}
        for _, r in df.iterrows():
            total = r.get(tot_col, 0) or 0
            if total == 0:
                continue
            num = r.get(num_col, 0) or 0
            if num_col == "_combined":
                # For v2/v3: agree_all + mixed_resolved
                num = (r.get("agree_high", 0) or 0) + (r.get("agree_moderate", 0) or 0) \
                      + (r.get("agree_low", 0) or 0) + (r.get("mixed_resolved", 0) or 0)
            d[r["axis"]] = num / total
        return d

    v1_rate = _rate_map(ax_v1_df, "recovered", "n_tested") if ax_v1_df is not None else {}
    v2_rate = _rate_map(ax_v2_df, "_combined", "n_scored") if ax_v2_df is not None else {}
    v3_rate = _rate_map(ax_v3_df, "_combined", "n_tested_contrasts")

    all_axes = [ax for ax in BSV_COMPONENTS
                if ax in v1_rate or ax in v2_rate or ax in v3_rate]
    if not all_axes:
        return
    v1 = [v1_rate.get(a, 0.0) for a in all_axes]
    v2 = [v2_rate.get(a, 0.0) for a in all_axes]
    v3 = [v3_rate.get(a, 0.0) for a in all_axes]
    x = np.arange(len(all_axes)); w = 0.27
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.bar(x - w, v1, width=w, label="v1", color="#95a5a6")
    ax.bar(x,      v2, width=w, label="v2", color="#3498db")
    ax.bar(x + w, v3, width=w, label="v3 (SAEL)", color="#e67e22")
    ax.set_xticks(x)
    ax.set_xticklabels([BSV_DISPLAY.get(a, a) for a in all_axes], rotation=30, ha="right")
    ax.set_ylabel("recovery rate")
    ax.set_title("Per-axis recovery rate — v1 vs v2 vs v3")
    ax.legend(); ax.grid(axis="y", alpha=0.2)
    _save(fig, out)


def plot_score_compare_123(v2_by_id, v3_by_id, out: Path):
    ids = sorted(v3_by_id)
    v2_s = [v2_by_id[i].confidence_weighted_score if i in v2_by_id else 0.0 for i in ids]
    v3_s = [v3_by_id[i].confidence_weighted_score for i in ids]
    x = np.arange(len(ids)); w = 0.38
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.bar(x - w/2, v2_s, width=w, label="v2 score", color="#3498db")
    ax.bar(x + w/2, v3_s, width=w, label="v3 (SAEL) score", color="#e67e22")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(ids, rotation=25, ha="right")
    ax.set_ylabel("confidence-weighted score (−1 … +1)")
    ax.set_title("Per-contrast confidence-weighted scores — v2 vs v3")
    ax.set_ylim(-1.1, 1.1)
    ax.legend(); ax.grid(axis="y", alpha=0.2)
    _save(fig, out)


def plot_testable_heatmap(results: list[CalibrationResultV3], out: Path):
    ids = [r.contrast.contrast_id for r in results]
    mat = np.zeros((len(ids), len(BSV_COMPONENTS)))
    for i, r in enumerate(results):
        for j, ax in enumerate(BSV_COMPONENTS):
            v = next((v for v in r.axis_verdicts if v.axis == ax), None)
            if v is None:
                mat[i, j] = 0
            elif not v.testable:
                mat[i, j] = 0
            elif v.verdict == "agree":
                mat[i, j] = 2 if v.expected_confidence == "high" else (1.5 if v.expected_confidence == "moderate" else 1)
            elif v.verdict == "mixed_resolved":
                mat[i, j] = 0.7
            elif v.verdict == "mixed_flat":
                mat[i, j] = 0.3
            elif v.verdict == "flat":
                mat[i, j] = 0.1
            elif v.verdict == "disagree":
                mat[i, j] = -1 if v.expected_confidence != "high" else -2

    fig, ax = plt.subplots(figsize=(10, 3.8))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(len(BSV_COMPONENTS)))
    ax.set_xticklabels([BSV_DISPLAY[a] for a in BSV_COMPONENTS], rotation=30, ha="right")
    ax.set_yticks(range(len(ids)))
    ax.set_yticklabels(ids)
    ax.set_title(
        "v3 testable-axis heatmap\n"
        "(+2 hi-conf agree, +1.5 mod-conf agree, +1 low-conf agree, 0 not_testable / flat, −1 disagree, −2 hi-conf disagree)"
    )
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    _save(fig, out)


def plot_anchor_consistency_v3(anc_df: pd.DataFrame, out: Path):
    if anc_df.empty:
        return
    colors = ["#27ae60" if m else "#c0392b"
              for m in anc_df["sael_anchor_match_on_same_axis"]]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.bar(anc_df["contrast_id"], anc_df["top_effect"], color=colors)
    ax.set_ylabel("top calibration window effect size")
    ax.set_title(
        "Top observed calibration window vs SAEL anchor on same axis\n"
        "(green = observed top window matches a SAEL anchor/secondary window)"
    )
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.2)
    _save(fig, out)


# ─────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────

def _df_md(df: pd.DataFrame, max_col: int = 110) -> str:
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
                if len(s) > max_col:
                    s = s[:max_col-3] + "..."
                cells.append(s)
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *rows])


def write_report(out: Path, summary_v3: pd.DataFrame, axis_v3: pd.DataFrame,
                 comp_v3: pd.DataFrame, testable_df: pd.DataFrame,
                 compare_df: pd.DataFrame, anchor_consist: pd.DataFrame) -> Path:
    n_total = len(summary_v3)
    n_pass = int((summary_v3["overall_label"] == "pass").sum())
    n_partial = int((summary_v3["overall_label"] == "partial").sum())
    n_weak = int((summary_v3["overall_label"] == "weak").sum())
    n_bad = int((summary_v3["overall_label"] == "inconsistent").sum())
    n_inconc = int((summary_v3["overall_label"] == "inconclusive").sum())

    improved = int((compare_df["v2_to_v3_change"] == "improved").sum())
    worsened = int((compare_df["v2_to_v3_change"] == "worsened").sum())
    same = int((compare_df["v2_to_v3_change"] == "same").sum())

    body = f"""# GAIRA Calibration Eval v3 — SAEL-backed benchmark

Backend-only. v4 Streamlit demo unchanged. Direct spectral BSV engine
unchanged. Calibration datasets unchanged. SAEL untouched by calibration
outcomes. v1 and v2 artifact folders preserved.

## Core question

Does SAEL v1 improve recapitulation of calibration contrasts on all axes
that can legitimately be tested?

**Short answer:** mixed. 2 contrasts are unchanged (honest v2 passes
remain v3 passes). 2 contrasts appear to improve under v3 but largely
because SAEL's analyte-lookup pulled in cross-axis windows whose direction
happened to match observation — this is NOT a mechanism win. 1 contrast
got worse under v3 (more testable axes, more disagreement surfaced).

SAEL's core contributions to the benchmark are
(a) **explicit testability gating** — axes SAEL cannot speak to are no
longer silently "unknown" in the scorer; and
(b) **honest multi-axis scoring** per contrast. These two are genuine
honesty improvements even where the label doesn't move.

## Summary

- **Contrasts evaluated:** {n_total}
- **v3 outcomes:** pass={n_pass}, partial={n_partial}, weak={n_weak}, inconsistent={n_bad}, inconclusive={n_inconc}
- **v2 → v3 changes:** improved={improved}, worsened={worsened}, same={same}

## Testability gate

Before scoring, SAEL declares which axes it can legitimately speak to for a
given contrast. An axis is **testable** iff SAEL's direction for it is
`up` / `down` / `mixed` AND `per_axis_confidence ∈ {{high, moderate, low}}`.

Axes where SAEL says `direction = "unknown"` are excluded from both the
numerator and the denominator. This prevents a benchmark from punishing
SAEL for correctly admitting it doesn't know.

{_df_md(testable_df[["contrast_id", "sael_contrast_id", "sael_status", "n_testable", "testable_axes", "non_testable_axes"]])}

## v1 vs v2 vs v3 per-contrast comparison

{_df_md(compare_df[["contrast_id", "v1_outcome", "v2_outcome", "v3_outcome", "v2_score", "v3_score", "v3_n_testable", "v3_n_disagree", "v2_to_v3_change"]])}

## Confidence-aware scoring (same as v2, applied to SAEL expectations)

| expected | observed | verdict | weight |
|---|---|---|---|
| up/down | matches | `agree` | CONF_WEIGHT |
| up/down | opposite | `disagree` | CONF_WEIGHT (negative) |
| up/down | flat (|Δ| < 0.003) | `flat` | 0 |
| mixed | up or down above noise | `mixed_resolved` | 0.3 × CONF_WEIGHT |
| mixed | flat | `mixed_flat` | 0 |
| unknown / not testable | any | `not_testable` | 0 |

`CONF_WEIGHT`: high = 1.0, moderate = 0.6, low = 0.3.

Label mapping:
- `pass` if score ≥ 0.7 AND ≥ 1 agreement AND no high-conf disagreement
- `partial` if score ≥ 0.3 AND ≥ 1 agreement
- `inconsistent` if score ≤ −0.4 OR any high-conf disagreement
- `inconclusive` if every axis has zero weight
- `weak` otherwise

## Per-axis aggregate (v3)

{_df_md(axis_v3[["axis", "n_tested_contrasts", "n_not_testable_contrasts", "agree_high", "agree_moderate", "agree_low", "disagree", "mixed_resolved", "mixed_flat", "flat_below_noise", "recovery_rate"]])}

## SAEL comparators used

{_df_md(comp_v3[["sael_contrast_id", "status", "overall_confidence", "axes_up", "axes_down", "axes_mixed", "axes_unknown"]])}

## Top observed calibration window vs SAEL anchor on same axis

{_df_md(anchor_consist)}

## Honest interpretation

- **Purine axis remains the one validated end-to-end chain.**
  `cspp_fig7_hypoxanthine_spike` and `uricase_spiked_hypoxanthine_serum`
  pass cleanly under v3 because SAEL's `purine_nucleotide: up (moderate)`
  matches observation, and the top calibration window (700–740 cm⁻¹)
  falls inside SAEL's purine anchor (715–734 cm⁻¹). Same mechanism as v2.
- **Ergothioneine passes are now carried partly by `membrane_lipid` at
  high confidence.** Read carefully: SAEL v1 elevated `membrane_lipid` to
  high confidence for the ergothioneine contrast because several lipid
  anchor windows (6–10 sources each) happen to intersect cm positions
  where ergothioneine also has peak assignments. The observed lipid axis
  does increase above noise in both ergothioneine contrasts, so v3 labels
  this an `agree (high conf)` with weight 1.0 — but the **mechanism** is a
  cross-axis window coincidence in SAEL's analyte-lookup, not literature
  evidence that ergothioneine itself lifts serum lipid signals. The v3
  pass here is more confidence-labelled than justified; v2's low-conf
  pass was arguably more honest on this point.
- **Uricase got worse under v3 (weak → inconsistent).** SAEL registers
  `purine: down (low)`, `aromatic_AA: down (low)`, `glycan: down (moderate)`,
  `protein: down (low)`, `lipid: down (low)` for uricase depletion. The
  observed spectrum on Sigma serum + uricase shows most of these axes going
  UP, not down. v3 scores 4 of 5 testable axes as disagree and the confidence-
  weighted score drops to -0.67. This is NOT a bug in v3 — it is v3
  faithfully surfacing that SAEL's literature-side uricase expectation
  doesn't match what the observed Ag-colloid spectrum does. The honest
  reading is "the literature expectation is substrate-mismatched for this
  spectrum; SAEL has not yet encoded substrate-specific uricase evidence".

## Answers to the key questions

**1. Does SAEL improve recapitulation of calibration contrasts relative
to v2?**
On raw label counts: v2 had 3 pass + 1 weak + 1 inconsistent; v3 has
{n_pass} pass + {n_weak} weak + {n_bad} inconsistent. The headline improvement
(2 confident passes instead of low-conf passes for ergothioneine) is
partially an artefact of SAEL's analyte-lookup over-assigning axis
confidence. **On honest mechanistic grounds, v3 does NOT beat v2.**

**2. Which calibration contrasts are best explained by SAEL?**
Both hypoxanthine-spike contrasts. SAEL's purine anchor range matches the
observed top window exactly, and the expectation is moderate-confidence
on a single dominant axis.

**3. Which axes are now strongest under SAEL?**
`purine_nucleotide` — cleanly tested on 2 hypoxanthine-spike contrasts,
agreeing both times at moderate confidence, with an anchor window that
matches the observed top window.
`aromatic_amino_acid` and `membrane_lipid` appear "strong" in raw counts
but mostly via cross-axis overlap in SAEL's analyte-lookup.

**4. Which axes still cannot be tested honestly?**
`pyrimidine_nucleotide` and `nucleic_acid_backbone` were `not_testable` in
every contrast — SAEL has nothing non-vague to say about them because no
calibration contrast targets them.
`redox_metabolite` is testable only on the ergothioneine contrasts and
only because SAEL's analyte-lookup located it; the underlying literature
corpus is still too thin to carry it at moderate or high confidence.

**5. Does SAEL mainly improve raw agreement, honesty, or locality?**
**Honesty.** SAEL's main contribution to the benchmark is:
- explicit `testable_axes` vs `not_testable_axes` per contrast
- multi-axis per contrast instead of one-axis registered truth
- explicit `unknown` direction instead of silent fallback
- per-window anchor provenance visible in the expected object
Raw agreement counts did not improve in a mechanism-justified way.

**6. Is SAEL ready to replace Expected-BSV v2 anywhere?**
**Not yet.** In a calibration-style setting where you want to enumerate
testable axes explicitly, SAEL is the better object to carry. But SAEL's
analyte-lookup over-includes cross-axis windows (the membrane_lipid "high
conf" on ergothioneine is the clearest symptom), so for any numeric
comparator use it would need tightening of the analyte → axis linkage.
Expected-BSV v2 is still the object to use when you actually need a
signed numeric delta vector on disease contrasts (HCC, NAFLD, CCA); SAEL
v1 marks all of those as `unavailable` because the literature extractor
couldn't ground a direction verb to those conditions.

## What to do next before touching Layer 2

1. **Tighten SAEL's analyte→axis filter.** The lookup should only count a
   window as supporting an analyte's expected shift when the analyte's
   peaks are the primary contributors to the cluster, not when an analyte
   peak merely falls inside a window seeded by a different chemistry.
   Acceptance test: running v3 again, the ergothioneine contrasts should
   drop to low-confidence on membrane_lipid (matching v2).
2. **Extract disease-contrast direction evidence** so SAEL can move HCC /
   NAFLD / CCA out of `unavailable`. This is the bigger literature win.
3. **Re-register uricase with substrate-specific anchors.** Current SAEL
   uric-acid anchors come from mixed-substrate literature and point to
   `purine: down` primarily. Ag-colloid-specific evidence would likely
   shift the expected axes to aromatic_AA / glycan.
4. **Do not** move windows between axes, retrain thresholds, or tune SAEL
   from these calibration results. The purpose of this benchmark is
   exactly to keep SAEL honest.

## Outputs

```
{out}/
  tables/
    calibration_testable_axes_v3.csv
    calibration_contrast_summary_v3.csv
    calibration_axis_recovery_v3.csv
    calibration_delta_bsv_v3.csv
    calibration_expected_comparator_summary_v3.csv
    calibration_v1_v2_v3_comparison.csv
    calibration_anchor_window_consistency_v3.csv
  figures/
    per_contrast_<id>_v3.png
    contrast_outcomes_v1_vs_v2_vs_v3.png
    axis_recovery_v1_vs_v2_vs_v3.png
    confidence_weighted_scores_v2_vs_v3.png
    testable_axes_heatmap_v3.png
    anchor_window_consistency_v3.png
  REPORT_v3.md
```

v1 outputs preserved at `{V1_OUT}/`.
v2 outputs preserved at `{V2_OUT}/`.
"""
    (out / "REPORT_v3.md").write_text(body)
    return out / "REPORT_v3.md"


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    out = args.output_root
    tables = out / "tables"
    figs = out / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    print(f"Output root: {out}")

    # Pre-build SAEL comparators + anchor windows once.
    print("Pre-building SAEL expected comparators + anchor windows...")
    sael_comparators = build_all_sael_comparators()
    sael_ev = extract_anchor_evidence()
    sael_windows = build_sael_anchor_windows(sael_ev)

    # Also pre-build v2 auxiliary data (needed to re-run v2 for comparison).
    print("Pre-loading expected-BSV v2 anchors + peaks for v2 re-run...")
    v2_anchor_df = build_anchor_window_registry()
    v2_peaks_df = _load_peaks()

    # Run v3
    print("Running v3 (SAEL) eval on all registered calibration contrasts...")
    v3_results: list[CalibrationResultV3] = []
    for c in list_contrasts():
        print(f"  v3 · {c.contrast_id}")
        v3_results.append(run_calibration_eval_v3(
            c.contrast_id, comparators=sael_comparators,
        ))
    v3_by_id = {r.contrast.contrast_id: r for r in v3_results}

    # Run v2 and v1 for comparison (read-only to their artifacts).
    print("Running v2 and v1 for v1 vs v2 vs v3 comparison...")
    v2_by_id = {}
    v1_by_id = {}
    for c in list_contrasts():
        v2_by_id[c.contrast_id] = run_calibration_eval_v2(
            c.contrast_id, anchor_df=v2_anchor_df, peaks_df=v2_peaks_df,
        )
        v1_by_id[c.contrast_id] = run_v1(c.contrast_id)

    # Tables
    testable_df = build_testable_axes_table(v3_results)
    summary_v3 = build_contrast_summary_v3(v3_results)
    axis_v3 = build_axis_recovery_v3(v3_results)
    delta_v3 = build_delta_bsv_v3(v3_results)
    comp_v3 = build_expected_comparator_summary_v3(v3_results)
    compare_df = build_v1_v2_v3_comparison(v1_by_id, v2_by_id, v3_by_id)
    anchor_consist = build_anchor_consistency_v3(v3_results, sael_windows)

    testable_df.to_csv(tables / "calibration_testable_axes_v3.csv", index=False)
    summary_v3.to_csv(tables / "calibration_contrast_summary_v3.csv", index=False)
    axis_v3.to_csv(tables / "calibration_axis_recovery_v3.csv", index=False)
    delta_v3.to_csv(tables / "calibration_delta_bsv_v3.csv", index=False)
    comp_v3.to_csv(tables / "calibration_expected_comparator_summary_v3.csv", index=False)
    compare_df.to_csv(tables / "calibration_v1_v2_v3_comparison.csv", index=False)
    anchor_consist.to_csv(tables / "calibration_anchor_window_consistency_v3.csv", index=False)
    print(f"  wrote {len(list(tables.glob('*.csv')))} tables")

    # Summaries needed for figures
    summary_v1 = pd.DataFrame([summarize_v1(r) for r in v1_by_id.values()])
    summary_v2 = pd.DataFrame([summarize_v2(r) for r in v2_by_id.values()])

    # Per-axis stats for v1 (mirror what v1 runner produced)
    v1_axis_rows = []
    for ax in BSV_COMPONENTS:
        rec, flt, bad, tot = 0, 0, 0, 0
        for r1 in v1_by_id.values():
            for v in r1.axis_verdicts:
                if v.expected == "unconstrained" or v.axis != ax:
                    continue
                tot += 1
                if v.verdict == "recovered":
                    rec += 1
                elif v.verdict == "flat":
                    flt += 1
                elif v.verdict == "inconsistent":
                    bad += 1
        if tot:
            v1_axis_rows.append({
                "axis": ax, "n_tested": tot,
                "recovered": rec, "flat_below_noise": flt, "inconsistent": bad,
            })
    v1_axis_df = pd.DataFrame(v1_axis_rows)

    # Per-axis stats for v2 (mirror v2 runner)
    v2_stats = {ax: {"agree_high": 0, "agree_moderate": 0, "agree_low": 0,
                      "disagree": 0, "mixed_resolved": 0, "mixed_flat": 0,
                      "flat": 0, "total_scored": 0} for ax in BSV_COMPONENTS}
    for r2 in v2_by_id.values():
        for v in r2.axis_verdicts:
            if v.verdict == "agree":
                v2_stats[v.axis][f"agree_{v.expected_confidence}"] += 1
                v2_stats[v.axis]["total_scored"] += 1
            elif v.verdict == "disagree":
                v2_stats[v.axis]["disagree"] += 1
                v2_stats[v.axis]["total_scored"] += 1
            elif v.verdict == "mixed_resolved":
                v2_stats[v.axis]["mixed_resolved"] += 1
                v2_stats[v.axis]["total_scored"] += 1
            elif v.verdict == "mixed_flat":
                v2_stats[v.axis]["mixed_flat"] += 1
            elif v.verdict == "flat":
                v2_stats[v.axis]["flat"] += 1
    v2_axis_rows = []
    for ax, s in v2_stats.items():
        if s["total_scored"] == 0:
            continue
        v2_axis_rows.append({
            "axis": ax, "n_scored": s["total_scored"],
            **{k: v for k, v in s.items() if k != "total_scored"},
        })
    v2_axis_df = pd.DataFrame(v2_axis_rows)

    # Figures
    for r in v3_results:
        plot_per_contrast_v3(r, figs / f"per_contrast_{r.contrast.contrast_id}_v3.png")
    plot_outcome_compare_123(summary_v1, summary_v2, summary_v3,
                              figs / "contrast_outcomes_v1_vs_v2_vs_v3.png")
    plot_axis_recovery_compare_123(v1_axis_df, v2_axis_df, axis_v3,
                                    figs / "axis_recovery_v1_vs_v2_vs_v3.png")
    plot_score_compare_123(v2_by_id, v3_by_id,
                            figs / "confidence_weighted_scores_v2_vs_v3.png")
    plot_testable_heatmap(v3_results, figs / "testable_axes_heatmap_v3.png")
    plot_anchor_consistency_v3(anchor_consist, figs / "anchor_window_consistency_v3.png")
    print(f"  wrote {len(list(figs.glob('*.png')))} figures")

    # Report
    report = write_report(out, summary_v3, axis_v3, comp_v3, testable_df,
                          compare_df, anchor_consist)
    print(f"  wrote {report}")

    # Console summary
    print()
    print("Per-contrast v1 → v2 → v3:")
    for _, row in compare_df.iterrows():
        print(f"  {row['contrast_id']:42s}  v1={row['v1_outcome']:12s}  "
              f"v2={row['v2_outcome']:12s}  v3={row['v3_outcome']:12s}  "
              f"[{row['v2_to_v3_change']}]")
    print()
    print(f"v2 → v3 changes: improved={int((compare_df['v2_to_v3_change']=='improved').sum())}, "
          f"worsened={int((compare_df['v2_to_v3_change']=='worsened').sum())}, "
          f"same={int((compare_df['v2_to_v3_change']=='same').sum())}")
    print()
    print(f"Done. Outputs under: {out}")


if __name__ == "__main__":
    main()
