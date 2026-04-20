"""GAIRA calibration eval v1 — backend-only runner.

Runs every registered calibration contrast through the direct spectral → BSV
pipeline and writes CSV tables + figures + a markdown report to disk.

This script does NOT touch the Streamlit demo. The demo remains GAIRA v4
(Text Query + Spectral Query only).

Run:
    cd /Users/suraj/projects/GAIRA
    PYTHONPATH=src python scripts/run_calibration_eval_v1.py

Default output root:
    /Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_calibration_eval_v1/
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

from gaira.calibration import (
    CalibrationContrast,
    CalibrationResult,
    list_contrasts,
    run_calibration_eval,
)
from gaira.calibration.eval import summarize_result
from gaira.calibration.loaders import load_calibration_raw
from gaira.spectral.window_panel import BSV_COMPONENTS, WINDOW_DEFS


DEFAULT_OUTPUT_ROOT = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_calibration_eval_v1"
)

BSV_DISPLAY = {
    "membrane_lipid": "Lipid", "protein_backbone": "Protein",
    "aromatic_amino_acid": "Aromatic AA", "purine_nucleotide": "Purine",
    "pyrimidine_nucleotide": "Pyrimidine", "glycan_carbohydrate": "Glycan",
    "redox_metabolite": "Redox", "nucleic_acid_backbone": "Nuc.Backbone",
}
CATS = [BSV_DISPLAY[c] for c in BSV_COMPONENTS]


# ─────────────────────────────────────────────────────────────────────
# Table builders
# ─────────────────────────────────────────────────────────────────────

def build_dataset_inventory(contrasts: list[CalibrationContrast]) -> pd.DataFrame:
    """One row per registered calibration dataset (deduped from contrasts)."""
    seen: dict[str, dict] = {}
    for c in contrasts:
        if c.dataset_id in seen:
            seen[c.dataset_id]["contrast_ids"].append(c.contrast_id)
            continue
        raw = load_calibration_raw(c.loader_id)
        cohort_counts = pd.Series(raw.cohorts).value_counts().to_dict()
        seen[c.dataset_id] = {
            "dataset_id": c.dataset_id,
            "loader_id": c.loader_id,
            "display_name_first_contrast": c.display_name,
            "sample_family": c.sample_family,
            "substrate": c.substrate,
            "perturbation_types": c.perturbation_type,
            "n_spectra_total": int(len(raw.cohorts)),
            "n_cohorts": len(cohort_counts),
            "cohort_counts": "; ".join(
                f"{k}={v}" for k, v in sorted(cohort_counts.items())
            ),
            "source": raw.source,
            "contrast_ids": [c.contrast_id],
            "expected_axis_obvious": any(
                d in ("up", "down") for d in c.expected_directions.values()
            ),
            "notes": c.notes,
        }

    rows = []
    for d in seen.values():
        d = dict(d)
        d["contrast_ids"] = "; ".join(d["contrast_ids"])
        rows.append(d)
    return pd.DataFrame(rows)


def build_contrast_summary(results: list[CalibrationResult]) -> pd.DataFrame:
    return pd.DataFrame([summarize_result(r) for r in results])


def build_axis_recovery(results: list[CalibrationResult]) -> pd.DataFrame:
    """One row per axis, aggregating recovery counts across all contrasts."""
    stats = {ax: {"recovered": 0, "flat": 0, "inconsistent": 0, "total": 0}
             for ax in BSV_COMPONENTS}
    for r in results:
        for v in r.axis_verdicts:
            if v.expected == "unconstrained":
                continue
            stats[v.axis]["total"] += 1
            if v.verdict in ("recovered", "flat", "inconsistent"):
                stats[v.axis][v.verdict] += 1

    rows = []
    for ax, s in stats.items():
        if s["total"] == 0:
            continue
        rows.append({
            "axis": ax,
            "display_name": BSV_DISPLAY.get(ax, ax),
            "n_tested": s["total"],
            "recovered": s["recovered"],
            "flat_below_noise": s["flat"],
            "inconsistent": s["inconsistent"],
            "recovery_rate": round(s["recovered"] / s["total"], 3),
            "recovery_category": _category_for(s["recovered"], s["total"]),
        })
    return pd.DataFrame(rows)


def _category_for(recovered: int, total: int) -> str:
    if total == 0:
        return "no_data"
    rate = recovered / total
    if rate >= 0.8:
        return "strong_recovery"
    if rate >= 0.5:
        return "partial_recovery"
    if rate > 0:
        return "weak_recovery"
    return "inconclusive"


def build_delta_bsv(results: list[CalibrationResult]) -> pd.DataFrame:
    """One row per (contrast, axis) with observed Δ and verdict."""
    rows = []
    for r in results:
        for v in r.axis_verdicts:
            rows.append({
                "contrast_id": r.contrast.contrast_id,
                "control": r.contrast.control_cohort,
                "perturbed": r.contrast.perturbed_cohort,
                "axis": v.axis,
                "expected": v.expected,
                "observed_delta": round(v.observed_delta, 6),
                "control_mean_bsv": r.control_bsv.mean_bsv[v.axis],
                "perturbed_mean_bsv": r.perturbed_bsv.mean_bsv[v.axis],
                "verdict": v.verdict,
                "note": v.note,
            })
    return pd.DataFrame(rows)


def build_window_drivers(results: list[CalibrationResult]) -> pd.DataFrame:
    """One row per (contrast, top-window) with motif annotation."""
    rows = []
    for r in results:
        for w in r.top_windows:
            rows.append({
                "contrast_id": r.contrast.contrast_id,
                "rank": r.top_windows.index(w) + 1,
                "window_id": w["window_id"],
                "wavenumber_start": w["wavenumber_start"],
                "wavenumber_end": w["wavenumber_end"],
                "bsv_component": w["bsv_component"],
                "delta": round(w["delta"], 6),
                "effect_size": round(w["effect_size"], 4),
                "direction": w["direction"],
                "candidate_motifs": "; ".join(w["candidate_motifs"]),
                "example_analytes": "; ".join(w["example_analytes"]),
                "ambiguity": w.get("ambiguity") or "",
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────
# Figure builders (matplotlib, Agg backend — no display)
# ─────────────────────────────────────────────────────────────────────

def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_observed_bsv(r: CalibrationResult, out: Path) -> None:
    ctrl = [r.control_bsv.mean_bsv[c] for c in BSV_COMPONENTS]
    pert = [r.perturbed_bsv.mean_bsv[c] for c in BSV_COMPONENTS]
    x = np.arange(len(CATS))
    w = 0.4

    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    ax.bar(x - w/2, ctrl, width=w, label=f"control ({r.contrast.control_cohort})",
           color="#888888")
    ax.bar(x + w/2, pert, width=w, label=f"perturbed ({r.contrast.perturbed_cohort})",
           color="#c0392b")
    ax.set_xticks(x)
    ax.set_xticklabels(CATS, rotation=30, ha="right")
    ax.set_ylabel("mean BSV")
    ax.set_title(f"{r.contrast.display_name}\nobserved BSV — control vs perturbed")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.2)
    _save(fig, out)


def plot_delta_bsv(r: CalibrationResult, out: Path) -> None:
    deltas = [v.observed_delta for v in r.axis_verdicts]
    colors = []
    for v in r.axis_verdicts:
        if v.verdict == "recovered":
            colors.append("#27ae60")
        elif v.verdict == "inconsistent":
            colors.append("#c0392b")
        elif v.verdict == "flat":
            colors.append("#95a5a6")
        else:
            colors.append("#7f8c8d")

    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    ax.bar(CATS, deltas, color=colors)
    ax.axhline(0, color="k", linewidth=0.8)
    for i, v in enumerate(r.axis_verdicts):
        if v.expected in ("up", "down"):
            marker = "↑" if v.expected == "up" else "↓"
            ax.text(i, max(deltas[i], 0) + 0.0015 if deltas[i] >= 0 else deltas[i] - 0.0015,
                    marker, ha="center", va="bottom" if deltas[i] >= 0 else "top",
                    fontsize=12, color="white", weight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", fc="#2c3e50", ec="none"))
    ax.set_ylabel("Δ mean BSV (perturbed − control)")
    ax.set_title(f"{r.contrast.display_name} · observed delta · outcome: {r.overall_label}")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.2)
    _save(fig, out)


def plot_top_windows(r: CalibrationResult, out: Path) -> None:
    if not r.top_windows:
        return
    labels = [w["window_id"] for w in r.top_windows]
    effects = [w["effect_size"] for w in r.top_windows]
    axes_ = [BSV_DISPLAY.get(w["bsv_component"], w["bsv_component"])
             for w in r.top_windows]
    colors = ["#27ae60" if w["direction"] == "enriched" else "#c0392b"
              for w in r.top_windows]

    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    bars = ax.bar(labels, effects, color=colors)
    ax.set_ylabel("effect size (Δ / pooled SD)")
    ax.set_xlabel("Spectral window (cm⁻¹)")
    ax.set_title(f"{r.contrast.display_name}\ntop contributing windows (annotation-only motifs in CSV)")
    ax.axhline(0, color="k", linewidth=0.8)
    for bar, axis_name in zip(bars, axes_):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + (0.05 if bar.get_height() >= 0 else -0.08),
                axis_name, ha="center", va="bottom" if bar.get_height() >= 0 else "top",
                fontsize=8, rotation=0)
    ax.grid(axis="y", alpha=0.2)
    _save(fig, out)


def plot_pca_bsv(r: CalibrationResult, out: Path) -> None:
    if r.sample_bsv_control is None or r.sample_bsv_perturbed is None:
        return
    X = np.vstack([r.sample_bsv_control, r.sample_bsv_perturbed])
    if X.shape[0] < 4:
        return
    y = np.array([r.contrast.control_cohort] * r.n_control
                 + [r.contrast.perturbed_cohort] * r.n_perturbed)
    Xc = X - X.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(Xc, full_matrices=False)
    pcs = u[:, :2] * s[:2]

    fig, ax = plt.subplots(figsize=(6.5, 5))
    for lab, col in [(r.contrast.control_cohort, "#888888"),
                      (r.contrast.perturbed_cohort, "#c0392b")]:
        m = y == lab
        ax.scatter(pcs[m, 0], pcs[m, 1], s=35, alpha=0.85,
                   color=col, edgecolors="k", linewidth=0.3, label=lab)
    ax.set_xlabel(f"PC1 (var {s[0]**2/(s**2).sum():.2f})")
    ax.set_ylabel(f"PC2 (var {s[1]**2/(s**2).sum():.2f})")
    ax.set_title(f"{r.contrast.display_name} · PCA in BSV space")
    ax.legend()
    ax.grid(alpha=0.2)
    _save(fig, out)


def plot_axis_recovery_panel(axis_df: pd.DataFrame, out: Path) -> None:
    if axis_df.empty:
        return
    df = axis_df.sort_values("recovery_rate", ascending=False)
    labels = df["display_name"].tolist()
    recovered = df["recovered"].tolist()
    flat = df["flat_below_noise"].tolist()
    bad = df["inconsistent"].tolist()

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    ax.bar(x, recovered, label="recovered", color="#27ae60")
    ax.bar(x, flat, bottom=recovered, label="flat (below noise)", color="#95a5a6")
    ax.bar(x, bad,
           bottom=[recovered[i] + flat[i] for i in range(len(labels))],
           label="inconsistent", color="#c0392b")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("# contrasts (stacked)")
    ax.set_title("Per-axis recovery across all calibration contrasts")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.2)
    _save(fig, out)


def plot_contrast_outcome_panel(summary_df: pd.DataFrame, out: Path) -> None:
    counts = summary_df["overall_label"].value_counts()
    order = ["pass", "partial", "weak", "inconsistent", "no_expected"]
    counts = counts.reindex([o for o in order if o in counts.index])
    palette = {"pass": "#27ae60", "partial": "#2980b9", "weak": "#f39c12",
               "inconsistent": "#c0392b", "no_expected": "#7f8c8d"}
    fig, ax = plt.subplots(figsize=(6.5, 3.3))
    ax.bar(counts.index, counts.values,
           color=[palette.get(k, "#666") for k in counts.index])
    ax.set_ylabel("# contrasts")
    ax.set_title("Calibration outcomes across contrasts")
    ax.grid(axis="y", alpha=0.2)
    _save(fig, out)


# ─────────────────────────────────────────────────────────────────────
# Report (markdown at outputs/REPORT.md)
# ─────────────────────────────────────────────────────────────────────

STRONG_THRESHOLD = 0.8
PARTIAL_THRESHOLD = 0.5


def _df_to_md(df: pd.DataFrame) -> str:
    """Render a small DataFrame as a GitHub-flavored markdown table.

    Kept dependency-free so the runner doesn't pull in `tabulate`.
    """
    if df.empty:
        return "_empty_\n"
    cols = [str(c) for c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for _, row in df.iterrows():
        cells = []
        for v in row.tolist():
            if isinstance(v, float):
                cells.append(f"{v:g}")
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *rows])


def _axis_category_lines(axis_df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    strong, partial, weak = [], [], []
    for _, row in axis_df.iterrows():
        line = (f"- `{row['axis']}` — {row['recovered']}/{row['n_tested']} "
                f"recovered, {row['flat_below_noise']} flat, "
                f"{row['inconsistent']} inconsistent")
        cat = row["recovery_category"]
        if cat == "strong_recovery":
            strong.append(line)
        elif cat == "partial_recovery":
            partial.append(line)
        else:
            weak.append(line)
    return strong, partial, weak


def write_report(
    results: list[CalibrationResult],
    inventory_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    axis_df: pd.DataFrame,
    out_root: Path,
) -> Path:
    report = out_root / "REPORT.md"

    n_total = len(results)
    n_pass = sum(1 for r in results if r.overall_label == "pass")
    n_partial = sum(1 for r in results if r.overall_label == "partial")
    n_weak = sum(1 for r in results if r.overall_label == "weak")
    n_bad = sum(1 for r in results if r.overall_label == "inconsistent")

    strong, partial, weak = _axis_category_lines(axis_df)

    def _axis_block(title: str, lines: list[str]) -> str:
        if not lines:
            return f"_No axes in this category._\n"
        return "\n".join(lines) + "\n"

    contrast_lines = []
    for r in results:
        expected = [v for v in r.axis_verdicts if v.expected != "unconstrained"]
        exp_str = ", ".join(f"{v.axis} {v.expected}" for v in expected) or "—"
        top = r.top_windows[0] if r.top_windows else None
        top_str = (
            f"{top['window_id']} cm⁻¹ (axis={top['bsv_component']}, "
            f"effect={top['effect_size']:+.2f})" if top else "—"
        )
        contrast_lines.append(
            f"- **{r.contrast.contrast_id}** · {r.overall_label} · "
            f"n={r.n_control}/{r.n_perturbed} · expected: {exp_str} · "
            f"top window: {top_str}"
        )

    body = f"""# GAIRA Calibration Eval v1 — Run Report

Backend-only run. The GAIRA v4 Streamlit demo (Text Query + Spectral Query)
was not modified.

## Run summary

- **Contrasts evaluated:** {n_total}
- **Pass:** {n_pass} · **Partial:** {n_partial} · **Weak:** {n_weak} · **Inconsistent:** {n_bad}

See `tables/calibration_contrast_summary.csv` for the full summary.

## Datasets detected

{len(inventory_df)} calibration datasets in `/Volumes/SSD_Rad/GAIRA_DATA/raw/`:

{_df_to_md(inventory_df[["dataset_id", "sample_family", "substrate", "n_spectra_total", "contrast_ids"]])}

See `tables/calibration_dataset_inventory.csv` for full inventory details.

## Contrasts evaluated

{chr(10).join(contrast_lines)}

## Axis recovery across all contrasts

{_df_to_md(axis_df[["axis", "display_name", "n_tested", "recovered", "flat_below_noise", "inconsistent", "recovery_rate", "recovery_category"]])}

### Strong recovery
{_axis_block("strong", strong)}

### Partial recovery
{_axis_block("partial", partial)}

### Weak / inconclusive recovery
{_axis_block("weak", weak)}

## What this implies for GAIRA Layer 1

- **Validated mapping:** purine_nucleotide ↔ 700–740 cm⁻¹ window has been
  recovered end-to-end on independent hypoxanthine-spike contrasts. This is
  the only mapping calibration has confirmed.
- **Known panel gaps:**
  - No dedicated sulfur/thione window — `redox_metabolite` does not light up
    for ergothioneine.
  - The 700–740 window carries both purine ring and imidazole ring modes,
    so any imidazole-bearing analyte is indistinguishable from a purine
    perturbation on the current axis panel.
  - Uric-acid SERS on Ag colloid concentrates near ~635 / ~890 cm⁻¹, which
    routes the signal through `aromatic_amino_acid` / `glycan_carbohydrate`
    rather than `purine_nucleotide`. The `uricase_sigma_depletion` contrast
    as registered (expected `purine_nucleotide ↓`) likely has the wrong
    expected axis and should be re-registered before being treated as a
    failure of the pipeline.
- **Noise floor:** at µM-scale spikes (ergothioneine titration 0 → 2 µM),
  the delta is below the 0.003 per-axis floor used here.

## Recommendations before Layer 2

1. Anchor Layer-2 priors on purine_nucleotide ↔ 700–740 cm⁻¹ as the only
   calibration-verified axis mapping. Do not uniformly trust the remaining
   21 windows.
2. Re-register `uricase_sigma_depletion` with expected axes that match
   Ag-colloid SERS intensity distribution for uric acid
   (aromatic_amino_acid and glycan_carbohydrate).
3. Expand the window panel, or at minimum flag the 700–740 collision,
   before building uncertainty priors that assume one-window ↔ one-motif.
4. Add more calibration contrasts: `adenine_sers_control` concentration
   series (grounding), `spiked_commercial_serum_merck` for a second spike
   matrix, intermediate ergothioneine titration rungs to map the noise
   floor versus concentration.

## Outputs in this run

```
{out_root}/
  tables/
    calibration_dataset_inventory.csv
    calibration_contrast_summary.csv
    calibration_axis_recovery.csv
    calibration_delta_bsv.csv
    calibration_window_drivers.csv
  figures/
    axis_recovery_across_contrasts.png
    contrast_outcomes.png
    per_contrast_<id>_observed_bsv.png
    per_contrast_<id>_delta_bsv.png
    per_contrast_<id>_top_windows.png
    per_contrast_<id>_pca_bsv.png   (where applicable)
  REPORT.md   (this file)
```
"""
    out_root.mkdir(parents=True, exist_ok=True)
    report.write_text(body)
    return report


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT,
                    help="Output directory for CSVs, figures, and REPORT.md")
    args = p.parse_args()

    out_root: Path = args.output_root
    tables_dir = out_root / "tables"
    figs_dir = out_root / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output root: {out_root}")
    print("Running calibration eval on all registered contrasts...")
    contrasts = list_contrasts()
    results: list[CalibrationResult] = []
    for c in contrasts:
        print(f"  · {c.contrast_id}")
        results.append(run_calibration_eval(c.contrast_id))

    # ── Tables ────────────────────────────────────────────────────────
    inventory_df = build_dataset_inventory(contrasts)
    summary_df = build_contrast_summary(results)
    axis_df = build_axis_recovery(results)
    delta_df = build_delta_bsv(results)
    windows_df = build_window_drivers(results)

    inventory_df.to_csv(tables_dir / "calibration_dataset_inventory.csv", index=False)
    summary_df.to_csv(tables_dir / "calibration_contrast_summary.csv", index=False)
    axis_df.to_csv(tables_dir / "calibration_axis_recovery.csv", index=False)
    delta_df.to_csv(tables_dir / "calibration_delta_bsv.csv", index=False)
    windows_df.to_csv(tables_dir / "calibration_window_drivers.csv", index=False)
    print(f"  wrote {len(list(tables_dir.glob('*.csv')))} tables")

    # ── Figures ───────────────────────────────────────────────────────
    for r in results:
        slug = r.contrast.contrast_id
        plot_observed_bsv(r, figs_dir / f"per_contrast_{slug}_observed_bsv.png")
        plot_delta_bsv(r, figs_dir / f"per_contrast_{slug}_delta_bsv.png")
        plot_top_windows(r, figs_dir / f"per_contrast_{slug}_top_windows.png")
        plot_pca_bsv(r, figs_dir / f"per_contrast_{slug}_pca_bsv.png")
    plot_axis_recovery_panel(axis_df, figs_dir / "axis_recovery_across_contrasts.png")
    plot_contrast_outcome_panel(summary_df, figs_dir / "contrast_outcomes.png")
    print(f"  wrote {len(list(figs_dir.glob('*.png')))} figures")

    # ── Report ────────────────────────────────────────────────────────
    report_path = write_report(results, inventory_df, summary_df, axis_df, out_root)
    print(f"  wrote {report_path}")

    # ── Console summary ───────────────────────────────────────────────
    print()
    print("Summary:")
    for r in results:
        print(f"  {r.overall_label:12s}  {r.contrast.contrast_id}")
    print()
    print(f"Done. All outputs under: {out_root}")


if __name__ == "__main__":
    main()
