"""Calibration evaluation orchestration.

Pipeline for one contrast:
  1. load_calibration_raw(loader_id)      — raw spectra + cohort labels
  2. preprocess_calibration(X, wn)        — AsLS + SG + L2
  3. filter to contrast's control + perturbed cohorts
  4. extract_window_features → project_to_bsv
  5. compute cohort-mean BSV + observed delta
  6. compute per-axis agreement against expected directions
  7. compute delta cosine if a full expected vector is available
  8. rank top contributing spectral windows + annotate

The scoring language intentionally uses:
  - "correct direction recovered" / "partial recovery"
  - "weak recovery" / "axis inconsistent with expectation"
No molecule-identity claims.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from gaira.calibration.band_annotation import annotate_top_windows
from gaira.calibration.loaders import CalibrationRaw, load_calibration_raw
from gaira.calibration.preprocessing import (
    CalibrationPreprocessed, preprocess_calibration,
)
from gaira.calibration.registry import CalibrationContrast, get_contrast
from gaira.spectral.band_drivers import compute_per_cohort_window_importance
from gaira.spectral.bsv_projection import (
    CohortBSV, compute_cohort_bsvs, project_to_bsv,
)
from gaira.spectral.window_panel import (
    BSV_COMPONENTS, extract_window_features,
)


# Magnitude threshold below which an axis change is treated as "flat" (noise).
# Spectra are L2-normalized so BSV components sum to ~1; a cohort-mean delta
# of 0.003 on a single axis is about 0.3% of the normalized signal — below
# typical between-spectrum SERS variability for fingerprint-region windows.
FLAT_DELTA_THRESHOLD = 0.003


@dataclass
class AxisVerdict:
    axis: str
    expected: str                   # "up" | "down" | "flat"
    observed_delta: float
    verdict: str                    # "recovered" | "inconsistent" | "flat" | "unconstrained"
    note: str = ""


@dataclass
class CalibrationResult:
    contrast: CalibrationContrast

    # Cohort-level spectral state
    control_bsv: CohortBSV
    perturbed_bsv: CohortBSV
    observed_delta: dict[str, float]       # perturbed - control, per BSV component

    # Scoring
    axis_verdicts: list[AxisVerdict]
    overall_label: str                      # "pass" | "partial" | "weak" | "inconsistent" | "no_expected"
    expected_axes_hit: int
    expected_axes_total: int

    # Band drivers + annotations (top windows perturbed vs control)
    top_windows: list[dict]                 # annotated (w/ candidate_motifs)

    # Bookkeeping
    n_control: int
    n_perturbed: int
    pipeline: str
    crop_range: tuple[float, float]
    notes: list[str] = field(default_factory=list)

    # Raw handles for downstream plotting (Streamlit)
    window_features_control: np.ndarray | None = None
    window_features_perturbed: np.ndarray | None = None
    sample_bsv_control: np.ndarray | None = None
    sample_bsv_perturbed: np.ndarray | None = None


def _score_axis(expected: str, delta: float) -> tuple[str, str]:
    """Return (verdict, note) given expected direction and observed delta."""
    mag = abs(delta)
    if expected == "flat":
        if mag < FLAT_DELTA_THRESHOLD:
            return "recovered", "change within noise floor, as expected"
        return "inconsistent", f"expected flat, observed |Δ|={mag:.3f}"

    if mag < FLAT_DELTA_THRESHOLD:
        return "flat", f"|Δ|={mag:.3f} below {FLAT_DELTA_THRESHOLD:.3f} noise floor"

    sign_ok = (delta > 0 and expected == "up") or (delta < 0 and expected == "down")
    if sign_ok:
        return "recovered", f"sign matches expected {expected}"
    return "inconsistent", f"expected {expected}, observed Δ={delta:+.3f}"


def _overall_label(verdicts: list[AxisVerdict]) -> str:
    expected_verdicts = [v for v in verdicts if v.expected != "unconstrained"]
    if not expected_verdicts:
        return "no_expected"

    recovered = sum(1 for v in expected_verdicts if v.verdict == "recovered")
    inconsistent = sum(1 for v in expected_verdicts if v.verdict == "inconsistent")
    total = len(expected_verdicts)

    if recovered == total:
        return "pass"
    if inconsistent == total:
        return "inconsistent"
    if recovered >= 1 and inconsistent == 0:
        return "partial"
    if recovered >= 1:
        return "partial"  # mix, but at least one recovered
    return "weak"


def run_calibration_eval(
    contrast_id: str,
    raw: CalibrationRaw | None = None,
    preprocessed: CalibrationPreprocessed | None = None,
) -> CalibrationResult:
    """Run the full calibration eval for one contrast.

    Optional `raw` / `preprocessed` lets a caller reuse cached computation
    across contrasts that share a loader (e.g. CSPP Fig 7 contrasts share
    the same raw CSV).
    """
    contrast = get_contrast(contrast_id)

    if raw is None:
        raw = load_calibration_raw(contrast.loader_id)

    # Restrict to the two relevant cohorts
    want = {contrast.control_cohort, contrast.perturbed_cohort}
    mask = np.isin(raw.cohorts, list(want))
    if mask.sum() == 0:
        raise ValueError(
            f"Contrast '{contrast_id}': no spectra match cohorts {want}. "
            f"Available cohorts: {sorted(set(raw.cohorts))}"
        )

    X_sub = raw.X[mask]
    cohorts_sub = raw.cohorts[mask]

    if preprocessed is None:
        preprocessed = preprocess_calibration(X_sub, raw.wavenumbers)
    else:
        # If a cached preprocessed was passed, it was computed on the full raw.
        preprocessed = CalibrationPreprocessed(
            X=preprocessed.X[mask],
            wavenumbers=preprocessed.wavenumbers,
            pipeline=preprocessed.pipeline,
            crop_range=preprocessed.crop_range,
        )

    # Window features + BSV
    window_features = extract_window_features(preprocessed.X, preprocessed.wavenumbers)
    bsv_matrix = project_to_bsv(window_features)
    cohort_bsvs = compute_cohort_bsvs(bsv_matrix, cohorts_sub)

    if contrast.control_cohort not in cohort_bsvs or contrast.perturbed_cohort not in cohort_bsvs:
        raise ValueError(
            f"Contrast '{contrast_id}': cohort BSVs missing after masking. "
            f"Got {list(cohort_bsvs)}."
        )

    ctrl = cohort_bsvs[contrast.control_cohort]
    pert = cohort_bsvs[contrast.perturbed_cohort]
    observed_delta = {
        comp: round(pert.mean_bsv[comp] - ctrl.mean_bsv[comp], 6)
        for comp in BSV_COMPONENTS
    }

    # Score each axis
    verdicts: list[AxisVerdict] = []
    for comp in BSV_COMPONENTS:
        expected = contrast.expected_directions.get(comp, "unconstrained")
        delta = observed_delta[comp]
        if expected == "unconstrained":
            verdicts.append(AxisVerdict(
                axis=comp, expected="unconstrained",
                observed_delta=delta, verdict="unconstrained",
                note="no expectation registered for this axis",
            ))
        else:
            v, note = _score_axis(expected, delta)
            verdicts.append(AxisVerdict(
                axis=comp, expected=expected, observed_delta=delta,
                verdict=v, note=note,
            ))

    hit = sum(1 for v in verdicts
              if v.expected not in ("unconstrained", "flat") and v.verdict == "recovered")
    # For "flat" expectation, "recovered" also counts as a hit; fold it in:
    hit += sum(1 for v in verdicts
               if v.expected == "flat" and v.verdict == "recovered")
    total = sum(1 for v in verdicts if v.expected != "unconstrained")

    # Band drivers + annotations
    band_imp = compute_per_cohort_window_importance(
        window_features, cohorts_sub, reference=contrast.control_cohort,
    )
    top = band_imp.get(contrast.perturbed_cohort, [])
    top_annotated = annotate_top_windows(top, top_n=6)

    ctrl_mask = cohorts_sub == contrast.control_cohort
    pert_mask = cohorts_sub == contrast.perturbed_cohort

    notes: list[str] = []
    if contrast.confound_axes:
        notes.append(
            "Confound axes flagged in registry: "
            + ", ".join(contrast.confound_axes)
            + " (signal may leak across these windows)."
        )
    if contrast.notes:
        notes.append(contrast.notes)

    return CalibrationResult(
        contrast=contrast,
        control_bsv=ctrl,
        perturbed_bsv=pert,
        observed_delta=observed_delta,
        axis_verdicts=verdicts,
        overall_label=_overall_label(verdicts),
        expected_axes_hit=hit,
        expected_axes_total=total,
        top_windows=top_annotated,
        n_control=int(ctrl_mask.sum()),
        n_perturbed=int(pert_mask.sum()),
        pipeline=preprocessed.pipeline,
        crop_range=preprocessed.crop_range,
        notes=notes,
        window_features_control=window_features[ctrl_mask],
        window_features_perturbed=window_features[pert_mask],
        sample_bsv_control=bsv_matrix[ctrl_mask],
        sample_bsv_perturbed=bsv_matrix[pert_mask],
    )


def summarize_result(r: CalibrationResult) -> dict:
    """Flat summary row for a results table."""
    return {
        "contrast_id": r.contrast.contrast_id,
        "display_name": r.contrast.display_name,
        "dataset_id": r.contrast.dataset_id,
        "perturbation_type": r.contrast.perturbation_type,
        "control": r.contrast.control_cohort,
        "perturbed": r.contrast.perturbed_cohort,
        "n_control": r.n_control,
        "n_perturbed": r.n_perturbed,
        "overall_label": r.overall_label,
        "axes_hit": f"{r.expected_axes_hit}/{r.expected_axes_total}",
        "pipeline": r.pipeline,
    }
