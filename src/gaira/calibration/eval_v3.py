"""Calibration eval v3 — evaluates contrasts against SAEL expected comparators.

Key differences from eval_v2.py:
  - Expected side = SAELExpectedComparator (anchor-based, per-axis direction
    with per-axis confidence, from gaira.sael). Not landscape-level averages.
  - Testability gating: axes where SAEL says `direction = "unknown"` are
    EXCLUDED from scoring (not counted as neither agreement nor
    disagreement; not in the denominator). The calibration contrast's
    outcome is computed only over the axes SAEL declares testable.
  - Multi-axis per contrast. SAEL typically registers several axes per
    contrast (each with its own direction + confidence); v3 scores every
    testable axis, not just a single registered axis.

Observed side is UNCHANGED: same direct spectral BSV pipeline, same
preprocessing, same cohort slicing as eval_v1 / eval_v2.

Nothing in this module trains SAEL or adjusts its anchors from calibration
outcomes. Calibration datasets remain strictly tests.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from gaira.calibration.band_annotation import annotate_top_windows
from gaira.calibration.eval import FLAT_DELTA_THRESHOLD
from gaira.calibration.loaders import CalibrationRaw, load_calibration_raw
from gaira.calibration.preprocessing import preprocess_calibration
from gaira.calibration.registry import CalibrationContrast, get_contrast
from gaira.sael.anchor_builder import build_sael_anchor_windows
from gaira.sael.bsv_derivation import (
    SAELExpectedComparator, derive_expected_comparators,
)
from gaira.sael.delta_builder import build_sael_expected_deltas
from gaira.sael.extractor import extract_anchor_evidence
from gaira.spectral.band_drivers import compute_per_cohort_window_importance
from gaira.spectral.bsv_projection import (
    CohortBSV, compute_cohort_bsvs, project_to_bsv,
)
from gaira.spectral.window_panel import (
    BSV_COMPONENTS, extract_window_features,
)


# ─────────────────────────────────────────────────────────────────────
# Calibration contrast → SAEL contrast_id
# Same coverage as eval_v2 (literature-side objects are unchanged).
# Lives here, NOT in registry.py, so the calibration registry stays
# identical and v1/v2 remain bit-exact.
# ─────────────────────────────────────────────────────────────────────

CONTRAST_TO_SAEL: dict[str, str] = {
    "cspp_fig7_hypoxanthine_spike":        "hypoxanthine_spike_literature",
    "uricase_spiked_hypoxanthine_serum":   "hypoxanthine_spike_literature",
    "cspp_fig7_ergothioneine_spike":       "ergothioneine_spike_literature",
    "ergothioneine_titration_top_vs_zero": "ergothioneine_spike_literature",
    "uricase_sigma_depletion":             "uricase_depletion_literature",
}


CONF_WEIGHT = {"high": 1.0, "moderate": 0.6, "low": 0.3}


# ─────────────────────────────────────────────────────────────────────
# Testability
# ─────────────────────────────────────────────────────────────────────

def testable_axes_for(
    comparator: SAELExpectedComparator,
) -> tuple[list[str], list[tuple[str, str]]]:
    """Return (testable_axes, non_testable_with_reason).

    Testable axis = direction ∈ {up, down, mixed} AND confidence ∈ {high, moderate, low}.
    Axes with direction = "unknown" OR "flat" are excluded — they cannot
    confirm or refute anything.
    """
    if comparator.status == "unavailable":
        return [], [(ax, "comparator status unavailable") for ax in BSV_COMPONENTS]
    testable: list[str] = []
    non_testable: list[tuple[str, str]] = []
    for ax in BSV_COMPONENTS:
        direction = comparator.expected_delta.get(ax, "unknown")
        conf = comparator.per_axis_confidence.get(ax, "low")
        if direction in ("up", "down", "mixed") and conf in ("high", "moderate", "low"):
            testable.append(ax)
        else:
            reason = (
                f"direction='{direction}'"
                + (f", confidence='{conf}'" if conf not in ("high", "moderate", "low") else "")
            )
            non_testable.append((ax, reason))
    return testable, non_testable


# ─────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────

def _sign_of(delta: float, threshold: float = FLAT_DELTA_THRESHOLD) -> str:
    if abs(delta) < threshold:
        return "flat"
    return "up" if delta > 0 else "down"


@dataclass
class AxisVerdictV3:
    axis: str
    testable: bool
    expected_direction: str
    expected_confidence: str
    observed_delta: float
    observed_sign: str
    verdict: str                         # agree | disagree | flat | mixed_resolved | mixed_flat | not_testable
    weight: float
    score: float
    note: str = ""


@dataclass
class CalibrationResultV3:
    contrast: CalibrationContrast
    comparator: SAELExpectedComparator

    control_bsv: CohortBSV
    perturbed_bsv: CohortBSV
    observed_delta: dict[str, float]

    testable_axes: list[str]
    non_testable_axes: list[tuple[str, str]]

    axis_verdicts: list[AxisVerdictV3]    # one per BSV axis; non-testable axes carry verdict="not_testable"
    confidence_weighted_score: float
    n_high_conf_agree: int
    n_moderate_conf_agree: int
    n_low_conf_agree: int
    n_disagree: int
    n_mixed_resolved: int
    n_mixed_flat: int
    n_flat: int
    n_not_testable: int

    overall_label: str                    # pass | partial | weak | inconsistent | inconclusive
    top_windows: list[dict]

    n_control: int
    n_perturbed: int
    pipeline: str
    crop_range: tuple[float, float]

    sample_bsv_control: np.ndarray | None = None
    sample_bsv_perturbed: np.ndarray | None = None


def _score_axis_testable(
    axis: str, expected_direction: str, expected_confidence: str,
    observed_delta: float,
) -> AxisVerdictV3:
    obs_sign = _sign_of(observed_delta)
    cw = CONF_WEIGHT.get(expected_confidence, 0.3)

    if expected_direction == "mixed":
        if obs_sign == "flat":
            return AxisVerdictV3(
                axis=axis, testable=True,
                expected_direction="mixed", expected_confidence=expected_confidence,
                observed_delta=observed_delta, observed_sign="flat",
                verdict="mixed_flat", weight=0.0, score=0.0,
                note="mixed expectation, observed below noise → consistent",
            )
        w = 0.3 * cw
        return AxisVerdictV3(
            axis=axis, testable=True,
            expected_direction="mixed", expected_confidence=expected_confidence,
            observed_delta=observed_delta, observed_sign=obs_sign,
            verdict="mixed_resolved", weight=w, score=+w,
            note=f"mixed expectation resolved to {obs_sign}",
        )

    # up / down
    if obs_sign == "flat":
        return AxisVerdictV3(
            axis=axis, testable=True,
            expected_direction=expected_direction, expected_confidence=expected_confidence,
            observed_delta=observed_delta, observed_sign="flat",
            verdict="flat", weight=0.0, score=0.0,
            note=f"|Δ|={abs(observed_delta):.3f} below noise — cannot confirm/refute",
        )
    if obs_sign == expected_direction:
        return AxisVerdictV3(
            axis=axis, testable=True,
            expected_direction=expected_direction, expected_confidence=expected_confidence,
            observed_delta=observed_delta, observed_sign=obs_sign,
            verdict="agree", weight=cw, score=+cw,
            note=f"sign matches expected {expected_direction}",
        )
    return AxisVerdictV3(
        axis=axis, testable=True,
        expected_direction=expected_direction, expected_confidence=expected_confidence,
        observed_delta=observed_delta, observed_sign=obs_sign,
        verdict="disagree", weight=cw, score=-cw,
        note=f"expected {expected_direction}, observed {obs_sign}",
    )


def _overall_label(verdicts: list[AxisVerdictV3], score: float) -> str:
    scorable = [v for v in verdicts if v.weight > 0]
    if not scorable:
        # No axis contributed to the score — all flat / mixed_flat / not_testable.
        return "inconclusive"

    n_agree = sum(1 for v in verdicts if v.verdict == "agree")
    has_hi_disagree = any(
        v.verdict == "disagree" and v.expected_confidence == "high"
        for v in verdicts
    )

    if score >= 0.7 and n_agree >= 1 and not has_hi_disagree:
        return "pass"
    if score <= -0.4 or has_hi_disagree:
        return "inconsistent"
    if score >= 0.3 and n_agree >= 1:
        return "partial"
    if score > 0 or n_agree >= 1:
        return "weak"
    return "weak"


# ─────────────────────────────────────────────────────────────────────
# Shared SAEL artefacts (built once per run)
# ─────────────────────────────────────────────────────────────────────

def build_all_sael_comparators() -> dict[str, SAELExpectedComparator]:
    ev = extract_anchor_evidence()
    wn = build_sael_anchor_windows(ev)
    ds = build_sael_expected_deltas(ev, wn)
    cs = derive_expected_comparators(ds)
    return {c.contrast_id: c for c in cs}


# ─────────────────────────────────────────────────────────────────────
# Main evaluator
# ─────────────────────────────────────────────────────────────────────

def run_calibration_eval_v3(
    contrast_id: str,
    comparators: dict[str, SAELExpectedComparator] | None = None,
    raw: CalibrationRaw | None = None,
) -> CalibrationResultV3:
    contrast = get_contrast(contrast_id)

    sael_id = CONTRAST_TO_SAEL.get(contrast_id)
    if sael_id is None:
        raise KeyError(
            f"No SAEL mapping for calibration contrast '{contrast_id}'. "
            "Add it to CONTRAST_TO_SAEL."
        )

    if comparators is None:
        comparators = build_all_sael_comparators()
    if sael_id not in comparators:
        raise KeyError(
            f"SAEL comparator '{sael_id}' was not produced by "
            "build_all_sael_comparators() — check SAEL pipeline."
        )
    comparator = comparators[sael_id]

    # Observed side — unchanged.
    if raw is None:
        raw = load_calibration_raw(contrast.loader_id)
    want = {contrast.control_cohort, contrast.perturbed_cohort}
    mask = np.isin(raw.cohorts, list(want))
    if mask.sum() == 0:
        raise ValueError(
            f"Contrast '{contrast_id}': no spectra match cohorts {want}. "
            f"Available: {sorted(set(raw.cohorts))}"
        )
    X_sub = raw.X[mask]
    cohorts_sub = raw.cohorts[mask]
    pre = preprocess_calibration(X_sub, raw.wavenumbers)
    wf = extract_window_features(pre.X, pre.wavenumbers)
    bsv_matrix = project_to_bsv(wf)
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

    # Testability
    testable, non_testable = testable_axes_for(comparator)

    # Score
    verdicts: list[AxisVerdictV3] = []
    for axis in BSV_COMPONENTS:
        if axis not in testable:
            reason = next((r for a, r in non_testable if a == axis), "unknown")
            verdicts.append(AxisVerdictV3(
                axis=axis, testable=False,
                expected_direction=comparator.expected_delta.get(axis, "unknown"),
                expected_confidence=comparator.per_axis_confidence.get(axis, "low"),
                observed_delta=observed_delta[axis], observed_sign=_sign_of(observed_delta[axis]),
                verdict="not_testable", weight=0.0, score=0.0,
                note=f"excluded from scoring: {reason}",
            ))
        else:
            direction = comparator.expected_delta[axis]
            conf = comparator.per_axis_confidence[axis]
            verdicts.append(_score_axis_testable(axis, direction, conf, observed_delta[axis]))

    total_w = sum(v.weight for v in verdicts)
    cw_score = round(sum(v.score for v in verdicts) / total_w, 3) if total_w > 0 else 0.0

    n_hi_agree = sum(1 for v in verdicts if v.verdict == "agree" and v.expected_confidence == "high")
    n_mo_agree = sum(1 for v in verdicts if v.verdict == "agree" and v.expected_confidence == "moderate")
    n_lo_agree = sum(1 for v in verdicts if v.verdict == "agree" and v.expected_confidence == "low")
    n_disagree = sum(1 for v in verdicts if v.verdict == "disagree")
    n_mixed_resolved = sum(1 for v in verdicts if v.verdict == "mixed_resolved")
    n_mixed_flat = sum(1 for v in verdicts if v.verdict == "mixed_flat")
    n_flat = sum(1 for v in verdicts if v.verdict == "flat")
    n_nt = sum(1 for v in verdicts if v.verdict == "not_testable")

    # Band drivers (unchanged observed-side)
    band_imp = compute_per_cohort_window_importance(
        wf, cohorts_sub, reference=contrast.control_cohort,
    )
    top = band_imp.get(contrast.perturbed_cohort, [])
    top_annotated = annotate_top_windows(top, top_n=6)

    ctrl_mask = cohorts_sub == contrast.control_cohort
    pert_mask = cohorts_sub == contrast.perturbed_cohort

    label = _overall_label(verdicts, cw_score)

    return CalibrationResultV3(
        contrast=contrast, comparator=comparator,
        control_bsv=ctrl, perturbed_bsv=pert,
        observed_delta=observed_delta,
        testable_axes=testable, non_testable_axes=non_testable,
        axis_verdicts=verdicts, confidence_weighted_score=cw_score,
        n_high_conf_agree=n_hi_agree,
        n_moderate_conf_agree=n_mo_agree,
        n_low_conf_agree=n_lo_agree,
        n_disagree=n_disagree,
        n_mixed_resolved=n_mixed_resolved,
        n_mixed_flat=n_mixed_flat,
        n_flat=n_flat, n_not_testable=n_nt,
        overall_label=label, top_windows=top_annotated,
        n_control=int(ctrl_mask.sum()), n_perturbed=int(pert_mask.sum()),
        pipeline=pre.pipeline, crop_range=pre.crop_range,
        sample_bsv_control=bsv_matrix[ctrl_mask],
        sample_bsv_perturbed=bsv_matrix[pert_mask],
    )


def summarize_v3(r: CalibrationResultV3) -> dict:
    return {
        "contrast_id": r.contrast.contrast_id,
        "display_name": r.contrast.display_name,
        "sael_contrast_id": r.comparator.contrast_id,
        "sael_status": r.comparator.status,
        "sael_overall_confidence": r.comparator.overall_confidence,
        "n_control": r.n_control,
        "n_perturbed": r.n_perturbed,
        "n_testable_axes": len(r.testable_axes),
        "testable_axes": "; ".join(r.testable_axes),
        "confidence_weighted_score": r.confidence_weighted_score,
        "n_high_conf_agree": r.n_high_conf_agree,
        "n_moderate_conf_agree": r.n_moderate_conf_agree,
        "n_low_conf_agree": r.n_low_conf_agree,
        "n_disagree": r.n_disagree,
        "n_mixed_resolved": r.n_mixed_resolved,
        "n_mixed_flat": r.n_mixed_flat,
        "n_flat": r.n_flat,
        "n_not_testable": r.n_not_testable,
        "overall_label": r.overall_label,
        "pipeline": r.pipeline,
    }
