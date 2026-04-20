"""Calibration eval v2 — evaluates each contrast against ExpectedComparatorV2.

Key differences vs eval.py (v1):
  - Expected axes come from the literature-side ExpectedComparatorV2 (with
    per-axis direction AND per-axis confidence), not from the calibration
    registry's single expected axis.
  - Multiple axes per contrast are scored. `mixed` directions are allowed
    and do not count as failures.
  - Confidence-aware scoring: agreement on a high-confidence axis weighs
    more than agreement on a low-confidence axis. Disagreement on a
    low-confidence axis is a soft penalty, not a hard failure.

OBSERVED SIDE IS UNCHANGED. We reuse the same loader + preprocessing +
direct spectral BSV pipeline from eval.py. This module only replaces the
expected side and the scoring rule.

No calibration data influences the expected comparator — the expected
side is literature-grounded, pre-built by the expected-BSV layer v2.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from gaira.calibration.band_annotation import annotate_top_windows
from gaira.calibration.eval import FLAT_DELTA_THRESHOLD, CalibrationResult
from gaira.calibration.loaders import CalibrationRaw, load_calibration_raw
from gaira.calibration.preprocessing import preprocess_calibration
from gaira.calibration.registry import CalibrationContrast, get_contrast
from gaira.expected.comparator_v2 import (
    ExpectedComparatorV2, build_expected_comparator_v2,
)
from gaira.spectral.band_drivers import compute_per_cohort_window_importance
from gaira.spectral.bsv_projection import (
    compute_cohort_bsvs, project_to_bsv, CohortBSV,
)
from gaira.spectral.window_panel import (
    BSV_COMPONENTS, extract_window_features,
)


# ─────────────────────────────────────────────────────────────────────
# Calibration contrast → ExpectedComparatorV2 contrast_id
# Placed here, NOT in registry.py, so the calibration registry stays
# unchanged and v1 behaviour is preserved.
# ─────────────────────────────────────────────────────────────────────

CONTRAST_TO_EXPECTED_DELTA: dict[str, str] = {
    "cspp_fig7_hypoxanthine_spike":        "hypoxanthine_spike_literature",
    "uricase_spiked_hypoxanthine_serum":   "hypoxanthine_spike_literature",
    "cspp_fig7_ergothioneine_spike":       "ergothioneine_spike_literature",
    "ergothioneine_titration_top_vs_zero": "ergothioneine_spike_literature",
    "uricase_sigma_depletion":             "uricase_depletion_literature",
}


# ─────────────────────────────────────────────────────────────────────
# Confidence weighting
# ─────────────────────────────────────────────────────────────────────

CONF_WEIGHT = {"high": 1.0, "moderate": 0.6, "low": 0.3}


# ─────────────────────────────────────────────────────────────────────
# Per-axis verdict in the v2 world
# ─────────────────────────────────────────────────────────────────────

@dataclass
class AxisVerdictV2:
    axis: str
    expected_direction: str              # up | down | flat | mixed | unknown
    expected_confidence: str             # high | moderate | low
    observed_delta: float
    observed_sign: str                   # up | down | flat
    verdict: str                         # agree | disagree | flat | mixed_resolved | mixed_flat | unconstrained
    weight: float                        # 0..1 contribution to overall score
    score: float                         # signed: +weight (agree), 0 (flat/unconstrained), -weight (disagree)
    note: str = ""


@dataclass
class CalibrationResultV2:
    contrast: CalibrationContrast
    comparator: ExpectedComparatorV2

    control_bsv: CohortBSV
    perturbed_bsv: CohortBSV
    observed_delta: dict[str, float]

    axis_verdicts: list[AxisVerdictV2]
    confidence_weighted_score: float     # in [-1, 1]; weighted mean of signed per-axis scores
    n_high_conf_agree: int
    n_moderate_conf_agree: int
    n_low_conf_agree: int
    n_disagree: int
    n_mixed_resolved: int
    n_mixed_flat: int
    n_flat: int

    overall_label: str                   # pass | partial | weak | inconsistent | inconclusive
    top_windows: list[dict]

    n_control: int
    n_perturbed: int
    pipeline: str
    crop_range: tuple[float, float]

    sample_bsv_control: np.ndarray | None = None
    sample_bsv_perturbed: np.ndarray | None = None


# ─────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────

def _sign_of(delta: float, threshold: float = FLAT_DELTA_THRESHOLD) -> str:
    if abs(delta) < threshold:
        return "flat"
    return "up" if delta > 0 else "down"


def _score_axis(
    axis: str,
    expected_direction: str,
    expected_confidence: str,
    observed_delta: float,
) -> AxisVerdictV2:
    """Return a per-axis v2 verdict.

    Rules:
      - flat observed + up/down expected: verdict=flat (weight 0, score 0).
        Signal is below noise floor so it cannot confirm or refute.
      - up/down matching sign: verdict=agree, score = +weight.
      - up/down opposite sign: verdict=disagree, score = -weight.
      - mixed expected + strong observed: verdict=mixed_resolved,
        weight = 0.3 × conf_weight, score = +weight.
        We give partial credit because direction emerged despite the
        registered ambiguity — but never full credit.
      - mixed expected + flat observed: verdict=mixed_flat, weight 0, score 0.
        Consistent with the registered ambiguity; not penalized.
      - flat/unknown expected: verdict=unconstrained, weight 0, score 0.
    """
    obs_sign = _sign_of(observed_delta)
    conf_w = CONF_WEIGHT.get(expected_confidence, 0.3)

    if expected_direction in ("flat", "unknown"):
        return AxisVerdictV2(
            axis=axis, expected_direction=expected_direction,
            expected_confidence=expected_confidence,
            observed_delta=observed_delta, observed_sign=obs_sign,
            verdict="unconstrained", weight=0.0, score=0.0,
            note="no directional expectation to score against",
        )

    if expected_direction == "mixed":
        if obs_sign == "flat":
            return AxisVerdictV2(
                axis=axis, expected_direction="mixed",
                expected_confidence=expected_confidence,
                observed_delta=observed_delta, observed_sign="flat",
                verdict="mixed_flat", weight=0.0, score=0.0,
                note="expected ambiguous and observed below noise floor — consistent",
            )
        w = 0.3 * conf_w
        return AxisVerdictV2(
            axis=axis, expected_direction="mixed",
            expected_confidence=expected_confidence,
            observed_delta=observed_delta, observed_sign=obs_sign,
            verdict="mixed_resolved", weight=w, score=+w,
            note=f"mixed expectation resolved to {obs_sign} observed",
        )

    # Directional expected (up / down)
    if obs_sign == "flat":
        return AxisVerdictV2(
            axis=axis, expected_direction=expected_direction,
            expected_confidence=expected_confidence,
            observed_delta=observed_delta, observed_sign="flat",
            verdict="flat", weight=0.0, score=0.0,
            note=f"|Δ|={abs(observed_delta):.3f} below noise — cannot confirm/refute",
        )

    if obs_sign == expected_direction:
        return AxisVerdictV2(
            axis=axis, expected_direction=expected_direction,
            expected_confidence=expected_confidence,
            observed_delta=observed_delta, observed_sign=obs_sign,
            verdict="agree", weight=conf_w, score=+conf_w,
            note=f"sign matches expected {expected_direction}",
        )

    return AxisVerdictV2(
        axis=axis, expected_direction=expected_direction,
        expected_confidence=expected_confidence,
        observed_delta=observed_delta, observed_sign=obs_sign,
        verdict="disagree", weight=conf_w, score=-conf_w,
        note=f"expected {expected_direction}, observed {obs_sign}",
    )


def _overall_label(
    verdicts: list[AxisVerdictV2], confidence_weighted_score: float,
) -> str:
    """Derive a v1-style label from the per-axis verdicts + overall score."""
    # Only axes with non-zero weight count for labelling.
    scorable = [v for v in verdicts if v.weight > 0]
    if not scorable:
        # Nothing to score on — purely mixed_flat / flat / unconstrained.
        return "inconclusive"

    n_agree = sum(1 for v in verdicts if v.verdict == "agree")
    n_disagree = sum(1 for v in verdicts if v.verdict == "disagree")

    # Hard-disagreement on a high-confidence axis caps the label.
    has_hi_disagree = any(
        v.verdict == "disagree" and v.expected_confidence == "high"
        for v in verdicts
    )

    if confidence_weighted_score >= 0.7 and n_agree >= 1 and not has_hi_disagree:
        return "pass"
    if confidence_weighted_score <= -0.4 or has_hi_disagree:
        return "inconsistent"
    if confidence_weighted_score >= 0.3 and n_agree >= 1:
        return "partial"
    if confidence_weighted_score > 0 or n_agree >= 1:
        return "weak"
    return "weak"


# ─────────────────────────────────────────────────────────────────────
# Main evaluator
# ─────────────────────────────────────────────────────────────────────

def run_calibration_eval_v2(
    contrast_id: str,
    raw: CalibrationRaw | None = None,
    anchor_df=None,
    peaks_df=None,
) -> CalibrationResultV2:
    contrast = get_contrast(contrast_id)

    expected_delta_id = CONTRAST_TO_EXPECTED_DELTA.get(contrast_id)
    if expected_delta_id is None:
        raise KeyError(
            f"No expected-delta mapping for calibration contrast "
            f"'{contrast_id}'. Add it to CONTRAST_TO_EXPECTED_DELTA."
        )

    comparator = build_expected_comparator_v2(
        expected_delta_id, anchor_df=anchor_df, peaks_df=peaks_df,
    )

    # Observed side — same pipeline as v1.
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
    window_features = extract_window_features(pre.X, pre.wavenumbers)
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

    # Score each axis against the v2 expected comparator.
    verdicts: list[AxisVerdictV2] = []
    for axis in BSV_COMPONENTS:
        expected_dir = comparator.expected_delta.get(axis, "unknown")
        expected_conf = comparator.per_axis_confidence.get(axis, "low")
        verdicts.append(_score_axis(
            axis, expected_dir, expected_conf, observed_delta[axis],
        ))

    total_weight = sum(v.weight for v in verdicts)
    if total_weight > 0:
        cw_score = round(sum(v.score for v in verdicts) / total_weight, 3)
    else:
        cw_score = 0.0

    # Tallies
    n_hi_agree = sum(1 for v in verdicts if v.verdict == "agree" and v.expected_confidence == "high")
    n_mo_agree = sum(1 for v in verdicts if v.verdict == "agree" and v.expected_confidence == "moderate")
    n_lo_agree = sum(1 for v in verdicts if v.verdict == "agree" and v.expected_confidence == "low")
    n_disagree = sum(1 for v in verdicts if v.verdict == "disagree")
    n_mixed_resolved = sum(1 for v in verdicts if v.verdict == "mixed_resolved")
    n_mixed_flat = sum(1 for v in verdicts if v.verdict == "mixed_flat")
    n_flat = sum(1 for v in verdicts if v.verdict == "flat")

    # Band drivers + annotations — unchanged from v1.
    band_imp = compute_per_cohort_window_importance(
        window_features, cohorts_sub, reference=contrast.control_cohort,
    )
    top = band_imp.get(contrast.perturbed_cohort, [])
    top_annotated = annotate_top_windows(top, top_n=6)

    ctrl_mask = cohorts_sub == contrast.control_cohort
    pert_mask = cohorts_sub == contrast.perturbed_cohort

    label = _overall_label(verdicts, cw_score)

    return CalibrationResultV2(
        contrast=contrast,
        comparator=comparator,
        control_bsv=ctrl,
        perturbed_bsv=pert,
        observed_delta=observed_delta,
        axis_verdicts=verdicts,
        confidence_weighted_score=cw_score,
        n_high_conf_agree=n_hi_agree,
        n_moderate_conf_agree=n_mo_agree,
        n_low_conf_agree=n_lo_agree,
        n_disagree=n_disagree,
        n_mixed_resolved=n_mixed_resolved,
        n_mixed_flat=n_mixed_flat,
        n_flat=n_flat,
        overall_label=label,
        top_windows=top_annotated,
        n_control=int(ctrl_mask.sum()),
        n_perturbed=int(pert_mask.sum()),
        pipeline=pre.pipeline,
        crop_range=pre.crop_range,
        sample_bsv_control=bsv_matrix[ctrl_mask],
        sample_bsv_perturbed=bsv_matrix[pert_mask],
    )


def summarize_v2(r: CalibrationResultV2) -> dict:
    return {
        "contrast_id": r.contrast.contrast_id,
        "display_name": r.contrast.display_name,
        "expected_delta_id": r.comparator.contrast_id,
        "expected_status": r.comparator.status,
        "expected_overall_confidence": r.comparator.overall_confidence,
        "n_control": r.n_control,
        "n_perturbed": r.n_perturbed,
        "confidence_weighted_score": r.confidence_weighted_score,
        "n_high_conf_agree": r.n_high_conf_agree,
        "n_moderate_conf_agree": r.n_moderate_conf_agree,
        "n_low_conf_agree": r.n_low_conf_agree,
        "n_disagree": r.n_disagree,
        "n_mixed_resolved": r.n_mixed_resolved,
        "n_mixed_flat": r.n_mixed_flat,
        "n_flat": r.n_flat,
        "overall_label": r.overall_label,
        "pipeline": r.pipeline,
    }
