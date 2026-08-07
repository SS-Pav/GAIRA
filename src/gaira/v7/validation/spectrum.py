"""GAIRA V7 — Phase 10: scientific input validation.

Runs before inference. Answers one question: *can the frozen engine say anything useful about
this input, and if so with what caveats?*

Three severities, and the distinction is load-bearing:

  ERROR    the engine cannot run. Nothing is returned but the diagnostics.
  WARNING  the engine runs and the interpretation is limited in a stated way.
  INFO     metadata and scope messages, including the sample-type scope note.

What this module does NOT do: claim a molecule is absent from the bank. Phase 09 established
that the engine has no validated open-set detection, and no amount of input checking creates one.
"""
from __future__ import annotations

import numpy as np

from gaira.v7.contracts import (Diagnostic, Modality, SampleType, Severity,
                                SUPPORTED_MODALITIES, VALIDATED_SAMPLE_TYPES, ValidationResult)

# The canonical window, restated here only to check coverage. The authority is the engine.
GRID_LO, GRID_HI = 450.0, 1800.0
MIN_POINTS = 32
MIN_COVERAGE_ERROR = 0.10      # below this the projection is meaningless
MIN_COVERAGE_WARN = 0.70
MIN_DISTINCT_FRACTION = 0.05   # intensity degeneracy floor


def _d(sev: Severity, code: str, msg: str, **detail) -> Diagnostic:
    return Diagnostic(severity=sev, code=code, message=msg, detail=detail)


def coverage(wavenumber: np.ndarray) -> float:
    """Fraction of the canonical 450–1800 cm⁻¹ window the input actually spans."""
    if wavenumber.size < 2:
        return 0.0
    lo, hi = float(np.min(wavenumber)), float(np.max(wavenumber))
    return float(max(0.0, min(hi, GRID_HI) - max(lo, GRID_LO)) / (GRID_HI - GRID_LO))


def validate(wavenumber, intensity, *, modality: Modality = Modality.RAMAN,
             sample_type: SampleType = SampleType.PURE,
             extra: list[Diagnostic] | None = None) -> ValidationResult:
    diags: list[Diagnostic] = list(extra or [])
    x = np.asarray(wavenumber, float).ravel()
    y = np.asarray(intensity, float).ravel()

    # ── scope, first: it applies whatever the numbers look like ──────────────
    if modality not in SUPPORTED_MODALITIES:
        diags.append(_d(Severity.ERROR, "scope.modality_unsupported",
                        f"modality '{modality.value}' is not supported by the V7 scientific "
                        f"core. V7 is Raman-only (decision A-09): a Raman motif dictionary "
                        f"reconstructs SERS of the same metabolites comfortably, so running "
                        f"{modality.value} through it produces confident numbers with no "
                        f"validated meaning. Set modality='raman' only if this really is a "
                        f"Raman measurement.",
                        modality=modality.value,
                        supported=[m.value for m in SUPPORTED_MODALITIES]))
    if sample_type not in VALIDATED_SAMPLE_TYPES:
        diags.append(_d(Severity.WARNING, "scope.sample_type_unvalidated",
                        f"sample_type '{sample_type.value}' is recorded as metadata but V7 has "
                        f"no validated interpretation capability for it. Every V7 number comes "
                        f"from pure reference compounds. The calculation is unchanged; the "
                        f"interpretation is not supported.",
                        sample_type=sample_type.value))

    # ── shape ────────────────────────────────────────────────────────────────
    if x.size != y.size:
        diags.append(_d(Severity.ERROR, "input.length_mismatch",
                        f"wavenumber has {x.size} points but intensity has {y.size}"))
        return _finish(diags, int(min(x.size, y.size)), None, None)
    if x.size < MIN_POINTS:
        diags.append(_d(Severity.ERROR, "input.too_few_points",
                        f"{x.size} points supplied; at least {MIN_POINTS} are required to "
                        f"resample onto the 676-bin canonical grid without inventing structure",
                        n_points=int(x.size), minimum=MIN_POINTS))
        return _finish(diags, int(x.size), None, None)

    if not np.isfinite(x).all():
        diags.append(_d(Severity.ERROR, "input.wavenumber_non_finite",
                        f"{int((~np.isfinite(x)).sum())} wavenumber values are NaN or infinite"))
    if not np.isfinite(y).all():
        n = int((~np.isfinite(y)).sum())
        diags.append(_d(Severity.ERROR if n > 0.2 * y.size else Severity.WARNING,
                        "input.intensity_non_finite",
                        f"{n} of {y.size} intensity values are NaN or infinite", n=n))

    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 2:
        return _finish(diags, int(x.size), None, None)
    xf, yf = x[finite], y[finite]
    rng = (float(xf.min()), float(xf.max()))

    if not (np.all(np.diff(xf) > 0) or np.all(np.diff(xf) < 0)):
        diags.append(_d(Severity.WARNING, "input.not_monotonic",
                        "the wavenumber axis is not monotonic; it will be sorted ascending "
                        "before resampling"))

    # ── spectral coverage ────────────────────────────────────────────────────
    cov = coverage(xf)
    if cov < MIN_COVERAGE_ERROR:
        diags.append(_d(Severity.ERROR, "coverage.insufficient",
                        f"the input spans {rng[0]:.0f}-{rng[1]:.0f} cm-1 and covers only "
                        f"{cov:.1%} of the canonical 450-1800 cm-1 window. Below "
                        f"{MIN_COVERAGE_ERROR:.0%} the projection describes zero-fill, not "
                        f"chemistry.", coverage=round(cov, 4), range_cm=rng))
    elif cov < MIN_COVERAGE_WARN:
        diags.append(_d(Severity.WARNING, "coverage.partial",
                        f"the input covers {cov:.1%} of the canonical window "
                        f"({rng[0]:.0f}-{rng[1]:.0f} cm-1). Missing regions are zero-filled, "
                        f"never extrapolated, and motifs whose diagnostic bands fall outside the "
                        f"measured range cannot activate.",
                        coverage=round(cov, 4), range_cm=rng))
    else:
        diags.append(_d(Severity.INFO, "coverage.ok",
                        f"covers {cov:.1%} of 450-1800 cm-1", coverage=round(cov, 4)))

    if rng[1] < GRID_LO or rng[0] > GRID_HI:
        diags.append(_d(Severity.ERROR, "coverage.disjoint",
                        f"the input range {rng[0]:.0f}-{rng[1]:.0f} cm-1 does not overlap the "
                        f"canonical window at all", range_cm=rng))

    # ── intensity structure ──────────────────────────────────────────────────
    if np.allclose(yf, 0.0):
        diags.append(_d(Severity.ERROR, "intensity.all_zero",
                        "every intensity value is zero"))
    elif float(np.ptp(yf)) == 0.0:
        diags.append(_d(Severity.ERROR, "intensity.constant",
                        f"the spectrum is constant at {float(yf[0]):.6g}; there is no band "
                        f"structure to project"))
    else:
        distinct = len(np.unique(np.round(yf, 12))) / yf.size
        if distinct < MIN_DISTINCT_FRACTION:
            diags.append(_d(Severity.WARNING, "intensity.degenerate",
                            f"only {distinct:.1%} of intensity values are distinct; the "
                            f"spectrum may be heavily quantised or truncated",
                            distinct_fraction=round(distinct, 4)))
        top = float(yf.max())
        clipped = int((yf >= top * (1 - 1e-12)).sum())
        if clipped > max(3, 0.01 * yf.size):
            diags.append(_d(Severity.WARNING, "intensity.clipped",
                            f"{clipped} points sit exactly at the maximum value; the detector "
                            f"may have saturated", n_at_maximum=clipped))
        if float(yf.min()) < 0:
            diags.append(_d(Severity.INFO, "intensity.negative",
                            f"{int((yf < 0).sum())} negative intensity values present; these "
                            f"are clipped to zero after baseline removal (P-02)"))

    step = float(np.median(np.abs(np.diff(xf)))) if xf.size > 1 else 0.0
    if step > 8.0:
        diags.append(_d(Severity.WARNING, "input.coarse_sampling",
                        f"median wavenumber step is {step:.1f} cm-1 against a canonical grid "
                        f"spacing of 2.0; resampling will interpolate heavily",
                        median_step_cm=round(step, 3)))
    return _finish(diags, int(x.size), rng, cov)


def _finish(diags, n_points, rng, cov) -> ValidationResult:
    can_run = not any(d.severity is Severity.ERROR for d in diags)
    return ValidationResult(ok=can_run, can_run=can_run, diagnostics=diags,
                            n_points=n_points, range_cm=rng, grid_coverage=cov)
