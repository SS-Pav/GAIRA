"""Audit script for the crop-before-interpolate preprocessing rule.

Exercises four representative cases against
:func:`gaira.spectral.crop_before_interpolate`:

  1. 300-2000 input  (broad; crop active; full coverage)
  2. 400-1800 input  (exact match; no crop; full coverage)
  3. 420-1750 input  (partial coverage on both sides; NaN masking)
  4. 1600-2200 input (minimal overlap; should raise InsufficientOverlapError)

Reports per case:
  - raw range
  - cropped range used
  - in-support interpolation range on master axis
  - partial_coverage flag
  - whether extrapolation was avoided (no finite values outside measured support)
  - InsufficientOverlapError if applicable

Run:
    cd /Users/suraj/projects/GAIRA
    PYTHONPATH=src .venv/bin/python scripts/audit_crop_before_interpolate_v1.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gaira.spectral import (
    CANONICAL_SUPPORT_CM1,
    canonical_master_axis,
    crop_before_interpolate,
    InsufficientOverlapError,
)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _synthetic_spectrum(lo: float, hi: float, step: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Build a synthetic raw spectrum over [lo, hi] with a handful of Gaussian
    peaks so the interpolated output is non-trivial.
    """
    wn = np.arange(lo, hi + step, step)
    centres = [500.0, 700.0, 1000.0, 1280.0, 1450.0, 1600.0, 1740.0]
    widths  = [30.0, 25.0, 20.0, 35.0, 22.0, 28.0, 18.0]
    y = np.zeros_like(wn)
    for c, w in zip(centres, widths):
        y += np.exp(-((wn - c) / w) ** 2)
    # add a linear baseline to make cropping consequential
    y += 0.5 + 0.001 * (wn - 400.0)
    return wn, y


def _fmt_range(x: np.ndarray) -> str:
    return f"[{x.min():.1f}, {x.max():.1f}]"


def _check(label: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {label}{(' — ' + detail) if detail else ''}")
    if not cond:
        raise AssertionError(f"audit failed: {label} — {detail}")


# ──────────────────────────────────────────────────────────────────────
# Cases
# ──────────────────────────────────────────────────────────────────────

def run_case(label: str, lo: float, hi: float,
             *, expect_partial: bool,
             expect_raise: bool = False,
             min_coverage: float = 0.5) -> None:
    print("\n" + "─" * 78)
    print(f" {label}")
    print("─" * 78)
    master_x = canonical_master_axis()
    master_min, master_max = float(master_x[0]), float(master_x[-1])
    print(f"  canonical support       = {CANONICAL_SUPPORT_CM1}")
    print(f"  master_x (range/pts)    = [{master_min}, {master_max}] / {master_x.size}")
    print(f"  raw spectrum input      = [{lo}, {hi}]")

    raw_wn, raw_y = _synthetic_spectrum(lo, hi)

    if expect_raise:
        try:
            _ = crop_before_interpolate(
                raw_wn, raw_y, master_x,
                partial_ok=True, min_coverage=min_coverage,
            )
        except InsufficientOverlapError as exc:
            print(f"  raised InsufficientOverlapError ✓")
            print(f"    message = {exc}")
            _check("expected insufficient-overlap raise", True)
            return
        _check("expected InsufficientOverlapError but no raise", False)
        return

    y_interp, cov = crop_before_interpolate(
        raw_wn, raw_y, master_x,
        partial_ok=True, min_coverage=min_coverage,
    )

    in_support_mask = np.isfinite(y_interp)
    in_support_x = master_x[in_support_mask]
    out_support_x = master_x[~in_support_mask]

    print(f"  cropped range used      = [{cov.cropped_min_cm1:.1f}, {cov.cropped_max_cm1:.1f}]")
    print(f"  interpolation range     = "
          f"{_fmt_range(in_support_x) if in_support_x.size else 'EMPTY'}")
    print(f"  partial_coverage flag   = {cov.partial_coverage}")
    print(f"  coverage_fraction       = {cov.coverage_fraction:.3f}")
    print(f"  n_interpolated_points   = {cov.n_interpolated_points}")
    print(f"  n_masked_out_of_support = {cov.n_masked_out_of_support}")

    # ── Invariants ────────────────────────────────────────────────────

    # 1. raw wavenumbers outside the canonical support must not leak into
    #    the cropped range
    _check(
        "crop respects master_x bounds",
        cov.cropped_min_cm1 >= master_min and cov.cropped_max_cm1 <= master_max,
        f"cropped=[{cov.cropped_min_cm1}, {cov.cropped_max_cm1}] master=[{master_min}, {master_max}]",
    )

    # 2. partial_coverage flag matches expectation
    _check(
        "partial_coverage flag matches expectation",
        cov.partial_coverage == expect_partial,
        f"got {cov.partial_coverage}, expected {expect_partial}",
    )

    # 3. no extrapolation: every finite master_x point must be inside the
    #    measured support (cropped range)
    if in_support_x.size:
        _check(
            "no extrapolation — every finite output is inside measured support",
            (in_support_x.min() >= cov.cropped_min_cm1 - 1e-9)
            and (in_support_x.max() <= cov.cropped_max_cm1 + 1e-9),
            f"finite range {_fmt_range(in_support_x)} vs measured "
            f"[{cov.cropped_min_cm1}, {cov.cropped_max_cm1}]",
        )

    # 4. every out-of-support master_x point must be NaN
    if out_support_x.size:
        _check(
            "out-of-support master_x points are NaN (no silent clamping)",
            np.isnan(y_interp[~in_support_mask]).all(),
            f"{(~np.isnan(y_interp[~in_support_mask])).sum()} non-NaN out-of-support points",
        )

    # 5. counts must sum to master_x.size
    _check(
        "coverage counts sum to master_x size",
        cov.n_interpolated_points + cov.n_masked_out_of_support == master_x.size,
    )


def main() -> None:
    print("\n[audit: crop_before_interpolate_v1]")
    print(f"canonical support = {CANONICAL_SUPPORT_CM1}")

    # Case A — broad; crop should activate; full coverage inside master axis
    run_case("Case A — 300-2000 input (broad; crop active)",
             300.0, 2000.0, expect_partial=False)

    # Case B — exact match; no crop on either side
    run_case("Case B — 400-1800 input (exact; no crop)",
             400.0, 1800.0, expect_partial=False)

    # Case C — partial coverage on both sides; NaN masking expected
    run_case("Case C — 420-1750 input (partial on both sides)",
             420.0, 1750.0, expect_partial=True)

    # Case D — only 1600-1800 overlaps (200 / 1400 = 14.3 % of master axis);
    # below default min_coverage=0.5 → InsufficientOverlapError
    run_case("Case D — 1600-2200 input (insufficient overlap, ~14% of master)",
             1600.0, 2200.0, expect_partial=True, expect_raise=True)

    # Extra: Case D with relaxed threshold should NOT raise (but still flags partial)
    print("\n" + "─" * 78)
    print(" Case D' — 1600-2200 input with min_coverage=0.1 (relaxed)")
    print("─" * 78)
    master_x = canonical_master_axis()
    raw_wn, raw_y = _synthetic_spectrum(1600.0, 2200.0)
    y_interp, cov = crop_before_interpolate(
        raw_wn, raw_y, master_x,
        partial_ok=True, min_coverage=0.1,
    )
    print(f"  partial_coverage        = {cov.partial_coverage} (expected True)")
    print(f"  coverage_fraction       = {cov.coverage_fraction:.3f} (expected ~0.14)")
    print(f"  finite output range     = {_fmt_range(master_x[np.isfinite(y_interp)])}")
    _check("relaxed threshold allows partial coverage through",
           cov.partial_coverage and 0.10 < cov.coverage_fraction < 0.20)

    # Extra: partial_ok=False rejects a partial spectrum above min_coverage
    print("\n" + "─" * 78)
    print(" Case E — partial_ok=False on 420-1750 input")
    print("─" * 78)
    raw_wn, raw_y = _synthetic_spectrum(420.0, 1750.0)
    try:
        crop_before_interpolate(raw_wn, raw_y, master_x, partial_ok=False)
    except InsufficientOverlapError as exc:
        print(f"  raised InsufficientOverlapError ✓")
        print(f"    message = {exc}")
        _check("partial_ok=False rejects partial coverage", True)

    print()
    print("─" * 78)
    print("[audit: crop_before_interpolate_v1] ALL CHECKS PASSED ✓")


if __name__ == "__main__":
    main()
