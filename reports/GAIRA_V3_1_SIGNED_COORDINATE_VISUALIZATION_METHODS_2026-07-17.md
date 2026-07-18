# GAIRA V3.1 — Signed-Coordinate Visualization Methods

**Date:** 2026-07-17

## The problem (audited)
V3's global coordinates are **signed** robust z-scores (`(raw − median)/MAD`, negatives common), but the biological-projection radar rendered them with `gaira_core/plotting.radar_figure`, whose radial axis starts at **0** (`range=[0, radial_max]`). Consequences, confirmed:
- Negative axis means (e.g. axes below the Ag-SERS reference median) are pushed to the plot centre / clipped at 0.
- Group means visually collapse toward the origin; only positive components remain visible.
- The `±4` display clip further distorts shape.
- Net effect: a **sparse, redox-heavy** radar that misrepresents a genuinely multiaxis signed profile.

This is a rendering fault, not a coordinate fault — the stored signed values were correct; the radar hid their sign.

## The fix (default global-coordinate display)
A **diverging horizontal bar plot** (`gaira_core/v3_1_views.diverging_figure`):
- one row per biochemical axis,
- grouped bars per cohort (group means),
- an explicit **zero reference line**,
- a **symmetric x-range** `[−xmax, +xmax]`,
- both positive and negative values fully visible,
- a fixed global scale (robust σ from the Ag-SERS reference median).

Used for both the frozen global coordinates and the cohort-relative effect profile. The zero-origin radar is retained ONLY for nonnegative raw BSV (clearly labelled).

## Tests (`tests/test_signed_global_visualization.py`)
- Frozen global EV cohort means contain negative axis means (signed).
- `diverging_figure` plotted data spans both signs; x-range symmetric about 0.
- A zero reference line is present.

## Note
No stored coordinate value was changed. This is a visualization correction. The frozen calibration content hash is identical to V3 (`tests/test_v3_raw_regression.py::test_frozen_calibration_unchanged_vs_v3`).
