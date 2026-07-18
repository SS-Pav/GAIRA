"""GAIRA Demo v3 — Global Biochemical Coordinates.

Adds a FROZEN, versioned global-coordinate layer on top of the unchanged V2
raw heuristic BSV:

    raw_bsv  --(frozen robust per-axis calibration)-->  global_coordinates

Design invariants (enforced here):
  * The raw BSV path (build_report) is never modified. `raw_bsv_from_spectrum`
    is a thin wrapper around the V2 engine and returns identical values.
  * The calibration (per-axis center/scale/quantiles) is LOADED from a frozen
    artifact. It is NEVER fitted at runtime. If the artifact is missing, callers
    get status "GLOBAL_COORDINATE_UNAVAILABLE" and the raw BSV is retained.
  * The transform is deterministic and label-free:
        unbounded_j = (raw_j - center_j) / scale_j
        display_j   = clip(unbounded_j, -clip, +clip)   # unbounded preserved
  * Three coordinate systems are kept explicitly separate and never overwrite
    each other: raw_bsv, global_biochemical_coordinates, cohort_relative_coordinates.

Cohort-relative coordinates (within-dataset z-score) are provided for the
clearly-labelled "exploratory" view ONLY — they are not the global state and
change with the comparison group by construction.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config as cfg

CALIBRATION_PATH = cfg.GENERATED_DIR / "global_coordinate_calibration_v1.json"

GLOBAL_UNAVAILABLE = "GLOBAL_COORDINATE_UNAVAILABLE"


# ─────────────────────────────────────────────────────────────────────
# Raw BSV projection (V2 engine, UNCHANGED)
# ─────────────────────────────────────────────────────────────────────

def raw_bsv_from_spectrum(wavenumber, intensity, *, substrate: str,
                          domain: str = "unspecified",
                          sample_id: str = "sample") -> dict[str, float]:
    """Project a spectrum to the raw 11-axis heuristic BSV via the V2 engine.

    This calls the unmodified `report_builder.build_report` and returns its
    `bsv` dict verbatim. No calibration is applied here.
    """
    from .report_builder import build_report
    rep = build_report(sample_id=sample_id, title=sample_id, domain=domain,
                       substrate=substrate,
                       wavenumber=np.asarray(wavenumber, float),
                       intensity=np.asarray(intensity, float))
    return {a: float(rep["bsv"].get(a, 0.0)) for a in cfg.BSV_AXES}


# ─────────────────────────────────────────────────────────────────────
# Frozen calibration
# ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GlobalCalibration:
    ontology_version: str
    calibration_version: str
    axes: tuple[str, ...]
    center: dict[str, float]
    scale: dict[str, float]
    q_low: dict[str, float]
    q_high: dict[str, float]
    clip: float
    transform: str
    scale_floor: float
    n_reference_analytes: int
    n_calibration_spectra: int
    n_biological_spectra: int
    matrix_composition: dict
    substrate_composition: dict
    limitations: str

    def is_valid(self) -> bool:
        return bool(self.center) and bool(self.scale) and len(self.axes) > 0


def load_calibration(path: Path | None = None) -> GlobalCalibration | None:
    """Load the frozen calibration artifact. Returns None if unavailable.

    NEVER fits a calibration — absence is reported, not silently repaired.
    """
    p = path or CALIBRATION_PATH
    try:
        if not (p.exists() and p.stat().st_size > 0):
            return None
        d = json.loads(p.read_text())
    except Exception:
        return None
    try:
        return GlobalCalibration(
            ontology_version=d["ontology_version"],
            calibration_version=d["calibration_version"],
            axes=tuple(d["axes"]),
            center={k: float(v) for k, v in d["axis_center"].items()},
            scale={k: float(v) for k, v in d["axis_scale"].items()},
            q_low={k: float(v) for k, v in d.get("reference_q_low", {}).items()},
            q_high={k: float(v) for k, v in d.get("reference_q_high", {}).items()},
            clip=float(d.get("clip", 4.0)),
            transform=d.get("transform", "robust_z"),
            scale_floor=float(d.get("scale_floor", 1e-6)),
            n_reference_analytes=int(d.get("n_reference_analytes", 0)),
            n_calibration_spectra=int(d.get("n_calibration_spectra", 0)),
            n_biological_spectra=int(d.get("n_biological_spectra", 0)),
            matrix_composition=d.get("matrix_composition", {}),
            substrate_composition=d.get("substrate_composition", {}),
            limitations=d.get("limitations", ""),
        )
    except (KeyError, TypeError, ValueError):
        return None


# ─────────────────────────────────────────────────────────────────────
# Apply frozen calibration  (raw -> global)
# ─────────────────────────────────────────────────────────────────────

def to_global(raw_bsv: dict[str, float], calib: GlobalCalibration
              ) -> dict[str, dict[str, float]]:
    """Deterministically map a raw BSV to global coordinates.

    Returns {axis: {"unbounded": float, "display": float}}.
    Never mutates raw_bsv.
    """
    out: dict[str, dict[str, float]] = {}
    for a in cfg.BSV_AXES:
        raw = float(raw_bsv.get(a, 0.0))
        c = calib.center.get(a, 0.0)
        s = max(calib.scale.get(a, 1.0), calib.scale_floor)
        z = (raw - c) / s
        disp = float(np.clip(z, -calib.clip, calib.clip))
        out[a] = {"unbounded": float(z), "display": disp}
    return out


def global_unbounded_vector(raw_bsv: dict[str, float],
                            calib: GlobalCalibration) -> np.ndarray:
    g = to_global(raw_bsv, calib)
    return np.array([g[a]["unbounded"] for a in cfg.BSV_AXES], float)


def global_display_dict(raw_bsv: dict[str, float],
                        calib: GlobalCalibration) -> dict[str, float]:
    g = to_global(raw_bsv, calib)
    return {a: g[a]["display"] for a in cfg.BSV_AXES}


# ─────────────────────────────────────────────────────────────────────
# Cohort-relative coordinates (EXPLORATORY ONLY — cohort-dependent!)
# ─────────────────────────────────────────────────────────────────────

def cohort_relative_zscores(raw_bsv_rows: list[dict[str, float]]
                            ) -> list[dict[str, float]]:
    """Within-set robust z-score of a *group* of raw BSVs.

    This is cohort-DEPENDENT by construction (center/scale come from the given
    group), so it is only valid for the labelled exploratory view. It is NOT a
    global coordinate and will change if the comparison set changes.
    """
    if not raw_bsv_rows:
        return []
    M = np.array([[float(r.get(a, 0.0)) for a in cfg.BSV_AXES]
                  for r in raw_bsv_rows], float)
    med = np.median(M, axis=0)
    mad = np.median(np.abs(M - med), axis=0) * 1.4826
    mad = np.where(mad < 1e-9, 1.0, mad)
    Z = (M - med) / mad
    return [{a: float(Z[i, j]) for j, a in enumerate(cfg.BSV_AXES)}
            for i in range(Z.shape[0])]
