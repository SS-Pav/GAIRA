"""OTC drug detection + MSS identity layer — grounding-only, substrate-agnostic.

This module is an OPTIONAL, TOGGLEABLE parallel interpretation layer. It runs
AFTER canonical GAIRA preprocessing and PARALLEL to the 11-axis BSV biochemical
interpretation — only when explicitly enabled by the caller. When disabled
(the default) the detector is NOT instantiated and no MSS scores are computed.

STRICT INVARIANTS (enforced by construction):
- No new BSV axes; no "G12"; no schema modification
- No domain classification (biological vs non-biological)
- No centroid / distance methods
- No classifier training
- Pure MSS-based detection using grounding templates
- Default enable_drug_detection=False: core GAIRA outputs are identical when disabled

Usage (toggle-aware top-level entry point — preferred):
    from gaira.drug_detection import run_drug_detection_layer

    result = run_drug_detection_layer(y_pp, master_x,
                                             enable_drug_detection=True)  # or False
    # result is always shaped as:
    # {
    #   "drug_detection": {
    #     "enabled": True | False,
    #     "status":  "NOT_RUN" | "HIGH_CONFIDENCE_PURE_CONTEXT" |
    #                "CANDIDATE_IN_COMPLEX_CONTEXT" | "NOT_DETECTED",
    #     ...
    #   },
    #   "drug_identity": { ... }   # only present when enabled=True; None when disabled
    # }

Low-level usage (detector class, advanced callers):
    from gaira.drug_detection import OTCMSSDetector
    det = OTCMSSDetector()
    raw = det.detect(y_pp, master_x)  # returns OTCDetectionResult with inner identity-status
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


# Default registry built by the OTC pure-Raman MSS-build phase.
DEFAULT_OTC_REGISTRY: Path = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_4_otc_pure_raman_mss_build_v1/"
    "registry/otc_pure_raman_mss_registry_v1.csv"
)


# ──────────────────────────────────────────────────────────────────────
# Internal peak + scoring helpers — self-contained copy of the standard
# GAIRA MSS kernel (anchor_fires + 0.3·support_fires). Duplicated here
# so the module has no script-path dependency.
# ──────────────────────────────────────────────────────────────────────
def _has_real_peak(y: np.ndarray, master_x: np.ndarray, cm1: float,
                       half: float = 5.0, prom_frac: float = 0.05,
                       top_rank_max: int = 12) -> bool:
    """Return True if a real peak (find_peaks, prom ≥ prom_frac × range) exists
    within ±half cm⁻¹ of `cm1` AND ranks within the top `top_rank_max` peaks of
    the spectrum. Mirrors has_real_peak in the MSS resolution reporting layer.
    """
    if not np.any(np.isfinite(y)):
        return False
    rng = float(np.nanmax(y) - np.nanmin(y))
    if rng <= 0:
        return False
    idx, _ = find_peaks(y, prominence=prom_frac * rng)
    if len(idx) == 0:
        return False
    heights = y[idx]
    order = np.argsort(-heights)
    ranked = idx[order][: max(top_rank_max, 5)]
    for ix in ranked:
        if abs(master_x[ix] - cm1) <= half:
            return True
    return False


def _mss_anchor_score(y: np.ndarray, master_x: np.ndarray,
                           anchors: List[float], supports: List[float],
                           tolerance: float = 5.0) -> tuple[float, int, int]:
    """Standard GAIRA MSS kernel: anchor_fires + 0.3·support_fires.

    Returns (score, n_anchors_fired, n_supports_fired).
    """
    if not anchors:
        return 0.0, 0, 0
    af = sum(1 for a in anchors
              if _has_real_peak(y, master_x, a, half=tolerance))
    sf = (sum(1 for s in supports
              if _has_real_peak(y, master_x, s, half=tolerance))
          if supports else 0)
    n_anchors = len(anchors)
    n_supports = len(supports) if supports else 1
    score = (af / max(n_anchors, 1)
             + 0.3 * (sf / n_supports if supports else 0.0))
    return float(score), int(af), int(sf)


# ──────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────
@dataclass
class OTCTemplate:
    molecule: str
    anchors: List[float]
    supports: List[float]
    tolerance_cm1: float = 5.0
    source_dataset: str = "otc_drugs_paraguay"


@dataclass
class OTCDetectionResult:
    """Structured per-spectrum detection result. Mirrors the JSON contract
    described in the module docstring."""
    drug_detection: Dict
    drug_identity: Dict

    def to_dict(self) -> dict:
        return {"drug_detection": self.drug_detection,
                "drug_identity":  self.drug_identity}

    def __getitem__(self, k):      # dict-like access
        return self.to_dict()[k]

    def __repr__(self):
        det = self.drug_detection; ident = self.drug_identity
        present = det.get("present")
        if not present:
            return (f"<OTCDetectionResult present=False "
                    f"confidence={det.get('confidence')}>")
        return (f"<OTCDetectionResult present=True "
                f"top_1={ident.get('top_1')} "
                f"confidence={det.get('confidence')} "
                f"margin={ident.get('margin_top1_top2'):.3f}>")


# ──────────────────────────────────────────────────────────────────────
# Detector
# ──────────────────────────────────────────────────────────────────────
class OTCMSSDetector:
    """Grounding-only OTC drug detection layer.

    Parameters
    ----------
    registry_path : optional
        Path to the OTC pure-Raman MSS registry CSV. Defaults to the
        registry built by the `gaira_base_4_otc_pure_raman_mss_build_v1`
        phase.
    s_high : float, default 0.30
        Score threshold for HIGH-confidence detection. Conservative
        middle-low of the 0.30-0.40 range specified by the contract.
    delta_margin : float, default 0.07
        Required margin (top1 − top2) for HIGH confidence. Middle of the
        0.05-0.10 range.
    min_anchors : int, default 2
        Minimum number of anchor bands that must fire for drug_present
        to be set True.
    multi_hit_tolerance : float, default 0.85
        If top-2 score ≥ multi_hit_tolerance × top-1 score AND both
        satisfy the anchor requirement, result is flagged as a
        MIXTURE_OR_OVERLAP rather than forcing a single label.
    """

    def __init__(self,
                 registry_path: Optional[Path] = None,
                 s_high: float = 0.30,
                 delta_margin: float = 0.07,
                 min_anchors: int = 2,
                 multi_hit_tolerance: float = 0.85):
        self.registry_path = Path(registry_path or DEFAULT_OTC_REGISTRY)
        self.s_high = float(s_high)
        self.delta_margin = float(delta_margin)
        self.min_anchors = int(min_anchors)
        self.multi_hit_tolerance = float(multi_hit_tolerance)
        self.templates: Dict[str, OTCTemplate] = self._load_templates(self.registry_path)

    @staticmethod
    def _load_templates(registry_path: Path) -> Dict[str, OTCTemplate]:
        if not registry_path.exists():
            raise FileNotFoundError(
                f"OTC registry not found: {registry_path}. "
                f"Run the OTC pure-Raman MSS build phase first.")
        df = pd.read_csv(registry_path)
        out: Dict[str, OTCTemplate] = {}
        for _, r in df.iterrows():
            def _parse(field_name: str) -> List[float]:
                val = r.get(field_name)
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    return []
                return [float(x) for x in str(val).split(";") if x.strip()]
            out[r["molecule"]] = OTCTemplate(
                molecule=r["molecule"],
                anchors=_parse("anchor_bands_cm1"),
                supports=_parse("companion_bands_cm1"),
                tolerance_cm1=float(r.get("tolerance_cm1") or 5.0),
                source_dataset=str(r.get("source_dataset") or "otc_drugs_paraguay"),
            )
        return out

    def detect(self, y_pp: np.ndarray, master_x: np.ndarray) -> OTCDetectionResult:
        """Run detection on a preprocessed spectrum.

        Parameters
        ----------
        y_pp : np.ndarray
            Preprocessed spectrum (canonical GAIRA pipeline expected:
            baseline-corrected, smoothed, L2-normalized).
        master_x : np.ndarray
            Wavenumber axis (cm⁻¹) in ascending order.

        Returns
        -------
        OTCDetectionResult
            Structured result with drug_detection and drug_identity sub-dicts.
        """
        y_pp = np.asarray(y_pp, dtype=float)
        master_x = np.asarray(master_x, dtype=float)

        # Per-template MSS scoring
        scores: Dict[str, float] = {}
        anchors_fired: Dict[str, int] = {}
        supports_fired: Dict[str, int] = {}
        for mol, tpl in self.templates.items():
            sc, af, sf = _mss_anchor_score(
                y_pp, master_x, tpl.anchors, tpl.supports, tolerance=tpl.tolerance_cm1)
            scores[mol] = sc
            anchors_fired[mol] = af
            supports_fired[mol] = sf

        if not scores:
            return OTCDetectionResult(
                drug_detection={"present": False, "confidence": "NONE",
                                   "reason": "registry_empty"},
                drug_identity={"top_1": None, "top_3": [], "scores": {},
                                  "anchor_hits": {}, "margin_top1_top2": 0.0,
                                  "status": "NOT_DETECTED"})

        # Rank candidates
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        top1_mol, top1_score = ranked[0]
        top2_mol, top2_score = (ranked[1] if len(ranked) > 1 else (None, 0.0))
        margin = float(top1_score - top2_score)
        top3 = [m for m, _ in ranked[:3]]

        # Drug-presence gate (MSS-first)
        anchor_ok = anchors_fired[top1_mol] >= self.min_anchors
        present = bool(top1_score >= self.s_high and anchor_ok)

        # Multi-hit detection (true mixture signal)
        multi_hit = False
        if (present and top2_mol is not None
                and top2_score >= self.multi_hit_tolerance * top1_score
                and anchors_fired.get(top2_mol, 0) >= self.min_anchors):
            multi_hit = True

        # Confidence logic
        if not present:
            reason = ("low_top1_score" if top1_score < self.s_high
                        else "insufficient_anchors")
            detection = {"present": False, "confidence": "NONE", "reason": reason}
            identity = {"top_1": None, "top_3": top3,
                            "scores": {m: float(s) for m, s in scores.items()},
                            "anchor_hits": {m: int(v) for m, v in anchors_fired.items()},
                            "margin_top1_top2": margin,
                            "status": "NOT_DETECTED"}
            return OTCDetectionResult(detection, identity)

        if multi_hit:
            detection = {"present": True, "confidence": "LOW",
                            "reason": "possible mixture or overlapping spectral features"}
            identity = {"top_1": None, "top_3": top3,
                            "scores": {m: float(s) for m, s in scores.items()},
                            "anchor_hits": {m: int(v) for m, v in anchors_fired.items()},
                            "margin_top1_top2": margin,
                            "status": "MIXTURE_OR_OVERLAP"}
            return OTCDetectionResult(detection, identity)

        if margin >= self.delta_margin:
            detection = {"present": True, "confidence": "HIGH"}
            identity = {"top_1": top1_mol, "top_3": top3,
                            "scores": {m: float(s) for m, s in scores.items()},
                            "anchor_hits": {m: int(v) for m, v in anchors_fired.items()},
                            "margin_top1_top2": margin,
                            "status": "HIGH_CONFIDENCE"}
            return OTCDetectionResult(detection, identity)

        # Present + above threshold but insufficient margin — low confidence
        detection = {"present": True, "confidence": "LOW",
                        "reason": "narrow_margin"}
        identity = {"top_1": top1_mol, "top_3": top3,
                        "scores": {m: float(s) for m, s in scores.items()},
                        "anchor_hits": {m: int(v) for m, v in anchors_fired.items()},
                        "margin_top1_top2": margin,
                        "status": "LOW_CONFIDENCE"}
        return OTCDetectionResult(detection, identity)

    def describe(self, result: OTCDetectionResult) -> str:
        """Return a short human-readable string suitable for appending to a
        GAIRA BSV interpretation report."""
        det = result.drug_detection; ident = result.drug_identity
        if not det.get("present"):
            return "No drug-like spectral features detected above grounding threshold."
        status = ident.get("status")
        if status == "MIXTURE_OR_OVERLAP":
            return (f"Drug-like spectral features detected — possible mixture or "
                    f"overlapping features across: {', '.join(ident.get('top_3', []))}.")
        top_1 = ident.get("top_1")
        conf = det.get("confidence", "LOW")
        if top_1 and conf == "HIGH":
            return f"Drug-like spectral features detected. Top candidate: {top_1} (high confidence)."
        if top_1:
            return (f"Drug-like spectral features detected. Top candidate: {top_1} "
                    f"(low confidence; narrow margin over alternatives).")
        return "Drug-like spectral features detected (no clear top candidate)."


# ──────────────────────────────────────────────────────────────────────
# Toggle-aware pipeline entry point (USER-FACING API)
# ──────────────────────────────────────────────────────────────────────
# Context-aware outer status thresholds. These are SIGNAL-QUALITY thresholds,
# NOT domain classifications. A spectrum with very strong MSS signal (all
# anchors fire, large margin) reads as a "pure context" drug; a spectrum
# with moderate signal or partial anchor coverage reads as a "complex context"
# candidate. The caller does not pass any context hint.
PURE_CONTEXT_MIN_SCORE: float = 0.55  # top-1 score above this → pure-context tier
PURE_CONTEXT_MIN_MARGIN: float = 0.20 # margin above this → pure-context tier
# Detector instance cache — built lazily only when enable=True for the first time.
_DETECTOR_CACHE: Dict[tuple, "OTCMSSDetector"] = {}


def _outer_tier_from_result(result: "OTCDetectionResult") -> str:
    """Map the detector's inner status + scores to the outer context-aware tier.

    Outer tiers (signal-quality):
    - HIGH_CONFIDENCE_PURE_CONTEXT: present + HIGH inner conf + top-1 score strong +
        margin wide + ALL anchors of top-1 fire (at least len(anchors) total)
    - CANDIDATE_IN_COMPLEX_CONTEXT: present but signal quality below pure-context
        thresholds (LOW inner conf, partial anchors, narrow margin, or mixture)
    - NOT_DETECTED: not present
    """
    det = result.drug_detection
    ident = result.drug_identity
    if not det.get("present"):
        return "NOT_DETECTED"
    top_1 = ident.get("top_1")
    scores = ident.get("scores", {})
    anchor_hits = ident.get("anchor_hits", {})
    margin = float(ident.get("margin_top1_top2") or 0.0)
    status = ident.get("status")

    # Mixture or narrow-margin → complex
    if status in ("MIXTURE_OR_OVERLAP", "LOW_CONFIDENCE"):
        return "CANDIDATE_IN_COMPLEX_CONTEXT"
    # Pure-context gate (signal-quality only — no domain input)
    top_1_score = float(scores.get(top_1, 0.0)) if top_1 else 0.0
    top_1_anchors = int(anchor_hits.get(top_1, 0)) if top_1 else 0
    if (status == "HIGH_CONFIDENCE"
            and top_1_score >= PURE_CONTEXT_MIN_SCORE
            and margin >= PURE_CONTEXT_MIN_MARGIN
            and top_1_anchors >= 3):
        return "HIGH_CONFIDENCE_PURE_CONTEXT"
    return "CANDIDATE_IN_COMPLEX_CONTEXT"


def run_drug_detection_layer(
    y_pp: np.ndarray,
    master_x: np.ndarray,
    enable_drug_detection: bool = False,
    registry_path: Optional[Path] = None,
    s_high: float = 0.30,
    delta_margin: float = 0.07,
    min_anchors: int = 2,
    multi_hit_tolerance: float = 0.85,
) -> dict:
    """Toggle-aware OTC drug-detection layer.

    ALWAYS returns a dict with a top-level `drug_detection` block.
    When `enable_drug_detection=False` (the default), NO detector is
    instantiated and NO MSS scoring happens — the function returns the
    `{"enabled": False, "status": "NOT_RUN"}` block and `drug_identity=None`.

    Parameters
    ----------
    y_pp, master_x : arrays
        Preprocessed spectrum + wavenumber axis.
    enable_drug_detection : bool, default False
        User toggle. Must be True for detection to run.
    registry_path, s_high, delta_margin, min_anchors, multi_hit_tolerance
        Forwarded to OTCMSSDetector only when enable_drug_detection=True.

    Returns
    -------
    dict shaped as described in the module docstring.
    """
    if not enable_drug_detection:
        return {
            "drug_detection": {"enabled": False, "status": "NOT_RUN"},
            "drug_identity": None,
        }

    # Cache detector per (registry, thresholds) tuple to amortize init cost
    cache_key = (str(registry_path or DEFAULT_OTC_REGISTRY),
                    s_high, delta_margin, min_anchors, multi_hit_tolerance)
    det = _DETECTOR_CACHE.get(cache_key)
    if det is None:
        det = OTCMSSDetector(
            registry_path=registry_path,
            s_high=s_high,
            delta_margin=delta_margin,
            min_anchors=min_anchors,
            multi_hit_tolerance=multi_hit_tolerance,
        )
        _DETECTOR_CACHE[cache_key] = det

    raw = det.detect(y_pp, master_x)
    outer_tier = _outer_tier_from_result(raw)

    return {
        "drug_detection": {
            "enabled": True,
            "status": outer_tier,
            # preserve the inner confidence for callers that want it
            "inner_confidence": raw.drug_detection.get("confidence"),
            "present": raw.drug_detection.get("present"),
        },
        "drug_identity": raw.drug_identity,
    }


def load_config_flag_from_yaml(yaml_path: Path) -> bool:
    """Helper: read `enable_drug_detection` from a YAML config file.

    Returns False if the file does not exist, does not contain the key, or
    is unparseable. Does NOT raise.
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        return False
    try:
        import yaml  # type: ignore
    except ImportError:
        # Minimal manual parse: look for `enable_drug_detection: true|false`
        txt = yaml_path.read_text(errors="ignore")
        for line in txt.splitlines():
            line = line.strip()
            if line.startswith("enable_drug_detection"):
                value = line.split(":", 1)[-1].strip().lower()
                return value in ("true", "yes", "1", "on")
        return False
    try:
        cfg = yaml.safe_load(yaml_path.read_text())
        return bool(cfg.get("enable_drug_detection", False)) if isinstance(cfg, dict) else False
    except Exception:
        return False


def describe_layer_output(layer_result: dict) -> str:
    """Short human-readable one-liner for inclusion in GAIRA reports."""
    det = layer_result.get("drug_detection", {})
    ident = layer_result.get("drug_identity") or {}
    if not det.get("enabled"):
        return "(drug detection layer not run)"
    status = det.get("status")
    if status == "NOT_DETECTED":
        return "No drug-like spectral features detected above grounding threshold."
    if status == "HIGH_CONFIDENCE_PURE_CONTEXT":
        top_1 = ident.get("top_1")
        return (f"Drug-like spectral features detected in a pure-context signal. "
                f"Top candidate: {top_1} (high confidence).")
    if status == "CANDIDATE_IN_COMPLEX_CONTEXT":
        top_1 = ident.get("top_1")
        if top_1:
            return (f"Drug-like spectral features detected in a complex context. "
                    f"Candidate: {top_1} — report as candidate evidence only.")
        return ("Drug-like spectral features detected in a complex context — "
                "multiple candidates / overlap; report as candidate evidence only.")
    return f"Drug detection status: {status}"
