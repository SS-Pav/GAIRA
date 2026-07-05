"""GAIRA drug-detection module (grounding-only, toggleable, parallel).

Default is OFF: `enable_drug_detection=False` everywhere. Core GAIRA outputs
are identical to baseline unless the caller explicitly enables this layer.
"""
from gaira.drug_detection.otc_mss_detector import (
    OTCMSSDetector,
    OTCDetectionResult,
    OTCTemplate,
    DEFAULT_OTC_REGISTRY,
    run_drug_detection_layer,
    load_config_flag_from_yaml,
    describe_layer_output,
    PURE_CONTEXT_MIN_SCORE,
    PURE_CONTEXT_MIN_MARGIN,
)

__all__ = [
    "OTCMSSDetector",
    "OTCDetectionResult",
    "OTCTemplate",
    "DEFAULT_OTC_REGISTRY",
    "run_drug_detection_layer",
    "load_config_flag_from_yaml",
    "describe_layer_output",
    "PURE_CONTEXT_MIN_SCORE",
    "PURE_CONTEXT_MIN_MARGIN",
]
