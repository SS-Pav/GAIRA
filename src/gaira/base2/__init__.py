"""gaira_base_2 — motif-aware successor to the frozen gaira_base axes-era engine.

Public API:

    from gaira.base2 import (
        # schema
        BIOLOGY_AXES_V11, GAIRA_BASE_AXES_V8, CONTROL_LANE,
        PROJECTION_V11_TO_V8, ALPHA_SUPPORTING,
        MotifSpec, AxisMapping, MotifDualStatus,
        MotifScore, AxisScore, AmbiguityLane, SpectrumResult,

        # engine
        score_spectrum, result_to_flat_dict, load_engine,

        # registry
        load_motif_registry, load_axis_mapping, load_dual_status,
        load_active_registry,

        # compatibility
        frozen_pilot_manifest, GAIRA_BASE_FROZEN_PILOT_FILES,

        # regime
        get_active_regime, AG_COLLOID_SERUM, REGIME_AG_COLLOID_SERUM,
    )

gaira_base is NOT modified by this package. See
``gaira_base_2_backward_compatibility_v1.md`` for exact guarantees.
"""
from gaira.base2.ambiguity import compute_ambiguity_lane  # noqa: F401
from gaira.base2.axis_engine import aggregate_to_11_axes  # noqa: F401
from gaira.base2.calibration_overlay import (  # noqa: F401
    AG_COLLOID_SERUM,
    DEFAULT_REGIME,
    REGIME_AG_COLLOID_SERUM,
    SUPPORTED_REGIMES,
    Regime,
    get_active_regime,
)
from gaira.base2.compatibility import (  # noqa: F401
    GAIRA_BASE_FROZEN_PILOT_FILES,
    frozen_pilot_manifest,
    sha256_file,
)
from gaira.base2.motif_engine import (  # noqa: F401
    compute_motif_activation,
    compute_motif_score,
    motif_belongs_to_ambiguity_lane,
    resolve_mapping_weight,
    resolve_status_calibration_weight,
    resolve_status_core_weight,
)
from gaira.base2.primitives import band_fires, band_intensity  # noqa: F401
from gaira.base2.projection import project_to_8_axes  # noqa: F401
from gaira.base2.registry import (  # noqa: F401
    DUAL_STATUS_TABLE,
    MAPPING_SKELETON_V1_1,
    MOTIF_REGISTRY_V1_2,
    load_active_registry,
    load_axis_mapping,
    load_dual_status,
    load_motif_registry,
)
from gaira.base2.report import (  # noqa: F401
    load_engine,
    result_to_flat_dict,
    score_spectrum,
)
from gaira.base2.schema import (  # noqa: F401
    ALPHA_SUPPORTING,
    AmbiguityLane,
    AxisMapping,
    AxisScore,
    BIOLOGY_AXES_V11,
    BAND_FLOOR,
    CALIBRATION_WEIGHT_BY_STATUS,
    CONTROL_LANE,
    CORE_WEIGHT_BY_STATUS,
    BandFamily,
    GAIRA_BASE_AXES_V8,
    MAPPING_WEIGHT_BY_TYPE,
    MotifDualStatus,
    MotifScore,
    MotifSpec,
    PROJECTION_V11_TO_V8,
    SpectrumResult,
)

__all__ = [
    # schema
    "ALPHA_SUPPORTING",
    "BAND_FLOOR",
    "BIOLOGY_AXES_V11",
    "CALIBRATION_WEIGHT_BY_STATUS",
    "CONTROL_LANE",
    "CORE_WEIGHT_BY_STATUS",
    "GAIRA_BASE_AXES_V8",
    "MAPPING_WEIGHT_BY_TYPE",
    "PROJECTION_V11_TO_V8",
    "AmbiguityLane",
    "AxisMapping",
    "AxisScore",
    "BandFamily",
    "MotifDualStatus",
    "MotifScore",
    "MotifSpec",
    "SpectrumResult",
    # engine
    "aggregate_to_11_axes",
    "compute_ambiguity_lane",
    "compute_motif_activation",
    "compute_motif_score",
    "motif_belongs_to_ambiguity_lane",
    "project_to_8_axes",
    "resolve_mapping_weight",
    "resolve_status_calibration_weight",
    "resolve_status_core_weight",
    "score_spectrum",
    "result_to_flat_dict",
    "load_engine",
    # primitives
    "band_fires",
    "band_intensity",
    # registry
    "DUAL_STATUS_TABLE",
    "MAPPING_SKELETON_V1_1",
    "MOTIF_REGISTRY_V1_2",
    "load_active_registry",
    "load_axis_mapping",
    "load_dual_status",
    "load_motif_registry",
    # compatibility
    "GAIRA_BASE_FROZEN_PILOT_FILES",
    "frozen_pilot_manifest",
    "sha256_file",
    # regime
    "AG_COLLOID_SERUM",
    "DEFAULT_REGIME",
    "REGIME_AG_COLLOID_SERUM",
    "SUPPORTED_REGIMES",
    "Regime",
    "get_active_regime",
]
