"""Calibration contrast registry.

Each entry is a controlled perturbation with a known expected biochemical
direction. These are NOT disease cohorts — they are validation targets for
GAIRA's direct spectral → BSV pipeline.

Expected directions use: "up", "down", or "flat" (explicitly expected to
not change). Axes not listed are unconstrained.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CalibrationContrast:
    contrast_id: str
    display_name: str
    dataset_id: str              # source dataset id (cross-ref to main registry)
    loader_id: str               # key into calibration.loaders
    sample_family: str           # e.g. "serum"
    substrate: str               # e.g. "plasmonic_paper_Ag", "Ag_colloid"
    perturbation_type: str       # "spiking" | "enzymatic_depletion" | "titration"
    control_cohort: str
    perturbed_cohort: str
    expected_directions: dict[str, str] = field(default_factory=dict)
    confound_axes: list[str] = field(default_factory=list)
    notes: str = ""
    provenance: str = ""


# ─────────────────────────────────────────────────────────────────────
# Registered calibration contrasts
# ─────────────────────────────────────────────────────────────────────

CALIBRATION_REGISTRY: list[CalibrationContrast] = [

    CalibrationContrast(
        contrast_id="cspp_fig7_hypoxanthine_spike",
        display_name="CSPP Fig7 — Hypoxanthine spike into serum",
        dataset_id="cspp_serum",
        loader_id="cspp_fig7",
        sample_family="serum",
        substrate="plasmonic_paper_Ag",
        perturbation_type="spiking",
        control_cohort="Bkg",
        perturbed_cohort="Hyp",
        expected_directions={
            "purine_nucleotide": "up",
        },
        confound_axes=["aromatic_amino_acid"],
        notes=(
            "Hypoxanthine is a purine. Its dominant SERS ring-breathing mode "
            "near 725 cm⁻¹ falls in the 700-740 window which maps directly to "
            "the purine_nucleotide axis. Aromatic_AA is a potential confound "
            "because ring modes ~620-660 cm⁻¹ can leak across."
        ),
        provenance="Zenodo 5644790 / CSPP serum methodology, Figure 7 subset.",
    ),

    CalibrationContrast(
        contrast_id="cspp_fig7_ergothioneine_spike",
        display_name="CSPP Fig7 — Ergothioneine spike into serum",
        dataset_id="cspp_serum",
        loader_id="cspp_fig7",
        sample_family="serum",
        substrate="plasmonic_paper_Ag",
        perturbation_type="spiking",
        control_cohort="Bkg",
        perturbed_cohort="Erg",
        expected_directions={
            # Ergothioneine is a sulfur-containing imidazole derivative.
            # Strong SERS near 720 cm⁻¹ (imidazole) leaks into purine window;
            # no clean redox_metabolite axis mapping in the 22-window panel.
            "purine_nucleotide": "up",
        },
        confound_axes=["aromatic_amino_acid", "redox_metabolite"],
        notes=(
            "Ergothioneine is a deliberate weak-identifiability case: its "
            "imidazole ring mode near 720 cm⁻¹ falls in the purine_nucleotide "
            "window, and its thione/sulfur signature has no clean mapping in "
            "the current 8-axis panel. This contrast is expected to reveal a "
            "panel limitation rather than a pipeline failure."
        ),
        provenance="Zenodo 5644790 / CSPP serum methodology, Figure 7 subset.",
    ),

    CalibrationContrast(
        contrast_id="uricase_sigma_depletion",
        display_name="Uricase depletion — commercial serum (Sigma)",
        dataset_id="serum_ag_colloids",
        loader_id="serum_ag_uricase",
        sample_family="serum",
        substrate="Ag_colloid",
        perturbation_type="enzymatic_depletion",
        control_cohort="SerumSigma",
        perturbed_cohort="SerumSigma+Enzyme",
        expected_directions={
            "purine_nucleotide": "down",
        },
        confound_axes=["aromatic_amino_acid", "glycan_carbohydrate"],
        notes=(
            "Uricase converts uric acid (a purine derivative, dominant serum "
            "SERS analyte ~635/890/1130 cm⁻¹) to allantoin. Purine_nucleotide "
            "axis should drop. Uric acid's ~635 cm⁻¹ ring mode falls in the "
            "aromatic_AA window and ~890 in the glycan window — both may "
            "also decrease as confounds of the purine signal."
        ),
        provenance="Serum Ag colloids Zenodo archive, dataset uricase subset.",
    ),

    CalibrationContrast(
        contrast_id="uricase_spiked_hypoxanthine_serum",
        display_name="Hypoxanthine-spiked serum — Sigma vs spiked",
        dataset_id="serum_ag_colloids",
        loader_id="serum_ag_uricase",
        sample_family="serum",
        substrate="Ag_colloid",
        perturbation_type="spiking",
        control_cohort="SerumSigma",
        perturbed_cohort="Serumspiked",
        expected_directions={
            "purine_nucleotide": "up",
        },
        confound_axes=["aromatic_amino_acid"],
        notes=(
            "Hypoxanthine spike into commercial serum. Purine_nucleotide "
            "expected to increase vs unspiked Sigma serum."
        ),
        provenance="Serum Ag colloids Zenodo archive, dataset uricase subset.",
    ),

    CalibrationContrast(
        contrast_id="ergothioneine_titration_top_vs_zero",
        display_name="Ergothioneine titration — 2.0 µM vs 0.0 µM in serum",
        dataset_id="ergothioneine_serum",
        loader_id="ergothioneine_titration",
        sample_family="serum",
        substrate="Ag_colloid",
        perturbation_type="titration",
        control_cohort="erg_0p0_uM",
        perturbed_cohort="erg_2p0_uM",
        expected_directions={
            # Low-concentration titration at 2 µM is near the physiological
            # detection floor for SERS. Expect only weak, leaky signal — and
            # declare it up front.
            "purine_nucleotide": "up",
        },
        confound_axes=["aromatic_amino_acid", "protein_backbone"],
        notes=(
            "Low-concentration (µM) ergothioneine titration. Expected to be "
            "a weak-recovery case: the analyte's imidazole ring mode ~720 "
            "cm⁻¹ lights purine_nucleotide, but at 2 µM the effect size may "
            "be below the between-spectrum serum variability. Included to "
            "quantify the floor."
        ),
        provenance="Zenodo 13791050 — ergothioneine-in-serum SERS calibration.",
    ),
]

_BY_ID = {c.contrast_id: c for c in CALIBRATION_REGISTRY}


def list_contrasts() -> list[CalibrationContrast]:
    return list(CALIBRATION_REGISTRY)


def get_contrast(contrast_id: str) -> CalibrationContrast:
    if contrast_id not in _BY_ID:
        raise KeyError(
            f"Unknown calibration contrast '{contrast_id}'. "
            f"Registered: {sorted(_BY_ID)}"
        )
    return _BY_ID[contrast_id]
