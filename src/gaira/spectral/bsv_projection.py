"""
BSV projection — maps spectral window features to 8-component biochemical
state vectors. Pure spectral derivation, no literature priors.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gaira.spectral.window_panel import BSV_COMPONENTS, WINDOW_DEFS


@dataclass
class CohortBSV:
    """BSV profile for a single cohort."""
    cohort: str
    n_spectra: int
    mean_bsv: dict[str, float]        # component -> mean value
    std_bsv: dict[str, float]         # component -> std
    sample_bsv: np.ndarray             # (n_spectra, 8)


def project_to_bsv(
    window_features: np.ndarray,
) -> np.ndarray:
    """Project window features to 8-component BSV.

    Each BSV component = mean of its mapped windows.

    Args:
        window_features: Shape (n_spectra, 22).

    Returns:
        BSV matrix, shape (n_spectra, 8).
    """
    n = window_features.shape[0]
    bsv = np.zeros((n, len(BSV_COMPONENTS)))

    for ci, comp in enumerate(BSV_COMPONENTS):
        win_indices = [j for j, (_, _, _, c) in enumerate(WINDOW_DEFS) if c == comp]
        if win_indices:
            bsv[:, ci] = window_features[:, win_indices].mean(axis=1)

    return bsv


def compute_cohort_bsvs(
    bsv_matrix: np.ndarray,
    cohorts: np.ndarray,
) -> dict[str, CohortBSV]:
    """Compute BSV profiles per cohort.

    Args:
        bsv_matrix: Shape (n_spectra, 8).
        cohorts: Shape (n_spectra,) string labels.

    Returns:
        {cohort_name: CohortBSV}
    """
    result = {}
    for cohort in sorted(set(cohorts)):
        mask = cohorts == cohort
        samples = bsv_matrix[mask]
        means = samples.mean(axis=0)
        stds = samples.std(axis=0)

        result[cohort] = CohortBSV(
            cohort=cohort,
            n_spectra=int(mask.sum()),
            mean_bsv={BSV_COMPONENTS[i]: round(float(means[i]), 6) for i in range(8)},
            std_bsv={BSV_COMPONENTS[i]: round(float(stds[i]), 6) for i in range(8)},
            sample_bsv=samples,
        )

    return result


def compute_deltas(
    cohort_bsvs: dict[str, CohortBSV],
    reference: str = "healthy_control",
) -> dict[str, dict[str, float]]:
    """Compute BSV deltas relative to a reference cohort.

    Args:
        cohort_bsvs: Output from compute_cohort_bsvs().
        reference: Name of the reference cohort.

    Returns:
        {cohort_name: {component: delta_value}}
    """
    if reference not in cohort_bsvs:
        return {}

    ref_bsv = cohort_bsvs[reference]
    deltas = {}
    for cohort, cbsv in cohort_bsvs.items():
        if cohort == reference:
            continue
        deltas[cohort] = {
            comp: round(cbsv.mean_bsv[comp] - ref_bsv.mean_bsv[comp], 6)
            for comp in BSV_COMPONENTS
        }

    return deltas
