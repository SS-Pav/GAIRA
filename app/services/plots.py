from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import sparse
from scipy.sparse.linalg import spsolve


sns.set_theme(style="whitegrid", context="talk")


def _as_numpy(values: Sequence[float]) -> np.ndarray:
    return np.asarray(list(values), dtype=float)


def _fallback_asls(values: np.ndarray, lam: float = 1e6, p: float = 0.01, niter: int = 15) -> np.ndarray:
    length = len(values)
    if length < 3:
        return np.zeros_like(values, dtype=float)
    diff = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(length - 2, length), format="csc")
    weights = np.ones(length, dtype=float)
    baseline = np.zeros_like(values, dtype=float)
    for _ in range(niter):
        weight_matrix = sparse.spdiags(weights, 0, length, length)
        system = weight_matrix + lam * (diff.T @ diff)
        baseline = spsolve(system, weights * values)
        weights = p * (values > baseline) + (1.0 - p) * (values < baseline)
    return np.asarray(baseline, dtype=float)


def apply_visual_baseline_correction(
    x_values: Sequence[float],
    y_values: Sequence[float],
) -> tuple[list[float], list[float], str]:
    x_arr = _as_numpy(x_values)
    y_arr = _as_numpy(y_values)
    if len(x_arr) < 10:
        return x_arr.tolist(), y_arr.tolist(), "none"

    try:
        from pybaselines.whittaker import asls as pybaselines_asls  # type: ignore

        baseline, _ = pybaselines_asls(y_arr, lam=1e6, p=0.01)
        method = "AsLS"
    except Exception:
        baseline = _fallback_asls(y_arr)
        method = "AsLS-fallback"

    corrected = y_arr - np.asarray(baseline, dtype=float)
    return x_arr.tolist(), corrected.tolist(), method


def _prepare_display_values(
    x_values: Sequence[float],
    y_values: Sequence[float],
    apply_baseline_correction: bool,
) -> tuple[np.ndarray, np.ndarray, str | None]:
    x_arr = _as_numpy(x_values)
    y_arr = _as_numpy(y_values)
    if not apply_baseline_correction:
        return x_arr, y_arr, None
    corrected_x, corrected_y, method = apply_visual_baseline_correction(x_arr.tolist(), y_arr.tolist())
    return _as_numpy(corrected_x), _as_numpy(corrected_y), method


def plot_spectrum(
    x_values: Sequence[float],
    y_values: Sequence[float],
    *,
    title: str,
    color: str = "#1d4ed8",
    apply_baseline_correction: bool = False,
) -> tuple[plt.Figure, str | None]:
    x_arr, y_arr, method = _prepare_display_values(x_values, y_values, apply_baseline_correction)
    fig, ax = plt.subplots(figsize=(8.8, 4.1))
    ax.plot(x_arr, y_arr, lw=2.0, color=color)
    ax.set_title(title)
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Intensity")
    return fig, method


def plot_spectrum_pair(
    left_x: Sequence[float],
    left_y: Sequence[float],
    right_x: Sequence[float],
    right_y: Sequence[float],
    *,
    left_label: str,
    right_label: str,
    show_difference: bool = False,
    apply_baseline_correction: bool = False,
) -> tuple[plt.Figure, str | None]:
    left_x_arr, left_y_arr, method = _prepare_display_values(left_x, left_y, apply_baseline_correction)
    right_x_arr, right_y_arr, _ = _prepare_display_values(right_x, right_y, apply_baseline_correction)
    fig, ax = plt.subplots(figsize=(9.4, 4.5))
    ax.plot(left_x_arr, left_y_arr, lw=2.0, color="#2563eb", label=left_label)
    ax.plot(right_x_arr, right_y_arr, lw=2.0, color="#dc2626", label=right_label)
    if show_difference:
        diff = right_y_arr - left_y_arr
        ax.plot(left_x_arr[: len(diff)], diff, lw=1.4, color="#111827", alpha=0.55, label="Difference")
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Intensity")
    ax.legend(loc="upper right", frameon=False)
    return fig, method


def plot_theme_bars(
    labels: list[str],
    values: list[float],
    *,
    title: str,
    color: str = "#0f766e",
    x_label: str = "Score",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.2, max(3.0, 0.56 * len(labels))))
    sns.barplot(x=values, y=labels, orient="h", color=color, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("")
    for index, value in enumerate(values):
        ax.text(float(value) + 0.01, index, f"{value:.2f}", va="center", fontsize=10)
    return fig


def plot_signed_bars(
    labels: list[str],
    values: list[float],
    *,
    title: str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.2, max(3.0, 0.56 * len(labels))))
    colors = ["#dc2626" if value > 0 else "#2563eb" for value in values]
    ax.barh(labels, values, color=colors)
    ax.axvline(0.0, color="#6b7280", lw=1.0)
    ax.set_title(title)
    ax.set_xlabel("H0T shift  <  0  >  CTR shift")
    ax.set_ylabel("")
    for index, value in enumerate(values):
        offset = 0.01 if value >= 0 else -0.01
        ha = "left" if value >= 0 else "right"
        ax.text(float(value) + offset, index, f"{value:.2f}", va="center", ha=ha, fontsize=10)
    return fig
