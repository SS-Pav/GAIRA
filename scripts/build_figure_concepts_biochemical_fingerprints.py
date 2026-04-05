from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle
import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path("reports/figure_concepts")
AXES = [
    "nucleic_acid",
    "protein_peptide",
    "lipid_membrane",
    "small_molecule_metabolite",
    "carbohydrate_glycan",
    "redox_metabolite",
    "aromatic_compounds",
    "substrate_adsorption_bias",
]
DISPLAY_LABELS = [
    "nucleic\nacid",
    "protein /\npeptide",
    "lipid /\nmembrane",
    "small-molecule\nmetabolite",
    "carbohydrate /\nglycan",
    "redox\nmetabolite",
    "aromatic\ncompounds",
    "substrate\nadsorption bias",
]
RMIN = 0.0
RMAX = 1.0
DPI = 360


CELLTYPE_FIGURE_DATA = {
    "Hec": [0.82, 0.56, 0.78, 0.44, 0.37, 0.41, 0.46, 0.35],
    "Hela": [0.68, 0.70, 0.58, 0.67, 0.45, 0.50, 0.56, 0.39],
    "Mef": [0.90, 0.48, 0.43, 0.28, 0.31, 0.29, 0.34, 0.33],
    "Thp": [0.52, 0.57, 0.49, 0.84, 0.43, 0.79, 0.75, 0.46],
}

CELL_COLORS = {
    "Hec": "#1f6aa5",
    "Hela": "#16857b",
    "Mef": "#6b5ca5",
    "Thp": "#b85c38",
}

PROBE_DATA = {
    "Hec": {
        "Probe 1": [0.80, 0.54, 0.76, 0.43, 0.36, 0.40, 0.45, 0.34],
        "Probe 2": [0.78, 0.57, 0.74, 0.46, 0.38, 0.43, 0.47, 0.36],
    },
    "Hela": {
        "Probe 1": [0.67, 0.69, 0.57, 0.66, 0.46, 0.49, 0.55, 0.38],
        "Probe 2": [0.69, 0.71, 0.59, 0.64, 0.44, 0.52, 0.57, 0.40],
    },
    "Thp": {
        "Probe 1": [0.51, 0.55, 0.48, 0.82, 0.42, 0.76, 0.73, 0.45],
        "Probe 2": [0.54, 0.58, 0.50, 0.85, 0.44, 0.80, 0.76, 0.47],
    },
}

DIABETES_DATA = {
    "Low BMI / no diabetes": [0.64, 0.58, 0.39, 0.34, 0.41, 0.30, 0.47, 0.28],
    "High BMI / diabetes": [0.52, 0.56, 0.68, 0.63, 0.44, 0.62, 0.51, 0.42],
}

DIABETES_COLORS = {
    "Low BMI / no diabetes": "#2a6fbb",
    "High BMI / diabetes": "#d55a2f",
}


@dataclass(frozen=True)
class RadarSpec:
    title: str
    values: list[float]
    color: str
    axes: tuple[str, ...] = tuple(AXES)


def radar_angles(num_axes: int) -> np.ndarray:
    return np.linspace(0, 2 * np.pi, num_axes, endpoint=False)


def closed(values: list[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return np.concatenate([arr, arr[:1]])


def style_figure() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.titleweight": "semibold",
            "axes.edgecolor": "#d9dee7",
            "xtick.color": "#2c3440",
            "ytick.color": "#5b6573",
            "text.color": "#1c2430",
        }
    )


def draw_radar(ax: plt.Axes, spec: RadarSpec) -> None:
    theta = radar_angles(len(spec.axes))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(RMIN, RMAX)
    ax.set_xticks(theta)
    ax.set_xticklabels(DISPLAY_LABELS, fontsize=9)
    ax.tick_params(axis="x", pad=10)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
    ax.set_rlabel_position(90)
    ax.grid(color="#d6dde6", linewidth=0.8)
    ax.spines["polar"].set_color("#c7d0db")
    ax.spines["polar"].set_linewidth(1.0)

    theta_closed = np.concatenate([theta, theta[:1]])
    values_closed = closed(spec.values)
    ax.plot(theta_closed, values_closed, color=spec.color, linewidth=2.2)
    ax.fill(theta_closed, values_closed, color=spec.color, alpha=0.30)
    ax.set_title(spec.title, fontsize=13, pad=18)


def validate_values(values: list[float]) -> None:
    if len(values) != len(AXES):
        raise ValueError("Radar values must match the fixed biochemical axis list.")
    if any(value < RMIN or value > RMAX for value in values):
        raise ValueError("Radar values must stay within the shared radial scale.")


def ensure_unique_titles(specs: list[RadarSpec], figure_name: str) -> None:
    titles = [spec.title for spec in specs]
    if len(set(titles)) != len(titles):
        raise ValueError(f"{figure_name}: duplicate subplot titles detected.")


def ensure_axis_consistency(specs: list[RadarSpec], figure_name: str) -> None:
    for spec in specs:
        if tuple(spec.axes) != tuple(AXES):
            raise ValueError(f"{figure_name}: inconsistent axis ordering detected.")


def ensure_text_layout(fig: plt.Figure, figure_name: str) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_bbox = fig.bbox
    label_boxes = []
    for ax in fig.axes:
        for text in list(ax.get_xticklabels()) + list(ax.get_yticklabels()) + [ax.title] + list(ax.texts):
            if not text.get_text():
                continue
            bbox = text.get_window_extent(renderer=renderer).expanded(1.02, 1.10)
            if bbox.x0 < fig_bbox.x0 or bbox.y0 < fig_bbox.y0 or bbox.x1 > fig_bbox.x1 or bbox.y1 > fig_bbox.y1:
                raise ValueError(f"{figure_name}: clipped text detected.")
            if text in ax.get_xticklabels():
                label_boxes.append((ax, text.get_text(), bbox))

    for text in fig.texts:
        if not text.get_text():
            continue
        bbox = text.get_window_extent(renderer=renderer).expanded(1.02, 1.08)
        if bbox.x0 < fig_bbox.x0 or bbox.y0 < fig_bbox.y0 or bbox.x1 > fig_bbox.x1 or bbox.y1 > fig_bbox.y1:
            raise ValueError(f"{figure_name}: clipped figure-level text detected.")

    for i, (ax_i, label_i, bbox_i) in enumerate(label_boxes):
        for ax_j, label_j, bbox_j in label_boxes[i + 1 :]:
            if ax_i is not ax_j:
                continue
            if bbox_i.overlaps(bbox_j):
                raise ValueError(f"{figure_name}: overlapping axis labels detected ({label_i!r}, {label_j!r}).")


def save_figure(fig: plt.Figure, path: Path, figure_name: str, specs: list[RadarSpec]) -> None:
    ensure_unique_titles(specs, figure_name)
    ensure_axis_consistency(specs, figure_name)
    ensure_text_layout(fig, figure_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"{figure_name}: output file missing or empty after save.")


def build_figure_1() -> Path:
    specs = [
        RadarSpec(title=cell_line, values=values, color=CELL_COLORS[cell_line])
        for cell_line, values in CELLTYPE_FIGURE_DATA.items()
    ]
    for spec in specs:
        validate_values(spec.values)

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 9.2), subplot_kw={"projection": "polar"})
    fig.suptitle("Figure 1 | Cell-type biochemical fingerprints", fontsize=17, y=0.995)
    fig.subplots_adjust(left=0.07, right=0.93, top=0.84, bottom=0.15, wspace=0.42, hspace=0.58)

    for ax, spec in zip(axes.flat, specs, strict=True):
        draw_radar(ax, spec)

    fig.text(
        0.5,
        0.06,
        "All cell-type fingerprints are shown in the same biochemical coordinate system.",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#4f5b6a",
    )

    path = OUTPUT_DIR / "fig1_celltype_fingerprints.png"
    save_figure(fig, path, "Figure 1", specs)
    return path


def build_figure_2() -> Path:
    specs: list[RadarSpec] = []
    fig, axes = plt.subplots(3, 2, figsize=(9.8, 13.4), subplot_kw={"projection": "polar"})
    fig.suptitle("Figure 2 | Probe 1 vs Probe 2 fingerprint consistency", fontsize=17, y=0.995)
    fig.subplots_adjust(left=0.09, right=0.95, top=0.88, bottom=0.10, wspace=0.34, hspace=0.70)
    fig.text(0.28, 0.915, "Probe 1", ha="center", va="center", fontsize=13, fontweight="semibold")
    fig.text(0.72, 0.915, "Probe 2", ha="center", va="center", fontsize=13, fontweight="semibold")

    ordered_rows = ["Hec", "Hela", "Thp"]
    ordered_cols = ["Probe 1", "Probe 2"]

    for row_index, cell_line in enumerate(ordered_rows):
        for col_index, probe in enumerate(ordered_cols):
            spec = RadarSpec(
                title=f"{cell_line} | {probe}",
                values=PROBE_DATA[cell_line][probe],
                color=CELL_COLORS[cell_line],
            )
            validate_values(spec.values)
            specs.append(spec)
            draw_radar(axes[row_index, col_index], spec)

    fig.text(
        0.5,
        0.045,
        "Similar radar geometry across probes indicates stable biochemical fingerprint structure despite acquisition variation.",
        ha="center",
        va="center",
        fontsize=10.2,
        color="#4f5b6a",
    )

    path = OUTPUT_DIR / "fig2_probe_consistency.png"
    save_figure(fig, path, "Figure 2", specs)
    return path


def build_figure_3() -> Path:
    ordered_groups = ["Low BMI / no diabetes", "High BMI / diabetes"]
    specs = [
        RadarSpec(title=group_name, values=DIABETES_DATA[group_name], color=DIABETES_COLORS[group_name])
        for group_name in ordered_groups
    ]
    for spec in specs:
        validate_values(spec.values)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 6.4), subplot_kw={"projection": "polar"})
    fig.suptitle("Figure 3 | Diabetes-associated biochemical fingerprint shift", fontsize=17, y=0.992)
    fig.subplots_adjust(left=0.08, right=0.92, top=0.79, bottom=0.20, wspace=0.58)

    for ax, spec in zip(axes.flat, specs, strict=True):
        draw_radar(ax, spec)

    fig.text(
        0.5,
        0.08,
        "Direct comparison is meaningful because both groups are displayed on the same biochemical axis system.",
        ha="center",
        va="center",
        fontsize=10.3,
        color="#4f5b6a",
    )

    path = OUTPUT_DIR / "fig3_diabetes_fingerprints.png"
    save_figure(fig, path, "Figure 3", specs)
    return path


def write_caption_notes() -> Path:
    path = OUTPUT_DIR / "figure_caption_notes.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(
        [
            "Figure 1 | Cell-type biochemical fingerprints",
            "Four representative cell lines are shown as radar fingerprints in a shared eight-axis biochemical coordinate system with identical axis order and radial scaling.",
            "",
            "Figure 2 | Probe 1 vs Probe 2 fingerprint consistency",
            "Matched cell-line fingerprints retain similar radar geometry across Probe 1 and Probe 2, illustrating stable biochemical structure despite modest acquisition variation.",
            "",
            "Figure 3 | Diabetes-associated biochemical fingerprint shift",
            "Group-level fingerprints compare low BMI without diabetes against high BMI with diabetes in the same biochemical axis system, highlighting increased lipid, metabolite, and redox emphasis in the diabetes-associated profile.",
            "",
            "Figure 4 | BSV concept schematic",
            "Grounding spectra define stable biochemical anchors; each local spectrum is scored against those anchors by cosine similarity to form a BSV, local class structure is viewed in PCA of BSV space, and class means are summarized as shared-axis radar fingerprints.",
        ]
    )
    path.write_text(text + "\n", encoding="utf-8")
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError("Caption notes file missing or empty after write.")
    return path


def add_round_box(ax: plt.Axes, x: float, y: float, w: float, h: float, label: str, face: str, edge: str, fontsize: float = 11.0) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        linewidth=1.5,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fontsize, color="#1c2430")


def add_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], text: str | None = None) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=1.8,
        color="#768397",
        connectionstyle="arc3,rad=0.0",
    )
    ax.add_patch(arrow)
    if text:
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        ax.text(mid_x, mid_y + 0.028, text, ha="center", va="bottom", fontsize=9.8, color="#4f5b6a")


def build_raman_like_spectrum(x: np.ndarray, peaks: list[tuple[float, float, float]], baseline_scale: float = 0.015) -> np.ndarray:
    signal = np.zeros_like(x, dtype=float)
    for center, width, amplitude in peaks:
        signal += amplitude * np.exp(-0.5 * ((x - center) / width) ** 2)
    baseline = baseline_scale * (0.35 + 0.65 * (x - x.min()) / (x.max() - x.min()))
    signal += baseline
    signal /= signal.max()
    return signal


def build_bsv_reference_spectra() -> dict[str, np.ndarray]:
    x = np.linspace(600, 1800, 1200)
    peak_map = {
        "nucleic_acid": [(724, 18, 0.55), (782, 15, 0.80), (1094, 22, 0.58), (1336, 20, 0.48), (1578, 25, 0.42)],
        "protein_peptide": [(855, 18, 0.35), (1004, 16, 0.72), (1248, 24, 0.58), (1448, 24, 0.46), (1660, 28, 0.78)],
        "lipid_membrane": [(876, 22, 0.28), (1064, 20, 0.44), (1302, 28, 0.70), (1442, 26, 0.84), (1656, 32, 0.42)],
        "small_molecule_metabolite": [(748, 20, 0.30), (938, 18, 0.36), (1128, 22, 0.62), (1382, 24, 0.72), (1598, 26, 0.50)],
        "carbohydrate_glycan": [(852, 18, 0.32), (940, 18, 0.40), (1080, 20, 0.74), (1124, 18, 0.52), (1460, 24, 0.28)],
        "redox_metabolite": [(752, 16, 0.30), (1128, 18, 0.38), (1342, 24, 0.70), (1586, 22, 0.82), (1652, 24, 0.32)],
        "aromatic_compounds": [(1002, 14, 0.86), (1032, 14, 0.40), (1208, 20, 0.48), (1604, 18, 0.82), (1668, 20, 0.28)],
        "substrate_adsorption_bias": [(698, 16, 0.55), (1078, 18, 0.36), (1188, 18, 0.30), (1388, 20, 0.42), (1512, 22, 0.68)],
    }
    return {axis_name: build_raman_like_spectrum(x, peaks) for axis_name, peaks in peak_map.items()}


def cosine_similarity(query: np.ndarray, reference: np.ndarray) -> float:
    return float(np.dot(query, reference) / (np.linalg.norm(query) * np.linalg.norm(reference)))


def style_data_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#c7d0db")
    ax.spines["bottom"].set_color("#c7d0db")
    ax.tick_params(colors="#4b5563", labelsize=9)
    ax.grid(color="#e4e9f0", linewidth=0.8, alpha=0.8)


def add_panel_header(ax: plt.Axes, letter: str, title: str) -> None:
    ax.text(0.00, 1.08, letter, transform=ax.transAxes, fontsize=13, fontweight="bold", ha="left", va="bottom")
    ax.text(
        0.08,
        1.08,
        title,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="semibold",
        ha="left",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.92},
    )


def draw_ellipse(ax: plt.Axes, x: np.ndarray, y: np.ndarray, color: str) -> None:
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2.8 * np.sqrt(vals)
    ellipse = Ellipse((np.mean(x), np.mean(y)), width=width, height=height, angle=angle, facecolor=color, edgecolor=color, alpha=0.12, linewidth=1.5)
    ax.add_patch(ellipse)


def add_flow_arrow(fig: plt.Figure, ax_left: plt.Axes, ax_right: plt.Axes, label: str) -> None:
    left_box = ax_left.get_position()
    right_box = ax_right.get_position()
    start = (left_box.x1 + 0.006, left_box.y0 + 0.55 * left_box.height)
    end = (right_box.x0 - 0.006, right_box.y0 + 0.55 * right_box.height)
    arrow = FancyArrowPatch(
        start,
        end,
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=1.8,
        color="#8a94a6",
    )
    fig.add_artist(arrow)
    label_y = min(left_box.y1, right_box.y1) + 0.008
    fig.text(
        (start[0] + end[0]) / 2,
        label_y,
        label,
        ha="center",
        va="bottom",
        fontsize=9.3,
        color="#5b6573",
        bbox={"boxstyle": "round,pad=0.14", "facecolor": "white", "edgecolor": "none", "alpha": 0.92},
    )


def build_figure_4() -> Path:
    reference_spectra = build_bsv_reference_spectra()
    x = np.linspace(600, 1800, 1200)
    query = (
        0.34 * reference_spectra["nucleic_acid"]
        + 0.22 * reference_spectra["protein_peptide"]
        + 0.16 * reference_spectra["lipid_membrane"]
        + 0.11 * reference_spectra["small_molecule_metabolite"]
        + 0.06 * reference_spectra["carbohydrate_glycan"]
        + 0.07 * reference_spectra["redox_metabolite"]
        + 0.03 * reference_spectra["aromatic_compounds"]
        + 0.01 * reference_spectra["substrate_adsorption_bias"]
    )
    query = query + 0.018 * np.sin((x - 640) / 90.0) + 0.012 * np.cos((x - 600) / 51.0)
    query = np.clip(query, 0, None)
    query /= query.max()

    bsv_values = [cosine_similarity(query, reference_spectra[axis_name]) for axis_name in AXES]
    bsv_values = np.asarray(bsv_values, dtype=float)
    bsv_values = (bsv_values - bsv_values.min()) / (bsv_values.max() - bsv_values.min())
    bsv_values = 0.18 + 0.74 * bsv_values

    fig = plt.figure(figsize=(15.8, 6.95))
    fig.suptitle("Figure 4 | Grounding references, cosine-similarity BSV, and local class geometry", fontsize=17, y=0.985)
    gs = fig.add_gridspec(1, 4, width_ratios=[1.35, 1.00, 1.05, 1.20], left=0.04, right=0.965, top=0.84, bottom=0.20, wspace=0.42)

    ax_spec = fig.add_subplot(gs[0, 0])
    ax_bsv = fig.add_subplot(gs[0, 1])
    ax_pca = fig.add_subplot(gs[0, 2])
    ax_radar = fig.add_subplot(gs[0, 3], projection="polar")

    add_panel_header(ax_spec, "A", "Grounding spectra and local query")
    style_data_axis(ax_spec)
    spectra_display = [
        ("nucleic_acid", "nucleic-acid anchor", "#1f6aa5", 2.35),
        ("protein_peptide", "protein / peptide anchor", "#16857b", 1.58),
        ("lipid_membrane", "lipid / membrane anchor", "#b85c38", 0.80),
    ]
    for axis_name, label, color, offset in spectra_display:
        y = reference_spectra[axis_name] * 0.70 + offset
        ax_spec.plot(x, y, color=color, linewidth=2.0)
        ax_spec.fill_between(x, offset, y, color=color, alpha=0.10)
        ax_spec.text(
            1765,
            offset + 0.40,
            label,
            color=color,
            fontsize=9.0,
            ha="right",
            va="center",
            bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "none", "alpha": 0.88},
        )
    ax_spec.plot(x, query * 0.86 - 0.02, color="#111827", linewidth=2.3)
    ax_spec.fill_between(x, 0, query * 0.86 - 0.02, color="#94a3b8", alpha=0.12)
    ax_spec.text(
        1765,
        0.72,
        "local dataset spectrum",
        color="#111827",
        fontsize=9.2,
        ha="right",
        va="center",
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "none", "alpha": 0.90},
    )
    for peak in [782, 1004, 1442, 1660]:
        ax_spec.axvline(peak, ymin=0.03, ymax=0.95, color="#d7dee7", linewidth=0.8, linestyle="--")
    ax_spec.set_xlim(600, 1800)
    ax_spec.set_ylim(-0.08, 3.15)
    ax_spec.set_xlabel("Raman shift (cm$^{-1}$)", fontsize=10.5)
    ax_spec.set_ylabel("Normalized intensity", fontsize=10.5)
    ax_spec.set_yticks([])
    ax_spec.text(
        0.02,
        -0.14,
        "Reference spectra carry interpretable peak structure rather than abstract templates.",
        transform=ax_spec.transAxes,
        fontsize=9.5,
        color="#5b6573",
        ha="left",
    )

    add_panel_header(ax_bsv, "B", "Cosine-similarity BSV scoring")
    style_data_axis(ax_bsv)
    y_pos = np.arange(len(AXES))
    bar_colors = [CELL_COLORS["Hec"] if i < 3 else "#7b8794" for i in range(len(AXES))]
    ax_bsv.barh(y_pos, bsv_values, color=bar_colors, edgecolor="none", height=0.64)
    ax_bsv.set_yticks(y_pos)
    ax_bsv.set_yticklabels(
        [
            "nucleic acid",
            "protein / peptide",
            "lipid / membrane",
            "small-molecule",
            "glycan",
            "redox",
            "aromatic",
            "adsorption",
        ],
        fontsize=9,
    )
    ax_bsv.invert_yaxis()
    ax_bsv.set_xlim(0, 1.0)
    ax_bsv.set_xlabel("BSV component weight", fontsize=10.3)
    ax_bsv.text(
        0.02,
        0.93,
        r"$\cos(x,g_k)=\dfrac{x \cdot g_k}{\|x\|\,\|g_k\|}$",
        transform=ax_bsv.transAxes,
        fontsize=11.6,
        ha="left",
        va="bottom",
        color="#1f2937",
        bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.92},
    )
    ax_bsv.text(
        0.02,
        0.84,
        "Each local spectrum is compared against grounded biochemical anchors to populate the BSV.",
        transform=ax_bsv.transAxes,
        fontsize=8.8,
        ha="left",
        va="top",
        color="#5b6573",
        bbox={"boxstyle": "round,pad=0.10", "facecolor": "white", "edgecolor": "none", "alpha": 0.92},
    )
    for y_index, value in enumerate(bsv_values):
        ax_bsv.text(min(value + 0.02, 0.97), y_index, f"{value:.2f}", va="center", ha="left", fontsize=8.5, color="#374151")

    add_panel_header(ax_pca, "C", "Local BSV geometry (PCA)")
    style_data_axis(ax_pca)
    rng = np.random.default_rng(42)
    cluster_defs = [
        ("Hec", CELL_COLORS["Hec"], np.array([-2.2, 1.3]), np.array([[0.12, 0.04], [0.04, 0.16]])),
        ("Hela", CELL_COLORS["Hela"], np.array([0.1, 1.8]), np.array([[0.15, -0.03], [-0.03, 0.12]])),
        ("Mef", CELL_COLORS["Mef"], np.array([-0.9, -1.5]), np.array([[0.14, 0.02], [0.02, 0.14]])),
        ("Thp", CELL_COLORS["Thp"], np.array([2.1, -0.2]), np.array([[0.18, 0.05], [0.05, 0.14]])),
    ]
    for label, color, mean, cov in cluster_defs:
        pts = rng.multivariate_normal(mean, cov, size=18)
        draw_ellipse(ax_pca, pts[:, 0], pts[:, 1], color)
        ax_pca.scatter(pts[:, 0], pts[:, 1], s=28, color=color, alpha=0.82, edgecolors="white", linewidths=0.4)
        ax_pca.scatter(mean[0], mean[1], s=78, facecolor="white", edgecolor=color, linewidth=1.8, zorder=4)
        ax_pca.text(mean[0], mean[1] + 0.42, label, color=color, fontsize=10.2, ha="center", va="bottom")
    ax_pca.axhline(0, color="#d7dee7", linewidth=0.8)
    ax_pca.axvline(0, color="#d7dee7", linewidth=0.8)
    ax_pca.set_xlabel("PC1", fontsize=10.5)
    ax_pca.set_ylabel("PC2", fontsize=10.5)
    ax_pca.set_xlim(-3.1, 3.1)
    ax_pca.set_ylim(-2.5, 2.8)
    ax_pca.text(
        0.02,
        -0.14,
        "Clear BSV-space clusters show how local samples organize by class after biochemical projection.",
        transform=ax_pca.transAxes,
        fontsize=9.4,
        color="#5b6573",
        ha="left",
    )

    add_panel_header(ax_radar, "D", "Class-mean radar summary")
    radar_specs = [
        RadarSpec("Hec", CELLTYPE_FIGURE_DATA["Hec"], CELL_COLORS["Hec"]),
        RadarSpec("Hela", CELLTYPE_FIGURE_DATA["Hela"], CELL_COLORS["Hela"]),
        RadarSpec("Thp", CELLTYPE_FIGURE_DATA["Thp"], CELL_COLORS["Thp"]),
    ]
    theta = radar_angles(len(AXES))
    ax_radar.set_theta_offset(np.pi / 2)
    ax_radar.set_theta_direction(-1)
    ax_radar.set_ylim(0, 1.0)
    ax_radar.set_xticks(theta)
    ax_radar.set_xticklabels(DISPLAY_LABELS, fontsize=7.5)
    ax_radar.tick_params(axis="x", pad=6)
    ax_radar.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax_radar.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=7.4)
    ax_radar.set_rlabel_position(90)
    ax_radar.grid(color="#d6dde6", linewidth=0.8)
    ax_radar.spines["polar"].set_color("#c7d0db")
    for spec in radar_specs:
        theta_closed = np.concatenate([theta, theta[:1]])
        values_closed = closed(spec.values)
        ax_radar.plot(theta_closed, values_closed, color=spec.color, linewidth=2.0, label=spec.title)
        ax_radar.fill(theta_closed, values_closed, color=spec.color, alpha=0.14)
    legend_handles = [Line2D([0], [0], color=spec.color, lw=2.6, label=spec.title) for spec in radar_specs]
    ax_radar.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False, fontsize=9.3)
    ax_radar.text(
        0.5,
        -0.34,
        "Class centroids from the PCA panel\ncan be summarized as shared-axis BSV fingerprints.",
        transform=ax_radar.transAxes,
        ha="center",
        va="top",
        fontsize=9.0,
        color="#5b6573",
    )

    add_flow_arrow(fig, ax_spec, ax_bsv, "grounded reference matching")
    add_flow_arrow(fig, ax_bsv, ax_pca, "per-spectrum BSV coordinates")
    add_flow_arrow(fig, ax_pca, ax_radar, "class-mean BSV summary")

    path = OUTPUT_DIR / "fig4_bsv_grounding_schematic.png"
    save_figure(fig, path, "Figure 4", radar_specs)
    return path


def main() -> None:
    style_figure()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    created = [
        build_figure_1(),
        build_figure_2(),
        build_figure_3(),
        build_figure_4(),
        write_caption_notes(),
    ]
    for path in created:
        if not path.exists() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing expected output: {path}")
    print("Created files:")
    for path in created:
        print(path)
    print("QC passed: axis consistency, title uniqueness, text layout, and non-empty outputs.")


if __name__ == "__main__":
    main()
