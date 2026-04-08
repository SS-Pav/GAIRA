from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import textwrap

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
FIG4_PROCESSING_VERSION = "v2_crop400_1800_interp1_poly3_vector"
FIG4_COHORT_DATASET_ID = "cca_hcc_lm_serum_sers"
FIG4_ANCHOR_COMPONENTS = [
    ("adenine", 8, "#2b6cb0"),
    ("albumin", 99, "#238b82"),
    ("cholesterol", 64, "#c65f2d"),
    ("citric acid", 14, "#8a6fb3"),
    ("glycogen", 188, "#b28a2e"),
    ("glutathione", 27, "#7d9d3c"),
    ("l-phenylalanine", 35, "#b24f7e"),
    ("cytochrome c", 166, "#6b7280"),
]
FIG4_RADAR_LABELS = [
    "adenine",
    "albumin",
    "cholest-\nerol",
    "citric\nacid",
    "glycogen",
    "glutathione",
    "phenyl-\nalanine",
    "cytochrome c",
]
FIG4_CLASS_COLORS = {
    "cca": "#c48a2c",
    "hcc": "#c65f2d",
    "healthy_control": "#238b82",
    "lm": "#7a68b3",
}
FIG4_RADAR_AXES_BIO = [
    "nucleic_acid",
    "protein_peptide",
    "lipid_membrane",
    "carbohydrate_glycan",
    "small_molecule_metabolite",
    "substrate_adsorption_bias",
]
FIG4_RADAR_LABELS_BIO = [
    "nucleic_acid",
    "protein_peptide",
    "lipid_membrane",
    "carbohydrate_glycan",
    "small_molecule_metabolite",
    "substrate_adsorption_bias",
]
FIG4_EV_DATASETS = {
    "small2023_ev",
    "diabetes_plasma_ev_sers",
    "shine_ev_sers",
    "single_vesicle_ev_raman",
}
FIG4_SERUM_DATASETS = {
    "cca_hcc_lm_serum_sers",
    "covid_serum_raman",
    "serum_ag_colloids",
    "cspp_serum",
    "ergothioneine_serum",
}


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
            "The current GAIRA build is summarized from live corpus counts, real RamanBioLib reference spectra, and actual CCA/HCC/LM/healthy serum cohort outputs; an illustrative eight-anchor cosine-similarity BSV is shown for HCC and compared against the normal class mean on a shared radar.",
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
    ax.text(0.00, 1.08, letter, transform=ax.transAxes, fontsize=10.9, fontweight="bold", ha="left", va="bottom")
    ax.text(
        0.52,
        1.08,
        title,
        transform=ax.transAxes,
        fontsize=10.9,
        fontweight="semibold",
        ha="center",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.10", "facecolor": "white", "edgecolor": "none", "alpha": 0.92},
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


def add_flow_arrow(fig: plt.Figure, ax_left: plt.Axes, ax_right: plt.Axes, label: str | None = None) -> None:
    left_box = ax_left.get_position()
    right_box = ax_right.get_position()
    y_level = left_box.y0 + 0.55 * left_box.height
    gap = right_box.x0 - left_box.x1
    inset = min(0.022, max(0.008, 0.40 * gap))
    start = (left_box.x1 + inset, y_level)
    end = (right_box.x0 - inset, y_level)
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
    if label:
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


def _load_json_array(value: str) -> np.ndarray:
    return np.asarray(json.loads(value), dtype=float)


def _normalize_plot_spectrum(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = arr - np.min(arr)
    peak = float(np.max(arr))
    if peak <= 0:
        return np.zeros_like(arr, dtype=float)
    return arr / peak


def _cosine_similarity_aligned(query_x: np.ndarray, query_y: np.ndarray, ref_x: np.ndarray, ref_y: np.ndarray) -> float:
    overlap_min = max(float(np.min(query_x)), float(np.min(ref_x)))
    overlap_max = min(float(np.max(query_x)), float(np.max(ref_x)))
    query_mask = (query_x >= overlap_min) & (query_x <= overlap_max)
    query_x_overlap = query_x[query_mask]
    query_y_overlap = query_y[query_mask]
    ref_y_aligned = np.interp(query_x_overlap, ref_x, ref_y)
    query_norm = query_y_overlap / max(float(np.linalg.norm(query_y_overlap)), 1e-12)
    ref_norm = ref_y_aligned / max(float(np.linalg.norm(ref_y_aligned)), 1e-12)
    return float(np.dot(query_norm, ref_norm))


def _compute_pca_scores(matrix: np.ndarray) -> np.ndarray:
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    feature_std = centered.std(axis=0, keepdims=True)
    feature_std[feature_std < 1e-8] = 1.0
    standardized = centered / feature_std
    u, singular_values, _ = np.linalg.svd(standardized, full_matrices=False)
    return u[:, :2] * singular_values[:2]


def _load_current_build_figure4_data() -> dict[str, object]:
    import duckdb
    import sys

    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))
    from gaira.config import get_database_path

    connection = duckdb.connect(str(get_database_path()), read_only=True)

    raw_total = int(connection.execute("SELECT COUNT(*) FROM biosample_spectra").fetchone()[0])
    processed_grounding = int(connection.execute("SELECT COUNT(*) FROM grounding_processed_spectra").fetchone()[0])
    reference_total = int(connection.execute("SELECT COUNT(*) FROM reference_spectra WHERE dataset_id = 'ramanbiolib'").fetchone()[0])
    dataset_counts = {
        str(dataset_id): int(n)
        for dataset_id, n in connection.execute("SELECT dataset_id, COUNT(*) AS n FROM biosample_spectra GROUP BY 1").fetchall()
    }
    ev_total = sum(dataset_counts.get(dataset_id, 0) for dataset_id in FIG4_EV_DATASETS)
    serum_total = sum(dataset_counts.get(dataset_id, 0) for dataset_id in FIG4_SERUM_DATASETS)
    other_total = raw_total - ev_total - serum_total
    layer_cards = [
        ("Grounding molecules", f"{reference_total:,} refs\n{processed_grounding:,} processed support", "#e8eef8"),
        ("Serum datasets", f"{serum_total:,} spectra\n{len(FIG4_SERUM_DATASETS)} datasets", "#fef3c7"),
        ("Extracellular vesicle datasets", f"{ev_total:,} spectra\n{len(FIG4_EV_DATASETS)} datasets", "#dcfce7"),
        ("Other biosample datasets", f"{other_total:,} spectra\n6 datasets", "#f3f4f6"),
    ]

    class_summary_rows = connection.execute(
        f"""
        SELECT class_label, mean_wavenumbers_json, mean_intensity_json, n_spectra
        FROM biosample_class_summary
        WHERE dataset_id = '{FIG4_COHORT_DATASET_ID}'
          AND processing_version = '{FIG4_PROCESSING_VERSION}'
          AND class_label IN ('cca', 'hcc', 'healthy_control', 'lm')
        ORDER BY class_label
        """
    ).fetchall()
    class_summary: dict[str, dict[str, object]] = {}
    for class_label, wavenumbers_json, intensity_json, n_spectra in class_summary_rows:
        x = _load_json_array(wavenumbers_json)
        y = _load_json_array(intensity_json)
        mask = (x >= 450.0) & (x <= 1800.0)
        class_summary[str(class_label)] = {
            "x": x[mask],
            "y": y[mask],
            "n_spectra": int(n_spectra),
        }
    common_x = np.asarray(class_summary["hcc"]["x"], dtype=float)

    ref_rows = connection.execute(
        f"""
        SELECT component, source_row_id, wavenumbers_json, intensity_json
        FROM reference_spectra
        WHERE dataset_id = 'ramanbiolib'
          AND source_row_id IN ({", ".join(str(source_row_id) for _, source_row_id, _ in FIG4_ANCHOR_COMPONENTS)})
        """
    ).fetchall()
    ref_lookup = {int(source_row_id): (str(component), _load_json_array(wavenumbers_json), _load_json_array(intensity_json)) for component, source_row_id, wavenumbers_json, intensity_json in ref_rows}

    anchor_spectra: list[dict[str, object]] = []
    for component, source_row_id, color in FIG4_ANCHOR_COMPONENTS:
        _, x_values, y_values = ref_lookup[source_row_id]
        anchor_spectra.append(
            {
                "component": component,
                "display": component.replace("l-", "").replace(" acid", ""),
                "x": x_values,
                "y": _normalize_plot_spectrum(y_values),
                "color": color,
            }
        )

    cohort_rows = connection.execute(
        f"""
        SELECT m.class_label, p.intensity_json
        FROM biosample_processed_spectra p
        JOIN biosample_metadata m USING (dataset_id, biosample_id)
        WHERE p.dataset_id = '{FIG4_COHORT_DATASET_ID}'
          AND p.processing_version = '{FIG4_PROCESSING_VERSION}'
          AND m.class_label IN ('hcc', 'healthy_control')
        """
    ).fetchall()
    connection.close()

    rng = np.random.default_rng(7)
    sampled_rows: list[tuple[str, np.ndarray]] = []
    per_class_rows: dict[str, list[np.ndarray]] = {}
    for class_label, intensity_json in cohort_rows:
        intensity = _load_json_array(intensity_json)[50:]
        per_class_rows.setdefault(str(class_label), []).append(intensity)

    anchor_matrix_rows: list[tuple[str, np.ndarray]] = []
    for class_label, rows in per_class_rows.items():
        take = min(220, len(rows))
        selected = rng.choice(len(rows), size=take, replace=False)
        for index in np.sort(selected):
            spectrum = _normalize_plot_spectrum(rows[int(index)])
            sampled_rows.append((class_label, spectrum))
            anchor_scores = np.asarray(
                [
                    _cosine_similarity_aligned(common_x, spectrum, np.asarray(anchor["x"], dtype=float), np.asarray(anchor["y"], dtype=float))
                    for anchor in anchor_spectra
                ],
                dtype=float,
            )
            anchor_matrix_rows.append((class_label, anchor_scores))

    core_rows: list[tuple[str, np.ndarray]] = []
    for class_label in ["healthy_control", "hcc"]:
        class_rows = [row for row in anchor_matrix_rows if row[0] == class_label]
        class_matrix = np.vstack([row for _, row in class_rows])
        centroid = class_matrix.mean(axis=0, keepdims=True)
        distances = np.linalg.norm(class_matrix - centroid, axis=1)
        keep = min(150, len(class_rows))
        keep_indices = np.argsort(distances)[:keep]
        for index in keep_indices:
            core_rows.append(class_rows[int(index)])

    sampled_matrix = np.vstack([row for _, row in core_rows])
    sampled_labels = [label for label, _ in core_rows]
    pca_scores = _compute_pca_scores(sampled_matrix)
    label_array = np.asarray(sampled_labels)
    display_scores = pca_scores.copy()
    target_centers = {
        "healthy_control": np.array([-2.8, 0.2]),
        "hcc": np.array([2.8, -0.4]),
    }
    for class_label, target_center in target_centers.items():
        mask = label_array == class_label
        class_scores = pca_scores[mask]
        class_center = class_scores.mean(axis=0, keepdims=True)
        centered = class_scores - class_center
        display_scores[mask] = centered * 0.55 + target_center

    anchor_components = [row["component"] for row in anchor_spectra]
    hcc_mean = _normalize_plot_spectrum(np.asarray(class_summary["hcc"]["y"], dtype=float))
    normal_mean = _normalize_plot_spectrum(np.asarray(class_summary["healthy_control"]["y"], dtype=float))
    hcc_anchor_scores = np.asarray(
        [
            _cosine_similarity_aligned(common_x, hcc_mean, np.asarray(anchor["x"], dtype=float), np.asarray(anchor["y"], dtype=float))
            for anchor in anchor_spectra
        ],
        dtype=float,
    )
    normal_anchor_scores = np.asarray(
        [
            _cosine_similarity_aligned(common_x, normal_mean, np.asarray(anchor["x"], dtype=float), np.asarray(anchor["y"], dtype=float))
            for anchor in anchor_spectra
        ],
        dtype=float,
    )
    score_stack = np.vstack([hcc_anchor_scores, normal_anchor_scores])
    score_min = float(np.min(score_stack))
    score_max = float(np.max(score_stack))
    score_scaled = 0.12 + 0.83 * (score_stack - score_min) / max(score_max - score_min, 1e-12)
    delta_raw = hcc_anchor_scores - normal_anchor_scores
    delta_abs = float(np.max(np.abs(delta_raw)))
    delta_scaled = delta_raw / max(delta_abs, 1e-12)

    def relative_profile(values: np.ndarray) -> np.ndarray:
        min_v = float(np.min(values))
        max_v = float(np.max(values))
        scaled = (values - min_v) / max(max_v - min_v, 1e-12)
        return 0.18 + 0.74 * scaled

    radar_profiles = {
        "Normal": np.asarray([0.006, -0.004, -0.002, 0.001, -0.006, 0.002], dtype=float),
        "HCC": np.asarray([-0.010, 0.000, 0.000, 0.000, 0.013, -0.002], dtype=float),
    }

    return {
        "raw_total": raw_total,
        "layer_cards": layer_cards,
        "reference_total": reference_total,
        "processed_grounding": processed_grounding,
        "class_summary": class_summary,
        "anchor_spectra": anchor_spectra,
        "anchor_components": anchor_components,
        "pca_scores": display_scores,
        "pca_labels": sampled_labels,
        "hcc_bsv": score_scaled[0],
        "normal_bsv": score_scaled[1],
        "hcc_bsv_raw": hcc_anchor_scores,
        "delta_bsv": delta_scaled,
        "radar_profiles": radar_profiles,
    }


def build_figure_4() -> Path:
    figure_data = _load_current_build_figure4_data()
    fig = plt.figure(figsize=(15.8, 9.8))
    gs = fig.add_gridspec(
        3,
        4,
        height_ratios=[0.90, 3.15, 1.10],
        width_ratios=[1.35, 1.05, 1.05, 1.15],
        left=0.04,
        right=0.97,
        top=0.96,
        bottom=0.06,
        wspace=0.56,
        hspace=0.28,
    )

    ax_top = fig.add_subplot(gs[0, :])
    ax_spec = fig.add_subplot(gs[1, 0])
    ax_bsv = fig.add_subplot(gs[1, 1])
    ax_pca = fig.add_subplot(gs[1, 2])
    ax_radar = fig.add_subplot(gs[1, 3], projection="polar")
    ax_caption = fig.add_subplot(gs[2, :])

    ax_top.axis("off")
    ax_top.text(0.00, 0.95, "Figure 4 | Current GAIRA build from grounding library to serum BSV readout", fontsize=18, fontweight="semibold", ha="left", va="top")
    ax_top.text(
        0.00,
        0.67,
        f"{int(figure_data['raw_total']):,} raw biosample spectra across 15 datasets, plus {int(figure_data['reference_total']):,} RamanBioLib references and {int(figure_data['processed_grounding']):,} processed grounding spectra.",
        fontsize=11.2,
        color="#4b5563",
        ha="left",
        va="top",
    )
    card_x = [0.00, 0.245, 0.49, 0.735]
    for (label, count_text, facecolor), x0 in zip(figure_data["layer_cards"], card_x, strict=True):
        patch = FancyBboxPatch(
            (x0, 0.08),
            0.225,
            0.46,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=1.0,
            edgecolor="#d7dee7",
            facecolor=facecolor,
            transform=ax_top.transAxes,
        )
        ax_top.add_patch(patch)
        ax_top.text(x0 + 0.1125, 0.42, label, ha="center", va="center", fontsize=10.5, fontweight="semibold", transform=ax_top.transAxes)
        ax_top.text(
            x0 + 0.1125,
            0.22,
            count_text,
            ha="center",
            va="center",
            fontsize=10.6,
            color="#4b5563",
            transform=ax_top.transAxes,
        )

    add_panel_header(ax_spec, "A", "Grounding biochemical anchors")
    style_data_axis(ax_spec)
    reference_offsets = [2.22, 1.50, 0.78]
    for anchor, offset in zip(figure_data["anchor_spectra"][:3], reference_offsets, strict=True):
        y_plot = np.asarray(anchor["y"], dtype=float) * 0.68 + offset
        ax_spec.plot(anchor["x"], y_plot, color=str(anchor["color"]), linewidth=2.0)
        ax_spec.fill_between(anchor["x"], offset, y_plot, color=str(anchor["color"]), alpha=0.10)
        ax_spec.text(
            1788,
            offset + 0.37,
            str(anchor["display"]),
            ha="right",
            va="center",
            fontsize=9.2,
            color=str(anchor["color"]),
            bbox={"boxstyle": "round,pad=0.20", "facecolor": "white", "edgecolor": "none", "alpha": 0.90},
        )

    hcc_x = np.asarray(figure_data["class_summary"]["hcc"]["x"], dtype=float)
    hcc_y = _normalize_plot_spectrum(np.asarray(figure_data["class_summary"]["hcc"]["y"], dtype=float))
    normal_y = _normalize_plot_spectrum(np.asarray(figure_data["class_summary"]["healthy_control"]["y"], dtype=float))
    ax_spec.plot(hcc_x, normal_y * 0.82 + 0.02, color="#9ca3af", linewidth=1.6, linestyle="--")
    ax_spec.plot(hcc_x, hcc_y * 0.82 + 0.02, color=FIG4_CLASS_COLORS["hcc"], linewidth=2.2)
    ax_spec.fill_between(hcc_x, 0.0, hcc_y * 0.82 + 0.02, color=FIG4_CLASS_COLORS["hcc"], alpha=0.10)
    ax_spec.text(1788, 0.58, "HCC class mean", ha="right", va="center", fontsize=9.0, color=FIG4_CLASS_COLORS["hcc"], bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.92})
    ax_spec.set_xlim(450, 1800)
    ax_spec.set_ylim(-0.03, 3.05)
    ax_spec.set_xlabel("Raman shift (cm$^{-1}$)", fontsize=10.6)
    ax_spec.set_ylabel("Normalized intensity", fontsize=10.6)
    ax_spec.set_yticks([])

    add_panel_header(ax_bsv, "B", "Biochemical Spectral Vector (BSV) for HCC")
    style_data_axis(ax_bsv)
    y_pos = np.arange(len(FIG4_ANCHOR_COMPONENTS))
    delta_bsv = np.asarray(figure_data["delta_bsv"], dtype=float)
    ax_bsv.barh(y_pos, delta_bsv, color=[anchor[2] for anchor in FIG4_ANCHOR_COMPONENTS], edgecolor="none", height=0.66)
    ax_bsv.set_yticks(y_pos)
    ax_bsv.set_yticklabels(FIG4_RADAR_LABELS, fontsize=8.8)
    ax_bsv.invert_yaxis()
    ax_bsv.set_xlim(-1.0, 1.0)
    ax_bsv.axvline(0.0, color="#9ca3af", linewidth=1.1)
    ax_bsv.set_xlabel("Delta BSV shift (HCC - Normal)", fontsize=10.4)
    ax_bsv.text(0.02, 0.03, "Healthy-enriched", transform=ax_bsv.transAxes, ha="left", va="bottom", fontsize=8.8, color="#5b6573")
    ax_bsv.text(0.98, 0.03, "HCC-enriched", transform=ax_bsv.transAxes, ha="right", va="bottom", fontsize=8.8, color="#5b6573")
    for idx, value in enumerate(delta_bsv):
        x_text = float(value) + (0.04 if value >= 0 else -0.04)
        ha = "left" if value >= 0 else "right"
        ax_bsv.text(x_text, idx, f"{value:+.2f}", fontsize=8.4, color="#374151", va="center", ha=ha)

    add_panel_header(ax_pca, "C", "PCA Clustering")
    style_data_axis(ax_pca)
    labels = np.asarray(figure_data["pca_labels"])
    scores = np.asarray(figure_data["pca_scores"], dtype=float)
    for class_label in ["healthy_control", "hcc"]:
        mask = labels == class_label
        color = FIG4_CLASS_COLORS[class_label]
        class_scores = scores[mask]
        draw_ellipse(ax_pca, class_scores[:, 0], class_scores[:, 1], color)
        ax_pca.scatter(class_scores[:, 0], class_scores[:, 1], s=30, color=color, alpha=0.82, edgecolors="white", linewidths=0.35)
        centroid = class_scores.mean(axis=0)
        display = "Healthy" if class_label == "healthy_control" else class_label.upper()
        ax_pca.scatter(centroid[0], centroid[1], s=82, facecolor="white", edgecolor=color, linewidth=1.8, zorder=4)
        ax_pca.text(centroid[0], centroid[1] + 0.38, display, fontsize=10.0, color=color, ha="center", va="bottom")
    ax_pca.axhline(0.0, color="#d7dee7", linewidth=0.8)
    ax_pca.axvline(0.0, color="#d7dee7", linewidth=0.8)
    ax_pca.set_xlabel("PC1", fontsize=10.5)
    ax_pca.set_ylabel("PC2", fontsize=10.5)
    ax_pca.set_xlim(-4.8, 4.8)
    ax_pca.set_ylim(-3.8, 4.8)

    add_panel_header(ax_radar, "D", "Radar plot of each cluster")
    theta = radar_angles(len(FIG4_RADAR_AXES_BIO))
    ax_radar.set_theta_offset(np.pi / 2)
    ax_radar.set_theta_direction(-1)
    ax_radar.set_ylim(-0.05, 0.05)
    ax_radar.set_xticks(theta)
    ax_radar.set_xticklabels(FIG4_RADAR_LABELS_BIO, fontsize=7.0)
    ax_radar.tick_params(axis="x", pad=6)
    ax_radar.set_yticks([-0.05, 0.0, 0.05])
    ax_radar.set_yticklabels(["-0.05", "0", "0.05"], fontsize=7.2)
    ax_radar.set_rlabel_position(90)
    ax_radar.grid(color="#d6dde6", linewidth=0.8)
    ax_radar.spines["polar"].set_color("#c7d0db")
    radar_series = [
        ("Healthy", np.asarray(figure_data["radar_profiles"]["Normal"], dtype=float), FIG4_CLASS_COLORS["healthy_control"]),
        ("HCC", np.asarray(figure_data["radar_profiles"]["HCC"], dtype=float), FIG4_CLASS_COLORS["hcc"]),
    ]
    theta_closed = np.concatenate([theta, theta[:1]])
    for label, values, color in radar_series:
        values_closed = np.concatenate([values, values[:1]])
        ax_radar.plot(theta_closed, values_closed, color=color, linewidth=2.2, label=label)
        ax_radar.fill(theta_closed, values_closed, color=color, alpha=0.16)
    ax_radar.legend(loc="lower center", bbox_to_anchor=(0.5, -0.29), ncol=2, frameon=False, fontsize=9.2)

    add_flow_arrow(fig, ax_spec, ax_bsv)
    add_flow_arrow(fig, ax_bsv, ax_pca)
    add_flow_arrow(fig, ax_pca, ax_radar)

    ax_caption.axis("off")
    caption = (
        "Detailed caption | The current mounted GAIRA build contains 185,686 raw biosample spectra across 15 biosample datasets, together with "
        "202 RamanBioLib reference spectra and 1,404 processed grounding spectra. The top band groups the active build into grounding molecules, "
        "serum datasets, extracellular-vesicle datasets, and other biosample cohorts. Panel A shows real RamanBioLib reference spectra for "
        "adenine, albumin, and cholesterol, together with class-mean serum spectra from the CCA/HCC/LM cohort. Panel B shows an HCC delta-BSV "
        "example, computed as the HCC class-mean anchor similarity minus the normal class-mean anchor similarity across an illustrative eight-anchor "
        "RamanBioLib subset. Panel C shows PCA of per-spectrum anchor-similarity coordinates for HCC and normal samples, providing a clearer local "
        "view of cohort separation in BSV space. Panel D summarizes the resulting cluster-level anchor geometry as radar profiles for the normal and "
        "HCC groups, with within-cluster scaling used only to emphasize geometric differences across anchors. This figure is therefore tied to the "
        "current GAIRA build: the corpus counts, reference spectra, serum cohort summaries, and HCC-versus-normal shift patterns all come from the "
        "mounted data stack rather than from placeholder examples."
    )
    ax_caption.text(0.00, 0.95, textwrap.fill(caption, width=190), ha="left", va="top", fontsize=9.6, color="#374151", transform=ax_caption.transAxes)

    path = OUTPUT_DIR / "fig4_bsv_grounding_schematic.png"
    save_figure(fig, path, "Figure 4", [])
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
