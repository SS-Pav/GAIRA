"""GAIRA BSV v2 Visualizations — improved radar overlay + delta bars.

Pure matplotlib. No LLM. Streamlit-compatible.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from graph.bsv_v1_scoring import BSVComparison, BSVVector

# ── component display names (v2 refined) ────────────────────
_DISPLAY_NAMES = {
    "membrane_lipid": "Membrane\nLipid",
    "protein_amide": "Protein /\nAmide",
    "aromatic_amino_acid": "Aromatic\nAmino Acid",
    "purine_nucleotide": "Purine\nNucleotide",
    "pyrimidine_nucleotide": "Pyrimidine\nNucleotide",
    "glycan_carbohydrate": "Glycan /\nCarbohydrate",
    "redox_thiol_metabolite": "Redox / Thiol\nMetabolite",
    "phosphate_backbone": "Phosphate\nBackbone",
    # Legacy v1 names (fallback)
    "protein_backbone": "Protein /\nAmide",
    "nucleic_acid_backbone": "Phosphate\nBackbone",
    "redox_metabolite": "Redox / Thiol\nMetabolite",
}


def _label(name: str) -> str:
    return _DISPLAY_NAMES.get(name, name.replace("_", " ").title())


def _label_flat(name: str) -> str:
    """Single-line label for bar charts."""
    return _label(name).replace("\n", " ")


def render_radar_plot(comparison: BSVComparison) -> plt.Figure:
    """Render radar with query + comparator overlay for comparative queries."""
    q = comparison.query_bsv
    active = [(c, i) for i, c in enumerate(q.components) if c.coverage_note != "absent"]
    if len(active) < 3:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "Insufficient components for radar", ha="center", va="center", fontsize=12)
        ax.set_axis_off()
        return fig

    labels = [_label(c.name) for c, _ in active]
    q_vals = [c.normalized_score for c, _ in active]
    indices = [i for _, i in active]
    N = len(labels)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles_closed = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_facecolor("#fafafa")
    fig.patch.set_facecolor("white")

    # Grid styling
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=7, color="#aaa")
    for spine in ax.spines.values():
        spine.set_color("#ddd")

    # Query polygon
    q_closed = q_vals + [q_vals[0]]
    ax.fill(angles_closed, q_closed, alpha=0.20, color="#1565C0")
    ax.plot(angles_closed, q_closed, "o-", linewidth=2.2, color="#1565C0", markersize=6,
            label=q.query_condition.replace("_", " "), zorder=3)

    # Comparator polygon
    if comparison.comparator_bsv:
        c_comps = comparison.comparator_bsv.components
        c_vals = [c_comps[i].normalized_score for i in indices]
        c_closed = c_vals + [c_vals[0]]
        ax.fill(angles_closed, c_closed, alpha=0.12, color="#E53935")
        ax.plot(angles_closed, c_closed, "s--", linewidth=1.8, color="#E53935", markersize=5,
                label=comparison.comparator_bsv.query_condition.replace("_", " "), zorder=2)

    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=8, ha="center")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.12), fontsize=9, framealpha=0.9)
    ax.set_title("Biochemical State Vector", fontsize=13, fontweight="bold", pad=20)

    plt.tight_layout()
    return fig


def render_delta_plot(comparison: BSVComparison) -> plt.Figure | None:
    """Improved delta bar chart: sorted by |delta|, annotated, with flat-line note."""
    if not comparison.delta_components:
        return None

    # Sort by absolute delta (largest shifts first)
    deltas = sorted(comparison.delta_components, key=lambda x: -abs(x["delta"]))
    labels = [_label_flat(d["component"]) for d in deltas]
    values = [d["delta"] for d in deltas]

    max_abs = max(abs(v) for v in values) if values else 0.1

    # Color coding
    colors = []
    for v in values:
        if v > 0.03:
            colors.append("#1565C0")
        elif v < -0.03:
            colors.append("#E53935")
        else:
            colors.append("#BDBDBD")

    fig, ax = plt.subplots(figsize=(8, max(3.5, len(labels) * 0.45)))
    fig.patch.set_facecolor("white")

    bars = ax.barh(labels, values, color=colors, edgecolor="white", height=0.55, zorder=2)
    ax.axvline(0, color="#333", linewidth=1.0, zorder=1)

    # Dynamic x-range
    xlim = max(0.15, max_abs * 1.4)
    ax.set_xlim(-xlim, xlim)

    ax.set_xlabel("Delta (query − comparator)", fontsize=10)
    q_name = comparison.query_bsv.query_condition.replace("_", " ")
    c_name = comparison.query_bsv.comparator_condition.replace("_", " ")
    ax.set_title(f"BSV Delta: {q_name} vs {c_name}", fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    ax.tick_params(labelsize=9)

    # Value annotations
    for bar, val in zip(bars, values):
        if abs(val) > 0.01:
            offset = xlim * 0.04
            ax.text(val + (offset if val > 0 else -offset),
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:+.3f}", ha="left" if val > 0 else "right",
                    va="center", fontsize=8, color="#333")

    # Note if all deltas are tiny
    if max_abs < 0.05:
        ax.text(0, -0.5, "Note: query and comparator BSVs are very similar under current weighting",
                ha="center", va="top", fontsize=8, color="#888", style="italic",
                transform=ax.get_xaxis_transform())

    plt.tight_layout()
    return fig
