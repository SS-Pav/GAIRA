"""Publication figures for the V6 demo (matplotlib, position-first, one accent).

Every figure is a pure function of engine outputs — no hidden state, no fitting.
The MSS-hierarchy figure is the demo's centerpiece: it shows how a biochemical
state decomposes Radar -> MSS motifs -> latent components.
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import rcParams

from . import theme as T

rcParams.update(T.mpl_rc())

# short display labels for the 11 biochemical themes (radar axes)
THEME_SHORT = {
    "nucleic_purine": "Purine", "nucleic_pyrimidine": "Pyrimidine",
    "protein_peptide": "Protein", "aromatic_amino_acid": "Aromatic AA",
    "lipid_acyl": "Lipid acyl", "sterol_membrane": "Sterol",
    "saccharide_glycan": "Glycan", "organic_acid_metabolism": "Organic acid",
    "sulfur_antioxidant": "Sulfur", "heme_porphyrin": "Heme", "redox_broad": "Redox",
}


# ── Page 1: architecture flow ──
def architecture_diagram():
    stages = [
        ("Reference spectra", "375 pure Raman · 167 analytes"),
        ("Frozen Raman Atlas", "NMF k=24 · dictionary held fixed"),
        ("Latent Raman motifs", "24 components (mathematical)"),
        ("Molecular Spectral Signatures", "13 validated motifs (chemistry)"),
        ("Biochemical State Vector", "13 themes · elevation + confidence"),
        ("Domain-aware interpretation", "serum / EV / buffer / tissue"),
        ("Evidence report + radar", "provenance · OOD · caveats"),
    ]
    highlight = 3  # the MSS layer — the interpretive centrepiece
    fig, ax = plt.subplots(figsize=(6.4, 8.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, len(stages) * 1.5 + 1.15); ax.axis("off")
    n = len(stages)
    for i, (title, sub) in enumerate(stages):
        y = (n - i - 1) * 1.5 + 0.6
        is_hi = i == highlight
        face = "#eef5fa" if not is_hi else "#fdeef0"
        edge = T.PRIMARY if not is_hi else T.UP
        box = FancyBboxPatch((1.2, y), 7.6, 1.02, boxstyle="round,pad=0.02,rounding_size=0.12",
                             facecolor=face, edgecolor=edge, linewidth=1.6 if is_hi else 1.1)
        ax.add_patch(box)
        ax.text(5.0, y + 0.66, title, ha="center", va="center", fontsize=11.5,
                fontweight="700", color=T.INK)
        ax.text(5.0, y + 0.28, sub, ha="center", va="center", fontsize=8.6, color=T.MUTED)
        if is_hi:
            ax.text(8.95, y + 0.51, "centerpiece", ha="left", va="center", fontsize=8,
                    color=T.UP, fontweight="700", rotation=90)
        if i < n - 1:
            ax.add_patch(FancyArrowPatch((5.0, y), (5.0, y - 0.48), arrowstyle="-|>",
                         mutation_scale=14, color=T.FAINT, linewidth=1.3))
    ax.text(5.0, n * 1.5 + 0.7, "Frozen coordinate system · deterministic · provenance-preserving",
            ha="center", va="center", fontsize=8.8, color=T.FAINT, style="italic")
    fig.tight_layout()
    return fig


# ── biochemical theme radar (the engine's canonical radar backend) ──
def _draw_radar(ax, radar_axes, score_key="score", color=T.PRIMARY, labelsize=9,
                ref_axes=None):
    """Draw the engine radar onto a polar axis. Optional faint reference overlay."""
    vals = [a[score_key] for a in radar_axes]
    labels = [THEME_SHORT.get(a["theme"], a["theme"]) for a in radar_axes]
    ang = np.linspace(0, 2 * np.pi, len(vals), endpoint=False)
    ang_c = np.concatenate([ang, ang[:1]]); v_c = np.array(vals + vals[:1])
    peak = max(vals + ([max(a[score_key] for a in ref_axes)] if ref_axes else []))
    vmax = peak * 1.18 if peak > 0 else 1.0
    ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
    ax.set_ylim(0, vmax)
    ax.set_yticks([vmax / 2]); ax.set_yticklabels([f"{vmax/2:.2f}"], color=T.FAINT, fontsize=7.5)
    ax.set_xticks(ang); ax.set_xticklabels(labels, fontsize=labelsize, color=T.INK)
    if ref_axes is not None:
        rv = [a[score_key] for a in ref_axes]; rv_c = np.array(rv + rv[:1])
        ax.plot(ang_c, rv_c, color=T.FAINT, linewidth=1.0, linestyle="--")
    ax.plot(ang_c, v_c, color=color, linewidth=2.0)
    ax.fill(ang_c, v_c, color=color, alpha=0.16)
    ax.scatter(ang, vals, color=color, s=20, zorder=5)
    ax.grid(color=T.GRID, linewidth=0.7)
    ax.spines["polar"].set_color(T.PANEL_EDGE)


def radar(radar_axes, title="Biochemical State Vector", score_key="score", ref_axes=None):
    """Plot the engine's radar (out.radar['axes']). Uses composition share by
    default — it stays informative in- AND out-of-domain, unlike the tanh display
    value which saturates for far-OOD SERS inputs."""
    fig, ax = plt.subplots(figsize=(5.6, 5.6), subplot_kw={"polar": True})
    _draw_radar(ax, radar_axes, score_key=score_key, ref_axes=ref_axes)
    ax.set_title(title + "  ·  evidence share", fontsize=12.0, color=T.INK, pad=18)
    fig.tight_layout()
    return fig


# ── signed delta radar (perturbation vs baseline; shared centred scale) ──
def delta_radar(delta_axes, shared_max, title="ΔBSV vs baseline", subtitle=""):
    """Signed radar: theme deltas mapped to radius with 0 at the mid-ring, positive
    (increase) outward, negative (decrease) inward. Shared symmetric scale across the
    experiment — NO per-sample rescaling. Lobes out = up, in = down."""
    themes = [a["theme"] for a in delta_axes]
    deltas = np.array([a["delta"] for a in delta_axes])
    labels = [THEME_SHORT.get(t, t) for t in themes]
    M = shared_max if shared_max > 1e-9 else 1.0
    r = 0.5 + 0.5 * np.clip(deltas / M, -1, 1)
    ang = np.linspace(0, 2 * np.pi, len(themes), endpoint=False)
    ang_c = np.concatenate([ang, ang[:1]]); r_c = np.concatenate([r, r[:1]])
    fig, ax = plt.subplots(figsize=(5.9, 5.9), subplot_kw={"polar": True})
    ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1); ax.set_ylim(0, 1)
    ax.set_xticks(ang); ax.set_xticklabels(labels, fontsize=9, color=T.INK)
    ax.set_yticks([0.5]); ax.set_yticklabels([])                      # zero ring only
    ax.plot(ang_c, np.full_like(ang_c, 0.5), color=T.MUTED, linewidth=1.2, linestyle="--")
    ax.plot(ang_c, r_c, color=T.INK, linewidth=1.4)
    for a_, rr, d in zip(ang, r, deltas):
        ax.plot([a_, a_], [0.5, rr], color=(T.UP if d >= 0 else T.DOWN), linewidth=3.0,
                solid_capstyle="round")
        ax.scatter(a_, rr, color=(T.UP if d >= 0 else T.DOWN), s=28, zorder=5)
    ax.grid(color=T.GRID, linewidth=0.6)
    ax.spines["polar"].set_color(T.PANEL_EDGE)
    ax.set_title(f"{title}\n{subtitle}", fontsize=11.8, color=T.INK, pad=22)
    ax.text(np.pi / 2, 1.02, f"±{M:.3f}", fontsize=7.5, color=T.FAINT, ha="center")
    fig.tight_layout()
    return fig


def dose_delta_radar_grid(delta_axes_list, levels, shared_max, target_theme, analyte):
    """QC + teaching figure: ΔBSV delta radars for every dose, side by side on a shared
    centred scale. Verifies the radar evolves smoothly and interpretably with dose."""
    n = len(levels)
    ncol = min(n, 6)
    idx = np.linspace(0, n - 1, ncol).round().astype(int)
    M = shared_max if shared_max > 1e-9 else 1.0
    fig, axes = plt.subplots(1, ncol, figsize=(2.35 * ncol, 2.7),
                             subplot_kw={"polar": True})
    axes = np.atleast_1d(axes)
    for ax, i in zip(axes, idx):
        da = delta_axes_list[i]
        themes = [a["theme"] for a in da]
        deltas = np.array([a["delta"] for a in da])
        r = 0.5 + 0.5 * np.clip(deltas / M, -1, 1)
        ang = np.linspace(0, 2 * np.pi, len(themes), endpoint=False)
        ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1); ax.set_ylim(0, 1)
        ax.set_xticks([]); ax.set_yticks([0.5]); ax.set_yticklabels([])
        ax.plot(np.append(ang, ang[0]), np.full(len(ang) + 1, 0.5), color=T.MUTED,
                linewidth=0.8, linestyle="--")
        for a_, rr, d in zip(ang, r, deltas):
            ax.plot([a_, a_], [0.5, rr], color=(T.UP if d >= 0 else T.DOWN), linewidth=2.2)
        # mark the target-theme spoke
        ti = themes.index(target_theme)
        ax.scatter(ang[ti], r[ti], s=30, color=T.INK, zorder=6, marker="*")
        ax.set_title(f"{levels[i]:.2f} µM", fontsize=9, color=T.INK, pad=4)
        ax.grid(color=T.GRID, linewidth=0.5); ax.spines["polar"].set_color(T.PANEL_EDGE)
    fig.suptitle(f"{analyte.capitalize()} — ΔBSV delta radar across dose "
                 f"(★ = {THEME_SHORT.get(target_theme, target_theme)}; shared scale ±{M:.3f})",
                 fontsize=11, fontweight="700", color=T.INK, y=1.08)
    fig.tight_layout()
    return fig


def mechanism_curves(levels, redistribution, target_elev, target_name,
                     xlabel="concentration (µM)", title="Mechanism: redistribution vs evidence"):
    """Dual-axis: component-redistribution index (left) vs target-theme elevation (right).
    Explains why biochemical evidence can rise while the trajectory curves."""
    fig, ax1 = plt.subplots(figsize=(7.2, 3.8))
    ax1.plot(levels, redistribution, "-o", color=T.UP, markersize=4, linewidth=2.0,
             label="redistribution R(d)=1−cos")
    ax1.set_xlabel(xlabel); ax1.set_ylabel("redistribution index", color=T.UP)
    ax1.tick_params(axis="y", labelcolor=T.UP)
    ax2 = ax1.twinx()
    ax2.plot(levels, target_elev, "-s", color=T.PRIMARY, markersize=4, linewidth=2.0,
             label=target_name)
    ax2.set_ylabel(f"{target_name} composition", color=T.PRIMARY)
    ax2.tick_params(axis="y", labelcolor=T.PRIMARY); ax2.grid(False)
    ax1.set_title(title, fontsize=11.5, pad=8)
    fig.tight_layout()
    return fig


def pairwise_trajectory(x, y, levels, xlabel, ylabel, title="Trajectory on interpretable axes"):
    """Dose trajectory on two biologically meaningful axes (not unsupervised PCA)."""
    x = np.asarray(x, float); y = np.asarray(y, float); levels = np.asarray(levels, float)
    fig, ax = plt.subplots(figsize=(6.0, 4.8))
    ax.plot(x, y, "-", color=T.FAINT, linewidth=1.2, zorder=1)
    sc = ax.scatter(x, y, c=levels, cmap="viridis", s=70, zorder=3, edgecolor="white", linewidth=0.6)
    ax.scatter(x[0], y[0], s=150, facecolor="none", edgecolor=T.INK, linewidth=1.4, zorder=4)
    ax.annotate("", xy=(x[-1], y[-1]), xytext=(x[-2], y[-2]),
                arrowprops=dict(arrowstyle="-|>", color=T.INK, lw=1.5))
    cb = fig.colorbar(sc, ax=ax, shrink=0.8); cb.set_label("dose (µM)", fontsize=9)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title, fontsize=11.5, pad=8)
    fig.tight_layout()
    return fig


# ── MSS hierarchy — the centerpiece ──
def mss_hierarchy(activations, title="Molecular Spectral Signatures"):
    """Horizontal motif bars by elevation; diverging colour = increase/decrease."""
    acts = [a for a in activations if not a.non_biochemical]
    acts = sorted(acts, key=lambda a: a.elevation)
    names = [a.name for a in acts]
    elev = np.array([a.elevation for a in acts])
    conf = np.array([a.confidence for a in acts])
    colors = [T.UP if e >= 0 else T.DOWN for e in elev]
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    y = np.arange(len(acts))
    ax.barh(y, elev, color=colors, alpha=0.9, height=0.62,
            edgecolor=T.SURFACE, linewidth=1.2)
    ax.axvline(0, color=T.MUTED, linewidth=1.0)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9.5, color=T.INK)
    ax.set_xlabel("Motif elevation vs pure-Raman reference (signed z)")
    ax.set_title(title, fontsize=12.5, color=T.INK, pad=10)
    # annotate confidence as a faint marker at the bar end
    for yi, (e, c) in enumerate(zip(elev, conf)):
        ax.text(e + (0.02 if e >= 0 else -0.02) * max(abs(elev).max(), 1),
                yi, f"c={c:.2f}", va="center", ha="left" if e >= 0 else "right",
                fontsize=7.4, color=T.FAINT)
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.16)
    fig.tight_layout()
    return fig


# ── 24-component fingerprint ──
def component_fingerprint(coord, highlight=None, title="Latent Raman fingerprint (24 components)"):
    coord = np.asarray(coord, float)
    fig, ax = plt.subplots(figsize=(7.6, 2.8))
    x = np.arange(len(coord))
    base = [T.PRIMARY_SOFT] * len(coord)
    if highlight:
        for j in highlight:
            base[j] = T.PRIMARY
    ax.bar(x, coord, color=base, width=0.78)
    ax.set_xticks(x); ax.set_xticklabels([f"c{j}" for j in x], fontsize=6.8, color=T.MUTED)
    ax.set_ylabel("evidence share")
    ax.set_title(title, fontsize=11.5, pad=8)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig


# ── a component's frozen basis spectrum ──
def basis_spectrum(grid, spectrum, bands=None, title="Basis spectrum"):
    fig, ax = plt.subplots(figsize=(7.6, 2.9))
    ax.plot(grid, spectrum, color=T.PRIMARY, linewidth=1.4)
    ax.fill_between(grid, 0, spectrum, color=T.PRIMARY, alpha=0.10)
    if bands:
        for b in bands:
            ax.axvline(b, color=T.UP, linewidth=0.8, alpha=0.55, linestyle="--")
    ax.set_xlabel("Raman shift (cm$^{-1}$)"); ax.set_ylabel("loading")
    ax.set_title(title, fontsize=11.5, pad=8)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig


# ── NMF-native component views (Part 1) ──
def _classical_mds(D):
    """Deterministic classical (Torgerson) MDS to 2-D from a distance matrix."""
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    w, V = np.linalg.eigh(B)
    order = np.argsort(-w)[:2]
    L = np.sqrt(np.clip(w[order], 0, None))
    Y = V[:, order] * L
    for i in range(2):                                   # deterministic sign
        if Y[np.argmax(np.abs(Y[:, i])), i] < 0:
            Y[:, i] = -Y[:, i]
    return Y


def component_similarity_map(D, theme_by_comp, title="NMF component similarity map (MDS)"):
    """2-D map of the 24 components (classical MDS on basis-spectrum cosine distance),
    coloured by dominant theme. Visualises the learned representation — NOT inference."""
    Y = _classical_mds(D)
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    themes = sorted(set(theme_by_comp.values()))
    for t in themes:
        idx = [j for j in range(len(D)) if theme_by_comp[j] == t]
        ax.scatter(Y[idx, 0], Y[idx, 1], s=90, color=THEME_TINT_HEX(t), alpha=0.9,
                   edgecolor="white", linewidth=0.8, label=THEME_SHORT.get(t, t), zorder=3)
    for j in range(len(D)):
        ax.annotate(f"c{j}", (Y[j, 0], Y[j, 1]), fontsize=7.6, color=T.INK, ha="center",
                    va="center", zorder=4)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("MDS-1"); ax.set_ylabel("MDS-2")
    ax.set_title(title, fontsize=12.0, pad=8)
    ax.legend(fontsize=7.6, loc="upper left", bbox_to_anchor=(1.0, 1.0), title="dominant theme")
    fig.tight_layout()
    return fig


def component_dendrogram(D, theme_by_comp, title="NMF component hierarchy (basis-spectrum similarity)"):
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import squareform
    Z = linkage(squareform(D, checks=False), method="average")
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    labels = [f"c{j}·{THEME_SHORT.get(theme_by_comp[j], '')[:6]}" for j in range(len(D))]
    dendrogram(Z, labels=labels, ax=ax, color_threshold=0.6 * D.max(),
               above_threshold_color=T.FAINT, leaf_font_size=7.5)
    ax.set_title(title, fontsize=11.5, pad=8); ax.set_ylabel("cosine distance")
    ax.grid(axis="x", visible=False)
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(90)
    fig.tight_layout()
    return fig


def THEME_TINT_HEX(theme):
    return T.THEME_TINT.get(theme, "#9aa4b2")


# ── corpus source / excitation breakdown ──
def corpus_breakdown(sources, excitations, title="Reference corpus composition"):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.8, 3.0))
    s = dict(sorted(sources.items(), key=lambda kv: kv[1]))
    a1.barh(list(s.keys()), list(s.values()), color=T.PRIMARY, height=0.6)
    for i, v in enumerate(s.values()):
        a1.text(v, i, f" {v}", va="center", fontsize=8.5, color=T.MUTED)
    a1.set_title("Raman sources (spectra)", fontsize=10.5); a1.grid(axis="y", visible=False)
    ex = {str(int(float(k))): v for k, v in excitations.items()}
    ex = dict(sorted(ex.items(), key=lambda kv: -kv[1])[:6])
    a2.bar(list(ex.keys()), list(ex.values()), color=T.SEQ[3], width=0.7)
    a2.set_title("Excitation lines (nm)", fontsize=10.5); a2.grid(axis="x", visible=False)
    a2.tick_params(axis="x", labelsize=8.5)
    fig.suptitle(title, fontsize=12.0, fontweight="700", color=T.INK, y=1.04)
    fig.tight_layout()
    return fig


# ── spectral collisions map ──
def band_collision_map(motifs, tol=16.0, title="Spectral collisions: where motifs share bands"):
    """Each biochemical motif's characteristic bands as ticks on a shared cm⁻¹ axis.
    Regions claimed by >=3 motifs are shaded — the physical reason multiple motifs
    contribute to one spectral region ('peak != molecule')."""
    mot = [m for m in motifs if not m.non_biochemical]
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    all_bands = []
    for i, m in enumerate(mot):
        for b in m.bands_cm:
            ax.scatter(b, i, s=26, color=T.PRIMARY, zorder=3)
            all_bands.append(b)
    all_bands = np.array(all_bands)
    # find collision hotspots: grid points where >=3 DISTINCT motifs have a band within tol,
    # then merge contiguous hotspot cells into single shaded spans (avoids striping).
    grid = np.arange(450, 1801, 4)
    def n_motifs_near(g):
        return sum(any(abs(b - g) <= tol for b in m.bands_cm) for m in mot)
    hot = np.array([n_motifs_near(g) >= 4 for g in grid])   # genuinely congested regions only
    spans, start = [], None
    for gi, h in enumerate(list(hot) + [False]):
        if h and start is None:
            start = grid[gi]
        elif not h and start is not None:
            spans.append((start, grid[gi - 1])); start = None
    for lo, hi in spans:
        ax.axvspan(lo - tol, hi + tol, color=T.UP, alpha=0.12, zorder=0)
    ax.set_yticks(range(len(mot)))
    ax.set_yticklabels([m.name for m in mot], fontsize=8.6, color=T.INK)
    ax.set_xlabel("Raman shift (cm$^{-1}$)")
    ax.set_xlim(440, 1810)
    ax.set_title(title, fontsize=12.0, pad=10)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


# ── calibration dose-response ──
def dose_response(levels, scores, xlabel="dose", ylabel="theme composition",
                  title="Dose-response"):
    """Faint replicate cloud + bold per-dose mean line (handles many replicates/dose)."""
    levels = np.asarray(levels, float); scores = np.asarray(scores, float)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.scatter(levels, scores, s=10, color=T.PRIMARY_SOFT, alpha=0.5, zorder=1)
    uniq = np.array(sorted(np.unique(levels)))
    mean = np.array([scores[levels == u].mean() for u in uniq])
    ax.plot(uniq, mean, "-o", color=T.PRIMARY, markersize=5, linewidth=1.8, zorder=3)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title, fontsize=11.5, pad=8)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Page 4 — Calibration figures
# ══════════════════════════════════════════════════════════════════════════════

def _arrow_between(fig, ax_from, ax_to, direction="right"):
    """Draw a connector arrow between two axes in figure coordinates."""
    b0, b1 = ax_from.get_position(), ax_to.get_position()
    if direction == "right":
        xy = (b1.x0, (b1.y0 + b1.y1) / 2); xytext = (b0.x1, (b0.y0 + b0.y1) / 2)
    else:  # down
        xy = ((b1.x0 + b1.x1) / 2, b1.y1); xytext = ((b0.x0 + b0.x1) / 2, b0.y0)
    fig.add_artist(FancyArrowPatch(
        xytext, xy, transform=fig.transFigure, arrowstyle="-|>", mutation_scale=16,
        color=T.PRIMARY, linewidth=1.5, zorder=50, alpha=0.8))


def experimental_schematic(analyte, substrate, laser):
    """Small analyte → colloid → SERS → GAIRA flow schematic."""
    steps = [analyte.capitalize(), f"{substrate} colloids", f"{laser} nm SERS", "GAIRA engine"]
    fig, ax = plt.subplots(figsize=(7.6, 1.35)); ax.axis("off")
    ax.set_xlim(0, len(steps)); ax.set_ylim(0, 1)
    for i, s in enumerate(steps):
        hi = i == len(steps) - 1
        ax.add_patch(FancyBboxPatch((i + 0.06, 0.2), 0.88, 0.6,
                     boxstyle="round,pad=0.02,rounding_size=0.08",
                     facecolor="#fdeef0" if hi else "#eef5fa",
                     edgecolor=T.UP if hi else T.PRIMARY, linewidth=1.3))
        ax.text(i + 0.5, 0.5, s, ha="center", va="center", fontsize=10,
                fontweight="700" if hi else "600", color=T.INK)
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((i + 0.94, 0.5), (i + 1.06, 0.5), arrowstyle="-|>",
                         mutation_scale=12, color=T.FAINT, linewidth=1.3))
    fig.tight_layout()
    return fig


def representative_spectra(specs, grid, labels, bands=None,
                           title="Representative spectra (atlas reconstruction)"):
    """Stacked low/med/high reconstructed spectra with the target motif's bands."""
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    off = np.nanmax(specs) * 0.62
    for i, (sp, lab) in enumerate(zip(specs, labels)):
        y = sp + i * off
        ax.plot(grid, y, color=T.SEQ[min(2 + i, len(T.SEQ) - 1)], linewidth=1.5)
        ax.text(grid[-1], y[-1], f"  {lab}", va="center", fontsize=8.6, color=T.INK)
    if bands:
        for b in bands:
            ax.axvline(b, color=T.UP, linewidth=0.7, alpha=0.4, linestyle="--")
    ax.set_xlabel("Raman shift (cm$^{-1}$)"); ax.set_yticks([])
    ax.set_ylabel("reconstruction (offset)")
    ax.set_title(title, fontsize=11.5, pad=8); ax.grid(axis="x", visible=False)
    ax.set_xlim(grid[0], grid[-1] + 90)
    fig.tight_layout()
    return fig


def mss_evolution(levels, evo, target_id, names, xlabel="concentration (µM)",
                  title="MSS motif evolution with dose"):
    """Central panel: each motif's elevation vs dose; target motif emphasised."""
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    # rank motifs by dynamic range; show target + top movers
    ranked = sorted(evo, key=lambda m: -(np.ptp(evo[m])))
    show = [target_id] + [m for m in ranked if m != target_id][:4]
    for m in show:
        is_t = m == target_id
        ax.plot(levels, evo[m], "-o", markersize=4,
                color=T.UP if is_t else T.SEQ[3], alpha=1.0 if is_t else 0.55,
                linewidth=2.4 if is_t else 1.3, zorder=5 if is_t else 2,
                label=("★ " if is_t else "") + names.get(m, m))
    ax.set_xlabel(xlabel); ax.set_ylabel("motif elevation (signed z)")
    ax.set_title(title, fontsize=12.0, pad=8)
    ax.legend(fontsize=8.2, loc="best")
    fig.tight_layout()
    return fig


def dose_response_langmuir(levels, mean_scores, rep_levels, rep_scores, fit,
                           analyte="analyte", theme="theme", rho=None):
    """Dose-response: replicate cloud + per-dose means + Langmuir fit overlay."""
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.scatter(rep_levels, rep_scores, s=12, color=T.PRIMARY_SOFT, alpha=0.5,
               zorder=1, label="replicates")
    ax.plot(levels, mean_scores, "o", color=T.PRIMARY, markersize=6, zorder=3,
            label="per-dose mean")
    if fit is not None:
        xf, yf, popt, r2, K = fit
        ax.plot(xf, yf, "-", color=T.UP, linewidth=2.0, zorder=4,
                label=f"Langmuir fit (R²={r2:.2f}, K={K:.2g} µM)")
    ax.set_xlabel(f"{analyte} concentration (µM)")
    ax.set_ylabel(f"{THEME_SHORT.get(theme, theme)} evidence share")
    ttl = f"{analyte.capitalize()} → {THEME_SHORT.get(theme, theme)} dose-response"
    if rho is not None and np.isfinite(rho):
        ttl += f"   ρ={rho:.2f}"
    ax.set_title(ttl, fontsize=12.0, pad=8)
    ax.legend(fontsize=8.4, loc="lower right")
    fig.tight_layout()
    return fig


def component_evolution(levels, comp_series, top_js, xlabel="concentration (µM)",
                        title="Latent component evolution"):
    """Which components rise/fall with dose. Colour encodes DIRECTION — rising
    (warm) vs falling (cool) — so redistribution (some up, some down) vs scaling
    (one up, rest flat) is legible at a glance. Lines are end-labelled, no legend."""
    levels = np.asarray(levels, float)
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    up_shades = ["#b2182b", "#d6604d", "#e8917c"]
    dn_shades = ["#2166ac", "#4a90c2", "#7fb0d6"]
    rising = sorted([j for j in top_js if comp_series[-1, j] >= comp_series[0, j]],
                    key=lambda j: -(comp_series[-1, j] - comp_series[0, j]))
    falling = sorted([j for j in top_js if comp_series[-1, j] < comp_series[0, j]],
                     key=lambda j: (comp_series[-1, j] - comp_series[0, j]))
    for grp, shades in ((rising, up_shades), (falling, dn_shades)):
        for k, j in enumerate(grp):
            c = shades[min(k, len(shades) - 1)]
            ax.plot(levels, comp_series[:, j], "-o", markersize=3.5, color=c, linewidth=1.8)
            ax.text(levels[-1], comp_series[-1, j], f" c{j}", va="center", fontsize=8.4,
                    color=c, fontweight="700")
    ax.set_xlabel(xlabel); ax.set_ylabel("component evidence share")
    ax.set_title(title, fontsize=11.5, pad=8)
    ax.set_xlim(levels.min(), levels.max() + 0.10 * np.ptp(levels))
    ax.text(0.02, 0.96, "▲ rising", transform=ax.transAxes, color=up_shades[0],
            fontsize=8.5, fontweight="700", va="top")
    ax.text(0.02, 0.88, "▼ falling", transform=ax.transAxes, color=dn_shades[0],
            fontsize=8.5, fontweight="700", va="top")
    fig.tight_layout()
    return fig


def trajectory_2d(proj, levels, var, ref_cloud=None, cmap_label="dose (µM)",
                  title="Dose trajectory through BSV space"):
    """2-D BSV trajectory coloured by dose (PCA for visualisation only)."""
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    if ref_cloud is not None:
        ax.scatter(ref_cloud[:, 0], ref_cloud[:, 1], s=10, color=T.GRID, zorder=1,
                   label="reference cloud")
    ax.plot(proj[:, 0], proj[:, 1], "-", color=T.FAINT, linewidth=1.2, zorder=2)
    sc = ax.scatter(proj[:, 0], proj[:, 1], c=levels, cmap="viridis", s=60, zorder=3,
                    edgecolor="white", linewidth=0.6)
    ax.scatter(proj[0, 0], proj[0, 1], s=140, facecolor="none", edgecolor=T.INK,
               linewidth=1.4, zorder=4)
    cb = fig.colorbar(sc, ax=ax, shrink=0.8); cb.set_label(cmap_label, fontsize=9)
    ax.set_xlabel(f"BSV-PC1 ({var[0]:.0%})"); ax.set_ylabel(f"BSV-PC2 ({var[1]:.0%})")
    ax.set_title(title, fontsize=11.5, pad=8)
    fig.tight_layout()
    return fig


# ── THE signature figure: the full reasoning cascade at one concentration ──
def reasoning_cascade(bridge, coord, dose_label="", domain="buffer"):
    """Spectrum → Components → MSS → BSV → Radar, all for one query. The iconic
    GAIRA figure: move the concentration slider and every panel updates together."""
    out, acts = bridge.bsv_and_mss(coord, domain=domain)
    grid, spec = bridge.reconstruct(coord)
    comp = np.asarray(out.bsv.component_coord)
    bio_acts = sorted([a for a in acts if not a.non_biochemical],
                      key=lambda a: a.elevation)[-8:]
    bio_theme = sorted(out.bsv.biochemical_themes().items(), key=lambda kv: kv[1])[-8:]

    fig = plt.figure(figsize=(15.5, 7.4))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.05], hspace=0.55, wspace=0.42,
                          left=0.05, right=0.975, top=0.88, bottom=0.09)
    ax_spec = fig.add_subplot(gs[0, 0]); ax_comp = fig.add_subplot(gs[0, 1])
    ax_mss = fig.add_subplot(gs[0, 2]); ax_bsv = fig.add_subplot(gs[0, 3])
    ax_rad = fig.add_subplot(gs[1, 1:3], projection="polar")
    ax_lbl = fig.add_subplot(gs[1, 0]); ax_stat = fig.add_subplot(gs[1, 3])

    # 1 · spectrum
    ax_spec.plot(grid, spec, color=T.PRIMARY, linewidth=1.4)
    ax_spec.fill_between(grid, 0, spec, color=T.PRIMARY, alpha=0.10)
    ax_spec.set_title("1 · Spectrum", fontsize=11, color=T.INK)
    ax_spec.set_xlabel("cm$^{-1}$", fontsize=8.5); ax_spec.set_yticks([])
    ax_spec.tick_params(labelsize=7.5); ax_spec.grid(axis="x", visible=False)

    # 2 · components
    ax_comp.bar(range(len(comp)), comp, color=T.PRIMARY_SOFT, width=0.8)
    top3 = np.argsort(-comp)[:3]
    for j in top3:
        ax_comp.bar(j, comp[j], color=T.PRIMARY, width=0.8)
    ax_comp.set_title("2 · Latent components", fontsize=11, color=T.INK)
    ax_comp.set_xlabel("component", fontsize=8.5); ax_comp.tick_params(labelsize=7.5)
    ax_comp.set_xticks([0, 8, 16, 23]); ax_comp.grid(axis="x", visible=False)

    # 3 · MSS motifs (the emphasised layer)
    y = range(len(bio_acts)); el = [a.elevation for a in bio_acts]
    ax_mss.barh(list(y), el, color=[T.UP if e >= 0 else T.DOWN for e in el],
                height=0.66, edgecolor=T.SURFACE, linewidth=0.8)
    ax_mss.axvline(0, color=T.MUTED, linewidth=0.8)
    ax_mss.set_yticks(list(y)); ax_mss.set_yticklabels([a.name for a in bio_acts], fontsize=7.6)
    ax_mss.set_title("3 · MSS motifs", fontsize=11, color=T.UP, fontweight="700")
    ax_mss.tick_params(labelsize=7.5); ax_mss.grid(axis="y", visible=False)

    # 4 · BSV themes
    yt = range(len(bio_theme))
    ax_bsv.barh(list(yt), [v for _, v in bio_theme], color=T.SECONDARY, height=0.66)
    ax_bsv.set_yticks(list(yt))
    ax_bsv.set_yticklabels([THEME_SHORT.get(t, t) for t, _ in bio_theme], fontsize=7.6)
    ax_bsv.set_title("4 · Biochemical State Vector", fontsize=11, color=T.INK)
    ax_bsv.set_xlabel("evidence share", fontsize=8.5); ax_bsv.tick_params(labelsize=7.5)
    ax_bsv.grid(axis="y", visible=False)

    # 5 · radar
    _draw_radar(ax_rad, out.radar["axes"], labelsize=8.2)
    ax_rad.set_title("5 · Radar — one visualization of the BSV", fontsize=11, color=T.INK, pad=16)

    # dose label + stats
    for a in (ax_lbl, ax_stat):
        a.axis("off")
    ax_lbl.text(0.5, 0.62, dose_label, ha="center", va="center", fontsize=15,
                fontweight="700", color=T.INK)
    ax_lbl.text(0.5, 0.32, "concentration", ha="center", va="center", fontsize=9, color=T.FAINT)
    b = out.bsv
    stat = (f"confidence   {b.overall_confidence:.2f}\n"
            f"OOD score    {b.ood_score:.2f}\n"
            f"background   {b.non_biochemical.get('background_matrix', 0):.2f}\n"
            f"top motif    {bio_acts[-1].name}")
    ax_stat.text(0.0, 0.6, stat, ha="left", va="center", fontsize=9.2, color=T.MUTED,
                 family="monospace")

    fig.suptitle("The GAIRA reasoning cascade", fontsize=14.5, fontweight="700",
                 color=T.INK, x=0.05, ha="left", y=0.97)
    # connectors (after layout is fixed by add_gridspec explicit positions)
    _arrow_between(fig, ax_spec, ax_comp); _arrow_between(fig, ax_comp, ax_mss)
    _arrow_between(fig, ax_mss, ax_bsv); _arrow_between(fig, ax_bsv, ax_rad, direction="down")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Page 7 — Future DART (CONCEPTUAL figures — clearly labelled, no measurements)
# ══════════════════════════════════════════════════════════════════════════════

def dart_concept_ladder():
    """static point → dose trajectory → DART (potential×time) trajectory."""
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.9))
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2)
    axes[0].scatter([0], [0.1], s=120, color=T.PRIMARY, zorder=3)
    axes[0].set_title("Static spectrum\n→ one BSV point", fontsize=9.5)
    t = np.linspace(0, 1, 8)
    axes[1].plot(t - 0.5, 0.7 * t - 0.3, "-o", color=T.PRIMARY, markersize=4)
    axes[1].set_title("Dose series\n→ BSV trajectory", fontsize=9.5)
    th = np.linspace(0, 2.3 * np.pi, 60)
    axes[2].plot(0.7 * np.cos(th) * (1 - th / 20), 0.7 * np.sin(th) * (1 - th / 20),
                 color=T.UP, linewidth=1.8)
    axes[2].set_title("DART: potential × time\n→ dynamic trajectory", fontsize=9.5)
    fig.suptitle("CONCEPTUAL — how a static point becomes a trajectory", fontsize=10.5,
                 fontweight="700", color=T.MUTED, y=1.06)
    fig.tight_layout()
    return fig


def trajectory_gallery():
    """Eight conceptual electrochemical-trajectory classes (illustrative only)."""
    specs = ["scaling", "redistribution", "reversible loop", "hysteresis",
             "threshold", "delayed kinetics", "irreversible drift", "adsorption switch"]
    fig, axes = plt.subplots(2, 4, figsize=(8.4, 4.4))
    t = np.linspace(0, 1, 80)
    for ax, name in zip(axes.ravel(), specs):
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(name, fontsize=9.2, color=T.INK)
        if name == "scaling":
            ax.plot(t, t, color=T.PRIMARY, lw=2)
        elif name == "redistribution":
            ax.plot(t, np.sin(t * 2.2), color=T.PRIMARY, lw=2)
        elif name == "reversible loop":
            th = np.linspace(0, 2 * np.pi, 80); ax.plot(np.cos(th), np.sin(th) * 0.6, color=T.UP, lw=2)
        elif name == "hysteresis":
            ax.plot(t, t ** 2, color=T.DOWN, lw=2); ax.plot(t, np.sqrt(t), color=T.UP, lw=2)
        elif name == "threshold":
            ax.plot(t, (t > 0.5).astype(float), color=T.PRIMARY, lw=2)
        elif name == "delayed kinetics":
            ax.plot(t, 1 - np.exp(-6 * np.clip(t - 0.3, 0, 1)), color=T.PRIMARY, lw=2)
        elif name == "irreversible drift":
            ax.plot(t, t, color=T.DOWN, lw=2); ax.plot(t, t * 0.4 + 0.1, color=T.UP, lw=2,
                                                       linestyle="--")
        else:  # adsorption switch
            ax.plot(t, 0.5 + 0.4 * np.sign(np.sin(t * 12)), color=T.PRIMARY, lw=1.6)
    fig.suptitle("CONCEPTUAL trajectory vocabulary — illustrative shapes, not measurements",
                 fontsize=10.5, fontweight="700", color=T.MUTED, y=1.02)
    fig.tight_layout()
    return fig


def dart_data_model():
    """The future DART acquisition tensor + processing path (schematic)."""
    fig, ax = plt.subplots(figsize=(8.0, 2.4)); ax.axis("off")
    axes_lbl = ["intensity", "×  wavenumber", "×  potential", "×  time", "×  electrode", "×  waveform"]
    for i, s in enumerate(axes_lbl):
        ax.add_patch(FancyBboxPatch((i * 1.34 + 0.05, 1.05), 1.24, 0.6,
                     boxstyle="round,pad=0.02,rounding_size=0.08", facecolor="#fdeef0",
                     edgecolor=T.UP, linewidth=1.1))
        ax.text(i * 1.34 + 0.67, 1.35, s, ha="center", va="center", fontsize=8.8, color=T.INK)
    path = ["DART spectrum sequence", "frozen-atlas projection / t", "component → MSS → BSV / t",
            "electrochemical-response readout"]
    for i, s in enumerate(path):
        ax.add_patch(FancyBboxPatch((i * 2.05 + 0.05, 0.1), 1.9, 0.55,
                     boxstyle="round,pad=0.02,rounding_size=0.08", facecolor="#eef5fa",
                     edgecolor=T.PRIMARY, linewidth=1.0))
        ax.text(i * 2.05 + 1.0, 0.37, s, ha="center", va="center", fontsize=8.0, color=T.INK)
        if i < len(path) - 1:
            ax.annotate("", xy=(i * 2.05 + 2.0, 0.37), xytext=(i * 2.05 + 1.95, 0.37),
                        arrowprops=dict(arrowstyle="-|>", color=T.FAINT, lw=1.2))
    ax.set_xlim(0, 8.1); ax.set_ylim(0, 1.8)
    ax.text(4.05, 0.85, "each time point → the SAME frozen coordinate system", ha="center",
            fontsize=8.4, color=T.FAINT, style="italic")
    fig.tight_layout()
    return fig


# ── uricase depletion: difference (before/after) ──
def difference_bars(labels, before, after, title="Difference (after − before)",
                    xlabel="Δ share", diverging=True):
    """Signed difference waterfall — components, motifs, or themes."""
    diff = np.asarray(after, float) - np.asarray(before, float)
    order = np.argsort(diff)
    fig, ax = plt.subplots(figsize=(7.2, max(3.0, 0.42 * len(labels))))
    y = range(len(diff))
    colors = [T.UP if diff[i] >= 0 else T.DOWN for i in order]
    ax.barh(list(y), diff[order], color=colors, height=0.66, edgecolor=T.SURFACE, linewidth=0.8)
    ax.axvline(0, color=T.MUTED, linewidth=0.9)
    ax.set_yticks(list(y)); ax.set_yticklabels([labels[i] for i in order], fontsize=8.5)
    ax.set_xlabel(xlabel); ax.set_title(title, fontsize=11.5, pad=8)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def difference_spectrum(grid, before, after, bands=None,
                        title="Difference spectrum (after − before)"):
    fig, ax = plt.subplots(figsize=(7.6, 3.0))
    diff = np.asarray(after) - np.asarray(before)
    ax.axhline(0, color=T.MUTED, linewidth=0.8)
    ax.fill_between(grid, 0, diff, where=diff >= 0, color=T.UP, alpha=0.5)
    ax.fill_between(grid, 0, diff, where=diff < 0, color=T.DOWN, alpha=0.5)
    ax.plot(grid, diff, color=T.INK, linewidth=0.9)
    if bands:
        for bb in bands:
            ax.axvline(bb, color=T.FAINT, linewidth=0.7, linestyle="--", alpha=0.6)
    ax.set_xlabel("Raman shift (cm$^{-1}$)"); ax.set_ylabel("Δ reconstruction")
    ax.set_title(title, fontsize=11.5, pad=8); ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Page 5 — Serum Spike Stress Test figures
# ══════════════════════════════════════════════════════════════════════════════

TIER_COLOR = {"strong": "#2f7d4f", "partial": "#b7791f", "poor": "#b23a48"}


def recoverability_cascade():
    """bulk abundance ≠ surface occupancy ≠ hotspot ≠ recoverable signal."""
    steps = ["Bulk\nabundance", "Surface\noccupancy", "Hotspot\ncontribution",
             "Recoverable\nSERS signal"]
    fig, ax = plt.subplots(figsize=(7.8, 1.7)); ax.axis("off")
    ax.set_xlim(0, len(steps)); ax.set_ylim(0, 1)
    fade = ["#eef5fa", "#dbe9f2", "#cfe0ec", "#c3d8e8"]
    for i, s in enumerate(steps):
        ax.add_patch(FancyBboxPatch((i + 0.06, 0.15), 0.88, 0.7,
                     boxstyle="round,pad=0.02,rounding_size=0.08",
                     facecolor=fade[i], edgecolor=T.PRIMARY, linewidth=1.1))
        ax.text(i + 0.5, 0.5, s, ha="center", va="center", fontsize=9.5,
                fontweight="600", color=T.INK)
        if i < len(steps) - 1:
            ax.text(i + 1.0, 0.5, "≠", ha="center", va="center", fontsize=15,
                    color=T.BAD, fontweight="700")
    fig.tight_layout()
    return fig


def recoverability_scatter(df, annotate=None):
    """All 53 analytes: identity match (x) vs replicate consistency (y), by tier."""
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for tier in ("poor", "partial", "strong"):
        d = df[df.tier == tier]
        ax.scatter(d.cos_spike_vs_pureSERS, d.replicate_direction_cos,
                   s=30 + 400 * d.spike_displacement_norm, color=TIER_COLOR[tier],
                   alpha=0.75, edgecolor="white", linewidth=0.6,
                   label=f"{tier} (n={len(d)})")
    ax.axvline(0.35, color=T.FAINT, linestyle="--", linewidth=0.8)
    ax.axvline(0.10, color=T.FAINT, linestyle=":", linewidth=0.8)
    for a in (annotate or []):
        r = df[df.analyte == a]
        if len(r):
            ax.annotate(a, (r.cos_spike_vs_pureSERS.iloc[0], r.replicate_direction_cos.iloc[0]),
                        fontsize=8.2, color=T.INK, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("identity recovery  ·  cos(serum-spike, pure-SERS fingerprint)")
    ax.set_ylabel("replicate direction consistency")
    ax.set_title("Serum recoverability of 53 spiked analytes  ·  marker size = displacement",
                 fontsize=11.0, pad=8)
    ax.legend(fontsize=8.4, loc="lower left")
    fig.tight_layout()
    return fig


def recoverability_heatmap(analytes, matrix, themes, title="ΔBSV per analyte × theme"):
    m = np.asarray(matrix, float)
    vmax = np.abs(m).max() or 1.0
    fig, ax = plt.subplots(figsize=(7.8, max(3.2, 0.34 * len(analytes))))
    im = ax.imshow(m, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(themes)))
    ax.set_xticklabels([THEME_SHORT.get(t, t) for t in themes], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(analytes))); ax.set_yticklabels(analytes, fontsize=8)
    cb = fig.colorbar(im, ax=ax, shrink=0.7); cb.set_label("Δ evidence share", fontsize=8.5)
    ax.set_title(title, fontsize=11.0, pad=8)
    fig.tight_layout()
    return fig


def confidence_limitation(cdf, title="Confidence does not track recoverability"):
    """Engine confidence vs identity recovery — the key limitation to surface."""
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for tier in ("poor", "partial", "strong"):
        d = cdf[cdf.tier == tier]
        ax.scatter(d.cos, d.confidence, s=45, color=TIER_COLOR[tier], alpha=0.75,
                   edgecolor="white", linewidth=0.6, label=tier)
    # trend
    if len(cdf) > 2:
        z = np.polyfit(cdf.cos, cdf.confidence, 1)
        xs = np.linspace(cdf.cos.min(), cdf.cos.max(), 20)
        ax.plot(xs, np.polyval(z, xs), "--", color=T.FAINT, linewidth=1.2,
                label=f"slope {z[0]:+.03f}")
    ax.set_xlabel("identity recovery  ·  cos(serum-spike, pure-SERS)")
    ax.set_ylabel("engine overall confidence")
    ax.set_title(title, fontsize=11.0, pad=8)
    ax.legend(fontsize=8.4, loc="best")
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Page 6 — Biological Studies figures
# ══════════════════════════════════════════════════════════════════════════════

GROUP_COLORS = [T.PRIMARY, T.UP, T.GOOD, T.WARN]


def multi_radar(series, title="Group biochemical state (absolute atlas position)"):
    """Overlay several groups' absolute-BSV radars (composition share)."""
    peak = max(max(a["score"] for a in s["axes"]) for s in series)
    vmax = peak * 1.18 if peak > 0 else 1.0
    fig, ax = plt.subplots(figsize=(5.8, 5.8), subplot_kw={"polar": True})
    labels = [THEME_SHORT.get(a["theme"], a["theme"]) for a in series[0]["axes"]]
    ang = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    ang_c = np.concatenate([ang, ang[:1]])
    ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1); ax.set_ylim(0, vmax)
    ax.set_yticks([vmax / 2]); ax.set_yticklabels([f"{vmax/2:.2f}"], color=T.FAINT, fontsize=7.5)
    ax.set_xticks(ang); ax.set_xticklabels(labels, fontsize=8.6, color=T.INK)
    for k, s in enumerate(series):
        v = [a["score"] for a in s["axes"]]; vc = np.array(v + v[:1])
        c = s.get("color", GROUP_COLORS[k % len(GROUP_COLORS)])
        ax.plot(ang_c, vc, color=c, linewidth=2.0, label=s["name"])
        ax.fill(ang_c, vc, color=c, alpha=0.10)
    ax.grid(color=T.GRID, linewidth=0.7); ax.spines["polar"].set_color(T.PANEL_EDGE)
    ax.set_title(title, fontsize=11.5, color=T.INK, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.12), fontsize=8.6)
    fig.tight_layout()
    return fig


def forest_plot(rows, a, b, title="ΔBSV effect sizes with 95% bootstrap CI"):
    """Signed theme difference (a−b) with bootstrap CI, coloured by FDR significance,
    Cliff's delta annotated. Effect size is emphasised over the p-value."""
    rows = sorted(rows, key=lambda r: r["delta"])
    y = np.arange(len(rows))
    delta = [r["delta"] for r in rows]
    lo = [r["delta"] - r["ci_lo"] for r in rows]; hi = [r["ci_hi"] - r["delta"] for r in rows]
    colors = [T.GOOD if r["sig"] else T.FAINT for r in rows]
    fig, ax = plt.subplots(figsize=(7.4, max(3.0, 0.42 * len(rows))))
    ax.errorbar(delta, y, xerr=[lo, hi], fmt="none", ecolor=T.FAINT, elinewidth=1.2, capsize=3, zorder=1)
    ax.scatter(delta, y, c=colors, s=55, zorder=3, edgecolor="white", linewidth=0.6)
    ax.axvline(0, color=T.MUTED, linewidth=0.9)
    ax.set_yticks(y); ax.set_yticklabels([THEME_SHORT.get(r["theme"], r["theme"]) for r in rows],
                                         fontsize=9)
    for yi, r in enumerate(rows):
        ax.text(max(hi[yi] + r["delta"], r["delta"]) + 0.001 * (1 if r["delta"] >= 0 else 1),
                yi, f"  δ={r['cliffs_delta']:+.2f}", va="center", fontsize=7.4, color=T.MUTED)
    ax.set_xlabel(f"Δ evidence share ({a} − {b})")
    ax.set_title(title + f"  ·  green = FDR q<0.05", fontsize=10.8, pad=8)
    ax.grid(axis="y", visible=False); ax.margins(x=0.22)
    fig.tight_layout()
    return fig


def bio_pca(proj, groups, var, title="Sample space (PCA of BSV)"):
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    for k, g in enumerate(sorted(set(groups))):
        m = np.array(groups) == g
        c = GROUP_COLORS[k % len(GROUP_COLORS)]
        ax.scatter(proj[m, 0], proj[m, 1], s=34, color=c, alpha=0.65,
                   edgecolor="white", linewidth=0.4, label=g)
        # group centroid
        ax.scatter(proj[m, 0].mean(), proj[m, 1].mean(), s=180, color=c, marker="X",
                   edgecolor=T.INK, linewidth=1.0, zorder=5)
    ax.set_xlabel(f"BSV-PC1 ({var[0]:.0%})"); ax.set_ylabel(f"BSV-PC2 ({var[1]:.0%})")
    ax.set_title(title, fontsize=11.5, pad=8); ax.legend(fontsize=8.6, loc="best")
    fig.tight_layout()
    return fig


def group_quality(art, title="Data-quality panel"):
    """OOD / confidence / background-share distributions per group (box)."""
    groups = sorted(set(art["group"]))
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 3.0))
    for ax, key, lab in zip(axes, ["ood", "conf", "bg"], ["OOD", "confidence", "matrix share"]):
        data = [art[key][art["group"] == g] for g in groups]
        bp = ax.boxplot(data, patch_artist=True, widths=0.6)
        for k, box in enumerate(bp["boxes"]):
            box.set(facecolor=GROUP_COLORS[k % len(GROUP_COLORS)], alpha=0.55)
        for med in bp["medians"]:
            med.set(color=T.INK, linewidth=1.2)
        ax.set_xticklabels(groups, fontsize=7.6, rotation=15); ax.set_title(lab, fontsize=10)
        ax.grid(axis="x", visible=False)
    fig.suptitle(title, fontsize=11.5, fontweight="700", y=1.05)
    fig.tight_layout()
    return fig


def sample_heatmap(Z, groups, labels, title="Sample-level BSV heatmap (z-scored, display only)"):
    """Rows = samples (grouped), cols = themes/motifs, z-scored for VISUALIZATION."""
    groups = np.array(groups)
    order = np.argsort(groups, kind="stable")
    Zo = Z[order]; go = groups[order]
    uniq = list(dict.fromkeys(go))
    gcol = {g: GROUP_COLORS[i % len(GROUP_COLORS)] for i, g in enumerate(uniq)}
    fig, (axg, ax) = plt.subplots(1, 2, figsize=(7.8, max(3.4, 0.045 * len(Zo))),
                                  gridspec_kw={"width_ratios": [0.03, 1]}, sharey=True)
    axg.imshow(np.array([[uniq.index(g)] for g in go]), aspect="auto",
               cmap=plt.matplotlib.colors.ListedColormap([gcol[g] for g in uniq]))
    axg.set_xticks([]); axg.set_yticks([]); axg.set_ylabel("samples", fontsize=9)
    vmax = np.abs(Zo).max() or 1.0
    im = ax.imshow(Zo, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([THEME_SHORT.get(l, l) for l in labels], rotation=45, ha="right", fontsize=8)
    ax.set_yticks([])
    cb = fig.colorbar(im, ax=ax, shrink=0.6); cb.set_label("z (display)", fontsize=8)
    handles = [plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=gcol[g], markersize=8,
                          label=g) for g in uniq]
    ax.legend(handles=handles, fontsize=8, loc="upper right", bbox_to_anchor=(1.35, 1.0))
    fig.suptitle(title, fontsize=11.0, fontweight="700", y=1.02)
    fig.tight_layout()
    return fig


def paired_slope(a, b, paired, name, title="Paired change per stratum"):
    """Longitudinal slope plot: baseline→later per stratum (e.g. Day0→Day2 per dose)."""
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for i, (s, row) in enumerate(sorted(paired.items())):
        y0, y1 = row[b], row[a]
        c = GROUP_COLORS[i % len(GROUP_COLORS)]
        ax.plot([0, 1], [y0, y1], "-o", color=c, markersize=6, linewidth=1.8, label=s)
        ax.annotate(s, (1, y1), fontsize=8.5, color=c, xytext=(6, 0), textcoords="offset points",
                    va="center")
    ax.set_xticks([0, 1]); ax.set_xticklabels([b, a], fontsize=10)
    ax.set_ylabel(f"{THEME_SHORT.get(name, name)} composition")
    ax.set_title(title, fontsize=11.5, pad=8); ax.set_xlim(-0.2, 1.35)
    ax.legend(fontsize=8, title="stratum", loc="best")
    fig.tight_layout()
    return fig


def distance_bars(dist, title="Group difference vs biological heterogeneity"):
    fig, ax = plt.subplots(figsize=(5.6, 2.6))
    ax.barh(["between-group\ndistance", "within-group\nvariability"],
            [dist["between_group"], dist["within_group"]],
            color=[T.UP, T.FAINT], height=0.6)
    ax.set_title(f"{title}   ·   ratio = {dist['ratio']:.2f}", fontsize=10.5, pad=6)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def balanced_bars(themes, za, zb, a, b, title="Exploratory balanced view (standardized, display only)"):
    """Standardized per-theme deviation per group — reveals structure a dominant axis
    would otherwise obscure. Visualization only; canonical BSV unchanged."""
    y = np.arange(len(themes)); w = 0.4
    fig, ax = plt.subplots(figsize=(7.0, max(3.0, 0.42 * len(themes))))
    ax.barh(y - w / 2, za, w, color=T.UP, label=a)
    ax.barh(y + w / 2, zb, w, color=T.PRIMARY, label=b)
    ax.axvline(0, color=T.MUTED, linewidth=0.9)
    ax.set_yticks(y); ax.set_yticklabels([THEME_SHORT.get(t, t) for t in themes], fontsize=9)
    ax.set_xlabel("standardized theme deviation (z)")
    ax.set_title(title, fontsize=10.8, pad=8); ax.legend(fontsize=8.5); ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def study_centroid_map(sc, title="Cross-study biochemical centroids (BSV space)"):
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    proj, labels, ood = sc["proj"], sc["labels"], sc["ood"]
    sctr = ax.scatter(proj[:, 0], proj[:, 1], c=ood, cmap="magma_r", s=120,
                      edgecolor=T.INK, linewidth=0.8, zorder=3)
    for (x, y), lab in zip(proj, labels):
        ax.annotate(lab, (x, y), fontsize=7.8, color=T.INK, xytext=(5, 4),
                    textcoords="offset points")
    cb = fig.colorbar(sctr, ax=ax, shrink=0.8); cb.set_label("mean OOD", fontsize=9)
    ax.set_xlabel(f"BSV-PC1 ({sc['var'][0]:.0%})"); ax.set_ylabel(f"BSV-PC2 ({sc['var'][1]:.0%})")
    ax.set_title(title, fontsize=11.0, pad=8)
    fig.tight_layout()
    return fig


# ── compare all three trajectory classes in one BSV space ──
def compare_trajectories(trajs, title="Three perturbation classes in BSV space"):
    """trajs: list of dicts {name, proj (n,2), color, marker}."""
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    for tr in trajs:
        p = tr["proj"]
        ax.plot(p[:, 0], p[:, 1], "-", color=tr["color"], linewidth=1.5, alpha=0.7, zorder=2)
        ax.scatter(p[:, 0], p[:, 1], color=tr["color"], s=40, zorder=3,
                   edgecolor="white", linewidth=0.5, label=tr["name"], marker=tr.get("marker", "o"))
        ax.scatter(p[0, 0], p[0, 1], s=120, facecolor="none", edgecolor=T.INK,
                   linewidth=1.3, zorder=4)
        ax.annotate("", xy=p[-1], xytext=p[-2] if len(p) > 1 else p[0],
                    arrowprops=dict(arrowstyle="-|>", color=tr["color"], lw=1.6))
    ax.set_xlabel("BSV-PC1"); ax.set_ylabel("BSV-PC2")
    ax.set_title(title, fontsize=12.0, pad=8)
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    return fig
