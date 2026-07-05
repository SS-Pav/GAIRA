#!/usr/bin/env python3
"""GAIRA diabetes manuscript — publication-quality figure regeneration.

This script does NOT change any underlying analysis. It reads the outputs from
the latest `results/diabetes_gaira_audit_*/` run, recomputes per-patient MSS
fires (needed for Figure 3 bootstrap CI) using the same overrides, and
produces manuscript-ready:

  Figure 1  — normalized biochemical state radar, OWD vs NWD
  Figure 2  — normalized biochemical state radar, 4 subgroups
  Figure 3  — differential grounded analyte evidence forest plot (NEW)
  Figure 4  — differential biochemical states forest plot (renamed)
  Supplementary Table S1  — interpretation of GAIRA biochemical states
  figure_captions.md      — long-form captions per figure

Output: `publication_figures_v2/` inside the latest audit results directory.

Terminology follows the manuscript convention:
    "Mechanistic z-score"  → "Normalized biochemical state"
    "Raw BSV magnitude"    → "Absolute biochemical state"
    "fire score"           → "Grounded evidence score"
    "Analyte hits"         → "Grounded analyte evidence"
    "Biochemical family"   → "Associated biochemical class"
    axis names take the "-associated" suffix
"""
from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy import stats
from scipy.io import loadmat

DEMO_ROOT = Path("/Users/suraj/projects/GAIRA/gaira_demo_reasoning_v1")
ANALYSIS  = Path(__file__).resolve().parent
sys.path.insert(0, str(DEMO_ROOT))
sys.path.insert(0, str(ANALYSIS))

from gaira_core import config as gcfg
from gaira_core.data_loader import MOLECULES
from gaira_core.mss_scoring import score_all as mss_score_all
from _diabetes_overrides import build_report_diabetes    # noqa: E402

warnings.filterwarnings("ignore")
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────
#  PATH DISCOVERY
# ─────────────────────────────────────────────────────────────────────
RESULTS_ROOT = Path("/Users/suraj/projects/GAIRA/results")
audit_dirs = sorted([d for d in RESULTS_ROOT.iterdir()
                        if d.is_dir() and d.name.startswith("diabetes_gaira_audit_")])
if not audit_dirs:
    raise SystemExit("no diabetes_gaira_audit_* directory found; run the audit first.")
AUDIT_DIR = audit_dirs[-1]
OUT = AUDIT_DIR / "publication_figures_v2"
OUT.mkdir(exist_ok=True)
print(f"[pub-figs] reading from   {AUDIT_DIR}")
print(f"[pub-figs] writing to     {OUT}")


# ─────────────────────────────────────────────────────────────────────
#  TERMINOLOGY MAPS
# ─────────────────────────────────────────────────────────────────────
PUB_AXIS_LABELS = {
    "G01_purine_nucleotide":         "Purine-associated\n(Nucleic acids)",
    "G02_purine_metabolite":         "Purine-associated\n(Metabolism)",
    "G03_pyrimidine_nucleotide":     "Pyrimidine-\nassociated",
    "G04_nucleic_acid_phosphate":    "Nucleic acid\nbackbone-associated",
    "G05_glycan_carbohydrate":       "Glycan-\nassociated",
    "G06_protein_peptide_backbone":  "Protein-\nassociated",
    "G07_aromatic_residue":          "Aromatic biomolecule-\nassociated",
    "G08_lipid_acyl_membrane":       "Membrane lipid-\nassociated",
    "G09_sterol_neutral_lipid":      "Sterol-\nassociated",
    "G10_sulfur_thiol_redox":        "Redox-\nassociated",
    "G11_metabolic_small_molecule":  "Small molecule-\nassociated",
}
PUB_AXIS_ONE_LINE = {
    "G01_purine_nucleotide":         "Purine-associated (Nucleic acids)",
    "G02_purine_metabolite":         "Purine-associated (Metabolism)",
    "G03_pyrimidine_nucleotide":     "Pyrimidine-associated",
    "G04_nucleic_acid_phosphate":    "Nucleic acid backbone-associated",
    "G05_glycan_carbohydrate":       "Glycan-associated",
    "G06_protein_peptide_backbone":  "Protein-associated",
    "G07_aromatic_residue":          "Aromatic biomolecule-associated",
    "G08_lipid_acyl_membrane":       "Membrane lipid-associated",
    "G09_sterol_neutral_lipid":      "Sterol-associated",
    "G10_sulfur_thiol_redox":        "Redox-associated",
    "G11_metabolic_small_molecule":  "Small molecule-associated",
}

# Associated biochemical class per axis (used for family palette in forest plots)
AXIS_CLASS = {
    "G01_purine_nucleotide":       "Nucleic acid",
    "G02_purine_metabolite":       "Nucleic acid",
    "G03_pyrimidine_nucleotide":   "Nucleic acid",
    "G04_nucleic_acid_phosphate":  "Nucleic acid",
    "G05_glycan_carbohydrate":     "Glycan",
    "G06_protein_peptide_backbone":"Protein",
    "G07_aromatic_residue":        "Protein",
    "G08_lipid_acyl_membrane":     "Membrane lipid",
    "G09_sterol_neutral_lipid":    "Sterol / neutral lipid",
    "G10_sulfur_thiol_redox":      "Redox",
    "G11_metabolic_small_molecule":"Small-molecule metabolism",
}
CLASS_COLOR = {
    "Nucleic acid":            "#4E79A7",  # Tableau-Nature style muted blue
    "Glycan":                  "#59A14F",
    "Protein":                 "#F28E2B",
    "Membrane lipid":          "#17BECF",
    "Sterol / neutral lipid":  "#8CD17D",
    "Redox":                   "#E15759",
    "Small-molecule metabolism": "#B07AA1",
}

# 2-group and 4-subgroup colors — colorblind-safe (Wong palette flavour)
COLOR_OWD = "#D62728"
COLOR_NWD = "#1F77B4"
SUBGROUP_COLORS = {
    "Asian Impact":    "#D62728",
    "Asian Strong-D":  "#1F77B4",
    "White Impact":    "#F58518",
    "White Strong-D":  "#2CA02C",
}


# ─────────────────────────────────────────────────────────────────────
#  PUBLICATION STYLE
# ─────────────────────────────────────────────────────────────────────
PUB_STYLE = {
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size":          11,
    "axes.titlesize":     14,
    "axes.labelsize":     12,
    "axes.linewidth":     1.0,
    "axes.edgecolor":     "#1F2937",
    "axes.labelcolor":    "#111827",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "xtick.labelsize":    11,
    "ytick.labelsize":    11,
    "xtick.color":        "#111827",
    "ytick.color":        "#111827",
    "legend.fontsize":    11,
    "legend.frameon":     False,
    "figure.facecolor":   "white",
    "figure.dpi":         120,
    "savefig.dpi":        600,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.15,
    "lines.linewidth":    1.8,
    "pdf.fonttype":       42,       # editable text in PDF/SVG
    "ps.fonttype":        42,
    "svg.fonttype":       "none",
}
plt.rcParams.update(PUB_STYLE)


def _save(fig, name: str):
    """Save a figure in PDF, SVG and 600-dpi PNG."""
    for ext in ("pdf", "svg", "png"):
        transparent = (ext != "png")
        fig.savefig(OUT / f"{name}.{ext}", transparent=transparent)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
#  DATA LOADERS (from the audit run)
# ─────────────────────────────────────────────────────────────────────
bsv = pd.read_csv(AUDIT_DIR / "diabetes_gaira_scores_per_sample.csv")
stats_2 = pd.read_csv(AUDIT_DIR / "diabetes_group_summary_2group.csv")
stats_4 = pd.read_csv(AUDIT_DIR / "diabetes_group_summary_4subgroup.csv")
analyte = pd.read_csv(AUDIT_DIR / "diabetes_analyte_hits.csv")
zscore_2 = pd.read_csv(AUDIT_DIR / "diabetes_zscore_2group.csv")
zscore_4 = pd.read_csv(AUDIT_DIR / "diabetes_zscore_4subgroup.csv")


# ─────────────────────────────────────────────────────────────────────
#  RE-RUN THE PIPELINE JUST TO PRODUCE PER-PATIENT MSS FIRES
#  (needed for Figure 3 bootstrap CI)
# ─────────────────────────────────────────────────────────────────────
SSD_RAW = Path("/Volumes/SSD_Rad/GAIRA_DATA/raw/diabetes_plasma_ev_sers/extracted")
CAL_PIX = np.array([263, 367, 492, 512, 590, 782, 872, 887], dtype=float)
CAL_WN  = np.array([620.9, 795.8, 1001.4, 1031.8, 1155.3, 1450.5, 1583.1, 1602.3])
CROP_PIX_START, CROP_PIX_END = 162, 898

def _diabetes_wavenumbers() -> np.ndarray:
    coef = np.polyfit(CAL_PIX, CAL_WN, 3)
    full = np.arange(1, 1651)
    return np.polyval(coef, full)[CROP_PIX_START - 1 : CROP_PIX_END]

def _load_raw() -> dict[str, np.ndarray]:
    meta = pd.read_csv(SSD_RAW / "patient_data.csv")
    out = {}
    for prefix, mat in [("2151", "RawDataImpact.mat"), ("32113", "RawDataStrong.mat")]:
        m = loadmat(SSD_RAW / mat, squeeze_me=True)
        sp = m["smoothed_spectra"]
        pids = sorted(meta[meta["filename"].str.startswith(prefix)]["filename"].tolist())
        for i in range(min(len(sp), len(pids))):
            arr = np.asarray(sp[i], dtype=float)
            if arr.ndim == 2 and arr.shape[0] == 737:
                arr = arr.T
            out[pids[i]] = arr
    return out

def _interp(wn_native: np.ndarray, y: np.ndarray) -> np.ndarray:
    grid = np.linspace(gcfg.WAVENUMBER_MIN, gcfg.WAVENUMBER_MAX, gcfg.WAVENUMBER_N)
    order = np.argsort(wn_native)
    return np.interp(grid, wn_native[order], y[order], left=0.0, right=0.0)


def compute_per_patient_mss() -> pd.DataFrame:
    """Reproduces the audit script's per-patient MSS fires. Cached to CSV."""
    cache = OUT / "_cache_per_patient_mss.csv"
    if cache.exists():
        return pd.read_csv(cache)
    print("[pub-figs] recomputing per-patient MSS fires for Figure 3 bootstrap …")
    wn_native = _diabetes_wavenumbers()
    grid = np.linspace(gcfg.WAVENUMBER_MIN, gcfg.WAVENUMBER_MAX, gcfg.WAVENUMBER_N)
    spectra = _load_raw()
    rows = []
    for _, r in bsv.iterrows():
        pid = r["patient_id"]
        raw = spectra.get(pid)
        if raw is None:
            continue
        mean = raw.mean(axis=0)
        on_grid = np.clip(_interp(wn_native, mean), 0.0, None)
        rep = build_report_diabetes(
            sample_id=pid, title="", domain="extracellular_vesicle",
            substrate="Ag colloid SERS", wavenumber=grid, intensity=on_grid,
        )
        # Re-score MSS on the processed spectrum (same as inside build_report)
        pi = np.asarray(rep["spectrum"]["processed_intensity"])
        fires = mss_score_all(grid, pi)
        for mol_id, fire in fires.items():
            rows.append({
                "patient_id":    pid,
                "group_2":       r["group_2"],
                "subgroup_4":    r["subgroup_4"],
                "molecule_id":   mol_id,
                "molecule_name": MOLECULES[mol_id].name,
                "primary_axis":  MOLECULES[mol_id].primary_axis,
                "anchor_score":  float(fire.anchor_score),
                "support_score": float(fire.support_score),
                "anti_score":    float(fire.anti_score),
                "grounded_evidence_score": float(fire.fire),
            })
    df = pd.DataFrame(rows)
    df.to_csv(cache, index=False)
    return df


# ─────────────────────────────────────────────────────────────────────
#  RADAR HELPERS
# ─────────────────────────────────────────────────────────────────────
def _shade_class_wedges(ax, radial_lim: tuple):
    axes = list(gcfg.BSV_AXES)
    N = len(axes)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    height = radial_lim[1] - radial_lim[0]
    for i, a in enumerate(axes):
        color = CLASS_COLOR.get(AXIS_CLASS[a], "#94A3B8")
        ax.bar([angles[i]], [height], width=2 * np.pi / N, bottom=radial_lim[0],
                 color=color, alpha=0.06, edgecolor="none", zorder=0)


def _radar_axis_labels(ax, sig_axes: set[str], effects: dict[str, float] | None = None):
    axes = list(gcfg.BSV_AXES)
    angles = np.linspace(0, 2 * np.pi, len(axes), endpoint=False)
    labels = []
    for i, a in enumerate(axes):
        base = PUB_AXIS_LABELS[a]
        if a in sig_axes and effects and a in effects:
            d = effects[a]
            star = "**" if abs(d) >= 1.0 else "*"
            labels.append(f"{base}\n{star} d = {d:+.2f}")
        else:
            labels.append(base)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold", color="#111827")
    # nudge label radial position out slightly
    for lbl, ang in zip(ax.get_xticklabels(), angles):
        lbl.set_rotation_mode("anchor")


def _radar_ticks(ax, radial_lim: tuple):
    if radial_lim[0] < 0:
        ticks = np.array([-1.5, -0.75, 0.0, 0.75, 1.5])
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{v:+.2f}" for v in ticks],
                             fontsize=9, color="#4B5563")
        # bold zero circle
        theta_full = np.linspace(0, 2 * np.pi, 360)
        ax.plot(theta_full, [0]*360, color="#374151", linewidth=1.4,
                 alpha=0.9, zorder=5)
    else:
        ticks = np.linspace(radial_lim[0], radial_lim[1], 5)[1:-1]
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{v:.2f}" for v in ticks],
                             fontsize=9, color="#4B5563")


def _radar_trace(ax, values_by_group: dict, colors: dict, line_width: float = 2.8):
    axes = list(gcfg.BSV_AXES)
    angles = np.linspace(0, 2 * np.pi, len(axes), endpoint=False)
    for label, vals in values_by_group.items():
        # strip trailing "(n=..)" for color lookup
        key = label.split(" (")[0]
        color = colors.get(key, "#64748B")
        r = np.array([vals.get(a, 0.0) for a in axes], dtype=float)
        r_c = np.append(r, r[0])
        th = np.append(angles, angles[0])
        ax.plot(th, r_c, color=color, linewidth=line_width, label=label, zorder=3)
        ax.fill(th, r_c, color=color, alpha=0.13, zorder=2)


def _polar_setup(ax, radial_lim: tuple):
    ax.set_ylim(radial_lim)
    ax.grid(color="#CBD5E1", linewidth=0.7, alpha=0.85)
    ax.spines["polar"].set_color("#94A3B8")
    ax.spines["polar"].set_linewidth(1.0)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)


# ─────────────────────────────────────────────────────────────────────
#  FIGURE 1 — normalized biochemical state, OWD vs NWD
# ─────────────────────────────────────────────────────────────────────
def figure_1():
    n_owd = int((bsv["group_2"] == "OWD").sum())
    n_nwd = int((bsv["group_2"] == "NWD").sum())
    z = zscore_2.set_index("cohort")

    sig_axes = set(stats_2.loc[stats_2["q_value_fdr_bh"] < 0.05, "axis"])
    effects  = dict(zip(stats_2["axis"], stats_2["cohens_d"]))

    fig = plt.figure(figsize=(11.5, 10.0))
    ax = fig.add_subplot(111, polar=True)
    _polar_setup(ax, (-1.8, 1.8))
    _shade_class_wedges(ax, (-1.8, 1.8))

    vals = {
        f"OWD (n={n_owd})": z.loc["OWD"].to_dict(),
        f"NWD (n={n_nwd})": z.loc["NWD"].to_dict(),
    }
    _radar_trace(ax, vals, {"OWD": COLOR_OWD, "NWD": COLOR_NWD}, line_width=2.8)
    _radar_axis_labels(ax, sig_axes, effects)
    _radar_ticks(ax, (-1.8, 1.8))

    # Title + subtitle
    fig.text(0.5, 0.965,
              "Figure 1  ·  GAIRA biochemical state distinguishes obesity-associated "
              "and non-obesity-associated diabetes",
              ha="center", fontsize=14, weight="bold", color="#0F172A")
    fig.text(0.5, 0.925,
              "Normalized biochemical state — relative deviation from the pooled cohort "
              "biochemical reference state",
              ha="center", fontsize=11.5, color="#334155", style="italic")

    # Legend: outside upper-right, larger
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.06), fontsize=11.5,
                title="Cohort", title_fontsize=11.5)

    fig.text(0.5, 0.03,
              "Significant axes marked ** for |d| ≥ 1.0 and * for q < 0.05 "
              "(Mann–Whitney with Benjamini–Hochberg correction). Radial background "
              "wedges are colored by associated biochemical class.",
              ha="center", fontsize=9.5, color="#475569", style="italic")

    _save(fig, "figure_1_biochemical_state_2group")


# ─────────────────────────────────────────────────────────────────────
#  FIGURE 2 — normalized biochemical state, 4 subgroups
# ─────────────────────────────────────────────────────────────────────
def figure_2():
    sub = bsv[bsv["subgroup_4"].notna()]
    counts = sub["subgroup_4"].value_counts().to_dict()
    z = zscore_4.set_index("cohort")

    sig_axes = set(stats_4.loc[stats_4["q_value_fdr_bh"] < 0.05, "axis"])

    fig = plt.figure(figsize=(12.5, 10.5))
    ax = fig.add_subplot(111, polar=True)
    _polar_setup(ax, (-1.5, 1.5))
    _shade_class_wedges(ax, (-1.5, 1.5))

    vals = {}
    order = ["White Impact", "Asian Impact", "White Strong-D", "Asian Strong-D"]
    for s in order:
        if s not in z.index: continue
        vals[f"{s} (n={counts.get(s, 0)})"] = z.loc[s].to_dict()

    _radar_trace(ax, vals, SUBGROUP_COLORS, line_width=2.4)
    _radar_axis_labels(ax, sig_axes, effects=None)
    _radar_ticks(ax, (-1.5, 1.5))

    fig.text(0.5, 0.965,
              "Figure 2  ·  GAIRA biochemical states across diabetes subgroups",
              ha="center", fontsize=14, weight="bold", color="#0F172A")
    fig.text(0.5, 0.925,
              "Normalized biochemical state — relative deviation from the pooled "
              "cohort biochemical reference state",
              ha="center", fontsize=11.5, color="#334155", style="italic")

    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.06), fontsize=11,
                title="Subgroup", title_fontsize=11)

    fig.text(0.5, 0.03,
              "Significant axes marked * (Kruskal–Wallis q < 0.05, "
              "Benjamini–Hochberg correction). Radial background wedges are "
              "colored by associated biochemical class.",
              ha="center", fontsize=9.5, color="#475569", style="italic")

    _save(fig, "figure_2_biochemical_state_4subgroup")


# ─────────────────────────────────────────────────────────────────────
#  FIGURE 3 — Differential grounded analyte evidence (NEW)
# ─────────────────────────────────────────────────────────────────────
def figure_3(mss_pp: pd.DataFrame):
    """Forest plot ranking analytes by the OWD − NWD standardized difference of
    their grounded evidence score, with 95% bootstrap CI and confidence tier."""
    axes = MOLECULES

    def _cohens_d(a, b):
        a, b = np.asarray(a), np.asarray(b)
        if len(a) < 2 or len(b) < 2:
            return np.nan
        sa, sb = np.var(a, ddof=1), np.var(b, ddof=1)
        pooled = np.sqrt(((len(a) - 1) * sa + (len(b) - 1) * sb)
                            / max(1, len(a) + len(b) - 2))
        return (a.mean() - b.mean()) / max(pooled, 1e-12)

    rng = np.random.default_rng(42)
    rows = []
    for mol_id, ref in axes.items():
        sub = mss_pp[mss_pp["molecule_id"] == mol_id]
        owd = sub[sub["group_2"] == "OWD"]["grounded_evidence_score"].to_numpy()
        nwd = sub[sub["group_2"] == "NWD"]["grounded_evidence_score"].to_numpy()
        if len(owd) < 3 or len(nwd) < 3:
            continue
        d_med = _cohens_d(owd, nwd)
        # bootstrap
        ds = np.empty(2000)
        for i in range(2000):
            ds[i] = _cohens_d(rng.choice(owd, size=len(owd), replace=True),
                                 rng.choice(nwd, size=len(nwd), replace=True))
        d_lo, d_hi = np.nanpercentile(ds, [2.5, 97.5])
        try:
            _, p = stats.mannwhitneyu(owd, nwd, alternative="two-sided")
        except ValueError:
            p = np.nan
        rows.append({
            "molecule_id":     mol_id,
            "molecule_name":   ref.name,
            "biochemical_class": AXIS_CLASS.get(ref.primary_axis, "Other"),
            "primary_axis":    ref.primary_axis,
            "d_median":        d_med,
            "d_lo":            d_lo,
            "d_hi":            d_hi,
            "p_value":         p,
            "n_owd":           len(owd),
            "n_nwd":           len(nwd),
            "owd_mean_gscore": float(np.mean(owd)),
            "nwd_mean_gscore": float(np.mean(nwd)),
        })
    df = pd.DataFrame(rows)
    # BH-FDR
    p = df["p_value"].fillna(1.0).to_numpy()
    n = len(p); order = np.argsort(p); q = np.empty(n); prev = 1.0
    for i in range(n - 1, -1, -1):
        prev = min(prev, p[order[i]] * n / (i + 1))
        q[order[i]] = min(1.0, prev)
    df["q_value_fdr_bh"] = q

    # Confidence tier — parity with the mainline analyte table but expressed for pub
    #    High:   |d| ≥ 1.0 AND q < 0.05 AND OWD mean g-score > 0.03 (evidence above pipeline floor)
    #    Medium: |d| ≥ 0.4 AND OWD mean g-score > 0.02
    #    Low:    otherwise
    def _tier(r):
        d = abs(r["d_median"])
        ok_q = r["q_value_fdr_bh"] < 0.05
        anchor_lvl = max(r["owd_mean_gscore"], r["nwd_mean_gscore"])
        if d >= 1.0 and ok_q and anchor_lvl > 0.03:
            return "High"
        if d >= 0.4 and anchor_lvl > 0.02:
            return "Medium"
        return "Low"
    df["confidence"] = df.apply(_tier, axis=1)
    df = df.sort_values("d_median")     # NWD-side at top, OWD-side at bottom

    TIER_COLOR = {"High": "#059669", "Medium": "#F59E0B", "Low": "#94A3B8"}
    df.to_csv(OUT / "figure_3_data.csv", index=False)

    n_rows = len(df)
    fig, ax = plt.subplots(figsize=(9.5, max(6.2, n_rows * 0.55)))
    y = np.arange(n_rows)
    for i, (_, r) in enumerate(df.iterrows()):
        col = TIER_COLOR.get(r["confidence"], "#94A3B8")
        ax.plot([r["d_lo"], r["d_hi"]], [i, i], color=col, linewidth=2.4, alpha=0.9, solid_capstyle="round")
        ax.scatter([r["d_median"]], [i], color=col, s=95, edgecolor="white",
                     linewidth=1.0, zorder=5)
    ax.axvline(0, color="#374151", linewidth=1.0, alpha=0.8)
    ax.axvspan(-1.0, 1.0, color="#E2E8F0", alpha=0.28, zorder=0)

    # y-axis labels: analyte + direction (plain-text, publication-safe)
    y_labels = []
    for _, r in df.iterrows():
        direction = "Higher in OWD" if r["d_median"] > 0 else "Higher in NWD"
        star = "**" if r["q_value_fdr_bh"] < 0.001 else ("*" if r["q_value_fdr_bh"] < 0.05 else "")
        y_labels.append(f"{r['molecule_name']}  {star}\n{direction}  ·  {r['biochemical_class']}")
    ax.set_yticks(y)
    ax.set_yticklabels(y_labels, fontsize=10.5)

    # cosmetics
    ax.set_xlabel("Standardized effect size (Cohen's d), OWD − NWD",
                    fontsize=12, weight="bold")
    ax.set_xlim(-3.3, 3.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#94A3B8")
    ax.spines["bottom"].set_color("#94A3B8")
    ax.tick_params(axis="both", length=3, color="#94A3B8")

    # Title
    fig.text(0.02, 0.965,
              "Figure 3  ·  Differential grounded analyte evidence between "
              "obesity-associated and non-obesity-associated diabetes",
              ha="left", fontsize=13.5, weight="bold", color="#0F172A")
    fig.text(0.02, 0.930,
              "Analytes ranked by the standardized difference of the grounded "
              "evidence score, with 95% bootstrap CIs.",
              ha="left", fontsize=10.5, color="#334155", style="italic")

    # Confidence legend
    handles = [plt.Line2D([0], [0], marker="o", color=TIER_COLOR[t], linestyle="",
                              markersize=10, label=t)
                for t in ("High", "Medium", "Low")]
    ax.legend(handles=handles, loc="lower right", title="Evidence confidence",
                title_fontsize=10.5, fontsize=10, frameon=False)

    fig.text(0.02, 0.02,
              "Effect direction: 'Higher in OWD' = elevated grounded evidence score in the "
              "obesity-associated cohort; 'Higher in NWD' = elevated in the non-obesity-associated cohort. "
              "Analytes are latent themes; interpretation is class-level.",
              ha="left", fontsize=9.5, color="#475569", style="italic")

    plt.tight_layout(rect=(0, 0.03, 1, 0.90))
    _save(fig, "figure_3_differential_grounded_analyte_evidence")
    return df


# ─────────────────────────────────────────────────────────────────────
#  FIGURE 4 — differential biochemical states forest (renamed)
# ─────────────────────────────────────────────────────────────────────
def figure_4():
    rng = np.random.default_rng(42)
    rows = []
    for a in gcfg.BSV_AXES:
        owd = bsv.loc[bsv["group_2"] == "OWD", a].to_numpy()
        nwd = bsv.loc[bsv["group_2"] == "NWD", a].to_numpy()
        if len(owd) < 3 or len(nwd) < 3:
            continue

        def _cd(x, y):
            sa, sb = np.var(x, ddof=1), np.var(y, ddof=1)
            pooled = np.sqrt(((len(x) - 1) * sa + (len(y) - 1) * sb)
                                / max(1, len(x) + len(y) - 2))
            return (x.mean() - y.mean()) / max(pooled, 1e-12)

        d_med = _cd(owd, nwd)
        ds = np.empty(2000)
        for i in range(2000):
            ds[i] = _cd(rng.choice(owd, size=len(owd), replace=True),
                          rng.choice(nwd, size=len(nwd), replace=True))
        d_lo, d_hi = np.nanpercentile(ds, [2.5, 97.5])
        rows.append({
            "axis":  a,
            "label": PUB_AXIS_ONE_LINE[a],
            "class": AXIS_CLASS[a],
            "d_med": d_med, "d_lo": d_lo, "d_hi": d_hi,
        })
    df = pd.DataFrame(rows).merge(stats_2[["axis", "q_value_fdr_bh"]], on="axis")
    df = df.sort_values("d_med")

    fig, ax = plt.subplots(figsize=(10.2, 7.6))
    y = np.arange(len(df))
    for i, (_, r) in enumerate(df.iterrows()):
        col = CLASS_COLOR[r["class"]]
        ax.plot([r["d_lo"], r["d_hi"]], [i, i], color=col, linewidth=2.6,
                 alpha=0.9, solid_capstyle="round")
        ax.scatter([r["d_med"]], [i], color=col, s=100, edgecolor="white",
                     linewidth=1.0, zorder=5)
    ax.axvline(0, color="#374151", linewidth=1.0, alpha=0.85)
    ax.axvspan(-1.0, 1.0, color="#E2E8F0", alpha=0.25, zorder=0)

    labels = []
    for _, r in df.iterrows():
        star = "**" if r["q_value_fdr_bh"] < 0.001 else ("*" if r["q_value_fdr_bh"] < 0.05 else "")
        labels.append(f"{r['label']} {star}")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)

    ax.set_xlabel("Standardized effect size (Cohen's d), OWD − NWD",
                    fontsize=12, weight="bold")
    ax.set_xlim(-3.3, 3.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#94A3B8")
    ax.spines["bottom"].set_color("#94A3B8")
    ax.tick_params(axis="both", length=3, color="#94A3B8")

    fig.text(0.02, 0.965,
              "Figure 4  ·  Differential biochemical states between "
              "obesity-associated and non-obesity-associated diabetes",
              ha="left", fontsize=13.5, weight="bold", color="#0F172A")
    fig.text(0.02, 0.930,
              "Standardized effect size (Cohen's d), with 95% bootstrap CIs.",
              ha="left", fontsize=10.5, color="#334155", style="italic")

    # Legend
    handles = [mpatches.Patch(facecolor=CLASS_COLOR[c], label=c)
                for c in CLASS_COLOR]
    ax.legend(handles=handles, loc="lower right", title="Associated biochemical class",
                title_fontsize=10.5, fontsize=9.5, frameon=False)

    plt.tight_layout(rect=(0, 0.02, 1, 0.90))
    _save(fig, "figure_4_differential_biochemical_states")


# ─────────────────────────────────────────────────────────────────────
#  SUPPLEMENTARY TABLE S1 + FIGURE CAPTIONS
# ─────────────────────────────────────────────────────────────────────
S1_ROWS = [
    {
        "Biochemical state":           "Purine-associated (Nucleic acids)",
        "Representative grounded analytes": "Adenine, adenine-rich nucleic acid contributions",
        "Dominant Raman bands (cm⁻¹)": "720–735 (ring breathing), 1335, 1485",
        "Associated biochemical interpretation":
            "Spectral evidence consistent with purine-nucleobase contributions to nucleic-acid backbone signal.",
        "Confidence notes":
            "Class-level only; substrate rule dampens Ag-SERS purine amplification ×0.65 to prevent molecule-level overclaim.",
    },
    {
        "Biochemical state":           "Purine-associated (Metabolism)",
        "Representative grounded analytes": "Uric acid, hypoxanthine, xanthine",
        "Dominant Raman bands (cm⁻¹)": "640 + 891 (uric-acid doublet), 720 (shared ring)",
        "Associated biochemical interpretation":
            "Spectral evidence consistent with oxidised purine catabolites; overlaps purine-nucleic-acid signal at 720 cm⁻¹.",
        "Confidence notes":
            "Class-level; molecule-level call requires the 640 + 891 doublet co-fire.",
    },
    {
        "Biochemical state":           "Pyrimidine-associated",
        "Representative grounded analytes": "Cytosine-, thymine-, uracil-rich contributions",
        "Dominant Raman bands (cm⁻¹)": "780 (ring breathing)",
        "Associated biochemical interpretation":
            "Spectral evidence consistent with pyrimidine ring-associated nucleic-acid contributions.",
        "Confidence notes":
            "Sensitive to Ag colloid background; class-level interpretation only.",
    },
    {
        "Biochemical state":           "Nucleic acid backbone-associated",
        "Representative grounded analytes": "Phosphate-backbone-associated features",
        "Dominant Raman bands (cm⁻¹)": "1080 (PO₂⁻ symmetric stretch)",
        "Associated biochemical interpretation":
            "Spectral evidence consistent with nucleic-acid phosphate backbone; overlaps glycan C–O in the same region.",
        "Confidence notes":
            "Class-level; co-band with nucleobase support required for stronger inference.",
    },
    {
        "Biochemical state":           "Glycan-associated",
        "Representative grounded analytes": "Glucose, other monosaccharide contributions",
        "Dominant Raman bands (cm⁻¹)": "517, 845, 1125 (C–O / C–C / anomeric)",
        "Associated biochemical interpretation":
            "Spectral evidence consistent with glycan / carbohydrate-related contributions to the EV corona.",
        "Confidence notes":
            "Overlap with lactate 845 cm⁻¹ requires co-band at 1125 for molecule-level specificity.",
    },
    {
        "Biochemical state":           "Protein-associated",
        "Representative grounded analytes": "Albumin- and peptide-backbone-associated features",
        "Dominant Raman bands (cm⁻¹)": "1230–1300 (amide III), 1655 (amide I)",
        "Associated biochemical interpretation":
            "Spectral evidence consistent with protein/peptide backbone contributions to the plasma-EV corona.",
        "Confidence notes":
            "Amide-I overlaps with lipid C=C; substrate physics rule soft-dampens in protein-dominant matrices.",
    },
    {
        "Biochemical state":           "Aromatic biomolecule-associated",
        "Representative grounded analytes": "Tyrosine-, phenylalanine-, tryptophan-rich features",
        "Dominant Raman bands (cm⁻¹)": "1003 (Phe ring), 830/850 (Tyr doublet)",
        "Associated biochemical interpretation":
            "Spectral evidence consistent with aromatic amino-acid contributions.",
        "Confidence notes":
            "Phe 1003 is a strong, isolated marker; Tyr doublet ratio is environment-sensitive.",
    },
    {
        "Biochemical state":           "Membrane lipid-associated",
        "Representative grounded analytes": "Oleic-acid-like unsaturated acyl chains, membrane-lipid contributions",
        "Dominant Raman bands (cm⁻¹)": "1265, 1440 (CH₂), 1655 (C=C), 1745 (ester carbonyl)",
        "Associated biochemical interpretation":
            "Spectral evidence consistent with acyl / membrane-lipid contributions.",
        "Confidence notes":
            "1655 overlaps amide-I; class-level unless multiple lipid-anchor bands co-fire.",
    },
    {
        "Biochemical state":           "Sterol-associated",
        "Representative grounded analytes": "Cholesterol, cholesteryl-ester-associated features",
        "Dominant Raman bands (cm⁻¹)": "548 (sterol ring), 700, 1130, 1665",
        "Associated biochemical interpretation":
            "Spectral evidence consistent with sterol / neutral-lipid contributions, distinct from acyl lipids.",
        "Confidence notes":
            "The 548 cm⁻¹ ring deformation is required for sterol-specific interpretation.",
    },
    {
        "Biochemical state":           "Redox-associated",
        "Representative grounded analytes": "Ergothioneine, glutathione, other thiol/thione contributions",
        "Dominant Raman bands (cm⁻¹)": "490–505 (C–S / thione), 720 (imidazole support)",
        "Associated biochemical interpretation":
            "Spectral evidence consistent with sulfur/thiol/redox-active biochemical contributions.",
        "Confidence notes":
            "Motif window tightened to 490–505 cm⁻¹ and the Ag-SERS thiol boost gated by required 720 cm⁻¹ imidazole co-band; anchor at 495 alone is not sufficient.",
    },
    {
        "Biochemical state":           "Small molecule-associated",
        "Representative grounded analytes": "Lactate, acetate, pyruvate-like small-molecule contributions",
        "Dominant Raman bands (cm⁻¹)": "845 + 925 (C–C–O), 1043, 1090, 1455",
        "Associated biochemical interpretation":
            "Spectral evidence consistent with small-molecule metabolite contributions.",
        "Confidence notes":
            "Overlap with glycan anomeric C–H requires 845 + 925 co-fire for lactate-specific interpretation.",
    },
]


def write_supp_table_and_captions(fig3_df: pd.DataFrame):
    # Supplementary Table S1 — CSV + Markdown
    s1 = pd.DataFrame(S1_ROWS)
    s1.to_csv(OUT / "supplementary_table_s1.csv", index=False)

    md = ["# Supplementary Table S1. Interpretation of GAIRA biochemical states",
            "",
            "| " + " | ".join(s1.columns) + " |",
            "| " + " | ".join(["---"] * len(s1.columns)) + " |"]
    for _, r in s1.iterrows():
        md.append("| " + " | ".join(str(r[c]).replace("|", "/") for c in s1.columns) + " |")
    md.append("")
    md.append("*Each biochemical state is a latent theme inferred from Raman/SERS spectral "
                "evidence, not a direct measurement of a single molecule. Representative "
                "grounded analytes are curated Raman-band anchor sets that the GAIRA motif "
                "library uses to score each biochemical state; they are illustrative of the "
                "biochemical class and should not be interpreted as identity claims.*")
    (OUT / "supplementary_table_s1.md").write_text("\n".join(md) + "\n")

    # Figure captions
    caps = f"""# Figure captions — GAIRA diabetes manuscript

## Figure 1. GAIRA-derived biochemical state distinguishes obesity-associated (OWD) and non-obesity-associated (NWD) diabetes

Radar plot showing normalized biochemical states inferred from Raman/SERS spectra
using the GAIRA grounded inference framework. Each axis represents a latent
biochemical state associated with a broad biochemical class rather than a single
molecular species. Values are expressed as deviations from the pooled cohort
biochemical reference state, allowing comparison across biochemical states.
Positive values indicate relative enrichment and negative values indicate
relative depletion. Axes marked ** remained significant after Benjamini–Hochberg
correction (q < 0.05). Cohort: OWD (Impact) n = {(bsv["group_2"] == "OWD").sum()};
NWD (Strong-D) n = {(bsv["group_2"] == "NWD").sum()}. Statistics: per-patient
Mann–Whitney with Benjamini–Hochberg FDR correction.

## Figure 2. GAIRA biochemical states across diabetes subgroups

Radar plot showing normalized biochemical states for the four diabetes
subgroups relative to the pooled cohort biochemical reference state. Each axis
summarizes spectral evidence associated with a broad biochemical class.
Significant axes were identified using Kruskal–Wallis testing with
Benjamini–Hochberg correction. The figure highlights subgroup-specific
biochemical phenotypes beyond the binary obesity-associated versus
non-obesity-associated comparison. Subgroup sizes: {bsv[bsv["subgroup_4"].notna()]["subgroup_4"].value_counts().to_dict()}.

## Figure 3. Differential grounded analyte evidence between obesity-associated and non-obesity-associated diabetes

Forest plot summarizing the analytes contributing most strongly to the observed
biochemical differences between groups. Effect sizes represent standardized
differences in grounded evidence scores, with 95% bootstrap confidence
intervals. Confidence categories integrate spectral similarity, grounding
evidence, and domain-aware weighting. The analytes shown should be interpreted
as the strongest supported contributors to the observed biochemical states
rather than definitive molecular identifications. Analytes are latent themes;
interpretation is class-level. ** denotes q < 0.001; * denotes q < 0.05
(Mann–Whitney with Benjamini–Hochberg correction). Bootstrap: 2000 resamples,
percentile CI.

## Figure 4. Differential biochemical states between obesity-associated and non-obesity-associated diabetes

Forest plot showing standardized effect sizes (Cohen's d) for each GAIRA
biochemical state. Positive values indicate greater biochemical state activity
in obesity-associated diabetes, whereas negative values indicate greater
activity in non-obesity-associated diabetes. Error bars represent 95%
bootstrap confidence intervals (2000 resamples, percentile CI). Colors
indicate the associated biochemical class. Biochemical states are interpreted
as latent spectral themes associated with broad biochemical classes rather
than direct measurements of individual molecules. ** denotes q < 0.001; *
denotes q < 0.05 after Benjamini–Hochberg correction.

## Supplementary Table S1. Interpretation of GAIRA biochemical states

For each of the eleven GAIRA biochemical states, the table lists the
representative grounded analytes used in scoring, the dominant Raman bands
that anchor each state, the biochemical interpretation associated with the
state, and confidence caveats. Interpretations are class-level; the
"representative grounded analytes" are illustrative anchor sets and are not
identity claims.
"""
    (OUT / "figure_captions.md").write_text(caps.strip() + "\n")


# ─────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────
def main():
    print("[pub-figs] Figure 1 — 2-group biochemical state radar")
    figure_1()
    print("[pub-figs] Figure 2 — 4-subgroup biochemical state radar")
    figure_2()
    mss_pp = compute_per_patient_mss()
    print("[pub-figs] Figure 3 — differential grounded analyte evidence")
    fig3_df = figure_3(mss_pp)
    print("[pub-figs] Figure 4 — differential biochemical states (renamed)")
    figure_4()
    print("[pub-figs] Supplementary Table S1 + captions")
    write_supp_table_and_captions(fig3_df)
    print(f"\n[pub-figs] DONE → {OUT}")


if __name__ == "__main__":
    main()
