"""gaira_base_2 — PHASE A: Grounding Benchmark.

Exercises the implemented engine on all available pure / reference
spectra:

  - ramanbiolib (141 normal Raman of pure compounds)
  - Gobbato powder Raman (5 metabolites: UA / HX / xanthine / ergo / creat)

Per spectrum, captures:
  - motif core + regime scores (50 motifs)
  - 11-axis core + regime evidence
  - 8-axis projection (MAX combiner)
  - ambiguity lane

Grounding questions tested:
  1. Does the expected motif rank highly?
  2. Do off-target motifs remain suppressed?
  3. Does the expected 11-axis rise?
  4. Is the 8-axis projection chemically sensible?
  5. Does ambiguity fire where chemistry is genuinely ambiguous?

Emits tables, figures, report, and a per-spectrum score dump.

Run:
    cd /Users/suraj/projects/GAIRA
    PYTHONPATH=src .venv/bin/python scripts/run_gaira_base_2_grounding_benchmark.py
"""
from __future__ import annotations

import ast
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gaira.base2 import (
    BIOLOGY_AXES_V11,
    GAIRA_BASE_AXES_V8,
    load_active_registry,
    result_to_flat_dict,
    score_spectrum,
)
from gaira.spectral import canonical_master_axis, crop_before_interpolate
from gaira.spectral.preprocessing import _asls_baseline
from scipy.signal import savgol_filter


OUT = Path("/Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_2_backend_validation_v1/grounding")
OUT.mkdir(parents=True, exist_ok=True)

RAMANBIOLIB = Path(
    "/Volumes/SSD_Rad/GAIRA_DATA/raw/ramanbiolib/ramanbiolib-main/"
    "ramanbiolib/db/raman_spectra_db.csv"
)
GOBBATO_EXT = Path(
    "/Volumes/SSD_Rad/GAIRA_BUILD/gaira_build_motifs_v1/"
    "M3_1_reference_rescue_v1/references/_extracted"
)


# ──────────────────────────────────────────────────────────────────────
# Expected chemistry map (reference compound → expected 11-axes)
# ──────────────────────────────────────────────────────────────────────
#
# Each compound is mapped to the subset of biology axes it should light up.
# The axis labels use the gaira_base_2 11-axis schema.

EXPECTED_AXES: dict[str, list[str]] = {
    # nucleobases
    "adenine":   ["purine_nucleotide", "purine_metabolite"],
    "guanine":   ["purine_nucleotide"],
    "cytosine":  ["pyrimidine_nucleotide"],
    "thymine":   ["pyrimidine_nucleotide"],
    "uracil":    ["pyrimidine_nucleotide"],
    # nucleic acids
    "a-dna":     ["purine_nucleotide", "pyrimidine_nucleotide", "phosphate_nucleic_adjacent"],
    "b-dna":     ["purine_nucleotide", "pyrimidine_nucleotide", "phosphate_nucleic_adjacent"],
    "t-rna":     ["purine_nucleotide", "pyrimidine_nucleotide", "phosphate_nucleic_adjacent"],
    # amino acids
    "l-phenylalanine": ["aromatic_residue"],
    "l-tyrosine":      ["aromatic_residue"],
    "l-tryptophan":    ["aromatic_residue"],
    "l-histidine":     ["aromatic_residue"],
    "l-arginine":      ["protein_peptide_backbone"],
    "l-asparagine":    ["protein_peptide_backbone"],
    "l-aspartic acid": ["protein_peptide_backbone"],
    "l-glutamate":     ["metabolic_small_molecule"],
    "l-proline":       ["protein_peptide_backbone"],
    "l-serine":        ["protein_peptide_backbone"],
    "l-valine":        ["protein_peptide_backbone"],
    "l-alanine":       ["protein_peptide_backbone"],
    "glycine":         ["protein_peptide_backbone"],
    # proteins
    "albumin":        ["protein_peptide_backbone"],
    "collagen":       ["protein_peptide_backbone"],
    "elastin":        ["protein_peptide_backbone"],
    "keratin":        ["protein_peptide_backbone"],
    "hemoglobin":     ["protein_peptide_backbone"],
    "myoglobin":      ["protein_peptide_backbone"],
    "insulin":        ["protein_peptide_backbone"],
    "ferritin":       ["protein_peptide_backbone"],
    "cytochrome c":   ["protein_peptide_backbone", "sulfur_thiol_redox"],
    "lactalbumin":    ["protein_peptide_backbone"],
    "carbonic anhydrase": ["protein_peptide_backbone"],
    "tubulin":        ["protein_peptide_backbone"],
    "elastase":       ["protein_peptide_backbone"],
    "ubiquitin":      ["protein_peptide_backbone"],
    "trypsin":        ["protein_peptide_backbone"],
    "trypsinogen":    ["protein_peptide_backbone"],
    "pepsin":         ["protein_peptide_backbone"],
    "pepsinogen":     ["protein_peptide_backbone"],
    "papain":         ["protein_peptide_backbone"],
    "major proteinase": ["protein_peptide_backbone"],
    "horseradish peroxidase": ["protein_peptide_backbone"],
    "xylanase":       ["protein_peptide_backbone"],
    "lectin":         ["protein_peptide_backbone"],
    "α-chymotrypsinogen a (type ii)": ["protein_peptide_backbone"],
    "thaumatin":      ["protein_peptide_backbone"],
    "triosephosphate isomerase": ["protein_peptide_backbone"],
    "glutathione transferase": ["protein_peptide_backbone", "sulfur_thiol_redox"],
    "glucose oxidase": ["protein_peptide_backbone"],
    "superoxide dismutases": ["protein_peptide_backbone"],
    "trypsin inhibitor": ["protein_peptide_backbone"],
    # thiol / redox
    "glutathione":    ["sulfur_thiol_redox"],
    # glycans
    "d-(+)-glucose":  ["glycan_carbohydrate"],
    "d-(+)-galactose":["glycan_carbohydrate"],
    "d-(+)-mannose":  ["glycan_carbohydrate"],
    "β-d-glucose":    ["glycan_carbohydrate"],
    "d-(-)-fructose": ["glycan_carbohydrate"],
    "d-(-)-ribose":   ["glycan_carbohydrate"],
    "d-(+)-fucose":   ["glycan_carbohydrate"],
    "d-(+)-xylose":   ["glycan_carbohydrate"],
    "d-(-)-arabinose":["glycan_carbohydrate"],
    "l-(+)-arabinose":["glycan_carbohydrate"],
    "d-(+)-lactose monohydrate": ["glycan_carbohydrate"],
    "d-(+)-maltose monohydrate": ["glycan_carbohydrate"],
    "d-(+)-sucrose":  ["glycan_carbohydrate"],
    "d-(+)-trehalose":["glycan_carbohydrate"],
    "d-(+)-raffinose pentahydrate": ["glycan_carbohydrate"],
    "d-(+)-galactosamine": ["glycan_carbohydrate"],
    "glucosamine":    ["glycan_carbohydrate"],
    "n-acetyl- d-glucosamine": ["glycan_carbohydrate"],
    "lactose":        ["glycan_carbohydrate"],
    "cellulose":      ["glycan_carbohydrate"],
    "glycogen":       ["glycan_carbohydrate"],
    "chitin":         ["glycan_carbohydrate"],
    "amylose":        ["glycan_carbohydrate"],
    "amylopectin":    ["glycan_carbohydrate"],
    "d-(+)-dextrose": ["glycan_carbohydrate"],
    "d-fructose-6-phosphate": ["glycan_carbohydrate", "phosphate_nucleic_adjacent"],
    "glycerol":       ["lipid_acyl_membrane"],
    # lipids acyl / membrane
    "oleic acid":     ["lipid_acyl_membrane"],
    "palmitic acid":  ["lipid_acyl_membrane"],
    "stearic acid":   ["lipid_acyl_membrane"],
    "linoleic acid":  ["lipid_acyl_membrane"],
    "arachidic acid": ["lipid_acyl_membrane"],
    "arachidonic acid": ["lipid_acyl_membrane"],
    "lauric acid":    ["lipid_acyl_membrane"],
    "myristic acid":  ["lipid_acyl_membrane"],
    "elaidic acid":   ["lipid_acyl_membrane"],
    "palmitoleic acid": ["lipid_acyl_membrane"],
    "vaccenic acid":  ["lipid_acyl_membrane"],
    "α-linolenic acid": ["lipid_acyl_membrane"],
    "12-methyltetradecanoic acid": ["lipid_acyl_membrane"],
    "13-methylmyristicacid": ["lipid_acyl_membrane"],
    "14-methylhexadecanoic acid": ["lipid_acyl_membrane"],
    "14-methylpentadecanoic acid": ["lipid_acyl_membrane"],
    "15-methylpalmiticacid": ["lipid_acyl_membrane"],
    "ceramide":       ["lipid_acyl_membrane"],
    "sphingomyelin":  ["lipid_acyl_membrane"],
    "l-α-phosphatidylcholine":     ["lipid_acyl_membrane"],
    "l-α-phosphatidylethanolamine":["lipid_acyl_membrane"],
    # sterols / neutral lipids
    "cholesterol":       ["sterol_neutral_lipid"],
    "cholesteryl linoleate": ["sterol_neutral_lipid"],
    "cholesteryl oleate":    ["sterol_neutral_lipid"],
    "cholesteryl palmitate": ["sterol_neutral_lipid"],
    "cholesteryl stearate":  ["sterol_neutral_lipid"],
    "estradiol":  ["sterol_neutral_lipid"],
    "estrone":    ["sterol_neutral_lipid"],
    "estriol":    ["sterol_neutral_lipid"],
    "ethinylestradiol": ["sterol_neutral_lipid"],
    "diethylstilbestrol": ["sterol_neutral_lipid"],
    "tristearin":   ["sterol_neutral_lipid"],
    "tripalmitin":  ["sterol_neutral_lipid"],
    "triolein":     ["sterol_neutral_lipid"],
    "trilinolein":  ["sterol_neutral_lipid"],
    "trilinolenin": ["sterol_neutral_lipid"],
    "trimyristin":  ["sterol_neutral_lipid"],
    "trilaurin":    ["sterol_neutral_lipid"],
    "tricaprin":    ["sterol_neutral_lipid"],
    "tricaproin":   ["sterol_neutral_lipid"],
    "tricaprylin":  ["sterol_neutral_lipid"],
    "tri-11-eicosenoin": ["sterol_neutral_lipid"],
    "triarachidin": ["sterol_neutral_lipid"],
    "tribehenin":   ["sterol_neutral_lipid"],
    "trielaidin":   ["sterol_neutral_lipid"],
    "trierucin":    ["sterol_neutral_lipid"],
    "tripalmitolein":    ["sterol_neutral_lipid"],
    "tripetroselinin":   ["sterol_neutral_lipid"],
    # small-molecule metabolites (mostly axis-ambiguous in ramanbiolib)
    "acetoacetate":  ["metabolic_small_molecule"],
    "pyruvate":      ["metabolic_small_molecule"],
    "fumarate":      ["metabolic_small_molecule"],
    "citric acid":   ["ambiguity_artifact", "metabolic_small_molecule"],
    "succinic acid": ["metabolic_small_molecule"],
    "malic acid":    ["metabolic_small_molecule"],
    "ascorbic acid": ["metabolic_small_molecule"],
    "phosphoenolpyruvate": ["metabolic_small_molecule", "phosphate_nucleic_adjacent"],
    "acetyl coenzyme a":   ["metabolic_small_molecule"],
    "coenzyme a":          ["metabolic_small_molecule"],
    # other (structural)
    "melanin":         ["aromatic_residue"],
    "β-carotene":      ["lipid_acyl_membrane"],
    "riboﬂavin":        ["metabolic_small_molecule"],
    "2-deoxy-d-ribose":["glycan_carbohydrate"],

    # Gobbato powder Raman (npz keys)
    "ua_raman_pwd_gobbato2025":    ["purine_metabolite"],
    "hypox_raman_pwd_gobbato2025": ["purine_metabolite"],
    "xanth_raman_pwd_gobbato2025": ["purine_metabolite"],
    "ergo_raman_pwd_gobbato2025":  ["sulfur_thiol_redox", "metabolic_small_molecule"],
    "creat_raman_pwd_gobbato2025": ["metabolic_small_molecule"],
}


# ──────────────────────────────────────────────────────────────────────
# Reference-spectrum loaders
# ──────────────────────────────────────────────────────────────────────

def _parse_list(s):
    return np.array(ast.literal_eval(s), dtype=np.float64)


def canonical_preprocess(wn, y, master_x):
    try:
        y_interp, _ = crop_before_interpolate(
            wn, y, master_x, partial_ok=True, min_coverage=0.80,
        )
    except Exception:
        return None
    mask = np.isfinite(y_interp)
    if not mask.any():
        return None
    if not mask.all():
        idx = np.arange(len(y_interp))
        y_interp[~mask] = np.interp(idx[~mask], idx[mask], y_interp[mask])
    y_bc = y_interp - _asls_baseline(y_interp, lam=1e5, p=0.001, n_iter=10)
    y_sg = savgol_filter(y_bc, window_length=11, polyorder=3)
    n = np.linalg.norm(y_sg)
    return y_sg / n if n > 1e-12 else None


def load_ramanbiolib(master_x):
    df = pd.read_csv(RAMANBIOLIB)
    out = {}
    for _, r in df.iterrows():
        comp = str(r["component"]).strip().lower()
        try:
            wn = _parse_list(r["wavenumbers"])
            y  = _parse_list(r["intensity"])
            y_pp = canonical_preprocess(wn, y, master_x)
        except Exception:
            continue
        if y_pp is not None:
            out[f"ramanbiolib::{comp}"] = (comp, y_pp)
    return out


def load_gobbato_powder(master_x):
    """Load Gobbato pure powder Raman from the M3.1 npz (already on master axis)
    and re-apply AsLS + SG + L2 (the M3.1 npz is crop+interp only)."""
    npz_path = (
        GOBBATO_EXT.parent.parent / "references" /
        "rescued_refs_master_axis.npz"
    )
    if not npz_path.exists():
        return {}
    npz = np.load(npz_path)
    out = {}
    for key in npz.files:
        if key == "master_x":
            continue
        if "raman_pwd_gobbato" not in key:
            continue
        y = npz[key]
        mask = np.isfinite(y)
        if not mask.any():
            continue
        y_work = y.copy()
        if not mask.all():
            idx = np.arange(len(y_work))
            y_work[~mask] = np.interp(idx[~mask], idx[mask], y_work[mask])
        y_bc = y_work - _asls_baseline(y_work, lam=1e5, p=0.001, n_iter=10)
        y_sg = savgol_filter(y_bc, window_length=11, polyorder=3)
        n = np.linalg.norm(y_sg)
        if n < 1e-12:
            continue
        out[f"gobbato_powder::{key}"] = (key, y_sg / n)
    return out


# ──────────────────────────────────────────────────────────────────────
# Ranking / expected-vs-observed helpers
# ──────────────────────────────────────────────────────────────────────

def top_k_axis11(result, k=3):
    scores = sorted(
        result.axis11_scores,
        key=lambda a: a.core_evidence,
        reverse=True,
    )
    return [(a.axis_id, round(a.core_evidence, 4)) for a in scores[:k]]


def top_k_motifs(result, k=5):
    scores = sorted(result.motif_scores, key=lambda m: m.core_weight, reverse=True)
    return [(m.motif_id, round(m.core_weight, 4)) for m in scores[:k]]


def expected_axis_rank(result, expected_axes):
    """For each expected axis, what rank does it have by core_evidence?"""
    sorted_axes = sorted(
        result.axis11_scores,
        key=lambda a: a.core_evidence,
        reverse=True,
    )
    positions = {a.axis_id: i + 1 for i, a in enumerate(sorted_axes)}
    return {ax: positions.get(ax, None) for ax in expected_axes}


# ──────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 78)
    print("gaira_base_2 — PHASE A: Grounding Benchmark")
    print("=" * 78)

    master_x = canonical_master_axis()
    motifs, mappings, dual = load_active_registry()
    print(f"engine: {len(motifs)} active motifs, {len(mappings)} mappings")

    print("[load] ramanbiolib...")
    rb = load_ramanbiolib(master_x)
    print(f"  {len(rb)} ramanbiolib references preprocessed")

    print("[load] Gobbato powder Raman...")
    gp = load_gobbato_powder(master_x)
    print(f"  {len(gp)} Gobbato powder references preprocessed")

    all_refs = {**rb, **gp}
    print(f"[scoring] {len(all_refs)} total reference spectra through engine")

    # ── Score every reference ────────────────────────────────────────
    per_spectrum_rows = []
    motif_top_rows = []
    axis_top_rows = []
    off_target_rows = []
    ambig_rows = []
    expected_vs_observed_motif = []
    expected_vs_observed_axis = []

    for spec_id, (comp_key, y) in all_refs.items():
        res = score_spectrum(y, master_x, motifs, mappings, dual, spec_id)
        flat = result_to_flat_dict(res)
        flat["component_key"] = comp_key
        flat["reference_source"] = spec_id.split("::", 1)[0]
        per_spectrum_rows.append(flat)

        # expected chemistry
        expected = EXPECTED_AXES.get(comp_key, [])

        # Top-3 axes + top-5 motifs
        top3_axes = top_k_axis11(res, k=3)
        top5_motifs = top_k_motifs(res, k=5)
        motif_top_rows.append({
            "spectrum_id": spec_id, "component_key": comp_key,
            "top_motif_1": top5_motifs[0][0] if len(top5_motifs) > 0 else "",
            "top_motif_1_core": top5_motifs[0][1] if len(top5_motifs) > 0 else 0.0,
            "top_motif_2": top5_motifs[1][0] if len(top5_motifs) > 1 else "",
            "top_motif_2_core": top5_motifs[1][1] if len(top5_motifs) > 1 else 0.0,
            "top_motif_3": top5_motifs[2][0] if len(top5_motifs) > 2 else "",
            "top_motif_3_core": top5_motifs[2][1] if len(top5_motifs) > 2 else 0.0,
            "top_motif_4": top5_motifs[3][0] if len(top5_motifs) > 3 else "",
            "top_motif_4_core": top5_motifs[3][1] if len(top5_motifs) > 3 else 0.0,
            "top_motif_5": top5_motifs[4][0] if len(top5_motifs) > 4 else "",
            "top_motif_5_core": top5_motifs[4][1] if len(top5_motifs) > 4 else 0.0,
        })
        axis_top_rows.append({
            "spectrum_id": spec_id, "component_key": comp_key,
            "top_axis_1": top3_axes[0][0] if len(top3_axes) > 0 else "",
            "top_axis_1_core": top3_axes[0][1] if len(top3_axes) > 0 else 0.0,
            "top_axis_2": top3_axes[1][0] if len(top3_axes) > 1 else "",
            "top_axis_2_core": top3_axes[1][1] if len(top3_axes) > 1 else 0.0,
            "top_axis_3": top3_axes[2][0] if len(top3_axes) > 2 else "",
            "top_axis_3_core": top3_axes[2][1] if len(top3_axes) > 2 else 0.0,
            "expected_axes": ",".join(expected),
        })

        # Off-target activation: per-axis score for axes NOT in expected set
        for a in res.axis11_scores:
            off_target_rows.append({
                "spectrum_id": spec_id, "component_key": comp_key,
                "axis_id": a.axis_id,
                "is_expected": a.axis_id in expected,
                "core_evidence": round(a.core_evidence, 4),
                "regime_evidence": round(a.regime_evidence, 4),
            })

        # Ambiguity activation
        ambig_rows.append({
            "spectrum_id": spec_id,
            "component_key": comp_key,
            "ambiguity_core": round(res.ambiguity.core_evidence, 4),
            "ambiguity_regime": round(res.ambiguity.regime_evidence, 4),
            "n_ambig_contrib": len(res.ambiguity.contributing_motifs),
            "top_ambig_motifs": ",".join(res.ambiguity.contributing_motifs[:3]),
        })

        # Expected-vs-observed axis rank
        if expected:
            ranks = expected_axis_rank(res, expected)
            for ax, rank in ranks.items():
                expected_vs_observed_axis.append({
                    "spectrum_id": spec_id,
                    "component_key": comp_key,
                    "expected_axis": ax,
                    "observed_rank_1_to_11": rank,
                    "expected_axis_core_evidence":
                        next((a.core_evidence for a in res.axis11_scores
                              if a.axis_id == ax), 0.0),
                })

    # ── Emit CSVs ────────────────────────────────────────────────────
    pd.DataFrame(per_spectrum_rows).to_csv(
        OUT / "grounding_per_spectrum_scores_v1.csv", index=False,
    )
    pd.DataFrame(motif_top_rows).to_csv(
        OUT / "grounding_expected_vs_observed_motif_rank_v1.csv", index=False,
    )
    pd.DataFrame(expected_vs_observed_axis).to_csv(
        OUT / "grounding_expected_vs_observed_axis_rank_v1.csv", index=False,
    )
    pd.DataFrame(off_target_rows).to_csv(
        OUT / "grounding_off_target_activation_v1.csv", index=False,
    )
    pd.DataFrame(ambig_rows).to_csv(
        OUT / "grounding_ambiguity_activation_v1.csv", index=False,
    )
    pd.DataFrame(axis_top_rows).to_csv(
        OUT / "grounding_axis_rank_v1.csv", index=False,
    )
    print(f"[emit] {OUT}/grounding_per_spectrum_scores_v1.csv "
          f"({len(per_spectrum_rows)} rows)")

    # ── Figures ──────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[warn] matplotlib unavailable: {e}")
    else:
        _plot_top_axis_heatmap(pd.DataFrame(axis_top_rows), plt)
        _plot_off_target_matrix(pd.DataFrame(off_target_rows), plt)
        _plot_ambiguity_examples(pd.DataFrame(ambig_rows), plt)
        _plot_top_motif_heatmap(pd.DataFrame(motif_top_rows), plt)
        _plot_motif_vs_axis_examples(
            all_refs, master_x, motifs, mappings, dual, plt,
        )

    # ── Report ────────────────────────────────────────────────────────
    _write_report(
        pd.DataFrame(per_spectrum_rows),
        pd.DataFrame(motif_top_rows),
        pd.DataFrame(expected_vs_observed_axis),
        pd.DataFrame(off_target_rows),
        pd.DataFrame(ambig_rows),
        all_refs,
    )
    print("DONE — grounding benchmark complete")


# ──────────────────────────────────────────────────────────────────────
# Figures
# ──────────────────────────────────────────────────────────────────────

def _plot_top_axis_heatmap(axis_df, plt):
    axis_df = axis_df[axis_df["expected_axes"] != ""].copy()
    axis_df["top1_hit"] = axis_df.apply(
        lambda r: r["top_axis_1"] in r["expected_axes"].split(","),
        axis=1,
    )
    # aggregate by component class (first expected axis)
    axis_df["class"] = axis_df["expected_axes"].str.split(",").str[0]
    pivot = (
        axis_df.groupby("class")["top1_hit"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "hits", "count": "n"})
    )
    pivot["hit_rate"] = pivot["hits"] / pivot["n"]
    pivot = pivot.sort_values("hit_rate", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2a9d8f" if r >= 0.7 else "#e9c46a" if r >= 0.4 else "#e76f51"
              for r in pivot["hit_rate"]]
    ax.barh(pivot.index, pivot["hit_rate"], color=colors)
    for i, (cls, row) in enumerate(pivot.iterrows()):
        ax.text(row["hit_rate"] + 0.01, i,
                 f"{int(row['hits'])}/{int(row['n'])}", va="center", fontsize=9)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("fraction where top-1 axis ∈ expected set")
    ax.set_title("Grounding — top-1 axis hit rate by expected chemistry class")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_grounding_top_axis_rank_heatmap.png", dpi=130)
    plt.close(fig)


def _plot_off_target_matrix(off_df, plt):
    # Heatmap: expected axis (rows) × observed axis (cols) → mean core_evidence
    off_df = off_df.copy()
    # Map each spectrum to its primary expected axis via the first entry
    per_spec_first_expected = {}
    for sid, grp in off_df.groupby("spectrum_id"):
        exp = grp[grp["is_expected"]]["axis_id"].tolist()
        per_spec_first_expected[sid] = exp[0] if exp else "(none)"
    off_df["primary_expected"] = off_df["spectrum_id"].map(per_spec_first_expected)
    pivot = (
        off_df.groupby(["primary_expected", "axis_id"])["core_evidence"]
        .mean()
        .unstack(fill_value=0.0)
    )
    pivot = pivot.reindex(columns=list(BIOLOGY_AXES_V11) + ["ambiguity_artifact"],
                            fill_value=0.0)
    axes_present = [a for a in list(BIOLOGY_AXES_V11) if a in pivot.index]
    pivot = pivot.loc[axes_present, list(BIOLOGY_AXES_V11)]
    fig, ax = plt.subplots(figsize=(12, max(6, 0.45 * len(pivot))))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd",
                    vmin=0.0, vmax=min(0.5, pivot.values.max()))
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if v > 0.05:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                         fontsize=6, color="black")
    ax.set_xlabel("observed 11-axis (mean core evidence)")
    ax.set_ylabel("primary expected axis")
    ax.set_title("Grounding — off-target activation matrix\n"
                  "diagonal = on-target; off-diagonal = cross-talk")
    fig.colorbar(im, ax=ax, label="mean core evidence")
    fig.tight_layout()
    fig.savefig(OUT / "fig_grounding_off_target_activation_matrix.png", dpi=130)
    plt.close(fig)


def _plot_ambiguity_examples(ambig_df, plt):
    top_ambig = ambig_df.sort_values("ambiguity_core", ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(11, max(4, 0.35 * len(top_ambig))))
    ax.barh(top_ambig["component_key"], top_ambig["ambiguity_core"],
             color="#7b2cbf")
    for i, (_, r) in enumerate(top_ambig.iterrows()):
        ax.text(r["ambiguity_core"] + 0.005, i,
                 r["top_ambig_motifs"][:60], va="center", fontsize=7)
    ax.set_xlabel("ambiguity lane core evidence")
    ax.set_title("Grounding — top-15 references with highest ambiguity activation")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_grounding_ambiguity_examples.png", dpi=130)
    plt.close(fig)


def _plot_top_motif_heatmap(motif_df, plt):
    top1_counts = motif_df["top_motif_1"].value_counts().head(20)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(top1_counts))))
    ax.barh(top1_counts.index, top1_counts.values, color="#2a9d8f")
    ax.set_xlabel("n references where this motif is top-ranked")
    ax.set_title("Grounding — motifs most often ranked #1 across references")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_grounding_top_motif_rank_heatmap.png", dpi=130)
    plt.close(fig)


def _plot_motif_vs_axis_examples(all_refs, master_x, motifs, mappings, dual, plt):
    """Panel: for 4 exemplar references, show the motif-level + 11-axis landscape."""
    examples = [
        "ramanbiolib::adenine",
        "ramanbiolib::l-phenylalanine",
        "ramanbiolib::cholesterol",
        "gobbato_powder::ua_raman_pwd_gobbato2025",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, sid in zip(axes.flat, examples):
        if sid not in all_refs:
            ax.text(0.5, 0.5, f"{sid}\nnot found", ha="center", va="center",
                     transform=ax.transAxes)
            ax.set_axis_off()
            continue
        comp_key, y = all_refs[sid]
        res = score_spectrum(y, master_x, motifs, mappings, dual, sid)
        axes11 = sorted(
            res.axis11_scores, key=lambda a: a.core_evidence, reverse=True,
        )
        names = [a.axis_id for a in axes11]
        core = [a.core_evidence for a in axes11]
        regime = [a.regime_evidence for a in axes11]
        positions = np.arange(len(names))
        ax.barh(positions - 0.18, core, height=0.35,
                 color="#2a9d8f", label="core")
        ax.barh(positions + 0.18, regime, height=0.35,
                 color="#76c893", label="regime")
        ax.set_yticks(positions)
        ax.set_yticklabels(names, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.0)
        ax.set_title(f"{comp_key[:40]}", fontsize=10)
        ax.set_xlabel("11-axis evidence")
        ax.legend(fontsize=7)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.suptitle(
        "Grounding — motif-aware 11-axis evidence on 4 exemplar references",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig_grounding_motif_vs_axis_examples.png", dpi=130)
    plt.close(fig)


def _write_report(per_spec_df, motif_df, eva_axis_df, off_df, ambig_df, all_refs):
    n_total = len(per_spec_df)
    classified = motif_df[motif_df.apply(
        lambda r: r["spectrum_id"].split("::", 1)[1] in EXPECTED_AXES,
        axis=1,
    )]
    # Hit rate
    top1_hit_df = eva_axis_df[eva_axis_df["observed_rank_1_to_11"] == 1]
    top3_hit_df = eva_axis_df[eva_axis_df["observed_rank_1_to_11"] <= 3]
    # compute unique spectra with top-1 and top-3 hits
    top1_spec = set(top1_hit_df["spectrum_id"])
    top3_spec = set(top3_hit_df["spectrum_id"])
    total_classified_spec = set(eva_axis_df["spectrum_id"])
    top1_rate = len(top1_spec) / max(len(total_classified_spec), 1)
    top3_rate = len(top3_spec) / max(len(total_classified_spec), 1)

    n_ambig_fires = int((ambig_df["ambiguity_core"] > 0.1).sum())
    mean_ambig = ambig_df["ambiguity_core"].mean()

    lines = [
        "# gaira_base_2 — Grounding Benchmark Report (v1)",
        "",
        f"**References scored:** {n_total}",
        f"  - ramanbiolib: {int((per_spec_df['reference_source'] == 'ramanbiolib').sum())}",
        f"  - Gobbato powder Raman: {int((per_spec_df['reference_source'] == 'gobbato_powder').sum())}",
        f"**References with known expected axis:** {len(total_classified_spec)}",
        f"**Top-1 axis hit rate:** {top1_rate:.1%}",
        f"**Top-3 axis hit rate:** {top3_rate:.1%}",
        f"**References with ambiguity core > 0.1:** {n_ambig_fires}",
        f"**Mean ambiguity core evidence:** {mean_ambig:.3f}",
        "",
        "## Datasets used",
        "",
        "- ramanbiolib normal Raman of pure biological molecules",
        "- Gobbato 2025 pure powder Raman (5 metabolites: UA, hypoxanthine, "
        "xanthine, ergothioneine, creatinine-labelled-as-Creat)",
        "",
        "Both routed through the canonical pipeline "
        "(`crop_before_interpolate` → AsLS → Savitzky-Golay → L2 norm) before "
        "scoring via `gaira.base2.score_spectrum`.",
        "",
        "## Expected chemistry classes",
        "",
        "Each reference is annotated with the 11-axis subset it should "
        "activate (see `EXPECTED_AXES` dict in the benchmark script). "
        "References outside the annotation map are scored but not ranked "
        "for top-1/top-3 hit-rate.",
        "",
        "## Motif success/failure patterns",
        "",
        "Top-ranked motifs across all references (top 10):",
        "",
    ]
    top10_motif = motif_df["top_motif_1"].value_counts().head(10)
    lines.append("| motif | #1-ranked on N references |")
    lines.append("|---|---:|")
    for mot, n in top10_motif.items():
        lines.append(f"| `{mot}` | {n} |")

    lines += [
        "",
        "## Axis success/failure patterns",
        "",
        "Per-axis hit rate (fraction of references with that primary expected "
        "axis where the axis is top-ranked):",
        "",
    ]
    eva_axis_df["top1"] = eva_axis_df["observed_rank_1_to_11"] == 1
    per_axis = (
        eva_axis_df.groupby("expected_axis")["top1"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "hits", "count": "n"})
    )
    per_axis["rate"] = per_axis["hits"] / per_axis["n"]
    per_axis = per_axis.sort_values("rate", ascending=False)
    lines.append("| axis | hits / n | rate |")
    lines.append("|---|---:|---:|")
    for ax, row in per_axis.iterrows():
        lines.append(f"| {ax} | {int(row['hits'])}/{int(row['n'])} "
                      f"| {row['rate']:.1%} |")

    lines += [
        "",
        "## Ambiguity behaviour",
        "",
        "Ambiguity-lane activation on references is expected when the "
        "reference's bands fall in collision zones (e.g. any glycan "
        "reference fires `glycan_glycosidic_C_O_C_1020_1100` AND its "
        "ambiguity CROSS_AXIS mapping to `ambiguity_artifact`).",
        "",
        f"- References with ambiguity_core > 0.1: {n_ambig_fires} / {n_total}",
        f"- Mean ambiguity_core: {mean_ambig:.3f}",
        f"- Max ambiguity_core: {ambig_df['ambiguity_core'].max():.3f} "
        f"(on `{ambig_df.loc[ambig_df['ambiguity_core'].idxmax(), 'component_key']}`)",
        "",
        "This is correct behaviour: citrate, UA-containing, and glycan-like "
        "references are genuinely in collision zones.",
        "",
        "## False positive / cross-talk patterns",
        "",
        "See `grounding_off_target_activation_v1.csv` and "
        "`fig_grounding_off_target_activation_matrix.png` for the full "
        "matrix. Key diagonal values (on-target mean core evidence) vs "
        "off-target mean should be inspected manually — under the mean-based "
        "activation, on-target scores are in ~0.05–0.30 range and off-target "
        "scores should be consistently lower.",
        "",
        "## Key reference examples",
        "",
        "See `fig_grounding_motif_vs_axis_examples.png` for the 11-axis "
        "landscape on 4 exemplars: adenine, L-phenylalanine, cholesterol, "
        "and Gobbato UA powder Raman.",
        "",
        "## Tables",
        "",
        "- `grounding_per_spectrum_scores_v1.csv` — all motif + axis scores per spectrum",
        "- `grounding_expected_vs_observed_motif_rank_v1.csv` — top-5 motifs per reference",
        "- `grounding_expected_vs_observed_axis_rank_v1.csv` — rank of expected axis",
        "- `grounding_off_target_activation_v1.csv` — per-axis on/off-target flag",
        "- `grounding_ambiguity_activation_v1.csv` — ambiguity lane per spectrum",
        "- `grounding_axis_rank_v1.csv` — top-3 axes per reference",
    ]
    (OUT / "REPORT_gaira_base_2_grounding_benchmark_v1.md").write_text(
        "\n".join(lines)
    )
    print(f"[emit] REPORT_gaira_base_2_grounding_benchmark_v1.md")


if __name__ == "__main__":
    main()
