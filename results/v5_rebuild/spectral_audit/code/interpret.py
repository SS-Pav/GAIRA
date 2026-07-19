"""Part 11 — spectroscopic interpretation generator.

Composes 2-5 sentences per analyte AS A RAMAN SPECTROSCOPIST would, strictly from
the MEASURED quantities of this audit (peak correspondence, shifts, intensity
redistribution, band-level results, replicate reproducibility) plus a small curated
table of well-established characteristic bands. Nothing about machine learning.

Every numeric claim in the generated text traces to a computed value; curated band
names are only invoked when a DETECTED peak falls within tolerance of a known band.
"""
from __future__ import annotations
import numpy as np

# curated, well-established characteristic bands (cm-1) -> short mode label.
# Used ONLY to name a band when a detected peak lies within +/-12 cm-1.
KNOWN_BANDS = {
    "phenylalanine": [(1002, "ring-breathing nu12"), (1032, "ring C-H in-plane"), (1206, "C-C ring")],
    "tyrosine": [(830, "Fermi doublet"), (850, "Fermi doublet"), (1615, "ring C=C")],
    "tryptophan": [(759, "indole ring breathing"), (1010, "indole"), (1340, "indole"), (1552, "indole C=C")],
    "adenine": [(725, "purine ring breathing"), (1330, "C-N stretch"), (1480, "imidazole")],
    "guanine": [(650, "purine ring"), (1360, "purine"), (1480, "imidazole")],
    "hypoxanthine": [(725, "purine ring breathing"), (1370, "purine")],
    "xanthine": [(660, "purine ring"), (1370, "purine")],
    "urate": [(635, "ring deformation"), (1000, "ring"), (1130, "C-N stretch"), (1400, "ring/C-O")],
    "uracil": [(785, "ring breathing"), (1235, "ring")],
    "thymine": [(745, "ring breathing"), (1370, "CH3 deformation")],
    "cytosine": [(795, "ring breathing"), (1290, "ring")],
    "glutathione": [(1400, "COO- symmetric stretch"), (1040, "C-N")],
    "cysteine": [(680, "C-S stretch"), (1400, "COO- symmetric stretch")],
    "glucose": [(1060, "C-O/C-C"), (1120, "C-O-C"), (915, "anomeric C-H")],
    "citrate": [(1400, "COO- symmetric stretch"), (950, "C-C")],
    "lactate": [(1045, "C-O stretch"), (1420, "COO- symmetric stretch")],
    "creatinine": [(680, "ring"), (1410, "CH2/C-N")],
    "riboflavin": [(1350, "isoalloxazine"), (1580, "isoalloxazine C=C")],
    "cholesterol": [(1440, "CH2 scissoring"), (1670, "C=C stretch")],
    "oleate": [(1440, "CH2 scissoring"), (1655, "cis C=C stretch")],
    "stearate": [(1440, "CH2 scissoring"), (1130, "C-C skeletal")],
    "urea": [(1000, "C-N symmetric stretch"), (1600, "NH2 bend")],
}
COO_REGION = (1380, 1430)     # carboxylate symmetric stretch — classic Ag-adsorption handle


def _name_band(analyte, wn, tol=12.0):
    for pos, label in KNOWN_BANDS.get(analyte, []):
        if abs(wn - pos) <= tol:
            return f"{pos:.0f} cm⁻¹ ({label})"
    return None


def _fmt(x, nd=2):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def interpret_analyte(analyte, family, rec, rows, bands, grid):
    matched = [r for r in rows if r["kind"] == "matched"]
    r_only = [r for r in rows if r["kind"] == "raman_only"]
    s_only = [r for r in rows if r["kind"] == "sers_only"]
    n_r, n_s, n_m = rec["n_raman_peaks"], rec["n_sers_peaks"], rec["n_matched"]
    mas = rec["mean_abs_shift"]; cos = rec["cosine"]
    rank = rec.get("red_peak_rank_corr"); redist = rec.get("red_intensity_redistribution_index")
    ceiling = rec.get("reproducibility_ceiling")
    sents = []

    # 1. overall correspondence
    recov = rec["matched_pct_of_raman"]
    sents.append(
        f"Of {n_r} prominent Raman bands, {n_m} ({recov:.0f}%) have an Ag-SERS counterpart within "
        f"{12:.0f} cm⁻¹ (peak F1 {_fmt(rec['peak_f1'])}), while the full-profile cosine is only "
        f"{cos:+.2f} — the band inventory therefore agrees far better than the overall spectral shape.")

    # 2. named preserved bands
    named = []
    for r in sorted(matched, key=lambda x: -(x["raman_prom"] or 0))[:3]:
        nb = _name_band(analyte, r["raman_peak"])
        if nb:
            named.append(f"{nb} → {r['sers_peak']:.0f} cm⁻¹ ({r['shift']:+.0f})")
    if named:
        sents.append("Characteristic bands are retained: " + "; ".join(named) + ".")
    elif matched:
        strong = sorted(matched, key=lambda x: -(x["raman_prom"] or 0))[:3]
        sents.append("Strongest retained correspondences: " + "; ".join(
            f"{r['raman_peak']:.0f}→{r['sers_peak']:.0f} cm⁻¹ ({r['shift']:+.0f})" for r in strong) + ".")

    # 3. shift behaviour
    if np.isfinite(mas):
        if mas <= 4:
            sents.append(f"Band positions are essentially conserved (mean |Δν| {mas:.1f} cm⁻¹, "
                         f"median {_fmt(rec['median_shift'],1)} cm⁻¹), i.e. within the 2 cm⁻¹ grid and "
                         "typical of physisorption without strong bond perturbation.")
        elif mas <= 8:
            sents.append(f"Modest systematic displacement (mean |Δν| {mas:.1f} cm⁻¹, median "
                         f"{_fmt(rec['median_shift'],1)} cm⁻¹) consistent with chemisorption/charge "
                         "transfer at the Ag surface rather than a change of vibrational identity.")
        else:
            sents.append(f"Larger displacements (mean |Δν| {mas:.1f} cm⁻¹, max "
                         f"{_fmt(rec['max_abs_shift'],1)} cm⁻¹) suggest strong surface interaction or "
                         "ambiguous band assignment in a congested region.")

    # 4. intensity redistribution vs position (the mechanism question)
    if redist is not None and np.isfinite(redist):
        if rank is not None and np.isfinite(rank) and rank < 0.3 and recov >= 50:
            sents.append(f"Relative intensities are strongly reordered (peak-rank ρ {_fmt(rank)}, "
                         f"redistribution index {_fmt(redist)}): the same modes are observed but "
                         "re-weighted by SERS surface selection rules, not replaced.")
        elif rank is not None and np.isfinite(rank) and rank >= 0.5:
            sents.append(f"Relative band intensities are comparatively well preserved "
                         f"(peak-rank ρ {_fmt(rank)}, redistribution {_fmt(redist)}), an unusually "
                         "faithful transfer for a colloidal SERS measurement.")
        else:
            sents.append(f"Intensity transfer is partial (peak-rank ρ {_fmt(rank)}, redistribution "
                         f"{_fmt(redist)}), with enhancement favouring a subset of modes.")

    # 5. surface-chemistry / anomaly notes
    notes = []
    coo = [r for r in matched + s_only if (r["sers_peak"] or 0) and COO_REGION[0] <= r["sers_peak"] <= COO_REGION[1]]
    if coo and family in ("amino_acid", "organic_acid", "cofactor", "purine", "lipid"):
        notes.append("a band in the 1380–1430 cm⁻¹ carboxylate symmetric-stretch window is present in "
                     "Ag-SERS, the expected signature of COO⁻ anchoring to the colloid")
    if n_s and len(s_only) / n_s > 0.5:
        notes.append(f"{len(s_only)}/{n_s} Ag-SERS bands have no Raman counterpart, which may reflect "
                     "surface-activated modes, adsorbate reorientation, or residual citrate/colloid features")
    if n_r and len(r_only) / n_r > 0.5:
        notes.append(f"{len(r_only)}/{n_r} Raman bands are not recovered, consistent with modes whose "
                     "polarisability change lies unfavourably relative to the surface normal")
    if rec.get("raman_multi_source"):
        notes.append("Raman replicates span two sources (Gobbato powder + RamanBioLib), so within-Raman "
                     "spread here is partly inter-instrument, not pure measurement noise")
    if ceiling is not None and np.isfinite(ceiling) and ceiling < 0.6:
        notes.append(f"replicate reproducibility is itself limited (within-modality cosine ceiling "
                     f"{ceiling:.2f}), which caps any achievable cross-modal agreement")
    if notes:
        sents.append("Notes: " + "; ".join(notes) + ".")

    return " ".join(sents[:5])
