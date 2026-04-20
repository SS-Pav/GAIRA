"""Expected-delta registry — contrast-specific literature-grounded shifts.

Each `ExpectedDelta` represents "for this contrast, these axes are expected to
shift, in what direction, with what confidence, supported by which anchor
windows, and with what ambiguity".

Sources:
  - `outputs/landscape_v4/condition_differential_profile.csv` for disease
    contrasts where landscape v4 has encoded a delta (HCC, NAFLD_NASH,
    hepatitis, liver_cancer_unspecified — the only four currently present).
  - `peak_assignments` molecule mentions for calibration-style contrasts
    where the perturbation is a specific analyte (hypoxanthine spike,
    uricase depletion, ergothioneine titration). We use literature
    evidence of the analyte's own SERS fingerprint to encode expected
    direction on the relevant axis — NOT calibration spectra.

Direction encoding:
  "up" / "down" / "flat" / "mixed" / "unknown"

Confidence:
  "high" — direct contrast in source, strong anchor match, low ambiguity
  "moderate" — contrast present but evidence diffuse OR anchor ambiguous
  "low" — sparse support, or contrast inferred rather than direct
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path

import pandas as pd

from gaira.expected.axis_mapping import BSV_AXES


@dataclass
class ContrastAxisExpectation:
    axis: str
    direction: str                # up | down | flat | mixed | unknown
    confidence: str               # high | moderate | low
    anchor_windows: list[tuple[float, float]] = field(default_factory=list)
    ambiguity_notes: list[str] = field(default_factory=list)
    source_ids: list[str] = field(default_factory=list)
    rationale: str = ""


@dataclass
class ExpectedDelta:
    contrast_id: str
    condition_a: str              # the "perturbed" / disease side
    condition_b: str              # the reference
    matrix: str
    substrate_context: str
    status: str                    # direct | approximate | unavailable
    overall_confidence: str       # high | moderate | low | none
    expected_axes: list[ContrastAxisExpectation] = field(default_factory=list)
    ambiguity_summary: str = ""
    rationale: str = ""
    provenance: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────
# Contrast registry: what we want expected-deltas for
# ─────────────────────────────────────────────────────────────────────

_DISEASE_CONTRASTS: list[dict] = [
    {"id": "hcc_vs_healthy_serum",
     "a": "HCC", "b": "healthy_control",
     "matrix": "serum", "substrate": "mixed (Au / Ag)",
     "rationale_prefix":
         "Hepatocellular carcinoma serum vs healthy control; landscape v4 "
         "derives axis-level deltas from aggregated Raman/SERS evidence."},
    {"id": "nafld_vs_healthy_serum",
     "a": "NAFLD_NASH", "b": "healthy_control",
     "matrix": "serum", "substrate": "mixed (Raman / SERS)",
     "rationale_prefix":
         "NAFLD/NASH vs healthy serum; landscape v4 encodes the "
         "disease-minus-healthy shift from evidence aggregation."},
    {"id": "hepatitis_vs_healthy_serum",
     "a": "hepatitis", "b": "healthy_control",
     "matrix": "serum", "substrate": "mixed",
     "rationale_prefix":
         "Hepatitis vs healthy serum, landscape v4 aggregated delta."},
    {"id": "liver_cancer_unspecified_vs_healthy_serum",
     "a": "liver_cancer_unspecified", "b": "healthy_control",
     "matrix": "serum", "substrate": "mixed",
     "rationale_prefix":
         "Unspecified liver cancer vs healthy; coarser than the HCC-specific "
         "object and typically of lower confidence."},
]

# For controlled-perturbation contrasts we do NOT have a condition_differential
# row; we encode the expected direction from the literature fingerprint of the
# spiked/depleted analyte, not from calibration spectra. This keeps the
# expected layer literature-grounded.
_CALIBRATION_LITERATURE_CONTRASTS: list[dict] = [
    {"id": "hypoxanthine_spike_literature",
     "a": "serum_plus_hypoxanthine", "b": "serum_baseline",
     "matrix": "serum", "substrate": "plasmonic paper Ag / Ag colloid",
     "analyte_molecule_tokens": ("hypoxanthine", "adenine"),
     "expected_axis_directions": {"purine_nucleotide": "up"},
     "rationale":
         "Hypoxanthine is a purine. Literature peak assignments place its "
         "ring-breathing mode in the 700–740 cm⁻¹ window (axis: "
         "purine_nucleotide). Spiking a serum matrix with hypoxanthine "
         "is expected to raise signal on that axis."},
    {"id": "uricase_depletion_literature",
     "a": "serum_plus_uricase", "b": "serum_baseline",
     "matrix": "serum", "substrate": "Ag colloid",
     "analyte_molecule_tokens": ("uric acid", "urate"),
     "expected_axis_directions": {
         "purine_nucleotide": "down",
         "aromatic_amino_acid": "mixed",  # uric acid ring ~635 falls here
         "glycan_carbohydrate": "mixed",  # uric acid peak ~890 falls here
     },
     "rationale":
         "Uricase converts uric acid to allantoin. Uric acid's peak assignments "
         "straddle the 635, 890, and 1130 cm⁻¹ regions — so depletion should "
         "reduce the purine axis primarily but with substrate-dependent leakage "
         "into aromatic_amino_acid and glycan_carbohydrate windows."},
    {"id": "ergothioneine_spike_literature",
     "a": "serum_plus_ergothioneine", "b": "serum_baseline",
     "matrix": "serum", "substrate": "Ag colloid (cAg)",
     "analyte_molecule_tokens": ("ergothioneine",),
     "expected_axis_directions": {
         "redox_metabolite": "up",
         "purine_nucleotide": "mixed",  # imidazole ring mode overlaps 720
     },
     "rationale":
         "Ergothioneine is a sulfur-containing imidazole metabolite. "
         "Literature assignments cluster in the metabolites group and "
         "overlap the 720 cm⁻¹ region. Expected direction on "
         "redox_metabolite is up, but the imidazole ring mode ~720 cm⁻¹ "
         "co-occupies the purine window, so direction on purine_nucleotide "
         "is deliberately left as 'mixed'."},
]


# ─────────────────────────────────────────────────────────────────────
# Registry public view
# ─────────────────────────────────────────────────────────────────────

CONTRAST_REGISTRY = _DISEASE_CONTRASTS + _CALIBRATION_LITERATURE_CONTRASTS


def _load_differential_profile() -> pd.DataFrame:
    p = Path(__file__).resolve().parents[3] / "outputs" / "landscape_v4" / "condition_differential_profile.csv"
    if not p.exists():
        return pd.DataFrame(columns=["condition", "component", "delta", "direction"])
    return pd.read_csv(p)


def _anchor_windows_for(axis: str, anchor_df: pd.DataFrame) -> list[tuple[float, float]]:
    sub = anchor_df[(anchor_df["axis"] == axis)
                      & (anchor_df["classification"].isin(["anchor", "secondary"]))]
    return [(float(r["start_cm"]), float(r["end_cm"])) for _, r in sub.iterrows()]


def _ambiguity_for(axis: str, anchor_df: pd.DataFrame) -> list[str]:
    sub = anchor_df[(anchor_df["axis"] == axis)
                      & (anchor_df["classification"] == "ambiguous")]
    return [
        f"window {r['start_cm']:.0f}–{r['end_cm']:.0f}: "
        f"ambiguity_score={r['ambiguity_score']:.2f}"
        for _, r in sub.iterrows()
    ]


def _build_disease_delta(
    spec: dict, diff_df: pd.DataFrame, anchor_df: pd.DataFrame,
) -> ExpectedDelta:
    rows = diff_df[diff_df["condition"] == spec["a"]]
    axes_exp: list[ContrastAxisExpectation] = []
    if rows.empty:
        status = "unavailable"
        overall_conf = "none"
        ambiguity = f"No landscape-v4 delta row for {spec['a']}."
    else:
        status = "direct"
        for axis in BSV_AXES:
            r = rows[rows["component"] == axis]
            if r.empty:
                direction = "unknown"
                conf = "low"
                rationale_axis = f"axis {axis} not listed in landscape v4 for {spec['a']}"
            else:
                direction = str(r.iloc[0]["direction"])
                delta_mag = abs(float(r.iloc[0]["delta"]))
                # Confidence scales with both delta magnitude AND anchor support.
                has_anchor = (
                    (anchor_df["axis"] == axis)
                    & (anchor_df["classification"] == "anchor")
                ).any()
                if direction == "flat":
                    conf = "low"
                elif delta_mag >= 0.7 and has_anchor:
                    conf = "high"
                elif delta_mag >= 0.3 and has_anchor:
                    conf = "moderate"
                elif delta_mag >= 0.3:
                    conf = "moderate"
                else:
                    conf = "low"
                rationale_axis = (
                    f"landscape-v4 delta={r.iloc[0]['delta']:+.2f}; "
                    f"anchor present: {has_anchor}"
                )
            axes_exp.append(ContrastAxisExpectation(
                axis=axis, direction=direction, confidence=conf,
                anchor_windows=_anchor_windows_for(axis, anchor_df),
                ambiguity_notes=_ambiguity_for(axis, anchor_df),
                source_ids=[],  # disease contrasts don't currently track per-source provenance
                rationale=rationale_axis,
            ))
        n_with_direction = sum(1 for a in axes_exp if a.direction in ("up", "down"))
        overall_conf = "moderate" if n_with_direction >= 2 else "low"
        ambiguity = (
            f"Landscape v4 has {n_with_direction} non-flat axes for {spec['a']} "
            "(coarse aggregation; use with caution)."
        )
    return ExpectedDelta(
        contrast_id=spec["id"],
        condition_a=spec["a"], condition_b=spec["b"],
        matrix=spec["matrix"], substrate_context=spec["substrate"],
        status=status, overall_confidence=overall_conf,
        expected_axes=axes_exp,
        ambiguity_summary=ambiguity,
        rationale=spec["rationale_prefix"],
        provenance=["outputs/landscape_v4/condition_differential_profile.csv"],
    )


def _build_calibration_literature_delta(
    spec: dict, peaks_df: pd.DataFrame, anchor_df: pd.DataFrame,
) -> ExpectedDelta:
    # Match peaks for the analyte.
    toks = tuple(t.lower() for t in spec["analyte_molecule_tokens"])
    mask = peaks_df["assigned_molecule"].fillna("").str.lower().apply(
        lambda s: any(t in s for t in toks)
    )
    rel = peaks_df[mask]
    provenance = sorted(rel["source_id"].dropna().unique().tolist())

    axes_exp: list[ContrastAxisExpectation] = []
    declared = spec["expected_axis_directions"]
    for axis in BSV_AXES:
        direction = declared.get(axis, "flat")
        axis_peaks = rel[rel["peak_cm"].between(1, 3500)]  # tolerant range filter
        # Axis-specific source count for confidence calibration:
        if direction in ("up", "down"):
            has_anchor = (
                (anchor_df["axis"] == axis)
                & (anchor_df["classification"] == "anchor")
            ).any()
            n_src = rel[rel["source_id"].notna()]["source_id"].nunique()
            if n_src >= 2 and has_anchor:
                conf = "moderate"   # literature-only, never "high" for single-analyte deltas
            elif n_src >= 1:
                conf = "low"
            else:
                conf = "low"
        elif direction == "mixed":
            conf = "low"
        else:
            conf = "low"

        anchor_list = _anchor_windows_for(axis, anchor_df)
        ambiguity = _ambiguity_for(axis, anchor_df)
        rationale = ""
        if direction in ("up", "down", "mixed") and len(rel):
            peak_summary = "; ".join(
                f"{r['assigned_molecule']} @ {r['peak_cm']:.0f} cm⁻¹"
                for _, r in rel.head(4).iterrows()
            )
            rationale = f"analyte peak evidence: {peak_summary}"

        axes_exp.append(ContrastAxisExpectation(
            axis=axis, direction=direction, confidence=conf,
            anchor_windows=anchor_list,
            ambiguity_notes=ambiguity,
            source_ids=provenance if direction != "flat" else [],
            rationale=rationale,
        ))

    n_nonflat = sum(1 for a in axes_exp if a.direction in ("up", "down"))
    overall_conf = "moderate" if n_nonflat >= 1 and provenance else "low"
    status = "approximate"  # always approximate — we're using analyte fingerprints, not contrast prose
    ambiguity_summary = (
        f"Calibration-literature contrast. Expected direction encoded from "
        f"{len(rel)} matching peak_assignments rows across "
        f"{len(provenance)} sources; substrate/matrix match not verified."
    )

    return ExpectedDelta(
        contrast_id=spec["id"],
        condition_a=spec["a"], condition_b=spec["b"],
        matrix=spec["matrix"], substrate_context=spec["substrate"],
        status=status, overall_confidence=overall_conf,
        expected_axes=axes_exp,
        ambiguity_summary=ambiguity_summary,
        rationale=spec["rationale"],
        provenance=provenance,
    )


def build_expected_delta_objects(
    anchor_df: pd.DataFrame,
    peaks_df: pd.DataFrame,
) -> list[ExpectedDelta]:
    """Build the full list of ExpectedDelta objects.

    Args:
        anchor_df: output of build_anchor_window_registry
        peaks_df: raw peak_assignments dataframe
    """
    diff_df = _load_differential_profile()
    out: list[ExpectedDelta] = []
    for spec in _DISEASE_CONTRASTS:
        out.append(_build_disease_delta(spec, diff_df, anchor_df))
    for spec in _CALIBRATION_LITERATURE_CONTRASTS:
        out.append(_build_calibration_literature_delta(spec, peaks_df, anchor_df))
    return out


def to_serializable(delta: ExpectedDelta) -> dict:
    d = asdict(delta)
    # tuples → lists for JSON
    for axis_obj in d["expected_axes"]:
        axis_obj["anchor_windows"] = [list(w) for w in axis_obj["anchor_windows"]]
    return d
