"""Per-axis evidence audit.

For each BSV axis, aggregate signals from:
  - peak_assignments (local, peak-level evidence)
  - knowledge_chunks (prose, diffuse evidence, organised by `section`)
  - condition_differential_profile (contrast-explicit directionality)
  - calibration_axis_recovery.csv (downstream benchmark, READ-ONLY; never
    influences the audit's own scoring — included only as a column).

Locality score = fraction of axis-related evidence that is peak-level
(vs prose-level). Anchor-hint match rate = fraction of peak rows whose
peak_cm lies inside one of the canonical anchor ranges.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import duckdb
import pandas as pd

from gaira.expected.axis_mapping import (
    BSV_AXES, AXIS_ANCHOR_HINTS, assigned_row_to_axis,
)


DEFAULT_DB_PATH = Path("/Volumes/SSD_Rad/GAIRA_DATA/interim/gaira.duckdb")

# Map knowledge_chunks.section labels to the BSV axis they plausibly
# discuss. Unmapped sections contribute to the "diffuse / cross-cutting"
# pool and are counted but not attributed to any single axis.
_SECTION_TO_AXIS = {
    "protein_regions":             "protein_backbone",
    "amide_regions":               "protein_backbone",
    "lipid_regions":               "membrane_lipid",
    "ch_regions":                  "membrane_lipid",
    "aromatic_regions":            "aromatic_amino_acid",
    "nucleic_acid_regions":        "nucleic_acid_backbone",
    "carbohydrate_regions":        "glycan_carbohydrate",
    "choline_nucleic_overlap_regions": "nucleic_acid_backbone",
    "upper_700_900_overlap_regions":   "purine_nucleotide",
    "low_wavenumber_regions":          "redox_metabolite",
    "low_wavenumber_sers_cautions":    "redox_metabolite",
}


@dataclass
class AxisAuditRow:
    axis: str
    n_peak_rows: int
    n_sources: int
    n_molecules: int
    distinct_molecules_top5: str
    peak_cm_min: float
    peak_cm_max: float
    peak_cm_median: float
    share_high_conf: float
    share_medium_conf: float
    share_low_conf: float
    anchor_hint_hit_rate: float      # fraction of peak rows within canonical anchor hints
    matrix_serum_count: int          # rows with matrix_context containing "serum"
    matrix_ev_count: int              # rows with "extracellular vesicles" / "ev"
    matrix_biofluid_count: int        # rows with "biofluid"
    substrate_sers_count: int         # rows with "SERS"
    substrate_raman_count: int        # rows with "Raman"
    n_prose_chunks: int               # knowledge_chunks rows mapped to this axis
    n_conditions_explicit: int        # conditions with non-flat delta in condition_differential_profile
    n_conditions_up: int
    n_conditions_down: int
    locality_score: float             # n_peak_rows / (n_peak_rows + n_prose_chunks)
    support_strength: str              # "strong" | "moderate" | "sparse"
    calibration_status: str            # from calibration_axis_recovery.csv
    notes: str


def _support_strength(n_peak_rows: int, share_high_plus_medium: float) -> str:
    if n_peak_rows >= 40 and share_high_plus_medium >= 0.75:
        return "strong"
    if n_peak_rows >= 15:
        return "moderate"
    return "sparse"


def _load_peak_assignments(db_path: Path) -> pd.DataFrame:
    with duckdb.connect(str(db_path), read_only=True) as con:
        df = con.execute("""
            SELECT assignment_id, source_id, peak_cm, tolerance_cm,
                   assigned_molecule, assigned_group, matrix_context,
                   confidence_text, evidence_text
            FROM peak_assignments
        """).df()
    df["axis"] = df.apply(
        lambda r: assigned_row_to_axis(
            r["assigned_group"], r["assigned_molecule"], r["evidence_text"],
        ),
        axis=1,
    )
    return df


def _load_chunks(db_path: Path) -> pd.DataFrame:
    with duckdb.connect(str(db_path), read_only=True) as con:
        df = con.execute("""
            SELECT chunk_id, source_id, section, chunk_text
            FROM knowledge_chunks
        """).df()
    df["axis"] = df["section"].map(_SECTION_TO_AXIS)
    return df


def _load_differential_profile() -> pd.DataFrame:
    p = Path(__file__).resolve().parents[3] / "outputs" / "landscape_v4" / "condition_differential_profile.csv"
    if not p.exists():
        return pd.DataFrame(columns=["condition", "component", "delta", "direction"])
    return pd.read_csv(p)


def _load_calibration_recovery() -> pd.DataFrame:
    p = Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_calibration_eval_v1"
            "/tables/calibration_axis_recovery.csv")
    if not p.exists():
        return pd.DataFrame(columns=["axis", "recovery_category"])
    return pd.read_csv(p)


def _anchor_hit_rate(peaks: pd.Series, axis: str) -> float:
    hints = AXIS_ANCHOR_HINTS.get(axis, [])
    if not hints or peaks.empty:
        return 0.0
    hit = 0
    for p in peaks:
        for lo, hi, _ in hints:
            if lo <= p < hi:
                hit += 1
                break
    return round(hit / len(peaks), 3)


def _count_if_contains(series: pd.Series, token: str) -> int:
    return int(series.fillna("").str.lower().str.contains(token).sum())


def build_axis_audit(db_path: Path = DEFAULT_DB_PATH) -> pd.DataFrame:
    peaks = _load_peak_assignments(db_path)
    chunks = _load_chunks(db_path)
    diff = _load_differential_profile()
    calib = _load_calibration_recovery()

    calib_by_axis = dict(zip(calib.get("axis", []), calib.get("recovery_category", [])))

    rows: list[AxisAuditRow] = []
    for axis in BSV_AXES:
        axis_peaks = peaks[peaks["axis"] == axis]
        axis_chunks = chunks[chunks["axis"] == axis]

        n_peak = len(axis_peaks)
        n_prose = len(axis_chunks)
        n_sources = int(axis_peaks["source_id"].nunique()) if n_peak else 0
        n_molecules = int(axis_peaks["assigned_molecule"].nunique()) if n_peak else 0

        if n_peak:
            confs = axis_peaks["confidence_text"].fillna("unknown").value_counts(normalize=True)
            share_h = float(confs.get("high", 0.0))
            share_m = float(confs.get("medium", 0.0))
            share_l = float(confs.get("low", 0.0))
            pk = axis_peaks["peak_cm"].dropna()
            pk_min = float(pk.min()) if not pk.empty else 0.0
            pk_max = float(pk.max()) if not pk.empty else 0.0
            pk_med = float(pk.median()) if not pk.empty else 0.0
            anchor_rate = _anchor_hit_rate(pk, axis)
            top_mols = "; ".join(
                axis_peaks["assigned_molecule"].value_counts().head(5).index.tolist()
            )
            mx = axis_peaks["matrix_context"]
            matrix_serum = _count_if_contains(mx, "serum")
            matrix_ev = _count_if_contains(mx, "extracellular vesicle") + _count_if_contains(mx, "vesicle")
            # Count biofluids exclusive of serum/EV where they don't already appear
            matrix_biofluid = int(
                (mx.fillna("").str.lower().str.contains("biofluid")
                 & ~mx.fillna("").str.lower().str.contains("serum|vesicle")).sum()
            )
            substrate_sers = _count_if_contains(mx, "sers")
            substrate_raman = int(
                (mx.fillna("").str.lower().str.contains("raman")
                 & ~mx.fillna("").str.lower().str.contains("sers")).sum()
            )
        else:
            share_h = share_m = share_l = 0.0
            pk_min = pk_max = pk_med = 0.0
            anchor_rate = 0.0
            top_mols = ""
            matrix_serum = matrix_ev = matrix_biofluid = 0
            substrate_sers = substrate_raman = 0

        axis_diff = diff[diff["component"] == axis]
        n_conds_up = int((axis_diff["direction"] == "up").sum())
        n_conds_down = int((axis_diff["direction"] == "down").sum())
        n_conds_explicit = n_conds_up + n_conds_down

        locality = round(n_peak / max(n_peak + n_prose, 1), 3)
        support = _support_strength(n_peak, share_h + share_m)

        rows.append(AxisAuditRow(
            axis=axis,
            n_peak_rows=n_peak,
            n_sources=n_sources,
            n_molecules=n_molecules,
            distinct_molecules_top5=top_mols,
            peak_cm_min=round(pk_min, 1),
            peak_cm_max=round(pk_max, 1),
            peak_cm_median=round(pk_med, 1),
            share_high_conf=round(share_h, 3),
            share_medium_conf=round(share_m, 3),
            share_low_conf=round(share_l, 3),
            anchor_hint_hit_rate=anchor_rate,
            matrix_serum_count=matrix_serum,
            matrix_ev_count=matrix_ev,
            matrix_biofluid_count=matrix_biofluid,
            substrate_sers_count=substrate_sers,
            substrate_raman_count=substrate_raman,
            n_prose_chunks=n_prose,
            n_conditions_explicit=n_conds_explicit,
            n_conditions_up=n_conds_up,
            n_conditions_down=n_conds_down,
            locality_score=locality,
            support_strength=support,
            calibration_status=calib_by_axis.get(axis, "not_tested"),
            notes="",
        ))

    # Also record a synthetic "ambiguous / unmapped" row for transparency.
    ambig = peaks[peaks["axis"].isna()]
    if len(ambig):
        rows.append(AxisAuditRow(
            axis="_ambiguous_unmapped",
            n_peak_rows=len(ambig),
            n_sources=int(ambig["source_id"].nunique()),
            n_molecules=int(ambig["assigned_molecule"].nunique()),
            distinct_molecules_top5="; ".join(
                ambig["assigned_molecule"].value_counts().head(5).index.tolist()
            ),
            peak_cm_min=round(float(ambig["peak_cm"].min()), 1),
            peak_cm_max=round(float(ambig["peak_cm"].max()), 1),
            peak_cm_median=round(float(ambig["peak_cm"].median()), 1),
            share_high_conf=0.0, share_medium_conf=0.0, share_low_conf=0.0,
            anchor_hint_hit_rate=0.0,
            matrix_serum_count=_count_if_contains(ambig["matrix_context"], "serum"),
            matrix_ev_count=_count_if_contains(ambig["matrix_context"], "vesicle"),
            matrix_biofluid_count=0,
            substrate_sers_count=_count_if_contains(ambig["matrix_context"], "sers"),
            substrate_raman_count=0,
            n_prose_chunks=0,
            n_conditions_explicit=0, n_conditions_up=0, n_conditions_down=0,
            locality_score=1.0,
            support_strength="ambiguous",
            calibration_status="n/a",
            notes="Mixed-Vibrational + unclassifiable Metabolite/AminoAcid rows",
        ))

    return pd.DataFrame([asdict(r) for r in rows])
