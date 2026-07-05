"""Derive v3-specific demo tables.

v1 data is reused unchanged for pure-molecule layer, atlas, and calibration
raw ΔBSV. v3 adds a small set of *summary* tables that make the evidence
layers explicit.

Outputs (under streamlit_apps/gaira_demo_v3/data/):
  - grounding_layer_summary.csv
  - literature_evidence_layer.csv
  - calibration_metadata_v3.csv
  - regression_registry.csv

Run:
    cd /Users/suraj/projects/GAIRA
    PYTHONPATH=src .venv/bin/python streamlit_apps/gaira_demo_v3/build_v3_assets.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

V1_DATA = ROOT / "streamlit_apps" / "gaira_demo" / "data"
V3_OUT = Path(__file__).resolve().parent / "data"
V3_OUT.mkdir(parents=True, exist_ok=True)

ATLAS_FULL = ROOT / "config" / "spectral_anchor_windows_v1.csv"


# ──────────────────────────────────────────────────────────────────────
# 1. Grounding layer summary — three distinct evidence layers
# ──────────────────────────────────────────────────────────────────────

def build_grounding_layer_summary() -> None:
    summary = pd.read_csv(V1_DATA / "grounding_corpus_summary.csv")
    m = dict(zip(summary["metric"], summary["value"]))
    mol_index = pd.read_csv(V1_DATA / "grounding_molecule_index.csv")
    n_families = mol_index["family"].nunique()

    atlas = pd.read_csv(ATLAS_FULL)
    all_ids = set()
    for s in atlas["supporting_source_ids"].dropna():
        for x in s.split(";"):
            x = x.strip()
            if x:
                all_ids.add(x)
    paper_ids = {s for s in all_ids if s.startswith("src_paper_")}
    core_ids = {s for s in all_ids if not s.startswith("src_paper_")}

    rows = [
        {
            "layer": "pure_molecule",
            "title": "Pure-molecule reference layer",
            "metric_a_label": "Pure-molecule spectra",
            "metric_a_value": int(m.get("n_molecule_spectra", 0)),
            "metric_b_label": "Unique molecules",
            "metric_b_value": int(mol_index["component"].nunique()),
            "metric_c_label": "Families",
            "metric_c_value": int(n_families),
            "datasets": "RamanBioLib (202 spectra); amino-acid Raman grounding xlsx (available, not yet auto-ingested)",
            "note": "Spectra are already preprocessed and normalized upstream. Used directly for BSV projection and per-molecule display.",
        },
        {
            "layer": "literature_linked",
            "title": "Literature-linked molecular evidence layer",
            "metric_a_label": "Unique supporting sources",
            "metric_a_value": int(len(all_ids)),
            "metric_b_label": "Literature papers",
            "metric_b_value": int(len(paper_ids)),
            "metric_c_label": "Core references (src_001–005)",
            "metric_c_value": int(len(core_ids)),
            "datasets": "Phase B literature corpus; 5 core references link to pure-molecule vetting.",
            "note": "Each atlas band is backed by one or more sources. Paper IDs carry phaseB / phaseB2 provenance tags.",
        },
        {
            "layer": "atlas",
            "title": "Raman physics atlas layer",
            "metric_a_label": "Atlas bands",
            "metric_a_value": int(m.get("n_atlas_bands", 0)),
            "metric_b_label": "Canonical axes",
            "metric_b_value": int(m.get("n_atlas_axes", 0)),
            "metric_c_label": "Spectral range (cm⁻¹)",
            "metric_c_value": f"{int(m.get('wavenumber_min_cm1', 0))}–{int(m.get('wavenumber_max_cm1', 0))}",
            "datasets": "Bands derived from the literature-linked layer + core refs.",
            "note": "Coverage is current and uneven across axes — will be refined in future work.",
        },
    ]
    pd.DataFrame(rows).to_csv(V3_OUT / "grounding_layer_summary.csv", index=False)


# ──────────────────────────────────────────────────────────────────────
# 2. Literature evidence layer — per-source breakdown
# ──────────────────────────────────────────────────────────────────────

def build_literature_evidence_layer() -> None:
    atlas = pd.read_csv(ATLAS_FULL)
    # explode supporting_source_ids
    rows = []
    for _, r in atlas.iterrows():
        srcs = [x.strip() for x in str(r.get("supporting_source_ids", "")).split(";") if x.strip()]
        for src in srcs:
            rows.append({
                "source_id": src,
                "window_id": r["window_id"],
                "primary_axis": r["primary_axis"],
                "classification": r["classification"],
            })
    long = pd.DataFrame(rows)
    grp = (
        long.groupby("source_id")
        .agg(
            n_bands_supported=("window_id", "count"),
            n_axes_touched=("primary_axis", "nunique"),
            anchor_bands=("classification", lambda s: (s == "anchor").sum()),
            ambiguous_bands=("classification", lambda s: (s == "ambiguous").sum()),
        )
        .reset_index()
    )
    grp["kind"] = grp["source_id"].apply(
        lambda s: "literature_paper" if s.startswith("src_paper_") else "core_reference"
    )
    grp = grp.sort_values(["kind", "n_bands_supported"], ascending=[True, False])
    grp.to_csv(V3_OUT / "literature_evidence_layer.csv", index=False)


# ──────────────────────────────────────────────────────────────────────
# 3. Calibration metadata v3 — human-readable labels + context
# ──────────────────────────────────────────────────────────────────────

CALIBRATION_RICH = {
    "cspp_fig7_hypoxanthine_spike": {
        "rich_label": "Serum baseline vs Hypoxanthine spike",
        "baseline_label": "Serum baseline (CSPP Fig 7)",
        "perturbed_label": "Serum + Hypoxanthine spike",
        "analyte": "Hypoxanthine",
        "matrix": "Serum",
        "substrate": "Plasmonic paper — Ag colloid",
        "perturbation_type": "Metabolite spike",
        "concentration_info": "Spiking level fixed per CSPP Fig 7 protocol",
        "behavior_class": "approximate / moderate",
        "caveat": "Atlas has limited direct hypoxanthine windows beyond purine band.",
    },
    "cspp_fig7_ergothioneine_spike": {
        "rich_label": "Serum baseline vs Ergothioneine spike",
        "baseline_label": "Serum baseline (CSPP Fig 7)",
        "perturbed_label": "Serum + Ergothioneine spike",
        "analyte": "Ergothioneine",
        "matrix": "Serum",
        "substrate": "Plasmonic paper — Ag colloid",
        "perturbation_type": "Metabolite spike",
        "concentration_info": "Spiking level fixed per CSPP Fig 7 protocol",
        "behavior_class": "direct / high-confidence",
        "caveat": "Lipid, Protein, Redox axes are SAEL-testable.",
    },
    "uricase_sigma_depletion": {
        "rich_label": "Commercial serum — Uricase untreated vs treated",
        "baseline_label": "Untreated serum (Sigma)",
        "perturbed_label": "Uricase-treated serum",
        "analyte": "Uric acid (depleted)",
        "matrix": "Serum (commercial)",
        "substrate": "Ag colloid",
        "perturbation_type": "Enzymatic depletion",
        "concentration_info": "Endpoint comparison (not a titration)",
        "behavior_class": "approximate / moderate — flagged inconsistent",
        "caveat": "Verdict is inconsistent against SAEL expectation; see v3 report.",
    },
    "uricase_spiked_hypoxanthine_serum": {
        "rich_label": "Uricase-treated serum — baseline vs Hypoxanthine spike",
        "baseline_label": "Uricase-treated serum",
        "perturbed_label": "Uricase-treated serum + Hypoxanthine spike",
        "analyte": "Hypoxanthine (post-depletion)",
        "matrix": "Serum (uricase-treated)",
        "substrate": "Ag colloid",
        "perturbation_type": "Metabolite spike (post-depletion)",
        "concentration_info": "Endpoint comparison",
        "behavior_class": "approximate / moderate",
        "caveat": "Only purine axis is above noise; other axes are flat.",
    },
    "ergothioneine_titration_top_vs_zero": {
        "rich_label": "Ergothioneine titration — 0.0 µM vs 2.0 µM endpoints",
        "baseline_label": "Serum + 0.0 µM Ergothioneine",
        "perturbed_label": "Serum + 2.0 µM Ergothioneine",
        "analyte": "Ergothioneine",
        "matrix": "Serum",
        "substrate": "Ag colloid",
        "perturbation_type": "Titration endpoints",
        "concentration_info": "Endpoints of an 11-level ladder (0.0 → 2.0 µM, step 0.2 µM)",
        "behavior_class": "direct / high-confidence",
        "caveat": "Full ladder is available in the Regression tab.",
    },
}


def build_calibration_metadata() -> None:
    conds = pd.read_csv(V1_DATA / "calibration_conditions.csv")
    rows = []
    for _, r in conds.iterrows():
        cid = r["contrast_id"]
        rich = CALIBRATION_RICH.get(cid, {})
        rows.append({
            "contrast_id": cid,
            "v1_display_name": r["display_name"],
            "rich_label": rich.get("rich_label", r["display_name"]),
            "baseline_label": rich.get("baseline_label", "Baseline"),
            "perturbed_label": rich.get("perturbed_label", "Perturbed"),
            "analyte": rich.get("analyte", "—"),
            "matrix": rich.get("matrix", "—"),
            "substrate": rich.get("substrate", "—"),
            "perturbation_type": rich.get("perturbation_type", "—"),
            "concentration_info": rich.get("concentration_info", "—"),
            "behavior_class": rich.get("behavior_class", "—"),
            "caveat": rich.get("caveat", ""),
        })
    pd.DataFrame(rows).to_csv(V3_OUT / "calibration_metadata_v3.csv", index=False)


# ──────────────────────────────────────────────────────────────────────
# 4. Regression registry — explicit supported-vs-not table
# ──────────────────────────────────────────────────────────────────────

def build_regression_registry() -> None:
    rows = [
        {
            "dataset_id": "ergothioneine_titration",
            "display_name": "Ergothioneine titration — 0.0 → 2.0 µM in serum",
            "n_levels": 11,
            "n_replicates_per_level": 5,
            "units": "µM",
            "data_path": "streamlit_apps/gaira_demo/data/ergothioneine_bsv_per_concentration.csv",
            "supported": True,
            "reason_if_unsupported": "",
            "notes": "Only true ordered ladder currently wired through the GAIRA BSV pipeline.",
        },
        {
            "dataset_id": "uricase_sigma_depletion",
            "display_name": "Uricase depletion (untreated vs treated)",
            "n_levels": 2,
            "n_replicates_per_level": 5,
            "units": "—",
            "data_path": "",
            "supported": False,
            "reason_if_unsupported": "Endpoint comparison, not an ordered series. Stays in calibration.",
            "notes": "",
        },
        {
            "dataset_id": "cspp_fig7_hypoxanthine_spike",
            "display_name": "CSPP Fig 7 — Hypoxanthine spike",
            "n_levels": 2,
            "n_replicates_per_level": 50,
            "units": "—",
            "data_path": "",
            "supported": False,
            "reason_if_unsupported": "Single spike level; not a concentration ladder.",
            "notes": "",
        },
        {
            "dataset_id": "cspp_fig7_ergothioneine_spike",
            "display_name": "CSPP Fig 7 — Ergothioneine spike",
            "n_levels": 2,
            "n_replicates_per_level": 50,
            "units": "—",
            "data_path": "",
            "supported": False,
            "reason_if_unsupported": "Single spike level; not a concentration ladder.",
            "notes": "",
        },
        {
            "dataset_id": "adenine_sers_ladder",
            "display_name": "Adenine SERS ladder (raw CSVs)",
            "n_levels": 6,
            "n_replicates_per_level": 1,
            "units": "ng/mL → µg/mL",
            "data_path": "/Volumes/SSD_Rad/GAIRA_DATA/raw/adenine_sers_control",
            "supported": False,
            "reason_if_unsupported": "Raw CSVs present but not yet wired through the GAIRA BSV preprocessing + projection pipeline; would require a dedicated loader before safe demo use.",
            "notes": "Candidate for a future v4 addition.",
        },
    ]
    pd.DataFrame(rows).to_csv(V3_OUT / "regression_registry.csv", index=False)


def main() -> None:
    print(f"[v3] writing to {V3_OUT}")
    build_grounding_layer_summary()
    build_literature_evidence_layer()
    build_calibration_metadata()
    build_regression_registry()
    print("[v3] done")


if __name__ == "__main__":
    main()
