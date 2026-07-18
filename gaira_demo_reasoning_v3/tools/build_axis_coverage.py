"""GAIRA Demo v3 — per-axis reference-space coverage table.

Combines, per ontology axis:
  * unique reference analytes / measured spectra / direct sources / literature
    (from the V2-derived grounding evidence table + corpus summary),
  * calibration datasets that exercise the axis (via ontology MSS analytes),
  * biological datasets that OCCUPY the axis (from the frozen reference samples),
  * ontology grounding/independence status.

NA is preserved (never shown as 0). Flags axes with insufficient grounding.
Output: data/generated/axis_reference_coverage_v1.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEMO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DEMO_ROOT))
from gaira_core import config as cfg                # noqa: E402
from gaira_core.ontology import load_ontology       # noqa: E402

GEN = cfg.GENERATED_DIR

# calibration datasets available in V3 and the analytes they exercise
CALIBRATION_ANALYTE_DATASETS = {
    "adenine": "adenine (6 conc, live Ag-SERS)",
    "ergothioneine": "ergothioneine (55 spectra, live Ag-SERS)",
    "uric_acid": "uric-acid/hypoxanthine/uricase (SAEL contrasts, cached)",
    "hypoxanthine": "uric-acid/hypoxanthine/uricase (SAEL contrasts, cached)",
}


def main() -> int:
    onto = load_ontology()

    grounding = None
    gp = GEN / "per_axis_grounding_evidence.csv"
    if gp.exists():
        grounding = pd.read_csv(gp, keep_default_na=False).set_index("axis")

    ref = None
    rp = GEN / "global_coordinate_reference_samples_v1.csv"
    if rp.exists():
        ref = pd.read_csv(rp)

    rows = []
    for ax in cfg.BSV_AXES:
        a = onto.axis(ax)
        gr = grounding.loc[ax] if grounding is not None and ax in grounding.index else None

        uniq = gr["unique_reference_analytes"] if gr is not None else "NA"

        # calibration datasets exercising this axis
        cal = sorted({CALIBRATION_ANALYTE_DATASETS[m]
                      for m in (a.contributing_mss_analytes if a else ())
                      if m in CALIBRATION_ANALYTE_DATASETS})

        # biological datasets occupying this axis (>50% of samples with raw>1e-3)
        occ = []
        if ref is not None:
            col = f"raw_{ax}"
            for ds in ("serum_liver", "ev_diabetes"):
                sub = ref[ref["dataset"] == ds]
                if len(sub):
                    frac = float((sub[col].to_numpy(float) > 1e-3).mean())
                    if frac >= 0.5:
                        occ.append(f"{ds}({frac:.0%})")

        status = a.grounding_status if a else "unknown"
        insufficient = status in ("derived_split", "insufficiently_grounded")

        rows.append({
            "axis": ax,
            "axis_short": cfg.axis_short(ax),
            "unique_reference_analytes": uniq,
            "measured_reference_spectra": "NA (corpus-level: 160)",
            "direct_spectral_sources": "NA (corpus-level: 12 unique)",
            "supporting_literature_sources": "NA (corpus-level: 16 unique)",
            "calibration_datasets": "; ".join(cal) if cal else "none",
            "biological_datasets_occupying": "; ".join(occ) if occ else "none",
            "ontology_independence_status": status,
            "insufficient_grounding_flag": insufficient,
        })

    out = pd.DataFrame(rows)
    out.to_csv(GEN / "axis_reference_coverage_v1.csv", index=False)
    print("wrote", GEN / "axis_reference_coverage_v1.csv")
    print(out[["axis_short", "unique_reference_analytes", "calibration_datasets",
               "biological_datasets_occupying", "ontology_independence_status"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
