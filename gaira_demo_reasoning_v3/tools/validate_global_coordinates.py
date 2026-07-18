"""GAIRA Demo v3 — validation & diagnostics runner (report numbers).

Prints (and optionally JSON-dumps) the quantitative results used in the V3
reports: variance/dominance before vs after calibration, redox dominance,
disease effect sizes (labels used only AFTER the frozen fit), nuisance
associations, and SHINE projection with its dimensional caveat.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEMO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DEMO_ROOT))
from gaira_core import config as cfg                    # noqa: E402
from gaira_core import global_coordinates as gc         # noqa: E402
from gaira_core import coordinate_validation as cv       # noqa: E402


def main():
    calib = gc.load_calibration()
    df = cv.load_reference_samples()
    if calib is None or df is None:
        print("GLOBAL COORDINATE UNAVAILABLE — artifacts missing")
        return 2

    out = {}

    # ── variance / dominance before vs after ──
    va = cv.variance_before_after(df)
    out["variance_before_after"] = va.to_dict(orient="records")
    print("=== per-axis variance rank (1 = largest); raw vs global ===")
    print(va[["axis_short", "raw_var_rank", "global_var_rank",
              "raw_dyn_range", "global_dyn_range"]].to_string(index=False))

    # ── redox dominance ──
    rd = cv.redox_dominance(df)
    out["redox_dominance"] = rd
    print("\n=== redox dominance (G10) ===")
    print(f"raw dyn range={rd['raw_dynamic_range']:.4f} global dyn range={rd['global_dynamic_range']:.3f} "
          f"| raw var rank={rd['raw_variance_rank']} -> global var rank={rd['global_variance_rank']} "
          f"| global max|z|={rd['global_max_abs']:.2f}")

    # ── disease effect sizes (post-fit; labels NOT used in calibration) ──
    print("\n=== serum-liver effect sizes (global coords, Cohen's d, HA vs cancer) ===")
    serum = df[df.dataset == "serum_liver"]
    out["serum_effect_sizes"] = {}
    for canc in ("CCA", "HCC", "LM"):
        es = cv.group_effect_sizes(serum, "label", canc, "HA")
        top = es.head(3)[["axis_short", "cohens_d"]].values.tolist()
        out["serum_effect_sizes"][f"{canc}_vs_HA"] = es.to_dict(orient="records")
        print(f"  {canc} vs HA top: " + ", ".join(f"{a} d={d:+.2f}" for a, d in top))

    print("\n=== EV-diabetes effect sizes (Impact vs Strong-D) ===")
    ev = df[df.dataset == "ev_diabetes"]
    es = cv.group_effect_sizes(ev, "label", "Impact", "Strong-D")
    out["ev_effect_sizes"] = es.to_dict(orient="records")
    print("  " + ", ".join(f"{r.axis_short} d={r.cohens_d:+.2f}" for r in es.head(3).itertuples()))

    # ── nuisance diagnostics ──
    print("\n=== nuisance association (eta^2 of dataset identity on global coords) ===")
    eta_ds = cv.nuisance_eta_squared(df, "dataset")
    out["nuisance_dataset_eta2"] = eta_ds.to_dict(orient="records")
    print(eta_ds.head(6).to_string(index=False))
    # matrix (serum vs EV) restricted to the two SERS biological sets
    bio = df[df.dataset.isin(["serum_liver", "ev_diabetes"])].copy()
    eta_mx = cv.nuisance_eta_squared(bio, "matrix")
    out["nuisance_matrix_eta2"] = eta_mx.to_dict(orient="records")
    print("\n=== matrix (serum vs EV) eta^2 on global coords ===")
    print(eta_mx.head(6).to_string(index=False))
    mean_ds_eta = float(eta_ds["eta_squared"].mean())
    out["mean_dataset_eta2"] = mean_ds_eta
    print(f"\nmean dataset-identity eta^2 across axes = {mean_ds_eta:.3f} "
          f"({'DATASET IDENTITY DOMINATES' if mean_ds_eta > 0.5 else 'moderate' if mean_ds_eta>0.2 else 'weak'})")

    # ── SHINE projection (legacy cached BSV remap -> global; NOT a recomputed projection) ──
    print("\n=== SHINE projection (legacy 3-axis remap -> global coords) ===")
    from gaira_core import data_loader as dl
    sh, ph = dl.load_pilot_cohorts("shine_liver_injury")
    shine_rows = []
    for _, r in sh.iterrows():
        raw = {a: float(r[a]) for a in cfg.BSV_AXES}
        g = gc.global_display_dict(raw, calib)
        nz = sum(1 for a in cfg.BSV_AXES if abs(raw[a]) > 1e-4)
        shine_rows.append({"cohort": r["cohort"], "raw_nonzero_axes": nz,
                           **{a: round(g[a], 2) for a in cfg.BSV_AXES}})
    out["shine_projection"] = shine_rows
    print(f"  SHINE cohorts projected: {len(shine_rows)}; raw nonzero axes per cohort = "
          f"{[x['raw_nonzero_axes'] for x in shine_rows]} (upstream 3-axis collapse preserved)")

    if "--json" in sys.argv:
        p = cfg.GENERATED_DIR / "v3_validation_diagnostics.json"
        p.write_text(json.dumps(out, indent=2, default=float))
        print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
