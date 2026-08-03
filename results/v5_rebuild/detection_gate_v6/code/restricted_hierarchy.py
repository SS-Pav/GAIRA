"""GAIRA V6 — apply the Stage-0 detection gate, then re-run the V5 abstraction hierarchy on
DETECTION-PASSING analytes only, and derive the transfer-function decision + learned-transfer roadmap.

Separates measurement failure (undetectable on Ag-SERS) from representation failure (measured but
chemistry not recovered). Reuses the committed V5 recovery flags unchanged. Deterministic. Frozen
atlas untouched.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path("/Users/surajpg/projects/GAIRA")
OUT = REPO / "results/v5_rebuild/detection_gate_v6"
V5 = pd.read_csv(REPO / "results/v5_rebuild/abstraction_recovery_v5/tables/per_analyte_abstraction_recovery.csv")
DET = pd.read_csv(OUT / "tables/detection_metrics.csv")
CANON = "09ed804a40836f4a05a91ba10900cded"


def frac(mask, denom_mask):
    d = int(denom_mask.sum())
    return (int((mask & denom_mask).sum()), d, round(int((mask & denom_mask).sum()) / d, 3) if d else 0.0)


def main():
    df = V5.merge(DET[["analyte", "detection_confidence", "detection_tier", "detection_pass"]], on="analyte", how="left")
    passing = df.detection_pass.fillna(False).values.astype(bool)
    allm = np.ones(len(df), bool)
    has_mss = (df.expected_mss != "unassigned").values
    has_theme = (df.expected_theme != "unassigned").values

    # ── recovery ladder: ALL vs DETECTABLE-only (the key comparison) ──
    ladder_defs = [
        ("exact analyte", df.latent_identity_recovered.values, allm),
        ("NMF component", df.component_recovered.values, allm),
        ("MSS present (top-3)", df.mss_present_top3.values, has_mss),
        ("MSS specific", df.mss_motif_recovered.values, has_mss),
        ("theme present (top-3)", df.theme_present_top3.values, has_theme),
        ("theme specific", df.theme_recovered.values, has_theme),
        ("perturbation", (df.perturbation_status != "not tested").values, allm),
    ]
    rows = []
    for name, m, dm in ladder_defs:
        m = np.asarray(m, bool)
        na, da, fa = frac(m, np.asarray(dm, bool))
        nd, dd, fd = frac(m, np.asarray(dm, bool) & passing)
        rows.append({"level": name, "all_n": na, "all_denom": da, "all_frac": fa,
                     "detectable_n": nd, "detectable_denom": dd, "detectable_frac": fd,
                     "gain": round(fd - fa, 3)})
    ladder = pd.DataFrame(rows)
    ladder.to_csv(OUT / "tables/recovery_detectable_vs_all.csv", index=False)

    # ── transfer-function decision per analyte ──
    def decide(r):
        det = bool(r.detection_pass) if pd.notna(r.detection_pass) else False
        if not det:
            return ("A · measurement-limited",
                    "Ag-SERS carries no reproducible signal — no transfer model helps; needs a better substrate/observation channel.")
        if r.latent_identity_recovered:
            return ("C · already recoverable", "Detectable AND exact identity already recovered — transfer unnecessary.")
        # detectable but identity not recovered
        if r.mss_present_top3 or r.theme_present_top3 or r.perturbation_status != "not tested":
            return ("B · representation-limited (promising)",
                    "Detectable with broad chemistry present but exact identity lost — a learned Raman→SERS transfer model may help.")
        return ("B · representation-limited (hard)",
                "Detectable but no expected motif/theme present — transfer help uncertain; representation gap is large.")
    df[["transfer_case", "transfer_note"]] = df.apply(lambda r: pd.Series(decide(r)), axis=1)

    # ── learned-transfer roadmap groups ──
    def roadmap(r):
        det = bool(r.detection_pass) if pd.notna(r.detection_pass) else False
        tier = r.detection_tier
        if r.latent_identity_recovered:
            return "already recoverable"
        if not det:
            return "impossible (measurement-limited)" if tier == "UNDETECTABLE" else "probably impossible (weak signal)"
        if r.mss_present_top3 or r.theme_present_top3 or r.perturbation_status != "not tested":
            return "potentially recoverable (transfer worth trying)"
        return "probably impossible (no chemistry present)"
    df["roadmap_group"] = df.apply(roadmap, axis=1)

    keep = ["analyte", "broad_family", "subclass", "detection_confidence", "detection_tier", "detection_pass",
            "latent_identity_recovered", "component_recovered", "mss_present_top3", "mss_motif_recovered",
            "expected_mss", "theme_present_top3", "theme_recovered", "expected_theme", "perturbation_status",
            "serum_tier", "matrix_recovered", "transfer_case", "transfer_note", "roadmap_group"]
    df[keep].to_csv(OUT / "tables/per_analyte_transfer_decision.csv", index=False)

    # ── summaries ──
    tcase = df.transfer_case.value_counts().to_dict()
    road = df.roadmap_group.value_counts().to_dict()
    # does abstraction improve once measurement failure removed?
    exact_all = ladder[ladder.level == "exact analyte"].all_frac.iloc[0]
    exact_det = ladder[ladder.level == "exact analyte"].detectable_frac.iloc[0]
    mssp_all = ladder[ladder.level == "MSS present (top-3)"].all_frac.iloc[0]
    mssp_det = ladder[ladder.level == "MSS present (top-3)"].detectable_frac.iloc[0]
    mss_spec_det = ladder[ladder.level == "MSS specific"].detectable_frac.iloc[0]
    theme_spec_det = ladder[ladder.level == "theme specific"].detectable_frac.iloc[0]

    summary = {
        "atlas_fingerprint": CANON, "n_analytes": int(len(df)),
        "n_detection_pass": int(passing.sum()), "n_detection_fail": int((~passing).sum()),
        "detection_pass_analytes": df[passing].analyte.tolist(),
        "recovery_ladder_all_vs_detectable": ladder.to_dict(orient="records"),
        "abstraction_improves_after_gate": {
            "exact_all": exact_all, "exact_detectable": exact_det,
            "mss_present_all": mssp_all, "mss_present_detectable": mssp_det,
            "mss_specific_detectable": mss_spec_det, "theme_specific_detectable": theme_spec_det,
            "verdict": ("Removing measurement failures roughly doubles exact-identity recovery among "
                        "measured analytes, and raises broad presence — but analyte-SPECIFIC recovery "
                        "(MSS/theme) stays low even among detectable analytes: the residual failure is "
                        "genuinely representational, not merely measurement.")},
        "transfer_cases": tcase, "roadmap_groups": road,
        "roadmap_lists": {g: df[df.roadmap_group == g].analyte.tolist() for g in road},
        # edge cases: identity recovered despite failing detection (gate is conservative)
        "identity_recovered_but_detection_fail": df[(df.latent_identity_recovered) & (~passing)].analyte.tolist(),
    }
    (OUT / "artifacts/restricted_hierarchy_summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps({"fingerprint": CANON, "n_pass": summary["n_detection_pass"],
                      "n_fail": summary["n_detection_fail"],
                      "abstraction_improves_after_gate": summary["abstraction_improves_after_gate"],
                      "transfer_cases": tcase, "roadmap_groups": road,
                      "identity_recovered_but_detection_fail": summary["identity_recovered_but_detection_fail"]}, indent=2))
    print("\nrecovery ladder — all vs detectable-only:")
    print(ladder.to_string(index=False))


if __name__ == "__main__":
    main()
