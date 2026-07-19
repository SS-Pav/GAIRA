"""Build Component Registry v1 — every latent component as a structured object.

Derived deterministically from FROZEN evidence:
  * Raman Reference Atlas v0.1 (basis spectra, fingerprint)
  * Component Audit (p1 inventory, p2 composition, p4 coherence)
  * Perturbation Response Audit (dose-response, response fingerprints, uricase)

Nothing is refitted. Every field records its provenance source.
"""
from __future__ import annotations
import sys, json, hashlib, time
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path("/Users/surajpg/projects/GAIRA"); sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "results/v5_rebuild/reference_atlas_audit/code"))
FROZEN = REPO / "results/v5_rebuild/foundation/artifacts"
AUDIT = REPO / "results/v5_rebuild/reference_atlas_audit/tables"
RESP = REPO / "results/v5_rebuild/perturbation_response/tables"
OUT = REPO / "results/v5_rebuild/engine_v1/artifacts"
OUT.mkdir(parents=True, exist_ok=True)
K = 24

# provenance tag helper
def prov(value, source):
    return {"value": value, "provenance": source}


def main():
    man = json.loads((FROZEN / "manifold.json").read_text())
    W = np.load(FROZEN / "manifold_components.npz")["components"]
    grid = np.load(FROZEN / "manifold_components.npz")["grid"]
    fp = hashlib.sha256(np.ascontiguousarray(W).tobytes()).hexdigest()[:32]
    assert fp == man["fingerprint"], "atlas fingerprint mismatch — refusing to build"

    inv = pd.read_csv(AUDIT / "p1_component_inventory.csv")
    comp = pd.read_csv(AUDIT / "p2_full_analyte_composition.csv")
    coh = pd.read_csv(AUDIT / "p4_chemical_coherence.csv").set_index("component")
    dose = pd.read_csv(RESP / "part1_component_dose_response.csv")
    fpr = pd.read_csv(RESP / "part2_response_fingerprints.csv")
    rob = pd.read_csv(RESP / "part11_component_robustness.csv").set_index("component")
    uric = json.loads((RESP / "part8_uricase.json").read_text())
    import atlas_audit as AA
    bands = AA.component_bands(W, grid)

    registry = {"schema": "component_registry_v1", "version": "1.0",
                "atlas_fingerprint": fp, "n_components": K,
                "basis_reference": "results/v5_rebuild/foundation/artifacts/manifold_components.npz",
                "generated_utc": None,  # left null: deterministic build, no wall-clock in artifact
                "provenance_note": "Every component field carries its source study. Basis spectra are "
                                   "referenced (not duplicated) from the frozen atlas npz.",
                "components": []}

    # serum-spike activation per component: which analytes activate it (from fingerprints)
    dcols = [f"d{j}" for j in range(K)]
    Fserum = fpr[dcols].values
    for j in range(K):
        r = inv[inv.component == j].iloc[0]
        cc = comp[comp.component == j].nlargest(8, "contribution_pct")
        loadings = [{"analyte": a, "contribution_pct": round(float(p), 2), "family": f}
                    for a, p, f in zip(cc.analyte, cc.contribution_pct, cc.chemical_family)]
        fam_share = (cc.groupby("chemical_family").contribution_pct.sum()
                     .sort_values(ascending=False))
        fam_share = (fam_share / fam_share.sum()).round(3).to_dict()

        d_resp = dose[(dose.component == j) & (dose.responsive)]
        dose_ev = [{"experiment": e.experiment, "spearman_rho": round(float(e.spearman_rho), 3),
                    "direction": e.direction, "effect_size": round(float(e.effect_size), 2),
                    "saturating_r2": (round(float(e.saturating_r2), 3)
                                      if "saturating_r2" in e._fields and pd.notna(e.saturating_r2) else None)}
                   for e in d_resp.itertuples()]
        # serum spike activators (top analytes by |delta| on this component)
        col = Fserum[:, j]
        up = np.argsort(-col)[:5]; down = np.argsort(col)[:5]
        spike_ev = {"top_activators": [{"analyte": fpr.analyte.iloc[i], "delta": round(float(col[i]), 4)}
                                       for i in up if col[i] > 0][:5],
                    "top_suppressors": [{"analyte": fpr.analyte.iloc[i], "delta": round(float(col[i]), 4)}
                                        for i in down if col[i] < 0][:5]}
        depletion_ev = {"uricase_delta": round(float(uric["delta_components"][str(j)]), 4),
                        "note": ("selectively decreased by urate depletion"
                                 if j in uric.get("components_decreased", []) else
                                 "increased by urate depletion" if j in uric.get("components_increased", [])
                                 else "stable under urate depletion")}

        stability = float(r.bootstrap_stability)
        purity = float(r.class_purity)
        n_dose = int(rob.loc[j, "n_dose_experiments_responsive"]) if j in rob.index else len(dose_ev)
        # interpretation: derive from the STRONGEST evidence, flagging label unreliability
        interp, conf, caveats = _interpret(j, r, loadings, fam_share, dose_ev, spike_ev)

        registry["components"].append({
            "component_id": prov(int(j), "frozen atlas index"),
            "dominant_raman_peaks_cm": prov([round(float(x), 0) for x in bands[j].position.values[:8]],
                                            "Component Audit p5 (find_peaks on basis loading)"),
            "spectral_bandwidth_cm": prov(round(float(bands[j].width_cm.median()), 1) if len(bands[j]) else None,
                                          "Component Audit p5"),
            "reference_analyte_loadings": prov(loadings, "Component Audit p2 composition"),
            "top_biochemical_families": prov(fam_share, "Component Audit p2 (family shares)"),
            "bootstrap_stability": prov(round(stability, 3), "Component Audit p1 (analyte bootstrap)"),
            "purity": prov(round(purity, 3), "Component Audit p4 (dominant-family fraction)"),
            "n_dose_experiments_responsive": prov(n_dose, "Perturbation Response part1"),
            "dose_response_evidence": prov(dose_ev, "Perturbation Response part1"),
            "serum_spike_evidence": prov(spike_ev, "Perturbation Response part2 fingerprints"),
            "depletion_evidence": prov(depletion_ev, "Perturbation Response part8 (uricase)"),
            "audit_label_v0_1": prov(r.primary_interpretation, "Component Audit p1 (coarse label — UNRELIABLE on low purity)"),
            "current_interpretation": prov(interp, "engine v1 synthesis of loading+dose+spike evidence"),
            "interpretation_confidence": prov(conf, "engine v1 rubric (stability x purity x perturbation agreement)"),
            "known_caveats": prov(caveats, "engine v1"),
            "outstanding_uncertainties": prov(_uncertainties(j, purity, stability), "engine v1"),
            "version_history": [{"version": "1.0", "change": "initial registry from frozen v0.1 evidence"}],
        })

    (OUT / "component_registry_v1.json").write_text(json.dumps(registry, indent=2, default=float))
    print(f"component_registry_v1.json written ({K} components, fingerprint {fp})")
    # quick confidence summary
    confs = pd.Series([c["interpretation_confidence"]["value"] for c in registry["components"]])
    print("interpretation confidence:", confs.value_counts().to_dict())


def _interpret(j, r, loadings, fam_share, dose_ev, spike_ev):
    top_fam = max(fam_share, key=fam_share.get) if fam_share else "unknown"
    top_analyte = loadings[0]["analyte"] if loadings else "n/a"
    # perturbation identity: is a chemically-specific analyte among strong spike activators?
    activators = [a["analyte"] for a in spike_ev["top_activators"]]
    interp = (f"Latent Raman motif whose reference loadings are dominated by {top_fam} chemistry "
              f"(top analyte {top_analyte}). ")
    if dose_ev:
        dirs = {e["direction"] for e in dose_ev}
        interp += f"Perturbation-responsive in {len(dose_ev)} dose experiment(s) ({'/'.join(dirs)}). "
    if activators:
        interp += f"Serum-spike activators: {', '.join(activators[:3])}."
    stab, pur = float(r.bootstrap_stability), float(r.class_purity)
    if stab >= 0.85 and pur >= 0.40 and dose_ev:
        conf = "high"
    elif stab >= 0.80 and (pur >= 0.30 or dose_ev):
        conf = "moderate"
    else:
        conf = "low"
    caveats = []
    if pur < 0.30:
        caveats.append("low chemical purity — the audit's coarse label is unreliable; trust the "
                       "reference loadings and perturbation identity over the label.")
    if stab < 0.75:
        caveats.append("below-threshold bootstrap stability; may be partly a fitting mixture.")
    if not dose_ev:
        caveats.append("no dose-response evidence; interpretation rests on static loadings only.")
    return interp, conf, caveats


def _uncertainties(j, purity, stability):
    u = []
    if purity < 0.5:
        u.append("shares evidence across multiple chemical families (mixture motif)")
    u.append("theme attribution is fractional (see component_theme_weights_v1)")
    u.append("responses are matrix-specific (pure vs serum consistency ~0)")
    return u


if __name__ == "__main__":
    main()
