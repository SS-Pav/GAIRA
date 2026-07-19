"""Build component_theme_weights_v1 — many-to-many component→theme mapping.

Each weight w[component, theme] combines THREE independent evidence lines, each
recorded so any weight can be explained:

  (1) reference-loading evidence  — the chemical families that load the component
      (Component Audit p2) mapped through family_theme_affinity.
  (2) spectral-band evidence      — overlap of the component's dominant Raman bands
      with the theme's characteristic bands.
  (3) perturbation evidence       — chemically-specific analytes that drive the
      component (Perturbation Response) mapped to their theme.

The three are combined with fixed, documented mixing weights and normalised so each
component's theme weights sum to 1 (fractional contribution). No weight is arbitrary.
"""
from __future__ import annotations
import sys, json, yaml
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path("/Users/surajpg/projects/GAIRA")
AUDIT = REPO / "results/v5_rebuild/reference_atlas_audit/tables"
RESP = REPO / "results/v5_rebuild/perturbation_response/tables"
ENG = REPO / "src/gaira/engine/data"
OUT = REPO / "results/v5_rebuild/engine_v1/artifacts"
sys.path.insert(0, str(REPO / "src"))
from gaira.foundation.families_raman import family_of
K = 24

# documented mixing of the three evidence lines (fixed, not tuned)
MIX = {"loading": 0.50, "spectral": 0.25, "perturbation": 0.25}
BAND_TOL = 20.0     # cm-1 window for band-to-theme matching

# analytes whose perturbation identity is chemically specific enough to carry a theme
PERTURBATION_THEME = {
    "adenine": "nucleic_purine", "hypoxanthine": "nucleic_purine", "guanine": "nucleic_purine",
    "xanthine": "nucleic_purine", "urate": "nucleic_purine",
    "ergothioneine": "sulfur_antioxidant",
}


def main():
    onto = yaml.safe_load((ENG / "biochemical_ontology_v2.yaml").read_text())
    theme_ids = [t["id"] for t in onto["themes"]]
    theme_bands = {t["id"]: t.get("characteristic_bands_cm", []) for t in onto["themes"]}
    fam_aff = onto["family_theme_affinity"]

    reg = json.loads((OUT / "component_registry_v1.json").read_text())
    dose = pd.read_csv(RESP / "part1_component_dose_response.csv")
    fpr = pd.read_csv(RESP / "part2_response_fingerprints.csv")
    dcols = [f"d{j}" for j in range(K)]

    rows = {}
    for comp in reg["components"]:
        j = comp["component_id"]["value"]
        # (1) loading evidence
        fam_share = comp["top_biochemical_families"]["value"]
        e_load = {t: 0.0 for t in theme_ids}
        for fam, share in fam_share.items():
            for th, a in fam_aff.get(fam, {}).items():
                e_load[th] += share * a
        # (2) spectral evidence
        peaks = comp["dominant_raman_peaks_cm"]["value"]
        e_spec = {t: 0.0 for t in theme_ids}
        for t in theme_ids:
            tb = theme_bands[t]
            if not tb or not peaks:
                continue
            hit = sum(1 for p in peaks if any(abs(p - b) <= BAND_TOL for b in tb))
            e_spec[t] = hit / max(1, len(peaks))
        # (3) perturbation evidence — analytes driving this component
        e_pert = {t: 0.0 for t in theme_ids}
        col = fpr[dcols].values[:, j]
        for a, name in zip(np.argsort(-col)[:5], fpr.analyte.values[np.argsort(-col)[:5]]):
            if col[a] <= 0:
                continue
            th = PERTURBATION_THEME.get(name)
            if th:
                e_pert[th] += float(col[a])
        # background / unknown themes are NOT evidenced by chemistry; assign to
        # components that are generic (many families, low purity) via a residual
        purity = comp["purity"]["value"]
        generic = (len(fam_share) >= 4) or (purity < 0.25)

        def _norm(d):
            s = sum(d.values())
            return {k: v / s for k, v in d.items()} if s > 1e-12 else d

        el, es, ep = _norm(e_load), _norm(e_spec), _norm(e_pert)
        combined = {t: MIX["loading"] * el.get(t, 0) + MIX["spectral"] * es.get(t, 0) +
                    MIX["perturbation"] * ep.get(t, 0) for t in theme_ids}
        # route residual (unexplained) mass to background (if generic) or unknown
        total = sum(combined.values())
        residual = max(0.0, 1.0 - total)
        combined["background_matrix" if generic else "unknown_mixed"] += residual
        combined = _norm(combined)

        # drop negligible themes, then renormalise the kept ones to a proper distribution
        kept = {t: w for t, w in combined.items() if w >= 0.02}
        ks = sum(kept.values())
        kept = {t: w / ks for t, w in kept.items()} if ks > 1e-12 else kept
        entries = []
        for t in theme_ids:
            if t not in kept:
                continue
            w = kept[t]
            entries.append({
                "theme": t, "weight": round(float(w), 4),
                "evidence": {
                    "reference_loading": round(float(el.get(t, 0)), 3),
                    "spectral_band": round(float(es.get(t, 0)), 3),
                    "perturbation": round(float(ep.get(t, 0)), 3),
                },
                "confidence": ("high" if (el.get(t, 0) > 0.4 and es.get(t, 0) > 0) else
                               "moderate" if (el.get(t, 0) > 0.3 or ep.get(t, 0) > 0.3) else "low"),
            })
        rows[j] = sorted(entries, key=lambda e: -e["weight"])

    out = {"schema": "component_theme_weights_v1", "version": "1.0",
           "ontology_version": onto["version"], "atlas_fingerprint": reg["atlas_fingerprint"],
           "mixing_weights": MIX, "band_tolerance_cm": BAND_TOL,
           "note": "Per component, theme weights sum to 1. Each carries the three evidence lines "
                   "that produced it. Background/unknown absorb residual unexplained mass.",
           "weights": {str(j): rows[j] for j in rows}}
    (OUT / "component_theme_weights_v1.json").write_text(json.dumps(out, indent=2))
    print("component_theme_weights_v1.json written")
    # sanity: each component sums to ~1
    for j in rows:
        s = sum(e["weight"] for e in rows[j])
        assert abs(s - 1.0) < 0.05, f"c{j} weights sum {s}"
    # show a few
    for j in [1, 3, 15, 17, 19]:
        top = rows[j][:3]
        print(f"  c{j}: " + ", ".join(f"{e['theme']}={e['weight']:.2f}" for e in top))


if __name__ == "__main__":
    main()
