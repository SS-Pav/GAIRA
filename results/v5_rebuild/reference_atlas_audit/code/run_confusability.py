"""P12 supplement — is low MSS uniqueness an atlas failure or correct chemistry?

Classifies each analyte's nearest neighbour in BSV space as:
  duplicate  — the same compound entered under two names / stereo-descriptors
  homolog    — same subfamily (e.g. the saturated triacylglycerol series, whose
               fingerprint-region Raman spectra genuinely differ only in CH2 count)
  family     — same chemical family but different subfamily
  distinct   — different chemistry (a genuine atlas confusion)

Only the last category counts against the atlas.
"""
from __future__ import annotations
import sys, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

REPO = Path("/Users/surajpg/projects/GAIRA"); sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from gaira.foundation.families_raman import family_of, _norm
import atlas_audit as AA

OUT = REPO / "results/v5_rebuild/reference_atlas_audit"
TAB = OUT / "tables"

# stereo/anomeric descriptors and trivial synonyms that denote the SAME molecule
SYNONYM = {"dextrose": "glucose", "levulose": "fructose", "uric acid": "urate",
           "glutamic acid": "glutamate", "aspartic acid": "aspartate",
           "riboﬂavin": "riboflavin"}


def canon(name: str) -> str:
    s = _norm(name)
    s = s.replace("(+)-", "").replace("(-)-", "").replace("(+)", "").replace("(-)", "").strip()
    s = SYNONYM.get(s, s)
    return s


def classify(a, b):
    if canon(a) == canon(b):
        return "duplicate"
    fa, fb = family_of(a), family_of(b)
    sa, sb = AA.subfamily(a), AA.subfamily(b)
    if fa == fb and sa == sb and sa != "unavailable":
        return "homolog"
    if fa == fb and fa != "unknown":
        return "family"
    # same coarse molecular class (e.g. fatty acid / triacylglycerol / phospholipid are
    # all lipids and share the dominant acyl-chain bands) is still explicable chemistry
    ca, cb = AA.molecular_class(a), AA.molecular_class(b)
    if ca == cb and ca != "unassigned":
        return "same_class"
    return "distinct"


def main():
    m = pd.read_csv(TAB / "p12_mss_readiness.csv")
    m["nn1"] = m.nearest_neighbours.apply(lambda x: eval(x)[0])
    m["nn1_sim"] = m.nn_similarity.apply(lambda x: eval(x)[0])
    m["nn_relation"] = [classify(a, b) for a, b in zip(m.analyte, m.nn1)]

    counts = m.nn_relation.value_counts().to_dict()
    low = m[m.signature_uniqueness < 0.15]
    low_counts = low.nn_relation.value_counts().to_dict()

    # uniqueness measured against only GENUINELY DIFFERENT chemistry
    genuine = m[m.nn_relation == "distinct"]
    summary = {
        "n_analytes": int(len(m)),
        "nearest_neighbour_relation_counts": counts,
        "n_low_uniqueness(<0.15)": int(len(low)),
        "low_uniqueness_relation_counts": low_counts,
        "fraction_of_low_uniqueness_explained_by_chemistry":
            round(float(sum(v for k, v in low_counts.items() if k != "distinct") / max(1, len(low))), 3),
        "class_level_resolution": "atlas separates molecular CLASS; within-class species are often not resolved",
        "median_uniqueness_all": round(float(m.signature_uniqueness.median()), 3),
        "median_uniqueness_vs_distinct_chemistry": round(float(genuine.signature_uniqueness.median()), 3)
            if len(genuine) else None,
        "n_genuine_confusions": int(len(genuine[genuine.signature_uniqueness < 0.15])),
        "genuine_confusion_examples": genuine.nsmallest(10, "signature_uniqueness")[
            ["analyte", "nn1", "nn1_sim"]].to_dict("records"),
    }
    m.to_csv(TAB / "p12_mss_readiness_with_relations.csv", index=False)
    (TAB / "p12_confusability_summary.json").write_text(json.dumps(summary, indent=2))

    print("=== MSS confusability: is low uniqueness chemistry or atlas failure? ===")
    print("nearest-neighbour relation (all analytes):", counts)
    print(f"low-uniqueness analytes (<0.15): {len(low)} — relations {low_counts}")
    print(f"  explained by duplicate/homolog/same-family chemistry: "
          f"{summary['fraction_of_low_uniqueness_explained_by_chemistry']:.0%}")
    print(f"median uniqueness — all: {summary['median_uniqueness_all']:.3f} | "
          f"vs genuinely distinct chemistry: {summary['median_uniqueness_vs_distinct_chemistry']}")
    print(f"genuine confusions (<0.15 and different chemistry): {summary['n_genuine_confusions']}")
    for r in summary["genuine_confusion_examples"][:6]:
        print(f"    {r['analyte']:30s} ~ {r['nn1']:30s} cos={r['nn1_sim']:.3f}")


if __name__ == "__main__":
    main()
