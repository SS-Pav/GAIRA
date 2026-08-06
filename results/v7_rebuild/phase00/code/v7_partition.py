"""GAIRA V7 Phase 00 — the frozen chemical-family partition and evaluation ontology.

The V7 specification listed three partition problems that had to be resolved before any
fitting: the `unknown` bucket (6 analytes — not a chemistry), the
`lipid`/`fatty_acid`/`triglyceride` overlap, and `polysaccharide` vs `saccharide`.

All three are already resolved by the **V6.3 cleaned ontology**, which was built and
statistically revalidated in the previous generation. V7 adopts it rather than inventing a
fourth ontology, for two reasons: it dissolves `unknown` into real chemistry, and it is the
ontology the V5 baseline was last measured under — so the Phase-07 comparison stays
like-for-like.

  fine (16 classes)  — the decomposition partition and the fine evaluation label
  broad (6 classes)  — the superclass evaluation label
  old  (18 classes)  — retained for continuity with published V5/V6 numbers only

The partition is an ORGANISATIONAL PRIOR. It is never a prediction target, never a term in
any local loss, and never an input to inference.
"""
from __future__ import annotations

import pandas as pd

import v7_paths as P

V63_AUDIT = P.V63 / "tables" / "v63_analyte_audit.csv"

# Written chemical rationale per fine class. A class with no rationale cannot be frozen.
FINE_CLASS_RATIONALE: dict[str, str] = {
    "peptide_protein": "Polypeptides and proteins; amide I/II/III backbone modes dominate.",
    "mono_oligosaccharide": "Mono- and oligosaccharides; C-O-C / C-O-H ring and glycosidic modes.",
    "free_amino_acid": "Free amino acids in zwitterionic form; COO-/NH3+ modes plus side-chain.",
    "fatty_acid": "Free fatty acids; long acyl chain C-C/CH2 modes, terminal carboxyl.",
    "acylglycerol": "Mono/di/triacylglycerols; acyl chain PLUS the ~1745 ester carbonyl and "
                    "~1160 C-O-C that separate them from free fatty acids.",
    "sterol_steroid": "Fused-ring sterols and steroids; ring-breathing and methyl modes.",
    "carboxylic_acid_metabolite": "Small organic acids of central metabolism; carboxylate modes.",
    "sulfur_thiol_cofactor": "Thiol / thioether / CoA-type cofactors; C-S and S-H modes.",
    "purine": "Purine ring systems; ring-breathing near 720-730 cm-1.",
    "pyrimidine": "Pyrimidine ring systems; ring modes near 780-800 cm-1.",
    "polysaccharide": "Glycosidically polymerised sugars; distinguished from monosaccharides "
                      "by glycosidic-linkage modes, and given a separate class because V6 "
                      "found a distinct polysaccharide_glycosidic motif.",
    "chromophore_pigment": "Conjugated pigments (carotenoid polyene, flavin, porphyrin); "
                           "strong resonance-enhanced conjugated-chain modes.",
    "phospholipid_sphingolipid": "Phospholipids and sphingolipids; phosphate head group plus "
                                 "acyl chain.",
    "nucleic_acid_polymer": "DNA/RNA polymers; backbone phosphate plus base modes.",
    "small_nitrogenous": "Small nitrogen-containing metabolites (urea, creatinine).",
    "phosphate_metabolite": "Phosphorylated small molecules and inorganic phosphate; "
                            "symmetric PO stretch near 980 cm-1.",
}

BROAD_CLASS_RATIONALE: dict[str, str] = {
    "protein_amino_acid": "Peptide, protein and free amino-acid chemistry.",
    "lipid": "Fatty acyl, acylglycerol, sterol and phospholipid chemistry.",
    "carbohydrate": "Mono-, oligo- and polysaccharide chemistry.",
    "energy_metabolism": "Carboxylic-acid and phosphate metabolites of central metabolism.",
    "nucleic": "Purine, pyrimidine and nucleic-acid-polymer chemistry.",
    "redox_cofactor": "Sulfur/thiol cofactors and conjugated redox pigments.",
}

# How each V7 partition problem was resolved, recorded for audit.
PARTITION_RESOLUTIONS = [
    {"problem": "`unknown` (6 analytes) is not a chemistry",
     "resolution": "Dissolved. The V6.3 fine ontology assigns every analyte to a real "
                   "chemical class; no `unknown` class exists in the frozen partition.",
     "status": "RESOLVED"},
    {"problem": "`lipid` (5) overlaps `fatty_acid` (12) and `triglyceride` (15)",
     "resolution": "Replaced by three chemically separated classes: fatty_acid (17), "
                   "acylglycerol (17) and phospholipid_sphingolipid (5). The ester carbonyl "
                   "(~1745) and C-O-C (~1160) are the spectroscopic basis for the split.",
     "status": "RESOLVED"},
    {"problem": "`polysaccharide` (5) vs `saccharide` (27)",
     "resolution": "Kept separate: mono_oligosaccharide (27) and polysaccharide (5). "
                   "Justification: glycosidic polymerisation is spectroscopically real and V6 "
                   "derived distinct glycan_co_network and polysaccharide_glycosidic motifs.",
     "status": "RESOLVED (kept separate, with rationale)"},
]


def load_v63_ontology() -> pd.DataFrame:
    df = pd.read_csv(V63_AUDIT)
    return df[["analyte", "old_family", "new_fine_family", "new_broad_superclass",
               "expected_theme", "primary_motif"]].copy()


def build_partition(canon: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map the V6.3 ontology onto canonical IDs and detect class conflicts.

    A conflict is a canonical ID whose surface forms were filed under different classes —
    exactly the `acetyl coenzyme a` (protein) vs `acetyl-coa` (cofactor) case. Under the V7
    class-partitioned decomposition an unresolved conflict would put one molecule into two
    independent local fits.
    """
    ont = load_v63_ontology().set_index("analyte")
    rows, conflicts = [], []
    for _, r in canon.iterrows():
        forms = r.surface_forms.split(";")
        got = {c: ont.loc[c].to_dict() for c in forms if c in ont.index}
        if not got:
            rows.append({"canonical_id": r.canonical_id, "fine_class": "", "broad_class": "",
                         "old_family": "", "expected_theme": "", "primary_motif": "",
                         "source_forms": r.surface_forms, "conflict": True})
            conflicts.append({"canonical_id": r.canonical_id, "field": "coverage",
                              "values": "", "resolution": "NOT COVERED BY ONTOLOGY"})
            continue
        out = {"canonical_id": r.canonical_id, "source_forms": r.surface_forms}
        conflict = False
        for field, col in (("fine_class", "new_fine_family"),
                           ("broad_class", "new_broad_superclass"),
                           ("old_family", "old_family"),
                           ("expected_theme", "expected_theme"),
                           ("primary_motif", "primary_motif")):
            vals = sorted({str(v[col]) for v in got.values() if pd.notna(v[col])})
            # the canonical form's own value wins; disagreement is recorded
            own = got.get(r.canonical_id, {}).get(col)
            out[field] = str(own) if pd.notna(own) else (vals[0] if vals else "")
            if len(vals) > 1:
                conflict = True
                conflicts.append({
                    "canonical_id": r.canonical_id, "field": field, "values": " | ".join(vals),
                    "resolution": f"canonical form '{r.canonical_id}' → '{out[field]}'"})
        out["conflict"] = conflict
        rows.append(out)
    part = pd.DataFrame(rows)
    return part, pd.DataFrame(conflicts)


def class_census(part: pd.DataFrame, canon: pd.DataFrame) -> pd.DataFrame:
    """Analyte and spectrum counts per fine class, plus Phase-02 viability."""
    m = part.merge(canon[["canonical_id", "n_spectra", "n_sources", "sources",
                          "n_excitations"]], on="canonical_id")
    rows = []
    for cls, g in m.groupby("fine_class"):
        n = len(g)
        k_max = n // 2
        src = pd.Series(";".join(g.sources).split(";")).value_counts()
        dom = float(src.iloc[0] / src.sum()) if len(src) else 0.0
        rows.append({
            "fine_class": cls,
            "broad_class": g.broad_class.mode().iat[0] if len(g) else "",
            "n_canonical_analytes": n,
            "n_spectra": int(g.n_spectra.sum()),
            "k_c_ceiling": k_max,
            "phase02_route": ("local_decomposition" if k_max >= 1 else "anchor_only"),
            "dominant_source": src.index[0] if len(src) else "",
            "dominant_source_fraction": round(dom, 3),
            "source_confounded": bool(dom >= 0.9 and n >= 3),
            "rationale": FINE_CLASS_RATIONALE.get(cls, ""),
        })
    return (pd.DataFrame(rows).sort_values("n_canonical_analytes", ascending=False)
            .reset_index(drop=True))
