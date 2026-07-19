"""GAIRA V5 — analyte synonym / abbreviation reconciliation (Phase 1.5).

Maps source-specific names (Gobbato abbreviations, RamanBioLib names, salt/hydrate
forms, capitalization) to one canonical analyte name so no duplicate analytes
remain. Deterministic. Extend as new sources are added.
"""
from __future__ import annotations
import re

# Gobbato filename abbreviation -> canonical analyte name
GOBBATO_ABBREV = {
    "AcCoA": "acetyl-coa", "Acetoacet": "acetoacetate", "Ala": "alanine", "Alb": "albumin",
    "Arg": "arginine", "Asc": "ascorbate", "Citric": "citrate", "CoA": "coenzyme a",
    "Creat": "creatinine", "Dfruct6P": "fructose-6-phosphate", "Ergo": "ergothioneine",
    "Fruct": "fructose", "Galact": "galactose", "Glut": "glutathione", "Glutamic": "glutamate",
    "Gly": "glycine", "Glycerol": "glycerol", "Gua": "guanine", "His": "histidine",
    "Hydroxypro": "hydroxyproline", "Ile": "isoleucine", "Lact": "lactate", "Leu": "leucine",
    "Mann": "mannose", "Methio": "methionine", "NacDgluc": "n-acetylglucosamine",
    "Oleic": "oleate", "PEP": "phosphoenolpyruvate", "PhInositol": "phosphatidylinositol",
    "Phe": "phenylalanine", "Phosph": "phosphate", "Pro": "proline", "Pyr": "pyruvate",
    "Ser": "serine", "Stearic": "stearate", "Thy": "thymine", "Triolein": "triolein",
    "Trp": "tryptophan", "Tyr": "tyrosine", "UA": "urate", "Ura": "uracil", "Val": "valine",
    "Xanth": "xanthine", "Ade": "adenine", "Chol": "cholesterol", "Cys": "cysteine",
    "Gluc": "glucose", "Glycogen": "glycogen", "Hypox": "hypoxanthine", "Ribo": "riboflavin",
    "Urea": "urea",
}

# explicit canonicalizations for cross-source name variants (RamanBioLib etc.)
_ALIAS = {
    "uric acid": "urate", "uric": "urate", "l-tryptophan": "tryptophan",
    "l-arginine": "arginine", "l-histidine": "histidine", "l-asparagine": "asparagine",
    "l-cysteine": "cysteine", "l-cystine": "cystine", "l-lysine": "lysine",
    "l-methionine": "methionine", "l-phenylalanine": "phenylalanine",
    "l-tyrosine": "tyrosine", "l-serine": "serine", "l-proline": "proline",
    "l-leucine": "leucine", "l-isoleucine": "isoleucine", "l-valine": "valine",
    "l-alanine": "alanine", "l-glutamic acid": "glutamate", "glutamic acid": "glutamate",
    "glutamic": "glutamate", "glutathione (reduced)": "glutathione",
    "d-glucose": "glucose", "cytochrome c": "cytochrome c", "l-cystathionine": "cystathionine",
    "ascorbic acid": "ascorbate", "citric acid": "citrate", "oleic acid": "oleate",
    "stearic acid": "stearate", "lactic acid": "lactate", "pyruvic acid": "pyruvate",
    "uric acid ": "urate",
}


def canonical(name: str) -> str:
    """Return the canonical analyte name (lowercase). Strips L-/D-/DL- prefixes,
    salt/hydrate suffixes, capitalization; applies the alias table."""
    s = str(name).strip().lower()
    s = re.sub(r"\s+", " ", s)
    if s in _ALIAS:
        return _ALIAS[s]
    # strip salt/hydrate forms
    s = re.sub(r"\b(hydrochloride|hcl|sodium|potassium|sulfate|sulphate|acetate salt|"
               r"dihydrate|monohydrate|anhydrous|disodium|hemisulfate)\b", "", s).strip()
    s = re.sub(r"^(l-|d-|dl-)", "", s).strip()
    s2 = s.replace(" acid", "").strip()
    if s in _ALIAS:
        return _ALIAS[s]
    if s2 in _ALIAS:
        return _ALIAS[s2]
    # normalize "...ic acid" -> "...ate" is analyte-specific; leave unless aliased
    return s
