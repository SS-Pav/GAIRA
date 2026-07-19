"""Chemical-family assignment for the Raman reference corpus.

Extends gaira.evidence.families (which covers only the small-metabolite Stage-B set)
with rule-based coverage for the RamanBioLib compound space: sugars, fatty acids,
triglycerides, sterols, phospholipids, proteins, nucleic acids, nucleotides and
cofactors. Assignment is from KNOWN CHEMISTRY (name//class rules), never inferred
from spectra, and is used only for post-hoc interpretation and evaluation.
"""
from __future__ import annotations
import re

from ..evidence.families import family_of as _base_family

PROTEIN_NAMES = {
    "albumin", "alb", "bovine serum albumin", "bsa", "lectin", "thaumatin", "pepsinogen",
    "carbonic anhydrase", "lysozyme", "insulin", "collagen", "hemoglobin", "haemoglobin",
    "myoglobin", "trypsin", "chymotrypsin", "casein", "globulin", "keratin", "avidin",
    "ferritin", "catalase", "elastin", "fibrinogen", "gelatin", "cytochrome c", "ovalbumin",
    "concanavalin a", "papain", "pepsin", "ribonuclease", "protamine", "histone",
}
NUCLEIC = {"dna", "rna", "a-dna", "b-dna", "z-dna", "t-rna", "trna", "mrna",
           "deoxyribonucleic acid", "ribonucleic acid"}
NUCLEOSIDES = {"adenosine", "guanosine", "cytidine", "uridine", "thymidine", "inosine",
               "deoxyadenosine", "deoxyguanosine", "deoxycytidine", "deoxythymidine"}
PURINES = {"adenine", "guanine", "hypoxanthine", "xanthine", "urate", "uric acid", "caffeine",
           "theophylline", "theobromine"}
PYRIMIDINES = {"uracil", "thymine", "cytosine", "orotic acid", "barbituric acid"}
COFACTORS = {"coenzyme a", "acetyl coenzyme a", "acetyl-coa", "nad", "nadh", "nadp", "nadph",
             "fad", "fmn", "riboflavin", "glutathione", "biotin", "thiamine", "pyridoxine",
             "folic acid", "cobalamin", "ubiquinone", "heme", "hemin", "chlorophyll",
             "ergothioneine", "atp", "adp", "amp", "gtp"}
STEROLS = {"cholesterol", "cholesteryl", "ergosterol", "stigmasterol", "sitosterol",
           "testosterone", "estradiol", "progesterone", "cortisol", "cholic acid"}
PHOSPHOLIPIDS = {"phosphatidylcholine", "phosphatidylinositol", "phosphatidylethanolamine",
                 "phosphatidylserine", "sphingomyelin", "ceramide", "lecithin", "cardiolipin"}
POLYSACCHARIDES = {"amylose", "amylopectin", "glycogen", "starch", "cellulose", "dextran",
                   "chitin", "chitosan", "inulin", "pectin", "agarose", "heparin",
                   "hyaluronic acid", "chondroitin"}

# strip only genuine stereochemistry prefixes: "(+)-", "(-)-", "d-", "l-", "dl-"
_STRIP = re.compile(r"^(\(\s*[+-]\s*\)\s*-?\s*|dl-|[dl]-)+", re.I)


def _norm(name: str) -> str:
    s = str(name).strip().lower().replace("\ufb02", "fl").replace("\ufb01", "fi")
    s = _STRIP.sub("", s).strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\b(monohydrate|dihydrate|pentahydrate|anhydrous|hydrate)\b", "", s).strip()
    return s


def family_of(name: str) -> str:
    """Coarse chemical family. Falls back to the Stage-B map, then to rules."""
    raw = str(name).strip().lower()
    base = _base_family(raw)
    if base != "unknown":
        return base
    s = _norm(raw)

    if s in NUCLEIC or s.endswith(" dna") or s.endswith(" rna"):
        return "nucleic_acid"
    if s in PROTEIN_NAMES or any(p in s for p in ("albumin", "globulin", "lectin", "enzyme")):
        return "protein"
    if s in NUCLEOSIDES:
        return "nucleoside"
    if s in PURINES:
        return "purine"
    if s in PYRIMIDINES:
        return "pyrimidine"
    if s in COFACTORS or "coenzyme" in s or s.endswith("phosphate") and "adenosine" in s:
        return "cofactor"
    if s in PHOSPHOLIPIDS or s.startswith("phosphatidyl") or "sphingo" in s or "ceramide" in s:
        return "phospholipid"
    if s in STEROLS or "cholester" in s or "sterol" in s:
        return "sterol"
    if s in POLYSACCHARIDES:
        return "polysaccharide"
    # triglycerides: tripalmitin, tristearin, triolein, trimyristin, triarachidin
    if re.match(r"^tri[a-z]+in$", s):
        return "triglyceride"
    # fatty acids: *anoic/*enoic/*oic acid, or a known fatty-acid stem + acid
    if re.search(r"(anoic|enoic|ynoic) acid$", s) or re.search(
            r"\b(palmitic|stearic|oleic|linoleic|linolenic|myristic|lauric|arachidic|"
            r"arachidonic|behenic|capric|caprylic|margaric|pentadecanoic|heptadecanoic)\b", s):
        return "fatty_acid"
    # sugars: *ose (but not cellulose/amylose handled above), plus amino/deoxy sugars
    if re.search(r"ose$", s) or "deoxyribose" in s or "galactosamine" in s or \
       "glucosamine" in s or "glucuronic" in s:
        return "saccharide"
    if re.search(r"(nucleotide|monophosphate|diphosphate|triphosphate)$", s):
        return "nucleotide"
    # enzymes and other named proteins
    if re.search(r"(ase|ases|globin|albumin|lectin|zyme|ogen|ubulin|biquitin)$", s) or \
       " proteinase" in s or "trypsin" in s or "chymotrypsin" in s or "dismutase" in s or \
       re.search(r"\b(elastase|peroxidase|transferase|oxidase|proteinase|protease|inhibitor)\b", s):
        return "protein"
    # steroid hormones and synthetic analogues
    if re.search(r"\b(estriol|estrone|estradiol|ethinylestradiol|diethylstilbestrol|"
                 r"androsterone|pregnenolone|aldosterone)\b", s) or s.startswith("estr"):
        return "sterol"
    # concatenated fatty-acid names lacking a space ("15-methylpalmiticacid")
    if re.search(r"(palmitic|stearic|myristic|oleic|lauric|arachidic|behenic|capric)acid$", s):
        return "fatty_acid"
    if re.search(r"\bcarotene\b|\blycopene\b", s):
        return "carotenoid"
    if s.endswith(" acid") or s.endswith("ate"):
        return "organic_acid"
    return "unknown"


# map a chemical family onto the vocabulary used by the literature peak-assignment
# table, so literature evidence can corroborate (not define) a theme.
FAMILY_TO_LIT = {
    "protein": "Proteins", "amino_acid": "AminoAcids", "nucleic_acid": "NucleicAcids",
    "nucleoside": "NucleicAcids", "nucleotide": "NucleicAcids", "purine": "NucleicAcids",
    "pyrimidine": "NucleicAcids", "fatty_acid": "Lipids-FattyAcids",
    "triglyceride": "Lipids-FattyAcids", "phospholipid": "Lipids-FattyAcids",
    "lipid": "Lipids-FattyAcids", "sterol": "Lipids-Hormones",
    "saccharide": "Saccharides-Monosaccharides", "polysaccharide": "Saccharides-Polysaccharides",
}
