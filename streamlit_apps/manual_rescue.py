"""GAIRA Rescue Ops Console — folder-based rescue workflow.

Tab 1: Browse blocked papers + open links for manual download
Tab 2: Scan a folder for downloaded PDFs, auto-match to blocked papers
Tab 3: Extract + Review + Approve evidence
Tab 4: Pipeline refresh
Tab 5: Structured evidence viewer
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
import streamlit as st

# ──────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────
V2 = Path("/Volumes/SSD_Rad/GAIRA_DATA/structured_evidence_v2")
STAGING = V2 / "staging" / "rescue_packet"
BLOCKED_REG = V2 / "registry" / "master_blocked_registry.csv"
RESCUE_LOG = V2 / "reports" / "phaseK1_rescue_status_update_log.csv"
EXT_RUNS = V2 / "processed" / "extraction_runs"

TIER_ORDER = ["critical_A", "critical_B", "high_value_rescue_later", "secondary"]
TERMINAL = {"rescued_and_ingested", "manual_failed"}
HIDDEN = {"rescued_and_ingested", "manual_failed", "deferred"}

# ──────────────────────────────────────────────────────────
# Data helpers (no caching — always fresh from disk)
# ──────────────────────────────────────────────────────────
def load_blocked() -> pd.DataFrame:
    if not BLOCKED_REG.exists(): return pd.DataFrame()
    df = pd.read_csv(BLOCKED_REG, dtype=str).fillna("")
    tr = {t: i for i, t in enumerate(TIER_ORDER)}
    df["_rank"] = df["tier"].map(tr).fillna(99).astype(int)
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0)
    return df.sort_values(["_rank", "score"], ascending=[True, False])

def _log(pid, action, old, new, agv=0, note=""):
    if old == new and action not in ("staged",): return
    row = {"timestamp": datetime.now().isoformat(), "paper_id": pid,
           "action": action, "old_status": old, "new_status": new,
           "agv_rows": str(agv), "note": note}
    hdr = not RESCUE_LOG.exists()
    with open(RESCUE_LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if hdr: w.writeheader()
        w.writerow(row)

def _set_status(pid, new):
    df = pd.read_csv(BLOCKED_REG, dtype=str).fillna("")
    m = df["paper_id"] == pid
    old = df.loc[m, "access_status"].iloc[0] if m.any() else ""
    if old != new and m.any():
        df.loc[m, "access_status"] = new
        df.to_csv(BLOCKED_REG, index=False)
    return old

# ──────────────────────────────────────────────────────────
# Extraction engine (K3A + K3B patched, verified M1)
# ──────────────────────────────────────────────────────────
ASSIGNMENT_VERBS = ("assigned to","attributed to","represents","represent",
    "corresponds to","corresponding to","associated with","due to","arises from","arising from")
NOISE_TERMS = ("doi","copyright","creativecommons","license","http","www.","et al")
TENTATIVE_TERMS = ("tentative","possibly","possible","may be","might be","likely","putative","region","broad")
SUBFAMILY_HINT_TERMS = (
    "amide","protein","lipid","dna","rna","nucleic","phenylalanine","tyrosine","tryptophan",
    "glycogen","carbohydrate","glycan","carotenoid","adenine","guanine","cytosine","thymine",
    "citrate","c-h","c=c","c-c","c-n","n-h","o-h","s-s","c=o","ch2","ch3","nh2","phosphate",
    "sulfate","cholesterol","collagen","hemoglobin","glucose","urea","creatinine","uric acid",
    "bilirubin","dopamine","serotonin","amino acid","peptide","nucleotide","saccharide",
    "stretching","bending","deformation","vibration","breathing","symmetric","asymmetric",
    "backbone","ring","methyl","methylene","carbonyl","ester","ether","hydroxyl","amine",
    "imine","thiol","disulfide","phosphodiester","glycosidic","deoxyribose","nucleobase",
    "pyrimidine","imidazole","purine ring")

def _extract_pdf(p):
    try: return subprocess.run(["pdftotext",str(p),"-"],capture_output=True,text=True,check=True).stdout
    except: return ""

def _classify(mn,orig,method):
    c=re.sub(r"\s+"," ",mn).strip(" ,;:."); l=c.lower(); ol=orig.lower()
    if len(c)<4 or sum(d.isdigit() for d in c)>=max(3,len(c)//3) or any(t in ol for t in NOISE_TERMS) or c.count("(")!=c.count(")"): return "reject_noise"
    if re.search(r"^(figure|fig\.|table|scheme)\s",l): return "reject_noise"
    if method=="text_regex" and not any(t in l for t in SUBFAMILY_HINT_TERMS): return "mention_only"
    if method=="table_assignment" and not any(t in l for t in SUBFAMILY_HINT_TERMS): return "reject_noise"
    if any(t in l for t in TENTATIVE_TERMS): return "validated_secondary"
    if method=="table_assignment": return "validated_secondary"
    return "validated_primary"

def _is_fp(t,pk):
    l=t.lower()
    for p in [r"\d+\s*mW",r"\d+\s*nm\b",r"\d+\s*RPM",r"\d+\s*°C",r"\d+\s*min",r"\d+\s*hours?",r"\d+\s*Torr",r"order of magnitude",r"acquisition time",r"randomness"]:
        if re.search(p,l): return True
    if re.search(rf"{int(pk)}\s*(?:nm|mW|RPM|°C|min)\b",t): return True
    return False

def _parse_peak(s):
    if "\u2013" in s or "\u2212" in s or "-" in s:
        parts=re.split(r"[\u2013\u2212\-]",s)
        nums=[float(p.strip()) for p in parts if re.match(r"^\d{3,4}(?:\.\d+)?$",p.strip())]
        if len(nums)==2: return (nums[0]+nums[1])/2
    return float(s)

def _clean(raw):
    c=re.split(r"[,\u2013\u2212\-]\s*\d{3,4}\s*(?:cm[\-\u2212\u2013]?\s*1|cm-1|cm\u22121)",raw)[0]
    c=re.sub(r"\s*\([a-z]{2,5}\)\s*$","",c)
    c=re.sub(r"^\([a-z]{2,5}\)[,\s]*","",c)
    c=re.sub(r"^(?:and\s+|or\s+|,\s*)","",c)
    c=re.sub(r"\s*[+]\s*$","",c)
    return c.strip(" ,;–\u2013-")

_CM1=r"(?:cm[\-\u2212\u2013]?\s*1|cm[\-\u2212\u2013]?1|cm-1|cm\u22121|cm\u20131)"
_PEAK=r"(?P<peak>\d{3,4}(?:[\u2013\u2212\-]\d{3,4})?(?:\.\d+)?)"
_AVERBS=(r"(?:explicitly\s+)?(?:assigned to|attributed to|represents?|corresponds?\s+to"
         r"|corresponding\s+to|associated with|due to|aris(?:es?|ing)\s+from)")

def _split_assignment(meaning: str, method: str) -> dict:
    """Split a meaning string into chemistry / biology / metadata layers."""
    raw = meaning
    meta_parts = []
    chem_mode = ""
    chem_group = ""
    chem_bond = ""
    bio_candidates = ""
    bio_theme = ""
    condition_context = ""

    # 1. Extract and strip IR/Raman intensity metadata: (IR vs, Ra w), (Ra m, IR s), etc.
    def _strip_meta(s):
        found = []
        # Parenthetical: (IR vs, Ra w)
        for m in re.finditer(r"\((?:IR|Ra|Raman)\s+[a-z\-]+(?:,\s*(?:IR|Ra|Raman)\s+[a-z\-]+)*\)", s, re.I):
            found.append(m.group(0))
        for f in found:
            s = s.replace(f, "").strip()
        # Leading: "IR vs, Raman w" without parens
        s = re.sub(r"^(?:IR|Ra|Raman)\s+(?:vs|s|m|w|vw|mw|ms|m-s|m-w)\b[,;\s]*", "", s, flags=re.I)
        return s.strip(" ,;"), found

    meaning_clean, meta_found = _strip_meta(meaning)
    meta_parts.extend(meta_found)

    # 2. Strip intensity/shoulder qualifiers
    for pat in [r"\bshoulder\b", r"\bstrong\b", r"\bweak\b", r"\bmedium\b",
                r"\bvery strong\b", r"\bvery weak\b", r"\bm-s\b", r"\bm-w\b"]:
        if re.search(pat, meaning_clean, re.I):
            meta_parts.append(re.search(pat, meaning_clean, re.I).group(0))
            meaning_clean = re.sub(pat, "", meaning_clean, flags=re.I).strip(" ,;")

    # 3. Identify vibrational mode terms
    VIB_MODES = ["stretching", "stretch", "bending", "bend", "deformation", "deform",
                 "breathing", "rocking", "twisting", "wagging", "scissoring",
                 "symmetric", "asymmetric", "antisymmetric", "overtone", "torsion",
                 "in-plane", "out-of-plane", "skeletal", "ring vibrat", "vibration",
                 "valence", "single bond", "double bond", "fermi resonance"]
    mc_low = meaning_clean.lower()
    modes = [v for v in VIB_MODES if v in mc_low]
    if modes:
        chem_mode = "; ".join(modes[:3])

    # 4. Identify functional groups / bonds (expanded)
    FUNC_GROUPS = {"C=O": "carbonyl", "C-H": "C-H", "N-H": "N-H", "O-H": "hydroxyl",
                   "S-S": "disulfide", "S-H": "thiol", "C-N": "C-N", "C-C": "C-C",
                   "C=C": "C=C", "C-O": "C-O", "C-O-C": "ether/glycosidic",
                   "P-O": "phosphate", "O-P-O": "phosphate", "S=O": "sulfonyl",
                   "N-CH3": "N-methyl", "C-S": "C-S", "C-Cl": "C-Cl", "C-Br": "C-Br",
                   "CH2": "methylene", "CH3": "methyl", "NH2": "amine", "NO2": "nitro",
                   "BH2": "borohydride", "C≡": "triple bond",
                   "amide I": "amide I", "amide II": "amide II", "amide III": "amide III",
                   "amid iii": "amide III", "amid ii": "amide II", "amid i": "amide I",
                   "amide": "amide", "carbonyl": "carbonyl", "hydroxyl": "hydroxyl",
                   "carboxyl": "carboxyl", "ester": "ester", "ether": "ether",
                   "phosphodiester": "phosphodiester", "glycosidic": "glycosidic",
                   "disulfide": "disulfide", "thiol": "thiol"}
    found_groups = []
    for bond, label in FUNC_GROUPS.items():
        if bond.lower() in mc_low or bond in meaning_clean:
            if label not in found_groups:
                found_groups.append(label)
    # Also detect ring systems
    RING_SYSTEMS = ["aromatic ring", "phenyl", "pyrrole", "imidazole", "indole",
                    "pyrimidine ring", "purine ring", "benzene", "ring"]
    for rs in RING_SYSTEMS:
        if rs in mc_low and "ring" not in found_groups:
            found_groups.append(rs)
            break
    chem_group = "; ".join(found_groups[:4]) if found_groups else ""

    # Extract bond/moiety from notation: ν(X), δ(X)
    notation_bonds = re.findall(r"[νδρτωγ]\(([^)]+)\)", meaning_clean)
    if notation_bonds:
        chem_bond = "; ".join(notation_bonds[:4])
        if not chem_group:
            chem_group = chem_bond

    # Also extract bonds from C-X patterns in text (e.g., "C−O−C vibration")
    text_bonds = re.findall(r"[A-Z][a-z]?[\-−–][A-Z][a-z]?(?:[\-−–][A-Z][a-z]?)?", meaning_clean)
    if text_bonds and not chem_bond:
        chem_bond = "; ".join(text_bonds[:3])

    # 5. Identify biomolecule candidates (expanded with amino acids, saccharides, lipids)
    BIO_MOLECULES = {
        # Amino acids
        "phenylalanine": "phenylalanine", "tyrosine": "tyrosine", "tryptophan": "tryptophan",
        "proline": "proline", "glycine": "glycine", "alanine": "alanine",
        "valine": "valine", "leucine": "leucine", "isoleucine": "isoleucine",
        "histidine": "histidine", "lysine": "lysine", "arginine": "arginine",
        "methionine": "methionine", "cysteine": "cysteine", "serine": "serine",
        "threonine": "threonine", "asparagine": "asparagine", "aspartate": "aspartate",
        "glutamate": "glutamate", "glutamine": "glutamine", "glutathione": "glutathione",
        "l-serine": "serine", "l-glu": "glutamate",
        # Nucleobases
        "adenine": "adenine", "guanine": "guanine", "cytosine": "cytosine",
        "thymine": "thymine", "uracil": "uracil",
        # Metabolites
        "cholesterol": "cholesterol", "glucose": "glucose", "urea": "urea",
        "creatinine": "creatinine", "uric acid": "uric acid", "bilirubin": "bilirubin",
        "biliverdin": "biliverdin", "dopamine": "dopamine", "serotonin": "serotonin",
        "citrate": "citrate", "lactate": "lactate", "pyruvate": "pyruvate",
        "caffeine": "caffeine", "paraxanthine": "paraxanthine", "ergothioneine": "ergothioneine",
        "dATP": "dATP", "deoxyadenosine": "deoxyadenosine",
        # Proteins
        "collagen": "collagen", "hemoglobin": "hemoglobin", "albumin": "albumin",
        "keratin": "keratin", "fibrinogen": "fibrinogen",
        # Carbohydrates
        "glycogen": "glycogen", "galactose": "galactose", "galactosamine": "galactosamine",
        "mannose": "mannose", "fucose": "fucose", "fructose": "fructose",
        # Lipids
        "phosphatidylcholine": "phosphatidylcholine", "phosphatidylinositol": "phosphatidylinositol",
        "phosphatidylserine": "phosphatidylserine", "sphingomyelin": "sphingomyelin",
        # Bacterial
        "N-acetyl-D-glucosamine": "N-acetyl-D-glucosamine",
        "N-acetylmuramic": "N-acetylmuramic acid",
        "peptidoglycan": "peptidoglycan",
        # Pigments / carotenoids
        "lumichrome": "lumichrome", "riboflavin": "riboflavin", "carotenoid": "carotenoid",
        "carotene": "carotenoid (carotene)", "β-carotene": "carotenoid (β-carotene)",
        "beta-carotene": "carotenoid (β-carotene)",
        "carnitine": "carnitine",
        # Saccharides (additional)
        "ribose": "ribose", "d-mannos": "mannose",
        # Nucleotides
        "adenosine monophosphate": "AMP (nucleotide)",
        "deoxythymidine monophosphate": "dTMP (nucleotide)",
        "amp": "AMP (nucleotide)", "dtmp": "dTMP (nucleotide)",
        "gmp": "GMP (nucleotide)", "cmp": "CMP (nucleotide)",
        "ump": "UMP (nucleotide)",
    }
    found_bio = []
    for mol, label in BIO_MOLECULES.items():
        if mol.lower() in mc_low and label not in found_bio:
            found_bio.append(label)
    # Purine shorthand: "A,G" or "A, G" (adenine/guanine)
    if re.search(r"\bA\s*,\s*G\b", meaning_clean):
        if "adenine" not in found_bio:
            found_bio.append("adenine")
        if "guanine" not in found_bio:
            found_bio.append("guanine")
    bio_candidates = "; ".join(found_bio[:4]) if found_bio else ""

    # 6. Identify biochemical themes (expanded with class-name forms)
    THEMES = {
        # Standard terms
        "protein": "protein", "lipid": "lipid", "nucleic acid": "nucleic acid",
        "carbohydrate": "carbohydrate", "amino acid": "amino acid",
        "purine": "purine metabolite", "pyrimidine": "pyrimidine",
        "peptide": "peptide", "metabolite": "metabolite",
        "cell wall": "cell wall", "membrane": "membrane", "backbone": "backbone",
        # CamelCase / class-name / underscore forms from grounding and curated sources
        "nucleicacid": "nucleic acid", "nucleic_acid": "nucleic acid",
        "aminoacid": "amino acid", "amino_acid": "amino acid",
        "saccharide": "carbohydrate", "monosaccharide": "carbohydrate",
        "polysaccharide": "carbohydrate", "disaccharide": "carbohydrate",
        "dna": "nucleic acid (DNA)", "rna": "nucleic acid (RNA)",
        "nucleotide": "nucleic acid (nucleotide)",
        "monophosphate": "nucleic acid (nucleotide)",
        "phospholipid": "lipid (phospholipid)",
        "fatty": "lipid (fatty acid)", "triglyceride": "lipid (triglyceride)",
        "carotenoid": "lipid-soluble pigment (carotenoid)",
        # Broader themes
        "typical for": "", "belongs to": "",  # these are context markers, not themes themselves
    }
    found_themes = []
    for term, theme in THEMES.items():
        if term in mc_low and theme and theme not in found_themes:
            found_themes.append(theme)
    bio_theme = "; ".join(found_themes[:3]) if found_themes else ""

    # 7. Detect condition/context side-channel phrases
    CONTEXT_PATTERNS = [
        (r"\b(?:BMI|group|cohort|patient|sample|disease|tumor|cancer|stage)\s*[-–]?\s*differentiating\b", "discriminative_context"),
        (r"\bEV-associated\s+spectral\s+region\b", "sample_context"),
        (r"\bserving as (?:an?\s+)?indicator\b", "discriminative_context"),
        (r"\bpoint(?:s|ing)?\s+to\s+(?:higher|lower|increased|decreased|elevated|reduced)\b", "cohort_context"),
        (r"\btypical for\b", "cohort_context"),
        (r"\bwere consistently detected across\b", "sample_context"),
        (r"\bcharacteristic (?:of|for) (?:the\s+)?(?:cancer|tumor|disease|healthy|normal|control)\b", "discriminative_context"),
        (r"\bindicator of\b", "discriminative_context"),
        (r"\b(?:oxidative stress|cellular environment|biological environment)\b", "condition_context"),
    ]
    found_ctx = []
    for pat, ctx_type in CONTEXT_PATTERNS:
        if re.search(pat, meaning_clean, re.I):
            if ctx_type not in found_ctx:
                found_ctx.append(ctx_type)
    condition_context = "; ".join(found_ctx) if found_ctx else ""

    # 8. Determine assignment level
    has_chem = bool(chem_mode or chem_group or chem_bond)
    has_bio = bool(bio_candidates or bio_theme)
    if has_chem and has_bio:
        level = "chemistry_plus_biomolecule"
    elif has_chem:
        level = "chemistry_only"
    elif has_bio:
        level = "theme_only"
    elif condition_context:
        level = "context_only"
    else:
        level = "unresolved"

    return {
        "vibrational_mode": chem_mode,
        "functional_group": chem_group,
        "bond_or_moiety": chem_bond,
        "biomolecule_candidates": bio_candidates,
        "biochemical_theme": bio_theme,
        "condition_context": condition_context,
        "annotation_metadata": "; ".join(meta_parts) if meta_parts else "",
        "assignment_level": level,
        "cleaned_meaning": meaning_clean.strip(),
    }


def run_extraction(text, pid):
    rows,seen=[],set()
    def _add(pk,mn,ev,method):
        if pk<100 or pk>4000: return
        mn=_clean(re.sub(r"\s+"," ",mn).strip(" ,;"))
        ev=re.sub(r"\s+"," ",ev).strip()
        if not mn or len(mn)<3 or _is_fp(ev,pk): return
        key=(round(pk),mn.lower()[:60])
        if key in seen: return
        seen.add(key)
        cl=_classify(mn,ev,method)
        if cl=="reject_noise": return
        split = _split_assignment(mn, method)
        row = {"peak_cm":pk,"meaning":split["cleaned_meaning"][:120],
               "evidence_text":ev[:300],"method":method,"classification":cl}
        row.update(split)
        rows.append(row)
    for m in re.finditer(_PEAK+r"\s*"+_CM1+r"\s*(?:[^.\n]{0,80}?)"+_AVERBS+r"\s+(?P<meaning>[^.;\n]{3,120})",text,re.I):
        _add(_parse_peak(m.group("peak")),m.group("meaning"),m.group(0),"text_assignment")
    for m in re.finditer(_PEAK+r"\s*"+_CM1+r"(?P<tail>.{0,80}?)\.\s+(?:It\s+(?:has\s+been\s+|is\s+|was\s+)|This\s+(?:band\s+|peak\s+|feature\s+)?(?:has\s+been\s+|is\s+|was\s+)?)?"+_AVERBS+r"\s+(?P<meaning>[^.;]{3,150})",text,re.I):
        _add(_parse_peak(m.group("peak")),m.group("meaning"),m.group(0),"text_assignment_cross")
    for m in re.finditer(_AVERBS+r"\s+(?P<meaning>[^.;]{3,100}?)(?:\bat\s+)?"+_PEAK+r"\s*"+_CM1,text,re.I):
        _add(_parse_peak(m.group("peak")),m.group("meaning"),m.group(0),"text_assignment_rev")
    notation_peaks=set()
    _VC=r"[\u03BD\u03B4\u03C1\u03C4\u03C9\u03B3\u03C7v\u0076]"
    _VT=_VC+r"\([^)]{1,30}\)"; _VM=_VT+r"(?:\s*\+\s*"+_VT+r")*"
    _IPR=r"i\.\s*p\.\s*[a-z]+\s+ring\s+(?:breathing|deformation|stretch)\w*(?:\s*\+\s*"+_VT+r")?"
    for m in re.finditer(r"(?P<notation>"+_VM+r"|"+_IPR+r")\s*[\u2013\u2212\-]+\s*"+_PEAK+r"\s*"+_CM1,text,re.I):
        pk=_parse_peak(m.group("peak"))
        _add(pk,re.sub(r"\s+"," ",m.group("notation")).strip(),m.group(0),"notation_assignment")
        notation_peaks.add(round(pk))
    for line in text.splitlines():
        c=re.sub(r"\s+"," ",line).strip()
        m=re.match(r"^(?P<peak>\d{3,4}(?:\.\d+)?)\s+(?P<meaning>[A-Za-z][A-Za-z0-9 ,()=/\-\u2013\u2014]{3,80})$",c)
        if not m: continue
        pk=float(m.group("peak")); mn=m.group("meaning").strip(" ,;")
        if len(mn.split())>12 or re.search(r"\d{2,}",mn): continue
        _add(pk,mn,c,"table_assignment")
    for m in re.finditer(_PEAK+r"\s*"+_CM1+r"(?P<context>[^.;\n]{0,120})",text,re.I):
        pk=_parse_peak(m.group("peak"))
        if round(pk) in notation_peaks: continue
        ctx=_clean(re.sub(r"\s+"," ",m.group("context")).strip(" ,;"))
        if not ctx or len(ctx)<3 or _is_fp(ctx,pk): continue
        key=(round(pk),ctx.lower()[:60])
        if key in seen: continue
        if any(v in ctx.lower() for v in ASSIGNMENT_VERBS): continue
        if not any(t in ctx.lower() for t in SUBFAMILY_HINT_TERMS): continue
        seen.add(key)
        cl=_classify(ctx,ctx,"text_regex")
        if cl!="reject_noise":
            split=_split_assignment(ctx,"text_regex")
            row={"peak_cm":pk,"meaning":split["cleaned_meaning"][:120],"evidence_text":ctx[:300],"method":"text_regex","classification":cl}
            row.update(split)
            rows.append(row)
    return rows

_METHOD_LABELS = {"notation_assignment":"Notation","text_assignment":"Direct",
    "text_assignment_cross":"Cross-sent.","text_assignment_rev":"Reverse",
    "table_assignment":"Table","text_regex":"Contextual"}
_METHOD_RANK = {"notation_assignment":1,"text_assignment":2,"text_assignment_cross":3,
    "text_assignment_rev":4,"table_assignment":5,"text_regex":6}
_STATUS_COLORS = {"rescued_and_ingested":"#1b7f3a","extracted_high_signal":"#1565c0",
    "extracted_low_signal":"#6d4c9e","rescued_local_asset_present":"#e65100",
    "manual_failed":"#b3261e","deferred":"#9e6b14"}

def _badge(s):
    c=_STATUS_COLORS.get(s,"#6b7280"); l=s.replace("_"," ").title()
    return f"<span style='display:inline-block;padding:0.2rem 0.6rem;border-radius:999px;background:{c};color:white;font-size:0.82rem;font-weight:600'>{l}</span>"

# ══════════════════════════════════════════════════════════
# APP
# ══════════════════════════════════════════════════════════
st.set_page_config(page_title="GAIRA Rescue Ops", layout="wide")
st.title("GAIRA Rescue Ops Console")

t1,t2,t3,t4,t5 = st.tabs(["Blocked Registry","Folder Scan","Extract + Review","Pipeline","Evidence Viewer"])

# ══════════════════════════════════════════════════════════
# TAB 1 — BLOCKED REGISTRY BROWSER (with checkboxes + working set)
# ══════════════════════════════════════════════════════════
with t1:
    all_df = load_blocked()
    if all_df.empty: st.warning("No blocked registry."); st.stop()

    # ── Global counts (always from full registry, not filtered) ──
    g1,g2,g3,g4,g5 = st.columns(5)
    g1.metric("Total", len(all_df))
    g2.metric("Critical A", len(all_df[all_df["tier"]=="critical_A"]))
    g3.metric("Critical B", len(all_df[all_df["tier"]=="critical_B"]))
    g4.metric("Ingested", len(all_df[all_df["access_status"]=="rescued_and_ingested"]))
    g5.metric("Remaining", len(all_df[~all_df["access_status"].isin(HIDDEN)]))

    # ── Visibility toggle ──
    show_done = st.toggle("Show completed / ingested", value=False, key="t1_done")
    df = all_df if show_done else all_df[~all_df["access_status"].isin(HIDDEN)]

    st.subheader(f"Queue ({len(df)} papers)")

    # ── Filters ──
    c1,c2,c3,c4 = st.columns(4)
    with c1: tf=st.multiselect("Tier",[t for t in TIER_ORDER if t in df["tier"].unique()],default=[t for t in ["critical_A","critical_B"] if t in df["tier"].unique()])
    with c2: pf=st.multiselect("Publisher",sorted(df["publisher"].unique()))
    with c3: mf=st.multiselect("Modality",sorted(df["modality"].unique()))
    with c4: af=st.multiselect("Access Status",sorted(df["access_status"].unique()))
    filt=df.copy()
    if tf: filt=filt[filt["tier"].isin(tf)]
    if pf: filt=filt[filt["publisher"].isin(pf)]
    if mf: filt=filt[filt["modality"].isin(mf)]
    if af: filt=filt[filt["access_status"].isin(af)]

    st.caption(f"Showing {len(filt)} of {len(df)} papers after filters")

    # ── Checkbox table ──
    show_cols=["paper_id","title","publisher","tier","modality","biosample",
               "condition","expected_AGV_yield","access_status"]
    show_cols=[c for c in show_cols if c in filt.columns]
    tbl = filt[show_cols].reset_index(drop=True).copy()
    tbl.insert(0, "select", False)

    edited = st.data_editor(tbl, use_container_width=True, height=450,
        column_config={
            "select": st.column_config.CheckboxColumn("Sel", width="small", default=False),
            "title": st.column_config.TextColumn("Title", width="large"),
            "tier": st.column_config.TextColumn("Tier", width="small"),
        },
        disabled=[c for c in show_cols], key="t1_editor")

    selected = list(edited.loc[edited["select"], "paper_id"])

    # ── Working set controls ──
    st.divider()
    w1,w2,w3 = st.columns(3)
    ws = st.session_state.get("working_set", [])
    with w1:
        if st.button(f"Set Working Set ({len(selected)})", disabled=len(selected)==0):
            st.session_state["working_set"] = selected
            st.success(f"Working set: {len(selected)} papers")
            st.rerun()
    with w2:
        st.markdown(f"**Working set: {len(ws)} papers**")
    with w3:
        if st.button("Clear Working Set", disabled=len(ws)==0):
            st.session_state["working_set"] = []
            st.rerun()

# ══════════════════════════════════════════════════════════
# TAB 2 — FOLDER SCAN + MATCH
# ══════════════════════════════════════════════════════════
with t2:
    st.subheader("Scan Folder for Downloaded Papers")
    folder = st.text_input("Folder path", value=str(Path.home()/"Downloads"), key="scan_folder")
    if st.button("Scan", key="scan_btn"):
        fp = Path(folder)
        if not fp.is_dir():
            st.error(f"Not a directory: {folder}")
        else:
            files = sorted(fp.rglob("*.pdf")) + sorted(fp.rglob("*.txt"))
            st.info(f"Found {len(files)} PDF/TXT files")
            if files:
                df = load_blocked()
                # Try matching each file to blocked papers
                matches = []
                for fpath in files:
                    fname = fpath.stem.lower()
                    best_pid, best_score, best_title = "", 0.0, ""
                    for _, row in df.iterrows():
                        pid = row["paper_id"]
                        doi = (row.get("doi","") or "").replace("/","_").replace(".","_").lower()
                        title = (row.get("title","") or "").lower()
                        # DOI match
                        if doi and doi[:15] in fname:
                            best_pid, best_score, best_title = pid, 1.0, row.get("title","")
                            break
                        # Title similarity
                        sim = SequenceMatcher(None, fname[:40], title[:40]).ratio()
                        if sim > best_score:
                            best_pid, best_score, best_title = pid, sim, row.get("title","")
                    status = "matched" if best_score >= 0.5 else "unresolved"
                    matches.append({"file": fpath.name, "path": str(fpath),
                                    "matched_paper": best_pid if status == "matched" else "",
                                    "confidence": round(best_score, 2),
                                    "matched_title": best_title[:60] if status == "matched" else "",
                                    "status": status})

                mdf = pd.DataFrame(matches)
                matched = mdf[mdf["status"] == "matched"]
                unresolved = mdf[mdf["status"] == "unresolved"]

                st.metric("Matched", len(matched))
                st.metric("Unresolved", len(unresolved))

                if not matched.empty:
                    st.subheader("Matched Files")
                    st.dataframe(matched[["file","matched_paper","confidence","matched_title"]].reset_index(drop=True),
                                 use_container_width=True)

                if not unresolved.empty:
                    st.subheader("Unresolved Files")
                    st.dataframe(unresolved[["file","confidence"]].reset_index(drop=True),
                                 use_container_width=True)

                    # Manual assignment
                    st.subheader("Manual Assignment")
                    for _, row in unresolved.iterrows():
                        with st.expander(row["file"]):
                            assign_to = st.selectbox(f"Assign to paper", ["(skip)"] + df["paper_id"].tolist(),
                                                     key=f"assign_{row['file']}")
                            if assign_to != "(skip)" and st.button(f"Stage {row['file']}", key=f"stage_manual_{row['file']}"):
                                target = STAGING / assign_to
                                target.mkdir(parents=True, exist_ok=True)
                                src = Path(row["path"])
                                dst = target / f"{assign_to}_manuscript{src.suffix}"
                                import shutil
                                shutil.copy2(src, dst)
                                if src.suffix == ".pdf":
                                    txt = _extract_pdf(dst)
                                    if txt:
                                        (target / f"{assign_to}_full_text.txt").write_text(txt)
                                old = _set_status(assign_to, "rescued_local_asset_present")
                                _log(assign_to, "staged", old, "rescued_local_asset_present",
                                     note=f"manual assignment from {row['file']}")
                                st.success(f"Staged {row['file']} → {assign_to}")

                # Auto-stage matched files
                if not matched.empty:
                    if st.button("Stage all matched files", key="stage_all"):
                        import shutil
                        staged = 0
                        for _, row in matched.iterrows():
                            pid = row["matched_paper"]
                            target = STAGING / pid
                            target.mkdir(parents=True, exist_ok=True)
                            src = Path(row["path"])
                            dst = target / f"{pid}_manuscript{src.suffix}"
                            shutil.copy2(src, dst)
                            if src.suffix == ".pdf":
                                txt = _extract_pdf(dst)
                                if txt:
                                    (target / f"{pid}_full_text.txt").write_text(txt)
                            old = _set_status(pid, "rescued_local_asset_present")
                            _log(pid, "staged", old, "rescued_local_asset_present",
                                 note=f"folder scan match from {row['file']}")
                            staged += 1
                        st.success(f"Staged {staged} files. Refresh browser.")

# ══════════════════════════════════════════════════════════
# TAB 3 — EXTRACT + REVIEW + APPROVE
# ══════════════════════════════════════════════════════════
with t3:
    df = load_blocked()
    if df.empty: st.stop()
    ws = st.session_state.get("working_set", [])
    if ws:
        pids = ws
        st.caption(f"Working set: {len(ws)} papers")
    else:
        pids = df["paper_id"].tolist()
    sel = st.selectbox("Select paper", pids, index=0, key="t3_sel")
    paper = df[df["paper_id"]==sel].iloc[0]
    status = paper.get("access_status","unknown")
    is_term = status in TERMINAL
    target = STAGING / sel

    st.markdown(f"**Status:** {_badge(status)}", unsafe_allow_html=True)
    if is_term: st.info(f"Paper is **{status.replace('_',' ')}**.")
    st.subheader(paper.get("title",sel))
    a,b = st.columns(2)
    with a:
        doi=paper.get("doi",""); url=paper.get("canonical_url","")
        st.markdown(f"**DOI:** [{doi}]({url})" if url else f"**DOI:** {doi}")
        st.markdown(f"**Publisher:** {paper.get('publisher','')} | **Tier:** {paper.get('tier','')}")
        st.markdown(f"**Modality:** {paper.get('modality','')} | **Biosample:** {paper.get('biosample','')}")
    with b:
        st.markdown(f"**Condition:** {paper.get('condition','')}")
        st.markdown(f"**Expected yield:** {paper.get('expected_AGV_yield','')}")
        st.markdown(f"**Gap:** {paper.get('gap_filled','')}")
    st.divider()

    # Upload or load staged
    if not is_term:
        ms=st.file_uploader("Manuscript PDF",type=["pdf"],key=f"ms_{sel}")
        si=st.file_uploader("Supplement (opt)",type=["pdf"],key=f"si_{sel}")
        if st.button("Stage",type="primary",key=f"stg_{sel}"):
            if not ms: st.error("Upload a PDF.")
            else:
                target.mkdir(parents=True,exist_ok=True); tc=""
                p=target/f"{sel}_manuscript.pdf"; p.write_bytes(ms.getvalue()); tc=_extract_pdf(p)
                if si: p2=target/f"{sel}_supplement.pdf"; p2.write_bytes(si.getvalue()); tc+="\n\n--- SI ---\n\n"+_extract_pdf(p2)
                if tc: (target/f"{sel}_full_text.txt").write_text(tc)
                wc=len(tc.split()) if tc else 0
                st.success(f"Staged. {wc:,} words.")
                old=_set_status(sel,"rescued_local_asset_present")
                _log(sel,"staged",old,"rescued_local_asset_present",note=f"{wc} words")
                st.session_state[f"txt_{sel}"]=tc
        st.divider()

    # Load from disk
    txt_disk = target/f"{sel}_full_text.txt"
    if txt_disk.exists() and f"txt_{sel}" not in st.session_state:
        st.session_state[f"txt_{sel}"] = txt_disk.read_text()

    # Extract
    has_txt = f"txt_{sel}" in st.session_state
    if has_txt and not is_term:
        st.subheader("Extraction")
        is_ext = status in {"extracted_high_signal","extracted_low_signal"}
        if is_ext: st.warning(f"Already extracted ({status}).")
        if st.button("Force Re-extract" if is_ext else "Run Extraction",key=f"ext_{sel}"):
            text=st.session_state[f"txt_{sel}"]
            results=run_extraction(text,sel)
            agv=[r for r in results if r["classification"] in ("validated_primary","validated_secondary")]
            st.session_state[f"agv_{sel}"]=agv
            n=len(agv); yc="high" if n>=10 else "medium" if n>=3 else "low" if n>=1 else "none"
            ns="extracted_high_signal" if n>=5 else "extracted_low_signal"
            old=_set_status(sel,ns); _log(sel,"extracted",old,ns,agv=n,note=f"{yc} yield")
            if agv:
                ep=target/f"{sel}_extraction.csv"
                with open(ep,"w",newline="") as f:
                    w=csv.DictWriter(f,fieldnames=list(agv[0].keys())); w.writeheader(); w.writerows(agv)

    # Load extraction from disk
    agv = st.session_state.get(f"agv_{sel}")
    if not agv:
        ext_csv = target/f"{sel}_extraction.csv"
        if ext_csv.exists():
            with open(ext_csv) as f:
                agv = list(csv.DictReader(f))
                for r in agv: r["peak_cm"]=float(r["peak_cm"])
            st.session_state[f"agv_{sel}"] = agv

    if agv:
        n=len(agv); yc="high" if n>=10 else "medium" if n>=3 else "low" if n>=1 else "none"
        st.divider()
        st.subheader(f"Review ({n} AGV, {yc})")
        x1,x2,x3,x4=st.columns(4)
        x1.metric("AGV",n); x2.metric("Yield",yc)
        x3.metric("Primary",sum(1 for a in agv if a.get("classification")=="validated_primary"))
        # Count structured vs legacy rows
        has_struct = sum(1 for a in agv if a.get("vibrational_mode") or a.get("functional_group") or a.get("biomolecule_candidates"))
        x4.metric("Structured", f"{has_struct}/{n}")

        rdf=pd.DataFrame(agv)
        rdf["quality"]=rdf["method"].map(_METHOD_LABELS).fillna("Other")
        rdf["_s"]=rdf["method"].map(_METHOD_RANK).fillna(9)
        rdf=rdf.sort_values(["_s","peak_cm"]).drop(columns=["_s"])

        # Ensure structured columns exist (handle legacy CSVs gracefully)
        for col in ["vibrational_mode","functional_group","bond_or_moiety",
                     "biomolecule_candidates","biochemical_theme","annotation_metadata","assignment_level"]:
            if col not in rdf.columns:
                rdf[col] = ""
        rdf = rdf.fillna("")

        # Display structured review table
        display_cols = ["peak_cm","meaning","assignment_level","functional_group",
                        "biomolecule_candidates","biochemical_theme","quality","classification"]
        st.dataframe(rdf[display_cols],use_container_width=True,hide_index=True,
            column_config={
                "peak_cm":st.column_config.NumberColumn("Peak",format="%.1f"),
                "meaning":st.column_config.TextColumn("Meaning",width="medium"),
                "assignment_level":st.column_config.TextColumn("Level",width="small"),
                "functional_group":st.column_config.TextColumn("Chem Group",width="small"),
                "biomolecule_candidates":st.column_config.TextColumn("Biomolecule",width="small"),
                "biochemical_theme":st.column_config.TextColumn("Theme",width="small"),
                "quality":st.column_config.TextColumn("Quality",width="small"),
                "classification":st.column_config.TextColumn("Class",width="small"),
            })

        # Show annotation metadata if any rows have it
        meta_rows = rdf[rdf["annotation_metadata"].astype(str).str.len() > 0]
        if not meta_rows.empty:
            with st.expander(f"Annotation metadata ({len(meta_rows)} rows)"):
                st.dataframe(meta_rows[["peak_cm","annotation_metadata"]].reset_index(drop=True),
                             use_container_width=True, hide_index=True)
        st.divider()
        st.caption("Evidence NOT updated until you click Approve.")
        d1,d2,d3,d4=st.columns(4)
        with d1:
            already=status=="rescued_and_ingested"
            if st.button("Approve + Ingest" if not already else "Already Ingested",type="primary",key=f"app_{sel}",disabled=already):
                old=_set_status(sel,"rescued_and_ingested")
                _log(sel,"approved",old,"rescued_and_ingested",agv=n)
                st.success(f"Approved. Refresh browser.")
        with d2:
            if st.button("Reject",key=f"rej_{sel}"):
                st.session_state.pop(f"agv_{sel}",None)
                old=_set_status(sel,"rescued_local_asset_present")
                _log(sel,"rejected",old,"rescued_local_asset_present",note="user rejected")
                st.warning("Rejected. Refresh.")
        with d3:
            if st.button("SI Follow-up",key=f"si_{sel}_btn"):
                old=_set_status(sel,"needs_si_followup"); _log(sel,"si_followup",old,"needs_si_followup")
                st.info("Marked SI. Refresh.")
        with d4:
            if st.button("Save Later",key=f"later_{sel}"): st.info("Saved.")

    st.divider()
    st.subheader("Quick Actions")
    q1,q2=st.columns(2)
    with q1:
        if st.button("Manual Failed",key=f"fail_{sel}",disabled=is_term):
            old=_set_status(sel,"manual_failed"); _log(sel,"failed",old,"manual_failed"); st.warning("Failed. Refresh.")
    with q2:
        if st.button("Defer",key=f"def_{sel}",disabled=is_term):
            old=_set_status(sel,"deferred"); _log(sel,"deferred",old,"deferred"); st.info("Deferred. Refresh.")

# ══════════════════════════════════════════════════════════
# TAB 4 — PIPELINE REFRESH
# ══════════════════════════════════════════════════════════
with t4:
    st.subheader("Pipeline Refresh")
    st.caption("Recompute neighborhoods, motifs, and condition links from current evidence table.")
    # Show current state from latest refresh files
    for fname, label in [("phaseO3_batch1_refreshed_neighborhoods.csv","Neighborhoods"),
                          ("phaseO3_batch1_refreshed_motifs.csv","Motifs"),
                          ("phaseO3_batch1_refreshed_condition_neighborhood_links.csv","Cond-Nbhd Links"),
                          ("phaseO3_batch1_refreshed_condition_motif_links.csv","Cond-Motif Links")]:
        p = EXT_RUNS / fname
        if not p.exists():
            # Fallback to phaseM1
            p = EXT_RUNS / fname.replace("phaseN3B", "phaseM1")
        if p.exists():
            n = sum(1 for _ in open(p)) - 1
            st.metric(label, n)
    st.info("To refresh, run the pipeline recompute script externally and reload this page.")

# ══════════════════════════════════════════════════════════
# TAB 5 — STRUCTURED EVIDENCE VIEWER
# ══════════════════════════════════════════════════════════
with t5:
    st.subheader("Structured Evidence")
    view = st.radio("View", ["Neighborhoods","Motifs","Condition Links","Structured Evidence"], horizontal=True)

    if view == "Neighborhoods":
        p = EXT_RUNS / "phaseO3_batch1_refreshed_neighborhoods.csv"
        if not p.exists():
            p = EXT_RUNS / "phaseM1_refreshed_neighborhoods.csv"
        if p.exists():
            ndf = pd.read_csv(p)
            st.metric("Total", len(ndf))
            filt_sf = st.multiselect("Subfamily", sorted(ndf["dominant_subfamily"].unique()), key="ev_sf")
            if filt_sf: ndf = ndf[ndf["dominant_subfamily"].isin(filt_sf)]
            st.dataframe(ndf.sort_values("canonical_peak_cm").reset_index(drop=True),
                         use_container_width=True, height=500)
        else: st.info("No neighborhood data. Run pipeline refresh.")

    elif view == "Motifs":
        p = EXT_RUNS / "phaseO3_batch1_refreshed_motifs.csv"
        if not p.exists():
            p = EXT_RUNS / "phaseM1_refreshed_motifs.csv"
        if p.exists():
            mdf = pd.read_csv(p)
            st.metric("Total", len(mdf))
            st.dataframe(mdf.reset_index(drop=True), use_container_width=True, height=400)
        else: st.info("No motif data.")

    elif view == "Condition Links":
        for fname, label in [("phaseO3_batch1_refreshed_condition_neighborhood_links.csv","Condition-Neighborhood"),
                              ("phaseO3_batch1_refreshed_condition_motif_links.csv","Condition-Motif")]:
            p = EXT_RUNS / fname
            if not p.exists():
                p = EXT_RUNS / fname.replace("phaseN3B", "phaseM1")
            if p.exists():
                cdf = pd.read_csv(p)
                st.subheader(label)
                st.metric("Links", len(cdf))
                st.dataframe(cdf.reset_index(drop=True), use_container_width=True, height=300)

    elif view == "Structured Evidence":
        # Load the N1 structured adjunct if available
        struct_path = EXT_RUNS / "phaseN1_active_evidence_update.csv"
        if struct_path.exists():
            sdf = pd.read_csv(struct_path, dtype=str).fillna("")
            sdf["peak_cm"] = pd.to_numeric(sdf["peak_cm"], errors="coerce")
            st.metric("Total structured rows", len(sdf))

            # Filters
            fc1, fc2 = st.columns(2)
            with fc1:
                lvl_filter = st.multiselect("Assignment Level",
                    sorted(sdf["assignment_level"].unique()), key="ev_lvl")
            with fc2:
                theme_filter = st.multiselect("Biochemical Theme",
                    sorted([t for t in sdf["biochemical_theme"].unique() if t]), key="ev_theme")

            if lvl_filter: sdf = sdf[sdf["assignment_level"].isin(lvl_filter)]
            if theme_filter: sdf = sdf[sdf["biochemical_theme"].isin(theme_filter)]

            show = ["peak_cm","cleaned_meaning","assignment_level","functional_group",
                    "biomolecule_candidates","biochemical_theme","condition_context",
                    "vibrational_mode","source_id"]
            show = [c for c in show if c in sdf.columns]
            st.dataframe(sdf[show].sort_values("peak_cm").reset_index(drop=True),
                         use_container_width=True, height=500, hide_index=True,
                         column_config={
                             "peak_cm": st.column_config.NumberColumn("Peak", format="%.1f"),
                             "cleaned_meaning": st.column_config.TextColumn("Meaning", width="medium"),
                         })
        else:
            st.info("No structured evidence adjunct. Run phaseN1 pipeline.")
