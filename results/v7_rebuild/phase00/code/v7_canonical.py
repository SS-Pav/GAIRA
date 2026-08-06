"""GAIRA V7 Phase 00 — canonical molecule identity.

V7's statistical unit is the canonical molecule, so identity resolution is not
housekeeping: it is the foundation of the whole balancing argument, and it is the
primary defence against cross-validation leakage (risk R-09).

Design decision, stated once because it governs everything downstream:

    Canonical identity is a METADATA LAYER, not a corpus edit.

The fitting corpus stays at 375 spectra over 167 surface analytes, so the frozen atlas
reproduces bit-exactly. The canonical table adds a `canonical_id` column collapsing
surface forms onto molecules, and **cross-validation groups by `canonical_id`**. That
gives leakage protection without redefining the corpus.

Merge policy
  1. Mechanical  — Unicode NFKC + whitespace + case. This is what collapses the U+FB02
                   ligature in `riboﬂavin`; without it the corpus holds two riboflavins.
  2. Declared    — the 13 duplicate pairs established by the V6.3 ontology revalidation,
                   each re-audited here with a chemical justification.
  3. Protected   — enantiomers and anomers are NEVER merged, whatever the string
                   similarity. `(+)-arabinose` and `(-)-arabinose` are different molecules.
  4. Flagged     — near-misses that are neither clearly the same nor clearly different are
                   recorded as UNRESOLVED with a recommendation, never merged silently.
"""
from __future__ import annotations

import re
from collections import defaultdict

import pandas as pd

from v7_corpus import nfkc

# ── 2. Declared merges (V6.3 `duplicate_of`, re-audited) ──────────────────────
# surface_form -> (canonical_target, merge_class, justification)
DECLARED_MERGES: dict[str, tuple[str, str, str]] = {
    "riboﬂavin": ("riboflavin", "unicode_ligature",
                  "U+FB02 'ﬂ' ligature; NFKC-identical to riboflavin. Same molecule."),
    "ure": ("urea", "truncation",
            "Truncated spreadsheet header for urea."),
    "alb": ("albumin", "truncation",
            "Truncated spreadsheet header for albumin."),
    "gluth": ("glutathione", "truncation",
              "Truncated spreadsheet header for glutathione."),
    "acetyl coenzyme a": ("acetyl-coa", "orthographic",
                          "Same molecule written long-form vs hyphenated abbreviation. "
                          "ALSO a class-assignment conflict: filed under 'protein' in one "
                          "source and 'cofactor' in the other; the merge resolves it."),
    "aspartic acid": ("aspartate", "protonation_state",
                      "Conjugate acid/base pair of one molecule; indistinguishable as a "
                      "solid-state Raman reference at this resolution. ALSO a class "
                      "conflict: 'organic_acid' vs 'amino_acid'."),
    "n-acetyl- d-glucosamine": ("n-acetylglucosamine", "orthographic",
                                "Stray space and D- prefix; same molecule."),
    "(+)-dextrose": ("(+)-glucose", "synonym",
                     "Dextrose is the common name for D-(+)-glucose."),
    "glucose": ("(+)-glucose", "stereo_prefix_generic",
                "Unprefixed 'glucose' in these reference libraries denotes the natural "
                "D-(+) form. Merged; the distinct anomer β-D-glucose is NOT merged."),
    "fructose": ("(-)-fructose", "stereo_prefix_generic",
                 "Unprefixed 'fructose' denotes the natural D-(−) form."),
    "galactose": ("(+)-galactose", "stereo_prefix_generic",
                  "Unprefixed 'galactose' denotes the natural D-(+) form."),
    "mannose": ("(+)-mannose", "stereo_prefix_generic",
                "Unprefixed 'mannose' denotes the natural D-(+) form."),
    "lactose": ("(+)-lactose", "stereo_prefix_generic",
                "Unprefixed 'lactose' denotes the natural (+) form; both from the same "
                "source library at different excitations."),
}

# ── 3. Protected: never merged regardless of string similarity ────────────────
PROTECTED_DISTINCT: list[tuple[str, str, str]] = [
    ("(+)-arabinose", "(-)-arabinose",
     "Enantiomers. Distinct molecules; D- and L-arabinose are not interchangeable."),
    ("(+)-glucose", "β-d-glucose",
     "Anomers. β-D-glucose is a distinct anomeric form with its own reference spectrum."),
    ("(-)-ribose", "2-deoxy-d-ribose",
     "Ribose vs 2-deoxyribose — different molecules (the 2'-OH is the RNA/DNA distinction)."),
]

# ── 4. Flagged near-misses: recorded, NOT merged ──────────────────────────────
FLAGGED_UNRESOLVED: list[tuple[str, str, str, str]] = [
    ("carotene", "β-carotene",
     "Loose-key collision across two sources; both filed as chromophore_pigment. "
     "'Carotene' most likely denotes the β isomer (the common commercial form), but the "
     "source spreadsheet does not say so, and α-carotene is a real alternative.",
     "NOT MERGED. Recommend resolving from the source datasheet before Phase 02; if they "
     "are the same molecule this is one leaked spectrum, which is a small but real "
     "inflation of any within-carotenoid result."),
]


def loose_key(s: str) -> str:
    """Aggressive key for near-miss detection only — never used to merge automatically."""
    return re.sub(r"[^a-z0-9]", "", nfkc(s))


def build_canonical_table(meta: pd.DataFrame,
                          family_of: dict[str, str] | None = None) -> pd.DataFrame:
    """One row per canonical molecule, with every observed surface form mapped to it."""
    surface = sorted(meta.analyte.unique().tolist())

    # step 1 — mechanical NFKC normalisation
    nfkc_map = {a: nfkc(a) for a in surface}
    by_nfkc: dict[str, list[str]] = defaultdict(list)
    for a, n in nfkc_map.items():
        by_nfkc[n].append(a)

    # step 2 — resolve each surface form to a canonical target, then take the closure
    def target(a: str) -> str:
        seen = set()
        cur = a
        while cur in DECLARED_MERGES and cur not in seen:
            seen.add(cur)
            cur = DECLARED_MERGES[cur][0]
        return cur

    # NFKC-equal forms collapse onto the lexicographically first, then merges apply
    canon_of: dict[str, str] = {}
    for n, forms in by_nfkc.items():
        rep = sorted(forms)[0]
        for f in forms:
            canon_of[f] = rep
    canon_of = {a: target(canon_of[a]) for a in surface}
    # a merge target may itself be an NFKC alias; resolve once more
    canon_of = {a: canon_of.get(c, c) for a, c in canon_of.items()}

    groups: dict[str, list[str]] = defaultdict(list)
    for a, c in canon_of.items():
        groups[c].append(a)

    rows = []
    for cid in sorted(groups):
        forms = sorted(groups[cid])
        sub = meta[meta.analyte.isin(forms)]
        merge_kinds = sorted({DECLARED_MERGES[f][1] for f in forms if f in DECLARED_MERGES}
                             | ({"unicode_nfkc"} if len(set(nfkc_map[f] for f in forms)) <
                                len(forms) else set()))
        rows.append({
            "canonical_id": cid,
            "canonical_name": nfkc(cid),
            "n_surface_forms": len(forms),
            "surface_forms": ";".join(forms),
            "aliases": ";".join(f for f in forms if f != cid),
            "merge_kinds": ";".join(merge_kinds),
            "chemical_class": (family_of or {}).get(cid, ""),
            "n_spectra": int(len(sub)),
            "n_sources": int(sub.source.nunique()),
            "sources": ";".join(sorted(sub.source.unique().tolist())),
            "n_excitations": int(sub.excitation_nm.nunique()),
            "excitations": ";".join(str(x) for x in sorted(sub.excitation_nm.unique().tolist())),
            "cross_source_merge": bool(len(forms) > 1 and sub.source.nunique() > 1),
        })
    return pd.DataFrame(rows).sort_values("canonical_id").reset_index(drop=True)


def alias_table(meta: pd.DataFrame, canon: pd.DataFrame) -> pd.DataFrame:
    """Every observed surface form, its canonical ID, and why it maps there."""
    to_cid = {}
    for _, r in canon.iterrows():
        for f in r.surface_forms.split(";"):
            to_cid[f] = r.canonical_id
    rows = []
    for a in sorted(meta.analyte.unique().tolist()):
        cid = to_cid[a]
        if a == cid:
            kind, why = "canonical", "Canonical form."
        elif a in DECLARED_MERGES:
            kind, why = DECLARED_MERGES[a][1], DECLARED_MERGES[a][2]
        elif nfkc(a) == nfkc(cid):
            kind, why = "unicode_nfkc", "NFKC-identical to the canonical form."
        else:
            kind, why = "unresolved", "Mapped by NFKC grouping."
        sub = meta[meta.analyte == a]
        rows.append({"surface_form": a, "canonical_id": cid, "is_alias": a != cid,
                     "merge_kind": kind, "justification": why,
                     "n_spectra": int(len(sub)),
                     "sources": ";".join(sorted(sub.source.unique().tolist()))})
    return pd.DataFrame(rows)


def near_miss_audit(meta: pd.DataFrame, canon: pd.DataFrame) -> pd.DataFrame:
    """Every string-similar pair that was NOT merged, with the decision and its reason.

    Every non-merge is an explicit decision, recorded so a reader can disagree with it.
    """
    to_cid = {}
    for _, r in canon.iterrows():
        for f in r.surface_forms.split(";"):
            to_cid[f] = r.canonical_id

    surface = sorted(meta.analyte.unique().tolist())
    by_key: dict[str, list[str]] = defaultdict(list)
    for a in surface:
        by_key[loose_key(a)].append(a)

    protected = {tuple(sorted((a, b))): why for a, b, why in PROTECTED_DISTINCT}
    flagged = {tuple(sorted((a, b))): (why, rec) for a, b, why, rec in FLAGGED_UNRESOLVED}

    rows = []
    for key, forms in sorted(by_key.items()):
        if len(forms) < 2:
            continue
        for i in range(len(forms)):
            for j in range(i + 1, len(forms)):
                a, b = forms[i], forms[j]
                pair = tuple(sorted((a, b)))
                merged = to_cid[a] == to_cid[b]
                if merged:
                    decision, reason = "MERGED", "Resolved to one canonical ID."
                elif pair in protected:
                    decision, reason = "NOT_MERGED_PROTECTED", protected[pair]
                elif pair in flagged:
                    decision, reason = "NOT_MERGED_UNRESOLVED", flagged[pair][0]
                else:
                    decision, reason = "NOT_MERGED", "Distinct canonical IDs; no merge declared."
                rows.append({"form_a": a, "form_b": b, "loose_key": key,
                             "canonical_a": to_cid[a], "canonical_b": to_cid[b],
                             "decision": decision, "reason": reason})

    # protected / flagged pairs that the loose key did not catch are still recorded
    seen = {tuple(sorted((r["form_a"], r["form_b"]))) for r in rows}
    for a, b, why in PROTECTED_DISTINCT:
        if tuple(sorted((a, b))) not in seen and a in to_cid and b in to_cid:
            rows.append({"form_a": a, "form_b": b, "loose_key": "",
                         "canonical_a": to_cid[a], "canonical_b": to_cid[b],
                         "decision": "NOT_MERGED_PROTECTED", "reason": why})
    for a, b, why, _ in FLAGGED_UNRESOLVED:
        if tuple(sorted((a, b))) not in seen and a in to_cid and b in to_cid:
            rows.append({"form_a": a, "form_b": b, "loose_key": "",
                         "canonical_a": to_cid[a], "canonical_b": to_cid[b],
                         "decision": "NOT_MERGED_UNRESOLVED", "reason": why})
    return pd.DataFrame(rows).sort_values(["decision", "form_a"]).reset_index(drop=True)


def leakage_report(meta: pd.DataFrame, canon: pd.DataFrame) -> dict:
    """How much cross-source leakage the canonicalisation actually removed."""
    multi = canon[canon.n_surface_forms > 1]
    cross = multi[multi.cross_source_merge]
    return {
        "n_surface_analytes": int(meta.analyte.nunique()),
        "n_canonical_ids": int(len(canon)),
        "n_merged_surface_forms": int(meta.analyte.nunique() - len(canon)),
        "n_canonical_ids_with_aliases": int(len(multi)),
        "n_cross_source_merges": int(len(cross)),
        "cross_source_merged_ids": sorted(cross.canonical_id.tolist()),
        "spectra_affected_by_cross_source_merge": int(cross.n_spectra.sum()),
        "interpretation": (
            "Each cross-source merge is a molecule that appeared in two reference libraries "
            "under two spellings. Under naive surface-name grouping those spectra land in "
            "different CV folds and the same molecule is scored against itself."
        ),
    }
