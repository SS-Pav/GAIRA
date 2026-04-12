"""GAIRA Scoring Layer v1.7 — motif differential + cross-study signal stability.

No LLM. Deterministic. Adds:
- Motif-level differential scoring (not just theme-level)
- Per-study direction tracking
- Cross-study stability metrics
- Stability labels (STABLE / MIXED / UNSTABLE / INSUFFICIENT)
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from graph.phaseC1_query_engine import GraphResult


_GENERIC_HUBS = {"stretch", "stretching", "vibration", "breathing", "deformation", "ring"}
_BROAD_THEMES = {"protein", "lipid", "nucleic acid", "carbohydrate", "amino acid", "membrane"}
_GENERIC_DOWNWEIGHT = 0.5
_BROAD_SPECIFICITY_PENALTY = 0.4

_MIN_COMPARATOR_EVIDENCE = 5
_MIN_COMPARATOR_SOURCES = 2
_MIN_THEME_SUPPORT_FOR_ENRICHMENT = 3
_MIN_COMPARATOR_DIRECT_FOR_CONTRAST = 2
_ENRICHMENT_NORM_THRESHOLD = 2.0
_DEPLETION_NORM_THRESHOLD = 0.5
_STABILITY_THRESHOLD_STABLE = 0.7
_STABILITY_THRESHOLD_MIXED = 0.4
_MIN_COMPARABLE_STUDIES = 2


# ── dataclasses ─────────────────────────────────────────────

@dataclass
class ScoredTheme:
    theme: str
    support_score: float
    specificity_score: float
    final_score: float
    direct_count: int
    inferred_count: int
    motif_count: int
    source_diversity: int
    confidence: str
    is_broad: bool = False
    query_direct: int = 0
    query_sources: int = 0
    query_motifs: int = 0
    comp_direct: int = 0
    comp_sources: int = 0
    comp_motifs: int = 0
    enrichment_ratio: float = 0.0
    contrast_score: float = 0.0
    query_direct_norm: float = 0.0
    comp_direct_norm: float = 0.0
    norm_enrichment_ratio: float = 0.0
    evidence_balance: float = 0.0
    interpretation: str = "associated"
    coverage_flag: str = ""


@dataclass
class MotifDifferential:
    """Motif-level differential with cross-study stability."""
    motif_id: str
    subfamily: str
    family: str
    # Counts
    q_direct: int = 0
    q_sources: int = 0
    c_direct: int = 0
    c_sources: int = 0
    # Ratios
    raw_ratio: float = 0.0
    norm_ratio: float = 0.0
    balance: float = 0.0
    # Classification
    interpretation: str = "associated"
    coverage_flag: str = ""
    # Stability (cross-study)
    studies_total: int = 0
    studies_enriched: int = 0
    studies_depleted: int = 0
    studies_shared: int = 0
    studies_non_comparable: int = 0
    directional_agreement: float = 0.0
    stability_label: str = "INSUFFICIENT"


@dataclass
class ScoredMotif:
    motif_id: str
    subfamily: str
    family: str
    member_count: int
    condition_enrichment: str
    comparator_members: int = 0
    motif_interpretation: str = "query-associated"
    coverage_flag: str = ""


@dataclass
class ComparatorSummary:
    query_condition: str
    comparator_condition: str
    inferred_sample_type: str
    query_evidence: int
    query_sources: int
    query_motifs: int
    comparator_evidence: int
    comparator_sources: int
    comparator_adequacy: str
    evidence_balance: float
    scope_mode: str = "broad"
    sample_scope: str = ""
    domain_scope: str = ""
    scope_fallback: str = ""


@dataclass
class ScoredResult:
    query_type: str
    query_mode: str
    matched_entity: str
    comparator: str
    themes: list[ScoredTheme]
    top_biomolecules: list[dict]
    top_functional_groups: list[dict]
    motif_summary: list[ScoredMotif]
    motif_differentials: list[MotifDifferential]   # NEW in C1.7
    evidence_sample: list[dict]
    evidence_count: int
    source_count: int
    caveats: list[str]
    comparator_summary: ComparatorSummary | None = None


# ── helpers ─────────────────────────────────────────────────

def _evidence_balance(a: int, b: int) -> float:
    total = a + b
    return round(2.0 * min(a, b) / total, 3) if total > 0 else 0.0


def _select_quality_evidence(gr: GraphResult, max_rows: int = 8) -> list[dict]:
    candidates = []
    seen = set()
    for r in gr.raw_records:
        eid = r.get("evidence_id")
        if not eid or eid in seen:
            continue
        seen.add(eid)
        meaning = r.get("meaning", "")
        level = r.get("level", "")
        q = (min(len(meaning)/40.0, 2.0) +
             {"chemistry_plus_biomolecule": 3, "theme_only": 2, "chemistry_only": 1.5}.get(level, 0) +
             (-1.0 if len(meaning) < 10 else 0) +
             (-2.0 if any(kw in meaning.lower() for kw in ["classifier","pca","accuracy"]) else 0))
        candidates.append({"evidence_id": eid, "peak_cm": r.get("peak_cm") or r.get("wavenumber"),
                           "meaning": meaning, "level": level, "source": r.get("source",""), "_q": q})
    candidates.sort(key=lambda x: -x["_q"])
    selected = []
    su: dict[str, int] = {}
    for c in candidates:
        if len(selected) >= max_rows:
            break
        s = c.get("source","")
        if su.get(s, 0) >= 2 and len(candidates) > max_rows:
            continue
        su[s] = su.get(s, 0) + 1
        selected.append({k: v for k, v in c.items() if k != "_q"})
    return selected


def _count_theme_data(records):
    data = {}
    for rec in records:
        theme = rec.get("theme") or rec.get("direct_theme")
        if not theme:
            continue
        if theme not in data:
            data[theme] = {"direct": 0, "sources": set(), "motifs": set(), "eids": set()}
        d = data[theme]
        eid = rec.get("evidence_id")
        if eid:
            d["direct"] += 1; d["eids"].add(eid)
        src = rec.get("source")
        if src: d["sources"].add(src)
        mid = rec.get("motif_id")
        if mid: d["motifs"].add(mid)
    return data


def _count_motifs(records):
    return Counter(r.get("motif_id") for r in records if r.get("motif_id"))


def _get_study_id(record: dict) -> str:
    """Extract a study-level identifier. Uses source as the study unit."""
    return record.get("source", "") or record.get("evidence_id", "unknown")


# ── motif differential + stability ──────────────────────────

def _build_motif_differentials(gr: GraphResult) -> list[MotifDifferential]:
    """Build motif-level differentials with per-study stability analysis."""
    if gr.query_mode not in ("pairwise", "one_vs_rest"):
        return []

    total_q = gr.total_evidence_rows or 1
    total_c = gr.comparator_evidence_count or 1

    # Group by motif: query side
    q_motif_data: dict[str, dict] = defaultdict(lambda: {
        "direct": 0, "sources": set(), "by_study": defaultdict(int),
        "subfamily": "", "family": "",
    })
    for rec in gr.raw_records:
        mid = rec.get("motif_id")
        if not mid:
            continue
        d = q_motif_data[mid]
        d["direct"] += 1
        src = rec.get("source", "")
        if src:
            d["sources"].add(src)
        study = _get_study_id(rec)
        d["by_study"][study] += 1
        if not d["subfamily"]:
            d["subfamily"] = rec.get("motif_subfamily", "")
            d["family"] = rec.get("motif_family", "")

    # Group by motif: comparator side
    c_motif_data: dict[str, dict] = defaultdict(lambda: {
        "direct": 0, "sources": set(), "by_study": defaultdict(int),
    })
    for rec in gr.comparator_records:
        mid = rec.get("motif_id")
        if not mid:
            continue
        d = c_motif_data[mid]
        d["direct"] += 1
        src = rec.get("source", "")
        if src:
            d["sources"].add(src)
        study = _get_study_id(rec)
        d["by_study"][study] += 1

    # All motif IDs
    all_motifs = sorted(set(list(q_motif_data.keys()) + list(c_motif_data.keys())),
                        key=lambda m: -q_motif_data[m]["direct"] if m in q_motif_data else 0)

    results = []
    for mid in all_motifs[:20]:  # top 20
        qd = q_motif_data.get(mid, {"direct": 0, "sources": set(), "by_study": defaultdict(int),
                                      "subfamily": "", "family": ""})
        cd = c_motif_data.get(mid, {"direct": 0, "sources": set(), "by_study": defaultdict(int)})

        q_dir = qd["direct"]
        q_src = len(qd["sources"])
        c_dir = cd["direct"]
        c_src = len(cd["sources"])

        # Ratios
        raw_ratio = round((q_dir + 1) / (c_dir + 1), 2)
        q_norm = q_dir / total_q
        c_norm = c_dir / total_c if total_c > 0 else 0.0
        norm_ratio = round(q_norm / max(c_norm, 0.0001), 2)
        balance = _evidence_balance(q_dir, c_dir)

        # Classification
        flag = ""
        if q_dir < _MIN_THEME_SUPPORT_FOR_ENRICHMENT:
            interp = "associated"; flag = "low_query_support"
        elif c_dir == 0:
            interp = "not_comparable"; flag = "comparator_absent"
        elif c_dir < _MIN_COMPARATOR_DIRECT_FOR_CONTRAST:
            interp = "associated"; flag = "insufficient_comparator"
        elif norm_ratio >= _ENRICHMENT_NORM_THRESHOLD and raw_ratio >= 1.5:
            interp = "enriched"
        elif norm_ratio <= _DEPLETION_NORM_THRESHOLD and raw_ratio <= 0.67:
            interp = "depleted"
        elif 0.7 <= norm_ratio <= 1.4:
            interp = "shared"
        else:
            interp = "associated"

        # ── STABILITY: per-study direction tracking ─────────
        all_studies = set(list(qd["by_study"].keys()) + list(cd["by_study"].keys()))
        s_enriched = 0; s_depleted = 0; s_shared = 0; s_non_comp = 0

        for study in all_studies:
            sq = qd["by_study"].get(study, 0)
            sc = cd["by_study"].get(study, 0)

            if sc == 0 and sq == 0:
                s_non_comp += 1
            elif sc == 0:
                s_non_comp += 1  # cannot determine direction without comparator
            elif sq == 0:
                s_depleted += 1
            else:
                study_ratio = sq / sc
                if study_ratio >= 2.0:
                    s_enriched += 1
                elif study_ratio <= 0.5:
                    s_depleted += 1
                else:
                    s_shared += 1

        comparable = len(all_studies) - s_non_comp
        if comparable >= _MIN_COMPARABLE_STUDIES:
            agreement = round(max(s_enriched, s_depleted, s_shared) / comparable, 2)
            if agreement >= _STABILITY_THRESHOLD_STABLE:
                stab_label = "STABLE"
            elif agreement >= _STABILITY_THRESHOLD_MIXED:
                stab_label = "MIXED"
            else:
                stab_label = "UNSTABLE"
        else:
            agreement = 0.0
            stab_label = "INSUFFICIENT"

        results.append(MotifDifferential(
            motif_id=mid, subfamily=qd["subfamily"], family=qd["family"],
            q_direct=q_dir, q_sources=q_src, c_direct=c_dir, c_sources=c_src,
            raw_ratio=raw_ratio, norm_ratio=norm_ratio, balance=balance,
            interpretation=interp, coverage_flag=flag,
            studies_total=len(all_studies), studies_enriched=s_enriched,
            studies_depleted=s_depleted, studies_shared=s_shared,
            studies_non_comparable=s_non_comp,
            directional_agreement=agreement, stability_label=stab_label,
        ))

    results.sort(key=lambda x: (-x.q_direct, -x.directional_agreement))
    return results


# ── main scoring ───────────────────────────────────────────

def score_result(gr: GraphResult) -> ScoredResult:
    is_comp = gr.query_mode in ("pairwise", "one_vs_rest") and gr.comparator_records

    q_themes = _count_theme_data(gr.raw_records)
    c_themes = _count_theme_data(gr.comparator_records) if is_comp else {}
    inferred_counts = Counter(r.get("inferred_theme") for r in gr.raw_records if r.get("inferred_theme"))

    total_q_ev = gr.total_evidence_rows or 1
    total_c_ev = gr.comparator_evidence_count or 1
    total_th = len(q_themes) or 1

    comp_adequacy = "adequate"
    if is_comp:
        if gr.comparator_evidence_count < 2: comp_adequacy = "very_sparse"
        elif gr.comparator_evidence_count < _MIN_COMPARATOR_EVIDENCE: comp_adequacy = "sparse"
        elif gr.comparator_source_count < _MIN_COMPARATOR_SOURCES: comp_adequacy = "sparse"

    global_balance = _evidence_balance(total_q_ev, gr.comparator_evidence_count) if is_comp else 0.0

    scored_themes = []
    for theme, td in q_themes.items():
        inf = inferred_counts.get(theme, 0)
        support = td["direct"]*3.0 + inf*1.5 + len(td["motifs"])*2.0 + min(len(td["sources"]),10)
        ev_share = len(td["eids"]) / total_q_ev
        spec_raw = ev_share / max(1.0/total_th, 0.01)
        is_broad = theme.lower().split("(")[0].strip() in _BROAD_THEMES
        if is_broad: spec_raw *= _BROAD_SPECIFICITY_PENALTY
        specificity = round(min(spec_raw, 5.0), 2)
        final = round(math.sqrt(support * max(specificity, 0.1)), 1)
        if theme.lower() in _GENERIC_HUBS: final *= _GENERIC_DOWNWEIGHT
        conf = ("high" if td["direct"]>=10 and len(td["sources"])>=3 and specificity>=0.5
                else ("medium" if td["direct"]>=3 or len(td["sources"])>=2 else "low"))

        q_dir=td["direct"]; q_src=len(td["sources"]); q_mot=len(td["motifs"])
        c_dir=0;c_src=0;c_mot=0; raw_e=0.0;contrast=0.0;q_n=0.0;c_n=0.0;ne=0.0;bal=0.0
        interp="associated"; cov=""

        if is_comp:
            ct=c_themes.get(theme,{})
            c_dir=ct.get("direct",0) if ct else 0
            c_src=len(ct.get("sources",set())) if ct else 0
            c_mot=len(ct.get("motifs",set())) if ct else 0
            raw_e=round((q_dir+1)/(c_dir+1),2); contrast=round(support-c_dir*3.0,1)
            q_n=round(q_dir/total_q_ev,4); c_n=round(c_dir/total_c_ev,4) if total_c_ev>0 else 0.0
            ne=round(q_n/max(c_n,0.0001),2); bal=_evidence_balance(q_dir,c_dir)

            if q_dir<_MIN_THEME_SUPPORT_FOR_ENRICHMENT: interp="associated"; cov="low_query_support"
            elif c_dir==0: interp="associated"; cov="comparator_absent"; conf="low"
            elif c_dir<_MIN_COMPARATOR_DIRECT_FOR_CONTRAST: interp="associated"; cov="insufficient_comparator"
            elif ne>=_ENRICHMENT_NORM_THRESHOLD and raw_e>=1.5: interp="enriched"
            elif ne<=_DEPLETION_NORM_THRESHOLD and raw_e<=0.67: interp="depleted"
            elif 0.7<=ne<=1.4: interp="shared"
            if interp=="enriched": final*=1.3
            elif interp=="depleted": final*=0.5
            if cov in ("comparator_absent","insufficient_comparator") and conf=="high": conf="medium"

        scored_themes.append(ScoredTheme(
            theme=theme,support_score=round(support,1),specificity_score=specificity,
            final_score=round(final,1),direct_count=td["direct"],inferred_count=inf,
            motif_count=len(td["motifs"]),source_diversity=len(td["sources"]),
            confidence=conf,is_broad=is_broad,
            query_direct=q_dir,query_sources=q_src,query_motifs=q_mot,
            comp_direct=c_dir,comp_sources=c_src,comp_motifs=c_mot,
            enrichment_ratio=raw_e,contrast_score=contrast,
            query_direct_norm=q_n,comp_direct_norm=c_n,norm_enrichment_ratio=ne,
            evidence_balance=bal,interpretation=interp,coverage_flag=cov,
        ))
    scored_themes.sort(key=lambda x: -x.final_score)

    top_bio = sorted(gr.biomolecules, key=lambda x: -x["count"])[:8]
    top_fg = [{**fg, "generic_flag": fg["functional_group"].lower() in _GENERIC_HUBS}
              for fg in sorted(gr.functional_groups, key=lambda x: -x["count"])[:8]]

    # Legacy motif summary (kept for backward compat)
    q_mc = _count_motifs(gr.raw_records)
    c_mc = _count_motifs(gr.comparator_records) if is_comp else {}
    mm={}
    for r in gr.raw_records:
        mid=r.get("motif_id")
        if mid and mid not in mm: mm[mid]={"sf":r.get("motif_subfamily",""),"fam":r.get("motif_family","")}
    motif_summary=[]
    for mid,cnt in sorted(q_mc.items(),key=lambda x:-x[1])[:12]:
        m=mm.get(mid,{}); ccnt=c_mc.get(mid,0)
        enr="broadly-shared" if cnt>=20 else ("condition-enriched" if cnt>=3 else "sparse")
        mf=""
        if is_comp:
            if ccnt==0: interp2="query-associated"; mf="comparator_absent"
            elif ccnt<2 and cnt>=3: interp2="query-associated"; mf="insufficient_comparator"
            elif cnt>ccnt*2 and ccnt>=2: interp2="enriched"
            elif ccnt>cnt*2 and cnt>=2: interp2="comparator-associated"
            else: interp2="shared"
        else: interp2="query-associated"
        motif_summary.append(ScoredMotif(motif_id=mid,subfamily=m.get("sf",""),family=m.get("fam",""),
                                          member_count=cnt,condition_enrichment=enr,
                                          comparator_members=ccnt,motif_interpretation=interp2,coverage_flag=mf))

    # NEW: Motif differentials with stability
    motif_diffs = _build_motif_differentials(gr)

    evidence_sample = _select_quality_evidence(gr)

    comp_summary = None
    if is_comp:
        comp_summary = ComparatorSummary(
            query_condition=gr.matched_entity, comparator_condition=gr.comparator or "all_other",
            inferred_sample_type=gr.inferred_sample_type or "unknown",
            query_evidence=gr.total_evidence_rows, query_sources=gr.source_count,
            query_motifs=len(q_mc), comparator_evidence=gr.comparator_evidence_count,
            comparator_sources=gr.comparator_source_count,
            comparator_adequacy=comp_adequacy, evidence_balance=global_balance,
            scope_mode=gr.scope_mode, sample_scope=gr.sample_scope,
            domain_scope=gr.domain_scope, scope_fallback=gr.scope_fallback,
        )

    caveats = _generate_caveats(scored_themes, top_fg, motif_summary, motif_diffs,
                                 gr, is_comp, comp_adequacy, comp_summary)

    return ScoredResult(
        query_type=gr.query_type, query_mode=gr.query_mode,
        matched_entity=gr.matched_entity, comparator=gr.comparator,
        themes=scored_themes, top_biomolecules=top_bio, top_functional_groups=top_fg,
        motif_summary=motif_summary, motif_differentials=motif_diffs,
        evidence_sample=evidence_sample,
        evidence_count=gr.total_evidence_rows, source_count=gr.source_count,
        caveats=caveats, comparator_summary=comp_summary,
    )


def _generate_caveats(themes, fgs, motifs, motif_diffs, gr, is_comp, comp_adequacy, comp_summary):
    caveats = []
    if gr.total_evidence_rows < 5: caveats.append("Low evidence support.")
    if gr.source_count < 2: caveats.append("Fewer than 2 independent sources.")
    broad_top = [t for t in themes[:3] if t.is_broad]
    if len(broad_top) >= 2:
        caveats.append(f"Top themes ({', '.join(t.theme for t in broad_top)}) are broad biochemical classes.")
    generic_fgs = [f for f in fgs[:3] if f.get("generic_flag")]
    if generic_fgs: caveats.append("Dominant functional group(s) are generic.")

    if is_comp:
        if comp_adequacy == "very_sparse":
            caveats.append(f"Comparator very sparse ({comp_summary.comparator_evidence if comp_summary else '?'} rows).")
        elif comp_adequacy == "sparse":
            caveats.append(f"Comparator sparse ({comp_summary.comparator_evidence if comp_summary else '?'} rows).")
        absent = [t for t in themes[:6] if t.coverage_flag == "comparator_absent"]
        if absent: caveats.append(f"Comparator absent for: {', '.join(t.theme for t in absent[:3])}.")
        if comp_summary and comp_summary.evidence_balance < 0.3:
            caveats.append(f"Evidence balance low ({comp_summary.evidence_balance:.2f}).")

        # Stability-based caveats
        unstable = [d for d in motif_diffs if d.stability_label == "UNSTABLE"]
        mixed = [d for d in motif_diffs if d.stability_label == "MIXED"]
        if unstable:
            caveats.append(f"Unstable cross-study signal for: {', '.join(d.subfamily or d.motif_id for d in unstable[:3])}.")
        if mixed and not any("Mixed" in c for c in caveats):
            n = len(mixed)
            caveats.append(f"{n} motif(s) show mixed cross-study consistency.")
    else:
        broad_m = [m for m in motifs[:5] if m.condition_enrichment == "broadly-shared"]
        if len(broad_m) >= 3: caveats.append("Most motifs broadly shared.")
        low_spec = [t for t in themes[:3] if t.specificity_score < 0.3]
        if low_spec: caveats.append("Top themes have low specificity.")
    return caveats
