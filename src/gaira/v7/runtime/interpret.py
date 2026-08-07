"""GAIRA V7 — Phase 10: deterministic interpretation text.

Template-driven, seeded by nothing, produced by no language model. The same result always yields
the same sentence, which is what makes it quotable in a report.

Every clause is bounded by what Phase 09 measured. There is no sentence here that says a molecule
was identified, that a chemistry is present at some amount, or that a biological state was
inferred, because none of those is supported.
"""
from __future__ import annotations


def _quality(ev: float) -> str:
    if ev >= 0.90:
        return "very well represented by the frozen Raman atlas"
    if ev >= 0.75:
        return "well represented by the frozen Raman atlas"
    if ev >= 0.50:
        return "moderately represented by the frozen Raman atlas"
    return "poorly represented by the frozen Raman atlas"


def _pretty(axis: str) -> str:
    return axis.replace("_", " ")


def interpretation(csm_ev: float, chem_top, retrieval_top, conf, warnings: list[str]) -> str:
    """One paragraph, assembled from measured quantities only."""
    parts: list[str] = []
    parts.append(
        f"The query is {_quality(csm_ev)} (CSM explained variance = {csm_ev:.3f}).")

    if chem_top:
        first = chem_top[0]
        rest = ", ".join(_pretty(a.axis) for a in chem_top[1:3])
        s = (f"The strongest relative Chemistry Evidence is associated with "
             f"{_pretty(first.axis)} ({first.share:.0%} of total evidence, calibrated "
             f"confidence {first.calibrated_probability:.2f})")
        if rest:
            s += f", followed by {rest}"
        parts.append(s + ".")

    if retrieval_top:
        names = [h.molecule for h in retrieval_top[:3]]
        sims = retrieval_top[0].similarity
        parts.append(
            f"Grounded Evidence Retrieval returns {', '.join(names[:-1])} and {names[-1]} as the "
            f"nearest reference spectra (top similarity {sims:.3f}). These candidates should be "
            f"interpreted as reference analogues rather than definitive molecular "
            f"identifications." if len(names) > 1 else
            f"Grounded Evidence Retrieval returns {names[0]} as the nearest reference spectrum "
            f"(similarity {sims:.3f}), to be interpreted as a reference analogue rather than a "
            f"definitive molecular identification.")

    if conf.unknown_warning or conf.outlier_warning:
        # Name the condition that actually fired. The two `unknown` triggers mean opposite
        # things: a poorly explained spectrum is a coverage problem, while a small margin on a
        # well-explained spectrum usually means two near-identical references — a stereoisomer
        # pair, say — and is not a defect at all.
        why: list[str] = []
        if conf.reconstruction_explained_variance < 0.50:
            why.append(f"the atlas explains only {conf.reconstruction_explained_variance:.0%} "
                       f"of the spectrum")
        if conf.retrieval_margin < 0.01:
            why.append(f"the top two candidates differ by only {conf.retrieval_margin:.4f}, so "
                       f"the ranking between them is not decisive")
        if conf.outlier_warning:
            why.append("the residual or active-component count is outside the expected range")
        flags = [n for n, f in (("unknown", conf.unknown_warning),
                                ("outlier", conf.outlier_warning)) if f]
        parts.append(
            f"The engine raised the {' and '.join(flags)} flag because "
            f"{'; and '.join(why) if why else 'a confidence condition was met'}. This is not "
            f"evidence that the true molecule is absent from the reference bank, which V7 "
            f"cannot determine.")
    elif conf.overall < 0.50:
        parts.append(
            f"Overall confidence is low ({conf.overall:.2f}). Treat the ordering as indicative "
            f"rather than settled.")

    if warnings:
        parts.append("Input notes: " + "; ".join(warnings[:3]) + ".")
    return " ".join(parts)


def comparison_text(label_a: str, label_b: str, csm_cos: float, chem_cos: float,
                    deltas, shared: list[str], rank_agreement: float) -> str:
    up = sorted(deltas, key=lambda d: -d.delta)[:2]
    down = sorted(deltas, key=lambda d: d.delta)[:2]
    s = [f"{label_a} and {label_b} have CSM cosine similarity {csm_cos:.3f} and Chemistry "
         f"Evidence cosine similarity {chem_cos:.3f}."]
    if up and up[0].delta > 0:
        s.append(f"Relative evidence is higher in {label_b} for "
                 f"{', '.join(_pretty(d.axis) for d in up if d.delta > 0)}")
    if down and down[0].delta < 0:
        s.append(f"and higher in {label_a} for "
                 f"{', '.join(_pretty(d.axis) for d in down if d.delta < 0)}.")
    elif up and up[0].delta > 0:
        s[-1] += "."
    s.append(f"The two top-10 retrieval sets overlap at Jaccard {rank_agreement:.2f}"
             + (f", sharing {', '.join(shared[:3])}." if shared else ", sharing no molecules."))
    s.append("These are differences in spectral motif evidence, chemistry evidence and reference "
             "neighbourhoods. V7 does not license a claim about biological state change.")
    return " ".join(s)
