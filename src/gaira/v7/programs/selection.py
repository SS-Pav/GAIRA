"""GAIRA V7 — Phase 07: the model-selection rule, declared before any model was fitted.

Written down here, in code, so that it cannot be adjusted after the numbers arrive. Four phases
of V7 have now been decided by a rule of this shape and three of them returned *discard*; the
rule is the reason those verdicts are believable.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12

# ── PRE-REGISTERED DECISION RULE ─────────────────────────────────────────────
# Stage 1 — hard floors. A candidate that fails any of these is not eligible, however good it
# looks elsewhere. Stability, sparsity and reconstruction gains do not buy exemption (P-18).
# CORRECTION, made after the first run and before re-inspecting any result. The floors below
# originally omitted two constraints that the phase brief states explicitly, and the omission was
# a specification bug rather than a threshold worth arguing about: the rule as first written
# selected K = 16 over a 16-dimensional input, where NMF learns a permutation of the identity —
# EV 1.000, bootstrap 0.997, and every "programme" equal to exactly one chemistry class. That is
# not a compression and it is not a programme layer. The two missing constraints are added here,
# both derived from statements that predate any Phase 07 result:
#   * "A programme should NOT equal one chemistry class"  (brief, Scientific constraints)
#   * BSV2 is a COMPRESSION of the Chemistry Evidence layer (brief, Scientific objective)
# The compression bound is the input's own effective rank, 12.12, measured in Phase 06 — above
# it, a factorisation is not compressing information at all.
FLOORS = {
    # Carried forward from success criterion S-28, frozen in the post-Phase-05 architecture
    # update. BSV2 must retain half of what it compresses on BOTH axes.
    "information_retained_vs_chemistry_evidence": 0.50,
    "heldout_chemistry_retention": 0.50,
    # A programme set in which two programmes load on the same chemistries is over-parameterised;
    # a set where one programme wins for most spectra is a background with decorations.
    "max_pairwise_overlap": 0.90,          # upper bound
    "max_dominance": 0.60,                 # upper bound
    # Non-negativity is a hard architectural constraint (P-02): a negative amount of a
    # biochemical programme is meaningless. PCA and ICA are therefore controls, never candidates.
    "non_negative_activations": True,
    # Compression must be real. The Chemistry Evidence matrix has effective rank 12.12 (Phase 06);
    # a K at or above that is a rotation, not a compression.
    "max_K": 12,
    # Soft membership. A programme placing almost all of its loading on one chemistry axis IS
    # that chemistry class under another name.
    "max_single_axis_share": 0.90,
}
# Stage 2 — among eligible candidates, maximise the PRODUCT of held-out chemistry retention and
# bootstrap programme stability. A product, so a candidate must be both informative and
# reproducible; neither can compensate for the other. This is the exact shape of rule that
# discarded Meta Components in Phase 04.5.
OBJECTIVE = "heldout_chemistry_retention * bootstrap_stability"
# Stage 3 — ties within 0.01 of the best are broken toward the SMALLER K. Parsimony, and a
# defence against reading noise as structure at K near the effective rank of the input.
TIE_TOLERANCE = 0.01


def eligible(row: dict) -> tuple[bool, str]:
    """Apply the floors. Returns (eligible, reason-if-not)."""
    if not row.get("non_negative_activations", False):
        return False, "signed activations — control only, never a candidate (P-02)"
    if row["information_retained_vs_chemistry_evidence"] < \
            FLOORS["information_retained_vs_chemistry_evidence"]:
        return False, (f"information retained {row['information_retained_vs_chemistry_evidence']:.3f} "
                       f"< floor {FLOORS['information_retained_vs_chemistry_evidence']}")
    if row["heldout_chemistry_retention"] < FLOORS["heldout_chemistry_retention"]:
        return False, (f"held-out chemistry retention {row['heldout_chemistry_retention']:.3f} "
                       f"< floor {FLOORS['heldout_chemistry_retention']}")
    if row["max_pairwise_overlap"] > FLOORS["max_pairwise_overlap"]:
        return False, (f"two programmes overlap at {row['max_pairwise_overlap']:.3f} "
                       f"> {FLOORS['max_pairwise_overlap']} — duplicated programme")
    if row["dominance"] > FLOORS["max_dominance"]:
        return False, (f"one programme dominates {row['dominance']:.3f} of spectra "
                       f"> {FLOORS['max_dominance']} — a background, not a programme set")
    if row["K"] > FLOORS["max_K"]:
        return False, (f"K = {int(row['K'])} exceeds the input's effective rank "
                       f"({FLOORS['max_K']}) — a rotation, not a compression")
    if row.get("max_single_axis_share", 0.0) > FLOORS["max_single_axis_share"]:
        return False, (f"a programme places {row['max_single_axis_share']:.3f} of its loading on "
                       f"one chemistry axis > {FLOORS['max_single_axis_share']} — that programme "
                       f"IS a chemistry class under another name")
    return True, ""


def select(tab: "pd.DataFrame") -> dict:
    """Apply the rule to the full sweep and return the decision with its audit trail."""
    import pandas as pd
    t = tab.copy()
    verdicts = [eligible(r._asdict() if hasattr(r, "_asdict") else dict(r))
                for _, r in t.iterrows()]
    t["eligible"] = [v[0] for v in verdicts]
    t["ineligible_reason"] = [v[1] for v in verdicts]
    t["objective"] = t.heldout_chemistry_retention * t.bootstrap_stability
    ok = t[t.eligible]
    if not len(ok):
        return {"decision": "NO ELIGIBLE CANDIDATE", "family": None, "K": None,
                "rule": OBJECTIVE, "table": t,
                "rationale": ("no (family, K) combination cleared the pre-registered floors; "
                              "BSV2 is not adopted")}
    best = ok.sort_values("objective", ascending=False).iloc[0]
    near = ok[ok.objective >= best.objective - TIE_TOLERANCE]
    chosen = near.sort_values(["K", "objective"], ascending=[True, False]).iloc[0]
    return {"decision": "ADOPT" if chosen.objective > 0 else "NO ELIGIBLE CANDIDATE",
            "family": str(chosen.family), "K": int(chosen.K),
            "objective_value": float(chosen.objective),
            "n_eligible": int(len(ok)), "n_within_tie_tolerance": int(len(near)),
            "rule": OBJECTIVE, "floors": FLOORS, "tie_tolerance": TIE_TOLERANCE,
            "rationale": (f"{chosen.family} at K={int(chosen.K)}: objective "
                          f"{chosen.objective:.4f} = held-out chemistry retention "
                          f"{chosen.heldout_chemistry_retention:.3f} x bootstrap stability "
                          f"{chosen.bootstrap_stability:.3f}; smallest K within "
                          f"{TIE_TOLERANCE} of the best of {len(ok)} eligible candidates"),
            "table": t}
