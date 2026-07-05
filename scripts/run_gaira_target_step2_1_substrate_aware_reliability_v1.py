"""GAIRA — Step 2.1 Substrate-Aware Reliability (v1).

Post-analysis reclassification layer.

Consumes:
  - Step 2 statistical reliability matrix
    (.../step2_axis_reliability_v1/axis_reliability_matrix.csv)
  - Stage 2 per-pilot substrate overlays
    (.../stage2_substrate_overlay_v1/<pilot>_axis_substrate_overlay.csv)

Produces:
  - axis_reliability_matrix_step2_1.csv
  - axis_reliability_summary_step2_1.csv
  - axis_reliability_changes_step2_to_2_1.csv
  - fig_step2_vs_step2_1_heatmap.png
  - fig_step2_1_change_reason_panel.png
  - fig_step2_1_carry_forward_axes.png
  - REPORT_step2_1_substrate_aware_reliability_v1.md

Hard rules enforced:
  - never recompute pilots
  - never alter BSV / ΔBSV / effect sizes / scorer / atlas / windows
  - never overwrite Step 2 outputs (file checksum gate verifies this)
  - tier reclassification is deterministic and rule-traced

Run:
    cd /Users/suraj/projects/GAIRA
    PYTHONPATH=src .venv/bin/python scripts/run_gaira_target_step2_1_substrate_aware_reliability_v1.py
"""
from __future__ import annotations

import csv
import hashlib
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ──────────────────────────────────────────────────────────────────────
# Paths / configuration
# ──────────────────────────────────────────────────────────────────────

PILOT_ROOT     = Path("/Volumes/SSD_Rad/GAIRA_BUILD/gaira_target_pilot")
STEP2_DIR      = PILOT_ROOT / "step2_axis_reliability_v1"
STAGE2_DIR     = PILOT_ROOT / "stage2_substrate_overlay_v1"
OUT_DIR        = PILOT_ROOT / "step2_1_substrate_aware_reliability_v1"

STEP2_MATRIX_CSV = STEP2_DIR / "axis_reliability_matrix.csv"

# Map Step 2 pilot_name → Stage 2 overlay CSV path
PILOT_TO_OVERLAY: dict[str, Path] = {
    "pilot1_hcc_holdout": STAGE2_DIR / "pilot1_hcc_axis_substrate_overlay.csv",
    "pilot2b_cca_raw":    STAGE2_DIR / "pilot2b_cca_axis_substrate_overlay.csv",
    # (Step 2 does not include pilot3; if it ever does, add a mapping here.)
}

# Tier vocabulary (Step 2 → Step 2.1 parallel)
TIER_T1 = "TIER_1_ROBUST"
TIER_T2 = "TIER_2_CONTEXTUAL"
TIER_T3 = "TIER_3_UNRELIABLE"          # generic Step 2.1 T3 label
TIER_T3_UNINTERPRETABLE = "TIER_3_UNINTERPRETABLE"  # specific cause: substrate conflict

# Step 2.1 reliability class
PRIMARY_CARRY = "PRIMARY_CARRY_FORWARD"
SECONDARY     = "SECONDARY_CONTEXT_ONLY"
EXCLUDE       = "EXCLUDE_PRIMARY_CLAIMS"

# Reclassification rule labels
RULE_NO_CHANGE         = "NO_CHANGE_NEUTRAL"
RULE_CONFLICT_OVERRIDE = "RULE1_CONFLICT_OVERRIDE"
RULE_BIAS_DOWNGRADE    = "RULE2_SUBSTRATE_BIAS_DOWNGRADE"
RULE_PRESERVED_UPGRADE = "RULE3_PHYSICS_AWARE_UPGRADE_PRESERVED"
RULE_AMBIGUOUS_CAP     = "RULE5_AMBIGUOUS_CAP_AT_T2"


# ──────────────────────────────────────────────────────────────────────
# IO helpers
# ──────────────────────────────────────────────────────────────────────

def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(8192), b""): h.update(c)
    return h.hexdigest()


def _read_csv(p: Path) -> list[dict[str, str]]:
    with p.open() as f: return list(csv.DictReader(f))


def _write_csv(p: Path, header: list[str], rows: list[list]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(header)
        for r in rows: w.writerow(r)


def _gate(before: dict, after: dict) -> None:
    diff = [k for k in before if after.get(k) != before[k]]
    if diff:
        raise RuntimeError("UPSTREAM FILES MUTATED:\n  " + "\n  ".join(diff))


def _snapshot_inputs() -> dict[str, str]:
    files: list[Path] = []
    for p in STEP2_DIR.iterdir():
        if p.is_file(): files.append(p)
    for ov in PILOT_TO_OVERLAY.values():
        if ov.exists(): files.append(ov)
    return {str(p): _sha(p) for p in files}


# ──────────────────────────────────────────────────────────────────────
# Tier-step helpers
# ──────────────────────────────────────────────────────────────────────

_TIER_RANK = {TIER_T1: 1, TIER_T2: 2, TIER_T3: 3, "TIER_3_UNSTABLE": 3,
              TIER_T3_UNINTERPRETABLE: 3}


def _normalise_step2_tier(s: str) -> str:
    """Step 2 uses TIER_3_UNSTABLE; Step 2.1 uses TIER_3_UNRELIABLE for the
    generic T3 bucket. Preserve Step 2's label as-is (read-only from Step 2)."""
    return s.strip()


def _step2_1_tier_to_class(t: str) -> str:
    if t == TIER_T1: return PRIMARY_CARRY
    if t == TIER_T2: return SECONDARY
    return EXCLUDE   # any T3 variant


def _downgrade_one_level(t: str) -> str:
    if t == TIER_T1: return TIER_T2
    if t == TIER_T2: return TIER_T2     # no change per Rule 2
    return TIER_T3   # already at T3


# ──────────────────────────────────────────────────────────────────────
# Rule engine (deterministic)
# ──────────────────────────────────────────────────────────────────────

@dataclass
class RuleOutcome:
    new_tier: str
    rule_applied: str
    reason_short: str
    reason_long: str


def _apply_rules(
    step2_tier: str,
    visibility: str,
    abundance: str,
    conflict_flag: bool,
    unresolved_assignment_flag: bool,
    substrate_adjusted_class: str,
    insufficient_n: int,
    weighted_n: int,
) -> RuleOutcome:
    """Apply Step 2.1 reclassification rules in deterministic priority order.

    Priority (top wins):
      Rule 1  — conflict_flag OR unresolved_assignment_flag → TIER_3_UNINTERPRETABLE
      Rule 3  — original=T1 + suppressed-with-STRONGER → preserve T1 + annotate
      Rule 5  — substrate_adjusted_class=AMBIGUOUS (no conflict) → cap at T2
      Rule 2  — visibility in {enhanced, biased, suppressed} + abundance caveat → downgrade one level
      Rule 4  — neutral / unchanged / inconclusive → no change
    """

    # Rule 1 — conflict overrides
    if conflict_flag or unresolved_assignment_flag:
        return RuleOutcome(
            new_tier=TIER_T3_UNINTERPRETABLE,
            rule_applied=RULE_CONFLICT_OVERRIDE,
            reason_short="conflict / unresolved_assignment_flag → tier 3 uninterpretable",
            reason_long=(
                "Substrate engine flagged this axis with CONFLICTING literature "
                "evidence (or an unresolved assignment); strong statistical effect "
                "cannot be carried forward as a clean biological interpretation. "
                "Step 2 tier preserved separately for audit."
            ),
        )

    # Rule 3 — biological-overcoming-suppression preserves T1
    # Trigger: original=T1, substrate_adjusted_class=STRONGER, visibility implies
    # under-observation (suppressed / biased — both indicate Ag-colloid suppression
    # context for axes like glycan).
    if (step2_tier == TIER_T1
            and substrate_adjusted_class == "STRONGER"
            and visibility in {"suppressed", "biased"}
            and not (conflict_flag or unresolved_assignment_flag)):
        return RuleOutcome(
            new_tier=TIER_T1,
            rule_applied=RULE_PRESERVED_UPGRADE,
            reason_short="PHYSICS_AWARE_UPGRADE — biology overcomes substrate suppression; T1 preserved",
            reason_long=(
                "Original Step 2 Tier 1 effect occurs on a substrate axis whose "
                "canonical visibility is suppressed (or competing-bias). Substrate "
                "engine classifies this as STRONGER — observed signal is conservative "
                "relative to true biology. Tier 1 preserved with PHYSICS_AWARE_UPGRADE "
                "annotation."
            ),
        )

    # Rule 5 — AMBIGUOUS without explicit conflict_flag → cap at T2
    # (e.g. citrate baseline / non_biological visibility, or CONVERGED both directions)
    if substrate_adjusted_class == "AMBIGUOUS":
        if step2_tier == TIER_T1:
            return RuleOutcome(
                new_tier=TIER_T2,
                rule_applied=RULE_AMBIGUOUS_CAP,
                reason_short="substrate_adjusted_class=AMBIGUOUS → capped at T2",
                reason_long=(
                    "Substrate engine classification is AMBIGUOUS (substrate-artifact "
                    "contribution or converged-both-directions). T1 cannot be claimed; "
                    "downgraded to T2 contextual."
                ),
            )
        # already T2/T3 — leave as is (no upgrade), but tag the rule for audit
        return RuleOutcome(
            new_tier=_normalise_step2_tier(step2_tier) if "TIER_3" not in step2_tier else TIER_T3,
            rule_applied=RULE_AMBIGUOUS_CAP,
            reason_short="substrate AMBIGUOUS — already ≤ T2; no further change",
            reason_long="AMBIGUOUS but original tier already T2 or T3; preserved.",
        )

    # Rule 2 — substrate bias downgrade
    bias_visibilities = {"enhanced", "biased", "suppressed"}
    bias_abundances = {
        "may_overestimate_abundance",
        "may_underestimate_abundance",
        "abundance_not_directly_inferable",
    }
    if (visibility in bias_visibilities
            and abundance in bias_abundances
            and not (conflict_flag or unresolved_assignment_flag)):
        new_tier = _downgrade_one_level(step2_tier) if "TIER_3" not in step2_tier else TIER_T3
        if new_tier != step2_tier:
            return RuleOutcome(
                new_tier=new_tier,
                rule_applied=RULE_BIAS_DOWNGRADE,
                reason_short=f"substrate bias caveat (visibility=`{visibility}`, abundance=`{abundance}`) → downgrade",
                reason_long=(
                    f"Substrate engine reports visibility=`{visibility}` and "
                    f"abundance=`{abundance}`. Signal may be real but direct "
                    "abundance interpretation is not fully trustworthy; downgrade "
                    "one tier."
                ),
            )
        # downgrade is a no-op (already T2 or T3 reflecting bias)
        return RuleOutcome(
            new_tier=_normalise_step2_tier(step2_tier) if "TIER_3" not in step2_tier else TIER_T3,
            rule_applied=RULE_NO_CHANGE,
            reason_short="substrate bias present; original tier already ≤ T2",
            reason_long=(
                f"Visibility=`{visibility}`, abundance=`{abundance}`, but the original "
                "Step 2 tier already reflects a context / unstable classification."
            ),
        )

    # Rule 4 — neutral / unchanged → keep tier
    return RuleOutcome(
        new_tier=_normalise_step2_tier(step2_tier) if "TIER_3" not in step2_tier else TIER_T3,
        rule_applied=RULE_NO_CHANGE,
        reason_short="no substrate caveat triggers — Step 2 tier preserved",
        reason_long=(
            f"substrate_adjusted_class=`{substrate_adjusted_class}`, "
            f"visibility=`{visibility}`, abundance=`{abundance}`, conflict_flag=false; "
            "no Step 2.1 reclassification rule fires."
        ),
    )


# ──────────────────────────────────────────────────────────────────────
# Build the matrix
# ──────────────────────────────────────────────────────────────────────

def _truthy(s: str) -> bool:
    return str(s).strip().lower() in {"true", "1", "yes"}


def build_matrix() -> list[dict]:
    step2_rows = _read_csv(STEP2_MATRIX_CSV)
    out_rows: list[dict] = []
    overlay_cache: dict[str, dict[str, dict[str, str]]] = {}

    for r in step2_rows:
        pilot = r["pilot_name"]
        if pilot not in PILOT_TO_OVERLAY:
            # Skip pilots without a Stage 2 overlay; document explicitly.
            out_rows.append({
                **r,
                "_overlay_status": "NO_OVERLAY_AVAILABLE",
                "_outcome": RuleOutcome(
                    new_tier=_normalise_step2_tier(r["tier"])
                              if "TIER_3" not in r["tier"] else TIER_T3,
                    rule_applied="NO_OVERLAY_PASSTHROUGH",
                    reason_short="no Stage 2 overlay for this pilot — Step 2 tier passed through",
                    reason_long="Step 2.1 cannot reclassify without a Stage 2 overlay; passthrough.",
                ),
                "_overlay_row": {},
            })
            continue

        if pilot not in overlay_cache:
            overlay_cache[pilot] = {
                row["axis"]: row for row in _read_csv(PILOT_TO_OVERLAY[pilot])
            }
        ov = overlay_cache[pilot].get(r["axis"], {})

        outcome = _apply_rules(
            step2_tier=_normalise_step2_tier(r["tier"]),
            visibility=ov.get("visibility_tag", "") or "",
            abundance=ov.get("abundance_interpretation", "") or "",
            conflict_flag=_truthy(ov.get("conflict_flag", "false")),
            unresolved_assignment_flag=_truthy(ov.get("unresolved_assignment_flag", "false")),
            substrate_adjusted_class=ov.get("interpretation_shift", "UNCHANGED") or "UNCHANGED",
            insufficient_n=int(ov.get("insufficient_effect_count", "0") or 0),
            weighted_n=int(ov.get("weighted_effect_count", "0") or 0),
        )

        out_rows.append({
            **r,
            "_overlay_status": "OK",
            "_overlay_row": ov,
            "_outcome": outcome,
        })

    return out_rows


# ──────────────────────────────────────────────────────────────────────
# Output writers
# ──────────────────────────────────────────────────────────────────────

MATRIX_HEADER = [
    "pilot_name", "dataset_name", "compare_class", "axis",
    "step2_tier", "step2_1_tier", "step2_1_reliability_class",
    "statistical_basis_summary",
    "substrate_visibility_tag", "abundance_interpretation",
    "conflict_flag", "unresolved_assignment_flag",
    "substrate_adjusted_class",
    "reclassification_rule_applied", "reason_short", "reason_long",
]


def write_matrix(rows: list[dict]) -> Path:
    out = OUT_DIR / "axis_reliability_matrix_step2_1.csv"
    csv_rows: list[list] = []
    for r in rows:
        ov = r.get("_overlay_row", {}) or {}
        oc: RuleOutcome = r["_outcome"]
        csv_rows.append([
            r["pilot_name"], r["dataset_name"], r["compare_class"], r["axis"],
            r["tier"],                         # original Step 2 tier (preserved)
            oc.new_tier,
            _step2_1_tier_to_class(oc.new_tier),
            r.get("reason_short", ""),
            ov.get("visibility_tag", ""),
            ov.get("abundance_interpretation", ""),
            ov.get("conflict_flag", ""),
            ov.get("unresolved_assignment_flag", ""),
            ov.get("interpretation_shift", ""),
            oc.rule_applied, oc.reason_short, oc.reason_long,
        ])
    _write_csv(out, MATRIX_HEADER, csv_rows)
    return out


SUMMARY_HEADER = [
    "pilot_name", "dataset_name", "compare_class",
    "n_axes", "n_step2_T1", "n_step2_T2", "n_step2_T3",
    "n_step2_1_T1", "n_step2_1_T2", "n_step2_1_T3",
    "promoted", "unchanged", "downgraded",
    "carry_forward_axes_step2_1",
]


def write_summary(rows: list[dict]) -> Path:
    out = OUT_DIR / "axis_reliability_summary_step2_1.csv"
    by_pilot: dict[tuple, list[dict]] = {}
    for r in rows:
        key = (r["pilot_name"], r["dataset_name"], r["compare_class"])
        by_pilot.setdefault(key, []).append(r)

    csv_rows: list[list] = []
    for (pilot, ds, cmp_cls), rs in by_pilot.items():
        n = len(rs)
        s2_counts = Counter(rs_["tier"] for rs_ in rs)
        s2_1_counts = Counter(rs_["_outcome"].new_tier for rs_ in rs)
        promoted = unchanged = downgraded = 0
        carry_forward: list[str] = []
        for rs_ in rs:
            r2 = _TIER_RANK.get(rs_["tier"], 99)
            r2_1 = _TIER_RANK.get(rs_["_outcome"].new_tier, 99)
            if r2_1 < r2:
                promoted += 1
            elif r2_1 > r2:
                downgraded += 1
            else:
                unchanged += 1
            if rs_["_outcome"].new_tier == TIER_T1:
                carry_forward.append(rs_["axis"])
        csv_rows.append([
            pilot, ds, cmp_cls, n,
            s2_counts.get(TIER_T1, 0),
            s2_counts.get(TIER_T2, 0),
            s2_counts.get("TIER_3_UNSTABLE", 0) + s2_counts.get(TIER_T3, 0),
            s2_1_counts.get(TIER_T1, 0),
            s2_1_counts.get(TIER_T2, 0),
            s2_1_counts.get(TIER_T3, 0) + s2_1_counts.get(TIER_T3_UNINTERPRETABLE, 0),
            promoted, unchanged, downgraded,
            "; ".join(carry_forward) if carry_forward else "—",
        ])
    _write_csv(out, SUMMARY_HEADER, csv_rows)
    return out


CHANGES_HEADER = [
    "pilot_name", "axis", "step2_tier", "step2_1_tier", "change_type",
    "reclassification_rule_applied", "notes",
]


def write_changes(rows: list[dict]) -> Path:
    out = OUT_DIR / "axis_reliability_changes_step2_to_2_1.csv"
    csv_rows: list[list] = []
    for r in rows:
        oc: RuleOutcome = r["_outcome"]
        r2 = _TIER_RANK.get(r["tier"], 99)
        r2_1 = _TIER_RANK.get(oc.new_tier, 99)
        if oc.rule_applied == RULE_PRESERVED_UPGRADE:
            change_type = "PRESERVED_WITH_PHYSICS_UPGRADE"
        elif r2_1 > r2:
            change_type = "DOWNGRADED"
        elif r2_1 < r2:
            change_type = "PROMOTED"        # not expected under v1 rules
        else:
            change_type = "UNCHANGED"
        csv_rows.append([
            r["pilot_name"], r["axis"], r["tier"], oc.new_tier,
            change_type, oc.rule_applied, oc.reason_short,
        ])
    _write_csv(out, CHANGES_HEADER, csv_rows)
    return out


# ──────────────────────────────────────────────────────────────────────
# Figures
# ──────────────────────────────────────────────────────────────────────

_TIER_COLOR = {
    TIER_T1: "#3F8E3F",                # green
    TIER_T2: "#D8A24E",                # amber
    TIER_T3: "#B04848",                # red
    "TIER_3_UNSTABLE": "#B04848",
    TIER_T3_UNINTERPRETABLE: "#7A1F1F",   # darker red
}
_RULE_COLOR = {
    RULE_NO_CHANGE:         "#888888",
    RULE_CONFLICT_OVERRIDE: "#7A1F1F",
    RULE_BIAS_DOWNGRADE:    "#C04444",
    RULE_PRESERVED_UPGRADE: "#3F8E3F",
    RULE_AMBIGUOUS_CAP:     "#B07A2A",
    "NO_OVERLAY_PASSTHROUGH": "#444444",
}


def fig_step2_vs_step2_1_heatmap(rows: list[dict], out_dir: Path) -> Path:
    # one row per (pilot, axis); 2 columns: Step 2 tier, Step 2.1 tier
    labels = [f"{r['pilot_name'].split('_',1)[0]} · {r['axis']}" for r in rows]
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(rows) + 1))

    # Map tiers → numeric bins for imshow
    tier_to_bin = {TIER_T1: 1, TIER_T2: 2, TIER_T3: 3,
                   "TIER_3_UNSTABLE": 3, TIER_T3_UNINTERPRETABLE: 4}
    tier_palette = ["#FFFFFF", _TIER_COLOR[TIER_T1], _TIER_COLOR[TIER_T2],
                    _TIER_COLOR[TIER_T3], _TIER_COLOR[TIER_T3_UNINTERPRETABLE]]
    cmap = matplotlib.colors.ListedColormap(tier_palette)
    norm = matplotlib.colors.BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], cmap.N)
    M = np.array([
        [tier_to_bin.get(r["tier"], 0), tier_to_bin.get(r["_outcome"].new_tier, 0)]
        for r in rows
    ])
    ax.imshow(M, aspect="auto", cmap=cmap, norm=norm)

    # Cell text
    for i, r in enumerate(rows):
        for j, t in enumerate([r["tier"], r["_outcome"].new_tier]):
            short = (
                "T1" if t == TIER_T1
                else "T2" if t == TIER_T2
                else "T3*" if t == TIER_T3_UNINTERPRETABLE
                else "T3"
            )
            ax.text(j, i, short, ha="center", va="center", fontsize=9,
                    color="w" if t in (TIER_T1, TIER_T3, "TIER_3_UNSTABLE",
                                       TIER_T3_UNINTERPRETABLE) else "k",
                    fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Step 2\n(statistical)", "Step 2.1\n(substrate-aware)"], fontsize=10)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(labels, fontsize=8)
    ax.set_title("Step 2 vs Step 2.1 — tier per (pilot × axis)\n"
                 "(T3* = TIER_3_UNINTERPRETABLE, conflict-driven)", fontsize=11)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=_TIER_COLOR[TIER_T1], label="T1 ROBUST → PRIMARY_CARRY_FORWARD"),
        plt.Rectangle((0, 0), 1, 1, color=_TIER_COLOR[TIER_T2], label="T2 CONTEXTUAL → SECONDARY_CONTEXT_ONLY"),
        plt.Rectangle((0, 0), 1, 1, color=_TIER_COLOR[TIER_T3], label="T3 UNRELIABLE / UNSTABLE → EXCLUDE"),
        plt.Rectangle((0, 0), 1, 1, color=_TIER_COLOR[TIER_T3_UNINTERPRETABLE],
                      label="T3* UNINTERPRETABLE (substrate conflict) → EXCLUDE"),
    ]
    ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.15),
              ncol=1, fontsize=8)
    fig.tight_layout()
    out = out_dir / "fig_step2_vs_step2_1_heatmap.png"
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    return out


def fig_change_reason_panel(rows: list[dict], out_dir: Path) -> Path:
    counts: Counter = Counter(r["_outcome"].rule_applied for r in rows)
    order = [
        RULE_NO_CHANGE, RULE_PRESERVED_UPGRADE, RULE_BIAS_DOWNGRADE,
        RULE_AMBIGUOUS_CAP, RULE_CONFLICT_OVERRIDE, "NO_OVERLAY_PASSTHROUGH",
    ]
    order = [k for k in order if counts.get(k, 0) > 0]
    fig, ax = plt.subplots(figsize=(8.5, 4))
    colors = [_RULE_COLOR.get(k, "#666") for k in order]
    x = np.arange(len(order))
    bars = ax.bar(x, [counts[k] for k in order], color=colors, alpha=0.85)
    for b, k in zip(bars, order):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.15,
                str(counts[k]), ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("# (pilot, axis) cells")
    ax.set_title("Step 2.1 reclassification — reason breakdown")
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=20, ha="right", fontsize=8)
    fig.tight_layout()
    out = out_dir / "fig_step2_1_change_reason_panel.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    return out


def fig_carry_forward_axes(rows: list[dict], out_dir: Path) -> Path:
    pilots = sorted({r["pilot_name"] for r in rows})
    classes = [PRIMARY_CARRY, SECONDARY, EXCLUDE]
    M = np.zeros((len(pilots), len(classes)), dtype=int)
    cells: dict[tuple[int, int], list[str]] = {}
    for i, p in enumerate(pilots):
        rs = [r for r in rows if r["pilot_name"] == p]
        for r in rs:
            cls = _step2_1_tier_to_class(r["_outcome"].new_tier)
            j = classes.index(cls)
            M[i, j] += 1
            cells.setdefault((i, j), []).append(r["axis"])
    fig, ax = plt.subplots(figsize=(11, 1.2 + 0.6 * len(pilots)))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "carry", ["#F5F5F5", "#3F8E3F"], N=256)
    im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=0, vmax=max(1, M.max()))
    for (i, j), axs in cells.items():
        ax.text(j, i, f"n={M[i, j]}\n" + ", ".join(axs),
                ha="center", va="center", fontsize=8,
                color="k" if M[i, j] < M.max() * 0.6 else "w")
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(
        ["PRIMARY\nCARRY_FORWARD", "SECONDARY\nCONTEXT_ONLY", "EXCLUDE\nPRIMARY_CLAIMS"],
        fontsize=9)
    ax.set_yticks(range(len(pilots))); ax.set_yticklabels(pilots, fontsize=9)
    ax.set_title("Step 2.1 carry-forward sets per pilot")
    fig.colorbar(im, ax=ax, label="# axes")
    fig.tight_layout()
    out = out_dir / "fig_step2_1_carry_forward_axes.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    return out


# ──────────────────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────────────────

RULE_TEXT = {
    RULE_CONFLICT_OVERRIDE: (
        "**Rule 1 — conflict overrides.** If `conflict_flag=true` OR "
        "`unresolved_assignment_flag=true` → set Step 2.1 tier = "
        "`TIER_3_UNINTERPRETABLE` regardless of Step 2 tier. Strong statistical "
        "effect with unresolved literature assignment cannot be carried forward "
        "as a clean biological interpretation."
    ),
    RULE_BIAS_DOWNGRADE: (
        "**Rule 2 — substrate bias downgrades.** If "
        "`visibility_tag` ∈ {enhanced, biased, suppressed} AND no conflict AND "
        "`abundance_interpretation` ∈ {may_overestimate, may_underestimate, "
        "abundance_not_directly_inferable} → downgrade one tier "
        "(T1→T2, T2→T2, T3→T3). Signal may be real but direct abundance "
        "interpretation is not fully trustworthy."
    ),
    RULE_PRESERVED_UPGRADE: (
        "**Rule 3 — biological-overcoming-suppression preserves T1.** If "
        "`step2_tier=TIER_1_ROBUST` AND `substrate_adjusted_class=STRONGER` AND "
        "visibility indicates suppressed / under-observed substrate context AND "
        "no conflict → preserve T1 with `PHYSICS_AWARE_UPGRADE` annotation. "
        "Observed signal is conservative relative to likely biology."
    ),
    RULE_NO_CHANGE: (
        "**Rule 4 — neutral / unchanged keeps tier.** If no conflict, no "
        "meaningful substrate caveat, and `substrate_adjusted_class=UNCHANGED` "
        "(or equivalent) → keep Step 2 tier."
    ),
    RULE_AMBIGUOUS_CAP: (
        "**Rule 5 — ambiguous axes cap at T2.** If "
        "`substrate_adjusted_class=AMBIGUOUS` (substrate-artifact / both-direction "
        "CONVERGED) without an explicit conflict_flag → cap Step 2.1 tier at T2."
    ),
}


def write_report(matrix_rows: list[dict], summary_csv: Path,
                 changes_csv: Path) -> Path:
    L: list[str] = []
    L.append("# GAIRA — Step 2.1 Substrate-Aware Reliability (v1)")
    L.append("")
    L.append(
        "_Post-analysis reclassification layer. Consumes Step 2 statistical "
        "reliability + Stage 2 substrate overlays. Annotation-only — no pilot "
        "numerics changed (verified by SHA-256 checksum gate over both Step 2 "
        "and Stage 2 inputs)._"
    )
    L.append("")

    # A. Why Step 2.1 exists
    L.append("## A. Why Step 2.1 exists")
    L.append("")
    L.append(
        "- **Step 2** = *statistical* axis reliability (effect size, CI, "
        "stability, entanglement, sensitivity). Tells us whether a number is "
        "stable. Does not say whether the number means what it appears to."
    )
    L.append(
        "- **Step 2.1** = *physics-aware* reliability — Step 2 tiers filtered "
        "through Stage 2 substrate overlays (visibility / abundance / conflict). "
        "Tells us whether a stable number can be trusted as a biological "
        "interpretation under the declared substrate."
    )
    L.append(
        "- The two layers are kept side-by-side: `step2_tier` is preserved "
        "verbatim in every row of the Step 2.1 matrix; `step2_1_tier` adds the "
        "physics-aware reading."
    )
    L.append("")

    # B. Rule set
    L.append("## B. Rule set")
    L.append("")
    L.append(
        "Rules are evaluated in priority order. The first rule whose "
        "preconditions match wins; only one rule fires per (pilot, axis) cell."
    )
    L.append("")
    for r in (RULE_CONFLICT_OVERRIDE, RULE_PRESERVED_UPGRADE, RULE_AMBIGUOUS_CAP,
              RULE_BIAS_DOWNGRADE, RULE_NO_CHANGE):
        L.append(f"1. {RULE_TEXT[r]}")
    L.append("")

    # C. Per-pilot reclassification
    L.append("## C. Per-pilot reclassification")
    L.append("")
    pilots = sorted({r["pilot_name"] for r in matrix_rows})
    for p in pilots:
        rs = [r for r in matrix_rows if r["pilot_name"] == p]
        L.append(f"### `{p}`")
        L.append("")
        L.append("| axis | step2 tier | step2.1 tier | rule | reason |")
        L.append("|---|:---:|:---:|:---:|---|")
        for r in rs:
            oc: RuleOutcome = r["_outcome"]
            L.append(f"| `{r['axis']}` | `{r['tier']}` | `{oc.new_tier}` | "
                     f"`{oc.rule_applied}` | {oc.reason_short} |")
        # Quick narrative
        unchanged = [r["axis"] for r in rs if r["_outcome"].rule_applied == RULE_NO_CHANGE]
        downgr    = [r["axis"] for r in rs
                     if r["_outcome"].rule_applied in (RULE_BIAS_DOWNGRADE,
                                                       RULE_AMBIGUOUS_CAP,
                                                       RULE_CONFLICT_OVERRIDE)
                     and _TIER_RANK[r["_outcome"].new_tier] > _TIER_RANK[r["tier"]]]
        preserved = [r["axis"] for r in rs
                     if r["_outcome"].rule_applied == RULE_PRESERVED_UPGRADE]
        L.append("")
        L.append(f"- unchanged (Rule 4 / no rule fired): {', '.join(f'`{a}`' for a in unchanged) or '—'}")
        L.append(f"- preserved with physics-aware upgrade (Rule 3): "
                 f"{', '.join(f'`{a}`' for a in preserved) or '—'}")
        L.append(f"- downgraded by Step 2.1 (Rules 1 / 2 / 5): "
                 f"{', '.join(f'`{a}`' for a in downgr) or '—'}")
        L.append("")

    # D. Cross-pilot carry-forward
    L.append("## D. Cross-pilot carry-forward set")
    L.append("")
    L.append("| pilot | PRIMARY_CARRY_FORWARD | SECONDARY_CONTEXT_ONLY | EXCLUDE_PRIMARY_CLAIMS |")
    L.append("|---|---|---|---|")
    for p in pilots:
        rs = [r for r in matrix_rows if r["pilot_name"] == p]
        prim = [r["axis"] for r in rs
                if _step2_1_tier_to_class(r["_outcome"].new_tier) == PRIMARY_CARRY]
        sec  = [r["axis"] for r in rs
                if _step2_1_tier_to_class(r["_outcome"].new_tier) == SECONDARY]
        exc  = [r["axis"] for r in rs
                if _step2_1_tier_to_class(r["_outcome"].new_tier) == EXCLUDE]
        L.append(f"| `{p}` | {', '.join(f'`{a}`' for a in prim) or '—'} | "
                 f"{', '.join(f'`{a}`' for a in sec) or '—'} | "
                 f"{', '.join(f'`{a}`' for a in exc) or '—'} |")
    L.append("")

    # Cross-pilot intersection
    pilot_primaries: dict[str, set[str]] = {
        p: {r["axis"] for r in matrix_rows
            if r["pilot_name"] == p
            and _step2_1_tier_to_class(r["_outcome"].new_tier) == PRIMARY_CARRY}
        for p in pilots
    }
    if len(pilot_primaries) >= 2:
        common = set.intersection(*pilot_primaries.values())
        union  = set.union(*pilot_primaries.values())
        L.append(f"- **Axes that survive as PRIMARY_CARRY_FORWARD in ALL pilots**: "
                 f"{', '.join(f'`{a}`' for a in sorted(common)) or '— none —'}")
        only_in_one = sorted(union - common)
        if only_in_one:
            L.append(f"- Axes that are PRIMARY in at least one but not all pilots: "
                     f"{', '.join(f'`{a}`' for a in only_in_one)}")
        L.append("")

    # E. Most important scientific consequence
    L.append("## E. Most important scientific consequence")
    L.append("")
    consequences: list[str] = []
    # auto-derive from the data
    for p in pilots:
        rs_by_axis = {r["axis"]: r for r in matrix_rows if r["pilot_name"] == p}
        for axis, rec in rs_by_axis.items():
            oc: RuleOutcome = rec["_outcome"]
            if oc.rule_applied == RULE_PRESERVED_UPGRADE:
                consequences.append(
                    f"- **{p} × `{axis}`** — preserved at TIER_1 with PHYSICS_AWARE_UPGRADE: "
                    f"the observed signal is conservative relative to likely biology "
                    f"(substrate suppresses this channel)."
                )
            elif oc.rule_applied == RULE_CONFLICT_OVERRIDE and rec["tier"] == TIER_T1:
                consequences.append(
                    f"- **{p} × `{axis}`** — was Step 2 TIER_1_ROBUST but is now "
                    f"TIER_3_UNINTERPRETABLE because substrate evidence is "
                    f"CONFLICTING. Strong |d| cannot be claimed as biology."
                )
            elif oc.rule_applied == RULE_BIAS_DOWNGRADE and rec["tier"] == TIER_T1:
                consequences.append(
                    f"- **{p} × `{axis}`** — was Step 2 TIER_1_ROBUST, downgraded to "
                    f"TIER_2 because the substrate axis carries a bias caveat "
                    f"(visibility=`{rec['_overlay_row'].get('visibility_tag','')}`, "
                    f"abundance=`{rec['_overlay_row'].get('abundance_interpretation','')}`)."
                )
            elif oc.rule_applied == RULE_AMBIGUOUS_CAP and rec["tier"] == TIER_T1:
                consequences.append(
                    f"- **{p} × `{axis}`** — was Step 2 TIER_1_ROBUST, capped at TIER_2 "
                    f"because the substrate engine reads this axis as AMBIGUOUS "
                    f"(non-biological / both-direction CONVERGED)."
                )
    if consequences:
        for c in consequences:
            L.append(c)
    else:
        L.append("- _no T1 axis was reclassified by Step 2.1._")
    L.append("")

    # F. What remains unchanged
    L.append("## F. What remains unchanged")
    L.append("")
    L.append("- BSV / ΔBSV per spectrum")
    L.append("- per-axis effect sizes, CIs, sample-level consistency, batch-level "
             "consistency, entanglement metrics")
    L.append("- the canonical Step 2 reliability matrix (preserved verbatim under "
             "`step2_axis_reliability_v1/`)")
    L.append("- scorer / atlas / axes / windows / preprocessing")
    L.append("- Stage 2 overlay outputs (Step 2.1 reads them, never writes them)")
    L.append("")
    L.append("Step 2.1 is purely **classification on top of classification** — "
             "the underlying measured geometry is untouched.")
    L.append("")

    # Outputs index
    L.append("## Outputs")
    L.append("")
    L.append(f"- `{Path(summary_csv).name}` — per-pilot reclassification census")
    L.append(f"- `{Path(changes_csv).name}` — per-axis change_type and rule applied")
    L.append("- `axis_reliability_matrix_step2_1.csv` — full per-(pilot, axis) matrix")
    L.append("- `fig_step2_vs_step2_1_heatmap.png`")
    L.append("- `fig_step2_1_change_reason_panel.png`")
    L.append("- `fig_step2_1_carry_forward_axes.png`")

    out = OUT_DIR / "REPORT_step2_1_substrate_aware_reliability_v1.md"
    out.write_text("\n".join(L))
    return out


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n[Step 2.1 substrate-aware reliability v1]")
    print("─" * 76)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    before = _snapshot_inputs()

    rows = build_matrix()
    print(f"loaded {len(rows)} (pilot, axis) cells from Step 2 matrix")

    matrix_csv  = write_matrix(rows)
    summary_csv = write_summary(rows)
    changes_csv = write_changes(rows)
    print(f"  wrote {matrix_csv.name}")
    print(f"  wrote {summary_csv.name}")
    print(f"  wrote {changes_csv.name}")

    fig_a = fig_step2_vs_step2_1_heatmap(rows, OUT_DIR)
    fig_b = fig_change_reason_panel(rows, OUT_DIR)
    fig_c = fig_carry_forward_axes(rows, OUT_DIR)
    print(f"  wrote {fig_a.name}, {fig_b.name}, {fig_c.name}")

    report = write_report(rows, summary_csv, changes_csv)
    print(f"  wrote {report.name}")

    after = _snapshot_inputs()
    _gate(before, after)
    print("[gate] Step 2 + Stage 2 input checksums unchanged ✓")

    print()
    print("─" * 76)
    print("[Step 2.1] complete")
    print(f"  outputs: {OUT_DIR}")


if __name__ == "__main__":
    main()
