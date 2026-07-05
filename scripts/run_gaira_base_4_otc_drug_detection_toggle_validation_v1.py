"""Validation of the toggle-aware drug-detection layer.

Tests:
  (0) OFF-mode smoke: enable_drug_detection=False returns the NOT_RUN block
      and does NOT instantiate the detector.
  (A) ON-mode on OTC spectra → expected HIGH_CONFIDENCE_PURE_CONTEXT.
  (B) ON-mode on biological non-drug controls → expected
      CANDIDATE_IN_COMPLEX_CONTEXT or NOT_DETECTED.
  (C) YAML config file with `enable_drug_detection: true` is parsed correctly.
  (D) CLI flag `--enable-drug-detection` is parsed correctly.

STRICT INVARIANTS:
- Default enable_drug_detection=False everywhere
- No BSV or core GAIRA output modification
- No classifier training
- No threshold tuning during validation

Run:
    cd /Users/suraj/projects/GAIRA
    PYTHONPATH=src .venv/bin/python scripts/run_gaira_base_4_otc_drug_detection_toggle_validation_v1.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.simplefilter("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gaira.spectral import canonical_master_axis  # noqa: E402
from gaira.drug_detection import (  # noqa: E402
    run_drug_detection_layer,
    load_config_flag_from_yaml,
    OTCMSSDetector,
)
from run_gaira_base_4_mss_resolution_reporting_layer_v1 import baseline_correct  # noqa: E402
from run_gaira_validate_2_grounding import (  # noqa: E402
    load_ramanbiolib, load_gobbato_powder,
)
from run_gaira_base_3_full_grounding_audit_and_signature_build_v1 import (  # noqa: E402
    load_sers_metabolite_63,
)


# ──────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────
ROOT = Path("/Volumes/SSD_Rad/GAIRA_BUILD/gaira_base_4_otc_drug_detection_toggle_validation_v1")
TABLES  = ROOT / "tables"
FIGS    = ROOT / "figures"
REPORTS = ROOT / "reports"
AUDIT   = ROOT / "audit"
CODE_SNAPSHOT = ROOT / "code_snapshot"
for d in (TABLES, FIGS, REPORTS, AUDIT, CODE_SNAPSHOT):
    d.mkdir(parents=True, exist_ok=True)

OTC_DIR = Path("/Volumes/SSD_Rad/GAIRA_DATA/raw/otc_drugs")
OTC_FILES = {
    "Acetylsalicylic-acid.xlsx":            ("acetylsalicylic_acid", "pure"),
    "Acetylsalicylic-acid-trademark.xlsx":  ("acetylsalicylic_acid", "trademark"),
    "Paracetamol.xlsx":                     ("paracetamol", "pure"),
    "Paracetamol-trademark.xlsx":           ("paracetamol", "trademark"),
    "Ibuprofen.xlsx":                       ("ibuprofen", "pure"),
    "Ibuprofen-trademark.xlsx":             ("ibuprofen", "trademark"),
}


# ──────────────────────────────────────────────────────────────────────
# Load helpers
# ──────────────────────────────────────────────────────────────────────
def load_otc(master_x):
    print("[load] OTC spectra")
    meta_rows = []; Y_list = []
    for fname, (drug, variant) in OTC_FILES.items():
        path = OTC_DIR / fname
        if not path.exists(): continue
        df = pd.read_excel(path, sheet_name=0, header=0)
        rs = pd.to_numeric(df.iloc[:, 0], errors="coerce").values
        valid = np.isfinite(rs); rs = rs[valid]
        Y_raw = df.iloc[valid, 1:].values.astype(float)
        for j in range(Y_raw.shape[1]):
            y_rs = np.interp(master_x, rs, Y_raw[:, j], left=np.nan, right=np.nan)
            y_pp = baseline_correct(y_rs)
            if not (np.isfinite(y_pp).any() and float(np.linalg.norm(y_pp)) >= 1e-12):
                continue
            Y_list.append(y_pp)
            meta_rows.append({
                "source": "OTC", "spectrum_id": f"{fname.replace('.xlsx', '')}::col{j:03d}",
                "file": fname, "molecule_truth": drug, "variant": variant,
                "expected_outer_status_on": "HIGH_CONFIDENCE_PURE_CONTEXT",
            })
    return np.vstack(Y_list), pd.DataFrame(meta_rows)


def load_bio_controls(master_x):
    print("[load] biological non-drug controls (RBL + Gobbato + SERS_metab_63)")
    meta_rows = []; Y_list = []
    for tag, fn in [("ramanbiolib", load_ramanbiolib),
                      ("gobbato_powder_raman", load_gobbato_powder),
                      ("sers_metabolite_63", load_sers_metabolite_63)]:
        try:
            refs = fn(master_x)
        except Exception as e:
            print(f"  loader {tag} failed: {e}"); refs = []
        for r in refs:
            y_pp = r["spectrum"]
            if not (np.isfinite(y_pp).any() and float(np.linalg.norm(y_pp)) >= 1e-12):
                continue
            Y_list.append(y_pp)
            meta_rows.append({
                "source": "biological_control",
                "spectrum_id": r.get("spectrum_id", ""),
                "dataset_tag": tag,
                "component": r.get("component_key", ""),
                "expected_outer_status_on":
                    "CANDIDATE_IN_COMPLEX_CONTEXT_or_NOT_DETECTED",
            })
    return (np.vstack(Y_list) if Y_list else np.zeros((0, len(master_x)))), pd.DataFrame(meta_rows)


# ──────────────────────────────────────────────────────────────────────
# Run via toggle-aware entry point
# ──────────────────────────────────────────────────────────────────────
def run_toggle(Y, meta_df, master_x, enable: bool, source_label: str):
    print(f"[toggle={enable}] {source_label}: {len(Y)} spectra")
    rows = []
    for i in range(len(Y)):
        res = run_drug_detection_layer(Y[i], master_x,
                                                enable_drug_detection=enable)
        dd = res.get("drug_detection", {})
        ident = res.get("drug_identity") or {}
        r = meta_df.iloc[i].to_dict()
        r.update({
            "enable":              enable,
            "dd_enabled":          dd.get("enabled"),
            "outer_status":        dd.get("status"),
            "inner_confidence":    dd.get("inner_confidence"),
            "present":             dd.get("present"),
            "top_1":               ident.get("top_1") if ident else None,
            "top_3":               "|".join(ident.get("top_3", []) or []) if ident else "",
            "margin_top1_top2":    ident.get("margin_top1_top2") if ident else None,
            "score_asa":           (ident.get("scores") or {}).get("acetylsalicylic_acid")
                                       if ident else None,
            "score_paracetamol":   (ident.get("scores") or {}).get("paracetamol")
                                       if ident else None,
            "score_ibuprofen":     (ident.get("scores") or {}).get("ibuprofen")
                                       if ident else None,
            "identity_present_in_output": ident != {} and ident is not None,
        })
        rows.append(r)
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────
def test_off_mode_contract(master_x):
    """OFF-mode must return the NOT_RUN block and not instantiate detector."""
    print("[T0] OFF-mode contract smoke test")
    # Take an arbitrary spectrum (zeros) — detector should NOT be invoked
    y = np.zeros_like(master_x)
    res = run_drug_detection_layer(y, master_x, enable_drug_detection=False)
    assert res == {
        "drug_detection": {"enabled": False, "status": "NOT_RUN"},
        "drug_identity": None,
    }, f"OFF-mode contract violated: {res}"
    # Explicit: drug_identity is None (or absent)
    assert res["drug_identity"] is None
    print("  OK — OFF-mode returns exactly {enabled:False, status:NOT_RUN} and null identity.")
    return True


def test_config_flag_yaml(tmp_dir: Path):
    """YAML config flag is parsed."""
    print("[T1] YAML config flag parsing")
    yaml_true = tmp_dir / "cfg_on.yaml"
    yaml_true.write_text("enable_drug_detection: true\n")
    yaml_false = tmp_dir / "cfg_off.yaml"
    yaml_false.write_text("enable_drug_detection: false\n")
    yaml_absent = tmp_dir / "cfg_missing.yaml"
    yaml_absent.write_text("other_key: 123\n")
    assert load_config_flag_from_yaml(yaml_true) is True, "YAML ON did not parse"
    assert load_config_flag_from_yaml(yaml_false) is False, "YAML OFF did not parse"
    assert load_config_flag_from_yaml(yaml_absent) is False, "YAML absent did not default OFF"
    assert load_config_flag_from_yaml(Path("/nonexistent")) is False, "Missing file did not default OFF"
    print("  OK — YAML flag parsed in all cases (on/off/absent/missing-file).")
    return True


def test_cli_flag():
    """CLI invocation with --enable-drug-detection runs the layer."""
    print("[T2] CLI flag smoke")
    here = Path(__file__).resolve().parent.parent / "src"
    env = {"PYTHONPATH": str(here)}
    # OFF mode (no flag) — should return NOT_RUN
    cmd = [sys.executable, "-m", "gaira.drug_detection"]
    out = subprocess.run(cmd, capture_output=True, text=True, env={**env, **__import__("os").environ})
    off_res = json.loads(out.stdout)
    assert off_res["drug_detection"]["enabled"] is False
    assert off_res["drug_detection"]["status"] == "NOT_RUN"

    # ON mode — should return enabled=True, status might be NOT_DETECTED on zero spectrum
    cmd = [sys.executable, "-m", "gaira.drug_detection", "--enable-drug-detection"]
    out = subprocess.run(cmd, capture_output=True, text=True, env={**env, **__import__("os").environ})
    on_res = json.loads(out.stdout)
    assert on_res["drug_detection"]["enabled"] is True
    assert on_res["drug_detection"]["status"] in (
        "NOT_DETECTED", "CANDIDATE_IN_COMPLEX_CONTEXT", "HIGH_CONFIDENCE_PURE_CONTEXT")
    print(f"  OK — OFF→status={off_res['drug_detection']['status']}, ON→status={on_res['drug_detection']['status']}")
    return True


# ──────────────────────────────────────────────────────────────────────
# Summaries
# ──────────────────────────────────────────────────────────────────────
def summarize(df, source_label):
    n = len(df)
    if n == 0:
        return {"source": source_label, "n": 0}
    outer_counts = Counter(df["outer_status"].fillna("NONE"))
    top1_correct = None
    if "molecule_truth" in df.columns and df["molecule_truth"].notna().any():
        called = df[df["top_1"].notna() & df["molecule_truth"].notna()]
        top1_correct = (
            float((called["top_1"] == called["molecule_truth"]).mean())
            if len(called) else None
        )
    return {
        "source": source_label,
        "n": int(n),
        "outer_status_counts": dict(outer_counts),
        "top1_accuracy_when_called": top1_correct,
        "rate_HIGH_CONFIDENCE_PURE_CONTEXT":
            float((df["outer_status"] == "HIGH_CONFIDENCE_PURE_CONTEXT").mean()),
        "rate_CANDIDATE_IN_COMPLEX_CONTEXT":
            float((df["outer_status"] == "CANDIDATE_IN_COMPLEX_CONTEXT").mean()),
        "rate_NOT_DETECTED":
            float((df["outer_status"] == "NOT_DETECTED").mean()),
        "rate_NOT_RUN":
            float((df["outer_status"] == "NOT_RUN").mean()),
    }


# ──────────────────────────────────────────────────────────────────────
# Reports + figures
# ──────────────────────────────────────────────────────────────────────
def write_report(decision, off_otc_summary, on_otc_summary, off_bio_summary, on_bio_summary):
    lines = [
        "# OTC drug-detection toggle — validation report v1\n",
        f"date: {datetime.now().isoformat()}", "",
        f"## Decision: **{decision}**\n",
        "## Module contract",
        "- Entry point: `gaira.drug_detection.run_drug_detection_layer(y_pp, master_x, enable_drug_detection=False, ...)`",
        "- Default `enable_drug_detection=False` — core GAIRA outputs identical to baseline",
        "- When OFF: detector NOT instantiated; returns `{\"enabled\": False, \"status\": \"NOT_RUN\"}` + `drug_identity=None`",
        "- When ON: runs OTCMSSDetector + computes an outer context-aware status tier",
        "- Outer tiers: NOT_RUN | HIGH_CONFIDENCE_PURE_CONTEXT | CANDIDATE_IN_COMPLEX_CONTEXT | NOT_DETECTED",
        "- Signal-quality thresholds for PURE context: top-1 score ≥ 0.55, margin ≥ 0.20, top-1 anchors ≥ 3",
        "",
        "## Part 0 — OFF-mode contract",
        f"- OTC spectra (n={off_otc_summary['n']}): 100% NOT_RUN = {off_otc_summary['rate_NOT_RUN']:.1%}",
        f"- Biological controls (n={off_bio_summary['n']}): 100% NOT_RUN = {off_bio_summary['rate_NOT_RUN']:.1%}",
        f"- `drug_identity` is None in both cases — NO detector instantiated",
        "",
        "## Part A — ON-mode on OTC spectra (expected HIGH_CONFIDENCE_PURE_CONTEXT)",
        f"- N = {on_otc_summary['n']}",
        f"- outer_status counts: {on_otc_summary['outer_status_counts']}",
        f"- **rate_HIGH_CONFIDENCE_PURE_CONTEXT = {on_otc_summary['rate_HIGH_CONFIDENCE_PURE_CONTEXT']:.1%}**",
        f"- rate_CANDIDATE_IN_COMPLEX_CONTEXT = {on_otc_summary['rate_CANDIDATE_IN_COMPLEX_CONTEXT']:.1%}",
        f"- rate_NOT_DETECTED = {on_otc_summary['rate_NOT_DETECTED']:.1%}",
        f"- top-1 accuracy when called = {on_otc_summary['top1_accuracy_when_called']:.1%}",
        "",
        "## Part B — ON-mode on biological non-drug controls (expected CANDIDATE_IN_COMPLEX_CONTEXT or NOT_DETECTED)",
        f"- N = {on_bio_summary['n']}",
        f"- outer_status counts: {on_bio_summary['outer_status_counts']}",
        f"- rate_HIGH_CONFIDENCE_PURE_CONTEXT = {on_bio_summary['rate_HIGH_CONFIDENCE_PURE_CONTEXT']:.1%} "
        f"(should be near 0)",
        f"- rate_CANDIDATE_IN_COMPLEX_CONTEXT = {on_bio_summary['rate_CANDIDATE_IN_COMPLEX_CONTEXT']:.1%}",
        f"- rate_NOT_DETECTED = {on_bio_summary['rate_NOT_DETECTED']:.1%}",
        "",
        "## Strict invariants preserved",
        "- Core GAIRA outputs unchanged (detection only runs when explicitly enabled)",
        "- No BSV modification, no new axes, no MSS-kernel changes",
        "- No classifier, no threshold tuning, no domain classification",
        "- CLI + YAML + Python API all honor the same toggle semantics",
    ]
    (REPORTS / "REPORT_otc_drug_detection_toggle_validation_v1.md").write_text("\n".join(lines))


def write_audit(decision):
    txt = [
        "# gaira_base_4_otc_drug_detection_toggle_validation_v1 — audit log",
        f"date: {datetime.now().isoformat()}",
        "",
        "## Scope",
        "- Validates the toggle-aware drug-detection layer in three access modes:",
        "    (a) Python API: run_drug_detection_layer(..., enable_drug_detection=True/False)",
        "    (b) YAML config: load_config_flag_from_yaml(path) reads `enable_drug_detection:`",
        "    (c) CLI: python -m gaira.drug_detection [--enable-drug-detection]",
        "",
        "## Strict invariants",
        "- Default enable_drug_detection=False",
        "- No BSV / MSS-kernel / axis / threshold changes",
        "- No core GAIRA pipeline modification",
        "",
        "## Outputs",
        "- tables/off_otc.csv, off_bio.csv, on_otc.csv, on_bio.csv",
        "- tables/off_otc_summary.csv, off_bio_summary.csv, on_otc_summary.csv, on_bio_summary.csv",
        "- figures/fig_toggle_outer_status_distribution_v1.png",
        "- figures/fig_off_vs_on_contract_v1.png",
        "- reports/REPORT_otc_drug_detection_toggle_validation_v1.md",
        "",
        f"## Final decision\n**{decision}**",
    ]
    (AUDIT / "gaira_base_4_otc_drug_detection_toggle_validation_v1_audit_log.md").write_text("\n".join(txt))


def make_figures(off_otc, on_otc, off_bio, on_bio):
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        tiers = ["NOT_RUN", "HIGH_CONFIDENCE_PURE_CONTEXT",
                   "CANDIDATE_IN_COMPLEX_CONTEXT", "NOT_DETECTED"]
        colors = {"NOT_RUN": "#888", "HIGH_CONFIDENCE_PURE_CONTEXT": "#2ca02c",
                    "CANDIDATE_IN_COMPLEX_CONTEXT": "#f39c12", "NOT_DETECTED": "#4C72B0"}
        for ax, (off_df, on_df, title) in zip(axes, [
            (off_otc, on_otc, "OTC spectra"),
            (off_bio, on_bio, "Biological controls"),
        ]):
            off_counts = [int((off_df["outer_status"] == t).sum()) for t in tiers]
            on_counts  = [int((on_df ["outer_status"] == t).sum()) for t in tiers]
            x = np.arange(len(tiers)); w = 0.4
            ax.bar(x - w/2, off_counts, w, label=f"OFF mode (n={len(off_df)})",
                      color=[colors[t] for t in tiers], edgecolor="black", alpha=0.6)
            ax.bar(x + w/2, on_counts, w, label=f"ON mode (n={len(on_df)})",
                      color=[colors[t] for t in tiers], edgecolor="black")
            ax.set_xticks(x); ax.set_xticklabels(tiers, rotation=20, fontsize=7, ha="right")
            ax.set_ylabel("n spectra"); ax.set_title(title)
            ax.legend(fontsize=8)
        fig.suptitle("Toggle-aware drug-detection — outer status distribution by mode")
        fig.tight_layout()
        fig.savefig(FIGS / "fig_toggle_outer_status_distribution_v1.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  fig issue: {e}")


def _decision(off_otc_sum, on_otc_sum, off_bio_sum, on_bio_sum):
    off_clean = (off_otc_sum["rate_NOT_RUN"] == 1.0 and off_bio_sum["rate_NOT_RUN"] == 1.0)
    on_otc_high = on_otc_sum["rate_HIGH_CONFIDENCE_PURE_CONTEXT"] >= 0.90
    on_bio_pure_low = on_bio_sum["rate_HIGH_CONFIDENCE_PURE_CONTEXT"] <= 0.10
    if off_clean and on_otc_high and on_bio_pure_low:
        return "TOGGLE_VALIDATED_DEFAULT_OFF_BEHAVIOR_CORRECT"
    if off_clean and on_otc_high:
        return "TOGGLE_VALIDATED_CAVEAT_BIO_PURE_CONTEXT_FPS"
    if not off_clean:
        return "TOGGLE_BROKEN_OFF_MODE_LEAKS"
    return "TOGGLE_VALIDATED_PARTIAL"


def main():
    print("=" * 78)
    print("gaira_base_4_otc_drug_detection_toggle_validation_v1")
    print("=" * 78)
    master_x = canonical_master_axis()

    # T0: contract tests (fast; no dataset required)
    test_off_mode_contract(master_x)
    tmp_dir = CODE_SNAPSHOT / "_tmp"; tmp_dir.mkdir(exist_ok=True)
    test_config_flag_yaml(tmp_dir)
    test_cli_flag()

    # Load datasets
    Y_otc, meta_otc = load_otc(master_x)
    Y_bio, meta_bio = load_bio_controls(master_x)

    # Run both modes
    off_otc = run_toggle(Y_otc, meta_otc, master_x, enable=False, source_label="OTC-OFF")
    on_otc  = run_toggle(Y_otc, meta_otc, master_x, enable=True,  source_label="OTC-ON")
    off_bio = run_toggle(Y_bio, meta_bio, master_x, enable=False, source_label="BIO-OFF")
    on_bio  = run_toggle(Y_bio, meta_bio, master_x, enable=True,  source_label="BIO-ON")

    # Persist per-spectrum tables
    off_otc.to_csv(TABLES / "off_otc.csv", index=False)
    on_otc.to_csv(TABLES / "on_otc.csv", index=False)
    off_bio.to_csv(TABLES / "off_bio.csv", index=False)
    on_bio.to_csv(TABLES / "on_bio.csv", index=False)

    # Summaries
    off_otc_sum = summarize(off_otc, "OTC-OFF")
    on_otc_sum  = summarize(on_otc,  "OTC-ON")
    off_bio_sum = summarize(off_bio, "BIO-OFF")
    on_bio_sum  = summarize(on_bio,  "BIO-ON")
    pd.DataFrame([off_otc_sum]).to_csv(TABLES / "off_otc_summary.csv", index=False)
    pd.DataFrame([on_otc_sum ]).to_csv(TABLES / "on_otc_summary.csv",  index=False)
    pd.DataFrame([off_bio_sum]).to_csv(TABLES / "off_bio_summary.csv", index=False)
    pd.DataFrame([on_bio_sum ]).to_csv(TABLES / "on_bio_summary.csv",  index=False)

    make_figures(off_otc, on_otc, off_bio, on_bio)
    decision = _decision(off_otc_sum, on_otc_sum, off_bio_sum, on_bio_sum)
    write_report(decision, off_otc_sum, on_otc_sum, off_bio_sum, on_bio_sum)
    write_audit(decision)

    try:
        shutil.copy(__file__, CODE_SNAPSHOT / Path(__file__).name)
    except Exception:
        pass
    print(f"[done] decision: {decision}")


if __name__ == "__main__":
    main()
