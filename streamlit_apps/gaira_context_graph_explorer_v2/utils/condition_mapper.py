"""Map raw `condition_A` strings + dataset hints → specific condition labels.

Rules are loaded from `config/condition_mapping.yaml`. The first matching
rule wins. Used to enrich the events table with a `specific_condition`
column for v2's cohort-aware views.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import yaml


def load_rules(yaml_path: Path) -> list[dict]:
    if not yaml_path.exists():
        return []
    with yaml_path.open() as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("specific_conditions", [])


def _match(rule: dict, condition_a: str, dataset: str,
           comparison_type: str) -> bool:
    # dataset substr filter
    ds_subs = rule.get("dataset_substr")
    if ds_subs:
        if not any(s.lower() in dataset.lower() for s in ds_subs):
            return False
    # comparison_type filter
    ct_in = rule.get("comparison_type_in")
    if ct_in and comparison_type not in ct_in:
        return False
    # regex first (more specific)
    rx = rule.get("regex")
    if rx and re.search(rx, condition_a):
        return True
    # contains list (case-insensitive)
    contains = rule.get("contains")
    if contains:
        ca = condition_a.lower()
        for tok in contains:
            if tok.lower() in ca:
                return True
        # If only contains was set, this rule needs at least one hit
        if rx is None and ds_subs is None and ct_in is None:
            return False
        # If we got here and only had dataset/comparison restrictions but no
        # regex/contains hit, we still consider it a fallback hit (the dataset
        # gate is enough).
    # If the rule has only dataset_substr / comparison_type filters and we
    # passed those, treat it as a match.
    return contains is None and rx is None


def derive_specific_condition(condition_a: object, dataset: str,
                               comparison_type: object,
                               rules: list[dict]) -> tuple[str, str, str]:
    """Return (specific_condition, mapped_sample_type, mapped_condition_family).

    `mapped_sample_type` / `mapped_condition_family` come from the matching
    rule when set (allows correcting EV vs serum tags). Empty string when
    the rule didn't override.
    """
    ca = str(condition_a) if condition_a is not None else ""
    ct = str(comparison_type) if comparison_type is not None else ""
    for rule in rules:
        if _match(rule, ca, dataset, ct):
            return (str(rule.get("label", "unmapped")),
                    str(rule.get("sample_type", "")),
                    str(rule.get("condition_family", "")))
    return ("unmapped", "", "")


def attach_specific_conditions(events: pd.DataFrame,
                                rules: list[dict]) -> pd.DataFrame:
    if events is None or events.empty:
        return events
    out_labels: list[str] = []
    out_st: list[str] = []
    out_cf: list[str] = []
    for _, r in events.iterrows():
        lbl, st, cf = derive_specific_condition(
            r.get("condition_A", ""),
            str(r.get("dataset", "")),
            r.get("comparison_type", ""),
            rules)
        out_labels.append(lbl)
        out_st.append(st or str(r.get("sample_type", "")))
        out_cf.append(cf or str(r.get("condition_family", "")))
    out = events.copy()
    out["specific_condition"] = out_labels
    out["sample_type_v2"] = out_st
    out["condition_family_v2"] = out_cf
    return out
