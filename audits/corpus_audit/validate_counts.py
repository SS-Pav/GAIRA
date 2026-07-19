"""Reconciliation checks over the audit CSVs. Ensures totals add up and role
separation holds. Read-only. Prints PASS/FAIL and writes data_audit/reconciliation.json.
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

DA = Path("/Users/surajpg/projects/GAIRA/data_audit")
checks = []


def chk(name, cond, detail=""):
    checks.append({"check": name, "pass": bool(cond), "detail": detail})


# grounding: 202 rows, 141 unique
gt = json.loads((DA/"grounding_totals.json").read_text())
chk("202_table_rows", gt["n_rows_202_table"] == 202, f"rows={gt['n_rows_202_table']}")
chk("202_unique_analytes_141", gt["unique_analyte_names"] == 141, f"unique={gt['unique_analyte_names']}")
chk("202_duplicates_61", gt["duplicate_analyte_names"] == 61, f"dups={gt['duplicate_analyte_names']}")
chk("measured_grounding_160_summary", gt["total_measured_reference_spectra"] == 160,
    f"summary-level measured={gt['total_measured_reference_spectra']} (DB grounding_metadata=468)")

# calibration registry: 7 core datasets (+OTC extra)
cal = pd.read_csv(DA/"calibration_dataset_registry.csv")
chk("calibration_datasets_ge7", len(cal) >= 7, f"n={len(cal)}")

# axis coverage: exactly 11 axes; 3 supportive/partial
ax = pd.read_csv(DA/"axis_calibration_coverage.csv")
chk("axis_coverage_11", len(ax) == 11, f"n={len(ax)}")
sup = ax[ax["verdict"].str.contains("supportive")]
chk("axes_with_support_3", len(sup) == 3, f"supportive axes={sup['axis'].tolist()}")

# biological: independent human samples ~760
bio = pd.read_csv(DA/"biological_dataset_registry.csv")
chk("biological_datasets_13", len(bio) == 13, f"n={len(bio)}")

# canonical registry role separation
can = pd.read_csv(DA/"canonical_dataset_registry.csv")
roles = can["top_level_role"].value_counts().to_dict()
chk("canonical_has_all_roles",
    all(r in roles for r in ["molecular_grounding", "calibration", "biological_mixtures", "supporting_literature"]),
    str(roles))

# corpus totals separation: 5 roles, spectrum != analyte != patient
tot = pd.read_csv(DA/"gaira_corpus_totals.csv")
chk("corpus_totals_5_roles", len(tot) == 5, f"roles={tot['role'].tolist()}")

# big-number reconciliation: augmented + technical dominate biological
chk("big_number_augmentation_flag",
    any("AUGMENT" in str(x).upper() for x in bio["known_leakage_or_duplication_risk"]),
    "small2023 flagged augmented")

# runtime: demo does NOT read the duckdb; src_gaira does
rt = pd.read_csv(DA/"runtime_dataset_usage.csv")
duck = rt[rt["dataset"].str.contains("duckdb")]
chk("demo_does_not_read_duckdb", (duck["v3_1_tab"].str.contains("NONE").all()) if len(duck) else False,
    "gaira.duckdb v3_1_tab=NONE")

# substrate: production engine dormant
sub = pd.read_csv(DA/"substrate_physics_rules.csv")
dormant = sub[sub["classification"].str.contains("engine")]
chk("prod_substrate_dormant", (dormant["used_in_src_gaira"].str.contains("no_importer").any()) if len(dormant) else False,
    "src/gaira substrate engine has no importer")

n_pass = sum(c["pass"] for c in checks)
(DA/"reconciliation.json").write_text(json.dumps({"checks": checks, "passed": n_pass, "total": len(checks)}, indent=2))
for c in checks:
    print(("PASS" if c["pass"] else "FAIL"), c["check"], "—", c["detail"])
print(f"\n{n_pass}/{len(checks)} reconciliation checks passed")
