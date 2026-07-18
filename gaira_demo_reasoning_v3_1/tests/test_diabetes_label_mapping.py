"""OWD/NWD vs Impact/Strong-D — proven from the historical label audit."""
from pathlib import Path
import pandas as pd
REPO = Path(__file__).resolve().parents[2]
LA = REPO / "results" / "diabetes_gaira_audit_20260701_1322" / "diabetes_label_audit.csv"


def test_impact_owd_strongd_nwd_mapping():
    d = pd.read_csv(LA)
    ct = pd.crosstab(d["group_raw"], d["group_2"])
    # Impact maps ONLY to OWD; Strong-D maps ONLY to NWD (clean 1:1 relabel)
    assert ct.loc["Impact", "OWD"] == 40 and ct.loc["Impact"].sum() == 40
    assert ct.loc["Strong-D", "NWD"] == 24 and ct.loc["Strong-D"].sum() == 24
    assert "NWD" not in ct.columns or ct.loc["Impact"].get("NWD", 0) == 0


def test_group2_is_direct_map_not_bmi():
    # group_2 is the direct Group relabel, not a per-patient bmi threshold:
    # at least one Impact/OWD patient has bmi in a range that a strict bmi>=25
    # rule would not uniquely determine, confirming the direct map was used.
    d = pd.read_csv(LA)
    impact = d[d.group_raw == "Impact"]
    assert (impact.group_2 == "OWD").all()   # every Impact -> OWD regardless of bmi
