"""Reference-space coverage: NA preserved (never shown as 0), grounding status
present, resolved axes carry integer analyte counts, insufficiency flagged."""
from pathlib import Path

import pandas as pd

DEMO_ROOT = Path(__file__).resolve().parent.parent
from gaira_core import config as cfg
from gaira_core.ontology import ontology

COVERAGE = DEMO_ROOT / "data" / "generated" / "axis_reference_coverage_v1.csv"


def test_coverage_table_shape_and_na():
    assert COVERAGE.exists(), "axis coverage artifact missing"
    df = pd.read_csv(COVERAGE, keep_default_na=False)
    assert len(df) == 11
    # NA preserved as literal text, never 0
    na_axes = df[df["unique_reference_analytes"] == "NA"]["axis_short"].tolist()
    assert set(na_axes) >= {"Purine-nuc", "Purine-met", "Lipid", "Sterol", "Redox", "Metabolite"}
    # resolved axes carry integer counts
    for axis_short, expected in [("Pyrimidine", "13"), ("Nuc-phosphate", "31"),
                                 ("Glycan", "25"), ("Protein", "81"), ("Aromatic", "12")]:
        v = df[df.axis_short == axis_short]["unique_reference_analytes"].iloc[0]
        assert str(v) == expected, f"{axis_short}: {v} != {expected}"


def test_grounding_status_matches_ontology():
    df = pd.read_csv(COVERAGE, keep_default_na=False).set_index("axis")
    onto = ontology()
    for ax in cfg.BSV_AXES:
        assert df.loc[ax, "ontology_independence_status"] == onto.status_of(ax)


def test_insufficient_flag_present():
    df = pd.read_csv(COVERAGE, keep_default_na=False)
    flagged = df[df["insufficient_grounding_flag"] == True]["axis_short"].tolist()  # noqa: E712
    # the three legacy split families must be flagged
    assert set(flagged) >= {"Purine-nuc", "Purine-met", "Lipid", "Sterol", "Redox", "Metabolite"}
