"""Reproduce the later historical (1322) cohort z-score table numerically."""
from pathlib import Path
import numpy as np, pandas as pd
DEMO = Path(__file__).resolve().parent.parent
REPO = DEMO.parent
from gaira_core import config as cfg
AXES = list(cfg.BSV_AXES)
H = REPO / "results" / "diabetes_gaira_audit_20260701_1322"


def _cohort_z(bsv, col):
    pm = bsv[AXES].mean(); ps = bsv[AXES].std(ddof=1).replace(0, np.nan)
    return {c: ((sub[AXES].mean() - pm) / ps).to_dict() for c, sub in bsv.groupby(col)}


def test_zscore_2group_reproduced_within_1e9():
    bsv = pd.read_csv(H / "diabetes_gaira_scores_per_sample.csv")
    saved = pd.read_csv(H / "diabetes_zscore_2group.csv").set_index("cohort")
    repro = _cohort_z(bsv, "group_2")
    mx = 0.0
    for coh in saved.index:
        for a in AXES:
            mx = max(mx, abs(float(saved.loc[coh, a]) - float(repro[coh][a])))
    assert mx <= 1e-9, f"z-score reproduction drift {mx:.2e}"


def test_group_means_and_axis_ordering_reproduced():
    bsv = pd.read_csv(H / "diabetes_gaira_scores_per_sample.csv")
    gs = pd.read_csv(H / "diabetes_group_summary_2group.csv")
    means = bsv.groupby("group_2")[AXES].mean()
    for _, r in gs.iterrows():
        assert abs(means.loc["OWD", r["axis"]] - r["mean_OWD"]) <= 1e-9
        assert abs(means.loc["NWD", r["axis"]] - r["mean_NWD"]) <= 1e-9
    # strongest effect is sterol (not redox)
    top = gs.reindex(gs["cohens_d"].abs().sort_values(ascending=False).index).iloc[0]
    assert top["axis"] == "G09_sterol_neutral_lipid"
