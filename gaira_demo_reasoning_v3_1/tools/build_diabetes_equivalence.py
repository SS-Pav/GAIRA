"""GAIRA V3.1 — diabetes three-path equivalence experiment.

Answers the core question: did the historical "better" EV-diabetes radar come
from AXIS-WISE NORMALIZATION, from a DIFFERENT UPSTREAM BSV ENGINE, or both?

Three BSV construction paths on the EV-diabetes cohort:
  Path B — V3 plain build_report (the current unchanged demo engine)
  Path A — build_report_diabetes (historical engine: tightened G10 490-505 +
           co-band-gated Ag-SERS thiol boost)  [analysis/_diabetes_overrides.py]
  Historical tables — 1322 (later, = Path A engine) and 1304 (earlier, = plain).

Path A and Path B are computed on the SAME 63 spectra (sample_query_spectra.csv,
the source V3 uses), so their difference isolates the ENGINE effect. Historical
saved tables used .mat-derived mean spectra, so historical-vs-V3 residuals also
carry an INPUT-SPECTRA effect (reported separately).

Also reproduces the exact historical cohort z-score normalization
(z = (cohort_mean − pool_mean)/pool_sd, ddof=1, pooled over all 63) and five
normalization variants, and quantifies redox dominance under each.

Read-only w.r.t. the historical folders. Outputs -> data/generated/diabetes_equivalence/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEMO_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = DEMO_ROOT.parent
sys.path.insert(0, str(DEMO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "analysis"))

from gaira_core import config as cfg                       # noqa: E402
from gaira_core import global_coordinates as gc            # noqa: E402
from gaira_core.report_builder import build_report          # plain V3 engine
from _diabetes_overrides import build_report_diabetes       # historical engine

AXES = list(cfg.BSV_AXES)
OUT = cfg.GENERATED_DIR / "diabetes_equivalence"
OUT.mkdir(parents=True, exist_ok=True)
H1322 = REPO_ROOT / "results" / "diabetes_gaira_audit_20260701_1322"
H1304 = REPO_ROOT / "results" / "diabetes_gaira_audit_20260701_1304"
GRID = np.linspace(cfg.WAVENUMBER_MIN, cfg.WAVENUMBER_MAX, cfg.WAVENUMBER_N)


def load_ev_spectra():
    p = cfg.EV_DIABETES_TABLES / "sample_query_spectra.csv"
    sp = pd.read_csv(p)
    rows = []
    for _, r in sp.iterrows():
        wn = np.asarray(json.loads(r["wavenumbers_json"]), float)
        y = np.asarray(json.loads(r["intensity_json"]), float)
        o = np.argsort(wn)
        yg = np.clip(np.interp(GRID, wn[o], y[o], left=0, right=0), 0, None)
        rows.append((str(r["sample_id"]), str(r["class_label"]), yg))
    return rows


def cohort_z(bsv_df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Exact historical formula (run_diabetes_gaira_audit._compute_zscore):
    z = (cohort_mean − pool_mean) / pool_sd  ; pool over ALL rows, ddof=1."""
    pool_mean = bsv_df[AXES].mean()
    pool_sd = bsv_df[AXES].std(ddof=1).replace(0, np.nan)
    out = []
    for coh, sub in bsv_df.groupby(group_col):
        z = (sub[AXES].mean() - pool_mean) / pool_sd
        out.append({"cohort": coh, "n": len(sub), **z.to_dict()})
    return pd.DataFrame(out)


def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    sp = np.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1)) / (len(a)+len(b)-2))
    return float((a.mean()-b.mean())/sp) if sp > 1e-12 else 0.0


def redox_rank(means_by_axis: dict, redox="G10_sulfur_thiol_redox") -> int:
    order = sorted(AXES, key=lambda a: abs(means_by_axis[a]), reverse=True)
    return order.index(redox) + 1


def main():
    spectra = load_ev_spectra()
    # ── Path A / Path B on identical spectra ──
    recs = []
    for sid, label, y in spectra:
        b = build_report(sample_id=sid, title=sid, domain="extracellular_vesicle",
                         substrate="Ag colloid SERS", wavenumber=GRID, intensity=y)["bsv"]
        a = build_report_diabetes(sample_id=sid, title=sid, domain="extracellular_vesicle",
                                  substrate="Ag colloid SERS", wavenumber=GRID, intensity=y)["bsv"]
        rec = {"sample_id": sid, "class_label": label,
               "group_2": "OWD" if label == "Impact" else "NWD"}
        for ax in AXES:
            rec[f"pathB_plain_{ax}"] = float(b[ax])
            rec[f"pathA_diabetes_{ax}"] = float(a[ax])
        recs.append(rec)
    per = pd.DataFrame(recs)
    per.to_csv(OUT / "path_comparison_per_sample.csv", index=False)

    # ── historical 1304 vs 1322 (patient_id merge; exact) ──
    h04 = pd.read_csv(H1304 / "diabetes_gaira_scores_per_sample.csv")
    h22 = pd.read_csv(H1322 / "diabetes_gaira_scores_per_sample.csv")
    m = h04[["patient_id"] + AXES].merge(h22[["patient_id"] + AXES], on="patient_id",
                                         suffixes=("_1304", "_1322"))
    hist_cmp = [{"axis": ax, "axis_short": cfg.axis_short(ax),
                 "max_abs_1304_vs_1322": float((m[f"{ax}_1304"] - m[f"{ax}_1322"]).abs().max()),
                 "mean_1304": float(m[f"{ax}_1304"].mean()), "mean_1322": float(m[f"{ax}_1322"].mean())}
                for ax in AXES]
    pd.DataFrame(hist_cmp).to_csv(OUT / "historical_v1_vs_v2_analysis_comparison.csv", index=False)

    # ── historical_vs_v3_raw_bsv_per_sample: align V3 recomputed (Path A) to 1322 by
    #    nearest-neighbor within group (same engine) ──
    align_rows = []
    for grp_hist, grp_v3 in [("OWD", "OWD"), ("NWD", "NWD")]:
        H = h22[h22["group_2"] == grp_hist].reset_index(drop=True)
        V = per[per["group_2"] == grp_v3].reset_index(drop=True)
        Hm = H[AXES].to_numpy(float)
        Vm = V[[f"pathA_diabetes_{a}" for a in AXES]].to_numpy(float)
        used = set()
        for i in range(len(V)):
            dists = np.linalg.norm(Hm - Vm[i], axis=1)
            for j in np.argsort(dists):
                if j not in used:
                    used.add(j); break
            row = {"group_2": grp_hist, "v3_sample_id": V.iloc[i]["sample_id"],
                   "hist_patient_id": H.iloc[j]["patient_id"], "match_dist": float(dists[j])}
            for a in AXES:
                row[f"hist1322_{a}"] = float(H.iloc[j][a])
                row[f"v3_pathA_{a}"] = float(Vm[i, list(AXES).index(a)])
                row[f"v3_pathB_{a}"] = float(V.iloc[i][f"pathB_plain_{a}"])
            align_rows.append(row)
    align = pd.DataFrame(align_rows)
    align.to_csv(OUT / "historical_vs_v3_raw_bsv_per_sample.csv", index=False)

    # ── axiswise correlations ──
    corr = []
    for ax in AXES:
        A = per[f"pathA_diabetes_{ax}"].to_numpy(float)
        B = per[f"pathB_plain_{ax}"].to_numpy(float)
        def _p(x, y):
            return float(np.corrcoef(x, y)[0, 1]) if x.std() > 1e-12 and y.std() > 1e-12 else np.nan
        def _s(x, y):
            return _p(pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy())
        # engine effect (A vs B, same spectra)
        # input+engine (v3 pathA vs hist1322, aligned)
        hA = align[f"hist1322_{ax}"].to_numpy(float); vA = align[f"v3_pathA_{ax}"].to_numpy(float)
        corr.append({
            "axis": ax, "axis_short": cfg.axis_short(ax),
            "engine_pearson_A_vs_B": _p(A, B), "engine_spearman_A_vs_B": _s(A, B),
            "engine_max_abs_A_vs_B": float(np.max(np.abs(A - B))),
            "engine_mean_abs_A_vs_B": float(np.mean(np.abs(A - B))),
            "v3A_vs_hist1322_pearson": _p(vA, hA),
            "v3A_vs_hist1322_max_abs": float(np.max(np.abs(vA - hA))),
        })
    corr_df = pd.DataFrame(corr)
    corr_df.to_csv(OUT / "axiswise_correlations.csv", index=False)

    # ── group effect comparison across paths + normalization variants ──
    def group_means(df, prefix):
        return {grp: {a: float(sub[f"{prefix}{a}"].mean()) for a in AXES}
                for grp, sub in df.groupby("group_2")}
    def group_d(df, prefix):
        return {a: cohens_d(df[df.group_2=="OWD"][f"{prefix}{a}"], df[df.group_2=="NWD"][f"{prefix}{a}"])
                for a in AXES}

    eff_rows = []
    # raw plain (B), raw diabetes (A)
    for name, prefix, df in [("rawB_plain", "pathB_plain_", per), ("rawA_diabetes", "pathA_diabetes_", per)]:
        gm = group_means(df, prefix); gd = group_d(df, prefix)
        for a in AXES:
            eff_rows.append({"variant": name, "axis": a, "axis_short": cfg.axis_short(a),
                             "mean_OWD": gm["OWD"][a], "mean_NWD": gm["NWD"][a], "cohens_d": gd[a]})
    # historical 1322 & 1304 (from saved per-sample)
    for name, hdf in [("historical_1322", h22), ("historical_1304", h04)]:
        gd = {a: cohens_d(hdf[hdf.group_2=="OWD"][a], hdf[hdf.group_2=="NWD"][a]) for a in AXES}
        gm = {grp: {a: float(sub[a].mean()) for a in AXES} for grp, sub in hdf.groupby("group_2")}
        for a in AXES:
            eff_rows.append({"variant": name, "axis": a, "axis_short": cfg.axis_short(a),
                             "mean_OWD": gm["OWD"][a], "mean_NWD": gm["NWD"][a], "cohens_d": gd[a]})
    pd.DataFrame(eff_rows).to_csv(OUT / "group_effect_comparison.csv", index=False)

    # ── normalization variants on Path A (historical engine) ──
    perA = per.rename(columns={f"pathA_diabetes_{a}": a for a in AXES})
    variants = {}
    # A. raw
    variants["raw"] = {grp: {a: float(sub[a].mean()) for a in AXES} for grp, sub in perA.groupby("group_2")}
    # B. historical cohort z-score (exact formula) on Path A
    hz = cohort_z(perA, "group_2").set_index("cohort")
    variants["historical_cohort_z"] = {c: {a: float(hz.loc[c, a]) for a in AXES} for c in hz.index}
    # C. robust cohort-relative z (median/MAD)
    med = perA[AXES].median(); mad = (perA[AXES] - med).abs().median() * 1.4826
    mad = mad.replace(0, np.nan)
    variants["robust_cohort_z"] = {grp: {a: float(((sub[a].mean() - med[a]) / mad[a])) for a in AXES}
                                    for grp, sub in perA.groupby("group_2")}
    # D. V3 frozen global coordinate (unbounded)
    calib = gc.load_calibration()
    if calib is not None:
        gcoords = perA.copy()
        for a in AXES:
            gcoords[f"g_{a}"] = (perA[a] - calib.center[a]) / max(1e-9, 1)  # placeholder to vectorize below
        gm_g = {}
        for grp, sub in perA.groupby("group_2"):
            gm_g[grp] = {a: float(np.mean([(v - calib.center[a]) / calib.scale[a] for v in sub[a]])) for a in AXES}
        variants["v3_frozen_global"] = gm_g
        # E. cohort-standardized global (z of global coords within diabetes)
        gall = pd.DataFrame({a: [(v - calib.center[a]) / calib.scale[a] for v in perA[a]] for a in AXES})
        gall["group_2"] = perA["group_2"].values
        gpool_mean = gall[AXES].mean(); gpool_sd = gall[AXES].std(ddof=1).replace(0, np.nan)
        variants["cohort_standardized_global"] = {
            grp: {a: float((sub[a].mean() - gpool_mean[a]) / gpool_sd[a]) for a in AXES}
            for grp, sub in gall.groupby("group_2")}

    # EXACT historical reproduction (from the saved 1322 BSV, not recomputed spectra) —
    # this is what drives the default EV view so it matches the audited figure verbatim.
    hz_exact = cohort_z(h22.assign(**{c: h22[c] for c in AXES}), "group_2").set_index("cohort")
    variants["historical_cohort_z_exact"] = {c: {a: float(hz_exact.loc[c, a]) for a in AXES}
                                             for c in hz_exact.index}
    # bundle the exact historical 2-group stats (Cohen's d + Mann-Whitney + BH q) for the UI
    try:
        gs = pd.read_csv(H1322 / "diabetes_group_summary_2group.csv")
        gs.to_csv(OUT / "historical_2group_stats_exact.csv", index=False)
    except Exception:
        pass

    # redox dominance per variant + tidy CSV
    nrows, redox_summary = [], {}
    for vname, gm in variants.items():
        # OWD-NWD signed profile magnitude & redox rank on |OWD| profile
        owd = gm.get("OWD", {a: np.nan for a in AXES})
        redox_summary[vname] = {"redox_rank_OWD_profile": redox_rank({a: owd[a] for a in AXES})}
        for grp in gm:
            for a in AXES:
                nrows.append({"variant": vname, "cohort": grp, "axis": a,
                              "axis_short": cfg.axis_short(a), "value": gm[grp][a]})
    pd.DataFrame(nrows).to_csv(OUT / "normalization_variants.csv", index=False)

    # ── reproduce historical zscore_2group EXACTLY from historical 1322 BSV ──
    hz_repro = cohort_z(h22.assign(**{c: h22[c] for c in AXES}), "group_2")
    saved = pd.read_csv(H1322 / "diabetes_zscore_2group.csv")
    repro_max = 0.0
    for _, sr in saved.iterrows():
        rr = hz_repro[hz_repro["cohort"] == sr["cohort"]].iloc[0]
        for a in AXES:
            repro_max = max(repro_max, abs(float(sr[a]) - float(rr[a])))

    # ── self-contained 2-group stats on Path A (Cohen's d + Mann-Whitney + BH) ──
    from scipy import stats as _st
    stat_rows = []
    for a in AXES:
        x = per[per.group_2 == "OWD"][f"pathA_diabetes_{a}"].to_numpy(float)
        y = per[per.group_2 == "NWD"][f"pathA_diabetes_{a}"].to_numpy(float)
        try:
            U, p = _st.mannwhitneyu(x, y, alternative="two-sided")
        except Exception:
            U, p = np.nan, np.nan
        stat_rows.append({"axis": a, "axis_short": cfg.axis_short(a),
                          "mean_OWD": float(x.mean()), "mean_NWD": float(y.mean()),
                          "delta_OWD_minus_NWD": float(x.mean() - y.mean()),
                          "cohens_d": cohens_d(x, y), "mannwhitney_U": float(U), "p_value": float(p)})
    sdf = pd.DataFrame(stat_rows)
    # Benjamini-Hochberg
    ps = sdf["p_value"].to_numpy(float)
    order = np.argsort(ps); ranks = np.empty_like(order); ranks[order] = np.arange(1, len(ps) + 1)
    q = ps * len(ps) / ranks
    # enforce monotonicity
    q_sorted = q[order]; q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_final = np.empty_like(q); q_final[order] = np.clip(q_sorted, 0, 1)
    sdf["q_value_fdr_bh"] = q_final
    sdf.sort_values("cohens_d", key=lambda s: s.abs(), ascending=False)\
       .to_csv(OUT / "diabetes_2group_stats_pathA_v31.csv", index=False)

    summary = {
        "n_samples": len(per), "n_OWD": int((per.group_2 == "OWD").sum()),
        "n_NWD": int((per.group_2 == "NWD").sum()),
        "engine_effect_axes_nonzero": [a for a in AXES
                                       if float(np.max(np.abs(per[f"pathA_diabetes_{a}"] - per[f"pathB_plain_{a}"]))) > 1e-9],
        "engine_max_abs_by_axis": {a: float(np.max(np.abs(per[f"pathA_diabetes_{a}"] - per[f"pathB_plain_{a}"]))) for a in AXES},
        "historical_1304_vs_1322_nonzero_axes": [r["axis"] for r in hist_cmp if r["max_abs_1304_vs_1322"] > 1e-9],
        "historical_zscore_reproduction_max_abs": repro_max,
        "redox_rank_by_variant": {v: redox_summary[v]["redox_rank_OWD_profile"] for v in redox_summary},
        "label_mapping": {"Impact": "OWD", "Strong-D": "NWD",
                          "rule": "direct group_raw->group_2 map (run_diabetes_gaira_audit.py:205); "
                                  "bmi>=25 rule exists (_map_bmi_group) but was NOT used for group_2"},
    }
    (OUT / "equivalence_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print("=== engine effect (Path A diabetes vs Path B plain, same spectra) ===")
    print("nonzero axes:", summary["engine_effect_axes_nonzero"])
    print("G10 engine max abs:", round(summary["engine_max_abs_by_axis"]["G10_sulfur_thiol_redox"], 4))
    print("historical 1304 vs 1322 nonzero axes:", summary["historical_1304_vs_1322_nonzero_axes"])
    print("historical zscore reproduction max abs diff:", repro_max)
    print("redox rank by variant (1=most dominant):", summary["redox_rank_by_variant"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
