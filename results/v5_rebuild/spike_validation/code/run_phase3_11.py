"""Phases 3-11 — projection into the FROZEN atlas, trajectories, reproducibility,
component activation, serum-vs-pure, linearity, OOD coherence, controls.

The atlas is never modified: its fingerprint is verified before and after.
"""
from __future__ import annotations
import sys, json, time, hashlib, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

REPO = Path("/Users/surajpg/projects/GAIRA"); sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from gaira.foundation import serialization as SER, dataset as DS
import spike_lib as SL

OUT = REPO / "results/v5_rebuild/spike_validation"
TAB, ART = OUT / "tables", OUT / "artifacts"
FROZEN = REPO / "results/v5_rebuild/foundation/artifacts"
AXES = json.loads((FROZEN / "manifold.json").read_text()).get("axes", [])


def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def theme_coords(Z):
    """Hierarchical BSV: sum component shares within each frozen theme axis."""
    cols, names = [], []
    for ax in AXES:
        cols.append(np.clip(Z[:, ax["components"]], 0, None).sum(axis=1))
        names.append(f"axis{ax['axis']}_{ax['tentative_theme']}")
    return (np.vstack(cols).T if cols else np.zeros((len(Z), 0))), names


def ood(Z_ref_unit, Z, k=5):
    U = SL._unit(np.clip(Z, 0, None))
    S = U @ Z_ref_unit.T
    return 1.0 - np.sort(S, axis=1)[:, -k:].mean(axis=1)


def main():
    t0 = time.time()
    atlas = SER.load_frozen_manifold(FROZEN)
    fp0 = atlas.meta["fingerprint"]
    log(f"FROZEN atlas {atlas.name} k={atlas.k} fingerprint {fp0}")

    # in-domain Raman reference cloud (for OOD)
    corpus = DS.load_reference_corpus()
    Zref = atlas.coordinates(corpus.X, normalise=True)
    Zref_u = SL._unit(np.clip(Zref, 0, None))
    ref_analytes = corpus.meta.analyte.values

    blob = np.load(ART / "processed_spectra.npz")
    data = {}
    for k in SL.LOADERS:
        key = f"X_{k}"
        if key in blob and (ART / f"meta_{k}.csv").exists():
            data[k] = (blob[key], pd.read_csv(ART / f"meta_{k}.csv"))

    # ── Phase 3: project everything ──
    log("Phase 3 — projecting into the frozen atlas …")
    proj = {}
    all_rows = []
    for name, (X, m) in data.items():
        Z = atlas.coordinates(X, normalise=True)
        T, tnames = theme_coords(Z)
        o = ood(Zref_u, Z)
        proj[name] = (Z, T, tnames, m)
        df = pd.DataFrame(Z, columns=[f"c{j}" for j in range(atlas.k)])
        for i, tn in enumerate(tnames):
            df[tn] = T[:, i]
        df["ood_score"] = o
        for c in m.columns:
            df[c] = m[c].values
        df.to_csv(TAB / f"phase3_projection_{name}.csv", index=False)
        all_rows.append(pd.DataFrame({"dataset": name, "ood_score": o}))
        log(f"  {name}: {len(Z)} spectra | median OOD {np.median(o):.3f}")
    pd.concat(all_rows).groupby("dataset").ood_score.describe().to_csv(TAB / "phase10_ood_summary.csv")

    # ── Phase 4/5/8/10: dose-response trajectories ──
    log("Phase 4/5/8 — trajectory, reproducibility, linearity …")
    traj_rows, rep_rows = [], []
    # ILS adenine, stratified by substrate x laser (each is its own experiment)
    Z, T, tnames, m = proj["ils_adenine"]
    for (sub, las), idx in m.groupby(["substrate", "laser_nm"]).groups.items():
        pos = [m.index.get_loc(i) for i in idx]
        mm = m.iloc[pos]; Zi = Z[pos]
        concs = mm.conc_uM.values
        if len(np.unique(concs)) < 4:
            continue
        keys, M = SL.group_means(Zi, concs)
        tm = SL.trajectory_metrics(np.array(keys, float), M)
        null = SL.monotonicity_null(np.array(keys, float), Zi, concs, n_perm=500, seed=0)
        fits = SL.dose_response_fits(np.array(keys, float), tm["distance_from_control"])
        traj_rows.append({"experiment": f"ils_adenine::{sub}@{las}nm", "analyte": "adenine",
                          "n_spectra": len(pos), "n_levels": tm["n_levels"],
                          "monotonicity_rho": tm["monotonicity_rho"],
                          "monotonicity_p_perm": null["p_value"],
                          "null_mean_abs_rho": null["null_mean_abs_rho"],
                          "straightness": tm["straightness"],
                          "mean_step_cosine": tm["mean_step_cosine"],
                          "net_displacement": tm["net_displacement"],
                          "total_path": tm["total_path_length"],
                          "best_dose_model": fits.get("best_model"),
                          "linear_r2": fits.get("linear_r2"), "log_r2": fits.get("log_r2"),
                          "saturating_r2": fits.get("saturating_r2"),
                          "median_ood": float(np.median(ood(Zref_u, Zi)))})
        # replicate reproducibility per concentration
        for c in np.unique(concs):
            sel = concs == c
            if sel.sum() < 2:
                continue
            Zc = Zi[sel]
            U = SL._unit(Zc); C = U @ U.T; iu = np.triu_indices(len(U), 1)
            base = M[0]
            disp = SL._unit(Zc - base)
            Cd = disp @ disp.T
            rep_rows.append({"experiment": f"ils_adenine::{sub}@{las}nm", "conc_uM": float(c),
                             "n": int(sel.sum()), "coord_cos_mean": float(C[iu].mean()),
                             "direction_cos_mean": float(Cd[iu].mean()) if len(U) > 1 else np.nan,
                             "coord_var": float(np.mean(np.var(Zc, axis=0)))})
    # ergothioneine
    Z, T, tnames, m = proj["ergothioneine"]
    concs = m.conc_uM.values
    if len(np.unique(concs)) >= 4:
        keys, M = SL.group_means(Z, concs)
        tm = SL.trajectory_metrics(np.array(keys, float), M)
        null = SL.monotonicity_null(np.array(keys, float), Z, concs, n_perm=500, seed=0)
        fits = SL.dose_response_fits(np.array(keys, float), tm["distance_from_control"])
        traj_rows.append({"experiment": "ergothioneine_calibration", "analyte": "ergothioneine",
                          "n_spectra": len(Z), "n_levels": tm["n_levels"],
                          "monotonicity_rho": tm["monotonicity_rho"],
                          "monotonicity_p_perm": null["p_value"],
                          "null_mean_abs_rho": null["null_mean_abs_rho"],
                          "straightness": tm["straightness"],
                          "mean_step_cosine": tm["mean_step_cosine"],
                          "net_displacement": tm["net_displacement"],
                          "total_path": tm["total_path_length"],
                          "best_dose_model": fits.get("best_model"),
                          "linear_r2": fits.get("linear_r2"), "log_r2": fits.get("log_r2"),
                          "saturating_r2": fits.get("saturating_r2"),
                          "median_ood": float(np.median(ood(Zref_u, Z)))})
    pd.DataFrame(traj_rows).to_csv(TAB / "phase4_8_trajectories.csv", index=False)
    pd.DataFrame(rep_rows).to_csv(TAB / "phase5_replicate_reproducibility.csv", index=False)
    log(f"  {len(traj_rows)} dose-response experiments analysed")

    # ── Phase 7: serum spike vs pure analyte (the decisive test) ──
    log("Phase 7 — serum spike vs pure analyte direction …")
    Zs, Ts, _, ms = proj["spiked_serum"]
    Zb, _, _, mb = proj["serum_baseline"]
    Zp, _, _, mp = proj["pure_sers"]
    base_serum = np.nan_to_num(Zb).mean(axis=0)
    # pure-analyte reference direction: pure Ag-SERS analyte minus the mean pure-SERS cloud
    pure_centroid = np.nan_to_num(Zp).mean(axis=0)
    # Raman atlas reference position per analyte (in-domain anchor)
    raman_pos = {a: np.nan_to_num(Zref[ref_analytes == a]).mean(axis=0)
                 for a in np.unique(ref_analytes)}
    raman_centroid = np.nan_to_num(Zref).mean(axis=0)

    rows = []
    for a, idx in ms.groupby("analyte").groups.items():
        pos = [ms.index.get_loc(i) for i in idx]
        d_spike = SL.displacement(Zs[pos], Zb)
        rec = {"analyte": a, "n_spike": len(pos),
               "spike_conc_uM": float(ms.iloc[pos].conc_uM.iloc[0]),
               "spike_displacement_norm": d_spike["norm"],
               "replicate_direction_cos": d_spike["replicate_direction_cos"]}
        pidx = mp.index[mp.analyte == a]
        if len(pidx):
            ppos = [mp.index.get_loc(i) for i in pidx]
            v_pure = np.nan_to_num(Zp[ppos]).mean(axis=0) - pure_centroid
            rec["pure_sers_available"] = True
            rec["cos_spike_vs_pureSERS"] = SL.cos(d_spike["vector"], v_pure)
            rec["angle_spike_vs_pureSERS_deg"] = SL.angle_deg(d_spike["vector"], v_pure)
            rec["distance_ratio_spike_over_pure"] = d_spike["norm"] / (np.linalg.norm(v_pure) + 1e-12)
        else:
            rec["pure_sers_available"] = False
        if a in raman_pos:
            v_raman = raman_pos[a] - raman_centroid
            rec["raman_reference_available"] = True
            rec["cos_spike_vs_pureRaman"] = SL.cos(d_spike["vector"], v_raman)
            rec["angle_spike_vs_pureRaman_deg"] = SL.angle_deg(d_spike["vector"], v_raman)
        else:
            rec["raman_reference_available"] = False
        rows.append(rec)
    sp = pd.DataFrame(rows)

    # null: spike displacement vs a MISMATCHED analyte's pure direction
    rng = np.random.default_rng(0)
    nulls_sers, nulls_raman = [], []
    for a, idx in ms.groupby("analyte").groups.items():
        pos = [ms.index.get_loc(i) for i in idx]
        v = SL.displacement(Zs[pos], Zb)["vector"]
        others = [o for o in mp.analyte.unique() if o != a]
        for b in rng.choice(others, size=min(10, len(others)), replace=False):
            ppos = [mp.index.get_loc(i) for i in mp.index[mp.analyte == b]]
            nulls_sers.append(SL.cos(v, np.nan_to_num(Zp[ppos]).mean(axis=0) - pure_centroid))
        ro = [o for o in raman_pos if o != a]
        for b in rng.choice(ro, size=min(10, len(ro)), replace=False):
            nulls_raman.append(SL.cos(v, raman_pos[b] - raman_centroid))
    sp.to_csv(TAB / "phase7_serum_vs_pure.csv", index=False)

    p7 = {
        "n_analytes": int(len(sp)),
        "matched_cos_vs_pureSERS_median": float(sp.cos_spike_vs_pureSERS.median()),
        "null_cos_vs_pureSERS_median": float(np.median(nulls_sers)),
        "matched_cos_vs_pureSERS_mean": float(sp.cos_spike_vs_pureSERS.mean()),
        "null_cos_vs_pureSERS_mean": float(np.mean(nulls_sers)),
        "p_matched_gt_null_sers": float((np.sum(np.array(nulls_sers) >=
                                                sp.cos_spike_vs_pureSERS.median()) + 1) /
                                        (len(nulls_sers) + 1)),
        "matched_cos_vs_pureRaman_median": float(sp.cos_spike_vs_pureRaman.median()),
        "null_cos_vs_pureRaman_median": float(np.median(nulls_raman)),
        "median_angle_vs_pureSERS_deg": float(sp.angle_spike_vs_pureSERS_deg.median()),
        "median_distance_ratio": float(sp.distance_ratio_spike_over_pure.median()),
        "median_replicate_direction_cos": float(sp.replicate_direction_cos.median()),
        "n_analytes_cos_above_null_p05": int((sp.cos_spike_vs_pureSERS >
                                              np.percentile(nulls_sers, 95)).sum()),
    }
    (TAB / "phase7_summary.json").write_text(json.dumps(p7, indent=2))
    log(f"  spike vs pure-SERS cosine: matched {p7['matched_cos_vs_pureSERS_median']:+.3f} "
        f"vs null {p7['null_cos_vs_pureSERS_median']:+.3f} | "
        f"{p7['n_analytes_cos_above_null_p05']}/{p7['n_analytes']} analytes beat the 95th pct null")

    # ── Phase 6: component activation ──
    log("Phase 6 — component activation …")
    act_rows = []
    for a, idx in ms.groupby("analyte").groups.items():
        pos = [ms.index.get_loc(i) for i in idx]
        d = np.nan_to_num(Zs[pos]).mean(0) - base_serum
        sd = np.nan_to_num(Zs[pos]).std(0) + 1e-9
        for j in range(atlas.k):
            act_rows.append({"analyte": a, "component": j, "delta": float(d[j]),
                             "effect_size": float(d[j] / sd[j]),
                             "direction": "up" if d[j] > 0 else "down"})
    act = pd.DataFrame(act_rows)
    act.to_csv(TAB / "phase6_component_activation.csv", index=False)

    # ── Phase 9: mixture behaviour ──
    mix = {"feasible": False,
           "reason": ("Each analyte was spiked into serum individually at a single "
                      "concentration; no combinatorial A+B spike exists in this corpus, so "
                      "Delta(A+B) vs Delta(A)+Delta(B) cannot be tested. The uricase experiment "
                      "is a depletion, not a mixture.")}
    (TAB / "phase9_mixture.json").write_text(json.dumps(mix, indent=2))

    # ── Phase 11: controls ──
    log("Phase 11 — controls and drift …")
    ctrl = {}
    Zb_ = np.nan_to_num(Zb)
    U = SL._unit(Zb_); C = U @ U.T; iu = np.triu_indices(len(U), 1)
    ctrl["unspiked_serum"] = {"n": int(len(Zb_)), "coord_cos_mean": float(C[iu].mean()),
                              "median_ood": float(np.median(ood(Zref_u, Zb_)))}
    # ILS blanks (conc == 0)
    Zi, _, _, mi = proj["ils_adenine"]
    bl = mi.conc_uM.values == 0
    if bl.sum() > 2:
        Zbl = np.nan_to_num(Zi[bl])
        Ub = SL._unit(Zbl); Cb = Ub @ Ub.T; iub = np.triu_indices(len(Ub), 1)
        # drift across batches
        drift = {}
        for b, g in mi[bl].groupby("batch"):
            gp = [mi.index.get_loc(i) for i in g.index]
            drift[str(b)] = float(np.linalg.norm(np.nan_to_num(Zi[gp]).mean(0) - Zbl.mean(0)))
        ctrl["ils_blanks"] = {"n": int(bl.sum()), "coord_cos_mean": float(Cb[iub].mean()),
                              "median_ood": float(np.median(ood(Zref_u, Zbl))),
                              "batch_drift_from_grand_mean": drift,
                              "max_batch_drift": float(max(drift.values())) if drift else np.nan}
    # uricase depletion direction
    Zu, _, _, mu = proj["uricase"]
    if "condition" in mu:
        g = {c: np.nan_to_num(Zu[[mu.index.get_loc(i) for i in idx]]).mean(0)
             for c, idx in mu.groupby("condition").groups.items()}
        if "spiked" in g and "spiked+uricase" in g:
            v_dep = g["spiked+uricase"] - g["spiked"]
            urate_dir = (raman_pos.get("urate", raman_centroid) - raman_centroid)
            ctrl["uricase_depletion"] = {
                "displacement_norm": float(np.linalg.norm(v_dep)),
                "cos_vs_urate_raman_direction": SL.cos(v_dep, urate_dir),
                "interpretation_note": ("enzymatic removal of urate should move AWAY from the "
                                        "urate direction, i.e. a NEGATIVE cosine is the "
                                        "chemically expected sign"),
            }
    # isotopic specificity control
    Ziso, _, _, miso = proj["isotopic"]
    if "condition" in miso:
        gg = {c: np.nan_to_num(Ziso[[miso.index.get_loc(i) for i in idx]]).mean(0)
              for c, idx in miso.groupby("condition").groups.items()}
        if "UA" in gg and "UAiso" in gg:
            ctrl["isotopic_UA_vs_UAiso"] = {
                "coordinate_distance": float(np.linalg.norm(gg["UA"] - gg["UAiso"])),
                "cosine": SL.cos(gg["UA"], gg["UAiso"])}
    (TAB / "phase11_controls.json").write_text(json.dumps(ctrl, indent=2, default=float))
    log(f"  controls: {json.dumps({k: (v.get('median_ood') if isinstance(v, dict) else v) for k, v in ctrl.items()}, default=str)}")

    # ── verify atlas untouched ──
    fp1 = hashlib.sha256(np.ascontiguousarray(atlas.components).tobytes()).hexdigest()[:32]
    assert fp1 == fp0, "ATLAS MUTATED — study invalid"
    (ART / "study_manifest.json").write_text(json.dumps({
        "atlas": {"k": atlas.k, "fingerprint": fp0, "verified_unchanged": True},
        "datasets": {k: int(len(v[1])) for k, v in data.items()},
        "phase7": p7, "controls": ctrl, "mixture": mix,
        "runtime_s": round(time.time() - t0, 1),
    }, indent=2, default=str))
    log(f"atlas fingerprint re-verified unchanged | runtime {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
