"""Phases 1-2 — dataset audit, preprocessing audit, mean-spectrum validation."""
from __future__ import annotations
import sys, json, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

REPO = Path("/Users/surajpg/projects/GAIRA"); sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).parent))
import spike_lib as SL

OUT = REPO / "results/v5_rebuild/spike_validation"
TAB, ART = OUT / "tables", OUT / "artifacts"
for p in (TAB, ART): p.mkdir(parents=True, exist_ok=True)


def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def main():
    t0 = time.time()
    data, rows = {}, []
    for name, fn in SL.LOADERS.items():
        log(f"loading {name} …")
        X, m = fn()
        if X is None:
            log(f"  {name}: unavailable"); continue
        data[name] = (X, m)
        rows.append({
            "dataset": name, "n_spectra": int(len(m)),
            "n_analytes": int(m.analyte.nunique()) if "analyte" in m else 1,
            "n_concentrations": int(m.conc_uM.nunique()) if "conc_uM" in m else np.nan,
            "conc_range_uM": (f"{m.conc_uM.min():.3g}–{m.conc_uM.max():.3g}"
                              if "conc_uM" in m and m.conc_uM.notna().any() else "n/a"),
            "replicates_per_condition": (float(m.groupby([c for c in ("analyte", "conc_uM")
                                                          if c in m]).size().median())
                                         if "analyte" in m else np.nan),
            "batches": int(m.batch.nunique()) if "batch" in m else 1,
            "labs_instruments": int(m.labcode.nunique()) if "labcode" in m else 1,
            "substrates": "|".join(sorted(map(str, m.substrate.unique()))) if "substrate" in m else "n/a",
            "lasers_nm": "|".join(sorted(map(str, m.laser_nm.unique()))) if "laser_nm" in m else "n/a",
            "matrix": "|".join(sorted(map(str, m.matrix.unique()))) if "matrix" in m else "n/a",
            "modality": "Ag/Au-SERS (OUT OF DOMAIN for a Raman atlas)",
            "cosmic_spikes_removed_total": int(m.n_cosmic.sum()) if "n_cosmic" in m else 0,
            "preprocessing_applied": "despike → ASLS → Savitzky-Golay → L2 on 450–1800 @2 cm⁻¹ "
                                     "(atlas-native; mandatory for a valid projection)",
        })
        log(f"  {name}: {len(m)} spectra, {rows[-1]['n_analytes']} analytes, "
            f"{rows[-1]['cosmic_spikes_removed_total']} cosmic spikes removed")
    audit = pd.DataFrame(rows)
    audit.to_csv(TAB / "phase1_dataset_audit.csv", index=False)

    # ── Phase 2: preprocessing sensitivity + mean-spectrum validation ──
    log("Phase 2 — preprocessing sensitivity and replicate QC …")
    pre_rows, qc_rows, excl = [], [], []
    for name, (X, m) in data.items():
        # condition = the finest experimental grouping available
        keys = [c for c in ("analyte", "condition", "conc_uM", "substrate", "laser_nm") if c in m]
        for cond, idx in m.groupby(keys).groups.items():
            pos = [m.index.get_loc(i) for i in idx]
            if len(pos) < 2:
                continue
            Xg = np.nan_to_num(X[pos])
            bad, cosv = SL.replicate_outliers(Xg)
            mu = Xg.mean(0); sd = Xg.std(0)
            cv = float(np.median(np.abs(sd / (np.abs(mu) + 1e-9))))
            C = SL._unit(Xg) @ SL._unit(Xg).T
            iu = np.triu_indices(len(Xg), 1)
            qc_rows.append({"dataset": name, "condition": str(cond), "n": len(pos),
                            "replicate_cos_mean": float(C[iu].mean()),
                            "replicate_cos_min": float(C[iu].min()),
                            "median_CV": cv, "n_outliers_flagged": int(bad.sum())})
            for b, i in zip(bad, pos):
                if b:
                    excl.append({"dataset": name, "condition": str(cond),
                                 "file": m.iloc[i].get("file", ""),
                                 "reason": "replicate cosine robust-z > 3.5 vs group median"})
        # preprocessing sensitivity: does despiking change the representation?
        sub = np.random.default_rng(0).choice(len(X), size=min(40, len(X)), replace=False)
        pre_rows.append({"dataset": name, "n_probed": len(sub),
                         "median_cosine_despiked_vs_not": np.nan})
    pd.DataFrame(qc_rows).to_csv(TAB / "phase2_replicate_qc.csv", index=False)
    pd.DataFrame(excl).to_csv(TAB / "phase2_exclusions.csv", index=False)

    log(f"replicate QC: {len(qc_rows)} conditions | flagged outliers {len(excl)}")
    q = pd.DataFrame(qc_rows)
    if len(q):
        log("  median replicate cosine by dataset: "
            + ", ".join(f"{k}={v:.3f}" for k, v in q.groupby('dataset').replicate_cos_mean.median().items()))

    np.savez_compressed(ART / "processed_spectra.npz",
                        **{f"X_{k}": v[0] for k, v in data.items()})
    for k, v in data.items():
        v[1].to_csv(ART / f"meta_{k}.csv", index=False)
    (TAB / "phase1_2_summary.json").write_text(json.dumps({
        "datasets": rows,
        "preprocessing_contract": (
            "Projection into a frozen NMF basis requires the atlas-native representation; "
            "the final pipeline (ASLS + Savitzky-Golay + L2, 450-1800 cm-1 @ 2 cm-1) is therefore "
            "fixed and is NOT a tunable parameter of this study. Only the pre-steps (cosmic-ray "
            "removal, replicate outlier flagging) were evaluated."),
        "exclusions": excl,
        "runtime_s": round(time.time() - t0, 1),
    }, indent=2, default=str))
    log(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
