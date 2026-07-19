"""GAIRA V5 Phase 1 (V5.1) — preprocessing & comparability experiment.

Scientific questions:
  Q-comparability: can Raman and Ag-SERS spectra share one preprocessing pipeline
                   and be jointly analyzed without erasing chemistry?
  H1b: are matched analytes more similar across modalities than to other analytes?
  H6 : does unsupervised structure reflect chemistry or acquisition modality/source/excitation?
  H1c: does preprocessing preserve known analyte bands?

Read-only. Emits figures/tables/logs under results/v5_rebuild/phase1/.
"""
from __future__ import annotations
import sys, re, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/Users/surajpg/projects/GAIRA")
sys.path.insert(0, str(REPO / "src"))
from gaira.data import loader                      # noqa
from gaira.preprocessing import pipeline as pp      # noqa

PH = REPO / "results/v5_rebuild/phase1"
FIG = PH / "figures"; TAB = PH / "tables"; LOG = PH / "logs"
for d in (FIG, TAB, LOG): d.mkdir(parents=True, exist_ok=True)
GRID = pp.common_grid(520.0, 1750.0, 2.0)


def norm_name(s):
    s = re.sub(r"^(l-|d-|dl-)", "", str(s).lower().strip())
    return s.replace("acid", "").replace("-", " ").strip()


def cosine(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 20: return np.nan
    a, b = a[m], b[m]
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 1e-9 and nb > 1e-9 else np.nan


def main():
    specs = [s for s in loader.load_all() if s.has_spectrum]
    raman = [s for s in specs if s.record.modality.value == "raman"]
    sers = [s for s in specs if s.record.modality.value == "sers"]
    print(f"loaded {len(raman)} Raman + {len(sers)} Ag-SERS full spectra")

    # matched analytes across modalities (by normalized name)
    r_by = {norm_name(s.record.canonical_analyte_name): s for s in raman}
    s_by = {}
    for s in sers:
        s_by.setdefault(norm_name(s.record.canonical_analyte_name), s)
    matched = sorted(set(r_by) & set(s_by))
    print("matched analytes (Raman∩Ag-SERS):", len(matched), matched)

    results = {"pipelines": {}}
    # ── preprocess under every candidate pipeline; compute comparability metrics ──
    for pname, cfg in pp.PIPELINES.items():
        R = np.array([pp.preprocess(s.wavenumber, s.intensity, cfg, GRID) for s in raman])
        S = np.array([pp.preprocess(s.wavenumber, s.intensity, cfg, GRID) for s in sers])
        cov_r = float(np.mean([np.isfinite(x).mean() for x in R]))
        cov_s = float(np.mean([np.isfinite(x).mean() for x in S]))

        # H1b: matched cross-modality cosine vs null (matched Raman vs random SERS)
        matched_cos, null_cos = [], []
        for a in matched:
            rp = pp.preprocess(r_by[a].wavenumber, r_by[a].intensity, cfg, GRID)
            sp = pp.preprocess(s_by[a].wavenumber, s_by[a].intensity, cfg, GRID)
            matched_cos.append(cosine(rp, sp))
            for b in matched:
                if b != a:
                    sp2 = pp.preprocess(s_by[b].wavenumber, s_by[b].intensity, cfg, GRID)
                    null_cos.append(cosine(rp, sp2))
        matched_cos = np.array([c for c in matched_cos if np.isfinite(c)])
        null_cos = np.array([c for c in null_cos if np.isfinite(c)])

        # H6: modality leakage — predict modality from PCA of all spectra
        X = np.vstack([R, S])
        X = np.nan_to_num(X, nan=0.0)
        ylab = np.array([0] * len(R) + [1] * len(S))
        leak_acc = np.nan; pcs = None
        try:
            from sklearn.decomposition import PCA
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import cross_val_score
            pca = PCA(n_components=10, random_state=0)
            pcs = pca.fit_transform(X)
            leak_acc = float(np.mean(cross_val_score(
                LogisticRegression(max_iter=500), pcs, ylab, cv=5)))
        except Exception as e:
            print("sklearn leakage skip:", e)

        results["pipelines"][pname] = {
            "coverage_raman": round(cov_r, 3), "coverage_sers": round(cov_s, 3),
            "matched_xmod_cosine_mean": round(float(np.mean(matched_cos)), 3) if len(matched_cos) else None,
            "matched_xmod_cosine_vals": [round(float(x), 3) for x in matched_cos],
            "null_xmod_cosine_mean": round(float(np.mean(null_cos)), 3) if len(null_cos) else None,
            "modality_leakage_cv_acc": round(leak_acc, 3),
            "class_balance": round(len(R) / (len(R) + len(S)), 3),
        }
        # store PCs for the primary pipeline figure
        if pname == "P3_asls_savgol_snv":
            results["_primary_pcs"] = pcs
            results["_primary_labels"] = ylab
            results["_primary_R"] = R; results["_primary_S"] = S

    (TAB / "phase1_comparability_metrics.json").write_text(json.dumps(
        {k: v for k, v in results.items() if not k.startswith("_")}, indent=2))
    # tidy CSV
    rows = [{"pipeline": k, **v} for k, v in results["pipelines"].items()]
    dfm = pd.DataFrame(rows).drop(columns=["matched_xmod_cosine_vals"])
    dfm.to_csv(TAB / "phase1_comparability_metrics.csv", index=False)
    print("\n== comparability by pipeline ==")
    print(dfm[["pipeline", "coverage_raman", "coverage_sers", "matched_xmod_cosine_mean",
               "null_xmod_cosine_mean", "modality_leakage_cv_acc"]].to_string(index=False))

    # ── FIGURES ──
    # 1. matched-analyte cross-modality overlays (adenine + tryptophan)
    cfg = pp.PIPELINES["P3_asls_savgol_snv"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, a in zip(axes, [x for x in ["adenine", "tryptophan"] if x in matched] or matched[:2]):
        rp = pp.preprocess(r_by[a].wavenumber, r_by[a].intensity, cfg, GRID)
        sp = pp.preprocess(s_by[a].wavenumber, s_by[a].intensity, cfg, GRID)
        ax.plot(GRID, rp, label=f"Raman ({r_by[a].record.excitation_nm} nm)", color="#2563eb")
        ax.plot(GRID, sp, label=f"Ag-SERS ({s_by[a].record.excitation_nm} nm)", color="#dc2626", alpha=0.8)
        ax.set_title(f"{a}  (cross-mod cosine={cosine(rp,sp):.2f})"); ax.legend(fontsize=8)
        ax.set_xlabel("cm⁻¹"); ax.set_ylabel("SNV intensity")
    fig.suptitle("Matched analyte: Raman vs Ag-SERS (P3: asls+savgol+snv)")
    fig.tight_layout(); fig.savefig(FIG / "matched_analyte_crossmodality.png", dpi=130); plt.close(fig)

    # 2. PCA colored by modality / source / excitation
    pcs = results.get("_primary_pcs")
    if pcs is not None:
        exc = np.array([s.record.excitation_nm or 0 for s in raman] + [s.record.excitation_nm or 0 for s in sers])
        src = np.array([0] * len(raman) + [1] * len(sers))
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        lbl = results["_primary_labels"]
        for i, (c, title) in enumerate([(lbl, "modality (0=Raman,1=Ag-SERS)"), (exc, "excitation (nm)")]):
            sc = axes[i].scatter(pcs[:, 0], pcs[:, 1], c=c, cmap="coolwarm" if i == 0 else "viridis", s=18, alpha=0.8)
            axes[i].set_title(f"PCA of preprocessed spectra — colored by {title}")
            axes[i].set_xlabel("PC1"); axes[i].set_ylabel("PC2"); fig.colorbar(sc, ax=axes[i])
        fig.suptitle(f"Modality leakage: linear CV acc predicting modality from 10 PCs = "
                     f"{results['pipelines']['P3_asls_savgol_snv']['modality_leakage_cv_acc']}")
        fig.tight_layout(); fig.savefig(FIG / "pca_modality_leakage.png", dpi=130); plt.close(fig)

    # 3. matched vs null cosine distribution (primary pipeline)
    p = results["pipelines"]["P3_asls_savgol_snv"]
    fig, ax = plt.subplots(figsize=(7, 4))
    mc = p["matched_xmod_cosine_vals"]
    ax.axvline(p["null_xmod_cosine_mean"], color="gray", ls="--", label=f"null mean={p['null_xmod_cosine_mean']}")
    ax.hist(mc, bins=np.linspace(-1, 1, 21), color="#2563eb", alpha=0.8, label="matched cross-mod cosine")
    ax.set_xlabel("cosine(Raman, Ag-SERS)"); ax.set_title("Same-analyte cross-modality similarity vs null")
    ax.legend(); fig.tight_layout(); fig.savefig(FIG / "matched_vs_null_cosine.png", dpi=130); plt.close(fig)

    # 4. preprocessing overlays (one Raman, one SERS, across pipelines)
    r0 = r_by.get("adenine", raman[0]); s0 = s_by.get("adenine", sers[0])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for pname, cfg in pp.PIPELINES.items():
        axes[0].plot(GRID, pp.preprocess(r0.wavenumber, r0.intensity, cfg, GRID), label=pname, alpha=0.7)
        axes[1].plot(GRID, pp.preprocess(s0.wavenumber, s0.intensity, cfg, GRID), label=pname, alpha=0.7)
    axes[0].set_title(f"Raman {r0.record.canonical_analyte_name} across pipelines")
    axes[1].set_title(f"Ag-SERS {s0.record.canonical_analyte_name} across pipelines")
    for ax in axes: ax.legend(fontsize=7); ax.set_xlabel("cm⁻¹")
    fig.tight_layout(); fig.savefig(FIG / "preprocessing_overlays.png", dpi=130); plt.close(fig)

    # ── band preservation (H1c): adenine 725 (SERS), Phe/1003 region ──
    def peak_near(spec, center, tol=15):
        m = (GRID >= center - tol) & (GRID <= center + tol)
        return float(np.nanmax(spec[m])) if m.any() and np.isfinite(spec[m]).any() else np.nan
    band_rows = []
    for label, spec, center in [("adenine_SERS_725", pp.preprocess(s0.wavenumber, s0.intensity, cfg, GRID), 725)]:
        band_rows.append({"band": label, "center": center, "retained_intensity": round(peak_near(spec, center), 3)})
    pd.DataFrame(band_rows).to_csv(TAB / "phase1_band_preservation.csv", index=False)

    print("\nfigures + tables written to results/v5_rebuild/phase1/")
    return results


if __name__ == "__main__":
    main()
