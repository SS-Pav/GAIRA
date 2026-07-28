"""Foundation audit — Part 9/10: run the six validation datasets IN ORDER through the
FROZEN Raman atlas (test only, never train). Deterministic. Writes
foundation_audit/tables/validation_results.json + figures.

Order:
  1. Pure Gobbato Raman analytes      (in-domain sanity)
  2. Pure Gobbato SERS analytes       (Raman->SERS transfer)
  3. Adenine concentration series     (dose / Langmuir / redistribution)
  4. Ergothioneine concentration series (dose / scaling)
  5. Serum spike-in                   (recoverability, direction agreement)
  6. Uricase depletion                (difference BSV, purine localisation)
"""
from __future__ import annotations
import sys, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

REPO = Path("/Users/surajpg/projects/GAIRA")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "results/v5_rebuild/spike_validation/code"))
from gaira.engine import GAIRAEngine
from gaira.engine.mss import MSSLayer
from gaira.foundation import dataset as DS
from gaira.data.synonyms import canonical
import spike_lib as SL

AUD = REPO / "results/v5_rebuild/foundation_audit"
TAB, FIG = AUD / "tables", AUD / "figures"
INK, ACC, RED, GRN = "#1b2430", "#2a6f97", "#b2182b", "#2f7d4f"

eng = GAIRAEngine()
mss = MSSLayer.from_engine(eng)
THEMES = eng.builder.onto.biochemical_theme_ids
atlas = eng.atlas


def coords(V):
    """L1 24-coords for a stack of atlas-grid vectors (NNLS onto frozen components)."""
    return atlas.coordinates(np.atleast_2d(np.nan_to_num(V)))


def bsv_of(coord, domain="buffer"):
    return eng.infer(coordinates=np.asarray(coord, float), domain=domain).bsv


def theme_vec(bsv):
    return np.array([bsv.composition[t] for t in THEMES])


def dominant_theme(coord, domain="buffer"):
    b = bsv_of(coord, domain)
    tv = theme_vec(b)
    return THEMES[int(np.argmax(tv))], float(tv.max()), float(b.ood_score)


R = {}

# ── 1. Pure Gobbato Raman (in-domain) ─────────────────────────────────
corpus = DS.load_reference_corpus()
m = corpus.meta
gob = m.source == "gobbato_raman_metabolites"
Zg = coords(corpus.X[gob.values])
ood_raman = [float(bsv_of(z).ood_score) for z in Zg]
# dominant themes per Gobbato analyte
dt = {}
for a, z in zip(m.analyte[gob].values, Zg):
    t, w, o = dominant_theme(z)
    dt.setdefault(a, []).append(t)
R["1_gobbato_raman"] = {
    "n_spectra": int(gob.sum()), "n_analytes": int(m.analyte[gob].nunique()),
    "mean_ood": round(float(np.mean(ood_raman)), 4),
    "median_ood": round(float(np.median(ood_raman)), 4),
    "note": "in-domain (training) — expected LOW OOD",
    "example_dominant_themes": {a: max(set(v), key=v.count) for a, v in list(dt.items())[:12]},
}
print("1. Gobbato Raman: mean OOD", R["1_gobbato_raman"]["mean_ood"], flush=True)

# ── 2. Pure Gobbato SERS transfer ─────────────────────────────────────
Xs, rs = SL.load_pure_sers()
if Xs is not None:
    Zs = coords(Xs)
    ood_sers = [float(bsv_of(z, "buffer").ood_score) for z in Zs]
    # per-analyte mean coords, both modalities
    def per_analyte(Z, names):
        out = {}
        for a in pd.unique(names):
            out[canonical(a)] = Z[np.asarray(names) == a].mean(0)
        return out
    raman_by = per_analyte(Zg, m.analyte[gob].values)
    sers_by = per_analyte(Zs, rs.analyte.values)
    shared = sorted(set(raman_by) & set(sers_by))
    rows = []
    for a in shared:
        r_, s_ = raman_by[a], sers_by[a]
        cc = float(np.dot(r_, s_) / (np.linalg.norm(r_) * np.linalg.norm(s_) + 1e-12))
        tr, _, _ = dominant_theme(r_); ts, _, _ = dominant_theme(s_)
        rows.append({"analyte": a, "coord_cosine": round(cc, 4),
                     "raman_theme": tr, "sers_theme": ts, "theme_preserved": tr == ts})
    tr_df = pd.DataFrame(rows).sort_values("coord_cosine", ascending=False)
    tr_df.to_csv(TAB / "validation_transfer_pairs.csv", index=False)
    R["2_gobbato_sers_transfer"] = {
        "n_sers_spectra": int(len(Xs)), "n_matched_analytes": len(shared),
        "sers_mean_ood": round(float(np.mean(ood_sers)), 4),
        "raman_mean_ood": R["1_gobbato_raman"]["mean_ood"],
        "median_coord_cosine": round(float(tr_df.coord_cosine.median()), 4),
        "n_theme_preserved": int(tr_df.theme_preserved.sum()),
        "most_preserved": tr_df.head(5)[["analyte", "coord_cosine"]].values.tolist(),
        "least_preserved": tr_df.tail(5)[["analyte", "coord_cosine"]].values.tolist(),
    }
    print("2. SERS transfer: median cos", R["2_gobbato_sers_transfer"]["median_coord_cosine"],
          "| SERS OOD", R["2_gobbato_sers_transfer"]["sers_mean_ood"], flush=True)


# ── dose helper ───────────────────────────────────────────────────────
def dose_analysis(X, rec, theme_id, method_filter=None):
    if method_filter is not None:
        keep = method_filter(rec)
        X, rec = X[keep.values], rec[keep].reset_index(drop=True)
    Z = coords(X)
    concs = rec.conc_uM.values
    keys, M = SL.group_means(Z, concs)                       # per-dose mean coords
    ti = THEMES.index(theme_id)
    theme_series = [float(theme_vec(bsv_of(mc))[ti]) for mc in M]
    traj = SL.trajectory_metrics(np.array(keys, float), M)
    fits = SL.dose_response_fits(np.array(keys, float), traj["distance_from_control"])
    mono = SL.monotonicity_null(concs, Z, concs, n_perm=500)
    # redistribution: how many components change sign of step across the series
    steps = np.diff(M, axis=0)
    n_signflip = int(np.sum(np.any(np.sign(steps[:-1]) != np.sign(steps[1:]), axis=0))) if len(steps) > 1 else 0
    return {"levels_uM": [float(k) for k in keys], "n_per_level": [int((concs == k).sum()) for k in keys],
            "theme_series": [round(v, 4) for v in theme_series], "theme": theme_id,
            "straightness": round(traj["straightness"], 3),
            "monotonicity_rho": round(traj["monotonicity_rho"], 3),
            "monotonicity_p": round(mono["p_value"], 4),
            "best_dose_model": fits.get("best_model"),
            "saturating_r2": round(fits.get("saturating_r2", float("nan")), 3),
            "saturating_K_uM": round(fits.get("saturating_K", float("nan")), 3),
            "n_components_redistributing": n_signflip}, keys, M, theme_series


# ── 3. Adenine dose (cAg@785, purine) ─────────────────────────────────
Xa, ra = SL.load_ils_adenine()
if Xa is not None:
    def cag785(r): return (r.substrate.astype(str).str.contains("cAg")) & (r.laser_nm == 785)
    ad, ak, aM, aser = dose_analysis(Xa, ra, "nucleic_purine", cag785)
    R["3_adenine_dose"] = ad
    print("3. Adenine: purine rho", ad["monotonicity_rho"], "model", ad["best_dose_model"],
          "redistrib comps", ad["n_components_redistributing"], flush=True)

# ── 4. Ergothioneine dose (sulfur) ────────────────────────────────────
Xe, re_ = SL.load_ergothioneine()
if Xe is not None:
    eg, ek, eM, eser = dose_analysis(Xe, re_, "sulfur_antioxidant")
    R["4_ergothioneine_dose"] = eg
    print("4. Ergothioneine: sulfur rho", eg["monotonicity_rho"], "model", eg["best_dose_model"],
          "straightness", eg["straightness"], flush=True)

# dose figure
if "3_adenine_dose" in R and "4_ergothioneine_dose" in R:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    a = R["3_adenine_dose"]; ax[0].plot(a["levels_uM"], a["theme_series"], "-o", color=ACC)
    ax[0].set_title(f"Adenine → purine theme (ρ={a['monotonicity_rho']}, {a['best_dose_model']})")
    ax[0].set_xlabel("adenine µM"); ax[0].set_ylabel("nucleic_purine share")
    e = R["4_ergothioneine_dose"]; ax[1].plot(e["levels_uM"], e["theme_series"], "-o", color=GRN)
    ax[1].set_title(f"Ergothioneine → sulfur theme (ρ={e['monotonicity_rho']}, {e['best_dose_model']})")
    ax[1].set_xlabel("ergothioneine µM"); ax[1].set_ylabel("sulfur_antioxidant share")
    for a_ in ax:
        for s in ("top", "right"): a_.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "validation_dose.png", dpi=130); plt.close(fig)

# ── 5. Serum spike-in recoverability (use committed validated table) ──
p7 = REPO / "results/v5_rebuild/spike_validation/tables/phase7_serum_vs_pure.csv"
if p7.exists():
    d7 = pd.read_csv(p7)
    col = "cos_spike_vs_pureSERS" if "cos_spike_vs_pureSERS" in d7 else d7.columns[-1]
    strong = d7[d7[col] >= 0.35]; partial = d7[(d7[col] >= 0.10) & (d7[col] < 0.35)]
    poor = d7[d7[col] < 0.10]
    R["5_serum_spike"] = {
        "n_analytes": int(len(d7)),
        "strong_recovery": int(len(strong)), "moderate_recovery": int(len(partial)),
        "weak_recovery": int(len(poor)),
        "median_direction_agreement": round(float(d7[col].median()), 3),
        "strong_examples": sorted(strong.analyte.tolist())[:10] if "analyte" in d7 else [],
        "note": "direction agreement = cos(serum-spike Δ, pure-SERS fingerprint); "
                "committed phase7_serum_vs_pure",
    }
    print("5. Serum: strong", len(strong), "moderate", len(partial), "weak", len(poor), flush=True)

# ── 6. Uricase depletion (difference BSV) ─────────────────────────────
Xu, ru = SL.load_uricase()
if Xu is not None:
    Zu = coords(Xu)
    cond = ru.get("condition", pd.Series(["?"] * len(ru)))
    # spiked vs spiked+uricase (labels vary — detect by 'uricase' token)
    is_uri = ru.apply(lambda r: "uricase" in str(r.to_dict()).lower(), axis=1)
    if is_uri.any() and (~is_uri).any():
        base = Zu[(~is_uri).values].mean(0); dep = Zu[is_uri.values].mean(0)
        bb, bd = bsv_of(base, "serum"), bsv_of(dep, "serum")
        dtheme = {t: round(float(bd.composition[t] - bb.composition[t]), 4) for t in THEMES}
        worst = sorted(dtheme.items(), key=lambda kv: kv[1])[:3]
        R["6_uricase_depletion"] = {
            "n_spectra": int(len(Xu)), "n_uricase": int(is_uri.sum()), "n_base": int((~is_uri).sum()),
            "delta_theme": dtheme,
            "purine_delta": dtheme.get("nucleic_purine"),
            "most_decreased_themes": worst,
            "localises_to_purine": bool(min(dtheme, key=dtheme.get) == "nucleic_purine"),
        }
        print("6. Uricase: purine Δ", dtheme.get("nucleic_purine"),
              "most decreased", worst[0], flush=True)

(TAB / "validation_results.json").write_text(json.dumps(R, indent=2, default=str))
print("\nWROTE validation_results.json")
