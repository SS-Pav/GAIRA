"""Phase 12 — figures + manuscript-quality report/PDF."""
from __future__ import annotations
import sys, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap

REPO = Path("/Users/surajpg/projects/GAIRA"); sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).parent))
import spike_lib as SL
from gaira.foundation.families_raman import family_of

OUT = REPO / "results/v5_rebuild/spike_validation"
TAB, FIG, ART = OUT / "tables", OUT / "figures", OUT / "artifacts"
FIG.mkdir(parents=True, exist_ok=True)
PDF_PATH = REPO / "GAIRA_V5_Serum_Spike_Projection_Validation.pdf"

P, S = "#2563EB", "#D97706"
INK, MUTED, GRIDC = "#1f2328", "#6B7280", "#E5E7EB"
DIV = LinearSegmentedColormap.from_list("div", [P, "#F3F4F6", S])
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9,
                     "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "axes.grid": True,
                     "grid.color": GRIDC, "grid.linewidth": 0.5, "xtick.color": MUTED,
                     "ytick.color": MUTED, "text.color": INK, "axes.labelcolor": INK,
                     "axes.titlecolor": INK, "legend.frameon": False, "figure.facecolor": "white"})
PAGE = (8.5, 11.0)


def _wrap(t, n=104):
    out, line = [], ""
    for w in str(t).split():
        if len(line) + len(w) + 1 > n:
            out.append(line); line = w
        else:
            line = (line + " " + w).strip()
    if line: out.append(line)
    return out


def text_page(pdf, title, blocks, subtitle=None):
    fig = plt.figure(figsize=PAGE); y = 0.95
    fig.text(0.07, y, title, fontsize=16, fontweight="bold", color=INK); y -= 0.032
    if subtitle:
        fig.text(0.07, y, subtitle, fontsize=9.5, color=MUTED); y -= 0.028
    for kind, txt in blocks:
        if y < 0.06:
            pdf.savefig(fig); plt.close(fig); fig = plt.figure(figsize=PAGE); y = 0.95
        if kind == "h":
            y -= 0.013; fig.text(0.07, y, txt, fontsize=10.5, fontweight="bold", color=INK); y -= 0.024
        elif kind == "b":
            for ln in _wrap(txt):
                fig.text(0.07, y, ln, fontsize=8.5, color=INK); y -= 0.0163
            y -= 0.007
        elif kind == "m":
            fig.text(0.075, y, txt, fontsize=7.1, color=INK, family="DejaVu Sans Mono"); y -= 0.0136
    pdf.savefig(fig); plt.close(fig)


def main():
    traj = pd.read_csv(TAB / "phase4_8_trajectories.csv")
    rep = pd.read_csv(TAB / "phase5_replicate_reproducibility.csv")
    sp = pd.read_csv(TAB / "phase7_serum_vs_pure.csv")
    sp["family"] = sp.analyte.map(family_of)
    p7 = json.loads((TAB / "phase7_summary.json").read_text())
    ctrl = json.loads((TAB / "phase11_controls.json").read_text())
    audit = pd.read_csv(TAB / "phase1_dataset_audit.csv")
    qc = pd.read_csv(TAB / "phase2_replicate_qc.csv")
    act = pd.read_csv(TAB / "phase6_component_activation.csv")
    ood = pd.read_csv(TAB / "phase10_ood_summary.csv", index_col=0)
    man = json.loads((ART / "study_manifest.json").read_text())
    mix = json.loads((TAB / "phase9_mixture.json").read_text())
    ils = pd.read_csv(TAB / "phase3_projection_ils_adenine.csv")
    erg = pd.read_csv(TAB / "phase3_projection_ergothioneine.csv")

    # ── F1 preprocessing cascade ──
    fs = SL._zip_dir("SERS metabolites/")[:1]
    fig, axs = plt.subplots(1, 5, figsize=(15, 3.0))
    if fs:
        fn, wn, y = fs[0]
        yd, ns = SL.despike(y, wn=wn)
        from gaira.preprocessing import pipeline as pp
        base = pp.BASELINES["asls"](yd)
        ybc = yd - base
        ysm = pp.SMOOTHERS["savgol"](ybc)
        yfin, _ = SL.to_atlas(wn, y)
        for ax, (d, t) in zip(axs, [((wn, y), "raw"), ((wn, yd), f"cosmic removed (n={ns})"),
                                    ((wn, ybc), "ASLS baseline removed"),
                                    ((wn, ysm), "Savitzky–Golay"),
                                    ((SL.GRID, yfin), "resampled + L2 (atlas-native)")]):
            ax.plot(d[0], d[1], lw=0.6, color=P)
            ax.set_title(t, fontsize=8); ax.set_xlabel("cm⁻¹", fontsize=7)
            ax.tick_params(labelsize=6); ax.set_yticks([])
            if t.startswith("raw") or "cosmic" in t:
                ax.set_xlim(400, 1900)
    fig.suptitle("Phase 2 — preprocessing cascade (pure Ag-SERS example). The final step is "
                 "atlas-native and mandatory for a valid projection.", fontsize=8.5)
    fig.tight_layout(); fig.savefig(FIG / "f1_preprocessing.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # ── F2 dose-response trajectories ──
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
    ax = axs[0]
    for (sub, las), g in ils.groupby(["substrate", "laser_nm"]):
        gm = g.groupby("conc_uM")[[f"c{j}" for j in range(24)]].mean()
        base = gm.iloc[0].values
        d = np.linalg.norm(gm.values - base, axis=1)
        ax.plot(gm.index, d, "-o", ms=3, lw=1.2, label=f"{sub}@{las}")
    ax.set_xlabel("adenine concentration (µM)"); ax.set_ylabel("distance from blank in atlas")
    ax.set_title("Phase 4 — ILS adenine dose-response"); ax.legend(fontsize=6.5)
    ax = axs[1]
    gm = erg.groupby("conc_uM")[[f"c{j}" for j in range(24)]].mean()
    d = np.linalg.norm(gm.values - gm.iloc[0].values, axis=1)
    ax.plot(gm.index, d, "-o", ms=4, color=S)
    ax.set_xlabel("ergothioneine concentration"); ax.set_ylabel("distance from lowest")
    ax.set_title("Phase 4 — ergothioneine calibration")
    ax = axs[2]
    x = np.arange(len(traj))
    ax.bar(x, traj.monotonicity_rho, color=P, label="observed |ρ|")
    ax.bar(x, traj.null_mean_abs_rho, color=S, alpha=0.75, width=0.45, label="permutation null")
    ax.set_xticks(x); ax.set_xticklabels([e.replace("ils_adenine::", "") for e in traj.experiment],
                                         rotation=40, ha="right", fontsize=6)
    ax.set_ylabel("Spearman ρ (distance vs concentration)")
    ax.set_title("Monotonicity vs label-permutation null (all p=0.002)"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "f2_dose_response.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # ── F3 reproducibility + smoothness ──
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.0))
    ax = axs[0]
    for e, g in rep.groupby("experiment"):
        ax.plot(g.conc_uM, g.direction_cos_mean, "-o", ms=3, lw=1, label=e.replace("ils_adenine::", ""))
    ax.set_xlabel("concentration (µM)"); ax.set_ylabel("replicate displacement-direction cosine")
    ax.set_title("Phase 5 — direction reproducibility"); ax.legend(fontsize=6)
    ax = axs[1]
    ax.scatter(traj.straightness, traj.mean_step_cosine, s=50,
               c=[P if "cAg" in e or "cAu" in e else S for e in traj.experiment])
    for _, r in traj.iterrows():
        ax.annotate(r.experiment.replace("ils_adenine::", "")[:12],
                    (r.straightness, r.mean_step_cosine), fontsize=6,
                    xytext=(3, 3), textcoords="offset points")
    ax.axhline(0, ls="--", c=MUTED, lw=1)
    ax.set_xlabel("straightness (net / path)"); ax.set_ylabel("mean consecutive-step cosine")
    ax.set_title("Trajectory smoothness (negative = zig-zag)")
    ax = axs[2]
    ax.bar(range(len(ood)), ood["50%"], color=[P if i == "pure_sers" else S for i in ood.index])
    ax.set_xticks(range(len(ood))); ax.set_xticklabels(ood.index, rotation=40, ha="right", fontsize=6.5)
    ax.set_ylabel("median OOD distance"); ax.set_title("Phase 10 — all datasets are out of domain")
    fig.tight_layout(); fig.savefig(FIG / "f3_reproducibility_ood.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # ── F4 Phase 7 decisive test ──
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
    ax = axs[0]
    ax.hist(sp.cos_spike_vs_pureSERS.dropna(), bins=20, color=P, alpha=0.9, label="matched analyte")
    ax.axvline(p7["null_cos_vs_pureSERS_median"], ls="--", c=S, lw=1.6, label="mismatched null (median)")
    ax.axvline(p7["matched_cos_vs_pureSERS_median"], ls="-", c=INK, lw=1.4, label="matched median")
    ax.set_xlabel("cos(spike displacement, pure-analyte direction)")
    ax.set_ylabel("analytes"); ax.legend(fontsize=6.5)
    ax.set_title("Phase 7 — overall: no directional agreement")
    ax = axs[1]
    fam_col = {"purine": "#7C3AED", "cofactor": "#DB2777", "amino_acid": "#059669",
               "saccharide": P, "lipid": S}
    for f, g in sp.groupby("family"):
        ax.scatter(g.spike_displacement_norm, g.cos_spike_vs_pureSERS, s=34,
                   color=fam_col.get(f, MUTED), label=f, alpha=0.85)
    for _, r in sp.nlargest(5, "cos_spike_vs_pureSERS").iterrows():
        ax.annotate(r.analyte, (r.spike_displacement_norm, r.cos_spike_vs_pureSERS),
                    fontsize=6.5, xytext=(4, 2), textcoords="offset points")
    ax.axhline(0, ls="--", c=MUTED, lw=1)
    ax.set_xlabel("spike displacement magnitude"); ax.set_ylabel("direction agreement")
    ax.set_title("Analytes that move more, move correctly (r = +0.54)")
    ax.legend(fontsize=6, ncol=2)
    ax = axs[2]
    top = sp.nlargest(10, "cos_spike_vs_pureSERS").iloc[::-1]
    ax.barh(range(len(top)), top.cos_spike_vs_pureSERS,
            color=[fam_col.get(f, MUTED) for f in top.family])
    ax.set_yticks(range(len(top))); ax.set_yticklabels(top.analyte, fontsize=6.5)
    ax.axvline(p7["null_cos_vs_pureSERS_median"], ls="--", c=S, lw=1.2)
    ax.set_xlabel("cos vs pure-analyte direction")
    ax.set_title("The responders are strong Ag adsorbers")
    fig.tight_layout(); fig.savefig(FIG / "f4_serum_vs_pure.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # ── F5 component activation + controls ──
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.2))
    piv = act.pivot_table(index="analyte", columns="component", values="delta")
    sel = piv.loc[piv.abs().max(axis=1).nlargest(28).index]
    v = np.abs(sel.values).max()
    im = axs[0].imshow(sel.values, aspect="auto", cmap=DIV, vmin=-v, vmax=v)
    axs[0].set_yticks(range(len(sel))); axs[0].set_yticklabels(sel.index, fontsize=5.5)
    axs[0].set_xlabel("component"); axs[0].set_title("Phase 6 — component activation (spike − serum)")
    axs[0].grid(False); plt.colorbar(im, ax=axs[0], fraction=0.03)
    ax = axs[1]
    labels, vals, cols = [], [], []
    if "ils_blanks" in ctrl:
        for b, d in ctrl["ils_blanks"]["batch_drift_from_grand_mean"].items():
            labels.append(f"ILS blank batch {b}"); vals.append(d); cols.append(P)
    if "uricase_depletion" in ctrl:
        labels.append("uricase depletion\n|displacement|")
        vals.append(ctrl["uricase_depletion"]["displacement_norm"]); cols.append(S)
    if "isotopic_UA_vs_UAiso" in ctrl:
        labels.append("urate vs\nisotopic urate"); vals.append(ctrl["isotopic_UA_vs_UAiso"]["coordinate_distance"])
        cols.append(MUTED)
    ax.bar(range(len(vals)), vals, color=cols)
    ax.set_xticks(range(len(vals))); ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylabel("coordinate displacement")
    ax.set_title("Phase 11 — controls: no batch drift; depletion moves; isotopologues close")
    fig.tight_layout(); fig.savefig(FIG / "f5_activation_controls.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # ═══ PDF ═══
    with PdfPages(PDF_PATH) as pdf:
        fig = plt.figure(figsize=PAGE)
        fig.text(0.5, 0.80, "Serum Spike-in Projection Validation", ha="center", fontsize=22,
                 fontweight="bold", color=INK)
        fig.text(0.5, 0.757, "GAIRA V5 — controlled perturbations through the frozen Raman atlas",
                 ha="center", fontsize=12.5, color=MUTED)
        ax = fig.add_axes([0.12, 0.42, 0.76, 0.27]); ax.axis("off")
        rows = [["Atlas", f"NMF k=24, FROZEN (fingerprint {man['atlas']['fingerprint'][:12]}…, verified)"],
                ["Perturbation spectra", f"{sum(man['datasets'].values())} across "
                                         f"{len(man['datasets'])} controlled datasets"],
                ["Dose-response", "ILS adenine (6 substrate×laser arms, 15 labs) + ergothioneine"],
                ["Serum spikes", "51 analytes × 5 replicates into pooled serum"],
                ["Modality", "Ag/Au-SERS — OUT OF DOMAIN for a Raman atlas by construction"],
                ["Dose-response result", "monotonic & saturating in 7/7 arms (ρ 0.93–1.00, p=0.002)"],
                ["Serum-spike result", "no overall directional agreement; 6 strong Ag adsorbers succeed"],
                ["Atlas modified?", "No — verified byte-identical before and after"]]
        t = ax.table(cellText=rows, colWidths=[0.30, 0.70], loc="center", cellLoc="left")
        t.auto_set_font_size(False); t.set_fontsize(8.3); t.scale(1, 1.6)
        for (r, c), cell in t.get_celld().items():
            cell.set_edgecolor(GRIDC)
            if c == 0: cell.set_text_props(fontweight="bold")
        fig.text(0.5, 0.33, "PRINCIPAL FINDING", ha="center", fontsize=11, fontweight="bold", color=INK)
        fig.text(0.5, 0.185,
                 "Concentration is registered; chemical identity mostly is not. Pure-analyte dose series\n"
                 "move monotonically and saturate in every arm, far above a label-permutation null, and\n"
                 "enzymatic depletion moves in the chemically correct direction. But serum spikes at\n"
                 "physiological levels show no directional agreement with their own pure-analyte reference\n"
                 "— except for strong silver adsorbers (hypoxanthine, xanthine, guanine, ergothioneine).",
                 ha="center", fontsize=9.3, color=INK)
        fig.text(0.5, 0.06, "Validation study · atlas unmodified · nothing pushed", ha="center",
                 fontsize=8, color=MUTED)
        pdf.savefig(fig); plt.close(fig)

        text_page(pdf, "Executive summary", [
            ("h", "Question"),
            ("b", "Does a known biochemical perturbation move a spectrum through the frozen Raman "
                  "reference manifold in a physically meaningful, reproducible way? The atlas, its "
                  "preprocessing, its basis and its ontology were frozen throughout; the study only "
                  "projects."),
            ("h", "A constraint that frames every result"),
            ("b", "Every controlled perturbation dataset in GAIRA is Ag- or Au-SERS, while the atlas is "
                  "built from pure Raman references. All projections here are therefore OUT OF DOMAIN by "
                  "construction (median OOD 0.15–0.28 versus in-domain Raman references). Per the study "
                  "design this is not treated as failure; the question is whether motion remains "
                  "internally coherent despite it."),
            ("h", "What the data support"),
            ("b", f"1. Dose-response is real and strong. In all seven arms (six ILS adenine "
                  f"substrate×laser combinations across 15 laboratories, plus ergothioneine) distance "
                  f"from the blank rises monotonically with concentration: Spearman ρ = "
                  f"{traj.monotonicity_rho.min():.2f}–{traj.monotonicity_rho.max():.2f} against a "
                  f"label-permutation null of ≈{traj.null_mean_abs_rho.mean():.2f}, p = 0.002 in every arm."),
            ("b", "2. The dose relationship is saturating, not linear. A Langmuir-type saturating model "
                  "beat linear and logarithmic fits in 7/7 arms — the expected behaviour for analytes "
                  "competing for a finite number of adsorption sites on a colloid surface."),
            ("b", f"3. Replicate displacements are reproducible: median direction cosine "
                  f"{rep.direction_cos_mean.median():.2f} across concentrations and "
                  f"{sp.replicate_direction_cos.median():.2f} for the serum spikes."),
            ("b", "4. Controls behave correctly. Unspiked serum replicates agree at 0.999; ILS blanks "
                  "show no systematic batch drift (max 0.024); enzymatic urate depletion moves AWAY from "
                  "the urate direction (cos −0.61), which is the chemically expected sign; and urate "
                  "versus isotopically-labelled urate remain close (cos 0.87), as isotopologues should."),
            ("h", "What the data do NOT support"),
            ("b", f"5. Serum spikes do not, in general, move toward their own pure-analyte reference. "
                  f"Median cos(spike displacement, pure-SERS direction) = "
                  f"{p7['matched_cos_vs_pureSERS_median']:+.3f} versus a mismatched-analyte null of "
                  f"{p7['null_cos_vs_pureSERS_median']:+.3f}; only {p7['n_analytes_cos_above_null_p05']} "
                  f"of {p7['n_analytes']} analytes exceed the 95th percentile of that null. Chemical "
                  f"identity is largely NOT recovered from a physiological-level spike in serum."),
            ("b", "6. Trajectories are monotonic in distance but not smooth in direction: mean "
                  "consecutive-step cosine ranges from +0.43 to −0.26, i.e. several arms zig-zag while "
                  "still advancing. Monotonic magnitude is not the same as a coherent path."),
            ("h", "Interpretation (distinguished from the observations above)"),
            ("b", "The analytes that do succeed are chemically the ones that should: hypoxanthine (0.89), "
                  "xanthine (0.74), guanine (0.41) — purines that chemisorb to silver through ring "
                  "nitrogen lone pairs — and ergothioneine (0.53), whose thione sulfur binds silver "
                  "strongly. Displacement magnitude and direction agreement correlate at r = +0.54. The "
                  "consistent reading is that the atlas registers a perturbation when the analyte "
                  "generates enough SERS signal to rise above the colloid background, and is blind to it "
                  "otherwise. That is a property of the Ag-colloid measurement, not of the atlas."),
            ("h", "Speculation (explicitly labelled)"),
            ("b", "If the limiting factor is surface competition rather than the manifold, then spikes "
                  "measured at higher effective surface coverage — or with a background-suppressed "
                  "acquisition — might recover directional agreement for weaker adsorbers. This study "
                  "cannot test that claim and does not assert it."),
        ], f"{sum(man['datasets'].values())} spectra · atlas frozen and verified unchanged")

        for name, cap in [
            ("f1_preprocessing.png", "Figure 1 — Phase 2 preprocessing cascade. Cosmic-ray removal is "
             "applied only where the wavenumber axis oversamples the narrowest real band; on the "
             "3 cm⁻¹ ILS axis it is declined rather than risk deleting genuine signal."),
            ("f2_dose_response.png", "Figure 2 — Phase 4/8 dose-response. Distance from the blank rises "
             "monotonically and saturates in every arm, far above a label-permutation null."),
            ("f3_reproducibility_ood.png", "Figure 3 — Phase 5/10. Replicate direction reproducibility, "
             "trajectory smoothness (negative step cosine indicates zig-zag), and out-of-domain distance "
             "for every dataset."),
            ("f4_serum_vs_pure.png", "Figure 4 — Phase 7, the decisive test. Overall there is no "
             "directional agreement between a serum spike and its pure-analyte reference; the exceptions "
             "are strong silver adsorbers, and displacement magnitude predicts direction agreement."),
            ("f5_activation_controls.png", "Figure 5 — Phase 6/11. Component activation per spiked "
             "analyte, and the control panel: no batch drift, depletion moves, isotopologues stay close."),
        ]:
            fig = plt.figure(figsize=PAGE)
            img = plt.imread(FIG / name)
            axi = fig.add_axes([0.04, 0.42, 0.92, 0.46]); axi.imshow(img); axi.axis("off")
            fig.text(0.07, 0.94, cap.split("—")[0].strip(), fontsize=13, fontweight="bold", color=INK)
            y = 0.37
            for ln in _wrap(cap, 100):
                fig.text(0.07, y, ln, fontsize=8.7, color=INK); y -= 0.017
            pdf.savefig(fig); plt.close(fig)

        # data + results tables
        b1 = [("h", "Phase 1 — dataset audit"),
              ("m", f"{'dataset':18s} {'n':>5s} {'analytes':>8s} {'levels':>7s} {'labs':>5s} "
                    f"{'substrates':>22s} {'lasers':>9s}")]
        for _, r in audit.iterrows():
            b1.append(("m", f"{r.dataset:18s} {int(r.n_spectra):>5d} {int(r.n_analytes):>8d} "
                            f"{('' if pd.isna(r.n_concentrations) else int(r.n_concentrations)):>7} "
                            f"{int(r.labs_instruments):>5d} {str(r.substrates)[:22]:>22s} "
                            f"{str(r.lasers_nm):>9s}"))
        b1 += [("h", "Phase 2 — preprocessing decisions"),
               ("b", "Projection into a frozen NMF basis is only meaningful in the representation the "
                     "basis was fitted in, so the atlas-native pipeline (ASLS → Savitzky-Golay → L2 on "
                     "450–1800 cm⁻¹ @ 2 cm⁻¹) is mandatory and was NOT tuned. Only the pre-steps were "
                     "evaluated."),
               ("b", "Cosmic-ray removal was initially over-aggressive: a naive large-residual rule "
                     "flagged 13% of points on the 3 cm⁻¹ ILS axis, i.e. it was deleting real bands. The "
                     "rule was replaced by a sharpness test that is applied only when the axis "
                     "oversamples the narrowest plausible band (≤2 cm⁻¹/point). On the ILS axis "
                     "despiking is therefore declined and recorded as skipped; on the 1.7 cm⁻¹ B&WTek "
                     "axis it removes 0.09% of points."),
               ("m", f"replicate cosine by dataset (median): " +
                     ", ".join(f"{k}={v:.3f}" for k, v in
                               qc.groupby('dataset').replicate_cos_mean.median().items())),
               ("b", f"{len(pd.read_csv(TAB / 'phase2_exclusions.csv'))} replicate spectra were flagged "
                     f"as robust-z outliers versus their group median. They are documented in "
                     f"phase2_exclusions.csv and were retained in the analysis; no spectrum was removed "
                     f"on the basis of it being inconvenient.")]
        text_page(pdf, "Datasets and preprocessing", b1)

        b2 = [("h", "Phase 4/8 — trajectory and dose-response model"),
              ("m", f"{'experiment':26s} {'lev':>4s} {'rho':>6s} {'p':>6s} {'null':>6s} "
                    f"{'straight':>9s} {'step-cos':>9s} {'model':>11s} {'linR2':>6s}")]
        for _, r in traj.iterrows():
            b2.append(("m", f"{r.experiment[:26]:26s} {int(r.n_levels):>4d} {r.monotonicity_rho:>6.3f} "
                            f"{r.monotonicity_p_perm:>6.3f} {r.null_mean_abs_rho:>6.3f} "
                            f"{r.straightness:>9.3f} {r.mean_step_cosine:>9.3f} "
                            f"{str(r.best_dose_model):>11s} {r.linear_r2:>6.3f}"))
        b2 += [("h", "Phase 9 — mixture behaviour"), ("b", mix["reason"]),
               ("h", "Phase 11 — controls"),
               ("m", f"unspiked serum: replicate cos {ctrl['unspiked_serum']['coord_cos_mean']:.4f}, "
                     f"OOD {ctrl['unspiked_serum']['median_ood']:.3f}")]
        if "ils_blanks" in ctrl:
            b2.append(("m", f"ILS blanks (n={ctrl['ils_blanks']['n']}): replicate cos "
                            f"{ctrl['ils_blanks']['coord_cos_mean']:.3f}, max batch drift "
                            f"{ctrl['ils_blanks']['max_batch_drift']:.4f} — no systematic drift"))
        if "uricase_depletion" in ctrl:
            b2.append(("m", f"uricase depletion: |Δ| {ctrl['uricase_depletion']['displacement_norm']:.3f}, "
                            f"cos vs urate direction {ctrl['uricase_depletion']['cos_vs_urate_raman_direction']:+.3f} "
                            f"(negative = chemically expected)"))
        if "isotopic_UA_vs_UAiso" in ctrl:
            b2.append(("m", f"urate vs isotopic urate: distance "
                            f"{ctrl['isotopic_UA_vs_UAiso']['coordinate_distance']:.3f}, cos "
                            f"{ctrl['isotopic_UA_vs_UAiso']['cosine']:.3f}"))
        text_page(pdf, "Trajectories, mixtures and controls", b2)

        b3 = [("h", "Phase 7 — serum spike versus pure analyte"),
              ("m", f"matched cos (median)      {p7['matched_cos_vs_pureSERS_median']:+.4f}"),
              ("m", f"mismatched null (median)  {p7['null_cos_vs_pureSERS_median']:+.4f}"),
              ("m", f"median angle              {p7['median_angle_vs_pureSERS_deg']:.1f} deg"),
              ("m", f"median distance ratio     {p7['median_distance_ratio']:.3f}"),
              ("m", f"analytes beating 95th pct null   {p7['n_analytes_cos_above_null_p05']}/{p7['n_analytes']}"),
              ("h", "Analytes with directional agreement"),
              ("m", f"{'analyte':16s} {'family':18s} {'conc µM':>8s} {'cos':>7s} {'angle':>7s} {'|Δ|':>7s}")]
        for _, r in sp.nlargest(10, "cos_spike_vs_pureSERS").iterrows():
            b3.append(("m", f"{r.analyte[:16]:16s} {str(r.family)[:18]:18s} {r.spike_conc_uM:>8.1f} "
                            f"{r.cos_spike_vs_pureSERS:>7.3f} {r.angle_spike_vs_pureSERS_deg:>7.1f} "
                            f"{r.spike_displacement_norm:>7.3f}"))
        b3 += [("h", "Limitations"),
               ("b", "• Every perturbation dataset is Ag/Au-SERS projected into a Raman atlas; nothing "
                     "here validates in-domain Raman behaviour."),
               ("b", "• Each serum spike has a single concentration, so no in-serum dose-response and no "
                     "combinatorial mixture test (Phase 9) is possible."),
               ("b", "• Distance-from-control can increase for reasons unrelated to the analyte "
                     "(colloid aggregation state, laser power, surface fouling). The permutation null "
                     "controls for concentration-label assignment but not for a confound that varies "
                     "monotonically with concentration by design."),
               ("b", "• Prior GAIRA work established these Ag-SERS spectra are background-dominated; the "
                     "serum-spike null is consistent with that and should not be read as a property of "
                     "the Raman atlas."),
               ("h", "Recommended next experiments"),
               ("b", "1. An in-domain Raman dose-response (pure analyte, Raman, several concentrations) "
                     "to separate 'the atlas cannot track concentration' from 'Ag-SERS is the limiting "
                     "factor'. No such dataset currently exists in GAIRA and this is the single most "
                     "informative missing measurement."),
               ("b", "2. Serum spikes at a concentration series rather than one level, so in-serum "
                     "dose-response and detection thresholds can be estimated per analyte."),
               ("b", "3. Blank-colloid difference acquisition, to test directly whether background "
                     "suppression restores directional agreement for weak adsorbers.")]
        text_page(pdf, "Serum spike test, limitations and next steps", b3)

        d = pdf.infodict()
        d["Title"] = "GAIRA V5 — Serum Spike-in Projection Validation"
    print(f"PDF written: {PDF_PATH} ({PDF_PATH.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
