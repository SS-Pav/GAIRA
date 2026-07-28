"""Foundation audit — Part 6: complete per-component audit.

For each of the 24 frozen NMF components, gathers (from the frozen artifacts only):
basis spectrum + top bands, top reference-analyte loadings + chemical classes, source
datasets, contribution entropy, purity, bootstrap stability, variance share, linked
biochemical themes (ontology W), linked MSS motifs (MSS M), and collision/redundancy
notes. Renders one markdown page per component, a 24-panel basis grid, a per-component
basis figure, a global summary CSV and the global classification.

Deterministic; reads results/v5_rebuild/{foundation,engine_v1}/artifacts. Writes to
foundation_audit/{components,figures,tables,reports}. Modifies nothing frozen.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/Users/surajpg/projects/GAIRA"); sys.path.insert(0, str(REPO / "src"))
from gaira.engine import GAIRAEngine
from gaira.engine.mss import MSSLayer
from gaira.foundation.families_raman import family_of

AUD = REPO / "results/v5_rebuild/foundation_audit"
COMP, FIG, TAB, REP = AUD / "components", AUD / "figures", AUD / "tables", AUD / "reports"
for p in (COMP, FIG, TAB, REP): p.mkdir(parents=True, exist_ok=True)
INK = "#1b2430"; ACC = "#2a6f97"


def entropy_eff(weights):
    w = np.asarray(weights, float); w = w[w > 0]; w = w / (w.sum() + 1e-12)
    if len(w) == 0:
        return 0.0
    H = -np.sum(w * np.log(w + 1e-12))
    return float(np.exp(H))                       # effective number of contributors


def main():
    eng = GAIRAEngine()
    reg = eng.builder.reg
    onto = eng.builder.onto
    mss = MSSLayer.from_engine(eng)
    H, grid = reg.basis()                          # (24, 676), (676,)
    stats = json.loads((REPO / "results/v5_rebuild/foundation/artifacts/manifold.json").read_text())["stats"]
    pcv = np.array(stats["per_component_variance"])
    k = H.shape[0]

    # component-component cosine similarity (redundancy)
    Hn = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-12)
    S = Hn @ Hn.T; np.fill_diagonal(S, 0.0)

    # 24-panel basis grid
    fig, axes = plt.subplots(6, 4, figsize=(15, 16))
    for j, ax in enumerate(axes.ravel()):
        ax.plot(grid, H[j], color=ACC, lw=0.8)
        lab = reg.value(j, "audit_label_v0_1")
        ax.set_title(f"c{j} · {lab} (stab {reg.stability(j):.2f})", fontsize=8.5)
        ax.tick_params(labelsize=6)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.suptitle("The 24 frozen NMF basis spectra (H) — Raman motifs, 450–1800 cm⁻¹", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.985]); fig.savefig(FIG / "basis_grid_24.png", dpi=110); plt.close(fig)

    rows = []
    for j in range(k):
        c = reg.get(j)
        bands = reg.value(j, "dominant_raman_peaks_cm")
        loads = reg.value(j, "reference_analyte_loadings")
        top_load = loads[:8]
        fams = [family_of(l["analyte"]) for l in top_load]
        fam_counts = pd.Series(fams).value_counts().to_dict()
        eff = entropy_eff([l["contribution_pct"] for l in loads])
        purity = reg.value(j, "purity")
        stab = reg.stability(j)
        # themes (ontology W row)
        wrow = onto.W[j]
        theme_order = np.argsort(-wrow)
        top_themes = [(onto.theme_ids[t], float(wrow[t])) for t in theme_order if wrow[t] > 0.02][:5]
        # linked MSS motifs (component weight in M column)
        linked = [(mss.motifs[mi].id, float(mss.M[j, mi])) for mi in range(mss.M.shape[1])
                  if mss.M[j, mi] > 0]
        linked.sort(key=lambda t: -t[1])
        # redundancy
        nn = int(np.argmax(S[j])); nn_sim = float(S[j, nn])
        rows.append({
            "component": j, "audit_label": reg.value(j, "audit_label_v0_1"),
            "top_theme": top_themes[0][0] if top_themes else "unassigned",
            "top_theme_w": round(top_themes[0][1], 3) if top_themes else 0.0,
            "n_themes>0.1": int((wrow > 0.1).sum()),
            "purity": round(purity, 3), "stability": round(stab, 3),
            "variance_share": round(float(pcv[j]), 4),
            "eff_contributors": round(eff, 1),
            "top_analyte": top_load[0]["analyte"] if top_load else "",
            "top_family": max(fam_counts, key=fam_counts.get) if fam_counts else "",
            "n_families_top8": len(set(fams)),
            "nearest_comp": nn, "nearest_cos": round(nn_sim, 3),
            "n_linked_motifs": len(linked),
            "confidence": reg.value(j, "interpretation_confidence"),
            "n_dose_responsive": reg.value(j, "n_dose_experiments_responsive"),
        })

        # per-component basis figure
        f, ax = plt.subplots(figsize=(7.2, 2.6))
        ax.plot(grid, H[j], color=ACC, lw=1.0)
        for b in bands:
            ax.axvline(b, color="#b2182b", lw=0.7, ls=":")
            ax.text(b, ax.get_ylim()[1] * 0.92, f"{int(b)}", fontsize=6.5, color="#b2182b",
                    ha="center", rotation=90)
        ax.set_xlabel("Raman shift (cm⁻¹)", fontsize=8); ax.set_ylabel("loading", fontsize=8)
        ax.set_title(f"Component c{j} basis spectrum", fontsize=9.5, color=INK)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        f.tight_layout(); f.savefig(FIG / f"component_c{j:02d}.png", dpi=120); plt.close(f)

        # per-component markdown page
        theme_lines = "\n".join(
            f"  - `{t}` — weight **{w:.3f}**" + (f"  ·  {onto.themes[t].get('short','')}" if t in onto.themes else "")
            for t, w in top_themes)
        load_lines = "\n".join(
            f"  - {l['analyte']} — {l['contribution_pct']:.2f}%  ({family_of(l['analyte'])})"
            for l in top_load)
        motif_lines = ("\n".join(f"  - {mid} — component weight {w:.3f}" for mid, w in linked)
                       if linked else "  - (none above threshold)")
        collide = []
        if len(set(fams[:5])) >= 3:
            collide.append(f"top-5 analytes span {len(set(fams[:5]))} chemical families "
                           f"({', '.join(sorted(set(fams[:5])))}) — a mixed/collision component")
        if reg.value(j, "audit_label_v0_1") != (top_themes[0][0].split('_')[0] if top_themes else ""):
            pass
        if nn_sim > 0.85:
            collide.append(f"basis cosine {nn_sim:.2f} to c{nn} — possible redundancy")
        cav = reg.value(j, "known_caveats")
        page = f"""# Component c{j}

**Audit label:** `{reg.value(j,'audit_label_v0_1')}`  ·  **interpretation confidence:** {reg.value(j,'interpretation_confidence')}

![basis](../figures/component_c{j:02d}.png)

**Interpretation (registry).** {reg.value(j,'current_interpretation')}

| metric | value |
|---|---|
| bootstrap stability | **{stab:.3f}** |
| purity (theme) | {purity:.3f} |
| variance share | {pcv[j]:.4f} |
| effective # contributing analytes | {eff:.1f} |
| dominant Raman bands (cm⁻¹) | {', '.join(str(int(b)) for b in bands)} |
| top chemical family (top-8 loadings) | {max(fam_counts, key=fam_counts.get) if fam_counts else '—'} |
| dose-responsive experiments | {reg.value(j,'n_dose_experiments_responsive')} |
| nearest component (basis cosine) | c{nn} ({nn_sim:.3f}) |

**Top reference-analyte loadings**
{load_lines}

**Linked biochemical themes** (component→theme weights, ontology v2)
{theme_lines if theme_lines else '  - (none > 0.02)'}

**Linked MSS motifs** (component→motif weights, MSS v1)
{motif_lines}

**Collision / redundancy notes**
{chr(10).join('  - ' + x for x in collide) if collide else '  - none flagged'}

**Known caveats (registry).** {cav if cav else '—'}
"""
        (COMP / f"component_c{j:02d}.md").write_text(page)

    df = pd.DataFrame(rows)
    df.to_csv(TAB / "component_audit_summary.csv", index=False)

    # ── global classification ──
    def classify(r):
        tags = []
        tf = r["top_theme"]
        if r["top_theme_w"] >= 0.45 and r["purity"] >= 0.35:
            tags.append("clean")
        if r["n_families_top8"] >= 4 or r["purity"] < 0.22:
            tags.append("mixed/ambiguous")
        if r["nearest_cos"] >= 0.85:
            tags.append("redundant")
        if r["nearest_cos"] < 0.55:
            tags.append("unique")
        return tags
    df["tags"] = df.apply(classify, axis=1)
    prot = df[df.top_theme.str.contains("protein")].component.tolist()
    puri = df[df.top_theme.str.contains("purine")].component.tolist()
    lipid = df[df.top_theme.str.contains("lipid|sterol")].component.tolist()
    sacc = df[df.top_theme.str.contains("saccharide|glycan")].component.tolist()
    clean = df[df.tags.apply(lambda t: "clean" in t)].component.tolist()
    ambig = df[df.tags.apply(lambda t: "mixed/ambiguous" in t)].component.tolist()
    redun = df[df.tags.apply(lambda t: "redundant" in t)].component.tolist()
    uniq = df[df.tags.apply(lambda t: "unique" in t)].component.tolist()

    glob = {
        "n_components": k,
        "mean_stability": round(float(df.stability.mean()), 3),
        "min_stability": round(float(df.stability.min()), 3),
        "protein_dominated": prot, "purine_dominated": puri,
        "lipid_sterol_dominated": lipid, "saccharide_dominated": sacc,
        "chemically_clean": clean, "mixed_ambiguous": ambig,
        "redundant": redun, "unique": uniq,
        "max_pairwise_basis_cosine": round(float(S.max()), 3),
    }
    (TAB / "component_global_classification.json").write_text(json.dumps(glob, indent=2))
    print(json.dumps(glob, indent=2))
    print("\nper-component summary:")
    print(df[["component", "audit_label", "top_theme", "top_theme_w", "purity",
              "stability", "variance_share", "n_families_top8", "nearest_comp",
              "nearest_cos", "tags"]].to_string(index=False))


if __name__ == "__main__":
    main()
