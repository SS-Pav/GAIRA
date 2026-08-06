#!/usr/bin/env python3
"""GAIRA V7 — Phase 03: emergent biochemical theme discovery.

    49 frozen CSMs + frozen Phase 02.5 geometry
        ↓  5 soft-membership models, K swept 2–15, no chemistry label visible
    label-free criteria + band-based admissibility veto
        ↓  smallest admissible K on the contiguous Pareto plateau
    nine validations
        ↓  post-hoc naming, with Unknown allowed
    themes: soft, sparse, overlapping, with bridges and unassigned CSMs preserved

Nothing upstream is refitted. Output location resolves through `gaira.v7.io.PhaseOutputs`,
so a run can be redirected with GAIRA_V7_OUTPUT_ROOT without editing this file.

    python results/v7_rebuild/phase03/code/run_phase03.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))            # repo/src
sys.path.insert(0, str(HERE.parents[1] / "phase00/code"))   # shared v7 path helpers

import v7_paths as P                                            # noqa: E402
from gaira.v7.io import PhaseOutputs, frozen_root               # noqa: E402
from gaira.v7.themes import criteria as CRIT                    # noqa: E402
from gaira.v7.themes import models as MOD                       # noqa: E402
from gaira.v7.themes import validation as VAL                   # noqa: E402
from gaira.v7.themes.registry import Theme, ThemeRegistry       # noqa: E402

warnings.filterwarnings("ignore")

PHASE, PHASE_NAME = "03", "Emergent biochemical theme discovery"
OUT = PhaseOutputs(PHASE).ensure()
FROZEN = frozen_root()
P01, P02, P025 = FROZEN / "phase01", FROZEN / "phase02", FROZEN / "phase02_5"
ATLAS_FP = "09ed804a40836f4a05a91ba10900cded"
LSM_FP = "208482d6f7178b5b8f16cace91be55b0"
CSM_FP = "0b4aa550ccefed3edabdbde5bae11c8d"
SEED, K_NN = 0, 5
LOG: list[str] = []


def log(m):
    line = f"[phase03] {m}"
    print(line, flush=True)
    LOG.append(line)


def wtab(df, name, where=None):
    p = (where or OUT.tables) / name
    df.to_csv(p, index=False)
    return {"artifact_id": name, "path": OUT.rel(p), "sha256": P.sha256_file(p), "rows": len(df)}


def wjson(obj, name, where=None):
    p = (where or OUT.artifacts) / name
    p.write_text(json.dumps(obj, indent=2, default=_ser))
    return {"artifact_id": name, "path": OUT.rel(p), "sha256": P.sha256_file(p)}


def _ser(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def main() -> int:
    outputs: list[dict] = []
    t0 = datetime.now(timezone.utc)

    # ── 0. architecture check and fingerprint gate ───────────────────────────
    log("architecture check — themes are derived FROM CSMs, never asserted over them (L-05); "
        "chemistry names only (P-07); soft overlapping membership retained")
    Hfroz = np.asarray(np.load(P.FOUNDATION / "manifold_components.npz")["components"], float)
    fp_atlas = P.sha256_array(Hfroz)
    s01 = json.loads((P01 / "PHASE_STATE.json").read_text())
    s02 = json.loads((P02 / "PHASE_STATE.json").read_text())
    s025 = json.loads((P025 / "PHASE_STATE.json").read_text())
    checks = {
        "atlas": (fp_atlas, ATLAS_FP),
        "lsm_registry": (s01["registry_fingerprint"], LSM_FP),
        "csm_dictionary": (s02["csm_fingerprint"], CSM_FP),
    }
    for k, (got, want) in checks.items():
        if got != want:
            log(f"ABORT: {k} fingerprint {got} != frozen {want}")
            return 1
    if s025["status"] != "COMPLETE" or s025["themes_created"]:
        log("ABORT: Phase 02.5 is not COMPLETE, or has created themes")
        return 1
    log(f"  frozen: atlas {fp_atlas} · LSM {LSM_FP} · CSM {CSM_FP} · "
        f"geometry {s025['primary_spectral_metric']}/{s025['primary_geometry']}")

    # ── 1. frozen inputs ─────────────────────────────────────────────────────
    z = np.load(P02 / "artifacts/csm_dictionary_v1.npz", allow_pickle=True)
    X = np.asarray(z["CSM"], float)
    csm_ids = [str(s) for s in z["csm_ids"]]
    grid = np.asarray(z["grid"], float)
    creg = json.loads((P02 / "artifacts/csm_registry_v1.json").read_text())
    by_id = {c["csm_id"]: c for c in creg["csms"]}
    csm_class = [(by_id[c]["supporting_classes"][0]
                  if len(by_id[c]["supporting_classes"]) == 1 else "multi") for c in csm_ids]

    g = np.load(P025 / "artifacts/geometry_v1.npz", allow_pickle=True)
    lsm_ids = [str(s) for s in g["motif_ids"]]
    lsm2csm = {l["lsm_id"]: c["csm_id"] for c in creg["csms"] for l in c["contributing_lsms"]}
    e = np.load(P025 / "artifacts/embeddings_v1.npz", allow_pickle=True)
    roles = pd.read_csv(P025 / "tables/graph_roles_v1.csv")
    priors = json.loads((P025 / "artifacts/phase03_geometry_priors.json").read_text())

    # LSM-level geometry lifted to CSM level. 48 of 49 CSMs are a single LSM, so this is very
    # nearly the identity; the one merged CSM takes the mean of its two LSM rows.
    idx_of_lsm = {m: i for i, m in enumerate(lsm_ids)}
    members = {c: [idx_of_lsm[l["lsm_id"]] for l in by_id[c]["contributing_lsms"]]
               for c in csm_ids}
    Dl = g["D_primary_metric"]
    D = np.zeros((len(csm_ids), len(csm_ids)))
    for a, ca in enumerate(csm_ids):
        for b, cb in enumerate(csm_ids):
            D[a, b] = float(np.mean(Dl[np.ix_(members[ca], members[cb])]))
    np.fill_diagonal(D, 0.0)
    coords = np.array([e["diffusion"][members[c]].mean(0) for c in csm_ids])
    A = np.zeros_like(D)
    for i in range(len(csm_ids)):
        for j in np.argsort(D[i])[1:K_NN + 1]:
            A[i, j] = A[j, i] = 1.0
    geom_bridges = {lsm2csm.get(m, m) for m in roles[roles.is_bridge].motif}
    geom_isolates = {lsm2csm.get(m, m) for m in roles[roles.is_isolated].motif}
    log(f"inputs: {X.shape} CSM spectra · geometry {D.shape} · "
        f"{len(geom_bridges)} geometry bridges · {len(geom_isolates)} geometry isolates · "
        f"{len(priors['priors'])} Phase 02.5 priors (evidence, not truth)")

    src_of, exc_of = _provenance(by_id, csm_ids)
    folds = np.array([hash(c) % 5 for c in csm_ids])   # placeholder; replaced below
    folds = _folds(csm_ids, 5)

    # ── 2. model × K sweep — NO chemistry label is visible in this block ──────
    log("sweeping 5 membership models × K = 2–15 (label-free criteria only)")
    # First pass: fit everything and collect band assignments, so family specificity can be
    # estimated across the whole sweep before any theme is judged on its families.
    fits: dict[tuple[str, int], dict] = {}
    for model in MOD.MODELS:
        for K in CRIT.K_RANGE:
            try:
                fits[(model, int(K))] = MOD.fit(model, K, X, D, coords, A)
            except Exception as exc:                       # pragma: no cover
                log(f"  {model} K={K} failed: {exc}")
    SPEC = CRIT.family_specificity(
        [[CRIT.admissibility(t, grid) for t in o["themes"]] for o in fits.values()])
    log("  mode-family specificity (higher = more discriminating): "
        + ", ".join(f"{k} {v:.2f}" for k, v in sorted(SPEC.items(), key=lambda kv: -kv[1])))
    rows = []
    store: dict[tuple[str, int], dict] = {}
    for model in MOD.MODELS:
        for K in CRIT.K_RANGE:
            def fit_fn(Xs, model=model, K=K):
                n = Xs.shape[0]
                if Xs.shape[0] == X.shape[0]:
                    return MOD.fit(model, K, Xs, D, coords, A)
                keep = _rows_of(X, Xs)
                return MOD.fit(model, K, Xs, D[np.ix_(keep, keep)], coords[keep],
                               A[np.ix_(keep, keep)])
            out = fits.get((model, int(K)))
            if out is None:
                continue
            S, Th = out["S"], out["themes"]
            adm = [CRIT.admissibility(t, grid, SPEC) for t in Th]
            dist = CRIT.theme_set_distinct(Th, adm)
            info = CRIT.information_retained(X, S, Th)
            deg = CRIT.membership_degenerate(S)
            r = {
                "model": model, "K": int(K),
                "information_retained": info,
                "max_theme_share": deg["max_theme_share"],
                "effective_K": deg["effective_K"],
                "degenerate": bool(info < CRIT.DEGENERATE_INFORMATION or deg["degenerate"]),
                "themes_distinct": bool(dist["distinct"]),
                "n_duplicate_theme_pairs": len(dist["duplicate_pairs"]),
                "heldout_reconstruction": CRIT.heldout_reconstruction(X, fit_fn, folds),
                "stability": CRIT.stability(fit_fn, X, n_boot=12, seed=SEED),
                "spectral_coherence": CRIT.spectral_coherence(X, S),
                "compression": 1.0 - K / X.shape[0],
                "calibration_ece": CRIT.calibration_ece(S, X, Th),
                "membership_sparsity": CRIT.membership_sparsity(S),
                "n_admissible_themes": int(sum(a["admissible"] for a in adm)),
                "chemically_admissible": bool(all(a["admissible"] for a in adm)),
            }
            rows.append(r)
            store[(model, int(K))] = out
        mine = [r for r in rows if r["model"] == model]
        deg = [r["K"] for r in mine if r["degenerate"]]
        adm_ks = [r["K"] for r in mine
                  if r["chemically_admissible"] and r["themes_distinct"] and not r["degenerate"]]
        log(f"  {model:24s} admissible+distinct at K = {adm_ks if adm_ks else 'none'}"
            + (f"   [degenerate at K = {deg}]" if deg else ""))
    sweep = pd.DataFrame(rows)
    outputs.append(wtab(sweep, "model_k_sweep_v1.csv"))

    # ── 3. select the model, then K ──────────────────────────────────────────
    per_model = []
    for model in MOD.MODELS:
        sub = [r for r in rows if r["model"] == model]
        if not sub:
            continue
        sel = CRIT.select_K(sub)
        comp = CRIT.composite(sub)
        per_model.append({"model": model, **{k: v for k, v in sel.items() if k != "rationale"},
                          "n_degenerate_K": int(sum(r["degenerate"] for r in sub)),
                          "best_composite": float(np.nanmax(np.where(
                              [r["chemically_admissible"] and r["themes_distinct"]
                               and not r["degenerate"] for r in sub], comp, np.nan)))
                          if any(r["chemically_admissible"] and r["themes_distinct"]
                                 and not r["degenerate"] for r in sub) else float("nan"),
                          "rationale": sel["rationale"]})
    model_tab = pd.DataFrame(per_model)
    outputs.append(wtab(model_tab, "model_comparison_v1.csv"))
    live = model_tab[model_tab.status == "PASS"]
    if live.empty:
        log("ABORT: no model produces a chemically admissible K anywhere in the sweep")
        wjson({"model_comparison": per_model}, "k_selection_v1.json")
        return 2
    winner = live.sort_values("best_composite", ascending=False).iloc[0]
    model, K = str(winner.model), int(winner.K)
    sel = CRIT.select_K([r for r in rows if r["model"] == model])
    log(f"SELECTED: {model}, K = {K}")
    log(f"  {sel['rationale']}")
    outputs.append(wjson({"selected_model": model, "selected_K": K, "selection": sel,
                          "model_comparison": per_model,
                          "criteria": {k: {"weight": v[0], "direction": "max" if v[1] > 0
                                           else "min"} for k, v in CRIT.CRITERIA.items()},
                          "labels_used_in_selection": False},
                         "k_selection_v1.json"))

    fit = store[(model, K)]
    S, Th = fit["S"], fit["themes"]
    Th = np.clip(Th, 0.0, None)
    Th = Th / (np.linalg.norm(Th, axis=1, keepdims=True) + 1e-12)

    def refit(Xs):
        keep = _rows_of(X, Xs)
        return MOD.fit(model, K, Xs, D[np.ix_(keep, keep)], coords[keep], A[np.ix_(keep, keep)])

    # ── 4. validation ────────────────────────────────────────────────────────
    log("validating")
    boot = VAL.bootstrap_stability(refit, X, S, n_boot=40, seed=SEED)
    loo = VAL.leave_one_out_stability(refit, X, S)
    nbc = VAL.neighbour_consistency(S, D, K_NN)
    modu = VAL.theme_modularity(S, D, csm_ids, K_NN, n_null=40, seed=SEED)
    rec = VAL.reconstruction_comparison(X, Th)
    rob = VAL.robustness(refit, X, S, _groups(src_of) | {f"exc::{k}": v for k, v
                                                          in _groups(exc_of).items()})
    outputs.append(wtab(rob, "robustness_v1.csv", OUT.validation))
    grad = VAL.theme_gradients(S, coords, csm_ids, n_perm=300, seed=SEED)
    outputs.append(wtab(grad, "theme_gradients_v1.csv", OUT.validation))
    roles_tab = VAL.membership_roles(S, csm_ids, X, Th, geom_bridges, geom_isolates)
    outputs.append(wtab(roles_tab, "membership_roles_v1.csv"))
    hier = VAL.infer_hierarchy(Th, S)
    outputs.append(wjson(hier, "hierarchy_v1.json"))
    log(f"  bootstrap {boot['mean']:.3f} (min {boot['min']:.3f}) · LOO {loo['mean']:.3f} · "
        f"kNN agreement {nbc['knn_agreement']:.3f} vs chance {nbc['chance']:.3f}")
    log(f"  modularity {modu['observed']:.3f} vs null {modu['null_mean']:.3f} "
        f"(p = {modu['p_empirical']:.3f}) · EV theme basis {rec['ev_theme_basis']:.3f} vs "
        f"CSM basis {rec['ev_csm_basis']:.3f}")
    log(f"  hierarchy: {hier['n_levels']} level(s) inferred · "
        f"{int(grad.is_gradient.sum())} of {len(grad)} theme×coordinate pairs are gradients")
    log(f"  roles: {dict(roles_tab.role.value_counts())}")

    # ── 5. POST HOC — labels revealed here for the first time ────────────────
    log("POST HOC: revealing curated chemistry for interpretation and agreement only")
    onto = VAL.ontology_agreement(S, csm_class, n_perm=500, seed=SEED)
    value = VAL.value_over_csm(S, X, Th, csm_class, D, K_NN)
    log(f"  ontology AMI {onto['ami']:.3f} (adjusted — the honest statistic); NMI "
        f"{onto['nmi']:.3f} against a permutation null of {onto['null_mean']:.3f}, "
        f"p = {onto['p_empirical']:.4f}")
    log(f"  value over CSM layer: retrieval {value['retrieval_theme_coordinates']:.3f} "
        f"(themes) vs {value['retrieval_csm_basis']:.3f} (CSMs) — "
        f"{'ADDS VALUE' if value['theme_layer_adds_value'] else 'DOES NOT'}")
    outputs.append(wjson({"ontology_agreement": onto, "value_over_csm_layer": value,
                          "revealed_after_K_selected": True}, "post_hoc_v1.json"))

    # ── 6. build the registry ────────────────────────────────────────────────
    reg = ThemeRegistry(model, K, sel, {
        "atlas_fingerprint": fp_atlas, "lsm_registry_fingerprint": LSM_FP,
        "csm_dictionary_fingerprint": CSM_FP,
        "geometry_metric": s025["primary_spectral_metric"],
        "geometry_fusion": s025["primary_geometry"]})
    reg.unassigned_csms = roles_tab[roles_tab.role == "unassigned"].csm_id.tolist()
    reg.bridge_csms = roles_tab[roles_tab.role == "bridge"].csm_id.tolist()

    from scipy.stats import entropy as _ent
    for k in range(K):
        adm = CRIT.admissibility(Th[k], grid, SPEC)
        order = np.argsort(-S[:, k])
        mem = [(csm_ids[i], float(S[i, k])) for i in order if S[i, k] >= 0.15]
        brid = [c for c, _ in mem if c in set(reg.bridge_csms)]
        gsub = grad[(grad.theme == k) & grad.is_gradient]
        rr = rob[rob.testable]
        t = Theme(
            theme_id=f"Theme-{k + 1:02d}", index=k, spectrum=Th[k],
            dominant_bands=adm["bands_cm1"], band_assignments=adm["assignments"],
            mode_families=adm["mode_families"],
            dominant_families=adm["dominant_families"],
            family_concentration=adm["family_concentration"],
            chemically_admissible=adm["admissible"],
            assigned_fraction=adm["assigned_fraction"],
            member_csms=[c for c, _ in mem], member_memberships=[v for _, v in mem],
            bridge_csms=brid, n_supporting_csms=len(mem),
            mean_membership=float(S[:, k].mean()),
            membership_entropy=float(_ent(S[:, k] + 1e-12)),
            geometry_evidence={
                "mean_pairwise_distance": round(float(_mean_within(D, order[:max(len(mem), 2)])), 4),
                "knn_agreement": round(float(np.mean([nbc["per_csm"][i] for i in order[:len(mem)]]))
                                       if mem else 0.0, 4),
                "modularity_p": modu["p_empirical"]},
            spectral_evidence={
                "n_dominant_bands": len(adm["bands_cm1"]),
                "family_concentration": round(adm["family_concentration"], 3),
                "dominant_families": ",".join(adm["dominant_families"]),
                "all_mode_families": ",".join(adm["mode_families"])},
            biochemical_evidence={},                 # filled at naming
            bootstrap_stability=float(boot["per_theme"][k]),
            loo_stability=float(loo["per_theme"][k]),
            gradient={"n_gradient_coords": int(len(gsub)),
                      "strongest": (f"DC{int(gsub.iloc[0].diffusion_coord)} "
                                    f"rho={gsub.iloc[0].spearman:.2f}") if len(gsub) else "none"},
            source_robust=bool(rr.theme_recovery.min() >= 0.60) if len(rr) else True,
        )
        reg.add(t)

    # ── 7. counter-evidence, alternatives, acceptance ────────────────────────
    for t in reg.themes:
        ce, alt = [], []
        if t.bootstrap_stability < 0.75:
            ce.append(f"bootstrap recovery only {t.bootstrap_stability:.2f}")
        if t.loo_stability < 0.80:
            ce.append(f"leave-one-out recovery only {t.loo_stability:.2f}")
        if t.n_supporting_csms < 3:
            ce.append(f"only {t.n_supporting_csms} CSMs carry membership >= 0.15")
        if not t.chemically_admissible:
            ce.append("bands do not name a coherent chemistry")
        if t.family_concentration < 0.70:
            ce.append(f"only {t.family_concentration:.2f} of band prominence sits in its two "
                      f"dominant mode families — partly a mixture")
        if not t.source_robust:
            ce.append("does not survive leave-one-source-out at 0.60")
        if t.gradient["n_gradient_coords"] == 0:
            ce.append("membership shows no significant gradient over the manifold")
        if not ce:
            ce.append("none found by the nine checks; the theme survives every one")
        alt.append("shared Raman physics rather than shared biochemistry — tested by the "
                   "mode-family constraint and by band assignability")
        if "aliphatic" in t.dominant_families:
            alt.append("CH2/CH3 scissoring is the most common band in biological Raman; "
                       "aliphatic membership alone is weak evidence of lipid biology")
        if "ring" in t.dominant_families:
            alt.append("aromatic and heterocyclic ring modes are shared by purines, "
                       "pyrimidines, aromatic amino acids and pigments alike")
        t.counter_evidence, t.alternative_explanations = ce, alt
        fails = []
        if t.bootstrap_stability < 0.60:
            fails.append(f"bootstrap recovery {t.bootstrap_stability:.2f} < 0.60")
        if not t.chemically_admissible:
            fails.append("not chemically admissible")
        if t.n_supporting_csms < 2:
            fails.append("fewer than two supporting CSMs")
        if fails:
            t.status, t.rejection_reason = "rejected", "; ".join(fails)
    n_rej = sum(t.status == "rejected" for t in reg.themes)
    log(f"acceptance: {len(reg.accepted)} accepted · {n_rej} rejected")

    # ── 8. naming — only now, and only chemistry ─────────────────────────────
    for t in reg.themes:
        name, definition, conf = _name(t, S, csm_ids, csm_class, X, grid)
        t.name, t.chemical_definition, t.name_confidence = name, definition, conf
        t.biochemical_evidence = {
            "dominant_classes": ";".join(_top_classes(S[:, t.index], csm_class, 3)),
            "ontology_ami_overall": round(onto["ami"], 3),
            "naming_confidence": round(conf, 2)}
        t.confidence = float(np.clip(
            0.35 * t.bootstrap_stability + 0.25 * t.loo_stability
            + 0.20 * conf + 0.20 * (1.0 if t.chemically_admissible else 0.0), 0, 1))
        t.limitations = ([f"interpretation weak — named {t.name}"] if conf < 0.5 else []) + \
                        ([f"{len(t.bridge_csms)} of its CSMs are bridges shared with other "
                          f"themes"] if t.bridge_csms else [])
        log(f"  {t.theme_id}: {t.name:34s} conf {t.confidence:.2f}  "
            f"n_csm {t.n_supporting_csms:2d}  bands {[int(b) for b in t.dominant_bands]}")

    # ── 9. artefacts ─────────────────────────────────────────────────────────
    inv = reg.check_invariants(S, csm_ids)
    outputs.append(wtab(pd.DataFrame(inv), "theme_invariants_v1.csv", OUT.validation))
    np.savez_compressed(OUT.artifacts / "theme_membership_v1.npz", S=S, THEMES=reg.basis(),
                        csm_ids=np.array(csm_ids, dtype=object),
                        theme_ids=np.array([t.theme_id for t in reg.themes], dtype=object),
                        grid=grid, D_csm=D, coords=coords)
    outputs.append({"artifact_id": "theme_membership_v1.npz",
                    "path": OUT.rel(OUT.artifacts / "theme_membership_v1.npz"),
                    "sha256": P.sha256_file(OUT.artifacts / "theme_membership_v1.npz")})
    outputs.append(wtab(reg.table(), "theme_catalogue_v1.csv"))
    memb = pd.DataFrame(S, index=csm_ids, columns=[t.theme_id for t in reg.themes])
    outputs.append(wtab(memb.reset_index().rename(columns={"index": "csm_id"}),
                        "theme_membership_v1.csv"))
    outputs.append(wjson(_registry_json(reg, S, csm_ids, hier, boot, loo, nbc, modu, rec,
                                        value, onto, roles_tab, grad),
                         "theme_registry_v1.json"))
    _write_yaml(reg, S, csm_ids, OUT.artifacts / "theme_registry_v1.yaml")
    outputs.append({"artifact_id": "theme_registry_v1.yaml",
                    "path": OUT.rel(OUT.artifacts / "theme_registry_v1.yaml"),
                    "sha256": P.sha256_file(OUT.artifacts / "theme_registry_v1.yaml")})

    # ── 10. gates and state ──────────────────────────────────────────────────
    gates = [
        _g("frozen inputs verified", True, "atlas, LSM registry, CSM dictionary, geometry"),
        _g("themes derived from CSMs, not asserted over them", True,
           "membership fitted on the frozen CSM dictionary (L-05)"),
        _g("no chemistry label used before K was selected", True,
           "labels revealed at step 5, after selection and validation"),
        _g("K justified on a Pareto frontier", sel["status"] == "PASS", sel["rationale"][:110]),
        _g("every theme chemically admissible",
           all(t.chemically_admissible for t in reg.accepted),
           f"{sum(t.chemically_admissible for t in reg.accepted)}/{len(reg.accepted)}"),
        _g("no disease/pathway/process/phenotype name (P-07)",
           all(i["status"] == "PASS" for i in inv if "disease" in i["invariant"]), "checked"),
        _g("soft membership retained; no forced single parent",
           all(i["status"] == "PASS" for i in inv if "single parent" in i["invariant"]),
           f"{int((S.max(1) < 0.999).sum())} CSMs have split membership"),
        _g("C-08 invariants hold", all(i["status"] == "PASS" for i in inv),
           f"{len(inv)} checked"),
        _g("bootstrap stability", boot["mean"] >= 0.70,
           f"mean {boot['mean']:.3f}, min {boot['min']:.3f}"),
        _g("theme layer value over CSM layer measured and reported", True, value["verdict"]),
        _g("bridges and unassigned CSMs preserved, not forced", True,
           f"{len(reg.bridge_csms)} bridges, {len(reg.unassigned_csms)} unassigned"),
    ]
    outputs.append(wtab(pd.DataFrame(gates), "phase03_gates_v1.csv", OUT.validation))
    all_pass = all(x["status"] == "PASS" for x in gates)
    log(f"gates: {sum(x['status'] == 'PASS' for x in gates)}/{len(gates)} PASS")

    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=P.REPO,
                                capture_output=True, text=True).stdout.strip())
    wjson({"schema": "gaira_v7_phase_manifest_v1", "phase": PHASE, "phase_name": PHASE_NAME,
           "built_utc": t0.isoformat(),
           "output_root": str(OUT.root), "redirectable_via": "GAIRA_V7_OUTPUT_ROOT",
           "frozen_inputs": {k: v[0] for k, v in checks.items()},
           "label_firewall": {"labels_used_before_K_selection": False,
                              "revealed_at_step": 5},
           "selected_model": model, "selected_K": K, "seed": SEED,
           "theme_fingerprint": reg.fingerprint(),
           "outputs": outputs, "gates": gates, "code_dirty": dirty,
           "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                           "pandas": pd.__version__}},
          "phase_03_manifest_v1.json")
    (OUT.root / "PHASE_STATE.json").write_text(json.dumps({
        "schema": "gaira_v7_phase_state_v1", "phase": PHASE, "phase_name": PHASE_NAME,
        "status": "COMPLETE" if all_pass else "GATE_FAILED",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "atlas_fingerprint": fp_atlas, "lsm_registry_fingerprint": LSM_FP,
        "csm_dictionary_fingerprint": CSM_FP,
        "selected_model": model, "K": K, "theme_fingerprint": reg.fingerprint(),
        "themes": reg.summary(),
        "bootstrap_mean": round(boot["mean"], 4), "loo_mean": round(loo["mean"], 4),
        "ontology_ami": round(onto["ami"], 4), "ontology_nmi": round(onto["nmi"], 4),
        "ontology_p": onto["p_empirical"],
        "theme_layer_adds_value": value["theme_layer_adds_value"],
        "n_hierarchy_levels": hier["n_levels"],
        "gates_passed": sum(x["status"] == "PASS" for x in gates), "gates_total": len(gates),
    }, indent=2))
    (OUT.logs / "phase03_run.log").write_text("\n".join(LOG))
    log("PHASE 03 " + ("COMPLETE" if all_pass else "GATE FAILED"))
    return 0 if all_pass else 3


# ── helpers ──────────────────────────────────────────────────────────────────
def _g(name, ok, detail):
    return {"gate": name, "status": "PASS" if ok else "FAIL", "detail": detail}


def _rows_of(X, Xs):
    """Indices of `Xs` rows within `X` — the resample bookkeeping the models need."""
    key = {tuple(np.round(r, 10)): i for i, r in enumerate(X)}
    return [key[tuple(np.round(r, 10))] for r in Xs]


def _folds(ids, n):
    """Deterministic folds by position — no hashing, which varies per process."""
    return np.array([i % n for i in range(len(ids))])


def _mean_within(D, idx):
    idx = list(idx)
    if len(idx) < 2:
        return 0.0
    return float(np.mean([D[a, b] for i, a in enumerate(idx) for b in idx[i + 1:]]))


def _provenance(by_id, csm_ids):
    canon = pd.read_csv(frozen_root() / "phase00/tables/canonical_analytes_v1.csv")
    so = {r.canonical_id: str(r.sources).split(";") for r in canon.itertuples()}
    eo = {r.canonical_id: str(r.excitations).split(";") for r in canon.itertuples()}

    def dom(cid, tbl):
        cnt = {}
        for a in by_id[cid]["supporting_analytes"]:
            for v in tbl.get(a, []):
                cnt[v] = cnt.get(v, 0) + 1
        return sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[0][0] if cnt else "?"
    return [dom(c, so) for c in csm_ids], [dom(c, eo) for c in csm_ids]


def _groups(vals):
    out = {}
    for i, v in enumerate(vals):
        out.setdefault(v, []).append(i)
    return out


def _top_classes(w, classes, n):
    agg = {}
    for c, x in zip(classes, w):
        agg[c] = agg.get(c, 0.0) + float(x)
    return [f"{k}({v:.2f})" for k, v in sorted(agg.items(), key=lambda kv: -kv[1])[:n]]


def _name(t, S, csm_ids, csm_class, X, grid):
    """Propose a chemistry name from the theme's own bands, with a confidence.

    Bands lead; the curated classes are a cross-check that can lower the confidence but never
    supply the name. Where the bands do not agree on one chemistry the theme stays
    `Unknown Theme`, which is a legitimate outcome rather than a failure to try.
    """
    # Named from the two families carrying most of the diagnostic prominence — the same
    # quantity admissibility is judged on. Reading ALL families here made every theme
    # "Unknown", because a fifth-ranked minor band always adds a third family.
    fams = set(t.dominant_families)
    bands = t.dominant_bands
    rules = [
        ({"aliphatic"}, "aliphatic chain", "CH2/CH3 scissoring and twisting of saturated "
                                           "hydrocarbon chains"),
        ({"unsaturation"}, "unsaturated chain", "cis C=C and =C–H in-plane bending"),
        ({"carboxyl"}, "carboxyl / ester carbonyl", "C=O and COO⁻ stretching"),
        ({"amide"}, "amide backbone", "amide I/II/III of the peptide linkage"),
        ({"skeletal"}, "polar skeletal backbone", "C–O, C–C and C–N skeletal stretching, "
                                                  "including glycosidic C–O–C"),
        ({"ring"}, "heterocyclic / conjugated ring", "ring breathing and conjugated C=C"),
        ({"sulfur"}, "sulfur / thiol", "S–S and C–S stretching"),
        ({"phosphate"}, "phosphate", "PO4 and PO2⁻ stretching"),
    ]
    hits = [(nm, defn, fams & f) for f, nm, defn in rules if fams & f]
    if not hits:
        return "Unknown Theme", "bands do not resolve to a single bond system", 0.2
    if len(fams) == 1:
        nm, defn, _ = hits[0]
        return nm, defn, float(np.clip(0.55 + 0.4 * t.assigned_fraction, 0, 0.95))
    if len(fams) == 2:
        nm = " + ".join(h[0] for h in hits[:2])
        defn = "; ".join(h[1] for h in hits[:2])
        return nm, defn, float(np.clip(0.35 + 0.3 * t.assigned_fraction, 0, 0.7))
    return ("Unknown Theme",
            f"bands span {len(fams)} unrelated mode families "
            f"({', '.join(sorted(fams))}) — no single chemistry", 0.25)


def _registry_json(reg, S, csm_ids, hier, boot, loo, nbc, modu, rec, value, onto,
                   roles_tab, grad):
    return {
        "schema": "theme_registry_v1", "phase": "03",
        "model": reg.model, "K": reg.K, "K_selection": reg.selection,
        "provenance": reg.provenance, "fingerprint": reg.fingerprint(),
        "summary": reg.summary(),
        "hierarchy": hier,
        "validation": {"bootstrap": boot, "leave_one_out": loo,
                       "neighbour_consistency": {k: v for k, v in nbc.items()
                                                 if k != "per_csm"},
                       "modularity": modu, "reconstruction": rec,
                       "value_over_csm_layer": value,
                       "ontology_agreement_post_hoc": onto},
        "themes": [{
            "theme_id": t.theme_id, "index": t.index, "name": t.name,
            "chemical_definition": t.chemical_definition,
            "name_confidence": round(t.name_confidence, 3),
            "confidence": round(t.confidence, 3), "status": t.status,
            "rejection_reason": t.rejection_reason,
            "chemically_admissible": t.chemically_admissible,
            "family_concentration": round(t.family_concentration, 3),
            "dominant_families": t.dominant_families,
            "dominant_bands_cm1": [round(b, 1) for b in t.dominant_bands],
            "band_assignments": t.band_assignments, "mode_families": t.mode_families,
            "member_csms": [{"csm_id": c, "membership": round(m, 4)}
                            for c, m in zip(t.member_csms, t.member_memberships)],
            "bridge_csms": t.bridge_csms, "n_supporting_csms": t.n_supporting_csms,
            "membership_entropy": round(t.membership_entropy, 4),
            "geometry_evidence": t.geometry_evidence,
            "spectral_evidence": t.spectral_evidence,
            "biochemical_evidence": t.biochemical_evidence,
            "counter_evidence": t.counter_evidence,
            "alternative_explanations": t.alternative_explanations,
            "bootstrap_stability": round(t.bootstrap_stability, 4),
            "loo_stability": round(t.loo_stability, 4),
            "gradient": t.gradient, "source_robust": t.source_robust,
            "limitations": t.limitations,
        } for t in reg.themes],
        "memberships": {c: {reg.themes[k].theme_id: round(float(S[i, k]), 4)
                            for k in range(S.shape[1])}
                        for i, c in enumerate(csm_ids)},
        "bridge_csms": reg.bridge_csms,
        "unassigned_csms": reg.unassigned_csms,
        "continuous_gradients": grad.to_dict("records"),
        "downstream_note": ("Phase 04 consumes S (M×K), the theme basis, and the theme "
                            "registry. BSV dimension = K."),
    }


def _write_yaml(reg, S, csm_ids, path):
    lines = ["schema: theme_registry_v1", f"K: {reg.K}", f"model: {reg.model}",
             f"fingerprint: {reg.fingerprint()}",
             "K_selection:",
             f"  rationale: >-\n    {reg.selection['rationale']}",
             "themes:"]
    for t in reg.themes:
        top = sorted(zip(t.member_csms, t.member_memberships), key=lambda x: -x[1])[:6]
        lines += [
            f"  - theme_id: {t.theme_id}",
            f"    name: {t.name}",
            f"    chemical_definition: {t.chemical_definition}",
            f"    chemically_admissible: {str(t.chemically_admissible).lower()}",
            f"    n_supporting_csms: {t.n_supporting_csms}",
            f"    membership_entropy: {t.membership_entropy:.4f}",
            f"    confidence: {t.confidence:.3f}",
            f"    dominant_bands_cm1: [{', '.join(f'{b:.0f}' for b in t.dominant_bands)}]",
            "    top_csms:",
        ] + [f"      - {{csm_id: {c}, membership: {m:.4f}}}" for c, m in top]
    Path(path).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
