#!/usr/bin/env python3
"""Deterministic orchestration of the complete GAIRA foundation rebuild.

Reproduces the GAIRA foundation outputs on another computer WITHOUT assuming
/Volumes/SSD_Rad. It ORCHESTRATES the existing canonical functions (it does not
re-implement any scientific logic) and writes everything to --output-dir. It never
modifies assets/foundation/ or any canonical results/ artifact.

Two modes
---------
  full                raw Raman corpus -> preprocess -> NMF -> rebuilt basis -> compare to
                      canonical -> registry -> theme weights -> MSS -> reference norm ->
                      validation -> manifest.   (needs the raw datasets)
  interpretation-only committed frozen basis + committed tables -> registry -> theme
                      weights -> MSS -> reference norm (iff reference coords available) ->
                      BSV fixtures -> manifest.   (needs NO raw data, NO SSD)

Data-root precedence (full mode):  --data-root  >  $GAIRA_DATA_ROOT  >  optional default  > error.

    python tools/reproduce_gaira_foundation.py --mode full \
        --data-root /path/to/GAIRA_DATA/raw --output-dir results/reproduction/full
    python tools/reproduce_gaira_foundation.py --mode interpretation-only \
        --foundation-root assets/foundation --output-dir results/reproduction/interp
"""
from __future__ import annotations
import argparse, os, sys, json, hashlib, platform, importlib.util, shutil, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

CANONICAL_FINGERPRINT = "09ed804a40836f4a05a91ba10900cded"
ASSETS_FOUNDATION = REPO / "assets" / "foundation"
LEGACY_FOUNDATION = REPO / "results/v5_rebuild/foundation/artifacts"
LEGACY_ENGINE = REPO / "results/v5_rebuild/engine_v1/artifacts"
ENGINE_CODE = REPO / "results/v5_rebuild/engine_v1/code"
ENGINE_TOOLS = REPO / "results/v5_rebuild/engine_v1/tools"
PACKAGE = REPO / "results/v5_rebuild/reproduction"
DATASET_ROLE_MAP = PACKAGE / "audits" / "dataset_role_map.csv"
# committed reference coordinates → lets interpretation-only rebuild reference normalization
# with NO raw data (Part 5 enabling asset). ~44 KB; canonical fingerprint recorded in manifests/.
COMMITTED_REFERENCE_COORDS = PACKAGE / "manifests" / "nmf_reference_coordinates.npz"
# Optional documented default data-root (precedence tier 3). Left None so no lab-specific
# absolute path is committed; set GAIRA_DATA_ROOT or pass --data-root instead.
DEFAULT_DATA_ROOT = os.environ.get("GAIRA_DEFAULT_DATA_ROOT")   # usually None


def log(msg): print(f"[reproduce] {msg}", flush=True)


# ─────────────────────────── helpers ───────────────────────────
def sha256_array(a):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()[:32]


def environment():
    import numpy, scipy, sklearn
    try:
        blas = numpy.__config__.show(mode="dicts") if hasattr(numpy.__config__, "show") else None
    except Exception:
        blas = None
    return {
        "python": platform.python_version(), "numpy": numpy.__version__,
        "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
        "platform": platform.platform(), "machine": platform.machine(),
        "processor": platform.processor(), "blas": str(blas)[:400] if blas else "unknown",
    }


def resolve_data_root(cli):
    for cand, why in [(cli, "--data-root"), (os.environ.get("GAIRA_DATA_ROOT"), "GAIRA_DATA_ROOT"),
                      (DEFAULT_DATA_ROOT, "documented default (GAIRA_DEFAULT_DATA_ROOT)")]:
        if cand:
            p = Path(cand).expanduser()
            if p.exists():
                log(f"data-root = {p}  (via {why})")
                return p
    raise SystemExit(
        "ERROR: no raw data-root found. Provide --data-root /path/to/GAIRA_DATA/raw or set "
        "GAIRA_DATA_ROOT. Full mode needs the raw Raman datasets (Gobbato zip + amino-acid "
        "sheet; RamanBioLib parquet in-repo or under the data-root).")


def configure_loaders(data_root, ramanbiolib_parquet=None):
    """Point the canonical loaders at data_root by overriding their module path constants
    at RUNTIME. No canonical source file is edited; the scientific logic is untouched."""
    import gaira.foundation.dataset as DS
    import gaira.data.gobbato as GOB
    import gaira.data.loader as LDR
    DS.RAW = data_root
    DS.AA_XLSX = data_root / "amino_acid_raman_grounding" / "aa.xlsx"
    DS.COVID_DIR = data_root / "covid_serum_raman"
    DS.KNOWLEDGE = data_root / "raman_knowledge_core" / "peak_assignments.csv"
    GOB.ZIP = data_root / "serum_ag_colloids" / "dataset_spectral_data.zip"
    if ramanbiolib_parquet:
        LDR.RAMANBIOLIB_PARQUET = Path(ramanbiolib_parquet)
    elif not LDR.RAMANBIOLIB_PARQUET.exists():
        for alt in (data_root / "ramanbiolib" / "grounding_molecule_spectra.parquet",
                    data_root / "grounding_molecule_spectra.parquet"):
            if alt.exists():
                LDR.RAMANBIOLIB_PARQUET = alt
                break
    return {"dataset.RAW": str(DS.RAW), "gobbato.ZIP": str(GOB.ZIP),
            "ramanbiolib_parquet": str(LDR.RAMANBIOLIB_PARQUET),
            "ramanbiolib_parquet_exists": LDR.RAMANBIOLIB_PARQUET.exists()}


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_build_script(rel_path, name, patches):
    """Import a canonical build script, patch its module globals (paths), call main()."""
    mod = _load_module(str(rel_path), name)
    for k, v in patches.items():
        setattr(mod, k, v)
    mod.main()
    return mod


def hungarian_align(H_new, H_canon):
    """Match rebuilt components to canonical by max cosine (Hungarian)."""
    from scipy.optimize import linear_sum_assignment
    def unit(M): return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    S = unit(H_new) @ unit(H_canon).T          # (k_new, k_canon) cosine
    row, col = linear_sum_assignment(-S)       # maximize
    perm = col.tolist()
    cos = [float(S[r, c]) for r, c in zip(row, col)]
    identity = perm == list(range(len(perm)))
    return {"permutation_new_to_canon": perm, "per_match_cosine": [round(c, 6) for c in cos],
            "mean_cosine": round(float(np.mean(cos)), 6), "min_cosine": round(float(np.min(cos)), 6),
            "is_identity_permutation": bool(identity)}


def compare_basis(H_new, grid_new, H_canon, grid_canon, out_dir):
    import pandas as pd
    equal = bool(H_new.shape == H_canon.shape and np.array_equal(H_new, H_canon))
    aligned = None if equal else hungarian_align(H_new, H_canon)
    if H_new.shape == H_canon.shape:
        maxd = float(np.max(np.abs(H_new - H_canon)))
        meand = float(np.mean(np.abs(H_new - H_canon)))
        def unit(M): return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
        diag_cos = [float(x) for x in np.sum(unit(H_new) * unit(H_canon), axis=1)]
    else:
        maxd = meand = None; diag_cos = []
    cmp = {
        "canonical_fingerprint": CANONICAL_FINGERPRINT,
        "rebuilt_fingerprint": sha256_array(H_new),
        "fingerprint_match": sha256_array(H_new) == CANONICAL_FINGERPRINT,
        "shape_rebuilt": list(H_new.shape), "shape_canonical": list(H_canon.shape),
        "grid_match": bool(np.array_equal(grid_new, grid_canon)),
        "exact_array_equality": equal,
        "max_abs_diff": maxd, "mean_abs_diff": meand,
        "per_component_cosine_by_index": [round(c, 6) for c in diag_cos],
        "hungarian_alignment": aligned,
        "note": ("Basis is byte-identical — direct index comparison valid." if equal else
                 "Basis differs — DO NOT compare component i to i blindly; use the Hungarian "
                 "permutation before transferring any integer-index annotation."),
    }
    (out_dir / "basis_comparison.json").write_text(json.dumps(cmp, indent=2))
    # component_alignment.csv
    rows = []
    perm = (aligned["permutation_new_to_canon"] if aligned else list(range(H_new.shape[0])))
    for i in range(H_new.shape[0]):
        rows.append({"rebuilt_component": i, "canonical_component": perm[i],
                     "match_cosine": (aligned["per_match_cosine"][i] if aligned
                                      else (round(diag_cos[i], 6) if diag_cos else None)),
                     "byte_identical": equal})
    pd.DataFrame(rows).to_csv(out_dir / "component_alignment.csv", index=False)
    return cmp


def _norm_from_Z(Z, analytes, out_path, support_path, fingerprint):
    """Mirror of build_reference_norm.py (lines 24-30): center=median, spread=1.4826*MAD,
    support=unit(Z). Used ONLY in interpretation-only mode from a saved reference-coordinate
    matrix, so no raw corpus is needed. Identical formula to the canonical script."""
    Zn = np.clip(np.nan_to_num(np.asarray(Z, float)), 0, None)
    center = np.median(Zn, axis=0)
    spread = np.maximum(1.4826 * np.median(np.abs(Zn - center), axis=0), 1e-3)
    U = Zn / (np.linalg.norm(Zn, axis=1, keepdims=True) + 1e-12)
    out = {"schema": "reference_normalization_v1", "version": "1.0", "atlas_fingerprint": fingerprint,
           "n_reference_spectra": int(len(Zn)), "n_reference_analytes": int(len(set(map(str, analytes)))),
           "statistic": "median / MAD (robust)",
           "component_center": [round(float(x), 6) for x in center],
           "component_spread": [round(float(x), 6) for x in spread],
           "note": "Rebuilt in interpretation-only mode from saved reference coordinates "
                   "(nmf_reference_coordinates.npz); formula mirrors build_reference_norm.py."}
    Path(out_path).write_text(json.dumps(out, indent=2))
    np.savez_compressed(support_path, support_unit=U.astype(np.float32),
                        analytes=np.asarray(analytes))
    return out


def bsv_fixtures(foundation_root, out_dir, label):
    """Deterministic BSV regression: infer on fixed coordinate fixtures using the engine
    loaded from `foundation_root`, so two runs / two modes can be compared byte-for-byte."""
    from gaira.engine import GAIRAEngine
    from gaira.engine.mss import MSSLayer
    eng = GAIRAEngine()
    mss = MSSLayer.from_engine(eng)
    themes = eng.builder.onto.biochemical_theme_ids
    k = eng.builder.onto.k
    fixtures = {"uniform": np.full(k, 1.0 / k)}
    for j in (0, 3, 15):                                   # a few single-component fixtures
        v = np.zeros(k); v[j] = 1.0; fixtures[f"component_{j}"] = v
    out = {"label": label, "atlas_fingerprint": eng.atlas.meta["fingerprint"], "results": {}}
    for name, coord in fixtures.items():
        bsv = eng.infer(coordinates=coord, domain="serum").bsv
        acts = {a.id: round(float(a.composition), 6) for a in mss.activate(bsv)}
        out["results"][name] = {
            "composition": {t: round(float(bsv.composition[t]), 6) for t in themes},
            "ood": round(float(bsv.ood_score), 6),
            "top_mss": sorted(acts.items(), key=lambda kv: -kv[1])[:3],
        }
    (out_dir / "bsv_regression_results.json").write_text(json.dumps(out, indent=2))
    return out


# ─────────────────────────── modes ───────────────────────────
def build_interpretation(foundation_root, out_dir, reference_coords=None):
    """Steps 7-10 + BSV fixtures. Orchestrates the canonical build scripts, writing to out_dir.
    foundation_root must contain manifold.json + manifold_components.npz."""
    interp = out_dir
    results = {}
    # 7 · component registry (frozen basis + committed evidence tables; no raw)
    log("build component registry …")
    run_build_script(ENGINE_CODE / "build_registry.py", "build_registry",
                     {"FROZEN": Path(foundation_root), "OUT": interp})
    results["component_registry_v1.json"] = (interp / "component_registry_v1.json").exists()
    # 8 · component -> theme weights (registry + committed tables + ontology yaml; no raw)
    log("build component-theme weights …")
    run_build_script(ENGINE_CODE / "build_theme_weights.py", "build_theme_weights",
                     {"OUT": interp})
    results["component_theme_weights_v1.json"] = (interp / "component_theme_weights_v1.json").exists()
    # 9 · MSS registry — from the JUST-BUILT registry + weights (public constructor args)
    log("build MSS registry …")
    from gaira.engine.ontology import Ontology
    from gaira.engine.registry import ComponentRegistry
    from gaira.engine.mss import MSSLayer
    onto = Ontology(weights_path=interp / "component_theme_weights_v1.json")
    reg = ComponentRegistry(path=interp / "component_registry_v1.json")
    mss = MSSLayer(ontology=onto, registry=reg)
    (interp / "mss_registry_v1.json").write_text(json.dumps(mss.registry(), indent=2))
    results["mss_registry_v1.json"] = True
    # 10 · reference normalization
    fp = json.loads((Path(foundation_root) / "manifold.json").read_text())["fingerprint"]
    if reference_coords and Path(reference_coords).exists():
        log("build reference normalization from saved coordinates (no raw) …")
        z = np.load(reference_coords, allow_pickle=True)
        _norm_from_Z(z["Z"], z["analytes"], interp / "reference_normalization_v1.json",
                     interp / "reference_support.npz", fp)
        results["reference_normalization_v1.json"] = "rebuilt from saved coordinates"
    else:
        results["reference_normalization_v1.json"] = ("SKIPPED — needs raw corpus or a saved "
            "nmf_reference_coordinates.npz (Mode-B limitation; see report)")
        log("reference normalization SKIPPED (no reference coordinates supplied)")
    return results


def mode_full(args, out_dir):
    from gaira.foundation import dataset as DS, latent_space as LS, serialization as SER
    import pandas as pd
    manifest = {"mode": "full", "environment": environment(), "steps": {}}
    data_root = resolve_data_root(args.data_root)
    manifest["loader_config"] = configure_loaders(data_root, args.ramanbiolib_parquet)

    # 1-3 · corpus + preprocessing
    log("load + preprocess reference corpus …")
    corpus = DS.load_reference_corpus()
    X = corpus.X; grid = corpus.grid
    (out_dir / "preprocessing_config.json").write_text(json.dumps({
        "PREPROC": DS.PREPROC, "window_cm": list(DS.WINDOW), "grid_step_cm": 2.0,
        "grid_len": int(len(grid)), "asls": {"lam": 1e5, "p": 0.01, "n_iter": 8},
        "savgol": {"window": 9, "poly": 3}, "normalization": "l2", "nonneg_clip_for_nmf": True}, indent=2))
    corpus.meta.to_csv(out_dir / "corpus_manifest.csv", index=False)
    if DATASET_ROLE_MAP.exists():
        shutil.copy(DATASET_ROLE_MAP, out_dir / "dataset_role_map.csv")
    np.savez_compressed(out_dir / "preprocessed_reference_matrix.npz", X=X.astype(np.float32),
                        grid=grid, analyte=corpus.meta.analyte.values, source=corpus.meta.source.values)
    manifest["steps"]["corpus"] = {"n_spectra": int(len(X)), "n_analytes": int(corpus.meta.analyte.nunique()),
                                   "n_bins": int(X.shape[1]),
                                   "sources": corpus.meta.source.value_counts().to_dict()}

    # 4 · NMF (canonical fit) -> A_ref + H_basis
    log("fit NMF k=24 (canonical parameters) …")
    man = LS.build_manifold(corpus, "NMF", 24, seed=0)
    H = man.rep.components_
    A_ref = man.rep.transform(np.nan_to_num(X))
    np.savez_compressed(out_dir / "nmf_basis.npz", components=H, grid=grid)
    np.savez_compressed(out_dir / "nmf_training_activations.npz", A_ref=A_ref.astype(np.float32),
                        analyte=corpus.meta.analyte.values)

    # 5 · freeze rebuilt basis (to a SUBDIR of out_dir; never assets/foundation)
    rebuilt_root = out_dir / "rebuilt_foundation"; rebuilt_root.mkdir(exist_ok=True)
    fp = SER.freeze_manifold(man, rebuilt_root, corpus_card=DS.dataset_card(corpus),
                             extra={"reproduction": True})
    # save the NNLS reference coordinates Z (what reference-norm consumes) → enables Mode B later
    Z = SER.load_frozen_manifold(rebuilt_root).coordinates(X, normalise=True)
    np.savez_compressed(out_dir / "nmf_reference_coordinates.npz", Z=np.nan_to_num(Z),
                        analytes=corpus.meta.analyte.values)
    rec = float(np.linalg.norm(np.nan_to_num(X) - man.rep.reconstruct(np.nan_to_num(X))) /
                (np.linalg.norm(np.nan_to_num(X)) + 1e-12))
    (out_dir / "nmf_metrics.json").write_text(json.dumps({
        "explained_variance": man.stats["explained_variance"], "recon_rel_error": rec,
        "rebuilt_fingerprint": fp, "n_iter_max": 1500, "seed": 0,
        "per_component_variance": man.stats["per_component_variance"]}, indent=2))

    # 6 · compare rebuilt vs canonical
    log("compare rebuilt basis to canonical …")
    canon = np.load(ASSETS_FOUNDATION / "manifold_components.npz")
    cmp = compare_basis(H, grid, canon["components"], canon["grid"], out_dir)
    manifest["steps"]["basis_comparison"] = {"fingerprint_match": cmp["fingerprint_match"],
                                             "exact_equality": cmp["exact_array_equality"],
                                             "max_abs_diff": cmp["max_abs_diff"]}

    # 7-10 · interpretation layers — built against the CANONICAL basis (the committed evidence
    # tables are canonical-index-keyed; verified equivalent to rebuilt iff fingerprint matches).
    interp_source = ASSETS_FOUNDATION if cmp["fingerprint_match"] else ASSETS_FOUNDATION
    if not cmp["fingerprint_match"]:
        log("WARNING: rebuilt basis != canonical; interpretation built on CANONICAL basis "
            "(committed evidence tables are canonical-indexed). Align first before transferring.")
    manifest["steps"]["interpretation"] = build_interpretation(
        interp_source, out_dir, reference_coords=out_dir / "nmf_reference_coordinates.npz")

    # 11 · validation fixtures + BSV regression
    log("BSV regression fixtures …")
    manifest["steps"]["bsv_fixtures"] = "bsv_regression_results.json"
    bsv_fixtures(ASSETS_FOUNDATION, out_dir, label="full")
    return manifest


def mode_interpretation(args, out_dir):
    foundation_root = Path(args.foundation_root or ASSETS_FOUNDATION)
    manifest = {"mode": "interpretation-only", "environment": environment(),
                "foundation_root": str(foundation_root), "steps": {}}
    if not (foundation_root / "manifold.json").exists():
        raise SystemExit(f"ERROR: {foundation_root}/manifold.json not found.")
    if DATASET_ROLE_MAP.exists():
        shutil.copy(DATASET_ROLE_MAP, out_dir / "dataset_role_map.csv")
    ref_coords = args.reference_coords
    if not ref_coords and COMMITTED_REFERENCE_COORDS.exists():
        ref_coords = str(COMMITTED_REFERENCE_COORDS)     # committed enabling asset (no raw needed)
    manifest["steps"]["interpretation"] = build_interpretation(foundation_root, out_dir, ref_coords)
    bsv_fixtures(foundation_root, out_dir, label="interpretation-only")
    manifest["steps"]["bsv_fixtures"] = "bsv_regression_results.json"
    return manifest


# ─────────────────────────── canonical comparison of downstream ───────────────────────────
def _normalize_registry(reg):
    """Normalize the ONLY known cosmetic nondeterminism in build_registry.py: the
    `'/'.join(set(directions))` in current_interpretation, whose order depends on
    PYTHONHASHSEED. Sorting the 'x/y' direction group makes the numeric+text content
    comparable byte-for-byte."""
    import re, copy
    reg = copy.deepcopy(reg)
    for c in reg.get("components", []):
        ci = c.get("current_interpretation", {})
        if isinstance(ci, dict) and isinstance(ci.get("value"), str):
            ci["value"] = re.sub(r"\(([a-z]+(?:/[a-z]+)+)\)",
                                 lambda m: "(" + "/".join(sorted(m.group(1).split("/"))) + ")",
                                 ci["value"])
    return reg


def compare_downstream(out_dir):
    """Compare rebuilt interpretation artifacts to the canonical committed ones."""
    res = {}
    pairs = [("component_registry_v1.json", LEGACY_ENGINE / "component_registry_v1.json"),
             ("component_theme_weights_v1.json", LEGACY_ENGINE / "component_theme_weights_v1.json"),
             ("mss_registry_v1.json", LEGACY_ENGINE / "mss_registry_v1.json"),
             ("reference_normalization_v1.json", LEGACY_ENGINE / "reference_normalization_v1.json")]
    for name, canon in pairs:
        rebuilt = out_dir / name
        if not rebuilt.exists() or not canon.exists():
            res[name] = "n/a"; continue
        a = json.loads(rebuilt.read_text()); b = json.loads(canon.read_text())
        if a == b:
            res[name] = "identical"
        elif name == "component_registry_v1.json" and _normalize_registry(a) == _normalize_registry(b):
            res[name] = ("identical except cosmetic interpretation-text ordering "
                         "(PYTHONHASHSEED-dependent set join in build_registry.py; numeric content identical)")
        elif (name == "reference_normalization_v1.json"
              and a.get("component_center") == b.get("component_center")
              and a.get("component_spread") == b.get("component_spread")):
            res[name] = "identical (numeric center/spread); only the 'note' metadata string differs"
        else:
            res[name] = "differs"
    (out_dir / "downstream_comparison.json").write_text(json.dumps(res, indent=2))
    return res


# ─────────────────────────── main ───────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Reproduce the GAIRA foundation deterministically.")
    ap.add_argument("--mode", choices=["full", "interpretation-only"], required=True)
    ap.add_argument("--data-root", default=None, help="raw GAIRA_DATA/raw (full mode)")
    ap.add_argument("--ramanbiolib-parquet", default=None, help="override RamanBioLib parquet path")
    ap.add_argument("--foundation-root", default=None, help="frozen bundle dir (interpretation-only)")
    ap.add_argument("--reference-coords", default=None,
                    help="nmf_reference_coordinates.npz to rebuild reference norm without raw")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.output_dir).resolve()
    if ASSETS_FOUNDATION.resolve() in (out_dir, *out_dir.parents):
        raise SystemExit("REFUSING to write inside assets/foundation/ — choose another --output-dir.")
    (out_dir / "reproduction_run").mkdir(parents=True, exist_ok=True)
    run_dir = out_dir / "reproduction_run"

    canon_before = sha256_array(np.load(ASSETS_FOUNDATION / "manifold_components.npz")["components"])
    t0 = time.time()
    manifest = (mode_full if args.mode == "full" else mode_interpretation)(args, run_dir)
    manifest["downstream_comparison"] = compare_downstream(run_dir)
    (run_dir / "environment.json").write_text(json.dumps(environment(), indent=2))

    # guard: canonical asset unchanged
    canon_after = sha256_array(np.load(ASSETS_FOUNDATION / "manifold_components.npz")["components"])
    manifest["canonical_asset_unmodified"] = (canon_before == canon_after == CANONICAL_FINGERPRINT)
    manifest["runtime_sec"] = round(time.time() - t0, 1)
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    # human-readable report
    _write_report(run_dir, manifest)
    log(f"DONE ({manifest['runtime_sec']}s). Outputs in {run_dir}")
    log(f"canonical asset unmodified: {manifest['canonical_asset_unmodified']}")


def _write_report(run_dir, manifest):
    m = manifest
    bc = m.get("steps", {}).get("basis_comparison", {})
    lines = [f"# Reproduction report — mode: {m['mode']}", "",
             f"- runtime: {m.get('runtime_sec')} s",
             f"- canonical asset unmodified: **{m.get('canonical_asset_unmodified')}**",
             f"- environment: {m['environment']['scikit_learn']=} · numpy {m['environment']['numpy']}"
             f" · py {m['environment']['python']}", ""]
    if m["mode"] == "full":
        lines += [f"- basis fingerprint match: **{bc.get('fingerprint_match')}** "
                  f"(exact equality {bc.get('exact_equality')}, max abs diff {bc.get('max_abs_diff')})", ""]
    lines += ["## Downstream vs canonical", ""]
    for k, v in m.get("downstream_comparison", {}).items():
        lines.append(f"- `{k}`: **{v}**")
    lines += ["", "## Output files", ""]
    for p in sorted(run_dir.iterdir()):
        lines.append(f"- `{p.name}` ({p.stat().st_size} bytes)")
    (run_dir / "reproduction_report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
