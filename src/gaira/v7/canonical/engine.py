"""GAIRA V7 — the canonical inference engine.

    engine = GAIRAEngine.load()
    result = engine.infer(spectrum)

One frozen path, fixed by Phase 08's decision and not negotiable at runtime:

    spectrum → canonical preprocessing → LSM projection → CSM projection
             → molecular retrieval (CSM similarity) → 16-axis Chemistry Evidence
             → calibrated radar → interpretation report

**Not on this path**, and not importable from this module: BSV2, PCA, UMAP, clustering, latent
geometry, continuous coordinates, the legacy theme layer, the legacy BSV. Phase 06.5 showed
geometry does not improve inference; Phase 07 adopted BSV2 as an offline representation only;
Phase 08 showed chemistry reranking adds exactly zero to molecular ranking. Each of those is a
measured decision, and the engine encodes them rather than re-litigating them.

The engine holds **no mutable state**. Every artefact is loaded once, fingerprint-checked, and
never written to; `infer` is a pure function of its argument and the frozen atlas.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

EPS = 1e-12
EXPECTED_FINGERPRINTS = {
    "atlas": "09ed804a40836f4a05a91ba10900cded",
    "lsm": "208482d6f7178b5b8f16cace91be55b0",
    "csm": "0b4aa550ccefed3edabdbde5bae11c8d",
    "engine": "20d8bd99ce71f45a125c6a2b1d719e51",
}
# The canonical preprocessing grid, frozen in Phase 00 and unchanged since V5.
GRID_LO, GRID_HI, GRID_STEP, N_BINS = 450.0, 1800.0, 2.0, 676


class FrozenArtifactError(RuntimeError):
    """Raised when a fingerprint does not match. The engine must not run on a changed atlas."""


@dataclass(frozen=True)
class PreprocessingSummary:
    n_input_points: int
    input_range: tuple[float, float]
    resampled_to: str
    baseline_method: str
    smoothing: str
    normalisation: str
    n_peaks: int
    signal_quality: float
    snr_estimate: float
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class InferenceReport:
    """Everything the engine knows about one spectrum. Immutable by construction."""
    preprocessing: PreprocessingSummary
    lsm: dict
    csm: dict
    retrieval: dict
    chemistry: dict
    confidence: dict
    provenance: dict
    atlas_fingerprint: str

    def to_dict(self) -> dict:
        def _s(o):
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, tuple):
                return list(o)
            return o
        return json.loads(json.dumps({
            "preprocessing": self.preprocessing.__dict__, "lsm": self.lsm, "csm": self.csm,
            "retrieval": self.retrieval, "chemistry": self.chemistry,
            "confidence": self.confidence, "provenance": self.provenance,
            "atlas_fingerprint": self.atlas_fingerprint}, default=_s))


class GAIRAEngine:
    """The frozen atlas plus the fixed inference path. Construct with `GAIRAEngine.load()`."""

    def __init__(self, *, grid, H_lsm, lsm_ids, CSM, csm_ids, csm_records, ref_A, ref_E,
                 ref_labels, ref_classes, chem_model, calibrator, axis_names, class_of,
                 fingerprints, atlas_fingerprint):
        self._grid = np.asarray(grid, float)
        self._H_lsm, self._lsm_ids = np.asarray(H_lsm, float), tuple(lsm_ids)
        self._CSM, self._csm_ids = np.asarray(CSM, float), tuple(csm_ids)
        self._csm_records = tuple(csm_records)
        self._ref_A, self._ref_E = np.asarray(ref_A, float), np.asarray(ref_E, float)
        self._ref_labels, self._ref_classes = tuple(ref_labels), tuple(ref_classes)
        self._chem_model, self._calibrator = chem_model, calibrator
        self._axis_names, self._class_of = tuple(axis_names), dict(class_of)
        self.fingerprints = dict(fingerprints)
        self.atlas_fingerprint = atlas_fingerprint

    # ── construction ─────────────────────────────────────────────────────────
    @classmethod
    def load(cls, frozen_root: Path | None = None, verify: bool = True) -> "GAIRAEngine":
        """Load every frozen artefact and verify its fingerprint before returning."""
        import pandas as pd
        from gaira.v7.io import frozen_root as default_root
        from gaira.v7.chemistry import evidence as CHEM, registry as REG
        from gaira.v7.chemistry import calibration as CAL

        F = Path(frozen_root) if frozen_root else default_root()
        csm_reg = json.loads((F / "phase02/artifacts/csm_registry_v1.json").read_text())
        st01 = json.loads((F / "phase01/PHASE_STATE.json").read_text())
        st05 = json.loads((F / "phase05/PHASE_STATE.json").read_text())
        got = {"atlas": st01["atlas_fingerprint"], "lsm": st01["registry_fingerprint"],
               "csm": csm_reg["fingerprint"], "engine": st05["engine_fingerprint"]}
        if verify:
            bad = {k: (got[k], v) for k, v in EXPECTED_FINGERPRINTS.items() if got.get(k) != v}
            if bad:
                raise FrozenArtifactError(
                    "frozen artefact fingerprints do not match; the engine will not run on a "
                    f"changed atlas: {bad}")

        z1 = np.load(F / "phase01/artifacts/lsm_dictionary_v1.npz", allow_pickle=True)
        z2 = np.load(F / "phase02/artifacts/csm_dictionary_v1.npz", allow_pickle=True)
        br = np.load(F / "phase01/artifacts/balanced_references_v1.npz", allow_pickle=True)
        X = np.asarray(br["X"], float)
        y = np.array([str(s) for s in br["canonical_id"]])
        part = pd.read_csv(F / "phase00/tables/chemical_partition_v1.csv")
        class_of = dict(zip(part.canonical_id, part.fine_class))
        canon = pd.read_csv(F / "phase00/tables/canonical_analytes_v1.csv")
        broad_of = dict(zip(canon.canonical_id, canon.broad_class))

        from gaira.v7.inference import projection as PRJ
        CSM = np.asarray(z2["CSM"], float)
        A = PRJ.project(X, CSM)
        cls_arr = np.array([class_of[m] for m in y])
        cfg = dict(json.loads((F / "phase06/artifacts/chemistry_evidence_model_v1.json"
                               ).read_text())["config"])
        fam = cfg.pop("family")
        chem_model = (CHEM.fit_D(A, y, cls_arr, broad_of=broad_of, **cfg)
                      if fam == "D_hierarchical" else
                      CHEM.fit(fam, A, y, cls_arr, **cfg))
        E = CHEM.predict(chem_model, A)
        cal_name = json.loads((F / "phase06/artifacts/chemistry_evidence_calibrator_v1.json"
                               ).read_text())["method"]
        calibrator = CAL.Calibrator(cal_name).fit(E, cls_arr)

        mols = sorted(set(y.tolist()))
        ref_A = np.vstack([A[y == m].mean(axis=0) for m in mols])
        ref_E = np.vstack([E[y == m].mean(axis=0) for m in mols])
        atlas_fp = hashlib.md5(json.dumps(
            [got, list(mols), cal_name, cfg, fam], sort_keys=True).encode()).hexdigest()
        return cls(grid=np.asarray(br["grid"], float), H_lsm=z1["H"],
                    lsm_ids=[str(s) for s in z1["motif_ids"]], CSM=CSM,
                    csm_ids=[str(s) for s in z2["csm_ids"]], csm_records=csm_reg["csms"],
                    ref_A=ref_A, ref_E=ref_E, ref_labels=mols,
                    ref_classes=[class_of[m] for m in mols], chem_model=chem_model,
                    calibrator=calibrator, axis_names=list(REG.CLASS_ORDER),
                    class_of=class_of, fingerprints=got, atlas_fingerprint=atlas_fp)

    # ── the pipeline ─────────────────────────────────────────────────────────
    def preprocess(self, wavenumbers, intensities) -> tuple[np.ndarray, PreprocessingSummary]:
        """Canonical preprocessing: crop → resample → asLS baseline → SG smooth → L2.

        Identical in kind to the V5 specification and unchanged since Phase 00. A spectrum that
        does not cover the canonical window is resampled with zero fill and a warning; it is
        never silently extrapolated.
        """
        from scipy.signal import find_peaks, savgol_filter
        w = np.asarray(wavenumbers, float)
        v = np.asarray(intensities, float)
        warn = []
        if w.min() > GRID_LO + 20 or w.max() < GRID_HI - 20:
            warn.append(f"input range {w.min():.0f}-{w.max():.0f} does not cover the canonical "
                        f"window {GRID_LO:.0f}-{GRID_HI:.0f}; missing regions are zero-filled")
        if len(w) < 100:
            warn.append(f"only {len(w)} points supplied; resampling to {N_BINS} bins")
        r = np.interp(self._grid, w, v, left=0.0, right=0.0)
        base = _asls(r)
        r = np.clip(r - base, 0.0, None)
        r = savgol_filter(r, 9, 3)
        r = np.clip(r, 0.0, None)
        n = float(np.linalg.norm(r))
        if n < EPS:
            warn.append("spectrum is empty after preprocessing")
        r = r / (n + EPS)
        pk, _ = find_peaks(r / (r.max() + EPS), prominence=0.02)
        noise = float(np.median(np.abs(np.diff(r))))
        snr = float(r.max() / (noise + EPS))
        if snr < 20:
            warn.append(f"low SNR estimate ({snr:.1f}); interpretation is unreliable")
        return r, PreprocessingSummary(
            n_input_points=int(len(w)), input_range=(float(w.min()), float(w.max())),
            resampled_to=f"{GRID_LO:.0f}-{GRID_HI:.0f} cm-1, {GRID_STEP:.1f} step, "
                         f"{N_BINS} bins",
            baseline_method="asymmetric least squares", smoothing="Savitzky-Golay (9, 3)",
            normalisation="L2", n_peaks=int(len(pk)),
            signal_quality=float(np.clip(snr / 100.0, 0, 1)), snr_estimate=snr,
            warnings=tuple(warn))

    def project_lsm(self, x) -> dict:
        from scipy.optimize import nnls
        a = nnls(self._H_lsm.T, np.clip(np.asarray(x, float), 0, None))[0]
        rec = a @ self._H_lsm
        resid = float(np.linalg.norm(x - rec))
        ev = float(1.0 - (resid ** 2) / (float((np.asarray(x) ** 2).sum()) + EPS))
        o = np.argsort(-a)[:8]
        return {"activation": a, "reconstruction": rec,
                "reconstruction_error": resid, "explained_variance": ev,
                "n_active": int((a > 1e-9).sum()),
                "top": [{"lsm_id": self._lsm_ids[int(j)], "weight": float(a[j]),
                         "share": float(a[j] / (a.sum() + EPS))}
                        for j in o if a[j] > 1e-9]}

    def project_csm(self, x) -> dict:
        from gaira.v7.inference import projection as PRJ
        a = PRJ.project(np.atleast_2d(x), self._CSM)[0]
        d = PRJ.diagnostics(np.atleast_2d(x), a[None, :], self._CSM)
        o = np.argsort(-a)[:8]
        return {"activation": a, "explained_variance": float(d["explained_variance"][0]),
                "residual_fraction": float(d["residual_fraction"][0]),
                "sparsity": float(d["component_sparsity"][0]),
                "entropy": float(d["activation_entropy"][0]),
                "n_active": int(d["n_active_csms"][0]),
                "top": [{"csm_id": self._csm_ids[int(j)], "weight": float(a[j]),
                         "share": float(a[j] / (a.sum() + EPS)),
                         "dominant_bands": [float(b) for b in
                                            self._csm_records[int(j)].get("dominant_bands", [])],
                         "band_assignment": self._csm_records[int(j)].get("band_assignment", ""),
                         "lsms": [l["lsm_id"] if isinstance(l, dict) else str(l)
                                  for l in self._csm_records[int(j)].get(
                                      "contributing_lsms", [])]}
                        for j in o if a[j] > 1e-9]}

    def retrieve(self, a_csm, top_k: int = 10) -> dict:
        """Molecular retrieval by CSM cosine — Phase 08 outcome A, unchanged.

        Every returned score reconciles: the similarity is an inner product of unit vectors and
        the listed CSM contributions sum to it exactly.
        """
        q = np.asarray(a_csm, float)
        qn = q / (np.linalg.norm(q) + EPS)
        Rn = self._ref_A / (np.linalg.norm(self._ref_A, axis=1, keepdims=True) + EPS)
        s = np.clip(Rn @ qn, 0.0, 1.0)
        order = np.argsort(-s)
        srt = np.sort(s)
        out = []
        for rank, j in enumerate(order[:top_k], start=1):
            contrib = qn * Rn[j]
            cs = np.argsort(-contrib)[:5]
            out.append({
                "rank": rank, "molecule": self._ref_labels[int(j)],
                "chemistry_class": self._ref_classes[int(j)],
                "similarity": float(s[j]),
                "supporting_csms": [
                    {"csm_id": self._csm_ids[int(k)], "contribution": float(contrib[k]),
                     "share_of_similarity": float(contrib[k] / (s[j] + EPS)),
                     "lsms": [l["lsm_id"] if isinstance(l, dict) else str(l)
                              for l in self._csm_records[int(k)].get("contributing_lsms", [])],
                     "diagnostic_bands": [float(b) for b in
                                          self._csm_records[int(k)].get("dominant_bands", [])]}
                    for k in cs if contrib[k] > 1e-6],
                "contribution_sum": float(contrib.sum()),
                "reconciles": bool(abs(contrib.sum() - s[j]) < 1e-9)})
        return {"top": out, "margin": float(srt[-1] - srt[-2]),
                "n_candidates": int(len(self._ref_labels)),
                "all_similarities": s}

    def chemistry(self, a_csm) -> dict:
        from gaira.v7.chemistry import evidence as CHEM
        E = CHEM.predict(self._chem_model, np.atleast_2d(a_csm))
        P = self._calibrator.transform(E)
        e, p = E[0], P[0]
        order = np.argsort(-e)
        return {"axis_names": list(self._axis_names),
                "evidence": e, "calibrated_probability": p,
                "evidence_l1": e / (e.sum() + EPS),
                "top": [{"axis": self._axis_names[int(j)], "evidence": float(e[j]),
                         "share": float(e[j] / (e.sum() + EPS)),
                         "calibrated_probability": float(p[j]), "rank": r + 1}
                        for r, j in enumerate(order[:5])],
                "predicted_class": self._axis_names[int(order[0])],
                "margin": float(np.sort(e)[-1] - np.sort(e)[-2]),
                "entropy": float(-(np.where(e > 0, (e / (e.sum() + EPS)) *
                                            np.log(e / (e.sum() + EPS) + EPS), 0)).sum()
                                 / np.log(len(e)))}

    def infer(self, spectrum, wavenumbers=None, top_k: int = 10,
              already_preprocessed: bool = False) -> InferenceReport:
        """The whole path. Deterministic, stateless, and identical alone or in a batch.

        `already_preprocessed=True` skips the preprocessing stage for a spectrum that is already
        on the canonical grid and normalised — the frozen corpus, for instance. Running asLS a
        second time on a baseline-corrected spectrum would subtract a second small baseline, so
        the flag exists to keep validation honest rather than to save time.
        """
        v = np.asarray(spectrum, float)
        w = np.asarray(wavenumbers, float) if wavenumbers is not None else self._grid
        if v.shape[0] != w.shape[0]:
            raise ValueError(f"spectrum has {v.shape[0]} points but {w.shape[0]} wavenumbers")
        if already_preprocessed:
            if v.shape[0] != N_BINS:
                raise ValueError(f"already_preprocessed needs {N_BINS} bins, got {v.shape[0]}")
            from scipy.signal import find_peaks
            x = np.clip(v, 0.0, None)
            x = x / (np.linalg.norm(x) + EPS)
            pk, _ = find_peaks(x / (x.max() + EPS), prominence=0.02)
            noise = float(np.median(np.abs(np.diff(x))))
            pre = PreprocessingSummary(
                n_input_points=int(len(w)), input_range=(float(w.min()), float(w.max())),
                resampled_to="supplied on the canonical grid", baseline_method="(skipped)",
                smoothing="(skipped)", normalisation="L2 (re-applied)", n_peaks=int(len(pk)),
                signal_quality=float(np.clip(x.max() / (noise + EPS) / 100.0, 0, 1)),
                snr_estimate=float(x.max() / (noise + EPS)),
                warnings=("preprocessing skipped: caller asserts the spectrum is canonical",))
        else:
            x, pre = self.preprocess(w, v)
        lsm = self.project_lsm(x)
        csm = self.project_csm(x)
        ret = self.retrieve(csm["activation"], top_k)
        chem = self.chemistry(csm["activation"])
        conf = self._confidence(pre, csm, ret, chem)
        prov = self._provenance(lsm, csm, ret, chem)
        return InferenceReport(preprocessing=pre, lsm=_strip(lsm), csm=_strip(csm),
                               retrieval=_strip(ret), chemistry=_strip(chem),
                               confidence=conf, provenance=prov,
                               atlas_fingerprint=self.atlas_fingerprint)

    # ── derived reporting ────────────────────────────────────────────────────
    def _confidence(self, pre, csm, ret, chem) -> dict:
        """Confidence, coverage and the two warnings that matter.

        Deliberately pessimistic: reconstruction quality multiplies everything, because a
        spectrum the atlas cannot explain should not produce a confident answer regardless of
        how well its residual happens to match a reference.
        """
        ev = csm["explained_variance"]
        top1 = ret["top"][0]["similarity"] if ret["top"] else 0.0
        top3 = float(np.mean([t["similarity"] for t in ret["top"][:3]])) if ret["top"] else 0.0
        coverage = float(np.clip(ev, 0, 1))
        overall = float(np.clip(ev, 0, 1) * top1)
        unknown = bool(ev < 0.50 or ret["margin"] < 0.01)
        outlier = bool(csm["residual_fraction"] > 0.50 or csm["n_active"] <= 1)
        return {"overall": overall, "evidence_coverage": coverage,
                "top1_confidence": float(top1), "top3_confidence": float(top3),
                "retrieval_margin": float(ret["margin"]),
                "chemistry_confidence": float(max(chem["calibrated_probability"])),
                "reconstruction_explained_variance": float(ev),
                "unknown_warning": unknown, "outlier_warning": outlier,
                "notes": tuple(
                    ([f"reconstruction explains only {ev:.0%} of the spectrum; the atlas may "
                      "not cover this chemistry"] if ev < 0.50 else []) +
                    ([f"top-1 and top-2 differ by {ret['margin']:.4f}; the identification is "
                      "not decisive"] if ret["margin"] < 0.01 else []) +
                    list(pre.warnings))}

    def _provenance(self, lsm, csm, ret, chem) -> dict:
        """The full tree: spectrum → LSMs → CSMs → chemistry → molecules."""
        return {
            "root": "spectrum",
            "lsm_layer": [{"id": t["lsm_id"], "weight": t["weight"]} for t in lsm["top"]],
            "csm_layer": [{"id": t["csm_id"], "weight": t["weight"], "lsms": t["lsms"],
                           "bands": t["dominant_bands"],
                           "assignment": t["band_assignment"]} for t in csm["top"]],
            "chemistry_layer": [{"axis": t["axis"], "evidence": t["evidence"]}
                                for t in chem["top"]],
            "molecule_layer": [{"molecule": t["molecule"], "similarity": t["similarity"],
                                "class": t["chemistry_class"],
                                "via_csms": [c["csm_id"] for c in t["supporting_csms"]]}
                               for t in ret["top"][:5]],
            "atlas_fingerprint": self.atlas_fingerprint,
        }

    # ── introspection ────────────────────────────────────────────────────────
    @property
    def grid(self):
        return self._grid.copy()

    @property
    def reference_molecules(self):
        return list(self._ref_labels)

    @property
    def chemistry_axes(self):
        return list(self._axis_names)

    def __repr__(self):
        return (f"GAIRAEngine(atlas={self.atlas_fingerprint[:12]}…, "
                f"{len(self._lsm_ids)} LSMs, {len(self._csm_ids)} CSMs, "
                f"{len(self._ref_labels)} molecules, {len(self._axis_names)} chemistry axes)")


def _strip(d: dict) -> dict:
    """Drop the bulky intermediate arrays a report does not need to carry."""
    return {k: v for k, v in d.items()
            if k not in ("reconstruction", "all_similarities")}


def _asls(y, lam: float = 1e5, p: float = 0.01, n_iter: int = 10) -> np.ndarray:
    """Asymmetric least-squares baseline (Eilers & Boelens 2005)."""
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    n = len(y)
    D = sparse.diags([1.0, -2.0, 1.0], [0, -1, -2], shape=(n, n - 2))
    DD = lam * D.dot(D.T)
    w = np.ones(n)
    z = np.zeros(n)
    for _ in range(n_iter):
        W = sparse.spdiags(w, 0, n, n)
        z = spsolve((W + DD).tocsc(), w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z

