"""GAIRA V7 — Phase 04: the canonical inference pathway.

One function, one direction, no fitting. `FrozenAtlas` loads every frozen layer and verifies
its fingerprint; `project_spectrum` walks a spectrum up the hierarchy and returns a
`SpectrumState`.

**Three invariants this module exists to guarantee:**

1. *No fitting.* Nothing here calls `fit`, `fit_transform` or `partial_fit`, and nothing draws
   a random number. The static check in the tests enforces it.
2. *Batch independence.* Every quantity for a spectrum depends only on that spectrum and the
   frozen atlas. A spectrum's output is identical alone and in a batch of a thousand.
3. *Determinism.* Same input, same atlas, bit-identical output.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import aggregation as AGG
from . import geometry as GEO
from . import projection as PRJ
from .state import SpectrumState, assign_confidence

EPS = 1e-12
ENGINE_VERSION = "v7_engine_v1"


@dataclass
class FrozenAtlas:
    """Every frozen layer, loaded once and fingerprint-verified."""
    grid: np.ndarray
    H_lsm: np.ndarray
    lsm_ids: list[str]
    CSM: np.ndarray
    csm_ids: list[str]
    csm_members: dict[str, list[str]]
    csm_registry: dict
    S: np.ndarray
    theme_ids: list[str]
    theme_names: list[str]
    theme_basis: np.ndarray
    theme_accepted: np.ndarray
    theme_confidence: np.ndarray
    D_ref: np.ndarray
    coords_ref: np.ndarray
    bridge_csms: set
    bsv_frame: dict
    config: dict
    fingerprints: dict

    @classmethod
    def load(cls, frozen_root: Path, config: dict, expected: dict) -> "FrozenAtlas":
        F = Path(frozen_root)
        z = np.load(F / "phase01/artifacts/lsm_dictionary_v1.npz", allow_pickle=True)
        c = np.load(F / "phase02/artifacts/csm_dictionary_v1.npz", allow_pickle=True)
        t = np.load(F / "phase03/artifacts/theme_membership_v1.npz", allow_pickle=True)
        creg = json.loads((F / "phase02/artifacts/csm_registry_v1.json").read_text())
        treg = json.loads((F / "phase03/artifacts/theme_registry_v1.json").read_text())
        s01 = json.loads((F / "phase01/PHASE_STATE.json").read_text())
        s02 = json.loads((F / "phase02/PHASE_STATE.json").read_text())
        s03 = json.loads((F / "phase03/PHASE_STATE.json").read_text())
        got = {"lsm": s01["registry_fingerprint"], "csm": s02["csm_fingerprint"],
               "theme": s03["theme_fingerprint"], "atlas": s01["atlas_fingerprint"]}
        for k, want in expected.items():
            if got.get(k) != want:
                raise RuntimeError(f"frozen {k} fingerprint {got.get(k)} != expected {want} — "
                                   f"the atlas has changed; refusing to run")
        by = {x["csm_id"]: x for x in creg["csms"]}
        csm_ids = [str(s) for s in c["csm_ids"]]
        themes = treg["themes"]
        return cls(
            grid=np.asarray(c["grid"], float),
            H_lsm=np.asarray(z["H"], float), lsm_ids=[str(s) for s in z["motif_ids"]],
            CSM=np.asarray(c["CSM"], float), csm_ids=csm_ids,
            csm_members={cid: [l["lsm_id"] for l in by[cid]["contributing_lsms"]]
                         for cid in csm_ids},
            csm_registry=by,
            S=np.asarray(t["S"], float), theme_ids=[str(s) for s in t["theme_ids"]],
            theme_names=[x["name"] for x in themes],
            theme_basis=np.asarray(t["THEMES"], float),
            theme_accepted=np.array([x["status"] == "accepted" for x in themes]),
            theme_confidence=np.array([x["confidence"] for x in themes], float),
            D_ref=np.asarray(t["D_csm"], float), coords_ref=np.asarray(t["coords"], float),
            bridge_csms=set(treg.get("bridge_csms", [])),
            bsv_frame={}, config=config,
            fingerprints={**got, "atlas": expected.get("atlas", "")})

    def with_frame(self, frame: dict) -> "FrozenAtlas":
        self.bsv_frame = frame
        return self


def preprocessing_hash(config: dict) -> str:
    """Hash of the engine configuration, over the *decisions* only.

    Calibration data carried in the config — the reference residual scale, for instance — is
    excluded: it is derived from the frozen corpus, not a choice, and including a 375-element
    array would make the hash depend on floating-point formatting rather than on the setup.
    """
    decisions = {k: v for k, v in config.items() if not isinstance(v, np.ndarray)}
    return hashlib.sha256(json.dumps(decisions, sort_keys=True,
                                     default=str).encode()).hexdigest()[:16]


def quality_control(x: np.ndarray, grid: np.ndarray) -> dict:
    """Measured before anything is interpreted, so a bad spectrum cannot look confident.

    The second-difference noise estimate is the one Phase 00 arrived at after a first-
    difference version scored every curated reference identically — on library spectra a
    first difference measures band sharpness, not noise.
    """
    x = np.asarray(x, float)
    finite = np.isfinite(x)
    xf = np.where(finite, x, 0.0)
    d2 = np.diff(xf, 2)
    noise = float(np.median(np.abs(d2)) * 1.4826 / np.sqrt(6.0)) + EPS
    snr = float(xf.max() / noise)
    return {"grid_coverage": float(finite.mean()),
            "snr": snr,
            "quality": float(np.clip(np.log10(max(snr, 1.0)) / 2.5, 0.0, 1.0)),
            "n_negative_bins": int((x < 0).sum()),
            "saturated": bool(np.isclose(xf, xf.max()).sum() > 5)}


def project_spectrum(x: np.ndarray, atlas: FrozenAtlas, spectrum_id: str = "unknown",
                     canonical_id: str | None = None) -> SpectrumState:
    """One spectrum, all the way up. Projection only; nothing is fitted."""
    cfg = atlas.config
    x = np.asarray(x, float).ravel()
    qc = quality_control(x, atlas.grid)

    # ── LSM ──────────────────────────────────────────────────────────────────
    a_lsm = PRJ.project(x, atlas.H_lsm, cfg["projection_method"],
                        **cfg.get("projection_kwargs", {}))[0]
    rec_lsm = a_lsm @ atlas.H_lsm
    ev_lsm = float(np.clip(1.0 - ((x - rec_lsm) ** 2).sum() / ((x ** 2).sum() + EPS), 0, None))

    # ── CSM ──────────────────────────────────────────────────────────────────
    a_csm = AGG.lsm_to_csm(a_lsm[None, :], atlas.lsm_ids, atlas.csm_members, atlas.csm_ids,
                           cfg["aggregation_method"], x=x[None, :], CSM=atlas.CSM)[0]
    u_csm = AGG.csm_uncertainty(a_lsm[None, :], atlas.lsm_ids, atlas.csm_members,
                                atlas.csm_ids)[0]
    rec_csm = a_csm @ atlas.CSM
    ev_csm = float(np.clip(1.0 - ((x - rec_csm) ** 2).sum() / ((x ** 2).sum() + EPS), 0, None))

    # ── themes ───────────────────────────────────────────────────────────────
    t_all = AGG.theme_activation(a_csm[None, :], atlas.S, cfg["theme_mode"],
                                 atlas.theme_confidence, atlas.theme_accepted,
                                 topk=cfg.get("theme_topk", 2))[0]
    t_acc = t_all[atlas.theme_accepted]
    rejected_mass = float(AGG.rejected_theme_to_uncertainty(t_all[None, :],
                                                            atlas.theme_accepted)[0])
    p = t_acc / (t_acc.sum() + EPS)
    ent = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    ent_n = ent / (np.log(max(len(t_acc), 2)) + EPS)

    # ── geometry ─────────────────────────────────────────────────────────────
    d_new = _csm_distance(a_csm, atlas)
    coords = GEO.extend(cfg["geometry_extension"], d_new[None, :], atlas.D_ref,
                        atlas.coords_ref, cfg.get("knn", 5))[0]
    dens = float(GEO.local_density(d_new[None, :], cfg.get("knn", 5))[0])
    ood_geo = float(GEO.ood_score(d_new[None, :], atlas.D_ref, cfg.get("knn", 5))[0])
    ood_res = GEO.residual_ood(x, atlas.CSM, a_csm, cfg.get("reference_residuals"))
    # The residual score is primary: the geometric one is computed on the reconstruction, which
    # lies inside the dictionary cone by construction and cannot see out-of-domain chemistry.
    ood = ood_res
    bridge = float(GEO.bridge_proximity(d_new[None, :], atlas.csm_ids, atlas.bridge_csms,
                                        cfg.get("knn", 5))[0])
    nearest = GEO.nearest_references(d_new[None, :], atlas.csm_ids, cfg.get("knn", 5))[0]
    loc_conf = float(GEO.local_confidence(np.array([dens]), np.array([ood]),
                                          np.array([bridge]))[0])

    # ── BSV ──────────────────────────────────────────────────────────────────
    bsv, names = AGG.build_bsv(t_acc[None, :], np.array([1.0 - ev_csm]),
                               np.array([rejected_mass]), np.array([bridge]),
                               cfg["bsv_variant"])
    bsv = bsv[0]
    elev = (AGG.bsv_elevation(bsv[None, :], atlas.bsv_frame)[0]
            if atlas.bsv_frame else np.zeros_like(bsv))

    residual = {"explained_variance": ev_csm, "ev_lsm_basis": ev_lsm,
                "unexplained_fraction": float(1.0 - ev_csm),
                "residual_norm": float(np.linalg.norm(x - rec_csm))}
    unc = {"theme_entropy": ent, "theme_entropy_normalised": float(ent_n),
            "rejected_theme_mass": rejected_mass,
            "mean_csm_disagreement": float(u_csm.mean()),
            "geometry_local_confidence": loc_conf,
            "method": "entropy of accepted-theme activations; CSM-internal disagreement; "
                      "rejected-theme mass; geometric locality"}
    conf, tier = assign_confidence(qc, {"ood_score": ood}, residual, unc)

    flags = []
    if ood > 2.0:
        flags.append("out_of_domain")
    if ev_csm < 0.35:
        flags.append("poorly_reconstructed")
    if bridge >= 0.5:
        flags.append("bridge_neighbourhood")
    if rejected_mass > 0.25:
        flags.append("mass_on_rejected_theme")
    if qc["grid_coverage"] < 0.99:
        flags.append("incomplete_grid")

    top_csm = np.argsort(-a_csm)[:5]
    mols = _nearest_molecules(a_csm, atlas, top=5)
    return SpectrumState(
        spectrum_id=spectrum_id,
        atlas_fingerprint=atlas.fingerprints.get("atlas", ""),
        lsm_fingerprint=atlas.fingerprints["lsm"],
        csm_fingerprint=atlas.fingerprints["csm"],
        theme_fingerprint=atlas.fingerprints["theme"],
        preprocessing_config_hash=preprocessing_hash(cfg),
        engine_version=ENGINE_VERSION,
        qc=qc, lsm_activations=a_lsm, csm_activations=a_csm, csm_uncertainty=u_csm,
        theme_activations_all=t_all, theme_activations=t_acc,
        bsv=bsv, bsv_axis_names=names, bsv_elevation=elev,
        geometry_coords=coords, nearest_csms=nearest,
        nearest_molecules=mols,
        nearest_reference_spectra=[],
        bridge_memberships={"bridge_proximity": bridge,
                            "bridge_csms_in_neighbourhood":
                                [n["id"] for n in nearest if n["id"] in atlas.bridge_csms]},
        ood={"ood_score": ood, "ood_residual": ood_res, "ood_geometric": ood_geo,
             "local_density": dens,
             "distance_to_support": float(d_new.min()), "local_confidence": loc_conf},
        residual=residual, uncertainty=unc, confidence=conf, confidence_tier=tier,
        evidence={"top_csms": [{"csm_id": atlas.csm_ids[i],
                                "activation": float(a_csm[i])} for i in top_csm],
                  "top_themes": [{"theme_id": atlas.theme_ids[
                      np.where(atlas.theme_accepted)[0][j]],
                      "name": atlas.theme_names[np.where(atlas.theme_accepted)[0][j]],
                      "activation": float(t_acc[j])}
                      for j in np.argsort(-t_acc)[:3]]},
        provenance={"true_canonical_id": canonical_id,
                    "chain": "spectrum → LSM → CSM → theme → BSV → geometry",
                    "engine": ENGINE_VERSION},
        flags=flags)


def _csm_distance(a_csm: np.ndarray, atlas: FrozenAtlas) -> np.ndarray:
    """Distance from a spectrum to each CSM, on the scale the frozen geometry uses.

    The frozen `D_ref` is a Wasserstein distance between CSM *spectra*. A new spectrum's
    distance to a CSM is computed the same way, so the extension places it on the same scale
    rather than on a new one.
    """
    from scipy.stats import wasserstein_distance
    g = atlas.grid
    span = float(g.max() - g.min()) or 1.0
    rec = np.clip(a_csm[:, None] * atlas.CSM, 0, None).sum(axis=0)
    p = rec / (rec.sum() + EPS)
    out = np.zeros(atlas.CSM.shape[0])
    for i, c in enumerate(atlas.CSM):
        q = np.clip(c, 0, None)
        q = q / (q.sum() + EPS)
        out[i] = wasserstein_distance(g, g, p, q) / span
    return out


def _nearest_molecules(a_csm: np.ndarray, atlas: FrozenAtlas, top: int = 5) -> list[dict]:
    """Molecules ranked by how much of the spectrum's CSM activation their CSMs carry.

    Retrieval through the hierarchy rather than around it: a molecule scores because the CSMs
    it supports are activated, which keeps the answer explainable.
    """
    score: dict[str, float] = {}
    for i, cid in enumerate(atlas.csm_ids):
        if a_csm[i] <= 0:
            continue
        rec = atlas.csm_registry.get(cid, {})
        mols = rec.get("supporting_analytes", [])
        if not mols:
            continue
        w = a_csm[i] / len(mols)
        for m in mols:
            score[m] = score.get(m, 0.0) + w
    ranked = sorted(score.items(), key=lambda kv: -kv[1])[:top]
    tot = sum(score.values()) + EPS
    return [{"canonical_id": m, "score": float(v), "share": float(v / tot)} for m, v in ranked]
