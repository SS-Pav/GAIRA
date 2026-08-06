"""GAIRA V7 — Phase 05: the canonical inference engine.

    spectrum → canonical preprocessing → non-negative CSM projection → 49-d activation
             → { analyte retrieval | chemistry class | evidence profile | provenance | uncertainty }

Everything upstream is frozen and simply read. The engine fits nothing at inference time, so a
given spectrum produces a bit-for-bit identical report on every run.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np

from . import calibration, evidence, openset, projection, provenance, retrieval

EPS = 1e-12


@dataclass
class InferenceReport:
    """One spectrum's complete answer, including what the engine could not answer."""
    activation: np.ndarray
    diagnostics: dict
    top_molecules: list[tuple[str, float]]
    confidence: float
    margin: float
    entropy: float
    chemistry_class: tuple[str, float]
    class_top3: list[tuple[str, float]]
    evidence_profile: dict
    provenance: list[dict] = field(default_factory=list)
    rejected: bool = False
    rejection_score: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "activation": [float(x) for x in self.activation],
            "diagnostics": {k: float(v) for k, v in self.diagnostics.items()},
            "top_molecules": [[m, float(s)] for m, s in self.top_molecules],
            "confidence": float(self.confidence), "margin": float(self.margin),
            "entropy": float(self.entropy),
            "chemistry_class": [self.chemistry_class[0], float(self.chemistry_class[1])],
            "class_top3": [[c, float(s)] for c, s in self.class_top3],
            "evidence_profile": self.evidence_profile,
            "provenance": self.provenance,
            "rejected": bool(self.rejected),
            "rejection_score": float(self.rejection_score),
            "notes": self.notes,
        }


class CanonicalEngine:
    """The frozen engine. Construction loads frozen artifacts; `infer` only reads them."""

    def __init__(self, CSM, csm_records, grid, ref_bank, ref_labels, ref_classes,
                 axis_map, axis_unassigned, axis_spec, calibrator, metric="cosine",
                 cov_inv=None, ref_channels=None, reject_threshold=None, ref_mean=None):
        self.CSM, self.csm_records, self.grid = CSM, csm_records, np.asarray(grid, float)
        self.ref_bank, self.ref_labels = ref_bank, list(ref_labels)
        self.ref_classes = list(ref_classes)
        self.M, self.unassigned, self.spec = axis_map, axis_unassigned, axis_spec
        self.calibrator, self.metric, self.cov_inv = calibrator, metric, cov_inv
        self.ref_channels, self.reject_threshold = ref_channels, reject_threshold
        self.ref_mean = ref_mean
        self.axis_index = {a: i for i, a in enumerate(evidence.AXIS_NAMES)}

    # ── the pipeline ─────────────────────────────────────────────────────────
    def infer(self, X: np.ndarray, top_k: int = 5) -> list[InferenceReport]:
        X = np.atleast_2d(np.asarray(X, float))
        A = projection.project(X, self.CSM)
        D = projection.diagnostics(X, A, self.CSM)
        ret = retrieval.retrieve(A, self.ref_bank, self.ref_labels, self.metric,
                                 self.cov_inv, k=max(top_k, 5))
        conf = self.calibrator.transform(ret["similarity"])
        prof = evidence.profile(A, self.M, self.spec, D["explained_variance"])
        chan = openset.channel_scores(A, D, self.ref_bank, self.cov_inv, self.ref_mean)
        rej = (openset.joint_score(chan, self.ref_channels) if self.ref_channels
               else np.zeros(len(A)))
        reports = []
        for i in range(len(A)):
            S = ret["similarity"][i]
            cls_score: dict[str, float] = {}
            for c, s in zip(self.ref_classes, S):
                cls_score[c] = max(cls_score.get(c, -np.inf), float(s))
            cls_rank = sorted(cls_score.items(), key=lambda kv: (-kv[1], kv[0]))
            mols = [(self.ref_labels[j], float(S[j])) for j in ret["order"][i][:top_k]]
            active = [a for a in evidence.AXIS_NAMES
                      if prof["magnitude"][i][self.axis_index[a]] > 0.02]
            prov = [provenance.axis_chain(a, A[i], self.M, self.csm_records, self.axis_index)
                    for a in active]
            notes = []
            if D["explained_variance"][i] < 0.5:
                notes.append("low reconstruction: the frozen atlas explains <50% of this spectrum")
            if float(prof["magnitude"][i].max()) > 0.6 and prof["support"][i].max() < 2:
                notes.append("dominant axis rests on a single CSM")
            rejected = bool(self.reject_threshold is not None and rej[i] > self.reject_threshold)
            if rejected:
                notes.append("REJECTED: evidence is outside the frozen atlas's domain; "
                             "molecule identity is not reported")
            reports.append(InferenceReport(
                activation=A[i],
                diagnostics={k: float(np.asarray(v)[i]) for k, v in D.items()
                             if k != "reconstruction"},
                top_molecules=[] if rejected else mols,
                confidence=float(conf[i]), margin=float(ret["margin"][i]),
                entropy=float(ret["entropy"][i]),
                chemistry_class=(cls_rank[0][0], float(cls_rank[0][1])),
                class_top3=[(c, float(s)) for c, s in cls_rank[:3]],
                evidence_profile={
                    "axes": list(evidence.AXIS_NAMES),
                    "magnitude": prof["magnitude"][i].tolist(),
                    "coverage": prof["coverage"][i].tolist(),
                    "confidence": prof["confidence"][i].tolist(),
                    "support": prof["support"][i].tolist(),
                    "unassigned_mass": float((A[i] @ self.unassigned) / (A[i].sum() + EPS)),
                },
                provenance=prov, rejected=rejected, rejection_score=float(rej[i]),
                notes=notes))
        return reports

    def fingerprint(self) -> str:
        """Determinism anchor: hashes every frozen object the engine's answer depends on."""
        h = hashlib.md5()
        for arr in (self.CSM, self.ref_bank, self.M, self.unassigned, self.spec, self.grid):
            h.update(np.ascontiguousarray(np.asarray(arr, float)).round(10).tobytes())
        h.update(json.dumps([self.ref_labels, self.ref_classes, self.metric,
                             self.calibrator.method], sort_keys=True).encode())
        return h.hexdigest()
