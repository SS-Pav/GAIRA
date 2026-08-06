"""GAIRA V7 — Phase 02: the CSM registry (contract C-07).

Indexed by CSM, with the LSM -> CSM inverse map kept alongside so provenance resolves in both
directions. A registry that can only be read forwards cannot answer "what happened to LSM x",
which is the question every falsification check starts from.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from .csm import CSM

SCHEMA = "gaira_v7_csm_registry_v1"


class CSMRegistry:
    def __init__(self, integration_method: str, selected_threshold: float,
                 consensus_operator: str, atlas_build: str = ""):
        self.integration_method = integration_method
        self.selected_threshold = selected_threshold
        self.consensus_operator = consensus_operator
        self.atlas_build = atlas_build
        self._csms: list[CSM] = []
        self.n_rejected_merges = 0        # proposals that failed falsification and were undone
        self.n_lsms_reverted = 0          # LSMs returned to separate CSMs by those rejections

    def add(self, csm: CSM) -> None:
        if any(c.csm_id == csm.csm_id for c in self._csms):
            raise ValueError(f"duplicate csm_id {csm.csm_id}")
        self._csms.append(csm)

    # ── access ───────────────────────────────────────────────────────────────
    @property
    def csms(self) -> list[CSM]:
        return list(self._csms)

    @property
    def accepted(self) -> list[CSM]:
        return [c for c in self._csms if c.status != "rejected"]

    def by_id(self, csm_id: str) -> CSM:
        for c in self._csms:
            if c.csm_id == csm_id:
                return c
        raise KeyError(csm_id)

    def lsm_to_csm(self) -> dict[str, str]:
        return {lsm: c.csm_id for c in self._csms for lsm in c.contributing_lsms}

    def dictionary(self) -> np.ndarray:
        return np.array([c.spectrum for c in self._csms])

    def table(self) -> pd.DataFrame:
        return pd.DataFrame([c.to_row() for c in self._csms])

    # ── invariants (C-07) ────────────────────────────────────────────────────
    def check_invariants(self, all_lsm_ids: list[str]) -> list[dict]:
        out = []

        def chk(name, ok, detail=""):
            out.append({"invariant": name, "status": "PASS" if ok else "FAIL",
                        "detail": detail})

        ids = [c.csm_id for c in self._csms]
        chk("csm_id unique", len(set(ids)) == len(ids))
        chk("CSM >= 0", all((c.spectrum >= 0).all() for c in self._csms))
        chk("is_singleton <=> n_lsms == 1",
            all(c.is_singleton == (c.n_lsms == 1) for c in self._csms))
        chk("is_anchored implies justification",
            all((not c.is_anchored) or bool(c.anchor_justification) for c in self._csms))

        assigned = [lsm for c in self._csms for lsm in c.contributing_lsms]
        chk("every LSM assigned to exactly one CSM",
            sorted(assigned) == sorted(all_lsm_ids),
            f"{len(assigned)} assignments over {len(all_lsm_ids)} LSMs; "
            f"duplicates={len(assigned) - len(set(assigned))}; "
            f"missing={sorted(set(all_lsm_ids) - set(assigned))}")
        chk("every CSM resolves to classes", all(c.supporting_classes for c in self._csms))
        chk("every CSM resolves to analytes", all(c.supporting_analytes for c in self._csms))
        chk("n_lsms matches contributing list",
            all(c.n_lsms == len(c.contributing_lsms) for c in self._csms))
        chk("weights sum to 1 per CSM",
            all(abs(sum(c.contributing_lsm_weights) - 1.0) < 1e-6 for c in self._csms))
        return out

    # ── serialisation ────────────────────────────────────────────────────────
    def fingerprint(self) -> str:
        D = self.dictionary()
        h = hashlib.sha256(np.ascontiguousarray(D).tobytes())
        h.update("|".join(c.csm_id + ",".join(c.contributing_lsms)
                          for c in self._csms).encode())
        return h.hexdigest()[:32]

    def summary(self) -> dict:
        acc = self.accepted
        return {
            "schema": SCHEMA,
            "integration_method": self.integration_method,
            "selected_threshold": self.selected_threshold,
            "consensus_operator": self.consensus_operator,
            "M": len(self._csms),
            "n_accepted": len(acc),
            "n_rejected_merges": self.n_rejected_merges,
            "n_lsms_reverted": self.n_lsms_reverted,
            "n_singletons": sum(c.is_singleton for c in self._csms),
            "n_cross_class": sum(c.is_cross_class for c in self._csms),
            "n_anchored": sum(c.is_anchored for c in self._csms),
            "mean_cohesion": round(float(np.mean([c.cohesion for c in self._csms])), 4),
            "mean_uncertainty": round(float(np.mean([c.uncertainty for c in self._csms])), 4),
            "fingerprint": self.fingerprint(),
        }
