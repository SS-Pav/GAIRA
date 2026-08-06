"""GAIRA V7 — Phase 03: the Theme object and registry (contract C-08).

A theme is a spectrum plus the complete case for and against it. The counter-evidence field is
not decoration: a theme with no recorded counter-evidence has not been examined.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

SCHEMA = "theme_registry_v1"
GRID_BINS = 676
# Words that would make a theme name biology rather than chemistry (P-07). Checked, not trusted.
FORBIDDEN_NAME_TOKENS = (
    "cancer", "tumour", "tumor", "disease", "diabet", "sepsis", "infection", "inflamm",
    "pathway", "metabolism", "signalling", "signaling", "apoptosis", "proliferat",
    "phenotype", "prognos", "diagnos", "biomarker", "syndrome", "patient", "healthy",
    "malignan", "benign", "stage", "grade", "response", "process",
)


@dataclass
class Theme:
    theme_id: str                       # Theme-01 … assigned before any name exists
    index: int
    spectrum: np.ndarray
    dominant_bands: list[float]
    band_assignments: list[str]
    mode_families: list[str]
    dominant_families: list[str]
    family_concentration: float
    chemically_admissible: bool
    assigned_fraction: float
    # membership
    member_csms: list[str]
    member_memberships: list[float]
    bridge_csms: list[str]
    n_supporting_csms: int
    mean_membership: float
    membership_entropy: float
    # evidence
    geometry_evidence: dict = field(default_factory=dict)
    spectral_evidence: dict = field(default_factory=dict)
    biochemical_evidence: dict = field(default_factory=dict)
    counter_evidence: list[str] = field(default_factory=list)
    alternative_explanations: list[str] = field(default_factory=list)
    # validation
    bootstrap_stability: float = float("nan")
    loo_stability: float = float("nan")
    gradient: dict = field(default_factory=dict)
    source_robust: bool = True
    # post hoc
    name: str = ""
    chemical_definition: str = ""
    name_confidence: float = 0.0
    confidence: float = 0.0
    status: str = "accepted"
    rejection_reason: str | None = None
    limitations: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.spectrum = np.asarray(self.spectrum, float)
        if self.spectrum.shape != (GRID_BINS,):
            raise ValueError(f"{self.theme_id}: spectrum must be ({GRID_BINS},)")
        if (self.spectrum < 0).any():
            raise ValueError(f"{self.theme_id}: theme spectra must be non-negative (C-08)")
        if self.name:
            check_name(self.name)

    def to_row(self) -> dict:
        d = asdict(self)
        d.pop("spectrum")
        for k in ("member_csms", "bridge_csms", "band_assignments", "mode_families",
                  "dominant_families",
                  "counter_evidence", "alternative_explanations", "limitations"):
            d[k] = ";".join(map(str, d[k]))
        d["member_memberships"] = ";".join(f"{v:.3f}" for v in d["member_memberships"])
        d["dominant_bands"] = ";".join(f"{b:.0f}" for b in d["dominant_bands"])
        for k in ("geometry_evidence", "spectral_evidence", "biochemical_evidence", "gradient"):
            d[k] = "; ".join(f"{a}={b}" for a, b in d[k].items())
        return d


def check_name(name: str) -> None:
    """P-07: themes name chemistry. Biology is downstream and never propagates upstream."""
    low = name.lower()
    for tok in FORBIDDEN_NAME_TOKENS:
        if tok in low:
            raise ValueError(
                f"theme name {name!r} contains {tok!r} — themes name chemistry only; no "
                f"disease, pathway, process or phenotype (P-07)")


class ThemeRegistry:
    def __init__(self, model: str, K: int, selection: dict, provenance: dict):
        self.model = model
        self.K = K
        self.selection = selection
        self.provenance = provenance
        self._themes: list[Theme] = []
        self.unassigned_csms: list[str] = []
        self.bridge_csms: list[str] = []

    def add(self, t: Theme) -> None:
        if any(x.theme_id == t.theme_id for x in self._themes):
            raise ValueError(f"duplicate theme_id {t.theme_id}")
        self._themes.append(t)

    @property
    def themes(self) -> list[Theme]:
        return list(self._themes)

    @property
    def accepted(self) -> list[Theme]:
        return [t for t in self._themes if t.status == "accepted"]

    def basis(self) -> np.ndarray:
        return np.array([t.spectrum for t in self._themes])

    def table(self) -> pd.DataFrame:
        return pd.DataFrame([t.to_row() for t in self._themes])

    def fingerprint(self) -> str:
        h = hashlib.sha256(np.ascontiguousarray(self.basis()).tobytes())
        h.update("|".join(t.theme_id + ",".join(t.member_csms) for t in self._themes).encode())
        return h.hexdigest()[:32]

    def check_invariants(self, S: np.ndarray, all_csms: list[str]) -> list[dict]:
        out = []

        def chk(name, ok, detail=""):
            out.append({"invariant": name, "status": "PASS" if ok else "FAIL",
                        "detail": detail})

        chk("S >= 0", bool((S >= 0).all()))
        chk("S rows sum to 1", bool(np.allclose(S.sum(axis=1), 1.0, atol=1e-6)),
            f"max deviation {np.abs(S.sum(axis=1) - 1).max():.2e}")
        chk("S shape is (M, K)", S.shape == (len(all_csms), len(self._themes)),
            f"{S.shape} vs ({len(all_csms)}, {len(self._themes)})")
        chk("theme spectra non-negative", all((t.spectrum >= 0).all() for t in self._themes))
        chk("theme_id unique", len({t.theme_id for t in self._themes}) == len(self._themes))
        chk("membership is sparse",
            bool(np.sort(S, axis=1)[:, -2:].sum(axis=1).mean() >= 0.60),
            f"mean top-2 mass {np.sort(S, axis=1)[:, -2:].sum(axis=1).mean():.3f}")
        chk("no CSM forced to a single parent",
            bool((S.max(axis=1) < 0.999).any()),
            "at least one CSM has genuinely split membership")
        chk("no theme name refers to disease/pathway/process/phenotype",
            all(_name_ok(t.name) for t in self._themes))
        chk("every theme resolves to CSMs", all(t.member_csms or t.status != "accepted"
                                                for t in self._themes))
        chk("every theme carries counter-evidence",
            all(t.counter_evidence for t in self.accepted),
            "an accepted theme with no recorded counter-evidence has not been examined")
        return out

    def summary(self) -> dict:
        acc = self.accepted
        return {
            "schema": SCHEMA, "model": self.model, "K": self.K,
            "n_accepted": len(acc),
            "n_rejected": sum(t.status == "rejected" for t in self._themes),
            "n_unassigned_csms": len(self.unassigned_csms),
            "n_bridge_csms": len(self.bridge_csms),
            "n_named": sum(bool(t.name) for t in acc),
            "n_unknown": sum(t.name.lower().startswith("unknown") for t in acc),
            "mean_confidence": round(float(np.mean([t.confidence for t in acc])), 4) if acc else 0.0,
            "fingerprint": self.fingerprint(),
        }


def _name_ok(name: str) -> bool:
    try:
        check_name(name or "")
        return True
    except ValueError:
        return False
