"""GAIRA Spectral Physics Atlas loader.

Reads the Phase-4.1 atlas YAMLs and exposes constraints for a given
(wavenumber, modality, matrix) tuple. Deterministic, inspectable,
extensible to future bands.

The loader is a *constraint layer*: it does NOT score, does NOT
modify primitives, does NOT commit axes. It returns rules that a
validator or BSV scorer must obey.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_ATLAS_ROOT = Path(
    os.environ.get(
        "GAIRA_ATLAS_ROOT",
        "/Volumes/SSD_Rad/GAIRA_BUILD/atlas/phase4_1_outputs",
    )
)

# Source-spectrum id prefixes → modality / matrix inference. Kept
# explicit rather than heuristic so an auditor can reason about it.
_MODALITY_MATRIX_BY_PREFIX: dict[str, tuple[str, str]] = {
    "rbl": ("raman", "pure_compound"),
    "aa": ("raman", "pure_compound"),
    "erg": ("sers", "pure_compound_aqueous"),
}


def _infer_modality_matrix(source_spectrum_id: str | None) -> tuple[str, str]:
    if not source_spectrum_id:
        return ("unknown", "unknown")
    prefix = source_spectrum_id.split("_", 1)[0]
    return _MODALITY_MATRIX_BY_PREFIX.get(prefix, ("unknown", "unknown"))


# ----------------------------------------------------------------
# Phase 4.1.1 chemistry-family registry
# ----------------------------------------------------------------
# Maps a *component keyword* (found as a substring of the GAIRA
# source_spectrum_id) to the set of vibrational chemistry families
# that that component can physically produce. Companion detection
# and axis commit both consult this registry so that a peak in a
# companion window is only counted as chemistry-X evidence if the
# reference is actually capable of producing chemistry-X.
#
# Families are deliberately coarse and chemistry-grounded. They are
# not axis labels; they are mechanistic tags.
#
# Keys are matched as substrings (lowercased) against the
# source_spectrum_id. The most specific match wins via a longest-
# substring rule handled in ``reference_chemistry_families``.

_REFERENCE_CHEMISTRY_FAMILIES: dict[str, set[str]] = {
    # --- purine / imidazole-carrying ---
    "adenine":                {"purine", "imidazole"},
    "guanine":                {"purine", "imidazole"},
    "a_dna":                  {"purine", "imidazole", "phosphate", "carbohydrate"},
    "b_dna":                  {"purine", "imidazole", "phosphate", "carbohydrate"},
    "t_rna":                  {"purine", "imidazole", "phosphate", "carbohydrate"},
    "coenzyme_a":             {"purine", "imidazole", "thiol_free",
                               "phosphate"},
    "acetyl_coenzyme_a":      {"purine", "imidazole", "thioester",
                               "phosphate"},

    # --- imidazole-bearing non-purine ---
    "l_histidine":            {"imidazole"},
    "histidine":              {"imidazole"},
    "ergothioneine":          {"imidazole", "thione",
                               "trimethylammonium", "carboxylate"},

    # --- pyrimidine nucleobases ---
    "cytosine":               {"pyrimidine", "amine"},
    "thymine":                {"pyrimidine", "carbonyl"},
    "uracil":                 {"pyrimidine", "carbonyl"},

    # --- free-thiol / sulfur metabolites ---
    "glutathione":            {"thiol_free", "peptide_backbone",
                               "carboxylate"},
    "cysteine":               {"thiol_free"},
    "cystine":                {"disulfide", "carboxylate"},
    "methionine":             {"thioether_methionine"},
    "aa_met":                 {"thioether_methionine"},
    "aa_cys":                 {"thiol_free"},

    # --- aromatic AAs ---
    "l_phenylalanine":        {"aromatic_amino_acid", "aromatic_phenyl"},
    "l_tryptophan":           {"aromatic_amino_acid", "indole"},
    "l_tyrosine":             {"aromatic_amino_acid", "phenol"},

    # --- non-aromatic, non-sulfur AAs ---
    # (these have NO sulfur chemistry; a 660 peak in them is a
    # side-chain or backbone mode, NOT C-S rotamer.)
    "aa_arg":                 {"basic_amino_acid", "guanidinium"},
    "aa_asp":                 {"acidic_amino_acid", "carboxylate"},
    "aa_gly":                 {"simple_amino_acid"},
    "aa_ala":                 {"simple_amino_acid"},
    "aa_l_glu":               {"acidic_amino_acid", "carboxylate"},
    "aa_ser":                 {"simple_amino_acid", "hydroxyl"},
    "aa_thr":                 {"simple_amino_acid", "hydroxyl"},
    "aa_lys":                 {"basic_amino_acid"},
    "aa_pro":                 {"simple_amino_acid"},
    "aa_val":                 {"simple_amino_acid"},
    "aa_leu":                 {"simple_amino_acid"},
    "aa_ile":                 {"simple_amino_acid"},

    # --- proteins (carry many chemistries) ---
    "albumin":                {"protein", "disulfide",
                               "aromatic_amino_acid", "peptide_backbone"},
    "lactalbumin":            {"protein", "disulfide",
                               "peptide_backbone"},
    "collagen":               {"protein", "peptide_backbone",
                               "aromatic_amino_acid"},
    "hemoglobin":             {"protein", "imidazole",
                               "aromatic_amino_acid", "peptide_backbone",
                               "porphyrin"},
    "myoglobin":              {"protein", "imidazole",
                               "aromatic_amino_acid", "peptide_backbone",
                               "porphyrin"},
    "cytochrome_c":           {"protein", "imidazole", "porphyrin",
                               "peptide_backbone"},
    "keratin":                {"protein", "disulfide",
                               "peptide_backbone"},
    "elastin":                {"protein", "peptide_backbone"},
    "insulin":                {"protein", "disulfide",
                               "peptide_backbone"},
    "ferritin":               {"protein", "peptide_backbone"},
    "carbonic_anhydrase":     {"protein", "imidazole", "peptide_backbone"},

    # --- lipids (no sulfur, no purine, no imidazole) ---
    "cholesterol":            {"sterol"},
    "triolein":               {"triglyceride", "unsaturated_lipid"},
    "trilinolein":            {"triglyceride", "unsaturated_lipid"},
    "phosphatidylcholine":    {"phospholipid", "trimethylammonium",
                               "phosphate", "unsaturated_lipid"},

    # --- carotenoid ---
    "beta_carotene":          {"carotenoid"},

    # --- carbohydrates (no sulfur, no purine) ---
    "d_glucose":              {"carbohydrate"},
    "d_ribose":               {"carbohydrate"},
    "d_fructose":             {"carbohydrate"},
    "d_sucrose":              {"carbohydrate"},
    "2_deoxy_d_ribose":       {"carbohydrate"},

    # --- small organic acids / misc metabolites ---
    "acetoacetate":           {"carboxylate", "carbonyl"},
    "citric_acid":            {"carboxylate"},
    "pyruvate":               {"carboxylate", "carbonyl"},
    "malic_acid":             {"carboxylate", "hydroxyl"},
    "succinic_acid":          {"carboxylate"},
    "fumarate":               {"carboxylate", "unsaturated_lipid"},
    "phosphoenolpyruvate":    {"carboxylate", "phosphate", "enol"},
}


# Companion-window validity: which chemistry families can physically
# produce a peak in each companion window. A companion is only
# counted as evidence when the reference's chemistry families
# intersect this set.
_COMPANION_VALID_CHEMISTRIES: dict[str, set[str]] = {
    # --- purine side ---
    "comp_N7C8_1320_1380":              {"purine"},
    "comp_adenine_in_plane_1485":       {"purine", "imidazole"},

    # --- imidazole side ---
    "comp_imidazole_ring_packing_1480_1496": {"imidazole"},
    "comp_histidine_pH_marker":              {"imidazole"},

    # --- sulfur side ---
    # S-S disulfide is specific to disulfide bonds only.
    # Free-thiol compounds (GSH, Cys, CoA) do NOT have S-S;
    # their sulfur signal lives at C-S 630-670 and S-H 2560-2600.
    "comp_sulfur_ss_450_540":           {"disulfide"},
    # C-S rotamer covers methionine thioether, cysteine/cystine,
    # glutathione free thiol, CoA thiol, acetyl-CoA thioester, and
    # ergothioneine thione (per the Phase 4.1.1 user-specified
    # valid-chemistry list).
    "comp_cs_gauche_trans_630_670":     {"thioether_methionine",
                                          "thiol_free", "disulfide",
                                          "thioester", "thione"},
    # Phase 4.1.1 addition: S-H stretch at ~2570 cm-1
    "comp_sh_stretch_2560_2600":        {"thiol_free"},
    # Phase 4.1.2 closure-pass additions ---------------------------
    # Ergothioneine 2-thione N-C-S bend + C-S stretch at ~484 cm-1
    # (Vidrio 2020, Schiaretti 2025). Chemistry-specific to thione.
    "comp_ergothioneine_thione_480_490": {"thione"},
    # Histidine imidazole ring stretch pair (N1-H ~1576 /
    # N3-H ~1596) per Ashikawa 2002. Valid for imidazole-carrying
    # chemistry (plain histidine or ergothioneine).
    "comp_histidine_ring_stretch_1570_1600": {"imidazole", "thione"},

    # ============================================================
    # Phase 4 Pilot Wave A additions
    # ============================================================
    # Band 1000-1010 (phe1003_vs_carotenoid1005) companions --------
    "comp_phe_ring_breathing_620":          {"aromatic_phenyl"},
    "comp_phe_in_plane_1033":               {"aromatic_phenyl"},
    "comp_phe_ring_stretch_1606":           {"aromatic_phenyl",
                                              "aromatic_amino_acid"},
    "comp_carotenoid_cc_1150_1165":         {"carotenoid"},
    "comp_carotenoid_cdoublec_1505_1535":   {"carotenoid"},
    "comp_tyrosine_fermi_830_850":          {"phenol",
                                              "aromatic_amino_acid"},
    # Band 1080-1140 (phosphate_vs_carbohydrate) companions --------
    "comp_phosphate_bend_780":              {"phosphate"},
    "comp_phosphate_asymstretch_1220_1240": {"phosphate"},
    "comp_carbohydrate_ch_850_940":         {"carbohydrate"},
    "comp_carbohydrate_coc_900_960":        {"carbohydrate"},
    "comp_lipid_ch2_trans_1060":            {"triglyceride",
                                              "unsaturated_lipid",
                                              "phospholipid", "sterol"},
    "comp_lipid_ch2_twist_1295":            {"triglyceride",
                                              "unsaturated_lipid",
                                              "phospholipid", "sterol"},
    # Band 1380-1450 (CH2 + carboxylate + amide_III + purine_edge) -
    "comp_ch2_scissor_envelope_1440":       {"triglyceride",
                                              "unsaturated_lipid",
                                              "phospholipid", "sterol"},
    "comp_carboxylate_sym_1380_1420":       {"carboxylate"},
    "comp_carboxylate_cooh_1700_1730":      {"carboxylate"},
    "comp_amide_III_core_1230_1280":        {"peptide_backbone",
                                              "protein"},
    # Band 1630-1680 (amideI_vs_lipid_cdoublebond) companions ------
    "comp_amide_II_1540_1560":              {"peptide_backbone",
                                              "protein"},
    "comp_amide_III_1240_1280":             {"peptide_backbone",
                                              "protein"},
    "comp_lipid_cc_double_1655_1665":       {"unsaturated_lipid",
                                              "triglyceride",
                                              "phospholipid"},
    "comp_ch2_scissor_1440":                {"triglyceride",
                                              "unsaturated_lipid",
                                              "phospholipid", "sterol"},
    "comp_ch_vinyl_3000_3020":              {"unsaturated_lipid",
                                              "carotenoid",
                                              "triglyceride"},
}


# Axis commit gates: an axis may only be committed to when the
# reference's chemistry families intersect this set. Used together
# with the per-axis companion requirements.
_AXIS_REQUIRED_CHEMISTRIES: dict[str, set[str]] = {
    "purine_nucleobase":            {"purine"},
    "sulfur_thiol_redox":           {"thiol_free", "disulfide",
                                     "thioether_methionine", "thione",
                                     "thioester"},
    "aromatic_amino_acid":          {"aromatic_amino_acid", "imidazole"},
    "protein_backbone_peptide":     {"protein", "peptide_backbone"},
    "pyrimidine_nucleobase":        {"pyrimidine"},
    "carbohydrate_glycan":          {"carbohydrate"},
    "aliphatic_lipid":              {"triglyceride", "phospholipid",
                                     "sterol", "unsaturated_lipid"},
    "unsaturated_lipid_carbonyl":   {"triglyceride", "phospholipid",
                                     "unsaturated_lipid"},
    "nucleic_acid_phosphate":       {"phosphate"},
    "small_organic_acid_metabolite": {"carboxylate"},
}


@dataclass
class BandConstraints:
    band_id: str
    canonical_window_cm: tuple[float, float]
    ambiguity_type: str
    candidate_axes: list[str]
    primary_mechanisms: list[str]
    requires_companions: dict[str, list[str]]
    forbidden_assignments: list[str]
    confidence_rules: dict[str, float]
    penalties: dict[str, float]
    hard_veto_rules: list[str]
    matrix_weights: dict[str, float]
    ambiguity_zone_id: str
    rationale: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "band_id": self.band_id,
            "canonical_window_cm": list(self.canonical_window_cm),
            "ambiguity_type": self.ambiguity_type,
            "candidate_axes": self.candidate_axes,
            "primary_mechanisms": self.primary_mechanisms,
            "requires_companions": self.requires_companions,
            "forbidden_assignments": self.forbidden_assignments,
            "confidence_rules": self.confidence_rules,
            "penalties": self.penalties,
            "hard_veto_rules": self.hard_veto_rules,
            "matrix_weights": self.matrix_weights,
            "ambiguity_zone_id": self.ambiguity_zone_id,
            "rationale": self.rationale,
        }


class AtlasLoader:
    """Loads and caches atlas constraints per zone.

    Zone registration is explicit (via ``register_zone``) so adding a
    new band in Phase 4.2 is a one-liner.
    """

    def __init__(self, atlas_root: Path | str = _DEFAULT_ATLAS_ROOT):
        self.atlas_root = Path(atlas_root)
        self._zones: dict[str, dict[str, Path]] = {}
        self._zone_cache: dict[str, dict[str, Any]] = {}
        self._register_builtin_zones()

    def _register_builtin_zones(self) -> None:
        z700 = self.atlas_root / "zone_700_740"
        if z700.exists():
            self.register_zone(
                "700_740",
                {
                    "band_atlas": z700 / "spectral_band_atlas_v1.yaml",
                    "ambiguity_rules": z700 / "ambiguity_rules_v1.yaml",
                    "companion_rules": z700 / "companion_rules_v1.yaml",
                    "validation_expectations":
                        z700 / "validation_expectations_v1.yaml",
                },
            )

        # Phase 4 Pilot Wave A zones live under the inspection dir.
        pilot_root = Path(
            "/Volumes/SSD_Rad/GAIRA_BUILD/inspection/phase4_pilot_wave_a"
        )
        for band_id, folder in (
            ("1000_1010", "band_1000_1010"),
            ("1080_1140", "band_1080_1140"),
            ("1380_1450", "band_1380_1450"),
            ("1630_1680", "band_1630_1680"),
        ):
            zdir = pilot_root / folder
            if zdir.exists() and (zdir / "band_atlas_v1.yaml").exists():
                self.register_zone(band_id, {
                    "band_atlas": zdir / "band_atlas_v1.yaml",
                    "ambiguity_rules": zdir / "ambiguity_rules_v1.yaml",
                    "companion_rules": zdir / "companion_rules_v1.yaml",
                    "validation_expectations":
                        zdir / "validation_expectations_v1.yaml",
                })

    def register_zone(self, band_id: str, files: dict[str, Path]) -> None:
        self._zones[band_id] = files

    def _load_zone(self, band_id: str) -> dict[str, Any]:
        if band_id in self._zone_cache:
            return self._zone_cache[band_id]
        files = self._zones[band_id]
        bundle: dict[str, Any] = {}
        for key, path in files.items():
            with open(path) as f:
                bundle[key] = yaml.safe_load(f)
        self._zone_cache[band_id] = bundle
        return bundle

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------

    def band_id_for_wavenumber(self, wavenumber_cm: float) -> str | None:
        """Return the atlas band_id whose canonical window contains
        the wavenumber, or None."""
        for band_id in self._zones:
            bundle = self._load_zone(band_id)
            win = bundle["band_atlas"]["band"]["canonical_window_cm"]
            if win[0] <= wavenumber_cm <= win[1]:
                return band_id
        return None

    def get_band_constraints(
        self,
        wavenumber_cm: float | None = None,
        modality: str | None = None,
        matrix: str | None = None,
        *,
        band_id: str | None = None,
        expected_window_cm: tuple[float, float] | None = None,
    ) -> dict[str, Any] | None:
        """Return the constraint dict for the band that covers the
        given wavenumber or expected-window.

        If ``band_id`` is explicit, uses that. Otherwise prefers a
        wavenumber, falling back to overlap with ``expected_window_cm``.
        Returns None if no zone matches.
        """
        if band_id is None:
            if wavenumber_cm is not None:
                band_id = self.band_id_for_wavenumber(wavenumber_cm)
            elif expected_window_cm is not None:
                for candidate in self._zones:
                    bundle = self._load_zone(candidate)
                    win = bundle["band_atlas"]["band"]["canonical_window_cm"]
                    if (expected_window_cm[0] < win[1]
                            and expected_window_cm[1] > win[0]):
                        band_id = candidate
                        break
        if band_id is None:
            return None
        return self._build_constraints(band_id, modality, matrix)

    def _build_constraints(
        self, band_id: str, modality: str | None, matrix: str | None
    ) -> dict[str, Any]:
        bundle = self._load_zone(band_id)
        band = bundle["band_atlas"]["band"]
        chemistries = bundle["band_atlas"]["chemistries"]
        band_level = bundle["band_atlas"]["band_level_downstream_constraints"]
        amb_rules = bundle["ambiguity_rules"]["rules"]
        # use loader-augmented companion list (includes Phase 4.1.1
        # S-H stretch companion) instead of raw bundle list.
        companions = self.get_companion_specs(band_id)

        # ambiguity_type label aggregated from rules
        amb_types = []
        for r in amb_rules:
            label = (
                "Type_A_irreducible_structural_inheritance"
                if r.get("irreducible_on_single_peak")
                else "Type_C_or_D_edge_or_context_reducible"
            )
            if r["rule_id"].endswith("triple"):
                label = "Type_A_three_way_irreducible"
            amb_types.append(label)
        if any("Type_A" in t for t in amb_types):
            ambiguity_type = "Type_A_structural_inheritance_plus_Type_B_SERS_drift"
        else:
            ambiguity_type = "Type_C_edge_only"

        # candidate axes aggregated across chemistries
        cand_axes: list[str] = []
        for ch in chemistries:
            for ax in ch.get("candidate_axes_linked", []):
                if ax not in cand_axes:
                    cand_axes.append(ax)

        primary_mechanisms = [ch["chemistry_id"] for ch in chemistries]

        # companion requirements keyed by the axis they unlock
        requires_companions: dict[str, list[str]] = {}
        for ch in chemistries:
            axis = ch["downstream_constraints"]["phase5_axis_definition"].get(
                "primary_axis"
            )
            comp_ids = ch["downstream_constraints"]["phase6_bsv_scoring"].get(
                "companion_ids", []
            )
            comp_windows = []
            for cid in comp_ids:
                for comp in companions:
                    if comp["companion_id"] == cid:
                        lo, hi = comp["window_cm"]
                        comp_windows.append(f"{lo}-{hi}")
                        break
            if axis and comp_windows:
                existing = requires_companions.get(axis, [])
                for cw in comp_windows:
                    if cw not in existing:
                        existing.append(cw)
                requires_companions[axis] = existing

        # Add an explicit ergothioneine entry even though it is not its
        # own axis yet, because Phase-6 scoring must distinguish it
        # from histidine-imidazole.
        requires_companions.setdefault(
            "ergothioneine",
            ["450-540", "920-940", "750-770"],
        )

        # Phase 4.1.1: S-H stretch companion is mandatory for the
        # sulfur axis in addition to S-S / C-S. Inject its window
        # into the sulfur requirements list if not already present.
        if band_id == "700_740":
            sulfur_list = requires_companions.setdefault(
                "sulfur_thiol_redox", []
            )
            for w in ("450-540", "630-670", "2560-2600"):
                if w not in sulfur_list:
                    sulfur_list.append(w)

        # forbidden assignments derived from hard-veto expectations
        forbidden = []
        for ch in chemistries:
            gv = ch["downstream_constraints"].get("gaira_validate", {})
            if gv.get("strict_rule_id") == "sulfur_requires_s_s_companion":
                forbidden.append("sulfur_from_700_740_only")
            for pairs in (gv.get("must_pair_with") or []):
                pass  # captured as companion requirements above
        # fallback: always forbid sulfur-only commit in this zone
        if "sulfur_from_700_740_only" not in forbidden:
            forbidden.append("sulfur_from_700_740_only")

        # confidence rules from band-level downstream constraints.
        # The two-companion ceiling is capped at 0.85 per the
        # Phase-4.1 red-team audit (atlas YAML carries 0.90 as a
        # policy prior; the audit recommended the cap). The cap is
        # applied here so Phase-6 calibration does not inherit the
        # uncalibrated prior.
        p6 = band_level["phase6_bsv_scoring"]
        confidence_rules = {
            "no_companion": float(p6["max_confidence_without_companion"]),
            "one_companion": float(p6["max_confidence_with_one_companion"]),
            "two_companion": min(
                float(p6["max_confidence_with_two_companions"]), 0.85
            ),
        }

        # penalties
        penalties = {
            "modality_mismatch": -0.15,
            "matrix_mismatch": -0.10,
            "multi_axis_overlap": -float(p6.get("multi_chemistry_penalty", 0.10)),
            "internal_only_evidence": -0.10,
        }

        # matrix weights, averaged per-chemistry and per-matrix-key
        matrix_weights: dict[str, float] = {}
        for ch in chemistries:
            mw = ch["downstream_constraints"]["phase5_axis_definition"].get(
                "matrix_weights", {}
            )
            for k, v in mw.items():
                matrix_weights.setdefault(k, float(v))

        # hard veto rules
        hard_veto = [
            "no_sulfur_commit_without_sulfur_companion",
            "no_single_axis_commit_without_atlas_exception_match",
        ]

        # human-readable rationale (for audit trails)
        rationale = [
            f"zone {band_id} spans canonical_window_cm={band['canonical_window_cm']} "
            "with 3 candidate chemistries: purine ring breathing, imidazole "
            "ring breathing, C-S stretch lower edge.",
            "Type-A structural-inheritance ambiguity (purine contains "
            "imidazole) is irreducible on a single peak.",
            "Type-B SERS geometry drift spreads adenine across 710-740 "
            "on AgNP substrates.",
            "Any sulfur commit from this band alone is vetoed "
            "(sulfur_requires_s_s_companion strict rule).",
        ]

        return BandConstraints(
            band_id=band["band_id"],
            canonical_window_cm=tuple(band["canonical_window_cm"]),
            ambiguity_type=ambiguity_type,
            candidate_axes=cand_axes,
            primary_mechanisms=primary_mechanisms,
            requires_companions=requires_companions,
            forbidden_assignments=forbidden,
            confidence_rules=confidence_rules,
            penalties=penalties,
            hard_veto_rules=hard_veto,
            matrix_weights=matrix_weights,
            ambiguity_zone_id=band.get("gaira_vault_crosslink", "").split("/")[-1].replace(".md", ""),
            rationale=rationale,
        ).to_dict()

    def get_companion_specs(self, band_id: str) -> list[dict[str, Any]]:
        bundle = self._load_zone(band_id)
        specs = list(bundle["companion_rules"]["companions"])
        # Phase 4.1.1: inject the S-H stretch companion at loader
        # level without modifying the frozen atlas YAML.
        # Phase 4.1.2: also inject ergothioneine thione + histidine
        # ring stretch companions per the closure-pass patches.
        if band_id == "700_740":
            def _has(cid):
                return any(c["companion_id"] == cid for c in specs)

            if not _has("comp_sh_stretch_2560_2600"):
                specs.append({
                    "companion_id": "comp_sh_stretch_2560_2600",
                    "display_name": "Free-thiol S-H stretch",
                    "window_cm": [2560, 2600],
                    "supported_by": ["loader_v1_1"],
                    "modality_scope": ["raman"],
                    "added_in": "phase_4_1_1",
                })
            if not _has("comp_ergothioneine_thione_480_490"):
                specs.append({
                    "companion_id": "comp_ergothioneine_thione_480_490",
                    "display_name":
                        "Ergothioneine 2-thione N-C-S bend + C-S stretch",
                    "window_cm": [478, 492],
                    "supported_by": ["loader_v1_2",
                                     "vidrio_2020",
                                     "schiaretti_2025"],
                    "modality_scope": ["sers", "raman"],
                    "added_in": "phase_4_1_2",
                })
            if not _has("comp_histidine_ring_stretch_1570_1600"):
                specs.append({
                    "companion_id":
                        "comp_histidine_ring_stretch_1570_1600",
                    "display_name":
                        "Histidine imidazole ring stretch "
                        "(N1-H ~1576 / N3-H ~1596)",
                    "window_cm": [1570, 1600],
                    "supported_by": ["loader_v1_2",
                                     "ashikawa_2002"],
                    "modality_scope": ["raman"],
                    "added_in": "phase_4_1_2",
                })
            # Phase 4.1.2: expand the C-S window upper bound from
            # 670 to 690 to capture canonical cysteine/methionine
            # C-S stretch at ~680 per Van Wart.
            for c in specs:
                if c["companion_id"] == "comp_cs_gauche_trans_630_670":
                    c["window_cm"] = [630, 690]
                    break
            # Phase 4.1.2: widen histidine pH marker window to
            # capture both L-histidine (1282) and 4-methylimidazole
            # N1-H (1304) and N3-H (1260) tautomer breathing modes.
            for c in specs:
                if c["companion_id"] == "comp_histidine_pH_marker":
                    c["window_cm"] = [1255, 1310]
                    break
        return specs

    # ------------------------------------------------------------
    # Phase 4.1.1 chemistry validity API
    # ------------------------------------------------------------

    def reference_chemistry_families(
        self, source_spectrum_id: str | None
    ) -> set[str]:
        """Return the set of vibrational chemistry families a
        reference can physically produce.

        Uses a longest-substring match against the registry keys
        so that ``rbl_007_acetyl_coenzyme_a`` prefers
        ``acetyl_coenzyme_a`` over ``coenzyme_a``.
        Unknown references yield the empty set (strict default).
        """
        if not source_spectrum_id:
            return set()
        sid = source_spectrum_id.lower()
        best_key: str | None = None
        for key in _REFERENCE_CHEMISTRY_FAMILIES:
            if key in sid:
                if best_key is None or len(key) > len(best_key):
                    best_key = key
        if best_key is None:
            return set()
        return set(_REFERENCE_CHEMISTRY_FAMILIES[best_key])

    def is_valid_companion(
        self,
        source_spectrum_id: str | None,
        companion_id: str,
        *,
        band_present: bool = True,
    ) -> tuple[bool, str]:
        """Return (valid, reason) for a companion claim.

        A companion is valid iff:
          1) the band is detected in the reference's spectrum
             (``band_present=True``), AND
          2) the reference's chemistry families intersect the
             companion's valid-chemistry set (chemistry plausibility).
        """
        if not band_present:
            return False, "band not present in reference"
        valid_chems = _COMPANION_VALID_CHEMISTRIES.get(companion_id)
        if valid_chems is None:
            return False, f"companion {companion_id} has no validity rule"
        ref_chems = self.reference_chemistry_families(source_spectrum_id)
        if not ref_chems:
            return False, (
                f"reference {source_spectrum_id} has no known chemistry "
                "families (strict default)"
            )
        common = ref_chems & valid_chems
        if not common:
            return False, (
                f"reference chemistry {sorted(ref_chems)} cannot "
                f"produce {companion_id} (requires one of "
                f"{sorted(valid_chems)})"
            )
        return True, (
            f"valid: reference chemistry {sorted(common)} "
            f"satisfies {companion_id}"
        )

    def reference_can_commit_to_axis(
        self, source_spectrum_id: str | None, axis: str
    ) -> tuple[bool, str]:
        """Return (can_commit, reason) for an axis commit.

        A reference may commit to an axis only if its chemistry
        families intersect the axis's required chemistries.
        """
        required = _AXIS_REQUIRED_CHEMISTRIES.get(axis)
        if required is None:
            return True, "axis has no chemistry restriction"
        ref_chems = self.reference_chemistry_families(source_spectrum_id)
        if not ref_chems:
            return False, (
                f"reference {source_spectrum_id} has no known chemistry "
                "families; cannot commit to any axis (strict default)"
            )
        if ref_chems & required:
            return True, (
                f"reference chemistry {sorted(ref_chems & required)} "
                f"satisfies axis {axis}"
            )
        return False, (
            f"reference chemistry {sorted(ref_chems)} cannot "
            f"support axis {axis} (requires {sorted(required)})"
        )

    def get_ambiguity_rules(self, band_id: str) -> list[dict[str, Any]]:
        bundle = self._load_zone(band_id)
        return bundle["ambiguity_rules"]["rules"]

    def get_validation_expectations(self, band_id: str) -> list[dict[str, Any]]:
        bundle = self._load_zone(band_id)
        return bundle["validation_expectations"]["expectations"]


# ----------------------------------------------------------------
# Module-level convenience (module singleton; deterministic)
# ----------------------------------------------------------------

@lru_cache(maxsize=1)
def _default_loader() -> AtlasLoader:
    return AtlasLoader()


def get_band_constraints(
    wavenumber_cm: float | None = None,
    modality: str | None = None,
    matrix: str | None = None,
    *,
    expected_window_cm: tuple[float, float] | None = None,
    band_id: str | None = None,
) -> dict[str, Any] | None:
    """Module-level accessor using a cached default loader."""
    return _default_loader().get_band_constraints(
        wavenumber_cm=wavenumber_cm,
        modality=modality,
        matrix=matrix,
        expected_window_cm=expected_window_cm,
        band_id=band_id,
    )


def infer_modality_matrix(source_spectrum_id: str | None) -> tuple[str, str]:
    """Expose the modality/matrix inference as a pure helper."""
    return _infer_modality_matrix(source_spectrum_id)
