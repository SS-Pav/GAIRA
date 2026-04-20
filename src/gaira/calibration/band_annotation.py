"""Band annotation — attach candidate motif / theme labels to 22-window IDs.

ANNOTATION ONLY. Does not participate in quantitative BSV computation; used
for interpretability and for flagging many-to-many ambiguity when a window
carries multiple plausible assignments.

Assignment source: standard Raman/SERS mode lists (Tu 1982, De Gelder et al.
2007, Movasaghi et al. 2007). Where two or more distinct biochemical motifs
overlap a window, the `ambiguity` field is set.
"""
from __future__ import annotations

from dataclasses import dataclass

from gaira.spectral.window_panel import WINDOW_DEFS


@dataclass(frozen=True)
class WindowAnnotation:
    window_id: str
    wavenumber_range: tuple[int, int]
    bsv_component: str
    candidate_motifs: list[str]      # plausible biochemical assignments
    example_analytes: list[str]      # concrete molecules commonly cited
    ambiguity: str | None            # short note if many-to-many


# Hand-curated from standard literature. Motifs are candidate labels, not
# exclusive assignments — e.g. the 700-740 window is listed as carrying both
# purine ring breathing AND methionine/cysteine C-S, a known ambiguity.
_ANNOTATIONS: dict[str, WindowAnnotation] = {
    "450-500": WindowAnnotation(
        "450-500", (450, 500), "redox_metabolite",
        candidate_motifs=["S-S disulfide stretch", "glucose ring"],
        example_analytes=["cystine", "glucose"],
        ambiguity="disulfide vs sugar — depends on matrix",
    ),
    "500-540": WindowAnnotation(
        "500-540", (500, 540), "redox_metabolite",
        candidate_motifs=["S-S disulfide stretch", "PO bending"],
        example_analytes=["cystine", "phospholipids"],
        ambiguity=None,
    ),
    "540-580": WindowAnnotation(
        "540-580", (540, 580), "unmapped",
        candidate_motifs=["weak matrix/background"],
        example_analytes=[],
        ambiguity="no dominant assignment — treated as unmapped",
    ),
    "580-620": WindowAnnotation(
        "580-620", (580, 620), "unmapped",
        candidate_motifs=["phenylalanine skeletal (weak)", "matrix"],
        example_analytes=[],
        ambiguity="no dominant assignment — treated as unmapped",
    ),
    "620-660": WindowAnnotation(
        "620-660", (620, 660), "aromatic_amino_acid",
        candidate_motifs=["phenylalanine ring", "C-C twist",
                           "uric acid ring breathing (~635)"],
        example_analytes=["phenylalanine", "uric_acid"],
        ambiguity="Phe and uric-acid ring modes co-occupy this window",
    ),
    "660-700": WindowAnnotation(
        "660-700", (660, 700), "purine_nucleotide",
        candidate_motifs=["tyrosine ring out-of-plane", "guanine"],
        example_analytes=["tyrosine", "guanine"],
        ambiguity=None,
    ),
    "700-740": WindowAnnotation(
        "700-740", (700, 740), "purine_nucleotide",
        candidate_motifs=["adenine ring breathing (~725)",
                           "hypoxanthine ring",
                           "imidazole ring breathing",
                           "methionine C-S (~704)"],
        example_analytes=["adenine", "hypoxanthine", "ergothioneine", "methionine"],
        ambiguity=("purine ring vs imidazole ring vs C-S stretch — "
                   "core confound region for calibration analytes"),
    ),
    "740-780": WindowAnnotation(
        "740-780", (740, 780), "pyrimidine_nucleotide",
        candidate_motifs=["tryptophan indole (~759)", "cytosine/uracil ring"],
        example_analytes=["tryptophan", "cytosine"],
        ambiguity="Trp and pyrimidine overlap",
    ),
    "780-820": WindowAnnotation(
        "780-820", (780, 820), "pyrimidine_nucleotide",
        candidate_motifs=["cytosine/uracil ring", "O-P-O backbone"],
        example_analytes=["cytosine", "RNA/DNA phosphate"],
        ambiguity=None,
    ),
    "820-860": WindowAnnotation(
        "820-860", (820, 860), "aromatic_amino_acid",
        candidate_motifs=["tyrosine Fermi doublet", "polysaccharide"],
        example_analytes=["tyrosine", "glycogen"],
        ambiguity=None,
    ),
    "860-920": WindowAnnotation(
        "860-920", (860, 920), "glycan_carbohydrate",
        candidate_motifs=["tryptophan (~876)",
                           "uric acid (~890)",
                           "glucose C-O-C", "proline ring"],
        example_analytes=["glucose", "uric_acid", "tryptophan"],
        ambiguity="uric acid ~890 overlaps glycan assignment",
    ),
    "920-980": WindowAnnotation(
        "920-980", (920, 980), "protein_backbone",
        candidate_motifs=["C-C backbone stretch", "glycogen"],
        example_analytes=["proteins", "glycogen"],
        ambiguity=None,
    ),
    "980-1020": WindowAnnotation(
        "980-1020", (980, 1020), "aromatic_amino_acid",
        candidate_motifs=["phenylalanine ring breathing (~1003)"],
        example_analytes=["phenylalanine"],
        ambiguity=None,
    ),
    "1020-1080": WindowAnnotation(
        "1020-1080", (1020, 1080), "nucleic_acid_backbone",
        candidate_motifs=["C-N stretch", "phosphodiester", "glycogen"],
        example_analytes=["nucleic_acids", "phospholipids"],
        ambiguity=None,
    ),
    "1080-1140": WindowAnnotation(
        "1080-1140", (1080, 1140), "glycan_carbohydrate",
        candidate_motifs=["C-C/C-O stretch (glucose)",
                           "PO2⁻ symmetric stretch",
                           "uric acid (~1130)"],
        example_analytes=["glucose", "phospholipids", "uric_acid"],
        ambiguity="sugar C-O vs phosphate vs uric-acid overlap",
    ),
    "1140-1200": WindowAnnotation(
        "1140-1200", (1140, 1200), "membrane_lipid",
        candidate_motifs=["lipid CH2 twist", "tyrosine"],
        example_analytes=["lipids", "tyrosine"],
        ambiguity=None,
    ),
    "1200-1260": WindowAnnotation(
        "1200-1260", (1200, 1260), "protein_backbone",
        candidate_motifs=["Amide III (β-sheet)", "C-H bend"],
        example_analytes=["proteins"],
        ambiguity=None,
    ),
    "1260-1320": WindowAnnotation(
        "1260-1320", (1260, 1320), "protein_backbone",
        candidate_motifs=["Amide III (α-helix)", "lipid δCH2"],
        example_analytes=["proteins", "lipids"],
        ambiguity=None,
    ),
    "1320-1380": WindowAnnotation(
        "1320-1380", (1320, 1380), "purine_nucleotide",
        candidate_motifs=["guanine/adenine N7-C8", "lipid δCH3"],
        example_analytes=["guanine", "adenine", "lipids"],
        ambiguity="purine vs lipid CH3 overlap",
    ),
    "1380-1450": WindowAnnotation(
        "1380-1450", (1380, 1450), "membrane_lipid",
        candidate_motifs=["δCH2/CH3 lipid", "COO⁻ symmetric"],
        example_analytes=["lipids", "carboxylates"],
        ambiguity=None,
    ),
    "1450-1520": WindowAnnotation(
        "1450-1520", (1450, 1520), "protein_backbone",
        candidate_motifs=["δCH2 scissors", "Amide II (weak)"],
        example_analytes=["proteins", "lipids"],
        ambiguity=None,
    ),
    "1520-1600": WindowAnnotation(
        "1520-1600", (1520, 1600), "aromatic_amino_acid",
        candidate_motifs=["tryptophan", "Amide II",
                           "uric acid (~1510, edge)", "ring C=C"],
        example_analytes=["tryptophan", "proteins", "uric_acid"],
        ambiguity="aromatic ring vs Amide II vs uric-acid edge",
    ),
}

# Sanity check: cover every window in WINDOW_DEFS
assert set(_ANNOTATIONS) == {wid for wid, *_ in WINDOW_DEFS}, (
    "band_annotation._ANNOTATIONS out of sync with WINDOW_DEFS"
)


def annotate_window(window_id: str) -> WindowAnnotation:
    return _ANNOTATIONS[window_id]


def annotate_top_windows(windows: list[dict], top_n: int = 6) -> list[dict]:
    """Attach motif annotations to the top-N windows from a band-driver list.

    Input is the output of band_drivers.compute_per_cohort_window_importance()
    — a list of {window_id, wavenumber_start, wavenumber_end, bsv_component,
    delta, effect_size, direction, ...} dicts, pre-sorted by |effect_size|.

    Returns the top_n entries annotated with candidate_motifs, example_analytes,
    and an ambiguity note.
    """
    out = []
    for w in windows[:top_n]:
        ann = _ANNOTATIONS.get(w["window_id"])
        if ann is None:
            continue
        out.append({
            **w,
            "candidate_motifs": list(ann.candidate_motifs),
            "example_analytes": list(ann.example_analytes),
            "ambiguity": ann.ambiguity,
        })
    return out
