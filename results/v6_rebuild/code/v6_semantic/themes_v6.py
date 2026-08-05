"""GAIRA V6 — chemical themes derived FROM MSS motifs (never from components).

A chemical theme at level K is a PARTITION of the biochemical MSS motifs into K
groups. Theme composition is the sum of the member motifs' activations:

    theme_t(x) = SUM_{m in t}  mss_m(x)        with mss = M^T . coord

so the full chain is  coord -> M -> mss -> T -> theme, a composition of two
non-negative linear maps. Because V6 MSS carries no theme information (see
mss_v6.py), this derivation is non-circular.

Five partition-generation methods are implemented and compared (Part 5):
    A manual expert hierarchy
    B agglomerative clustering on motif ACTIVATION correlation across the corpus
    C agglomerative clustering on motif SPECTRAL similarity (implied spectra)
    D ontology-aware clustering (chemical_class + shared exemplar families)
    E hybrid (mean of the B/C/D distance matrices)
"""
from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# ── Method A: the manual expert chemical hierarchy ───────────────────────────
# Read top-down: at each K, which motifs merge. Grounded in chemistry, fixed in
# advance of any evaluation.
MANUAL_LEVELS = {
    17: [["purine_ring_breathing"], ["oxopurine_carbonyl"], ["pyrimidine_ring"],
         ["nucleic_backbone_phosphate"], ["aromatic_ring_residue"], ["protein_amide_backbone"],
         ["amino_acid_zwitterion"], ["fatty_acyl_chain"], ["triglyceride_ester"],
         ["sterol_ring_system"], ["glycan_co_network"], ["polysaccharide_glycosidic"],
         ["carboxylate_organic_acid"], ["sulfur_heterocycle_thione"], ["porphyrin_macrocycle"],
         ["flavin_redox_cofactor"], ["carotenoid_polyene"]],
    14: [["purine_ring_breathing", "oxopurine_carbonyl"], ["pyrimidine_ring"],
         ["nucleic_backbone_phosphate"], ["aromatic_ring_residue"], ["protein_amide_backbone"],
         ["amino_acid_zwitterion"], ["fatty_acyl_chain"], ["triglyceride_ester"],
         ["sterol_ring_system"], ["glycan_co_network"], ["polysaccharide_glycosidic"],
         ["carboxylate_organic_acid"], ["sulfur_heterocycle_thione"],
         ["porphyrin_macrocycle", "flavin_redox_cofactor", "carotenoid_polyene"]],
    12: [["purine_ring_breathing", "oxopurine_carbonyl"], ["pyrimidine_ring"],
         ["nucleic_backbone_phosphate"], ["aromatic_ring_residue"], ["protein_amide_backbone"],
         ["amino_acid_zwitterion"], ["fatty_acyl_chain", "triglyceride_ester"],
         ["sterol_ring_system"], ["glycan_co_network"], ["polysaccharide_glycosidic"],
         ["carboxylate_organic_acid"], ["sulfur_heterocycle_thione", "porphyrin_macrocycle",
                                        "flavin_redox_cofactor", "carotenoid_polyene"]],
    10: [["purine_ring_breathing", "oxopurine_carbonyl"], ["pyrimidine_ring"],
         ["nucleic_backbone_phosphate"], ["aromatic_ring_residue"],
         ["protein_amide_backbone"], ["amino_acid_zwitterion"],
         ["fatty_acyl_chain", "triglyceride_ester", "sterol_ring_system"],
         ["glycan_co_network", "polysaccharide_glycosidic"], ["carboxylate_organic_acid"],
         ["sulfur_heterocycle_thione", "porphyrin_macrocycle", "flavin_redox_cofactor",
          "carotenoid_polyene"]],
    8: [["purine_ring_breathing", "oxopurine_carbonyl", "pyrimidine_ring"],
        ["nucleic_backbone_phosphate"], ["aromatic_ring_residue", "protein_amide_backbone"],
        ["amino_acid_zwitterion"], ["fatty_acyl_chain", "triglyceride_ester", "sterol_ring_system"],
        ["glycan_co_network", "polysaccharide_glycosidic"], ["carboxylate_organic_acid"],
        ["sulfur_heterocycle_thione", "porphyrin_macrocycle", "flavin_redox_cofactor",
         "carotenoid_polyene"]],
    6: [["purine_ring_breathing", "oxopurine_carbonyl", "pyrimidine_ring",
         "nucleic_backbone_phosphate"],
        ["aromatic_ring_residue", "protein_amide_backbone", "amino_acid_zwitterion"],
        ["fatty_acyl_chain", "triglyceride_ester", "sterol_ring_system"],
        ["glycan_co_network", "polysaccharide_glycosidic"],
        ["carboxylate_organic_acid"],
        ["sulfur_heterocycle_thione", "porphyrin_macrocycle", "flavin_redox_cofactor",
         "carotenoid_polyene"]],
    5: [["purine_ring_breathing", "oxopurine_carbonyl", "pyrimidine_ring",
         "nucleic_backbone_phosphate"],
        ["aromatic_ring_residue", "protein_amide_backbone", "amino_acid_zwitterion"],
        ["fatty_acyl_chain", "triglyceride_ester", "sterol_ring_system"],
        ["glycan_co_network", "polysaccharide_glycosidic"],
        ["carboxylate_organic_acid", "sulfur_heterocycle_thione", "porphyrin_macrocycle",
         "flavin_redox_cofactor", "carotenoid_polyene"]],
    4: [["purine_ring_breathing", "oxopurine_carbonyl", "pyrimidine_ring",
         "nucleic_backbone_phosphate"],
        ["aromatic_ring_residue", "protein_amide_backbone", "amino_acid_zwitterion"],
        ["fatty_acyl_chain", "triglyceride_ester", "sterol_ring_system"],
        ["glycan_co_network", "polysaccharide_glycosidic", "carboxylate_organic_acid",
         "sulfur_heterocycle_thione", "porphyrin_macrocycle", "flavin_redox_cofactor",
         "carotenoid_polyene"]],
    3: [["purine_ring_breathing", "oxopurine_carbonyl", "pyrimidine_ring",
         "nucleic_backbone_phosphate"],
        ["aromatic_ring_residue", "protein_amide_backbone", "amino_acid_zwitterion",
         "sulfur_heterocycle_thione", "porphyrin_macrocycle", "flavin_redox_cofactor"],
        ["fatty_acyl_chain", "triglyceride_ester", "sterol_ring_system", "glycan_co_network",
         "polysaccharide_glycosidic", "carboxylate_organic_acid", "carotenoid_polyene"]],
    2: [["purine_ring_breathing", "oxopurine_carbonyl", "pyrimidine_ring",
         "nucleic_backbone_phosphate", "aromatic_ring_residue", "protein_amide_backbone",
         "amino_acid_zwitterion", "sulfur_heterocycle_thione", "porphyrin_macrocycle",
         "flavin_redox_cofactor"],
        ["fatty_acyl_chain", "triglyceride_ester", "sterol_ring_system", "glycan_co_network",
         "polysaccharide_glycosidic", "carboxylate_organic_acid", "carotenoid_polyene"]],
}

MANUAL_NAMES = {
    "purine_ring_breathing": "Purine", "oxopurine_carbonyl": "Oxopurine",
    "pyrimidine_ring": "Pyrimidine", "nucleic_backbone_phosphate": "Phosphate ester",
    "aromatic_ring_residue": "Aromatic residue", "protein_amide_backbone": "Protein backbone",
    "amino_acid_zwitterion": "Free amino acid", "fatty_acyl_chain": "Fatty acyl",
    "triglyceride_ester": "Acylglycerol", "sterol_ring_system": "Sterol",
    "glycan_co_network": "Monosaccharide", "polysaccharide_glycosidic": "Polysaccharide",
    "carboxylate_organic_acid": "Organic acid", "sulfur_heterocycle_thione": "Sulfur metabolite",
    "porphyrin_macrocycle": "Porphyrin", "flavin_redox_cofactor": "Flavin",
    "carotenoid_polyene": "Carotenoid",
}


def name_group(members):
    """A readable theme name from its member motifs."""
    if len(members) == 1:
        return MANUAL_NAMES.get(members[0], members[0])
    key = frozenset(members)
    canned = {
        frozenset(["purine_ring_breathing", "oxopurine_carbonyl"]): "Purine",
        frozenset(["fatty_acyl_chain", "triglyceride_ester"]): "Acyl lipid",
        frozenset(["glycan_co_network", "polysaccharide_glycosidic"]): "Carbohydrate",
        frozenset(["aromatic_ring_residue", "protein_amide_backbone"]): "Protein",
        frozenset(["fatty_acyl_chain", "triglyceride_ester", "sterol_ring_system"]): "Lipid",
        frozenset(["purine_ring_breathing", "oxopurine_carbonyl", "pyrimidine_ring"]): "Nucleobase",
        frozenset(["purine_ring_breathing", "oxopurine_carbonyl", "pyrimidine_ring",
                   "nucleic_backbone_phosphate"]): "Nucleic acid",
        frozenset(["aromatic_ring_residue", "protein_amide_backbone",
                   "amino_acid_zwitterion"]): "Protein / amino acid",
        frozenset(["porphyrin_macrocycle", "flavin_redox_cofactor",
                   "carotenoid_polyene"]): "Conjugated cofactor",
        frozenset(["sulfur_heterocycle_thione", "porphyrin_macrocycle",
                   "flavin_redox_cofactor", "carotenoid_polyene"]): "Cofactor / redox",
    }
    if key in canned:
        return canned[key]
    return " + ".join(MANUAL_NAMES.get(m, m).split()[0] for m in members[:3]) + \
           ("…" if len(members) > 3 else "")


# ── distance matrices for methods B / C / D ─────────────────────────────────
def dist_activation(A):
    """1 - Pearson correlation of motif activations across the corpus."""
    C = np.corrcoef(A.T)
    C = np.nan_to_num(C, nan=0.0)
    D = 1.0 - C
    np.fill_diagonal(D, 0.0)
    return np.clip((D + D.T) / 2, 0, 2)


def dist_spectral(M, H):
    """1 - cosine between the motifs' implied Raman spectra (M[:,m]^T H)."""
    S = (M.T @ H)
    S = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-12)
    D = 1.0 - S @ S.T
    np.fill_diagonal(D, 0.0)
    return np.clip((D + D.T) / 2, 0, 2)


def dist_ontology(motifs, analyte_hits):
    """Chemical distance: 1 - Jaccard over (chemical_class, matched corpus analytes)."""
    n = len(motifs)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            same_class = motifs[i].chemical_class == motifs[j].chemical_class
            a, b = analyte_hits[i], analyte_hits[j]
            jac = len(a & b) / (len(a | b) + 1e-12)
            D[i, j] = 1.0 - (0.5 * same_class + 0.5 * jac)
    np.fill_diagonal(D, 0.0)
    return np.clip((D + D.T) / 2, 0, 2)


def partition_from_distance(D, K, ids, method="average"):
    """Agglomerative clustering into K groups; returns a list of member-id lists."""
    if K >= len(ids):
        return [[i] for i in ids]
    Z = linkage(squareform(D, checks=False), method=method)
    lab = fcluster(Z, K, criterion="maxclust")
    groups = {}
    for i, l in enumerate(lab):
        groups.setdefault(int(l), []).append(ids[i])
    return [sorted(v) for v in groups.values()]


def manual_partition(K, ids):
    """Nearest defined manual level at or above K, then merge the two closest-size
    groups until exactly K remain (deterministic)."""
    avail = sorted(MANUAL_LEVELS)
    src = min([k for k in avail if k >= K], default=max(avail))
    groups = [list(g) for g in MANUAL_LEVELS[src]]
    while len(groups) > K:
        groups.sort(key=len)
        groups[0] = sorted(groups[0] + groups[1])
        del groups[1]
    return [sorted(g) for g in groups]


# ── theme layer ─────────────────────────────────────────────────────────────
class ThemeLayer:
    """A K-theme chemical hierarchy over the biochemical MSS motifs."""

    def __init__(self, groups, motif_ids):
        self.groups = [sorted(g) for g in groups]
        self.motif_ids = list(motif_ids)
        self.K = len(self.groups)
        self.names = [name_group(g) for g in self.groups]
        # disambiguate duplicate names
        seen = {}
        for i, n in enumerate(self.names):
            if n in seen:
                seen[n] += 1
                self.names[i] = f"{n} {seen[n]}"
            else:
                seen[n] = 1
        idx = {m: i for i, m in enumerate(self.motif_ids)}
        self.T = np.zeros((len(self.motif_ids), self.K))
        for t, g in enumerate(self.groups):
            for m in g:
                if m in idx:
                    self.T[idx[m], t] = 1.0
        self.of_motif = {m: t for t, g in enumerate(self.groups) for m in g}

    def compose(self, mss_bio):
        """mss_bio: (n_motifs,) or (n, n_motifs) biochemical motif activations."""
        return np.atleast_2d(np.asarray(mss_bio, float)) @ self.T

    def as_dict(self):
        return {"K": self.K, "themes": [{"name": n, "motifs": g}
                                        for n, g in zip(self.names, self.groups)]}
