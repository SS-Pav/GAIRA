"""GAIRA V7 — validation of the Atlas Component Substructure layer.

The central scientific question Phase 01 must answer is not "did the code run" but:

    Does the motif layer resolve biochemical ambiguity that currently sits inside atlas
    components — does a component separate into chemically meaningful motifs, or is it
    genuinely one substructure?

Two things make that answerable rather than rhetorical:

  * chemistry is used ONLY here, in evaluation. No class label touched band definition,
    profile construction, linkage choice or cut choice, so "motifs align with chemistry"
    cannot be circular.
  * every alignment claim is measured against a **permutation null** using a
    chance-corrected statistic (adjusted mutual information). Raw purity rises simply by
    cutting a set into more pieces; AMI does not.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

N_PERM = 1000
SEED = 0


def chemical_alignment(results: list[dict], fine_of: dict, broad_of: dict,
                       n_perm: int = N_PERM, seed: int = SEED) -> pd.DataFrame:
    """Per component: do motif memberships align with chemistry beyond chance?"""
    rng = np.random.default_rng(seed)
    rows = []
    for r in results:
        if r["status"] == "NOT_ANALYSABLE":
            continue
        kept = [m for m in r["motifs"] if m.retained]
        if len(kept) < 2:
            rows.append({"component": r["component"], "status": r["status"],
                         "n_motifs": len(kept), "n_analytes": 0, "ami_fine": None,
                         "ari_fine": None, "ami_broad": None, "null_ami_mean": None,
                         "null_ami_p95": None, "p_permutation": None,
                         "significant": False, "n_distinct_classes": 0,
                         "n_distinct_dominant": 0})
            continue
        labels, fine, broad = [], [], []
        for j, m in enumerate(kept):
            for a in m.analytes:
                labels.append(j)
                fine.append(fine_of.get(a, ""))
                broad.append(broad_of.get(a, ""))
        labels = np.asarray(labels)
        ami = float(adjusted_mutual_info_score(fine, labels))
        ari = float(adjusted_rand_score(fine, labels))
        amib = float(adjusted_mutual_info_score(broad, labels))
        null = np.array([adjusted_mutual_info_score(list(rng.permutation(fine)), labels)
                         for _ in range(n_perm)], float)
        p = float((np.sum(null >= ami) + 1) / (n_perm + 1))
        rows.append({
            "component": r["component"], "status": r["status"], "n_motifs": len(kept),
            "n_analytes": int(len(labels)),
            "ami_fine": round(ami, 4), "ari_fine": round(ari, 4),
            "ami_broad": round(amib, 4),
            "null_ami_mean": round(float(null.mean()), 4),
            "null_ami_p95": round(float(np.percentile(null, 95)), 4),
            "p_permutation": round(p, 5), "significant": bool(p < 0.05),
            "n_distinct_classes": int(len(set(fine))),
            "n_distinct_dominant": int(len({m.dominant_class for m in kept})),
        })
    return pd.DataFrame(rows).sort_values("component").reset_index(drop=True)


def ambiguity_resolution(results: list[dict], fine_of: dict) -> pd.DataFrame:
    """The headline table: did a component that mixed chemistries separate them?

    `component_dominant_share` is the purity of the component taken whole;
    `weighted_motif_purity` is the size-weighted purity of its retained motifs. The gap is
    the resolution the motif layer actually delivers for that component.
    """
    rows = []
    for r in results:
        if r["status"] == "NOT_ANALYSABLE":
            continue
        parts = r.get("participant_ids", [])
        if not parts:
            continue
        cls = pd.Series([fine_of.get(a, "") for a in parts])
        vc = cls.value_counts(normalize=True)
        kept = [m for m in r["motifs"] if m.retained]
        best = max((m.purity for m in kept), default=float("nan"))
        denom = sum(m.n_analytes for m in kept)
        wmean = (sum(m.purity * m.n_analytes for m in kept) / denom) if denom else float("nan")
        rows.append({
            "component": r["component"], "status": r["status"],
            "n_participants": len(parts),
            "n_classes_in_component": int(cls.nunique()),
            "component_dominant_class": vc.index[0] if len(vc) else "",
            "component_dominant_share": round(float(vc.iloc[0]), 4) if len(vc) else None,
            "n_retained_motifs": len(kept),
            "best_motif_purity": round(float(best), 4) if kept else None,
            "weighted_motif_purity": round(float(wmean), 4) if kept else None,
            "purity_gain": (round(float(wmean - vc.iloc[0]), 4)
                            if kept and len(vc) else None),
            "distinct_motif_classes": ";".join(sorted({m.dominant_class for m in kept})),
        })
    return pd.DataFrame(rows).sort_values("component").reset_index(drop=True)


def purity_null(results: list[dict], fine_of: dict, n_draws: int = 500,
                seed: int = SEED) -> pd.DataFrame:
    """Size-matched random-motif null for purity.

    Purity rises mechanically as a set is cut into more pieces, so a raw purity gain over
    the whole component is not evidence on its own. This shuffles which analytes fall in
    which motif while holding the motif SIZE PROFILE fixed, giving the purity a random
    partition of the same shape would reach. The gain beyond that is the part that is about
    chemistry — the same logic as Phase 00's size-matched random ontologies.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for r in results:
        kept = [m for m in r["motifs"] if m.retained]
        if len(kept) < 2:
            continue
        members = [a for m in kept for a in m.analytes]
        cls = np.array([fine_of.get(a, "") for a in members])
        sizes = [m.n_analytes for m in kept]
        obs = sum(m.purity * m.n_analytes for m in kept) / sum(sizes)

        draws = []
        for _ in range(n_draws):
            perm = rng.permutation(len(cls))
            pos, tot = 0, 0.0
            for sz in sizes:
                grp = cls[perm[pos:pos + sz]]
                vals, cnt = np.unique(grp, return_counts=True)
                tot += cnt.max()
                pos += sz
            draws.append(tot / len(cls))
        draws = np.array(draws, float)
        p = float((np.sum(draws >= obs) + 1) / (n_draws + 1))
        rows.append({
            "component": r["component"], "n_motifs": len(kept),
            "n_analytes": int(len(cls)),
            "observed_weighted_purity": round(float(obs), 4),
            "null_purity_mean": round(float(draws.mean()), 4),
            "null_purity_p95": round(float(np.percentile(draws, 95)), 4),
            "gain_beyond_mechanical": round(float(obs - draws.mean()), 4),
            "p_permutation": round(p, 5),
            "significant": bool(p < 0.05),
        })
    return pd.DataFrame(rows).sort_values("component").reset_index(drop=True)


def redundancy_matrix(motifs) -> pd.DataFrame:
    """Pairwise cosine between retained motif spectra — the spectral-overlap check."""
    if not motifs:
        return pd.DataFrame()
    S = np.vstack([m.normalised() for m in motifs])
    ids = [m.motif_id for m in motifs]
    return pd.DataFrame(np.round(S @ S.T, 4), index=ids, columns=ids)


def redundancy_summary(motifs) -> dict:
    if len(motifs) < 2:
        return {"n_motifs": len(motifs), "max_offdiag_cosine": 0.0,
                "mean_offdiag_cosine": 0.0, "n_pairs_above_0.9": 0,
                "n_pairs_above_0.95": 0, "n_cross_component_pairs_above_0.9": 0}
    S = np.vstack([m.normalised() for m in motifs])
    C = S @ S.T
    iu = np.triu_indices(len(motifs), 1)
    off = C[iu]
    same = np.array([motifs[i].parent_component == motifs[j].parent_component
                     for i, j in zip(*iu)])
    return {
        "n_motifs": len(motifs),
        "max_offdiag_cosine": round(float(off.max()), 4),
        "mean_offdiag_cosine": round(float(off.mean()), 4),
        "n_pairs_above_0.9": int((off > 0.9).sum()),
        "n_pairs_above_0.95": int((off > 0.95).sum()),
        "n_cross_component_pairs_above_0.9": int(((off > 0.9) & (~same)).sum()),
    }


def coverage_report(registry, analyte_ids: list[str], spectra_of: dict) -> dict:
    """How much of the corpus the retained motif layer actually touches."""
    kept = registry.retained
    covered = {a for m in kept for a in m.analytes}
    n_spec_cov = sum(spectra_of.get(a, 0) for a in covered)
    n_spec_tot = sum(spectra_of.get(a, 0) for a in analyte_ids)
    per = pd.Series([len(registry.by_analyte(a)) for a in analyte_ids], index=analyte_ids)
    return {
        "n_analytes_total": len(analyte_ids),
        "n_analytes_covered": len(covered),
        "analyte_coverage": round(len(covered) / max(len(analyte_ids), 1), 4),
        "n_spectra_covered": int(n_spec_cov),
        "spectrum_coverage": round(n_spec_cov / max(n_spec_tot, 1), 4),
        "motifs_per_analyte_mean": round(float(per.mean()), 3),
        "motifs_per_analyte_median": float(per.median()),
        "n_analytes_with_no_motif": int((per == 0).sum()),
        "uncovered_analytes": sorted(set(analyte_ids) - covered),
    }


def reproducibility(discover_fn, subsets: dict[str, dict],
                    reference_results: list[dict]) -> pd.DataFrame:
    """Re-run discovery on data subsets and measure agreement with the full-corpus run.

    Agreement is the adjusted Rand index between motif memberships restricted to analytes
    present in both runs — chance-corrected, so a component that trivially yields one motif
    does not score as perfect reproducibility.
    """
    ref = {}
    for r in reference_results:
        kept = [m for m in r["motifs"] if m.retained]
        if len(kept) < 2:
            continue
        ref[r["component"]] = {a: j for j, m in enumerate(kept) for a in m.analytes}

    rows = []
    for name, kwargs in subsets.items():
        got = discover_fn(**kwargs)
        for r in got:
            k = r["component"]
            if k not in ref:
                continue
            kept = [m for m in r["motifs"] if m.retained]
            if len(kept) < 2:
                rows.append({"subset": name, "component": k, "n_shared": 0, "ari": None,
                             "note": "subset run gave <2 motifs"})
                continue
            sub = {a: j for j, m in enumerate(kept) for a in m.analytes}
            shared = sorted(set(ref[k]) & set(sub))
            if len(shared) < 4:
                rows.append({"subset": name, "component": k, "n_shared": len(shared),
                             "ari": None, "note": "too few shared analytes"})
                continue
            rows.append({"subset": name, "component": k, "n_shared": len(shared),
                         "ari": round(float(adjusted_rand_score(
                             [ref[k][a] for a in shared], [sub[a] for a in shared])), 4),
                         "note": ""})
    return pd.DataFrame(rows)


def motif_signature(results: list[dict]) -> str:
    """Content signature of a discovery run — used for the determinism check."""
    kept = sorted([m for r in results for m in r["motifs"] if m.retained],
                  key=lambda m: m.motif_id)
    h = hashlib.sha256()
    for m in kept:
        h.update(m.motif_id.encode("utf-8"))
        h.update(np.ascontiguousarray(m.spectrum, dtype=np.float64).tobytes())
    return h.hexdigest()[:32]


def determinism_check(discover_fn, kwargs: dict, n_runs: int = 3) -> dict:
    """Run discovery repeatedly and require byte-identical motif spectra."""
    sigs = [motif_signature(discover_fn(**kwargs)) for _ in range(n_runs)]
    return {"n_runs": n_runs, "signatures": sigs, "identical": len(set(sigs)) == 1}
