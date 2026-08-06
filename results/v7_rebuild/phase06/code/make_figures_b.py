#!/usr/bin/env python3
"""GAIRA V7 — Phase 06 figures, part B: radars, provenance, comparators, summary."""
from __future__ import annotations

import textwrap

import matplotlib.pyplot as plt
import numpy as np

INK, MUTED, LINE = "#1a1a1a", "#6b7280", "#9ca3af"
BLUE, GREEN, AMBER, RED, GREY = "#2563eb", "#15803d", "#b45309", "#b91c1c", "#4b5563"
PURPLE = "#7c3aed"


def radar(ax, e, conf, CL, SHORT, title, color=PURPLE, truth=None, pred=None):
    """One 16-spoke radar. Spoke weight encodes confidence; magnitude encodes evidence."""
    n = len(CL)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    v = e / (e.max() + 1e-12)
    cl = np.concatenate([v, v[:1]])
    aa = np.concatenate([ang, ang[:1]])
    ax.plot(aa, cl, color=color, lw=1.2, alpha=0.85)
    ax.fill(aa, cl, color=color, alpha=0.13)
    for k, (a, m, cf) in enumerate(zip(ang, v, conf)):
        w = 0.5 + 3.0 * float(cf)
        ax.plot([a, a], [0, m], color=color, lw=w, alpha=0.30 + 0.70 * float(cf),
                solid_capstyle="round")
        ax.plot([a], [m], "o", ms=2.0 + 4.0 * float(cf), color=color,
                alpha=0.30 + 0.70 * float(cf))
    if truth is not None:
        j = CL.index(truth)
        ax.plot([ang[j]], [v[j]], "o", ms=9, mfc="none", mec=GREEN, mew=1.6)
    if pred is not None and pred != truth:
        j = CL.index(pred)
        ax.plot([ang[j]], [v[j]], "x", ms=9, color=RED, mew=1.8)
    ax.set_xticks(ang)
    ax.set_xticklabels([SHORT[x].replace(" ", "\n") for x in CL], fontsize=5.2)
    ax.set_yticklabels([]); ax.set_ylim(0, 1.15)
    ax.set_title(title, fontsize=7.4, pad=11, color=INK)
    ax.grid(color=LINE, lw=0.35, alpha=0.6)


def _pick(c, CL):
    """Seven representative cases, chosen by rule rather than by eye."""
    conf, ok = c.P.max(axis=1), c.pred == c.cls
    mg = np.sort(c.E, axis=1)
    margin = mg[:, -1] - mg[:, -2]
    adj = {(a, b) for a, b in __import__("gaira.v7.chemistry.registry",
                                         fromlist=["x"]).ADJACENT}
    adj |= {(b, a) for a, b in adj}
    cases = []
    cases.append(("clear single class", int(np.argmax(np.where(ok, margin, -1)))))
    amb = [i for i in range(len(c.E)) if margin[i] < 0.05 and
           (c.cls[i], c.pred[i]) in adj or (ok[i] and margin[i] < 0.03)]
    cases.append(("ambiguous adjacent classes",
                  int(amb[0]) if amb else int(np.argmin(margin))))
    cases.append(("low reconstruction (EV)", int(np.argmin(c.ev))))
    wrong = np.where(~ok)[0]
    cases.append(("misclassified", int(wrong[int(np.argmax(conf[wrong]))]) if len(wrong) else 0))
    cases.append(("high-confidence correct", int(np.argmax(np.where(ok, conf, -1)))))
    cases.append(("low-confidence correct", int(np.argmin(np.where(ok, conf, 9)))))
    srcs = sorted(set(c.src.tolist()))
    alt = np.where(c.src == srcs[-1])[0]
    cases.append((f"cross-source ({srcs[-1]})", int(alt[len(alt) // 2]) if len(alt) else 0))
    return cases


def f16_radars(c, save, SHORT, CL):
    cases = _pick(c, CL)
    fig, axs = plt.subplots(2, 4, figsize=(13.2, 8.0), subplot_kw={"polar": True})
    for ax, (lab, i) in zip(axs.ravel(), cases):
        radar(ax, c.E[i], c.P[i] / (c.P[i].max() + 1e-12), CL, SHORT,
              f"{lab}\n{c.y[i]}\ntrue {SHORT[c.cls[i]]} · pred {SHORT[c.pred[i]]}\n"
              f"conf {c.P[i].max():.2f} · EV {c.ev[i]:.2f}",
              truth=c.cls[i], pred=c.pred[i])
    axs.ravel()[-1].axis("off")
    axs.ravel()[-1].set_frame_on(False)
    fig.text(0.78, 0.20, "green ring = TRUE class\nred cross = prediction, when wrong\n\n"
             "spoke thickness and marker size\nencode calibrated confidence\n\n"
             "radius is evidence RELATIVE to the\nstrongest axis — it is NOT a\n"
             "concentration and NOT a\ncomposition", fontsize=7.6, color=MUTED, va="top")
    fig.suptitle("Figure 16 · 16-axis Chemistry Evidence radars — seven representative cases",
                 x=0.035, ha="left", fontsize=11.5, weight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.subplots_adjust(hspace=0.55)
    save(fig, "F16_radar_examples")


def f17_bars(c, save, SHORT, CL):
    """The ordered-bar alternative: 16 spokes are cluttered, 16 sorted bars are not."""
    cases = _pick(c, CL)[:4]
    fig, axs = plt.subplots(1, 4, figsize=(13.0, 4.6))
    for ax, (lab, i) in zip(axs, cases):
        e = c.E[i]
        o = np.argsort(e)
        cols = [GREEN if CL[j] == c.cls[i] else (RED if CL[j] == c.pred[i] else GREY)
                for j in o]
        ax.barh(range(16), e[o], color=cols, alpha=0.9)
        for k, j in enumerate(o):
            ax.plot([0, e[j]], [k, k], color=cols[k],
                    lw=0.5 + 3.0 * float(c.P[i][j] / (c.P[i].max() + 1e-12)), alpha=0.5)
        ax.set_yticks(range(16)); ax.set_yticklabels([SHORT[CL[j]] for j in o], fontsize=6)
        ax.set_xlabel("evidence", fontsize=7.4)
        ax.set_title(f"{lab}\n{c.y[i]}", fontsize=7.6, loc="left")
        ax.tick_params(labelsize=6.5)
    fig.suptitle("Figure 17 · Ordered-bar alternative to the radar — same numbers, "
                 "no angular distortion\ngreen = true class · red = prediction when wrong · "
                 "bar-line weight encodes confidence",
                 x=0.035, ha="left", fontsize=11, weight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    save(fig, "F17_ordered_bars")


def f18_provenance(c, save, box, arrow, SHORT, CL):
    ex = c.prov["examples"][0]
    ch = ex["chains"][0]
    fig, ax = plt.subplots(figsize=(12.4, 5.6))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    box(ax, 0.01, 0.80, 0.17, 0.13,
        f"query spectrum\n{ex['spectrum_id']}\n{ex['molecule']}", "#f8fafc", GREY, 7.6, "bold")
    box(ax, 0.21, 0.80, 0.17, 0.13,
        f"chemistry class\n{SHORT[ch['class_id']]}\nevidence {ch['evidence']:.3f}",
        "#f5f3ff", PURPLE, 7.6, "bold")
    arrow(ax, (0.18, 0.865), (0.21, 0.865))
    ax.text(0.01, 0.755, f"aggregation {ch['aggregation']} · size correction "
            f"{ch['size_correction']} (weight {ch['size_weight']:.3f}) · "
            f"{ch['n_reference_molecules']} reference molecules of this class",
            fontsize=7.2, color=MUTED)
    for k, link in enumerate(ch["molecules"][:4]):
        yk = 0.60 - k * 0.155
        box(ax, 0.21, yk, 0.20, 0.125,
            f"{link['molecule']}\nsimilarity {link['similarity']:.3f}", "#eff6ff", BLUE, 7.2)
        arrow(ax, (0.295, 0.80), (0.31, yk + 0.125), LINE, 0.8)
        cs = link["supporting_csms"][:3]
        box(ax, 0.44, yk, 0.26, 0.125,
            "\n".join(f"{s['csm_id']}  {s['share_of_similarity']:.0%} of the similarity"
                      for s in cs) or "—", "#ecfdf5", GREEN, 6.8)
        arrow(ax, (0.41, yk + 0.062), (0.44, yk + 0.062), LINE, 0.8)
        lsms = sorted({l for s in cs for l in s["lsms"]})[:3]
        bands = sorted({b for s in cs for b in s["dominant_bands"]})[:6]
        box(ax, 0.73, yk, 0.26, 0.125,
            f"LSMs: {', '.join(lsms) if lsms else '—'}\n"
            f"bands: {', '.join(f'{b:.0f}' for b in bands)}", "#fffbeb", AMBER, 6.6)
        arrow(ax, (0.70, yk + 0.062), (0.73, yk + 0.062), LINE, 0.8)
    ax.text(0.01, 0.045,
            f"Exact decomposition: {c.prov['exact_decomposition']} · "
            f"{c.prov['n_chains_verified']} chains verified against the frozen registries · "
            f"{c.prov['n_broken']} broken. Every similarity is an inner product of the query's "
            "CSM activation with a named reference activation, so the listed CSMs sum back to "
            "the number shown.", fontsize=7.4, color=MUTED, style="italic")
    ax.set_title("Figure 18 · Provenance waterfall — chemistry class → molecules → CSMs → "
                 "LSMs → spectra", loc="left", fontsize=11.5, weight="bold", color=INK)
    save(fig, "F18_provenance_waterfall")


def f19_lowev(c, save, SHORT, CL):
    fig, axs = plt.subplots(1, 3, figsize=(12.2, 4.2),
                            gridspec_kw={"width_ratios": [1.0, 1.0, 1.2]})
    ax = axs[0]
    low = c.ev < 0.5
    ax.scatter(c.ev[~low], c.P.max(axis=1)[~low], s=10, color=GREY, alpha=0.5, label="EV ≥ 0.5")
    ax.scatter(c.ev[low], c.P.max(axis=1)[low], s=26, color=RED, alpha=0.85, label="EV < 0.5")
    ax.axvline(0.5, color=RED, ls="--", lw=1.0)
    ax.set_xlabel("CSM explained variance"); ax.set_ylabel("calibrated confidence")
    ax.legend(frameon=False, fontsize=7.4)
    ax.set_title(f"a · {int(low.sum())} low-EV spectra ({low.mean():.1%})", fontsize=9,
                 loc="left")
    ax = axs[1]
    f = c.fail
    grp = f.groupby("source").agg(n=("correct", "size"), acc=("correct", "mean"),
                                  ev=("explained_variance", "mean")).reset_index()
    ax.bar(range(len(grp)), grp.acc, color=BLUE, alpha=0.85)
    for k, r in grp.iterrows():
        ax.text(k, r.acc + 0.015, f"{r.acc:.2f}\nn={int(r.n)}", ha="center", fontsize=6.8)
    ax.set_xticks(range(len(grp)))
    ax.set_xticklabels([s.replace("_", "\n") for s in grp.source], fontsize=6.6)
    ax.set_ylim(0, 1.12); ax.set_ylabel("accuracy")
    ax.set_title("b · accuracy by source dataset", fontsize=9, loc="left")
    ax = axs[2]
    ax.axis("off")
    d = f[f.low_ev].sort_values("explained_variance").head(11)
    ax.text(0.0, 0.98, "the low-EV tail, by name", fontsize=8.6, weight="bold")
    ax.text(0.0, 0.90, f"{'molecule':26s}{'EV':>6s} {'class':>16s}  ok", fontsize=6.4,
            family="DejaVu Sans Mono", color=MUTED)
    yy = 0.83
    for _, r in d.iterrows():
        ax.text(0.0, yy, f"{r.canonical_id[:25]:26s}{r.explained_variance:6.3f} "
                f"{SHORT[r.true_class][:15]:>16s}  {'Y' if r.correct else 'N'}",
                fontsize=6.4, family="DejaVu Sans Mono",
                color=GREEN if r.correct else RED)
        yy -= 0.062
    ax.text(0.0, 0.06, f"accuracy on low-EV {f[f.low_ev].correct.mean():.3f} vs "
            f"{f[~f.low_ev].correct.mean():.3f} elsewhere", fontsize=7.4, color=INK,
            style="italic")
    fig.suptitle("Figure 19 · Low-EV cases and failure analysis", x=0.035, ha="left",
                 fontsize=11.5, weight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    save(fig, "F19_low_ev_failures")


def f20_layers(c, save, SHORT, CL):
    d = c.cmp.copy()
    order = ["raw_spectrum", "legacy_theme_bsv", "legacy_11_axis",
             "PHASE06_chemistry_evidence_16", "csm_49", "lsm_50", "phase05_hard_chemistry"]
    d = d.set_index("representation").reindex([o for o in order if o in
                                               set(d.representation)]).reset_index()
    fig, axs = plt.subplots(1, 2, figsize=(12.0, 4.6),
                            gridspec_kw={"width_ratios": [1.35, 1.0]})
    ax = axs[0]
    x = np.arange(len(d))
    cols = [PURPLE if "PHASE06" in r else (GREY if "legacy" in r or r == "raw_spectrum"
                                           else GREEN) for r in d.representation]
    ax.bar(x - 0.2, d.top1, 0.38, color=cols, alpha=0.95, label="top-1")
    ax.bar(x + 0.2, d.macro_f1, 0.38, color=cols, alpha=0.5, label="macro-F1")
    for k, r in d.iterrows():
        ax.text(k - 0.2, r.top1 + 0.012, f"{r.top1:.3f}", ha="center", fontsize=7,
                weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r}\n{int(dd)}-d" for r, dd in zip(d.representation, d.dim)],
                       rotation=28, ha="right", fontsize=6.8)
    ax.set_ylim(0, 1.06); ax.set_ylabel("chemistry class, unseen molecule")
    ax.legend(frameon=False, fontsize=7.6)
    ax.set_title("a · every semantic layer on identical outer folds", fontsize=9, loc="left")
    ax = axs[1]
    s = c.sem
    cols2 = [GREEN if b else AMBER for b in s.accuracy_comparable_to_curated]
    ax.bar(range(len(s)), s.chance_adjusted_top1, color=cols2, alpha=0.9)
    ax.set_xticks(range(len(s)))
    ax.set_xticklabels([f"{r}\nK={int(k)}" for r, k in zip(s.semantic_layer, s.n_classes)],
                       fontsize=7)
    for k, v in enumerate(s.chance_adjusted_top1):
        ax.text(k, v + 0.012, f"{v:.3f}", ha="center", fontsize=8, weight="bold")
    ax.set_ylim(0, 1.06); ax.set_ylabel("chance-adjusted top-1")
    ax.set_title("b · semantic comparators", fontsize=9, loc="left")
    ax.text(0.5, -0.42, "amber = NOT comparable: an unsupervised grouping defined by "
            "CSM-space\nproximity, predicted by CSM-space proximity, is near self-prediction",
            transform=ax.transAxes, ha="center", fontsize=6.8, color=AMBER)
    fig.suptitle("Figure 20 · Comparison with every legacy semantic layer, and with "
                 "unsupervised grouping", x=0.035, ha="left", fontsize=11.5, weight="bold",
                 color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    save(fig, "F20_layer_comparison")


def f21_ontology(c, save, SHORT, CL):
    fig, axs = plt.subplots(1, 2, figsize=(11.0, 4.4))
    import json
    from pathlib import Path
    ag = json.loads((Path(c.prov and "" or "") or
                     __import__("pathlib").Path(".")).as_posix() and
                    (__import__("pathlib").Path(
                        c.__dict__.get("_agree_path", "")) if False else "{}") or "{}") \
        if False else c.agree
    ax = axs[0]
    ks = ["adjusted_rand_curated_vs_unsupervised", "adjusted_mutual_info_curated_vs_unsupervised",
          "adjusted_rand_broad_vs_unsupervised"]
    lbl = ["ARI\ncurated 16 vs\nunsupervised 16", "AMI\ncurated 16 vs\nunsupervised 16",
           "ARI\nbroad 6 vs\nunsupervised 16"]
    ax.bar(range(3), [ag[k] for k in ks], color=[PURPLE, BLUE, GREY], alpha=0.9)
    for k in range(3):
        ax.text(k, ag[ks[k]] + 0.015, f"{ag[ks[k]]:.3f}", ha="center", fontsize=9,
                weight="bold")
    ax.set_xticks(range(3)); ax.set_xticklabels(lbl, fontsize=7)
    ax.set_ylim(0, 1.05); ax.set_ylabel("agreement (1 = identical, 0 = chance)")
    ax.set_title("a · is the curated ontology recoverable from the spectra alone?",
                 fontsize=9, loc="left")
    ax = axs[1]
    ax.axis("off")
    txt = ("ARI 0.595 and AMI 0.725 say the curated 16-class ontology is SUBSTANTIALLY but not "
           "fully recoverable from CSM activations without labels.\n\n"
           "What the curated layer adds is what the ~40% disagreement contains: distinctions a "
           "clustering does not make because they are chemical rather than spectral — "
           "acylglycerol vs fatty acid, purine vs pyrimidine, polysaccharide vs "
           "mono/oligosaccharide.\n\n"
           "What it costs is nothing in accuracy terms that can be fairly measured: the "
           "unsupervised comparator's apparently higher score is a structural artefact of "
           "predicting a proximity-defined grouping using proximity.\n\n"
           "The curated ontology is retained. It is nameable, it is frozen, it is the label "
           "space of the frozen success criteria, and it is the only one of the three that a "
           "spectroscopist can argue with.")
    yy = 0.96
    for line in txt.split("\n"):
        for w in textwrap.wrap(line, 62) or [""]:
            ax.text(0.0, yy, w, fontsize=7.4, color=INK)
            yy -= 0.052
    fig.suptitle("Figure 21 · Curated ontology versus unsupervised semantic comparator",
                 x=0.035, ha="left", fontsize=11.5, weight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    save(fig, "F21_ontology_comparator")


def f22_summary(c, save, box, SHORT, CL):
    fig = plt.figure(figsize=(11.6, 7.8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.05], hspace=0.42, wspace=0.36)
    s = c.s
    ax = fig.add_subplot(gs[0, 0])
    d = c.cmp.set_index("representation")
    pts = [("raw", "raw_spectrum", GREY), ("theme/BSV", "legacy_theme_bsv", GREY),
           ("11-axis", "legacy_11_axis", GREY), ("CSM 49", "csm_49", GREEN),
           ("chem. evidence 16", "PHASE06_chemistry_evidence_16", PURPLE)]
    for lab, key, col in pts:
        if key not in d.index:
            continue
        ax.plot(d.loc[key, "dim"], d.loc[key, "top1"], "o", ms=11, color=col)
        ax.annotate(lab, (d.loc[key, "dim"], d.loc[key, "top1"]),
                    textcoords="offset points", xytext=(8, -3), fontsize=7.4)
    ax.set_xscale("log"); ax.set_xlabel("dimension"); ax.set_ylabel("class top-1, unseen molecule")
    ax.set_ylim(0.4, 0.95)
    ax.set_title("a · accuracy against dimension", fontsize=9, loc="left")
    ax = fig.add_subplot(gs[0, 1])
    p = c.pc.set_index("class_id").reindex(CL).reset_index().sort_values("n")
    ax.scatter(p.n, p.f1, s=40, color=PURPLE, alpha=0.85)
    for _, r in p.iterrows():
        if r.f1 < 0.75 or r.n > 60:
            ax.annotate(SHORT[r.class_id], (r.n, r.f1), textcoords="offset points",
                        xytext=(5, -2), fontsize=6.2)
    ax.set_xlabel("class size (spectra)"); ax.set_ylabel("F1")
    ax.set_title("b · F1 tracks class size", fontsize=9, loc="left")
    ax = fig.add_subplot(gs[0, 2])
    ax.axis("off")
    rows = [("fine top-1", f"{s['performance']['top1']['value']:.3f}"),
            ("fine top-3", f"{s['performance']['top3']['value']:.3f}"),
            ("macro F1", f"{s['performance']['macro_f1']['value']:.3f}"),
            ("balanced accuracy", f"{s['performance']['balanced_accuracy']['value']:.3f}"),
            ("ECE / classwise", f"{s['calibration']['ece']:.3f} / "
                                f"{s['calibration']['classwise_ece']:.3f}"),
            ("replicate consistency", f"{s['soft_evidence']['replicate_consistency']:.3f}"),
            ("robustness retention", f"{[r for r in s['robustness'] if r['representation'] == 'chemistry_evidence_16'][0]['top1_retention']:.3f}"),
            ("novelty AUROC (mean)", f"{s['novelty']['mean_auroc']:.3f}"),
            ("broken provenance", f"{s['provenance']['broken']}"),
            ("effective rank", f"{s['soft_evidence']['effective_rank']:.2f} of 16")]
    ax.text(0.0, 1.0, "headline numbers", fontsize=9, weight="bold", va="top")
    for i, (k, v) in enumerate(rows):
        ax.text(0.0, 0.88 - i * 0.088, k, fontsize=7.6, color=INK)
        ax.text(1.0, 0.88 - i * 0.088, v, fontsize=7.6, color=GREEN, ha="right",
                family="DejaVu Sans Mono", weight="bold")
    ax = fig.add_subplot(gs[1, :])
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    g = c.gates
    ncol, w = 5, 0.185
    nrow = int(np.ceil(len(g) / ncol))
    h = min(0.21, (0.92 - 0.08) / nrow - 0.030)
    for i, (_, row) in enumerate(g.iterrows()):
        x = 0.012 + (i % ncol) * 0.197
        yy = 0.88 - (i // ncol) * (h + 0.030)
        ok = row.status == "PASS"
        box(ax, x, yy, w, h, "\n".join(textwrap.wrap(row.gate, 30)),
            "#ecfdf5" if ok else "#fef2f2", GREEN if ok else RED, 6.0)
    ax.text(0.012, 0.02, f"{int((g.status == 'PASS').sum())} of {len(g)} gates pass · "
            f"selected model {s['selected_model']['candidate']} · calibration "
            f"{s['calibration']['method']} · Raman only, no cross-modality data",
            fontsize=7.8, color=MUTED)
    fig.suptitle("Figure 22 · Phase 06 architecture summary", x=0.035, ha="left",
                 fontsize=11.5, weight="bold", color=INK)
    save(fig, "F22_summary")


def main(c, save, box, arrow, SHORT, CL):
    import json
    from pathlib import Path
    c.agree = json.loads((Path(__file__).resolve().parents[1] / "artifacts" /
                          "semantic_agreement_v1.json").read_text())
    f16_radars(c, save, SHORT, CL)
    f17_bars(c, save, SHORT, CL)
    f18_provenance(c, save, box, arrow, SHORT, CL)
    f19_lowev(c, save, SHORT, CL)
    f20_layers(c, save, SHORT, CL)
    f21_ontology(c, save, SHORT, CL)
    f22_summary(c, save, box, SHORT, CL)
