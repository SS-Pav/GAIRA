"""GAIRA V7 — Phase 10: the single report generator.

Called identically from the Python SDK, the CLI, the FastAPI service, the MCP tool server and
the Streamlit client. There is no second implementation anywhere, and a static test enforces it.

Deterministic: the same `InferenceResult` produces the same bytes, apart from the generation
timestamp, which is isolated in the metadata block so a caller can diff two reports meaningfully.
No language model is involved in any output.
"""
from __future__ import annotations

import base64
import html
import io
import json
import textwrap
from datetime import datetime, timezone

from gaira.v7.contracts import InferenceResult, ReportMetadata

from . import figures as FIG

SECTIONS = (
    "Sample metadata", "Engine metadata", "Input spectrum", "Preprocessing",
    "CSM representation", "Reconstruction and residual", "Grounded Evidence Retrieval",
    "Chemistry Evidence", "Confidence", "Scientific audit", "Provenance",
    "Interpretation summary", "Scope and limitations",
)

SCOPE_NOTES = (
    "Chemistry Evidence is RELATIVE. It is not a concentration, not an abundance, and not a "
    "mixture fraction.",
    "Retrieved molecules are reference analogues, not definitive identifications. Validated "
    "molecule top-1 is 0.6053.",
    "The engine provides no validated open-set detection. It cannot determine that the true "
    "molecule is absent from its 154-molecule bank.",
    "Pure Raman reference spectra only. SERS, serum, plasma, EV, bacteria and tissue behaviour "
    "is unmeasured in V7.",
    "The 16 chemistry classes are a curated cut through a continuum, not a discovered structure.",
    "Interpretation text is template-driven and deterministic. No language model is involved.",
)


def _meta(result: InferenceResult) -> ReportMetadata:
    return ReportMetadata(
        generated_utc=datetime.now(timezone.utc).isoformat(),
        sample_id=result.request_metadata.sample_id,
        sample_name=result.request_metadata.source_name,
        engine_version=result.engine.engine_version,
        atlas_fingerprint=result.engine.atlas_fingerprint)


def render_json(result: InferenceResult, title: str | None = None) -> str:
    return json.dumps({
        "schema": "gaira_v7_report_v1",
        "title": title or "GAIRA V7 Inference Report",
        "metadata": _meta(result).model_dump(mode="json"),
        "sections": list(SECTIONS),
        "result": result.model_dump(mode="json"),
        "scope_and_limitations": list(SCOPE_NOTES),
    }, indent=2, sort_keys=False)


def render_html(result: InferenceResult, title: str | None = None) -> str:
    m = _meta(result)
    e, chem, conf = result.engine, result.chemistry, result.confidence
    esc = html.escape

    def img(png: bytes, alt: str) -> str:
        return (f'<img alt="{esc(alt)}" style="width:100%;max-width:960px;height:auto" '
                f'src="data:image/png;base64,{base64.b64encode(png).decode()}">')

    rows = "".join(
        f"<tr><td>{h.rank}</td><td><b>{esc(h.molecule)}</b></td>"
        f"<td>{esc(h.chemistry_class)}</td><td>{h.similarity:.4f}</td>"
        f"<td>{', '.join(esc(c.csm_id) for c in h.supporting_csms[:3])}</td></tr>"
        for h in result.retrieval.top)
    axes = "".join(
        f"<tr><td>{a.rank}</td><td>{esc(a.axis.replace('_', ' '))}</td>"
        f"<td>{a.evidence:.4f}</td><td>{a.share:.1%}</td>"
        f"<td>{a.calibrated_probability:.3f}</td></tr>" for a in chem.top)
    diags = "".join(
        f'<li><code>{esc(d.code)}</code> <b>{d.severity.value}</b> — {esc(d.message)}</li>'
        for d in result.diagnostics) or "<li>none</li>"
    limits = "".join(f"<li>{esc(s)}</li>" for s in SCOPE_NOTES)

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>{esc(title or 'GAIRA V7 Inference Report')}</title>
<style>
 body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:1000px;
       margin:2rem auto;padding:0 1.2rem;color:#1a1a1a;line-height:1.55}}
 h1{{font-size:1.75rem;margin-bottom:.2rem}} h2{{font-size:1.05rem;margin-top:2rem;
   border-bottom:1px solid #d1d5db;padding-bottom:.3rem}}
 .sub{{color:#6b7280;margin-top:0}} table{{border-collapse:collapse;width:100%;font-size:.86rem}}
 th,td{{text-align:left;padding:.32rem .5rem;border-bottom:1px solid #eef0f3}}
 th{{color:#6b7280;font-weight:600}} code{{background:#f3f4f6;padding:.05rem .3rem;
   border-radius:3px;font-size:.82rem}}
 .note{{border-left:3px solid #b45309;padding:.5rem .9rem;background:#fffbeb;font-size:.88rem}}
 .kv td:first-child{{color:#6b7280;width:16rem}}
</style></head><body>
<h1>{esc(title or 'GAIRA V7 Inference Report')}</h1>
<p class="sub">Grounded Raman Biochemical Inference &middot; generated {esc(m.generated_utc[:19])} UTC
&middot; result digest <code>{esc(result.result_digest)}</code></p>

<h2>1&ndash;2. Sample and engine metadata</h2>
<table class="kv">
<tr><td>sample id</td><td>{esc(str(m.sample_id or '—'))}</td></tr>
<tr><td>sample name</td><td>{esc(str(m.sample_name or '—'))}</td></tr>
<tr><td>modality</td><td>{esc(result.request_metadata.modality.value)}</td></tr>
<tr><td>sample type</td><td>{esc(result.request_metadata.sample_type.value)}</td></tr>
<tr><td>excitation</td><td>{esc(str(result.request_metadata.excitation_nm or '—'))} nm</td></tr>
<tr><td>engine</td><td>{esc(e.engine_version)} (GAIRA {esc(e.gaira_version)})</td></tr>
<tr><td>atlas fingerprint</td><td><code>{esc(e.atlas_fingerprint)}</code></td></tr>
<tr><td>frozen fingerprints</td><td><code>{esc(json.dumps(e.fingerprints))}</code></td></tr>
<tr><td>atlas</td><td>{e.n_lsms} LSMs &middot; {e.n_csms} CSMs &middot; {e.n_molecules}
  molecules &middot; {e.n_chemistry_axes} chemistry axes</td></tr>
</table>

<h2>3&ndash;4. Input spectrum and preprocessing</h2>
{img(FIG.spectrum_panel(result), 'preprocessed spectrum')}
<table class="kv">
<tr><td>input points</td><td>{result.preprocessing.n_input_points}</td></tr>
<tr><td>input range</td><td>{result.preprocessing.input_range[0]:.0f}&ndash;{result.preprocessing.input_range[1]:.0f} cm<sup>-1</sup></td></tr>
<tr><td>resampled to</td><td>{esc(result.preprocessing.resampled_to)}</td></tr>
<tr><td>baseline</td><td>{esc(result.preprocessing.baseline_method)}</td></tr>
<tr><td>smoothing</td><td>{esc(result.preprocessing.smoothing)}</td></tr>
<tr><td>normalisation</td><td>{esc(result.preprocessing.normalisation)}</td></tr>
<tr><td>grid coverage</td><td>{result.preprocessing.grid_coverage:.1%}</td></tr>
<tr><td>peaks detected</td><td>{result.preprocessing.n_peaks}</td></tr>
<tr><td>SNR estimate</td><td>{result.preprocessing.snr_estimate:.1f}</td></tr>
</table>

<h2>5&ndash;6. CSM representation, reconstruction and residual</h2>
{img(FIG.csm_panel(result), 'CSM activation')}
{img(FIG.reconstruction_panel(result), 'reconstruction and residual')}

<h2>7. Grounded Evidence Retrieval</h2>
<p class="note">Candidates are retrieved reference analogues, not definitive molecular
identifications.</p>
{img(FIG.retrieval_panel(result), 'retrieval')}
<table><tr><th>rank</th><th>molecule</th><th>chemistry class</th><th>similarity</th>
<th>supporting CSMs</th></tr>{rows}</table>

<h2>8. Chemistry Evidence</h2>
{img(FIG.chemistry_panel(result), 'chemistry evidence')}
<table><tr><th>rank</th><th>axis</th><th>evidence</th><th>share</th>
<th>calibrated confidence</th></tr>{axes}</table>

<h2>9&ndash;10. Confidence and scientific audit</h2>
<table class="kv">
<tr><td>overall confidence</td><td>{conf.overall:.4f}</td></tr>
<tr><td>evidence coverage</td><td>{conf.evidence_coverage:.4f}</td></tr>
<tr><td>CSM explained variance</td><td>{conf.reconstruction_explained_variance:.4f}</td></tr>
<tr><td>retrieval margin</td><td>{conf.retrieval_margin:.4f}</td></tr>
<tr><td>chemistry confidence</td><td>{conf.chemistry_confidence:.4f}</td></tr>
<tr><td>unknown warning</td><td>{conf.unknown_warning}</td></tr>
<tr><td>outlier warning</td><td>{conf.outlier_warning}</td></tr>
<tr><td>all scores reconcile</td><td>{result.audit.all_scores_reconcile if result.audit else '—'}</td></tr>
</table>
<p><b>Diagnostics</b></p><ul>{diags}</ul>
<p class="note">{esc(result.audit.open_set_limitation) if result.audit else ''}</p>

<h2>11. Provenance</h2>
<p>spectrum &rarr; {len(result.provenance.lsm_layer) if result.provenance else 0} LSMs &rarr;
{len(result.provenance.csm_layer) if result.provenance else 0} CSMs &rarr;
{len(result.provenance.chemistry_layer) if result.provenance else 0} chemistry axes &rarr;
{len(result.provenance.molecule_layer) if result.provenance else 0} molecules &middot;
atlas <code>{esc(result.provenance.atlas_fingerprint) if result.provenance else '—'}</code></p>

<h2>12. Interpretation summary</h2>
<p>{esc(result.interpretation)}</p>

<h2>13. Scope and limitations</h2><ul>{limits}</ul>
</body></html>"""


def render_pdf(result: InferenceResult, title: str | None = None) -> bytes:
    """A five-page PDF assembled with matplotlib, so no extra dependency is introduced."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    m, e, chem, conf = _meta(result), result.engine, result.chemistry, result.confidence
    INK, MUTED, RULE, WARM = FIG.INK, FIG.MUTED, FIG.RULE, FIG.WARM
    PAGE = (8.27, 11.69)
    buf = io.BytesIO()

    def text_page(pdf, heading, blocks):
        fig = plt.figure(figsize=PAGE); fig.patch.set_facecolor("white")
        ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.text(0.08, 0.94, heading, fontsize=14, weight="bold", color=INK, va="top")
        ax.plot([0.08, 0.92], [0.918, 0.918], color=RULE, lw=1.0)
        y = 0.895
        for kind, payload in blocks:
            if kind == "h":
                y -= 0.012
                ax.text(0.08, y, payload, fontsize=10, weight="bold", color=INK, va="top")
                y -= 0.024
            elif kind == "kv":
                for k, v in payload:
                    ax.text(0.08, y, k, fontsize=8.4, color=MUTED, va="top")
                    ax.text(0.42, y, str(v), fontsize=8.4, color=INK, va="top")
                    y -= 0.0185
                y -= 0.008
            elif kind == "p":
                for line in textwrap.wrap(payload, 96):
                    ax.text(0.08, y, line, fontsize=8.6, color=INK, va="top"); y -= 0.0175
                y -= 0.008
            elif kind == "note":
                for line in textwrap.wrap(payload, 92):
                    ax.text(0.09, y, line, fontsize=8.2, color=WARM, va="top",
                            style="italic"); y -= 0.0170
                y -= 0.008
            elif kind == "bullets":
                for b in payload:
                    lines = textwrap.wrap(b, 92)
                    for i, line in enumerate(lines):
                        if i == 0:
                            ax.text(0.08, y, "—", fontsize=8.6, color=MUTED, va="top")
                        ax.text(0.10, y, line, fontsize=8.6, color=INK, va="top"); y -= 0.0175
                    y -= 0.004
                y -= 0.006
            elif kind == "table":
                header, trows, cols = payload
                for c, h in zip(cols, header):
                    ax.text(0.08 + c, y, h, fontsize=8.0, color=MUTED, weight="bold", va="top")
                y -= 0.020
                ax.plot([0.08, 0.92], [y + 0.008, y + 0.008], color=RULE, lw=0.6)
                for r in trows:
                    for c, cell in zip(cols, r):
                        ax.text(0.08 + c, y, str(cell), fontsize=8.0, color=INK, va="top")
                    y -= 0.0175
                y -= 0.010
        ax.text(0.08, 0.035, f"GAIRA V7 · atlas {e.atlas_fingerprint[:12]}… · digest "
                f"{result.result_digest[:12]}…", fontsize=7, color=MUTED)
        pdf.savefig(fig); plt.close(fig)

    def image_page(pdf, heading, pngs, note=None):
        fig = plt.figure(figsize=PAGE); fig.patch.set_facecolor("white")
        head = fig.add_axes([0, 0, 1, 1]); head.axis("off")
        head.set_xlim(0, 1); head.set_ylim(0, 1)
        head.text(0.08, 0.94, heading, fontsize=14, weight="bold", color=INK, va="top")
        head.plot([0.08, 0.92], [0.918, 0.918], color=RULE, lw=1.0)
        if note:
            yy = 0.078
            for line in textwrap.wrap(note, 96):
                head.text(0.08, yy, line, fontsize=7.8, color=WARM, style="italic"); yy -= 0.016
        # Stack from the top with a fixed gap rather than distributing over the column: a
        # single wide panel centred in a portrait page reads as a mistake.
        y = 0.90
        for png in pngs:
            img = mpimg.imread(io.BytesIO(png))
            ih, iw = img.shape[:2]
            w = 0.86
            h = w * PAGE[0] * ih / (iw * PAGE[1])
            if y - h < 0.12:
                h = max(y - 0.12, 0.05)
                w = h * PAGE[1] * iw / (ih * PAGE[0])
            a = fig.add_axes([(1 - w) / 2, y - h, w, h])
            a.imshow(img); a.axis("off")
            y -= h + 0.028
        head.text(0.08, 0.035, f"GAIRA V7 · digest {result.result_digest[:12]}…",
                  fontsize=7, color=MUTED)
        pdf.savefig(fig); plt.close(fig)

    with PdfPages(buf) as pdf:
        text_page(pdf, title or "GAIRA V7 Inference Report", [
            ("p", "Grounded Raman Biochemical Inference. A Raman spectrum projected into a "
                  "frozen biochemical motif atlas, with grounded reference evidence and an "
                  "interpretable 16-axis Chemistry Evidence profile."),
            ("h", "1. Sample metadata"),
            ("kv", [("sample id", m.sample_id or "—"), ("sample name", m.sample_name or "—"),
                    ("modality", result.request_metadata.modality.value),
                    ("sample type", result.request_metadata.sample_type.value),
                    ("excitation", f"{result.request_metadata.excitation_nm or '—'} nm"),
                    ("notes", (result.request_metadata.notes or "—")[:80])]),
            ("h", "2. Engine metadata"),
            ("kv", [("engine", e.engine_version), ("GAIRA version", e.gaira_version),
                    ("atlas fingerprint", e.atlas_fingerprint),
                    ("atlas (frozen)", e.fingerprints.get("atlas", "")),
                    ("LSM registry", e.fingerprints.get("lsm", "")),
                    ("CSM registry", e.fingerprints.get("csm", "")),
                    ("Phase 05 engine", e.fingerprints.get("engine", "")),
                    ("frozen assets verified", e.frozen_assets_verified),
                    ("atlas shape", f"{e.n_lsms} LSMs · {e.n_csms} CSMs · {e.n_molecules} "
                                    f"molecules · {e.n_chemistry_axes} axes"),
                    ("generated", m.generated_utc[:19] + " UTC"),
                    ("result digest", result.result_digest)]),
            ("h", "4. Preprocessing"),
            ("kv", [("input points", result.preprocessing.n_input_points),
                    ("input range", f"{result.preprocessing.input_range[0]:.0f}–"
                                    f"{result.preprocessing.input_range[1]:.0f} cm-1"),
                    ("resampled to", result.preprocessing.resampled_to),
                    ("baseline", result.preprocessing.baseline_method),
                    ("smoothing", result.preprocessing.smoothing),
                    ("normalisation", result.preprocessing.normalisation),
                    ("grid coverage", f"{result.preprocessing.grid_coverage:.1%}"),
                    ("peaks", result.preprocessing.n_peaks),
                    ("SNR estimate", f"{result.preprocessing.snr_estimate:.1f}")]),
        ])
        image_page(pdf, "3–6. Spectrum, representation and reconstruction",
                   [FIG.spectrum_panel(result), FIG.csm_panel(result),
                    FIG.reconstruction_panel(result)])
        image_page(pdf, "7. Grounded Evidence Retrieval", [FIG.retrieval_panel(result)],
                   note="Candidates are retrieved reference analogues, not definitive molecular "
                        "identifications. Validated molecule top-1 is 0.6053.")
        image_page(pdf, "8. Chemistry Evidence", [FIG.chemistry_panel(result)],
                   note="RELATIVE BIOCHEMICAL EVIDENCE — not a concentration, not an abundance, "
                        "not a mixture fraction.")
        text_page(pdf, "7–8. Evidence tables", [
            ("h", "Grounded Evidence Retrieval"),
            ("table", (["rank", "molecule", "chemistry class", "similarity"],
                       [[h.rank, h.molecule[:34], h.chemistry_class[:26], f"{h.similarity:.4f}"]
                        for h in result.retrieval.top],
                       [0.0, 0.06, 0.42, 0.72])),
            ("h", "Chemistry Evidence — all 16 axes"),
            ("table", (["axis", "evidence", "share", "calibrated"],
                       [[chem.axis_names[i].replace("_", " "), f"{chem.evidence[i]:.4f}",
                         f"{chem.evidence_l1[i]:.1%}",
                         f"{chem.calibrated_probability[i]:.3f}"]
                        for i in sorted(range(len(chem.axis_names)),
                                        key=lambda j: -chem.evidence[j])],
                       [0.0, 0.34, 0.52, 0.68])),
        ])
        text_page(pdf, "9–13. Confidence, audit, provenance and scope", [
            ("h", "9. Confidence"),
            ("kv", [("overall", f"{conf.overall:.4f}"),
                    ("evidence coverage", f"{conf.evidence_coverage:.4f}"),
                    ("CSM explained variance",
                     f"{conf.reconstruction_explained_variance:.4f}"),
                    ("retrieval margin", f"{conf.retrieval_margin:.4f}"),
                    ("chemistry confidence", f"{conf.chemistry_confidence:.4f}"),
                    ("unknown warning", conf.unknown_warning),
                    ("outlier warning", conf.outlier_warning)]),
            ("h", "10. Scientific audit"),
            ("kv", ([("CSM residual fraction", f"{result.audit.csm_residual_fraction:.4f}"),
                     ("active CSMs", result.audit.n_active_csms),
                     ("spectral coverage", f"{result.audit.spectral_coverage:.1%}"),
                     ("chemistry margin", f"{result.audit.chemistry_margin:.4f}"),
                     ("chemistry entropy", f"{result.audit.chemistry_entropy:.4f}"),
                     ("all scores reconcile", result.audit.all_scores_reconcile)]
                    if result.audit else [("audit", "not requested")])),
            ("note", result.audit.open_set_limitation if result.audit else ""),
            ("h", "11. Provenance"),
            ("p", f"spectrum → {len(result.provenance.lsm_layer) if result.provenance else 0} "
                  f"LSMs → {len(result.provenance.csm_layer) if result.provenance else 0} CSMs "
                  f"→ {len(result.provenance.chemistry_layer) if result.provenance else 0} "
                  f"chemistry axes → "
                  f"{len(result.provenance.molecule_layer) if result.provenance else 0} "
                  f"molecules. Every node resolves to diagnostic wavenumbers."),
            ("h", "12. Interpretation summary"),
            ("p", result.interpretation),
            ("h", "13. Scope and limitations"),
            ("bullets", list(SCOPE_NOTES)),
        ])
        d = pdf.infodict()
        d["Title"] = title or "GAIRA V7 Inference Report"
        d["Subject"] = "Grounded Raman biochemical inference"
        d["Keywords"] = f"GAIRA V7 Raman chemistry evidence {result.result_digest}"
    return buf.getvalue()


def render(result: InferenceResult, fmt: str = "pdf", title: str | None = None):
    fmt = fmt.lower()
    if fmt == "json":
        return render_json(result, title)
    if fmt == "html":
        return render_html(result, title)
    if fmt == "pdf":
        return render_pdf(result, title)
    raise ValueError(f"unknown report format {fmt!r}; expected json, html or pdf")
