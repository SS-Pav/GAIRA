"""GAIRA V7 — Grounded Raman Biochemical Inference.

A THIN client. It uploads, calls the runtime, and renders. It contains no scientific computation:
no NMF, no NNLS, no projection, no similarity, no calibration, no aggregation. Every number
displayed comes from an `InferenceResult` returned by `gaira.v7.runtime`, and
`tests/test_v7_phase10_parity.py` fails the build if a scientific import appears in this file.

Run:  streamlit run streamlit_apps/gaira_v7_console.py     (or: gaira streamlit)

Backend selection is configuration only. Set GAIRA_API_URL to point at a deployed FastAPI service
and nothing else changes — the request and result schemas are identical either way.
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from gaira.v7.adapters import PLANNED_FORMATS, load as load_spectrum   # noqa: E402
from gaira.v7.contracts import Modality, SampleType                    # noqa: E402

API_URL = os.environ.get("GAIRA_API_URL", "").strip()
MAX_UPLOAD_MB = 32
INK, MUTED, ACCENT, WARM, GOOD = "#1a1a1a", "#6b7280", "#1d4ed8", "#b45309", "#15803d"

st.set_page_config(page_title="GAIRA V7", page_icon="◈", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
  .block-container {padding-top: 2.2rem; max-width: 1280px;}
  h1 {font-size: 1.85rem !important; letter-spacing: -0.01em;}
  h2 {font-size: 1.15rem !important; margin-top: 1.6rem;}
  h3 {font-size: 0.98rem !important; color: #374151;}
  .lede {color:#6b7280; font-size:0.98rem; line-height:1.55; margin-bottom:0.4rem;}
  .scope {border-left:3px solid #b45309; background:#fffbeb; padding:0.6rem 0.9rem;
          font-size:0.86rem; border-radius:0 4px 4px 0; margin:0.6rem 0;}
  .ok {border-left:3px solid #15803d; background:#f0fdf4; padding:0.6rem 0.9rem;
       font-size:0.86rem; border-radius:0 4px 4px 0;}
  .mono {font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size:0.80rem;
         color:#374151;}
  [data-testid="stMetricValue"] {font-size: 1.28rem;}
  [data-testid="stMetricLabel"] {font-size: 0.78rem; color:#6b7280;}
  .stTabs [data-baseweb="tab"] {font-size: 0.92rem;}
</style>""", unsafe_allow_html=True)


# ── backend ──────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading the frozen GAIRA V7 atlas…")
def get_client():
    """Local engine or remote API. The same call signatures either way."""
    from gaira.v7 import GAIRA
    return GAIRA.remote(API_URL) if API_URL else GAIRA.shared()


@st.cache_data(show_spinner=False)
def engine_info_dict():
    return get_client().engine_info().model_dump(mode="json")


def run_inference(x, y, metadata, options):
    from gaira.v7.sdk import RemoteRejected, SpectrumRejected
    try:
        return get_client().infer(x, y, metadata, options), None
    except (SpectrumRejected, RemoteRejected) as rejected:
        return None, rejected


# ── shared rendering helpers ─────────────────────────────────────────────────
def scope_banner(md: dict) -> None:
    if md.get("modality", "raman") != "raman":
        st.error(f"**Modality `{md['modality']}` is not supported by the V7 scientific core.** "
                 "V7 is Raman-only. A Raman motif dictionary reconstructs SERS of the same "
                 "metabolites comfortably (Phase 04 measured AUROC 0.548), so running this "
                 "through the engine would produce confident numbers with no validated meaning. "
                 "Inference is blocked.")
    if md.get("sample_type", "pure") != "pure":
        st.markdown(f"<div class='scope'><b>Scope warning — sample type "
                    f"“{md['sample_type']}”.</b> Every V7 number was measured on pure reference "
                    f"compounds. The calculation is unchanged and the metadata is recorded, but "
                    f"V7 has <b>no validated interpretation capability</b> for this context. "
                    f"Read the result as spectral evidence, not as a domain finding.</div>",
                    unsafe_allow_html=True)


def diagnostics_block(diags: list[dict], title: str = "Input diagnostics") -> None:
    if not diags:
        return
    errs = [d for d in diags if d["severity"] == "error"]
    warns = [d for d in diags if d["severity"] == "warning"]
    infos = [d for d in diags if d["severity"] == "info"]
    for d in errs:
        st.error(f"**{d['code']}** — {d['message']}")
    for d in warns:
        st.warning(f"**{d['code']}** — {d['message']}")
    if infos:
        with st.expander(f"{title} — {len(infos)} informational", expanded=False):
            for d in infos:
                st.markdown(f"<span class='mono'>{d['code']}</span> — {d['message']}",
                            unsafe_allow_html=True)


def spectrum_figure(grid, values, title, colour=ACCENT, second=None, second_name=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grid, y=values, mode="lines", name="query",
                             line=dict(color=INK, width=1.2)))
    if second is not None:
        fig.add_trace(go.Scatter(x=grid, y=second, mode="lines", name=second_name or "second",
                                 line=dict(color=colour, width=1.2)))
    fig.update_layout(title=dict(text=title, font=dict(size=13)), height=280,
                      margin=dict(l=10, r=10, t=40, b=10), plot_bgcolor="white",
                      xaxis_title="wavenumber (cm⁻¹)", yaxis_title="intensity",
                      legend=dict(orientation="h", y=1.12, x=1, xanchor="right"),
                      xaxis=dict(gridcolor="#f1f3f5"), yaxis=dict(gridcolor="#f1f3f5"))
    return fig


def chemistry_bars(chem: dict, height=430):
    order = np.argsort(chem["evidence_l1"])
    names = [chem["axis_names"][i].replace("_", " ") for i in order]
    vals = [chem["evidence_l1"][i] for i in order]
    conf = [chem["calibrated_probability"][i] for i in order]
    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h", marker=dict(color=ACCENT),
        customdata=np.array(conf),
        hovertemplate="%{y}<br>relative evidence %{x:.3f}"
                      "<br>calibrated confidence %{customdata:.3f}<extra></extra>"))
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=36, b=40), plot_bgcolor="white",
                      title=dict(text="Relative Chemistry Evidence — ordered", font=dict(size=13)),
                      xaxis_title="share of total evidence",
                      xaxis=dict(gridcolor="#f1f3f5"), yaxis=dict(gridcolor="white"))
    return fig


def chemistry_radar(chem: dict, height=430):
    names = [a.replace("_", " ") for a in chem["axis_names"]]
    vals = list(chem["evidence_l1"])
    fig = go.Figure(go.Scatterpolar(r=vals + vals[:1], theta=names + names[:1], fill="toself",
                                    line=dict(color=ACCENT, width=1.6),
                                    fillcolor="rgba(29,78,216,0.16)",
                                    hovertemplate="%{theta}<br>%{r:.3f}<extra></extra>"))
    fig.update_layout(height=height, margin=dict(l=60, r=60, t=46, b=30),
                      title=dict(text="Relative Chemistry Evidence — radar", font=dict(size=13)),
                      polar=dict(radialaxis=dict(showticklabels=False, gridcolor="#e5e7eb"),
                                 angularaxis=dict(tickfont=dict(size=9))),
                      showlegend=False)
    return fig


def evidence_caveat() -> None:
    st.caption("**Relative biochemical evidence.** Not a concentration, not an abundance, not a "
               "mixture fraction. Absolute scale is removed by L2 normalisation in the first "
               "preprocessing stage.")


# ── sidebar ──────────────────────────────────────────────────────────────────
def sidebar() -> None:
    with st.sidebar:
        st.markdown("### GAIRA V7")
        st.caption("Grounded Raman Biochemical Inference")
        try:
            i = engine_info_dict()
        except Exception as exc:
            st.error(f"engine unavailable: {exc}")
            return
        st.markdown(f"<span class='mono'>atlas {i['atlas_fingerprint'][:16]}…</span>",
                    unsafe_allow_html=True)
        st.markdown(f"<span class='mono'>{i['n_csms']} CSMs · {i['n_molecules']} molecules · "
                    f"{i['n_chemistry_axes']} axes</span>", unsafe_allow_html=True)
        st.markdown("---")
        st.caption(f"backend: {'remote — ' + API_URL if API_URL else 'local engine'}")
        st.caption(f"frozen assets verified: {i['frozen_assets_verified']}")
        st.markdown("---")
        st.caption("**Scope.** Pure Raman reference spectra. Chemistry Evidence is relative. "
                   "Retrieved molecules are reference analogues, not identifications. No "
                   "validated open-set detection.")


# ── page: analyse ────────────────────────────────────────────────────────────
def page_analyse() -> None:
    st.markdown("## Analyze Spectrum")
    st.markdown("<p class='lede'>Upload a Raman spectrum to project it into a frozen "
                "biochemical motif atlas, retrieve grounded reference evidence, and generate an "
                "interpretable Chemistry Evidence profile.</p>", unsafe_allow_html=True)

    left, right = st.columns([1.15, 1])
    with left:
        up = st.file_uploader("Spectrum file", type=["csv", "tsv", "txt", "dat", "asc"],
                              help=f"Two columns: wavenumber and intensity. Max "
                                   f"{MAX_UPLOAD_MB} MB. Planned but not yet supported: "
                                   f"{', '.join(sorted(PLANNED_FORMATS))}.")
    with right:
        st.markdown("**Measurement metadata**")
        c1, c2 = st.columns(2)
        modality = c1.selectbox(
            "Modality", [m.value for m in Modality], index=0,
            format_func=lambda m: m if m == "raman" else f"{m} — unsupported by V7",
            help="Only Raman is supported. Selecting another value blocks inference rather "
                 "than running it silently.")
        sample_type = c2.selectbox("Sample type", [s.value for s in SampleType], index=0)
        c3, c4 = st.columns(2)
        sample_name = c3.text_input("Sample name", value="")
        excitation = c4.number_input("Excitation (nm)", min_value=0.0, max_value=2000.0,
                                     value=0.0, step=1.0,
                                     help="0 leaves it unrecorded")
        notes = st.text_input("Notes", value="")

    metadata = {"modality": modality, "sample_type": sample_type,
                "sample_id": sample_name or None, "source_name": (up.name if up else None),
                "excitation_nm": excitation or None, "notes": notes or None}
    scope_banner(metadata)

    if up is None:
        st.info("Upload a spectrum to begin.")
        return
    raw = up.getvalue()
    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        st.error(f"File is {len(raw) / 1e6:.1f} MB; the limit is {MAX_UPLOAD_MB} MB.")
        return

    parsed = load_spectrum(raw, up.name)
    pdiags = [d.model_dump(mode="json") for d in parsed.diagnostics]
    with st.expander("Raw file preview", expanded=False):
        st.code("\n".join(raw.decode("utf-8", errors="replace").splitlines()[:12]) or "(empty)",
                language="text")
    diagnostics_block(pdiags, "Parsing")
    if not parsed.ok:
        st.error("The file could not be parsed into a spectrum.")
        return
    st.caption(f"parsed {len(parsed.wavenumber)} points, "
               f"{parsed.wavenumber.min():.0f}–{parsed.wavenumber.max():.0f} cm⁻¹, "
               f"via `{parsed.source_format}`")

    if not st.button("Run GAIRA V7 Inference", type="primary", use_container_width=False):
        return
    with st.spinner("Projecting into the frozen atlas…"):
        result, rejected = run_inference(
            [float(v) for v in parsed.wavenumber], [float(v) for v in parsed.intensity],
            metadata, {"include_reconstruction": True, "top_k_molecules": 10})
    if rejected is not None:
        st.error(f"**Spectrum rejected.** {rejected}")
        val = getattr(rejected, "validation", None)
        if val is not None:
            diagnostics_block([d.model_dump(mode="json") for d in val.diagnostics], "Validation")
        return

    st.session_state["result"] = result.model_dump(mode="json")
    st.session_state["result_name"] = sample_name or up.name
    render_result(st.session_state["result"])


def render_result(r: dict) -> None:
    conf, chem, pre = r["confidence"], r["chemistry"], r["preprocessing"]
    st.markdown("---")
    m = st.columns(5)
    m[0].metric("Chemistry", chem["predicted_class"].replace("_", " "),
                help="highest relative evidence axis")
    m[1].metric("Top analogue", r["retrieval"]["top"][0]["molecule"][:20],
                f"{r['retrieval']['top'][0]['similarity']:.3f}")
    m[2].metric("Confidence", f"{conf['overall']:.3f}")
    m[3].metric("CSM explained variance", f"{conf['reconstruction_explained_variance']:.3f}")
    m[4].metric("Coverage", f"{pre['grid_coverage']:.0%}")

    if conf["unknown_warning"] or conf["outlier_warning"]:
        flags = [n for n, f in (("unknown", conf["unknown_warning"]),
                                ("outlier", conf["outlier_warning"])) if f]
        st.markdown(f"<div class='scope'><b>Engine warning: {', '.join(flags)}.</b> "
                    + " ".join(conf["notes"]) +
                    " This is not evidence that the true molecule is absent from the reference "
                    "bank — V7 cannot determine that.</div>", unsafe_allow_html=True)

    t = st.tabs(["Chemistry Evidence", "Grounded Evidence Retrieval", "Preprocessing",
                 "CSM representation", "Interpretation"])

    with t[0]:
        c1, c2 = st.columns([1, 1])
        c1.plotly_chart(chemistry_bars(chem), use_container_width=True)
        c2.plotly_chart(chemistry_radar(chem), use_container_width=True)
        evidence_caveat()
        df = pd.DataFrame({
            "axis": [a.replace("_", " ") for a in chem["axis_names"]],
            "evidence": chem["evidence"], "share": chem["evidence_l1"],
            "calibrated confidence": chem["calibrated_probability"]}
        ).sort_values("evidence", ascending=False).reset_index(drop=True)
        df.insert(0, "rank", np.arange(1, len(df) + 1))
        with st.expander("All 16 axes", expanded=False):
            st.dataframe(df, use_container_width=True, hide_index=True,
                         column_config={
                             "share": st.column_config.ProgressColumn(
                                 "share", format="%.3f", min_value=0.0, max_value=1.0),
                             "calibrated confidence": st.column_config.NumberColumn(
                                 format="%.3f")})

    with t[1]:
        st.markdown("<div class='scope'>Candidates are retrieved <b>reference analogues</b>, "
                    "not definitive molecular identifications. Validated molecule top-1 is "
                    "0.6053; 68 of 375 corpus queries are unretrievable by construction.</div>",
                    unsafe_allow_html=True)
        hits = r["retrieval"]["top"]
        st.dataframe(pd.DataFrame([{
            "rank": h["rank"], "reference molecule": h["molecule"],
            "chemistry class": h["chemistry_class"].replace("_", " "),
            "CSM similarity": h["similarity"],
            "supporting CSMs": ", ".join(c["csm_id"] for c in h["supporting_csms"][:3]),
            "reconciles": h["reconciles"]} for h in hits]),
            use_container_width=True, hide_index=True,
            column_config={"CSM similarity": st.column_config.ProgressColumn(
                format="%.4f", min_value=0.0, max_value=1.0)})
        pick = st.selectbox("Inspect a candidate",
                            [f"{h['rank']}. {h['molecule']}" for h in hits])
        h = hits[int(pick.split(".")[0]) - 1]
        st.markdown(f"**{h['molecule']}** — {h['chemistry_class'].replace('_', ' ')}, "
                    f"similarity {h['similarity']:.4f}")
        contrib = pd.DataFrame([{
            "CSM": c["csm_id"], "contribution": c["contribution"],
            "share of similarity": c["share_of_similarity"],
            "diagnostic bands (cm⁻¹)": ", ".join(f"{b:.0f}" for b in c["diagnostic_bands"][:5]),
            "supporting LSMs": ", ".join(c["contributing_lsms"][:4])}
            for c in h["supporting_csms"]])
        st.dataframe(contrib, use_container_width=True, hide_index=True)
        st.caption(f"Score reconciliation: contributions sum to {h['contribution_sum']:.6f} "
                   f"against a similarity of {h['similarity']:.6f} — "
                   f"{'exact, no hidden term' if h['reconciles'] else 'MISMATCH (a defect)'}.")

    with t[2]:
        if pre.get("grid") and pre.get("processed_intensity"):
            st.plotly_chart(spectrum_figure(pre["grid"], pre["processed_intensity"],
                                            "Canonical preprocessed spectrum"),
                            use_container_width=True)
        c = st.columns(4)
        c[0].metric("Input points", pre["n_input_points"])
        c[1].metric("Peaks", pre["n_peaks"])
        c[2].metric("SNR estimate", f"{pre['snr_estimate']:.1f}")
        c[3].metric("Grid coverage", f"{pre['grid_coverage']:.1%}")
        st.markdown(f"**Pipeline** — resampled to {pre['resampled_to']}; baseline: "
                    f"{pre['baseline_method']}; smoothing: {pre['smoothing']}; normalisation: "
                    f"{pre['normalisation']}.")
        for w in pre.get("warnings", []):
            st.caption(f"⚠︎ {w}")

    with t[3]:
        csm = r.get("csm")
        if not csm:
            st.info("CSM detail was not requested."); return
        act = np.asarray(csm["activation"])
        fig = go.Figure(go.Bar(x=np.arange(len(act)), y=act, marker=dict(color=ACCENT)))
        fig.update_layout(height=250, margin=dict(l=10, r=10, t=36, b=30), plot_bgcolor="white",
                          title=dict(text=f"CSM activation — {csm['n_active']} of {len(act)} "
                                          f"active", font=dict(size=13)),
                          xaxis_title="CSM index", yaxis_title="activation",
                          xaxis=dict(gridcolor="#f1f3f5"), yaxis=dict(gridcolor="#f1f3f5"))
        st.plotly_chart(fig, use_container_width=True)
        if pre.get("grid") and pre.get("processed_intensity") and csm.get("reconstruction"):
            st.plotly_chart(spectrum_figure(pre["grid"], pre["processed_intensity"],
                                            "Query and CSM reconstruction",
                                            second=csm["reconstruction"],
                                            second_name="reconstruction"),
                            use_container_width=True)
        c = st.columns(4)
        c[0].metric("Explained variance", f"{csm['explained_variance']:.4f}")
        c[1].metric("Residual fraction", f"{csm['residual_fraction']:.4f}")
        c[2].metric("Sparsity", f"{csm['sparsity']:.4f}")
        c[3].metric("Entropy", f"{csm['entropy']:.4f}")
        st.markdown("**Top consensus motifs and their diagnostic bands**")
        st.dataframe(pd.DataFrame([{
            "CSM": t_["motif_id"], "weight": t_["weight"], "share": t_["share"],
            "bands (cm⁻¹)": ", ".join(f"{b:.0f}" for b in t_["dominant_bands"][:6]),
            "assignment": t_["band_assignment"][:70]} for t_ in csm["top"]]),
            use_container_width=True, hide_index=True)
        if r.get("lsm"):
            with st.expander("Local Spectral Motifs (diagnostic; not consumed downstream)"):
                st.caption("The LSM projection is reported for interpretability. The **CSM** "
                           "activation is the canonical representation and the only one any "
                           "later stage reads.")
                st.dataframe(pd.DataFrame([{
                    "LSM": t_["motif_id"], "weight": t_["weight"], "share": t_["share"]}
                    for t_ in r["lsm"]["top"]]), use_container_width=True, hide_index=True)

    with t[4]:
        st.markdown(f"##### Interpretation")
        st.write(r["interpretation"])
        st.caption("Deterministic template text. No language model is involved.")
        st.markdown("---")
        fmt = st.radio("Report format", ["pdf", "html", "json"], horizontal=True)
        if st.button("Generate report"):
            from gaira.v7.contracts import InferenceResult
            with st.spinner("Rendering…"):
                payload = get_client().report(InferenceResult.model_validate(r), fmt=fmt)
            data = payload if isinstance(payload, bytes) else payload.encode("utf-8")
            st.download_button(f"Download .{fmt}", data=data,
                               file_name=f"gaira_v7_report_{r['result_digest'][:12]}.{fmt}",
                               mime={"pdf": "application/pdf", "html": "text/html",
                                     "json": "application/json"}[fmt])


# ── page: audit ──────────────────────────────────────────────────────────────
def page_audit() -> None:
    st.markdown("## Scientific Audit")
    r = st.session_state.get("result")
    if not r:
        st.info("Run an inference on the Analyze Spectrum page first.")
        return
    conf, audit = r["confidence"], r.get("audit")
    st.markdown("<p class='lede'>Everything a reviewer needs to decide how much weight this "
                "answer can bear.</p>", unsafe_allow_html=True)
    c = st.columns(4)
    c[0].metric("Overall confidence", f"{conf['overall']:.4f}")
    c[1].metric("CSM explained variance", f"{conf['reconstruction_explained_variance']:.4f}")
    c[2].metric("Top-hit margin", f"{conf['retrieval_margin']:.4f}")
    c[3].metric("Chemistry confidence", f"{conf['chemistry_confidence']:.4f}")
    if audit:
        c = st.columns(4)
        c[0].metric("Residual fraction", f"{audit['csm_residual_fraction']:.4f}")
        c[1].metric("Active CSMs", audit["n_active_csms"])
        c[2].metric("Spectral coverage", f"{audit['spectral_coverage']:.1%}")
        c[3].metric("Chemistry entropy", f"{audit['chemistry_entropy']:.4f}")
        st.markdown("**Score reconciliation** — "
                    + ("every retrieval score decomposes exactly into its listed CSM "
                       "contributions (difference < 1e-9). No hidden term."
                       if audit["all_scores_reconcile"] else
                       "**A score failed to reconcile. This is a defect, not a warning.**"))

    st.markdown("### Warnings")
    if conf["unknown_warning"] or conf["outlier_warning"]:
        for n in conf["notes"]:
            st.warning(n)
    else:
        st.markdown("<div class='ok'>No engine warnings raised.</div>", unsafe_allow_html=True)
    diagnostics_block(r.get("diagnostics", []), "Input and scope")

    st.markdown("### Open-set limitation")
    st.markdown(
        "<div class='scope'><b>The current V7 engine does not provide validated open-set "
        "molecule detection.</b> Phase 09 measured white noise reconstructing at CSM explained "
        "variance ≈ 0.61 — above the 0.50 <code>unknown</code> floor — with the flag firing on "
        "only 1 of 20 random spectra. Confidence separates it correctly (noise peaked at 0.495 "
        "against a corpus mean of 0.803).<br><br>Low confidence and poor evidence quality should "
        "be treated as <b>caution signals, not proof of novelty</b>. Read the confidence, not "
        "the flag.</div>", unsafe_allow_html=True)


# ── page: provenance ─────────────────────────────────────────────────────────
def page_provenance() -> None:
    st.markdown("## Evidence & Provenance")
    r = st.session_state.get("result")
    if not r:
        st.info("Run an inference on the Analyze Spectrum page first.")
        return
    prov = r.get("provenance")
    if not prov:
        st.info("Provenance was not requested for this result.")
        return
    st.markdown("<p class='lede'>Every conclusion resolves down to specific wavenumbers. This "
                "chain is what makes a GAIRA answer auditable rather than merely produced.</p>",
                unsafe_allow_html=True)
    st.markdown(f"<span class='mono'>spectrum → {len(prov['lsm_layer'])} LSMs → "
                f"{len(prov['csm_layer'])} CSMs → {len(prov['chemistry_layer'])} chemistry axes "
                f"→ {len(prov['molecule_layer'])} molecules</span>", unsafe_allow_html=True)

    cols = st.columns(4)
    with cols[0]:
        st.markdown("##### Local motifs")
        st.dataframe(pd.DataFrame([{"LSM": n["identifier"], "weight": round(n["weight"], 4)}
                                   for n in prov["lsm_layer"]]),
                     hide_index=True, use_container_width=True)
    with cols[1]:
        st.markdown("##### Consensus motifs")
        st.dataframe(pd.DataFrame([{
            "CSM": n["identifier"], "weight": round(n["weight"], 4),
            "bands": ", ".join(f"{b:.0f}" for b in n["detail"].get("bands", [])[:3])}
            for n in prov["csm_layer"]]), hide_index=True, use_container_width=True)
    with cols[2]:
        st.markdown("##### Chemistry axes")
        st.dataframe(pd.DataFrame([{"axis": n["identifier"].replace("_", " "),
                                    "evidence": round(n["weight"], 4)}
                                   for n in prov["chemistry_layer"]]),
                     hide_index=True, use_container_width=True)
    with cols[3]:
        st.markdown("##### Reference molecules")
        st.dataframe(pd.DataFrame([{
            "molecule": n["identifier"], "similarity": round(n["weight"], 4),
            "class": n["detail"].get("class", "")} for n in prov["molecule_layer"]]),
            hide_index=True, use_container_width=True)

    st.markdown("### Motif detail")
    for n in prov["csm_layer"]:
        with st.expander(f"{n['identifier']} — weight {n['weight']:.4f}"):
            st.markdown(f"**Diagnostic bands** "
                        f"{', '.join(f'{b:.0f}' for b in n['detail'].get('bands', [])) or '—'} "
                        f"cm⁻¹")
            st.markdown(f"**Band assignment** {n['detail'].get('assignment') or '—'}")
            st.markdown(f"**Contributing LSMs** "
                        f"{', '.join(n['detail'].get('lsms', [])) or '—'}")

    st.markdown("### Atlas identity")
    e = r["engine"]
    st.dataframe(pd.DataFrame([{"artefact": k, "fingerprint": v}
                               for k, v in e["fingerprints"].items()]
                              + [{"artefact": "derived atlas hash",
                                  "fingerprint": e["atlas_fingerprint"]},
                                 {"artefact": "result digest",
                                  "fingerprint": r["result_digest"]}]),
                 hide_index=True, use_container_width=True)


# ── page: compare ────────────────────────────────────────────────────────────
def page_compare() -> None:
    st.markdown("## Compare Spectra")
    st.markdown("<p class='lede'>Two spectra, each run independently through the complete "
                "engine, then compared in motif and chemistry space.</p>",
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    fa = c1.file_uploader("Spectrum A", type=["csv", "tsv", "txt", "dat", "asc"], key="cmp_a")
    fb = c2.file_uploader("Spectrum B", type=["csv", "tsv", "txt", "dat", "asc"], key="cmp_b")
    sample_type = st.selectbox("Sample type (both)", [s.value for s in SampleType], index=0)
    scope_banner({"modality": "raman", "sample_type": sample_type})
    if not (fa and fb):
        st.info("Upload two spectra to compare.")
        return
    pa, pb = load_spectrum(fa.getvalue(), fa.name), load_spectrum(fb.getvalue(), fb.name)
    if not (pa.ok and pb.ok):
        st.error("One or both files could not be parsed.")
        diagnostics_block([d.model_dump(mode="json") for d in pa.diagnostics + pb.diagnostics])
        return
    if not st.button("Compare", type="primary"):
        return
    from gaira.v7.sdk import RemoteRejected, SpectrumRejected
    md = {"sample_type": sample_type}
    try:
        with st.spinner("Running both spectra…"):
            cmp = get_client().compare(
                ([float(v) for v in pa.wavenumber], [float(v) for v in pa.intensity]),
                ([float(v) for v in pb.wavenumber], [float(v) for v in pb.intensity]),
                label_a=Path(fa.name).stem, label_b=Path(fb.name).stem,
                metadata_a=md, metadata_b=md)
    except (SpectrumRejected, RemoteRejected) as rejected:
        st.error(f"**Rejected.** {rejected}")
        return
    d = cmp.model_dump(mode="json")
    m = st.columns(3)
    m[0].metric("CSM cosine", f"{d['csm_cosine']:.4f}")
    m[1].metric("Chemistry cosine", f"{d['chemistry_cosine']:.4f}")
    m[2].metric("Top-10 overlap", f"{d['rank_agreement']:.2f}",
                f"{len(d['shared_top_molecules'])} shared")

    deltas = sorted(d["chemistry_delta"], key=lambda x: x["delta"])
    fig = go.Figure(go.Bar(
        x=[x["delta"] for x in deltas], y=[x["axis"].replace("_", " ") for x in deltas],
        orientation="h",
        marker=dict(color=[GOOD if x["delta"] >= 0 else WARM for x in deltas])))
    fig.update_layout(height=460, margin=dict(l=10, r=10, t=40, b=40), plot_bgcolor="white",
                      title=dict(text=f"Chemistry evidence: {d['label_b']} − {d['label_a']}",
                                 font=dict(size=13)),
                      xaxis=dict(gridcolor="#f1f3f5", zerolinecolor="#d1d5db"),
                      yaxis=dict(gridcolor="white"))
    st.plotly_chart(fig, use_container_width=True)
    evidence_caveat()

    c1, c2 = st.columns(2)
    for col, side, label in ((c1, d["a"], d["label_a"]), (c2, d["b"], d["label_b"])):
        with col:
            st.markdown(f"##### {label}")
            st.markdown(f"chemistry **{side['chemistry']['predicted_class'].replace('_', ' ')}** "
                        f"· confidence {side['confidence']['overall']:.3f}")
            st.dataframe(pd.DataFrame([{"rank": h["rank"], "molecule": h["molecule"],
                                        "similarity": round(h["similarity"], 4)}
                                       for h in side["retrieval"]["top"][:5]]),
                         hide_index=True, use_container_width=True)
    st.markdown("---")
    st.write(d["interpretation"])
    st.markdown(f"<div class='scope'>{d['scope_note']}</div>", unsafe_allow_html=True)


# ── page: engine ─────────────────────────────────────────────────────────────
def page_engine() -> None:
    st.markdown("## Engine Information")
    try:
        i = engine_info_dict()
    except Exception as exc:
        st.error(f"engine unavailable: {exc}")
        return
    st.markdown("<p class='lede'>The scientific architecture is frozen after Phase 09. These "
                "figures are quoted from committed validation artefacts; nothing on this page "
                "is recomputed.</p>", unsafe_allow_html=True)
    c = st.columns(4)
    c[0].metric("Local Spectral Motifs", i["n_lsms"])
    c[1].metric("Consensus Spectral Motifs", i["n_csms"])
    c[2].metric("Reference molecules", i["n_molecules"])
    c[3].metric("Chemistry axes", i["n_chemistry_axes"])

    st.markdown("### Identity")
    st.dataframe(pd.DataFrame(
        [{"artefact": "GAIRA version", "value": i["gaira_version"]},
         {"artefact": "engine", "value": i["engine_version"]},
         {"artefact": "atlas fingerprint (derived)", "value": i["atlas_fingerprint"]}]
        + [{"artefact": f"frozen {k}", "value": v} for k, v in i["fingerprints"].items()]
        + [{"artefact": "frozen assets verified", "value": str(i["frozen_assets_verified"])}]),
        hide_index=True, use_container_width=True)

    st.markdown("### Corpus")
    st.dataframe(pd.DataFrame([{"property": k.replace("_", " "), "value": v}
                               for k, v in i["corpus"].items()]),
                 hide_index=True, use_container_width=True)

    st.markdown("### Validated performance")
    st.caption("Molecule retrieval is leave-one-spectrum-out over 375 spectra against the full "
               "154-molecule bank. Chemistry figures are **held out** under molecule-grouped "
               "cross-validation — the model never saw the test molecule.")
    p = i["validated_performance"]
    c = st.columns(3)
    with c[0]:
        st.markdown("**Molecular retrieval**")
        st.dataframe(pd.DataFrame([
            {"metric": "top-1", "value": p["molecule_top1"]},
            {"metric": "top-3", "value": p["molecule_top3"]},
            {"metric": "top-5", "value": p["molecule_top5"]},
            {"metric": "top-10", "value": p["molecule_top10"]},
            {"metric": "MRR", "value": p["molecule_mrr"]},
            {"metric": "nDCG@5", "value": p["molecule_ndcg5"]}]),
            hide_index=True, use_container_width=True)
    with c[1]:
        st.markdown("**Chemistry Evidence (unseen molecules)**")
        st.dataframe(pd.DataFrame([
            {"metric": "top-1", "value": p["chemistry_top1_heldout"]},
            {"metric": "top-3", "value": p["chemistry_top3_heldout"]},
            {"metric": "macro F1", "value": p["chemistry_macro_f1_heldout"]},
            {"metric": "radar reproducibility", "value": p["radar_reproducibility"]}]),
            hide_index=True, use_container_width=True)
    with c[2]:
        st.markdown("**Representation and robustness**")
        st.dataframe(pd.DataFrame([
            {"metric": "CSM explained variance", "value": p["csm_mean_explained_variance"]},
            {"metric": "replicate consistency", "value": p["csm_replicate_consistency"]},
            {"metric": "radar under noise", "value": p["robustness_radar_cosine"]},
            {"metric": "chemistry under noise", "value": p["robustness_chemistry_top1"]},
            {"metric": "molecule under noise", "value": p["robustness_molecule_top1"]}]),
            hide_index=True, use_container_width=True)

    st.markdown("### Chemistry axes")
    st.markdown(" · ".join(a.replace("_", " ") for a in i["chemistry_axes"]))

    st.markdown("### Scope and known limitations")
    for lim in i["known_limitations"]:
        st.markdown(f"- {lim}")
    st.markdown(f"**Supported modalities** — {', '.join(i['supported_modalities'])}. "
                f"**Validated sample types** — {', '.join(i['validated_sample_types'])}. "
                "Everything else is an extension point with a defined contract and no "
                "implementation.")

    st.markdown("### Methodology")
    with st.expander("How a spectrum becomes an inference"):
        st.markdown("""
1. **Canonical preprocessing** — crop to 450–1800 cm⁻¹, resample to 676 bins at 2.0 cm⁻¹
   spacing, remove the fluorescence background by asymmetric least squares, smooth with a
   Savitzky–Golay filter, normalise to unit length. The last step removes absolute intensity,
   which is why no output can be read as a concentration.
2. **LSM projection** — express the spectrum as a non-negative sum of 50 learned basis spectra.
   Diagnostic only; no later stage reads it.
3. **CSM projection** — the same against the 49 merged consensus motifs. **These 49 numbers are
   the canonical representation.**
4. **Grounded retrieval** — cosine similarity against 154 reference molecules. Because a cosine
   is an inner product, each score decomposes exactly into per-motif contributions, which is
   what makes the ranking explainable.
5. **Chemistry Evidence** — collapse the activation onto 16 chemistry axes with a hierarchical
   model, then calibrate. The result is *relative* evidence.
""")
    with st.expander("Why nothing sits above the CSM layer"):
        st.markdown("""
Chemistry accuracy on molecules the model has never seen:

| representation | held-out top-1 |
|---|---|
| raw preprocessed spectrum | 0.608 |
| LSM activation | 0.850 |
| **CSM activation** | **0.855** |
| 11 grounded axes | 0.664 |
| themes | 0.405 |
| Meta Components | 0.392 |

Four independent attempts to build an abstraction above the CSM layer each lost information. A
fifth — a geometric coordinate layer — produced +0.016 that a paired significance test could not
distinguish from zero (McNemar p = 0.180). BSV2 exists but is a *derived description* of
Chemistry Evidence; it is not on the inference path.
""")


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    st.markdown("# GAIRA V7")
    st.markdown("<p class='lede'>Grounded Raman Biochemical Inference</p>",
                unsafe_allow_html=True)
    sidebar()
    tabs = st.tabs(["Analyze Spectrum", "Scientific Audit", "Evidence & Provenance",
                    "Compare Spectra", "Engine Information"])
    with tabs[0]:
        page_analyse()
    with tabs[1]:
        page_audit()
    with tabs[2]:
        page_provenance()
    with tabs[3]:
        page_compare()
    with tabs[4]:
        page_engine()


main()
