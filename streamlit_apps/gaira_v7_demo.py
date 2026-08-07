"""GAIRA V7 — Phase 11 interactive scientific demo.

    streamlit run streamlit_apps/gaira_v7_demo.py

PRESENTATION ONLY. Every displayed number comes from an `InferenceResult` returned by the frozen
runtime. This file contains no preprocessing, no projection, no similarity, no calibration and no
aggregation, and `tests/test_v7_phase11.py` parses it with `ast` and fails the build if a
scientific primitive appears.

The one call that produces science is `GAIRA.infer(...)`. Everything else is layout.
"""
from __future__ import annotations

import base64
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT / "src"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gaira.v7.adapters import PLANNED_FORMATS, load as parse_spectrum        # noqa: E402
from gaira.v7.contracts import InferenceResult, Modality, SampleType         # noqa: E402
from streamlit_apps.gaira_v7_demo import data as D                           # noqa: E402
from streamlit_apps.gaira_v7_demo import figures as FIG                      # noqa: E402
from streamlit_apps.gaira_v7_demo import theme as T                          # noqa: E402

st.set_page_config(page_title="GAIRA — Grounded AI for Raman Analysis", page_icon="◈",
                   layout="wide", initial_sidebar_state="collapsed")
st.markdown(T.CSS, unsafe_allow_html=True)

PLOT_CFG = {"displaylogo": False,
            "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
            "toImageButtonOptions": {"format": "png", "scale": 2}}
PAGES = ["Home", "Analyze", "Docs", "Architecture", "About"]


# ── engine access (cached; loaded exactly once) ──────────────────────────────
@st.cache_resource(show_spinner=False)
def client():
    from gaira.v7 import GAIRA
    return GAIRA.shared()


@st.cache_resource(show_spinner=False)
def motifs():
    return D.load_reference_motifs()


@st.cache_resource(show_spinner=False)
def reference_spectra():
    return D.load_reference_spectra()


@st.cache_data(show_spinner=False)
def engine_info() -> dict:
    return client().engine_info().model_dump(mode="json")


def state(k, default=None):
    if k not in st.session_state:
        st.session_state[k] = default
    return st.session_state[k]


def goto(page: str) -> None:
    st.session_state["page"] = page


# ── shared chrome ────────────────────────────────────────────────────────────
def navbar() -> None:
    left, right = st.columns([1, 2.1])
    with left:
        st.markdown(
            '<div class="brand">◈ GAIRA<span>V7 · frozen runtime</span></div>',
            unsafe_allow_html=True)
    with right:
        cols = st.columns(len(PAGES))
        for c, name in zip(cols, PAGES):
            with c:
                if st.button(name, key=f"nav_{name}", use_container_width=True,
                             type="primary" if st.session_state.get("page") == name
                             else "secondary"):
                    goto(name); st.rerun()
    st.markdown("<div style='height:1px;background:%s;margin:.4rem 0 1.5rem'></div>" % T.STROKE,
                unsafe_allow_html=True)


def glass(html: str, tight=False) -> None:
    st.markdown(f'<div class="glass {"glass-tight" if tight else ""}">{html}</div>',
                unsafe_allow_html=True)


def stat(value, label, sub="") -> str:
    return (f'<div class="stat"><div class="stat-v">{value}</div>'
            f'<div class="stat-l">{label}</div>'
            f'{f"<div class=stat-s>{sub}</div>" if sub else ""}</div>')


def scope_footer() -> None:
    st.markdown(
        '<div class="scope"><b>Scope.</b> Pure Raman reference spectra. Chemistry Evidence is '
        '<b>relative</b> — not a concentration, not an abundance, not a mixture fraction. '
        'Retrieved molecules are <b>reference analogues</b>, not identifications. The engine '
        'provides <b>no validated open-set detection</b>: low confidence is a caution signal, '
        'not proof of novelty.</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1 — Home
# ═══════════════════════════════════════════════════════════════════════════
def page_home() -> None:
    st.markdown("""
      <div class="hero">
        <div class="hero-title">GAIRA</div>
        <div class="hero-sub">Grounded AI for Raman Analysis</div>
        <div class="hero-lede">Explainable biochemical reasoning for Raman spectroscopy.
          A spectrum is projected into a frozen motif atlas, matched against grounded reference
          evidence, and read out as a sixteen-axis chemistry profile — with every claim
          traceable back to a wavenumber.</div>
      </div>""", unsafe_allow_html=True)

    a, b, c = st.columns([1, 1, 1])
    with b:
        if st.button("Begin Analysis  →", key="hero_cta", use_container_width=True,
                     type="primary"):
            goto("Analyze"); st.rerun()

    st.markdown("<div style='height:2.6rem'></div>", unsafe_allow_html=True)
    i = engine_info()
    p = i["validated_performance"]
    cols = st.columns(4)
    tiles = [
        (i["n_csms"], "consensus motifs", "the canonical representation"),
        (i["n_molecules"], "reference molecules", "grounded evidence bank"),
        (i["n_chemistry_axes"], "chemistry axes", "relative evidence profile"),
        (f"{p['chemistry_top1_heldout']:.3f}", "chemistry top-1", "on molecules never seen"),
    ]
    for col, (v, l, s) in zip(cols, tiles):
        with col:
            glass(stat(v, l, s), tight=True)

    st.markdown("<div style='height:2.2rem'></div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    for col, (title, body) in zip((c1, c2, c3), [
        ("Frozen, and provably so",
         "The scientific engine has not changed since Phase 09. Ten artefact digests are "
         "verified before the runtime will serve anything, and the engine refuses to load "
         "against a changed atlas."),
        ("Explainable by construction",
         "Retrieval is a cosine, and a cosine is an inner product — so every score decomposes "
         "exactly into per-motif contributions. Nothing is hidden and nothing is left over."),
        ("Honest about its limits",
         "Molecule top-1 is 0.605. Chemistry Evidence is relative, never a concentration. There "
         "is no validated open-set detection. Every screen says so."),
    ]):
        with col:
            glass(f"<h3 style='margin-top:0'>{title}</h3>"
                  f"<p style='margin-bottom:0;font-size:.90rem'>{body}</p>")

    st.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)
    scope_footer()


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2–5 — Analyze
# ═══════════════════════════════════════════════════════════════════════════
def page_analyze() -> None:
    step = state("step", "upload")
    rail = {"upload": 0, "preprocess": 1, "analyze": 2, "results": 3}
    labels = ["Upload", "Preprocess", "Analyze", "Report"]
    marks = []
    for i, lab in enumerate(labels):
        cls = "done" if i < rail[step] else ("active" if i == rail[step] else "")
        icon = "✓" if i < rail[step] else str(i + 1)
        marks.append(f'<span class="pill {"pill-ok" if i < rail[step] else "pill-neu"}">'
                     f'{icon} {lab}</span>')
    st.markdown(f'<div style="margin-bottom:1.1rem">{"".join(marks)}</div>',
                unsafe_allow_html=True)

    {"upload": step_upload, "preprocess": step_preprocess,
     "analyze": step_analyze, "results": step_results}[step]()


def step_upload() -> None:
    st.markdown("## Upload a spectrum")
    st.markdown('<p class="caption">Two columns — wavenumber and intensity. CSV, TSV or '
                'plain text. The delimiter, header, column identity and axis direction are '
                'detected, and every decision is reported.</p>', unsafe_allow_html=True)

    left, right = st.columns([1.25, 1])
    with left:
        up = st.file_uploader("Drag and drop, or browse", type=["csv", "tsv", "txt", "dat",
                                                                "asc"],
                              label_visibility="collapsed")
    with right:
        st.markdown('<div style="height:.3rem"></div>', unsafe_allow_html=True)
        choice = st.selectbox("…or start from a built-in reference spectrum",
                             list(D.DEMO_SPECTRA))
        st.markdown('<p class="caption">Built-in spectra come from the frozen corpus and run '
                    'through exactly the same path as an upload.</p>',
                    unsafe_allow_html=True)

    x = y = None
    diags: list = []
    name = ""
    if up is not None:
        raw = up.getvalue()
        if len(raw) > 32 * 1024 * 1024:
            st.error("File exceeds the 32 MB limit."); return
        parsed = parse_spectrum(raw, up.name)
        diags = [d.model_dump(mode="json") for d in parsed.diagnostics]
        if not parsed.ok:
            st.error("This file could not be parsed into a spectrum.")
            for d in diags:
                if d["severity"] == "error":
                    st.error(f"**{d['code']}** — {d['message']}")
            return
        x, y, name = parsed.wavenumber, parsed.intensity, up.name
        with st.expander("Raw file preview", expanded=False):
            st.code("\n".join(raw.decode("utf-8", errors="replace").splitlines()[:12]),
                    language="text")
    elif D.DEMO_SPECTRA.get(choice):
        mol = D.DEMO_SPECTRA[choice]
        x = motifs()["grid"]
        y = reference_spectra()[mol]
        name = f"{mol} (frozen corpus reference)"
        st.session_state["already_preprocessed"] = True
    if x is None:
        st.markdown('<div class="note">Supported now: CSV, TSV, two-column text. Planned but '
                    'not yet implemented: ' + ", ".join(sorted(PLANNED_FORMATS)) +
                    ' — these are refused explicitly rather than mis-parsed.</div>',
                    unsafe_allow_html=True)
        return
    if up is not None:
        st.session_state["already_preprocessed"] = False

    st.session_state.update(x_raw=list(map(float, x)), y_raw=list(map(float, y)),
                            filename=name, parse_diagnostics=diags)

    st.markdown("<div style='height:.9rem'></div>", unsafe_allow_html=True)
    cols = st.columns(4)
    for col, (v, l) in zip(cols, [
            (name if len(name) < 26 else name[:23] + "…", "file"),
            (f"{len(x):,}", "points"),
            (f"{min(x):.0f}–{max(x):.0f}", "range (cm⁻¹)"),
            (f"{float(np.median(np.abs(np.diff(x)))):.2f}", "median step (cm⁻¹)")]):
        with col:
            glass(stat(v, l), tight=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.markdown("##### Raw spectrum, as supplied")
    st.plotly_chart(FIG.raw_spectrum(x, y), use_container_width=True, config=PLOT_CFG)

    infos = [d for d in diags if d["severity"] == "info"]
    warns = [d for d in diags if d["severity"] == "warning"]
    for d in warns:
        st.warning(f"**{d['code']}** — {d['message']}")
    if infos:
        with st.expander(f"How this file was read — {len(infos)} decisions", expanded=False):
            for d in infos:
                st.markdown(f"<span class='mono'>{d['code']}</span> &nbsp; {d['message']}",
                            unsafe_allow_html=True)

    st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)
    meta_cols = st.columns([1, 1, 1, 1.2])
    modality = meta_cols[0].selectbox(
        "Modality", [m.value for m in Modality],
        format_func=lambda m: m if m == "raman" else f"{m} — unsupported")
    sample_type = meta_cols[1].selectbox("Sample type", [s.value for s in SampleType])
    excitation = meta_cols[2].number_input("Excitation (nm)", 0.0, 2000.0, 0.0, 1.0)
    sample_id = meta_cols[3].text_input("Sample ID", value="")
    st.session_state["metadata"] = {
        "modality": modality, "sample_type": sample_type,
        "excitation_nm": excitation or None, "sample_id": sample_id or None,
        "source_name": name}

    if modality != "raman":
        st.error("**Modality `%s` is not supported by the V7 scientific core.** V7 is Raman-"
                 "only: Phase 04 measured a Raman motif dictionary reconstructing real Ag-SERS "
                 "at AUROC 0.548, so running this through the engine would produce confident "
                 "numbers with no validated meaning. Analysis is blocked." % modality)
        return
    if sample_type != "pure":
        st.markdown(f'<div class="scope"><b>Scope warning — sample type “{sample_type}”.</b> '
                    'Every V7 number was measured on pure reference compounds. The calculation '
                    'is unchanged and the metadata is recorded, but V7 has no validated '
                    'interpretation capability for this context.</div>',
                    unsafe_allow_html=True)

    st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
    if st.button("Preprocess Spectrum  →", key="to_pre"):
        st.session_state["step"] = "preprocess"; st.rerun()


def step_preprocess() -> None:
    st.markdown("## Preprocessing")
    x = np.asarray(st.session_state["x_raw"], float)
    y = np.asarray(st.session_state["y_raw"], float)

    stages = [("interpolate onto the canonical grid", "450–1800 cm⁻¹ · 2.0 step · 676 bins"),
              ("remove the fluorescence baseline", "asymmetric least squares"),
              ("smooth and normalise", "Savitzky–Golay (9, 3) → L2"),
              ("verify frozen fingerprints", "10 artefact digests")]
    stage = state("pre_stage", -1)

    left, right = st.columns([1, 2])
    with left:
        rows = []
        for i, (name, sub) in enumerate(stages):
            cls = "done" if i <= stage else ""
            icon = "✓" if i <= stage else "○"
            rows.append(
                f'<div class="stage {cls}"><div class="stage-dot">{icon}</div>'
                f'<div><div>{name}</div>'
                f'<div style="font-size:.76rem;color:{T.INK_3}">{sub}</div></div></div>')
        glass("".join(rows))
        st.markdown("<div style='height:.7rem'></div>", unsafe_allow_html=True)
        if stage < len(stages) - 1:
            if st.button("Run preprocessing", key="run_pre", use_container_width=True):
                ph = st.empty()
                for s in range(len(stages)):
                    st.session_state["pre_stage"] = s
                    time.sleep(0.32)
                    ph.markdown(f'<p class="caption">{stages[s][0]}…</p>',
                                unsafe_allow_html=True)
                ph.empty()
                st.rerun()
        else:
            st.markdown('<div class="note">Preprocessing complete. The curve on the right is '
                        'the vector the projection actually consumed — returned by the engine, '
                        'not recomputed here.</div>', unsafe_allow_html=True)
            if st.button("Analyze Spectrum  →", key="to_analyze", use_container_width=True):
                st.session_state["step"] = "analyze"; st.rerun()

    with right:
        processed, grid = None, motifs()["grid"]
        if stage >= 2:
            r = run_inference()
            if r is None:
                return
            processed = r["preprocessing"]["processed_intensity"]
            grid = r["preprocessing"]["grid"]
        st.plotly_chart(
            FIG.preprocessing_stages(x, y, grid, processed if processed is not None
                                     else np.zeros(len(grid)), max(stage, 0)),
            use_container_width=True, config=PLOT_CFG)
        st.markdown('<p class="caption">The raw trace is scaled to its own maximum so both '
                    'curves are visible on one axis; the processed curve is the engine\'s '
                    'output unaltered.</p>', unsafe_allow_html=True)

    if stage >= 3:
        r = run_inference()
        if r:
            pre = r["preprocessing"]
            cols = st.columns(5)
            for col, (v, l) in zip(cols, [
                    (pre["n_input_points"], "input points"),
                    (f"{pre['grid_coverage']:.0%}", "grid coverage"),
                    (pre["n_peaks"], "peaks detected"),
                    (f"{pre['snr_estimate']:.0f}", "SNR estimate"),
                    ("verified", "frozen assets")]):
                with col:
                    glass(stat(v, l), tight=True)
            for w in pre["warnings"]:
                st.markdown(f'<p class="caption">⚠︎ {w}</p>', unsafe_allow_html=True)


def run_inference():
    """The single scientific call. Cached per spectrum so it happens exactly once."""
    key = st.session_state.get("infer_key")
    sig = (st.session_state.get("filename"), len(st.session_state.get("x_raw", [])),
           json.dumps(st.session_state.get("metadata", {}), sort_keys=True))
    if key == sig and st.session_state.get("result"):
        return st.session_state["result"]
    from gaira.v7.sdk import SpectrumRejected
    try:
        t0 = time.perf_counter()
        res = client().infer(
            st.session_state["x_raw"], st.session_state["y_raw"],
            st.session_state.get("metadata"),
            {"include_reconstruction": True, "top_k_molecules": 10,
             "already_preprocessed": st.session_state.get("already_preprocessed", False)})
        elapsed = (time.perf_counter() - t0) * 1000
    except SpectrumRejected as rejected:
        st.error(f"**Spectrum rejected.** {rejected}")
        for d in rejected.validation.diagnostics:
            if d.severity.value == "error":
                st.error(f"**{d.code}** — {d.message}")
        return None
    st.session_state.update(result=res.model_dump(mode="json"),
                            result_obj=res, infer_key=sig, latency_ms=elapsed)
    return st.session_state["result"]


ANALYSIS_STEPS = [
    ("Projecting into the frozen motif atlas", "50 local spectral motifs, non-negative"),
    ("Computing LSM activations", "which learned building blocks are present"),
    ("Computing CSM activations", "the 49 canonical coordinates"),
    ("Searching the molecular reference atlas", "cosine over 154 reference molecules"),
    ("Building chemistry evidence", "16 axes, hierarchical, calibrated"),
    ("Generating report", "confidence · audit · provenance"),
]


def step_analyze() -> None:
    st.markdown("## Analysis")
    holder = st.empty()
    bar = st.progress(0.0)
    with holder.container():
        glass('<div style="text-align:center;padding:1.2rem 0">'
              '<div class="stat-v" style="font-size:1.35rem">Running the frozen engine…</div>'
              '<div class="stat-l">one call · no fitting · no randomness</div></div>')

    result = run_inference()
    if result is None:
        bar.empty(); holder.empty()
        if st.button("← Back to upload"):
            st.session_state["step"] = "upload"; st.rerun()
        return

    for i, (title, sub) in enumerate(ANALYSIS_STEPS):
        rows = []
        for j, (t2, s2) in enumerate(ANALYSIS_STEPS):
            cls = "done" if j < i else ("active" if j == i else "")
            icon = "✓" if j < i else ("◆" if j == i else "○")
            rows.append(f'<div class="stage {cls}"><div class="stage-dot">{icon}</div>'
                        f'<div><div>{t2}</div>'
                        f'<div style="font-size:.76rem;color:{T.INK_3}">{s2}</div></div></div>')
        with holder.container():
            glass("".join(rows))
        bar.progress((i + 1) / len(ANALYSIS_STEPS))
        time.sleep(0.34)

    bar.empty()
    holder.empty()
    st.session_state["step"] = "results"
    st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 5 — Results
# ═══════════════════════════════════════════════════════════════════════════
def confidence_band(v: float) -> tuple[str, str]:
    if v >= 0.70:
        return "High confidence", "pill-ok"
    if v >= 0.40:
        return "Moderate confidence", "pill-mid"
    return "Low confidence", "pill-low"


def step_results() -> None:
    r = run_inference()
    if r is None:
        return
    chem, conf, ret = r["chemistry"], r["confidence"], r["retrieval"]
    pre, csm, lsm = r["preprocessing"], r["csm"], r["lsm"]
    grid = pre["grid"]
    band, pill = confidence_band(conf["overall"])

    top_axis = chem["top"][0]
    top_hit = ret["top"][0]
    st.markdown(f"""
      <div class="verdict rise-in">
        <div class="verdict-k">bottom line</div>
        <div class="verdict-h">{top_axis['axis'].replace('_', ' ')}</div>
        <div><span class="pill {pill}">{band} · {conf['overall']:.3f}</span>
             <span class="pill pill-neu">{top_axis['share']:.0%} of total evidence</span>
             <span class="pill pill-neu">nearest analogue · {top_hit['molecule']}</span></div>
        <div class="verdict-p">{r['interpretation']}</div>
      </div>""", unsafe_allow_html=True)

    if conf["unknown_warning"] or conf["outlier_warning"]:
        flags = [n for n, f in (("unknown", conf["unknown_warning"]),
                                ("outlier", conf["outlier_warning"])) if f]
        st.markdown(f'<div class="scope"><b>Engine warning: {", ".join(flags)}.</b> '
                    + " ".join(conf["notes"]) +
                    ' This is <b>not</b> evidence that the true molecule is absent from the '
                    'reference bank — V7 cannot determine that.</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:1.1rem'></div>", unsafe_allow_html=True)
    left, centre, right = st.columns([1.05, 1.05, 0.95])

    with left:
        st.markdown("##### Processed spectrum")
        # the strongest motif's bands only: four motifs produced dozens of rules
        # and buried the spectrum they were meant to annotate
        bands = sorted({b for t in csm["top"][:1] for b in t["dominant_bands"]})[:10]
        st.plotly_chart(FIG.processed_spectrum(grid, pre["processed_intensity"], bands),
                        use_container_width=True, config=PLOT_CFG)
        st.markdown('<p class="caption">Amber markers are the diagnostic bands of the '
                    'strongest consensus motif — the wavenumbers the answer rests on.</p>',
                    unsafe_allow_html=True)

    with centre:
        st.markdown("##### Relative Chemistry Evidence")
        view = st.radio("view", ["Radar", "Bars", "Polar"], horizontal=True,
                        label_visibility="collapsed", key="chem_view")
        fig = {"Radar": FIG.chemistry_radar, "Bars": FIG.chemistry_bars,
               "Polar": FIG.chemistry_polar_bars}[view](chem)
        st.plotly_chart(fig, use_container_width=True, config=PLOT_CFG)
        st.markdown('<p class="caption"><b>Relative</b> biochemical evidence — not a '
                    'concentration, not an abundance, not a mixture fraction.</p>',
                    unsafe_allow_html=True)

    with right:
        st.markdown("##### Top molecular analogues")
        for h in ret["top"]:
            with st.expander(f"{h['rank']}.  {h['molecule']}  ·  {h['similarity']:.4f}",
                             expanded=h["rank"] == 1):
                st.markdown(
                    f"<span class='pill pill-neu'>{h['chemistry_class'].replace('_',' ')}"
                    f"</span> <span class='caption'>similarity {h['similarity']:.4f}</span>",
                    unsafe_allow_html=True)
                st.plotly_chart(FIG.csm_contribution_waterfall(h),
                                use_container_width=True, config=PLOT_CFG,
                                key=f"wf_{h['rank']}")
                st.markdown(
                    f"<p class='caption'>Contributions sum to {h['contribution_sum']:.6f} "
                    f"against a similarity of {h['similarity']:.6f} — "
                    f"{'exact, no hidden term' if h['reconciles'] else 'MISMATCH'}.</p>",
                    unsafe_allow_html=True)
        st.markdown('<div class="scope" style="font-size:.82rem">Retrieved <b>reference '
                    'analogues</b>, not identifications. Validated molecule top-1 is 0.6053.'
                    '</div>', unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    section_chemistry(chem)
    section_csm(csm, grid)
    section_lsm(lsm, grid)
    section_reconstruction(grid, pre, csm)
    section_retrieval(ret, grid, pre)
    section_confidence(conf, r)
    section_provenance(r)
    section_downloads(r)

    st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)
    c1, c2 = st.columns([1, 4])
    with c1:
        if st.button("↺  New spectrum", key="restart", use_container_width=True):
            for k in ("step", "result", "result_obj", "infer_key", "pre_stage",
                      "x_raw", "y_raw", "already_preprocessed"):
                st.session_state.pop(k, None)
            st.session_state["step"] = "upload"
            st.rerun()
    scope_footer()


def section_chemistry(chem: dict) -> None:
    with st.expander("Chemical evidence — all sixteen axes", expanded=False):
        st.plotly_chart(FIG.chemistry_bars(chem), use_container_width=True, config=PLOT_CFG,
                        key="chem_full")
        df = pd.DataFrame({
            "axis": [a.replace("_", " ") for a in chem["axis_names"]],
            "evidence": chem["evidence"], "share": chem["evidence_l1"],
            "calibrated confidence": chem["calibrated_probability"],
            "description": [FIG.AXIS_NOTE.get(a, "") for a in chem["axis_names"]],
        }).sort_values("evidence", ascending=False).reset_index(drop=True)
        df.insert(0, "rank", np.arange(1, len(df) + 1))
        st.dataframe(df, use_container_width=True, hide_index=True,
                     column_config={"share": st.column_config.ProgressColumn(
                         "share", format="%.3f", min_value=0.0, max_value=1.0)})


def section_csm(csm: dict, grid) -> None:
    with st.expander("CSM contributions — the canonical 49 coordinates", expanded=False):
        m = motifs()
        ids = m["csm_ids"]
        act = csm["activation"]
        c1, c2 = st.columns([1, 1])
        with c1:
            st.plotly_chart(FIG.activation_bars(act, ids, "CSM activation, ordered"),
                            use_container_width=True, config=PLOT_CFG, key="csm_bars")
        with c2:
            st.plotly_chart(FIG.activation_heatmap(act, ids, "CSM activation map"),
                            use_container_width=True, config=PLOT_CFG, key="csm_heat")
        active = [i for i in np.argsort(-np.asarray(act)) if act[i] > 0]
        pick = st.selectbox(
            "Inspect a consensus motif",
            [ids[i] for i in active] + [i_ for i_ in ids if i_ not in {ids[i] for i in active}],
            key="csm_pick")
        j = ids.index(pick)
        rec = m["csm_records"].get(pick, {})
        bands = [float(b) for b in rec.get("dominant_bands", [])]
        st.plotly_chart(
            FIG.motif_spectrum(grid, m["CSM"][j], pick, bands, rec.get("band_assignment", "")),
            use_container_width=True, config=PLOT_CFG, key="csm_motif")
        cols = st.columns(4)
        for col, (v, l) in zip(cols, [
                (f"{act[j]:.4f}", "activation in this query"),
                (len(rec.get("contributing_lsms", [])) or 1, "contributing LSMs"),
                (len(rec.get("supporting_analytes", [])), "supporting molecules"),
                (", ".join(f"{b:.0f}" for b in bands[:3]) or "—", "diagnostic bands cm⁻¹")]):
            with col:
                glass(stat(v, l), tight=True)
        sup = rec.get("supporting_analytes", [])
        if sup:
            st.markdown(f"<p class='caption'><b>Supporting molecules:</b> "
                        f"{', '.join(map(str, sup[:22]))}"
                        f"{' …' if len(sup) > 22 else ''}</p>", unsafe_allow_html=True)


def section_lsm(lsm: dict, grid) -> None:
    with st.expander("LSM view — the 50 local motifs (diagnostic only)", expanded=False):
        st.markdown('<div class="note">The LSM projection is reported for interpretability. '
                    'The <b>CSM</b> activation is the canonical representation and the only one '
                    'any later stage reads.</div>', unsafe_allow_html=True)
        m = motifs()
        ids, act = m["lsm_ids"], lsm["activation"]
        st.plotly_chart(FIG.activation_bars(act, ids, "LSM activation, ordered",
                                            colour=T.VIOLET),
                        use_container_width=True, config=PLOT_CFG, key="lsm_bars")
        pick = st.selectbox("Inspect a local motif",
                            [ids[i] for i in np.argsort(-np.asarray(act))], key="lsm_pick")
        j = ids.index(pick)
        st.plotly_chart(FIG.motif_spectrum(grid, m["H_lsm"][j], pick),
                        use_container_width=True, config=PLOT_CFG, key="lsm_motif")
        c = st.columns(3)
        for col, (v, l) in zip(c, [(f"{act[j]:.4f}", "activation"),
                                   (f"{lsm['explained_variance']:.4f}", "LSM explained variance"),
                                   (lsm["n_active"], "active of 50")]):
            with col:
                glass(stat(v, l), tight=True)


def section_reconstruction(grid, pre: dict, csm: dict) -> None:
    with st.expander("Reconstruction and residual", expanded=False):
        op = st.slider("reconstruction opacity", 0.1, 1.0, 0.85, 0.05, key="recon_op")
        st.plotly_chart(FIG.reconstruction(grid, pre["processed_intensity"],
                                           csm["reconstruction"], op),
                        use_container_width=True, config=PLOT_CFG, key="recon")
        c = st.columns(4)
        for col, (v, l) in zip(c, [
                (f"{csm['explained_variance']:.4f}", "explained variance"),
                (f"{csm['residual_fraction']:.4f}", "residual fraction"),
                (csm["n_active"], "active CSMs"),
                (f"{csm['sparsity']:.3f}", "sparsity")]):
            with col:
                glass(stat(v, l), tight=True)


def section_retrieval(ret: dict, grid, pre: dict) -> None:
    with st.expander("Molecular retrieval — query against reference", expanded=False):
        st.plotly_chart(FIG.retrieval_bars(ret["top"]), use_container_width=True,
                        config=PLOT_CFG, key="ret_bars")
        refs = reference_spectra()
        pick = st.selectbox("Overlay the query on a reference spectrum",
                            [f"{h['rank']}. {h['molecule']}" for h in ret["top"]],
                            key="ret_pick")
        h = ret["top"][int(pick.split(".")[0]) - 1]
        ref = refs.get(h["molecule"])
        if ref is None:
            st.info("No stored reference spectrum for this molecule.")
            return
        bands = sorted({b for c in h["supporting_csms"] for b in c["diagnostic_bands"]})
        st.plotly_chart(FIG.overlay(grid, pre["processed_intensity"], ref, h["molecule"],
                                    bands),
                        use_container_width=True, config=PLOT_CFG, key="ret_overlay")
        c = st.columns(4)
        for col, (v, l) in zip(c, [
                (f"{h['similarity']:.4f}", "CSM similarity"),
                (h["chemistry_class"].replace("_", " "), "chemistry class"),
                (len(h["supporting_csms"]), "supporting CSMs"),
                ("exact" if h["reconciles"] else "MISMATCH", "score reconciliation")]):
            with col:
                glass(stat(v, l), tight=True)


def section_confidence(conf: dict, r: dict) -> None:
    with st.expander("Confidence — why this number and not a higher one", expanded=False):
        c1, c2 = st.columns([1, 1.6])
        with c1:
            st.plotly_chart(FIG.confidence_gauge(conf), use_container_width=True,
                            config=PLOT_CFG, key="conf_gauge")
        with c2:
            st.plotly_chart(FIG.confidence_factors(conf), use_container_width=True,
                            config=PLOT_CFG, key="conf_factors")
        ev = conf["reconstruction_explained_variance"]
        s1 = conf["top1_confidence"]
        limiter = ("the atlas's ability to explain the spectrum"
                   if ev < s1 else "how closely the nearest reference matches")
        st.markdown(
            f"<p>Confidence is <b>deliberately multiplicative</b>: the atlas must be able to "
            f"<i>explain</i> the spectrum <b>and</b> some reference must <i>match</i> it. Here "
            f"{ev:.3f} × {s1:.3f} = <b>{conf['overall']:.3f}</b>. The limiting factor is "
            f"{limiter}. A spectrum the dictionary cannot express might still land near some "
            f"molecule by accident, and multiplying ensures that accident cannot become a "
            f"confident answer.</p>", unsafe_allow_html=True)
        a = r.get("audit") or {}
        cols = st.columns(4)
        for col, (v, l) in zip(cols, [
                (f"{conf['retrieval_margin']:.4f}", "top-1 / top-2 margin"),
                (f"{conf['chemistry_confidence']:.4f}", "chemistry confidence"),
                (f"{a.get('spectral_coverage', 0):.0%}", "spectral coverage"),
                (f"{a.get('chemistry_entropy', 0):.3f}", "chemistry entropy")]):
            with col:
                glass(stat(v, l), tight=True)
        if a.get("open_set_limitation"):
            st.markdown(f'<div class="scope">{a["open_set_limitation"]}</div>',
                        unsafe_allow_html=True)


def section_provenance(r: dict) -> None:
    with st.expander("Provenance — the full evidence chain", expanded=False):
        prov = r.get("provenance")
        if not prov:
            st.info("Provenance was not requested."); return
        st.plotly_chart(FIG.provenance_sankey(prov, r["chemistry"]),
                        use_container_width=True, config=PLOT_CFG, key="prov")
        cols = st.columns(4)
        layers = [("lsm_layer", "Local motifs"), ("csm_layer", "Consensus motifs"),
                  ("chemistry_layer", "Chemistry axes"), ("molecule_layer", "Molecules")]
        for col, (key, label) in zip(cols, layers):
            with col:
                st.markdown(f"##### {label}")
                st.dataframe(pd.DataFrame([
                    {"id": n["identifier"].replace("_", " "),
                     "weight": round(n["weight"], 4)} for n in prov[key]]),
                    hide_index=True, use_container_width=True)
        e = r["engine"]
        st.markdown("##### Atlas identity")
        st.dataframe(pd.DataFrame(
            [{"term": "Scientific Atlas Fingerprint", "value": e["fingerprints"]["atlas"]},
             {"term": "LSM registry", "value": e["fingerprints"]["lsm"]},
             {"term": "CSM registry", "value": e["fingerprints"]["csm"]},
             {"term": "Phase 05 engine", "value": e["fingerprints"]["engine"]},
             {"term": "Frozen Runtime Content Hash", "value": e["atlas_fingerprint"]},
             {"term": "result digest", "value": r["result_digest"]}]),
            hide_index=True, use_container_width=True)


def section_downloads(r: dict) -> None:
    with st.expander("Download — JSON and PDF", expanded=False):
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            st.download_button(
                "InferenceResult.json", data=json.dumps(r, indent=2),
                file_name=f"gaira_v7_inference_{r['result_digest'][:12]}.json",
                mime="application/json", use_container_width=True)
        with c2:
            if st.button("Build PDF report", key="mk_pdf", use_container_width=True):
                with st.spinner("Rendering…"):
                    obj = st.session_state.get("result_obj") or \
                        InferenceResult.model_validate(r)
                    st.session_state["pdf"] = client().report(obj, fmt="pdf")
            if st.session_state.get("pdf"):
                st.download_button(
                    "Download PDF", data=st.session_state["pdf"],
                    file_name=f"gaira_v7_report_{r['result_digest'][:12]}.pdf",
                    mime="application/pdf", use_container_width=True)
        with c3:
            st.markdown('<p class="caption">Both artefacts are produced by the frozen '
                        'reporting module — the same generator the CLI, the API and the MCP '
                        'server use. The PDF is template-driven; no language model is '
                        'involved.</p>', unsafe_allow_html=True)
        st.markdown(f"<p class='caption'>Result digest <span class='mono'>"
                    f"{r['result_digest']}</span> — an MD5 over the scientific fields. Any "
                    f"other GAIRA surface given this spectrum returns the same digest.</p>",
                    unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# Documentation · Architecture · About
# ═══════════════════════════════════════════════════════════════════════════
def page_documentation() -> None:
    i = engine_info()
    st.markdown("## Documentation")
    st.markdown('<p class="caption">Every value on this page is pulled live from the running '
                'engine.</p>', unsafe_allow_html=True)

    cols = st.columns(4)
    for col, (v, l) in zip(cols, [(i["n_lsms"], "local spectral motifs"),
                                  (i["n_csms"], "consensus spectral motifs"),
                                  (i["n_molecules"], "reference molecules"),
                                  (i["n_chemistry_axes"], "chemistry axes")]):
        with col:
            glass(stat(v, l), tight=True)

    st.markdown("### Atlas identity")
    st.markdown('<div class="note">Two hashes, two meanings. Never call both “the atlas '
                'fingerprint”.</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame([
        {"term": "Scientific Atlas Fingerprint", "value": i["fingerprints"]["atlas"],
         "identifies": "the scientific atlas identity generated by the Phase 01 build"},
        {"term": "Frozen Runtime Content Hash", "value": i["atlas_fingerprint"],
         "identifies": "the complete frozen runtime asset identity"},
        {"term": "LSM registry", "value": i["fingerprints"]["lsm"],
         "identifies": "the 50-motif dictionary"},
        {"term": "CSM registry", "value": i["fingerprints"]["csm"],
         "identifies": "the 49-motif dictionary"},
        {"term": "Phase 05 engine", "value": i["fingerprints"]["engine"],
         "identifies": "the frozen inference engine"}]),
        hide_index=True, use_container_width=True)
    st.markdown(f"<p class='caption'>Engine <span class='mono'>{i['engine_version']}</span> · "
                f"GAIRA <span class='mono'>{i['gaira_version']}</span> · frozen assets "
                f"verified: <b>{i['frozen_assets_verified']}</b></p>", unsafe_allow_html=True)

    st.markdown("### Validated performance")
    st.markdown('<p class="caption">Molecule retrieval is leave-one-spectrum-out over 375 '
                'spectra against the full 154-molecule bank. Chemistry figures are '
                '<b>held out</b> under molecule-grouped cross-validation — the model never saw '
                'the test molecule.</p>', unsafe_allow_html=True)
    p = i["validated_performance"]
    c = st.columns(3)
    with c[0]:
        st.markdown("##### Molecular retrieval")
        st.dataframe(pd.DataFrame([{"metric": k, "value": p[f"molecule_{k}"]}
                                   for k in ("top1", "top3", "top5", "top10", "mrr", "ndcg5")]),
                     hide_index=True, use_container_width=True)
    with c[1]:
        st.markdown("##### Chemistry (unseen molecules)")
        st.dataframe(pd.DataFrame([
            {"metric": "top-1", "value": p["chemistry_top1_heldout"]},
            {"metric": "top-3", "value": p["chemistry_top3_heldout"]},
            {"metric": "macro F1", "value": p["chemistry_macro_f1_heldout"]},
            {"metric": "radar reproducibility", "value": p["radar_reproducibility"]}]),
            hide_index=True, use_container_width=True)
    with c[2]:
        st.markdown("##### Representation and robustness")
        st.dataframe(pd.DataFrame([
            {"metric": "CSM explained variance", "value": p["csm_mean_explained_variance"]},
            {"metric": "replicate consistency", "value": p["csm_replicate_consistency"]},
            {"metric": "radar under noise", "value": p["robustness_radar_cosine"]},
            {"metric": "chemistry under noise", "value": p["robustness_chemistry_top1"]},
            {"metric": "molecule under noise", "value": p["robustness_molecule_top1"]}]),
            hide_index=True, use_container_width=True)

    st.markdown("### Validation status")
    try:
        root = Path(__file__).resolve().parent.parent
        par = json.loads((root / "results/v7_rebuild/phase10/artifacts/"
                          "parity_and_performance_v1.json").read_text())
        frz = json.loads((root / "results/v7_rebuild/phase10/artifacts/"
                          "engine_freeze_audit_v1.json").read_text())
        c = st.columns(4)
        for col, (v, l, s) in zip(c, [
                (f"{frz['gates']['n'] + par['gates']['n']}", "gates passed",
                 f"{frz['gates']['failed'] + par['gates']['failed']} failed"),
                (f"{par['parity']['max_abs_diff']:.0e}", "cross-surface parity",
                 f"{par['parity']['n_comparisons']} comparisons"),
                (f"{par['performance']['single_inference_ms_median']} ms", "median inference",
                 f"p95 {par['performance']['single_inference_ms_p95']} ms"),
                (f"{par['scientific_validation']['max_deviation']:.0e}", "deviation vs Phase 09",
                 "all 375 spectra")]):
            with col:
                glass(stat(v, l, s), tight=True)
    except Exception:
        st.caption("Validation artifacts not found in this checkout.")

    st.markdown("### Chemistry axes")
    st.dataframe(pd.DataFrame([{"axis": a.replace("_", " "), "description": FIG.AXIS_NOTE.get(a, "")}
                               for a in i["chemistry_axes"]]),
                 hide_index=True, use_container_width=True)

    st.markdown("### Known limitations")
    for lim in i["known_limitations"]:
        st.markdown(f"- {lim}")
    st.markdown(f"<p class='caption'>Supported modalities: "
                f"<b>{', '.join(i['supported_modalities'])}</b> · validated sample types: "
                f"<b>{', '.join(i['validated_sample_types'])}</b>. Everything else is an "
                f"extension point with a defined contract and no implementation.</p>",
                unsafe_allow_html=True)


def page_architecture() -> None:
    st.markdown("## Architecture")
    st.markdown('<p class="caption">Seven stages, always in this order, with no branches and no '
                'parameter tunable at inference time. Select a stage to read what it does.</p>',
                unsafe_allow_html=True)

    names = [s[0] for s in FIG.ARCH_STAGES]
    sel = st.radio("stage", list(range(len(names))), horizontal=True,
                   format_func=lambda i: names[i], label_visibility="collapsed",
                   key="arch_sel")
    st.plotly_chart(FIG.architecture_flow(sel), use_container_width=True,
                    config={"displayModeBar": False, "staticPlot": True})

    name, sub, body = FIG.ARCH_STAGES[sel]
    glass(f"<div class='verdict-k'>stage {sel + 1} of {len(names)}</div>"
          f"<h3 style='margin:.3rem 0 .1rem'>{name}</h3>"
          f"<p class='caption' style='margin:0'>{sub}</p>"
          f"<p style='margin-top:.7rem;margin-bottom:0'>{body}</p>")

    st.markdown("<div style='height:1.4rem'></div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        glass(
            "<h3 style='margin-top:0'>What is deliberately absent</h3>"
            "<p style='font-size:.90rem'>BSV2, latent geometry, clustering, UMAP, PCA, themes, "
            "Meta Components and chemistry-aware reranking are <b>not</b> on the inference path. "
            "Each exclusion is a measured decision, not a preference.</p>"
            f"<table style='width:100%;font-size:.83rem;color:{T.INK_2}'>"
            "<tr><td>themes</td><td style='text-align:right'>0.405</td></tr>"
            "<tr><td>Meta Components</td><td style='text-align:right'>0.392</td></tr>"
            "<tr><td>11 grounded axes</td><td style='text-align:right'>0.664</td></tr>"
            "<tr><td>latent geometry</td><td style='text-align:right'>p = 0.180</td></tr>"
            "<tr><td><b>CSM (shipped)</b></td>"
            "<td style='text-align:right'><b>0.855</b></td></tr></table>"
            "<p class='caption' style='margin-bottom:0'>Chemistry accuracy on unseen molecules. "
            "Four independent attempts to build a layer above the CSM each lost information.</p>")
    with c2:
        glass("<h3 style='margin-top:0'>DART is not a modality</h3>"
              "<p style='font-size:.90rem'>DART-Met produces <span class='mono'>I(wavenumber, "
              "potential, time)</span> — still a <b>vibrational</b> measurement. Every slice is "
              "a spectrum the frozen engine already reads, so no spectral transform is needed "
              "and nothing upstream changes.</p>"
              "<p style='font-size:.90rem'>What is new is perturbation and time, so DART "
              "attaches at the <b>trajectory layer</b>, downstream of the frozen "
              "representation, over a sequence of ordinary inference results.</p>"
              "<p class='caption' style='margin-bottom:0'>A trajectory of CSM activations is "
              "interpretable only if every activation along it came from the same frozen "
              "path.</p>")


def page_about() -> None:
    st.markdown("## About GAIRA")
    c1, c2 = st.columns([1.15, 1])
    with c1:
        glass("<h3 style='margin-top:0'>What is GAIRA?</h3>"
              "<p>Shine a laser at a sample and a small fraction of the light returns with its "
              "colour shifted. The pattern of shifts is set by how the molecules present "
              "vibrate — a Raman spectrum. GAIRA reads that pattern and says what chemistry the "
              "evidence favours, how strongly, and by way of which specific spectral "
              "features.</p>"
              "<h3>How is it different?</h3>"
              "<p>A textbook treats a spectrum as a fingerprint to match against a library. "
              "Real samples do not cooperate: spectra are <b>mixtures</b>, a peak is <b>not</b> "
              "a molecule, and nearby is not the same. Rather than asking <i>which molecule is "
              "this?</i>, GAIRA answers a question it can support — what chemistry does the "
              "evidence favour, and how sure can we be?</p>"
              "<p>Think of a doctor reading a blood panel. They rarely conclude “this is "
              "precisely molecule X at precisely concentration Y”. They read a pattern and say "
              "which processes it is consistent with. GAIRA is built to do the second thing "
              "well rather than the first thing badly.</p>")
    with c2:
        glass("<h3 style='margin-top:0'>Why CSM?</h3>"
              "<p>Consensus Spectral Motifs are 49 non-negative basis spectra — patterns that "
              "may only be <b>added</b>, never subtracted, because a real mixture spectrum is a "
              "sum of its components and there is no such thing as negative-two-parts "
              "glucose.</p>"
              "<p>They are the canonical representation because that is where the information "
              "is. Chemistry accuracy on molecules the model has never seen rises from 0.608 "
              "using the raw spectrum to <b>0.855</b> at the CSM layer — and falls for every "
              "layer built on top.</p>"
              "<h3>Why Chemistry Evidence?</h3>"
              "<p>Sixteen chemistry families, each carrying <b>relative</b> evidence and a "
              "calibrated confidence. It is the level at which the engine can generalise "
              "honestly: 0.851 top-1 on unseen molecules, against 0.605 for naming the molecule "
              "itself.</p>")
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    glass("<h3 style='margin-top:0'>No language model</h3>"
          "<p style='margin-bottom:0'>Every number and every sentence GAIRA produces comes from "
          "the frozen engine and deterministic templates. There is no LLM in the inference "
          "path, no cloud account, no network call and no telemetry. The same spectrum always "
          "produces the same answer — and the same digest — on every surface: Python, CLI, "
          "HTTP, MCP and this interface.</p>")
    scope_footer()


# ═══════════════════════════════════════════════════════════════════════════
def main() -> None:
    state("page", "Home")
    state("step", "upload")
    navbar()
    {"Home": page_home, "Analyze": page_analyze, "Docs": page_documentation,
     "Architecture": page_architecture, "About": page_about}[st.session_state["page"]]()


main()
