"""GAIRA Manual Rescue Console — batch rescue workflow for blocked papers.

Working-set driven: select papers from queue -> process sequentially.
Upload -> Validate -> Stage -> Extract -> (optional) Ingest.
"""

from __future__ import annotations

import csv
import re
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

# ──────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────
V2_ROOT = Path("/Volumes/SSD_Rad/GAIRA_DATA/structured_evidence_v2")
REGISTRY = V2_ROOT / "registry"
REPORTS = V2_ROOT / "reports"
STAGING = V2_ROOT / "staging" / "rescue_packet"
BLOCKED_REGISTRY = REGISTRY / "master_blocked_registry.csv"
RESCUE_LOG = REPORTS / "phaseK1_rescue_status_update_log.csv"

# ──────────────────────────────────────────────────────────
# Extraction constants (mirrors literature_acquisition_pipeline.py)
# ──────────────────────────────────────────────────────────
ASSIGNMENT_VERBS = (
    "assigned to", "attributed to", "represents", "represent",
    "corresponds to", "corresponding to", "associated with",
    "due to", "arises from", "arising from",
)
NOISE_TERMS = (
    "doi", "copyright", "creativecommons", "license",
    "http", "www.", "et al",
)
TENTATIVE_TERMS = (
    "tentative", "possibly", "possible", "may be", "might be",
    "likely", "putative", "region", "broad",
)
SUBFAMILY_HINT_TERMS = (
    "amide", "protein", "lipid", "dna", "rna", "nucleic",
    "phenylalanine", "tyrosine", "tryptophan", "glycogen",
    "carbohydrate", "glycan", "carotenoid", "adenine", "guanine",
    "cytosine", "thymine", "citrate", "c-h", "c=c", "c-c", "c-n",
    "n-h", "o-h", "s-s", "c=o", "ch2", "ch3", "nh2", "phosphate",
    "sulfate", "cholesterol", "collagen", "hemoglobin", "glucose",
    "urea", "creatinine", "uric acid", "bilirubin", "dopamine",
    "serotonin", "amino acid", "peptide", "nucleotide", "saccharide",
    "stretching", "bending", "deformation", "vibration", "breathing",
    "symmetric", "asymmetric", "backbone", "ring", "methyl",
    "methylene", "carbonyl", "ester", "ether", "hydroxyl", "amine",
    "imine", "thiol", "disulfide", "phosphodiester", "glycosidic",
    "deoxyribose", "nucleobase",
)

TIER_ORDER = ["critical_A", "critical_B", "high_value_rescue_later", "secondary"]
COMPLETED_STATUSES = {
    "extracted_high_signal", "extracted_low_signal",
    "manual_failed", "rescued_and_ingested",
}

# ──────────────────────────────────────────────────────────
# Session state helpers
# ──────────────────────────────────────────────────────────
def _ws() -> list[str]:
    """Return the current working set (list of paper_ids)."""
    return st.session_state.setdefault("rescue_working_set", [])


def _ws_index() -> int:
    return st.session_state.setdefault("current_working_index", 0)


def _ws_set_index(i: int):
    ws = _ws()
    st.session_state["current_working_index"] = max(0, min(i, len(ws) - 1)) if ws else 0


# ──────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────
@st.cache_data(ttl=30)
def load_blocked() -> pd.DataFrame:
    if not BLOCKED_REGISTRY.exists():
        return pd.DataFrame()
    df = pd.read_csv(BLOCKED_REGISTRY, dtype=str).fillna("")
    tier_rank = {t: i for i, t in enumerate(TIER_ORDER)}
    df["_tier_rank"] = df["tier"].map(tier_rank).fillna(99).astype(int)
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0)
    return df.sort_values(["_tier_rank", "score"], ascending=[True, False])


def load_rescue_log() -> pd.DataFrame:
    if not RESCUE_LOG.exists():
        return pd.DataFrame(columns=[
            "timestamp", "paper_id", "action", "old_status",
            "new_status", "agv_rows", "note",
        ])
    return pd.read_csv(RESCUE_LOG, dtype=str).fillna("")


def append_rescue_log(paper_id: str, action: str, old_status: str,
                      new_status: str, agv_rows: int = 0, note: str = ""):
    row = {
        "timestamp": datetime.now().isoformat(),
        "paper_id": paper_id,
        "action": action,
        "old_status": old_status,
        "new_status": new_status,
        "agv_rows": str(agv_rows),
        "note": note,
    }
    write_header = not RESCUE_LOG.exists()
    with open(RESCUE_LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


def update_blocked_status(paper_id: str, new_status: str):
    df = pd.read_csv(BLOCKED_REGISTRY, dtype=str).fillna("")
    mask = df["paper_id"] == paper_id
    if mask.any():
        df.loc[mask, "access_status"] = new_status
        df.to_csv(BLOCKED_REGISTRY, index=False)
        st.cache_data.clear()


# ──────────────────────────────────────────────────────────
# Extraction engine (unchanged from K1)
# ──────────────────────────────────────────────────────────
def extract_pdf_text(pdf_path: Path) -> str:
    try:
        result = subprocess.run(
            ["pdftotext", str(pdf_path), "-"],
            capture_output=True, text=True, check=True,
        )
        return result.stdout
    except Exception:
        return ""


def classify_assignment(meaning: str, original: str, method: str):
    cleaned = re.sub(r"\s+", " ", meaning).strip(" ,;:.")
    lowered = cleaned.lower()
    orig_lower = original.lower()
    if (
        len(cleaned) < 4
        or sum(c.isdigit() for c in cleaned) >= max(3, len(cleaned) // 3)
        or any(t in orig_lower for t in NOISE_TERMS)
        or cleaned.count("(") != cleaned.count(")")
    ):
        return "reject_noise"
    if re.search(r"^(figure|fig\.|table|scheme)\s", lowered):
        return "reject_noise"
    if method == "text_regex" and not any(t in lowered for t in SUBFAMILY_HINT_TERMS):
        return "mention_only"
    if method == "table_assignment" and not any(t in lowered for t in SUBFAMILY_HINT_TERMS):
        return "reject_noise"
    if any(t in lowered for t in TENTATIVE_TERMS):
        return "validated_secondary"
    if method == "table_assignment":
        return "validated_secondary"
    return "validated_primary"


def is_false_positive(text: str, peak: float) -> bool:
    lowered = text.lower()
    for pat in [
        r"\d+\s*mW", r"\d+\s*nm\b", r"\d+\s*RPM", r"\d+\s*°C",
        r"\d+\s*min", r"\d+\s*hours?", r"\d+\s*Torr",
        r"order of magnitude", r"acquisition time", r"randomness",
    ]:
        if re.search(pat, lowered):
            return True
    if re.search(rf"{int(peak)}\s*(?:nm|mW|RPM|°C|min)\b", text):
        return True
    return False


def _parse_peak(peak_str: str) -> float:
    """Parse a peak string, handling ranges like '725-735' by taking midpoint."""
    if "\u2013" in peak_str or "\u2212" in peak_str or "-" in peak_str:
        parts = re.split(r"[\u2013\u2212\-]", peak_str)
        nums = [float(p.strip()) for p in parts if re.match(r"^\d{3,4}(?:\.\d+)?$", p.strip())]
        if len(nums) == 2:
            return (nums[0] + nums[1]) / 2
    return float(peak_str)


# Regex fragment matching cm-1 in various unicode forms
_CM1 = r"(?:cm[\-\u2212\u2013]?\s*1|cm[\-\u2212\u2013]?1|cm-1|cm\u22121|cm\u20131)"

# Peak pattern: single peak or range (e.g. 725–735)
_PEAK = r"(?P<peak>\d{3,4}(?:[\u2013\u2212\-]\d{3,4})?(?:\.\d+)?)"

# Assignment verb phrases
_AVERBS = (
    r"(?:explicitly\s+)?(?:assigned to|attributed to|represents?|corresponds?\s+to"
    r"|corresponding\s+to|associated with|due to|aris(?:es?|ing)\s+from)"
)


def run_extraction(text: str, paper_id: str) -> list[dict]:
    rows: list[dict] = []
    seen: set[tuple] = set()

    def _add(peak: float, meaning: str, evidence: str, method: str):
        if peak < 100 or peak > 4000:
            return
        meaning = re.sub(r"\s+", " ", meaning).strip(" ,;")
        evidence = re.sub(r"\s+", " ", evidence).strip()
        if is_false_positive(evidence, peak):
            return
        key = (round(peak), meaning.lower()[:60])
        if key in seen:
            return
        seen.add(key)
        cls = classify_assignment(meaning, evidence, method)
        if cls == "reject_noise":
            return
        rows.append({"peak_cm": peak, "meaning": meaning[:120],
                      "evidence_text": evidence[:300],
                      "method": method, "classification": cls})

    # ── Pass 1: same-sentence explicit assignment ──
    for m in re.finditer(
        _PEAK + r"\s*" + _CM1 + r"\s*(?:[^.\n]{0,80}?)" + _AVERBS
        + r"\s+(?P<meaning>[^.;\n]{3,120})",
        text, re.I,
    ):
        peak = _parse_peak(m.group("peak"))
        _add(peak, m.group("meaning"), m.group(0), "text_assignment")

    # ── Pass 1b: cross-sentence assignment bridging ──
    # Finds peaks near a sentence boundary where the assignment verb
    # starts in the next sentence (e.g. "... 725 cm-1 (Fig. 1).\nIt
    # has been assigned to ...").
    # The tail allows periods inside parenthetical refs like "(Fig. 1)".
    for m in re.finditer(
        _PEAK + r"\s*" + _CM1 + r"(?P<tail>.{0,80}?)\.\s+"
        + r"(?:It\s+(?:has\s+been\s+|is\s+|was\s+)|This\s+(?:band\s+|peak\s+|feature\s+)"
        + r"?(?:has\s+been\s+|is\s+|was\s+)?)?" + _AVERBS
        + r"\s+(?P<meaning>[^.;]{3,150})",
        text, re.I,
    ):
        peak = _parse_peak(m.group("peak"))
        meaning = m.group("meaning")
        _add(peak, meaning, m.group(0), "text_assignment_cross")

    # ── Pass 1c: "assigned to" before the peak ──
    # Patterns like: "features ... assigned to purine ... at 668, 730 cm-1"
    for m in re.finditer(
        _AVERBS + r"\s+(?P<meaning>[^.;]{3,100}?)"
        + r"(?:\bat\s+)?" + _PEAK + r"\s*" + _CM1,
        text, re.I,
    ):
        peak = _parse_peak(m.group("peak"))
        _add(peak, m.group("meaning"), m.group(0), "text_assignment_rev")

    # ── Pass 2: table-like rows ──
    for line in text.splitlines():
        cleaned = re.sub(r"\s+", " ", line).strip()
        m = re.match(
            r"^(?P<peak>\d{3,4}(?:\.\d+)?)\s+"
            r"(?P<meaning>[A-Za-z][A-Za-z0-9 ,()=/\-\u2013\u2014]{3,80})$",
            cleaned,
        )
        if not m:
            continue
        peak = float(m.group("peak"))
        meaning = m.group("meaning").strip(" ,;")
        if len(meaning.split()) > 12 or re.search(r"\d{2,}", meaning):
            continue
        _add(peak, meaning, cleaned, "table_assignment")

    # ── Pass 3: contextual regex (peak + nearby biochemical context) ──
    for m in re.finditer(
        _PEAK + r"\s*" + _CM1 + r"(?P<context>[^.;\n]{0,120})",
        text, re.I,
    ):
        peak = _parse_peak(m.group("peak"))
        ctx = re.sub(r"\s+", " ", m.group("context")).strip(" ,;")
        if not ctx or is_false_positive(ctx, peak):
            continue
        key = (round(peak), ctx.lower()[:60])
        if key in seen:
            continue
        # Skip if an assignment verb is present (already handled in Pass 1)
        if any(v in ctx.lower() for v in ASSIGNMENT_VERBS):
            continue
        if not any(t in ctx.lower() for t in SUBFAMILY_HINT_TERMS):
            continue
        seen.add(key)
        cls = classify_assignment(ctx, ctx, "text_regex")
        if cls == "reject_noise":
            continue
        rows.append({"peak_cm": peak, "meaning": ctx[:120],
                      "evidence_text": ctx[:300],
                      "method": "text_regex", "classification": cls})
    return rows


# ──────────────────────────────────────────────────────────
# Render: paper detail + upload + extract (shared by both modes)
# ──────────────────────────────────────────────────────────
def render_paper_rescue(selected_id: str, df: pd.DataFrame, auto_advance: bool):
    """Render the full rescue workflow for a single paper.

    Returns True if an action was taken and auto-advance is requested.
    """
    match = df[df["paper_id"] == selected_id]
    if match.empty:
        st.error(f"Paper {selected_id} not found in registry.")
        return False
    paper = match.iloc[0]
    should_advance = False

    # ── Detail header ──
    st.subheader(paper.get("title", selected_id))
    col_a, col_b = st.columns(2)
    with col_a:
        doi = paper.get("doi", "N/A")
        url = paper.get("canonical_url", "")
        st.markdown(f"**DOI:** [{doi}]({url})" if url else f"**DOI:** {doi}")
        st.markdown(f"**Publisher:** {paper.get('publisher', '')} | **Journal:** {paper.get('journal', '')}")
        st.markdown(f"**Tier:** {paper.get('tier', '')} | **Score:** {paper.get('score', '')}")
        st.markdown(f"**Modality:** {paper.get('modality', '')} | **Biosample:** {paper.get('biosample', '')}")
    with col_b:
        st.markdown(f"**Condition:** {paper.get('condition', '')}")
        st.markdown(f"**Expected yield:** {paper.get('expected_AGV_yield', '')}")
        st.markdown(f"**Gap filled:** {paper.get('gap_filled', '')}")
        st.markdown(f"**Access status:** {paper.get('access_status', '')}")

    target_dir = STAGING / selected_id
    st.markdown(f"**Upload target:** `{target_dir}`")
    st.divider()

    # ── Upload ──
    st.subheader("Upload Files")
    col_u1, col_u2 = st.columns(2)
    with col_u1:
        manuscript_file = st.file_uploader(
            "Manuscript PDF", type=["pdf"], key=f"ms_{selected_id}")
        supplement_file = st.file_uploader(
            "Supplementary PDF (optional)", type=["pdf"], key=f"si_{selected_id}")
    with col_u2:
        text_file = st.file_uploader(
            "Text file (manual copy, optional)", type=["txt"], key=f"txt_{selected_id}")
        figures_file = st.file_uploader(
            "Figures archive (optional)", type=["pdf", "zip"], key=f"fig_{selected_id}")
    st.divider()

    # ── Validate + Stage ──
    if st.button("Validate + Stage", type="primary", key=f"validate_{selected_id}"):
        if not manuscript_file and not text_file:
            st.error("Upload at least a manuscript PDF or text file.")
        else:
            target_dir.mkdir(parents=True, exist_ok=True)
            saved_files: list[str] = []
            text_content = ""

            if manuscript_file:
                ms_path = target_dir / f"{selected_id}_manuscript.pdf"
                ms_path.write_bytes(manuscript_file.getvalue())
                saved_files.append(str(ms_path))
                text_content = extract_pdf_text(ms_path)
            if supplement_file:
                si_path = target_dir / f"{selected_id}_supplement.pdf"
                si_path.write_bytes(supplement_file.getvalue())
                saved_files.append(str(si_path))
                text_content += "\n\n--- SUPPLEMENT ---\n\n" + extract_pdf_text(si_path)
            if text_file:
                txt_path = target_dir / f"{selected_id}_text.txt"
                txt_content = text_file.getvalue().decode("utf-8", errors="replace")
                txt_path.write_text(txt_content)
                saved_files.append(str(txt_path))
                if not text_content:
                    text_content = txt_content
            if figures_file:
                ext = figures_file.name.rsplit(".", 1)[-1]
                fig_path = target_dir / f"{selected_id}_figures.{ext}"
                fig_path.write_bytes(figures_file.getvalue())
                saved_files.append(str(fig_path))

            if text_content:
                (target_dir / f"{selected_id}_full_text.txt").write_text(text_content)

            word_count = len(text_content.split()) if text_content else 0
            raman_hits = len(re.findall(r"(?i)raman|sers|cm[\-\u2212\u2013]?\s*1", text_content))

            st.success(f"Staged {len(saved_files)} file(s)")
            c1, c2, c3 = st.columns(3)
            c1.metric("Words", f"{word_count:,}")
            c2.metric("Parseable", "Yes" if word_count > 500 else "No")
            c3.metric("Raman/SERS hits", raman_hits)

            update_blocked_status(selected_id, "rescued_local_asset_present")
            append_rescue_log(selected_id, "staged", paper.get("access_status", ""),
                              "rescued_local_asset_present",
                              note=f"{word_count} words, {raman_hits} raman mentions")
            st.session_state[f"staged_text_{selected_id}"] = text_content
            st.session_state[f"staged_{selected_id}"] = True

    # ── Load staged text from disk if available ──
    full_txt_on_disk = target_dir / f"{selected_id}_full_text.txt"
    if full_txt_on_disk.exists() and f"staged_text_{selected_id}" not in st.session_state:
        st.session_state[f"staged_text_{selected_id}"] = full_txt_on_disk.read_text()
        st.session_state[f"staged_{selected_id}"] = True

    # ── Run Extraction ──
    if st.session_state.get(f"staged_{selected_id}"):
        st.divider()
        st.subheader("Extraction")
        if st.button("Run Extraction", key=f"extract_{selected_id}"):
            text = st.session_state.get(f"staged_text_{selected_id}", "")
            if not text:
                st.error("No text content available.")
            else:
                assignments = run_extraction(text, selected_id)
                agv = [a for a in assignments
                       if a["classification"] in ("validated_primary", "validated_secondary")]
                agv_count = len(agv)
                yield_class = ("high" if agv_count >= 10 else "medium" if agv_count >= 3
                               else "low" if agv_count >= 1 else "none")

                c1, c2, c3 = st.columns(3)
                c1.metric("AGV Rows", agv_count)
                c2.metric("Yield", yield_class)
                c3.metric("Primary", sum(1 for a in agv if a["classification"] == "validated_primary"))

                if agv:
                    st.dataframe(pd.DataFrame(agv)[["peak_cm", "meaning", "method", "classification"]],
                                 use_container_width=True)
                    ext_path = target_dir / f"{selected_id}_extraction.csv"
                    with open(ext_path, "w", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=list(agv[0].keys()))
                        w.writeheader()
                        w.writerows(agv)

                new_status = "extracted_high_signal" if agv_count >= 5 else "extracted_low_signal"
                update_blocked_status(selected_id, new_status)
                append_rescue_log(selected_id, "extracted", "rescued_local_asset_present",
                                  new_status, agv_rows=agv_count, note=f"{yield_class} yield")
                if not agv:
                    st.warning("No assignment-grade evidence found. Consider SI/figure follow-up.")

    # ── Quick actions ──
    st.divider()
    cols = st.columns(4)
    with cols[0]:
        if st.button("Skip for now", key=f"skip_{selected_id}"):
            should_advance = auto_advance
            st.info("Skipped.")
    with cols[1]:
        if st.button("Manual failed", key=f"fail_{selected_id}"):
            update_blocked_status(selected_id, "manual_failed")
            append_rescue_log(selected_id, "manual_failed",
                              paper.get("access_status", ""), "manual_failed")
            st.warning("Marked as manual-failed.")
            should_advance = auto_advance
    with cols[2]:
        if st.button("Defer", key=f"later_{selected_id}"):
            update_blocked_status(selected_id, "deferred")
            append_rescue_log(selected_id, "deferred",
                              paper.get("access_status", ""), "deferred")
            st.info("Deferred.")
            should_advance = auto_advance
    with cols[3]:
        if st.button("Rescued + Ingested", key=f"ingested_{selected_id}"):
            update_blocked_status(selected_id, "rescued_and_ingested")
            append_rescue_log(selected_id, "ingested",
                              paper.get("access_status", ""), "rescued_and_ingested",
                              agv_rows=st.session_state.get(f"agv_count_{selected_id}", 0))
            st.success("Marked as rescued and ingested.")
            should_advance = auto_advance

    return should_advance


# ══════════════════════════════════════════════════════════
# APP LAYOUT
# ══════════════════════════════════════════════════════════
st.set_page_config(page_title="GAIRA Manual Rescue", layout="wide")
st.title("GAIRA Manual Rescue Console")

# Working set indicator in header
ws = _ws()
if ws:
    st.caption(f"Working Set: {len(ws)} papers | "
               f"Current: {_ws_index() + 1}/{len(ws)}")
else:
    st.caption("No working set. Select papers from the Rescue Queue.")

tab_queue, tab_detail, tab_history = st.tabs([
    "Rescue Queue", "Rescue + Extract", "Rescue History",
])

# ══════════════════════════════════════════════════════════
# TAB 1: RESCUE QUEUE (with multi-select + working set)
# ══════════════════════════════════════════════════════════
with tab_queue:
    df = load_blocked()
    if df.empty:
        st.warning("No blocked registry found.")
        st.stop()

    st.subheader(f"Blocked Papers ({len(df)} total)")

    # ── Filters ──
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        tier_opts = [t for t in TIER_ORDER if t in df["tier"].unique()]
        tier_filter = st.multiselect(
            "Tier", tier_opts,
            default=[t for t in ["critical_A", "critical_B"] if t in tier_opts])
    with col2:
        pub_filter = st.multiselect("Publisher", sorted(df["publisher"].unique()))
    with col3:
        mod_filter = st.multiselect("Modality", sorted(df["modality"].unique()))
    with col4:
        access_filter = st.multiselect("Access Status", sorted(df["access_status"].unique()))

    filtered = df.copy()
    if tier_filter:
        filtered = filtered[filtered["tier"].isin(tier_filter)]
    if pub_filter:
        filtered = filtered[filtered["publisher"].isin(pub_filter)]
    if mod_filter:
        filtered = filtered[filtered["modality"].isin(mod_filter)]
    if access_filter:
        filtered = filtered[filtered["access_status"].isin(access_filter)]

    # ── Metrics ──
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Showing", len(filtered))
    c2.metric("Critical A", len(filtered[filtered["tier"] == "critical_A"]))
    c3.metric("Critical B", len(filtered[filtered["tier"] == "critical_B"]))
    c4.metric("High Value", len(filtered[filtered["tier"] == "high_value_rescue_later"]))
    c5.metric("Secondary", len(filtered[filtered["tier"] == "secondary"]))

    # ── Selectable table ──
    display_cols = [
        "paper_id", "title", "publisher", "tier", "modality",
        "biosample", "condition", "expected_AGV_yield",
        "access_status", "gap_filled",
    ]
    show_cols = [c for c in display_cols if c in filtered.columns]
    table_df = filtered[show_cols].reset_index(drop=True).copy()
    table_df.insert(0, "select", False)

    edited = st.data_editor(
        table_df,
        use_container_width=True,
        height=450,
        column_config={
            "select": st.column_config.CheckboxColumn("Sel", width="small", default=False),
            "title": st.column_config.TextColumn("Title", width="large"),
            "tier": st.column_config.TextColumn("Tier", width="small"),
        },
        disabled=[c for c in show_cols],  # only checkbox is editable
        key="queue_editor",
    )

    # ── Working set controls ──
    selected_ids = list(edited.loc[edited["select"], "paper_id"])

    st.divider()
    ws_col1, ws_col2, ws_col3, ws_col4 = st.columns(4)

    with ws_col1:
        if st.button(f"Add {len(selected_ids)} selected to Working Set",
                     disabled=len(selected_ids) == 0):
            current = _ws()
            merged = list(dict.fromkeys(current + selected_ids))  # dedup, preserve order
            st.session_state["rescue_working_set"] = merged
            st.session_state["current_working_index"] = 0
            st.success(f"Working set: {len(merged)} papers")
            st.rerun()

    with ws_col2:
        if st.button("Add ALL filtered", disabled=len(filtered) == 0):
            all_ids = filtered["paper_id"].tolist()
            current = _ws()
            merged = list(dict.fromkeys(current + all_ids))
            st.session_state["rescue_working_set"] = merged
            st.session_state["current_working_index"] = 0
            st.success(f"Working set: {len(merged)} papers")
            st.rerun()

    with ws_col3:
        ws_size = len(_ws())
        st.markdown(f"**Working set: {ws_size} papers**")

    with ws_col4:
        if st.button("Clear Working Set", disabled=ws_size == 0):
            st.session_state["rescue_working_set"] = []
            st.session_state["current_working_index"] = 0
            st.rerun()


# ══════════════════════════════════════════════════════════
# TAB 2: RESCUE + EXTRACT (working set or manual)
# ══════════════════════════════════════════════════════════
with tab_detail:
    df = load_blocked()
    if df.empty:
        st.stop()

    ws = _ws()
    has_ws = len(ws) > 0

    # ── Mode toggle ──
    use_ws = st.toggle("Use Working Set", value=has_ws, disabled=not has_ws,
                       key="use_ws_toggle")

    if use_ws and ws:
        # ── Working set mode ──
        idx = _ws_index()
        total = len(ws)

        # Clamp index
        if idx >= total:
            idx = total - 1
            _ws_set_index(idx)

        # Resolve completed status for each paper in working set
        completed_ids = set()
        for pid in ws:
            row = df[df["paper_id"] == pid]
            if not row.empty and row.iloc[0].get("access_status", "") in COMPLETED_STATUSES:
                completed_ids.add(pid)
        remaining = total - len(completed_ids)

        # ── Progress bar + metrics ──
        progress = len(completed_ids) / total if total else 0
        st.progress(progress, text=f"Paper {idx + 1} of {total} | "
                    f"Completed: {len(completed_ids)} | Remaining: {remaining}")

        # ── Navigation ──
        nav_cols = st.columns([1, 1, 2, 1, 1])
        with nav_cols[0]:
            if st.button("Previous", disabled=idx <= 0, key="nav_prev"):
                _ws_set_index(idx - 1)
                st.rerun()
        with nav_cols[1]:
            if st.button("Next", disabled=idx >= total - 1, key="nav_next"):
                _ws_set_index(idx + 1)
                st.rerun()
        with nav_cols[2]:
            jump = st.number_input("Jump to", min_value=1, max_value=total,
                                   value=idx + 1, key="nav_jump")
            if jump != idx + 1:
                _ws_set_index(jump - 1)
                st.rerun()
        with nav_cols[3]:
            if st.button("Remove current", key="ws_remove_current"):
                ws.pop(idx)
                st.session_state["rescue_working_set"] = ws
                _ws_set_index(min(idx, len(ws) - 1))
                st.rerun()
        with nav_cols[4]:
            if st.button("Remove completed", key="ws_remove_completed"):
                ws_new = [p for p in ws if p not in completed_ids]
                st.session_state["rescue_working_set"] = ws_new
                _ws_set_index(0)
                st.rerun()

        st.divider()

        # Auto-advance toggle
        auto_advance = st.toggle("Auto-advance after action", value=True, key="auto_advance")

        # Render current paper
        selected_id = ws[idx]
        advanced = render_paper_rescue(selected_id, df, auto_advance)
        if advanced and idx < total - 1:
            _ws_set_index(idx + 1)
            st.rerun()

    else:
        # ── Manual selection mode (fallback) ──
        if has_ws:
            st.info("Working set available but toggled off. Using manual selection.")
        else:
            st.info("No working set. Select papers from the Rescue Queue tab, "
                    "or pick one below.")

        paper_ids = df["paper_id"].tolist()
        selected_id = st.selectbox("Select paper", paper_ids, index=0,
                                   key="manual_paper_select")
        render_paper_rescue(selected_id, df, auto_advance=False)


# ══════════════════════════════════════════════════════════
# TAB 3: RESCUE HISTORY
# ══════════════════════════════════════════════════════════
with tab_history:
    st.subheader("Rescue History")

    log_df = load_rescue_log()
    df = load_blocked()

    if not df.empty:
        status_counts = df["access_status"].value_counts()
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total blocked", len(df))
        c2.metric("Awaiting", int(status_counts.get("browser_only", 0))
                   + int(status_counts.get("metadata_only", 0))
                   + int(status_counts.get("awaiting_manual_download", 0)))
        c3.metric("Rescued", int(status_counts.get("rescued_local_asset_present", 0)))
        c4.metric("Extracted", int(status_counts.get("extracted_high_signal", 0))
                   + int(status_counts.get("extracted_low_signal", 0)))
        c5.metric("Ingested", int(status_counts.get("rescued_and_ingested", 0)))

    if log_df.empty:
        st.info("No rescue actions logged yet.")
    else:
        st.dataframe(
            log_df.sort_values("timestamp", ascending=False).reset_index(drop=True),
            use_container_width=True)

    if not df.empty:
        st.divider()
        st.subheader("Still Waiting")
        waiting_statuses = {"browser_only", "metadata_only", "token_denied",
                           "human_only", "awaiting_manual_download"}
        waiting = df[df["access_status"].isin(waiting_statuses)]
        if not waiting.empty:
            show = ["paper_id", "tier", "publisher", "modality",
                    "expected_AGV_yield", "access_status"]
            show = [c for c in show if c in waiting.columns]
            st.dataframe(waiting[show].head(50).reset_index(drop=True),
                         use_container_width=True)
        else:
            st.success("All papers have been attempted!")
