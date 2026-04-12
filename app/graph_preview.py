"""GAIRA Embedded Graph Preview v1.2 — cleaner layout, emphasized anchor, better labels.

Generates an interactive HTML graph preview from query results.
"""

from __future__ import annotations

from collections import Counter
from pyvis.network import Network

from graph.phaseC1_query_engine import GraphResult


# ── node styling ────────────────────────────────────────────
_NODE_STYLES = {
    "Condition":        {"color": "#FDD835", "shape": "diamond"},
    "Motif":            {"color": "#66BB6A", "shape": "dot"},
    "EvidenceRow":      {"color": "#90CAF9", "shape": "dot"},
    "FunctionalGroup":  {"color": "#FFCC80", "shape": "triangle"},
    "BondMoiety":       {"color": "#80CBC4", "shape": "triangle"},
    "VibrationalMode":  {"color": "#D7CCC8", "shape": "triangle"},
    "BiochemicalTheme": {"color": "#4DD0E1", "shape": "star"},
    "Biomolecule":      {"color": "#CE93D8", "shape": "star"},
    "Peak":             {"color": "#EF5350", "shape": "square"},
}

# ── edge styling ────────────────────────────────────────────
_EDGE_STYLES = {
    "DIRECTLY_SUPPORTS_THEME":      {"color": "#7B1FA2", "width": 2.0, "dashes": False},
    "DIRECTLY_SUPPORTS_BIOMOLECULE": {"color": "#AD1457", "width": 1.8, "dashes": False},
    "INFERRED_SUPPORTS_THEME":      {"color": "#CE93D8", "width": 1.5, "dashes": [5, 5]},
    "HAS_FUNCTIONAL_GROUP":         {"color": "#EF6C00", "width": 1.2, "dashes": False},
    "LINKED_TO":                    {"color": "#E53935", "width": 2.5, "dashes": False},
    "PART_OF_MOTIF":                {"color": "#43A047", "width": 1.0, "dashes": False},
}

_MAX_EVIDENCE = 12
_MAX_NODES = 50


def _clean_label(text: str, max_len: int = 20) -> str:
    """Make a node label readable: replace underscores, truncate."""
    text = text.replace("_", " ")
    if len(text) > max_len:
        return text[:max_len - 1] + "..."
    return text


def build_preview_graph(gr: GraphResult) -> str | None:
    """Build a PyVis HTML graph preview from a GraphResult."""
    if gr.total_evidence_rows == 0:
        return None

    net = Network(height="460px", width="100%", bgcolor="#16213e", font_color="#e0e0e0",
                  directed=True, notebook=False)
    net.barnes_hut(gravity=-4000, central_gravity=0.35, spring_length=140, spring_strength=0.04)

    added = set()

    def add_node(nid: str, label: str, ntype: str, size: int, title: str = ""):
        if nid in added or len(added) >= _MAX_NODES:
            return False
        style = _NODE_STYLES.get(ntype, {"color": "#888", "shape": "dot"})
        net.add_node(nid, label=_clean_label(label), title=f"{ntype}: {title or label}",
                     color=style["color"], size=size, shape=style["shape"],
                     font={"size": 11, "color": "#e0e0e0"})
        added.add(nid)
        return True

    def add_edge(src: str, tgt: str, rel: str):
        if src not in added or tgt not in added:
            return
        style = _EDGE_STYLES.get(rel, {"color": "#555", "width": 1.0, "dashes": False})
        net.add_edge(src, tgt, title=rel.replace("_", " "),
                     color=style["color"], width=style["width"],
                     dashes=style["dashes"], arrows="to")

    entity = gr.matched_entity

    # ── 1. ANCHOR NODE (large, prominent) ───────────────────
    if gr.query_type == "condition":
        add_node(f"ANCHOR", entity.replace("_", " "), "Condition", 40, entity)
    elif gr.query_type == "peak":
        add_node(f"ANCHOR", f"{entity} cm-1", "Peak", 35, f"Peak at {entity} cm-1")
    elif gr.query_type == "theme":
        add_node(f"ANCHOR", entity.replace("_", " "), "BiochemicalTheme", 35, entity)
    elif gr.query_type == "chemistry":
        add_node(f"ANCHOR", entity.replace("_", " "), "FunctionalGroup", 35, entity)

    # ── 2. THEMES (inner ring) ──────────────────────────────
    theme_counts = Counter(r.get("theme") or r.get("direct_theme") for r in gr.raw_records
                           if r.get("theme") or r.get("direct_theme"))
    for theme, cnt in theme_counts.most_common(5):
        tid = f"t_{theme}"
        size = min(28, 16 + cnt // 5)
        add_node(tid, theme, "BiochemicalTheme", size, f"{theme} ({cnt} evidence)")

    # ── 3. MOTIFS (middle ring) ─────────────────────────────
    motif_counts = Counter(r.get("motif_id") for r in gr.raw_records if r.get("motif_id"))
    motif_labels = {}
    for r in gr.raw_records:
        mid = r.get("motif_id")
        if mid and mid not in motif_labels:
            motif_labels[mid] = r.get("motif_subfamily") or mid
    for mid, cnt in motif_counts.most_common(6):
        size = min(24, 14 + cnt // 3)
        add_node(mid, motif_labels.get(mid, mid), "Motif", size, f"{motif_labels.get(mid,mid)} ({cnt} members)")

    # ── 4. BIOMOLECULES ─────────────────────────────────────
    bio_counts = Counter(r.get("biomolecule") for r in gr.raw_records if r.get("biomolecule"))
    for bio, cnt in bio_counts.most_common(4):
        add_node(f"b_{bio}", bio, "Biomolecule", min(22, 14 + cnt // 4), f"{bio} ({cnt})")

    # ── 5. FUNCTIONAL GROUPS ────────────────────────────────
    fg_counts = Counter(r.get("functional_group") for r in gr.raw_records if r.get("functional_group"))
    for fg, cnt in fg_counts.most_common(3):
        add_node(f"f_{fg}", fg, "FunctionalGroup", min(20, 14 + cnt // 5), f"{fg} ({cnt})")

    # ── 6. EVIDENCE ROWS (outer ring, sampled) ──────────────
    ev_seen = set()
    ev_n = 0
    for r in gr.raw_records:
        eid = r.get("evidence_id")
        if not eid or eid in ev_seen or ev_n >= _MAX_EVIDENCE:
            continue
        pk = r.get("peak_cm") or r.get("wavenumber") or ""
        meaning = (r.get("meaning") or "")[:35]
        label = str(pk) if pk else meaning[:12]
        if add_node(eid, label, "EvidenceRow", 10, f"{pk} cm-1: {meaning}"):
            ev_n += 1
            ev_seen.add(eid)

    # ── BUILD EDGES ─────────────────────────────────────────
    seen_edges = set()

    for r in gr.raw_records:
        eid = r.get("evidence_id", "")
        mid = r.get("motif_id", "")
        theme = r.get("theme") or r.get("direct_theme") or ""
        bio = r.get("biomolecule", "")
        fg = r.get("functional_group", "")

        # Evidence -> Theme
        if eid in added and theme:
            tid = f"t_{theme}"
            ek = (eid, tid)
            if ek not in seen_edges:
                add_edge(eid, tid, "DIRECTLY_SUPPORTS_THEME")
                seen_edges.add(ek)

        # Evidence -> Motif
        if eid in added and mid in added:
            ek = (eid, mid)
            if ek not in seen_edges:
                add_edge(eid, mid, "PART_OF_MOTIF")
                seen_edges.add(ek)

        # Evidence -> Biomolecule
        if eid in added and bio:
            bid = f"b_{bio}"
            ek = (eid, bid)
            if ek not in seen_edges:
                add_edge(eid, bid, "DIRECTLY_SUPPORTS_BIOMOLECULE")
                seen_edges.add(ek)

        # Evidence -> FG
        if eid in added and fg:
            fid = f"f_{fg}"
            ek = (eid, fid)
            if ek not in seen_edges:
                add_edge(eid, fid, "HAS_FUNCTIONAL_GROUP")
                seen_edges.add(ek)

        # Motif -> Anchor (condition queries)
        if mid in added and gr.query_type == "condition":
            ek = (mid, "ANCHOR")
            if ek not in seen_edges:
                add_edge(mid, "ANCHOR", "LINKED_TO")
                seen_edges.add(ek)

        # Inferred FG -> Theme
        inf_theme = r.get("inferred_theme", "")
        if fg and inf_theme:
            fid = f"f_{fg}"
            tid = f"t_{inf_theme}"
            ek = (fid, tid)
            if fid in added and tid in added and ek not in seen_edges:
                add_edge(fid, tid, "INFERRED_SUPPORTS_THEME")
                seen_edges.add(ek)

    return net.generate_html()


# ── Legend HTML ──────────────────────────────────────────────
LEGEND_HTML = """
<div style="background:#16213e; border-radius:8px; padding:10px 14px; margin:4px 0;">
  <div style="display:flex; flex-wrap:wrap; gap:14px; font-size:13px; color:#e0e0e0;">
    <span><span style="color:#FDD835;">&#9670;</span> Condition</span>
    <span><span style="color:#66BB6A;">&#9679;</span> Motif</span>
    <span><span style="color:#90CAF9;">&#9679;</span> Evidence</span>
    <span><span style="color:#4DD0E1;">&#9733;</span> Theme</span>
    <span><span style="color:#CE93D8;">&#9733;</span> Biomolecule</span>
    <span><span style="color:#FFCC80;">&#9650;</span> Func. Group</span>
    <span><span style="color:#EF5350;">&#9632;</span> Peak</span>
  </div>
  <div style="font-size:11px; color:#9e9e9e; margin-top:6px;">
    Solid line = direct evidence &nbsp;&bull;&nbsp; Dashed line = inferred mapping &nbsp;&bull;&nbsp; Red thick line = condition link
  </div>
  <div style="font-size:11px; color:#757575; margin-top:2px;">
    Simplified preview. Use Neo4j Browser for full graph inspection.
  </div>
</div>
"""
