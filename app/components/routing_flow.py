from __future__ import annotations

import streamlit as st


def render_routing_flow(predicted_family: str | None) -> None:
    family_label = predicted_family or "legacy"
    st.markdown(
        f"""
        <div style="display:grid;grid-template-columns:1fr;gap:10px;">
          <div style="border:1px solid #cbd5e1;border-radius:12px;padding:12px;background:#f8fafc;"><strong>Input metadata</strong><br/><span style="color:#475569;">sample type, modality, substrate, use-case</span></div>
          <div style="border:2px solid #2563eb;border-radius:12px;padding:12px;background:#eff6ff;"><strong>Query family assignment</strong><br/><span style="color:#1d4ed8;">{family_label}</span></div>
          <div style="border:1px solid #cbd5e1;border-radius:12px;padding:12px;background:#f8fafc;"><strong>Context / support buckets</strong><br/><span style="color:#475569;">domain context, tier-2 support, knowledge support</span></div>
          <div style="border:1px solid #cbd5e1;border-radius:12px;padding:12px;background:#f8fafc;"><strong>Inference engine</strong><br/><span style="color:#475569;">retrieval, reranking, evidence assembly</span></div>
          <div style="border:1px solid #cbd5e1;border-radius:12px;padding:12px;background:#f8fafc;"><strong>Biochemical theme layer</strong><br/><span style="color:#475569;">themes, cautions, what-not-to-claim</span></div>
          <div style="border:1px solid #cbd5e1;border-radius:12px;padding:12px;background:#f8fafc;"><strong>Output</strong><br/><span style="color:#475569;">evidence-linked interpretation</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
