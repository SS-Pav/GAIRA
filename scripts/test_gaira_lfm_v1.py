"""
GAIRA LFM v1 — Phase 1 Smoke Test

Sends a mock GAIRA text query through the full pipeline:
  mock evidence → prompt builder → Gemini → structured response

Run:
    python scripts/test_gaira_lfm_v1.py
"""
from __future__ import annotations

import sys
import time

from gaira.llm.gemini_client import generate_text
from gaira.llm.prompt_builder import build_prompt
from gaira.llm.response_schema import GAIRAResponse


# ── Mock query data ────────────────────────────────────────────────────

USER_QUERY = (
    "What biochemical changes are associated with HCC (hepatocellular carcinoma) "
    "in serum SERS spectra, and which spectral regions carry the strongest evidence?"
)

EVIDENCE = [
    "The 1000-1010 cm-1 region (phenylalanine ring breathing) shows consistent "
    "depletion in HCC serum across multiple Au-SERS studies, suggesting reduced "
    "aromatic amino acid content.",

    "Lipid-associated bands near 1440-1460 cm-1 (CH2 deformation) show variable "
    "direction: some studies report enrichment, others depletion. This likely "
    "reflects differences in substrate enhancement and sample preparation.",

    "The 1020-1080 cm-1 region (nucleic acid backbone, PO2 stretch) is frequently "
    "cited as HCC-enriched, but cross-substrate comparisons show this assignment "
    "is highly substrate-dependent (Au vs AgNP show opposite directions).",

    "Protein backbone amide III bands (1230-1300 cm-1) consistently show depletion "
    "in cancer sera across multiple disease types, suggesting this is a "
    "disease-general rather than HCC-specific signature.",

    "Glycan/carbohydrate-associated features near 860-920 cm-1 show moderate "
    "enrichment in CCA and LM but not strongly in HCC, potentially useful for "
    "differentiating liver cancer subtypes.",
]

PROVENANCE = [
    "Vornoli et al. 2022 — Au SERS, 72 HCC + 72 CTR, serum",
    "Lin et al. 2020 — AgNP SERS, 89 HCC + 96 CCA + 81 LM + 88 healthy, serum",
    "GAIRA BSV landscape v4 — cross-dataset delta analysis",
    "GAIRA spectral query v3 — CCA dataset broad BSV projection",
]

CAVEATS = [
    "Substrate-dependent enhancement means exact peak positions and intensities "
    "are not directly comparable across Au and AgNP SERS.",
    "Literature assignments frequently overclaim molecule specificity from "
    "single peaks without adequate controls.",
    "Cross-dataset delta directions can reverse between substrates — "
    "the nucleic_acid_backbone axis is the most prominent example.",
]

DOMAIN_CONTEXT = (
    "Sample type: human serum. "
    "Technique: SERS (surface-enhanced Raman spectroscopy). "
    "Substrates in evidence: Au nanoparticles, AgNP colloids. "
    "Disease focus: hepatocellular carcinoma (HCC) vs healthy controls."
)


# ── Run ────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("GAIRA LFM v1 — Phase 1 Smoke Test")
    print("=" * 70)

    # Build prompt
    print("\n[1] Building prompt...")
    prompt = build_prompt(
        user_query=USER_QUERY,
        evidence=EVIDENCE,
        provenance=PROVENANCE,
        caveats=CAVEATS,
        domain_context=DOMAIN_CONTEXT,
    )
    print(f"    Prompt length: {len(prompt)} chars")
    print(f"    Sections: {prompt.count('---') + 1}")

    # Show truncated prompt
    print("\n--- PROMPT (first 500 chars) ---")
    print(prompt[:500])
    print("... [truncated]")

    # Call Gemini
    print("\n[2] Calling Gemini (with fallback)...")
    t0 = time.time()
    try:
        result = generate_text(prompt)
    except Exception as e:
        print(f"    ERROR: {e}")
        sys.exit(1)
    elapsed = time.time() - t0
    raw = result.text
    print(f"    Response received: {len(raw)} chars in {elapsed:.1f}s")
    print(f"    Model used: {result.model_used}")
    print(f"    Fallback used: {result.fallback_used}")
    print(f"    Attempts: {len(result.attempts)}")

    # Parse response
    print("\n[3] Parsing response...")
    response = GAIRAResponse.from_raw(raw)
    print(f"    Parse success: {response.parse_success}")

    # Display
    print("\n" + "=" * 70)
    print("USER QUERY")
    print("=" * 70)
    print(USER_QUERY)

    print("\n" + "=" * 70)
    print("GAIRA RESPONSE")
    print("=" * 70)
    print(response.raw_text)

    print("\n" + "=" * 70)
    print("PARSED SECTIONS")
    print("=" * 70)
    for field in [
        "answer_summary",
        "biochemical_themes",
        "strongest_evidence",
        "supporting_evidence",
        "caveats",
        "confidence_notes",
    ]:
        val = getattr(response, field, "")
        status = "OK" if val else "EMPTY"
        preview = val[:120].replace("\n", " ") if val else ""
        print(f"  [{status:5s}] {field}: {preview}")

    print("\n" + "=" * 70)
    print(f"Phase 1 complete. Parse success: {response.parse_success}")
    print("=" * 70)


if __name__ == "__main__":
    main()
