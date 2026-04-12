# BSV v2 — Component Confidence Rules

## Confidence Labels

| Label | Icon | Criteria |
|---|---|---|
| **strong** | 🟢 | motif_count >= 3 AND stability = STABLE |
| **moderate** | 🟡 | motif_count >= 2 AND stability in (STABLE, MIXED) |
| **weak** | 🟠 | motif_count >= 1 (anything present but sparse) |
| **exploratory** | ⚪ | coverage = absent OR motif_count = 0 |

## Purpose
Confidence labels help the user understand which BSV components are well-supported by evidence and which are based on sparse or unstable motifs. A component with "strong" confidence is more reliable for interpretation than one labeled "exploratory."

## How It Affects Interpretation
- Confidence does NOT alter the numerical score
- It is a transparency layer — it tells the user how much to trust the score
- In the UI, confidence is shown as a colored icon + label in the BSV table
- In the explanation panel, confidence appears in the component summary text

## Design Principle
A component can have a high score but weak confidence (e.g., if one large motif drives it but stability is UNSTABLE). The score reflects the evidence; the confidence reflects the evidence quality.
