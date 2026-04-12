# D1 — Peak Matching Rules

## Principle
Contrast evidence is matched to EXISTING GAIRA peaks. We do NOT create new peak identities.

## Matching Logic
1. Round the reported peak value to the nearest integer
2. Search existing evidence rows within a ±5 cm-1 tolerance window
3. Return all matching evidence_item_ids (capped at 5 per directional entry)
4. Record the first match as `matched_peak_node_id` and the total as `peak_match_count`

## Tolerance: ±5 cm-1
This is the default tolerance. Rationale:
- Raman/SERS peak positions vary by 2-5 cm-1 across substrates, solvents, and instruments
- ±5 cm-1 covers typical reporting variation without crossing into adjacent peaks
- For the fingerprint region (400-1800 cm-1), peaks are typically spaced >10 cm-1 apart

## Unmatched Entries
If a directional statement references a peak with no existing evidence within tolerance:
- `matched_peak_node_id` = empty
- `peak_match_count` = 0
- The entry is still recorded but flagged as unmatched
- It should NOT be forced into a bad match

## Current Pilot Result
- 19/19 entries matched (100%) — because all directional language was found within existing evidence text, which inherently references peaks already in the corpus
