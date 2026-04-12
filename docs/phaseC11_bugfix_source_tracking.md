# Phase C1.1 Bugfix — Evidence Selection Source Tracking

## Root Cause
In `graph/phaseC1_scoring.py`, the function `_select_quality_evidence` initialized `sources_used` as a `set()` on line 104, but then called `.get(src, 0)` and dict-style assignment `sources_used[src] = ...` on lines 110-112. The `set` type does not support `.get()` with a default, causing `AttributeError: 'set' object has no attribute 'get'`.

This was introduced during the C1.1 source-diversity upgrade. The intent was a counter dict, but `set()` was written instead of `dict()`.

## Exact Fix
Changed line 104 from:
```python
sources_used = set()
```
to:
```python
sources_used: dict[str, int] = {}
```

No other lines changed. The `.get()` and `[src] = ...` usage on lines 110-112 was already correct for a dict — only the initialization was wrong.

## Why This Preserves C1.1 Behavior
The source-diversity logic works as designed:
- `sources_used` counts how many evidence rows have been selected from each source
- If a source already has 2 selected rows AND there are more candidates than `max_rows`, skip it
- This ensures the displayed evidence sample includes diverse sources without excluding any source entirely

## Verification
- Smoke-tested with simulated data: 5 candidates from 3 sources, max_rows=3
- Source cap at 2 works correctly
- Diversity preference works correctly
- No remaining set-vs-dict misuse in the scoring module
