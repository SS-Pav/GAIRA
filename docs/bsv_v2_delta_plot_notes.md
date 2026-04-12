# BSV v2 — Delta Plot Notes

## Changes
1. **Sorted by |delta|**: largest shifts shown first (most informative at top)
2. **Dynamic x-axis**: scales to actual delta range instead of fixed ±1.1
3. **Annotated values**: exact delta numbers shown next to bars
4. **Flat-line note**: when max |delta| < 0.05, a note explains the BSVs are very similar
5. **Color thresholds**: blue (positive > 0.03), red (negative < -0.03), gray (near zero)

## Reading the Plot
- Top bars = biggest biochemical state differences
- Blue bars = query condition has more evidence/support
- Red bars = comparator has more
- Gray bars = no meaningful difference at current resolution

## Honest Flat Results
When all deltas are tiny (< 0.05), the plot displays a note:
> "Note: query and comparator BSVs are very similar under current weighting"

This is scientifically honest — it means the conditions are not well-separated by the current evidence and component definitions.
