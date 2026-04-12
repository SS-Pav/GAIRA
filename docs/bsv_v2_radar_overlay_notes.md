# BSV v2 — Radar Overlay Notes

## Change
For comparative queries, both query AND comparator BSV polygons are now overlaid on the same radar.

## Visual Design
- **Blue polygon (solid line)**: query condition (e.g., HCC)
- **Red polygon (dashed line)**: comparator condition (e.g., healthy control)
- Both use matching index positions so components align perfectly
- Components with "absent" coverage are excluded from both polygons

## Technical Fix
The comparator BSV polygon uses the same component indices as the query, ensuring alignment even if some components have zero scores on one side.

## Grid Styling
- Light gray background (#fafafa)
- Subtle gridlines
- Y-axis labels: 0.25, 0.5, 0.75, 1.0
- Component labels use multi-line formatting for readability

## Single-Query Behavior
For non-comparative queries, only the query polygon is shown (same as v1).
