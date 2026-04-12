# BSV v2 — Side-by-Side Table Format

## Columns

| Column | Description |
|---|---|
| Component | BSV component name (title-cased, underscores removed) |
| Query | Normalized score for query condition (0-1) |
| Raw Q | Raw unnormalized query score |
| Comparator | Normalized score for comparator (comparative queries only) |
| Raw C | Raw unnormalized comparator score |
| Delta | Query minus comparator (signed, 3 decimal places) |
| Shift | "up" / "down" / "flat" |
| Stability | Dominant stability label from contributing motifs |
| Conf. | Confidence label with icon (🟢🟡🟠⚪) |
| Motifs | Number of contributing motifs |

## When Displayed
- Always shown for all queries
- For single-condition queries: Comparator/Raw C/Delta/Shift columns are omitted
- For comparative queries: all columns shown

## Sorting
Components are listed in definition order (not by score), matching the radar plot axes.
