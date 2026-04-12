# Landscape v3 — Trust Rules

## Support Tiers

| Tier | Criteria |
|---|---|
| **strong** | evidence >= 20, sources >= 3, biological condition |
| **moderate** | evidence >= 8, sources >= 2 |
| **weak** | evidence >= 3, sources >= 1 |
| **insufficient** | evidence > 0 but below weak threshold |
| **absent** | evidence = 0 |

## Trust Downgrades

| Condition | Effect |
|---|---|
| Single source (sources = 1) | strong → moderate |
| Non-biological label | strong/moderate → low |
| Zero BSV mapping | strong/moderate → low |

## Caution Notes
- `single-source`: only one paper contributes — cannot cross-validate
- `no BSV`: motif profile exists but no motifs map to BSV components — chemistry-only evidence

## In the App
- Trust label shown in coverage table
- Filters can exclude weak/insufficient
- Default: biological-only, sources >= 2, evidence >= 5, exclude single-source
