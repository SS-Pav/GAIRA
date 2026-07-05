"""gaira_base_3 — packet/subfamily ontology layer.

Architectural layer ABOVE gaira_base_2: introduces the packet (subfamily)
intermediate between motif and family. Motif scoring is unchanged
(reuses gaira_base_2 final-ranking-repair output); packet scoring is
the new primary output; family scoring is derived from packet scores.

  primitives -> motifs -> packets (NEW) -> families -> report
"""
