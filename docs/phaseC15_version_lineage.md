# GAIRA Streamlit App Version Lineage

## Versioned App Files

| File | Version | Key Feature |
|---|---|---|
| `app/gaira_query_demo_C14.py` | C1.4 | Pre-normalization comparative |
| `app/gaira_query_demo_C1_5.py` | C1.5 | Coverage-aware, normalized, test-validated |
| `app/gaira_query_demo_C1_6.py` | C1.6 | Scope-aware comparator (sample/domain) |
| `app/gaira_query_demo_c1_7.py` | C1.7 | Motif differential + signal stability |
| `app/gaira_query_demo_C1_8.py` | C1.8 | BSV v1 + radar/delta visualization |
| `app/gaira_query_demo_BSV_v2.py` | **BSV v2** | **Refined radar overlay + confidence + explanations** |
| `app/gaira_query_demo.py` | C1.5 | Main pointer (stable fallback) |

## How to Launch
```bash
streamlit run app/gaira_query_demo_BSV_v2.py # Latest with BSV v2
streamlit run app/gaira_query_demo_c1_7.py   # Motif differential
streamlit run app/gaira_query_demo_C1_6.py   # Scoped comparator
streamlit run app/gaira_query_demo.py         # Stable fallback (C1.5)
```

## C1.8 Notes
- Introduces BSV (Biochemical State Vector) as a higher-level abstraction
- 8 biochemical components: membrane_lipid, protein_backbone, aromatic_amino_acid, purine_nucleotide, pyrimidine_nucleotide, glycan_carbohydrate, redox_metabolite, nucleic_acid_backbone
- Radar plot (query vs comparator) + delta bar chart
- Graph-backed component explanations
- Preserves all prior motif differential + stability + theme outputs
