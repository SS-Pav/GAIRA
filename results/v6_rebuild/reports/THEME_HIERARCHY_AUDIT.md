# THEME_HIERARCHY_AUDIT

### How the V1 semantic hierarchy is wired, and where theme information leaks into MSS

*GAIRA V6 rebuild, first task. Read-only: the frozen atlas
`09ed804a40836f4a05a91ba10900cded` and every asset under `assets/foundation/` are
inputs, never outputs. Reproduce with
`python results/v6_rebuild/code/p01_audit_and_mss_rebuild.py`.*

---

## 0 · The headline

**MSS is not an independent layer in V1.** 25 % of every component→motif weight is copied
from the component→**theme** matrix. Averaged over the 70 contributor edges the theme term
supplies **13.4 %** of the raw score (max 63.9 %), and **15 of 70 edges (21 %)** exist only
because of it — they fall below the keep threshold when it is removed.

Any hierarchy that derives themes *from* MSS on the V1 layer is therefore **circular**:
the themes would be predicting a quantity that was partly built out of themes. This single
fact is why V6 must rebuild MSS before it can build a theme layer.

---

## 1 · The V1 hierarchy as implemented

```
                    ┌─────────────────────────────────────────┐
   spectrum ──► preprocess ──► NNLS onto frozen H ──► coord ∈ ℝ²⁴₊
                    └─────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
        ┌───────────────────────┐       ┌───────────────────────┐
        │  W  (24 × 13)         │       │  M  (24 × 13)         │
        │  component → theme    │       │  component → motif    │
        │  ontology weights     │       │  MSS weights          │
        └───────────┬───────────┘       └───────────┬───────────┘
                    ▼                               ▼
             BSV theme composition            MSS motif composition
                    │                               │
                    └──────────► PARALLEL ◄─────────┘
                        (MSS never feeds the BSV)

                    ▲                               │
                    │        ✗ LEAKAGE ✗            │
                    └───────────────────────────────┘
              W[j, parent_theme] is an INPUT to M[j, m]
```

Both maps read the same 24 coordinates and neither feeds the other **forward** — the
docstring's claim that MSS is "additive and parallel to the BSV" is true in that
direction. But the *construction* of M reads W, so the two are not independent.

### The exact code

| file : line | statement | role |
|---|---|---|
| `src/gaira/engine/mss.py:189` | `parent = m["parent_theme"]` | every motif declares a parent theme in the frozen YAML |
| `src/gaira/engine/mss.py:190` | `ti = self.onto.theme_index(parent)` | resolves it to a column of W |
| `src/gaira/engine/mss.py:195` | `theme = float(self.onto.W[j, ti])` | **reads the component→theme matrix** |
| `src/gaira/engine/mss.py:196` | `raw = self.wb*band + self.we*exemplar + self.wt*theme` | **0.40 / 0.35 / 0.25 — a quarter of the score is theme** |
| `src/gaira/engine/mss.py:209` | `breadth = mean(((band>0)+(exemplar>0)+(theme>0))/3)` | the theme indicator is one of three evidence lines |
| `src/gaira/engine/mss.py:169` | `purine_motif = parent_theme == "nucleic_purine"` | perturbation provenance is gated on the theme **label** |

The weights are read from the frozen spec at `mss.py:133`
(`d["weights"]["band"], ["exemplar"], ["theme"]`) and are recorded in
`assets/foundation/mss_motifs_v1.yaml` as `{band: 0.40, exemplar: 0.35, theme: 0.25}`.

### How components are mapped into MSS (V1)

For motif *m* and component *j*:

```
band_j,m     = |{b ∈ bands(m) : ∃ p ∈ peaks(j), |p − b| ≤ 16}| / |bands(m)|
exemplar_j,m = min(1, Σ contribution_pct of analytes of j matching exemplars(m) / 12)
theme_j,m    = W[j, parent_theme(m)]                              ← LEAKAGE
raw_j,m      = 0.40·band + 0.35·exemplar + 0.25·theme
keep if raw ≥ 0.15 · cap 6 contributors · normalise so Σ_j M[j,m] = 1
```

### How components are mapped into themes (V1)

Independently of MSS, from three evidence lines mixed 0.50 / 0.25 / 0.25
(`results/v5_rebuild/engine_v1/code/build_theme_weights.py`):

```
loading_j,t      = Σ_family share(j,family) · family_theme_affinity[family][t]
spectral_j,t     = |{p ∈ peaks(j) : ∃ b ∈ bands(t), |p − b| ≤ 20}| / |peaks(j)|
perturbation_j,t = Σ response of analytes driving j whose identity maps to t
W[j,t]           ∝ 0.50·loading + 0.25·spectral + 0.25·perturbation
residual → background_matrix (generic components) else unknown_mixed
```

So **W is built from component evidence, and M is built from component evidence *plus W***.

---

## 2 · Is there theme leakage into MSS?

**Yes.** Quantified over all 70 contributor edges:

| measure | value |
|---|---:|
| mean theme share of the raw score | **13.4 %** |
| median | 11.7 % |
| maximum | **63.9 %** |
| edges with no spectral and no chemical evidence at all | 0 of 70 |
| edges that fall below the keep threshold once the theme term is removed | **15 of 70 (21 %)** |

No edge is *purely* theme-derived, but a fifth of the MSS graph would not exist without
theme information. The leakage is not uniform: it concentrates on motifs whose parent
theme carries a large ontology weight, which is precisely where a circular hierarchy
would be most flattering to itself.

A second, structural consequence sits alongside it. Because `evidence_breadth` counts the
theme indicator as one of three lines — and because the three indicators are `np.bool_`,
whose `+` is logical OR — breadth evaluates to **exactly 1/3 for all 13 motifs**. MSS
confidence in V1 is therefore `stability × 1/3`: a rescaled stability carrying no
motif-discriminating information at all.

---

## 3 · What V6 changes

MSS is rebuilt from evidence that cannot reference a theme
(`results/v6_rebuild/code/v6_core/mss_v6.py`):

| line | V1 | V6 |
|---|---|---|
| band match | ✓ 0.40 | ✓ 0.30 |
| basis-spectrum cosine | — | **✓ 0.30 (new)** — cosine of the frozen basis spectrum against a Gaussian profile built from the motif's bands |
| exemplar loading | ✓ 0.35 | ✓ 0.30 |
| perturbation | provenance only | ✓ 0.10, with an ablation always reported |
| **theme weight** | **✓ 0.25** | **✗ removed** |
| evidence breadth | constant 1/3 (bug) | 0.58 – 0.92, computed with `int()` |

### Effect

| measure | V1 | V6 |
|---|---:|---:|
| mean band fidelity (does the motif's implied spectrum match its declared bands?) | 0.594 | **0.633** |
| motifs whose band fidelity improved | — | **12 of 13** |
| mean component stability | 0.812 | 0.817 |
| mean component-weight cosine V1 ↔ V6 | — | 0.960 |
| mean activation Spearman across the corpus | — | 0.887 |
| MSS confidence range | 0.254 – 0.300 | 0.475 – 0.723 |
| perturbation-ablation cosine (V6 vs V6 without perturbation) | — | 0.983 |

**MSS quality improves and its support is preserved.** The component sets barely move
(cosine 0.96, activation ρ 0.89), so V6 is a *purification* of the V1 layer rather than a
replacement of it — but it is now derivable without reference to any theme, which is the
precondition for the rest of the V6 rebuild. The perturbation ablation confirms the new
functional term is not driving the result (cosine 0.983 with it removed), so the
purine-heavy perturbation corpus does not bias the motif definitions.

---

## 4 · The V6 hierarchy

```
  spectrum → coord ∈ ℝ²⁴₊  ──M──►  17 MSS motifs  ──T──►  13 chemical themes
                            (spectroscopy)          (hard partition)

  theme(x) = Tᵀ · Mᵀ · coord(x)
```

Themes are now *defined as groupings of motifs*, so the chain is a composition of two
non-negative linear maps and every theme is, by construction, a statement about
spectroscopy. Biological-state themes are deliberately **not** implemented: they require
functional evidence a static Raman spectrum does not carry.

---

## 5 · Verdict

1. **Components → MSS (V1):** band match + exemplar loading + **component→theme weight**.
2. **Components → themes (V1):** family affinity + band overlap + perturbation identity.
3. **Theme leakage into MSS: yes** — 13.4 % of the mean raw score, 21 % of edges depend on
   it for their existence, and the confidence metric is degenerate.
4. **All leakage sites located** (§1) and removed in `v6_core/mss_v6.py`.
5. **Removing it does not damage MSS** — band fidelity and confidence both improve,
   stability and component support are unchanged.

Nothing frozen was modified. `assets/foundation/` is byte-identical.
