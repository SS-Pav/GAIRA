# GAIRA V7 — Mathematical Appendix

Every equation the canonical engine executes, and every metric used to judge it. Each section
gives the mathematics first and then a plain-English reading of what it actually does. Nothing
here is new to Phase 09; this is the formal statement of decisions taken in Phases 00–08.

**Notation.** A preprocessed spectrum is a vector $x \in \mathbb{R}^{676}_{\ge 0}$ with
$\lVert x \rVert_2 = 1$, indexed by the canonical grid
$g_b = 450 + 2b$ cm⁻¹ for $b = 0 \dots 675$. Matrices are upper case, vectors lower case,
$\odot$ is elementwise product, $\varepsilon = 10^{-12}$ guards every division.

---

## 1. Preprocessing

### 1.1 Resampling

$$ r_b \;=\; \operatorname{interp}\bigl(g_b;\, w,\, v\bigr), \qquad
r_b = 0 \ \text{ for } g_b < \min w \ \text{ or } \ g_b > \max w $$

Linear interpolation of the measured pairs $(w_i, v_i)$ onto the fixed grid, with **zero fill**
outside the measured range.

> *In plain terms.* Different instruments report different wavenumber points. Before any two
> spectra can be compared they must be expressed on the same ruler. Where a spectrum simply does
> not reach, the engine writes zero and says so, rather than inventing values by extrapolation —
> an invented value at 480 cm⁻¹ would be indistinguishable from a real measurement downstream.

### 1.2 Baseline removal — asymmetric least squares

Fluorescence adds a broad, slowly varying background. asLS finds a smooth curve $z$ that hugs the
spectrum from below by solving, iteratively,

$$ z \;=\; \arg\min_{z} \; \sum_b \omega_b (r_b - z_b)^2 \;+\; \lambda \sum_b (\Delta^2 z_b)^2 ,
\qquad
\omega_b \leftarrow \begin{cases} p & r_b > z_b \\ 1-p & r_b \le z_b \end{cases} $$

with $\lambda = 10^5$, $p = 0.01$, 10 iterations, and $\Delta^2$ the second-difference operator.
In closed form each iteration solves $(W + \lambda D^\top D)\,z = W r$ with $W =
\operatorname{diag}(\omega)$. The corrected spectrum is $\max(r - z,\, 0)$.

> *In plain terms.* Two forces are balanced. The first term pulls the curve toward the data, but
> **asymmetrically**: points above the curve are weighted 0.01 and points below 0.99, so peaks
> (which stick up) are largely ignored while the valleys between them pull hard. The second term,
> scaled by the large $\lambda$, punishes curvature, forcing the result to be smooth. What
> survives is the slow background rather than the sharp bands. The asymmetry is the whole trick:
> a symmetric fit would be dragged up into the peaks and would subtract away real signal.

### 1.3 Savitzky–Golay smoothing

Over each sliding window of 9 points, fit a cubic by least squares and replace the centre point
with the fitted value. Equivalently a fixed convolution $x = c * r$ whose kernel $c$ is determined
by the window and polynomial order alone.

> *In plain terms.* A moving average removes noise but also flattens peaks, which is exactly the
> wrong trade for spectroscopy. Fitting a small cubic instead lets the smoother follow a peak's
> curvature while still averaging out random jitter, so heights and positions survive.

### 1.4 L2 normalisation

$$ x \;=\; \frac{\max(r,0)}{\lVert \max(r,0) \rVert_2 + \varepsilon} $$

> *In plain terms.* Doubling the laser power or the integration time doubles every intensity.
> Scaling to unit length discards that overall size and keeps only the **shape** — which is what
> chemistry lives in. It is also why nothing downstream may be read as a concentration: the
> engine deliberately threw the scale away.

### 1.5 Quality statistics

$$ \widehat{\mathrm{SNR}} \;=\; \frac{\max_b x_b}{\operatorname{median}_b |x_{b+1} - x_b| +
\varepsilon}, \qquad
q = \operatorname{clip}(\widehat{\mathrm{SNR}}/100,\, 0,\, 1) $$

Peaks are located by prominence $\ge 0.02\,\max x$.

> *In plain terms.* The tallest peak is signal; the typical hop between neighbouring points is
> mostly noise. Their ratio is a serviceable signal-to-noise estimate. Using the *median* hop
> rather than the mean keeps real peaks — which produce a few enormous hops — from inflating the
> noise estimate.

---

## 2. The LSM dictionary — non-negative matrix factorisation

The 50 Local Spectral Motifs were learned in Phase 01, **within each chemistry class**, by
minimising

$$ \min_{W \ge 0,\, H \ge 0} \; \bigl\lVert X - W H \bigr\rVert_F^2 ,
\qquad X \in \mathbb{R}^{n \times 676}_{\ge 0},\ H \in \mathbb{R}^{k \times 676}_{\ge 0} $$

$H$ holds the motifs (basis spectra), $W$ their per-spectrum weights, and $\lVert\cdot\rVert_F$
is the Frobenius norm, i.e. the sum of squared entrywise errors. The engine loads $H$ frozen and
never refits it.

> *In plain terms.* This asks: what small set of building-block spectra can be **added together,
> never subtracted**, to reproduce the observed spectra? The non-negativity is not a technical
> convenience — it is principle P-02. A real Raman spectrum of a mixture is a sum of its
> components' spectra; there is no such thing as negative-two-parts glucose. Allowing negative
> weights (as PCA does) buys a better fit at the cost of components that mean nothing chemically.
> Learning the motifs *inside* each chemistry class rather than across the whole corpus means
> each motif describes a pattern that class actually contains, rather than one imposed on it by
> the corpus average.

---

## 3. The CSM dictionary — consensus over a spectral graph

Phase 02 merged the 50 LSMs into 49 Consensus Spectral Motifs by building a weighted graph on
motifs and cutting it.

### 3.1 Edge weight

Seven similarity features per pair of motifs are combined by a **weighted geometric mean**:

$$ W_{ij} \;=\; \prod_{f} \bigl(\max(F^{(f)}_{ij},\, 10^{-3})\bigr)^{\alpha_f},
\qquad \sum_f \alpha_f = 1 $$

with $\alpha$ = band overlap 0.25, spectral cosine 0.20, peak agreement 0.15, bootstrap
co-occurrence 0.15, substitutability 0.10, activation co-occurrence 0.10, provenance overlap 0.05.

> *In plain terms.* A geometric mean is a strict form of "all of the above". Under an arithmetic
> mean a spectral cosine of 0.95 could carry an edge whose every other channel was near zero —
> two motifs that merely look alike would be merged. Under a geometric mean any single near-zero
> channel drags the whole weight toward zero. This is the operational form of "a valid consensus
> requires multiple independent lines of evidence". The floor at $10^{-3}$ makes the maximum
> possible veto a **stated constant** rather than an accident of floating-point representation.

### 3.2 Thresholding and consensus

Edges below $\tau$ are removed and communities extracted by Louvain over 12 seeds. $\tau$ was
chosen by sweeping 0.05 → 0.90 and taking the midpoint of the widest interval over which the
partition itself does not change, against a band-permutation null with empirical per-edge
p-values.

> *In plain terms.* Any threshold produces *some* grouping; the question is whether that grouping
> is an artefact of the threshold. Sweeping and keeping only what survives a wide, contiguous
> range of thresholds — and only what beats a null built by shuffling bands — is how the phase
> distinguished structure from a choice of knob. When the single-threshold rule failed, the phase
> recorded the failure and moved to unanimous threshold consensus rather than quietly picking a
> value that looked good.

---

## 4. Projection onto the CSM dictionary

For a query $x$ and the frozen dictionary $C \in \mathbb{R}^{49 \times 676}_{\ge 0}$:

$$ a \;=\; \arg\min_{a \ge 0} \bigl\lVert x - a^\top C \bigr\rVert_2^2 $$

solved by non-negative least squares. Diagnostics, with $\hat{x} = a^\top C$:

$$ \mathrm{EV} = 1 - \frac{\lVert x - \hat{x}\rVert_2^2}{\lVert x \rVert_2^2 + \varepsilon},
\qquad
\rho = \frac{\lVert x - \hat{x}\rVert_2}{\lVert x \rVert_2 + \varepsilon},
\qquad
s = \frac{\#\{a_j \le 10^{-9}\}}{49} $$

$$ H(a) \;=\; -\frac{1}{\log 49}\sum_j \tilde a_j \log \tilde a_j,
\qquad \tilde a = \frac{a}{\sum_j a_j + \varepsilon} $$

> *In plain terms.* "Which mixture of the 49 frozen motifs best explains this spectrum, given
> that you cannot use a negative amount of anything?" **There is no regularisation parameter**,
> and that absence is deliberate: a penalty weight would be a quantity that could be tuned per
> spectrum, and a tunable quantity on the inference path is a place where results can be nudged.
> Explained variance is the fraction of the spectrum the dictionary accounts for; the residual
> fraction is what is left over; entropy near 0 means one motif dominates and near 1 means the
> activation is spread thin. These are also the raw material for the `unknown` warning — a
> spectrum the atlas cannot explain shows up **here**, in the residual, before any similarity is
> computed.

---

## 5. Molecular retrieval

Reference molecule $m$ is represented by the mean CSM activation of its spectra,
$R_m = \operatorname{mean}\{a_i : y_i = m\}$. With $\hat q = q/\lVert q\rVert$ and
$\hat R_m = R_m/\lVert R_m\rVert$:

$$ S_m \;=\; \operatorname{clip}\bigl(\hat R_m \cdot \hat q,\ 0,\ 1\bigr)
\;=\; \sum_{j=1}^{49} \underbrace{\hat q_j\, \hat R_{mj}}_{\text{contribution of CSM } j} $$

Candidates are ranked by $S_m$ descending. The margin is $S_{(1)} - S_{(2)}$ over the whole bank.

> *In plain terms.* Cosine similarity asks whether two activation patterns **point in the same
> direction**, ignoring how large they are. The second equality is the reason the engine can
> explain itself: an inner product is literally a sum of per-motif terms, so the similarity
> decomposes exactly into "how much each CSM contributed". Nothing is hidden and nothing is left
> over — the engine asserts $|\sum_j \hat q_j \hat R_{mj} - S_m| < 10^{-9}$ for every candidate it
> reports.

---

## 6. Chemistry Evidence — Model D over Model A

The selected model is `D:A_max_idf:lam0.5`, chosen by nested molecule-grouped cross-validation on
inner-fold macro F1 (modal across the five outer folds).

### 6.1 Fine level (Model A, `max` aggregation, `idf` size correction)

For class $c$ with reference molecules $M_c$:

$$ e_c \;=\; w_c \cdot \max_{m \in M_c} S_m,
\qquad
w_c \;=\; \frac{\log\bigl(N/|M_c|\bigr) + 1}{\operatorname{mean}_{c'}\bigl[\log(N/|M_{c'}|)+1\bigr]} $$

> *In plain terms.* A class's evidence is the similarity of its **single best-matching** member,
> not an average — averaging would penalise a chemically diverse class such as `free_amino_acid`
> for containing members unlike the query, which is not a defect. The idf weight offsets the
> opposite bias: with 80 peptide references and 3 nucleic-acid-polymer references, the large class
> would otherwise win simply by having more chances. The normalisation by the mean keeps the
> correction from changing the overall scale of the evidence vector — only its distribution.

### 6.2 Broad level and soft routing

Six curated superclasses from Phase 00, with prototype $P_b$ the mean activation of their
molecules. With $B_b = \cos(q, P_b)$ scaled so $\max_b B_b = 1$, and $\beta(c)$ the modal
superclass of class $c$:

$$ e_c \;\leftarrow\; e_c \cdot \bigl(B_{\beta(c)}\bigr)^{\lambda}, \qquad \lambda = 0.5 $$

> *In plain terms.* Coarse chemistry is easier to get right than fine chemistry, so it is used as
> a **hint** — but only as a hint. Because every $B_b > 0$, multiplying can never zero a fine
> class out; a molecule whose superclass was misjudged is penalised, not excluded. A hard filter
> would make every broad error permanently unrecoverable, which at six-way accuracy would lose a
> non-trivial share of queries outright. The exponent $\lambda = 0.5$ sets how strong the hint is
> and was selected by cross-validation, not chosen by hand.

### 6.3 The radar

$$ \tilde e \;=\; \frac{e}{\sum_c e_c + \varepsilon} $$

plotted on 16 spokes in the canonical class order.

> **$\tilde e_c$ is RELATIVE BIOCHEMICAL EVIDENCE.** It is not a concentration, not an abundance,
> and not a mixture fraction. The L2 normalisation in §1.4 destroyed absolute scale, and a
> similarity is not a quantity of material. A tall spoke means *"the spectrum carries evidence
> associated with this chemistry"*, nothing more.

---

## 7. Calibration

### 7.1 Temperature scaling (chemistry, $T = 0.4538$)

$$ p_c \;=\; \frac{\exp(e_c / T)}{\sum_{c'} \exp(e_{c'} / T)} $$

> *In plain terms.* One number rescales how sharply the evidence vector is turned into
> probabilities. $T < 1$ sharpens, $T > 1$ flattens. It cannot change the *ranking* — the winner
> is the winner at any $T$ — only how confident the engine claims to be. That is exactly the
> property wanted from a calibrator: fix the honesty of the numbers without touching the answer.

### 7.2 Expected calibration error

Partition predictions into $B = 10$ equal-width confidence bins:

$$ \mathrm{ECE} \;=\; \sum_{b=1}^{B} \frac{|\mathcal{B}_b|}{n}
\Bigl| \operatorname{acc}(\mathcal{B}_b) - \operatorname{conf}(\mathcal{B}_b) \Bigr| $$

> *In plain terms.* Of everything the engine called 80% likely, was it right about 80% of the
> time? ECE averages that gap across confidence levels. **ECE alone is dangerous**, and V7 learned
> this the hard way: a model that outputs the base rate for every input has near-perfect ECE and
> is useless. That is why selection uses log loss or Brier *subject to* floors on sharpness and
> discrimination — a constant predictor is disqualified before ECE is ever consulted.

### 7.3 Brier score, sharpness, discrimination

$$ \mathrm{Brier} = \frac{1}{n}\sum_i \lVert p_i - \mathbb{1}_{y_i} \rVert_2^2, \qquad
\mathrm{sharpness} = \operatorname{Var}_i(p_{i,\hat y}), \qquad
\mathrm{disc} = \Pr\bigl(p^{\text{correct}} > p^{\text{wrong}}\bigr) $$

> *In plain terms.* Brier rewards being both calibrated **and** decisive, because it punishes
> confident errors quadratically. Sharpness asks whether the confidences vary at all; a constant
> predictor scores zero. Discrimination asks whether the engine is more confident when it is right
> than when it is wrong. Together these three close the loophole ECE leaves open.

### 7.4 Risk–coverage

Sort by confidence descending; at threshold $t$,

$$ \mathrm{coverage}(t) = \frac{\#\{c_i \ge t\}}{n}, \qquad
\mathrm{risk}(t) = 1 - \operatorname{acc}\bigl(\{i : c_i \ge t\}\bigr) $$

> *In plain terms.* "If I only answer when I am confident, how often am I right, and how often do
> I answer at all?" This is the curve that tells an operator where to set an abstention rule.
> For GAIRA retrieval, answering only above margin 0.497 keeps 51% of spectra at 79% accuracy,
> against 61% accuracy at full coverage.

---

## 8. Retrieval metrics

For query $i$ let $\mathrm{rank}_i$ be the position of the true molecule.

$$ \mathrm{top}\text{-}k = \frac{1}{n}\sum_i \mathbb{1}[\mathrm{rank}_i \le k], \qquad
\mathrm{MRR} = \frac{1}{n}\sum_i \frac{1}{\mathrm{rank}_i} $$

$$ \mathrm{nDCG@}k = \frac{1}{n}\sum_i \frac{\mathrm{DCG}_i@k}{\mathrm{IDCG}@k},
\qquad
\mathrm{DCG}_i@k = \sum_{j=1}^{k} \frac{\mathrm{rel}_{ij}}{\log_2(j+1)} $$

> *In plain terms.* Top-$k$ is the blunt question — was the answer in the first $k$? MRR rewards
> being close: rank 1 scores 1.0, rank 2 scores 0.5, rank 10 scores 0.1, so a system that is
> usually second beats one that is usually tenth. nDCG additionally discounts by position with a
> logarithm and divides by the best achievable score, which makes it comparable across queries
> with different numbers of relevant items.

**The singleton convention.** 66 of 154 molecules have exactly one spectrum. Under
leave-one-spectrum-out their only representative leaves the bank, so the true answer is *absent* —
68 spectra, 18.1%. These are counted as misses at rank $n+1 = 154$. This is the conservative
choice; dropping them would raise reported top-1 from 0.605 to roughly 0.74 and would be
misleading, because the engine really did fail to name those molecules.

---

## 9. Classification metrics

$$ \mathrm{F1}_c = \frac{2 P_c R_c}{P_c + R_c}, \qquad
\text{macro F1} = \frac{1}{16}\sum_c \mathrm{F1}_c, \qquad
\text{balanced acc} = \frac{1}{16}\sum_c R_c $$

> *In plain terms.* Plain accuracy is dominated by the big classes — with 80 peptide spectra and
> 3 nucleic-acid-polymer spectra, a model could ignore the small classes entirely and still look
> good. Macro F1 gives every class one vote regardless of size, so the 7-spectrum
> `small_nitrogenous` class counts as much as the 80-spectrum `peptide_protein` class. That is
> why macro F1 (0.811) sits below top-1 (0.851): the engine is worse on the rare classes, and
> macro F1 refuses to let that hide.

**ROC and PR.** ROC plots true-positive rate against false-positive rate as a threshold sweeps;
AUC is the probability that a random positive outranks a random negative. PR plots precision
against recall, and average precision is the area beneath.

> *In plain terms.* ROC is optimistic when classes are imbalanced, because a large true-negative
> pool keeps the false-positive rate small no matter what. PR does not have that escape hatch, so
> for GAIRA's uneven 16 classes the PR curves (macro AP 0.983) are the more honest picture — and
> both are reported **in-sample** in this phase, with the caveat printed on the figures.

---

## 10. Uncertainty

### 10.1 Bootstrap confidence intervals

Resample **molecules** (not spectra) with replacement $B$ times; recompute the statistic; report
the 2.5th and 97.5th percentiles.

> *In plain terms.* Resampling molecules rather than spectra is the whole point. Replicate spectra
> of glucose are not independent evidence about how well the engine handles sugars — treating them
> as independent would shrink the interval to something the data does not support. Resampling at
> the molecule level asks the question that matters: *how much would this number move if I had
> drawn a different set of molecules?*

### 10.2 McNemar's test

For paired predictions, with $b$ and $c$ the discordant counts,

$$ \chi^2 = \frac{(|b - c| - 1)^2}{b + c} $$

> *In plain terms.* When two systems are run on the *same* spectra, only the cases where they
> disagree carry information; the ones both get right or both get wrong tell you nothing about
> which is better. McNemar looks solely at the disagreements. This test is the reason Phase 06.5
> did **not** adopt the geometry layer: the raw improvement of +0.016 came from six spectra, and
> McNemar returned p = 0.180.

### 10.3 Molecule-grouped cross-validation

Five folds partitioned so that **all spectra of a molecule fall in the same fold**.

> *In plain terms.* If two replicate spectra of cholesterol land on opposite sides of a split, the
> model is being tested on a molecule it has already seen, and it will look far better than it is.
> Grouping by molecule forces the honest question: *can this generalise to a molecule it has never
> encountered?* The gap between the in-sample 0.955 and the held-out 0.851 is exactly the size of
> the illusion this guards against — and it is why 0.851 is the number Phase 09 quotes.

---

## 11. Confidence composition

$$ \mathrm{confidence} \;=\; \operatorname{clip}(\mathrm{EV},0,1)\times S_{(1)} $$

$$
\texttt{unknown} \;=\; \bigl[\mathrm{EV} < 0.50\bigr] \lor \bigl[\text{margin} < 0.01\bigr],
\qquad
\texttt{outlier} \;=\; \bigl[\rho > 0.50\bigr] \lor \bigl[n_{\text{active}} \le 1\bigr]
$$

> *In plain terms.* The product is deliberately unforgiving. Both factors must be high: the atlas
> must be able to *explain* the spectrum **and** some reference must *match* it. A spectrum the
> dictionary cannot express might still land close to some molecule by accident, and the
> multiplication ensures that accident cannot become a confident answer. Thresholds are inherited
> from Phase 05 unchanged — adjusting them in a packaging phase would be tuning on the test set
> (P-13).

---

## 12. Perturbation models

Seven perturbations, five levels each, each re-normalised to unit L2 after corruption. $\tilde g$
is the grid mapped to $[0,1]$ and $\sigma$ the level.

| perturbation | model | levels |
|---|---|---|
| Gaussian noise | $x + \mathcal{N}\!\bigl(0,\ (\sigma \max x)^2 I\bigr)$ | 0.01, 0.02, 0.05, 0.10, 0.20 |
| shot noise | $x + \sigma' \sqrt{\max(x,0)} \odot \mathcal{N}(0, I)$ | 0.01, 0.02, 0.05, 0.10, 0.20 |
| baseline drift | $x + \tfrac{1}{2}\sigma \max(x)\,\bigl(a\tilde g + b\tilde g^2\bigr)$, $a,b \sim \mathcal{N}(0,1)$ | 0.05, 0.10, 0.20, 0.40, 0.80 |
| fluorescence | $x + \sigma \max(x)\,\widehat{P_3(\tilde g)}$, cubic with $\mathcal{N}(0,1)$ coefficients, min–max scaled | 0.05, 0.10, 0.20, 0.40, 0.80 |
| wavenumber shift | $x$ re-interpolated at $g + \delta$ | 1, 2, 4, 8, 16 cm⁻¹ |
| band broadening | Gaussian convolution of width $\sigma/\text{step}$ | 1, 2, 4, 8, 16 cm⁻¹ |
| peak dropout | a random $\sigma$-fraction of detected peaks linearly interpolated away | 0.05, 0.10, 0.20, 0.40, 0.60 |

> *In plain terms.* Each of these is a real way a spectrum degrades on a real instrument. Gaussian
> noise is detector noise. Shot noise scales with the square root of the signal because photons
> arrive as counts, so bright regions are noisier in absolute terms and quieter in relative ones.
> Drift is a slow instrumental ramp and fluorescence a broad polynomial background — both are
> what the asLS stage exists to remove, so these levels test whether it copes when the background
> is larger than it was designed for. A wavenumber shift is calibration error. Band broadening is
> resolution loss, and it is the corruption that most directly destroys band identity, which is
> why it degrades molecule retrieval faster than it degrades the radar. Peak dropout is the
> hardest case: whole bands simply absent, from a masked region, a detector defect, or genuine
> chemistry. Every perturbation is re-normalised afterwards, so none of them is secretly testing
> the intensity scale — L2 normalisation already removed that.

---

## 13. Fingerprints

$$ \mathrm{fp} = \mathrm{MD5}\bigl(\mathrm{canonical\ JSON\ serialisation}\bigr) $$

with sorted keys, so the hash depends on content and never on dictionary ordering.

> *In plain terms.* A fingerprint is a short string that changes if the artefact changes at all.
> The engine checks four of them at load and refuses to start on a mismatch. This is what makes
> "reproduces the frozen baseline exactly" a claim that can be *checked* rather than asserted —
> and it is why regenerating an upstream artefact is not a thing that can happen silently.
