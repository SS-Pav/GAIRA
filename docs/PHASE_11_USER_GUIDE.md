# GAIRA V7 — User Guide

A step-by-step guide to the interactive demo, written for someone who has a Raman spectrum and
wants to know what GAIRA can honestly tell them about it.

```bash
export PYTHONPATH=$PWD/src
streamlit run streamlit_apps/gaira_v7_demo.py        # → http://localhost:8501
```

---

## Before you start: what GAIRA answers

GAIRA does **not** answer *"which molecule is this?"* It answers a question it can support:
**what chemistry does the evidence favour, how strongly, and by way of which spectral features?**

Three facts govern that choice:

- **Spectra are mixtures, not fingerprints.** What reaches the detector is the sum of everything
  in the illuminated volume.
- **A peak is not a molecule.** A band near 1450 cm⁻¹ says a CH₂ group is bending, and thousands
  of biological molecules contain CH₂ groups.
- **Nearby is not the same.** Matching a peak to a molecule because the wavenumbers agree within
  a few units and the biology is plausible is not evidence.

Think of a doctor reading a blood panel: they read a pattern and say which processes it is
consistent with, and how sure they are.

---

## 1 · Home

The landing page states the four numbers that matter and three claims the rest of the app has to
live up to. Press **Begin Analysis**.

The top navigation — Home, Analyze, Docs, Architecture, About — is always available. There is no
sidebar.

## 2 · Upload

**What to supply.** Two columns: wavenumber and intensity. `.csv`, `.tsv`, `.txt`, `.dat`, `.asc`.
Up to 32 MB.

You do not need to normalise, baseline-correct, resample, or sort your data. GAIRA detects the
delimiter, whether there is a header, which columns are which, and whether the axis ascends or
descends — and **reports every decision it made** under *How this file was read*. Read that
expander at least once; it is how you find out that your file was read as semicolon-delimited
with a header, rather than discovering it from a strange plot.

**No file to hand?** Pick a built-in reference spectrum from the dropdown. They come from the
frozen corpus and run through exactly the same path.

**What you get immediately.** Filename, point count, wavenumber range, median step, and the raw
spectrum plotted — zoom, pan and hover all work.

**Metadata.** Optional, except for one field:

| field | effect |
|---|---|
| **Modality** | Only `raman` runs. Anything else is **blocked**, not warned about — see below |
| **Sample type** | Recorded and warned about. **Never changes a single number** |
| Excitation, Sample ID | Recorded and carried into the report |

> **Why a non-Raman modality is blocked.** Phase 04 measured the frozen Raman atlas reconstructing
> *real* Ag-SERS at AUROC 0.548 — chance. A non-negative Raman basis reconstructs SERS of the same
> metabolites comfortably, so a SERS spectrum run through this engine produces confident numbers
> with no validated meaning. Blocking is the honest behaviour.

> **Why a non-pure sample type only warns.** The arithmetic is identical whatever you select — a
> test runs one spectrum under four sample types and asserts the result digest does not move. What
> changes is what the result *means*: every V7 number was measured on pure reference compounds, so
> for serum, EV or tissue you are reading spectral evidence, not a domain finding.

## 3 · Preprocess

Press **Run preprocessing**. Four stages tick through on the left while the plot on the right
reveals what happened:

1. **interpolate onto the canonical grid** — 450–1800 cm⁻¹, 2.0 cm⁻¹ spacing, 676 bins
2. **remove the fluorescence baseline** — asymmetric least squares
3. **smooth and normalise** — Savitzky–Golay (9, 3), then L2
4. **verify frozen fingerprints** — ten artefact digests

Two curves are drawn and only two exist: what you supplied, and what the **engine returned**. The
raw trace is scaled to its own maximum so both fit on one axis — the caption says so. No
intermediate is invented for the animation.

> **The L2 step is why nothing here is a concentration.** Normalising to unit length discards
> absolute intensity, which is exactly what makes two spectra comparable — and exactly what makes
> a concentration unrecoverable.

## 4 · Analyze

Six stages, each tied to real output from one engine call:

```
Projecting into the frozen motif atlas
Computing LSM activations
Computing CSM activations
Searching the molecular reference atlas
Building chemistry evidence
Generating report
```

The call itself takes about **3.5 ms**. The sequence you watch is the reveal of stages that have
already completed — the app does not fake work, and it does not fake timing either.

## 5 · Reading the report

### The verdict card

The chemistry family with the strongest relative evidence, a confidence band, and a paragraph of
deterministic template text. **No language model wrote it.**

| band | meaning |
|---|---|
| High confidence ≥ 0.70 | the atlas explains the spectrum well *and* a reference matches it well |
| Moderate 0.40–0.70 | one of those two is weak — the Confidence section says which |
| Low < 0.40 | treat the ordering as indicative, not settled |

### The three columns

**Left — processed spectrum.** Amber markers are the diagnostic bands of the strongest consensus
motif: the wavenumbers the answer actually rests on.

**Centre — Chemistry Evidence.** Sixteen axes, switchable between Radar, Bars and Polar. **Bars
are the precision view; the radar is for recognising a shape at a glance.** Hover any axis for
its evidence, its calibrated confidence and a plain-English description of the family.

> **This is RELATIVE evidence.** Not a concentration. Not an abundance. Not a mixture fraction.
> A tall axis means *"the spectrum carries evidence associated with this chemistry"* and nothing
> more.

**Right — top molecular analogues.** Ten reference molecules ranked by similarity. Expand any one
to see its **score decomposition**: because a cosine is an inner product, the score splits exactly
into per-motif contributions, and the caption shows them summing back to the score.

> **These are reference *analogues*, not identifications.** Validated molecule top-1 is **0.6053**,
> and 68 of 375 corpus queries cannot be retrieved at all because their molecule has only one
> spectrum. A shortlist of three is often more useful than one confident guess.

### The expandable sections

| section | what to look for |
|---|---|
| **Chemical evidence** | all sixteen axes with descriptions and calibrated confidence |
| **CSM contributions** | the 49 canonical coordinates. Click any motif to see its spectrum, its diagnostic bands and the molecules that support it |
| **LSM view** | the 50 local motifs. Diagnostic only — no later stage reads them |
| **Reconstruction** | query, reconstruction and residual on synchronised axes, with an opacity slider. **The residual is where the atlas failed** — read it |
| **Molecular retrieval** | overlay your query on any reference spectrum, with matched bands marked |
| **Confidence** | why this number and not a higher one |
| **Provenance** | the full chain: spectrum → LSM → CSM → chemistry → molecule, plus atlas fingerprints |
| **Download** | `InferenceResult.json` and the PDF report |

### Understanding confidence

Confidence is **deliberately multiplicative**:

```
confidence  =  CSM explained variance  ×  top-1 similarity
```

Both must be high. The atlas must be able to **explain** your spectrum *and* some reference must
**match** it. A spectrum the dictionary cannot express might still land near some molecule by
accident, and multiplying ensures that accident cannot become a confident answer. The section
names which of the two factors is limiting.

### The two warnings

| flag | fires when |
|---|---|
| `unknown` | CSM explained variance < 0.50, **or** the top-1/top-2 margin < 0.01 |
| `outlier` | residual fraction > 0.50, **or** one or fewer active motifs |

Those two `unknown` triggers mean opposite things. A poorly explained spectrum is a coverage
problem. A small margin on a *well*-explained spectrum usually means two near-identical references
— a stereoisomer pair, say — and is not a defect at all. The interpretation text names which one
fired.

> **Neither flag is an unknown-molecule detector.** GAIRA has **no validated open-set detection**.
> Phase 09 measured white noise reconstructing at explained variance ≈ 0.61 — above the 0.50 floor
> — with the flag firing on only 1 of 20 random spectra. Confidence separated it correctly
> (noise peaked at 0.495 against a corpus mean of 0.803). **Read the confidence, not the flag.**

## 6 · Docs, Architecture, About

**Docs** — engine version, both atlas identities, corpus, validated performance and known
limitations, all pulled live from the running engine.

**Architecture** — the seven stages. Select any one to read what it does, plus what is
deliberately absent and why DART is not a modality.

**About** — what GAIRA is, how it differs, why CSM, why Chemistry Evidence, and why there is no
language model anywhere in the inference path.

---

## Getting a trustworthy answer

1. **Cover the window.** Below 70% coverage of 450–1800 cm⁻¹ you get a warning; below 10% the run
   is refused. Motifs whose diagnostic bands fall outside your measured range cannot activate.
2. **Read the residual before the radar.** If the reconstruction misses a strong band, the
   chemistry read is describing what the atlas *could* express, not what you measured.
3. **Prefer top-3 to top-1.** Validated top-1 is 0.605; top-3 is 0.763.
4. **Treat the radar as a shape, not a measurement.** Compare shapes between samples; do not read
   a radius as a quantity.
5. **Check the score decomposition** when a candidate surprises you. If one motif carries 95% of
   the similarity, the match rests on a single spectral feature.
6. **Download the JSON.** It carries the `result_digest` — every other GAIRA surface given the
   same spectrum returns the same digest, so a result is checkable rather than merely reported.

## What GAIRA will never tell you

- a concentration or an abundance
- a definitive molecular identification
- that a molecule is *absent* from its 154-molecule bank
- a biological or clinical interpretation
- anything validated about SERS, serum, plasma, EV, bacteria or tissue

If you need one of those, GAIRA is the wrong instrument, and it is designed to say so rather than
to guess.
