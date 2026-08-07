# GAIRA V7 — Demo Video Script

**Target 4:30. Screen recording at 1560×1000, dark room, no music under the narration.**
Every number spoken below is what the app actually shows for the built-in **cholesterol**
example; if you record a different spectrum, read the numbers off the screen instead.

Setup:
```bash
export PYTHONPATH=$PWD/src
streamlit run streamlit_apps/gaira_v7_demo.py
```

---

### 0:00 — 0:20 · Cold open

**[Home page, full screen. Hold still for two seconds before speaking.]**

> This is GAIRA. It reads a Raman spectrum and tells you what chemistry the evidence supports —
> and, just as importantly, what it doesn't.

**[Cursor drifts over the four stat tiles.]**

> Forty-nine consensus motifs. A hundred and fifty-four reference molecules. Sixteen chemistry
> axes. And point eight-five-one — chemistry accuracy on molecules the model has never seen.

### 0:20 — 0:45 · The claim

**[Slow scroll to the three cards.]**

> Three claims, and the rest of this demo has to live up to them. The engine is frozen and
> provably so. It's explainable by construction. And it's honest about its limits — molecule
> top-1 is point six-zero-five, and every screen says so.

**[Click **Begin Analysis**.]**

### 0:45 — 1:15 · Upload

**[Upload page.]**

> Two columns: wavenumber and intensity. You don't need to normalise, baseline-correct or even
> sort your data.

**[Select **Cholesterol (sterol)** from the dropdown.]**

> I'll use a built-in reference spectrum — it runs through exactly the same path as an upload.

**[Point at the four stat tiles, then the raw plot. Zoom into 1400–1500 cm⁻¹ and back out.]**

> Six hundred and seventy-six points, four-fifty to eighteen-hundred wavenumbers. This is the raw
> trace — fully interactive.

**[Open *How this file was read*.]**

> Every decision the parser made is reported. Delimiter, header, column identity, axis direction.
> Nothing is repaired silently.

**[Point at the Modality dropdown. Select `ag_sers`. Let the red block appear. Hold two seconds.]**

> And this is the part I want you to notice. Select a modality GAIRA hasn't validated, and it
> refuses to run. Not a warning — a block. Phase 04 measured this engine reconstructing real
> silver-SERS at AUROC point five-four-eight. That's chance. It would have produced confident
> numbers with no meaning, so it doesn't produce them at all.

**[Set it back to `raman`. Click **Preprocess Spectrum**.]**

### 1:15 — 1:50 · Preprocessing

**[Preprocess page. Click **Run preprocessing**. Let all four stages tick.]**

> Crop to the canonical window. Remove the fluorescence baseline by asymmetric least squares.
> Smooth, then normalise to unit length.

**[Point at the final blue curve.]**

> Two curves exist here and only two: what I supplied, and what the engine returned. Nothing is
> invented for the animation.

**[Pause on the last stat row.]**

> That last step — normalising to unit length — is why nothing downstream is a concentration.
> It's what makes two spectra comparable, and it's what makes a concentration unrecoverable.

**[Click **Analyze Spectrum**.]**

### 1:50 — 2:10 · Analysis

**[Let the six stages run. Don't rush this — it reads as the machine thinking.]**

> Projecting into the frozen atlas. Local motifs, then the forty-nine consensus motifs — the
> canonical representation. Searching a hundred and fifty-four reference molecules. Building
> chemistry evidence. Generating the report.

> The call itself takes about three and a half milliseconds. What you're watching is the reveal.

### 2:10 — 2:50 · The verdict

**[Results page. Hold on the verdict card.]**

> Sterol steroid. High confidence, point eight-nine-six. Fifty-five percent of total evidence.
> Nearest analogue: cholesterol.

**[Point at the paragraph.]**

> That paragraph is template-driven. No language model wrote it, and no language model is
> anywhere in this system.

**[Move to the left column, hover a couple of amber markers.]**

> The amber marks are the diagnostic bands of the strongest motif — the wavenumbers this answer
> actually rests on.

**[Centre column. Hover two or three radar spokes.]**

> Sixteen chemistry axes. Hover any one for its evidence, its calibrated confidence and what the
> family means.

**[Click **Bars**.]**

> Bars are the precision view; the radar is for recognising a shape.

**[Emphasise.]**

> And this is *relative* evidence. Not a concentration. Not an abundance. A tall axis means
> evidence associated with that chemistry is present. Nothing more.

### 2:50 — 3:20 · Explainability

**[Right column. The first candidate is already expanded.]**

> Ten reference analogues — analogues, not identifications. Validated top-1 is point six-zero-
> five, and the app says so right here.

**[Point at the score decomposition bars, then the caption underneath.]**

> This is the part I like. A cosine is an inner product, so the score splits *exactly* into
> per-motif contributions. Contributions sum to point nine-nine-nine-five-two-four against a
> similarity of point nine-nine-nine-five-two-four. Exact. No hidden term.

### 3:20 — 3:55 · Under the hood

**[Open **CSM contributions**. Pick a motif from the dropdown.]**

> The forty-nine canonical coordinates. Click any motif and you get its spectrum, its diagnostic
> bands, and the molecules that support it.

**[Open **Reconstruction**. Drag the opacity slider.]**

> Query, reconstruction, residual. The residual is where the atlas failed — read it before you
> trust the radar.

**[Open **Confidence**.]**

> And here's why the confidence is point eight-nine-six and not point ninety-five. It's a
> product: explained variance times top-one similarity. Both have to be high. A spectrum the
> dictionary can't express might still land near something by accident — multiplying makes sure
> that accident can't become a confident answer.

**[Scroll to the open-set banner. Slow down.]**

> And the limitation that matters most. GAIRA has no validated open-set detection. It cannot tell
> you a molecule is *absent* from its bank. White noise reconstructs above the warning floor.
> Read the confidence, not the flag. The app tells you that on the results page, not in a
> footnote.

### 3:55 — 4:15 · Provenance and download

**[Open **Provenance**. Let the Sankey render.]**

> Spectrum, to local motifs, to consensus motifs, to chemistry, to molecules. Every claim walks
> back to a wavenumber. And the atlas fingerprints travel with the result.

**[Open **Download**. Click **InferenceResult.json**, then build the PDF.]**

> JSON and a PDF, both from the frozen report generator — the same one the CLI, the API and the
> MCP server use.

**[Point at the result digest line.]**

> That digest is the checkable part. Every other GAIRA surface — Python, command line, HTTP, MCP
> — given this spectrum returns the same digest. Measured across seven surfaces at a maximum
> difference of exactly zero.

### 4:15 — 4:30 · Close

**[Navigate to **Architecture**. Click through two stages.]**

> Seven stages, no branches, nothing tunable at inference time.

**[Gesture at *What is deliberately absent*.]**

> And this is what's *not* here. Themes, meta-components, latent geometry, chemistry reranking —
> four independent attempts to build a layer above the consensus motifs. Each one lost
> information. So the engine ships the layer where the information actually is.

**[Cut back to Home. Hold on the hero for three seconds.]**

> GAIRA. Grounded AI for Raman analysis. Frozen, explainable, and honest about what it can't do.

---

## Production notes

- **Don't speed up the analysis sequence.** Its pacing is the whole point — it reads as reasoning.
- **Hold the scope warnings on screen for a full beat.** They are the most differentiating thing
  in the demo, and cutting them fast reads as hiding them.
- **Never say "identifies", "detects" or "diagnoses".** Say *retrieves*, *favours*, *is
  consistent with*.
- **Never call the radar a composition.** Say *relative evidence*.
- If recording a different spectrum, re-read every number off the screen. The point of the video
  is that the numbers are real.
