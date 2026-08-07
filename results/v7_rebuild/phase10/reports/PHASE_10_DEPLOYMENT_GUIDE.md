# GAIRA V7 — Deployment Guide

From a clean clone to a working inference in under five minutes. **No external volume, no cloud
account, no LLM, no network access.**

---

## 1. Install

```bash
git clone <repo> GAIRA && cd GAIRA
python -m venv .venv && source .venv/bin/activate

pip install numpy scipy scikit-learn pandas pyyaml          # core
pip install pydantic fastapi "uvicorn[standard]" httpx      # API + SDK
pip install matplotlib streamlit plotly                     # reports + UI
pip install mcp                                             # MCP server (optional)

export PYTHONPATH=$PWD/src
```

## 2. Verify the atlas

```bash
python -m gaira.v7.cli info
```

Expect `frozen assets verified: True`, atlas `2e43ddcca7d3be41c5f9da016fb8277f`, and
`50 LSMs · 49 CSMs · 154 molecules · 16 chemistry axes`.

**The frozen atlas is committed to the repository** — ten files, ~10 MB, all git-tracked under
`results/v7_rebuild/`. `GAIRAEngine.load()` was instrumented to confirm it opens those ten and
nothing else. **`/Volumes/SSD_Rad` is NOT required for Phase 10 inference**, and a test asserts
that every frozen asset resolves inside the repository.

## 3. Run

| | command | address |
|---|---|---|
| CLI | `python -m gaira.v7.cli infer spectrum.csv` | — |
| API | `gaira serve` · `python -m gaira.v7.api` | http://127.0.0.1:8000/docs |
| Streamlit | `gaira streamlit` · `streamlit run streamlit_apps/gaira_v7_console.py` | http://localhost:8501 |
| MCP | `gaira mcp` · `python -m gaira.v7.mcp` | stdio |

Install the `gaira` entry point with `pip install -e .` if you prefer it to `python -m`.

### Two terminals, the usual setup
```bash
# terminal 1
python -m gaira.v7.api

# terminal 2 — the UI talks to the API instead of loading its own engine
GAIRA_API_URL=http://localhost:8000 streamlit run streamlit_apps/gaira_v7_console.py
```

## 4. Python

```python
from gaira.v7 import GAIRA

gaira  = GAIRA.load()
x, y, diagnostics = GAIRA.read("spectrum.csv")
result = gaira.infer(x, y, metadata={"sample_id": "S-001", "excitation_nm": 785})

print(result.chemistry.predicted_class)
print(result.retrieval.top[0].molecule, result.retrieval.top[0].similarity)
print(result.interpretation)

open("report.pdf", "wb").write(gaira.report(result, fmt="pdf"))
```

Remote mode — identical results, engine elsewhere:
```python
gaira = GAIRA.remote("http://localhost:8000")
```

## 5. Docker (optional — not required for Phase 10 success)

```bash
docker compose up --build       # API :8000, Streamlit :8501
```

The image copies the frozen atlas from the repository and **fails the build** if verification
does not pass, so a broken image cannot ship. No machine-local path is baked in.

## 6. Input formats

**Supported** — CSV, TSV, two-column text (`.txt`, `.dat`, `.asc`), NumPy arrays, Python lists.
Automatic detection of delimiter, header, column identity, and axis direction; every decision is
reported as a diagnostic.

**Declared but not implemented** — `.spc`, `.jdx`/`.dx`, `.wdf`, `.sp`. These return an explicit
"planned format, not an implemented one" error rather than a confusing parse failure. Export to
CSV.

## 7. Limits

Upload 32 MB · spectrum 200,000 points · body 32 MB · report filenames are **derived** from the
result digest, never supplied by the caller. No route or tool accepts a filesystem path.

## 8. Troubleshooting

| symptom | cause |
|---|---|
| `FrozenAssetError` | an artefact under `results/v7_rebuild/` changed. Restore from git; do not repin. |
| `FrozenArtifactError` | a declared fingerprint changed — an upstream phase was re-run. |
| `422 spectrum_rejected` | validation failed. `detail.validation.diagnostics` says exactly why. |
| modality rejected | only `raman` is supported. This is deliberate; see the plugin architecture. |
| slow first request | the atlas loads once (~0.3 s), then inference is ~2 ms. |

## 9. What this deployment does *not* do

No LLM, no cloud service, no telemetry, no outbound network call, no database, no authentication
(bind to loopback, which is the default), and no SERS, serum, EV, bacteria or tissue
interpretation.
