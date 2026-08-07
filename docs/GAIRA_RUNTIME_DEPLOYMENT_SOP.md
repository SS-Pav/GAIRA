# GAIRA Runtime — Reproducibility SOP

How another researcher reproduces the GAIRA V7 engine from a fresh clone, on their own machine,
with no external volume, no cloud account and no network access beyond `pip`.

Every command below has been run against this repository. Where the earlier draft of this SOP
guessed an import path, the guess is corrected here and the correction is flagged.

---

## 1. Clone

```bash
git clone <repo-url> GAIRA
cd GAIRA
```

**The frozen atlas is committed.** Ten files, roughly 10 MB, under `results/v7_rebuild/`. There is
no separate model download, no Git LFS step and no external volume. `GAIRAEngine.load()` was
instrumented to confirm it opens exactly those ten files and nothing else.

## 2. Create an environment

Python 3.9 or newer; 3.12 is what this repository is developed against.

**macOS / Linux**
```bash
python -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Windows (cmd)**
```cmd
python -m venv .venv
.\.venv\Scripts\activate.bat
```

## 3. Install

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Then put `src/` on the path for the session:

```bash
export PYTHONPATH=$PWD/src            # macOS / Linux
$env:PYTHONPATH = "$PWD\src"          # Windows PowerShell
set PYTHONPATH=%CD%\src               # Windows cmd
```

Or install the package so the `gaira` command exists everywhere:

```bash
pip install -e .
```

## 4. Verify the runtime

```bash
pytest -q
```

**Read the expected count from the run rather than from this document** — it moves as phases are
added:

```bash
pytest -q --collect-only | tail -1        # how many tests exist in this checkout
```

Two outcomes are both correct, and which you get depends on one thing:

| you see | meaning |
|---|---|
| everything passes, one skip | you have the V5 raw-data volume mounted |
| ~1435 pass, ~33 skip, `test_gaira_base_2.py` errors | **normal** — those are legacy **V5** tests that need an external data volume |

The V5 tests are not part of the V7 runtime. To check only what this SOP deploys:

```bash
pytest -q -k v7
```

That must be **all green with no skips**. If it is not, stop here — something is wrong with the
checkout, not with your environment.

Then verify the atlas itself:

```bash
python -m gaira.v7.cli info
```

Expected:

```
frozen assets verified: True
atlas: 50 LSMs · 49 CSMs · 154 molecules · 16 chemistry axes
```

with two identities printed — see `results/v7_rebuild/phase10/reports/PHASE_10_ARCHITECTURE.md`
§4 for why there are two:

| term | value |
|---|---|
| **Scientific Atlas Fingerprint** (`frozen atlas`) | `09ed804a40836f4a05a91ba10900cded` |
| **Frozen Runtime Content Hash** (`atlas fingerprint`) | `2e43ddcca7d3be41c5f9da016fb8277f` |

## 5. Run the FastAPI service

```bash
python -m gaira.v7.api                       # or: gaira serve
```

Options: `--host`, `--port`, `--reload`, `--log-level`. It binds to `127.0.0.1:8000` by default —
loopback only, which is the only posture it has been reasoned about in.

Equivalent, if you prefer to invoke uvicorn yourself:

```bash
uvicorn gaira.v7.api.app:app --host 127.0.0.1 --port 8000
```

> **Correction to an earlier draft.** The ASGI application is `gaira.v7.api.app:app`, **not**
> `src.gaira.v7.runtime.api:app`. `src` is a source root on `PYTHONPATH`, not a package, and the
> HTTP layer lives in `gaira.v7.api`, not in `gaira.v7.runtime`.

Interactive documentation:

* Swagger UI — <http://127.0.0.1:8000/docs>
* ReDoc — <http://127.0.0.1:8000/redoc>
* OpenAPI JSON — <http://127.0.0.1:8000/openapi.json>

Smoke test:

```bash
curl -s http://127.0.0.1:8000/v1/health | python -m json.tool
curl -s http://127.0.0.1:8000/v1/engine | python -m json.tool | head -20
```

## 6. Run a Streamlit client

There are two, and both are canonical — they share one runtime and return identical numbers.

```bash
# the Phase 11 interactive demo (recommended)
streamlit run streamlit_apps/gaira_v7_demo.py      # or: gaira streamlit

# the Phase 10 operator console
streamlit run streamlit_apps/gaira_v7_console.py
```

> **Correction to an earlier draft.** There is no `app.py` at the repository root and no
> `src/gaira/v7/runtime/streamlit_app.py`. The entry points are the two files above.

To point a client at a deployed API instead of loading its own engine — **configuration only**:

```bash
GAIRA_API_URL=http://127.0.0.1:8000 streamlit run streamlit_apps/gaira_v7_console.py
```

## 7. Use the Python SDK

```python
from gaira.v7 import GAIRA

gaira = GAIRA.load()                    # verifies ten frozen digests, then loads the atlas

x, y, diagnostics = GAIRA.read("example.csv")
result = gaira.infer(x, y, metadata={"sample_id": "S-001", "excitation_nm": 785})

print(result.chemistry.predicted_class)
print(result.chemistry.evidence)                       # 16 relative-evidence values
print(result.retrieval.top[0].molecule, result.retrieval.top[0].similarity)
print(result.interpretation)

open("report.pdf", "wb").write(gaira.report(result, fmt="pdf"))
```

One-liner for a file:

```python
result = gaira.infer_file("example.csv")
```

Remote mode — identical results, engine elsewhere:

```python
gaira = GAIRA.remote("http://127.0.0.1:8000")
```

> **Correction to an earlier draft.** The client class is `GAIRA` in `gaira.v7` (implemented in
> `gaira.v7.sdk`), **not** `GAIRAClient` in `gaira.v7.runtime`. The method is `infer()` /
> `infer_file()`, **not** `analyze()`. The chemistry field is `result.chemistry`, **not**
> `result.chemistry_evidence`.

Inside a long-running process, prefer `GAIRA.shared()` — it returns the process-wide service so
the atlas loads exactly once.

## 8. MCP

```bash
python -m gaira.v7.mcp                  # or: gaira mcp
```

The server speaks MCP over **stdio**. It runs no language model and makes no network call; it is
a tool provider, and whatever consumes it lives entirely outside the process.

Client configuration (Claude Desktop, or any MCP-capable client):

```json
{
  "mcpServers": {
    "gaira-v7": {
      "command": "python",
      "args": ["-m", "gaira.v7.mcp"],
      "env": { "PYTHONPATH": "/absolute/path/to/GAIRA/src" }
    }
  }
}
```

Eight read-only tools:

| tool | returns |
|---|---|
| `gaira_engine_info` | versions, fingerprints, validated performance, **known limitations** |
| `gaira_validate_spectrum` | can this run, and with what caveats |
| `gaira_infer_spectrum` | the complete `InferenceResult` |
| `gaira_compare_spectra` | two spectra run independently, then compared |
| `gaira_get_molecular_evidence` | ranked analogues with per-motif score decomposition |
| `gaira_get_chemistry_evidence` | the 16 axes, calibrated |
| `gaira_explain_result` | audit, provenance, deterministic interpretation |
| `gaira_generate_report` | JSON or HTML (PDF via the HTTP API or CLI) |

Call `gaira_engine_info` first: it is the tool that states what the engine cannot do.

## 9. Expected outputs

| artefact | produced by | what it contains |
|---|---|---|
| **JSON inference** | SDK `result.model_dump()`, `POST /v1/infer`, `gaira infer --json`, MCP | preprocessing, LSM, CSM, retrieval, chemistry, confidence, audit, provenance, engine metadata, `result_digest` |
| **PDF report** | `gaira.report(result, "pdf")`, `POST /v1/report`, `gaira infer --report r.pdf` | a five-page template-driven report; no language model involved |
| **HTML report** | same generator, `fmt="html"` | self-contained, figures embedded |
| **Interactive visualisation** | either Streamlit client | spectrum, CSM activation, radar, retrieval, provenance, downloads |
| **API response** | `POST /v1/infer` | the same `InferenceResult`, serialised |

**The parity guarantee.** For one spectrum, every surface returns the same `result_digest` — an
MD5 over the scientific fields. Measured at **max |Δ| = 0.0** across seven surfaces (engine,
runtime service, SDK, CLI subprocess, HTTP API, MCP, Streamlit demo). If two surfaces ever
disagree, one of them is computing something, and that is a defect rather than a rounding
difference.

Verify it yourself:

```bash
python results/v7_rebuild/phase10/code/run_parity_and_performance.py
python results/v7_rebuild/phase11/code/run_phase11_validation.py
```

## 10. Troubleshooting

| symptom | cause and fix |
|---|---|
| `ModuleNotFoundError: gaira` | `PYTHONPATH` not set, or the venv is not active. `export PYTHONPATH=$PWD/src`, or `pip install -e .` |
| `ModuleNotFoundError: fastapi` / `pydantic` / `mcp` | an older `requirements.txt`. Re-run `pip install -r requirements.txt` |
| `FrozenAssetError` | a file under `results/v7_rebuild/` changed. Restore it: `git checkout -- results/v7_rebuild/`. **Do not re-pin the digests** — the check is the point |
| `FrozenArtifactError` | a *declared* fingerprint changed, meaning an upstream phase was re-run. Same fix |
| `[Errno 48] Address already in use` | port taken. `python -m gaira.v7.api --port 8010`, or `lsof -ti:8000 \| xargs kill` |
| Streamlit port clash | `streamlit run … --server.port 8502` |
| `422 spectrum_rejected` | validation refused the input. `detail.validation.diagnostics` says exactly which condition failed |
| `input.malformed_rows` | more than 10% of rows are not two numbers. Check the delimiter and any trailing metadata block |
| `input.duplicate_wavenumbers` | duplicate x-values; intensities are averaged per wavenumber and the count is reported |
| `coverage.insufficient` | the spectrum overlaps less than 10% of 450–1800 cm⁻¹. Below that the projection describes zero-fill, not chemistry |
| `input.format_not_supported` | `.spc`, `.jdx`, `.wdf`, `.sp` are **declared but not implemented**. Export to CSV — the refusal is deliberate, not a bug |
| modality rejected | only `raman` is supported. This is enforced, not advisory: Phase 04 measured a Raman dictionary reconstructing real Ag-SERS at AUROC 0.548 |
| `test_gaira_base_2.py` errors, ~33 skips | **expected without the V5 data volume.** Run `pytest -q -k v7` to check the V7 runtime alone |
| slow first request | the atlas loads once (~0.3 s), then inference is ~2–4 ms |
| PDF differs byte-for-byte between machines | matplotlib version. The *content* is reproducible; the bytes are reproducible on a fixed environment |

## 11. What this deployment does not do

No LLM, no cloud service, no telemetry, no outbound network call, no database, no authentication
(bind to loopback, which is the default), and no SERS, serum, plasma, EV, bacteria, tissue or
dynamic (DART) interpretation.

DART is **not** a modality: it is a dynamic perturbation protocol built on Raman/SERS
measurements — `I(wavenumber, potential, time)` is still a vibrational measurement — and it
attaches at the **trajectory layer**, downstream of the frozen representation. See
`results/v7_rebuild/phase10/reports/PHASE_10_PLUGIN_ARCHITECTURE.md`.
