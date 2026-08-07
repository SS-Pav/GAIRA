# GAIRA V7 — MCP Server Specification

`gaira.v7.mcp` · stdio · `gaira mcp` or `python -m gaira.v7.mcp`

**No language model runs in this server**, and it makes no network call. It is a tool provider;
whatever consumes it lives entirely outside the process. The permitted chain is strictly

```
        LLM  →  MCP  →  Frozen Runtime  →  Frozen Engine
```

and never `LLM → scientific computation`.

Client configuration:
```json
{"mcpServers": {"gaira-v7": {"command": "python", "args": ["-m", "gaira.v7.mcp"],
                             "env": {"PYTHONPATH": "/path/to/GAIRA/src"}}}}
```

---

## The eight tools

| tool | returns |
|---|---|
| `gaira_engine_info` | versions, fingerprints, corpus, validated performance, **known limitations** |
| `gaira_validate_spectrum` | `can_run` + three-severity diagnostics, without running |
| `gaira_infer_spectrum` | the complete `InferenceResult` |
| `gaira_compare_spectra` | two spectra run independently, then compared |
| `gaira_get_molecular_evidence` | ranked analogues with per-CSM score decomposition |
| `gaira_get_chemistry_evidence` | 16 axes, calibrated |
| `gaira_explain_result` | audit, provenance, preprocessing, deterministic interpretation |
| `gaira_generate_report` | JSON or HTML (PDF via the HTTP API or CLI, which can return binary) |

Every tool accepts either `{"wavenumber": [...], "intensity": [...]}` or `{"text": "<CSV>"}`,
parsed by the same adapters the CLI uses.

## Why the tools are coarse

An agent should be able to ask *"interpret this spectrum"* and get a scientifically coherent
answer. It should **not** be able to assemble its own inference path out of primitives. No tool
exposes NNLS, a raw dictionary matrix, or an arbitrary projection, because those are where a
caller could construct a result the engine never sanctioned. A test asserts that no tool name
contains `nnls`, `project`, `matrix`, `raw_activation`, `dictionary` or `eval`.

## Guarantees

- Every tool routes through `GAIRAService`; a static AST test fails the build if the MCP package
  references a scientific primitive or imports a scientific module.
- The narrow tools return **exactly** the corresponding slice of `gaira_infer_spectrum` for the
  same input, verified by digest equality.
- MCP results equal SDK and HTTP results to machine precision (measured: 0.0).
- Unsupported modality is rejected on this surface as on every other.
- Errors return structured JSON (`error`, `message`, and `validation` where applicable) rather
  than raising through the transport.

## Server instructions

The server advertises the scope in its MCP `instructions` field, so a connecting client receives
the caveats before its first call: relative evidence, reference analogues, Raman-only,
`sample_type` is metadata, and no validated open-set detection.
