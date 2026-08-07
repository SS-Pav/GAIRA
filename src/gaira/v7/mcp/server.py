"""GAIRA V7 — Phase 10: the MCP tool server.

Exposes the frozen engine as callable tools over stdio. **No language model runs here** and no
network call is made: the server is a provider, and whatever consumes it — an agent, a script, a
test harness — lives entirely outside this process.
"""
from __future__ import annotations

import json
import logging

from gaira.v7 import __version__
from gaira.v7.runtime.service import GAIRAService, SpectrumRejected

from .tools import TOOLS, call

log = logging.getLogger("gaira.v7.mcp")

INSTRUCTIONS = """GAIRA V7 — Grounded Raman Biochemical Inference.

Project a Raman spectrum into a frozen biochemical motif atlas, retrieve grounded reference
evidence, and obtain an interpretable 16-axis Chemistry Evidence profile.

Read these before using the results:

- Chemistry Evidence is RELATIVE. It is not a concentration, an abundance, or a mixture
  fraction. A tall axis means evidence associated with that chemistry is present, nothing more.
- Retrieved molecules are reference ANALOGUES, not identifications. Validated top-1 is 0.6053.
- Only modality 'raman' is supported. Any other modality is rejected, not silently run.
- sample_type is metadata only. V7 has validated interpretation for pure reference compounds
  and nothing else; a serum or EV sample produces a scope warning and unchanged arithmetic.
- The engine has NO validated open-set detection. It cannot tell you the true molecule is
  absent from its 154-molecule bank. Low confidence is a caution signal, not proof of novelty.
"""


async def serve() -> None:
    import mcp.types as types
    from mcp.server import Server
    from mcp.server.stdio import stdio_server

    svc = GAIRAService.instance()
    info = svc.engine_info()
    log.info("GAIRA V7 MCP server: atlas=%s molecules=%d",
             info.atlas_fingerprint[:12], info.n_molecules)

    server = Server("gaira-v7", version=__version__, instructions=INSTRUCTIONS)

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [types.Tool(name=t["name"], description=t["description"],
                           inputSchema=t["inputSchema"]) for t in TOOLS]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
        try:
            payload = call(name, arguments, svc)
        except SpectrumRejected as rejected:
            payload = {"error": "spectrum_rejected", "message": str(rejected),
                       "validation": rejected.validation.model_dump(mode="json")}
        except ValueError as exc:
            payload = {"error": "invalid_arguments", "message": str(exc)}
        return [types.TextContent(type="text",
                                  text=json.dumps(payload, indent=2, default=str))]

    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())
