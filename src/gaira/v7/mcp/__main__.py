"""Run the GAIRA V7 MCP server on stdio: `python -m gaira.v7.mcp`."""
from __future__ import annotations

import asyncio
import logging
import sys


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s %(name)s %(message)s")
    from .server import serve
    asyncio.run(serve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
