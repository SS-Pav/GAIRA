"""Run the GAIRA V7 API: `python -m gaira.v7.api [--host H] [--port P]`."""
from __future__ import annotations

import argparse


def main() -> int:
    import uvicorn
    p = argparse.ArgumentParser(prog="python -m gaira.v7.api",
                                description="GAIRA V7 HTTP inference service")
    p.add_argument("--host", default="127.0.0.1", help="bind address (default: loopback only)")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")
    p.add_argument("--log-level", default="info")
    a = p.parse_args()
    uvicorn.run("gaira.v7.api.app:app", host=a.host, port=a.port, reload=a.reload,
                log_level=a.log_level)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
