"""Entry-point shim — preserves `python -m src.server` without Dockerfile changes.

All business logic lives in src/mcp_server/. This file exists so that:
  - CMD ["python", "-m", "src.server"] in Dockerfile continues to work.
  - src/cli/ingest.py and experiments/ingest_smoke_test.py can continue to use
    `import src.server as server; server.sync_repository(...)` without changes.
"""

import os

# Import server module for route/tool registration side-effect — must come before mcp.run()
from src.mcp_server import server as _server_routes  # noqa: F401

# Re-export public symbols consumed by external callers (cli/ingest.py, ingest_smoke_test.py)
from src.mcp_server.container import mcp
from src.mcp_server.server import sync_repository  # noqa: F401 — re-exported for callers
from src.mcp_server.startup import logger

if __name__ == "__main__":
    transport = os.environ.get("MCP_TRANSPORT", "http")
    host = "0.0.0.0"
    port = int(os.environ.get("MCP_PORT", "9000"))
    logger.info(
        "Starting Developer Memory MCP server: transport=%s host=%s port=%d",
        transport, host, port,
    )
    mcp.run(transport=transport, host=host, port=port)
