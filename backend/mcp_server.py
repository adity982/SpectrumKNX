"""Model Context Protocol (MCP) server for SpectrumKNX.

Exposes the KNX telegram store to MCP clients (Claude Desktop, Cursor, …) over
the Streamable HTTP transport, mounted into the FastAPI app at ``/mcp``.

The tool logic lives in the shared ``knx_telegram_store.mcp`` package so it is
identical to the Home Assistant consumer; this module only wraps those
functions into MCP tools and wires them to SpectrumKNX's store.

Access is gated by ``MCP_MODE``:

- ``off`` — the endpoint is not mounted.
- ``read-only`` (default) — query/introspection tools only.
- ``read-write`` — additionally exposes bus/write tools (added in a later step).
"""

import json
import os
from dataclasses import asdict
from typing import Any

from knx_telegram_store.mcp import (
    LastValuesInput,
    QueryTelegramsInput,
)
from knx_telegram_store.mcp import (
    count_telegrams as lib_count_telegrams,
)
from knx_telegram_store.mcp import (
    get_last_values as lib_get_last_values,
)
from knx_telegram_store.mcp import (
    get_store_capabilities as lib_get_store_capabilities,
)
from knx_telegram_store.mcp import (
    get_store_stats as lib_get_store_stats,
)
from knx_telegram_store.mcp import (
    query_telegrams as lib_query_telegrams,
)
from mcp.server.fastmcp import FastMCP

import knx_daemon
from database import store

MCP_MODE = os.getenv("MCP_MODE", "read-only").strip().lower()
_VALID_MODES = ("off", "read-only", "read-write")
_PROJECT_SECTIONS = ("group_addresses", "devices", "topology", "locations", "functions")

KNX_API_PROMPT = """You have access to KNX home automation tools. KNX is a wired
building automation protocol. The installation may have an ETS project loaded
that defines group addresses (communication endpoints), data point types
(DPTs), and devices.

Key concepts:
- Group addresses (GAs) are communication endpoints, for example "1/2/3".
- Each GA has a Data Point Type defining its data format, such as DPT 9.001 for temperature in °C.
- Devices have individual addresses (IAs), for example "1.1.5".
- Telegrams are bus messages: GroupValueWrite, GroupValueRead, or GroupValueResponse.
- The ETS project defines topology, locations, and functions that group related GAs.

Use project resources for installation structure and history tools for stored
telegrams, statistics, and last-known values. Bus write tools are available
only when the server is explicitly configured in read-write mode."""


def mcp_enabled() -> bool:
    """Whether the MCP endpoint should be mounted."""
    return MCP_MODE in ("read-only", "read-write")


def write_tools_enabled() -> bool:
    """Whether bus/write tools may be exposed (read-write mode only)."""
    return MCP_MODE == "read-write"


def mcp_status() -> dict[str, Any]:
    """MCP state for the server-config/status API."""
    mode = MCP_MODE if MCP_MODE in _VALID_MODES else "off"
    return {"mode": mode, "enabled": mcp_enabled(), "write_tools": write_tools_enabled()}


def _project_resource(section: str | None = None) -> str:
    """Serialize a stable, read-only view of the loaded ETS project."""
    project = knx_daemon.global_knx_project
    if not project:
        payload: dict[str, Any] = {"status": "no_project_loaded"}
        if section is not None:
            payload[section] = {}
    elif section is None:
        payload = {"status": "ok", **{name: project.get(name, {}) for name in _PROJECT_SECTIONS}}
    else:
        payload = {"status": "ok", section: project.get(section, {})}
    return json.dumps(payload, default=str, sort_keys=True)


def _build_server() -> FastMCP:
    # stateless_http keeps each request self-contained — no server-side session
    # state to manage, which is all these read tools need and simplifies mounting.
    # streamable_http_path="/" so that mounting the app at "/mcp" serves the
    # endpoint at "/mcp" rather than the doubled-up "/mcp/mcp".
    mcp = FastMCP("spectrum-knx", stateless_http=True, streamable_http_path="/")

    @mcp.resource(
        "knx://project",
        name="ETS project structure",
        description="Loaded ETS group addresses, devices, topology, locations, and functions.",
        mime_type="application/json",
    )
    def project_structure() -> str:
        return _project_resource()

    @mcp.resource(
        "knx://project/group-addresses",
        name="ETS group addresses",
        description="Group addresses and their names, DPTs, and project metadata.",
        mime_type="application/json",
    )
    def project_group_addresses() -> str:
        return _project_resource("group_addresses")

    @mcp.resource(
        "knx://project/devices",
        name="ETS devices",
        description="Devices keyed by KNX individual address.",
        mime_type="application/json",
    )
    def project_devices() -> str:
        return _project_resource("devices")

    @mcp.resource(
        "knx://project/topology",
        name="ETS topology",
        description="ETS area and line topology.",
        mime_type="application/json",
    )
    def project_topology() -> str:
        return _project_resource("topology")

    @mcp.resource(
        "knx://project/locations",
        name="ETS locations",
        description="ETS building and room structure.",
        mime_type="application/json",
    )
    def project_locations() -> str:
        return _project_resource("locations")

    @mcp.prompt()
    def analyze_bus_traffic(hours: int = 1) -> str:
        """Analyze recent KNX traffic for anomalies, noisy addresses, and unusual values."""
        return (
            f"{KNX_API_PROMPT}\n\n"
            f"Analyze KNX bus traffic from the last {hours} hour(s). Use query_telegrams, "
            "count_telegrams, and get_store_stats. Summarize traffic volume, the busiest "
            "sources and destinations, repeated or unusual values, and actionable anomalies. "
            "Do not send or modify bus values."
        )

    @mcp.prompt()
    def find_group_addresses_without_dpts() -> str:
        """Find ETS group addresses that do not have an assigned DPT."""
        return (
            f"{KNX_API_PROMPT}\n\n"
            "Read the knx://project/group-addresses resource and list every group address "
            "without an assigned DPT. Group results by project range or function when that "
            "metadata is available. Do not infer a DPT; recommend candidates separately and "
            "explain what evidence would be needed to confirm them."
        )

    @mcp.tool()
    async def query_telegrams(
        start_time: str | None = None,
        end_time: str | None = None,
        sources: list[str] | None = None,
        destinations: list[str] | None = None,
        telegram_types: list[str] | None = None,
        directions: list[str] | None = None,
        dpt_mains: list[int] | None = None,
        limit: int = 100,
        offset: int = 0,
        order_descending: bool = True,
    ) -> dict[str, Any]:
        """Search stored KNX telegrams. Times are ISO-8601; address/type/direction
        filters are lists (OR within a filter, AND across filters)."""
        result = await lib_query_telegrams(
            store,
            QueryTelegramsInput(
                start_time=start_time,
                end_time=end_time,
                sources=sources or [],
                destinations=destinations or [],
                telegram_types=telegram_types or [],
                directions=directions or [],
                dpt_mains=dpt_mains or [],
                limit=limit,
                offset=offset,
                order_descending=order_descending,
            ),
        )
        return asdict(result)

    @mcp.tool()
    async def get_last_values(destinations: list[str] | None = None) -> dict[str, Any]:
        """Most recent telegram for each group address (optionally filtered to
        the given destinations)."""
        result = await lib_get_last_values(store, LastValuesInput(destinations=destinations or []))
        return {"telegrams": [asdict(t) for t in result]}

    @mcp.tool()
    async def get_store_stats() -> dict[str, Any]:
        """Telegram count, covered time range, on-disk size, backend and retention."""
        return asdict(await lib_get_store_stats(store))

    @mcp.tool()
    async def get_store_capabilities() -> dict[str, Any]:
        """What the telegram-store backend supports (time range, pagination, size, …)."""
        return asdict(await lib_get_store_capabilities(store))

    @mcp.tool()
    async def count_telegrams() -> dict[str, Any]:
        """Total number of stored telegrams."""
        return asdict(await lib_count_telegrams(store))

    @mcp.tool()
    async def get_server_config() -> dict[str, Any]:
        """SpectrumKNX connection/security configuration (passwords masked)."""
        return knx_daemon.get_server_config()

    return mcp


_fastmcp: FastMCP | None = None
_asgi_app: Any = None


def get_asgi_app() -> Any:
    """The Streamable HTTP ASGI app to mount, built once on first use."""
    global _fastmcp, _asgi_app
    if _asgi_app is None:
        _fastmcp = _build_server()
        _asgi_app = _fastmcp.streamable_http_app()
    return _asgi_app


def session_manager_run() -> Any:
    """Async context manager that runs the MCP session manager for the app's
    lifetime. Must be entered while the endpoint is mounted."""
    get_asgi_app()  # ensure the server (and its session manager) exists
    assert _fastmcp is not None
    return _fastmcp.session_manager.run()
