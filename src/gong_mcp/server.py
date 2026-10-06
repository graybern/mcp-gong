"""MCP stdio server wiring for the Gong tools.

Exposes exactly two tools to Claude — `list_calls` and `get_transcript` — and
nothing else. The validation/business logic below is plain, synchronous, and
free of `mcp` types so it can be unit tested directly; `create_server` only
adds the thin async glue the `mcp` SDK requires.
"""

from __future__ import annotations

import os
from datetime import date
from typing import Any, Optional

from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from .formatter import format_transcript, resolve_role
from .gong_client import GongClient

SERVER_NAME = "gong-mcp-server"

LIST_CALLS_TOOL = "list_calls"
GET_TRANSCRIPT_TOOL = "get_transcript"

MAX_DATE_RANGE_DAYS = 31


class ToolInputError(ValueError):
    """Raised when a tool call's arguments fail validation."""


def build_tool_definitions() -> list[types.Tool]:
    return [
        types.Tool(
            name=LIST_CALLS_TOOL,
            description=(
                "List Gong calls that started within a date range, with enough "
                "metadata (title, start time, account name) to identify a "
                "specific customer's call by cross-referencing a known date."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "from_date": {
                        "type": "string",
                        "description": "Start of the range, ISO date (YYYY-MM-DD).",
                    },
                    "to_date": {
                        "type": "string",
                        "description": "End of the range, ISO date (YYYY-MM-DD), inclusive.",
                    },
                },
                "required": ["from_date", "to_date"],
            },
        ),
        types.Tool(
            name=GET_TRANSCRIPT_TOOL,
            description=(
                "Download a single call's transcript as plain text, with each "
                "utterance labeled by speaker name and Customer/Internal role, "
                "in chronological order. By default the transcript text is "
                "returned inline. If 'output_path' is given, the server writes "
                "the transcript to that file itself and returns only a "
                "write-status summary (call_id, output_path, bytes_written) "
                "with no transcript text — use this to persist transcripts "
                "deterministically, without the text passing through the model."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "call_id": {
                        "type": "string",
                        "description": "Gong call ID, as returned by list_calls.",
                    },
                    "output_path": {
                        "type": "string",
                        "description": (
                            "Optional. Absolute filesystem path to write the "
                            "transcript to (parent directories are created "
                            "automatically). When set, the response omits the "
                            "transcript text and returns only a write-status "
                            "summary. Omit to receive the transcript inline."
                        ),
                    },
                },
                "required": ["call_id"],
            },
        ),
    ]


def validate_date_range(
    from_date: str, to_date: str, max_days: int = MAX_DATE_RANGE_DAYS
) -> None:
    """Raise ToolInputError if either date is malformed, the range is
    inverted, or it spans more than `max_days` (Gong has no server-side
    account filter, so an oversized range would pull excessive call volume
    in one request instead of Claude narrowing/paging by date itself)."""
    start = _parse_date(from_date, "from_date")
    end = _parse_date(to_date, "to_date")

    if end < start:
        raise ToolInputError(
            f"'to_date' ({to_date}) must not be before 'from_date' ({from_date})."
        )

    span_days = (end - start).days
    if span_days > max_days:
        raise ToolInputError(
            f"Date range spans {span_days} days, which exceeds the "
            f"{max_days}-day maximum. Narrow the range and issue multiple "
            "calls if needed."
        )


def _parse_date(value: str, field_name: str) -> date:
    if not isinstance(value, str):
        raise ToolInputError(f"'{field_name}' must be an ISO date string (YYYY-MM-DD).")
    try:
        return date.fromisoformat(value[:10])
    except ValueError as exc:
        raise ToolInputError(
            f"'{field_name}' must be an ISO date (YYYY-MM-DD), got {value!r}."
        ) from exc


def summarize_call(call: dict[str, Any]) -> dict[str, Any]:
    """Reduce a raw Gong call record to the lightweight fields Claude needs to
    disambiguate by customer + day, dropping everything else Gong returns.

    `/v2/calls/extensive` (what `list_calls` now queries) nests id/title/
    start time under `metaData`, with `parties` as a top-level sibling."""
    meta = call.get("metaData") or {}
    return {
        "call_id": meta.get("id"),
        "title": meta.get("title"),
        "start_time": meta.get("started"),
        "account_name": _extract_account_name(call),
        "participants_summary": _summarize_participants(call),
    }


def _extract_account_name(call: dict[str, Any]) -> Optional[str]:
    account = call.get("account")
    if isinstance(account, dict) and account.get("name"):
        return account["name"]
    for party in call.get("parties") or []:
        if party.get("affiliation") == "External" and party.get("accountName"):
            return party["accountName"]
    return None


def _summarize_participants(call: dict[str, Any]) -> str:
    labels = [
        f"{party.get('name') or 'Unknown Speaker'} ({resolve_role(party.get('affiliation'))})"
        for party in call.get("parties") or []
    ]
    return ", ".join(labels)


def handle_list_calls(gong_client: GongClient, arguments: dict[str, Any]) -> dict[str, Any]:
    from_date = arguments.get("from_date")
    to_date = arguments.get("to_date")
    if not from_date or not to_date:
        raise ToolInputError("Both 'from_date' and 'to_date' are required.")

    validate_date_range(from_date, to_date)

    calls = gong_client.list_calls(from_date, to_date)
    return {"calls": [summarize_call(call) for call in calls]}


def handle_get_transcript(gong_client: GongClient, arguments: dict[str, Any]) -> dict[str, Any]:
    call_id = arguments.get("call_id")
    if not call_id or not isinstance(call_id, str):
        raise ToolInputError("'call_id' is required and must be a non-empty string.")

    output_path = arguments.get("output_path")
    if output_path is not None and (
        not isinstance(output_path, str) or not output_path.strip()
    ):
        raise ToolInputError(
            "'output_path', when provided, must be a non-empty string."
        )

    data = gong_client.get_transcript(call_id)
    transcript_text = format_transcript(data["transcript"], data["parties"])

    if output_path:
        written_path, bytes_written = _write_transcript_file(output_path, transcript_text)
        return {
            "call_id": call_id,
            "output_path": written_path,
            "bytes_written": bytes_written,
            "status": "written",
        }

    return {"call_id": call_id, "transcript": transcript_text}


def _write_transcript_file(output_path: str, transcript_text: str) -> tuple[str, int]:
    """Write `transcript_text` to `output_path` (UTF-8), creating parent
    directories as needed, and return the absolute path plus bytes written.

    Writing happens here in the server process so the transcript text never
    has to be round-tripped through the model to reach disk — the caller only
    ever sees the returned status summary."""
    path = os.path.abspath(os.path.expanduser(output_path))
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    encoded = transcript_text.encode("utf-8")
    with open(path, "wb") as handle:
        handle.write(encoded)
    return path, len(encoded)


def dispatch_tool_call(
    gong_client: GongClient, name: str, arguments: dict[str, Any]
) -> dict[str, Any]:
    if name == LIST_CALLS_TOOL:
        return handle_list_calls(gong_client, arguments)
    if name == GET_TRANSCRIPT_TOOL:
        return handle_get_transcript(gong_client, arguments)
    raise ToolInputError(f"Unknown tool '{name}'.")


def create_server(gong_client: GongClient) -> Server:
    server = Server(SERVER_NAME)

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return build_tool_definitions()

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return dispatch_tool_call(gong_client, name, arguments)

    return server


async def run(gong_client: GongClient) -> None:
    server = create_server(gong_client)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream, server.create_initialization_options()
        )
