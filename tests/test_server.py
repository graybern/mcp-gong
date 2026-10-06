from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest
from mcp import types

from gong_mcp.gong_client import GongNotFoundError
from gong_mcp.server import (
    GET_TRANSCRIPT_TOOL,
    LIST_CALLS_TOOL,
    ToolInputError,
    build_tool_definitions,
    create_server,
    dispatch_tool_call,
    handle_get_transcript,
    handle_list_calls,
    summarize_call,
    validate_date_range,
)


# --- validate_date_range -----------------------------------------------------


def test_validate_date_range_accepts_range_within_limit():
    validate_date_range("2026-01-01", "2026-01-15")  # should not raise


def test_validate_date_range_rejects_malformed_date():
    with pytest.raises(ToolInputError, match="from_date"):
        validate_date_range("not-a-date", "2026-01-15")


def test_validate_date_range_rejects_inverted_range():
    with pytest.raises(ToolInputError, match="must not be before"):
        validate_date_range("2026-01-15", "2026-01-01")


def test_validate_date_range_rejects_oversized_range():
    with pytest.raises(ToolInputError, match="exceeds"):
        validate_date_range("2026-01-01", "2026-12-31")


def test_validate_date_range_accepts_custom_max_days():
    validate_date_range("2026-01-01", "2026-02-01", max_days=45)


# --- summarize_call -----------------------------------------------------------


def test_summarize_call_extracts_expected_fields():
    call = {
        "metaData": {
            "id": "call-1",
            "title": "Weekly sync",
            "started": "2026-01-05T15:00:00Z",
        },
        "parties": [
            {"speakerId": "s1", "name": "Jane Doe", "affiliation": "External", "accountName": "Acme Co"},
            {"speakerId": "s2", "name": "Alex Smith", "affiliation": "Internal"},
        ],
    }

    summary = summarize_call(call)

    assert summary == {
        "call_id": "call-1",
        "title": "Weekly sync",
        "start_time": "2026-01-05T15:00:00Z",
        "account_name": "Acme Co",
        "participants_summary": "Jane Doe (Customer), Alex Smith (Internal)",
    }


def test_summarize_call_handles_missing_optional_fields():
    summary = summarize_call({"metaData": {"id": "call-2"}})

    assert summary == {
        "call_id": "call-2",
        "title": None,
        "start_time": None,
        "account_name": None,
        "participants_summary": "",
    }


def test_summarize_call_handles_missing_metadata_key():
    summary = summarize_call({"parties": []})

    assert summary == {
        "call_id": None,
        "title": None,
        "start_time": None,
        "account_name": None,
        "participants_summary": "",
    }


# --- handle_list_calls ---------------------------------------------------------


def test_handle_list_calls_returns_summarized_calls():
    gong_client = MagicMock()
    gong_client.list_calls.return_value = [{"metaData": {"id": "call-1", "title": "Sync"}}]

    result = handle_list_calls(gong_client, {"from_date": "2026-01-01", "to_date": "2026-01-05"})

    assert result == {
        "calls": [
            {
                "call_id": "call-1",
                "title": "Sync",
                "start_time": None,
                "account_name": None,
                "participants_summary": "",
            }
        ]
    }
    gong_client.list_calls.assert_called_once_with("2026-01-01", "2026-01-05")


def test_handle_list_calls_rejects_missing_dates():
    gong_client = MagicMock()

    with pytest.raises(ToolInputError, match="required"):
        handle_list_calls(gong_client, {"from_date": "2026-01-01"})

    gong_client.list_calls.assert_not_called()


def test_handle_list_calls_rejects_oversized_range_before_calling_client():
    gong_client = MagicMock()

    with pytest.raises(ToolInputError, match="exceeds"):
        handle_list_calls(gong_client, {"from_date": "2026-01-01", "to_date": "2026-12-31"})

    gong_client.list_calls.assert_not_called()


# --- handle_get_transcript ------------------------------------------------------


def test_handle_get_transcript_returns_formatted_text():
    gong_client = MagicMock()
    gong_client.get_transcript.return_value = {
        "call_id": "call-1",
        "transcript": [{"speakerId": "s1", "sentences": [{"start": 0, "text": "hi"}]}],
        "parties": [{"speakerId": "s1", "name": "Jane Doe", "affiliation": "External"}],
    }

    result = handle_get_transcript(gong_client, {"call_id": "call-1"})

    assert result == {
        "call_id": "call-1",
        "transcript": "[00:00:00] Customer (Jane Doe): hi",
    }


def test_handle_get_transcript_writes_file_and_omits_text_when_output_path_given(tmp_path):
    gong_client = MagicMock()
    gong_client.get_transcript.return_value = {
        "call_id": "call-1",
        "transcript": [{"speakerId": "s1", "sentences": [{"start": 0, "text": "hi"}]}],
        "parties": [{"speakerId": "s1", "name": "Jane Doe", "affiliation": "External"}],
    }
    dest = tmp_path / "nested" / "call-1.txt"

    result = handle_get_transcript(
        gong_client, {"call_id": "call-1", "output_path": str(dest)}
    )

    expected_text = "[00:00:00] Customer (Jane Doe): hi"
    assert dest.read_text(encoding="utf-8") == expected_text
    assert "transcript" not in result
    assert result == {
        "call_id": "call-1",
        "output_path": str(dest),
        "bytes_written": len(expected_text.encode("utf-8")),
        "status": "written",
    }


def test_handle_get_transcript_rejects_blank_output_path():
    gong_client = MagicMock()

    with pytest.raises(ToolInputError, match="output_path"):
        handle_get_transcript(gong_client, {"call_id": "call-1", "output_path": "   "})

    gong_client.get_transcript.assert_not_called()


def test_get_transcript_schema_exposes_output_path():
    tools = {tool.name: tool for tool in build_tool_definitions()}
    props = tools[GET_TRANSCRIPT_TOOL].inputSchema["properties"]

    assert "output_path" in props
    assert "output_path" not in tools[GET_TRANSCRIPT_TOOL].inputSchema.get("required", [])


def test_handle_get_transcript_rejects_missing_call_id():
    gong_client = MagicMock()

    with pytest.raises(ToolInputError, match="call_id"):
        handle_get_transcript(gong_client, {})

    gong_client.get_transcript.assert_not_called()


def test_handle_get_transcript_rejects_non_string_call_id():
    gong_client = MagicMock()

    with pytest.raises(ToolInputError, match="call_id"):
        handle_get_transcript(gong_client, {"call_id": 12345})

    gong_client.get_transcript.assert_not_called()


def test_handle_get_transcript_propagates_not_found():
    gong_client = MagicMock()
    gong_client.get_transcript.side_effect = GongNotFoundError("no such call")

    with pytest.raises(GongNotFoundError):
        handle_get_transcript(gong_client, {"call_id": "missing"})


# --- dispatch_tool_call ----------------------------------------------------------


def test_dispatch_tool_call_routes_to_list_calls():
    gong_client = MagicMock()
    gong_client.list_calls.return_value = []

    result = dispatch_tool_call(
        gong_client, LIST_CALLS_TOOL, {"from_date": "2026-01-01", "to_date": "2026-01-02"}
    )

    assert result == {"calls": []}


def test_dispatch_tool_call_rejects_unknown_tool():
    gong_client = MagicMock()

    with pytest.raises(ToolInputError, match="Unknown tool"):
        dispatch_tool_call(gong_client, "delete_everything", {})


# --- tool registration / schema shape --------------------------------------------


def test_build_tool_definitions_exposes_exactly_two_tools():
    tools = build_tool_definitions()

    assert [tool.name for tool in tools] == [LIST_CALLS_TOOL, GET_TRANSCRIPT_TOOL]
    for tool in tools:
        assert tool.inputSchema["type"] == "object"


def test_build_tool_definitions_never_mentions_api_key():
    tools = build_tool_definitions()

    for tool in tools:
        blob = f"{tool.name} {tool.description} {tool.inputSchema}".lower()
        assert "api_key" not in blob
        assert "api key" not in blob


def test_create_server_registers_exactly_two_tools_end_to_end():
    gong_client = MagicMock()
    gong_client.list_calls.return_value = [{"metaData": {"id": "call-1"}}]
    server = create_server(gong_client)

    list_tools_handler = server.request_handlers[types.ListToolsRequest]
    result = asyncio.run(list_tools_handler(types.ListToolsRequest()))

    tool_names = [tool.name for tool in result.root.tools]
    assert tool_names == [LIST_CALLS_TOOL, GET_TRANSCRIPT_TOOL]


def test_create_server_call_tool_dispatches_and_returns_structured_content():
    gong_client = MagicMock()
    gong_client.list_calls.return_value = [{"metaData": {"id": "call-1"}}]
    server = create_server(gong_client)

    call_tool_handler = server.request_handlers[types.CallToolRequest]
    request = types.CallToolRequest(
        params=types.CallToolRequestParams(
            name=LIST_CALLS_TOOL,
            arguments={"from_date": "2026-01-01", "to_date": "2026-01-02"},
        )
    )
    result = asyncio.run(call_tool_handler(request))

    assert result.root.isError is False
    assert result.root.structuredContent["calls"][0]["call_id"] == "call-1"


def test_create_server_call_tool_surfaces_validation_error_without_leaking_key():
    gong_client = MagicMock()
    server = create_server(gong_client)

    call_tool_handler = server.request_handlers[types.CallToolRequest]
    request = types.CallToolRequest(
        params=types.CallToolRequestParams(
            name=LIST_CALLS_TOOL,
            arguments={"from_date": "2026-01-01", "to_date": "2026-12-31"},
        )
    )
    result = asyncio.run(call_tool_handler(request))

    assert result.root.isError is True
    error_text = result.root.content[0].text
    assert "exceeds" in error_text
    gong_client.list_calls.assert_not_called()
