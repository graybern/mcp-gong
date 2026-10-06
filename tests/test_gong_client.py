from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from gong_mcp.gong_client import (
    GongAPIError,
    GongAuthError,
    GongClient,
    GongNotFoundError,
    GongPaginationLimitError,
    GongRateLimitError,
)


def _response(status_code=200, json_data=None, headers=None, text=""):
    mock = MagicMock()
    mock.status_code = status_code
    mock.json.return_value = json_data or {}
    mock.headers = headers or {}
    mock.text = text
    return mock


def _client(session, **kwargs):
    return GongClient(
        access_key="test-key",
        access_key_secret="test-secret",
        session=session,
        sleep_func=lambda _seconds: None,
        **kwargs,
    )


def test_list_calls_single_page():
    session = MagicMock()
    session.request.return_value = _response(
        json_data={
            "calls": [{"metaData": {"id": "1"}}, {"metaData": {"id": "2"}}],
            "records": {"cursor": None},
        }
    )
    client = _client(session)

    calls = client.list_calls("2026-01-01", "2026-01-31")

    assert [c["metaData"]["id"] for c in calls] == ["1", "2"]
    session.request.assert_called_once()
    called = session.request.call_args
    assert called.args[:2] == ("POST", "https://api.gong.io/v2/calls/extensive")
    assert called.kwargs["json"] == {
        "filter": {"fromDateTime": "2026-01-01T00:00:00Z", "toDateTime": "2026-01-31T23:59:59Z"},
        "contentSelector": {"exposedFields": {"parties": True}},
    }
    assert called.kwargs["auth"] == ("test-key", "test-secret")


def test_list_calls_paginates_until_cursor_exhausted():
    session = MagicMock()
    session.request.side_effect = [
        _response(
            json_data={"calls": [{"metaData": {"id": "1"}}], "records": {"cursor": "page-2"}}
        ),
        _response(
            json_data={"calls": [{"metaData": {"id": "2"}}], "records": {"cursor": None}}
        ),
    ]
    client = _client(session)

    calls = client.list_calls("2026-01-01", "2026-01-31")

    assert [c["metaData"]["id"] for c in calls] == ["1", "2"]
    assert session.request.call_count == 2
    second_call_json = session.request.call_args_list[1].kwargs["json"]
    assert second_call_json["filter"]["cursor"] == "page-2"


def test_list_calls_pagination_safety_cap():
    session = MagicMock()
    session.request.return_value = _response(
        json_data={"calls": [], "records": {"cursor": "always-more"}}
    )
    client = _client(session, max_pages=3)

    with pytest.raises(GongPaginationLimitError):
        client.list_calls("2026-01-01", "2026-01-31")

    assert session.request.call_count == 3


def test_list_calls_auth_failure():
    session = MagicMock()
    session.request.return_value = _response(status_code=401, text="unauthorized")
    client = _client(session)

    with pytest.raises(GongAuthError):
        client.list_calls("2026-01-01", "2026-01-31")


def test_list_calls_rate_limit_retries_then_succeeds():
    session = MagicMock()
    session.request.side_effect = [
        _response(status_code=429, headers={"Retry-After": "0"}),
        _response(json_data={"calls": [{"id": "1"}], "records": {"cursor": None}}),
    ]
    client = _client(session)

    calls = client.list_calls("2026-01-01", "2026-01-31")

    assert [c["id"] for c in calls] == ["1"]
    assert session.request.call_count == 2


def test_list_calls_rate_limit_exhausts_retries():
    session = MagicMock()
    session.request.return_value = _response(status_code=429, headers={})
    client = _client(session, max_retries=2)

    with pytest.raises(GongRateLimitError):
        client.list_calls("2026-01-01", "2026-01-31")

    assert session.request.call_count == 3  # initial attempt + 2 retries


def test_list_calls_generic_server_error():
    session = MagicMock()
    session.request.return_value = _response(status_code=500, text="boom")
    client = _client(session)

    with pytest.raises(GongAPIError):
        client.list_calls("2026-01-01", "2026-01-31")


def test_get_transcript_merges_transcript_and_parties():
    session = MagicMock()
    session.request.side_effect = [
        _response(
            json_data={
                "callTranscripts": [
                    {
                        "callId": "call-1",
                        "transcript": [
                            {"speakerId": "s1", "sentences": [{"text": "hello"}]}
                        ],
                    }
                ]
            }
        ),
        _response(
            json_data={
                "calls": [
                    {
                        "metaData": {"id": "call-1"},
                        "parties": [
                            {"speakerId": "s1", "name": "Jane Doe", "affiliation": "Internal"}
                        ],
                    }
                ]
            }
        ),
    ]
    client = _client(session)

    result = client.get_transcript("call-1")

    assert result["call_id"] == "call-1"
    assert result["transcript"][0]["sentences"][0]["text"] == "hello"
    assert result["parties"][0]["name"] == "Jane Doe"

    first_call = session.request.call_args_list[0]
    assert first_call.args[:2] == ("POST", "https://api.gong.io/v2/calls/transcript")
    second_call = session.request.call_args_list[1]
    assert second_call.args[:2] == ("POST", "https://api.gong.io/v2/calls/extensive")


def test_get_transcript_not_found_raises():
    session = MagicMock()
    session.request.return_value = _response(json_data={"callTranscripts": []})
    client = _client(session)

    with pytest.raises(GongNotFoundError):
        client.get_transcript("missing-call")


def test_get_transcript_missing_parties_defaults_to_empty_list():
    session = MagicMock()
    session.request.side_effect = [
        _response(json_data={"callTranscripts": [{"callId": "call-1", "transcript": []}]}),
        _response(json_data={"calls": []}),
    ]
    client = _client(session)

    result = client.get_transcript("call-1")

    assert result["parties"] == []


def test_get_transcript_auth_failure_propagates():
    session = MagicMock()
    session.request.return_value = _response(status_code=403, text="forbidden")
    client = _client(session)

    with pytest.raises(GongAuthError):
        client.get_transcript("call-1")
