from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from gong_mcp.__main__ import LazyGongClient
from gong_mcp.secrets import OnePasswordNotAuthenticatedError


def test_constructing_lazy_client_does_not_fetch_credentials():
    provider = MagicMock(return_value=("key", "secret"))
    LazyGongClient(credentials_provider=provider, client_factory=MagicMock())
    provider.assert_not_called()


def test_first_call_fetches_credentials_once_and_reuses_client():
    provider = MagicMock(return_value=("key", "secret"))
    real_client = MagicMock()
    real_client.list_calls.return_value = ["call"]
    real_client.get_transcript.return_value = {"transcript": []}
    factory = MagicMock(return_value=real_client)

    lazy = LazyGongClient(credentials_provider=provider, client_factory=factory)
    assert lazy.list_calls("2026-10-01", "2026-10-02") == ["call"]
    assert lazy.get_transcript("abc") == {"transcript": []}
    lazy.list_calls("2026-10-01", "2026-10-02")

    provider.assert_called_once()
    factory.assert_called_once_with("key", "secret")
    real_client.list_calls.assert_called_with("2026-10-01", "2026-10-02")
    real_client.get_transcript.assert_called_once_with("abc")


def test_failed_fetch_is_not_cached_and_can_be_retried():
    provider = MagicMock(
        side_effect=[OnePasswordNotAuthenticatedError("dismissed"), ("key", "secret")]
    )
    real_client = MagicMock()
    lazy = LazyGongClient(
        credentials_provider=provider, client_factory=MagicMock(return_value=real_client)
    )

    with pytest.raises(OnePasswordNotAuthenticatedError):
        lazy.list_calls("2026-10-01", "2026-10-02")
    lazy.list_calls("2026-10-01", "2026-10-02")

    assert provider.call_count == 2
    real_client.list_calls.assert_called_once()
