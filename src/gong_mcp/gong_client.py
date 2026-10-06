"""Thin Basic-auth HTTP client for the Gong REST API.

Covers exactly what the MCP tool layer needs: listing calls in a date range
(paginated) and fetching a single call's transcript plus its parties
metadata. No write/admin endpoints are wrapped.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional

import requests

GONG_BASE_URL = "https://api.gong.io"

_DEFAULT_TIMEOUT = 30.0
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_MAX_PAGES = 500


class GongAPIError(RuntimeError):
    """Base error for failures talking to the Gong API."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class GongAuthError(GongAPIError):
    """Raised when Gong rejects the API key (HTTP 401/403)."""


class GongNotFoundError(GongAPIError):
    """Raised when a requested resource (call/transcript) doesn't exist."""


class GongRateLimitError(GongAPIError):
    """Raised when Gong's rate limit is exceeded and retries are exhausted."""


class GongPaginationLimitError(GongAPIError):
    """Raised when list_calls exceeds its pagination safety cap."""


class GongClient:
    def __init__(
        self,
        access_key: str,
        access_key_secret: str,
        base_url: str = GONG_BASE_URL,
        session: Optional[requests.Session] = None,
        timeout: float = _DEFAULT_TIMEOUT,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        max_pages: int = _DEFAULT_MAX_PAGES,
        sleep_func: Callable[[float], None] = time.sleep,
    ):
        self._access_key = access_key
        self._access_key_secret = access_key_secret
        self._base_url = base_url.rstrip("/")
        self._session = session or requests.Session()
        self._timeout = timeout
        self._max_retries = max_retries
        self._max_pages = max_pages
        self._sleep = sleep_func

    def list_calls(self, from_date: str, to_date: str) -> list[dict[str, Any]]:
        """Return call metadata for every call starting in [from_date, to_date],
        paginating through Gong's cursor as needed.

        Uses /v2/calls/extensive rather than the bare /v2/calls list
        endpoint: the bare endpoint only returns minimal fields (id,
        timestamps, duration) and omits title and parties, both of which
        callers need to identify a call by customer + day."""
        calls: list[dict[str, Any]] = []
        cursor: Optional[str] = None
        pages = 0
        from_datetime = _start_of_day(from_date)
        to_datetime = _end_of_day(to_date)

        while True:
            filter_body: dict[str, Any] = {
                "fromDateTime": from_datetime,
                "toDateTime": to_datetime,
            }
            if cursor:
                filter_body["cursor"] = cursor

            payload = self._request(
                "POST",
                "/v2/calls/extensive",
                json_body={
                    "filter": filter_body,
                    "contentSelector": {"exposedFields": {"parties": True}},
                },
            )
            calls.extend(payload.get("calls", []))
            pages += 1

            cursor = (payload.get("records") or {}).get("cursor")
            if not cursor:
                break
            if pages >= self._max_pages:
                raise GongPaginationLimitError(
                    f"list_calls exceeded {self._max_pages} pages for range "
                    f"{from_date}..{to_date}; narrow the date range."
                )

        return calls

    def get_transcript(self, call_id: str) -> dict[str, Any]:
        """Return the raw transcript segments and parties metadata for a call."""
        transcript_payload = self._request(
            "POST",
            "/v2/calls/transcript",
            json_body={"filter": {"callIds": [call_id]}},
        )
        transcripts = transcript_payload.get("callTranscripts", [])
        transcript_entry = next(
            (t for t in transcripts if t.get("callId") == call_id), None
        )
        if transcript_entry is None:
            raise GongNotFoundError(f"No transcript found for call '{call_id}'.")

        parties_payload = self._request(
            "POST",
            "/v2/calls/extensive",
            json_body={
                "filter": {"callIds": [call_id]},
                "contentSelector": {"exposedFields": {"parties": True}},
            },
        )
        extensive_calls = parties_payload.get("calls", [])
        call_entry = next(
            (c for c in extensive_calls if c.get("metaData", {}).get("id") == call_id),
            None,
        )
        parties = call_entry.get("parties", []) if call_entry else []

        return {
            "call_id": call_id,
            "transcript": transcript_entry.get("transcript", []),
            "parties": parties,
        }

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[dict[str, Any]] = None,
        json_body: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        url = f"{self._base_url}{path}"
        attempt = 0

        while True:
            response = self._session.request(
                method,
                url,
                params=params,
                json=json_body,
                auth=(self._access_key, self._access_key_secret),
                timeout=self._timeout,
            )

            if response.status_code == 429 and attempt < self._max_retries:
                self._sleep(_parse_retry_after(response))
                attempt += 1
                continue

            if response.status_code in (401, 403):
                raise GongAuthError(
                    f"Gong API rejected the request (HTTP {response.status_code}). "
                    "Check the API key.",
                    status_code=response.status_code,
                )

            if response.status_code == 404:
                raise GongNotFoundError(
                    f"Gong API returned 404 for {method} {path}.",
                    status_code=404,
                )

            if response.status_code == 429:
                raise GongRateLimitError(
                    f"Gong API rate limit exceeded for {method} {path} after "
                    f"{self._max_retries} retries.",
                    status_code=429,
                )

            if response.status_code >= 400:
                raise GongAPIError(
                    f"Gong API error {response.status_code} for {method} {path}: "
                    f"{_safe_body(response)}",
                    status_code=response.status_code,
                )

            return response.json()


def _start_of_day(date_str: str) -> str:
    """Expand a bare ISO date (YYYY-MM-DD) to the start of that day in UTC.
    Gong's /v2/calls/extensive rejects bare dates for fromDateTime/toDateTime
    with a JSON parse error — it requires a full ISO-8601 datetime. Strings
    that already carry a time component are passed through unchanged."""
    if "T" in date_str:
        return date_str
    return f"{date_str}T00:00:00Z"


def _end_of_day(date_str: str) -> str:
    """Expand a bare ISO date (YYYY-MM-DD) to the end of that day in UTC, so
    the range stays inclusive of the whole `to_date` day. See _start_of_day."""
    if "T" in date_str:
        return date_str
    return f"{date_str}T23:59:59Z"


def _parse_retry_after(response: requests.Response) -> float:
    header = response.headers.get("Retry-After")
    if header is None:
        return 1.0
    try:
        return max(float(header), 0.0)
    except ValueError:
        return 1.0


def _safe_body(response: requests.Response) -> str:
    try:
        return response.text[:500]
    except Exception:
        return "<unavailable>"
