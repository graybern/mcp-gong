"""Entry point: start the stdio server, fetching the Gong API key lazily.

Credentials are NOT fetched at process startup. Claude Desktop launches this
server, kills it, and relaunches it within a couple of seconds every time the
app opens, and every launch that touched 1Password cost the user a Touch ID /
MFA prompt. Deferring the `op` call to the first tool call means opening
Claude costs zero prompts, and the first Gong tool call in a session costs
exactly one.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Optional

from .gong_client import GongClient
from .secrets import get_gong_credentials
from .server import run


class LazyGongClient:
    """Stand-in for `GongClient` that builds the real client (and fetches
    credentials from 1Password) on first use.

    A failed fetch isn't cached, so a dismissed 1Password prompt can simply
    be retried by calling the tool again. Any `SecretRetrievalError` raised
    here surfaces to Claude as a tool error; its messages never contain
    secret values.
    """

    def __init__(
        self,
        credentials_provider: Callable[[], tuple[str, str]] = get_gong_credentials,
        client_factory: Callable[[str, str], GongClient] = GongClient,
    ):
        self._credentials_provider = credentials_provider
        self._client_factory = client_factory
        self._client: Optional[GongClient] = None

    def _get_client(self) -> GongClient:
        if self._client is None:
            access_key, access_key_secret = self._credentials_provider()
            self._client = self._client_factory(access_key, access_key_secret)
        return self._client

    def list_calls(self, *args: Any, **kwargs: Any) -> Any:
        return self._get_client().list_calls(*args, **kwargs)

    def get_transcript(self, *args: Any, **kwargs: Any) -> Any:
        return self._get_client().get_transcript(*args, **kwargs)


def main() -> None:
    asyncio.run(run(LazyGongClient()))


if __name__ == "__main__":
    main()
