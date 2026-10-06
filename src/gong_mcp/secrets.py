"""Fetches the Gong API credentials from 1Password via the `op` CLI.

Gong's Basic-auth API requires two values — an access key and an access key
secret — so this module fetches both from the two fields on the same
1Password item. The credentials are held only in this module's cache, set
once at server startup, and are never logged or returned from any function
other than the getters below.

The 1Password item reference (vault, item title, field labels) is
configurable rather than hardcoded. Resolution order for each of
vault/item/access-key-field/access-key-secret-field is:

    1. Explicit function argument (mainly for tests/programmatic overrides)
    2. Environment variable (GONG_OP_VAULT / GONG_OP_ITEM /
       GONG_OP_ACCESS_KEY_FIELD / GONG_OP_ACCESS_KEY_SECRET_FIELD)
    3. Config file (JSON), if GONG_OP_CONFIG_FILE points to one
    4. Built-in default (DEFAULT_VAULT / DEFAULT_ITEM /
       DEFAULT_ACCESS_KEY_FIELD / DEFAULT_ACCESS_KEY_SECRET_FIELD)

See README.md for the exact 1Password item shape this module expects.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Optional

DEFAULT_VAULT = "Employee"
DEFAULT_ITEM = "Gong API Key"
DEFAULT_ACCESS_KEY_FIELD = "username"
DEFAULT_ACCESS_KEY_SECRET_FIELD = "credential"

ENV_VAULT = "GONG_OP_VAULT"
ENV_ITEM = "GONG_OP_ITEM"
ENV_ACCESS_KEY_FIELD = "GONG_OP_ACCESS_KEY_FIELD"
ENV_ACCESS_KEY_SECRET_FIELD = "GONG_OP_ACCESS_KEY_SECRET_FIELD"
ENV_CONFIG_FILE = "GONG_OP_CONFIG_FILE"

_AUTH_ERROR_MARKERS = (
    "not signed in",
    "not currently signed in",
    "sign in",
    "authenticate",
    "session expired",
    "unauthorized",
)
_NOT_FOUND_MARKERS = (
    "isn't an item",
    "isn't a vault",
    "no item found",
    "no such item",
    "not found",
    "isn't a field",
)


class SecretRetrievalError(RuntimeError):
    """Base error for failures fetching Gong credentials from 1Password."""


class OnePasswordCliMissingError(SecretRetrievalError):
    """Raised when the `op` binary isn't installed or isn't on PATH."""


class OnePasswordNotAuthenticatedError(SecretRetrievalError):
    """Raised when the `op` CLI has no active/authenticated session."""


class OnePasswordItemNotFoundError(SecretRetrievalError):
    """Raised when the designated vault item or field can't be found."""


class OnePasswordConfigError(SecretRetrievalError):
    """Raised when GONG_OP_CONFIG_FILE is set but unreadable or malformed."""


_cached_credentials: Optional[tuple[str, str]] = None


def _load_config_file() -> dict:
    """Load vault/item/field overrides from the JSON file at GONG_OP_CONFIG_FILE.

    Returns an empty dict if the env var isn't set. Raises
    OnePasswordConfigError if it's set but the file is missing or invalid,
    since a config file the user explicitly pointed us at should never be
    silently ignored.
    """
    path = os.environ.get(ENV_CONFIG_FILE)
    if not path:
        return {}

    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise OnePasswordConfigError(
            f"{ENV_CONFIG_FILE} is set to '{config_path}', but no file exists "
            "there. Either create it or unset the environment variable."
        )

    try:
        data = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise OnePasswordConfigError(
            f"Failed to read/parse 1Password config file '{config_path}': {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise OnePasswordConfigError(
            f"1Password config file '{config_path}' must contain a JSON "
            "object with optional 'vault', 'item', 'access_key_field', and "
            "'access_key_secret_field' string keys, e.g. "
            '{"vault": "Eng", "item": "Gong API Key"}.'
        )
    return data


def _resolve_item_reference(
    vault: Optional[str],
    item: Optional[str],
    access_key_field: Optional[str],
    access_key_secret_field: Optional[str],
) -> tuple[str, str, str, str]:
    """Resolve vault/item/fields via explicit arg > env var > config file > default."""
    config = _load_config_file()
    resolved_vault = vault or os.environ.get(ENV_VAULT) or config.get("vault") or DEFAULT_VAULT
    resolved_item = item or os.environ.get(ENV_ITEM) or config.get("item") or DEFAULT_ITEM
    resolved_access_key_field = (
        access_key_field
        or os.environ.get(ENV_ACCESS_KEY_FIELD)
        or config.get("access_key_field")
        or DEFAULT_ACCESS_KEY_FIELD
    )
    resolved_access_key_secret_field = (
        access_key_secret_field
        or os.environ.get(ENV_ACCESS_KEY_SECRET_FIELD)
        or config.get("access_key_secret_field")
        or DEFAULT_ACCESS_KEY_SECRET_FIELD
    )
    return (
        resolved_vault,
        resolved_item,
        resolved_access_key_field,
        resolved_access_key_secret_field,
    )


def _op_item_get_fields(vault: str, item: str, fields: list[str]) -> dict[str, str]:
    """Fetch several fields from one 1Password item with a SINGLE `op` call.

    Each `op` subprocess is a separate 1Password authorization request (and,
    with desktop-app integration, a separate Touch ID / MFA prompt when the
    caller has no TTY, as under Claude Desktop). Fetching every field in one
    invocation keeps server startup to one prompt instead of one per field.

    Returns a mapping of field label -> value. Raises a `SecretRetrievalError`
    subclass describing exactly what went wrong.
    """
    command = [
        "op",
        "item",
        "get",
        item,
        "--vault",
        vault,
        "--fields",
        ",".join(f"label={field}" for field in fields),
        "--reveal",
        "--format",
        "json",
    ]
    fields_desc = ", ".join(f"'{field}'" for field in fields)

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise OnePasswordCliMissingError(
            "1Password CLI ('op') not found on PATH. Install it and sign in "
            "before starting the Gong MCP server."
        ) from exc

    if result.returncode != 0:
        stderr = result.stderr.strip()
        lowered = stderr.lower()

        if any(marker in lowered for marker in _AUTH_ERROR_MARKERS):
            raise OnePasswordNotAuthenticatedError(
                "1Password CLI is not authenticated. Run `op signin` and "
                f"retry. Details: {stderr}"
            )

        if any(marker in lowered for marker in _NOT_FOUND_MARKERS):
            raise OnePasswordItemNotFoundError(
                f"1Password item '{item}' (fields {fields_desc}) not found in "
                f"vault '{vault}'. This server is blocked until the Gong API "
                "credentials are stored in 1Password at this location — "
                f"create an item titled '{item}' in the '{vault}' vault with "
                f"fields labeled {fields_desc} (see README.md for the exact "
                "shape expected). If the credentials actually live somewhere "
                f"else, point this server at them via the {ENV_VAULT}/"
                f"{ENV_ITEM}/{ENV_ACCESS_KEY_FIELD}/"
                f"{ENV_ACCESS_KEY_SECRET_FIELD} environment variables or a "
                f"{ENV_CONFIG_FILE} config file instead of editing defaults "
                f"in code. Details: {stderr}"
            )

        raise SecretRetrievalError(
            f"1Password CLI exited with status {result.returncode} while "
            f"fetching the Gong {fields_desc} fields. Details: {stderr}"
        )

    try:
        parsed = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        # Deliberately don't include stdout: it may contain secret values.
        raise SecretRetrievalError(
            f"1Password CLI returned non-JSON output for item '{item}'."
        ) from None

    # `op` returns a single object when one field matches, a list otherwise.
    entries = parsed if isinstance(parsed, list) else [parsed]
    values: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        label = entry.get("label")
        value = entry.get("value")
        if label in fields and label not in values and isinstance(value, str):
            values[label] = value.strip()

    for field in fields:
        if field not in values:
            raise OnePasswordItemNotFoundError(
                f"1Password item '{item}' in vault '{vault}' has no field "
                f"labeled '{field}'. Override the label via "
                f"{ENV_ACCESS_KEY_FIELD}/{ENV_ACCESS_KEY_SECRET_FIELD} or "
                f"{ENV_CONFIG_FILE} if it's named differently."
            )
        if not values[field]:
            raise SecretRetrievalError(
                f"1Password item '{item}' (field '{field}') returned an empty value."
            )
    return values


def fetch_gong_credentials(
    vault: Optional[str] = None,
    item: Optional[str] = None,
    access_key_field: Optional[str] = None,
    access_key_secret_field: Optional[str] = None,
) -> tuple[str, str]:
    """Fetch the Gong (access_key, access_key_secret) pair from 1Password,
    bypassing the module cache.

    All arguments default to None, meaning "resolve from environment
    variables, then config file, then built-in default" (see
    `_resolve_item_reference`). Pass explicit values to override all of
    that, e.g. in tests.

    Both fields are read in one `op` invocation so startup triggers a single
    1Password authorization prompt.

    Raises a `SecretRetrievalError` subclass describing exactly what went
    wrong (missing CLI, unauthenticated session, or missing item/field) so
    the server can fail fast with a clear message before accepting any tool
    calls.
    """
    vault, item, access_key_field, access_key_secret_field = _resolve_item_reference(
        vault, item, access_key_field, access_key_secret_field
    )

    values = _op_item_get_fields(vault, item, [access_key_field, access_key_secret_field])
    return values[access_key_field], values[access_key_secret_field]


def get_gong_credentials(force_refresh: bool = False, **kwargs) -> tuple[str, str]:
    """Return the cached (access_key, access_key_secret) pair, fetching it
    from 1Password on first use."""
    global _cached_credentials
    if _cached_credentials is None or force_refresh:
        _cached_credentials = fetch_gong_credentials(**kwargs)
    return _cached_credentials
