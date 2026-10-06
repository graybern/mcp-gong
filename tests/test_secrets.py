from __future__ import annotations

import json
import subprocess
from unittest.mock import patch

import pytest

from gong_mcp import secrets


def _completed_process(returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(
        args=["op"], returncode=returncode, stdout=stdout, stderr=stderr
    )


def _op_json(*values):
    """Build a `subprocess.run` side effect that mimics
    `op item get ... --fields label=a,label=b --format json`, returning the
    given values in the order the labels were requested."""
    def run(command, **kwargs):
        labels = [part.split("=", 1)[1] for part in command[command.index("--fields") + 1].split(",")]
        fields = [
            {"id": label, "label": label, "type": "STRING", "value": value}
            for label, value in zip(labels, values)
        ]
        payload = fields[0] if len(fields) == 1 else fields
        return _completed_process(stdout=json.dumps(payload))
    return run


def _fields_arg(command):
    return command[command.index("--fields") + 1]


@pytest.fixture(autouse=True)
def reset_cache():
    secrets._cached_credentials = None
    yield
    secrets._cached_credentials = None


def test_fetch_gong_credentials_success():
    with patch(
        "gong_mcp.secrets.subprocess.run",
        side_effect=_op_json("access-key-1\n", "super-secret-key\n"),
    ) as mock_run:
        access_key, access_key_secret = secrets.fetch_gong_credentials(
            vault="Eng", item="Gong API Key"
        )

    assert access_key == "access-key-1"
    assert access_key_secret == "super-secret-key"

    # Exactly one `op` call => exactly one 1Password authorization prompt.
    assert mock_run.call_count == 1
    args = mock_run.call_args_list[0].args[0]
    assert args[:3] == ["op", "item", "get"]
    assert "Eng" in args
    assert _fields_arg(args) == "label=username,label=credential"
    assert args[-2:] == ["--format", "json"]


def test_fetch_gong_credentials_handles_fields_in_any_order():
    payload = json.dumps([
        {"label": "credential", "value": "the-secret"},
        {"label": "username", "value": "the-key"},
    ])
    with patch(
        "gong_mcp.secrets.subprocess.run",
        return_value=_completed_process(stdout=payload),
    ):
        assert secrets.fetch_gong_credentials() == ("the-key", "the-secret")


def test_fetch_gong_credentials_missing_field_raises():
    payload = json.dumps({"label": "username", "value": "the-key"})
    with patch(
        "gong_mcp.secrets.subprocess.run",
        return_value=_completed_process(stdout=payload),
    ):
        with pytest.raises(secrets.OnePasswordItemNotFoundError):
            secrets.fetch_gong_credentials()


def test_fetch_gong_credentials_non_json_output_does_not_echo_stdout():
    with patch(
        "gong_mcp.secrets.subprocess.run",
        return_value=_completed_process(stdout="plain-secret-value"),
    ):
        with pytest.raises(secrets.SecretRetrievalError) as exc_info:
            secrets.fetch_gong_credentials()
    assert "plain-secret-value" not in str(exc_info.value)


def test_fetch_gong_credentials_op_missing():
    with patch("gong_mcp.secrets.subprocess.run", side_effect=FileNotFoundError):
        with pytest.raises(secrets.OnePasswordCliMissingError):
            secrets.fetch_gong_credentials()


def test_fetch_gong_credentials_not_authenticated():
    stderr = "[ERROR] you are not currently signed in. Please run `op signin`"
    with patch(
        "gong_mcp.secrets.subprocess.run",
        return_value=_completed_process(returncode=1, stderr=stderr),
    ):
        with pytest.raises(secrets.OnePasswordNotAuthenticatedError):
            secrets.fetch_gong_credentials()


def test_fetch_gong_credentials_item_not_found():
    stderr = "[ERROR] \"Gong API Key\" isn't an item in the \"Eng\" vault."
    with patch(
        "gong_mcp.secrets.subprocess.run",
        return_value=_completed_process(returncode=1, stderr=stderr),
    ):
        with pytest.raises(secrets.OnePasswordItemNotFoundError):
            secrets.fetch_gong_credentials()


def test_fetch_gong_credentials_generic_failure():
    with patch(
        "gong_mcp.secrets.subprocess.run",
        return_value=_completed_process(returncode=1, stderr="boom"),
    ):
        with pytest.raises(secrets.SecretRetrievalError):
            secrets.fetch_gong_credentials()


def test_fetch_gong_credentials_empty_value_raises():
    with patch(
        "gong_mcp.secrets.subprocess.run",
        side_effect=_op_json("key", "   \n"),
    ):
        with pytest.raises(secrets.SecretRetrievalError):
            secrets.fetch_gong_credentials()


def test_get_gong_credentials_caches_result():
    with patch(
        "gong_mcp.secrets.subprocess.run", side_effect=_op_json("key-1", "secret-1")
    ) as mock_run:
        first = secrets.get_gong_credentials()
        second = secrets.get_gong_credentials()

    assert first == second == ("key-1", "secret-1")
    assert mock_run.call_count == 1


def test_get_gong_credentials_force_refresh_bypasses_cache():
    runs = iter([_op_json("key-1", "secret-1"), _op_json("key-2", "secret-2")])
    with patch(
        "gong_mcp.secrets.subprocess.run",
        side_effect=lambda command, **kw: next(runs)(command, **kw),
    ) as mock_run:
        first = secrets.get_gong_credentials()
        second = secrets.get_gong_credentials(force_refresh=True)

    assert first == ("key-1", "secret-1")
    assert second == ("key-2", "secret-2")
    assert mock_run.call_count == 2


def test_error_never_echoes_stdout_value():
    with patch(
        "gong_mcp.secrets.subprocess.run",
        return_value=_completed_process(returncode=1, stdout="should-not-appear", stderr="boom"),
    ):
        with pytest.raises(secrets.SecretRetrievalError) as exc_info:
            secrets.fetch_gong_credentials()

    assert "should-not-appear" not in str(exc_info.value)


def test_default_item_reference_used_when_unconfigured(monkeypatch):
    monkeypatch.delenv(secrets.ENV_VAULT, raising=False)
    monkeypatch.delenv(secrets.ENV_ITEM, raising=False)
    monkeypatch.delenv(secrets.ENV_ACCESS_KEY_FIELD, raising=False)
    monkeypatch.delenv(secrets.ENV_ACCESS_KEY_SECRET_FIELD, raising=False)
    monkeypatch.delenv(secrets.ENV_CONFIG_FILE, raising=False)

    with patch(
        "gong_mcp.secrets.subprocess.run", side_effect=_op_json("key", "secret")
    ) as mock_run:
        secrets.fetch_gong_credentials()

    first_args = mock_run.call_args_list[0].args[0]
    requested = _fields_arg(first_args).split(",")
    assert secrets.DEFAULT_ITEM in first_args
    assert secrets.DEFAULT_VAULT in first_args
    assert f"label={secrets.DEFAULT_ACCESS_KEY_FIELD}" in requested
    assert f"label={secrets.DEFAULT_ACCESS_KEY_SECRET_FIELD}" in requested


def test_env_vars_override_defaults(monkeypatch):
    monkeypatch.setenv(secrets.ENV_VAULT, "EnvVault")
    monkeypatch.setenv(secrets.ENV_ITEM, "EnvItem")
    monkeypatch.setenv(secrets.ENV_ACCESS_KEY_FIELD, "EnvAccessKeyField")
    monkeypatch.setenv(secrets.ENV_ACCESS_KEY_SECRET_FIELD, "EnvSecretField")
    monkeypatch.delenv(secrets.ENV_CONFIG_FILE, raising=False)

    with patch(
        "gong_mcp.secrets.subprocess.run", side_effect=_op_json("key", "secret")
    ) as mock_run:
        secrets.fetch_gong_credentials()

    first_args = mock_run.call_args_list[0].args[0]
    requested = _fields_arg(first_args).split(",")
    assert "EnvItem" in first_args
    assert "EnvVault" in first_args
    assert "label=EnvAccessKeyField" in requested
    assert "label=EnvSecretField" in requested


def test_config_file_used_when_no_env_vars(tmp_path, monkeypatch):
    monkeypatch.delenv(secrets.ENV_VAULT, raising=False)
    monkeypatch.delenv(secrets.ENV_ITEM, raising=False)
    monkeypatch.delenv(secrets.ENV_ACCESS_KEY_FIELD, raising=False)
    monkeypatch.delenv(secrets.ENV_ACCESS_KEY_SECRET_FIELD, raising=False)

    config_path = tmp_path / "gong_op_config.json"
    config_path.write_text(
        '{"vault": "FileVault", "item": "FileItem", '
        '"access_key_field": "FileAccessKeyField", '
        '"access_key_secret_field": "FileSecretField"}'
    )
    monkeypatch.setenv(secrets.ENV_CONFIG_FILE, str(config_path))

    with patch(
        "gong_mcp.secrets.subprocess.run", side_effect=_op_json("key", "secret")
    ) as mock_run:
        secrets.fetch_gong_credentials()

    first_args = mock_run.call_args_list[0].args[0]
    requested = _fields_arg(first_args).split(",")
    assert "FileItem" in first_args
    assert "FileVault" in first_args
    assert "label=FileAccessKeyField" in requested
    assert "label=FileSecretField" in requested


def test_env_vars_take_precedence_over_config_file(tmp_path, monkeypatch):
    config_path = tmp_path / "gong_op_config.json"
    config_path.write_text(
        '{"vault": "FileVault", "item": "FileItem", '
        '"access_key_field": "FileAccessKeyField", '
        '"access_key_secret_field": "FileSecretField"}'
    )
    monkeypatch.setenv(secrets.ENV_CONFIG_FILE, str(config_path))
    monkeypatch.setenv(secrets.ENV_VAULT, "EnvVault")
    monkeypatch.delenv(secrets.ENV_ITEM, raising=False)
    monkeypatch.delenv(secrets.ENV_ACCESS_KEY_FIELD, raising=False)
    monkeypatch.delenv(secrets.ENV_ACCESS_KEY_SECRET_FIELD, raising=False)

    with patch(
        "gong_mcp.secrets.subprocess.run", side_effect=_op_json("key", "secret")
    ) as mock_run:
        secrets.fetch_gong_credentials()

    first_args = mock_run.call_args_list[0].args[0]
    requested = _fields_arg(first_args).split(",")
    assert "EnvVault" in first_args
    assert "FileItem" in first_args
    assert "label=FileAccessKeyField" in requested
    assert "label=FileSecretField" in requested


def test_explicit_args_take_precedence_over_env_and_config(tmp_path, monkeypatch):
    config_path = tmp_path / "gong_op_config.json"
    config_path.write_text(
        '{"vault": "FileVault", "item": "FileItem", '
        '"access_key_field": "FileAccessKeyField", '
        '"access_key_secret_field": "FileSecretField"}'
    )
    monkeypatch.setenv(secrets.ENV_CONFIG_FILE, str(config_path))
    monkeypatch.setenv(secrets.ENV_VAULT, "EnvVault")
    monkeypatch.setenv(secrets.ENV_ITEM, "EnvItem")
    monkeypatch.setenv(secrets.ENV_ACCESS_KEY_FIELD, "EnvAccessKeyField")
    monkeypatch.setenv(secrets.ENV_ACCESS_KEY_SECRET_FIELD, "EnvSecretField")

    with patch(
        "gong_mcp.secrets.subprocess.run", side_effect=_op_json("key", "secret")
    ) as mock_run:
        secrets.fetch_gong_credentials(
            vault="ArgVault",
            item="ArgItem",
            access_key_field="ArgAccessKeyField",
            access_key_secret_field="ArgSecretField",
        )

    first_args = mock_run.call_args_list[0].args[0]
    requested = _fields_arg(first_args).split(",")
    assert "ArgVault" in first_args
    assert "ArgItem" in first_args
    assert "label=ArgAccessKeyField" in requested
    assert "label=ArgSecretField" in requested


def test_missing_config_file_raises_clear_error(monkeypatch):
    monkeypatch.setenv(secrets.ENV_CONFIG_FILE, "/nonexistent/path/gong_op_config.json")

    with pytest.raises(secrets.OnePasswordConfigError):
        secrets.fetch_gong_credentials()


def test_malformed_config_file_raises_clear_error(tmp_path, monkeypatch):
    config_path = tmp_path / "gong_op_config.json"
    config_path.write_text("not valid json")
    monkeypatch.setenv(secrets.ENV_CONFIG_FILE, str(config_path))

    with pytest.raises(secrets.OnePasswordConfigError):
        secrets.fetch_gong_credentials()


def test_item_not_found_error_message_mentions_configurability():
    stderr = "[ERROR] \"Gong API Key\" isn't an item in the \"Employee\" vault."
    with patch(
        "gong_mcp.secrets.subprocess.run",
        return_value=_completed_process(returncode=1, stderr=stderr),
    ):
        with pytest.raises(secrets.OnePasswordItemNotFoundError) as exc_info:
            secrets.fetch_gong_credentials()

    message = str(exc_info.value)
    assert secrets.ENV_VAULT in message
    assert secrets.ENV_ITEM in message
    assert secrets.ENV_ACCESS_KEY_FIELD in message
    assert secrets.ENV_ACCESS_KEY_SECRET_FIELD in message
    assert secrets.ENV_CONFIG_FILE in message
