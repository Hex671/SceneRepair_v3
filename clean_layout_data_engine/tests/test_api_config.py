from __future__ import annotations

import json

import pytest

from clean_layout_data_engine.run_pilot import _load_api_config


def test_api_config_loads_required_fields_without_mutating_source(tmp_path) -> None:
    path = tmp_path / "config.json"
    source = {
        "api": {
            "api_key": "test-secret-value",
            "base_url": "https://example.test/v1",
            "model": "test-model",
            "wire_api": "responses",
            "timeout_seconds": 45,
        }
    }
    path.write_text(json.dumps(source), encoding="utf-8")

    loaded = _load_api_config(path)

    assert loaded["api_key"] == "test-secret-value"
    assert loaded["timeout_seconds"] == 45.0
    assert json.loads(path.read_text(encoding="utf-8")) == source


def test_api_config_empty_key_error_does_not_echo_any_value(tmp_path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "api": {
                    "api_key": "",
                    "base_url": "https://example.test/v1",
                    "model": "test-model",
                    "wire_api": "responses",
                    "timeout_seconds": 45,
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"api\.api_key is empty"):
        _load_api_config(path)


def test_api_config_rejects_non_responses_wire_api(tmp_path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "api": {
                    "api_key": "test-secret-value",
                    "base_url": "https://example.test/v1",
                    "model": "test-model",
                    "wire_api": "chat_completions",
                    "timeout_seconds": 45,
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="wire_api must be 'responses'"):
        _load_api_config(path)
