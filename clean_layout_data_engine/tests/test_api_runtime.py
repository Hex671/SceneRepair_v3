from __future__ import annotations

from types import SimpleNamespace

import pytest

from clean_layout_data_engine import api_runtime


def _response(output_text):
    return SimpleNamespace(
        id="resp_test",
        model="test-model",
        status="completed",
        output_text=output_text,
    )


def test_structured_response_requires_nonempty_text() -> None:
    with pytest.raises(api_runtime.EmptyOutputTextError):
        api_runtime.parse_structured_response(_response(None))


def test_structured_response_requires_valid_json_object() -> None:
    with pytest.raises(api_runtime.InvalidStructuredJSONError):
        api_runtime.parse_structured_response(_response("not json"))
    with pytest.raises(api_runtime.StructuredOutputTypeError):
        api_runtime.parse_structured_response(_response("[]"))
    assert api_runtime.parse_structured_response(_response('{"ok": true}')) == {
        "ok": True
    }


def test_failure_record_is_classified_and_redacted() -> None:
    error = TypeError("bad api_key=sk-secret-value Bearer abc.def")

    record = api_runtime.api_failure_record(
        error, scene_id="scene", stage="semantic_review"
    )

    assert record["error_category"] == "client_type_error"
    assert "sk-secret-value" not in record["safe_message"]
    assert "abc.def" not in record["safe_message"]


def test_client_disables_sdk_retries(monkeypatch) -> None:
    captured = {}

    def fake_openai(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(api_runtime, "OpenAI", fake_openai)
    api_runtime.create_openai_client(
        {
            "api_key": "secret",
            "base_url": "https://example.test/v1",
            "timeout_seconds": 30,
        }
    )

    assert captured["max_retries"] == 0
