"""Safe Responses API client, structured parsing, and failure diagnostics."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

from clean_layout_data_engine.storage import atomic_write_json, utc_now


_SECRET_PATTERNS = (
    re.compile(r"(?i)sk-[a-z0-9_-]+"),
    re.compile(r"(?i)(bearer\s+)[a-z0-9._~+/=-]+"),
    re.compile(r"(?i)((?:api[_-]?key|authorization)\s*[:=]\s*)[^\s,;}]+"),
)


class StructuredResponseError(ValueError):
    """Base class for a completed response that cannot be consumed safely."""

    category = "malformed_response"

    def __init__(self, message: str, response: Any) -> None:
        super().__init__(message)
        self.response_id = getattr(response, "id", None)
        self.response_model = getattr(response, "model", None)
        self.response_status = getattr(response, "status", None)
        output_text = getattr(response, "output_text", None)
        self.output_text_type = type(output_text).__name__


class EmptyOutputTextError(StructuredResponseError):
    category = "empty_output_text"


class InvalidStructuredJSONError(StructuredResponseError):
    category = "invalid_structured_json"


class StructuredOutputTypeError(StructuredResponseError):
    category = "invalid_structured_type"


def create_openai_client(
    api: dict[str, Any],
    *,
    base_url: str | None = None,
    timeout_seconds: float | None = None,
) -> OpenAI:
    """Create a client with exactly one HTTP attempt per engine-level attempt."""
    return OpenAI(
        api_key=api["api_key"],
        base_url=base_url or api["base_url"],
        timeout=timeout_seconds or api["timeout_seconds"],
        max_retries=0,
    )


def parse_structured_response(response: Any) -> dict[str, Any]:
    output_text = getattr(response, "output_text", None)
    if not isinstance(output_text, str) or not output_text.strip():
        raise EmptyOutputTextError(
            "Responses API completed without non-empty output_text", response
        )
    try:
        value = json.loads(output_text)
    except json.JSONDecodeError as error:
        raise InvalidStructuredJSONError(
            f"Responses API output_text is not valid JSON: {error.msg}", response
        ) from error
    if not isinstance(value, dict):
        raise StructuredOutputTypeError(
            "Responses API structured output must be a JSON object", response
        )
    return value


def _safe_message(error: Exception, limit: int = 500) -> str:
    message = " ".join(str(error).split())
    for pattern in _SECRET_PATTERNS:
        if pattern.pattern.startswith("(?i)(bearer"):
            message = pattern.sub(r"\1[REDACTED]", message)
        elif "api[_-]?key" in pattern.pattern:
            message = pattern.sub(r"\1[REDACTED]", message)
        else:
            message = pattern.sub("[REDACTED]", message)
    return message[:limit]


def _error_category(error: Exception) -> str:
    if isinstance(error, StructuredResponseError):
        return error.category
    name = type(error).__name__
    if name in {"APITimeoutError", "TimeoutError"}:
        return "timeout"
    if name in {"APIConnectionError", "ConnectionError"}:
        return "connection"
    if name in {
        "APIStatusError",
        "BadRequestError",
        "InternalServerError",
        "RateLimitError",
    }:
        return "http_status"
    if isinstance(error, TypeError):
        return "client_type_error"
    return "unexpected_client_error"


def api_failure_record(
    error: Exception,
    *,
    scene_id: str,
    stage: str,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema_version": "clean_layout_api_failure_v1",
        "scene_id": scene_id,
        "stage": stage,
        "status": "retryable_api_failure",
        "error_type": type(error).__name__,
        "error_category": _error_category(error),
        "safe_message": _safe_message(error),
        "updated_at": utc_now(),
    }
    status_code = getattr(error, "status_code", None)
    if isinstance(status_code, int):
        record["status_code"] = status_code
    for name in (
        "response_id",
        "response_model",
        "response_status",
        "output_text_type",
    ):
        value = getattr(error, name, None)
        if value is not None:
            record[name] = value
    return record


def persist_api_failure(
    error: Exception,
    *,
    scene_id: str,
    stage: str,
    latest_path: Path,
    history_dir: Path,
) -> dict[str, Any]:
    record = api_failure_record(error, scene_id=scene_id, stage=stage)
    atomic_write_json(latest_path, record)
    history_name = f"{scene_id}_{time.time_ns()}.json"
    atomic_write_json(history_dir / history_name, record)
    return record


def mark_api_failure_resolved(path: Path, *, response: Any) -> None:
    if not path.exists():
        return
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(record, dict):
        return
    record.update(
        {
            "status": "resolved",
            "resolved_at": utc_now(),
            "resolved_by_response_id": getattr(response, "id", None),
        }
    )
    atomic_write_json(path, record)
