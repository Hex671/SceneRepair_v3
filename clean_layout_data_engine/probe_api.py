"""Safely test the configured Responses API without printing configuration values."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from openai import OpenAI

from clean_layout_data_engine.run_pilot import ENGINE_DIR, _load_api_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-config", type=Path, default=ENGINE_DIR / "config.json")
    args = parser.parse_args()
    try:
        api = _load_api_config(args.api_config)
        client = OpenAI(
            api_key=api["api_key"],
            base_url=api["base_url"],
            timeout=api["timeout_seconds"],
            max_retries=0,
        )
        response = client.responses.create(
            model=api["model"],
            input="Return exactly: API_OK",
            # This gateway rejects very small response budgets before the SDK can
            # consistently surface the upstream HTTP status.
            max_output_tokens=64,
        )
        if response.output_text.strip() != "API_OK":
            print(json.dumps({"ok": False, "error_type": "UnexpectedResponse"}))
            return 1
        print(json.dumps({"ok": True, "wire_api": "responses"}))
        return 0
    except Exception as error:
        status_code = getattr(error, "status_code", None)
        payload = {"ok": False, "error_type": type(error).__name__}
        if isinstance(status_code, int):
            payload["status_code"] = status_code
        print(json.dumps(payload))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
