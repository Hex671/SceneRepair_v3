from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any


DEFAULT_MODEL = "gpt-5.6"
DEFAULT_PROMPT = (
    "This is an API connectivity test. Reply in Chinese with exactly: "
    "GPT-5.6 API 调用成功"
)


def _safe_error_message(error: Exception, api_key: str) -> str:
    message = str(error).replace(api_key, "<redacted>")
    return message[:2000]


def _error_hint(error: Exception) -> str:
    status_code = getattr(error, "status_code", None)
    if status_code == 401:
        return "API key无效、已撤销，或没有被当前进程正确读取。"
    if status_code == 403:
        return "当前API项目没有该模型的访问权限。"
    if status_code == 404:
        return "模型名不存在，或当前API项目尚未开放该模型。"
    if status_code == 429:
        return "请求达到速率限制，或API项目没有可用额度/账单配置。"
    return "请检查网络、OpenAI SDK版本、模型权限和API项目账单状态。"


def _response_payload(response: Any, requested_model: str) -> dict[str, object]:
    return {
        "success": True,
        "requested_model": requested_model,
        "response_model": getattr(response, "model", None),
        "response_id": getattr(response, "id", None),
        "status": getattr(response, "status", None),
        "output_text": getattr(response, "output_text", ""),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test an OpenAI GPT-5.6 Responses API call."
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", DEFAULT_MODEL),
        help=f"API model ID (default: OPENAI_MODEL or {DEFAULT_MODEL})",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high", "xhigh", "max"),
        default="low",
    )
    parser.add_argument("--max-output-tokens", type=int, default=128)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    args = parser.parse_args()

    if args.max_output_tokens <= 0:
        parser.error("--max-output-tokens must be positive")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        print(
            "ERROR: 当前进程没有读取到OPENAI_API_KEY。\n"
            '请先在同一个PowerShell终端执行：$env:OPENAI_API_KEY="你的API密钥"',
            file=sys.stderr,
        )
        return 2

    try:
        from openai import OpenAI
    except ImportError:
        print(
            "ERROR: 当前Python环境没有安装OpenAI SDK。\n"
            "请执行：python -m pip install --upgrade openai",
            file=sys.stderr,
        )
        return 3

    client = OpenAI(api_key=api_key, timeout=args.timeout_seconds)
    try:
        response = client.responses.create(
            model=args.model,
            reasoning={"effort": args.reasoning_effort},
            input=args.prompt,
            max_output_tokens=args.max_output_tokens,
        )
    except Exception as error:
        payload = {
            "success": False,
            "requested_model": args.model,
            "error_type": type(error).__name__,
            "status_code": getattr(error, "status_code", None),
            "message": _safe_error_message(error, api_key),
            "hint": _error_hint(error),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1

    print(
        json.dumps(
            _response_payload(response, args.model),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
