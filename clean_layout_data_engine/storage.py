"""Crash-safe storage helpers for batch production artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(
        path, json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    )


def atomic_write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    lines = [json.dumps(value, ensure_ascii=False, sort_keys=True) for value in values]
    atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class BatchLock:
    """Exclusive create lock; stale locks require an explicit recovery flag."""

    def __init__(self, path: Path, *, recover_stale: bool = False) -> None:
        self.path = path
        self.recover_stale = recover_stale
        self._held = False

    def __enter__(self) -> "BatchLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.recover_stale and self.path.exists():
            self.path.unlink()
        try:
            descriptor = os.open(
                self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY
            )
        except FileExistsError as error:
            raise RuntimeError(
                f"batch lock exists at {self.path}; use --recover-stale-lock "
                "only after confirming no batch process is running"
            ) from error
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump({"pid": os.getpid(), "created_at": utc_now()}, stream)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        self._held = True
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._held:
            self.path.unlink(missing_ok=True)
            self._held = False
