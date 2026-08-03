from __future__ import annotations

import json

import pytest

from clean_layout_data_engine.storage import BatchLock, atomic_write_json, load_json


def test_atomic_json_replaces_complete_document(tmp_path) -> None:
    path = tmp_path / "nested" / "state.json"
    atomic_write_json(path, {"version": 1, "items": [1, 2]})
    atomic_write_json(path, {"version": 2, "items": [3]})

    assert load_json(path) == {"version": 2, "items": [3]}
    assert not list(path.parent.glob("*.tmp"))


def test_batch_lock_rejects_second_coordinator(tmp_path) -> None:
    path = tmp_path / ".production.lock"
    with BatchLock(path):
        with pytest.raises(RuntimeError, match="batch lock exists"):
            with BatchLock(path):
                pass
    assert not path.exists()


def test_batch_lock_recovery_is_explicit(tmp_path) -> None:
    path = tmp_path / ".production.lock"
    path.write_text(json.dumps({"pid": -1}), encoding="utf-8")
    with BatchLock(path, recover_stale=True):
        assert path.exists()
    assert not path.exists()
