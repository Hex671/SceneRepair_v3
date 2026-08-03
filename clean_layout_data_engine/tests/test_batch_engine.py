from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from clean_layout_data_engine import batch_engine
from clean_layout_data_engine.storage import atomic_write_json, atomic_write_text, load_json, sha256_file, utc_now


def test_target_allocation_is_exact_and_deterministic() -> None:
    weights = {key: 0.25 for key in batch_engine.ROOM_TYPES}
    assert batch_engine._allocate_targets(10, weights) == {
        "bedroom": 3,
        "living_room": 3,
        "dining_room": 2,
        "home_office": 2,
    }
    first = batch_engine._round_robin_jobs(
        "batch_001", {key: 2 for key in batch_engine.ROOM_TYPES}, 123
    )
    second = batch_engine._round_robin_jobs(
        "batch_001", {key: 2 for key in batch_engine.ROOM_TYPES}, 123
    )
    assert first == second
    assert len({value["seed"] for value in first}) == len(first)


def test_blocked_api_resume_preserves_pipeline_stage() -> None:
    assert (
        batch_engine._blocked_retry_state({"last_generation_exit_code": 3})
        == "generation_retryable"
    )


def test_runtime_failure_metadata_uses_stage_specific_record(tmp_path: Path) -> None:
    scene_id = "scene_001"
    review_path = (
        tmp_path / "reviews" / "semantic_failures" / f"{scene_id}.json"
    )
    atomic_write_json(
        review_path,
        {"error_type": "InternalServerError", "error_category": "http_status"},
    )
    atomic_write_json(
        tmp_path / "status" / f"{scene_id}.json",
        {"error_type": "OldAuthorError", "error_category": "connection"},
    )

    value = batch_engine._runtime_failure_metadata(tmp_path, "review", scene_id)

    assert value["error_type"] == "InternalServerError"
    assert value["error_category"] == "http_status"
    assert (
        batch_engine._blocked_retry_state({"last_review_exit_code": 3})
        == "review_retryable"
    )


def _production_args(tmp_path: Path) -> argparse.Namespace:
    contract_paths = {}
    for name in (
        "pipeline_config",
        "semantic_mapping",
        "layout_rules",
        "functional_rules",
        "functional_compatibility",
        "vocab",
        "hssd_lookup",
    ):
        path = tmp_path / f"{name}.json"
        path.write_text("{}\n", encoding="utf-8")
        contract_paths[name] = path
    production_config = tmp_path / "production.json"
    production_config.write_text(
        json.dumps(
            {
                "schema_version": "clean_layout_production_config_v1",
                "base_seed": 123,
                "default_target_count": 2,
                "room_type_weights": {key: 0.25 for key in batch_engine.ROOM_TYPES},
                "candidate_multiplier": 1.5,
                "wave_size": 2,
                "max_workers": 2,
                "launch_interval_seconds": 0,
                "max_generation_runs_per_candidate": 2,
                "max_review_runs_per_candidate": 2,
                "max_consecutive_runtime_api_failure_waves": 1,
                "runtime_api_retry_backoff_seconds": 0,
                "runtime_api_retry_backoff_multiplier": 1,
                "runtime_api_retry_max_seconds": 0,
                "review_enabled": True,
            }
        ),
        encoding="utf-8",
    )
    existing = tmp_path / "existing"
    existing.mkdir()
    return argparse.Namespace(
        batch_id="batch_001",
        output_root=tmp_path / "output",
        target_count=2,
        room_count=["bedroom=2"],
        max_workers=None,
        wave_size=None,
        pipeline=False,
        author_workers=None,
        reviewer_workers=None,
        speculative_buffer_per_room=None,
        recover_stale_lock=False,
        plan_only=False,
        status_only=False,
        api_config=tmp_path / "unread_api_config.json",
        production_config=production_config,
        project_root=tmp_path,
        existing_scenes=existing,
        **contract_paths,
    )


def test_pipeline_selector_keeps_small_room_buffer() -> None:
    jobs = [
        {
            "scene_id": f"{room}_{index}",
            "room_type": room,
            "state": "planned",
            "generation_runs": 0,
        }
        for index in range(3)
        for room in batch_engine.ROOM_TYPES
    ]

    selected = batch_engine._select_pipeline_generation_jobs(
        jobs,
        {room: 1 for room in batch_engine.ROOM_TYPES},
        max_runs=2,
        limit=8,
        speculative_buffer_per_room=1,
    )

    assert len(selected) == 8
    assert {
        room: sum(job["room_type"] == room for job in selected)
        for room in batch_engine.ROOM_TYPES
    } == {room: 2 for room in batch_engine.ROOM_TYPES}
    assert batch_engine._pipeline_worker_split(16) == (12, 4)
    assert batch_engine._pipeline_worker_split(6) == (4, 2)


def test_pipeline_max_workers_is_a_real_total_cap(tmp_path: Path) -> None:
    args = _production_args(tmp_path)
    production = load_json(args.production_config)
    production.update(
        {"max_workers": 16, "author_workers": 12, "reviewer_workers": 4}
    )

    args.max_workers = 8

    assert batch_engine._resolve_pipeline_workers(args, production) == (6, 2)

    args.author_workers = 7
    args.reviewer_workers = 2
    with pytest.raises(ValueError, match="exceed the max-workers cap"):
        batch_engine._resolve_pipeline_workers(args, production)


def test_run_process_enforces_coordinator_timeout(tmp_path: Path) -> None:
    stdout = tmp_path / "stdout.log"
    stderr = tmp_path / "stderr.log"

    code = batch_engine._run_process(
        [sys.executable, "-c", "import time; time.sleep(5)"],
        cwd=tmp_path,
        stdout_path=stdout,
        stderr_path=stderr,
        throttle=batch_engine._LaunchThrottle(0),
        timeout_seconds=0.05,
    )

    assert code == 3
    assert "Coordinator timeout" in stderr.read_text(encoding="utf-8")


def test_pipeline_selector_prefers_untried_candidates() -> None:
    retryable = {
        "scene_id": "bedroom_retry",
        "room_type": "bedroom",
        "state": "generation_retryable",
        "generation_runs": 0,
    }
    planned = {
        "scene_id": "bedroom_new",
        "room_type": "bedroom",
        "state": "planned",
        "generation_runs": 0,
    }

    selected = batch_engine._select_pipeline_generation_jobs(
        [retryable, planned],
        {room: (1 if room == "bedroom" else 0) for room in batch_engine.ROOM_TYPES},
        max_runs=2,
        limit=1,
        speculative_buffer_per_room=0,
    )

    assert [job["scene_id"] for job in selected] == ["bedroom_new"]


def test_pipeline_selector_prioritizes_room_with_least_inflight_coverage() -> None:
    jobs = [
        {
            "scene_id": "bedroom_active",
            "room_type": "bedroom",
            "state": "authoring",
            "generation_runs": 1,
        },
        {
            "scene_id": "bedroom_new",
            "room_type": "bedroom",
            "state": "planned",
            "generation_runs": 0,
        },
        {
            "scene_id": "dining_new",
            "room_type": "dining_room",
            "state": "planned",
            "generation_runs": 0,
        },
    ]
    targets = {
        room: (2 if room in {"bedroom", "dining_room"} else 0)
        for room in batch_engine.ROOM_TYPES
    }

    selected = batch_engine._select_pipeline_generation_jobs(
        jobs,
        targets,
        max_runs=2,
        limit=1,
        speculative_buffer_per_room=0,
    )

    assert [job["scene_id"] for job in selected] == ["dining_new"]


def test_offline_pipeline_overlaps_author_and_reviewer(monkeypatch, tmp_path) -> None:
    args = _production_args(tmp_path)
    args.target_count = 4
    args.room_count = ["bedroom=4"]
    production = load_json(args.production_config)
    production.update(
        {
            "pipeline_enabled": True,
            "max_workers": 3,
            "author_workers": 2,
            "reviewer_workers": 1,
            "speculative_buffer_per_room": 1,
        }
    )
    atomic_write_json(args.production_config, production)
    active = {"generation": 0, "review": 0}
    observed_overlap = False
    lock = threading.Lock()

    def fake_run_process(command, **kwargs):
        nonlocal observed_overlap
        stage = "review" if "clean_layout_data_engine.semantic_reviewer" in command else "generation"
        with lock:
            active[stage] += 1
            observed_overlap = observed_overlap or all(active.values())
        time.sleep(0.02)
        with lock:
            active[stage] -= 1
        return 0

    def fake_promote(args, batch_dir, staging, jobs, plan, max_generation_runs):
        accepted_counts = batch_engine._counts(jobs, "accepted")
        for index, job in enumerate(jobs):
            if job["state"] != "semantic_pass_pending_diversity":
                continue
            if accepted_counts[job["room_type"]] >= plan["room_type_targets"][job["room_type"]]:
                jobs[index] = batch_engine._save_job(
                    batch_dir, job, state="surplus_not_needed"
                )
                continue
            destination = (
                args.output_root / "accepted" / "scenes" / f"{job['scene_id']}.json"
            )
            atomic_write_text(destination, json.dumps({"scene_id": job["scene_id"]}) + "\n")
            jobs[index] = batch_engine._save_job(
                batch_dir,
                job,
                state="accepted",
                accepted_scene=str(destination),
                scene_sha256=sha256_file(destination),
                validation_path=str(tmp_path / "validation.json"),
                semantic_review=str(tmp_path / "review.json"),
                diversity_report=str(tmp_path / "diversity.json"),
                accepted_at=utc_now(),
            )
            accepted_counts[job["room_type"]] += 1
        return jobs

    monkeypatch.setattr(batch_engine, "_run_process", fake_run_process)
    monkeypatch.setattr(batch_engine, "_promote_reviewed_jobs", fake_promote)

    assert batch_engine.run_batch(args) == 0
    assert observed_overlap
    status = load_json(
        args.output_root / "batches" / args.batch_id / "batch_status.json"
    )
    assert status["accepted_count"] == 4
    assert status["target_met"]


def test_offline_batch_lifecycle_and_resume(monkeypatch, tmp_path) -> None:
    args = _production_args(tmp_path)
    dispatch_calls: list[tuple[str, list[str]]] = []

    def fake_dispatch(items, command_builder, *, stage, **kwargs):
        dispatch_calls.append((stage, [item["scene_id"] for item in items]))
        return {item["scene_id"]: 0 for item in items}

    def fake_promote(args, batch_dir, staging, jobs, plan, max_generation_runs):
        accepted_counts = batch_engine._counts(jobs, "accepted")
        for index, job in enumerate(jobs):
            if job["state"] != "semantic_pass_pending_diversity":
                continue
            if accepted_counts[job["room_type"]] >= plan["room_type_targets"][job["room_type"]]:
                continue
            destination = args.output_root / "accepted" / "scenes" / f"{job['scene_id']}.json"
            atomic_write_text(destination, json.dumps({"scene_id": job["scene_id"]}) + "\n")
            jobs[index] = batch_engine._save_job(
                batch_dir,
                job,
                state="accepted",
                accepted_scene=str(destination),
                scene_sha256=sha256_file(destination),
                validation_path=str(tmp_path / "validation.json"),
                semantic_review=str(tmp_path / "review.json"),
                diversity_report=str(tmp_path / "diversity.json"),
                accepted_at=utc_now(),
            )
            accepted_counts[job["room_type"]] += 1
        return jobs

    monkeypatch.setattr(batch_engine, "_dispatch", fake_dispatch)
    monkeypatch.setattr(batch_engine, "_promote_reviewed_jobs", fake_promote)

    assert batch_engine.run_batch(args) == 0
    batch_dir = args.output_root / "batches" / args.batch_id
    status = load_json(batch_dir / "batch_status.json")
    assert status["target_met"]
    assert status["accepted_count"] == 2
    assert [value[0] for value in dispatch_calls] == ["generation", "review"]
    assert len((args.output_root / "accepted" / "manifest.jsonl").read_text().splitlines()) == 2

    dispatch_calls.clear()
    assert batch_engine.run_batch(args) == 0
    assert dispatch_calls == []


def test_plan_only_does_not_read_api_config(tmp_path) -> None:
    args = _production_args(tmp_path)
    args.plan_only = True
    assert not args.api_config.exists()

    assert batch_engine.run_batch(args) == 0

    status = load_json(
        args.output_root / "batches" / args.batch_id / "batch_status.json"
    )
    assert status["status"] == "planned"
    assert not args.api_config.exists()

    status_path = (
        args.output_root / "batches" / args.batch_id / "batch_status.json"
    )
    before = status_path.read_bytes()
    args.plan_only = False
    args.status_only = True
    assert batch_engine.run_batch(args) == 0
    assert status_path.read_bytes() == before


def test_configuration_worker_error_stops_batch(monkeypatch, tmp_path) -> None:
    args = _production_args(tmp_path)

    def fake_dispatch(items, command_builder, *, stage, **kwargs):
        assert stage == "generation"
        return {item["scene_id"]: 2 for item in items}

    monkeypatch.setattr(batch_engine, "_dispatch", fake_dispatch)

    with pytest.raises(RuntimeError, match="batch stopped"):
        batch_engine.run_batch(args)

    batch_dir = args.output_root / "batches" / args.batch_id
    states = {
        load_json(path)["state"] for path in (batch_dir / "jobs").glob("*.json")
    }
    assert "blocked_configuration" in states


def test_uniform_runtime_api_failure_trips_circuit_breaker(monkeypatch, tmp_path) -> None:
    args = _production_args(tmp_path)

    def fake_dispatch(items, command_builder, *, stage, batch_dir, **kwargs):
        assert stage == "generation"
        for item in items:
            atomic_write_json(
                batch_dir / "staging" / "status" / f"{item['scene_id']}.json",
                {"error_type": "APIConnectionError"},
            )
        return {item["scene_id"]: 3 for item in items}

    monkeypatch.setattr(batch_engine, "_dispatch", fake_dispatch)

    with pytest.raises(RuntimeError, match="runtime API failure"):
        batch_engine.run_batch(args)

    batch_dir = args.output_root / "batches" / args.batch_id
    states = {
        load_json(path)["state"] for path in (batch_dir / "jobs").glob("*.json")
    }
    assert "blocked_api_runtime" in states
    assert "planned" in states


def test_resume_rejects_target_drift(monkeypatch, tmp_path) -> None:
    args = _production_args(tmp_path)
    args.plan_only = True
    assert batch_engine.run_batch(args) == 0

    args.plan_only = False
    args.target_count = 3
    with pytest.raises(ValueError, match="immutable batch plan"):
        batch_engine.run_batch(args)


def test_resume_rejects_engine_source_drift(monkeypatch, tmp_path) -> None:
    args = _production_args(tmp_path)
    args.plan_only = True
    assert batch_engine.run_batch(args) == 0

    batch_dir = args.output_root / "batches" / args.batch_id
    plan_path = batch_dir / "batch_plan.json"
    plan = load_json(plan_path)
    plan["engine_source_sha256"]["run_pilot.py"] = "changed"
    atomic_write_json(plan_path, plan)

    args.plan_only = False
    with pytest.raises(ValueError, match="source drift"):
        batch_engine.run_batch(args)


def test_diversity_failure_retries_with_feedback(monkeypatch, tmp_path) -> None:
    args = _production_args(tmp_path)
    batch_dir = args.output_root / "batches" / args.batch_id
    staging = batch_dir / "staging"
    scene_id = "batch_001_bedroom_0001"
    atomic_write_json(
        staging / "validation" / f"{scene_id}.json",
        {"checks": {"diversity_novelty_pass": False}},
    )
    job = {
        "scene_id": scene_id,
        "room_type": "bedroom",
        "state": "semantic_pass_pending_diversity",
        "generation_runs": 1,
    }
    plan = {"room_type_targets": {key: 1 for key in batch_engine.ROOM_TYPES}}

    monkeypatch.setattr(
        batch_engine.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    result = batch_engine._promote_reviewed_jobs(
        args, batch_dir, staging, [job], plan, max_generation_runs=2
    )

    assert result[0]["state"] == "generation_retryable"
    assert result[0]["diversity_retry_count"] == 1
    assert result[0]["last_stage"] == "diversity_retry"
