"""Recoverable production batch coordinator for clean-layout generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, as_completed, wait
from pathlib import Path
from typing import Any, Callable, Iterable

from clean_layout_data_engine.run_pilot import ENGINE_DIR
from clean_layout_data_engine.storage import (
    BatchLock,
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    load_json,
    sha256_file,
    utc_now,
)


ROOM_TYPES = ("bedroom", "living_room", "dining_room", "home_office")
ENGINE_SOURCE_FILES = (
    "batch_engine.py",
    "api_runtime.py",
    "run_pilot.py",
    "orientation_intent_compiler.py",
    "semantic_reviewer.py",
    "storage.py",
)
TERMINAL_STATES = {
    "accepted",
    "rejected_diversity",
    "rejected_generation",
    "rejected_semantic",
    "surplus_not_needed",
}
BATCH_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{2,63}$")


def _stable_seed(base_seed: int, batch_id: str, index: int) -> int:
    payload = f"{base_seed}:{batch_id}:{index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFFFFFF


def _engine_source_hashes() -> dict[str, str]:
    """Pin the executable parts of a batch, not only its JSON contracts."""
    return {
        name: sha256_file(ENGINE_DIR / name)
        for name in ENGINE_SOURCE_FILES
    }


def _assert_plan_integrity(plan: dict[str, Any]) -> None:
    """Refuse to dispatch work after any executable or contract drift."""
    for key, expected in plan["contract_sha256"].items():
        current = sha256_file(Path(plan["contract_paths"][key]))
        if current != expected:
            raise ValueError(
                f"batch contract drift detected for {key}; start a new batch"
            )
    expected_source_hashes = plan.get("engine_source_sha256")
    if expected_source_hashes is None:
        raise ValueError("batch plan predates source pinning; start a new batch")
    if _engine_source_hashes() != expected_source_hashes:
        raise ValueError("batch engine source drift detected; start a new batch")


def _allocate_targets(total: int, weights: dict[str, float]) -> dict[str, int]:
    if total <= 0:
        raise ValueError("target count must be positive")
    if set(weights) != set(ROOM_TYPES):
        raise ValueError(f"room_type_weights must contain exactly {ROOM_TYPES}")
    weight_sum = sum(float(weights[key]) for key in ROOM_TYPES)
    if weight_sum <= 0:
        raise ValueError("room type weights must sum to a positive value")
    raw = {key: total * float(weights[key]) / weight_sum for key in ROOM_TYPES}
    result = {key: math.floor(raw[key]) for key in ROOM_TYPES}
    remaining = total - sum(result.values())
    order = sorted(ROOM_TYPES, key=lambda key: -(raw[key] - result[key]))
    for key in order[:remaining]:
        result[key] += 1
    return result


def _parse_room_counts(values: list[str]) -> dict[str, int] | None:
    if not values:
        return None
    result = {key: 0 for key in ROOM_TYPES}
    seen: set[str] = set()
    for value in values:
        if "=" not in value:
            raise ValueError("--room-count values must use room_type=count")
        room_type, raw_count = value.split("=", 1)
        if room_type not in result or room_type in seen:
            raise ValueError(f"invalid or duplicate room type {room_type!r}")
        count = int(raw_count)
        if count < 0:
            raise ValueError("room counts must be non-negative")
        result[room_type] = count
        seen.add(room_type)
    if sum(result.values()) <= 0:
        raise ValueError("room counts must contain at least one scene")
    return result


def _round_robin_jobs(
    batch_id: str,
    capacities: dict[str, int],
    base_seed: int,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    room_indices = {key: 0 for key in ROOM_TYPES}
    while any(room_indices[key] < capacities[key] for key in ROOM_TYPES):
        for room_type in ROOM_TYPES:
            if room_indices[room_type] >= capacities[room_type]:
                continue
            room_indices[room_type] += 1
            global_index = len(jobs) + 1
            jobs.append(
                {
                    "scene_id": (
                        f"{batch_id}_{room_type}_{room_indices[room_type]:04d}"
                    ),
                    "room_type": room_type,
                    "room_index": room_indices[room_type],
                    "plan_index": global_index,
                    "seed": _stable_seed(base_seed, batch_id, global_index),
                }
            )
    return jobs


def _create_plan(args: argparse.Namespace, production: dict[str, Any]) -> dict[str, Any]:
    if not BATCH_ID_PATTERN.fullmatch(args.batch_id):
        raise ValueError(
            "batch id must be 3..64 lowercase letters, digits, '_' or '-', "
            "starting with a letter or digit"
        )
    explicit = _parse_room_counts(args.room_count)
    if explicit is None:
        target_count = args.target_count or int(production["default_target_count"])
        targets = _allocate_targets(target_count, production["room_type_weights"])
    else:
        targets = explicit
        target_count = sum(targets.values())
        if args.target_count is not None and args.target_count != target_count:
            raise ValueError("--target-count conflicts with --room-count total")
    multiplier = float(production["candidate_multiplier"])
    if multiplier < 1.0:
        raise ValueError("candidate_multiplier must be at least 1.0")
    capacities = {
        key: (0 if targets[key] == 0 else max(targets[key], math.ceil(targets[key] * multiplier)))
        for key in ROOM_TYPES
    }
    jobs = _round_robin_jobs(
        args.batch_id, capacities, int(production["base_seed"])
    )
    contract_paths = {
        "pipeline_config": args.pipeline_config,
        "production_config": args.production_config,
        "semantic_mapping": args.semantic_mapping,
        "layout_rules": args.layout_rules,
        "functional_rules": args.functional_rules,
        "functional_compatibility": args.functional_compatibility,
        "vocab": args.vocab,
        "hssd_lookup": args.hssd_lookup,
    }
    return {
        "schema_version": "clean_layout_batch_plan_v1",
        "batch_id": args.batch_id,
        "created_at": utc_now(),
        "target_accepted_count": target_count,
        "room_type_targets": targets,
        "candidate_capacities": capacities,
        "candidate_count": len(jobs),
        "jobs": jobs,
        "contract_paths": {
            key: str(path.resolve()) for key, path in contract_paths.items()
        },
        "contract_sha256": {
            key: sha256_file(path) for key, path in contract_paths.items()
        },
        "engine_source_sha256": _engine_source_hashes(),
        "existing_scenes": str(args.existing_scenes.resolve()),
        "formal_dataset_mutation_allowed": False,
    }


def _load_or_create_plan(
    args: argparse.Namespace,
    production: dict[str, Any],
    batch_dir: Path,
) -> dict[str, Any]:
    path = batch_dir / "batch_plan.json"
    if path.exists():
        plan = load_json(path)
        if plan.get("batch_id") != args.batch_id:
            raise ValueError("existing batch plan has a different batch_id")
        if (
            args.target_count is not None
            and args.target_count != plan["target_accepted_count"]
        ):
            raise ValueError("--target-count differs from the immutable batch plan")
        requested_counts = _parse_room_counts(args.room_count)
        if (
            requested_counts is not None
            and requested_counts != plan["room_type_targets"]
        ):
            raise ValueError("--room-count differs from the immutable batch plan")
        _assert_plan_integrity(plan)
        return plan
    plan = _create_plan(args, production)
    atomic_write_json(path, plan)
    return plan


def _job_path(batch_dir: Path, scene_id: str) -> Path:
    return batch_dir / "jobs" / f"{scene_id}.json"


def _load_job(batch_dir: Path, plan_job: dict[str, Any]) -> dict[str, Any]:
    path = _job_path(batch_dir, plan_job["scene_id"])
    if path.exists():
        return load_json(path)
    value = {
        "schema_version": "clean_layout_batch_job_v1",
        **plan_job,
        "state": "planned",
        "generation_runs": 0,
        "review_runs": 0,
        "updated_at": utc_now(),
    }
    atomic_write_json(path, value)
    return value


def _save_job(batch_dir: Path, job: dict[str, Any], **changes: Any) -> dict[str, Any]:
    value = {**job, **changes, "updated_at": utc_now()}
    atomic_write_json(_job_path(batch_dir, value["scene_id"]), value)
    return value


def _blocked_retry_state(job: dict[str, Any]) -> str:
    return (
        "review_retryable"
        if "last_review_exit_code" in job
        else "generation_retryable"
    )


def _runtime_failure_metadata(
    staging: Path, stage: str, scene_id: str
) -> dict[str, Any]:
    if stage == "review":
        path = staging / "reviews" / "semantic_failures" / f"{scene_id}.json"
    else:
        path = staging / "api_failures" / f"{scene_id}.json"
        if not path.exists():
            path = staging / "status" / f"{scene_id}.json"
    if not path.exists():
        return {}
    try:
        return load_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


class _LaunchThrottle:
    def __init__(self, interval_seconds: float) -> None:
        self.interval = max(0.0, interval_seconds)
        self.lock = threading.Lock()
        self.next_start = 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            delay = max(0.0, self.next_start - now)
            if delay:
                time.sleep(delay)
            self.next_start = time.monotonic() + self.interval


def _run_process(
    command: list[str],
    *,
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
    throttle: _LaunchThrottle,
    timeout_seconds: float | None = None,
) -> int:
    throttle.wait()
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout if isinstance(error.stdout, str) else ""
        stderr = error.stderr if isinstance(error.stderr, str) else ""
        stderr += (
            "\nCoordinator timeout: worker exceeded "
            f"{timeout_seconds:.1f} seconds.\n"
        )
        atomic_write_text(stdout_path, stdout)
        atomic_write_text(stderr_path, stderr)
        return 3
    atomic_write_text(stdout_path, result.stdout)
    atomic_write_text(stderr_path, result.stderr)
    return result.returncode


def _dispatch(
    items: list[dict[str, Any]],
    command_builder: Callable[[dict[str, Any]], list[str]],
    *,
    stage: str,
    batch_dir: Path,
    cwd: Path,
    max_workers: int,
    launch_interval: float,
) -> dict[str, int]:
    throttle = _LaunchThrottle(launch_interval)
    results: dict[str, int] = {}
    print(
        f"[batch] {stage} started for {len(items)} scene(s): "
        + ", ".join(item["scene_id"] for item in items),
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _run_process,
                command_builder(item),
                cwd=cwd,
                stdout_path=batch_dir / "logs" / stage / f"{item['scene_id']}.stdout.log",
                stderr_path=batch_dir / "logs" / stage / f"{item['scene_id']}.stderr.log",
                throttle=throttle,
            ): item["scene_id"]
            for item in items
        }
        for future in as_completed(futures):
            scene_id = futures[future]
            try:
                results[scene_id] = future.result()
            except Exception:
                results[scene_id] = 99
            print(
                f"[batch] {stage} finished: {scene_id} "
                f"exit_code={results[scene_id]}",
                flush=True,
            )
    return results


def _worker_command(args: argparse.Namespace, staging: Path, job: dict[str, Any]) -> list[str]:
    return [
        sys.executable,
        "-m",
        "clean_layout_data_engine.run_pilot",
        "--scene-id",
        job["scene_id"],
        "--room-type",
        job["room_type"],
        "--output-root",
        str(staging),
        "--seed",
        str(job["seed"]),
        "--api-config",
        str(args.api_config),
        "--config",
        str(args.pipeline_config),
        "--hssd-lookup",
        str(args.hssd_lookup),
        "--semantic-mapping",
        str(args.semantic_mapping),
        "--layout-rules",
        str(args.layout_rules),
        "--functional-rules",
        str(args.functional_rules),
        "--functional-compatibility",
        str(args.functional_compatibility),
        "--vocab",
        str(args.vocab),
        "--existing-scenes",
        str(args.existing_scenes),
        "--coverage-scenes",
        str(args.output_root / "accepted" / "scenes"),
        "--diversity-feedback",
        str(staging / "diversity" / f"{job['scene_id']}.json"),
    ]


def _review_command(args: argparse.Namespace, staging: Path, job: dict[str, Any]) -> list[str]:
    return [
        sys.executable,
        "-m",
        "clean_layout_data_engine.semantic_reviewer",
        "--scene-id",
        job["scene_id"],
        "--output-root",
        str(staging),
        "--api-config",
        str(args.api_config),
    ]


def _validator_command(
    args: argparse.Namespace,
    staging: Path,
    proposal: Path,
    global_accepted: Path,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "tools.validate_authored_scene",
        "--proposal",
        str(proposal),
        "--output-root",
        str(staging),
        "--hssd-lookup",
        str(args.hssd_lookup),
        "--semantic-mapping",
        str(args.semantic_mapping),
        "--layout-rules",
        str(args.layout_rules),
        "--functional-rules",
        str(args.functional_rules),
        "--functional-compatibility",
        str(args.functional_compatibility),
        "--vocab",
        str(args.vocab),
        "--existing-scenes",
        str(args.existing_scenes),
        "--existing-scenes",
        str(global_accepted),
    ]


def _counts(jobs: Iterable[dict[str, Any]], state: str) -> dict[str, int]:
    result = {key: 0 for key in ROOM_TYPES}
    for job in jobs:
        if job["state"] == state:
            result[job["room_type"]] += 1
    return result


def _write_manifests(
    output_root: Path,
    batch_dir: Path,
    plan: dict[str, Any],
    jobs: list[dict[str, Any]],
) -> dict[str, Any]:
    accepted = [job for job in jobs if job["state"] == "accepted"]
    rejected = [job for job in jobs if job["state"].startswith("rejected_")]
    pending = [
        job
        for job in jobs
        if job["state"] not in TERMINAL_STATES and job["state"] != "planned"
    ]
    manifests = batch_dir / "manifests"
    atomic_write_jsonl(manifests / "accepted.jsonl", accepted)
    atomic_write_jsonl(manifests / "rejected.jsonl", rejected)
    atomic_write_jsonl(manifests / "pending.jsonl", pending)
    accepted_counts = _counts(jobs, "accepted")
    target_met = all(
        accepted_counts[key] >= plan["room_type_targets"][key]
        for key in ROOM_TYPES
    )
    summary = {
        "schema_version": "clean_layout_batch_status_v1",
        "batch_id": plan["batch_id"],
        "status": "complete" if target_met else "incomplete",
        "target_met": target_met,
        "target_accepted_count": plan["target_accepted_count"],
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "pending_count": len(pending),
        "planned_unused_count": sum(job["state"] == "planned" for job in jobs),
        "room_type_targets": plan["room_type_targets"],
        "room_type_accepted": accepted_counts,
        "state_counts": {
            state: sum(job["state"] == state for job in jobs)
            for state in sorted({job["state"] for job in jobs})
        },
        "updated_at": utc_now(),
        "formal_dataset_modified": False,
        "accepted_dataset_root": str((output_root / "accepted" / "scenes").resolve()),
    }
    atomic_write_json(batch_dir / "batch_status.json", summary)
    global_manifest_path = output_root / "accepted" / "manifest.jsonl"
    global_rows: dict[str, dict[str, Any]] = {}
    if global_manifest_path.exists():
        for line in global_manifest_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                value = json.loads(line)
                global_rows[value["scene_id"]] = value
    for job in accepted:
        global_rows[job["scene_id"]] = {
            "scene_id": job["scene_id"],
            "room_type": job["room_type"],
            "batch_id": plan["batch_id"],
            "scene_path": job["accepted_scene"],
            "scene_sha256": job["scene_sha256"],
            "validation_path": job["validation_path"],
            "semantic_review_path": job["semantic_review"],
            "diversity_report_path": job["diversity_report"],
            "diversity_design": job.get("diversity_design"),
            "accepted_at": job["accepted_at"],
        }
    atomic_write_jsonl(
        global_manifest_path,
        (global_rows[key] for key in sorted(global_rows)),
    )
    return summary


def _select_generation_wave(
    jobs: list[dict[str, Any]],
    targets: dict[str, int],
    max_runs: int,
    wave_size: int,
) -> list[dict[str, Any]]:
    accepted = _counts(jobs, "accepted")
    selected: list[dict[str, Any]] = []
    room_selected = {key: 0 for key in ROOM_TYPES}
    candidates = sorted(
        (
        job
        for job in jobs
        if job["state"] in {"planned", "generation_retryable"}
        and job["generation_runs"] < max_runs
        ),
        key=lambda job: job["state"] != "planned",
    )
    while len(selected) < wave_size:
        changed = False
        for room_type in ROOM_TYPES:
            deficit = targets[room_type] - accepted[room_type] - room_selected[room_type]
            if deficit <= 0:
                continue
            job = next(
                (value for value in candidates if value["room_type"] == room_type),
                None,
            )
            if job is None:
                continue
            candidates.remove(job)
            selected.append(job)
            room_selected[room_type] += 1
            changed = True
            if len(selected) >= wave_size:
                break
        if not changed:
            break
    return selected


def _promote_reviewed_jobs(
    args: argparse.Namespace,
    batch_dir: Path,
    staging: Path,
    jobs: list[dict[str, Any]],
    plan: dict[str, Any],
    max_generation_runs: int,
) -> list[dict[str, Any]]:
    global_accepted = args.output_root / "accepted" / "scenes"
    global_accepted.mkdir(parents=True, exist_ok=True)
    accepted_counts = _counts(jobs, "accepted")
    for index, job in enumerate(jobs):
        if job["state"] != "semantic_pass_pending_diversity":
            continue
        room_type = job["room_type"]
        if accepted_counts[room_type] >= plan["room_type_targets"][room_type]:
            jobs[index] = _save_job(batch_dir, job, state="surplus_not_needed")
            continue
        command = _validator_command(
            args,
            staging,
            staging / "compiled_proposals" / f"{job['scene_id']}.json",
            global_accepted,
        )
        result = subprocess.run(
            command,
            cwd=args.project_root,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
        )
        atomic_write_text(
            batch_dir / "logs" / "diversity" / f"{job['scene_id']}.stdout.log",
            result.stdout,
        )
        atomic_write_text(
            batch_dir / "logs" / "diversity" / f"{job['scene_id']}.stderr.log",
            result.stderr,
        )
        validation_path = staging / "validation" / f"{job['scene_id']}.json"
        if result.returncode != 0 or not validation_path.exists():
            jobs[index] = _save_job(
                batch_dir, job, state="generation_retryable", last_stage="diversity_revalidation"
            )
            continue
        validation = load_json(validation_path)
        if not validation["checks"]["diversity_novelty_pass"]:
            report_path = staging / "diversity" / f"{job['scene_id']}.json"
            if job["generation_runs"] < max_generation_runs:
                jobs[index] = _save_job(
                    batch_dir,
                    job,
                    state="generation_retryable",
                    last_stage="diversity_retry",
                    diversity_report=str(report_path),
                    diversity_retry_count=int(job.get("diversity_retry_count", 0)) + 1,
                )
                continue
            jobs[index] = _save_job(
                batch_dir,
                job,
                state="rejected_diversity",
                diversity_report=str(report_path),
            )
            continue
        scene_path = staging / "scenes" / f"{job['scene_id']}.json"
        destination = global_accepted / scene_path.name
        atomic_write_text(destination, scene_path.read_text(encoding="utf-8"))
        brief_path = staging / "briefs" / f"{job['scene_id']}.json"
        diversity_design = None
        if brief_path.exists():
            diversity_design = load_json(brief_path).get("diversity_design")
        jobs[index] = _save_job(
            batch_dir,
            job,
            state="accepted",
            accepted_scene=str(destination),
            scene_sha256=sha256_file(destination),
            validation_path=str(validation_path),
            diversity_report=str(
                staging / "diversity" / f"{job['scene_id']}.json"
            ),
            diversity_design=diversity_design,
            accepted_at=utc_now(),
        )
        accepted_counts[room_type] += 1
    return jobs


def _pipeline_outstanding_counts(jobs: list[dict[str, Any]]) -> dict[str, int]:
    states = {
        "authoring",
        "hard_valid_pending_review",
        "review_retryable",
        "reviewing",
        "semantic_pass_pending_diversity",
    }
    return {
        room_type: sum(
            job["room_type"] == room_type and job["state"] in states
            for job in jobs
        )
        for room_type in ROOM_TYPES
    }


def _select_pipeline_generation_jobs(
    jobs: list[dict[str, Any]],
    targets: dict[str, int],
    max_runs: int,
    limit: int,
    speculative_buffer_per_room: int,
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    accepted = _counts(jobs, "accepted")
    outstanding = _pipeline_outstanding_counts(jobs)
    selected: list[dict[str, Any]] = []
    room_selected = {key: 0 for key in ROOM_TYPES}
    candidates = sorted(
        (
            job
            for job in jobs
            if job["state"] in {"planned", "generation_retryable"}
            and job["generation_runs"] < max_runs
        ),
        key=lambda job: job["state"] != "planned",
    )
    while len(selected) < limit:
        eligible: list[tuple[int, int, int, str, dict[str, Any]]] = []
        for room_order, room_type in enumerate(ROOM_TYPES):
            deficit = targets[room_type] - accepted[room_type]
            if deficit <= 0:
                continue
            desired_outstanding = deficit + speculative_buffer_per_room
            available = (
                desired_outstanding
                - outstanding[room_type]
                - room_selected[room_type]
            )
            if available <= 0:
                continue
            job = next(
                (value for value in candidates if value["room_type"] == room_type),
                None,
            )
            if job is None:
                continue
            eligible.append(
                (available, deficit, -room_order, room_type, job)
            )
        if not eligible:
            break
        _, _, _, room_type, job = max(eligible, key=lambda value: value[:3])
        candidates.remove(job)
        selected.append(job)
        room_selected[room_type] += 1
    return selected


def _pipeline_worker_split(total_workers: int) -> tuple[int, int]:
    if total_workers < 2:
        return total_workers, 0
    reviewer_workers = max(1, (total_workers + 3) // 4)
    return total_workers - reviewer_workers, reviewer_workers


def _resolve_pipeline_workers(
    args: argparse.Namespace, production: dict[str, Any]
) -> tuple[int, int]:
    total_workers = int(args.max_workers or production.get("max_workers", 16))
    default_author, default_reviewer = _pipeline_worker_split(total_workers)
    if args.author_workers is None and args.reviewer_workers is None:
        if args.max_workers is None:
            author_workers = int(
                production.get("author_workers", default_author)
            )
            reviewer_workers = int(
                production.get("reviewer_workers", default_reviewer)
            )
        else:
            author_workers, reviewer_workers = default_author, default_reviewer
    else:
        author_workers = int(
            args.author_workers
            if args.author_workers is not None
            else default_author
        )
        reviewer_workers = int(
            args.reviewer_workers
            if args.reviewer_workers is not None
            else default_reviewer
        )
    if author_workers < 1 or reviewer_workers < 1:
        raise ValueError("pipeline author/reviewer workers must both be positive")
    if author_workers + reviewer_workers > total_workers:
        raise ValueError(
            "pipeline author/reviewer workers exceed the max-workers cap: "
            f"{author_workers}+{reviewer_workers}>{total_workers}"
        )
    return author_workers, reviewer_workers


def _run_pipelined_jobs(
    args: argparse.Namespace,
    production: dict[str, Any],
    batch_dir: Path,
    staging: Path,
    plan: dict[str, Any],
    jobs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    author_workers, reviewer_workers = _resolve_pipeline_workers(
        args, production
    )
    speculative_buffer = int(
        args.speculative_buffer_per_room
        if args.speculative_buffer_per_room is not None
        else production.get("speculative_buffer_per_room", 2)
    )
    if speculative_buffer < 0:
        raise ValueError("speculative buffer per room must be non-negative")
    max_generation_runs = int(production["max_generation_runs_per_candidate"])
    max_review_runs = int(production["max_review_runs_per_candidate"])
    launch_interval = float(production["launch_interval_seconds"])
    max_api_failures = int(
        production.get("max_consecutive_runtime_api_failure_waves", 3)
    )
    backoff_seconds = float(
        production.get("runtime_api_retry_backoff_seconds", 15.0)
    )
    backoff_multiplier = float(
        production.get("runtime_api_retry_backoff_multiplier", 2.0)
    )
    backoff_max = float(production.get("runtime_api_retry_max_seconds", 60.0))
    worker_timeout = float(
        production.get("worker_process_timeout_seconds", 300.0)
    )
    if worker_timeout <= 0:
        raise ValueError("worker process timeout must be positive")
    throttle = _LaunchThrottle(launch_interval)
    active: dict[Future[int], tuple[str, str]] = {}
    api_failure_streak = 0
    api_failure_jobs: list[tuple[str, str]] = []
    configuration_error = False
    last_report: tuple[int, int, int] | None = None

    def job_index(scene_id: str) -> int:
        return next(
            index for index, value in enumerate(jobs)
            if value["scene_id"] == scene_id
        )

    def submit_generation(
        executor: ThreadPoolExecutor, job: dict[str, Any]
    ) -> None:
        index = job_index(job["scene_id"])
        jobs[index] = _save_job(
            batch_dir,
            jobs[index],
            state="authoring",
            generation_runs=jobs[index]["generation_runs"] + 1,
        )
        future = executor.submit(
            _run_process,
            _worker_command(args, staging, jobs[index]),
            cwd=args.project_root,
            stdout_path=(
                batch_dir / "logs" / "generation" / f"{job['scene_id']}.stdout.log"
            ),
            stderr_path=(
                batch_dir / "logs" / "generation" / f"{job['scene_id']}.stderr.log"
            ),
            throttle=throttle,
            timeout_seconds=worker_timeout,
        )
        active[future] = (job["scene_id"], "generation")

    def submit_review(executor: ThreadPoolExecutor, job: dict[str, Any]) -> None:
        index = job_index(job["scene_id"])
        jobs[index] = _save_job(
            batch_dir,
            jobs[index],
            state="reviewing",
            review_runs=jobs[index]["review_runs"] + 1,
        )
        future = executor.submit(
            _run_process,
            _review_command(args, staging, jobs[index]),
            cwd=args.project_root,
            stdout_path=(
                batch_dir / "logs" / "review" / f"{job['scene_id']}.stdout.log"
            ),
            stderr_path=(
                batch_dir / "logs" / "review" / f"{job['scene_id']}.stderr.log"
            ),
            throttle=throttle,
            timeout_seconds=worker_timeout,
        )
        active[future] = (job["scene_id"], "review")

    def apply_result(scene_id: str, stage: str, code: int) -> None:
        nonlocal api_failure_streak, api_failure_jobs, configuration_error
        index = job_index(scene_id)
        job = jobs[index]
        if code == 3:
            metadata = _runtime_failure_metadata(staging, stage, scene_id)
            if stage == "review":
                changes = {
                    "state": "review_retryable",
                    "review_runs": max(0, job["review_runs"] - 1),
                    "last_review_exit_code": 3,
                }
            else:
                changes = {
                    "state": "generation_retryable",
                    "generation_runs": max(0, job["generation_runs"] - 1),
                    "last_generation_exit_code": 3,
                }
            changes.update(
                {
                    "api_error_type": metadata.get(
                        "error_type", "unknown_runtime_api_error"
                    ),
                    "api_error_category": metadata.get(
                        "error_category", "unknown_runtime_api_failure"
                    ),
                }
            )
            jobs[index] = _save_job(batch_dir, job, **changes)
            api_failure_streak += 1
            api_failure_jobs.append((scene_id, stage))
            return

        api_failure_streak = 0
        api_failure_jobs = []
        clear_api = {
            "api_error_type": None,
            "api_error_category": None,
            "api_failure_waves": 0,
        }
        if code == 2:
            exit_key = (
                "last_review_exit_code"
                if stage == "review"
                else "last_generation_exit_code"
            )
            jobs[index] = _save_job(
                batch_dir,
                job,
                state="blocked_configuration",
                **{exit_key: 2},
            )
            configuration_error = True
            return
        if stage == "review":
            if code == 0:
                state = "semantic_pass_pending_diversity"
            elif code == 1:
                state = "rejected_semantic"
            elif job["review_runs"] < max_review_runs:
                state = "review_retryable"
            else:
                state = "rejected_semantic"
            jobs[index] = _save_job(
                batch_dir,
                job,
                state=state,
                last_review_exit_code=code,
                semantic_review=str(
                    staging / "reviews" / "semantic" / f"{scene_id}.json"
                ),
                **(clear_api if code == 0 else {}),
            )
            return
        if code == 0:
            state = "hard_valid_pending_review"
        elif job["generation_runs"] < max_generation_runs:
            state = "generation_retryable"
        else:
            state = "rejected_generation"
        jobs[index] = _save_job(
            batch_dir,
            job,
            state=state,
            last_generation_exit_code=code,
            **(clear_api if code == 0 else {}),
        )

    print(
        f"[batch] {plan['batch_id']} pipelined target="
        f"{plan['target_accepted_count']} author_workers={author_workers} "
        f"reviewer_workers={reviewer_workers} speculative_per_room="
        f"{speculative_buffer}",
        flush=True,
    )
    with (
        ThreadPoolExecutor(max_workers=author_workers) as author_executor,
        ThreadPoolExecutor(max_workers=reviewer_workers) as reviewer_executor,
    ):
        while True:
            _assert_plan_integrity(plan)
            jobs = _promote_reviewed_jobs(
                args,
                batch_dir,
                staging,
                jobs,
                plan,
                max_generation_runs,
            )
            summary = _write_manifests(args.output_root, batch_dir, plan, jobs)
            report = (
                summary["accepted_count"],
                summary["rejected_count"],
                summary["pending_count"],
            )
            if report != last_report:
                print(
                    f"[batch] status: accepted={report[0]}/"
                    f"{summary['target_accepted_count']}, rejected={report[1]}, "
                    f"pending={report[2]}, active={len(active)}",
                    flush=True,
                )
                last_report = report

            target_met = bool(summary["target_met"])
            if target_met and not active:
                for index, job in enumerate(jobs):
                    if job["state"] in {
                        "generation_retryable",
                        "hard_valid_pending_review",
                        "review_retryable",
                        "semantic_pass_pending_diversity",
                        "blocked_api_runtime",
                        "blocked_configuration",
                    }:
                        jobs[index] = _save_job(
                            batch_dir, job, state="surplus_not_needed"
                        )
                summary = _write_manifests(args.output_root, batch_dir, plan, jobs)
                break

            if configuration_error and not active:
                _write_manifests(args.output_root, batch_dir, plan, jobs)
                raise RuntimeError(
                    "pipeline worker reported a configuration/contract error; "
                    "batch stopped before further API calls"
                )
            if api_failure_streak >= max_api_failures and not active:
                for scene_id, stage in dict.fromkeys(api_failure_jobs):
                    index = job_index(scene_id)
                    job = jobs[index]
                    jobs[index] = _save_job(
                        batch_dir,
                        job,
                        state="blocked_api_runtime",
                        api_failure_waves=api_failure_streak,
                        api_failure_stage=stage,
                    )
                _write_manifests(args.output_root, batch_dir, plan, jobs)
                raise RuntimeError(
                    "consecutive runtime API failures reached the configured "
                    f"limit ({max_api_failures}); pipeline batch halted"
                )
            if not active and 0 < api_failure_streak < max_api_failures:
                delay = min(
                    backoff_max,
                    backoff_seconds
                    * backoff_multiplier ** (api_failure_streak - 1),
                )
                print(
                    f"[batch] API retry {api_failure_streak}/"
                    f"{max_api_failures} after {delay:.1f}s",
                    flush=True,
                )
                time.sleep(delay)

            scheduling_allowed = (
                not target_met
                and not configuration_error
                and api_failure_streak < max_api_failures
            )
            if scheduling_allowed:
                review_active = sum(stage == "review" for _, stage in active.values())
                reviewable = sorted(
                    (
                        job for job in jobs
                        if job["state"]
                        in {"hard_valid_pending_review", "review_retryable"}
                        and job["review_runs"] < max_review_runs
                    ),
                    key=lambda job: job["state"] != "hard_valid_pending_review",
                )
                for job in reviewable[: max(0, reviewer_workers - review_active)]:
                    submit_review(reviewer_executor, job)

                author_active = sum(
                    stage == "generation" for _, stage in active.values()
                )
                selected = _select_pipeline_generation_jobs(
                    jobs,
                    plan["room_type_targets"],
                    max_generation_runs,
                    max(0, author_workers - author_active),
                    speculative_buffer,
                )
                for job in selected:
                    submit_generation(author_executor, job)

            if not active:
                can_generate = bool(
                    _select_pipeline_generation_jobs(
                        jobs,
                        plan["room_type_targets"],
                        max_generation_runs,
                        1,
                        speculative_buffer,
                    )
                )
                can_review = any(
                    job["state"]
                    in {"hard_valid_pending_review", "review_retryable"}
                    and job["review_runs"] < max_review_runs
                    for job in jobs
                )
                if not can_generate and not can_review:
                    break
                continue

            done, _ = wait(tuple(active), return_when=FIRST_COMPLETED)
            for future in done:
                scene_id, stage = active.pop(future)
                try:
                    code = future.result()
                except Exception:
                    code = 99
                print(
                    f"[batch] {stage} finished: {scene_id} exit_code={code}",
                    flush=True,
                )
                apply_result(scene_id, stage, code)

    return jobs, summary


def run_batch(args: argparse.Namespace) -> int:
    production = load_json(args.production_config)
    if not production.get("review_enabled", False):
        raise ValueError("formal production requires review_enabled=true")
    max_workers = int(args.max_workers or production["max_workers"])
    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    launch_interval = float(production["launch_interval_seconds"])
    batch_dir = args.output_root / "batches" / args.batch_id
    staging = batch_dir / "staging"
    with BatchLock(
        args.output_root / ".production.lock",
        recover_stale=args.recover_stale_lock,
    ):
        if args.status_only:
            status_path = batch_dir / "batch_status.json"
            if not status_path.exists():
                raise ValueError(f"batch {args.batch_id!r} does not exist")
            print(json.dumps(load_json(status_path), ensure_ascii=False, indent=2))
            return 0
        plan = _load_or_create_plan(args, production, batch_dir)
        jobs = [_load_job(batch_dir, value) for value in plan["jobs"]]
        for index, job in enumerate(jobs):
            if job["state"] == "authoring":
                jobs[index] = _save_job(
                    batch_dir, job, state="generation_retryable", recovered=True
                )
            elif job["state"] == "reviewing":
                jobs[index] = _save_job(
                    batch_dir, job, state="review_retryable", recovered=True
                )
            elif job["state"] == "blocked_configuration":
                jobs[index] = _save_job(
                    batch_dir,
                    job,
                    state=_blocked_retry_state(job),
                    recovered=True,
                )
            elif job["state"] == "blocked_api_runtime":
                jobs[index] = _save_job(
                    batch_dir,
                    job,
                    state=_blocked_retry_state(job),
                    recovered=True,
                )
            elif job["state"] == "accepted":
                accepted_path = Path(job["accepted_scene"])
                if (
                    not accepted_path.exists()
                    or sha256_file(accepted_path) != job["scene_sha256"]
                ):
                    raise ValueError(
                        f"accepted artifact missing or changed for {job['scene_id']}"
                    )
        if args.plan_only:
            if any(job["state"] != "planned" for job in jobs):
                raise ValueError("--plan-only cannot be used after a batch has started")
            summary = _write_manifests(args.output_root, batch_dir, plan, jobs)
            summary["status"] = "planned"
            atomic_write_json(batch_dir / "batch_status.json", summary)
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 0
        if bool(production.get("pipeline_enabled", False) or getattr(args, "pipeline", False)):
            jobs, summary = _run_pipelined_jobs(
                args, production, batch_dir, staging, plan, jobs
            )
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 0 if summary["target_met"] else 1
        max_generation_runs = int(production["max_generation_runs_per_candidate"])
        max_review_runs = int(production["max_review_runs_per_candidate"])
        wave_size = int(getattr(args, "wave_size", None) or production["wave_size"])
        if wave_size < 1:
            raise ValueError("wave_size must be positive")
        max_runtime_api_failure_waves = int(
            production.get("max_consecutive_runtime_api_failure_waves", 3)
        )
        runtime_api_retry_backoff_seconds = float(
            production.get("runtime_api_retry_backoff_seconds", 15.0)
        )
        runtime_api_retry_backoff_multiplier = float(
            production.get("runtime_api_retry_backoff_multiplier", 2.0)
        )
        runtime_api_retry_max_seconds = float(
            production.get("runtime_api_retry_max_seconds", 60.0)
        )
        if max_runtime_api_failure_waves < 1:
            raise ValueError("max_consecutive_runtime_api_failure_waves must be positive")
        if runtime_api_retry_backoff_seconds < 0:
            raise ValueError("runtime_api_retry_backoff_seconds must be non-negative")
        if runtime_api_retry_backoff_multiplier < 1:
            raise ValueError("runtime_api_retry_backoff_multiplier must be at least one")
        if runtime_api_retry_max_seconds < runtime_api_retry_backoff_seconds:
            raise ValueError(
                "runtime_api_retry_max_seconds must be >= runtime_api_retry_backoff_seconds"
            )
        consecutive_runtime_api_failure_waves = 0

        def record_runtime_api_wave(
            current: list[dict[str, Any]],
            results: dict[str, int],
            *,
            stage: str,
        ) -> bool:
            """Return true when a full API-failure wave was retried or halted."""
            nonlocal consecutive_runtime_api_failure_waves
            runtime_failures = [
                job for job in current if results[job["scene_id"]] == 3
            ]
            if not runtime_failures:
                consecutive_runtime_api_failure_waves = 0
                return False

            count_key = "review_runs" if stage == "review" else "generation_runs"
            retry_state = "review_retryable" if stage == "review" else "generation_retryable"
            exit_key = "last_review_exit_code" if stage == "review" else "last_generation_exit_code"
            failed_scene_ids = {job["scene_id"] for job in runtime_failures}
            for index, job in enumerate(jobs):
                if job["scene_id"] not in failed_scene_ids:
                    continue
                # An unavailable provider says nothing about the candidate. Do not
                # spend its quality-attempt budget on a transport/runtime failure.
                jobs[index] = _save_job(
                    batch_dir,
                    job,
                    state=retry_state,
                    **{count_key: max(0, job[count_key] - 1), exit_key: 3},
                )

            if len(runtime_failures) != len(current):
                consecutive_runtime_api_failure_waves = 0
                return False

            consecutive_runtime_api_failure_waves += 1
            failure_metadata = [
                _runtime_failure_metadata(staging, stage, job["scene_id"])
                for job in runtime_failures
            ]
            error_types = {
                value.get("error_type") for value in failure_metadata
                if value.get("error_type")
            }
            error_categories = {
                value.get("error_category") for value in failure_metadata
                if value.get("error_category")
            }
            error_type = (
                next(iter(error_types))
                if len(error_types) == 1
                else (
                    "unknown_runtime_api_error"
                    if not error_types
                    else "mixed_runtime_api_errors"
                )
            )
            error_category = (
                next(iter(error_categories))
                if len(error_categories) == 1
                else (
                    "unknown_runtime_api_failure"
                    if not error_categories
                    else "mixed_runtime_api_failures"
                )
            )
            if consecutive_runtime_api_failure_waves >= max_runtime_api_failure_waves:
                for index, job in enumerate(jobs):
                    if job["scene_id"] in failed_scene_ids:
                        jobs[index] = _save_job(
                            batch_dir,
                            job,
                            state="blocked_api_runtime",
                            api_error_type=error_type,
                            api_error_category=error_category,
                            api_failure_waves=consecutive_runtime_api_failure_waves,
                        )
                _write_manifests(args.output_root, batch_dir, plan, jobs)
                raise RuntimeError(
                    "consecutive runtime API failure waves reached the configured "
                    f"limit ({max_runtime_api_failure_waves}; "
                    f"{error_category}/{error_type}); batch halted"
                )
            wait_seconds = min(
                runtime_api_retry_max_seconds,
                runtime_api_retry_backoff_seconds
                * runtime_api_retry_backoff_multiplier
                ** (consecutive_runtime_api_failure_waves - 1),
            )
            print(
                f"[batch] {stage} API runtime failure wave "
                f"{consecutive_runtime_api_failure_waves}/{max_runtime_api_failure_waves}; "
                f"retrying after {wait_seconds:.1f}s",
                flush=True,
            )
            time.sleep(wait_seconds)
            return True
        print(
            f"[batch] {plan['batch_id']} target={plan['target_accepted_count']} "
            f"candidate_capacity={plan['candidate_count']} workers={max_workers}",
            flush=True,
        )
        while True:
            _assert_plan_integrity(plan)
            reviewable = [
                job
                for job in jobs
                if job["state"] in {"hard_valid_pending_review", "review_retryable"}
                and job["review_runs"] < max_review_runs
            ]
            if reviewable:
                for job in reviewable:
                    index = jobs.index(job)
                    jobs[index] = _save_job(
                        batch_dir,
                        job,
                        state="reviewing",
                        review_runs=job["review_runs"] + 1,
                    )
                current = [jobs[next(i for i, value in enumerate(jobs) if value["scene_id"] == item["scene_id"])] for item in reviewable]
                review_results = _dispatch(
                    current,
                    lambda job: _review_command(args, staging, job),
                    stage="review",
                    batch_dir=batch_dir,
                    cwd=args.project_root,
                    max_workers=max_workers,
                    launch_interval=launch_interval,
                )
                if record_runtime_api_wave(current, review_results, stage="review"):
                    continue
                if any(code == 2 for code in review_results.values()):
                    for index, job in enumerate(jobs):
                        if review_results.get(job["scene_id"]) == 2:
                            jobs[index] = _save_job(
                                batch_dir,
                                job,
                                state="blocked_configuration",
                                last_review_exit_code=2,
                            )
                    _write_manifests(args.output_root, batch_dir, plan, jobs)
                    raise RuntimeError(
                        "review worker reported a configuration/contract error; "
                        "batch stopped before further API calls"
                    )
                for index, job in enumerate(jobs):
                    if job["scene_id"] not in review_results:
                        continue
                    code = review_results[job["scene_id"]]
                    if code == 3:
                        # record_runtime_api_wave already restored this candidate's
                        # budget and persisted its retryable state.
                        continue
                    if code == 0:
                        state = "semantic_pass_pending_diversity"
                    elif code == 1:
                        state = "rejected_semantic"
                    elif code == 3:
                        state = "review_retryable"
                    elif job["review_runs"] < max_review_runs:
                        state = "review_retryable"
                    else:
                        state = "rejected_semantic"
                    jobs[index] = _save_job(
                        batch_dir,
                        job,
                        state=state,
                        last_review_exit_code=code,
                        **(
                            {
                                "api_error_type": None,
                                "api_error_category": None,
                                "api_failure_waves": 0,
                            }
                            if code == 0
                            else {}
                        ),
                        semantic_review=str(
                            staging / "reviews" / "semantic" / f"{job['scene_id']}.json"
                        ),
                    )

            jobs = _promote_reviewed_jobs(
                args,
                batch_dir,
                staging,
                jobs,
                plan,
                max_generation_runs,
            )
            summary = _write_manifests(args.output_root, batch_dir, plan, jobs)
            print(
                f"[batch] status: accepted={summary['accepted_count']}/"
                f"{summary['target_accepted_count']}, rejected={summary['rejected_count']}, "
                f"pending={summary['pending_count']}",
                flush=True,
            )
            if summary["target_met"]:
                break

            selected = _select_generation_wave(
                jobs,
                plan["room_type_targets"],
                max_generation_runs,
                wave_size,
            )
            if not selected:
                break
            for job in selected:
                index = jobs.index(job)
                jobs[index] = _save_job(
                    batch_dir,
                    job,
                    state="authoring",
                    generation_runs=job["generation_runs"] + 1,
                )
            current = [jobs[next(i for i, value in enumerate(jobs) if value["scene_id"] == item["scene_id"])] for item in selected]
            generation_results = _dispatch(
                current,
                lambda job: _worker_command(args, staging, job),
                stage="generation",
                batch_dir=batch_dir,
                cwd=args.project_root,
                max_workers=max_workers,
                launch_interval=launch_interval,
            )
            if record_runtime_api_wave(current, generation_results, stage="generation"):
                continue
            if any(code == 2 for code in generation_results.values()):
                for index, job in enumerate(jobs):
                    if generation_results.get(job["scene_id"]) == 2:
                        jobs[index] = _save_job(
                            batch_dir,
                            job,
                            state="blocked_configuration",
                            last_generation_exit_code=2,
                        )
                _write_manifests(args.output_root, batch_dir, plan, jobs)
                raise RuntimeError(
                    "generation worker reported a configuration/contract error; "
                    "batch stopped before further API calls"
                )
            for index, job in enumerate(jobs):
                if job["scene_id"] not in generation_results:
                    continue
                code = generation_results[job["scene_id"]]
                if code == 3:
                    # record_runtime_api_wave already restored this candidate's
                    # budget and persisted its retryable state.
                    continue
                if code == 0:
                    state = "hard_valid_pending_review"
                elif code == 3:
                    state = "generation_retryable"
                elif job["generation_runs"] < max_generation_runs:
                    state = "generation_retryable"
                else:
                    state = "rejected_generation"
                jobs[index] = _save_job(
                    batch_dir,
                    job,
                    state=state,
                    last_generation_exit_code=code,
                    **(
                        {
                            "api_error_type": None,
                            "api_error_category": None,
                            "api_failure_waves": 0,
                        }
                        if code == 0
                        else {}
                    ),
                )

        summary = _write_manifests(args.output_root, batch_dir, plan, jobs)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if summary["target_met"] else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--target-count", type=int)
    parser.add_argument("--room-count", action="append", default=[])
    parser.add_argument("--max-workers", type=int)
    parser.add_argument("--wave-size", type=int)
    parser.add_argument("--pipeline", action="store_true")
    parser.add_argument("--author-workers", type=int)
    parser.add_argument("--reviewer-workers", type=int)
    parser.add_argument("--speculative-buffer-per-room", type=int)
    parser.add_argument("--recover-stale-lock", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--status-only", action="store_true")
    parser.add_argument("--api-config", type=Path, default=ENGINE_DIR / "config.json")
    parser.add_argument(
        "--pipeline-config", type=Path, default=ENGINE_DIR / "pipeline_config.json"
    )
    parser.add_argument(
        "--production-config", type=Path, default=ENGINE_DIR / "production_config.json"
    )
    parser.add_argument(
        "--project-root", type=Path, default=ENGINE_DIR.parent
    )
    parser.add_argument(
        "--hssd-lookup",
        type=Path,
        default=Path("../hssd-annotations/data/hssd_annotation_lookup.json.gz"),
    )
    parser.add_argument(
        "--semantic-mapping",
        type=Path,
        default=Path("data/clean_layouts_gpt56_v1/semantic_mapping.json"),
    )
    parser.add_argument(
        "--layout-rules",
        type=Path,
        default=Path("configs/authored_layout_validation_rules_v2.json"),
    )
    parser.add_argument(
        "--functional-rules",
        type=Path,
        default=Path("configs/functional_partner_rules_hssd_v1.json"),
    )
    parser.add_argument(
        "--functional-compatibility",
        type=Path,
        default=Path("configs/functional_partner_category_compatibility_v1.json"),
    )
    parser.add_argument(
        "--vocab",
        type=Path,
        default=Path("configs/furniture_vocab_bootstrap_v1.json"),
    )
    parser.add_argument(
        "--existing-scenes",
        type=Path,
        default=Path("data/clean_layouts_gpt56_v2_authored/scenes"),
    )
    return parser.parse_args()


def main() -> int:
    try:
        args = parse_args()
        args.project_root = args.project_root.resolve()
        for name in (
            "api_config",
            "pipeline_config",
            "production_config",
            "hssd_lookup",
            "semantic_mapping",
            "layout_rules",
            "functional_rules",
            "functional_compatibility",
            "vocab",
            "existing_scenes",
            "output_root",
        ):
            path = getattr(args, name)
            if not path.is_absolute():
                setattr(args, name, (args.project_root / path).resolve())
        return run_batch(args)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"Batch engine error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
