"""Run recoverable clean-layout batches until a global accepted-scene target."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from clean_layout_data_engine.storage import (
    BatchLock,
    atomic_write_json,
    load_json,
    utc_now,
)


STATE_FILENAME = "continuous_production_status.json"
LOCK_FILENAME = ".continuous_production.lock"


def count_accepted_scenes(output_root: Path) -> int:
    return sum(1 for _ in (output_root / "accepted" / "scenes").glob("*.json"))


def next_batch_index(output_root: Path, prefix: str) -> int:
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    indices = []
    for path in (output_root / "batches").glob(f"{prefix}_*"):
        match = pattern.fullmatch(path.name)
        if path.is_dir() and match:
            indices.append(int(match.group(1)))
    return max(indices, default=0) + 1


def batch_target_met(output_root: Path, batch_id: str) -> bool:
    path = output_root / "batches" / batch_id / "batch_status.json"
    return path.exists() and bool(load_json(path).get("target_met"))


def batch_has_configuration_block(output_root: Path, batch_id: str) -> bool:
    path = output_root / "batches" / batch_id / "batch_status.json"
    if not path.exists():
        return False
    states = load_json(path).get("state_counts", {})
    return int(states.get("blocked_configuration", 0)) > 0


def batch_has_api_runtime_block(output_root: Path, batch_id: str) -> bool:
    path = output_root / "batches" / batch_id / "batch_status.json"
    if not path.exists():
        return False
    states = load_json(path).get("state_counts", {})
    return int(states.get("blocked_api_runtime", 0)) > 0


def _write_state(path: Path, state: dict[str, Any], **updates: Any) -> None:
    state.update(updates)
    state["updated_at"] = utc_now()
    atomic_write_json(path, state)


def _wait_for_batch_lock(
    output_root: Path,
    state_path: Path,
    state: dict[str, Any],
    poll_seconds: float,
) -> None:
    lock_path = output_root / ".production.lock"
    announced = False
    while lock_path.exists():
        if not announced:
            print("[continuous] waiting for the active batch to release its lock", flush=True)
            announced = True
        _write_state(state_path, state, status="waiting_for_active_batch")
        time.sleep(poll_seconds)


def _run_verification(
    project_root: Path,
    output_root: Path,
    minimum_scenes: int,
    log_dir: Path,
) -> int:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"verify_{minimum_scenes:04d}.log"
    command = [
        sys.executable,
        "-m",
        "clean_layout_data_engine.verify_dataset",
        "--output-root",
        str(output_root),
        "--minimum-scenes",
        str(minimum_scenes),
    ]
    with log_path.open("w", encoding="utf-8", newline="\n") as stream:
        result = subprocess.run(
            command,
            cwd=project_root,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    print(
        f"[continuous] dataset verification minimum={minimum_scenes} "
        f"exit_code={result.returncode} log={log_path}",
        flush=True,
    )
    return result.returncode


def _run_batch(
    args: argparse.Namespace,
    batch_id: str,
    target_count: int,
    workers: int,
    invocation: int,
) -> tuple[int, Path]:
    reviewer_workers = max(1, (workers + 3) // 4)
    author_workers = workers - reviewer_workers
    log_dir = args.output_root / "continuous_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / (
        f"{batch_id}_run_{invocation:03d}_w{workers}_"
        f"a{author_workers}_r{reviewer_workers}.log"
    )
    command = [
        sys.executable,
        "-m",
        "clean_layout_data_engine.batch_engine",
        "--batch-id",
        batch_id,
        "--output-root",
        str(args.output_root),
        "--target-count",
        str(target_count),
        "--max-workers",
        str(workers),
        "--wave-size",
        str(workers),
        "--pipeline",
        "--author-workers",
        str(author_workers),
        "--reviewer-workers",
        str(reviewer_workers),
        "--speculative-buffer-per-room",
        str(args.speculative_buffer_per_room),
        "--existing-scenes",
        str(args.existing_scenes),
    ]
    print(
        f"[continuous] starting {batch_id} target={target_count} "
        f"author_workers={author_workers} reviewer_workers={reviewer_workers}",
        flush=True,
    )
    with log_path.open("w", encoding="utf-8", newline="\n") as stream:
        result = subprocess.run(
            command,
            cwd=args.project_root,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    print(
        f"[continuous] {batch_id} exit_code={result.returncode} log={log_path}",
        flush=True,
    )
    return result.returncode, log_path


def _tail_contains_fatal_error(path: Path) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8", errors="replace")[-12000:].lower()
    markers = (
        "contract drift",
        "source drift",
        "configuration/contract error",
        "differs from the immutable batch plan",
    )
    return any(marker in text for marker in markers)


def _initial_state(args: argparse.Namespace) -> dict[str, Any]:
    current_batch_id = args.initial_batch_id
    current_target = None
    if current_batch_id:
        plan_path = (
            args.output_root / "batches" / current_batch_id / "batch_plan.json"
        )
        if plan_path.exists():
            current_target = int(load_json(plan_path)["target_accepted_count"])
    return {
        "schema_version": "clean_layout_continuous_production_v1",
        "status": "starting",
        "target_total": args.target_total,
        "batch_size": args.batch_size,
        "preferred_workers": args.workers,
        "intermediate_workers": args.intermediate_workers,
        "fallback_workers": args.fallback_workers,
        "speculative_buffer_per_room": args.speculative_buffer_per_room,
        "current_workers": args.workers,
        "consecutive_supervisor_api_failures": 0,
        "current_batch_id": current_batch_id,
        "current_batch_target": current_target,
        "next_batch_index": next_batch_index(args.output_root, args.batch_prefix),
        "batch_invocations": {},
        "completed_batches": [],
        "abandoned_batches": [],
        "accepted_count": count_accepted_scenes(args.output_root),
        "created_at": utc_now(),
        "updated_at": utc_now(),
    }


def _load_or_create_state(args: argparse.Namespace, state_path: Path) -> dict[str, Any]:
    if not state_path.exists():
        state = _initial_state(args)
        atomic_write_json(state_path, state)
        return state
    state = load_json(state_path)
    expected = {
        "target_total": args.target_total,
        "batch_size": args.batch_size,
        "preferred_workers": args.workers,
        "intermediate_workers": args.intermediate_workers,
        "fallback_workers": args.fallback_workers,
        "speculative_buffer_per_room": args.speculative_buffer_per_room,
    }
    for key, value in expected.items():
        if state.get(key) != value:
            raise ValueError(
                f"existing supervisor state has {key}={state.get(key)!r}, "
                f"but command requested {value!r}"
            )
    return state


def run_continuous(args: argparse.Namespace) -> int:
    args.project_root = args.project_root.resolve()
    args.output_root = args.output_root.resolve()
    args.existing_scenes = args.existing_scenes.resolve()
    if args.target_total <= 0 or args.batch_size <= 0:
        raise ValueError("target total and batch size must be positive")
    if not 1 <= args.fallback_workers <= args.intermediate_workers <= args.workers:
        raise ValueError(
            "worker levels must satisfy fallback <= intermediate <= preferred"
        )
    if args.speculative_buffer_per_room < 0:
        raise ValueError("speculative buffer per room must be non-negative")
    if args.api_failure_cooldown_seconds < 30:
        raise ValueError("API failure cooldown must be at least 30 seconds")
    if args.api_failure_cooldown_multiplier < 1:
        raise ValueError("API failure cooldown multiplier must be at least one")
    if args.api_failure_cooldown_max_seconds < args.api_failure_cooldown_seconds:
        raise ValueError("maximum API cooldown must be >= base cooldown")
    if args.max_supervisor_api_failures < 1:
        raise ValueError("max supervisor API failures must be positive")

    args.output_root.mkdir(parents=True, exist_ok=True)
    state_path = args.output_root / STATE_FILENAME
    with BatchLock(
        args.output_root / LOCK_FILENAME,
        recover_stale=args.recover_stale_supervisor_lock,
    ):
        state = _load_or_create_state(args, state_path)
        print(
            f"[continuous] target={args.target_total} accepted="
            f"{count_accepted_scenes(args.output_root)} preferred_workers={args.workers}",
            flush=True,
        )
        while True:
            accepted = count_accepted_scenes(args.output_root)
            _write_state(state_path, state, accepted_count=accepted)
            if accepted >= args.target_total:
                verify_code = _run_verification(
                    args.project_root,
                    args.output_root,
                    args.target_total,
                    args.output_root / "continuous_logs" / "verification",
                )
                final_status = "complete" if verify_code == 0 else "verification_failed"
                _write_state(
                    state_path,
                    state,
                    status=final_status,
                    final_verification_exit_code=verify_code,
                )
                return 0 if verify_code == 0 else 2

            _wait_for_batch_lock(
                args.output_root,
                state_path,
                state,
                args.poll_seconds,
            )

            current_batch = state.get("current_batch_id")
            if current_batch and batch_target_met(args.output_root, current_batch):
                completed = list(state.get("completed_batches", []))
                if current_batch not in completed:
                    completed.append(current_batch)
                accepted = count_accepted_scenes(args.output_root)
                verify_code = _run_verification(
                    args.project_root,
                    args.output_root,
                    accepted,
                    args.output_root / "continuous_logs" / "verification",
                )
                if verify_code != 0:
                    _write_state(
                        state_path,
                        state,
                        status="verification_failed",
                        completed_batches=completed,
                        accepted_count=accepted,
                        last_verification_exit_code=verify_code,
                    )
                    return 2
                _write_state(
                    state_path,
                    state,
                    completed_batches=completed,
                    current_batch_id=None,
                    current_batch_target=None,
                    accepted_count=accepted,
                    status="batch_verified",
                )
                current_batch = None

            if not current_batch:
                index = max(
                    int(state.get("next_batch_index", 1)),
                    next_batch_index(args.output_root, args.batch_prefix),
                )
                current_batch = f"{args.batch_prefix}_{index:04d}"
                remaining = args.target_total - count_accepted_scenes(args.output_root)
                current_target = min(args.batch_size, remaining)
                _write_state(
                    state_path,
                    state,
                    current_batch_id=current_batch,
                    current_batch_target=current_target,
                    next_batch_index=index + 1,
                    status="batch_planned",
                )

            current_target = int(state["current_batch_target"])
            workers = int(state.get("current_workers", args.workers))
            invocations = dict(state.get("batch_invocations", {}))
            invocation = int(invocations.get(current_batch, 0)) + 1
            invocations[current_batch] = invocation
            _write_state(
                state_path,
                state,
                status="batch_running",
                current_workers=workers,
                batch_invocations=invocations,
            )
            code, log_path = _run_batch(
                args, current_batch, current_target, workers, invocation
            )

            if code == 0:
                _write_state(
                    state_path,
                    state,
                    status="batch_finished_pending_verification",
                    current_workers=args.workers,
                    consecutive_supervisor_api_failures=0,
                    last_batch_exit_code=code,
                )
                continue

            if code == 1:
                abandoned = list(state.get("abandoned_batches", []))
                if current_batch not in abandoned:
                    abandoned.append(current_batch)
                _write_state(
                    state_path,
                    state,
                    status="candidate_capacity_exhausted",
                    abandoned_batches=abandoned,
                    current_batch_id=None,
                    current_batch_target=None,
                    current_workers=args.workers,
                    consecutive_supervisor_api_failures=0,
                    last_batch_exit_code=code,
                )
                continue

            if (
                batch_has_configuration_block(args.output_root, current_batch)
                or _tail_contains_fatal_error(log_path)
            ):
                _write_state(
                    state_path,
                    state,
                    status="blocked_configuration",
                    last_batch_exit_code=code,
                    last_batch_log=str(log_path),
                )
                return 2

            if not batch_has_api_runtime_block(args.output_root, current_batch):
                _write_state(
                    state_path,
                    state,
                    status="blocked_unclassified_batch_failure",
                    last_batch_exit_code=code,
                    last_batch_log=str(log_path),
                )
                return 2

            current_workers = int(state.get("current_workers", args.workers))
            if current_workers > args.intermediate_workers:
                fallback = args.intermediate_workers
            else:
                fallback = args.fallback_workers
            failure_count = int(
                state.get("consecutive_supervisor_api_failures", 0)
            ) + 1
            if failure_count >= args.max_supervisor_api_failures:
                _write_state(
                    state_path,
                    state,
                    status="blocked_api_outage",
                    current_workers=fallback,
                    consecutive_supervisor_api_failures=failure_count,
                    last_batch_exit_code=code,
                    last_batch_log=str(log_path),
                )
                print(
                    f"[continuous] API circuit breaker opened after "
                    f"{failure_count} failed batch invocations; manual health "
                    "check required",
                    flush=True,
                )
                return 2
            cooldown_seconds = min(
                args.api_failure_cooldown_max_seconds,
                args.api_failure_cooldown_seconds
                * args.api_failure_cooldown_multiplier
                ** (failure_count - 1),
            )
            _write_state(
                state_path,
                state,
                status="api_cooldown",
                current_workers=fallback,
                consecutive_supervisor_api_failures=failure_count,
                last_batch_exit_code=code,
                last_batch_log=str(log_path),
                resume_after_seconds=cooldown_seconds,
            )
            print(
                f"[continuous] transient API failure; cooling down for "
                f"{cooldown_seconds:.0f}s and retrying with "
                f"{fallback} workers",
                flush=True,
            )
            time.sleep(cooldown_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--target-total", required=True, type=int)
    parser.add_argument("--batch-prefix", default="clean_v2_prod")
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--intermediate-workers", type=int, default=8)
    parser.add_argument("--fallback-workers", type=int, default=4)
    parser.add_argument("--speculative-buffer-per-room", type=int, default=2)
    parser.add_argument("--initial-batch-id")
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--api-failure-cooldown-seconds", type=float, default=120.0)
    parser.add_argument("--api-failure-cooldown-multiplier", type=float, default=2.0)
    parser.add_argument("--api-failure-cooldown-max-seconds", type=float, default=900.0)
    parser.add_argument("--max-supervisor-api-failures", type=int, default=3)
    parser.add_argument("--recover-stale-supervisor-lock", action="store_true")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
    )
    parser.add_argument(
        "--existing-scenes",
        type=Path,
        default=Path("data/clean_layout_production_v1/accepted/scenes"),
    )
    return parser.parse_args()


def main() -> int:
    try:
        args = parse_args()
        if not args.existing_scenes.is_absolute():
            args.existing_scenes = args.project_root / args.existing_scenes
        return run_continuous(args)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"Continuous production error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
