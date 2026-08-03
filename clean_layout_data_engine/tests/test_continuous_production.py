from pathlib import Path

from clean_layout_data_engine.continuous_production import (
    _initial_state,
    batch_has_api_runtime_block,
    batch_has_configuration_block,
    batch_target_met,
    count_accepted_scenes,
    next_batch_index,
)


def test_initial_state_records_all_adaptive_worker_levels(tmp_path: Path) -> None:
    args = type(
        "Args",
        (),
        {
            "initial_batch_id": None,
            "output_root": tmp_path,
            "target_total": 600,
            "batch_size": 48,
            "workers": 16,
            "intermediate_workers": 8,
            "fallback_workers": 4,
            "speculative_buffer_per_room": 2,
            "batch_prefix": "clean_v2_prod",
        },
    )()

    state = _initial_state(args)

    assert state["preferred_workers"] == 16
    assert state["intermediate_workers"] == 8
    assert state["fallback_workers"] == 4
    assert state["speculative_buffer_per_room"] == 2


def test_count_accepted_scenes_counts_only_scene_json(tmp_path: Path) -> None:
    scenes = tmp_path / "accepted" / "scenes"
    scenes.mkdir(parents=True)
    (scenes / "a.json").write_text("{}", encoding="utf-8")
    (scenes / "b.json").write_text("{}", encoding="utf-8")
    (scenes / "note.txt").write_text("ignore", encoding="utf-8")

    assert count_accepted_scenes(tmp_path) == 2


def test_next_batch_index_uses_only_matching_prefix(tmp_path: Path) -> None:
    batches = tmp_path / "batches"
    (batches / "clean_v2_prod_0004").mkdir(parents=True)
    (batches / "clean_v2_prod_0012").mkdir()
    (batches / "clean_v1_prod_0099").mkdir()
    (batches / "clean_v2_prod_bad").mkdir()

    assert next_batch_index(tmp_path, "clean_v2_prod") == 13


def test_batch_status_helpers(tmp_path: Path) -> None:
    batch = tmp_path / "batches" / "clean_v2_prod_0004"
    batch.mkdir(parents=True)
    (batch / "batch_status.json").write_text(
        '{"target_met": false, "state_counts": {"blocked_configuration": 1}}',
        encoding="utf-8",
    )

    assert not batch_target_met(tmp_path, "clean_v2_prod_0004")
    assert batch_has_configuration_block(tmp_path, "clean_v2_prod_0004")
    assert not batch_has_api_runtime_block(tmp_path, "clean_v2_prod_0004")

    (batch / "batch_status.json").write_text(
        '{"target_met": false, "state_counts": {"blocked_api_runtime": 1}}',
        encoding="utf-8",
    )
    assert batch_has_api_runtime_block(tmp_path, "clean_v2_prod_0004")
