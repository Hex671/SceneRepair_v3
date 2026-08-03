from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from scene_repair_v3 import (
    ActionRange,
    FurnitureRepairEngine,
    FurnitureRepairNetwork,
    ModelConfig,
    RUNTIME_PREDICTION_SCHEMA,
    SFURRules,
)
from scene_repair_v3.checkpoint import save_checkpoint
from scene_repair_v3.data import scene_to_input
from scene_repair_v3.repair import hard_constraint_report


def _config() -> ModelConfig:
    return ModelConfig(
        hidden_dim=32,
        heads=4,
        edge_dim=16,
        geometry_dim=24,
        category_dim=16,
        behavior_dim=24,
        dropout=0.0,
    )


def _raw_scene(scene_id: str = "runtime_scene") -> dict:
    return {
        "schema_version": "scene_repair_v3_clean_furniture_scene_v1",
        "scene_id": scene_id,
        "stage": "furniture",
        "coordinate_frame": "room_local_z_up",
        "quaternion_order": "wxyz",
        "yaw_range": "[-pi,pi)",
        "room_type": "bedroom",
        "length": 5.0,
        "width": 4.0,
        "openings": [],
        "furniture": [
            {
                "object_id": "chair_0",
                "hssd_id": None,
                "category": "chair",
                "family": "seating",
                "functions": ["sit"],
                "bbox": {"width": 0.8, "depth": 0.8, "height": 0.9},
                "x": 2.8,
                "y": 0.0,
                "z": 0.0,
                "yaw_rad": 0.0,
                "rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
                "movable": True,
                "validity": {
                    "bbox": True,
                    "transform": True,
                    "family": True,
                    "functions": True,
                },
                "custom_field": "preserved",
            },
            {
                "object_id": "bed_0",
                "hssd_id": None,
                "category": "bed",
                "family": "sleep_surface",
                "functions": ["sleep"],
                "bbox": {"width": 1.8, "depth": 1.2, "height": 0.7},
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "yaw_rad": 0.0,
                "rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
                "movable": True,
                "validity": {
                    "bbox": True,
                    "transform": True,
                    "family": True,
                    "functions": True,
                },
            },
        ],
    }


def _zero_model(vocabularies) -> FurnitureRepairNetwork:
    model = FurnitureRepairNetwork(vocabularies, _config()).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    return model


def test_runtime_engine_repairs_json_and_preserves_source(
    vocabularies,
) -> None:
    raw = _raw_scene()
    engine = FurnitureRepairEngine(
        _zero_model(vocabularies),
        vocabularies,
        None,
        ActionRange(3.0, math.pi),
    )

    prediction = engine.repair_scene_jsons([raw])[0]

    assert raw["furniture"][0]["x"] == 2.8
    assert prediction.repaired_scene_json["furniture"][0]["custom_field"] == "preserved"
    repaired = scene_to_input(prediction.repaired_scene_json)
    assert hard_constraint_report(repaired).total == 0
    audit = prediction.audit
    assert audit["schema_version"] == RUNTIME_PREDICTION_SCHEMA
    assert audit["hard_constraints"]["before"]["total"] > 0
    assert audit["hard_constraints"]["final"]["total"] == 0
    assert audit["refinement"]["converged"]
    assert audit["changed_count"] >= 1
    for row in prediction.repaired_scene_json["furniture"]:
        yaw = row["yaw_rad"]
        assert row["rotation_wxyz"] == pytest.approx(
            [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
        )
    assert engine.repair_scene_jsons([]) == ()


def test_runtime_engine_rejects_repairs_outside_action_range(
    vocabularies,
) -> None:
    raw = _raw_scene("out_of_range")
    raw["furniture"][0]["x"] = 6.0
    engine = FurnitureRepairEngine(
        _zero_model(vocabularies),
        vocabularies,
        None,
        ActionRange(3.0, math.pi),
    )

    with pytest.raises(RuntimeError, match="exceeds ActionRange"):
        engine.repair_scene_jsons([raw])


def test_sfur_runtime_requires_and_audits_functional_contract(
    vocabularies,
) -> None:
    root = Path(__file__).parents[1]
    rules = SFURRules.load(root / "configs/sfur_rules_v1.json")
    engine = FurnitureRepairEngine(
        _zero_model(vocabularies),
        vocabularies,
        None,
        ActionRange(3.0, math.pi),
        sfur_rules=rules,
        action_policy="dense_delta",
        semantic_passes=2,
    )
    raw = _raw_scene("sfur_runtime")

    with pytest.raises(ValueError, match="SFUR inference requires"):
        engine.repair_scene_jsons([raw])

    raw["audit"] = {
        "use_clearance_zones": [],
        "path_targets": [],
    }
    prediction = engine.repair_scene_jsons([raw])[0]

    assert prediction.audit["deployment"] == {
        "mode": "sfur",
        "action_policy": "dense_delta",
        "semantic_passes": 2,
        "functional_refinement": True,
    }
    functional = prediction.audit["functional_refinement"]
    assert isinstance(functional["iterations"], int)
    assert isinstance(functional["converged"], bool)
    assert functional["contract"]["schema_version"] == (
        "furniture_functional_refinement_v1"
    )
    assert isinstance(functional["sfur"]["scene_functional_pass"], bool)
    assert functional["sfur"]["hard_constraints_pass"]


def test_inference_cli_loads_checkpoint_and_writes_audited_outputs(
    tmp_path: Path,
    vocabularies,
) -> None:
    root = Path(__file__).parents[1]
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(
        checkpoint,
        _zero_model(vocabularies),
        ActionRange(3.0, math.pi),
        extra={"epoch": 7},
    )
    source = tmp_path / "input_scene.json"
    source.write_text(json.dumps(_raw_scene()), encoding="utf-8")
    output = tmp_path / "output"
    command = [
        sys.executable,
        "-m",
        "tools.infer_furniture",
        "--input",
        str(source),
        "--output-dir",
        str(output),
        "--checkpoint",
        str(checkpoint),
        "--vocab",
        str(root / "configs/furniture_vocab_bootstrap_v1.json"),
        "--functional-rules",
        str(root / "configs/functional_partner_rules_hssd_v1.json"),
        "--compatibility-rules",
        str(root / "configs/functional_partner_category_compatibility_v1.json"),
        "--device",
        "cpu",
    ]
    completed = subprocess.run(
        command,
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    manifest = json.loads((output / "repair_manifest.json").read_text())
    assert manifest["scene_count"] == 1
    assert manifest["hard_violations_before"] > 0
    assert manifest["hard_violations_final"] == 0
    assert manifest["all_scenes_feasible"]
    assert (output / "scenes/input_scene.repaired.json").is_file()
    assert (output / "audits/input_scene.repair_audit.json").is_file()

    refused = subprocess.run(
        command,
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert refused.returncode != 0
    assert "refusing to overwrite" in refused.stderr
