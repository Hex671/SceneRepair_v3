from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

from scene_repair_v3 import (
    FunctionalLayoutEvaluator,
    FunctionalConstraintRefinementLayer,
    FunctionalPartnerInput,
    FunctionalSceneContract,
    FurnitureInput,
    OpeningInput,
    RoomInput,
    SFURRules,
    ActionRange,
    SceneInput,
)


def _rules() -> SFURRules:
    return SFURRules.load(Path(__file__).parents[1] / "configs" / "sfur_rules_v1.json")


def _scene_and_contract() -> tuple[SceneInput, FunctionalSceneContract]:
    scene = SceneInput(
        room=RoomInput("dining_room", 6.0, 5.0),
        openings=(
            OpeningInput("door_0", "door", (-2.0, -2.1, 1.0), (0.9, 0.8, 2.0), (0.0, 1.0)),
        ),
        furniture=(
            FurnitureInput("table_0", 0.0, 0.0, 0.0, 1.4, 0.8, 0.8, "table"),
            FurnitureInput("chair_0", 0.0, -1.0, math.pi, 0.5, 0.5, 0.9, "chair"),
        ),
        functional_partners=(
            FunctionalPartnerInput("chair_0", "table_0", "none", "none"),
        ),
    )
    raw = {
        "furniture": [
            {"object_id": "table_0", "x": 0.0, "y": 0.0, "yaw_rad": 0.0},
            {"object_id": "chair_0", "x": 0.0, "y": -1.0, "yaw_rad": math.pi},
        ],
        "audit": {
            "use_clearance_zones": [
                {
                    "zone_id": "chair_0_rear_pullout",
                    "owner_id": "chair_0",
                    "center_xy_m": [0.0, -1.65],
                    "size_xy_m": [0.5, 0.5],
                    "yaw_rad": math.pi,
                    "allowed_overlap_object_ids": [],
                    "validity": True,
                }
            ],
            "path_targets": [
                {"target_id": "chair_0", "goal_xy_m": [0.0, -1.65]}
            ],
        },
    }
    return scene, FunctionalSceneContract.from_raw_scene(raw)


def test_sfur_accepts_functional_scene() -> None:
    scene, contract = _scene_and_contract()
    report = FunctionalLayoutEvaluator(_rules()).evaluate(scene, contract)
    assert report.scene_pass
    assert report.functional_score == 1.0
    assert report.relation_checks == 1
    assert report.clearance_checks == 1
    assert report.path_checks == 1


def test_sfur_rejects_wrong_functional_orientation_without_hard_violation() -> None:
    scene, contract = _scene_and_contract()
    chair = replace(scene.furniture[1], yaw_rad=0.0)
    report = FunctionalLayoutEvaluator(_rules()).evaluate(
        replace(scene, furniture=(scene.furniture[0], chair)), contract
    )
    assert report.hard_constraints_pass
    assert not report.scene_pass
    assert report.relation_pass_rate == 0.0
    assert any(value.kind == "functional_relation" for value in report.violations)


def test_sfur_rejects_blocked_use_zone() -> None:
    scene, contract = _scene_and_contract()
    blocker = FurnitureInput("cabinet_0", 0.0, -1.65, 0.0, 0.4, 0.3, 0.8, "cabinet")
    report = FunctionalLayoutEvaluator(_rules()).evaluate(
        replace(scene, furniture=(*scene.furniture, blocker)), contract
    )
    assert report.hard_constraints_pass
    assert not report.scene_pass
    assert report.clearance_pass_rate == 0.0
    assert any(value.kind == "use_clearance" for value in report.violations)


def test_functional_refinement_fixes_relation_without_clean_pose() -> None:
    scene, contract = _scene_and_contract()
    wrong_chair = replace(scene.furniture[1], x_m=1.5, y_m=-1.4, yaw_rad=0.0)
    proposed = replace(scene, furniture=(scene.furniture[0], wrong_chair))
    result = FunctionalConstraintRefinementLayer()(
        proposed,
        proposed,
        contract,
        _rules(),
        ActionRange(3.0, math.pi),
    )
    assert result.converged
    assert result.report.scene_pass
    assert result.iterations > 0
