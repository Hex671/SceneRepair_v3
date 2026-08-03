from __future__ import annotations

import math
from pathlib import Path

import pytest

from scene_repair_v3 import (
    FunctionalPartnerInput,
    FurnitureInput,
    FurnitureVocabularies,
    OpeningInput,
    RoomInput,
    SceneInput,
)


@pytest.fixture(scope="session")
def vocabularies() -> FurnitureVocabularies:
    path = Path(__file__).parents[1] / "configs" / "furniture_vocab_bootstrap_v1.json"
    return FurnitureVocabularies.load(path)


@pytest.fixture
def sample_scene() -> SceneInput:
    return SceneInput(
        room=RoomInput("bedroom", 5.0, 4.0),
        openings=(
            OpeningInput(
                "door_0",
                "door",
                (0.0, -1.7, 1.0),
                (1.0, 0.8, 2.0),
                (0.0, 1.0),
            ),
        ),
        furniture=(
            FurnitureInput(
                "bed_0",
                -1.0,
                0.0,
                0.0,
                2.0,
                1.6,
                0.6,
                "bed",
                family="sleep_surface",
                functions=("sleep",),
                family_valid=True,
                functions_valid=True,
                wall_anchor_direction="west",
                wall_anchor_mode="near_wall",
                wall_anchor_valid=True,
            ),
            FurnitureInput(
                "chair_0",
                1.0,
                0.2,
                math.pi,
                0.6,
                0.6,
                0.9,
                "chair",
                family="seating",
                functions=("sit",),
                family_valid=True,
                functions_valid=True,
            ),
        ),
        functional_partners=(
            FunctionalPartnerInput(
                "chair_0",
                "bed_0",
                "facing",
                "face_target",
                (0.5, 1.5),
            ),
        ),
    )
