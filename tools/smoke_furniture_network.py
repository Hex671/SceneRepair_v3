from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from scene_repair_v3 import (
    ActionRange,
    FunctionalPartnerInput,
    FurnitureGraphBuilder,
    FurnitureInput,
    FurnitureRepairNetwork,
    FurnitureVocabularies,
    OpeningInput,
    RoomInput,
    SceneInput,
    decode_actions,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Furniture v1 smoke forward pass")
    parser.add_argument(
        "--vocabulary",
        type=Path,
        default=Path("configs/furniture_vocab_bootstrap_v1.json"),
    )
    parser.add_argument("--max-translation-m", type=float, required=True)
    parser.add_argument("--max-yaw-deg", type=float, required=True)
    args = parser.parse_args()

    vocabularies = FurnitureVocabularies.load(args.vocabulary)
    scene = SceneInput(
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
                "chair_0", "bed_0", "facing", "face_target", (0.5, 1.5)
            ),
        ),
    )
    graph = FurnitureGraphBuilder(vocabularies).build(scene)
    model = FurnitureRepairNetwork(vocabularies).eval()
    with torch.no_grad():
        output = model(graph)
        decoded = decode_actions(
            output,
            ActionRange(
                args.max_translation_m,
                math.radians(args.max_yaw_deg),
            ),
        )
    payload = {
        "nodes": graph.num_nodes,
        "furniture": len(graph.furniture_indices),
        "action_type_logits_shape": list(output.action_type_logits.shape),
        "primary_delta_shape": list(output.primary_delta_normalized.shape),
        "decoded_action_types": decoded.action_types.tolist(),
        "decoded_delta_m_rad": decoded.delta_m_rad.tolist(),
        "checkpoint_contract": model.checkpoint_contract(),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
