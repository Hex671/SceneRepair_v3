"""Production inference adapter for the constraint-aware Furniture model."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .checkpoint import load_checkpoint
from .contracts import ActionRange, ActionType, ModelConfig
from .data import scene_to_input
from .functional_partners import FrozenFunctionalPartnerRules
from .graph import FurnitureGraphBuilder, SceneInput, collate_graphs
from .model import FurnitureRepairNetwork
from .repair import HardConstraintReport, hard_constraint_report
from .sfur import FunctionalSceneContract, SFURRules
from .vocabulary import FurnitureVocabularies


RUNTIME_PREDICTION_SCHEMA = "scene_repair_v3_furniture_runtime_prediction_v1"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _report_dict(report: HardConstraintReport) -> dict[str, int]:
    return {**asdict(report), "total": report.total}


def _yaw_quaternion_wxyz(yaw_rad: float) -> list[float]:
    return [math.cos(yaw_rad * 0.5), 0.0, 0.0, math.sin(yaw_rad * 0.5)]


def _action_from_delta(dx: float, dy: float, dyaw: float) -> ActionType:
    translation = math.hypot(dx, dy) > 1.0e-6
    rotation = abs(dyaw) > 1.0e-6
    if translation and rotation:
        return ActionType.BOTH
    if translation:
        return ActionType.TRANSLATE
    if rotation:
        return ActionType.ROTATE
    return ActionType.KEEP


@dataclass(frozen=True)
class RuntimeScenePrediction:
    repaired_scene_json: dict[str, Any]
    audit: dict[str, Any]


class FurnitureRepairEngine:
    """Strict checkpoint loader and JSON-to-JSON Furniture repair runtime."""

    def __init__(
        self,
        model: FurnitureRepairNetwork,
        vocabularies: FurnitureVocabularies,
        rules: FrozenFunctionalPartnerRules | None,
        action_range: ActionRange,
        *,
        device: torch.device | str = "cpu",
        checkpoint_path: str | Path | None = None,
        checkpoint_sha256: str | None = None,
        checkpoint_extra: Mapping[str, Any] | None = None,
        sfur_rules: SFURRules | None = None,
        action_policy: str = "hard_constraint_gated",
        semantic_passes: int = 1,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.vocabularies = vocabularies
        self.rules = rules
        self.action_range = action_range
        self.builder = FurnitureGraphBuilder(vocabularies)
        self.checkpoint_path = str(checkpoint_path) if checkpoint_path else None
        self.checkpoint_sha256 = checkpoint_sha256
        self.checkpoint_extra = dict(checkpoint_extra or {})
        self.sfur_rules = sfur_rules
        self.action_policy = action_policy
        self.semantic_passes = semantic_passes
        if semantic_passes <= 0:
            raise ValueError("semantic_passes must be positive")
        if sfur_rules is None and (
            action_policy != "hard_constraint_gated" or semantic_passes != 1
        ):
            raise ValueError(
                "semantic deployment requires SFUR rules and scene functional contracts"
            )

    @classmethod
    def load(
        cls,
        *,
        checkpoint: str | Path,
        vocab: str | Path,
        functional_rules: str | Path,
        compatibility_rules: str | Path,
        sfur_rules: str | Path | None = None,
        device: torch.device | str | None = None,
    ) -> "FurnitureRepairEngine":
        checkpoint_path = Path(checkpoint)
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError("checkpoint payload must be a mapping")
        contract = payload.get("contract")
        if not isinstance(contract, dict) or not isinstance(contract.get("config"), dict):
            raise ValueError("checkpoint is missing the model config contract")
        range_value = payload.get("action_range")
        if not isinstance(range_value, dict):
            raise ValueError("checkpoint is missing the action range contract")
        action_range = ActionRange(
            float(range_value["max_translation_m"]),
            float(range_value["max_yaw_rad"]),
        )
        vocabularies = FurnitureVocabularies.load(vocab)
        rules = FrozenFunctionalPartnerRules.load(
            functional_rules, compatibility_rules
        )
        model = FurnitureRepairNetwork(
            vocabularies, ModelConfig(**contract["config"])
        )
        selected_device = torch.device(
            device
            if device is not None
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
        if selected_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA inference was requested but CUDA is unavailable")
        model = model.to(selected_device)
        extra = load_checkpoint(
            checkpoint_path,
            model,
            action_range,
            map_location=selected_device,
        )
        deployment = extra.get("deployment") or {}
        if not isinstance(deployment, Mapping):
            raise ValueError("checkpoint deployment metadata must be a mapping")
        loaded_sfur_rules = SFURRules.load(sfur_rules) if sfur_rules else None
        action_policy = str(
            deployment.get(
                "action_policy",
                "dense_delta" if loaded_sfur_rules is not None else "hard_constraint_gated",
            )
        )
        semantic_passes = int(
            deployment.get("semantic_passes", 4 if loaded_sfur_rules is not None else 1)
        )
        return cls(
            model,
            vocabularies,
            rules,
            action_range,
            device=selected_device,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=_file_sha256(checkpoint_path),
            checkpoint_extra=extra,
            sfur_rules=loaded_sfur_rules,
            action_policy=action_policy,
            semantic_passes=semantic_passes,
        )

    @torch.no_grad()
    def repair_scene_jsons(
        self, raw_scenes: Sequence[Mapping[str, Any]]
    ) -> tuple[RuntimeScenePrediction, ...]:
        if not raw_scenes:
            return ()
        raw_values = [copy.deepcopy(dict(value)) for value in raw_scenes]
        source_scenes = [
            scene_to_input(value, rules=self.rules) for value in raw_values
        ]
        functional_contracts = None
        if self.sfur_rules is not None:
            for index, raw in enumerate(raw_values):
                audit = raw.get("audit")
                if not isinstance(audit, Mapping) or not all(
                    key in audit for key in ("use_clearance_zones", "path_targets")
                ):
                    raise ValueError(
                        "SFUR inference requires audit.use_clearance_zones and "
                        f"audit.path_targets for scene {raw.get('scene_id', index)!r}"
                    )
            functional_contracts = tuple(
                FunctionalSceneContract.from_raw_scene(value) for value in raw_values
            )
        graph = collate_graphs(
            [self.builder.build(scene) for scene in source_scenes]
        ).to(self.device)
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            complete = self.model.repair_batch(
                graph,
                source_scenes,
                self.action_range,
                action_policy=self.action_policy,
                semantic_passes=self.semantic_passes,
                functional_contracts=functional_contracts,
                functional_origin_scenes=(
                    source_scenes if functional_contracts is not None else None
                ),
                sfur_rules=self.sfur_rules,
            )

        predictions: list[RuntimeScenePrediction] = []
        for graph_id, (raw, source, repaired) in enumerate(
            zip(raw_values, source_scenes, complete.scenes)
        ):
            if not complete.refinement_converged[graph_id]:
                raise RuntimeError(
                    f"constraint refinement did not converge for scene "
                    f"{raw.get('scene_id', graph_id)!r}"
                )
            final_report = hard_constraint_report(repaired)
            if final_report.total:
                raise RuntimeError(
                    f"refusing residual hard constraints for scene "
                    f"{raw.get('scene_id', graph_id)!r}: {final_report}"
                )
            mask = complete.neural_output.furniture_graph_index == graph_id
            neural_actions = complete.neural_action_types[mask].detach().cpu().tolist()
            neural_deltas = complete.neural_delta_m_rad[mask].detach().float().cpu().tolist()
            repaired_raw = copy.deepcopy(raw)
            rows = repaired_raw.get("furniture")
            if not isinstance(rows, list) or len(rows) != len(repaired.furniture):
                raise ValueError("Furniture JSON rows do not match model output")

            object_audit: list[dict[str, Any]] = []
            for index, (row, before, after) in enumerate(
                zip(rows, source.furniture, repaired.furniture)
            ):
                if str(row.get("object_id")) != before.object_id:
                    raise ValueError("Furniture JSON object order changed during inference")
                dx = after.x_m - before.x_m
                dy = after.y_m - before.y_m
                dyaw = math.atan2(
                    math.sin(after.yaw_rad - before.yaw_rad),
                    math.cos(after.yaw_rad - before.yaw_rad),
                )
                if math.hypot(dx, dy) > self.action_range.max_translation_m + 1.0e-6:
                    raise RuntimeError(
                        f"final translation exceeds ActionRange for {before.object_id!r}"
                    )
                if abs(dyaw) > self.action_range.max_yaw_rad + 1.0e-6:
                    raise RuntimeError(
                        f"final yaw exceeds ActionRange for {before.object_id!r}"
                    )
                action = _action_from_delta(dx, dy, dyaw)
                row["x"] = after.x_m
                row["y"] = after.y_m
                row["yaw_rad"] = after.yaw_rad
                row["rotation_wxyz"] = _yaw_quaternion_wxyz(after.yaw_rad)
                object_audit.append(
                    {
                        "object_id": before.object_id,
                        "neural_raw_action": ActionType(neural_actions[index]).name,
                        "neural_raw_delta_m_rad": neural_deltas[index],
                        "final_action": action.name,
                        "final_delta_m_rad": [dx, dy, dyaw],
                        "pose_before": [before.x_m, before.y_m, before.yaw_rad],
                        "pose_after": [after.x_m, after.y_m, after.yaw_rad],
                    }
                )

            audit = {
                "schema_version": RUNTIME_PREDICTION_SCHEMA,
                "scene_id": str(raw.get("scene_id", "unknown")),
                "canonical_input_sha256": _canonical_sha256(raw),
                "checkpoint": {
                    "path": self.checkpoint_path,
                    "sha256": self.checkpoint_sha256,
                    "extra": self.checkpoint_extra,
                },
                "inference_contract": self.model.inference_contract(),
                "deployment": {
                    "mode": "sfur" if self.sfur_rules is not None else "hard_constraints",
                    "action_policy": self.action_policy,
                    "semantic_passes": complete.semantic_passes,
                    "functional_refinement": self.sfur_rules is not None,
                },
                "action_range": self.action_range.to_dict(),
                "device": str(self.device),
                "hard_constraints": {
                    "before": _report_dict(hard_constraint_report(source)),
                    "neural_raw": _report_dict(
                        complete.neural_reports[graph_id]
                    ),
                    "final": _report_dict(final_report),
                },
                "refinement": {
                    "iterations": complete.refinement_iterations[graph_id],
                    "converged": complete.refinement_converged[graph_id],
                    "proposal_scale": complete.refinement_proposal_scales[graph_id],
                },
                "functional_refinement": {
                    "iterations": (
                        complete.functional_refinement_iterations[graph_id]
                        if complete.functional_refinement_iterations else 0
                    ),
                    "converged": (
                        complete.functional_refinement_converged[graph_id]
                        if complete.functional_refinement_converged else None
                    ),
                    "contract": complete.functional_refinement_contract,
                    "sfur": (
                        complete.functional_reports[graph_id].to_dict()
                        if complete.functional_reports else None
                    ),
                },
                "changed_count": sum(
                    row["final_action"] != ActionType.KEEP.name
                    for row in object_audit
                ),
                "objects": object_audit,
            }
            predictions.append(RuntimeScenePrediction(repaired_raw, audit))
        return tuple(predictions)
