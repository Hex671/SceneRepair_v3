"""Publish a trained checkpoint with a verified SFUR deployment contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from scene_repair_v3.sfur import SFUR_RULE_SCHEMA, SFURRules


DEPLOYMENT_SCHEMA = "scene_repair_v3_furniture_sfur_deployment_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package a checkpoint only after a passing frozen SFUR evaluation"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--sfur-rules", type=Path, required=True)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--semantic-passes", type=int, default=4)
    parser.add_argument("--minimum-sfur", type=float, default=0.90)
    args = parser.parse_args()
    if args.semantic_passes <= 0:
        parser.error("--semantic-passes must be positive")
    if args.input.resolve() == args.output.resolve():
        parser.error("--output must differ from --input")

    rules = SFURRules.load(args.sfur_rules)
    evaluation = _load_json(args.evaluation)
    complete = evaluation.get("metrics", {}).get("complete_model", {}).get("overall", {})
    measured_sfur = float(complete.get("sfur", -1.0))
    measured_hard_pass = float(complete.get("hard_constraint_scene_pass_rate", -1.0))
    if measured_sfur <= args.minimum_sfur:
        raise ValueError(
            f"refusing release: SFUR {measured_sfur:.6f} is not above "
            f"{args.minimum_sfur:.6f}"
        )
    if measured_hard_pass != 1.0:
        raise ValueError(
            f"refusing release: hard-constraint pass rate is {measured_hard_pass:.6f}"
        )
    if int(evaluation.get("model_passes", -1)) != args.semantic_passes:
        raise ValueError("evaluation model_passes does not match the deployment profile")
    if evaluation.get("action_policy") != "dense_delta":
        raise ValueError("evaluation did not use dense_delta")
    if not evaluation.get("functional_refinement"):
        raise ValueError("evaluation did not enable functional refinement")

    source_sha256 = _sha256(args.input)
    expected_source_sha256 = evaluation.get("checkpoint_sha256")
    if expected_source_sha256 and expected_source_sha256 != source_sha256:
        raise ValueError("evaluation checkpoint hash does not match --input")

    payload = torch.load(args.input, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a mapping")
    extra = dict(payload.get("extra") or {})
    if extra.get("label_mode") != "semantic_restore_v1":
        raise ValueError("SFUR deployment requires semantic_restore_v1 training")
    deployment = {
        "schema_version": DEPLOYMENT_SCHEMA,
        "release_id": args.release_id,
        "action_policy": "dense_delta",
        "semantic_passes": args.semantic_passes,
        "functional_refinement": True,
        "requires_functional_scene_contract": True,
        "sfur_rule_schema": SFUR_RULE_SCHEMA,
        "sfur_rules_sha256": _sha256(args.sfur_rules),
        "functional_score_threshold": rules.functional_score_threshold,
    }
    extra.update(
        {
            "release_id": args.release_id,
            "deployment_action_policy": "dense_delta",
            "deployment": deployment,
            "source_checkpoint_sha256": source_sha256,
            "frozen_acceptance": {
                "evaluation_schema": evaluation.get("schema_version"),
                "evaluation_path": str(args.evaluation),
                "split": evaluation.get("split"),
                "seed": evaluation.get("seed"),
                "samples_per_scene": evaluation.get("samples_per_scene"),
                "scene_count": complete.get("scenes"),
                "scene_functional_passes": complete.get("scene_functional_passes"),
                "sfur": measured_sfur,
                "hard_constraint_scene_pass_rate": measured_hard_pass,
            },
        }
    )
    payload["extra"] = extra
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    result = {
        "release_id": args.release_id,
        "output": str(args.output),
        "output_sha256": _sha256(args.output),
        "source_sha256": source_sha256,
        "sfur": measured_sfur,
        "hard_constraint_scene_pass_rate": measured_hard_pass,
        "deployment": deployment,
    }
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
