"""Run the deployed constraint-aware Furniture model on scene JSON files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scene_repair_v3 import FurnitureRepairEngine
from scene_repair_v3.data import load_scene_json


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _input_paths(values: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        if value.is_dir():
            paths.extend(sorted(value.glob("*.json")))
        elif value.is_file():
            paths.append(value)
        else:
            raise FileNotFoundError(value)
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    if not unique:
        raise ValueError("no scene JSON files found")
    stems = [path.stem for path in unique]
    if len(stems) != len(set(stems)):
        raise ValueError("input file stems must be unique")
    return unique


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Constraint-aware Furniture JSON-to-JSON inference"
    )
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--functional-rules", type=Path, required=True)
    parser.add_argument("--compatibility-rules", type=Path, required=True)
    parser.add_argument(
        "--sfur-rules",
        type=Path,
        help=(
            "Enable the checkpoint's SFUR deployment profile. Input scenes must "
            "contain audit.use_clearance_zones and audit.path_targets."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    inputs = _input_paths(args.input)
    scene_dir = args.output_dir / "scenes"
    audit_dir = args.output_dir / "audits"
    destinations = [
        destination
        for source in inputs
        for destination in (
            scene_dir / f"{source.stem}.repaired.json",
            audit_dir / f"{source.stem}.repair_audit.json",
        )
    ]
    manifest_path = args.output_dir / "repair_manifest.json"
    existing = [path for path in (*destinations, manifest_path) if path.exists()]
    if existing and not args.overwrite:
        parser.error(
            f"refusing to overwrite {len(existing)} existing output files; "
            "pass --overwrite explicitly"
        )
    scene_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    engine = FurnitureRepairEngine.load(
        checkpoint=args.checkpoint,
        vocab=args.vocab,
        functional_rules=args.functional_rules,
        compatibility_rules=args.compatibility_rules,
        sfur_rules=args.sfur_rules,
        device=None if args.device == "auto" else args.device,
    )
    rows: list[dict[str, object]] = []
    total_before = 0
    total_final = 0
    sfur_passes = 0
    for start in range(0, len(inputs), args.batch_size):
        batch_paths = inputs[start : start + args.batch_size]
        predictions = engine.repair_scene_jsons(
            [load_scene_json(path) for path in batch_paths]
        )
        for source, prediction in zip(batch_paths, predictions):
            scene_path = scene_dir / f"{source.stem}.repaired.json"
            audit_path = audit_dir / f"{source.stem}.repair_audit.json"
            _write_json(scene_path, prediction.repaired_scene_json)
            _write_json(audit_path, prediction.audit)
            before = prediction.audit["hard_constraints"]["before"]["total"]
            final = prediction.audit["hard_constraints"]["final"]["total"]
            total_before += int(before)
            total_final += int(final)
            sfur = prediction.audit["functional_refinement"]["sfur"]
            if sfur is not None:
                sfur_passes += int(sfur["scene_functional_pass"])
            rows.append(
                {
                    "scene_id": prediction.audit["scene_id"],
                    "source": str(source),
                    "source_sha256": _sha256(source),
                    "repaired_scene": str(scene_path),
                    "audit": str(audit_path),
                    "violations_before": before,
                    "violations_final": final,
                    "changed_count": prediction.audit["changed_count"],
                    "sfur_pass": (
                        sfur["scene_functional_pass"] if sfur is not None else None
                    ),
                }
            )
        print(f"repaired {min(start + args.batch_size, len(inputs))}/{len(inputs)}")

    manifest = {
        "schema_version": "scene_repair_v3_furniture_runtime_manifest_v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": engine.checkpoint_sha256,
        "inference_contract": engine.model.inference_contract(),
        "scene_count": len(rows),
        "hard_violations_before": total_before,
        "hard_violations_final": total_final,
        "all_scenes_feasible": total_final == 0,
        "sfur_mode": args.sfur_rules is not None,
        "sfur_passes": sfur_passes if args.sfur_rules is not None else None,
        "sfur": (
            sfur_passes / len(rows) if args.sfur_rules is not None else None
        ),
        "scenes": rows,
    }
    _write_json(manifest_path, manifest)
    print(json.dumps({key: manifest[key] for key in (
        "scene_count",
        "hard_violations_before",
        "hard_violations_final",
        "all_scenes_feasible",
        "sfur_mode",
        "sfur",
    )}, indent=2))


if __name__ == "__main__":
    main()
