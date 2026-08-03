from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from scene_repair_v3 import FrozenFunctionalPartnerRules, FurnitureGraphBuilder, FurnitureVocabularies
from scene_repair_v3.data import load_scene_json, scene_to_input
from scene_repair_v3.dataset import freeze_scene_manifest


def _vocabulary(scenes_dir: Path, bootstrap_path: Path) -> dict[str, object]:
    bootstrap = json.loads(bootstrap_path.read_text(encoding="utf-8"))
    values = {key: {"UNK"} for key in (
        "room_types", "opening_kinds", "categories", "families", "functions"
    )}
    for path in sorted(scenes_dir.glob("*.json")):
        raw = load_scene_json(path)
        values["room_types"].add(str(raw["room_type"]))
        for opening in raw.get("openings") or ():
            values["opening_kinds"].add(str(opening.get("kind") or "UNK"))
        for furniture in raw.get("furniture") or ():
            values["categories"].add(str(furniture.get("category") or "UNK"))
            values["families"].add(str(furniture.get("family") or "UNK"))
            values["functions"].update(str(v) for v in (furniture.get("functions") or ()))
    result: dict[str, object] = {"schema_version": "furniture_vocab_clean512_v1"}
    for key, entries in values.items():
        result[key] = ["UNK", *sorted(value for value in entries if value != "UNK")]
    for key in ("anchor_directions", "anchor_modes", "orientation_modes", "target_strategies"):
        result[key] = bootstrap[key]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze and validate the clean512 Furniture training source")
    parser.add_argument("--scenes-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-vocab", type=Path, default=Path("configs/furniture_vocab_bootstrap_v1.json"))
    parser.add_argument("--functional-rules", type=Path, default=Path("configs/functional_partner_rules_hssd_v1.json"))
    parser.add_argument("--compatibility-rules", type=Path, default=Path("configs/functional_partner_category_compatibility_v1.json"))
    parser.add_argument("--expected-scenes", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260801)
    args = parser.parse_args()

    paths = sorted(args.scenes_dir.glob("*.json"))
    if len(paths) != args.expected_scenes:
        raise SystemExit(f"expected exactly {args.expected_scenes} scenes, found {len(paths)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    vocab_payload = _vocabulary(args.scenes_dir, args.bootstrap_vocab)
    vocab_path = args.output_dir / "furniture_vocab_clean512_v1.json"
    vocab_path.write_text(json.dumps(vocab_payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    vocab = FurnitureVocabularies.from_dict(vocab_payload)
    rules = FrozenFunctionalPartnerRules.load(args.functional_rules, args.compatibility_rules)
    builder = FurnitureGraphBuilder(vocab)

    room_types: Counter[str] = Counter()
    furniture_counts: Counter[int] = Counter()
    action_counts = Counter()
    functional_edges = 0
    for path in paths:
        raw = load_scene_json(path)
        scene = scene_to_input(raw, rules=rules)
        graph = builder.build(scene)
        room_types[scene.room.room_type] += 1
        furniture_counts[len(scene.furniture)] += 1
        functional_edges += graph.edges[2].edge_index.shape[1]
        action_counts.update(f.category for f in scene.furniture)

    manifest = freeze_scene_manifest(
        args.scenes_dir, args.output_dir / "clean512_manifest_v1.json", seed=args.seed
    )
    report = {
        "schema_version": "scene_repair_v3_training_preparation_report_v1",
        "scene_count": len(paths),
        "splits": manifest["splits"],
        "room_types": dict(sorted(room_types.items())),
        "furniture_count_histogram": {str(k): v for k, v in sorted(furniture_counts.items())},
        "category_counts": dict(sorted(action_counts.items())),
        "functional_partner_edges": functional_edges,
        "vocabulary_sha256": vocab.sha256,
        "functional_rules_sha256": rules.rules_sha256,
        "validated_graphs": len(paths),
    }
    (args.output_dir / "preparation_report.json").write_text(
        json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
