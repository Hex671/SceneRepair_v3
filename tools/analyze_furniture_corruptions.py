from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from torch.utils.data import DataLoader

from scene_repair_v3 import FurnitureVocabularies, FrozenFunctionalPartnerRules
from scene_repair_v3.dataset import FurnitureCorruptionDataset
from scene_repair_v3.data import CorruptionConfig
from scene_repair_v3.model import _furniture_constraint_summary


def _collate_counts(samples):
    actions = Counter()
    sampled_actions = Counter()
    target_k = Counter()
    sampled_k = Counter()
    violation_objects = 0
    violation_labeled_keep = 0
    non_violation_labeled_change = 0
    for sample in samples:
        actions.update(sample.action_types)
        sampled_actions.update(sample.sampled_action_types)
        target_k[sample.changed_count] += 1
        sampled_k[sample.sampled_changed_count] += 1
        summary = _furniture_constraint_summary(sample.graph)
        violation = (
            (summary[:, :4].min(dim=-1).values < 0.0)
            | (summary[:, 4] > 0.0)
            | (summary[:, 8] > 0.0)
        )
        target_change = sample.targets.action_types != 0
        violation_objects += int(violation.sum())
        violation_labeled_keep += int((violation & ~target_change).sum())
        non_violation_labeled_change += int((~violation & target_change).sum())
    return (
        actions, sampled_actions, target_k, sampled_k,
        violation_objects, violation_labeled_keep, non_violation_labeled_change,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit natural action and K distributions")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--functional-rules", type=Path, required=True)
    parser.add_argument("--compatibility-rules", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--samples-per-scene", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--label-mode",
        choices=("observable_v3", "semantic_v2", "legacy_random"),
        default="observable_v3",
    )
    args = parser.parse_args()

    vocab = FurnitureVocabularies.load(args.vocab)
    rules = FrozenFunctionalPartnerRules.load(args.functional_rules, args.compatibility_rules)
    dataset = FurnitureCorruptionDataset(
        args.manifest, split=args.split, vocabularies=vocab, rules=rules,
        base_seed=args.seed, samples_per_scene=args.samples_per_scene,
        corruption=CorruptionConfig(label_mode=args.label_mode),
    )
    actions = Counter()
    sampled_actions = Counter()
    target_k = Counter()
    sampled_k = Counter()
    violation_objects = 0
    violation_labeled_keep = 0
    non_violation_labeled_change = 0
    loader = DataLoader(
        dataset, batch_size=64, shuffle=False, num_workers=args.workers,
        persistent_workers=False, collate_fn=_collate_counts,
    )
    for (
        batch_actions, batch_sampled_actions, batch_target_k, batch_sampled_k,
        batch_violation_objects, batch_violation_labeled_keep,
        batch_non_violation_labeled_change,
    ) in loader:
        actions.update(batch_actions)
        sampled_actions.update(batch_sampled_actions)
        target_k.update(batch_target_k)
        sampled_k.update(batch_sampled_k)
        violation_objects += batch_violation_objects
        violation_labeled_keep += batch_violation_labeled_keep
        non_violation_labeled_change += batch_non_violation_labeled_change
    action_names = ("KEEP", "TRANSLATE", "ROTATE", "BOTH")
    target_non_keep = sum(actions[index] for index in (1, 2, 3))
    sampled_non_keep = sum(sampled_actions[index] for index in (1, 2, 3))
    report = {
        "schema_version": "scene_repair_v3_corruption_audit_v1",
        "split": args.split, "seed": args.seed,
        "samples_per_scene": args.samples_per_scene,
        "scene_samples": len(dataset),
        "target_action_counts": {action_names[i]: actions[i] for i in range(4)},
        "sampled_action_counts": {action_names[i]: sampled_actions[i] for i in range(4)},
        "target_k_histogram": {str(k): v for k, v in sorted(target_k.items())},
        "sampled_k_histogram": {str(k): v for k, v in sorted(sampled_k.items())},
        "semantic_retention_rate": target_non_keep / max(sampled_non_keep, 1),
        "violation_objects": violation_objects,
        "violation_labeled_keep": violation_labeled_keep,
        "non_violation_labeled_change": non_violation_labeled_change,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
