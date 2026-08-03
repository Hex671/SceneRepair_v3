"""Scene-level split and online Furniture corruption dataset."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from .data import CorruptionConfig, corrupt_scene, load_scene_json, collate_corrupted_samples
from .functional_partners import FrozenFunctionalPartnerRules
from .graph import FurnitureGraphBuilder
from .vocabulary import FurnitureVocabularies


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_scene_manifest(
    scenes_dir: str | Path,
    output_path: str | Path,
    *,
    seed: int = 20260801,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
) -> dict[str, Any]:
    paths = sorted(Path(scenes_dir).glob("*.json"))
    if not paths:
        raise ValueError(f"no scene JSON files found in {scenes_dir}")
    rows = [{"scene_id": load_scene_json(path).get("scene_id", path.stem),
             "path": path.name, "sha256": file_sha256(path)} for path in paths]
    rng = random.Random(seed)
    rng.shuffle(rows)
    train_end = int(len(rows) * train_fraction)
    val_end = train_end + int(len(rows) * val_fraction)
    for index, row in enumerate(rows):
        row["split"] = "train" if index < train_end else "val" if index < val_end else "test"
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_directory = Path(scenes_dir).resolve()
    source_directory_text = os.path.relpath(source_directory, destination.parent.resolve())
    payload = {
        "schema_version": "scene_repair_v3_clean_scene_manifest_v1",
        "seed": seed,
        "source_directory": source_directory_text,
        "scene_count": len(rows),
        "splits": {split: sum(row["split"] == split for row in rows) for split in ("train", "val", "test")},
        "scenes": rows,
    }
    destination.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return payload


class FurnitureCorruptionDataset(Dataset):
    """One fresh full-scene corruption per index and epoch seed."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        split: str,
        vocabularies: FurnitureVocabularies,
        rules: FrozenFunctionalPartnerRules | None,
        rules_path: str | Path | None = None,
        corruption: CorruptionConfig | None = None,
        base_seed: int = 20260801,
        epoch: int = 0,
        samples_per_scene: int = 1,
    ) -> None:
        payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != "scene_repair_v3_clean_scene_manifest_v1":
            raise ValueError("unsupported clean scene manifest")
        self.manifest_path = Path(manifest_path)
        self.scene_root = Path(payload["source_directory"])
        if not self.scene_root.is_absolute():
            self.scene_root = (self.manifest_path.parent / self.scene_root).resolve()
        self.rows = tuple(row for row in payload["scenes"] if row["split"] == split)
        if not self.rows:
            raise ValueError(f"manifest split {split!r} is empty")
        self.vocabularies = vocabularies
        self.rules = rules
        self.rules_path = str(rules_path) if rules_path else None
        self.builder = FurnitureGraphBuilder(vocabularies)
        self.corruption = corruption or CorruptionConfig()
        self.base_seed = int(base_seed)
        self.epoch = int(epoch)
        self.samples_per_scene = int(samples_per_scene)
        if self.samples_per_scene <= 0:
            raise ValueError("samples_per_scene must be positive")
        for row in self.rows:
            path = self.scene_root / row["path"]
            if file_sha256(path) != row["sha256"]:
                raise RuntimeError(f"scene changed after manifest freeze: {path}")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.rows) * self.samples_per_scene

    def __getitem__(self, index: int):
        scene_index = index % len(self.rows)
        corruption_index = index // len(self.rows)
        row = self.rows[scene_index]
        path = self.scene_root / row["path"]
        # A stable per-sample seed makes validation repeatable while every epoch
        # receives a new corruption for training.
        seed = (
            self.base_seed
            + self.epoch * 1_000_003
            + scene_index * 97_003
            + corruption_index * 7_919
        ) & ((1 << 63) - 1)
        raw = load_scene_json(path)
        return corrupt_scene(raw, seed=seed, builder=self.builder, rules=self.rules, config=self.corruption)


def make_collate_fn():
    return collate_corrupted_samples
