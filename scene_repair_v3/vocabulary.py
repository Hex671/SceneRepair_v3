from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


UNK = "UNK"


@dataclass(frozen=True)
class Vocabulary:
    """Frozen, collision-free categorical vocabulary with an explicit UNK row."""

    values: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.values or self.values[0] != UNK:
            raise ValueError("vocabulary index 0 must be UNK")
        if any(not value or value.strip() != value for value in self.values):
            raise ValueError("vocabulary values must be non-empty and trimmed")
        if len(set(self.values)) != len(self.values):
            raise ValueError("vocabulary values must be unique")

    @classmethod
    def from_values(cls, values: Iterable[str]) -> "Vocabulary":
        normalized = tuple(str(value).strip() for value in values)
        if not normalized or normalized[0] != UNK:
            normalized = (UNK, *tuple(value for value in normalized if value != UNK))
        return cls(normalized)

    def __len__(self) -> int:
        return len(self.values)

    @property
    def index(self) -> dict[str, int]:
        return {value: index for index, value in enumerate(self.values)}

    def encode(self, value: str | None, *, strict: bool = False) -> int:
        if value is None:
            return 0
        key = str(value).strip()
        result = self.index.get(key)
        if result is not None:
            return result
        if strict:
            raise ValueError(f"value {key!r} is outside the frozen vocabulary")
        return 0

    def multihot(self, values: Iterable[str], *, strict: bool = True) -> list[float]:
        result = [0.0] * len(self.values)
        for value in values:
            index = self.encode(value, strict=strict)
            if index == 0 and value != UNK:
                continue
            result[index] = 1.0
        return result


@dataclass(frozen=True)
class FurnitureVocabularies:
    schema_version: str
    room_types: Vocabulary
    opening_kinds: Vocabulary
    categories: Vocabulary
    families: Vocabulary
    functions: Vocabulary
    anchor_directions: Vocabulary
    anchor_modes: Vocabulary
    orientation_modes: Vocabulary
    target_strategies: Vocabulary

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "FurnitureVocabularies":
        fields = (
            "room_types",
            "opening_kinds",
            "categories",
            "families",
            "functions",
            "anchor_directions",
            "anchor_modes",
            "orientation_modes",
            "target_strategies",
        )
        values: dict[str, Vocabulary] = {}
        for field in fields:
            payload = raw.get(field)
            if not isinstance(payload, list):
                raise ValueError(f"vocabulary field {field!r} must be a list")
            values[field] = Vocabulary.from_values(str(item) for item in payload)
        return cls(str(raw.get("schema_version") or ""), **values)

    @classmethod
    def load(cls, path: str | Path) -> "FurnitureVocabularies":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("vocabulary document must be an object")
        return cls.from_dict(raw)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "room_types": list(self.room_types.values),
            "opening_kinds": list(self.opening_kinds.values),
            "categories": list(self.categories.values),
            "families": list(self.families.values),
            "functions": list(self.functions.values),
            "anchor_directions": list(self.anchor_directions.values),
            "anchor_modes": list(self.anchor_modes.values),
            "orientation_modes": list(self.orientation_modes.values),
            "target_strategies": list(self.target_strategies.values),
        }

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.to_dict(), ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
