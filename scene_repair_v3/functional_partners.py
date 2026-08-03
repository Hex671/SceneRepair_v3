from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

from .graph import FunctionalPartnerInput, FurnitureInput, SceneInput


FUNCTIONAL_PARTNER_RULE_SCHEMA = "hssd_functional_partner_rules_v1"
FUNCTIONAL_PARTNER_COMPATIBILITY_SCHEMA = (
    "hssd_functional_partner_category_compatibility_v1"
)
_WORDNET_SUFFIX = re.compile(r"\.n\.\d+$")
_CATEGORY_SEPARATOR = re.compile(r"[^a-z0-9]+")


def normalize_hssd_id(value: str | None) -> str | None:
    if value is None:
        return None
    result = str(value).strip().lower()
    if result.startswith("hssd:"):
        result = result[5:]
    return result or None


def normalize_partner_category(value: str) -> str:
    result = _WORDNET_SUFFIX.sub("", str(value).strip().lower())
    return _CATEGORY_SEPARATOR.sub("_", result).strip("_")


@dataclass(frozen=True)
class FrozenFunctionalPartnerRule:
    hssd_id: str
    source_category: str | None
    partner_categories: tuple[str, ...]

    @property
    def partner_category_keys(self) -> frozenset[str]:
        return frozenset(normalize_partner_category(v) for v in self.partner_categories)


@dataclass(frozen=True)
class FrozenTargetCategoryCompatibilityRule:
    rule_id: str
    annotated_partner_category: str
    target_category: str
    required_family: str
    required_functions: tuple[str, ...]

    def matches(self, partner_category: str, target: FurnitureInput) -> bool:
        return (
            normalize_partner_category(partner_category)
            == normalize_partner_category(self.annotated_partner_category)
            and normalize_partner_category(target.category)
            == normalize_partner_category(self.target_category)
            and target.family_valid
            and target.functions_valid
            and target.family == self.required_family
            and set(self.required_functions).issubset(target.functions)
        )


@dataclass(frozen=True)
class FrozenFunctionalPartnerMatch:
    source_id: str
    target_id: str
    source_hssd_id: str
    annotated_partner_category: str
    category_match_mode: str
    compatibility_rule_id: str | None


class FrozenFunctionalPartnerRules:
    """Read-only per-HSSD-ID partner annotations used before graph encoding."""

    def __init__(
        self,
        *,
        source_sha256: str,
        rules_sha256: str,
        asset_count_in_source: int,
        rules: Mapping[str, FrozenFunctionalPartnerRule],
        compatibility_rules: Sequence[FrozenTargetCategoryCompatibilityRule] = (),
        compatibility_rules_sha256: str | None = None,
    ) -> None:
        self.source_sha256 = source_sha256
        self.rules_sha256 = rules_sha256
        self.asset_count_in_source = int(asset_count_in_source)
        self._rules = MappingProxyType(dict(rules))
        self._compatibility_rules = tuple(compatibility_rules)
        self.compatibility_rules_sha256 = compatibility_rules_sha256
        self.compatibility_schema_version = (
            FUNCTIONAL_PARTNER_COMPATIBILITY_SCHEMA
            if compatibility_rules_sha256 is not None
            else None
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        compatibility_path: str | Path | None = None,
    ) -> "FrozenFunctionalPartnerRules":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("functional partner rule document must be an object")
        if raw.get("schema_version") != FUNCTIONAL_PARTNER_RULE_SCHEMA:
            raise ValueError("unsupported functional partner rule schema")
        policy = raw.get("policy")
        expected_policy = {
            "asset_keyed_only": True,
            "category_inheritance": False,
            "commonsense_completion": False,
            "missing_annotation_behavior": "emit_no_edge",
        }
        if policy != expected_policy:
            raise ValueError("functional partner rule policy is not strict v1")
        raw_rules = raw.get("rules")
        if not isinstance(raw_rules, dict):
            raise ValueError("functional partner rules must be an object")

        rules: dict[str, FrozenFunctionalPartnerRule] = {}
        serializable_rules: dict[str, dict[str, object]] = {}
        for raw_id, payload in raw_rules.items():
            hssd_id = normalize_hssd_id(raw_id)
            if hssd_id is None or hssd_id != raw_id:
                raise ValueError(f"non-canonical HSSD rule key {raw_id!r}")
            if not isinstance(payload, dict):
                raise ValueError(f"invalid functional partner rule for {hssd_id}")
            raw_partners = payload.get("partner_categories")
            if not isinstance(raw_partners, list) or not raw_partners:
                raise ValueError(f"empty functional partner annotation for {hssd_id}")
            partners = tuple(str(value).strip() for value in raw_partners)
            if any(not value for value in partners) or tuple(sorted(set(partners))) != partners:
                raise ValueError(f"partner categories are not canonical for {hssd_id}")
            source_category = payload.get("source_category")
            if source_category is not None:
                source_category = str(source_category).strip() or None
            rule = FrozenFunctionalPartnerRule(
                hssd_id=hssd_id,
                source_category=source_category,
                partner_categories=partners,
            )
            rules[hssd_id] = rule
            serializable_rules[hssd_id] = {
                "source_category": source_category,
                "partner_categories": list(partners),
            }

        declared_count = raw.get("annotated_asset_count")
        if declared_count != len(rules):
            raise ValueError("annotated asset count does not match frozen rules")
        rules_payload = json.dumps(
            serializable_rules,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        actual_rules_sha256 = hashlib.sha256(rules_payload).hexdigest()
        if raw.get("rules_sha256") != actual_rules_sha256:
            raise ValueError("functional partner rule checksum mismatch")
        source = raw.get("source")
        if not isinstance(source, dict) or not isinstance(source.get("sha256"), str):
            raise ValueError("functional partner source checksum is missing")
        compatibility_rules: tuple[FrozenTargetCategoryCompatibilityRule, ...] = ()
        compatibility_rules_sha256: str | None = None
        if compatibility_path is not None:
            compatibility_rules, compatibility_rules_sha256 = (
                _load_category_compatibility(compatibility_path)
            )
        return cls(
            source_sha256=source["sha256"],
            rules_sha256=actual_rules_sha256,
            asset_count_in_source=int(raw.get("asset_count_in_source", 0)),
            rules=rules,
            compatibility_rules=compatibility_rules,
            compatibility_rules_sha256=compatibility_rules_sha256,
        )

    def __len__(self) -> int:
        return len(self._rules)

    def rule_for(self, hssd_id: str | None) -> FrozenFunctionalPartnerRule | None:
        normalized = normalize_hssd_id(hssd_id)
        return self._rules.get(normalized) if normalized is not None else None

    def build_edges(
        self, furniture: Sequence[FurnitureInput]
    ) -> tuple[FunctionalPartnerInput, ...]:
        """Emit annotated category candidates; never inherit rules across assets."""
        return tuple(
            FunctionalPartnerInput(
                source_id=match.source_id,
                target_id=match.target_id,
                orientation_mode="none",
                target_strategy="none",
                desired_gap_range_m=None,
                valid=True,
            )
            for match in self.build_edge_matches(furniture)
        )

    def build_edge_matches(
        self, furniture: Sequence[FurnitureInput]
    ) -> tuple[FrozenFunctionalPartnerMatch, ...]:
        """Return exact annotated matches with frozen category-compatibility audit."""
        result: list[FrozenFunctionalPartnerMatch] = []
        seen: set[tuple[str, str]] = set()
        for source in furniture:
            rule = self.rule_for(source.hssd_id)
            if rule is None:
                continue
            for target in furniture:
                if source.object_id == target.object_id:
                    continue
                matched_partner: str | None = None
                match_mode: str | None = None
                compatibility_rule_id: str | None = None
                target_key = normalize_partner_category(target.category)
                for partner_category in rule.partner_categories:
                    if normalize_partner_category(partner_category) == target_key:
                        matched_partner = partner_category
                        match_mode = "exact_canonical_category"
                        break
                    compatibility = next(
                        (
                            value
                            for value in self._compatibility_rules
                            if value.matches(partner_category, target)
                        ),
                        None,
                    )
                    if compatibility is not None:
                        matched_partner = partner_category
                        match_mode = "frozen_semantic_compatibility"
                        compatibility_rule_id = compatibility.rule_id
                        break
                if matched_partner is None or match_mode is None:
                    continue
                signature = (source.object_id, target.object_id)
                if signature in seen:
                    continue
                seen.add(signature)
                result.append(
                    FrozenFunctionalPartnerMatch(
                        source_id=source.object_id,
                        target_id=target.object_id,
                        source_hssd_id=rule.hssd_id,
                        annotated_partner_category=matched_partner,
                        category_match_mode=match_mode,
                        compatibility_rule_id=compatibility_rule_id,
                    )
                )
        return tuple(result)

    def apply(self, scene: SceneInput) -> SceneInput:
        if scene.functional_partners:
            raise ValueError(
                "scene already contains FUNCTIONAL_PARTNER edges; refusing to mix sources"
            )
        return replace(scene, functional_partners=self.build_edges(scene.furniture))


def _load_category_compatibility(
    path: str | Path,
) -> tuple[tuple[FrozenTargetCategoryCompatibilityRule, ...], str]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("functional partner compatibility document must be an object")
    if raw.get("schema_version") != FUNCTIONAL_PARTNER_COMPATIBILITY_SCHEMA:
        raise ValueError("unsupported functional partner compatibility schema")
    expected_policy = {
        "source_asset_annotation_required": True,
        "semantic_evidence_required": True,
        "category_inheritance": False,
        "commonsense_completion": False,
    }
    if raw.get("policy") != expected_policy:
        raise ValueError("functional partner compatibility policy is not strict v1")
    rows = raw.get("rules")
    if not isinstance(rows, list) or not rows:
        raise ValueError("functional partner compatibility rules must be non-empty")
    rules: list[FrozenTargetCategoryCompatibilityRule] = []
    serializable: list[dict[str, object]] = []
    for value in rows:
        if not isinstance(value, dict):
            raise ValueError("invalid functional partner compatibility rule")
        required_functions = value.get("required_functions")
        if (
            not isinstance(required_functions, list)
            or not required_functions
            or sorted(set(required_functions)) != required_functions
        ):
            raise ValueError("required_functions must be sorted and non-empty")
        payload = {
            "rule_id": str(value.get("rule_id", "")).strip(),
            "annotated_partner_category": str(
                value.get("annotated_partner_category", "")
            ).strip(),
            "target_category": str(value.get("target_category", "")).strip(),
            "required_family": str(value.get("required_family", "")).strip(),
            "required_functions": [str(item).strip() for item in required_functions],
        }
        if any(not payload[key] for key in payload if key != "required_functions"):
            raise ValueError("compatibility rule fields must be non-empty")
        serializable.append(payload)
        rules.append(
            FrozenTargetCategoryCompatibilityRule(
                rule_id=str(payload["rule_id"]),
                annotated_partner_category=str(payload["annotated_partner_category"]),
                target_category=str(payload["target_category"]),
                required_family=str(payload["required_family"]),
                required_functions=tuple(payload["required_functions"]),
            )
        )
    if [value.rule_id for value in rules] != sorted(
        set(value.rule_id for value in rules)
    ):
        raise ValueError("compatibility rule ids must be unique and sorted")
    payload_bytes = json.dumps(
        serializable, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    actual_sha256 = hashlib.sha256(payload_bytes).hexdigest()
    if raw.get("rules_sha256") != actual_sha256:
        raise ValueError("functional partner compatibility checksum mismatch")
    return tuple(rules), actual_sha256
