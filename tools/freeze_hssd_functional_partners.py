from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "hssd_functional_partner_rules_v1"
SOURCE_FIELD = "interaction_clearance.functional_partners.partners"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_hssd_id(value: object) -> str:
    result = str(value).strip().lower()
    if result.startswith("hssd:"):
        result = result[5:]
    return result


def extract_rules(source_path: Path) -> dict[str, Any]:
    with gzip.open(source_path, "rt", encoding="utf-8") as stream:
        lookup = json.load(stream)
    if not isinstance(lookup, dict):
        raise ValueError("HSSD annotation lookup must be a JSON object")

    rules: dict[str, dict[str, object]] = {}
    for raw_id, raw_record in sorted(lookup.items()):
        if not isinstance(raw_record, dict):
            continue
        interaction = raw_record.get("interaction_clearance")
        if not isinstance(interaction, dict):
            continue
        annotation = interaction.get("functional_partners")
        if not isinstance(annotation, dict):
            continue
        raw_partners = annotation.get("partners")
        if not isinstance(raw_partners, list):
            continue
        partners = sorted(
            {
                value.strip()
                for value in raw_partners
                if isinstance(value, str) and value.strip()
            }
        )
        if not partners:
            continue
        hssd_id = _normalize_hssd_id(raw_id)
        if not hssd_id:
            continue
        source_category = annotation.get("cat") or raw_record.get("category")
        rules[hssd_id] = {
            "source_category": (
                str(source_category).strip() if source_category is not None else None
            ),
            "partner_categories": partners,
        }

    rules_payload = json.dumps(
        rules, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "file": source_path.name,
            "sha256": _sha256(source_path),
            "field": SOURCE_FIELD,
        },
        "policy": {
            "asset_keyed_only": True,
            "category_inheritance": False,
            "commonsense_completion": False,
            "missing_annotation_behavior": "emit_no_edge",
        },
        "asset_count_in_source": len(lookup),
        "annotated_asset_count": len(rules),
        "rules_sha256": hashlib.sha256(rules_payload).hexdigest(),
        "rules": rules,
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    workspace_root = repo_root.parent
    parser = argparse.ArgumentParser(
        description="Freeze existing per-asset HSSD functional partner annotations."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=(
            workspace_root
            / "hssd-annotations"
            / "data"
            / "hssd_annotation_lookup.json.gz"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "configs" / "functional_partner_rules_hssd_v1.json",
    )
    args = parser.parse_args()

    payload = extract_rules(args.source.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"froze {payload['annotated_asset_count']} annotated assets "
        f"from {payload['asset_count_in_source']} source assets to {args.output}"
    )


if __name__ == "__main__":
    main()
