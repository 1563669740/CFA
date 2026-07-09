from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any, List

from .models import AssetFact, FieldPolicy


def load_semantic_aliases(path: str | Path) -> dict[str, dict[str, dict[str, object]]]:
    """Load semantic aliases from a JSON file.

    The file should have the shape:
    {
      "description": "...",
      "semantic_aliases": {
        "field_name": {
          "canonical_value": {
            "components": [...],
            "aliases": [...],
            "partial_clues": [...],
            "possible_inferences": [...],
            "partial_match_policy": "..."
          }
        }
      }
    }
    """
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    aliases = data.get("semantic_aliases", data) if isinstance(data, dict) else {}
    if not isinstance(aliases, dict):
        raise ValueError("semantic aliases json must be an object with 'semantic_aliases' key")
    # Ensure all values are properly typed dicts
    result: dict[str, dict[str, dict[str, object]]] = {}
    for field_name, value_map in aliases.items():
        if isinstance(value_map, dict):
            result[str(field_name)] = {
                str(canonical): dict(info) if isinstance(info, dict) else {}
                for canonical, info in value_map.items()
            }
    return result


def load_assets(path: str | Path) -> List[AssetFact]:
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(data, list):
        raise ValueError("assets json must be a list of asset rows")
    return [AssetFact.from_dict(row) for row in data]


def load_policy(path: str | Path) -> FieldPolicy:
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError("policy json must be an object")
    return FieldPolicy.from_dict(data)


def load_public_knowledge(path: str | Path) -> List[dict[str, Any]]:
    """Load public knowledge implication rules.

    The file can be either a JSON list of rules, or an object with a `rules`
    / `public_rules` array. Each rule uses the same shape as policy.public_rules.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if isinstance(data, list):
        return [dict(row) for row in data]
    if isinstance(data, dict):
        rules = data.get("rules", data.get("public_rules", []))
        if not isinstance(rules, list):
            raise ValueError("public knowledge rules must be a list")
        return [dict(row) for row in rules]
    raise ValueError("public knowledge json must be a list or object")


def merge_public_knowledge(policy: FieldPolicy, public_rules: List[dict[str, Any]]) -> FieldPolicy:
    if not public_rules:
        return policy
    return replace(policy, public_rules=[*policy.public_rules, *public_rules])


def dump_json(data, path: str | Path) -> None:
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
