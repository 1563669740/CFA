from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Dict, Iterable, List, Sequence, Tuple

from .models import Anchor, AssetFact, FieldPolicy


class RuleBasedAnchorExtractor:
    """Extract technical anchors from user input and model output.

    The extractor is deterministic by default. It simulates the role an LLM/NER
    module would play by using fact values, configured aliases, and public rules.
    A real LLM extractor can later emit the same Anchor objects.
    """

    _CONFIDENTIAL_TEXT_FIELDS = {
        "secret_content",
        "secret_summary",
        "secret_keywords",
        "confidential_level",
        "attack_paraphrases",
    }

    def __init__(self, policy: FieldPolicy):
        self.policy = policy

    def extract(self, raw_answer: str, assets: Sequence[AssetFact], source: str = "output") -> List[Anchor]:
        return self.extract_segments([(source, raw_answer)], assets)

    def extract_segments(self, segments: Sequence[Tuple[str, str]], assets: Sequence[AssetFact]) -> List[Anchor]:
        anchors: List[Anchor] = []
        seen = set()

        for source, text in segments:
            if not text:
                continue

            # 1) Exact values from the restricted fact pool.
            for field_name in self.policy.field_order:
                values = sorted({a.get(field_name) for a in assets if a.get(field_name)}, key=len, reverse=True)
                for value in values:
                    if self._is_weak_confidential_value(field_name, value):
                        continue
                    for start, end, matched_text in self._find_all(text, value):
                        key = (source, field_name, value, start, end)
                        if key in seen:
                            continue
                        seen.add(key)
                        anchors.append(
                            self._make_anchor(field_name, matched_text, value, start, end, inferred=False, source=source)
                        )

            # 2) Configured aliases: phrase in text -> canonical fact value.
            for field_name, alias_map in self.policy.field_aliases.items():
                for canonical, aliases in alias_map.items():
                    all_terms = [canonical] + list(aliases)
                    for term in sorted(set(all_terms), key=len, reverse=True):
                        if self._is_weak_confidential_value(field_name, term):
                            continue
                        for start, end, matched_text in self._find_all(text, term):
                            key = (source, field_name, canonical, start, end)
                            if key in seen:
                                continue
                            seen.add(key)
                            anchors.append(
                                self._make_anchor(
                                    field_name,
                                    matched_text,
                                    canonical,
                                    start,
                                    end,
                                    inferred=False,
                                    source=source,
                                )
                            )

        anchors = self._deduplicate_overlaps(anchors)

        # 3) Public-rule implications: explicit anchor implies another protected value.
        anchors.extend(self._infer_from_public_rules(anchors))

        source_rank = {"input": 0, "output": 1}
        anchors.sort(key=lambda a: (source_rank.get(a.source, 9), a.start, a.end, a.field_name, a.inferred))
        return anchors

    def _is_weak_confidential_value(self, field_name: str, value: str) -> bool:
        if field_name not in self._CONFIDENTIAL_TEXT_FIELDS:
            return False
        compact = unicodedata.normalize("NFKC", str(value or "")).strip()
        compact = re.sub(r"[\s\W_]+", "", compact, flags=re.UNICODE)
        if not compact:
            return True
        if len(compact) < 4:
            return True
        return False

    def _make_anchor(
        self,
        field_name: str,
        text: str,
        canonical_value: str,
        start: int,
        end: int,
        inferred: bool,
        source_anchor_id: str | None = None,
        evidence: str = "",
        source: str = "output",
    ) -> Anchor:
        protected = field_name in self.policy.protected_fields
        anchor_type = self._anchor_type(field_name)
        aid = self._stable_anchor_id(field_name, canonical_value, start, end, inferred, source, source_anchor_id)
        return Anchor(
            id=aid,
            field_name=field_name,
            field_label=self.policy.label(field_name),
            text=text,
            canonical_value=canonical_value,
            start=start,
            end=end,
            anchor_type=anchor_type,
            protected=protected,
            inferred=inferred,
            source_anchor_id=source_anchor_id,
            evidence=evidence,
            source=source,
        )

    def _stable_anchor_id(
        self,
        field_name: str,
        canonical_value: str,
        start: int,
        end: int,
        inferred: bool,
        source: str,
        source_anchor_id: str | None,
    ) -> str:
        raw = "|".join(
            [field_name, canonical_value, str(start), str(end), str(inferred), source, source_anchor_id or ""]
        )
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10].upper()
        return f"A{digest}"

    def _anchor_type(self, field_name: str) -> str:
        if field_name in {"business_domain", "environment"}:
            return "范围锚点"
        if field_name in {"function_category", "system_name"}:
            return "功能/名称锚点"
        if field_name == "component_version":
            return "版本锚点"
        if field_name in {"risk_status", "disposition_status"}:
            return "状态锚点"
        if field_name == "remote_entry":
            return "入口锚点"
        return "事实锚点"

    def _find_all(self, text: str, needle: str) -> Iterable[Tuple[int, int, str]]:
        if not needle:
            return []
        flags = re.IGNORECASE if re.search(r"[A-Za-z]", needle) else 0
        pattern = self._literal_pattern(needle)
        matches = []
        for m in re.finditer(pattern, text, flags):
            matches.append((m.start(), m.end(), text[m.start() : m.end()]))
        return matches

    def _literal_pattern(self, needle: str) -> str:
        escaped = re.escape(needle)
        if re.search(r"[A-Za-z0-9]", needle):
            # Keep xz-utils 5.6.1 from matching xz-utils 5.6.10 or other longer tokens.
            return rf"(?<![A-Za-z0-9_.-]){escaped}(?![A-Za-z0-9_.-])"
        return escaped

    def _deduplicate_overlaps(self, anchors: List[Anchor]) -> List[Anchor]:
        """Remove exact duplicates only; keep meaningful nested anchors.

        Example: “航天测控生产区” contains “航天测控” and “生产区”; both are useful.
        """
        result: List[Anchor] = []
        seen = set()
        for a in anchors:
            key = (a.source, a.field_name, a.canonical_value, a.start, a.end, a.inferred)
            if key not in seen:
                seen.add(key)
                result.append(a)
        return result

    def _infer_from_public_rules(self, anchors: Sequence[Anchor]) -> List[Anchor]:
        inferred: List[Anchor] = []
        seen = {(a.source, a.field_name, a.canonical_value, a.start, a.end, a.inferred) for a in anchors}
        for rule in self.policy.public_rules:
            source_field = rule.get("field")
            source_values = set(rule.get("values", []))
            implies: Dict[str, str] = dict(rule.get("implies", {}))
            rule_name = str(rule.get("name", "public-rule"))
            for anchor in anchors:
                if anchor.field_name != source_field or anchor.canonical_value not in source_values:
                    continue
                for target_field, target_value in implies.items():
                    key = (anchor.source, target_field, target_value, anchor.start, anchor.end, True)
                    if key in seen:
                        continue
                    seen.add(key)
                    inferred.append(
                        self._make_anchor(
                            target_field,
                            f"{anchor.text} => {target_value}",
                            str(target_value),
                            anchor.start,
                            anchor.end,
                            inferred=True,
                            source_anchor_id=anchor.id,
                            evidence=f"{rule_name}: {anchor.canonical_value} implies {target_field}={target_value}",
                            source=anchor.source,
                        )
                    )
        return inferred
