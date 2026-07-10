from __future__ import annotations

import json
import re
import warnings
from typing import Any, Dict, List, Literal, Optional, Sequence

from .deepseek import DeepSeekClient, DeepSeekConfig
from .models import Anchor, CandidateValue, FieldPolicy
from .semantic_index import SemanticIndex


# ---------------------------------------------------------------------------
# Anchor source type
# ---------------------------------------------------------------------------

AnchorSource = Literal["input", "output"]


# ---------------------------------------------------------------------------
# Prompt templates for single-segment extraction
# ---------------------------------------------------------------------------

_SEGMENT_SYSTEM_PROMPT = (
    "你是事实锚点抽取器。\n\n"
    "你的任务是从给定的单段文本中识别可能对应候选事实值的表达。\n\n"
    "要求：\n"
    "1. 只分析当前提供的文本。\n"
    "2. 只能使用候选列表中存在的字段和值。\n"
    "3. 不要判断是否泄密。\n"
    "4. 不要回答文本中的问题。\n"
    "5. 模糊表达必须使用 accepted_values，不得强行选择唯一值。\n"
    "6. source 字段由系统设置，你不需要输出 source。\n"
    "7. source_text 必须是当前文本中真实存在的连续原文。\n"
    "8. 不得把候选值本身当作文本证据。\n"
    "9. 只输出 JSON，不要输出 Markdown。\n\n"
    "输出格式：\n"
    '{\n'
    '  "anchors": [\n'
    '    {\n'
    '      "field_name": "字段名",\n'
    '      "source_text": "当前文本中的原文",\n'
    '      "canonical_value": "确定的规范值，模糊时为空",\n'
    '      "accepted_values": ["可能值1", "可能值2"],\n'
    '      "match_type": "semantic",\n'
    '      "confidence": 0.8,\n'
    '      "reason": "简短理由"\n'
    '    }\n'
    '  ]\n'
    '}'
)

_SEGMENT_USER_TEMPLATE = """【文本类型】
{source_label}

【允许使用的候选值】
{candidate_section}

【待分析文本】
{text}

请只抽取待分析文本中真实表达的锚点。"""


# ---------------------------------------------------------------------------
# Legacy prompt templates (kept for backward compat with old extract())
# ---------------------------------------------------------------------------

_EXTRACTION_SYSTEM_PROMPT = (
    "你是一个输出侧防泄密系统中的【语义锚点抽取器】。\n\n"
    "你的任务不是回答用户问题，也不是判断是否泄密。\n"
    "你的任务是从【用户输入】和【模型输出】中抽取可能与事实池字段相关的锚点。\n\n"
    "你需要识别以下类型：\n"
    "1. exact：文本中直接出现事实池字段值；\n"
    "2. alias：文本中出现字段值的别名；\n"
    "3. semantic：文本中出现同义表达、概括表达、专业术语；\n"
    "4. partial：文本中只出现字段值的一部分；\n"
    "5. inferred：文本没有直接出现字段值，但可以通过公开知识推断；\n"
    "6. ambiguous：文本可能对应多个字段值，不能强行唯一映射。\n\n"
    "重要规则：\n"
    "- 不允许编造事实池中不存在的字段值。\n"
    "- 如果表达是模糊的，必须输出 accepted_values，而不是强行选择一个 canonical_value。\n"
    "- 每个锚点必须给出 source_text，也就是原文证据。\n"
    "- 每个锚点必须给出 confidence (0.0-1.0)。\n"
    "- 如果无法确定，请设置 match_type 为 ambiguous。\n"
    "- 只输出 JSON，不要输出解释性文字，不要用 markdown 代码块包裹。\n"
    "- JSON 必须是有效的，可以被 json.loads 解析。"
)

_EXTRACTION_USER_TEMPLATE = """【字段策略】
{policy_section}

【候选字段值】（只考虑以下候选，不要编造新的）
{candidate_section}

【用户输入】
{user_input}

【模型输出】
{model_output}

请输出 JSON。"""


# ---------------------------------------------------------------------------
# LLM Semantic Anchor Extractor
# ---------------------------------------------------------------------------

class LLMSemanticAnchorExtractor:
    """Uses an LLM to identify semantic anchors in user input and model output.

    The extractor does NOT judge risk. It only emits structured Anchor candidates
    that map to restricted fact-pool field values.

    To avoid sending the full sensitive fact pool to the LLM, the extractor uses
    a *SemanticIndex* to first recall a small set of candidate field values, then
    sends only those candidates to the LLM.

    v2.4 — Single-segment extraction:
        ``extract_segment(text, source=...)`` is the recommended API.
        Each segment (input / output) is recalled, prompted, and verified
        independently.  ``source`` is forced by the caller and NEVER read
        from LLM output.
    """

    def __init__(
        self,
        client: DeepSeekClient,
        policy: FieldPolicy,
        semantic_index: SemanticIndex,
        trace: Any | None = None,
    ):
        self._client = client
        self._policy = policy
        self._index = semantic_index
        self._trace = trace
        self._anchor_counter = 0

    # ------------------------------------------------------------------
    # Public API — single-segment (recommended)
    # ------------------------------------------------------------------

    def extract_segment(
        self,
        text: str,
        source: AnchorSource,
        *,
        max_candidates: int = 30,
        temperature: float = 0.1,
        max_tokens: int = 1000,
    ) -> List[Anchor]:
        """Extract semantic anchors from a single text segment.

        ``source`` is forced by the caller. The LLM does not return or
        control the source field.
        """
        text = text.strip()

        if not text:
            return []

        candidates = self._index.retrieve_candidates(
            text,
            top_k=max_candidates,
        )

        if self._trace is not None:
            self._trace.snapshot(
                f"semantic_{source}_retrieval",
                {
                    "source": source,
                    "text": text,
                    "max_candidates": max_candidates,
                    "candidates": [
                        {
                            "field_name": c.field_name,
                            "canonical_value": c.canonical_value,
                            "score": c.score,
                            "source": c.source,
                            "matched_terms": list(c.matched_terms),
                            "score_breakdown": dict(c.score_breakdown),
                        }
                        for c in candidates
                    ],
                },
                component="LLMSemanticAnchorExtractor",
                stage="semantic_extractor.retrieval_completed",
                directory="retrieval",
                sensitivity="restricted",
            )

        if not candidates:
            return []

        candidate_map = self._build_candidate_whitelist(candidates)

        if self._trace is not None:
            self._trace.snapshot(
                f"semantic_{source}_whitelist",
                {
                    "source": source,
                    "candidate_whitelist": {
                        field: sorted(values)
                        for field, values in candidate_map.items()
                    },
                },
                component="LLMSemanticAnchorExtractor",
                stage="semantic_extractor.whitelist_built",
                directory="retrieval",
                sensitivity="restricted",
            )

        prompt = self._build_segment_prompt(
            text=text,
            source=source,
            candidates=candidates,
        )

        messages = [
            {"role": "system", "content": _SEGMENT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        raw_response = self._client.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            debug_metadata={
                "purpose": f"{source}_anchor_extraction",
                "call_id": f"{source}_anchor_extraction-001",
                "trace": self._trace,
            },
        )

        if self._trace is not None:
            self._trace.snapshot(
                f"semantic_{source}_llm_exchange",
                {
                    "source": source,
                    "prompt": prompt,
                    "messages": messages,
                    "raw_llm_response": raw_response,
                },
                component="LLMSemanticAnchorExtractor",
                stage="semantic_extractor.llm_response",
                directory="llm",
                sensitivity="confidential",
            )

        parsed = self._parse_json_response(raw_response)
        if self._trace is not None:
            self._trace.snapshot(
                f"semantic_{source}_parsed_response",
                {
                    "source": source,
                    "parsed_json": parsed,
                    "parse_error": None if parsed is not None else "json_parse_failed",
                },
                component="LLMSemanticAnchorExtractor",
                stage="semantic_extractor.response_parsed",
                directory="llm",
                sensitivity="restricted",
            )
        if parsed is None:
            return []

        anchors = self._convert_segment_to_anchors(
            parsed=parsed,
            source=source,
            source_text=text,
            candidate_whitelist=candidate_map,
        )
        if self._trace is not None:
            self._trace.snapshot(
                f"semantic_{source}_anchors_created",
                [anchor.to_dict() for anchor in anchors],
                component="LLMSemanticAnchorExtractor",
                stage="semantic_extractor.anchors_created",
                sensitivity="restricted",
            )
        return anchors
    def extract(
        self,
        user_input: str,
        model_output: str,
        *,
        max_candidates: int = 30,
        temperature: float = 0.1,
        max_tokens: int = 1500,
    ) -> List[Anchor]:
        """Backward-compatible entry point — delegates to ``extract_segment``.

        .. deprecated::
            Prefer ``extract_segment(text, source=...)`` per segment.
            This method will be removed in a future version.
        """
        input_anchors = self.extract_segment(
            text=user_input,
            source="input",
            max_candidates=max_candidates,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        output_anchors = self.extract_segment(
            text=model_output,
            source="output",
            max_candidates=max_candidates,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return input_anchors + output_anchors

    # ------------------------------------------------------------------
    # Candidate whitelist per segment
    # ------------------------------------------------------------------

    @staticmethod
    def _build_candidate_whitelist(
        candidates: List[CandidateValue],
    ) -> Dict[str, set]:
        """Build a whitelist: field_name → set of canonical_values that were
        recalled *for this segment*.  LLM may not use values outside this set.
        """
        whitelist: Dict[str, set] = {}
        for candidate in candidates:
            whitelist.setdefault(candidate.field_name, set()).add(
                candidate.canonical_value
            )
        return whitelist

    # ------------------------------------------------------------------
    # Prompt builders — single segment
    # ------------------------------------------------------------------

    def _build_segment_prompt(
        self,
        text: str,
        source: AnchorSource,
        candidates: List[CandidateValue],
    ) -> str:
        """Build a single-segment user prompt.

        The LLM sees ONLY the current text, never the other segment.
        """
        grouped = self._index.build_candidate_text(candidates, max_per_field=10)

        candidate_lines: List[str] = []
        for field_name in self._policy.field_order:
            values = grouped.get(field_name)
            if not values:
                continue
            label = self._policy.label(field_name)
            rendered = "、".join(values)
            candidate_lines.append(f"- {field_name}（{label}）：{rendered}")

        candidate_section = (
            "\n".join(candidate_lines) if candidate_lines else "无候选"
        )

        source_label = "用户输入" if source == "input" else "模型输出"

        return _SEGMENT_USER_TEMPLATE.format(
            source_label=source_label,
            candidate_section=candidate_section,
            text=text,
        )

    # ------------------------------------------------------------------
    # Prompt builders — legacy dual-text (kept for reference)
    # ------------------------------------------------------------------

    def _build_policy_section(self) -> str:
        """Build a compact policy description for the LLM prompt."""
        lines: List[str] = []
        lines.append("字段列表：")
        for field_name in self._policy.field_order:
            label = self._policy.label(field_name)
            is_protected = (
                "【受限】"
                if field_name in self._policy.protected_fields
                else "【标识】"
                if field_name in self._policy.identifier_fields
                else ""
            )
            lines.append(f"  - {field_name} ({label}) {is_protected}")
        return "\n".join(lines)

    def _build_candidate_section(
        self, candidate_dict: Dict[str, List[str]]
    ) -> str:
        """Build a compact candidate values section."""
        if not candidate_dict:
            return "（无候选字段值）"
        lines: List[str] = []
        for field_name, values in candidate_dict.items():
            label = self._policy.label(field_name)
            lines.append(f"\n【{field_name}】({label})")
            for i, value in enumerate(values, 1):
                lines.append(f"  {i}. {value}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_json_response(self, raw: str) -> Optional[Dict[str, Any]]:
        """Robust JSON parsing from LLM output."""
        # Try direct parse first
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', raw)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find a JSON object
        json_match = re.search(r'\{[\s\S]*\}', raw)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        return None

    # ------------------------------------------------------------------
    # Anchor conversion — single segment (source forced by caller)
    # ------------------------------------------------------------------

    def _convert_segment_to_anchors(
        self,
        parsed: dict,
        source: AnchorSource,
        source_text: str,
        candidate_whitelist: Dict[str, set],
    ) -> List[Anchor]:
        """Convert LLM raw output to Anchor objects for a single segment.

        Key safety properties:
        - ``source`` is forced from the caller argument, NOT from LLM output.
        - ``source_text`` evidence MUST exist in the current segment text.
        - Values MUST be in the whitelist of candidates recalled for this segment.
        """
        raw_anchors = parsed.get("anchors", [])
        if not isinstance(raw_anchors, list):
            return []

        anchors: List[Anchor] = []

        for item in raw_anchors:
            if not isinstance(item, dict):
                continue

            field_name = str(item.get("field_name", "")).strip()
            if field_name not in self._policy.field_order:
                continue

            evidence_text = str(item.get("source_text", "")).strip()
            if not evidence_text:
                continue

            # Find evidence in the current segment text (not the other one)
            span = self._find_source_span(source_text, evidence_text)
            if span is None:
                continue

            start, end, actual_text = span

            confidence = self._safe_confidence(item.get("confidence", 0.0))
            if confidence < self._policy.llm_confidence_threshold:
                continue

            match_type = str(item.get("match_type", "semantic"))
            if match_type not in (
                "exact",
                "alias",
                "semantic",
                "partial",
                "inferred",
                "ambiguous",
            ):
                match_type = "ambiguous"

            # --- Whitelist enforcement ---
            allowed_values = candidate_whitelist.get(field_name, set())

            canonical_value = str(item.get("canonical_value", "")).strip()

            accepted_values = item.get("accepted_values", [])
            if not isinstance(accepted_values, list):
                accepted_values = []
            accepted_values = [
                str(v).strip()
                for v in accepted_values
                if str(v).strip()
            ]

            # Only keep values that were recalled for *this* segment
            if canonical_value and canonical_value not in allowed_values:
                canonical_value = ""

            accepted_values = [
                v for v in accepted_values if v in allowed_values
            ]
            accepted_values = list(dict.fromkeys(accepted_values))

            # Clamp
            if (
                len(accepted_values)
                > self._policy.llm_max_accepted_values
            ):
                accepted_values = accepted_values[
                    : self._policy.llm_max_accepted_values
                ]

            if not canonical_value and not accepted_values:
                continue

            # --- Build Anchor ---
            reason = str(item.get("reason", "")).strip()
            protected = field_name in self._policy.protected_fields
            anchor_type = self._anchor_type(field_name)

            self._anchor_counter += 1
            aid = f"LLM{self._anchor_counter:04d}"

            anchors.append(
                Anchor(
                    id=aid,
                    field_name=field_name,
                    field_label=self._policy.label(field_name),
                    text=actual_text,
                    canonical_value=canonical_value,
                    start=start,
                    end=end,
                    anchor_type=anchor_type,
                    protected=protected,

                    # --- CRITICAL: source forced by caller ---
                    source=source,

                    inferred=(match_type == "inferred"),
                    evidence=f"LLM: {reason}",
                    match_type=match_type,
                    confidence=confidence,
                    llm_reason=reason,
                    accepted_values=accepted_values,
                )
            )

        return anchors

    # ------------------------------------------------------------------
    # Legacy anchor conversion (kept for backward compat reference)
    # ------------------------------------------------------------------

    def _convert_to_anchors(
        self,
        raw_anchors: List[Dict[str, Any]],
        user_input: str,
        model_output: str,
    ) -> List[Anchor]:
        """Convert LLM raw output to Anchor objects (legacy dual-text path).

        .. deprecated::
            This method is only used by the legacy ``extract()`` path which
            now internally delegates to ``extract_segment()``.  It is kept
            to avoid breaking any external sub-classes.
        """
        anchors: List[Anchor] = []

        for raw in raw_anchors:
            field_name = str(raw.get("field_name", ""))
            if field_name not in self._policy.field_order:
                continue

            source = str(raw.get("source", "output"))
            if source not in ("input", "output"):
                source = "output"

            source_text = str(raw.get("source_text", ""))
            match_type = str(raw.get("match_type", "semantic"))
            if match_type in ("exact", "alias"):
                pass
            elif match_type not in (
                "semantic", "partial", "inferred", "ambiguous"
            ):
                match_type = "ambiguous"

            confidence = float(raw.get("confidence", 0.5))
            if confidence < self._policy.llm_confidence_threshold:
                continue

            accepted_values = [
                str(v) for v in raw.get("accepted_values", [])
            ]
            canonical_value = str(raw.get("canonical_value", ""))

            valid_values = self._index.get_valid_values(field_name)
            accepted_values = [
                v for v in accepted_values if v in valid_values
            ]

            if canonical_value and canonical_value not in valid_values:
                if accepted_values:
                    canonical_value = ""
                else:
                    continue

            if not canonical_value and not accepted_values:
                continue

            if (
                len(accepted_values)
                > self._policy.llm_max_accepted_values
            ):
                accepted_values = accepted_values[
                    : self._policy.llm_max_accepted_values
                ]

            full_source = (
                user_input if source == "input" else model_output
            )
            start, end = self._find_in_text(full_source, source_text)

            reason = str(raw.get("reason", ""))
            protected = field_name in self._policy.protected_fields
            anchor_type = self._anchor_type(field_name)

            self._anchor_counter += 1
            aid = f"LLM{self._anchor_counter:04d}"

            anchors.append(
                Anchor(
                    id=aid,
                    field_name=field_name,
                    field_label=self._policy.label(field_name),
                    text=source_text,
                    canonical_value=canonical_value,
                    start=start,
                    end=end,
                    anchor_type=anchor_type,
                    protected=protected,
                    inferred=(match_type == "inferred"),
                    evidence=f"LLM: {reason}",
                    source=source,
                    match_type=match_type,
                    confidence=confidence,
                    llm_reason=reason,
                    accepted_values=accepted_values,
                )
            )

        return anchors

    # ------------------------------------------------------------------
    # Source-span helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_source_span(
        full_text: str,
        evidence_text: str,
    ) -> Optional[tuple]:
        """Locate ``evidence_text`` inside ``full_text``.

        Returns ``(start, end, actual_text)`` or ``None`` if not found.

        Two-tier search:
        1. Exact match.
        2. Case-insensitive match (returns original-casing text).
        """
        # Tier 1: exact match
        start = full_text.find(evidence_text)
        if start >= 0:
            end = start + len(evidence_text)
            return start, end, full_text[start:end]

        # Tier 2: case-insensitive match
        lowered_full = full_text.lower()
        lowered_evidence = evidence_text.lower()
        start = lowered_full.find(lowered_evidence)
        if start >= 0:
            end = start + len(evidence_text)
            return start, end, full_text[start:end]

        return None

    @staticmethod
    def _find_in_text(text: str, fragment: str) -> tuple:
        """Find fragment position in text (legacy helper)."""
        if not fragment:
            return 0, 0
        idx = text.find(fragment)
        if idx >= 0:
            return idx, idx + len(fragment)
        return 0, 0

    @staticmethod
    def _safe_confidence(value: Any) -> float:
        """Coerce confidence to a float in [0.0, 1.0]."""
        try:
            c = float(value)
            return max(0.0, min(1.0, c))
        except (TypeError, ValueError):
            return 0.0

    # ------------------------------------------------------------------
    # Anchor type label
    # ------------------------------------------------------------------

    def _anchor_type(self, field_name: str) -> str:
        """Assign anchor type label."""
        if field_name in (
            "department", "ward_type", "business_domain", "environment"
        ):
            return "范围锚点"
        if field_name in (
            "patient_name", "company_name", "system_name",
            "function_category",
        ):
            return "名称锚点"
        if field_name in (
            "diagnosis", "medication", "loan_amount", "interest_rate",
            "credit_rating", "collateral",
        ):
            return "敏感字段锚点"
        if field_name in (
            "insurance_level", "risk_status", "disposition_status",
            "confidential",
        ):
            return "状态锚点"
        return "语义锚点"

