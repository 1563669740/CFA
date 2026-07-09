from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .models import Anchor, FieldPolicy, RiskFinding


_DEFAULT_SAFE_REPLACEMENTS: Dict[str, str] = {
    "system_name": "相关对象",
    "business_domain": "相关业务域",
    "environment": "相关环境",
    "function_category": "相关系统",
    "component_version": "受影响版本",
    "risk_status": "相关风险状态",
    "disposition_status": "相关处置状态",
    "remote_entry": "远程入口",
}

_DEFAULT_SAFE_HINT = "具体资产名称、系统类型、组件版本和处置状态应通过授权资产管理系统查询。"


class AnswerSanitizer:
    """Create X-replaced and safe versions of the model output.

    Sanitizing is driven by findings. User-input anchors may help detect risk,
    but only model-output text is rewritten.

    The sanitizer receives a *policy* so that it can use scenario-specific
    safe replacements and hint text instead of a single hard-coded set.
    """

    def __init__(self, policy: Optional[FieldPolicy] = None):
        self._policy = policy

    # ---------- helpers that delegate to policy when available ----------

    def _safe_replacement(self, field_name: str) -> str:
        if self._policy and self._policy.safe_replacements:
            return self._policy.safe_replacements.get(field_name, "相关事实")
        return _DEFAULT_SAFE_REPLACEMENTS.get(field_name, "相关事实")

    def _safe_hint(self) -> str:
        if self._policy and self._policy.safe_hint:
            return self._policy.safe_hint
        return _DEFAULT_SAFE_HINT

    def _identifier_field_names(self) -> set:
        if self._policy:
            return set(self._policy.identifier_fields)
        return {"system_name", "business_domain", "environment", "function_category"}

    def _effective_sensitive_field_names(self) -> set:
        """Return the effective set of sensitive field names that must be redacted.

        Priority:
        1. If policy.sensitive_fields is explicitly provided, use it.
        2. Otherwise, auto-compute: protected_fields ∪ identifier_fields ∪ {display_field}.

        This makes the sanitizer fully policy-driven — no hard-coded field names.
        """
        if not self._policy:
            return set()

        if self._policy.sensitive_fields:
            return set(self._policy.sensitive_fields)

        # Fallback: union of protected + identifier + display
        names: set[str] = set(self._policy.protected_fields)
        names.update(self._policy.identifier_fields)
        if self._policy.display_field:
            names.add(self._policy.display_field)
        return names

    # ----------------------------------------------------------------

    def make_x_replaced(self, raw_answer: str, findings: Sequence[RiskFinding]) -> str:
        text = raw_answer

        if findings:
            # Pre-processing: known-case rewrite for demo compatibility.
            text = self._rewrite_known_case_for_x(text)

            text, unresolved = self._apply_field_level_redaction(
                text,
                findings,
                x_mode=True,
            )

            # Fallback: replace target_asset_name (e.g. patient_name=王芳) with "X"
            # even if it was not included in dangerous_output_anchors.
            text = self._redact_target_asset_names(text, findings, x_mode=True)

            # 如果存在无法定位到原文片段的高风险语义锚点，直接退回 X 摘要。
            # 这样可以避免"看起来改写了，但语义线索还残留"的情况。
            if unresolved:
                text = self._make_x_summary(findings)

            return self._append_authorized_query_hint(text)
        else:
            text = self._replace_asset_ids(text)
            # v2.5: No findings = no risk, return text as-is without hint
            return text


    def make_safe_answer(self, raw_answer: str, findings: Sequence[RiskFinding]) -> str:
        text = raw_answer

        if findings:
            # Pre-processing: known-case rewrite for demo compatibility.
            text = self._rewrite_known_case_for_safe(text)

            text, unresolved = self._apply_field_level_redaction(
                text,
                findings,
                x_mode=False,
            )

            # Fallback: replace target_asset_name (e.g. patient_name=王芳) with safe
            # replacement even if it was not included in dangerous_output_anchors.
            text = self._redact_target_asset_names(text, findings, x_mode=False)

            # v2.6: Policy-level sensitive sweep as safety net.
            # This catches any sensitive field values that were missed by
            # findings-driven redaction (e.g., normalized anchors with start=-1).
            text = self._policy_level_sensitive_sweep(text)

            # Post-normalize to eliminate repeated safe placeholders
            # (e.g. "相关治疗方案相关治疗方案治疗" → "相关治疗方案")
            text = self._post_normalize_safe_text(text)

            # 严格兜底：如果危险锚点是语义抽取出来的，但无法在文本中可靠替换，
            # 不再放行原回答，而是返回字段级安全摘要。
            if unresolved:
                text = self._make_safe_summary(findings)

            return self._append_authorized_query_hint(text)
        else:
            text = self._replace_asset_ids(text)
            # v2.5: No findings = no risk, return text as-is without hint
            return text

    # -------- known-case rewrites (keep for aerospace compatibility) --------

    def _rewrite_known_case_for_x(self, text: str) -> str:
        text = re.sub(r"航天测控生产区", "X", text)
        text = self._replace_version(text, "xz-utils 5.6.1", "X")
        text = self._replace_version(text, "xz-utils 5.6.0", "X")
        text = text.replace("仍处于回滚窗口", "仍处于 X")
        text = text.replace("链路健康监测类服务", "相关服务")
        text = re.sub(r"依赖\s*SSH\s*运维入口", "存在远程运维入口", text, flags=re.IGNORECASE)
        text = re.sub(r"SSH\s*运维入口", "远程运维入口", text, flags=re.IGNORECASE)
        text = text.replace("已隔离的测试区资产", "已隔离或测试区资产")
        return text

    def _rewrite_known_case_for_safe(self, text: str) -> str:
        pattern = re.compile(
            r"对于航天测控生产区中使用\s*xz-utils\s*5\.6\.1、仍处于回滚窗口、且依赖\s*SSH\s*运维入口的链路健康监测类服务",
            flags=re.IGNORECASE,
        )
        text = pattern.sub("对已确认使用受影响版本且尚未完成处置的生产区系统", text)
        text = re.sub(
            r"航天测控生产区中使用\s*xz-utils\s*5\.6\.[01]",
            "生产区系统使用受影响版本",
            text,
            flags=re.IGNORECASE,
        )
        text = self._replace_version(text, "xz-utils 5.6.1", "受影响版本")
        text = self._replace_version(text, "xz-utils 5.6.0", "受影响版本")
        text = text.replace("链路健康监测类服务", "相关系统")
        text = text.replace("仍处于回滚窗口", "尚未完成处置")
        text = text.replace("回滚完成前", "处置完成前")
        text = re.sub(r"依赖\s*SSH\s*运维入口", "存在远程运维入口", text, flags=re.IGNORECASE)
        text = re.sub(r"SSH\s*运维入口", "远程运维入口", text, flags=re.IGNORECASE)
        text = text.replace("已隔离的测试区资产", "已隔离或测试区资产")
        text = text.replace("航天测控生产区", "生产区")
        return text

    # ---------- field-level redaction (v2.4) ----------

    def _apply_field_level_redaction(
        self,
        text: str,
        findings: Sequence[RiskFinding],
        *,
        x_mode: bool = False,
    ) -> Tuple[str, bool]:
        """
        Apply field-level redaction instead of literal anchor.text replacement only.

        Returns:
            (redacted_text, unresolved)

        unresolved=True means at least one dangerous output anchor could not be
        reliably located or rewritten. In that case the caller should fall back to
        a safe summary.
        """

        unresolved = False

        for anchor in self._dangerous_output_anchors(findings):
            replacement = "X" if x_mode else self._safe_replacement(anchor.field_name)

            before = text

            # 1. 优先用 span 替换，适用于规则抽取出的精确锚点。
            text = self._replace_anchor_span_if_valid(text, anchor, replacement)

            # 2. 再用 anchor.text / canonical / aliases / semantic aliases / versions 替换。
            text = self._replace_anchor_variants(text, anchor, replacement)

            # 3. 如果还没替换成功，首先确认锚点的危险变体是否仍存在于文本中。
            if text == before:
                if self._any_variant_present(text, anchor):
                    # 尝试句子级兜底。
                    sentence_redacted = self._replace_sentence_containing_anchor(
                        text,
                        anchor,
                        replacement,
                    )

                    if sentence_redacted != text:
                        text = sentence_redacted
                    else:
                        unresolved = True
                elif self._is_semantic_anchor_without_literal_match(text, anchor):
                    # 语义锚点：检测到危险信号但无法映射到具体文本片段，
                    # 标记为 unresolved 以便调用方回退到安全摘要。
                    unresolved = True
                # 否则：危险变体已被前处理（如 known-case rewrite）移除，该锚点已安全。

        # 无论有没有 findings，都顺手处理裸资产编号。
        text = self._replace_asset_ids(text)

        return text, unresolved

    # -------- span-based replacement --------

    def _replace_anchor_span_if_valid(
        self,
        text: str,
        anchor: Anchor,
        replacement: str,
    ) -> str:
        """
        Replace by anchor span when the span is valid.

        This is safer than global literal replacement for exact rule-based anchors.
        For normalized anchors (start=-1), fall back to anchor.text-based replacement.
        """

        if anchor.start is None or anchor.end is None:
            return text

        # Normalized anchors have start=-1, end=-1 — skip span replacement
        if anchor.start < 0 or anchor.end <= anchor.start:
            return text

        if anchor.end > len(text):
            return text

        span_text = text[anchor.start:anchor.end]

        if not span_text.strip():
            return text

        # 如果 span 与 anchor.text 对不上，不强行替换，避免错位。
        if anchor.text and span_text != anchor.text:
            return text

        return text[:anchor.start] + replacement + text[anchor.end:]

    # -------- semantic anchor detection --------

    def _is_semantic_anchor_without_literal_match(
        self, text: str, anchor: Anchor
    ) -> bool:
        """Return True if this anchor carries a dangerous canonical or accepted
        value that was detected via semantic/LLM extraction but none of its
        variants literally appear in `text`.  Such anchors can't be rewritten
        in-place and must trigger a summary fallback.

        Only semantic-origin anchors (match_type or anchor_type == 'semantic')
        trigger this path.  Rule-based anchors whose dangerous text was already
        handled by preprocessing are considered safe."""

        # Not a semantic anchor — if variants are gone, preprocessing handled it.
        if anchor.match_type != "semantic" and anchor.anchor_type != "semantic":
            return False

        if anchor.start >= 0 and anchor.end > anchor.start:
            return False

        if self._is_already_safe_placeholder(anchor):
            return False

        return not self._any_variant_present(text, anchor)

    # -------- variant presence check --------

    def _any_variant_present(self, text: str, anchor: Anchor) -> bool:
        """Return True if at least one dangerous variant of the anchor still
        appears in the text.  Used to decide whether unresolved should be set
        after the primary replacement passes have run."""

        variants = self._variant_terms_for_anchor(anchor)

        if anchor.field_name == "component_version":
            canonical = anchor.effective_canonical_value()
            m = re.search(r"\d+(?:\.\d+){1,3}", canonical or "")
            if m:
                parts = m.group(0).split(".")
                if len(parts) >= 2:
                    variants.append(".".join(parts[:2]))

        variants = [v for v in variants if v]
        text_lower = text.lower()

        for v in variants:
            if v.lower() in text_lower:
                # Also check against safe replacements — they count as "not dangerous"
                if not self._text_is_safe_value(v, anchor.field_name):
                    return True

        return False

    # -------- variant-based replacement --------

    def _replace_anchor_variants(
        self,
        text: str,
        anchor: Anchor,
        replacement: str,
    ) -> str:
        """
        Replace all known textual variants of a dangerous anchor.
        """

        patterns = self._redaction_patterns_for_anchor(anchor)

        # 长模式优先，避免先把短词替换掉导致长词匹配失败。
        patterns = sorted(set(patterns), key=len, reverse=True)

        for pattern in patterns:
            try:
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
            except re.error:
                continue

        return text

    # -------- pattern generation --------

    def _redaction_patterns_for_anchor(self, anchor: Anchor) -> List[str]:
        patterns: List[str] = []

        terms = self._variant_terms_for_anchor(anchor)

        for term in terms:
            term = str(term or "").strip()
            if not term:
                continue

            if anchor.field_name == "component_version":
                patterns.extend(self._version_patterns(term))
            else:
                patterns.append(self._literal_pattern(term))

        # 对版本字段额外用 canonical value 生成宽松模式。
        if anchor.field_name == "component_version":
            canonical = anchor.effective_canonical_value()
            patterns.extend(self._version_patterns(canonical))

        return [p for p in patterns if p]

    # -------- variant terms collection --------

    def _variant_terms_for_anchor(self, anchor: Anchor) -> List[str]:
        """
        Collect textual variants for one anchor:
        - original text
        - canonical value
        - accepted candidate values
        - field aliases
        - semantic aliases
        """

        terms: List[str] = []

        def add(value: str):
            value = str(value or "").strip()
            if value and value not in terms:
                terms.append(value)

        add(anchor.text)
        add(anchor.canonical_value)

        for value in anchor.accepted_values or []:
            add(value)

        if not self._policy:
            return terms

        field_name = anchor.field_name
        canonical = anchor.effective_canonical_value()

        # field_aliases: field_name -> canonical_value -> aliases
        alias_map = self._policy.field_aliases.get(field_name, {}) or {}

        if canonical in alias_map:
            for alias in alias_map.get(canonical, []):
                add(alias)

        # 有些配置里 canonical_value 可能为空，用 accepted_values 再查一次。
        for value in anchor.accepted_values or []:
            if value in alias_map:
                for alias in alias_map.get(value, []):
                    add(alias)

        # semantic_aliases: components / aliases / partial_clues
        semantic_map = self._policy.get_semantic_aliases(field_name)

        candidate_keys = [canonical] + list(anchor.accepted_values or [])

        for key in candidate_keys:
            if not key or key not in semantic_map:
                continue

            semantic = semantic_map[key]

            for item in semantic.aliases:
                add(item)

            for item in semantic.components:
                add(item)

            for item in semantic.partial_clues:
                add(item)

        return terms

    # -------- version fuzzy matching --------

    def _version_patterns(self, term: str) -> List[str]:
        """
        Build fuzzy regex patterns for version-like protected values.

        Examples:
            xz-utils 5.6.1
            xz utils v5.6.1
            5.6.1
            5.6 系列
            5.6 分支
        """

        term = str(term or "").strip()

        if not term:
            return []

        patterns: List[str] = []

        # 原始 literal 仍然保留。
        patterns.append(self._literal_pattern(term))

        # 抽取版本号：5.6 / 5.6.1 / 1.2.3.4
        version_match = re.search(r"(?P<ver>\d+(?:\.\d+){1,3})", term)

        if not version_match:
            return patterns

        version = version_match.group("ver")

        # 产品名部分：版本号前面的文本
        product_part = term[:version_match.start()].strip(" -_:/，,;；")

        version_re = re.escape(version).replace(r"\.", r"\.")

        # 版本号自身，比如 5.6.1。
        patterns.append(rf"(?<![A-Za-z0-9_.-])v?\s*{version_re}(?![A-Za-z0-9_.-])")

        # 产品 + 版本，比如 xz-utils 5.6.1 / xz utils v5.6.1。
        if product_part:
            product_re = re.escape(product_part)
            product_re = product_re.replace(r"\-", r"[\s_-]*")
            product_re = product_re.replace(r"\ ", r"[\s_-]+")

            patterns.append(
                rf"(?<![A-Za-z0-9_.-]){product_re}\s*[-_:]?\s*v?\s*{version_re}(?![A-Za-z0-9_.-])"
            )

        # major.minor 系列表达，比如 5.6 系列 / 5.6 分支 / 5.6 小版本。
        parts = version.split(".")

        if len(parts) >= 2:
            family = ".".join(parts[:2])
            family_re = re.escape(family).replace(r"\.", r"\.")

            patterns.append(
                rf"(?<![A-Za-z0-9_.-]){family_re}\s*(?:系列|分支|版本线|小版本|受影响版本|相关版本)(?![A-Za-z0-9_.-])"
            )

            patterns.append(
                rf"(?:{family_re}\s*系列中受影响的小版本|{family_re}\s*分支中的受影响版本)"
            )

        return patterns

    # -------- sentence-level fallback --------

    def _replace_sentence_containing_anchor(
        self,
        text: str,
        anchor: Anchor,
        replacement: str,
    ) -> str:
        """
        Sentence-level fallback.

        If exact variant replacement fails, redact the sentence that contains
        a recognizable variant.
        """

        variants = self._variant_terms_for_anchor(anchor)

        # 加入 canonical 中的版本 family，帮助定位"5.6 系列"这种句子。
        if anchor.field_name == "component_version":
            canonical = anchor.effective_canonical_value()
            m = re.search(r"\d+(?:\.\d+){1,3}", canonical or "")
            if m:
                parts = m.group(0).split(".")
                if len(parts) >= 2:
                    variants.append(".".join(parts[:2]))

        variants = [v for v in variants if v]

        if not variants:
            return text

        # 简单按中文/英文句末切分，保留分隔符。
        sentence_pattern = re.compile(r"[^。！？!?；;\n]+[。！？!?；;]?", flags=re.MULTILINE)

        def redact_sentence(m: re.Match) -> str:
            sentence = m.group(0)

            for v in variants:
                if v and v.lower() in sentence.lower():
                    return replacement

            return sentence

        return sentence_pattern.sub(redact_sentence, text)

    # -------- safe summary fallbacks --------

    def _make_safe_summary(self, findings: Sequence[RiskFinding]) -> str:
        """
        Build a deterministic safe summary when field-level redaction is uncertain.
        """

        field_names: List[str] = []

        for finding in findings:
            for field_name in finding.restored_fields or []:
                if field_name not in field_names:
                    field_names.append(field_name)

            for anchor in finding.anchors:
                if anchor.source == "output" and not anchor.inferred:
                    if anchor.field_name not in field_names:
                        field_names.append(anchor.field_name)

        labels = []

        for field_name in field_names:
            if self._policy:
                labels.append(self._policy.label(field_name))
            else:
                labels.append(field_name)

        if labels:
            field_text = "、".join(labels)
            return f"该回答涉及可能导致受限事实恢复的字段信息（{field_text}），具体内容已省略。"

        return "该回答涉及可能导致受限事实恢复的信息，具体内容已省略。"


    def _make_x_summary(self, findings: Sequence[RiskFinding]) -> str:
        return "X"

    # -------- safe placeholder detection --------

    def _is_already_safe_placeholder(self, anchor: Anchor) -> bool:
        """Return True only when *all* available textual representations are
        already safe placeholders — i.e. this anchor does not carry any
        dangerous values worth redacting."""

        # Check anchor.text
        anchor_text = str(anchor.text or "").strip()

        if anchor_text and not self._text_is_safe_value(anchor_text, anchor.field_name):
            return False

        # Check canonical_value
        canonical = str(anchor.canonical_value or "").strip()
        if canonical and not self._text_is_safe_value(canonical, anchor.field_name):
            return False

        # Check accepted_values
        for value in anchor.accepted_values or []:
            value = str(value or "").strip()
            if value and not self._text_is_safe_value(value, anchor.field_name):
                return False

        # All populated fields are already safe — skip this anchor.
        return anchor_text != "" or canonical != ""

    def _text_is_safe_value(self, text: str, field_name: str) -> bool:
        """Check whether a single text string is a known safe placeholder."""
        text = str(text or "").strip()

        if not text:
            return True

        safe_values = {
            "相关对象",
            "相关业务域",
            "相关环境",
            "相关系统",
            "受影响版本",
            "相关风险状态",
            "相关处置状态",
            "远程入口",
            "相关事实",
            "已省略",
        }

        if text in safe_values:
            return True

        if self._policy and self._policy.safe_replacements:
            if text == self._policy.safe_replacements.get(field_name):
                return True

        return False

    # ---------- dangerous anchors ----------

    def _dangerous_output_anchors(self, findings: Sequence[RiskFinding]) -> List[Anchor]:
        """Collect dangerous output anchors for sanitization.

        v2.4: Uses policy.sensitive_fields (or its auto-computed fallback) to
        determine which field disclosures are direct-sensitive.  Also includes
        contributing locator anchors (quasi-identifiers that actually narrowed
        the candidate set).
        """
        anchors: List[Anchor] = []
        seen = set()
        sensitive_fields = self._effective_sensitive_field_names()

        for finding in findings:
            if finding.risk_level not in {"MEDIUM", "HIGH", "CRITICAL"}:
                continue

            # Collect key anchor IDs for this finding (v2.3)
            key_ids: set[str] = set()
            if hasattr(finding, "key_anchor_ids") and finding.key_anchor_ids:
                key_ids = set(finding.key_anchor_ids)

            for anchor in finding.anchors:
                if anchor.source != "output" or anchor.inferred:
                    continue
                if self._is_already_safe_placeholder(anchor):
                    continue

                # Direct sensitive disclosure — driven by policy-sensitive fields
                is_direct_sensitive = (
                    anchor.protected
                    or anchor.field_name in sensitive_fields
                )

                # v2.3: Contributing locator anchor (quasi-identifier that
                # actually narrowed the candidate set)
                is_contributing_locator = (
                    key_ids and anchor.id in key_ids
                )

                if not is_direct_sensitive and not is_contributing_locator:
                    continue

                key = (anchor.field_name, anchor.text)
                if key in seen:
                    continue
                seen.add(key)
                anchors.append(anchor)

        return sorted(anchors, key=lambda a: len(a.text), reverse=True)

    # ---------- asset ID replacement ----------

    def _replace_asset_ids(self, text: str) -> str:
        """Replace bare asset IDs like G003, P004 when they appear in Safe Answer."""
        # Match known ID patterns: uppercase letter(s) followed by digits, bounded by non-alnum
        pattern = re.compile(r'(?<![A-Za-z0-9])([A-Z]{1,3}\d{2,4})(?![A-Za-z0-9])')
        def _replacer(m: re.Match) -> str:
            prefix = m.group(1)[0] if m.group(1) else ""
            return f"[{prefix}类编号]"
        return pattern.sub(_replacer, text)

    # ---------- low-level helpers ----------

    def _replace_version(self, text: str, term: str, replacement: str) -> str:
        return self._replace_literal(text, term, replacement)

    def _replace_literal(self, text: str, term: str, replacement: str) -> str:
        if not term:
            return text
        flags = re.IGNORECASE if re.search(r"[A-Za-z]", term) else 0
        return re.sub(self._literal_pattern(term), replacement, text, flags=flags)

    def _literal_pattern(self, term: str) -> str:
        escaped = re.escape(term)
        if re.search(r"[A-Za-z0-9]", term):
            return rf"(?<![A-Za-z0-9_.-]){escaped}(?![A-Za-z0-9_.-])"
        return escaped

    # ---------- target_asset_name fallback redaction ----------

    def _redact_target_asset_names(
        self,
        text: str,
        findings: Sequence[RiskFinding],
        *,
        x_mode: bool = False,
    ) -> str:
        """Fallback: replace target_asset_name in the text when display_field is
        a sensitive field.

        Gated by policy.sensitive_fields (or auto-computed fallback): only
        replaces when the display_field belongs to the effective sensitive set.
        This avoids over-redacting benign display values like product_name or
        department_name that may be public in some domains.

        For each finding, if target_asset_name is a non-trivial, non-placeholder
        string that still appears in the text, replace it with the safe replacement
        (or "X" in x_mode).
        """
        if not self._policy or not self._policy.display_field:
            return text

        display_field = self._policy.display_field
        sensitive_fields = self._effective_sensitive_field_names()

        # Gate: only redact if display_field is actually sensitive
        if display_field not in sensitive_fields:
            return text

        safe_display = self._safe_replacement(display_field)

        for finding in findings:
            target_name = getattr(finding, "target_asset_name", None)
            if not target_name:
                continue
            target_name = str(target_name).strip()
            if not target_name:
                continue
            # Skip if it's already a safe placeholder
            if self._text_is_safe_value(target_name, display_field):
                continue

            replacement = "X" if x_mode else safe_display
            text = self._replace_literal(text, target_name, replacement)

        return text

    # ---------- post-normalization (deduplicate repeated safe placeholders) ----------

    # ---------- v2.6 policy-level sensitive sweep ----------

    def _policy_level_sensitive_sweep(self, text: str) -> str:
        """Deterministic policy-level sweep for protected/sensitive field values.

        This runs AFTER field-level redaction as a safety net.  It applies
        regex-based replacement patterns for every protected field declared in
        the policy, ensuring that any values missed by findings-driven
        sanitization are still caught.

        The sweep is conservative — it only replaces patterns that match
        known asset values, avoiding over-redaction.
        """
        if not self._policy:
            return text

        # For each protected field, build replacement patterns from
        # all known asset values of that field.
        for field_name in self._policy.protected_fields:
            replacement = self._safe_replacement(field_name)

            # Collect all known values for this field from assets
            # (the sanitizer doesn't hold assets directly, so we rely
            # on policy field_aliases + protected field patterns)
            if field_name == "collateral":
                # Collateral phrases: need whole-phrase replacement to avoid
                # leaving fragments like "张江研发大楼" or "在建产线抵押"
                text = self._sweep_collateral_phrases(text, replacement)

            elif field_name == "loan_amount":
                text = self._sweep_amount_patterns(text, replacement)

            elif field_name == "interest_rate":
                text = self._sweep_rate_patterns(text, replacement)

            elif field_name == "credit_rating":
                text = self._sweep_rating_patterns(text, replacement)

            elif field_name in {"diagnosis", "medication", "treatment"}:
                # Healthcare fields: replace known values from aliases
                text = self._sweep_known_values(text, field_name, replacement)

        return text

    def _sweep_amount_patterns(self, text: str, replacement: str) -> str:
        """Replace Chinese RMB amounts in text."""
        # Pattern: 人民币2.8亿元, 2.8亿元, 5000万元, 5,000万元, etc.
        # Must include unit (亿/万) to avoid matching bare numbers
        text = re.sub(
            r"(?:人民币|RMB|CNY)?\s*\d[\d,，]*\.?\d*\s*(?:亿|万)\s*(?:元|块|CNY|RMB)?",
            replacement,
            text,
        )
        return text

    def _sweep_rate_patterns(self, text: str, replacement: str) -> str:
        """Replace LPR-based interest rate patterns."""
        # LPR+120bp, LPR＋120bp, LPR+120个基点, LPR + 120 BP, etc.
        text = re.sub(
            r"LPR\s*[\+\-加]\s*\d{2,4}\s*(?:个)?\s*(?:BP|bp|基点|个基点)?",
            replacement,
            text,
            flags=re.IGNORECASE,
        )
        # Also replace "年化4.65%" style
        text = re.sub(
            r"（?年化\d+\.?\d*%\s*(?:）?\)?)?",
            replacement,
            text,
        )
        return text

    def _sweep_rating_patterns(self, text: str, replacement: str) -> str:
        """Replace credit rating patterns like AA+, BB, etc."""
        # Match standalone credit ratings: AA+, AA-, BB, CCC, etc.
        # Must be bounded to avoid matching abbreviations like "AA" in "AAPL"
        text = re.sub(
            r"(?<![A-Za-z])(?:AAA|AA|A|BBB|BB|B|CCC|CC|C)\s*[+＋\-－]?(?:级)?(?![A-Za-z])",
            replacement,
            text,
        )
        # Also match "信用评级AA+" "主体评级BB" patterns
        text = re.sub(
            r"(?:信用|主体)评级(?:AAA|AA|A|BBB|BB|B|CCC|CC|C)\s*[+＋\-－]?",
            replacement,
            text,
        )
        return text

    def _sweep_collateral_phrases(self, text: str, replacement: str) -> str:
        """Replace whole collateral/mortgage phrases to avoid fragment leakage.

        Instead of naive substring replacement that leaves "张江研发大楼"
        or "在建产线抵押", we match complete descriptive phrases.
        """
        # Pattern 1: "有X担保函和Y抵押" → "有相关抵押担保安排"
        text = re.sub(
            r"有[^。；，,；;]{0,30}?(?:担保函|抵押|质押)[^。；，,；;]{0,10}",
            replacement,
            text,
        )
        # Pattern 2: "以X研发大楼和Y质押" → "以相关抵押担保安排"
        text = re.sub(
            r"以[^。；，,；;]{0,30}?(?:研发大楼|担保函|抵押|质押|产线|厂房)[^。；，,；;]{0,10}",
            replacement,
            text,
        )
        # Pattern 3: Residual standalone collateral components
        # "在建产线抵押", "张江研发大楼"
        collateral_keywords = (
            "担保函|抵押|质押|研发大楼|产线|厂房|设备|股权质押|专利权|专利质押"
            "|担保物|抵押物|质权"
        )
        text = re.sub(
            rf"[^。；，,；;]*?(?:{collateral_keywords})[^。；，,；;]*",
            replacement,
            text,
        )
        return text

    def _sweep_known_values(self, text: str, field_name: str, replacement: str) -> str:
        """Replace values from policy field_aliases for a given field."""
        if not self._policy:
            return text
        alias_map = self._policy.field_aliases.get(field_name, {}) or {}
        # Sort by length descending for longest match first
        for canonical, aliases in sorted(alias_map.items(), key=lambda x: -len(x[0])):
            all_terms = [canonical] + list(aliases)
            for term in sorted(set(all_terms), key=len, reverse=True):
                if len(term) < 2:
                    continue
                # Only replace if it's a standalone term (bounded)
                try:
                    pattern = re.escape(term)
                    text = re.sub(pattern, replacement, text)
                except re.error:
                    continue
        return text

    # ---------- post-normalization (deduplicate repeated safe placeholders) ----------

    def _post_normalize_safe_text(self, text: str) -> str:
        """Policy-driven normalization to deduplicate repeated safe placeholders.

        Uses the policy's safe_replacements values to build normalization
        patterns, so it works across domains without hard-coded placeholder text.

        Examples:
            "相关治疗方案相关治疗方案治疗" → "相关治疗方案"
            "某企业某企业" → "某企业"
        """
        # Build dynamic normalization patterns from the policy's safe_replacements
        safe_values: set[str] = set()
        if self._policy and self._policy.safe_replacements:
            safe_values.update(self._policy.safe_replacements.values())

        for safe_value in sorted(safe_values, key=len, reverse=True):
            if len(safe_value) <= 2:
                continue
            escaped = re.escape(safe_value)
            # Collapse 2+ consecutive repetitions
            text = re.sub(rf"({escaped}){{2,}}", safe_value, text)

        # Generic fallback: collapse any 相关* placeholder repeated 2+ times
        text = re.sub(r"(相关[^\s。，,；;]{1,8})(?:\1)+", r"\1", text)

        return text

    # ---------- hint append ----------

    def _append_authorized_query_hint(self, text: str) -> str:
        hint = self._safe_hint()
        normalized = text.strip()
        if hint in normalized:
            return normalized
        if normalized.endswith("。"):
            return normalized + hint
        return normalized + "。" + hint
