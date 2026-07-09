from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence

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

    # ----------------------------------------------------------------

    def make_x_replaced(self, raw_answer: str, findings: Sequence[RiskFinding]) -> str:
        text = raw_answer
        if findings:
            text = self._rewrite_known_case_for_x(text)
            for anchor in self._dangerous_output_anchors(findings):
                text = self._replace_literal(text, anchor.text, "X")
        return self._append_authorized_query_hint(text)

    def make_safe_answer(self, raw_answer: str, findings: Sequence[RiskFinding]) -> str:
        text = raw_answer
        if findings:
            text = self._rewrite_known_case_for_safe(text)
            for anchor in self._dangerous_output_anchors(findings):
                replacement = self._safe_replacement(anchor.field_name)
                text = self._replace_literal(text, anchor.text, replacement)
        else:
            # Even without findings, replace asset IDs that appear in text
            text = self._replace_asset_ids(text)
        return self._append_authorized_query_hint(text)

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

    # ---------- dangerous anchors ----------

    def _dangerous_output_anchors(self, findings: Sequence[RiskFinding]) -> List[Anchor]:
        """Collect dangerous output anchors for sanitization.

        v2.3: Also includes contributing locator anchors (quasi-identifiers
        that actually narrowed the candidate set) in addition to direct
        protected field disclosures.
        """
        anchors: List[Anchor] = []
        seen = set()
        id_fields = self._identifier_field_names()

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
                if anchor.text in {"受影响版本", "处置未完成", "回滚完成前"}:
                    continue

                # Direct sensitive disclosure
                is_display = (
                    self._policy is not None
                    and anchor.field_name == self._policy.display_field
                )
                is_id_field = anchor.field_name in id_fields
                is_direct_sensitive = anchor.protected or is_id_field or is_display

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

    def _append_authorized_query_hint(self, text: str) -> str:
        hint = self._safe_hint()
        normalized = text.strip()
        if hint in normalized:
            return normalized
        if normalized.endswith("。"):
            return normalized + hint
        return normalized + "。" + hint