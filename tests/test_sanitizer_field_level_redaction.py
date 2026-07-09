from __future__ import annotations

import unittest

from cfa_score.models import Anchor, FieldPolicy, RiskFinding
from cfa_score.sanitizer import AnswerSanitizer


def fake_finding(anchors, *, restored_fields=None, key_anchor_ids=None):
    return RiskFinding(
        target_asset_id="A001",
        target_asset_name="测试资产",
        risk_level="HIGH",
        score=0.9,
        reason="测试",
        restored_fact="测试事实",
        anchors=anchors,
        reduction_chain=[],
        minimal_combinations=[],
        restored_fields=restored_fields or [],
        key_anchor_ids=key_anchor_ids or [],
    )


DEFAULT_POLICY = FieldPolicy(
    protected_fields=["component_version", "disposition_status", "system_name", "remote_entry"],
    identifier_fields=["system_name", "business_domain", "environment", "function_category"],
    quasi_identifier_fields=[],
    field_order=[],
    field_labels={
        "component_version": "组件版本",
        "disposition_status": "处置状态",
        "system_name": "系统名称",
        "remote_entry": "远程入口",
    },
    field_weights={},
    field_aliases={
        "disposition_status": {
            "仍处于回滚窗口": ["回滚窗口", "回滚窗口内", "还在回滚窗口"],
        },
    },
    public_rules=[],
    safe_replacements={
        "component_version": "受影响版本",
        "disposition_status": "相关处置状态",
        "system_name": "相关对象",
        "remote_entry": "远程入口",
    },
)


class SanitizerFieldLevelRedactionTest(unittest.TestCase):
    """Tests for v2.4 field-level redaction that replaces canonical/alias/semantic
    variants instead of merely anchor.text."""

    def setUp(self):
        self.sanitizer = AnswerSanitizer(DEFAULT_POLICY)

    # ------------------------------------------------------------------
    # 1. canonical value replacement
    # ------------------------------------------------------------------

    def test_replaces_canonical_value_even_when_anchor_text_differs(self):
        anchor = Anchor(
            id="a1",
            field_name="component_version",
            field_label="组件版本",
            text="那个受影响的小版本",
            canonical_value="xz-utils 5.6.1",
            start=-1,
            end=-1,
            anchor_type="field_value",
            protected=True,
            source="output",
        )

        finding = fake_finding([anchor], restored_fields=["component_version"])
        text = "该系统使用 xz-utils 5.6.1，仍需处置。"
        safe = self.sanitizer.make_safe_answer(text, [finding])

        self.assertNotIn("xz-utils 5.6.1", safe)
        self.assertIn("受影响版本", safe)

    # ------------------------------------------------------------------
    # 2. version family expression replacement
    # ------------------------------------------------------------------

    def test_replaces_version_family_expression(self):
        anchor = Anchor(
            id="a1",
            field_name="component_version",
            field_label="组件版本",
            text="受影响版本",
            canonical_value="xz-utils 5.6.1",
            start=-1,
            end=-1,
            anchor_type="field_value",
            protected=True,
            source="output",
        )

        finding = fake_finding([anchor], restored_fields=["component_version"])

        text = "该对象使用的是 5.6 系列中受影响的小版本。"
        safe = self.sanitizer.make_safe_answer(text, [finding])

        self.assertNotIn("5.6 系列", safe)
        self.assertTrue("受影响版本" in safe or "已省略" in safe)

    # ------------------------------------------------------------------
    # 3. field alias replacement
    # ------------------------------------------------------------------

    def test_replaces_field_aliases(self):
        anchor = Anchor(
            id="a1",
            field_name="disposition_status",
            field_label="处置状态",
            text="尚未完成处置",
            canonical_value="仍处于回滚窗口",
            start=-1,
            end=-1,
            anchor_type="field_value",
            protected=True,
            source="output",
        )

        finding = fake_finding([anchor], restored_fields=["disposition_status"])

        text = "该系统还在回滚窗口内。"
        safe = self.sanitizer.make_safe_answer(text, [finding])

        self.assertNotIn("回滚窗口", safe)

    # ------------------------------------------------------------------
    # 4. fallback to safe summary when unresolved
    # ------------------------------------------------------------------

    def test_falls_back_to_safe_summary_when_unresolved(self):
        anchor = Anchor(
            id="a1",
            field_name="component_version",
            field_label="组件版本",
            text="语义锚点",
            canonical_value="xz-utils 5.6.1",
            start=-1,
            end=-1,
            anchor_type="semantic",
            protected=True,
            source="output",
            match_type="semantic",
        )

        finding = fake_finding([anchor], restored_fields=["component_version"])

        text = "这个回答通过隐晦方式暗示了内部版本。"
        safe = self.sanitizer.make_safe_answer(text, [finding])

        self.assertIn("具体内容已省略", safe)

    # ------------------------------------------------------------------
    # 5. span-based replacement works for valid spans
    # ------------------------------------------------------------------

    def test_replaces_by_span_when_valid(self):
        text = "使用的是 xz-utils 5.6.1 这个版本。"

        anchor = Anchor(
            id="a1",
            field_name="component_version",
            field_label="组件版本",
            text="xz-utils 5.6.1",
            canonical_value="xz-utils 5.6.1",
            start=text.index("xz-utils 5.6.1"),
            end=text.index("xz-utils 5.6.1") + len("xz-utils 5.6.1"),
            anchor_type="field_value",
            protected=True,
            source="output",
        )

        finding = fake_finding([anchor], restored_fields=["component_version"])

        safe = self.sanitizer.make_safe_answer(text, [finding])
        self.assertNotIn("xz-utils 5.6.1", safe)
        self.assertIn("受影响版本", safe)

    # ------------------------------------------------------------------
    # 6. already-safe placeholders are skipped
    # ------------------------------------------------------------------

    def test_skips_already_safe_placeholder_anchors(self):
        anchor = Anchor(
            id="a1",
            field_name="disposition_status",
            field_label="处置状态",
            text="相关处置状态",
            canonical_value="相关处置状态",
            start=-1,
            end=-1,
            anchor_type="field_value",
            protected=True,
            source="output",
        )

        finding = fake_finding([anchor], restored_fields=["disposition_status"])

        text = "该系统的处置状态为相关处置状态，暂无风险。"
        safe = self.sanitizer.make_safe_answer(text, [finding])

        # Should still contain the placeholder since it was skipped.
        self.assertIn("相关处置状态", safe)

    # ------------------------------------------------------------------
    # 7. X-mode replaces with "X"
    # ------------------------------------------------------------------

    def test_x_mode_replaces_with_x(self):
        anchor = Anchor(
            id="a1",
            field_name="system_name",
            field_label="系统名称",
            text="某资产名称",
            canonical_value="星箭链路健康监测平台",
            start=-1,
            end=-1,
            anchor_type="field_value",
            protected=True,
            source="output",
        )

        finding = fake_finding([anchor], restored_fields=["system_name"])

        text = "星箭链路健康监测平台需要升级。"
        x_result = self.sanitizer.make_x_replaced(text, [finding])

        self.assertIn("X", x_result)
        self.assertNotIn("星箭链路健康监测平台", x_result)


if __name__ == "__main__":
    unittest.main()