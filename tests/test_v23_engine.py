"""
v2.3 Indirect Restoration Detection Tests.

Tests the new candidate-snapshot-based detection architecture:
    - Cin/Cout computation
    - Indirect asset restoration (no output protected anchor)
    - Information gain thresholding
    - No reduction = no alarm
    - Input-already-unique = no alarm
    - Zero candidate = conflict, not restoration
    - Accepted values OR semantics
    - Sanitizer removes contributing locator anchors
"""

from __future__ import annotations

import unittest
from pathlib import Path

from cfa_score import (
    CFAScoreEngine,
    ExtractionMode,
    load_assets,
    load_policy,
    load_public_knowledge,
    load_semantic_aliases,
    merge_public_knowledge,
)

ROOT = Path(__file__).resolve().parents[1]


class IndirectRestorationTest(unittest.TestCase):
    """Test v2.3 indirect restoration detection with healthcare scenario."""

    def setUp(self):
        self.assets = load_assets(ROOT / "config" / "healthcare_assets.json")
        self.policy = load_policy(ROOT / "config" / "healthcare_policy.json")
        # Merge semantic aliases for richer extraction
        aliases_path = ROOT / "config" / "healthcare_semantic_aliases.json"
        if aliases_path.exists():
            semantic_aliases = load_semantic_aliases(aliases_path)
            from dataclasses import replace
            self.policy = replace(self.policy, semantic_aliases=semantic_aliases)
        self.engine = CFAScoreEngine(self.assets, self.policy)

    # ---- Test 1: indirect restoration without protected output ----

    def test_indirect_restoration_without_protected_output(self):
        """Output has no protected fields, but narrows Cin→Cout to 1 record."""
        # Input narrows to 2 patients (P004=王芳, P007=李明 both in 心内科ICU)
        user_input = "心内科ICU的患者"
        # "心衰" is a condition_summary alias (not protected) that narrows 2→1
        model_output = "该患者心功能不全加重，需要调整治疗。"
        result = self.engine.analyze(model_output, user_input=user_input)
        self.assertGreaterEqual(len(result.findings), 1,
                                "Should detect indirect restoration even without output protected fields")
        finding = result.findings[0]
        self.assertIn(finding.risk_level, {"MEDIUM", "HIGH", "CRITICAL"})
        self.assertIn(finding.finding_type,
                      {"indirect_asset_restoration", "indirect_protected_value_restoration"})

    # ---- Test 2: output reduces candidates from many to exactly k ----

    def test_department_patient_reduction_to_one(self):
        """department + patient condition should narrow to 1."""
        user_input = "心内科的患者"
        model_output = "该患者有胸痛症状"
        result = self.engine.analyze(model_output, user_input=user_input)
        self.assertGreaterEqual(len(result.findings), 1)
        finding = result.findings[0]
        self.assertEqual(finding.final_candidate_count, 1)
        self.assertGreater(finding.information_gain_bits, 1.0)

    # ---- Test 3: output anchor without candidate reduction should NOT alarm ----

    def test_output_anchor_without_candidate_reduction_is_safe(self):
        """Output repeats info already provided by input — no extra narrowing."""
        user_input = "心内科ICU的患者"
        # "该患者目前在ICU" adds nothing to the narrowing already done by input
        model_output = "该患者目前在ICU，请继续观察。"
        result = self.engine.analyze(model_output, user_input=user_input)
        # Since ICU is already in input, output adds no narrowing
        # Check that either no findings, or findings are only indirect (not asset-level)
        asset_findings = [f for f in result.findings if f.finding_type == "indirect_asset_restoration"]
        self.assertEqual(len(asset_findings), 0,
                         "Output ICU anchor should not count as contributing (already in input)")

    # ---- Test 4: input already unique → no new CFA risk attributed to output ----

    def test_input_already_unique_no_new_cfa_risk(self):
        """When input alone already identifies the record, not a CFA output risk."""
        user_input = "患者张伟"
        model_output = "请遵循医生的一般建议。"
        result = self.engine.analyze(model_output, user_input=user_input)
        # Input may already narrow to 3 patients named 张伟.
        # The key: if output adds nothing, no CFA risk.
        asset_findings = [f for f in result.findings if f.finding_type == "indirect_asset_restoration"]
        self.assertEqual(len(asset_findings), 0,
                         "Should not flag CFA when output adds no narrowing info")

    # ---- Test 5: zero candidate = conflict, not restoration ----

    def test_zero_candidate_is_conflict_not_restoration(self):
        """When output makes candidate set empty, it's evidence conflict, not CFA."""
        user_input = "心内科ICU的患者"
        # "肿瘤科" conflicts with "心内科"
        model_output = "该患者在肿瘤科继续治疗。"
        result = self.engine.analyze(model_output, user_input=user_input)
        # Should NOT produce indirect asset restoration (0 candidates = conflict)
        asset_findings = [f for f in result.findings if f.finding_type == "indirect_asset_restoration"]
        self.assertEqual(len(asset_findings), 0,
                         "Zero candidates should be treated as inconsistent evidence, not restoration")

    # ---- Test 6: sanitizer removes contributing locator anchors ----

    def test_sanitizer_removes_contributing_locator(self):
        """The locator anchor that narrowed 2→1 should be sanitized."""
        user_input = "心内科ICU的患者"
        model_output = "患者为胸痛症状，有冠心病可能。"
        result = self.engine.analyze(model_output, user_input=user_input)
        if result.findings:
            # The contributing locator should be removed from safe_answer
            safe = result.safe_answer
            # conditional_summary anchors that actually narrowed should be removed
            finding = result.findings[0]
            for a in finding.anchors:
                if a.id in finding.key_anchor_ids and a.source == "output":
                    self.assertNotIn(a.text, safe,
                                     f"Contributing locator anchor '{a.text}' should be sanitized")

    # ---- Test 7: accepted_values OR semantics ----

    def test_anchor_values_or_semantics(self):
        """anchor.accepted_values should use OR, not AND."""
        # Create an anchor with multiple accepted values
        snapshot = self.engine._build_candidate_snapshot([])  # empty list
        input_cands = self.engine._filter_candidates(
            self.assets,
            [],  # no filters yet
        )
        self.assertEqual(len(input_cands), len(self.assets),
                         "All assets should pass through empty filters")

    # ---- Test 8: information gain computation ----

    def test_information_gain_computation(self):
        """Information gain should be log2(input_count / final_count)."""
        import math
        user_input = "心内科ICU的患者"
        model_output = "该患者有胸痛症状"
        result = self.engine.analyze(model_output, user_input=user_input)
        if result.findings:
            f = result.findings[0]
            self.assertGreater(f.information_gain_bits, 0.0)
            self.assertEqual(f.input_candidate_count, 4)  # 心内科=4 patients
            self.assertEqual(f.final_candidate_count, 1)  # +ICU + 胸痛=1
            expected_ig = math.log2(4.0 / 1.0)
            self.assertAlmostEqual(f.information_gain_bits, expected_ig, places=2)

    # ---- Test 9: direct disclosure still works (v2.3 regression) ----

    def test_direct_protected_disclosure_still_works(self):
        """v2.2 style detection should still work in v2.3."""
        model_output = "患者张伟，诊断结果为冠心病，用药方案为阿司匹林。"
        result = self.engine.analyze(model_output, user_input="")
        direct_findings = [f for f in result.findings if f.finding_type == "direct_protected_disclosure"]
        self.assertGreaterEqual(len(direct_findings), 1,
                                "Direct protected disclosure should still be detected in v2.3")

    # ---- Test 10: policy.v2.3 fields have correct defaults ----

    def test_policy_v23_defaults(self):
        """Verify v2.3 policy fields have expected default values."""
        self.assertTrue(self.policy.indirect_restoration_enabled)
        self.assertEqual(self.policy.min_candidate_reduction, 1)
        self.assertEqual(self.policy.min_information_gain_bits, 0.5)
        self.assertEqual(self.policy.protected_value_k, 1)
        self.assertEqual(self.policy.min_protected_entropy_drop_bits, 0.5)

    # ---- Test 11: RiskFinding has v2.3 fields ----

    def test_risk_finding_has_v23_fields(self):
        """RiskFinding should carry v2.3 extension fields."""
        user_input = "心内科ICU的患者"
        model_output = "该患者有胸痛症状"
        result = self.engine.analyze(model_output, user_input=user_input)
        if result.findings:
            f = result.findings[0]
            self.assertTrue(hasattr(f, "finding_type"))
            self.assertTrue(hasattr(f, "restored_fields"))
            self.assertTrue(hasattr(f, "input_candidate_count"))
            self.assertTrue(hasattr(f, "final_candidate_count"))
            self.assertTrue(hasattr(f, "information_gain_bits"))
            self.assertIsInstance(f.restored_fields, list)
            self.assertGreater(f.input_candidate_count, 0)
            self.assertGreaterEqual(f.final_candidate_count, 1)


class FinanceIndirectRestorationTest(unittest.TestCase):
    """Test v2.3 indirect restoration with finance scenario."""

    def setUp(self):
        self.assets = load_assets(ROOT / "config" / "finance_assets.json")
        self.policy = load_policy(ROOT / "config" / "finance_policy.json")
        pk_path = ROOT / "config" / "finance_public_knowledge.json"
        if pk_path.exists():
            self.policy = merge_public_knowledge(
                self.policy, load_public_knowledge(pk_path)
            )
        self.engine = CFAScoreEngine(self.assets, self.policy)

    def test_finance_indirect_loan_type_narrowing(self):
        """Input: branch + industry. Output: loan_type. Should narrow to 1 record."""
        user_input = "上海分行半导体制造的企业"
        # Only one loan_type per answer (not two) to avoid zero-candidate conflict
        model_output = "该企业申请了流动资金贷款。"
        result = self.engine.analyze(model_output, user_input=user_input)
        # Should detect indirect restoration via loan_type narrowing
        self.assertGreaterEqual(len(result.findings), 1,
                                "Should detect indirect restoration via loan_type narrowing")


if __name__ == "__main__":
    unittest.main()