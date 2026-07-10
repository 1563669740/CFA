from __future__ import annotations

import hashlib
import itertools
import math
import re
import unicodedata
from itertools import combinations
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .anchor_verifier import AnchorVerifier
from .deepseek import DeepSeekClient
from .extractor import RuleBasedAnchorExtractor
from .llm_extractor import LLMSemanticAnchorExtractor
from .claim_builder import ClaimBuilder
from .models import (
    AnalysisResult,
    Anchor,
    AssetFact,
    AuditRecord,
    CandidateSnapshot,
    DisclosureClaim,
    FieldPolicy,
    ReductionStep,
    RestorationDecision,
    RiskFinding,
)
from .normalizers import FieldNormalizer
from .sanitizer import AnswerSanitizer
from .semantic_index import SemanticIndex


# ---------------------------------------------------------------------------
# Fallback safe answer when rewriting cannot eliminate risk
# ---------------------------------------------------------------------------

_FALLBACK_SAFE_ANSWER = (
    "该问题涉及可能受限的内部信息。"
    "请通过授权系统查询，或联系具备相应权限的人员处理。"
)

# ---------------------------------------------------------------------------
# v2.7 — Schema-generic defaults for slot / confirmation detection.
# These are fallbacks ONLY when the policy does not provide its own terms.
# ---------------------------------------------------------------------------

_DEFAULT_CONFIRMATION_TERMS = (
    "是", "对", "正确", "确实", "明确", "理解正确", "可以明确", "您的理解是正确",
    "会议决定", "决定", "暂不审议", "后续安排", "等待", "待", "正式上报", "正式报", "上报和审议",
)

_DEFAULT_REFUSAL_TERMS = (
    "无法确认", "不能确认", "无法核实", "不能核实", "无法提供", "不能提供", "无权", "无权限",
    "授权系统", "授权业务系统", "按权限查询", "联系具备相应权限", "不便透露", "不予披露",
)

_DEFAULT_COMMON_FRAGMENTS: Set[str] = {
    "项目", "议题", "会议", "决定", "审议", "正式", "上报", "等待", "完成", "情况", "后续", "安排",
    "可以", "明确", "根据", "提供", "内容", "相关", "您的", "理解", "正确", "目前", "需要", "进行",
}

_DEFAULT_SLOT_QUERY_TERMS = (
    "多少", "几分", "多少分", "是什么", "为多少", "是多少", "哪项", "哪个", "多久",
)

_DEFAULT_SLOT_ATTRIBUTE_TERMS = (
    "成绩", "考试", "分数", "结果", "结论", "期限", "金额", "预算", "经费", "比例",
    "数量", "容量", "级别", "密级", "类别", "属性", "要求", "安排", "期限",
)

_DEFAULT_SLOT_CONNECTOR_TERMS = (
    "请", "问", "请问", "请告知", "告知", "查询", "查一下", "一下", "根据", "信息",
    "同志", "的", "为", "是", "了", "吗", "呢", "啊", "请给我", "给我",
)

_DEFAULT_SLOT_OUTPUT_VALUE_PATTERN = r"\d+(?:\.\d+)?\s*(?:亿元|万元|元|%|％|年|个月|月|日|天|小时|分钟|分|人|台|条|项|个)?"

_DEFAULT_SLOT_VALUE_UNIT_HINTS: Dict[str, str] = {
    "成绩": "分",
    "考试": "分",
    "分数": "分",
    "预算": "元",
    "经费": "元",
    "金额": "元",
    "期限": "年",
    "脱密": "年",
}


# ---------------------------------------------------------------------------
# Extraction mode enum
# ---------------------------------------------------------------------------

class ExtractionMode:
    RULE_ONLY = "rule_only"
    RULE_PLUS_LLM = "rule_plus_llm"


# ---------------------------------------------------------------------------
# Anchor Merger
# ---------------------------------------------------------------------------

class AnchorMerger:
    """Merges rule-based and LLM anchors.

    Design principles:
    1. input and output are NEVER deduplicated against each other.
    2. Same span, same field+value, rule + LLM → merged into one.
    3. Different spans → all preserved (for highlighting / audit).
    4. Rule anchor serves as base; LLM confidence, reason, accepted_values
       are merged in.
    5. ``collapse_for_reasoning`` provides per-(source, field, value)
       deduplication for scoring, preventing double-counting.
    """

    _MATCH_TYPE_PRIORITY = {
        "exact": 6,
        "alias": 5,
        "semantic": 4,
        "partial": 3,
        "inferred": 2,
        "ambiguous": 1,
    }

    @classmethod
    def merge(
        cls,
        rule_anchors: list[Anchor],
        llm_anchors: list[Anchor],
    ) -> list[Anchor]:
        """Merge rule + LLM anchors.  input/output never collide."""
        merged: dict[tuple, Anchor] = {}

        for extractor, anchors in (
            ("rule", rule_anchors),
            ("llm", llm_anchors),
        ):
            for original in anchors:
                anchor = _deepcopy_anchor(original)
                cls._initialize_provenance(anchor, extractor)
                key = cls._instance_key(anchor)
                if key not in merged:
                    merged[key] = anchor
                    continue
                merged[key] = cls._merge_duplicate(merged[key], anchor)

        result = list(merged.values())
        result.sort(key=cls._sort_key)
        return result

    @classmethod
    def _instance_key(cls, anchor: Anchor) -> tuple:
        """Uniquely identify a text occurrence."""
        return (
            anchor.source,
            anchor.field_name,
            cls._primary_value_key(anchor),
            anchor.start,
            anchor.end,
            anchor.inferred,
        )

    @staticmethod
    def _primary_value_key(anchor: Anchor) -> tuple:
        if anchor.canonical_value:
            return ("canonical", anchor.canonical_value)
        return ("accepted", tuple(sorted(set(anchor.accepted_values))))

    @classmethod
    def _merge_duplicate(cls, base: Anchor, incoming: Anchor) -> Anchor:
        """Merge two anchors at the exact same (source, field, value, span, inferred)."""
        merged = _deepcopy_anchor(base)

        merged.extractor_sources = cls._ordered_union(
            base.extractor_sources, incoming.extractor_sources
        )
        merged.merged_anchor_ids = cls._ordered_union(
            base.merged_anchor_ids, incoming.merged_anchor_ids
        )
        merged.accepted_values = cls._ordered_union(
            list(base.accepted_values), list(incoming.accepted_values)
        )
        merged.confidence = max(base.confidence, incoming.confidence)
        merged.match_type = cls._stronger_match_type(base.match_type, incoming.match_type)
        merged.evidence = cls._merge_text_values(base.evidence, incoming.evidence)
        merged.llm_reason = cls._merge_text_values(base.llm_reason, incoming.llm_reason)
        merged.protected = base.protected or incoming.protected

        if not merged.canonical_value:
            merged.canonical_value = incoming.canonical_value
        if not merged.text:
            merged.text = incoming.text
        if not merged.field_label:
            merged.field_label = incoming.field_label
        if not merged.source_anchor_id:
            merged.source_anchor_id = incoming.source_anchor_id

        return merged

    @staticmethod
    def _initialize_provenance(anchor: Anchor, extractor: str) -> None:
        if extractor not in anchor.extractor_sources:
            anchor.extractor_sources.append(extractor)
        if anchor.id not in anchor.merged_anchor_ids:
            anchor.merged_anchor_ids.append(anchor.id)

    @classmethod
    def _stronger_match_type(cls, left: str, right: str) -> str:
        left_score = cls._MATCH_TYPE_PRIORITY.get(left, 0)
        right_score = cls._MATCH_TYPE_PRIORITY.get(right, 0)
        return left if left_score >= right_score else right

    @staticmethod
    def _ordered_union(left, right) -> list:
        result: list = []
        seen: set = set()
        for value in [*left, *right]:
            if not value or value in seen:
                continue
            seen.add(value)
            result.append(value)
        return result

    @staticmethod
    def _merge_text_values(left: str, right: str) -> str:
        values: list[str] = []
        for v in (left, right):
            v = v.strip()
            if v and v not in values:
                values.append(v)
        return " | ".join(values)

    @staticmethod
    def _sort_key(anchor: Anchor) -> tuple:
        return (
            0 if anchor.source == "input" else 1,
            anchor.start,
            anchor.end,
            anchor.field_name,
            anchor.canonical_value,
            anchor.inferred,
        )

    # ---------------------------------------------------------------
    # Reasoning collapse — per-(source, field, value) dedup for scoring
    # ---------------------------------------------------------------

    @classmethod
    def collapse_for_reasoning(cls, anchors: list[Anchor]) -> list[Anchor]:
        """Deduplicate anchors for scoring: same source+field+value → keep best."""
        selected: dict[tuple, Anchor] = {}
        for anchor in anchors:
            key = cls._claim_key(anchor)
            current = selected.get(key)
            if current is None:
                selected[key] = anchor
                continue
            if cls._reasoning_priority(anchor) > cls._reasoning_priority(current):
                selected[key] = anchor
        result = list(selected.values())
        result.sort(key=cls._sort_key)
        return result

    @staticmethod
    def _claim_key(anchor: Anchor) -> tuple:
        values = set(anchor.accepted_values)
        if anchor.canonical_value:
            values.add(anchor.canonical_value)
        return (
            anchor.source,
            anchor.field_name,
            tuple(sorted(values)),
            anchor.inferred,
        )

    @classmethod
    def _reasoning_priority(cls, anchor: Anchor) -> tuple:
        return (
            cls._MATCH_TYPE_PRIORITY.get(anchor.match_type, 0),
            anchor.confidence,
            len(anchor.text),
        )


def _deepcopy_anchor(anchor: Anchor) -> Anchor:
    """Shallow copy sufficient for Anchor (all fields are primitives/lists)."""
    return Anchor(
        id=anchor.id,
        field_name=anchor.field_name,
        field_label=anchor.field_label,
        text=anchor.text,
        canonical_value=anchor.canonical_value,
        start=anchor.start,
        end=anchor.end,
        anchor_type=anchor.anchor_type,
        protected=anchor.protected,
        inferred=anchor.inferred,
        evidence=anchor.evidence,
        source_anchor_id=anchor.source_anchor_id,
        source=anchor.source,
        match_type=anchor.match_type,
        confidence=anchor.confidence,
        llm_reason=anchor.llm_reason,
        accepted_values=list(anchor.accepted_values),
        extractor_sources=list(anchor.extractor_sources),
        merged_anchor_ids=list(anchor.merged_anchor_ids),
    )


# ---------------------------------------------------------------------------
# CFA Score Engine (v2.3 — indirect restoration detection)
# ---------------------------------------------------------------------------

class CFAScoreEngine:
    """Combination fact restoration risk detector.

    Supports three extraction modes:

    Mode 1 – Rule Only (baseline):
        RuleBasedAnchorExtractor only. Fast, deterministic, high precision.

    Mode 2 – Rule + LLM Extraction:
        Rule anchors + LLM semantic anchors. Higher recall.

    Mode 3 – Rule + LLM Extraction + LLM Rewrite + Secondary Check:
        Mode 2 + LLM-based safe rewriting + second CFA-Score pass.

    v2.3: Indirect restoration detection — risk can now be triggered
    even without output protected anchors, as long as output contributes
    new locator information that uniquely narrows the candidate set.
    """

    def __init__(
        self,
        assets: Sequence[AssetFact],
        policy: FieldPolicy,
        *,
        mode: str = ExtractionMode.RULE_ONLY,
        deepseek_client: Optional[DeepSeekClient] = None,
        trace: Any | None = None,
    ):
        if not assets:
            raise ValueError("assets must not be empty")
        self.assets = list(assets)
        self.policy = policy
        self._mode = mode
        self._trace = trace

        self.rule_extractor = RuleBasedAnchorExtractor(policy)
        self.sanitizer = AnswerSanitizer(policy)
        self.semantic_index = SemanticIndex(policy, assets)

        self._llm_extractor: Optional[LLMSemanticAnchorExtractor] = None
        self._verifier: Optional[AnchorVerifier] = None
        self._llm_rewriter_client: Optional[DeepSeekClient] = None

        if mode in (ExtractionMode.RULE_PLUS_LLM,) and deepseek_client is not None:
            self._init_llm_components(deepseek_client)

    # ------------------------------------------------------------------
    # Mode management
    # ------------------------------------------------------------------

    def enable_mode_2(self, deepseek_client: DeepSeekClient) -> None:
        self._mode = ExtractionMode.RULE_PLUS_LLM
        self._init_llm_components(deepseek_client)

    def enable_mode_3(self, deepseek_client: DeepSeekClient) -> None:
        self._mode = ExtractionMode.RULE_PLUS_LLM
        self._llm_rewriter_client = deepseek_client
        self._init_llm_components(deepseek_client)

    def _init_llm_components(self, client: DeepSeekClient) -> None:
        self._llm_extractor = LLMSemanticAnchorExtractor(
            client=client,
            policy=self.policy,
            semantic_index=self.semantic_index,
            trace=self._trace,
        )
        self._verifier = AnchorVerifier(
            policy=self.policy,
            assets=self.assets,
            semantic_index=self.semantic_index,
        )

    @property
    def mode(self) -> str:
        return self._mode

    # ==================================================================
    # v2.7 — Policy-driven term helpers (slot / confirmation)
    # ==================================================================

    def _effective_confirmation_terms(self) -> Tuple[str, ...]:
        """Get confirmation terms: policy first, then defaults."""
        if self.policy.confirmation_detection_enabled and self.policy.confirmation_terms:
            return tuple(self.policy.confirmation_terms)
        return _DEFAULT_CONFIRMATION_TERMS

    def _effective_refusal_terms(self) -> Tuple[str, ...]:
        """Get refusal terms: policy first, then defaults."""
        if self.policy.confirmation_detection_enabled and self.policy.refusal_terms:
            return tuple(self.policy.refusal_terms)
        return _DEFAULT_REFUSAL_TERMS

    def _effective_common_fragments(self) -> Set[str]:
        """Get common text fragments: policy first, then defaults."""
        if self.policy.confirmation_detection_enabled and self.policy.common_text_fragments:
            return set(self.policy.common_text_fragments)
        return _DEFAULT_COMMON_FRAGMENTS

    def _effective_slot_query_terms(self) -> Tuple[str, ...]:
        """Get slot query terms: policy first, then defaults."""
        if self.policy.slot_detection_enabled and self.policy.slot_query_terms:
            return tuple(self.policy.slot_query_terms)
        return _DEFAULT_SLOT_QUERY_TERMS

    def _effective_slot_attribute_terms(self) -> Tuple[str, ...]:
        """Get slot attribute terms: policy first, then defaults."""
        if self.policy.slot_detection_enabled and self.policy.slot_attribute_terms:
            return tuple(self.policy.slot_attribute_terms)
        return _DEFAULT_SLOT_ATTRIBUTE_TERMS

    def _effective_slot_connector_terms(self) -> Tuple[str, ...]:
        """Get slot connector terms: policy first, then defaults."""
        if self.policy.slot_detection_enabled and self.policy.slot_connector_terms:
            return tuple(self.policy.slot_connector_terms)
        return _DEFAULT_SLOT_CONNECTOR_TERMS

    def _effective_slot_output_value_re(self) -> re.Pattern:
        """Get slot output value regex: policy first, then defaults."""
        if self.policy.slot_detection_enabled and self.policy.slot_output_value_pattern:
            return re.compile(self.policy.slot_output_value_pattern)
        return re.compile(_DEFAULT_SLOT_OUTPUT_VALUE_PATTERN)

    def _effective_slot_value_unit_hints(self) -> Dict[str, str]:
        """Get slot value unit hints: policy first, then defaults."""
        if self.policy.slot_detection_enabled and self.policy.slot_value_unit_hints:
            return dict(self.policy.slot_value_unit_hints)
        return dict(_DEFAULT_SLOT_VALUE_UNIT_HINTS)

    def _allow_confirmation_detection(self) -> bool:
        """Check if confirmation detection is applicable for this policy.

        Generic gate: confirmation detection is allowed when:
        1. policy.confirmation_detection_enabled OR protected_fields contain
           any field with sensitive text content.
        2. For backward compat, always allow if secret_content/secret_summary in protected_fields.
        """
        if self.policy.confirmation_detection_enabled:
            return True
        # Backward compat: secret_content/secret_summary based policies
        has_secret = {"secret_content", "secret_summary"} & set(self.policy.protected_fields or [])
        return bool(has_secret)

    def _allow_slot_detection(self) -> bool:
        """Check if slot-fill detection is applicable for this policy.

        Generic gate: slot detection is allowed when:
        1. policy.slot_detection_enabled OR
        2. protected_fields contain secret_content/secret_summary (backward compat)
        3. OR any protected field that is NOT an identifier (e.g., score, amount, grade)
        """
        if self.policy.slot_detection_enabled:
            return True
        # Backward compat: secret_content/secret_summary based policies
        has_secret = {"secret_content", "secret_summary"} & set(self.policy.protected_fields or [])
        if has_secret:
            return True
        # Generic: any non-identifier protected field can be a slot target
        slot_targets = self.policy.field_slot_protected_fields()
        return len(slot_targets) > 0

    # ==================================================================
    # v2.3 — Anchor value helpers
    # ==================================================================

    @staticmethod
    def _anchor_values(anchor: Anchor) -> set[str]:
        """Return all accepted values for an anchor (OR semantics)."""
        values: set[str] = set()
        if anchor.canonical_value:
            values.add(anchor.canonical_value)
        values.update(v for v in anchor.accepted_values if v)
        return values

    def _asset_matches_anchor(self, asset: AssetFact, anchor: Anchor) -> bool:
        """Check whether a fact row matches an anchor.

        For inferred anchors, source anchor must also match.
        accepted_values are treated with OR semantics.
        """
        # For inferred anchors: source anchor must also match
        if anchor.inferred and anchor.source_anchor_id:
            # We need the anchor_by_id dict — use the standalone version that
            # accepts the anchor dict as a separate parameter
            pass
        # Multi-value OR match
        if anchor.accepted_values:
            asset_value = asset.get(anchor.field_name)
            return asset_value in anchor.accepted_values
        # Exact canonical match
        return asset.get(anchor.field_name) == str(anchor.canonical_value)

    def _asset_matches_anchor_with_id(
        self,
        asset: AssetFact,
        anchor: Anchor,
        anchor_by_id: Dict[str, Anchor],
    ) -> bool:
        """Like _asset_matches_anchor but with access to anchor_by_id for inferred anchors."""
        if anchor.inferred and anchor.source_anchor_id:
            source_anchor = anchor_by_id.get(anchor.source_anchor_id)
            if source_anchor is None:
                return False
            if asset.get(source_anchor.field_name) != str(source_anchor.canonical_value):
                return False

        accepted = self._anchor_values(anchor)
        if accepted:
            asset_value = asset.get(anchor.field_name)
            return asset_value in accepted

        return asset.get(anchor.field_name) == str(anchor.canonical_value)

    # ==================================================================
    # v2.3 — Candidate filtering (AND across anchors, OR within anchor)
    # ==================================================================

    def _filter_candidates(
        self,
        candidates: list[AssetFact],
        anchors: list[Anchor],
        anchor_by_id: Optional[Dict[str, Anchor]] = None,
    ) -> list[AssetFact]:
        """Apply a list of anchors as AND filters."""
        if anchor_by_id is None:
            anchor_by_id = {a.id: a for a in anchors}
        remaining = list(candidates)
        for anchor in anchors:
            if not self._anchor_values(anchor):
                continue
            remaining = [
                asset
                for asset in remaining
                if self._asset_matches_anchor_with_id(asset, anchor, anchor_by_id)
            ]
            if not remaining:
                break
        return remaining

    # ==================================================================
    # v2.3 — Candidate snapshot
    # ==================================================================

    def _build_candidate_snapshot(self, anchors: Sequence[Anchor]) -> CandidateSnapshot:
        """Compute Cin, Cout, and contributing output anchors."""
        all_anchors = list(anchors)
        anchor_by_id = {a.id: a for a in all_anchors}

        input_anchors = [a for a in all_anchors if a.source == "input"]
        output_anchors = [
            a for a in all_anchors
            if a.source == "output" and not a.inferred
        ]

        # Cin = candidates after input-only anchors
        input_candidates = self._filter_candidates(
            self.assets, input_anchors, anchor_by_id
        )

        # Cout = candidates after input + output anchors
        final_candidates = self._filter_candidates(
            input_candidates, output_anchors, anchor_by_id
        )

        # Find which output anchors actually reduced the set
        contributing = self._find_contributing_output_anchors(
            input_candidates, output_anchors, anchor_by_id
        )

        # Information gain in bits
        if input_candidates and final_candidates:
            ig = math.log2(len(input_candidates) / len(final_candidates))
        else:
            ig = 0.0

        return CandidateSnapshot(
            input_candidates=input_candidates,
            final_candidates=final_candidates,
            input_anchors=input_anchors,
            output_anchors=output_anchors,
            contributing_output_anchors=contributing,
            information_gain_bits=max(0.0, ig),
        )

    def _find_contributing_output_anchors(
        self,
        input_candidates: list[AssetFact],
        output_anchors: list[Anchor],
        anchor_by_id: Dict[str, Anchor],
    ) -> list[Anchor]:
        """Find output anchors that actually narrow the candidate set."""
        remaining = list(input_candidates)
        contributing: list[Anchor] = []

        ordered = sorted(
            output_anchors,
            key=lambda a: (a.start, a.end, a.field_name),
        )

        for anchor in ordered:
            after = self._filter_candidates(remaining, [anchor], anchor_by_id)
            if 0 < len(after) < len(remaining):
                contributing.append(anchor)
                remaining = after

        return contributing

    # ==================================================================
    # v2.3 — Restoration shape detection
    # ==================================================================

    def _detect_restoration_shape(
        self,
        snapshot: CandidateSnapshot,
    ) -> RestorationDecision:
        """Determine whether the anchor set exhibits a CFA risk pattern.

        Three trigger types:
        1. direct_protected_disclosure — output directly contains protected fields
        2. indirect_asset_restoration — output narrows candidate to ≤k records
        3. indirect_protected_value_restoration — output narrows a protected field
           to ≤protected_value_k distinct values
        """
        direct_protected = [
            a for a in snapshot.output_anchors
            if a.protected and not a.inferred
        ]
        if direct_protected:
            return RestorationDecision(
                detected=True,
                trigger_type="direct_protected_disclosure",
                restored_fields=sorted({a.field_name for a in direct_protected}),
                reason="模型输出直接包含受限字段信息。",
            )

        if not self.policy.indirect_restoration_enabled:
            return RestorationDecision(detected=False)

        if not snapshot.contributing_output_anchors:
            return RestorationDecision(detected=False)

        input_count = len(snapshot.input_candidates)
        final_count = len(snapshot.final_candidates)

        # 0 candidates = inconsistent evidence, OR multi-record conflict
        if final_count == 0:
            # v2.5: Check if 0 is due to multi-record conflict (multiple values of same field)
            # If so, still report as potentially dangerous (direct disclosure may apply)
            output_field_values: Dict[str, set[str]] = {}
            for a in snapshot.output_anchors:
                if not a.inferred:
                    output_field_values.setdefault(a.field_name, set()).add(
                        a.canonical_value or a.text
                    )
            has_conflict = any(len(v) >= 2 for v in output_field_values.values())
            if has_conflict and any(a.protected for a in snapshot.output_anchors):
                return RestorationDecision(
                    detected=True,
                    trigger_type="direct_protected_disclosure",
                    restored_fields=sorted({
                        a.field_name for a in snapshot.output_anchors if a.protected
                    }),
                    reason="模型输出包含多记录冲突且直接泄露受限字段。",
                )
            return RestorationDecision(
                detected=False,
                trigger_type="inconsistent_evidence",
                reason="组合锚点未匹配任何事实记录。",
            )

        reduction = input_count - final_count

        # 2. Indirect asset restoration
        unique_restoration = (
            final_count <= self.policy.uniqueness_k
            and reduction >= self.policy.min_candidate_reduction
            and snapshot.information_gain_bits >= self.policy.min_information_gain_bits
        )
        if unique_restoration:
            return RestorationDecision(
                detected=True,
                trigger_type="indirect_asset_restoration",
                restored_fields=list(self.policy.protected_fields),
                reason=(
                    f"模型输出新增线索将候选从 {input_count} 条压缩至 "
                    f"{final_count} 条，信息增益为 "
                    f"{snapshot.information_gain_bits:.2f} bit。"
                ),
            )

        # 3. Protected field value restored without unique asset
        restored_fields = self._find_restored_protected_fields(
            snapshot.input_candidates,
            snapshot.final_candidates,
        )
        if restored_fields:
            return RestorationDecision(
                detected=True,
                trigger_type="indirect_protected_value_restoration",
                restored_fields=restored_fields,
                reason=(
                    "模型输出虽然未唯一定位记录，"
                    "但已使部分受限字段收敛到安全阈值以内。"
                ),
            )

        return RestorationDecision(detected=False)

    # ==================================================================
    # v2.3 — Field entropy and protected field restoration
    # ==================================================================

    def _field_entropy(self, candidates: list[AssetFact], field_name: str) -> float:
        """Compute Shannon entropy for a field across a candidate set."""
        values = [asset.get(field_name) for asset in candidates if asset.get(field_name)]
        if not values:
            return 0.0
        counts = Counter(values)
        total = len(values)
        entropy = 0.0
        for count in counts.values():
            p = count / total
            entropy -= p * math.log2(p)
        return entropy

    def _find_restored_protected_fields(
        self,
        before: list[AssetFact],
        after: list[AssetFact],
    ) -> list[str]:
        """Find protected fields whose value set has collapsed to ≤k."""
        if not before or not after:
            return []
        restored: list[str] = []
        for field_name in self.policy.protected_fields:
            before_values = {asset.get(field_name) for asset in before if asset.get(field_name)}
            after_values = {asset.get(field_name) for asset in after if asset.get(field_name)}
            # Skip if already unique in input (not attributable to output)
            if len(before_values) <= self.policy.protected_value_k:
                continue
            if not after_values:
                continue
            if len(after_values) <= self.policy.protected_value_k:
                entropy_before = self._field_entropy(before, field_name)
                entropy_after = self._field_entropy(after, field_name)
                entropy_drop = entropy_before - entropy_after
                if entropy_drop >= self.policy.min_protected_entropy_drop_bits:
                    restored.append(field_name)
        return restored

    # ==================================================================
    # v2.3 — Minimal restoration combination search
    # ==================================================================

    def _combo_restores_unique_fact(
        self,
        combo: list[Anchor],
        target_asset: AssetFact,
        restored_field: str,
        anchor_by_id: Dict[str, Anchor],
    ) -> bool:
        """Check whether a combo restores a unique fact.

        The combo must:
        1. Contain at least one output anchor that covers the protected field.
        2. Narrow the full fact universe to exactly the target_asset.
        """
        # Must have an output anchor providing the protected field value
        disclosure_anchors = [
            anchor
            for anchor in combo
            if anchor.source == "output"
            and anchor.protected
            and anchor.field_name == restored_field
            and not anchor.inferred
        ]
        if not disclosure_anchors:
            return False

        candidates = self._filter_candidates(
            list(self.assets),
            combo,
            anchor_by_id,
        )

        return (
            len(candidates) == 1
            and candidates[0].id == target_asset.id
        )

    def _minimal_restoration_combinations(
        self,
        input_anchors: list[Anchor],
        output_anchors: list[Anchor],
        anchor_by_id: Dict[str, Anchor],
        decision: RestorationDecision,
        *,
        max_combo_size: int = 6,
        max_results: int = 30,
    ) -> list[list[Anchor]]:
        """
        Find inclusion-minimal anchor combinations that are sufficient to trigger
        the same restoration decision, AND pass counterfactual necessity:
        removing any single anchor from the combo must break the unique binding.

        A valid combination must contain at least one output anchor, otherwise the
        risk is caused by user input alone and should not be attributed to model output.

        v2.8: Removed the direct_protected_disclosure shortcut.  Every minimal
        combo must now pass the counterfactual necessity check — the combo
        restores a unique fact, but dropping any single member no longer does.
        """

        usable_inputs = [
            a for a in self._unique_filters(input_anchors)
            if not a.inferred
        ]

        usable_outputs = [
            a for a in self._unique_filters(output_anchors)
            if a.source == "output" and not a.inferred
        ]

        if not usable_outputs:
            return []

        # For direct disclosure, we build a target_asset and restored_field
        # from the output anchors so we can run counterfactual checks.
        target_asset: Optional[AssetFact] = None
        restored_field: str = ""

        if decision.trigger_type == "direct_protected_disclosure":
            protected_outputs = [a for a in usable_outputs if a.protected]
            if not protected_outputs:
                return []
            # Try to bind to a single asset via all protected outputs
            for asset in self.assets:
                matches = self._anchors_matching_asset(asset, protected_outputs, anchor_by_id)
                if len(matches) == len(protected_outputs):
                    target_asset = asset
                    break
            if target_asset is None:
                # No single asset matches all protected outputs —
                # still try per-anchor binding.
                for anchor in protected_outputs:
                    for asset in self.assets:
                        if self._asset_matches_anchor_with_id(asset, anchor, anchor_by_id):
                            target_asset = asset
                            restored_field = anchor.field_name
                            break
                    if target_asset is not None:
                        break
            else:
                restored_field = protected_outputs[0].field_name
        else:
            # For indirect restoration, find target from final_candidates
            input_candidates = self._filter_candidates(
                list(self.assets),
                usable_inputs,
                anchor_by_id,
            )
            final_candidates = self._filter_candidates(
                input_candidates,
                usable_outputs,
                anchor_by_id,
            )
            if len(final_candidates) == 1:
                target_asset = final_candidates[0]
                restored_field = decision.restored_fields[0] if decision.restored_fields else ""

        if target_asset is None or not restored_field:
            # Fall back to pre-v2.8 behavior for indirect restoration
            # where we can't pin a single target asset
            pass

        pool = usable_inputs + usable_outputs
        output_ids = {a.id for a in usable_outputs}

        minimal: list[list[Anchor]] = []
        minimal_id_sets: list[frozenset[str]] = []

        upper = min(max_combo_size, len(pool))

        for size in range(1, upper + 1):
            for combo_tuple in combinations(pool, size):
                combo = list(combo_tuple)
                combo_ids = frozenset(a.id for a in combo)

                # The model output must contribute at least one anchor.
                if not (combo_ids & output_ids):
                    continue

                # If an already accepted smaller minimal combo is contained in this
                # combo, this combo is not minimal.
                if any(prev.issubset(combo_ids) for prev in minimal_id_sets):
                    continue

                if not self._combo_triggers_same_restoration(
                    combo,
                    anchor_by_id,
                    decision,
                ):
                    continue

                # v2.8: Counterfactual necessity check.
                # When a target_asset is available, verify that removing *any*
                # single anchor from the combo breaks the unique binding.
                if target_asset is not None and restored_field:
                    if not all(
                        not self._combo_restores_unique_fact(
                            [item for item in combo if item.id != anchor.id],
                            target_asset,
                            restored_field,
                            anchor_by_id,
                        )
                        for anchor in combo
                    ):
                        continue

                minimal.append(combo)
                minimal_id_sets.append(combo_ids)

                if len(minimal) >= max_results:
                    return minimal

        return minimal

    # ==================================================================
    # v2.3 — Combination trigger check
    # ==================================================================

    def _combo_triggers_same_restoration(
        self,
        combo: list[Anchor],
        anchor_by_id: Dict[str, Anchor],
        decision: RestorationDecision,
    ) -> bool:
        """
        Check whether a candidate anchor combination is sufficient to trigger
        the same restoration type as the global decision.
        """

        combo_inputs = [
            a for a in combo
            if a.source == "input"
        ]

        combo_outputs = [
            a for a in combo
            if a.source == "output" and not a.inferred
        ]

        if not combo_outputs:
            return False

        input_candidates = self._filter_candidates(
            list(self.assets),
            combo_inputs,
            anchor_by_id,
        )

        if not input_candidates:
            return False

        final_candidates = self._filter_candidates(
            input_candidates,
            combo_outputs,
            anchor_by_id,
        )

        if not final_candidates:
            return False

        input_count = len(input_candidates)
        final_count = len(final_candidates)

        if decision.trigger_type == "direct_protected_disclosure":
            # For direct disclosure, output must contain at least one
            # protected anchor, and the combo must produce a unique result
            # (or narrow to ≤k) to prove the disclosure matters.
            has_protected_output = any(
                a.protected and not a.inferred
                for a in combo_outputs
            )
            if not has_protected_output:
                return False
            # Must narrow to ≤k unique candidates.  When input was already 1,
            # the disclosure is real but its "uniqueness" is not attributable
            # to the output alone — that determination is made later by the
            # counterfactual necessity check.
            return (
                final_count <= self.policy.uniqueness_k
                and final_count < input_count
            )

        # Output must actually reduce the candidate set.
        # Otherwise the restoration is already caused by user input alone.
        if final_count >= input_count:
            return False

        information_gain_bits = math.log2(input_count / final_count)

        if decision.trigger_type == "indirect_asset_restoration":
            return (
                final_count <= self.policy.uniqueness_k
                and (input_count - final_count) >= self.policy.min_candidate_reduction
                and information_gain_bits >= self.policy.min_information_gain_bits
            )

        if decision.trigger_type == "indirect_protected_value_restoration":
            restored_fields = self._find_restored_protected_fields(
                input_candidates,
                final_candidates,
            )

            if not restored_fields:
                return False

            expected = set(decision.restored_fields or self.policy.protected_fields)
            return bool(set(restored_fields) & expected)

        return False

    # ==================================================================
    # v2.3 — Key anchor summary helper
    # ==================================================================

    def _summarize_key_anchors(
        self,
        anchors: Sequence[Anchor],
    ) -> list[str]:
        summary: list[str] = []
        seen = set()

        for a in anchors:
            item = (
                f"{self._source_label(a.source)}/"
                f"{a.field_label}:"
                f"{a.effective_canonical_value()}"
            )

            if item in seen:
                continue

            seen.add(item)
            summary.append(item)

        return summary

    # ==================================================================
    # v2.3 — Scoring for indirect restoration
    # ==================================================================

    def _build_score_breakdown(
        self,
        snapshot: CandidateSnapshot,
        decision: RestorationDecision,
    ) -> dict:
        """Compute CFA-Score and expose every weighted scoring component."""
        input_count = len(snapshot.input_candidates)
        final_count = len(snapshot.final_candidates)

        if decision.trigger_type in (
            "direct_protected_disclosure",
            "direct_confidential_value_match",
            "input_hypothesis_confirmation",
            "input_output_slot_restoration",
        ):
            base_score = 55.0
        elif decision.trigger_type == "indirect_asset_restoration":
            base_score = 50.0
        else:
            base_score = 40.0

        if final_count <= 0:
            uniqueness_raw = 0.0
        elif final_count <= self.policy.uniqueness_k:
            uniqueness_raw = 1.0
        else:
            uniqueness_raw = 1.0 / final_count

        gain_raw = min(1.0, max(0.0, snapshot.information_gain_bits) / 3.0)

        anchor_confidence_raw = 0.0
        anchor_details = []
        if snapshot.contributing_output_anchors:
            weighted_values = []
            for anchor in snapshot.contributing_output_anchors:
                match_weight = self.policy.match_type_weight(anchor.match_type)
                weighted_confidence = anchor.confidence * match_weight
                weighted_values.append(weighted_confidence)
                anchor_details.append({
                    "anchor_id": anchor.id,
                    "field_name": anchor.field_name,
                    "field_label": anchor.field_label,
                    "canonical_value": anchor.effective_canonical_value(),
                    "match_type": anchor.match_type,
                    "confidence": round(anchor.confidence, 4),
                    "match_type_weight": round(match_weight, 4),
                    "weighted_confidence": round(weighted_confidence, 4),
                })
            anchor_confidence_raw = sum(weighted_values) / len(weighted_values)

        uniqueness_contribution = 25.0 * uniqueness_raw
        gain_contribution = 15.0 * gain_raw
        confidence_contribution = 10.0 * min(1.0, anchor_confidence_raw)
        final_score = min(
            100.0,
            base_score
            + uniqueness_contribution
            + gain_contribution
            + confidence_contribution,
        )

        return {
            "formula_version": "CFA-Score-v2.3",
            "formula": "Base + 25×U + 15×G + 10×C",
            "trigger_type": decision.trigger_type,
            "base": {
                "raw": base_score,
                "contribution": base_score,
            },
            "uniqueness": {
                "input_candidate_count": input_count,
                "final_candidate_count": final_count,
                "uniqueness_k": self.policy.uniqueness_k,
                "raw": round(uniqueness_raw, 4),
                "weight": 25.0,
                "contribution": round(uniqueness_contribution, 2),
            },
            "information_gain": {
                "bits": round(snapshot.information_gain_bits, 4),
                "normalized": round(gain_raw, 4),
                "normalization": "min(1, bits / 3)",
                "weight": 15.0,
                "contribution": round(gain_contribution, 2),
            },
            "anchor_confidence": {
                "raw": round(anchor_confidence_raw, 4),
                "weight": 10.0,
                "contribution": round(confidence_contribution, 2),
                "anchors": anchor_details,
            },
            "final_score": round(final_score, 2),
            "risk_thresholds": {
                "critical": 85,
                "high": 70,
                "medium": 45,
            },
        }

    def _score_restoration(
        self,
        snapshot: CandidateSnapshot,
        decision: RestorationDecision,
    ) -> float:
        """Compute CFA-Score with indirect-restoration awareness."""
        breakdown = self._build_score_breakdown(snapshot, decision)
        return breakdown["final_score"]

    def _risk_level(self, score: float) -> str:
        if score >= 85:
            return "CRITICAL"
        elif score >= 70:
            return "HIGH"
        elif score >= 45:
            return "MEDIUM"
        else:
            return "LOW"

    # ==================================================================
    # v2.3 — Main finding construction (snapshot-based)
    # ==================================================================

    def _build_findings(self, anchors: Sequence[Anchor]) -> List[RiskFinding]:
        """Build RiskFindings from a global candidate snapshot.

        v2.3: Instead of iterating per-asset, we first compute the global
        candidate snapshot (Cin/Cout) and detect restoration shape.
        Then findings are created for each affected asset.
        """
        all_anchors = list(anchors)
        anchor_by_id = {a.id: a for a in all_anchors}

        snapshot = self._build_candidate_snapshot(all_anchors)
        decision = self._detect_restoration_shape(snapshot)

        if not decision.detected:
            return []

        findings: List[RiskFinding] = []

        # Select which assets to iterate.
        # When final_candidates is non-empty, iterate those.
        # When empty (e.g., output anchors over-constrained), fall back to
        # input_candidates, or all assets for direct disclosure.
        target_assets = list(snapshot.final_candidates) if snapshot.final_candidates else list(snapshot.input_candidates)
        if not target_assets and decision.trigger_type == "direct_protected_disclosure":
            target_assets = list(self.assets)

        # Compute minimal restoration combinations
        minimal_combos = self._minimal_restoration_combinations(
            snapshot.input_anchors,
            snapshot.output_anchors,
            anchor_by_id,
            decision,
        )

        # v2.8: Determine anchor verification status
        anchor_status = ""
        anchor_status_reason = ""

        if minimal_combos:
            minimal_combinations = [
                [a.id for a in combo]
                for combo in minimal_combos
            ]

            key_anchors: list[Anchor] = []
            seen_anchor_ids = set()

            for combo in minimal_combos:
                for a in combo:
                    if a.id in seen_anchor_ids:
                        continue
                    seen_anchor_ids.add(a.id)
                    key_anchors.append(a)

            key_anchor_ids_all = [a.id for a in key_anchors]
            key_summary_dedup = self._summarize_key_anchors(key_anchors)

            key_output_anchors = [
                a for a in key_anchors
                if a.source == "output"
            ]

            anchor_status = "unique_restoration_verified"

        else:
            # Fallback: keep old behavior only if minimal search fails.
            # This prevents breaking existing demo behavior.
            minimal_combinations = []

            key_anchors = (
                list(snapshot.input_anchors)
                + list(snapshot.contributing_output_anchors)
            )

            key_anchor_ids_all = [a.id for a in key_anchors]
            key_summary_dedup = self._summarize_key_anchors(key_anchors)

            key_output_anchors = list(snapshot.contributing_output_anchors)

            # v2.8: When direct_protected_disclosure is detected but no
            # minimal combos pass counterfactual verification, flag it.
            if decision.trigger_type == "direct_protected_disclosure":
                anchor_status = "unique_restoration_not_verified"
                input_count = len(snapshot.input_candidates)
                final_count = len(snapshot.final_candidates)
                if input_count <= 1 and final_count <= 1:
                    anchor_status_reason = (
                        f"加入模型输出前候选已经为 {input_count} 条；加入输出后仍为 "
                        f"{final_count} 条，信息增益为 "
                        f"{snapshot.information_gain_bits:.2f} bit，"
                        f"无法证明任何锚点导致了唯一化。"
                    )
                else:
                    anchor_status_reason = (
                        f"检测到受保护值泄露，但未通过删除锚点反事实验证唯一还原。"
                        f"输入候选 {input_count} 条，最终候选 {final_count} 条。"
                    )

        # When final_candidates is empty but detection triggered (e.g.
        # direct disclosure), use input_candidates count for scoring
        # so that findings are not LOW-filtered.
        effective_final = snapshot.final_candidates if snapshot.final_candidates else snapshot.input_candidates
        effective_input = snapshot.input_candidates if snapshot.input_candidates else list(self.assets)
        # Recompute info gain with effective counts
        if effective_input and effective_final:
            effective_ig = max(0.0, math.log2(len(effective_input) / len(effective_final)))
        else:
            effective_ig = 0.0
        effective_snapshot = CandidateSnapshot(
            input_candidates=effective_input,
            final_candidates=effective_final,
            input_anchors=snapshot.input_anchors,
            output_anchors=snapshot.output_anchors,
            contributing_output_anchors=key_output_anchors,
            information_gain_bits=effective_ig,
        )

        score_breakdown = self._build_score_breakdown(effective_snapshot, decision)
        score = score_breakdown["final_score"]

        for asset in target_assets:
            matching_anchors = self._anchors_matching_asset(asset, key_anchors, anchor_by_id)
            chain = self._reduction_chain(matching_anchors, anchor_by_id)

            level = self._risk_level(score)
            if level == "LOW":
                continue

            target_name = asset.display_name(self.policy.display_field)

            findings.append(
                RiskFinding(
                    target_asset_id=asset.id,
                    target_asset_name=target_name,
                    risk_level=level,
                    score=score,
                    reason=decision.reason,
                    restored_fact=self._restored_fact(asset, matching_anchors),
                    anchors=matching_anchors,
                    reduction_chain=chain,
                    minimal_combinations=minimal_combinations,
                    key_anchor_ids=key_anchor_ids_all,
                    key_anchor_summary=key_summary_dedup,
                    finding_type=decision.trigger_type,
                    target_asset_ids=[a.id for a in snapshot.final_candidates],
                    restored_fields=decision.restored_fields,
                    input_candidate_count=len(snapshot.input_candidates),
                    final_candidate_count=len(snapshot.final_candidates),
                    information_gain_bits=snapshot.information_gain_bits,
                    score_breakdown=score_breakdown,
                    anchor_status=anchor_status,
                    anchor_status_reason=anchor_status_reason,
                )
            )

        # Sort by score descending
        findings.sort(
            key=lambda f: (f.score, len(f.key_anchor_ids), len(f.anchors)),
            reverse=True,
        )
        return findings

    # ==================================================================
    # Anchor-to-asset matching (supports accepted_values)
    # ==================================================================

    def _anchors_matching_asset(
        self,
        asset: AssetFact,
        anchors: Sequence[Anchor],
        anchor_by_id: Dict[str, Anchor],
    ) -> List[Anchor]:
        matched: List[Anchor] = []
        seen = set()
        for anchor in anchors:
            if not self._asset_matches_anchor_with_id(asset, anchor, anchor_by_id):
                continue
            key = (
                anchor.source,
                anchor.field_name,
                anchor.canonical_value or str(anchor.accepted_values),
                anchor.start,
                anchor.end,
                anchor.inferred,
            )
            if key not in seen:
                matched.append(anchor)
                seen.add(key)
        matched.sort(
            key=lambda a: (
                self._field_rank(a.field_name),
                self._source_rank(a.source),
                a.start,
                a.end,
                a.inferred,
            )
        )
        return matched

    # ==================================================================
    # Reduction chain (supports match_symbol for semantic anchors)
    # ==================================================================

    def _reduction_chain(
        self,
        anchors: Sequence[Anchor],
        anchor_by_id: Dict[str, Anchor],
    ) -> List[ReductionStep]:
        remaining = list(self.assets)
        chain: List[ReductionStep] = []
        unique_filters = self._unique_filters(anchors)
        for anchor in unique_filters:
            before = len(remaining)
            remaining = [
                fact
                for fact in remaining
                if self._asset_matches_anchor_with_id(fact, anchor, anchor_by_id)
            ]
            chain.append(
                ReductionStep(
                    field_name=anchor.field_name,
                    field_label=anchor.field_label,
                    anchor_text=anchor.text,
                    canonical_value=anchor.effective_canonical_value(),
                    before_count=before,
                    after_count=len(remaining),
                    remaining_asset_ids=[fact.id for fact in remaining],
                    match_symbol=anchor.match_symbol(),
                )
            )
            if not remaining:
                break
        return chain

    # ==================================================================
    # Unique filters (deduplication)
    # ==================================================================

    def _unique_filters(self, anchors: Sequence[Anchor]) -> List[Anchor]:
        unique_filters: List[Anchor] = []
        seen_filters = set()
        for anchor in sorted(
            anchors,
            key=lambda x: (
                self._field_rank(x.field_name),
                self._source_rank(x.source),
                x.start,
                x.end,
                x.inferred,
            ),
        ):
            key = (anchor.field_name, anchor.canonical_value or str(anchor.accepted_values))
            if key in seen_filters:
                continue
            seen_filters.add(key)
            unique_filters.append(anchor)
        return unique_filters

    # ==================================================================
    # Restoration fact text
    # ==================================================================

    def _restored_fact(self, asset: AssetFact, anchors: Sequence[Anchor]) -> str:
        anchor_fields = {a.field_name for a in anchors}
        if "secret_content" in anchor_fields and asset.get("secret_content"):
            return f"{self.policy.label('secret_content')}={asset.get('secret_content')}"

        protected_parts = []
        for field_name in self.policy.protected_fields:
            if any(a.field_name == field_name for a in anchors):
                protected_parts.append(
                    f"{self.policy.label(field_name)}={asset.get(field_name)}"
                )
        if not protected_parts:
            protected_parts.append("受控字段状态被间接还原")
        return (
            f"{asset.display_name(self.policy.display_field)}："
            + "，".join(protected_parts)
        )

    # ==================================================================
    # Ranking helpers
    # ==================================================================

    def _field_rank(self, field_name: str) -> int:
        try:
            return self.policy.field_order.index(field_name)
        except ValueError:
            return len(self.policy.field_order) + 1

    def _source_rank(self, source: str) -> int:
        return {"input": 0, "output": 1}.get(source, 9)

    def _source_label(self, source: str) -> str:
        return {"input": "用户输入", "output": "模型输出"}.get(source, source)

    # ==================================================================
    # Unified extraction helper (used by both first pass and secondary check)
    # ==================================================================

    def _extract_anchors_for_pass(
        self,
        *,
        user_input: str,
        model_output: str,
    ) -> List[Anchor]:
        """Extract anchors for one CFA analysis pass.

        Important:
        - Always run rule-based extraction on BOTH user_input and model_output.
        - LLM extraction is optional.
        - If LLM extraction fails, fall back to rule anchors, not empty anchors.
        - v2.5: Also runs normalized extraction for fuzzy-matched amounts, rates, ratings.
        - This helper must be used by both the first pass and the secondary pass.
        """

        segments: List[Tuple[str, str]] = []

        if user_input and user_input.strip():
            segments.append(("input", user_input))

        segments.append(("output", model_output or ""))

        # 1. Rule-based anchors: always include input + output
        rule_anchors = self.rule_extractor.extract_segments(
            segments,
            self.assets,
        )
        if self._trace is not None:
            self._trace.snapshot(
                "rule_anchors",
                {
                    "segments": [
                        {"source": source, "text": text}
                        for source, text in segments
                    ],
                    "anchors": [anchor.to_dict() for anchor in rule_anchors],
                },
                component="CFAScoreEngine",
                stage="cfa.rule_extraction",
                sensitivity="restricted",
            )

        # 2. LLM anchors: optional recall enhancement
        llm_anchors: List[Anchor] = []

        if self._mode == ExtractionMode.RULE_PLUS_LLM and self._llm_extractor is not None:
            try:
                raw_input_anchors: list[Anchor] = []
                raw_output_anchors: list[Anchor] = []

                if user_input and user_input.strip():
                    raw_input_anchors = self._llm_extractor.extract_segment(
                        text=user_input,
                        source="input",
                    )

                if model_output and model_output.strip():
                    raw_output_anchors = self._llm_extractor.extract_segment(
                        text=model_output,
                        source="output",
                    )

                if self._verifier is not None:
                    verified_input = self._verifier.verify_segment_all(
                        raw_input_anchors,
                        source_text=user_input,
                        expected_source="input",
                    )

                    verified_output = self._verifier.verify_segment_all(
                        raw_output_anchors,
                        source_text=model_output,
                        expected_source="output",
                    )

                    llm_anchors = verified_input + verified_output
                else:
                    llm_anchors = raw_input_anchors + raw_output_anchors

            except Exception:
                # LLM extraction failure must not remove rule-based input anchors.
                llm_anchors = []

        merged = AnchorMerger.merge(rule_anchors, llm_anchors)
        if self._trace is not None:
            self._trace.snapshot(
                "merged_anchors",
                {
                    "rule_anchor_count": len(rule_anchors),
                    "llm_anchor_count": len(llm_anchors),
                    "merged_anchor_count": len(merged),
                    "anchors": [anchor.to_dict() for anchor in merged],
                },
                component="CFAScoreEngine",
                stage="cfa.anchor_merge",
                sensitivity="restricted",
            )

        # 3. v2.5: Normalized anchors for fuzzy-matched amounts, rates, ratings, collateral
        normalized_anchors = self._extract_normalized_anchors(user_input, model_output)
        # Merge normalized anchors with existing (avoid duplicates by span)
        all_anchors = self._merge_normalized_anchors(merged, normalized_anchors)
        if self._trace is not None:
            self._trace.snapshot(
                "all_anchors",
                {
                    "normalized_anchor_count": len(normalized_anchors),
                    "anchor_count": len(all_anchors),
                    "anchors": [anchor.to_dict() for anchor in all_anchors],
                },
                component="CFAScoreEngine",
                stage="cfa.anchor_lifecycle_complete",
                sensitivity="restricted",
            )

        return all_anchors

    def _extract_normalized_anchors(
        self,
        user_input: str,
        model_output: str,
    ) -> List[Anchor]:
        """Extract anchors via field normalizers (amount, rate, rating, collateral components)."""
        normalizer = FieldNormalizer(self.policy, self.assets)

        anchors: List[Anchor] = []
        anchor_index = 0

        for source, text in [("input", user_input), ("output", model_output)]:
            if not text or not text.strip():
                continue

            # 1. Amount/rate/rating/date normalization
            for nv in normalizer.extract_normalized_anchors(text, source=source):
                anchor = self._normalized_value_to_anchor(nv, text, source, anchor_index)
                anchor_index += 1
                anchors.append(anchor)

            # 2. Collateral component matching
            if "collateral" in self.policy.field_order:
                for nv in normalizer.match_collateral_components(text, "collateral"):
                    anchor = self._normalized_value_to_anchor(nv, text, source, anchor_index)
                    anchor_index += 1
                    anchors.append(anchor)

        return anchors

    def _normalized_value_to_anchor(
        self,
        nv,
        text: str,
        source: str,
        idx: int,
    ) -> Anchor:
        """Convert a NormalizedValue to an Anchor."""
        import hashlib
        raw_id = f"NV|{source}|{nv.field_name}|{nv.raw_text}|{idx}"
        digest = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:10].upper()
        return Anchor(
            id=f"N{digest}",
            field_name=nv.field_name,
            field_label=self.policy.label(nv.field_name),
            text=nv.raw_text,
            canonical_value=nv.canonical_value,
            start=-1,
            end=-1,
            anchor_type="事实锚点",
            protected=nv.protected,
            inferred=False,
            evidence=nv.evidence,
            source=source,
            match_type=nv.match_type,
            confidence=nv.confidence,
        )

    def _merge_normalized_anchors(
        self,
        merged_anchors: List[Anchor],
        normalized_anchors: List[Anchor],
    ) -> List[Anchor]:
        """Merge normalized anchors, avoiding exact duplicate field+value pairs."""
        if not normalized_anchors:
            return merged_anchors

        # Build set of existing (source, field, value) keys
        existing: set[tuple] = set()
        for a in merged_anchors:
            key = (a.source, a.field_name, a.canonical_value)
            existing.add(key)

        result = list(merged_anchors)
        for na in normalized_anchors:
            key = (na.source, na.field_name, na.canonical_value)
            if key in existing:
                continue
            existing.add(key)
            result.append(na)

        # Sort by source, position, field
        result.sort(
            key=lambda a: (
                0 if a.source == "input" else 1,
                max(0, a.start),
                a.field_name,
            )
        )
        return result

    # ==================================================================
    # v2.5 — Claim-level detection (multi-claim architecture)
    # ==================================================================

    def _has_multi_record_conflict(
        self,
        output_anchors: list[Anchor],
    ) -> bool:
        """Check if output contains multiple distinct records (discriminator field conflicts)."""
        field_values: Dict[str, set[str]] = {}
        discriminator_fields = {
            "loan_type", "loan_amount", "interest_rate", "collateral",
            "diagnosis", "medication", "treatment", "meeting_time",
            "meeting_title", "participant", "asset_type", "asset_value",
        }
        for anchor in output_anchors:
            if anchor.inferred:
                continue
            # Check all fields for multi-value conflicts (not just discriminators)
            field_values.setdefault(anchor.field_name, set()).add(
                anchor.canonical_value or anchor.text
            )
        for field_name, values in field_values.items():
            if len(values) >= 2:
                return True
        return False

    def _build_claim_based_findings(
        self,
        anchors: Sequence[Anchor],
        user_input: str,
        model_output: str,
    ) -> List[RiskFinding]:
        """v2.5: Build findings per disclosure claim, then UNION them.

        This replaces the global AND-based detection for multi-record answers.
        Each claim independently binds candidates and detects risk.
        Findings from different claims are unioned, not intersected.
        """
        all_anchors = list(anchors)
        anchor_by_id = {a.id: a for a in all_anchors}
        input_anchors = [a for a in all_anchors if a.source == "input" and not a.inferred]
        output_anchors = [a for a in all_anchors if a.source == "output" and not a.inferred]

        # Phase 1: Build claims using ClaimBuilder
        claim_builder = ClaimBuilder(self.policy, self.assets)
        claims = claim_builder.build_claims(
            model_output=model_output,
            anchors=all_anchors,
            user_input=user_input,
        )

        if not claims:
            # Fallback: treat the entire output as a single claim
            claims = [
                DisclosureClaim(
                    claim_id="CL_GLOBAL",
                    text=model_output,
                    start=0,
                    end=len(model_output),
                    context_anchors=list(input_anchors),
                    local_output_anchors=list(output_anchors),
                    candidate_assets=list(self.assets),
                    candidate_asset_ids=[a.id for a in self.assets],
                )
            ]

        # Phase 2: Detect risk per claim and union findings
        all_findings: List[RiskFinding] = []
        seen_finding_keys: set[tuple] = set()

        for claim in claims:
            claim_findings = self._detect_claim_risk(claim, anchor_by_id)
            for finding in claim_findings:
                # Deduplicate across claims
                # v2.6: For direct_protected_disclosure, the dedup key must include
                # the specific anchor field_name and value, otherwise all protected
                # fields within one claim collapse to a single finding (only the
                # first one survives).  For other finding types the original key
                # (type + target_ids + restored_fields) is still correct.
                if finding.finding_type == "direct_protected_disclosure":
                    # Include first anchor's field+value to differentiate
                    first_anchor = finding.anchors[0] if finding.anchors else None
                    anchor_sig = (
                        first_anchor.field_name,
                        first_anchor.effective_canonical_value(),
                    ) if first_anchor else ("unknown", str(finding.target_asset_ids))
                    key = (
                        finding.finding_type,
                        anchor_sig,
                    )
                else:
                    key = (
                        finding.finding_type,
                        str(finding.target_asset_ids),
                        str(finding.restored_fields),
                    )
                if key in seen_finding_keys:
                    continue
                seen_finding_keys.add(key)
                all_findings.append(finding)

        # Sort by score descending
        all_findings.sort(
            key=lambda f: (f.score, len(f.key_anchor_ids), len(f.anchors)),
            reverse=True,
        )
        return all_findings

    def _detect_claim_risk(
        self,
        claim: DisclosureClaim,
        anchor_by_id: Dict[str, Anchor],
    ) -> List[RiskFinding]:
        """Detect CFA risk for a single disclosure claim.

        Risk priority:
        1. Direct protected disclosure in output
        2. Indirect asset restoration (candidate narrowed to ≤k)
        3. Indirect protected value restoration
        4. Protected field convergence without unique match
        """
        findings: List[RiskFinding] = []

        # Priority 1: Direct protected disclosure — does NOT require unique binding
        protected_output = [
            a for a in claim.local_output_anchors
            if a.protected and not a.inferred
        ]
        if protected_output:
            direct_input_candidates = self._filter_candidates_by_context_anchors(
                claim.context_anchors,
                anchor_by_id,
            )
            if not direct_input_candidates:
                direct_input_candidates = list(self.assets)

            # Direct disclosure: build an aggregate finding when one claim binds to one
            # concrete asset, then keep per-anchor findings for field-level redaction.
            if len(claim.candidate_assets) == 1:
                aggregate_asset = claim.candidate_assets[0]
                aggregate_anchors = self._anchors_matching_asset(
                    aggregate_asset,
                    protected_output,
                    anchor_by_id,
                )
                aggregate_fields = sorted({a.field_name for a in aggregate_anchors if a.protected})
                if len(aggregate_fields) >= 2:
                    aggregate_decision = RestorationDecision(
                        detected=True,
                        trigger_type="direct_protected_disclosure",
                        restored_fields=aggregate_fields,
                        reason="模型输出直接包含多项受限字段信息。",
                    )
                    aggregate_snapshot = CandidateSnapshot(
                        input_candidates=direct_input_candidates,
                        final_candidates=[aggregate_asset],
                        input_anchors=list(claim.context_anchors),
                        output_anchors=list(protected_output),
                        contributing_output_anchors=list(aggregate_anchors),
                        information_gain_bits=0.0,
                    )
                    aggregate_breakdown = self._build_score_breakdown(
                        aggregate_snapshot,
                        aggregate_decision,
                    )
                    aggregate_score = aggregate_breakdown["final_score"]
                    findings.append(
                        RiskFinding(
                            target_asset_id=aggregate_asset.id,
                            target_asset_name=aggregate_asset.display_name(self.policy.display_field),
                            risk_level=self._risk_level(aggregate_score),
                            score=aggregate_score,
                            reason="模型输出直接包含多项受限字段信息。",
                            restored_fact=self._restored_fact(aggregate_asset, aggregate_anchors),
                            anchors=aggregate_anchors,
                            reduction_chain=self._reduction_chain(aggregate_anchors, anchor_by_id),
                            minimal_combinations=[[a.id for a in aggregate_anchors]],
                            key_anchor_ids=[a.id for a in aggregate_anchors],
                            key_anchor_summary=self._summarize_key_anchors(aggregate_anchors),
                            finding_type="direct_protected_disclosure",
                            target_asset_ids=[aggregate_asset.id],
                            restored_fields=aggregate_fields,
                            input_candidate_count=len(direct_input_candidates),
                            final_candidate_count=1,
                            information_gain_bits=0.0,
                            score_breakdown=aggregate_breakdown,
                        )
                    )

            # Direct disclosure: findings per protected anchor
            for anchor in protected_output:
                # Try to bind to a candidate asset if possible
                target_asset = None
                if claim.candidate_assets:
                    # Find assets where this field matches
                    for asset in claim.candidate_assets:
                        if asset.get(anchor.field_name) in self._anchor_values(anchor):
                            target_asset = asset
                            break

                asset_id = target_asset.id if target_asset else "unknown"
                asset_name = (
                    target_asset.display_name(self.policy.display_field)
                    if target_asset
                    else f"直接泄露:{anchor.field_label}={anchor.effective_canonical_value()}"
                )
                protected_field_names = sorted(
                    {a.field_name for a in protected_output}
                )
                target_ids = [a.id for a in claim.candidate_assets] if claim.candidate_assets else []
                if target_asset:
                    target_ids = [target_asset.id]
                direct_final_candidates = [target_asset] if target_asset else list(claim.candidate_assets)
                if not direct_final_candidates:
                    direct_final_candidates = list(direct_input_candidates)
                direct_decision = RestorationDecision(
                    detected=True,
                    trigger_type="direct_protected_disclosure",
                    restored_fields=protected_field_names,
                    reason=f"模型输出直接包含受限字段: {anchor.field_label}={anchor.effective_canonical_value()}",
                )
                direct_snapshot = CandidateSnapshot(
                    input_candidates=direct_input_candidates,
                    final_candidates=direct_final_candidates,
                    input_anchors=list(claim.context_anchors),
                    output_anchors=list(protected_output),
                    contributing_output_anchors=[anchor],
                    information_gain_bits=0.0,
                )
                direct_breakdown = self._build_score_breakdown(direct_snapshot, direct_decision)
                score = direct_breakdown["final_score"]
                level = self._risk_level(score)
                if level == "LOW":
                    continue

                findings.append(
                    RiskFinding(
                        target_asset_id=asset_id,
                        target_asset_name=asset_name,
                        risk_level=level,
                        score=score,
                        reason=f"模型输出直接包含受限字段: {anchor.field_label}={anchor.effective_canonical_value()}",
                        restored_fact=f"{anchor.field_label}={anchor.effective_canonical_value()}",
                        anchors=[anchor],
                        reduction_chain=[],
                        minimal_combinations=[[anchor.id]],
                        key_anchor_ids=[anchor.id],
                        key_anchor_summary=[f"模型输出/{anchor.field_label}:{anchor.effective_canonical_value()}"],
                        finding_type="direct_protected_disclosure",
                        target_asset_ids=target_ids,
                        restored_fields=protected_field_names,
                        input_candidate_count=len(direct_input_candidates),
                        final_candidate_count=len(direct_final_candidates),
                        information_gain_bits=0.0,
                        score_breakdown=direct_breakdown,
                    )
                )
            if findings:
                return findings

        # Priority 2-4: Indirect restoration via candidate binding
        if not claim.candidate_assets:
            return findings

        input_candidates = self._filter_candidates_by_context_anchors(
            claim.context_anchors,
            anchor_by_id,
        )
        final_candidates = claim.candidate_assets

        input_count = len(input_candidates) if input_candidates else len(self.assets)
        final_count = len(final_candidates)

        if final_count == 0:
            return findings

        if final_count >= input_count:
            return findings

        # Information gain
        if input_count > 0 and final_count > 0:
            ig = math.log2(input_count / final_count)
        else:
            ig = 0.0

        # Priority 2: Indirect asset restoration
        if (
            final_count <= self.policy.uniqueness_k
            and (input_count - final_count) >= self.policy.min_candidate_reduction
            and ig >= self.policy.min_information_gain_bits
        ):
            # Build findings for each narrowed asset
            snapshot = CandidateSnapshot(
                input_candidates=input_candidates or list(self.assets),
                final_candidates=final_candidates,
                input_anchors=list(claim.context_anchors),
                output_anchors=list(claim.local_output_anchors),
                contributing_output_anchors=list(claim.local_output_anchors),
                information_gain_bits=ig,
            )
            decision = RestorationDecision(
                detected=True,
                trigger_type="indirect_asset_restoration",
                restored_fields=list(self.policy.protected_fields),
                reason=f"模型输出新增线索将候选从 {input_count} 条压缩至 {final_count} 条，信息增益为 {ig:.2f} bit。",
            )

            score_breakdown = self._build_score_breakdown(snapshot, decision)
            score = score_breakdown["final_score"]

            for asset in final_candidates:
                matching = self._anchors_matching_asset(
                    asset, claim.local_output_anchors, anchor_by_id
                )
                chain = self._reduction_chain(matching, anchor_by_id)
                level = self._risk_level(score)
                if level == "LOW":
                    continue
                findings.append(
                    RiskFinding(
                        target_asset_id=asset.id,
                        target_asset_name=asset.display_name(self.policy.display_field),
                        risk_level=level,
                        score=score,
                        reason=decision.reason,
                        restored_fact=self._restored_fact(asset, matching),
                        anchors=matching,
                        reduction_chain=chain,
                        minimal_combinations=[[a.id for a in claim.local_output_anchors]],
                        key_anchor_ids=[a.id for a in claim.local_output_anchors],
                        key_anchor_summary=self._summarize_key_anchors(claim.local_output_anchors),
                        finding_type=decision.trigger_type,
                        target_asset_ids=[a.id for a in final_candidates],
                        restored_fields=decision.restored_fields,
                        input_candidate_count=input_count,
                        final_candidate_count=final_count,
                        information_gain_bits=ig,
                        score_breakdown=score_breakdown,
                    )
                )
            return findings

        # Priority 3-4: Protected field convergence
        before_set = input_candidates or list(self.assets)
        restored_fields = self._find_restored_protected_fields(before_set, final_candidates)
        if restored_fields:
            snapshot = CandidateSnapshot(
                input_candidates=before_set,
                final_candidates=final_candidates,
                input_anchors=list(claim.context_anchors),
                output_anchors=list(claim.local_output_anchors),
                contributing_output_anchors=list(claim.local_output_anchors),
                information_gain_bits=ig,
            )
            decision = RestorationDecision(
                detected=True,
                trigger_type="indirect_protected_value_restoration",
                restored_fields=restored_fields,
                reason="模型输出使部分受限字段收敛到安全阈值以内。",
            )
            score_breakdown = self._build_score_breakdown(snapshot, decision)
            score = score_breakdown["final_score"]

            for asset in final_candidates:
                matching = self._anchors_matching_asset(
                    asset, claim.local_output_anchors, anchor_by_id
                )
                chain = self._reduction_chain(matching, anchor_by_id)
                level = self._risk_level(score)
                if level == "LOW":
                    continue
                findings.append(
                    RiskFinding(
                        target_asset_id=asset.id,
                        target_asset_name=asset.display_name(self.policy.display_field),
                        risk_level=level,
                        score=score,
                        reason=decision.reason,
                        restored_fact=self._restored_fact(asset, matching),
                        anchors=matching,
                        reduction_chain=chain,
                        minimal_combinations=[[a.id for a in claim.local_output_anchors]],
                        key_anchor_ids=[a.id for a in claim.local_output_anchors],
                        key_anchor_summary=self._summarize_key_anchors(claim.local_output_anchors),
                        finding_type=decision.trigger_type,
                        target_asset_ids=[a.id for a in final_candidates],
                        restored_fields=decision.restored_fields,
                        input_candidate_count=input_count,
                        final_candidate_count=final_count,
                        information_gain_bits=ig,
                        score_breakdown=score_breakdown,
                    )
                )
            return findings

        return findings

    # ==================================================================
    # Confidential input-hypothesis confirmation detection
    # ==================================================================

    def _build_input_hypothesis_confirmation_findings(
        self,
        anchors: Sequence[Anchor],
        *,
        model_output: str,
    ) -> List[RiskFinding]:
        """Detect when a model confirms a unique protected fact supplied by input.

        v2.7: Gate is now policy-driven via _allow_confirmation_detection().
        For backward compat, secret_content/secret_summary policies are always
        allowed.  For new generic policies, set confirmation_detection_enabled=true
        in the policy JSON.
        """
        if not self._allow_confirmation_detection():
            return []
        if self._output_refuses_or_deflects(model_output):
            return []

        all_anchors = list(anchors)
        anchor_by_id = {a.id: a for a in all_anchors}
        input_anchors = [a for a in all_anchors if a.source == "input" and not a.inferred]
        protected_input_anchors = [
            a for a in input_anchors
            if a.protected and self._anchor_values(a)
        ]
        if not protected_input_anchors:
            return []

        input_candidates = self._filter_candidates(
            list(self.assets),
            input_anchors,
            anchor_by_id,
        )
        if len(input_candidates) != 1:
            return []

        target_asset = input_candidates[0]
        matching_protected = [
            a for a in protected_input_anchors
            if self._asset_matches_anchor_with_id(target_asset, a, anchor_by_id)
        ]
        if not matching_protected:
            return []

        restored_fields = self._restored_fields_for_confirmation(
            target_asset,
            matching_protected,
        )
        if not restored_fields:
            return []

        signals = self._confirmation_signals_for_asset(
            model_output,
            target_asset,
        )
        if not signals:
            return []
        output_anchor = self._make_confirmation_output_anchor(
            model_output,
            target_asset,
            restored_fields[0],
        )
        finding_anchors = list(matching_protected)
        if output_anchor is not None:
            finding_anchors.append(output_anchor)

        chain_anchors = [
            a for a in input_anchors
            if self._asset_matches_anchor_with_id(target_asset, a, anchor_by_id)
        ]
        chain = self._reduction_chain(chain_anchors, anchor_by_id)
        chain.append(
            ReductionStep(
                field_name="model_confirmation",
                field_label="模型输出确认",
                anchor_text="；".join(signals[:3]),
                canonical_value="确认/复述用户输入中的受限事实",
                before_count=1,
                after_count=1,
                remaining_asset_ids=[target_asset.id],
                match_symbol="≈",
            )
        )

        total_count = max(1, len(self.assets))
        confirmation_snapshot = CandidateSnapshot(
            input_candidates=list(self.assets),
            final_candidates=[target_asset],
            input_anchors=list(chain_anchors),
            output_anchors=[output_anchor] if output_anchor is not None else [],
            contributing_output_anchors=[output_anchor] if output_anchor is not None else [],
            information_gain_bits=round(math.log2(total_count), 4) if total_count > 1 else 0.0,
        )
        confirmation_decision = RestorationDecision(
            detected=True,
            trigger_type="input_hypothesis_confirmation",
            restored_fields=restored_fields,
            reason="用户输入已唯一指向受限事实，模型输出又确认或复述该事实的核心内容。",
        )
        score_breakdown = self._build_score_breakdown(confirmation_snapshot, confirmation_decision)
        score = score_breakdown["final_score"]
        key_summary = self._summarize_key_anchors(matching_protected)
        key_summary.extend(f"模型输出/确认信号:{signal}" for signal in signals[:4])
        if output_anchor is not None:
            key_summary.append(f"模型输出/{output_anchor.field_label}:确认/复述")

        finding = RiskFinding(
            target_asset_id=target_asset.id,
            target_asset_name=target_asset.display_name(self.policy.display_field),
            risk_level=self._risk_level(score),
            score=score,
            reason=(
                "用户输入已唯一指向受限事实，模型输出又确认或复述该事实的核心内容，"
                "使受限事实被组合还原。"
            ),
            restored_fact=self._restored_fact(target_asset, matching_protected),
            anchors=finding_anchors,
            reduction_chain=chain,
            minimal_combinations=[[a.id for a in finding_anchors]],
            key_anchor_ids=[a.id for a in finding_anchors],
            key_anchor_summary=key_summary,
            finding_type="input_hypothesis_confirmation",
            target_asset_ids=[target_asset.id],
            restored_fields=restored_fields,
            input_candidate_count=total_count,
            final_candidate_count=1,
            information_gain_bits=round(math.log2(total_count), 4) if total_count > 1 else 0.0,
            score_breakdown=score_breakdown,
        )
        return [finding]

    def _restored_fields_for_confirmation(
        self,
        asset: AssetFact,
        protected_input_anchors: Sequence[Anchor],
    ) -> List[str]:
        fields: List[str] = []
        anchor_fields = {a.field_name for a in protected_input_anchors}

        if "secret_content" in anchor_fields and asset.get("secret_content"):
            fields.append("secret_content")
        elif "secret_summary" in anchor_fields and asset.get("secret_content"):
            fields.append("secret_content")

        for anchor in protected_input_anchors:
            field_name = anchor.field_name
            if field_name not in fields and asset.get(field_name):
                fields.append(field_name)

        return fields

    def _make_confirmation_output_anchor(
        self,
        model_output: str,
        asset: AssetFact,
        field_name: str,
    ) -> Optional[Anchor]:
        raw_id = f"CONFIRM|{asset.id}|{field_name}|{model_output[:80]}"
        digest = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:10].upper()
        return Anchor(
            id=f"C{digest}",
            field_name=field_name,
            field_label=self.policy.label(field_name),
            text=str(model_output or "")[:120],
            canonical_value=asset.get(field_name),
            start=-1,
            end=-1,
            anchor_type="确认锚点",
            protected=True,
            inferred=False,
            evidence="模型输出确认或复述用户输入中的受限事实",
            source="output",
            match_type="semantic",
            confidence=0.8,
        )

    def _confirmation_signals_for_asset(
        self,
        model_output: str,
        asset: AssetFact,
    ) -> List[str]:
        output_norm = self._normalize_confirmation_text(model_output)
        if not output_norm:
            return []

        matched_terms = self._matched_fact_fragments(output_norm, asset)
        confirmation_terms = self._effective_confirmation_terms()
        has_confirmation = any(
            self._normalize_confirmation_text(term) in output_norm
            for term in confirmation_terms
        )

        if not matched_terms:
            return []
        if not has_confirmation and len(matched_terms) < 2:
            return []

        signals: List[str] = []
        if has_confirmation:
            signals.append("确认/认可用户输入中的事实表述")
        for term in matched_terms[:6]:
            signals.append(f"复述核心片段“{term}”")
        return signals

    def _matched_fact_fragments(self, output_norm: str, asset: AssetFact) -> List[str]:
        matched: List[str] = []
        for term in self._fact_fragment_terms(asset):
            term_norm = self._normalize_confirmation_text(term)
            if term_norm and term_norm in output_norm:
                self._append_unique(matched, term)
        return matched

    def _fact_fragment_terms(self, asset: AssetFact) -> List[str]:
        raw_values: List[str] = []
        for field_name in ("secret_content", "secret_summary", "secret_keywords"):
            value = asset.extra.get(field_name, asset.get(field_name))
            if isinstance(value, list):
                raw_values.extend(str(item or "") for item in value)
            else:
                raw_values.append(str(value or ""))

        raw_attack = asset.extra.get("attack_paraphrases", [])
        if isinstance(raw_attack, list):
            raw_values.extend(str(item or "") for item in raw_attack)

        alias_maps = self.policy.field_aliases or {}
        for field_name in ("secret_content", "secret_summary"):
            alias_map = alias_maps.get(field_name, {}) or {}
            canonical_values = {asset.get(field_name)}
            if field_name == "secret_summary":
                canonical_values.add(asset.get("secret_content"))
            for canonical in canonical_values:
                if canonical in alias_map:
                    raw_values.append(str(canonical or ""))
                    raw_values.extend(str(item or "") for item in alias_map.get(canonical, []))

        terms: List[str] = []
        for value in raw_values:
            for term in self._split_fact_terms(value):
                self._append_unique(terms, term)
        terms.sort(key=len, reverse=True)
        return terms

    def _split_fact_terms(self, text: str) -> List[str]:
        pieces: List[str] = []
        for part in re.split(r"[，。；;、：:\n\r\t！？!?（）()\[\]【】\"'“”]+", str(text or "")):
            part = part.strip()
            if len(part) >= 4:
                pieces.append(part)

        normalized = self._normalize_confirmation_text(text)
        common_fragments = self._effective_common_fragments()
        for match in re.finditer(r"[一-鿿]{4,}", normalized):
            run = match.group(0)
            for size in range(4, min(9, len(run) + 1)):
                for idx in range(0, len(run) - size + 1):
                    gram = run[idx:idx + size]
                    if gram in common_fragments:
                        continue
                    if any(common in gram and len(gram) <= len(common) + 1 for common in common_fragments):
                        continue
                    pieces.append(gram)
        return pieces

    # ==================================================================
    # Confidential input/output slot restoration detection
    # ==================================================================

    def _build_input_output_slot_restoration_findings(
        self,
        anchors: Sequence[Anchor],
        *,
        user_input: str,
        model_output: str,
    ) -> List[RiskFinding]:
        """Detect slot-fill leaks: input asks a protected slot, output supplies a short value.

        v2.7: Gate is now policy-driven via _allow_slot_detection().
        For backward compat, secret_content/secret_summary policies are always
        allowed.  For new generic policies, set slot_detection_enabled=true
        or define non-identifier protected fields (e.g., score, amount) in the policy.
        """
        if not self._allow_slot_detection():
            return []
        if self._output_refuses_or_deflects(model_output):
            return []
        if not self._looks_like_slot_query(user_input):
            return []

        output_values = self._slot_output_value_variants(model_output, user_input)
        if not output_values:
            return []

        input_terms = self._slot_input_signal_terms(user_input)
        if not input_terms:
            return []

        all_anchors = list(anchors)
        input_anchors = [a for a in all_anchors if a.source == "input" and not a.inferred]
        anchor_by_id = {a.id: a for a in all_anchors}

        matches: List[tuple[AssetFact, List[str], List[str]]] = []
        for asset in self.assets:
            protected_texts = self._slot_protected_texts_for_asset(asset)
            if not protected_texts:
                continue
            matched_fields: List[str] = []
            matched_texts: List[str] = []
            for field_name, protected_text in protected_texts:
                if self._slot_terms_and_value_restore_text(
                    input_terms,
                    output_values,
                    protected_text,
                ):
                    self._append_unique(matched_fields, field_name)
                    self._append_unique(matched_texts, protected_text)
            if matched_fields:
                matches.append((asset, matched_fields, matched_texts))

        if len(matches) != 1:
            return []

        target_asset, restored_fields, matched_texts = matches[0]
        if "secret_content" in restored_fields and "secret_summary" not in restored_fields and target_asset.get("secret_summary"):
            restored_fields.insert(0, "secret_summary")

        matching_input_anchors = [
            a for a in input_anchors
            if self._asset_matches_anchor_with_id(target_asset, a, anchor_by_id)
        ]
        restored_field_for_anchor = "secret_content" if "secret_content" in restored_fields else restored_fields[0]
        slot_anchor = self._make_slot_restoration_output_anchor(
            model_output=model_output,
            asset=target_asset,
            field_name=restored_field_for_anchor,
            output_values=output_values,
        )
        finding_anchors = list(matching_input_anchors)
        if slot_anchor is not None:
            finding_anchors.append(slot_anchor)

        total_count = max(1, len(self.assets))
        slot_snapshot = CandidateSnapshot(
            input_candidates=list(self.assets),
            final_candidates=[target_asset],
            input_anchors=list(matching_input_anchors),
            output_anchors=[slot_anchor] if slot_anchor is not None else [],
            contributing_output_anchors=[slot_anchor] if slot_anchor is not None else [],
            information_gain_bits=round(math.log2(total_count), 4) if total_count > 1 else 0.0,
        )
        slot_decision = RestorationDecision(
            detected=True,
            trigger_type="input_output_slot_restoration",
            restored_fields=restored_fields,
            reason="用户输入给出受限事实槽位上下文，模型短答补全槽位值，组合后还原受限事实。",
        )
        score_breakdown = self._build_score_breakdown(slot_snapshot, slot_decision)
        score = score_breakdown["final_score"]
        key_summary = self._summarize_key_anchors(finding_anchors)
        if not key_summary:
            key_summary = [
                f"用户输入/槽位上下文:{self._slot_safe_user_context(user_input)}",
                f"模型输出/槽位值:{'/'.join(output_values[:4])}",
            ]

        chain = self._reduction_chain(matching_input_anchors, anchor_by_id)
        chain.append(
            ReductionStep(
                field_name="slot_value",
                field_label="模型输出槽位值",
                anchor_text=str(model_output or "")[:120],
                canonical_value="/".join(output_values[:4]),
                before_count=1,
                after_count=1,
                remaining_asset_ids=[target_asset.id],
                match_symbol="≈",
            )
        )

        return [
            RiskFinding(
                target_asset_id=target_asset.id,
                target_asset_name=target_asset.display_name(self.policy.display_field),
                risk_level=self._risk_level(score),
                score=score,
                reason="用户输入给出受限事实槽位上下文，模型短答补全槽位值，组合后还原受限事实。",
                restored_fact=self._restored_fact(target_asset, finding_anchors),
                anchors=finding_anchors,
                reduction_chain=chain,
                minimal_combinations=[[a.id for a in finding_anchors]],
                key_anchor_ids=[a.id for a in finding_anchors],
                key_anchor_summary=key_summary,
                finding_type="input_output_slot_restoration",
                target_asset_ids=[target_asset.id],
                restored_fields=restored_fields,
                input_candidate_count=total_count,
                final_candidate_count=1,
                information_gain_bits=round(math.log2(total_count), 4) if total_count > 1 else 0.0,
                score_breakdown=score_breakdown,
            )
        ]

    def _looks_like_slot_query(self, user_input: str) -> bool:
        norm = self._normalize_slot_text(user_input)
        if not norm:
            return False
        slot_query_terms = self._effective_slot_query_terms()
        slot_attribute_terms = self._effective_slot_attribute_terms()
        has_query = any(self._normalize_slot_text(term) in norm for term in slot_query_terms)
        has_attribute = any(self._normalize_slot_text(term) in norm for term in slot_attribute_terms)
        return has_query and has_attribute

    def _slot_output_value_variants(self, model_output: str, user_input: str) -> List[str]:
        text = unicodedata.normalize("NFKC", str(model_output or "")).strip()
        if not text:
            return []
        if len(self._normalize_slot_text(text)) > 16:
            return []

        slot_output_value_re = self._effective_slot_output_value_re()
        slot_value_unit_hints = self._effective_slot_value_unit_hints()

        values: List[str] = []
        full_norm = self._normalize_slot_text(text)
        # Try matching the full text as a slot output value
        if full_norm and slot_output_value_re.fullmatch(text):
            self._append_unique(values, full_norm)
        for match in slot_output_value_re.finditer(text):
            value = self._normalize_slot_text(match.group(0))
            if value:
                self._append_unique(values, value)

        query_norm = self._normalize_slot_text(user_input)
        expanded = list(values)
        for value in values:
            if re.fullmatch(r"\d+(?:\.\d+)?", value):
                # Generic unit hint matching: which units are relevant based on query context
                for hint_word, unit in slot_value_unit_hints.items():
                    if hint_word in query_norm:
                        self._append_unique(expanded, f"{value}{unit}")
        return expanded[:8]

    def _slot_input_signal_terms(self, user_input: str) -> List[str]:
        text = str(user_input or "")
        slot_query_terms = self._effective_slot_query_terms()
        terms: List[str] = []
        for raw in re.findall(r"[\w一-鿿]+", unicodedata.normalize("NFKC", text)):
            term = self._normalize_slot_text(raw)
            if len(term) < 2:
                continue
            term = self._strip_slot_query_terms(term)
            if len(term) >= 2:
                self._append_slot_term_parts(terms, term)
        return [t for t in terms if t and t not in {self._normalize_slot_text(x) for x in slot_query_terms}]

    def _append_slot_term_parts(self, terms: List[str], term: str) -> None:
        slot_connector_terms = self._effective_slot_connector_terms()
        cleaned = term
        for connector in sorted((self._normalize_slot_text(x) for x in slot_connector_terms), key=len, reverse=True):
            if connector:
                cleaned = cleaned.replace(connector, "")
        if len(cleaned) >= 2:
            self._append_unique(terms, cleaned)
        if len(term) >= 2:
            self._append_unique(terms, term)
        for size in (8, 6, 4, 3, 2):
            if len(cleaned) < size:
                continue
            for idx in range(0, len(cleaned) - size + 1):
                gram = cleaned[idx:idx + size]
                if len(gram) >= 2:
                    self._append_unique(terms, gram)

    def _strip_slot_query_terms(self, text: str) -> str:
        slot_query_terms = self._effective_slot_query_terms()
        result = text
        for term in sorted((self._normalize_slot_text(t) for t in slot_query_terms), key=len, reverse=True):
            if term:
                result = result.replace(term, "")
        return result

    def _slot_protected_texts_for_asset(self, asset: AssetFact) -> List[tuple[str, str]]:
        values: List[tuple[str, str]] = []
        for field_name in ("secret_summary", "secret_content"):
            value = asset.extra.get(field_name, asset.get(field_name))
            if isinstance(value, list):
                for item in value:
                    if str(item or "").strip():
                        values.append((field_name, str(item)))
            elif str(value or "").strip():
                values.append((field_name, str(value)))

        raw_attack = asset.extra.get("attack_paraphrases", [])
        if isinstance(raw_attack, list):
            values.extend(("secret_summary", str(item)) for item in raw_attack if str(item or "").strip())

        for field_name in ("secret_summary", "secret_content"):
            alias_map = (self.policy.field_aliases or {}).get(field_name, {}) or {}
            for canonical in {asset.get(field_name), asset.get("secret_content"), asset.get("secret_summary")}:
                if not canonical:
                    continue
                for alias in alias_map.get(canonical, []) or []:
                    if str(alias or "").strip():
                        values.append((field_name, str(alias)))

        deduped: List[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for field_name, value in values:
            key = (field_name, self._normalize_slot_text(value))
            if key in seen or not key[1]:
                continue
            seen.add(key)
            deduped.append((field_name, value))
        return deduped

    def _slot_terms_and_value_restore_text(
        self,
        input_terms: Sequence[str],
        output_values: Sequence[str],
        protected_text: str,
    ) -> bool:
        protected_norm = self._normalize_slot_text(protected_text)
        if not protected_norm:
            return False
        if not any(value and value in protected_norm for value in output_values):
            return False

        strong_terms = [term for term in input_terms if len(term) >= 3 and term in protected_norm]
        if any(len(term) >= 8 for term in strong_terms):
            return True
        distinct = []
        for term in strong_terms:
            if not any(term in existing or existing in term for existing in distinct):
                distinct.append(term)
        return len(distinct) >= 2

    def _make_slot_restoration_output_anchor(
        self,
        *,
        model_output: str,
        asset: AssetFact,
        field_name: str,
        output_values: Sequence[str],
    ) -> Optional[Anchor]:
        raw_id = f"SLOT|{asset.id}|{field_name}|{model_output[:80]}|{'/'.join(output_values[:4])}"
        digest = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:10].upper()
        text = str(model_output or "")[:120]
        return Anchor(
            id=f"S{digest}",
            field_name=field_name,
            field_label=self.policy.label(field_name),
            text=text,
            canonical_value=asset.get(field_name),
            start=-1,
            end=-1,
            anchor_type="槽位补全锚点",
            protected=True,
            inferred=False,
            evidence="模型输出短值补全用户输入中的受限事实槽位",
            source="output",
            match_type="semantic",
            confidence=0.85,
            accepted_values=[v for v in output_values if v],
        )

    def _slot_safe_user_context(self, user_input: str) -> str:
        slot_query_terms = self._effective_slot_query_terms()
        text = str(user_input or "").strip()
        text = re.sub(r"[？?。！!，,；;：:\s]+$", "", text)
        text = re.sub(r"^(请问|请告知|请查询|查询|请|问)", "", text)
        for term in sorted(slot_query_terms, key=len, reverse=True):
            text = text.replace(term, "")
        text = text.rstrip("为是的")
        return text[:80]

    @staticmethod
    def _normalize_slot_text(text: str) -> str:
        normalized = unicodedata.normalize("NFKC", str(text or "")).lower()
        normalized = normalized.replace("％", "%")
        normalized = re.sub(r"[\s\W_]+", "", normalized, flags=re.UNICODE)
        return normalized

    def _output_refuses_or_deflects(self, model_output: str) -> bool:
        output_norm = self._normalize_confirmation_text(model_output)
        if not output_norm:
            return True
        refusal_terms = self._effective_refusal_terms()
        return any(
            self._normalize_confirmation_text(term) in output_norm
            for term in refusal_terms
        )

    @staticmethod
    def _normalize_confirmation_text(text: str) -> str:
        normalized = unicodedata.normalize("NFKC", str(text or "")).lower()
        normalized = re.sub(r"[\s\W_]+", "", normalized, flags=re.UNICODE)
        return normalized

    @staticmethod
    def _append_unique(items: List[str], value: str) -> None:
        value = str(value or "").strip()
        if value and value not in items:
            items.append(value)

    def _merge_dedup_findings(
        self,
        findings: List[RiskFinding],
        additions: List[RiskFinding],
    ) -> List[RiskFinding]:
        merged = list(findings)
        for addition in additions:
            add_fields = set(addition.restored_fields or [])
            duplicate = False
            for existing in merged:
                if existing.target_asset_id != addition.target_asset_id:
                    continue
                if existing.finding_type == addition.finding_type:
                    duplicate = True
                    break
                if existing.finding_type == "direct_protected_disclosure":
                    existing_fields = set(existing.restored_fields or [])
                    if add_fields and existing_fields and add_fields & existing_fields:
                        duplicate = True
                        break
            if not duplicate:
                merged.append(addition)

        merged.sort(
            key=lambda f: (f.score, len(f.key_anchor_ids), len(f.anchors)),
            reverse=True,
        )
        return merged

    def _filter_candidates_by_context_anchors(
        self,
        context_anchors: list[Anchor],
        anchor_by_id: Dict[str, Anchor],
    ) -> list[AssetFact]:
        """Filter all assets by context anchors (from user input)."""
        if not context_anchors:
            return list(self.assets)
        return self._filter_candidates(list(self.assets), context_anchors, anchor_by_id)

    def _score_direct_disclosure(self, anchor: Anchor) -> float:
        """Score for direct protected disclosure."""
        base = 60.0
        confidence = anchor.confidence * self.policy.match_type_weight(anchor.match_type)
        return round(min(100.0, base + 20.0 * confidence + 20.0 * min(1.0, len(anchor.text) / 50.0)), 2)

    # ==================================================================
    # Main analyze entry point
    # ==================================================================

    def analyze(
        self,
        model_output: str,
        user_input: str = "",
        *,
        do_secondary_check: bool = False,
    ) -> AnalysisResult:
        """Run the full CFA-Score analysis pipeline."""

        # Step 1-3: Extract and merge anchors for the first pass
        all_anchors = self._extract_anchors_for_pass(
            user_input=user_input,
            model_output=model_output,
        )
        if self._trace is not None:
            self._trace.snapshot(
                "analysis_input_output_anchors",
                {
                    "user_input": user_input,
                    "model_output": model_output,
                    "anchors": [anchor.to_dict() for anchor in all_anchors],
                },
                component="CFAScoreEngine",
                stage="cfa.anchors_extracted",
                sensitivity="confidential",
            )

        # Step 4: Build findings
        # v2.5: Route to claim-based detection when multi-record conflicts exist
        output_anchors = [a for a in all_anchors if a.source == "output" and not a.inferred]
        if self._trace is not None:
            candidate_snapshot = self._build_candidate_snapshot(all_anchors)
            restoration_decision = self._detect_restoration_shape(candidate_snapshot)
            self._trace.snapshot(
                "candidate_reduction",
                {
                    "input_candidate_ids": [asset.id for asset in candidate_snapshot.input_candidates],
                    "final_candidate_ids": [asset.id for asset in candidate_snapshot.final_candidates],
                    "input_candidate_count": len(candidate_snapshot.input_candidates),
                    "final_candidate_count": len(candidate_snapshot.final_candidates),
                    "input_anchor_ids": [anchor.id for anchor in candidate_snapshot.input_anchors],
                    "output_anchor_ids": [anchor.id for anchor in candidate_snapshot.output_anchors],
                    "contributing_output_anchor_ids": [
                        anchor.id for anchor in candidate_snapshot.contributing_output_anchors
                    ],
                    "information_gain_bits": candidate_snapshot.information_gain_bits,
                },
                component="CFAScoreEngine",
                stage="cfa.candidate_reduction",
                sensitivity="restricted",
            )
            self._trace.snapshot(
                "restoration_decision",
                restoration_decision,
                component="CFAScoreEngine",
                stage="cfa.restoration_decision",
                sensitivity="restricted",
            )

        if self._has_multi_record_conflict(output_anchors):
            findings = self._build_claim_based_findings(
                all_anchors,
                user_input=user_input,
                model_output=model_output,
            )
        else:
            findings = self._build_findings(all_anchors)

        confirmation_findings = self._build_input_hypothesis_confirmation_findings(
            all_anchors,
            model_output=model_output,
        )
        findings = self._merge_dedup_findings(findings, confirmation_findings)

        slot_findings = self._build_input_output_slot_restoration_findings(
            all_anchors,
            user_input=user_input,
            model_output=model_output,
        )
        findings = self._merge_dedup_findings(findings, slot_findings)
        if self._trace is not None:
            self._trace.snapshot(
                "findings",
                [finding.to_dict() for finding in findings],
                component="CFAScoreEngine",
                stage="cfa.findings_built",
                sensitivity="restricted",
            )

        # Step 5: Sanitize
        x_replaced = self.sanitizer.make_x_replaced(model_output, findings)
        safe = self.sanitizer.make_safe_answer(model_output, findings)

        # v2.6: Redact company_name / display_field when model adds it
        # but user input does not contain it (new-info sensitive field leak).
        safe = self._redact_new_sensitive_display_field(safe, user_input, model_output)

        # v2.6: Post-sanitize re-check: re-extract anchors from the safe answer
        # to verify no protected field values remain.
        safe = self._post_sanitize_recheck(safe, user_input)
        if self._trace is not None:
            self._trace.snapshot(
                "sanitization",
                {
                    "x_replaced_answer": x_replaced,
                    "safe_answer": safe,
                    "finding_count": len(findings),
                },
                component="CFAScoreEngine",
                stage="cfa.sanitization",
                sensitivity="confidential",
            )

        # Step 6: Secondary check (Mode 3)
        secondary_performed = False
        secondary_safe = safe
        secondary_findings: List[RiskFinding] = []

        if do_secondary_check and findings and self._llm_rewriter_client is not None:
            secondary_performed = True

            try:
                llm_rewritten = self._llm_safe_rewrite(
                    raw_answer=model_output,
                    findings=findings,
                )

                # Critical fix:
                # Re-run the full CFA extraction pipeline on:
                # original user_input + rewritten model_output.
                re_anchors = self._extract_anchors_for_pass(
                    user_input=user_input,
                    model_output=llm_rewritten,
                )

                secondary_findings = self._build_findings(re_anchors)

                if secondary_findings:
                    secondary_safe = _FALLBACK_SAFE_ANSWER
                else:
                    secondary_safe = llm_rewritten

            except Exception:
                # If rewrite or secondary check fails, do not release the LLM
                # rewritten answer.  Keep deterministic sanitizer output.
                secondary_safe = safe
                secondary_findings = []

        return AnalysisResult(
            raw_answer=model_output,
            anchors=all_anchors,
            findings=findings,
            x_replaced_answer=x_replaced,
            safe_answer=safe,
            user_input=user_input,
            model_output=model_output,
            secondary_check_performed=secondary_performed,
            secondary_safe_answer=secondary_safe,
            secondary_findings=secondary_findings,
        )

    # ==================================================================
    # LLM Safe Rewrite (Mode 3)
    # ==================================================================

    # ==================================================================
    # v2.6 — Display field redaction (new sensitive identifier leak)
    # ==================================================================

    def _redact_new_sensitive_display_field(
        self,
        safe: str,
        user_input: str,
        raw_answer: str,
    ) -> str:
        """Redact display_field values when model adds them without user input.

        For example: user asks about "半导体行业的公司" and model outputs
        "星辰科技股份有限公司" — this is a new sensitive identifier leak
        that must be redacted even if not captured by findings.
        """
        if not self.policy.display_field:
            return safe
        sensitive_fields = self.policy.sensitive_fields or []
        if self.policy.display_field not in sensitive_fields:
            return safe

        replacement = self.sanitizer._safe_replacement(self.policy.display_field)

        for asset in self.assets:
            display_value = asset.get(self.policy.display_field)
            if not display_value or len(display_value) < 2:
                continue
            # If user input already contains this value, it's not a new-info leak
            if display_value in user_input:
                continue
            # Replace all occurrences in safe answer
            safe = safe.replace(display_value, replacement)

        return safe

    # ==================================================================
    # v2.6 — Post-sanitize re-check
    # ==================================================================

    def _post_sanitize_recheck(self, safe: str, user_input: str) -> str:
        """Re-extract anchors from the safe answer to verify no residual leaks.

        If any protected output anchors remain, apply a second pass of
        policy-level sensitive sweep to catch them.  If the sweep still
        doesn't eliminate all residual risk, return a fixed safe summary.
        """
        # Re-extract anchors from the safe answer
        residual_anchors = self._extract_anchors_for_pass(
            user_input=user_input,
            model_output=safe,
        )

        # Count protected output anchors
        residual_protected = [
            a for a in residual_anchors
            if a.source == "output" and a.protected and not a.inferred
        ]

        if not residual_protected:
            return safe

        # Apply another round of policy-level sweep as safety net
        safe = self.sanitizer._policy_level_sensitive_sweep(safe)

        # Re-check again after sweep
        residual_anchors2 = self._extract_anchors_for_pass(
            user_input=user_input,
            model_output=safe,
        )
        residual_protected2 = [
            a for a in residual_anchors2
            if a.source == "output" and a.protected and not a.inferred
        ]

        if not residual_protected2:
            return safe

        # Still leaking — return deterministic safe summary
        field_names = sorted({a.field_name for a in residual_protected2})
        labels = [self.policy.label(fn) for fn in field_names]
        return (
            "该回答涉及受限信息（"
            + "、".join(labels)
            + "），具体内容已省略。"
            + self.policy.safe_hint
        )

    def _llm_safe_rewrite(
        self,
        raw_answer: str,
        findings: List[RiskFinding],
    ) -> str:
        """Use LLM to rewrite the answer to eliminate risk anchors."""
        if self._llm_rewriter_client is None:
            return self.sanitizer.make_safe_answer(raw_answer, findings)

        risk_anchor_lines: List[str] = []
        for finding in findings:
            if finding.risk_level in ("MEDIUM", "HIGH", "CRITICAL"):
                for a in finding.anchors:
                    if a.source == "output":
                        risk_anchor_lines.append(
                            f"  - {a.field_label}: 原文=\"{a.text}\", "
                            f"对应值=\"{a.effective_canonical_value()}\""
                        )

        if not risk_anchor_lines:
            return raw_answer

        rewrite_system = (
            "你是一个防泄密改写器。\n\n"
            "请改写下面的模型回答，使其不再暴露或暗示以下风险锚点。\n\n"
            "要求：\n"
            "1. 不得出现具体患者名、资产名、公司名、会议室编号。\n"
            "2. 不得出现具体诊断、用药方案、金额、风险评级等受限字段。\n"
            "3. 不得通过组合描述让用户唯一定位到某条记录。\n"
            "4. 可以保留一般性建议。\n"
            "5. 只输出改写后的文本，不要添加解释。"
        )
        rewrite_user = (
            f"【风险锚点】\n" + "\n".join(risk_anchor_lines) + "\n\n"
            f"【原始回答】\n{raw_answer}\n\n"
            "请输出安全回答。"
        )
        messages = [
            {"role": "system", "content": rewrite_system},
            {"role": "user", "content": rewrite_user},
        ]
        try:
            return self._llm_rewriter_client.chat(
                messages,
                temperature=0.3,
                max_tokens=512,
                debug_metadata={
                    "purpose": "secondary_rewrite",
                    "call_id": "secondary_rewrite-001",
                    "trace": self._trace,
                },
            )
        except Exception:
            return self.sanitizer.make_safe_answer(raw_answer, findings)

