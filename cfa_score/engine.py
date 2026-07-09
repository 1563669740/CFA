from __future__ import annotations

import itertools
import math
from itertools import combinations
from collections import Counter
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .anchor_verifier import AnchorVerifier
from .deepseek import DeepSeekClient
from .extractor import RuleBasedAnchorExtractor
from .llm_extractor import LLMSemanticAnchorExtractor
from .claim_builder import ClaimBuilder
from .models import (
    AnalysisResult,
    Anchor,
    AssetFact,
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
    ):
        if not assets:
            raise ValueError("assets must not be empty")
        self.assets = list(assets)
        self.policy = policy
        self._mode = mode

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
        the same restoration decision.

        A valid combination must contain at least one output anchor, otherwise the
        risk is caused by user input alone and should not be attributed to model output.
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

        # Direct protected disclosure is special:
        # each directly disclosed protected output anchor is itself a minimal combo.
        if decision.trigger_type == "direct_protected_disclosure":
            return [
                [a]
                for a in usable_outputs
                if a.protected
            ][:max_results]

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

                if self._combo_triggers_same_restoration(
                    combo,
                    anchor_by_id,
                    decision,
                ):
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

    def _score_restoration(
        self,
        snapshot: CandidateSnapshot,
        decision: RestorationDecision,
    ) -> float:
        """Compute CFA-Score with indirect-restoration awareness."""
        input_count = len(snapshot.input_candidates)
        final_count = len(snapshot.final_candidates)
        if final_count <= 0:
            return 0.0

        uniqueness_score = (
            1.0
            if final_count <= self.policy.uniqueness_k
            else 1.0 / final_count
        )
        gain_score = min(1.0, snapshot.information_gain_bits / 3.0)

        # Average output anchor confidence
        output_confidence = 0.0
        if snapshot.contributing_output_anchors:
            output_confidence = sum(
                a.confidence * self.policy.match_type_weight(a.match_type)
                for a in snapshot.contributing_output_anchors
            ) / len(snapshot.contributing_output_anchors)

        if decision.trigger_type == "direct_protected_disclosure":
            base = 55.0
        elif decision.trigger_type == "indirect_asset_restoration":
            base = 50.0
        else:
            base = 40.0

        score = (
            base
            + 25.0 * uniqueness_score
            + 15.0 * gain_score
            + 10.0 * min(1.0, output_confidence)
        )
        return round(min(100.0, score), 2)

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

        for asset in target_assets:
            matching_anchors = self._anchors_matching_asset(asset, key_anchors, anchor_by_id)
            chain = self._reduction_chain(matching_anchors, anchor_by_id)

            score = self._score_restoration(effective_snapshot, decision)
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

        # 3. v2.5: Normalized anchors for fuzzy-matched amounts, rates, ratings, collateral
        normalized_anchors = self._extract_normalized_anchors(user_input, model_output)
        # Merge normalized anchors with existing (avoid duplicates by span)
        all_anchors = self._merge_normalized_anchors(merged, normalized_anchors)

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
            # Direct disclosure: findings per protected anchor
            for anchor in protected_output:
                score = self._score_direct_disclosure(anchor)
                level = self._risk_level(score)
                if level == "LOW":
                    continue

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
                        input_candidate_count=len(self.assets),
                        final_candidate_count=len(claim.candidate_assets),
                        information_gain_bits=0.0,
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

            for asset in final_candidates:
                matching = self._anchors_matching_asset(
                    asset, claim.local_output_anchors, anchor_by_id
                )
                chain = self._reduction_chain(matching, anchor_by_id)
                score = self._score_restoration(snapshot, decision)
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
                    )
                )
            return findings

        # Priority 3-4: Protected field convergence
        before_set = input_candidates or list(self.assets)
        restored_fields = self._find_restored_protected_fields(before_set, final_candidates)
        if restored_fields:
            for asset in final_candidates:
                matching = self._anchors_matching_asset(
                    asset, claim.local_output_anchors, anchor_by_id
                )
                chain = self._reduction_chain(matching, anchor_by_id)
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
                score = self._score_restoration(snapshot, decision)
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
                    )
                )
            return findings

        return findings

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

        # Step 4: Build findings
        # v2.5: Route to claim-based detection when multi-record conflicts exist
        output_anchors = [a for a in all_anchors if a.source == "output" and not a.inferred]
        if self._has_multi_record_conflict(output_anchors):
            findings = self._build_claim_based_findings(
                all_anchors,
                user_input=user_input,
                model_output=model_output,
            )
        else:
            findings = self._build_findings(all_anchors)

        # Step 5: Sanitize
        x_replaced = self.sanitizer.make_x_replaced(model_output, findings)
        safe = self.sanitizer.make_safe_answer(model_output, findings)

        # v2.6: Redact company_name / display_field when model adds it
        # but user input does not contain it (new-info sensitive field leak).
        safe = self._redact_new_sensitive_display_field(safe, user_input, model_output)

        # v2.6: Post-sanitize re-check: re-extract anchors from the safe answer
        # to verify no protected field values remain.
        safe = self._post_sanitize_recheck(safe, user_input)

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
                messages, temperature=0.3, max_tokens=512
            )
        except Exception:
            return self.sanitizer.make_safe_answer(raw_answer, findings)