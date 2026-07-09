from __future__ import annotations

from typing import Dict, List, Literal, Optional, Sequence

from .models import Anchor, AssetFact, FieldPolicy
from .semantic_index import SemanticIndex


AnchorSource = Literal["input", "output"]


class AnchorVerifier:
    """Verifies LLM-generated anchors against the fact pool and policy.

    This is a critical safety layer that guards against LLM hallucinations:

    1. field_name must exist in policy.field_order
    2. canonical_value or accepted_values must exist in the fact pool
    3. source_text must be findable in user_input or model_output
    4. protected flag is derived from policy, NOT from LLM output
    5. confidence below threshold → discard
    6. excessive accepted_values → treat as ambiguous/weak anchor

    v2.4 — Single-segment verification:
        ``verify_segment(anchor, source_text=..., expected_source=...)`` is the
        recommended API.  Source is forced by the caller and evidence must exist
        in the supplied ``source_text`` (not the other segment).
    """

    def __init__(
        self,
        policy: FieldPolicy,
        assets: Sequence[AssetFact],
        semantic_index: SemanticIndex,
    ):
        self._policy = policy
        self._assets = list(assets)
        self._index = semantic_index

        # Pre-build valid values lookup
        self._valid_values: Dict[str, set] = {}
        for asset in self._assets:
            for field_name in self._policy.field_order:
                value = asset.get(field_name)
                if value:
                    self._valid_values.setdefault(field_name, set()).add(value)

    # ------------------------------------------------------------------
    # Public API — single-segment verification (recommended)
    # ------------------------------------------------------------------

    def verify_segment(
        self,
        anchor: Anchor,
        *,
        source_text: str,
        expected_source: AnchorSource,
        candidate_whitelist: Dict[str, set] | None = None,
    ) -> Optional[Anchor]:
        """Verify a single anchor against one specific source segment.

        Key safety properties:
        - ``source`` MUST equal ``expected_source`` (forced by caller).
        - Evidence text MUST exist in ``source_text`` (not the other segment).
        - Values in ``candidate_whitelist`` (if provided) constrain accepted values.
        """
        # Rule 0: source must match expected — forced by caller
        if anchor.source != expected_source:
            return None

        # Rule 1: field_name must be valid
        field_name = anchor.field_name
        if field_name not in self._policy.field_order:
            return None

        # Rule 2: source_text must exist in the supplied source_text
        evidence_text = anchor.text
        if anchor.match_type != "inferred":
            actual_span = self._find_source_span(source_text, evidence_text)
            if actual_span is None:
                return None
            start, end, evidence_text = actual_span
        else:
            start = anchor.start
            end = anchor.end

        # Rule 3: validate canonical_value and accepted_values against fact pool
        valid_values = self._valid_values.get(field_name, set())

        # Apply candidate whitelist if provided
        allowed_values = valid_values
        if candidate_whitelist is not None:
            whitelist_for_field = candidate_whitelist.get(field_name, set())
            allowed_values = valid_values & whitelist_for_field

        canonical_value = anchor.canonical_value
        accepted_values = list(anchor.accepted_values)

        # If canonical_value exists but is not allowed, move it to accepted
        # BEFORE filtering so the whitelist can also reject it.
        if canonical_value and canonical_value not in allowed_values:
            if canonical_value not in accepted_values:
                accepted_values.append(canonical_value)
            canonical_value = ""

        # Filter accepted_values to only those in allowed_values
        accepted_values = [v for v in accepted_values if v in allowed_values]

        # Rule 4: must have at least one valid mapping
        if not canonical_value and not accepted_values:
            return None

        # Rule 5: confidence must meet threshold
        if anchor.confidence < self._policy.llm_confidence_threshold:
            return None

        # Rule 6: if accepted_values is too broad, downgrade match_type to ambiguous
        match_type = anchor.match_type
        if len(accepted_values) > self._policy.llm_max_accepted_values:
            match_type = "ambiguous"

        # Rule 7: protected flag must come from policy, not LLM
        protected = field_name in self._policy.protected_fields

        # Rebuild anchor with verified data
        verified = Anchor(
            id=anchor.id,
            field_name=field_name,
            field_label=self._policy.label(field_name),
            text=evidence_text,
            canonical_value=canonical_value,
            start=start,
            end=end,
            anchor_type=anchor.anchor_type,
            protected=protected,
            inferred=anchor.inferred,
            evidence=anchor.evidence,
            source_anchor_id=anchor.source_anchor_id,
            # --- CRITICAL: source forced by caller ---
            source=expected_source,
            match_type=match_type,
            confidence=anchor.confidence,
            llm_reason=anchor.llm_reason,
            accepted_values=accepted_values,
        )

        return verified

    def verify_segment_all(
        self,
        anchors: Sequence[Anchor],
        *,
        source_text: str,
        expected_source: AnchorSource,
        candidate_whitelist: Dict[str, set] | None = None,
    ) -> List[Anchor]:
        """Verify a batch of anchors against one source segment."""
        verified: List[Anchor] = []
        for anchor in anchors:
            result = self.verify_segment(
                anchor,
                source_text=source_text,
                expected_source=expected_source,
                candidate_whitelist=candidate_whitelist,
            )
            if result is not None:
                verified.append(result)
        return verified

    # ------------------------------------------------------------------
    # Public API — legacy dual-text (backward compat)
    # ------------------------------------------------------------------

    def verify(
        self,
        llm_anchor: Anchor,
        user_input: str,
        model_output: str,
    ) -> Optional[Anchor]:
        """Verify a single LLM anchor. Returns None if it fails verification.

        .. deprecated::
            Prefer ``verify_segment(anchor, source_text=..., expected_source=...)``.
        """
        # Rule 1: field_name must be valid
        field_name = llm_anchor.field_name
        if field_name not in self._policy.field_order:
            return None

        # Rule 2: source_text must exist in the claimed source
        source = llm_anchor.source
        source_text_ref = user_input if source == "input" else model_output
        source_text = llm_anchor.text

        # For inferred anchors, skip text-in-source check (they're derived, not literal)
        if llm_anchor.match_type != "inferred":
            if source_text and source_text not in source_text_ref:
                # Try case-insensitive match
                lower_text = source_text.lower()
                lower_ref = source_text_ref.lower()
                if lower_text not in lower_ref:
                    return None
                # Adjust text to the actual casing from source
                idx = lower_ref.index(lower_text)
                source_text = source_text_ref[idx : idx + len(source_text)]

        # Rule 3: validate canonical_value and accepted_values against fact pool
        valid_values = self._valid_values.get(field_name, set())

        canonical_value = llm_anchor.canonical_value
        accepted_values = list(llm_anchor.accepted_values)

        # Filter accepted_values to only those that exist in fact pool
        accepted_values = [v for v in accepted_values if v in valid_values]

        # If canonical_value exists but is not valid, try to move it to accepted_values
        if canonical_value and canonical_value not in valid_values:
            if canonical_value not in accepted_values:
                accepted_values.append(canonical_value)
            canonical_value = ""

        # Rule 4: must have at least one valid mapping
        if not canonical_value and not accepted_values:
            return None

        # Rule 5: confidence must meet threshold
        if llm_anchor.confidence < self._policy.llm_confidence_threshold:
            return None

        # Rule 6: if accepted_values is too broad, downgrade match_type to ambiguous
        match_type = llm_anchor.match_type
        if len(accepted_values) > self._policy.llm_max_accepted_values:
            match_type = "ambiguous"

        # Rule 7: protected flag must come from policy, not LLM
        protected = field_name in self._policy.protected_fields

        # Rebuild anchor with verified data
        verified = Anchor(
            id=llm_anchor.id,
            field_name=field_name,
            field_label=self._policy.label(field_name),
            text=source_text,
            canonical_value=canonical_value,
            start=llm_anchor.start,
            end=llm_anchor.end,
            anchor_type=llm_anchor.anchor_type,
            protected=protected,
            inferred=llm_anchor.inferred,
            evidence=llm_anchor.evidence,
            source_anchor_id=llm_anchor.source_anchor_id,
            source=source,
            match_type=match_type,
            confidence=llm_anchor.confidence,
            llm_reason=llm_anchor.llm_reason,
            accepted_values=accepted_values,
        )

        return verified

    def verify_all(
        self,
        llm_anchors: Sequence[Anchor],
        user_input: str,
        model_output: str,
    ) -> List[Anchor]:
        """Verify a batch of LLM anchors. Returns only the verified ones.

        .. deprecated::
            Prefer ``verify_segment_all(anchors, source_text=..., expected_source=...)``.
        """
        verified: List[Anchor] = []
        for anchor in llm_anchors:
            result = self.verify(anchor, user_input, model_output)
            if result is not None:
                verified.append(result)
        return verified

    # ------------------------------------------------------------------
    # Source-span helper
    # ------------------------------------------------------------------

    @staticmethod
    def _find_source_span(
        full_text: str,
        evidence_text: str,
    ) -> Optional[tuple]:
        """Locate ``evidence_text`` inside ``full_text``.

        Returns ``(start, end, actual_text)`` or ``None`` if not found.
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